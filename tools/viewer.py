#!/usr/bin/env python3
"""Live camera feed beside live per-link CSI waterfalls, both at 30 fps.

Each panel shows the last 20 packets on that link, one packet per column, newest
on the right, always full. No integration and no slot grid.

This is deliberately a monitor, not a measurement. CSI and video are aligned
offline from the timestamps recorded with every packet and every frame (see
capture.py), which is where UART delay and jitter can be handled properly.
Earlier versions tried to reconstruct real timing on screen and only manufactured
artefacts: a grid anchored to the current time blanked cells that jitter had
merely displaced, and rounding each inter-packet gap accumulated error, both
reporting far more loss than the recordings actually contain.

Because the x axis is packet order, the 20 columns span whatever time those 20
packets took. That span is measured and displayed, since it is the only remaining
sign of a link slowing down.

Each link gets its own panel. Pooling several transmitters into one waterfall
would be denser and wrong: consecutive columns would then be different physical
links depending on who held the token, so a handoff would read as a channel
change. TX down the rows, RX across the columns.

Each packet is normalised before display: a z-score per column, computed separately
within each of the three 64-wide CSI fields and ignoring the dead guard-band
subcarriers. That strips the packet's overall level -- AGC, distance, per-board
gain -- and leaves the frequency-selective shape the body actually modulates, and it
makes a quiet link as readable as a loud one. Scaling all 192 together instead just
encodes which field a subcarrier is in: their gains differ about fivefold. The colour range is then fixed in sigma
units, never fitted to the current frame: a scale that rescales itself makes a
quiet moment look identical to a loud one. **This is display only; every recording
stores raw amplitudes.**

The radio is driven from here too. Both this and the role commands need the same
serial port, and two processes cannot own it at once.

Which board transmits is switchable live from the button bar: round-robin, or any
one board pinned as the sole transmitter.

Recording is driven from the GUI: type a name, press REC, press STOP. Output is
identical to capture.py -- both call the same writer -- so one process can
monitor and record at once, which matters because they cannot share the ports.

A scripted protocol can drive a whole session: a YAML lists the takes in order,
each gets a lead-in, a fixed recording, then a rest, and a large cue card shows
the instruction and countdown to whoever is standing in the array. Pick one with
the "Protocol…" button or --protocol; its takes are written to a folder named
after the YAML, so a session stays together.

  python3 live_viewer.py
  python3 live_viewer.py --tx C --prefix take1
  python3 live_viewer.py --tx C --protocol ../../../../protocols/poses.yaml
"""

import argparse
import os
import re
import pathlib
import sys
import threading
import time
from collections import deque

import numpy as np
import serial

from PyQt5.Qt import *
from PyQt5 import QtCore
import pyqtgraph as pg

from capture import (Camera, CsiStream, JpegWriter, LABEL, default_outdir,
                            discover, find_colour_node, owner_of, resolve_prefix,
                            write_capture, yuyv_to_rgb)

pg.setConfigOptions(imageAxisOrder='row-major')

NPKT = 20             # columns on screen: the last 20 packets, one packet each
# ESP32 HT40 CSI is three 64-wide fields (LLTF | HT-LTF | STBC-HT-LTF) whose gains
# differ by ~5x. Normalising across all 192 at once mostly encodes *which field* a
# subcarrier belongs to and buries the within-field shape, so each is scaled alone.
SUB_BLOCK = 64
INK, SURFACE, MUTED = '#e8e8e6', '#1b1b1d', '#8b8a86'


LEAD, REC, GAP = 'lead', 'rec', 'gap'
PHASE_BG = {LEAD: '#7f6000', REC: '#7f1010', GAP: '#33333a'}


def default_projdir():
    """Project root, for the protocol picker. Same reasoning as default_outdir:
    inside the container __file__ cannot identify the checkout, so the mount wins."""
    if os.path.isdir('/workspace/tools'):
        return '/workspace'
    return str(pathlib.Path(__file__).resolve().parents[1])


def link_name(tx, rx):
    return f'{LABEL.get(tx[-5:], tx[-5:])}→{LABEL.get(rx[-5:], rx[-5:])}'


def safe_name(name):
    """Take names become filenames, so collapse whitespace and drop separators.
    A name with a space is legal but turns every later shell glob into a quoting
    problem, and a name containing "/" would silently write outside the session."""
    return re.sub(r'\s+', '_', name.strip()).replace('/', '_').replace('\\', '_')


def load_protocol(path):
    """Read a protocol YAML into (timing dict, [(name, instruction), ...]).

    Takes may be plain strings or mappings with name/instruction, because a quick
    protocol is often just a list of names and forcing the verbose form for that is
    friction. Duplicate names are rejected: each take writes <name>.npz, so a repeat
    would silently destroy the earlier take's data halfway through a session.

    `repeats: N` runs the whole set N times rather than repeating each take back to
    back, and suffixes the round number (neutral0 ... neutral1 ...). Cycling the set
    is the point: consecutive recordings of one pose share whatever the subject and
    the room were doing at that moment, so repeats taken minutes apart are far more
    independent samples than repeats taken ten seconds apart.
    """
    import yaml
    with open(path) as fh:
        doc = yaml.safe_load(fh) or {}
    if isinstance(doc, list):                       # bare list of takes
        doc = {'takes': doc}
    timing = {k: float(doc.get(k, d)) for k, d in
              (('lead_in', 3.0), ('duration', 5.0), ('gap', 3.0))}
    base = []
    for item in doc.get('takes') or []:
        if isinstance(item, str):
            base.append((safe_name(item), item))
        elif isinstance(item, dict) and item.get('name'):
            base.append((safe_name(str(item['name'])),
                         str(item.get('instruction', item['name']))))
        else:
            raise ValueError(f'bad take entry: {item!r}')
    if not base:
        raise ValueError(f'{path} lists no takes')
    seen = [n for n, _ in base]
    dupes = {n for n in seen if seen.count(n) > 1}
    if dupes:
        raise ValueError(f'duplicate take names would overwrite each other: {sorted(dupes)}')

    repeats = max(1, int(doc.get('repeats', 1)))
    takes = ([(f'{n}{r}', ins) for r in range(repeats) for n, ins in base]
             if repeats > 1 else list(base))
    timing['repeats'] = repeats
    timing['nbase'] = len(base)
    return timing, takes


class Cue(QWidget):
    """Big full-screen-ish cue card. The subject is across the room and cannot read
    the viewer's status line, so the instruction and countdown have to be legible
    at several metres."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Capture protocol')
        self.resize(1100, 720)
        lay = QVBoxLayout(self)
        self.step = QLabel('', alignment=QtCore.Qt.AlignCenter)
        self.step.setStyleSheet('font-size: 30px; color: #dddddd;')
        self.head = QLabel('', alignment=QtCore.Qt.AlignCenter)
        self.head.setStyleSheet('font-size: 66px; font-weight: bold; color: white;')
        self.head.setWordWrap(True)
        self.count = QLabel('', alignment=QtCore.Qt.AlignCenter)
        self.count.setStyleSheet('font-size: 190px; font-weight: bold; color: white;')
        self.sub = QLabel('', alignment=QtCore.Qt.AlignCenter)
        self.sub.setStyleSheet('font-size: 28px; color: #eeeeee;')
        self.sub.setWordWrap(True)
        for w in (self.step, self.head, self.count, self.sub):
            lay.addWidget(w)
        self.set_phase(GAP)

    def set_phase(self, phase):
        self.setStyleSheet(f'QWidget {{ background: {PHASE_BG[phase]}; }}')

    def show_state(self, phase, step, head, secs, sub):
        self.set_phase(phase)
        self.step.setText(step)
        self.head.setText(head)
        self.count.setText('' if secs is None else str(int(np.ceil(max(secs, 0)))))
        self.sub.setText(sub)


class Live(QWidget):
    def __init__(self, boards, cam, args):
        super().__init__()
        self.boards, self.cam, self.args = boards, cam, args
        self.macs = sorted(boards)
        self.stop = threading.Event()
        self.t0 = time.time()

        # (t, amp) per ordered link, only ever holding the visible window
        self.buf = {}
        self.lock = threading.Lock()
        self.frame = None
        self.frame_times = deque(maxlen=60)
        # Last time each board delivered a CSI line. A dropped USB port does not
        # raise: readline() just returns empty forever, so the reader thread spins
        # and the board goes silent with nothing anywhere reporting it. That cost a
        # 100-take session, so silence is now a first-class, visible state.
        self.last_seen = {m: time.time() for m in self.macs}
        self.read_errors = 0
        self.clipped = 0
        # CLOCK_MONOTONIC to wall clock, measured once: the two drift far too slowly
        # to matter across a run, and re-measuring per frame would inject exactly the
        # scheduling jitter the driver timestamp exists to avoid.
        self.mono_offset = float(np.median([time.time() - time.monotonic()
                                            for _ in range(9)]))

        # cool (below the packet's own mean) -> dark (at it) -> warm (above)
        stops = [(0.00, (0x7d, 0xd3, 0xff)), (0.25, (0x2f, 0x7d, 0xc4)),
                 (0.50, (0x14, 0x14, 0x18)), (0.75, (0xc9, 0x6a, 0x1e)),
                 (1.00, (0xff, 0xc8, 0x66))]
        xs = np.linspace(0, 1, 256)
        self.lut = np.stack(
            [np.interp(xs, [p for p, _ in stops], [c[k] for _, c in stops])
             for k in range(3)], axis=1).astype(np.uint8)

        self.setWindowTitle('CSI + camera, live')
        self.resize(1720, 940)
        self.setStyleSheet(f'QWidget {{ background: {SURFACE}; color: {INK}; }}')
        root = QHBoxLayout(self)

        # ---- camera ----
        left = QVBoxLayout()
        self.cam_label = QLabel(alignment=QtCore.Qt.AlignCenter)
        self.cam_label.setMinimumWidth(880)
        cap = QLabel(f'RealSense colour · {cam.w}x{cam.h} @ {cam.fps:g} fps')
        cap.setStyleSheet(f'color: {MUTED}; font-size: 13px;')
        left.addWidget(cap)
        left.addWidget(self.cam_label, 1)
        self.stats = QLabel('')
        self.stats.setStyleSheet(f'color: {MUTED}; font-size: 12px;')
        left.addWidget(self.stats)
        root.addLayout(left, 3)

        # ---- CSI grid ----
        right = QVBoxLayout()
        head = QLabel(f'Per-link CSI · every subcarrier · last {NPKT} packets, '
                      f'normalised per packet')
        head.setStyleSheet(f'color: {INK}; font-size: 15px; font-weight: bold;')
        right.addWidget(head)

        # ---- who transmits: round-robin, or one board pinned ----
        self.by_label = {LABEL.get(m[-5:], m): m for m in self.macs}
        bar = QHBoxLayout()
        bar.addWidget(QLabel('Transmitter:'))
        self.tx_buttons = {}
        group = QButtonGroup(self)
        group.setExclusive(True)
        for name in ['round-robin'] + sorted(self.by_label):
            b = QPushButton(name)
            b.setCheckable(True)
            b.setStyleSheet(
                'QPushButton { padding: 6px 14px; font-size: 13px; border: 1px solid #444; '
                f'border-radius: 4px; color: {INK}; background: #2a2a2d; }}'
                'QPushButton:checked { background: #2a78d6; border-color: #2a78d6; '
                'color: white; font-weight: bold; }')
            b.clicked.connect(lambda _c, n=name: self.set_tx(n))
            group.addButton(b)
            bar.addWidget(b)
            self.tx_buttons[name] = b
        bar.addStretch(1)
        right.addLayout(bar)

        self.role = QLabel('')
        self.role.setStyleSheet(f'color: {INK}; font-size: 14px;')
        right.addWidget(self.role)

        # ---- capture ----
        self.rec = None            # None when idle; a dict of state while recording
        self.proto = None          # scripted protocol state, None when not running
        self.cue = None
        cap_bar = QHBoxLayout()
        cap_bar.addWidget(QLabel('Capture:'))
        self.prefix_edit = QLineEdit(args.prefix)
        self.prefix_edit.setStyleSheet(
            f'padding: 6px; font-size: 13px; color: {INK}; background: #2a2a2d; '
            'border: 1px solid #444; border-radius: 4px;')
        self.prefix_edit.setMinimumWidth(160)
        cap_bar.addWidget(self.prefix_edit)
        self.rec_btn = QPushButton('● REC')
        self.rec_btn.setMinimumWidth(130)
        self.rec_btn.clicked.connect(self.toggle_record)
        cap_bar.addWidget(self.rec_btn)
        self.proto_btn = QPushButton('▶ RUN PROTOCOL')
        self.proto_btn.setStyleSheet(
            'QPushButton { padding: 6px 14px; font-size: 13px; border: 1px solid #444; '
            f'border-radius: 4px; color: {INK}; background: #2a2a2d; }}')
        self.proto_btn.clicked.connect(self.toggle_protocol)
        self.proto_btn.setEnabled(bool(args.protocol))
        cap_bar.addWidget(self.proto_btn)
        pick = QPushButton('Protocol…')
        pick.setStyleSheet(
            'QPushButton { padding: 6px 12px; font-size: 13px; border: 1px solid #444; '
            f'border-radius: 4px; color: {INK}; background: #2a2a2d; }}')
        pick.clicked.connect(self.pick_protocol)
        cap_bar.addWidget(pick)
        cap_bar.addStretch(1)
        right.addLayout(cap_bar)
        self.style_rec_button()

        self.rec_status = QLabel('')
        self.proto_label = QLabel('')
        self.proto_label.setStyleSheet(f'color: {MUTED}; font-size: 12px;')
        self.rec_status.setStyleSheet(f'color: {MUTED}; font-size: 13px;')
        self.health = QLabel('')
        self.health.setStyleSheet('font-size: 14px; font-weight: bold;')
        self.health.setWordWrap(True)
        right.addWidget(self.health)
        right.addWidget(self.proto_label)
        right.addWidget(self.rec_status)
        self.show_protocol()

        self.grid = QGridLayout()
        self.grid.setSpacing(6)
        right.addLayout(self.grid, 1)
        self.note = QLabel('')
        self.note.setStyleSheet(f'color: {MUTED}; font-size: 12px;')
        self.note.setWordWrap(True)
        right.addWidget(self.note)
        root.addLayout(right, 2)

        self.panels = {}
        rxs = [m for m in self.macs]
        for r, tx in enumerate(self.macs):
            for c, rx in enumerate([m for m in rxs if m != tx]):
                self.panels[(tx, rx)] = self.make_panel(link_name(tx, rx), r, c)
                self.buf[(tx, rx)] = deque()

        # Selection lives in one place; the radio thread reads it every round, so a
        # click takes effect on the next one rather than needing a restart.
        start = args.tx if args.tx in self.by_label else 'round-robin'
        self.tx_sel = start
        self.tx_buttons[start].setChecked(True)

        for m in self.macs:
            boards[m].reset_input_buffer()
            threading.Thread(target=self.reader, args=(m,), daemon=True).start()
        threading.Thread(target=self.radio, daemon=True).start()
        self.cam.start()
        threading.Thread(target=self.grabber, daemon=True).start()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.on_tick)
        self.timer.start(int(1000 / args.refresh))

    def make_panel(self, title, r, c):
        box = QVBoxLayout()
        lab = QLabel(title)
        lab.setStyleSheet(f'color: {INK}; font-size: 12px; font-weight: bold;')
        gl = pg.GraphicsLayoutWidget()
        gl.setBackground(SURFACE)
        vb = gl.addViewBox()
        vb.setMouseEnabled(False, False)
        vb.invertY(False)
        img = pg.ImageItem()
        vb.addItem(img)
        vb.setAspectLocked(False)
        gl.setMinimumHeight(120)
        w = QWidget()
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(lab)
        box.addWidget(gl, 1)
        w.setLayout(box)
        self.grid.addWidget(w, r, c)
        return img, vb, lab

    # ---- inputs ----

    def reader(self, rx):
        """One board's byte stream. Nothing short of the port closing may end this
        loop: if it exits, that board goes silent for the rest of the session while
        looking perfectly healthy on the wire, and only a restart brings it back.
        A _csv.Error escaping the old line parser did exactly that -- it is not a
        ValueError, so the handler missed it -- and it is the real reason boards
        "died" twice. Everything after the read is contained for the same reason.
        """
        ser = self.boards[rx]
        st = CsiStream()
        while not self.stop.is_set():
            try:
                data = ser.read(ser.in_waiting or 1)
            except (serial.SerialException, OSError):
                return
            except Exception:
                self.read_errors += 1
                continue
            if not data:
                continue
            try:
                recs, _lines = st.feed(data)
                now = time.time()
                for tx, lts, rssi, amp, clipped in recs:
                    key = (tx, rx)
                    self.last_seen[rx] = now
                    self.clipped += clipped
                    rec = self.rec   # single read: stop_record may clear it mid-loop
                    if rec is not None:
                        rec['recs'][rx].append((now, tx, lts, rssi, amp))
                    with self.lock:
                        if key not in self.buf:
                            continue
                        # Host arrival drives the span readout; the board's hardware
                        # receive time rides along because it is what the offline
                        # alignment uses, matching what capture.py writes.
                        self.buf[key].append((now, lts / 1e6, amp))
                        cut = now - self.args.hold
                        while self.buf[key] and self.buf[key][0][0] < cut:
                            self.buf[key].popleft()
            except Exception:
                self.read_errors += 1
        self.read_errors += st.bad

    # ---- capture ----

    def style_rec_button(self):
        on = self.rec is not None
        self.rec_btn.setText('■ STOP' if on else '● REC')
        self.rec_btn.setStyleSheet(
            'QPushButton { padding: 6px 14px; font-size: 13px; font-weight: bold; '
            'border-radius: 4px; border: 1px solid %s; background: %s; color: %s; }'
            % (('#d03b3b', '#d03b3b', 'white') if on else ('#444', '#2a2a2d', INK)))

    def toggle_record(self):
        if self.rec is None:
            self.start_record()
        else:
            self.stop_record()

    def silent_boards(self, now):
        """Boards that should be receiving but have not for --silent-after seconds.
        The pinned transmitter is excluded: it is not supposed to receive at all."""
        active_tx = None if self.tx_sel == 'round-robin' else self.by_label[self.tx_sel]
        out = []
        for m in self.macs:
            if m == active_tx:
                continue
            if now - self.last_seen.get(m, 0) > self.args.silent_after:
                out.append(LABEL.get(m[-5:], m[-5:]))
        return sorted(out)

    def session_dir(self):
        """Takes from a protocol go in <outdir>/<yaml stem>/, so one session's files
        stay together instead of scattering loose names across the data folder."""
        return pathlib.Path(self.args.protocol).stem

    def show_protocol(self):
        if self.args.protocol:
            self.proto_label.setText(
                f'protocol <b>{os.path.basename(self.args.protocol)}</b> → '
                f'{self.args.outdir}/{self.session_dir()}/')
        else:
            self.proto_label.setText('no protocol chosen — press "Protocol…" to pick one')
        self.rec_status.setText(f'idle · manual takes save to {self.args.outdir}/')

    def pick_protocol(self):
        if self.proto is not None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, 'Choose a capture protocol', self.args.protocol or self.args.projdir,
            'Protocol YAML (*.yaml *.yml);;All files (*)')
        if not path:
            return
        try:
            _timing, takes = load_protocol(path)        # fail here, not mid-session
        except (OSError, ValueError) as e:
            self.proto_label.setText(f'<span style="color:#d03b3b">{e}</span>')
            return
        self.args.protocol = path
        self.proto_btn.setEnabled(True)
        self.show_protocol()
        self.rec_status.setText(f'{len(takes)} takes loaded: '
                                + ', '.join(n for n, _ in takes[:6])
                                + (' …' if len(takes) > 6 else ''))

    def toggle_protocol(self):
        if self.proto is None:
            self.start_protocol()
        else:
            self.abort_protocol('stopped by operator')

    def start_protocol(self):
        silent = self.silent_boards(time.time())
        if silent:
            self.rec_status.setText(
                f'<span style="color:#ff5555">refusing to start: no data from '
                f'{", ".join(silent)}. Fix it first — a protocol run with a silent '
                f'board produces takes missing every link into it.</span>')
            return
        try:
            timing, takes = load_protocol(self.args.protocol)
        except (OSError, ValueError) as e:
            self.rec_status.setText(f'<span style="color:#d03b3b">protocol: {e}</span>')
            return
        self.proto = dict(timing=timing, takes=takes, i=0, phase=LEAD,
                          until=time.time() + timing['lead_in'])
        self.cue = Cue()
        self.cue.show()
        self.cue.raise_()
        self.proto_btn.setText('■ ABORT')
        self.rec_btn.setEnabled(False)
        self.prefix_edit.setEnabled(False)
        self.proto_tick()

    def abort_protocol(self, why, keep_cue=False):
        if self.rec is not None:
            self.stop_record()          # never leave a half-written take on disk
        self.proto = None
        if self.cue is not None and not keep_cue:
            self.cue.close()
            self.cue = None
        self.proto_btn.setText('▶ RUN PROTOCOL')
        self.rec_btn.setEnabled(True)
        self.prefix_edit.setEnabled(True)
        self.rec_status.setText(f'protocol {why}')

    def proto_tick(self):
        """Advance lead-in -> record -> gap -> next take. Driven from the same 30 Hz
        timer as the display, so the cue card and the waterfalls never disagree."""
        p = self.proto
        if p is None:
            return
        now = time.time()

        # Stop the moment a board goes quiet. Refusing to *start* is not enough: a
        # board that dies seconds into a run leaves every remaining take missing all
        # links into it, and the subject is standing in the array unable to see the
        # screen. One ruined take beats twenty-three.
        gone = self.silent_boards(now)
        if gone:
            self.cue.show_state(
                REC, 'CAPTURE STOPPED', f'NO DATA FROM {", ".join(gone)}', None,
                'Its serial link died mid-run, not its radio. '
                'Replug that board and start the protocol again.')
            self.abort_protocol(f'ABORTED at take {p["i"] + 1}/{len(p["takes"])} '
                                f'— {", ".join(gone)} went silent', keep_cue=True)
            return
        t = p['timing']
        name, instruction = p['takes'][p['i']]
        left = p['until'] - now

        if left <= 0:
            if p['phase'] == LEAD:
                self.start_record(f'{self.session_dir()}/{name}')
                p['phase'], p['until'] = REC, now + t['duration']
            elif p['phase'] == REC:
                self.stop_record()
                p['phase'], p['until'] = GAP, now + t['gap']
            else:
                p['i'] += 1
                if p['i'] >= len(p['takes']):
                    return self.abort_protocol(
                        f'finished — {len(p["takes"])} takes written')
                p['phase'], p['until'] = LEAD, now + t['lead_in']
            name, instruction = p['takes'][p['i']]
            left = p['until'] - now

        step = f'take {p["i"] + 1} of {len(p["takes"])}'
        if t['repeats'] > 1:
            step += f'   ·   round {p["i"] // int(t["nbase"]) + 1}/{int(t["repeats"])}'
        step += f'   ·   {name}'
        nxt = (p['takes'][p['i'] + 1][0] if p['i'] + 1 < len(p['takes']) else 'finish')
        if p['phase'] == LEAD:
            self.cue.show_state(LEAD, step, instruction, left, 'GET READY — recording starts at 0')
        elif p['phase'] == REC:
            self.cue.show_state(REC, step, instruction, left, '● RECORDING — hold')
        else:
            self.cue.show_state(GAP, step, 'REST', left, f'next: {nxt}')

    def start_record(self, name=None):
        name = (name or self.prefix_edit.text().strip() or 'run')
        prefix = resolve_prefix(name, self.args.outdir)
        own = owner_of(prefix)
        # t0 is stamped before any thread can append, so every timestamp written is
        # relative to a single instant rather than to whenever a thread first woke.
        self.rec = dict(prefix=prefix, name=name, t0=time.time(), own=own,
                        recs={m: [] for m in self.macs}, frames=[], idx=0,
                        jpeg=JpegWriter(f'{prefix}_frames', self.args.quality,
                                        self.args.encoders, self.args.queue, own))
        self.prefix_edit.setEnabled(False)
        self.style_rec_button()

    def stop_record(self):
        rec, self.rec = self.rec, None      # readers see None immediately and stop
        self.prefix_edit.setEnabled(False)
        self.rec_btn.setEnabled(False)
        self.rec_status.setText('writing …')
        QApplication.processEvents()
        # A reader that read `rec` just before the clear can still be mid-append, and
        # a list that grows while write_capture iterates it raises. Wait past the
        # 0.3 s serial timeout, then snapshot, so the writer sees a frozen copy.
        time.sleep(0.35)
        recs = {k: list(v) for k, v in rec['recs'].items()}
        frames = list(rec['frames'])
        rec['jpeg'].close()
        meta = dict(t0_epoch=rec['t0'], mode=self.tx_sel,
                    round_duration=self.args.round_duration,
                    width=self.cam.w, height=self.cam.h, fps_requested=self.cam.fps,
                    frame_dir=f'{rec["prefix"]}_frames', jpeg_quality=self.args.quality,
                    boards={m: LABEL.get(m[-5:], '?') for m in self.macs},
                    driver_monotonic_ts=bool(self.cam.monotonic),
                    dropped_encode=rec['jpeg'].dropped)
        path, report, ft = write_capture(rec['prefix'], recs, frames,
                                         rec['t0'], meta, rec['own'])
        dur = ft[-1] - ft[0] if len(ft) > 1 else 0.0
        npkt = sum(len(v) for v in recs.values())
        msg = (f'wrote <b>{path}</b> · {len(ft)} frames '
               f'({len(ft) / max(dur, 1e-9):.1f} fps) · {npkt} packets · {dur:.1f}s')
        if rec['jpeg'].dropped:
            msg += f' · <span style="color:#d03b3b">{rec["jpeg"].dropped} frames unencoded</span>'
        self.rec_status.setText(msg)
        print(f'wrote {path} and {rec["prefix"]}_frames/  '
              f'({len(ft)} frames, {npkt} packets, {dur:.1f}s)', flush=True)
        self.prefix_edit.setEnabled(True)
        self.rec_btn.setEnabled(True)
        self.style_rec_button()

    def set_tx(self, name):
        """Retarget the token. Buffers are dropped because they belong to the old
        arrangement -- otherwise panels for links that just went silent would keep
        showing their last packets as though they were still live."""
        self.tx_sel = name
        with self.lock:
            for k in self.buf:
                self.buf[k].clear()

    def issue(self, tx):
        """One board takes the token, everyone else listens for it."""
        try:
            self.boards[tx].write(b'TX\n')
            for m in self.macs:
                if m != tx:
                    self.boards[m].write(f'RX {tx.replace(":", "")}\n'.encode())
        except (serial.SerialException, OSError):
            return False
        return True

    def radio(self):
        i, last = 0, None
        while not self.stop.is_set():
            sel = self.tx_sel
            if sel == 'round-robin':
                tx = self.macs[i % len(self.macs)]
                i += 1
                if not self.issue(tx):
                    return
                last = None
                self.stop.wait(self.args.round_duration)
            else:
                # Pinned: issue only on change. Re-sending TX to the board that
                # already holds it restarts its ping timer and emits another
                # ROLE_TX line for no benefit.
                if sel != last:
                    if not self.issue(self.by_label[sel]):
                        return
                    last = sel
                self.stop.wait(0.1)

    def grabber(self):
        while not self.stop.is_set():
            try:
                got = self.cam.read()
            except OSError:
                return
            if got is None:
                continue
            seq, ts, buf = got
            rec = self.rec
            if rec is not None:
                # Same clock discipline as capture_synced: the driver's DMA-completion
                # timestamp, not the moment this thread got round to looking.
                wall = ts + self.mono_offset if self.cam.monotonic else time.time()
                rec['frames'].append((rec['idx'], seq, wall))
                rec['jpeg'].submit(rec['idx'], buf, self.cam.w, self.cam.h)
                rec['idx'] += 1
            rgb = yuyv_to_rgb(buf, self.cam.w, self.cam.h)
            with self.lock:
                self.frame = rgb
                self.frame_times.append(time.time())

    # ---- render ----

    def last_packets(self, key, now):
        """[n_sub, NPKT] of the last NPKT packets, newest in the right column.

        Simply the last NPKT packets, one per column, newest on the right. No slot
        grid and no attempt to represent elapsed time, because this display is a
        monitor rather than a measurement: alignment between CSI and video is done
        offline from the recorded timestamps, where UART delay and jitter can be
        handled properly. Trying to reconstruct timing live only ever produced
        artefacts -- a fixed grid blanked cells that jitter had merely displaced,
        and cumulative gap rounding stretched time.

        Still no integration: a column is one packet exactly as it arrived.

        Returns the matrix, how many columns are populated, and the wall-clock span
        those packets cover -- with time off the axis, the span is the only thing
        that still shows a link slowing down.
        """
        with self.lock:
            items = list(self.buf.get(key, ()))[-NPKT:]
        if not items:
            return None, 0, 0.0
        n = len(items[-1][2])
        items = [it for it in items if len(it[2]) == n]
        if not items:
            return None, 0, 0.0

        out = np.full((n, NPKT), np.nan, dtype=np.float32)
        block = np.stack([it[2] for it in items], axis=1).astype(np.float32)
        # Per-packet normalisation, display only -- what is recorded stays raw.
        # Each column is z-scored across its own subcarriers, which removes that
        # packet's overall level (AGC, distance, per-board gain) and leaves the
        # frequency-selective *shape*, which is where the body's effect lives. It
        # also makes panels comparable: without it a loud link is bright and a quiet
        # one is dark regardless of what either is doing.
        # Guard bands and DC sit at ~0 all the time, so they are kept out of the
        # mean/sigma -- left in they drag every packet's statistics. They are still
        # *drawn*, at whatever z-score that leaves them (hard against the cool end),
        # so all 192 rows are on screen and the guard structure is visible rather
        # than being a hole you have to know about.
        dead = block.mean(axis=1) < 1.0
        z = np.full_like(block, np.nan)
        for b0 in range(0, n, SUB_BLOCK):
            sl = slice(b0, min(b0 + SUB_BLOCK, n))
            live = ~dead[sl]
            if live.sum() < 4:
                continue
            seg = block[sl]
            mu = seg[live].mean(axis=0, keepdims=True)
            sd = np.maximum(seg[live].std(axis=0, keepdims=True), 1e-3)
            z[sl] = (seg - mu) / sd
        out[:, NPKT - block.shape[1]:] = z
        span = (items[-1][0] - items[0][0]) if len(items) > 1 else 0.0
        return out, block.shape[1], span

    def to_rgba(self, M):
        """Colour-map z-scores on a fixed +/- sigma range.

        Diverging and dark-centred, because after per-packet normalisation zero is
        meaningful (this subcarrier sits at the packet's own average) and the
        interesting cells are the deviations -- so they are the bright ones. The
        range is fixed, never fitted to the current frame: a scale that rescales
        itself makes a quiet moment look identical to a loud one.

        Columns are only unfilled before 20 packets have arrived; those are black
        rather than transparent, which showed the panel background and read as a
        real value rather than "nothing yet".
        """
        good = np.isfinite(M)
        lim = self.args.sigma
        norm = np.zeros(M.shape, dtype=np.float32)
        np.divide(M + lim, 2 * lim, out=norm, where=good)
        idx = np.clip(norm * 255, 0, 255).astype(np.uint8)
        rgba = np.zeros(M.shape + (4,), dtype=np.uint8)
        rgba[..., :3] = np.where(good[..., None], self.lut[idx], 0)
        rgba[..., 3] = 255
        return rgba

    def on_tick(self):
        now = time.time()

        # Camera first and unconditionally, independent of the CSI panels.
        with self.lock:
            frame = self.frame
            ftimes = list(self.frame_times)
        if frame is not None:
            h, w, _ = frame.shape
            # Keep a name on the buffer: QImage does not copy, and letting the
            # temporary die before scaled() renders torn or empty frames.
            raw = frame.tobytes()
            qi = QImage(raw, w, h, 3 * w, QImage.Format_RGB888)
            self.cam_label.setPixmap(QPixmap.fromImage(qi).scaled(
                self.cam_label.width(), self.cam_label.height(),
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

        sel = self.tx_sel
        active_tx = None if sel == 'round-robin' else self.by_label[sel]

        fills, gots, rates = [], [], []
        for key, (img, vb, lab) in self.panels.items():
            live = active_tx is None or key[0] == active_tx
            lab.setStyleSheet(f'color: {INK if live else "#4a4a4c"}; font-size: 12px; '
                              f'font-weight: {"bold" if live else "normal"};')
            M, ncol, span = self.last_packets(key, now)
            if live:
                fills.append(ncol)
                gots.append(span)
                with self.lock:
                    rates.append(len(self.buf[key]) / max(self.args.hold, 1e-9))
            if M is None:
                img.clear()
                continue
            img.setImage(self.to_rgba(M), autoLevels=False)
            vb.setRange(xRange=(0, NPKT), yRange=(0, M.shape[0]), padding=0)

        if active_tx is None:
            self.role.setText('<b>Round-robin</b> — every board takes the token in turn; '
                              'all 12 links carry data')
        else:
            rx = ', '.join(sorted(l for l in self.by_label if l != sel))
            self.role.setText(
                f'<b>TX = {sel}</b> &nbsp;·&nbsp; RX = {rx} &nbsp;·&nbsp; '
                f'only the {sel}→* row can carry data; the other 9 panels are '
                f'dimmed because those links do not exist right now. '
                f'<span style="color:{MUTED}">Boards show it too: {sel} blue, the rest red.</span>')

        silent = self.silent_boards(now)
        if silent:
            self.health.setText(
                '<span style="color:#ff5555">⚠ NO DATA from '
                + ', '.join(silent) + f' for &gt;{self.args.silent_after:.0f}s — '
                'its serial link is down, not its radio. Replug it.</span>')
        else:
            self.health.setText('')

        if self.proto is not None:
            self.proto_tick()

        rec = self.rec
        if rec is not None:
            el = now - rec['t0']
            npkt = sum(len(v) for v in rec['recs'].values())
            drop = rec['jpeg'].dropped
            self.rec_status.setText(
                f'<span style="color:#e04b4b">● RECORDING</span> '
                f'<b>{rec["name"]}</b> · {el:5.1f}s · {len(rec["frames"])} frames · '
                f'{npkt} packets'
                + (f' · <span style="color:#d03b3b">{drop} unencoded</span>' if drop else ''))

        cam_fps = 0.0
        if len(ftimes) > 2:
            cam_fps = (len(ftimes) - 1) / max(ftimes[-1] - ftimes[0], 1e-9)
        self.stats.setText(
            f'camera {cam_fps:4.1f} fps   ·   per-packet z-score, fixed '
            f'±{self.args.sigma:g}σ   ·   refresh {self.args.refresh} Hz'
            + (f'   ·   {self.read_errors} bad frames' if self.read_errors else '')
            + (f'   ·   <span style="color:#ff5555">{self.clipped} clipped</span>'
               if self.clipped else ''))
        mean_cols = float(np.mean(fills)) if fills else 0.0
        live_spans = [x for x in gots if x > 0]
        mean_span = float(np.mean(live_spans)) if live_spans else 0.0
        mean_rate = float(np.mean(rates)) if rates else 0.0
        self.note.setText(
            f'<b>{mean_cols:.0f}/{NPKT}</b> columns · these {NPKT} packets span '
            f'<b>{mean_span * 1000:.0f} ms</b> · ~{mean_rate:.0f} pkt/s per link · '
            f'z-scored within each {SUB_BLOCK}-wide CSI field, guard bands excluded '
            f'from the statistics but still drawn. '
            + (f'Round-robin: with the token shared across {len(self.panels)} links '
               f'each is sampled every {self.args.round_duration * len(self.macs) * 1000:.0f} ms, '
               'so the same 20 columns cover a much longer span. Timestamps are what '
               'align this to video, not the columns.'
               if self.tx_sel == 'round-robin' else
               f'Pinned TX: {len(self.macs) - 1} links at the full ping rate; '
               'the other 9 links are idle.'))

    def closeEvent(self, ev):
        if self.proto is not None:
            self.abort_protocol('window closed')
        # Never drop a run on the floor because the window was closed.
        if self.rec is not None:
            self.stop_record()
        self.stop.set()
        time.sleep(0.35)
        try:
            self.cam.close()
        except OSError:
            pass
        for m in self.macs:
            try:
                self.boards[m].write(b'IDENT OFF\n')
                self.boards[m].close()
            except (serial.SerialException, OSError):
                pass
        ev.accept()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--refresh', type=float, default=30.0, help='GUI redraw rate (Hz)')
    ap.add_argument('--tx', default='round-robin',
                    help='"round-robin", or a board label (A/B/C/D) to pin the '
                         'transmitter. Switchable in the GUI at any time.')
    ap.add_argument('--round-duration', type=float, default=0.05)
    ap.add_argument('--prefix', default='run', help='default capture name in the GUI')
    ap.add_argument('--outdir', default=None,
                    help='directory for captures (default: the repo data/ folder)')
    ap.add_argument('--silent-after', type=float, default=3.0,
                    help='seconds without a packet before a board is called silent')
    ap.add_argument('--protocol', default=None,
                    help='YAML listing takes to record back to back, in order')
    ap.add_argument('--protocol-dir', dest='projdir', default=None,
                    help='where the protocol picker starts browsing (default: project root)')
    ap.add_argument('--quality', type=int, default=85)
    ap.add_argument('--encoders', type=int, default=3)
    ap.add_argument('--queue', type=int, default=120)
    ap.add_argument('--hold', type=float, default=6.0,
                    help='seconds of packets kept per link (packet mode needs depth)')
    ap.add_argument('--sigma', type=float, default=2.5,
                    help='colour range in standard deviations of the per-packet z-score')
    args = ap.parse_args()
    if args.outdir is None:
        args.outdir = default_outdir()
    if args.projdir is None:
        args.projdir = default_projdir()

    dev = find_colour_node() if args.device == 'auto' else args.device
    if dev is None:
        raise SystemExit('no RealSense colour node found (camera plugged in? '
                         '"c 81:* rmw" in device_cgroup_rules?)')
    boards = discover()
    if len(boards) < 2:
        raise SystemExit(f'need >= 2 boards, found {len(boards)}')
    print(f'{len(boards)} boards: {[LABEL.get(m[-5:], m) for m in sorted(boards)]}',
          flush=True)
    print(f'camera {dev}', flush=True)

    cam = Camera(dev, args.width, args.height, args.fps)
    app = QApplication(sys.argv)
    w = Live(boards, cam, args)
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
