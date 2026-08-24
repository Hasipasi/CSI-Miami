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

from capture import (SUB_INDEX, CsiStream, JpegWriter, LABEL,
                            default_camera_device, default_outdir, derive_fields,
                            discover, field_bounds, open_camera, owner_of,
                            parse_scan_line, rank_channels, resolve_prefix,
                            write_capture)

pg.setConfigOptions(imageAxisOrder='row-major')

NPKT = 20             # columns on screen: the last 20 packets, one packet each
# ESP32 HT40 CSI is three 64-wide fields (LLTF | HT-LTF | STBC-HT-LTF) whose gains
# differ by ~5x. Normalising across all 192 at once mostly encodes *which field* a
# subcarrier belongs to and buries the within-field shape, so each is scaled alone.
SUB_BLOCK = 64
INK, SURFACE, MUTED = '#e9edf1', '#101218', '#8d93a0'
PANEL, BORDER, ACCENT = '#191c24', '#2b3040', '#3b82f6'


LEAD, REC, GAP = 'lead', 'rec', 'gap'
PHASE_BG = {LEAD: '#7f6000', REC: '#7f1010', GAP: '#33333a'}


def default_projdir():
    """Project root, for the protocol picker. Same reasoning as default_outdir:
    inside the container __file__ cannot identify the checkout, so the mount wins."""
    if os.path.isdir('/workspace/tools'):
        return '/workspace'
    return str(pathlib.Path(__file__).resolve().parents[1])


def build_lut(name):
    """256-entry RGB lookup table for the waterfalls.

    `jet` is the classic radio-waterfall rainbow (the MATLAB piecewise-linear
    formula, so no matplotlib import for one colormap). Know what it does to
    z-scored data: the scale midpoint -- "this subcarrier sits at the packet's own
    mean" -- lands on bright green rather than something recessive, so an utterly
    quiet link still looks lively. `diverge` is the previous map (cool -> dark ->
    warm), which keeps the midpoint dark so only deviations glow; it is the more
    honest map for spotting motion, and stays available for exactly that reason.
    """
    xs = np.linspace(0, 1, 256)
    if name == 'jet':
        def seg(c):
            return np.clip(1.5 - np.abs(4 * xs - c), 0, 1)
        rgb = np.stack([seg(3), seg(2), seg(1)], axis=1)
        return (rgb * 255).astype(np.uint8)
    stops = [(0.00, (0x7d, 0xd3, 0xff)), (0.25, (0x2f, 0x7d, 0xc4)),
             (0.50, (0x14, 0x14, 0x18)), (0.75, (0xc9, 0x6a, 0x1e)),
             (1.00, (0xff, 0xc8, 0x66))]
    return np.stack(
        [np.interp(xs, [p for p, _ in stops], [c[k] for _, c in stops])
         for k in range(3)], axis=1).astype(np.uint8)


class StatTile(QFrame):
    """One number with a small caption. The GUI's job is these numbers; prose about
    what they mean lives in the README."""

    def __init__(self, caption):
        super().__init__()
        self.setStyleSheet(
            'QFrame { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, '
            f'stop:0 #1e222c, stop:1 {PANEL}); '
            f'border: 1px solid {BORDER}; border-radius: 11px; }}')
        self.setMinimumWidth(112)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 8, 14, 8)
        lay.setSpacing(1)
        self.val = QLabel('—')
        self.cap = QLabel(caption.upper())
        self.cap.setStyleSheet(
            f'font-size: 9px; letter-spacing: 1px; color: {MUTED}; '
            'background: transparent; border: none;')
        self.set('—')
        lay.addWidget(self.val)
        lay.addWidget(self.cap)

    def set(self, text, color=None):
        self.val.setText(text)
        self.val.setStyleSheet(f'font-size: 24px; font-weight: bold; '
                               f'color: {color or INK}; background: transparent; '
                               'border: none;')


GOOD, WARN, BAD = '#34d399', '#fbbf24', '#f87171'


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
        # Whatever width the boards are actually sending: 192 amplitudes or 30 I/Q,
        # depending on the firmware they are running. Read off the wire rather than
        # assumed, so the caption cannot claim a layout the data does not have.
        self.n_sub = 0
        # Radio on/off. The boards stream ~86 KB/s each the whole time they are
        # running, which is noise on the bench and heat in the boards when nobody is
        # looking; this is the switch that shuts it up without killing the viewer.
        self.csi_on = True
        # Field-boundary markers, drawn once the wire says how wide a frame is.
        self._panel_fields = None   # (mode, field split) the panels were last built for
        # 'spectrum': amplitude on the y axis, one curve per packet with fading
        # trails -- the natural view at 30 subcarriers, where a waterfall is a thin
        # colour strip. 'waterfall': the classic time x subcarrier image.
        self.view_mode = 'spectrum'
        # One fixed scale for both views: the min/max boxes (and FIT) set the
        # spectrum's y axis AND the waterfall's colour map, so a value reads the
        # same everywhere and nothing rescales itself while you watch.
        self.wf_min = float(args.wf_min)
        self.wf_max = float(args.wf_max)
        # Per-link scale overrides, set by FIT: a near link and a far link differ
        # hugely in level, and one shared scale leaves the far panels flat while the
        # near ones clip. The boxes stay the global default; editing them clears
        # the per-link fits (explicit global intent).
        self.link_scale = {}
        # The observed amplitude range since start / last recalibrate: the reference
        # for choosing the fixed scale, shown beside the boxes.
        self.seen_min = None
        self.seen_max = None
        # Spectrum level lock: measured on this rig, 11.7 of the 12.2% per-packet
        # amplitude flutter is the WHOLE curve moving together (AGC re-selection the
        # Q8.8 compensation cannot fully undo, per-packet TX/PHY scaling); the shape
        # itself moves only 4.3%. Locking divides each packet by its own median and
        # re-anchors to the link's running level: the bounce goes, the shape stays.
        self.level_lock = False
        self.level_ref = {}
        # Running mean level per subcarrier, and the field split derived from it.
        # Measured rather than tabulated because the tabulated layout is wrong (see
        # derive_fields), and because it has to survive a bandwidth change at runtime.
        self.amp_mean = None
        self.fields = None
        self.fields_at = 0.0
        self.scan = None           # channel survey state, None when not scanning
        self.bw = args.bw          # what the boards are believed to be running
        self.sub_sel = 166         # firmware boot default (CONFIG_SUB_COUNT)
        self.rate_val = 243        # firmware boot default (CONFIG_SEND_FREQUENCY)
        # CLOCK_MONOTONIC to wall clock, measured once: the two drift far too slowly
        # to matter across a run, and re-measuring per frame would inject exactly the
        # scheduling jitter the driver timestamp exists to avoid.
        self.mono_offset = float(np.median([time.time() - time.monotonic()
                                            for _ in range(9)]))

        self.lut = build_lut(args.cmap)

        self.setWindowTitle('CSI + camera, live')
        # Size to the screen that is actually there, not to the rig PC's monitor;
        # minimums below are small enough that the whole window can be dragged down.
        scr = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1720, scr.width() - 40), min(940, scr.height() - 60))
        # One sheet themes everything; widgets carry no inline styles of their own,
        # which is also what makes the hover/press states below reach every control.
        self.setStyleSheet(f'''
            QWidget {{ background: {SURFACE}; color: {INK}; font-size: 13px; }}
            QPushButton {{ padding: 7px 16px; border-radius: 9px;
                           border: 1px solid {BORDER}; background: {PANEL}; }}
            QPushButton:hover {{ background: #232733; border-color: #3d4356; }}
            QPushButton:pressed {{ background: #0c0e13; }}
            QPushButton:checked {{ background: {ACCENT}; border-color: {ACCENT};
                                   color: white; font-weight: bold; }}
            QPushButton:disabled {{ color: {MUTED}; }}
            QLineEdit {{ padding: 7px 9px; background: {PANEL}; color: {INK};
                         border: 1px solid {BORDER}; border-radius: 9px; }}
            QLineEdit:focus {{ border-color: {ACCENT}; }}
        ''')
        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)

        # ---- camera ----
        left = QVBoxLayout()
        self.cam_label = QLabel(alignment=QtCore.Qt.AlignCenter)
        self.cam_label.setMinimumWidth(360)   # the video scales; the layout decides
        cap = QLabel(f'{cam.name} · {cam.w}x{cam.h} @ {cam.fps:g} fps')
        cap.setStyleSheet(f'color: {MUTED}; font-size: 13px;')
        left.addWidget(cap)
        left.addWidget(self.cam_label, 1)

        root.addLayout(left, 3)

        # ---- CSI grid ----
        right = QVBoxLayout()
        # Numbers, not prose: live health of the rig in one row.
        strip = QHBoxLayout()
        self.tiles = {}
        for key, caption in (('rate', 'per-link rate'), ('deliv', 'delivery'),
                             ('loss', 'radio loss'), ('gap', 'gap p99'),
                             ('wire', 'wire util'), ('drops', 'drops (board·host)'),
                             ('cam', 'camera')):
            t = StatTile(caption)
            strip.addWidget(t)
            self.tiles[key] = t
        strip.addStretch(1)
        right.addLayout(strip)
        self.tiles_at = 0.0
        self.board_stats = {}       # rx mac -> parsed STATS counters

        # ---- who transmits: round-robin, or one board pinned ----
        self.by_label = {LABEL.get(m[-5:], m): m for m in self.macs}
        bar = QHBoxLayout()
        bar.addWidget(QLabel('TX'))
        self.tx_buttons = {}
        group = QButtonGroup(self)
        group.setExclusive(True)
        for name in ['round-robin'] + sorted(self.by_label):
            b = QPushButton(name)
            b.setCheckable(True)
            b.clicked.connect(lambda _c, n=name: self.set_tx(n))
            group.addButton(b)
            bar.addWidget(b)
            self.tx_buttons[name] = b
        bar.addSpacing(20)
        self.csi_btn = QPushButton('■ STOP CSI')
        self.csi_btn.setMinimumWidth(140)
        self.csi_btn.clicked.connect(self.toggle_csi)
        bar.addWidget(self.csi_btn)
        self.scan_btn = QPushButton('⚡ FIND BEST CHANNEL')
        self.scan_btn.setMinimumWidth(190)
        self.scan_btn.clicked.connect(self.start_scan)
        bar.addWidget(self.scan_btn)
        self.bw_btn = QPushButton('')
        self.bw_btn.setMinimumWidth(120)
        self.bw_btn.clicked.connect(self.toggle_bw)
        self.bw_btn.setText(f'BW: {self.bw} MHz')
        bar.addWidget(self.bw_btn)
        bar.addStretch(1)
        right.addLayout(bar)

        # ---- radio detail: subcarrier count and ping rate ----
        # One row because they are one decision: the UART carries ~92 KB/s, so the
        # frame size (set by SUB) fixes the highest rate that fits. The hint shows the
        # measured-clean pairing for each width so the coupling is visible where the
        # buttons are, not just in the README.
        radio_bar = QHBoxLayout()
        radio_bar.addWidget(QLabel('SUB'))
        self.sub_buttons = {}
        sub_group = QButtonGroup(self)
        sub_group.setExclusive(True)
        for n, label in ((30, '30'), (114, '114'), (166, '166'), (0, '192 (all)')):
            b = QPushButton(label)
            b.setCheckable(True)
            b.clicked.connect(lambda _c, k=n: self.set_sub(k))
            sub_group.addButton(b)
            radio_bar.addWidget(b)
            self.sub_buttons[n] = b
        self.sub_buttons[166].setChecked(True)
        radio_bar.addSpacing(16)
        radio_bar.addWidget(QLabel('Hz'))
        self.rate_buttons = {}
        rate_group = QButtonGroup(self)
        rate_group.setExclusive(True)
        for key in ('low', 'mid', 'high', 'max'):
            b = QPushButton('')
            b.setCheckable(True)
            b.clicked.connect(lambda _c, k=key: self.pick_rate_preset(k))
            rate_group.addButton(b)
            radio_bar.addWidget(b)
            self.rate_buttons[key] = b
        # Boot RATE 243 sits nearest 50% of the boot config's ceiling; labels are
        # filled in once tx_sel exists (refresh_rate_buttons at init tail).
        self.rate_preset = 'mid'
        self.rate_buttons['mid'].setChecked(True)
        self.view_btn = QPushButton('View: spectrum')
        self.view_btn.clicked.connect(self.toggle_view)
        radio_bar.addWidget(self.view_btn)
        self.lock_btn = QPushButton('Plot: raw')
        self.lock_btn.clicked.connect(self.toggle_level_lock)
        radio_bar.addWidget(self.lock_btn)
        radio_bar.addSpacing(16)
        self.wfmin_edit = QLineEdit(f'{self.wf_min:g}')
        self.wfmin_edit.setFixedWidth(84)
        self.wfmin_edit.returnPressed.connect(self.set_wf_scale)
        radio_bar.addWidget(self.wfmin_edit)
        self.wfmax_edit = QLineEdit(f'{self.wf_max:g}')
        self.wfmax_edit.setFixedWidth(84)
        self.wfmax_edit.returnPressed.connect(self.set_wf_scale)
        radio_bar.addWidget(self.wfmax_edit)
        fit = QPushButton('FIT')
        fit.setMaximumWidth(50)
        fit.clicked.connect(self.fit_wf_scale)
        radio_bar.addWidget(fit)
        self.seen_label = QLabel('')
        self.seen_label.setStyleSheet(f'color: {MUTED}; font-size: 11px;')
        radio_bar.addWidget(self.seen_label)
        radio_bar.addStretch(1)
        right.addLayout(radio_bar)

        self.role = QLabel('')
        self.role.setStyleSheet(f'color: {INK}; font-size: 14px;')
        right.addWidget(self.role)
        self.scan_label = QLabel('')
        self.scan_label.setStyleSheet(
            f'color: {MUTED}; font-size: 12px; font-family: Menlo, monospace;')
        self.scan_label.setWordWrap(True)
        right.addWidget(self.scan_label)

        # ---- capture ----
        self.rec = None            # None when idle; a dict of state while recording
        self.proto = None          # scripted protocol state, None when not running
        self.cue = None
        cap_bar = QHBoxLayout()

        self.prefix_edit = QLineEdit(args.prefix)
        self.prefix_edit.setMinimumWidth(160)
        cap_bar.addWidget(self.prefix_edit)
        self.rec_btn = QPushButton('● REC')
        self.rec_btn.setMinimumWidth(130)
        self.rec_btn.clicked.connect(self.toggle_record)
        cap_bar.addWidget(self.rec_btn)
        self.proto_btn = QPushButton('▶ RUN PROTOCOL')
        self.proto_btn.clicked.connect(self.toggle_protocol)
        self.proto_btn.setEnabled(bool(args.protocol))
        cap_bar.addWidget(self.proto_btn)
        pick = QPushButton('Protocol…')
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
        self.grid.setSpacing(10)
        right.addLayout(self.grid, 1)
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
        self.refresh_rate_buttons()
        self.style_csi_button()

        for m in self.macs:
            boards[m].reset_input_buffer()
            threading.Thread(target=self.reader, args=(m,), daemon=True).start()
        threading.Thread(target=self.radio, daemon=True).start()
        self.cam.start()
        threading.Thread(target=self.grabber, daemon=True).start()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.on_tick)
        self.timer.start(int(1000 / args.refresh))

    def current_fields(self):
        """The field split to normalise and draw by: measured if enough packets have
        arrived, otherwise the layout fallback so the first repaint is not one block."""
        if self.fields:
            return self.fields
        return field_bounds(self.n_sub) if self.n_sub else [(0, 1)]

    NTRAIL = 9        # newest packet bold, eight ghosts behind it
    def set_wf_scale(self):
        """Fixed waterfall colour bounds, editable live. The seen-so-far range beside
        the boxes is the reference: set them just outside it and the whole dynamic
        range is spent on real signal."""
        try:
            lo = float(self.wfmin_edit.text())
            hi = float(self.wfmax_edit.text())
        except ValueError:
            lo, hi = self.wf_min, self.wf_max
        if hi <= lo:
            lo, hi = self.wf_min, self.wf_max
        self.wf_min, self.wf_max = lo, hi
        self.link_scale.clear()          # typed bounds mean: everyone, this scale
        self.wfmin_edit.setText(f'{lo:g}')
        self.wfmax_edit.setText(f'{hi:g}')

    def fit_wf_scale(self):
        """Snap the fixed colour bounds to the range actually seen. The signal here
        sits around a third of the way up a 0-150 scale, so its ~10% flutter spans
        about two colour steps -- which is why a wide fixed scale reads as "not
        moving". Fitting spends the whole colormap on the real range; the scale
        stays fixed afterwards until fitted or edited again."""
        # Per LINK, from the CURRENT buffers: each panel gets bounds hugged to its
        # own signal, so a 3 m link and a 4.2 m diagonal are both readable at once.
        with self.lock:
            snap = {k: [it[2] for it in v] for k, v in self.buf.items() if len(v) > 3}
        n = 0
        for k, vals in snap.items():
            A = np.concatenate(vals)
            lo, hi = float(np.floor(A.min())), float(np.ceil(A.max()))
            if hi <= lo:
                hi = lo + 1
            self.link_scale[k] = (lo, hi)
            n += 1
        if n:
            self.rec_status.setText(f'scale fitted per link ({n} links)')

    def toggle_level_lock(self):
        """raw: exactly what the boards deliver and the recording stores. norm: the
        postprocessed view -- per-packet level normalisation, the same treatment a
        model's preprocessing applies, so the plots show what training data will
        look like. Display only in both positions."""
        self.level_lock = not self.level_lock
        self.lock_btn.setText('Plot: norm' if self.level_lock else 'Plot: raw')

    def toggle_view(self):
        self.view_mode = 'waterfall' if self.view_mode == 'spectrum' else 'spectrum'
        self.view_btn.setText(f'View: {self.view_mode}')
        self._panel_fields = None            # force a rebuild on the next tick

    @staticmethod
    def sub_positions(n):
        """x coordinates for an n-wide frame: TRUE subcarrier numbers when the width
        matches a known table, so the spectrum's x axis is honest about the uneven
        spacing (30 spans 66-190; 166 jumps the guard bands). Row index otherwise."""
        idx = SUB_INDEX.get(n)
        return np.array(idx, float) if idx else np.arange(n, dtype=float)

    def rebuild_spectrum(self):
        def style(plt):
            plt.setMouseEnabled(False, False)
            plt.hideButtons()
            for ax in ('left', 'bottom'):
                plt.getAxis(ax).setPen(MUTED)
                plt.getAxis(ax).setTextPen(MUTED)

        def trails(plt, bright, base):
            out = []
            for i in range(self.NTRAIL):
                if i == self.NTRAIL - 1:
                    pen = pg.mkPen(bright, width=2)
                else:
                    a = int(30 + 130 * i / max(self.NTRAIL - 2, 1))
                    pen = pg.mkPen(base + (a,), width=1)
                out.append(plt.plot([], [], pen=pen))
            return out

        for pan in self.panels.values():
            pan['gl'].clear()
            pan['cells'] = []
            plt = pan['gl'].addPlot()
            style(plt)
            plt.setLabel('left', '|CSI|', color=MUTED)
            plt.setLabel('bottom', 'subcarrier', color=MUTED)
            pan['plot'] = plt
            pan['curves'] = trails(plt, '#7dd3ff', (125, 211, 255))

    def draw_spectrum(self, key, pan):
        with self.lock:
            items = list(self.buf.get(key, ()))[-self.NTRAIL:]
        if not items:
            for c in pan['curves']:
                c.setData([], [])
            return
        n = len(items[-1][2])
        items = [it for it in items if len(it[2]) == n]
        xs = self.sub_positions(n)

        if self.level_lock:
            ref = self.level_ref.get(key)
            med = float(np.median(items[-1][2])) or 1.0
            ref = med if ref is None else ref + 0.02 * (med - ref)
            self.level_ref[key] = ref
        k = len(items)
        for i, c in enumerate(pan['curves']):
            j = i - (self.NTRAIL - k)            # oldest ghost first, newest last
            if j < 0:
                c.setData([], [])
                continue
            y = items[j][2]
            if self.level_lock:
                m = float(np.median(y))
                if m > 0:
                    y = y * (ref / m)
            c.setData(xs, y)
        lo, hi = self.link_scale.get(key, (self.wf_min, self.wf_max))
        pan['plot'].setYRange(lo, hi, padding=0)
        pan['plot'].setXRange(xs[0], xs[-1], padding=0.01)

    def rebuild_panels(self, bounds):
        """One sub-plot per CSI field in every link panel, not one stacked image.

        The fields are the same channel through different internal gains, so
        stacking them in one image spends most of the colour range encoding which
        field a row is in. Separate plots give each field the full range, and the
        seam between plots is the boundary -- no marker line needed on top of the
        data. Built from the *measured* split (derive_fields), so it is two plots
        at HT40 and HT20 alike, and re-splits itself when SUB or BW changes.

        Highest-frequency field on top, and each plot's height proportional to how
        many subcarriers it holds, so the vertical axis still reads as frequency.
        """
        for pan in self.panels.values():
            pan['gl'].clear()
            pan['cells'] = []
            for i, (lo, hi) in enumerate(reversed(bounds)):
                if len(bounds) > 1:
                    pan['gl'].addLabel(f'{lo}-{hi - 1}', row=i, col=0,
                                       size='7pt', color=MUTED)
                vb = pan['gl'].addViewBox(row=i, col=1)
                vb.setMouseEnabled(False, False)
                vb.invertY(False)
                vb.setAspectLocked(False)
                img = pg.ImageItem()
                vb.addItem(img)
                pan['gl'].ci.layout.setRowStretchFactor(i, max(hi - lo, 1))
                pan['cells'].append((img, vb, lo, hi))

    def make_panel(self, title, r, c):
        box = QVBoxLayout()
        lab = QLabel(title)
        lab.setStyleSheet(f'color: {INK}; font-size: 12px; font-weight: bold;')
        gl = pg.GraphicsLayoutWidget()
        gl.setBackground(SURFACE)
        gl.ci.layout.setSpacing(4)
        gl.ci.setContentsMargins(0, 0, 0, 0)
        gl.setMinimumHeight(120)
        w = QWidget()
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(lab)
        box.addWidget(gl, 1)
        w.setLayout(box)
        self.grid.addWidget(w, r, c)
        return dict(gl=gl, lab=lab, cells=[])

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
                recs, lines = st.feed(data)
                for ln in lines:
                    t = ln.decode(errors='ignore').strip()
                    if t.startswith('STATS,'):
                        self.on_stats_line(rx, t)
                    elif self.scan is not None:
                        self.on_scan_line(rx, t)
                now = time.time()
                for tx, lts, rssi, amp, clipped, iq, gmeta in recs:
                    key = (tx, rx)
                    self.last_seen[rx] = now
                    self.clipped += clipped
                    self.n_sub = len(amp)
                    # Slow EMA: the field split is a property of the hardware layout,
                    # not of what is moving in the room, so it should not chase a
                    # person walking through the beam.
                    if amp.size:
                        lo, hi = float(amp.min()), float(amp.max())
                        if self.seen_max is None or hi > self.seen_max:
                            self.seen_max = hi
                        if self.seen_min is None or lo < self.seen_min:
                            self.seen_min = lo
                    if self.amp_mean is None or len(self.amp_mean) != len(amp):
                        self.amp_mean = amp.astype(np.float64)
                    else:
                        self.amp_mean *= 0.995
                        self.amp_mean += 0.005 * amp
                    rec = self.rec   # single read: stop_record may clear it mid-loop
                    if rec is not None:
                        rec['recs'][rx].append((now, tx, lts, rssi, amp, iq, gmeta))
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
            'QPushButton { padding: 9px 22px; font-size: 15px; font-weight: bold; '
            'border-radius: 11px; border: 1px solid %s; background: %s; color: %s; }'
            'QPushButton:hover { background: %s; }'
            % (('#ef4444', '#ef4444', 'white', '#f87171') if on
               else (ACCENT, PANEL, INK, '#232733')))

    def toggle_record(self):
        if self.rec is None:
            self.start_record()
        else:
            self.stop_record()

    def silent_boards(self, now):
        """Boards that should be receiving but have not for --silent-after seconds.
        The pinned transmitter is excluded: it is not supposed to receive at all."""
        if not self.csi_on or self.scan is not None:
            return []          # parked on purpose: not the failure this alarm is for
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
        if not self.csi_on:
            self.rec_status.setText(
                '<span style="color:#ff5555">CSI is stopped — press START CSI before '
                'running a protocol.</span>')
            return
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
        if not self.csi_on:
            self.rec_status.setText(
                '<span style="color:#ff5555">CSI is stopped — press START CSI first, '
                'or this take would contain video and no packets.</span>')
            return
        name = (name or self.prefix_edit.text().strip() or 'run')
        prefix = resolve_prefix(name, self.args.outdir)
        own = owner_of(prefix)
        # t0 is stamped before any thread can append, so every timestamp written is
        # relative to a single instant rather than to whenever a thread first woke.
        self.rec = dict(prefix=prefix, name=name, t0=time.time(), own=own,
                        recs={m: [] for m in self.macs}, frames=[], idx=0,
                        jpeg=JpegWriter(f'{prefix}_frames', self.args.quality,
                                        self.args.encoders, self.args.queue, own,
                                        self.cam.to_rgb))
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

    SCAN_DWELL_MS = 500        # per channel; 13 channels, so ~6.5 s a board

    def start_scan(self):
        """Survey 2.4 GHz on every board, then move the rig to the quietest block.

        The radio is parked first and the survey only issued a moment later: the
        radio thread may be mid-round, and a TX command landing while a board is
        scanning would sit in its UART buffer and take effect afterwards, leaving a
        board transmitting when the host thinks nothing is.
        """
        if self.scan is not None:
            return
        if self.rec is not None or self.proto is not None:
            self.rec_status.setText(
                '<span style="color:#ff5555">stop the recording first — a survey '
                'retunes the radio and takes the boards off channel.</span>')
            return
        self.csi_on = False
        self.style_csi_button()
        self.scan = dict(phase='parking', until=time.time() + 0.4,
                         data={m: {} for m in self.macs}, done=set())
        self.scan_btn.setEnabled(False)
        self.scan_label.setText('parking the radio …')

    def on_stats_line(self, board, line):
        """STATS,framedrops=N,textdrops=N,sendfail=N,heap=N — the board's own drop
        counters, the only place wire loss is distinguishable from radio loss."""
        out = {}
        for part in line.split(',')[1:]:
            if '=' in part:
                k, _, v = part.partition('=')
                try:
                    out[k] = int(v)
                except ValueError:
                    pass
        with self.lock:
            self.board_stats[board] = out

    def on_scan_line(self, board, line):
        """Called from a reader thread, so everything here takes the lock."""
        got = parse_scan_line(line)
        if got is None:
            return
        with self.lock:
            sc = self.scan
            if sc is None:
                return
            if got[0] == 'ch':
                _k, ch, pkts, nbytes, rssi_mean, rssi_max = got
                sc['data'].setdefault(board, {})[ch] = dict(
                    pkts=pkts, bytes=nbytes, rssi_mean=rssi_mean, rssi_max=rssi_max)
            else:
                sc['done'].add(board)

    def scan_tick(self, now):
        sc = self.scan
        if sc is None:
            return
        if sc['phase'] == 'parking':
            if now < sc['until']:
                return
            for m in self.macs:
                try:
                    self.boards[m].write(f'SCAN {self.SCAN_DWELL_MS}\n'.encode())
                except (serial.SerialException, OSError):
                    pass
            sc['phase'] = 'running'
            # 13 channels plus room for the command to be picked up and the replies
            # to drain; a board that never finishes must not wedge the button.
            sc['until'] = now + 13 * self.SCAN_DWELL_MS / 1000.0 + 6.0
            return

        with self.lock:
            done = len(sc['done'])
            counts = {m: len(v) for m, v in sc['data'].items()}
        if done < len(self.macs) and now < sc['until']:
            self.scan_label.setText(
                'scanning 1-13 · ' + ' · '.join(
                    f'{LABEL.get(m[-5:], m)} {counts.get(m, 0)}/13' for m in self.macs))
            return
        self.finish_scan(timed_out=done < len(self.macs))

    def finish_scan(self, timed_out):
        with self.lock:
            per_board, self.scan = self.scan['data'], None
        # Boards sit metres apart and hear different neighbours, so the survey is
        # pooled: bytes summed (total airtime the rig has to share) and RSSI taken at
        # its worst (whoever is loudest anywhere will desense that board).
        stats = {}
        for board, chans in per_board.items():
            for ch, d in chans.items():
                a = stats.setdefault(ch, dict(bytes=0, pkts=0, rssi_max=-128))
                a['bytes'] += d['bytes']
                a['pkts'] += d['pkts']
                a['rssi_max'] = max(a['rssi_max'], d['rssi_max'])
        self.scan_btn.setEnabled(True)
        # A survey re-ran by hand recalibrates even when the verdict is "stay put":
        # the boards were just retuned across 13 channels and back, and anything
        # accumulated through that sweep is polluted.
        self.recalibrate()
        if not stats:
            self.scan_label.setText(
                '<span style="color:#ff5555">survey returned nothing — do the boards '
                'have firmware with SCAN?</span>')
            return
        ranked = rank_channels(stats)
        best, best_bytes, best_rssi, span = ranked[0]
        cur = next((r for r in ranked if r[0] == self.args.channel), None)
        # One line on screen; the full per-channel table goes to stdout below, where
        # it can be scrolled back rather than crowding the controls.
        moved = best != self.args.channel
        if timed_out:
            verdict = '<b style="color:#fbbf24">survey timed out</b> — all boards on SCAN firmware?'
        elif cur and cur[1] > best_bytes:
            pct = 100.0 * (cur[1] - best_bytes) / max(cur[1], 1)
            verdict = (f'⚡ <b>ch{best}</b> · {pct:.0f}% less airtime than '
                       f'ch{self.args.channel}' + (' · moving' if moved else ''))
        else:
            verdict = f'⚡ <b>ch{self.args.channel}</b> already best'
        self.scan_label.setText(verdict)
        # Move only on a meaningful win: your last survey ranked the top five
        # placements within 1.5% of each other -- a statistical tie -- and moving
        # the whole rig on that is churn, not optimisation.
        if moved and cur and cur[1] > best_bytes:
            pct = 100.0 * (cur[1] - best_bytes) / max(cur[1], 1)
            if pct < 5.0:
                self.scan_label.setText(
                    f'⚡ <b>ch{self.args.channel}</b> kept — best alternative ch{best} '
                    f'is only {pct:.0f}% better (tie)')
                return
        print(f'channel survey: best={best} span={span[0]}-{span[-1]} bytes={best_bytes} '
              f'rssi={best_rssi}; ranked={[(r[0], r[1]) for r in ranked[:5]]}', flush=True)
        if best != self.args.channel:
            self.set_channel(best)

    def busy_reason(self):
        """Why the radio must not be reconfigured right now, or None if it may be."""
        if self.rec is not None or self.proto is not None:
            return ('stop the recording first — changing the frame layout mid-take '
                    'writes a take whose packets disagree about their own width.')
        if self.scan is not None:
            return 'wait for the survey to finish — it is retuning the radio itself.'
        return None

    # Measured-clean rate for each width at BW 40 (the knee minus ~10%; see NOTES.md).
    # Rate presets as fractions of the live wire ceiling. Buttons, not a type box:
    # the ceiling moves with SUB, BW and MODE, and a number typed for one config is
    # a saturation accident waiting in another. MAX is 90% of ceiling -- the
    # operating point every clean benchmark on this rig was measured at.
    RATE_FRACTIONS = {'low': 0.25, 'mid': 0.50, 'high': 0.75, 'max': 0.90}

    def frame_width(self, sel=None):
        """Subcarriers per frame for a SUB selection, accounting for HT20's fallback:
        the 30/166 tables index up to subcarrier 190, which does not exist in a
        128-wide HT20 report, so the firmware sends everything instead."""
        sel = self.sub_sel if sel is None else sel
        if self.bw == 20:
            return 128
        return {30: 30, 114: 114, 166: 166, 0: 192}[sel]

    def wire_ceiling(self, sel=None):
        """The hard frame rate the wire can carry at this width and MODE. A pinned
        receiver carries the full rate; in round-robin each receiver is silent while
        it holds the token, so its wire carries only (n-1)/n of the total -- measured:
        RATE 1600 round-robin on 2 boards sustains ~93% where pinned saturates at
        1071. The firmware itself caps RATE at 2000."""
        per_frame = 26 + 2 * self.frame_width(sel)             # v3: 24 hdr + 2 sum
        ceil = 92160 / per_frame
        if self.tx_sel == 'round-robin' and len(self.macs) > 1:
            ceil *= len(self.macs) / (len(self.macs) - 1)
        return min(int(ceil), 2000)

    def preset_hz(self, key):
        return max(1, int(self.RATE_FRACTIONS[key] * self.wire_ceiling()))

    def refresh_rate_buttons(self):
        for key, b in self.rate_buttons.items():
            b.setText(f'{key.upper()} {self.preset_hz(key)}')

    def pick_rate_preset(self, key):
        why = self.busy_reason()
        if why:
            self.rec_status.setText(f'<span style="color:#ff5555">{why}</span>')
            self.rate_buttons[self.rate_preset].setChecked(True)
            return
        self.rate_preset = key
        self.apply_rate_preset(announce=True)

    def apply_rate_preset(self, announce=False):
        """Re-resolve the active preset against the CURRENT ceiling and send it.
        Called on every SUB/BW/mode change, so MAX means max-for-this-config,
        always -- pinning a board that was at round-robin MAX would otherwise slam
        one receiver's wire at ~170%."""
        self.refresh_rate_buttons()
        if self.rec is not None or self.proto is not None:
            return                       # never retune mid-take
        hz = self.preset_hz(self.rate_preset)
        if self.send_rate(hz) and announce:
            self.rec_status.setText(
                f'rate → {hz} Hz ({self.rate_preset.upper()} · '
                f'ceiling {self.wire_ceiling()} Hz here)')

    def send_rate(self, hz):
        ok = True
        for m in self.macs:
            try:
                self.boards[m].write(f'RATE {hz}\n'.encode())
            except (serial.SerialException, OSError):
                ok = False
        if ok:
            self.rate_val = hz
        return ok

    def set_sub(self, n):
        """Switch every board's subcarrier count (0 = all 192) AND retune the rate.

        Coupled on purpose: the two share one UART budget, so a rate that was clean
        at 30 subcarriers saturates the wire at 166. Nothing corrupts past the knee
        -- measured -- but delivery collapses, and sustained print pressure is the
        documented path to a board whose command task wedges (fgets returning NULL
        under RX overrun). Leaving a stale rate armed after a width change is how
        that happens by accident."""
        why = self.busy_reason()
        if why:
            self.rec_status.setText(f'<span style="color:#ff5555">{why}</span>')
            self.sub_buttons[self.sub_sel if self.sub_sel in self.sub_buttons else 166].setChecked(True)
            return
        ok = True
        for m in self.macs:
            try:
                self.boards[m].write(f'SUB {n}\n'.encode())
            except (serial.SerialException, OSError):
                ok = False
        if ok:
            self.sub_sel = n
            self.apply_rate_preset()
            self.rec_status.setText(
                f'subcarriers → {self.frame_width()} · rate → {self.rate_val} Hz '
                f'({self.rate_preset.upper()} · ceiling {self.wire_ceiling()} Hz)')
            self.recalibrate()

    def toggle_bw(self):
        """Flip the whole rig between 20 and 40 MHz.

        Neither is simply better, which is why it is a button: 40 MHz carries more
        subcarriers, 20 MHz is narrow enough to escape a congested band entirely --
        measured here, HT20 on a quiet channel delivered 99.7% against HT40's 91%.
        A recording's frames say which was active (n_sub 128 against 166/192).
        """
        why = self.busy_reason()
        if why:
            self.rec_status.setText(f'<span style="color:#ff5555">{why}</span>')
            return
        want = 20 if self.bw == 40 else 40
        ok = True
        for m in self.macs:
            try:
                self.boards[m].write(f'BW {want}\n'.encode())
            except (serial.SerialException, OSError):
                ok = False
        if not ok:
            return
        self.bw = want
        self.bw_btn.setText(f'BW: {want} MHz')
        self.apply_rate_preset()
        # The subcarrier layout just changed: stale columns and a mean built on the
        # old width would both mislead, and the field split must be re-derived.
        self.recalibrate()

    def recalibrate(self):
        """Forget everything derived from past packets: waterfall columns, the running
        amplitude mean, and the field split measured from it. Called on every manual
        survey and radio retune, because all of it describes the *previous* channel --
        a mean carried across a channel change normalises the new channel's shape
        against the old one's until the EMA catches up, which looks like the display
        drifting for no reason."""
        with self.lock:
            for k in self.buf:
                self.buf[k].clear()
        self.amp_mean, self.fields = None, None
        self.fields_at = 0.0
        self.seen_min = self.seen_max = None
        self.link_scale.clear()

    def set_channel(self, ch):
        """Move every board together. A board left behind hears nothing and shows up
        as a dead serial link rather than as a channel mismatch."""
        ok = True
        for m in self.macs:
            try:
                self.boards[m].write(f'CHAN {ch}\n'.encode())
            except (serial.SerialException, OSError):
                ok = False
        if ok:
            self.args.channel = ch
            self.recalibrate()
        return ok

    def style_csi_button(self):
        on = self.csi_on
        self.csi_btn.setText('■ STOP CSI' if on else '▶ START CSI')
        self.csi_btn.setStyleSheet(
            'QPushButton { padding: 7px 16px; font-weight: bold; '
            'border-radius: 9px; border: 1px solid %s; background: %s; color: %s; }'
            % ((BORDER, PANEL, INK) if on else ('#10b981', '#10b981', 'white')))

    def toggle_csi(self):
        if self.csi_on:
            # Refused rather than allowed-and-warned: halting the radio mid-take
            # writes a take whose packet count silently stops partway, which looks
            # like a dead board later and is indistinguishable from one in the file.
            if self.rec is not None or self.proto is not None:
                self.rec_status.setText(
                    '<span style="color:#ff5555">stop the recording first — halting '
                    'the radio mid-take would leave a take with missing packets.</span>')
                return
            self.csi_on = False
            with self.lock:
                for k in self.buf:
                    self.buf[k].clear()
        else:
            self.csi_on = True
            # The boards cannot have been "seen" while they were parked, so clear the
            # clock too: otherwise resuming trips the silent-board alarm instantly.
            self.last_seen = {m: time.time() for m in self.macs}
        self.style_csi_button()

    def set_tx(self, name):
        """Retarget the token. Buffers are dropped because they belong to the old
        arrangement -- otherwise panels for links that just went silent would keep
        showing their last packets as though they were still live."""
        self.tx_sel = name
        with self.lock:
            for k in self.buf:
                self.buf[k].clear()
        # The ceiling depends on the mode (a pinned receiver carries the full rate,
        # round-robin receivers share it), so the preset re-resolves on every switch.
        self.apply_rate_preset(announce=True)

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

    def idle_all(self):
        """Park every board as a receiver of nobody, which stops all transmission.

        become_rx() stops that board's ping timer, so once no board holds the token
        nothing is sent and the UART goes quiet. The peer is the all-zero MAC because
        that is the firmware's own "matches nothing" filter (see become_tx), so a
        packet already in flight is discarded rather than logged against whatever the
        board happened to be listening to before.
        """
        try:
            for m in self.macs:
                self.boards[m].write(b'RX 000000000000\n')
        except (serial.SerialException, OSError):
            return False
        return True

    def radio(self):
        i, last = 0, None
        stats_at = 0.0
        while not self.stop.is_set():
            # Counters live on the boards; poll them onto the strip. Cheap: one
            # 6-byte command and one ~60-byte reply per board every 3 s.
            if time.time() - stats_at > 3.0:
                stats_at = time.time()
                for m in self.macs:
                    try:
                        self.boards[m].write(b'STATS\n')
                    except (serial.SerialException, OSError):
                        pass
            if not self.csi_on:
                # Issued once per stop, not per loop: repeating it would put a command
                # on the wire 10x a second for no effect.
                if last != 'off':
                    if not self.idle_all():
                        return
                    last = 'off'
                self.stop.wait(0.1)
                continue
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
            rgb = self.cam.to_rgb(buf, self.cam.w, self.cam.h)
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
        # Raw, gain-compensated amplitude — exactly what is recorded. The z-scoring
        # this used to do solved two problems that no longer exist: AGC level (now
        # compensated via the v2 gain header) and the cross-field gain step (fields
        # now get separate sub-plots with separate colour ceilings).
        out[:, NPKT - block.shape[1]:] = block
        span = (items[-1][0] - items[0][0]) if len(items) > 1 else 0.0
        return out, block.shape[1], span

    def on_tick(self):
        now = time.time()

        if self.scan is not None:
            self.scan_tick(now)

        # Re-derive occasionally rather than per frame: it is a scan over the whole
        # width, and the answer only moves when the radio is reconfigured.
        if self.amp_mean is not None and now - self.fields_at > 2.0:
            self.fields_at = now
            am = self.amp_mean.copy()
            if am.size and np.isfinite(am).all():
                self.fields = derive_fields(am)

        # The field split moves when SUB, the bandwidth, or the derivation changes;
        # the panels are rebuilt -- one sub-plot per field -- whenever it does.
        want = (self.view_mode,) + tuple(self.current_fields())
        if want != self._panel_fields:
            if self.view_mode == 'spectrum':
                self.rebuild_spectrum()
            else:
                self.rebuild_panels(self.current_fields())
            self._panel_fields = want

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
        for key, pan in self.panels.items():
            live = active_tx is None or key[0] == active_tx
            pan['lab'].setStyleSheet(
                f'color: {INK if live else "#4a4a4c"}; font-size: 12px; '
                f'font-weight: {"bold" if live else "normal"};')
            M, ncol, span = self.last_packets(key, now)
            if live:
                fills.append(ncol)
                gots.append(span)
                with self.lock:
                    rates.append(len(self.buf[key]) / max(self.args.hold, 1e-9))
            if self.view_mode == 'spectrum':
                self.draw_spectrum(key, pan)
                continue
            if M is None:
                for img, _vb, _lo, _hi in pan['cells']:
                    img.clear()
                continue
            # Absolute values, one colour ceiling per field cell: the fields differ
            # ~5x in gain, so a shared ceiling would leave the LLTF permanently dark.
            # Each ceiling moves on the same asymmetric-EMA + ladder as the spectrum
            # y axis, so colours mean the same thing from one packet to the next.
            n = M.shape[0]
            for ci, (img, vb, lo, hi) in enumerate(pan['cells']):
                lo, hi = min(lo, n), min(hi, n)
                if hi <= lo:
                    img.clear()
                    continue
                seg = M[lo:hi]
                good = np.isfinite(seg)
                if not good.any():
                    img.clear()
                    continue
                if self.level_lock:
                    # Same treatment as the spectrum: divide each column (packet) by
                    # its own median and re-anchor, so the 11.7% common-mode AGC
                    # bounce stops striping the image and what remains is shape. The
                    # reference is kept here too -- the spectrum's updater only runs
                    # in spectrum mode.
                    med = np.nanmedian(seg, axis=0, keepdims=True)
                    newest = float(med[0, -1]) if np.isfinite(med[0, -1]) else 0.0
                    if newest > 0:
                        ref = self.level_ref.get(key)
                        ref = newest if ref is None else ref + 0.02 * (newest - ref)
                        self.level_ref[key] = ref
                        seg = np.where(med > 0, seg * (ref / np.maximum(med, 1e-9)), seg)
                # Fixed scale, per link once FIT has run: near and far links sit
                # at very different levels, and each panel's colours span its own
                # fitted bounds (or the global boxes before any fit).
                slo, shi = self.link_scale.get(key, (self.wf_min, self.wf_max))
                span = max(shi - slo, 1e-6)
                idx = np.clip((np.nan_to_num(seg) - slo) / span * 255,
                              0, 255).astype(np.uint8)
                rgba = np.zeros(seg.shape + (4,), np.uint8)
                rgba[..., :3] = np.where(good[..., None], self.lut[idx], 0)
                rgba[..., 3] = 255
                img.setImage(rgba, autoLevels=False)
                vb.setRange(xRange=(0, NPKT), yRange=(0, hi - lo), padding=0)

        if not self.csi_on:
            self.role.setText('<b style="color:#ff9d3b">■ CSI stopped</b> — START CSI resumes')
        elif active_tx is None:
            self.role.setText(f'<b>round-robin</b> · {len(self.panels)} links')
        else:
            rx = ', '.join(sorted(l for l in self.by_label if l != sel))
            self.role.setText(f'<b>TX {sel} → {rx}</b> · {len(self.macs) - 1} of '
                              f'{len(self.panels)} links live')

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

        if now - self.tiles_at > 0.5:
            self.tiles_at = now
            self.update_tiles(now, ftimes)

    def update_tiles(self, now, ftimes):
        """The strip: everything is measured over the last --hold seconds of what the
        boards actually delivered, the same window the panels draw."""
        with self.lock:
            links = {k: (np.array([it[0] for it in v]),
                         np.array([it[1] for it in v]))
                     for k, v in self.buf.items() if len(v) > 3}
            stats = {m: dict(d) for m, d in self.board_stats.items()}
        # Rates over each link's actual data span, not the nominal window: after a
        # TX swap the buffers restart empty, and count-over-window ramps up for a
        # full --hold seconds -- pinning A at 1000 Hz read ~500 mid-ramp. Span-based
        # rates snap to the truth within a second.
        per = {}
        for k, (h, _l) in links.items():
            span = h[-1] - h[0]
            if span > 0.3:
                per[k] = (len(h) - 1) / span
        rates = list(per.values())
        total_rate = sum(rates)
        self.tiles['rate'].set(f'{np.mean(rates):.0f} Hz' if rates else '—')

        if self.csi_on and self.rate_val and rates:
            deliv = 100.0 * total_rate / self.rate_val
            self.tiles['deliv'].set(f'{min(deliv, 100):.1f}%',
                                    GOOD if deliv >= 97 else WARN if deliv >= 90 else BAD)
        else:
            self.tiles['deliv'].set('—')

        # Radio loss from the boards' hardware receive clocks: consecutive-arrival
        # steps measured in ping periods. Steps past 8 periods are the round-robin
        # blind gap (schedule, not loss) and are excluded.
        sent = got = 0
        for _h, lts in links.values():
            st = np.rint(np.diff(np.sort(lts)) * self.rate_val).astype(int)
            st = st[(st >= 1) & (st <= 8)]
            sent += int(st.sum())
            got += len(st)
        if sent:
            loss = 100.0 * (sent - got) / sent
            self.tiles['loss'].set(f'{loss:.1f}%',
                                   GOOD if loss <= 1 else WARN if loss <= 5 else BAD)
        else:
            self.tiles['loss'].set('—')

        gaps = [np.percentile(np.diff(h) * 1000, 99) for h, _ in links.values()
                if len(h) > 10]
        self.tiles['gap'].set(f'{max(gaps):.0f} ms' if gaps else '—')

        # Each receiving board owns its own 92.16 KB/s wire; show the busiest.
        frame_bytes = 26 + 2 * max(self.n_sub, 1)
        by_rx = {}
        for k, r in per.items():
            by_rx[k[1]] = by_rx.get(k[1], 0) + r
        util = max((r * frame_bytes / 92160 for r in by_rx.values()), default=0)
        self.tiles['wire'].set(f'{100 * util:.0f}%',
                               None if util < 0.9 else WARN if util < 0.98 else BAD)

        board = sum(d.get('framedrops', 0) + d.get('sendfail', 0)
                    for d in stats.values())
        host = self.read_errors + self.clipped
        self.tiles['drops'].set(f'{board}·{host}', None if not (board or host) else BAD)

        cam_fps = 0.0
        if len(ftimes) > 2:
            cam_fps = (len(ftimes) - 1) / max(ftimes[-1] - ftimes[0], 1e-9)
        self.tiles['cam'].set(f'{cam_fps:.0f} fps',
                              None if cam_fps > 25 else WARN if cam_fps > 15 else BAD)

        if self.seen_max is not None:
            self.seen_label.setText(
                f'seen {self.seen_min:.0f} – {self.seen_max:.0f}')

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
    ap.add_argument('--device', default='auto',
                    help='V4L2 node (/dev/videoN) or an AVFoundation index (0, 1, …); '
                         'auto picks the RealSense on Linux and camera 0 on macOS')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--refresh', type=float, default=30.0, help='GUI redraw rate (Hz)')
    ap.add_argument('--bw', type=int, default=40, choices=(20, 40),
                    help='the bandwidth the firmware boots on (CONFIG_WIFI_BANDWIDTH)')
    ap.add_argument('--channel', type=int, default=13,
                    help='the 2.4 GHz channel the firmware boots on, so the survey can '
                         'say what a move would gain (CONFIG_LESS_INTERFERENCE_CHANNEL)')
    ap.add_argument('--tx', default='round-robin',
                    help='"round-robin", or a board label (A/B/C/D) to pin the '
                         'transmitter. Switchable in the GUI at any time.')
    ap.add_argument('--round-duration', type=float, default=0.025,
                    help='seconds each board holds the transmit token; sets the blind '
                         'gap between a link\'s bursts. 25 ms is the measured optimum '
                         'for 166 subcarriers, 12.5 ms for 30 -- see capture.py.')
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
    ap.add_argument('--wf-min', dest='wf_min', type=float, default=0.0,
                    help='waterfall colour scale floor, absolute |CSI| units')
    ap.add_argument('--wf-max', dest='wf_max', type=float, default=150.0,
                    help='waterfall colour scale ceiling, absolute |CSI| units '
                         '(int8 I/Q tops out near 180; this rig peaks ~130)')
    ap.add_argument('--cmap', default='jet', choices=('jet', 'diverge'),
                    help='waterfall colormap: jet (radio classic), or diverge '
                         '(dark at the packet mean, so only deviations glow)')
    args = ap.parse_args()
    if args.outdir is None:
        args.outdir = default_outdir()
    if args.projdir is None:
        args.projdir = default_projdir()

    dev = default_camera_device() if args.device == 'auto' else args.device
    if dev is None:
        raise SystemExit('no camera found. On the Linux rig: is the RealSense plugged '
                         'in, and does this container have "c 81:* rmw"? Pass --device '
                         'to name one explicitly.')
    boards = discover()
    if len(boards) < 2:
        raise SystemExit(f'need >= 2 boards, found {len(boards)}')
    labels = [LABEL.get(m[-5:], m) for m in sorted(boards)]
    n_links = len(boards) * (len(boards) - 1)
    print(f'{len(boards)} boards: {labels} -> {n_links} links', flush=True)
    if len(boards) < len(LABEL):
        # Missing boards are silent in the panels -- they simply are not drawn -- and
        # a short session recorded without noticing is exactly the failure the health
        # banner exists to prevent. Say it once, up front, where it cannot be missed.
        missing = sorted(set(LABEL.values()) - set(labels))
        print(f'note: {", ".join(missing)} not connected; recording {n_links} of '
              f'{len(LABEL) * (len(LABEL) - 1)} links', flush=True)

    cam = open_camera(dev, args.width, args.height, args.fps)
    print(f'camera {cam.name} {cam.w}x{cam.h} @ {cam.fps:g} fps', flush=True)
    app = QApplication(sys.argv)
    w = Live(boards, cam, args)
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
