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

The display is raw by default. Floor can capture the current empty-room response for
each link and flatten its frequency profile from then on; the reference stays fixed
until X is pressed or Floor captures it again. The lower Norm toggle separately removes each
packet's common-mode level bounce. **Both treatments are display only; every
recording stores raw amplitudes.**

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

from capture import (C5_MACS, SUB_INDEX, CsiStream, JpegWriter, LABEL,
                     default_camera_device, default_outdir, derive_fields,
                     discover, field_bounds, open_camera, owner_of,
                     parse_scan_line, rank_channels, resolve_prefix,
                     write_capture)

pg.setConfigOptions(imageAxisOrder='row-major')


_QPushButton = QPushButton


class QPushButton(_QPushButton):
    """Reserve room for the bold checked state used by every selector."""

    def sizeHint(self):
        size = super().sizeHint()
        size.setWidth((size.width() * 105 + 99) // 100)
        return size

    def minimumSizeHint(self):
        return self.sizeHint()


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

    clicked = QtCore.pyqtSignal()

    def __init__(self, caption):
        super().__init__()
        self.setStyleSheet(
            'QFrame { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, '
            f'stop:0 #1e222c, stop:1 {PANEL}); '
            f'border: 1px solid {BORDER}; border-radius: 11px; }}')
        self.setMinimumWidth(88)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 8, 14, 8)
        lay.setSpacing(1)
        self.val = QLabel('—')
        self.cap = QLabel(caption.upper())
        self.val.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.cap.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
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

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(ev)


class FlowLayout(QLayout):
    """A compact left-to-right layout that wraps widgets as its width changes."""

    def __init__(self, parent=None, margin=0, spacing=7):
        super().__init__(parent)
        self.items = []
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)

    def addItem(self, item):
        self.items.append(item)

    def count(self):
        return len(self.items)

    def itemAt(self, index):
        return self.items[index] if 0 <= index < len(self.items) else None

    def takeAt(self, index):
        return self.items.pop(index) if 0 <= index < len(self.items) else None

    def expandingDirections(self):
        return QtCore.Qt.Orientations(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self.items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(),
                            margins.top() + margins.bottom())

    def _layout(self, rect, test_only):
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(),
                             -margins.right(), -margins.bottom())
        x, y, line_height = area.x(), area.y(), 0
        spacing = self.spacing()
        for item in self.items:
            hint = item.sizeHint()
            next_x = x + hint.width() + spacing
            if line_height and next_x - spacing > area.right() + 1:
                x = area.x()
                y += line_height + spacing
                next_x = x + hint.width() + spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + margins.bottom()


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
        if secs is None:
            countdown = ''
        else:
            remaining = int(np.ceil(max(secs, 0)))
            countdown = (f'{remaining // 60}:{remaining % 60:02d}'
                         if remaining >= 60 else str(remaining))
        self.count.setText(countdown)
        self.sub.setText(sub)


class Live(QWidget):
    def __init__(self, boards, cam, args):
        super().__init__()
        self.boards, self.cam, self.args = boards, cam, args
        self.macs = sorted(boards)
        self.c5_rig = all(m in C5_MACS for m in self.macs)
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
        # Empty-room frequency-response equalisation. Each per-link vector is a
        # multiplier captured from the current visible packets: applying it makes
        # the captured profile horizontal while preserving that link's typical
        # amplitude, so the existing absolute plot bounds remain useful. NaNs mark
        # dead/null carriers which cannot be divided safely.
        self.profile_norm = False
        self.norm_scale = {}
        # Running mean level per subcarrier, and the field split derived from it.
        # Measured rather than tabulated because the tabulated layout is wrong (see
        # derive_fields), and because it has to survive a bandwidth change at runtime.
        self.amp_mean = None
        self.fields = None
        self.fields_at = 0.0
        self.scan = None           # channel survey state, None when not scanning
        self.band = '2.4'          # firmware always boots here; --band retunes below
        self.channels = {'2.4': args.channel, '5.6': 120}
        self.bandwidths = {'2.4': args.bw, '5.6': 20}
        self.bw = args.bw          # what the boards are believed to be running
        self.sub_sel = 0 if self.c5_rig else 30
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
            QGroupBox {{ border: 1px solid {BORDER}; border-radius: 11px;
                         margin-top: 12px; padding: 10px 8px 8px 8px; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px;
                                font-size: 10px; font-weight: bold; letter-spacing: 1px; }}
            QGroupBox#recordingSetup {{ border-color: #75552b; }}
            QGroupBox#recordingSetup::title {{ color: {WARN}; }}
            QGroupBox#displayOnly {{ border-color: #245d63; }}
            QGroupBox#displayOnly::title {{ color: #67e8f9; }}
            QGroupBox#captureControls {{ border-color: #315c99; }}
            QGroupBox#captureControls::title {{ color: #93c5fd; }}
            QTabWidget::pane {{ border: none; }}
            QTabBar::tab {{ padding: 6px 14px; color: {MUTED}; background: {PANEL};
                            border: 1px solid {BORDER}; border-bottom: none;
                            border-top-left-radius: 8px; border-top-right-radius: 8px; }}
            QTabBar::tab:selected {{ color: white; background: {ACCENT};
                                     border-color: {ACCENT}; font-weight: bold; }}
        ''')
        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)

        # ---- camera ----
        left = QVBoxLayout()
        # Live health belongs with the camera overview rather than the radio controls.
        self.tiles = {}
        tile_rows = (
            (('rate', 'per-link rate'), ('gap', 'gap p99'),
             ('drops', 'drops (board·host)')),
            (('deliv', 'delivery'), ('loss', 'radio loss'), ('wire', 'wire util')),
        )
        for specs in tile_rows:
            strip = QHBoxLayout()
            for key, caption in specs:
                t = StatTile(caption)
                strip.addWidget(t, 1)
                self.tiles[key] = t
            left.addLayout(strip)
        self.tiles_at = 0.0
        self.board_stats = {}       # rx mac -> parsed STATS counters
        self.link_health = {}       # directed link -> (one-second Hz, loss or None)
        self.drop_board_base = {}
        self.drop_host_base = 0
        self.tiles['drops'].setCursor(QtCore.Qt.PointingHandCursor)
        self.tiles['drops'].setToolTip('Click to reset the displayed drop counters')
        self.tiles['drops'].clicked.connect(self.reset_drop_stats)

        self.cam_label = QLabel(alignment=QtCore.Qt.AlignCenter)
        self.cam_label.setMinimumWidth(360)   # the video scales; the layout decides
        self.cam_fps = QLabel('—', self.cam_label)
        self.cam_fps.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.cam_fps.setStyleSheet(
            f'font-size: 17px; font-weight: bold; color: {INK}; '
            'background-color: rgba(16, 18, 24, 185); '
            'border: 1px solid rgba(255, 255, 255, 35); border-radius: 7px; '
            'padding: 4px 8px;')
        cap = QLabel(f'{cam.name} · {cam.w}x{cam.h} @ {cam.fps:g} fps')
        cap.setStyleSheet(f'color: {MUTED}; font-size: 13px;')
        left.addWidget(cap)
        left.addWidget(self.cam_label, 1)

        root.addLayout(left, 1)

        # ---- CSI grid ----
        right = QVBoxLayout()
        control_tabs = QTabWidget()
        self.control_tabs = control_tabs
        setup_box = QGroupBox()
        setup_box.setObjectName('recordingSetup')
        setup_layout = FlowLayout(setup_box, margin=8)

        # ---- who transmits: round-robin, or one board pinned ----
        self.by_label = {LABEL.get(m[-5:], m): m for m in self.macs}
        setup_layout.addWidget(QLabel('TX'))
        self.tx_buttons = {}
        group = QButtonGroup(self)
        group.setExclusive(True)
        for name in ['round-robin'] + sorted(self.by_label):
            b = QPushButton(name)
            b.setCheckable(True)
            b.clicked.connect(lambda _c, n=name: self.set_tx(n))
            group.addButton(b)
            setup_layout.addWidget(b)
            self.tx_buttons[name] = b
        setup_layout.addWidget(QLabel('Hz'))
        self.rate_buttons = {}
        rate_group = QButtonGroup(self)
        rate_group.setExclusive(True)
        for hz in (300, 600, 900):
            b = QPushButton(str(hz))
            b.setCheckable(True)
            b.clicked.connect(lambda _c, rate=hz: self.pick_rate_preset(rate))
            rate_group.addButton(b)
            setup_layout.addWidget(b)
            self.rate_buttons[hz] = b
        self.rate_preset = 300
        self.rate_buttons[300].setChecked(True)

        self.csi_btn = QPushButton('■ STOP CSI')
        self.csi_btn.setMinimumWidth(147)
        self.csi_btn.clicked.connect(self.toggle_csi)
        setup_layout.addWidget(self.csi_btn)
        setup_layout.addWidget(QLabel('SUB'))
        self.sub_buttons = {}
        sub_group = QButtonGroup(self)
        sub_group.setExclusive(True)
        sub_choices = ((0, 'ALL (C5)'),) if self.c5_rig else ((30, '30'), (114, '114'), (166, '166'))
        for n, label in sub_choices:
            b = QPushButton(label)
            b.setCheckable(True)
            b.clicked.connect(lambda _c, k=n: self.set_sub(k))
            sub_group.addButton(b)
            setup_layout.addWidget(b)
            self.sub_buttons[n] = b
        self.sub_buttons[self.sub_sel].setChecked(True)
        setup_layout.addWidget(QLabel('BAND'))
        self.band_buttons = {}
        band_group = QButtonGroup(self)
        band_group.setExclusive(True)
        for band in ('2.4', '5.6'):
            b = QPushButton(f'{band} GHz')
            b.setCheckable(True)
            b.clicked.connect(lambda _c, selected=band: self.set_band(selected))
            if band == '5.6':
                b.setToolTip('ESP32-C5 only · 5600 MHz primary (Wi-Fi channel 120)')
                b.setEnabled(self.c5_rig)
            band_group.addButton(b)
            setup_layout.addWidget(b)
            self.band_buttons[band] = b
        self.band_buttons[args.band].setChecked(True)
        self.scan_btn = QPushButton('⚡ FIND BEST CHANNEL')
        self.scan_btn.setMinimumWidth(200)
        self.scan_btn.clicked.connect(self.start_scan)
        self.scan_result_timer = QtCore.QTimer(self)
        self.scan_result_timer.setSingleShot(True)
        self.scan_result_timer.timeout.connect(self.reset_scan_button)
        setup_layout.addWidget(self.scan_btn)
        setup_layout.addWidget(QLabel('BW'))
        self.bw_buttons = {}
        bw_group = QButtonGroup(self)
        bw_group.setExclusive(True)
        for mhz in (20, 40):
            b = QPushButton(str(mhz))
            b.setCheckable(True)
            b.clicked.connect(lambda _c, width=mhz: self.set_bw(width))
            bw_group.addButton(b)
            setup_layout.addWidget(b)
            self.bw_buttons[mhz] = b
        self.bw_buttons[self.bw].setChecked(True)
        control_tabs.addTab(setup_box, 'CSI')

        # ---- display-only controls ----
        display_box = QGroupBox()
        display_box.setObjectName('displayOnly')
        visual_flow = FlowLayout(display_box, margin=8)
        floor_control = QWidget()
        floor_control.setFixedSize(131, 36)
        self.norm_btn = QPushButton('Floor', floor_control)
        self.norm_btn.setFixedSize(80, 36)
        self.norm_btn.move(51, 0)
        self.norm_btn.setCheckable(True)
        self.norm_btn.setToolTip(
            'Capture a new per-link empty-room floor for the display. '
            'Recordings always stay raw.')
        self.norm_btn.clicked.connect(self.capture_profile_norm)
        self.norm_off_btn = QPushButton('X', floor_control)
        self.norm_off_btn.setFixedSize(28, 14)
        self.norm_off_btn.move(103, 0)
        self.norm_off_btn.setStyleSheet(
            'QPushButton { padding: 0; font-size: 10px; font-weight: bold; '
            'color: white; background: #991b2f; border: 1px solid #e05268; '
            'border-radius: 7px; }'
            'QPushButton:hover { background: #be2440; border-color: #fb7185; }'
            'QPushButton:pressed { background: #701326; }'
            'QPushButton:disabled { color: #606572; background: #22252e; '
            'border-color: #353946; }')
        self.norm_off_btn.setEnabled(False)
        self.norm_off_btn.setToolTip(
            'Turn off empty-room normalization and return to the raw display.')
        self.norm_off_btn.clicked.connect(self.disable_profile_norm)
        self.norm_off_btn.raise_()
        visual_flow.addWidget(floor_control)
        visual_flow.addWidget(QLabel('VIEW'))
        self.view_buttons = {}
        view_group = QButtonGroup(self)
        view_group.setExclusive(True)
        for mode in ('spectrum', 'waterfall'):
            b = QPushButton(mode)
            b.setCheckable(True)
            b.clicked.connect(lambda _c, selected=mode: self.set_view(selected))
            view_group.addButton(b)
            visual_flow.addWidget(b)
            self.view_buttons[mode] = b
        self.view_buttons[self.view_mode].setChecked(True)
        self.lock_btn = QPushButton('Norm')
        self.lock_btn.setCheckable(True)
        self.lock_btn.setToolTip(
            'Toggle per-packet level locking for the display only. '
            'Recordings always stay raw.')
        self.lock_btn.clicked.connect(self.toggle_level_lock)
        visual_flow.addWidget(self.lock_btn)
        visual_flow.addWidget(QLabel('PLOT RANGE'))
        self.wfmin_edit = QLineEdit(f'{self.wf_min:g}')
        self.wfmin_edit.setFixedWidth(84)
        self.wfmin_edit.returnPressed.connect(self.set_wf_scale)
        visual_flow.addWidget(self.wfmin_edit)
        self.wfmax_edit = QLineEdit(f'{self.wf_max:g}')
        self.wfmax_edit.setFixedWidth(84)
        self.wfmax_edit.returnPressed.connect(self.set_wf_scale)
        visual_flow.addWidget(self.wfmax_edit)
        fit = QPushButton('FIT')
        fit.setMaximumWidth(53)
        fit.setToolTip('Fit per-link display ranges. Recordings always stay raw.')
        fit.clicked.connect(self.fit_wf_scale)
        visual_flow.addWidget(fit)
        self.seen_label = QLabel('')
        self.seen_label.setStyleSheet(f'color: {MUTED}; font-size: 11px;')
        visual_flow.addWidget(self.seen_label)
        control_tabs.addTab(display_box, 'VIZ')

        self.role = QLabel('')
        self.role.setStyleSheet(f'color: {INK}; font-size: 14px;')
        self.scan_label = QLabel('')
        self.scan_label.setStyleSheet(
            f'color: {MUTED}; font-size: 12px; font-family: Menlo, monospace;')
        self.scan_label.setWordWrap(True)

        # ---- capture ----
        self.rec = None            # None when idle; a dict of state while recording
        self.proto = None          # scripted protocol state, None when not running
        self.cue = None
        self.proto_label = QLabel('')
        self.proto_label.setStyleSheet(f'color: {MUTED}; font-size: 12px;')
        self.proto_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        capture_box = QGroupBox()
        capture_box.setObjectName('captureControls')
        cap_bar = FlowLayout(capture_box, margin=8)

        self.prefix_edit = QLineEdit(args.prefix)
        self.prefix_edit.setMinimumWidth(160)
        cap_bar.addWidget(self.prefix_edit)
        self.rec_btn = QPushButton('● REC')
        self.rec_btn.setMinimumWidth(137)
        self.rec_btn.clicked.connect(self.toggle_record)
        cap_bar.addWidget(self.rec_btn)
        self.proto_btn = QPushButton('▶ RUN PROTOCOL')
        self.proto_btn.clicked.connect(self.toggle_protocol)
        self.proto_btn.setEnabled(bool(args.protocol))
        cap_bar.addWidget(self.proto_btn)
        cap_bar.addWidget(self.proto_label)
        pick = QPushButton('Protocol…')
        pick.clicked.connect(self.pick_protocol)
        cap_bar.addWidget(pick)
        control_tabs.addTab(capture_box, 'REC')
        self.style_rec_button()

        self.rec_status = QLabel('')
        self.rec_status.setStyleSheet(f'color: {MUTED}; font-size: 13px;')
        self.health = QLabel('')
        self.health.setStyleSheet('font-size: 14px; font-weight: bold;')
        self.health.setWordWrap(True)
        right.addWidget(control_tabs)
        right.addWidget(self.health)
        right.addWidget(self.rec_status)
        self.show_protocol()
        control_tabs.currentChanged.connect(self.resize_control_tabs)

        self.grid = QGridLayout()
        self.grid.setSpacing(10)
        self.grid_host = QWidget()
        self.grid_host.setLayout(self.grid)
        self.grid_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right.addWidget(self.grid_host, 1)
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
        self.arrange_panels()
        self.set_band(args.band, force=True)
        self.set_sub(self.sub_sel)
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
        QtCore.QTimer.singleShot(0, self.resize_csi_grid)
        QtCore.QTimer.singleShot(0, self.resize_control_tabs)

    def current_fields(self):
        """The field split to normalise and draw by: measured if enough packets have
        arrived, otherwise the layout fallback so the first repaint is not one block."""
        if self.fields:
            return self.fields
        return field_bounds(self.n_sub) if self.n_sub else [(0, 1)]

    def resize_csi_grid(self):
        """Let the plot grid fill all space left by the responsive control tab."""
        self.grid_host.setMaximumHeight(16777215)

    def resize_control_tabs(self):
        """Match the tab frame to the active responsive control layout."""
        page = self.control_tabs.currentWidget()
        if page is None:
            return
        content_width = max(self.control_tabs.width() - 4, 1)
        content_height = page.heightForWidth(content_width)
        if content_height < 0:
            content_height = page.sizeHint().height()
        self.control_tabs.setFixedHeight(
            self.control_tabs.tabBar().sizeHint().height() + content_height + 4)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'grid_host'):
            self.resize_csi_grid()
            self.resize_control_tabs()

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

    def fit_wf_scale(self, _checked=False):
        """Snap the fixed colour bounds to the range actually seen. The signal here
        sits around a third of the way up a 0-150 scale, so its ~10% flutter spans
        about two colour steps -- which is why a wide fixed scale reads as "not
        moving". Fitting spends the whole colormap on the real range; the scale
        stays fixed afterwards until fitted or edited again."""
        # Per LINK, from the CURRENT buffers: each panel gets bounds hugged to its
        # own signal, so a 3 m link and a 4.2 m diagonal are both readable at once.
        with self.lock:
            snap = {
                k: [it[2] for it in list(v)[-NPKT:]]
                for k, v in self.buf.items() if len(v) > 3
            }
        n = 0
        for k, vals in snap.items():
            width = len(vals[-1])
            vals = [v for v in vals if len(v) == width]
            if not vals:
                continue
            A = np.stack(vals, axis=1)
            A = self.normalised_signal(k, A)
            A = A[np.isfinite(A)]
            if not A.size:
                continue
            lo, hi = float(np.floor(A.min())), float(np.ceil(A.max()))
            if hi <= lo:
                hi = lo + 1
            self.link_scale[k] = (lo, hi)
            n += 1
        if n:
            self.rec_status.setText(f'scale fitted per link ({n} links)')
        return n

    def capture_profile_norm(self, _checked=False):
        """Capture fixed empty-room frequency-response equalisation.

        The median of the visible packet window is robust to an occasional noisy
        packet. Every live subcarrier is scaled to that profile's median level, so
        the room response becomes a horizontal line without moving the plot to an
        unfamiliar 0/1 scale. Dead carriers are left as gaps instead of amplifying
        their near-zero noise. This only touches values on their way to the plots.
        """
        with self.lock:
            snap = {
                k: [it[2].copy() for it in list(v)[-NPKT:]]
                for k, v in self.buf.items() if len(v) > 3
            }
        scales = {}
        for key, vals in snap.items():
            n = len(vals[-1])
            vals = [v for v in vals if len(v) == n]
            if len(vals) < 4:
                continue
            profile = np.median(np.stack(vals), axis=0).astype(np.float64)
            positive = profile[np.isfinite(profile) & (profile > 0)]
            if not positive.size:
                continue
            floor = 0.05 * float(np.median(positive))
            valid = np.isfinite(profile) & (profile > max(floor, 1e-9))
            if not valid.any():
                continue
            level = float(np.median(profile[valid]))
            scale = np.full(profile.shape, np.nan, dtype=np.float64)
            scale[valid] = level / profile[valid]
            scales[key] = scale

        if not scales:
            # A failed recapture must not turn off a floor that was already active.
            self.norm_btn.setChecked(self.profile_norm)
            self.rec_status.setText(
                '<span style="color:#ff5555">Floor needs at least four current '
                'packets on a link</span>')
            return

        self.norm_scale = scales
        self.profile_norm = True
        self.level_ref.clear()
        # Floor is a recapture button, not a conventional toggle: clicking an
        # already-blue button captures again and must leave it blue.
        self.norm_btn.setChecked(True)
        self.norm_off_btn.setEnabled(True)
        fitted = self.fit_wf_scale()
        self.rec_status.setText(
            f'empty-room floor captured ({len(scales)} links) · '
            f'scale fitted ({fitted} links) · display only')

    def disable_profile_norm(self, _checked=False, announce=True):
        """Turn off display normalization; the captured reference is discarded."""
        self.profile_norm = False
        self.norm_scale.clear()
        self.level_ref.clear()
        self.norm_btn.setChecked(False)
        self.norm_off_btn.setEnabled(False)
        if announce:
            self.rec_status.setText('empty-room floor off')

    def normalised_signal(self, key, values):
        """Return display values with the captured frequency profile removed."""
        scale = self.norm_scale.get(key) if self.profile_norm else None
        a = np.asarray(values)
        if scale is None or not a.ndim or a.shape[0] != len(scale):
            return a
        if a.ndim == 1:
            return a * scale
        return a * scale[:, None]

    def toggle_level_lock(self, checked):
        """raw: exactly what the boards deliver and the recording stores. norm: the
        postprocessed view -- per-packet level normalisation, the same treatment a
        model's preprocessing applies, so the plots show what training data will
        look like. Display only in both positions."""
        self.level_lock = checked

    def set_view(self, mode):
        if mode == self.view_mode:
            return
        self.view_mode = mode
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
            newest = self.normalised_signal(key, items[-1][2])
            med = float(np.nanmedian(newest))
            med = med if np.isfinite(med) and med > 0 else 1.0
            ref = med if ref is None else ref + 0.02 * (med - ref)
            self.level_ref[key] = ref
        k = len(items)
        for i, c in enumerate(pan['curves']):
            j = i - (self.NTRAIL - k)            # oldest ghost first, newest last
            if j < 0:
                c.setData([], [])
                continue
            y = self.normalised_signal(key, items[j][2])
            if self.level_lock:
                m = float(np.nanmedian(y))
                if np.isfinite(m) and m > 0:
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
        lab.setContentsMargins(6, 2, 6, 2)
        lab.setStyleSheet(f'color: {INK}; font-size: 12px; font-weight: bold;')
        gl = pg.GraphicsLayoutWidget()
        gl.setBackground(SURFACE)
        gl.ci.layout.setSpacing(4)
        gl.ci.setContentsMargins(0, 0, 0, 0)
        gl.setMinimumHeight(0)
        gl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        w = QWidget()
        w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(lab)
        box.addWidget(gl, 1)
        w.setLayout(box)
        w.setObjectName('linkPanel')
        self.grid.addWidget(w, r, c)
        return dict(widget=w, gl=gl, lab=lab, title=title, cells=[], row=r, col=c)

    def arrange_panels(self):
        """Give pinned-TX links the plot area and collapse inactive links to labels."""
        for pan in self.panels.values():
            self.grid.removeWidget(pan['widget'])
        for row in range(4):
            self.grid.setRowStretch(row, 0)
        for col in range(3):
            self.grid.setColumnStretch(col, 1)

        if self.tx_sel == 'round-robin':
            for pan in self.panels.values():
                pan['widget'].setMinimumHeight(0)
                pan['widget'].setMaximumHeight(16777215)
                pan['widget'].setStyleSheet('QWidget#linkPanel { border: none; }')
                pan['gl'].show()
                self.grid.addWidget(pan['widget'], pan['row'], pan['col'])
            for row in range(4):
                self.grid.setRowStretch(row, 1)
            return

        active_tx = self.by_label[self.tx_sel]
        active = [pan for key, pan in self.panels.items() if key[0] == active_tx]
        inactive = [pan for key, pan in self.panels.items() if key[0] != active_tx]
        for col, pan in enumerate(active):
            pan['widget'].setMinimumHeight(0)
            pan['widget'].setMaximumHeight(16777215)
            pan['widget'].setStyleSheet('QWidget#linkPanel { border: none; }')
            pan['gl'].show()
            self.grid.addWidget(pan['widget'], 0, col)
        self.grid.setRowStretch(0, 1)
        for i, pan in enumerate(inactive):
            pan['gl'].hide()
            pan['widget'].setFixedHeight(34)
            pan['widget'].setStyleSheet(
                f'QWidget#linkPanel {{ background: {PANEL}; border: 1px solid {BORDER}; '
                'border-radius: 6px; }')
            self.grid.addWidget(pan['widget'], 1 + i // 3, i % 3)

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
                    if rec is not None and now >= rec['t0']:
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
                          until=time.time() + timing['lead_in'], record_rate_armed=False)
        self.cue = Cue()
        self.cue.show()
        self.cue.raise_()
        self.proto_btn.setText('■ ABORT')
        self.rec_btn.setEnabled(False)
        self.prefix_edit.setEnabled(False)
        self.apply_protocol_idle_rate()
        self.proto_tick()

    def abort_protocol(self, why, keep_cue=False):
        if self.rec is not None:
            self.stop_record()          # never leave a half-written take on disk
        self.proto = None
        self.apply_rate_preset()
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

        # Let the boards return to the selected capture rate before timestamps enter
        # the take, while still giving them nearly all of each lead-in and gap to rest.
        if p['phase'] == LEAD and not p['record_rate_armed'] and left <= 1.0:
            self.apply_rate_preset(force=True)
            p['record_rate_armed'] = True

        if left <= 0:
            if p['phase'] == LEAD:
                self.start_record(f'{self.session_dir()}/{name}')
                p['phase'], p['until'] = REC, now + t['duration']
            elif p['phase'] == REC:
                self.stop_record()
                self.apply_protocol_idle_rate()
                p['phase'], p['until'] = GAP, now + t['gap']
            else:
                p['i'] += 1
                if p['i'] >= len(p['takes']):
                    return self.abort_protocol(
                        f'finished — {len(p["takes"])} takes written')
                p['phase'], p['until'] = LEAD, now + t['lead_in']
                p['record_rate_armed'] = False
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
            self.cue.show_state(REC, step, instruction, left, '● RECORDING')
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
        t0_ns = time.time_ns()
        self.rec = dict(prefix=prefix, name=name, t0=t0_ns / 1e9, t0_ns=t0_ns, own=own,
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
        meta = dict(t0_epoch=rec['t0'], t0_epoch_ns=rec['t0_ns'], mode=self.tx_sel,
                    round_duration=self.args.round_duration,
                    wifi_band_ghz=self.band, wifi_channel=self.channels[self.band],
                    wifi_bandwidth_mhz=self.bw,
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
        if self.band != '2.4':
            self.rec_status.setText('switch to 2.4 GHz before running its channel survey')
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
        self.scan_result_timer.stop()
        self.scan_btn.setText('⚡ SCANNING…')
        self.scan_btn.setEnabled(False)
        self.scan_label.setText('parking the radio …')

    def reset_scan_button(self):
        if self.scan is None:
            self.refresh_scan_availability()

    def refresh_scan_availability(self):
        """The existing survey ranks 2.4 GHz channels 1-13 only."""
        on_2g = self.band == '2.4'
        self.scan_btn.setEnabled(on_2g and self.scan is None)
        self.scan_btn.setText('⚡ FIND BEST CHANNEL' if on_2g else 'SCAN: 2.4 GHz ONLY')
        self.scan_btn.setToolTip(
            '' if on_2g else 'Switch to 2.4 GHz to survey channels 1-13.')

    def show_scan_result(self, text):
        self.scan_btn.setText(text)
        self.scan_result_timer.start(5000)

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

    def reset_drop_stats(self):
        """Zero the displayed counters without rebooting or changing raw data."""
        with self.lock:
            self.drop_board_base = {
                m: d.get('framedrops', 0) + d.get('sendfail', 0)
                for m, d in self.board_stats.items()
            }
            self.drop_host_base = self.read_errors + self.clipped
        self.tiles['drops'].set('0·0')

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
            self.show_scan_result('NO CHANNEL FOUND')
            return
        ranked = rank_channels(stats)
        best, best_bytes, best_rssi, span = ranked[0]
        self.show_scan_result(f'CH {best} FOUND')
        current = self.channels['2.4']
        cur = next((r for r in ranked if r[0] == current), None)
        # One line on screen; the full per-channel table goes to stdout below, where
        # it can be scrolled back rather than crowding the controls.
        moved = best != current
        if timed_out:
            verdict = '<b style="color:#fbbf24">survey timed out</b> — all boards on SCAN firmware?'
        elif cur and cur[1] > best_bytes:
            pct = 100.0 * (cur[1] - best_bytes) / max(cur[1], 1)
            verdict = (f'⚡ <b>ch{best}</b> · {pct:.0f}% less airtime than '
                       f'ch{current}' + (' · moving' if moved else ''))
        else:
            verdict = f'⚡ <b>ch{current}</b> already best'
        self.scan_label.setText(verdict)
        # Move only on a meaningful win: your last survey ranked the top five
        # placements within 1.5% of each other -- a statistical tie -- and moving
        # the whole rig on that is churn, not optimisation.
        if moved and cur and cur[1] > best_bytes:
            pct = 100.0 * (cur[1] - best_bytes) / max(cur[1], 1)
            if pct < 5.0:
                self.scan_label.setText(
                    f'⚡ <b>ch{current}</b> kept — best alternative ch{best} '
                    f'is only {pct:.0f}% better (tie)')
                return
        print(f'channel survey: best={best} span={span[0]}-{span[-1]} bytes={best_bytes} '
              f'rssi={best_rssi}; ranked={[(r[0], r[1]) for r in ranked[:5]]}', flush=True)
        if best != current:
            self.set_channel(best)

    def busy_reason(self):
        """Why the radio must not be reconfigured right now, or None if it may be."""
        if self.rec is not None or self.proto is not None:
            return ('stop the recording first — changing the frame layout mid-take '
                    'writes a take whose packets disagree about their own width.')
        if self.scan is not None:
            return 'wait for the survey to finish — it is retuning the radio itself.'
        return None

    def frame_width(self, sel=None):
        """Subcarriers per frame for a SUB selection, accounting for HT20's fallback:
        the 30/166 tables index up to subcarrier 190, which does not exist in a
        128-wide HT20 report, so the firmware sends everything instead."""
        sel = self.sub_sel if sel is None else sel
        if sel == self.sub_sel and self.n_sub:
            return self.n_sub
        if self.c5_rig:
            return 57 if self.bw == 20 else 117
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
        return int(key)

    def protocol_idle_hz(self):
        """Temporary protocol rest rate: 25 Hz per RR link, 100 Hz pinned."""
        return 25 * len(self.macs) if self.tx_sel == 'round-robin' else 100

    def apply_protocol_idle_rate(self):
        """Lower the rate between takes without changing the selected preset."""
        self.send_rate(self.protocol_idle_hz())

    def refresh_rate_buttons(self):
        divisor = len(self.macs) if self.tx_sel == 'round-robin' else 1
        for key, b in self.rate_buttons.items():
            b.setText(str(key // divisor))

    def pick_rate_preset(self, key):
        why = self.busy_reason()
        if why:
            self.rec_status.setText(f'<span style="color:#ff5555">{why}</span>')
            self.rate_buttons[self.rate_preset].setChecked(True)
            return
        self.rate_preset = key
        self.apply_rate_preset(announce=True)

    def apply_rate_preset(self, announce=False, force=False):
        """Send the selected fixed rate after a button or radio-layout change."""
        self.refresh_rate_buttons()
        if (self.rec is not None or self.proto is not None) and not force:
            return                       # never retune mid-take
        hz = self.preset_hz(self.rate_preset)
        if self.send_rate(hz) and announce:
            self.rec_status.setText(
                f'rate → {hz} Hz · wire ceiling {self.wire_ceiling()} Hz here')

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
            selected = self.sub_sel if self.sub_sel in self.sub_buttons else next(iter(self.sub_buttons))
            self.sub_buttons[selected].setChecked(True)
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
                f'(wire ceiling {self.wire_ceiling()} Hz)')
            self.recalibrate()

    def set_band(self, want, force=False):
        """Move the whole C5 rig between 2.4 GHz and channel 120 (5.6 GHz)."""
        if want not in self.band_buttons:
            return False
        if want == '5.6' and not self.c5_rig:
            self.rec_status.setText(
                '<span style="color:#ff5555">5.6 GHz requires an all-ESP32-C5 rig.</span>')
            self.band_buttons[self.band].setChecked(True)
            return False
        if want == self.band:
            self.band_buttons[want].setChecked(True)
            self.refresh_scan_availability()
            return True
        why = self.busy_reason()
        if why and not force:
            self.rec_status.setText(f'<span style="color:#ff5555">{why}</span>')
            self.band_buttons[self.band].setChecked(True)
            return False
        ok = True
        for m in self.macs:
            try:
                self.boards[m].write(f'BAND {want}\n'.encode())
            except (serial.SerialException, OSError):
                ok = False
        if not ok:
            self.band_buttons[self.band].setChecked(True)
            return False
        self.band = want
        self.bw = self.bandwidths[want]
        self.band_buttons[want].setChecked(True)
        self.bw_buttons[self.bw].setChecked(True)
        self.bw_buttons[40].setEnabled(want == '2.4')
        self.bw_buttons[40].setToolTip(
            '' if want == '2.4' else 'The verified ESP32-C5 5.6 GHz CSI mode is 20 MHz.')
        self.refresh_scan_availability()
        self.recalibrate()
        self.rec_status.setText(
            f'band → {want} GHz · channel {self.channels[want]} · {self.bw} MHz')
        return True

    def set_bw(self, want):
        """Set the whole rig to 20 or 40 MHz.

        Neither is simply better: 40 MHz carries more
        subcarriers, 20 MHz is narrow enough to escape a congested band entirely --
        measured here, HT20 on a quiet channel delivered 99.7% against HT40's 91%.
        A recording's frames say which was active (n_sub 128 against 166/192).
        """
        if want == self.bw:
            return
        if self.band == '5.6' and want != 20:
            self.rec_status.setText(
                '<span style="color:#ff5555">5.6 GHz uses the verified 20 MHz C5 '
                'CSI mode.</span>')
            self.bw_buttons[self.bw].setChecked(True)
            return
        why = self.busy_reason()
        if why:
            self.rec_status.setText(f'<span style="color:#ff5555">{why}</span>')
            self.bw_buttons[self.bw].setChecked(True)
            return
        ok = True
        for m in self.macs:
            try:
                self.boards[m].write(f'BW {want}\n'.encode())
            except (serial.SerialException, OSError):
                ok = False
        if not ok:
            self.bw_buttons[self.bw].setChecked(True)
            return
        self.bw = want
        self.bandwidths[self.band] = want
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
        self.n_sub = 0
        self.fields_at = 0.0
        self.seen_min = self.seen_max = None
        self.link_scale.clear()
        self.disable_profile_norm(announce=False)

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
            self.channels[self.band] = ch
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
        self.arrange_panels()
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
                # A frame already queued in the driver can predate the button press;
                # exclude it so every shared-origin timestamp is non-negative.
                if wall >= rec['t0']:
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
        self.cam_fps.adjustSize()
        self.cam_fps.move(max(self.cam_label.width() - self.cam_fps.width() - 12, 0), 12)
        self.cam_fps.raise_()

        sel = self.tx_sel
        active_tx = None if sel == 'round-robin' else self.by_label[sel]

        for key, pan in self.panels.items():
            live = active_tx is None or key[0] == active_tx
            rate, loss = self.link_health.get(key, (0.0, None))
            if live:
                loss_text = 'loss —' if loss is None else f'{loss:.1f}% loss'
                pan['lab'].setText(f'{pan["title"]} · {rate:.0f} Hz · {loss_text}')
                label_color = (INK if loss is None else
                               GOOD if loss <= 1 else WARN if loss <= 5 else BAD)
            else:
                pan['lab'].setText(f'{pan["title"]} · inactive')
                label_color = '#4a4a4c'
            pan['lab'].setStyleSheet(
                f'color: {label_color}; font-size: 12px; '
                f'font-weight: {"bold" if live else "normal"};')
            if not live:
                if self.view_mode == 'spectrum':
                    for curve in pan['curves']:
                        curve.setData([], [])
                else:
                    for img, _vb, _lo, _hi in pan['cells']:
                        img.clear()
                continue
            M, _ncol, _span = self.last_packets(key, now)
            if self.view_mode == 'spectrum':
                self.draw_spectrum(key, pan)
                continue
            if M is None:
                for img, _vb, _lo, _hi in pan['cells']:
                    img.clear()
                continue
            M = self.normalised_signal(key, M)
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
            self.role.setText(
                f'<b>round-robin</b> · {len(self.panels)} links · '
                f'{self.band} GHz ch{self.channels[self.band]} · {self.bw} MHz')
        else:
            rx = ', '.join(sorted(l for l in self.by_label if l != sel))
            self.role.setText(f'<b>TX {sel} → {rx}</b> · {len(self.macs) - 1} of '
                              f'{len(self.panels)} links live · {self.band} GHz '
                              f'ch{self.channels[self.band]} · {self.bw} MHz')

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
        """Update live health; the Hz and delivery tiles use a one-second window."""
        active_tx = (None if self.tx_sel == 'round-robin'
                     else self.by_label[self.tx_sel])
        with self.lock:
            links = {k: (np.array([it[0] for it in v]),
                         np.array([it[1] for it in v]))
                     for k, v in self.buf.items()
                     if v and (active_tx is None or k[0] == active_tx)}
            stats = {m: dict(d) for m, d in self.board_stats.items()}
        # A literal trailing one-second count: responsive enough to show a rate
        # change immediately and independent of the multi-second plot buffer.
        cutoff = now - 1.0
        per = {k: float(np.count_nonzero(h >= cutoff))
               for k, (h, _l) in links.items()}
        rates = list(per.values())
        total_rate = sum(rates)
        self.tiles['rate'].set(f'{np.mean(rates):.0f} Hz' if rates else '—')

        if self.csi_on and self.rate_val and rates:
            expected = self.rate_val * max(len(self.macs) - 1, 1)
            deliv = 100.0 * total_rate / expected
            self.tiles['deliv'].set(f'{min(deliv, 100):.1f}%',
                                    GOOD if deliv >= 97 else WARN if deliv >= 90 else BAD)
        else:
            self.tiles['deliv'].set('—')

        # Radio loss over the same trailing second as the Hz tile, using the boards'
        # hardware receive clocks. Consecutive-arrival steps are measured in ping
        # periods; steps past 8 periods are the round-robin blind gap (schedule, not
        # loss) and are excluded.
        sent = got = 0
        per_loss = {}
        for key, (h, lts) in links.items():
            recent = lts[h >= cutoff]
            st = np.rint(np.diff(np.sort(recent)) * self.rate_val).astype(int)
            st = st[(st >= 1) & (st <= 8)]
            link_sent, link_got = int(st.sum()), len(st)
            sent += link_sent
            got += link_got
            if link_sent:
                per_loss[key] = 100.0 * (link_sent - link_got) / link_sent
        self.link_health = {
            key: (rate, per_loss.get(key)) for key, rate in per.items()
        }
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

        board = 0
        for m, d in stats.items():
            current = d.get('framedrops', 0) + d.get('sendfail', 0)
            base = self.drop_board_base.get(m, 0)
            if current < base:        # this board rebooted since the display reset
                self.drop_board_base[m] = current
                base = current
            board += current - base
        host = self.read_errors + self.clipped - self.drop_host_base
        self.tiles['drops'].set(f'{board}·{host}', None if not (board or host) else BAD)

        cam_fps = 0.0
        if len(ftimes) > 2:
            cam_fps = (len(ftimes) - 1) / max(ftimes[-1] - ftimes[0], 1e-9)
        fps_color = INK if cam_fps > 25 else WARN if cam_fps > 15 else BAD
        self.cam_fps.setText(f'{cam_fps:.0f} fps')
        self.cam_fps.setStyleSheet(
            f'font-size: 17px; font-weight: bold; color: {fps_color}; '
            'background-color: rgba(16, 18, 24, 185); '
            'border: 1px solid rgba(255, 255, 255, 35); border-radius: 7px; '
            'padding: 4px 8px;')

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
    ap.add_argument('--band', default='2.4', choices=('2.4', '5.6'),
                    help='initial radio selection; 5.6 requires ESP32-C5 boards')
    ap.add_argument('--channel', type=int, default=13,
                    help='the 2.4 GHz channel the firmware boots on, so the survey can '
                         'say what a move would gain (CONFIG_LESS_INTERFERENCE_CHANNEL)')
    ap.add_argument('--tx', default='round-robin',
                    help='"round-robin", or a discovered board label to pin the '
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
    if all(m in C5_MACS for m in boards):
        expected = {'F', 'G'}
    elif all(m not in C5_MACS for m in boards):
        expected = {'A', 'B', 'C', 'D'}
    else:
        expected = None
    missing = sorted(expected - set(labels)) if expected else []
    if missing:
        # Missing boards are silent in the panels -- they simply are not drawn -- and
        # a short session recorded without noticing is exactly the failure the health
        # banner exists to prevent. Say it once, up front, where it cannot be missed.
        print(f'note: {", ".join(missing)} not connected; recording {n_links} of '
              f'{len(expected) * (len(expected) - 1)} links', flush=True)

    cam = open_camera(dev, args.width, args.height, args.fps)
    print(f'camera {cam.name} {cam.w}x{cam.h} @ {cam.fps:g} fps', flush=True)
    app = QApplication(sys.argv)
    w = Live(boards, cam, args)
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
