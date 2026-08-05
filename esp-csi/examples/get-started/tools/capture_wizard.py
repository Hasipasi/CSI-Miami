#!/usr/bin/env python3
"""Guided CSI capture: runs a fixed protocol with large on-screen prompts and
mirrors each phase onto the boards' LEDs.

The LEDs matter more than the screen here -- the operator is in the room with the
boards and often cannot see the PC, and for the empty-room phase has to leave
entirely. Colour code, shown on every board:

    RED     stay out / do not move
    YELLOW  get into position
    BLUE    stand still
    GREEN   move
    WHITE   finished

An earlier attempt at this measurement was spoiled by a "static" baseline that
turned out not to be static (a monotonic drift across the whole window), which
made the motion-versus-static comparison meaningless. So this tool captures a
genuinely empty room as its reference, separates "person present but still" from
"person moving", and checks each baseline for drift before trusting it.

  python3 capture_wizard.py --mode fixedtx --prefix run1
"""

import argparse
import csv
import json
import re
import sys
import threading
import time
from io import StringIO

import numpy as np
import serial
import serial.tools.list_ports

from PyQt5.Qt import *
from PyQt5 import QtCore

WCH_VID = 0x1A86
BAUD = 921600
BOOT_MAC_RE = re.compile(r'Board MAC ([0-9a-fA-F:]{17})')
N_META = 25

RED, YELLOW, BLUE, GREEN, WHITE, OFF = (
    (40, 0, 0), (40, 30, 0), (0, 0, 40), (0, 40, 0), (30, 30, 30), (0, 0, 0))

# (key, seconds, headline, detail, led, save?)
PHASES = [
    ('leave',  20, 'LEAVE THE ROOM NOW',      'Close the door behind you. Boards turn RED.',      RED,    False),
    ('empty',  30, 'EMPTY ROOM - STAY OUT',   'Reference capture. Do not enter.',                 RED,    True),
    ('enter',  20, 'COME BACK IN',            'Stand in the middle of the array.',                YELLOW, False),
    ('still',  25, 'STAND COMPLETELY STILL',  'Arms down. Breathe normally. Do not shift weight.', BLUE,  True),
    ('move',   25, 'MOVE',                    'Arms out and down, turn around, squat and stand.', GREEN,  True),
]


def discover():
    ports = sorted(p.device for p in serial.tools.list_ports.comports() if p.vid == WCH_VID)
    boards, lock = {}, threading.Lock()

    def probe(port):
        try:
            ser = serial.Serial(port, BAUD, timeout=0.3)
        except (serial.SerialException, OSError):
            return
        ser.setDTR(False)
        ser.setRTS(True)
        time.sleep(0.1)
        ser.setRTS(False)
        mac, start = None, time.time()
        while time.time() - start < 3:
            m = BOOT_MAC_RE.search(ser.readline().decode(errors='ignore'))
            if m:
                mac = m.group(1).lower()
                break
        if mac:
            with lock:
                boards[mac] = ser
        else:
            ser.close()

    ts = [threading.Thread(target=probe, args=(p,)) for p in ports]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return boards


def parse(line):
    try:
        row = next(csv.reader(StringIO(line)))
        if len(row) != N_META:
            return None
        n = int(row[-3])
        vals = json.loads(row[-1])
        if n != len(vals) or n < 1:
            return None
        return row[2].lower(), np.array(vals, dtype=np.float32)
    except (ValueError, json.JSONDecodeError, IndexError, StopIteration):
        return None


class Wizard(QWidget):
    def __init__(self, boards, mode, prefix, round_s):
        super().__init__()
        self.boards = boards
        self.macs = sorted(boards)
        self.mode = mode
        self.prefix = prefix
        self.round_s = round_s
        self.phase_i = -1
        self.recs = {m: [] for m in self.macs}
        self.collecting = False
        self.t0 = time.time()
        self.results = {}

        self.resize(1100, 650)
        self.setWindowTitle('CSI capture wizard')

        lay = QVBoxLayout(self)
        self.head = QLabel('READY', alignment=QtCore.Qt.AlignCenter)
        self.head.setStyleSheet('font-size: 64px; font-weight: bold; color: white;')
        self.head.setWordWrap(True)
        self.count = QLabel('', alignment=QtCore.Qt.AlignCenter)
        self.count.setStyleSheet('font-size: 150px; font-weight: bold; color: white;')
        self.detail = QLabel('Press SPACE or click Start', alignment=QtCore.Qt.AlignCenter)
        self.detail.setStyleSheet('font-size: 26px; color: #dddddd;')
        self.detail.setWordWrap(True)
        self.status = QLabel('', alignment=QtCore.Qt.AlignCenter)
        self.status.setStyleSheet('font-size: 18px; color: #bbbbbb;')
        self.start_btn = QPushButton('Start')
        self.start_btn.setStyleSheet('font-size: 28px; padding: 14px;')
        self.start_btn.clicked.connect(self.next_phase)

        for w in (self.head, self.count, self.detail, self.status, self.start_btn):
            lay.addWidget(w)

        self.set_bg('#222222')
        self.set_leds(OFF)

        self.stop_evt = threading.Event()
        for m in self.macs:
            boards[m].reset_input_buffer()
            threading.Thread(target=self.reader, args=(m,), daemon=True).start()
        self.arm_radio()

        self.tick = QtCore.QTimer()
        self.tick.timeout.connect(self.on_tick)
        self.tick.start(100)
        self.deadline = None

    # ---- radio ----

    def arm_radio(self):
        """fixedtx: one board transmits for the whole run. roundrobin: rotate."""
        if self.mode == 'fixedtx':
            tx = self.macs[0]
            self.boards[tx].write(b'TX\n')
            for m in self.macs:
                if m != tx:
                    self.boards[m].write(f'RX {tx.replace(":", "")}\n'.encode())
        else:
            self.rr_i = 0
            self.rr = QtCore.QTimer()
            self.rr.timeout.connect(self.rr_step)
            self.rr.start(int(self.round_s * 1000))
            self.rr_step()

    def rr_step(self):
        tx = self.macs[self.rr_i % len(self.macs)]
        self.rr_i += 1
        self.boards[tx].write(b'TX\n')
        for m in self.macs:
            if m != tx:
                self.boards[m].write(f'RX {tx.replace(":", "")}\n'.encode())

    def reader(self, rx):
        ser = self.boards[rx]
        while not self.stop_evt.is_set():
            try:
                raw = ser.readline()
            except (serial.SerialException, OSError):
                break
            if not raw.startswith(b'CSI_AMP'):
                continue
            if not self.collecting:
                continue
            p = parse(raw.decode(errors='ignore').strip())
            if p:
                self.recs[rx].append((time.time() - self.t0, p[0], p[1]))

    def set_leds(self, rgb):
        for m in self.macs:
            try:
                self.boards[m].write(f'LED {rgb[0]},{rgb[1]},{rgb[2]}\n'.encode())
            except (serial.SerialException, OSError):
                pass

    def set_bg(self, css):
        self.setStyleSheet(f'QWidget {{ background: {css}; }}')

    # ---- protocol ----

    def next_phase(self):
        if self.phase_i >= len(PHASES):
            return  # already finished; SPACE must not walk off the end of PHASES
        if self.phase_i >= 0:
            self.finish_phase()
        self.phase_i += 1
        if self.phase_i >= len(PHASES):
            return self.done()

        key, secs, head, detail, led, _save = PHASES[self.phase_i]
        self.start_btn.hide()
        self.head.setText(head)
        self.detail.setText(detail)
        self.set_bg({RED: '#7f1010', YELLOW: '#7f6000', BLUE: '#10307f',
                     GREEN: '#0d6b1e', WHITE: '#444444'}[led])
        self.set_leds(led)
        self.recs = {m: [] for m in self.macs}
        self.t0 = time.time()
        self.collecting = True
        self.deadline = time.time() + secs

    def finish_phase(self):
        key, _s, _h, _d, _led, save = PHASES[self.phase_i]
        self.collecting = False
        if save:
            path = f'{self.prefix}_{key}.npz'
            n = save_npz(self.recs, path)
            self.results[key] = (path, n)

    def on_tick(self):
        if self.deadline is None:
            return
        left = self.deadline - time.time()
        if left <= 0:
            self.next_phase()
            return
        self.count.setText(str(int(np.ceil(left))))
        n = sum(len(v) for v in self.recs.values())
        nxt = (PHASES[self.phase_i + 1][2] if self.phase_i + 1 < len(PHASES) else 'FINISH')
        self.status.setText(f'{n} packets captured   |   next: {nxt}')

    def done(self):
        self.deadline = None
        self.collecting = False
        self.set_leds(WHITE)
        self.set_bg('#444444')
        self.head.setText('DONE')
        self.count.setText('')
        lines = [f'{k}: {p} ({n} links)' for k, (p, n) in self.results.items()]
        self.detail.setText('\n'.join(lines) + '\n\nYou can close this window.')
        self.status.setText('')
        self.stop_evt.set()


def save_npz(recs, path):
    out = {}
    for rx, items in recs.items():
        by_tx = {}
        for t, tx, amp in items:
            by_tx.setdefault(tx, []).append((t, amp))
        for tx, seq in by_tx.items():
            if len(seq) < 5:
                continue
            lens = {}
            for _t, a in seq:
                lens[len(a)] = lens.get(len(a), 0) + 1
            n = max(lens, key=lens.get)
            seq = [(t, a) for t, a in seq if len(a) == n]
            out[f'{tx}|{rx}|t'] = np.array([t for t, _ in seq], dtype=np.float32)
            out[f'{tx}|{rx}|a'] = np.stack([a for _, a in seq])
    np.savez_compressed(path, **out)
    return sum(1 for k in out if k.endswith('|t'))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['fixedtx', 'roundrobin'], default='fixedtx')
    ap.add_argument('--round-duration', type=float, default=0.05)
    ap.add_argument('--prefix', default='run')
    args = ap.parse_args()

    boards = discover()
    if len(boards) < 2:
        print(f'need >= 2 boards, found {len(boards)}')
        sys.exit(1)
    print(f'{len(boards)} boards: {sorted(boards)}')

    app = QApplication(sys.argv)
    w = Wizard(boards, args.mode, args.prefix, args.round_duration)

    QShortcut(QKeySequence('Space'), w, w.next_phase)
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
