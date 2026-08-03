#!/usr/bin/env python3
# -*-coding:utf-8-*-
"""Round-robin CSI capture with live amplitude visualization.

Combines what roundrobin_control.py and csi_data_read_parse.py each do
separately, because they can't run as separate processes at the same time --
both need to own the same serial connection per board (one to write TX/RX
commands, the other to read CSI_DATA). Here a single connection per board is
opened once and shared: a reader thread pulls CSI lines off it continuously,
while the scheduler (on the same QTimer driving the GUI) writes role commands
to it periodically.

Unlike csi_data_read_parse.py's rolling 3s waterfall, this window is fixed to
[0, --duration] and does not scroll -- the whole run is meant to be reviewed
at once. Each board's own panel gets a blue band over the time ranges where
that board itself held the TX token (so the visible gaps in its own capture,
which happen while it's transmitting rather than listening, are explained
rather than looking like dropouts). The window stays open after the run ends;
close it manually when done.
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
from PyQt5.QtCore import pyqtSignal, QThread
from pyqtgraph import PlotWidget
import pyqtgraph as pg

pg.setConfigOptions(imageAxisOrder='row-major')

WCH_VID = 0x1A86
BAUD = 921600
BOOT_MAC_RE = re.compile(r'Board MAC ([0-9a-fA-F:]{17})')
DATA_COLUMNS_NAMES_C5C6 = ['type', 'id', 'mac', 'rssi', 'rate', 'noise_floor', 'fft_gain', 'agc_gain',
                           'channel', 'local_timestamp', 'sig_len', 'rx_state', 'len', 'first_word', 'data']
DATA_COLUMNS_NAMES = ['type', 'id', 'mac', 'rssi', 'rate', 'sig_mode', 'mcs', 'bandwidth', 'smoothing',
                      'not_sounding', 'aggregation', 'stbc', 'fec_coding', 'sgi', 'noise_floor', 'ampdu_cnt',
                      'channel', 'secondary_channel', 'local_timestamp', 'ant', 'sig_len', 'rx_state', 'len',
                      'first_word', 'data']


def discover_boards():
    ports = sorted(p.device for p in serial.tools.list_ports.comports() if p.vid == WCH_VID)
    boards = {}
    for port in ports:
        ser = serial.Serial(port, BAUD, timeout=0.2)
        ser.setDTR(False)
        ser.setRTS(True)
        time.sleep(0.1)
        ser.setRTS(False)
        mac = None
        start = time.time()
        while time.time() - start < 3:
            line = ser.readline().decode(errors='ignore')
            m = BOOT_MAC_RE.search(line)
            if m:
                mac = m.group(1).lower()
                break
        if mac is None:
            print(f'[{port}] no "Board MAC" boot line seen -- skipping (wrong/no firmware?)')
            ser.close()
            continue
        print(f'[{port}] MAC {mac}')
        boards[port] = {'serial': ser, 'mac': mac}
    return boards


class BoardState:
    def __init__(self, port, ser, mac):
        self.port = port
        self.ser = ser
        self.mac = mac
        self.label = port.rsplit('/', 1)[-1]
        self.lock = threading.Lock()
        # Both records and tx_intervals are stamped with host arrival time, deliberately
        # on the SAME clock. The board's own local_timestamp looks more precise, but it
        # can't be compared against role changes without calibrating two clocks against
        # each other, and getting that wrong makes correctly-captured records appear to
        # land inside the board's own TX span. Since CSI lines and role-change lines
        # arrive interleaved on one FIFO serial stream read by one thread, host arrival
        # order is exact for deciding which side of a role change a record falls on.
        self.records = []          # (elapsed_s, amplitude ndarray)
        self.tx_intervals = []     # [start_s, end_s_or_None]
        self.rssis = []            # per-record RSSI, for comparing reception quality across boards
        self.implicit_closes = 0   # TX intervals closed by inference because a ROLE_RX marker was lost


class ReaderThread(QThread):
    def __init__(self, board: BoardState, t0: float, stop_event: threading.Event):
        super().__init__()
        self.board = board
        self.t0 = t0
        self.stop_event = stop_event

    def run(self):
        ser = self.board.ser
        while not self.stop_event.is_set():
            try:
                raw = ser.readline()
            except (serial.SerialException, OSError):
                break
            if not raw:
                continue
            now = time.time() - self.t0
            line = raw.decode(errors='ignore').strip()
            if not line.startswith('CSI_AMP'):
                # Role changes are taken from the board's own confirmation lines, not from
                # when the PC sent the command -- the command still has to cross the UART
                # and be processed, and at 100ms rounds that lag is a large fraction of a
                # round. Using send-time made legitimately-received records appear to fall
                # inside the board's own TX span.
                if line.startswith('ROLE_TX'):
                    with self.board.lock:
                        self.board.tx_intervals.append([now, None])
                elif line.startswith('ROLE_RX'):
                    with self.board.lock:
                        if self.board.tx_intervals and self.board.tx_intervals[-1][1] is None:
                            self.board.tx_intervals[-1][1] = now
                continue
            try:
                csi_data = next(csv.reader(StringIO(line)))
                if len(csi_data) != len(DATA_COLUMNS_NAMES) and len(csi_data) != len(DATA_COLUMNS_NAMES_C5C6):
                    continue
                n_sub = int(csi_data[-3])
                csi_raw = json.loads(csi_data[-1])
                if n_sub != len(csi_raw) or n_sub < 1:
                    continue
                amp = np.array(csi_raw, dtype=np.float64)  # firmware already sent amplitudes
                rssi = int(csi_data[3])
            except (ValueError, json.JSONDecodeError, IndexError, StopIteration):
                continue

            with self.board.lock:
                # A CSI record proves this board was receiving, so any TX interval still
                # open here must have ended -- the firmware cannot emit CSI while is_tx.
                # Without this, one corrupted/dropped ROLE_RX marker leaves an interval
                # open and it swallows the rest of the run.
                if self.board.tx_intervals and self.board.tx_intervals[-1][1] is None:
                    self.board.tx_intervals[-1][1] = now
                    self.board.implicit_closes += 1
                self.board.records.append((now, amp))
                self.board.rssis.append(rssi)


class RoundRobinViewer(QWidget):
    def __init__(self, boards: dict, duration_s: float, round_s: float, warmup_s: float = 0.0,
                 show_tx_bands: bool = True):
        super().__init__()
        self.boards = boards
        self.duration_s = duration_s
        self.round_s = round_s
        self.warmup_s = warmup_s
        self.show_tx_bands = show_tx_bands
        self.t0 = time.time()
        self.stop_event = threading.Event()
        self._warmup_pending = warmup_s > 0

        # Stacked vertically (one full-width row per board) rather than side by side:
        # every panel shares the same time axis, so this gives the time dimension the
        # full window width and makes the boards directly comparable row-to-row.
        n = len(boards)
        panel_width = 1400
        panel_height = 260
        self.resize(panel_width, panel_height * n)
        self.setWindowTitle(f'Round-Robin CSI Amplitude ({duration_s:.0f}s)')

        self.panels = {}
        waterfall_cmap = pg.colormap.get('viridis')
        for idx, (port, board) in enumerate(boards.items()):
            widget = PlotWidget(self)
            widget.setGeometry(QtCore.QRect(0, idx * panel_height, panel_width, panel_height))
            widget.setTitle(f'CSI Amplitude - {board.label}')
            widget.setLabel('left', 'Subcarrier Index')
            widget.setLabel('bottom', 'Time (s)')
            widget.setXRange(0, duration_s, padding=0)

            img = pg.ImageItem()
            widget.addItem(img)
            img.setColorMap(waterfall_cmap)
            colorbar = pg.ColorBarItem(colorMap=waterfall_cmap, label='Amplitude', values=(0, 180))
            colorbar.setImageItem(img, insert_in=widget.getPlotItem())

            self.panels[port] = {'widget': widget, 'img': img, 'tx_regions': []}

        self.readers = []
        for port, board in boards.items():
            reader = ReaderThread(board, self.t0, self.stop_event)
            reader.start()
            self.readers.append(reader)

        self.ports = list(boards.keys())
        self.current_tx_idx = -1
        self._round_count = 0
        self.commanded_tx_counts = {p: 0 for p in boards}  # ground truth for marker-loss check

        self.round_timer = pg.QtCore.QTimer()
        self.round_timer.timeout.connect(self.do_round)
        first_interval_s = warmup_s if warmup_s > 0 else round_s
        self.round_timer.start(int(first_interval_s * 1000))
        self.do_round()  # kick off round 0 immediately instead of waiting one interval

        self.draw_timer = pg.QtCore.QTimer()
        self.draw_timer.timeout.connect(self.redraw)
        self.draw_timer.start(200)

        self.stop_timer = pg.QtCore.QTimer()
        self.stop_timer.setSingleShot(True)
        self.stop_timer.timeout.connect(self.finish_run)
        self.stop_timer.start(int(duration_s * 1000))

    def do_round(self):
        self._round_count += 1
        self.current_tx_idx = (self.current_tx_idx + 1) % len(self.ports)
        tx_port = self.ports[self.current_tx_idx]
        tx_board = self.boards[tx_port]
        now = time.time() - self.t0
        # tx_intervals are recorded by the reader threads from each board's own
        # confirmation lines, not here -- see ReaderThread.run().
        tx_mac_hex = tx_board.mac.replace(':', '')

        tx_board.ser.write(b'TX\n')
        for port, board in self.boards.items():
            if port != tx_port:
                board.ser.write(f'RX {tx_mac_hex}\n'.encode())

        self.commanded_tx_counts[tx_port] += 1
        print(f'{now:6.2f}s  TX -> {tx_board.label} ({tx_board.mac})')

        if self._warmup_pending and self._round_count >= 2:
            # round 0 fired manually at construction time (still using the timer's
            # warmup-length interval); this is round 1, the timer's first real firing
            # after that warmup gap -- now switch it to the fast steady-state interval.
            self._warmup_pending = False
            self.round_timer.setInterval(int(self.round_s * 1000))

    def redraw(self):
        for port, board in self.boards.items():
            with board.lock:
                records = list(board.records)
            if len(records) >= 2:
                lengths = {len(amp) for _, amp in records}
                target_len = max(lengths, key=lambda l: sum(1 for _, a in records if len(a) == l))
                filtered = [(t, amp) for t, amp in records if len(amp) == target_len]
                if len(filtered) >= 2:
                    times = np.array([t for t, _ in filtered])
                    data = np.array([amp for _, amp in filtered]).T  # [subcarrier, time]
                    img = self.panels[port]['img']
                    img.setImage(data, autoLevels=False)
                    img.setRect(QtCore.QRectF(times[0], 0, max(times[-1] - times[0], 0.01), target_len))
                    self.panels[port]['widget'].setLabel('left', f'Subcarrier Index (0-{target_len - 1})')

            self._update_tx_regions(port, board)

    def _update_tx_regions(self, port, board):
        if not self.show_tx_bands:
            return
        panel = self.panels[port]
        with board.lock:
            intervals = [tuple(iv) for iv in board.tx_intervals]
        while len(panel['tx_regions']) < len(intervals):
            region = pg.LinearRegionItem(brush=(0, 80, 255, 90), movable=False, pen=pg.mkPen((0, 120, 255), width=1))
            panel['widget'].addItem(region)
            panel['tx_regions'].append(region)
        for region, (start, end) in zip(panel['tx_regions'], intervals):
            region.setRegion((start, end if end is not None else (time.time() - self.t0)))

    def finish_run(self):
        self.round_timer.stop()
        self.stop_event.set()  # reader threads exit their loop on next readline() timeout
        now = time.time() - self.t0
        for board in self.boards.values():
            with board.lock:
                if board.tx_intervals and board.tx_intervals[-1][1] is None:
                    board.tx_intervals[-1][1] = now
        self.redraw()  # one final draw with the closed-off intervals
        self._report_self_capture_leakage()
        print(f'\nRun complete ({self.duration_s:.0f}s). Window stays open -- close it manually when done.')

    def _report_self_capture_leakage(self):
        """Reports role-marker loss in BOTH directions: ROLE_TX markers that never arrived
        (interval never opened -- shows up as a missing blue band, which is what makes the
        rotation look out of order) and ROLE_RX markers that never arrived (interval left
        open, closed by inference). An earlier version only tracked the latter and reported
        0% while a quarter of the ROLE_TX markers were in fact being lost."""
        print('\nRole-marker reliability:')
        for port, board in self.boards.items():
            with board.lock:
                observed = len(board.tx_intervals)
                implicit = board.implicit_closes
            expected = self.commanded_tx_counts.get(port, 0)
            missing = max(expected - observed, 0)
            pct = (missing / expected * 100) if expected else 0.0
            flag = '  <-- ROLE_TX markers lost' if missing else ''
            print(f'  [{board.label}] TX turns commanded {expected}, intervals seen {observed} '
                  f'({missing} missing, {pct:.0f}%); {implicit} closed by inference{flag}')

        print('\nReception quality per board:')
        for port, board in self.boards.items():
            with board.lock:
                rssis = list(board.rssis)
                n = len(board.records)
            rate = n / self.duration_s
            if rssis:
                print(f'  [{board.label}] {n} records ({rate:.0f}/s), RSSI avg {np.mean(rssis):.1f} dBm '
                      f'(min {min(rssis)}, max {max(rssis)})')
            else:
                print(f'  [{board.label}] no records')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--duration', type=float, default=10.0, help='Total run length in seconds (default: 10)')
    parser.add_argument('-d', '--round-duration', type=float, default=1.0, help='Seconds per TX turn (default: 1.0)')
    parser.add_argument('--no-tx-bands', dest='show_tx_bands', action='store_false',
                         help='Hide the blue TX-span overlays (they are still tracked, and the '
                              'self-capture leak check still runs) -- useful for viewing the raw amplitude data')
    parser.add_argument('-w', '--warmup', type=float, default=0.0,
                         help='If set, the first round runs this long (seconds) before switching to '
                              '--round-duration for the rest -- lets things settle before a fast handoff rate')
    args = parser.parse_args()

    boards_raw = discover_boards()
    if len(boards_raw) < 2:
        print(f'Found {len(boards_raw)} board(s) -- need at least 2. Exiting.')
        sys.exit(1)

    boards = {port: BoardState(port, b['serial'], b['mac']) for port, b in boards_raw.items()}
    print(f'\n{len(boards)} boards. Round duration {args.round_duration}s, total run {args.duration}s.\n')

    app = QApplication(sys.argv)
    window = RoundRobinViewer(boards, args.duration, args.round_duration, args.warmup, args.show_tx_bands)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
