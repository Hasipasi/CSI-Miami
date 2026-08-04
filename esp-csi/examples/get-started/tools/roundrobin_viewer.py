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
        # (elapsed_s, transmitter_mac, amplitude). The transmitter matters: binning purely
        # by time averages packets from whichever board happened to hold the token in that
        # window, so a bin spanning a handoff blends two physically different links into
        # one column. Keyed by TX mac, each link is binned separately and stays a real
        # single-link measurement regardless of round duration.
        self.records = []
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
                tx_mac = csi_data[2].lower()
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
                self.board.records.append((now, tx_mac, amp))
                self.board.rssis.append(rssi)


class RoundRobinViewer(QWidget):
    def __init__(self, boards: dict, duration_s: float, round_s: float, warmup_s: float = 0.0,
                 show_tx_bands: bool = True, bin_hz: float = 10.0):
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

        # Uniform time grid for the waterfalls, one column per integration bin. Each bin
        # averages every packet that landed in it (~4 packets per 100ms bin at 50Hz with
        # an 80% RX duty cycle), which fills in the single-packet dropouts that made the
        # image look speckled. Bins with no packets at all stay empty -- notably a board's
        # own TX turns, which should read as genuinely blank rather than averaged-over.
        self.col_dt = 1.0 / bin_hz
        self.n_cols = max(int(round(duration_s * bin_hz)), 1)

        self.panels = {}
        waterfall_cmap = pg.colormap.get('viridis')
        self.lut = waterfall_cmap.getLookupTable(0.0, 1.0, 256)[:, :3]
        for idx, (port, board) in enumerate(boards.items()):
            widget = PlotWidget(self)
            widget.setGeometry(QtCore.QRect(0, idx * panel_height, panel_width, panel_height))
            widget.setTitle(f'CSI Amplitude - {board.label}  (rows: one block per transmitter)')
            widget.setLabel('left', 'Transmitter')
            widget.setLabel('bottom', 'Time (s)')
            widget.setXRange(0, duration_s, padding=0)

            img = pg.ImageItem()
            widget.addItem(img)
            img.setColorMap(waterfall_cmap)
            colorbar = pg.ColorBarItem(colorMap=waterfall_cmap, label='Amplitude', values=(0, 180))
            colorbar.setImageItem(img, insert_in=widget.getPlotItem())

            # Every board except this one -- a receiver never captures its own transmissions.
            peers = [(b.mac, b.label) for p, b in boards.items() if p != port]
            self.panels[port] = {'widget': widget, 'img': img, 'tx_regions': [],
                                 'peers': peers, 'separators': []}

        self.readers = []
        for port, board in boards.items():
            # Drop anything buffered from discovery or a previous run, or a stale ROLE_TX
            # gets attributed to this run and shows up as an extra TX interval.
            board.ser.reset_input_buffer()
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
            panel = self.panels[port]
            if len(records) >= 2:
                lengths = {}
                for _, _, amp in records:
                    lengths[len(amp)] = lengths.get(len(amp), 0) + 1
                target_len = max(lengths, key=lengths.get)

                # Group by transmitter first, so each link is binned on its own and a bin
                # spanning a handoff can't average two different links together.
                by_tx = {}
                for t, mac, amp in records:
                    if len(amp) == target_len:
                        by_tx.setdefault(mac, []).append((t, amp))

                peers = panel['peers']
                composite = np.full((len(peers) * target_len, self.n_cols), np.nan)
                for pi, (mac, _label) in enumerate(peers):
                    link = by_tx.get(mac)
                    if not link:
                        continue
                    acc = np.zeros((target_len, self.n_cols))
                    cnt = np.zeros(self.n_cols)
                    for t, amp in link:
                        c = int(t / self.col_dt)
                        if 0 <= c < self.n_cols:
                            acc[:, c] += amp
                            cnt[c] += 1
                    # mean over each bin; bins that caught nothing stay NaN -> transparent
                    block = np.where(cnt[None, :] > 0, acc / np.maximum(cnt, 1)[None, :], np.nan)
                    composite[pi * target_len:(pi + 1) * target_len, :] = block

                img = panel['img']
                img.setImage(self._to_rgba(composite), autoLevels=False)
                img.setRect(QtCore.QRectF(0, 0, self.duration_s, len(peers) * target_len))
                panel['widget'].getAxis('left').setTicks(
                    [[(pi * target_len + target_len / 2.0, label) for pi, (_m, label) in enumerate(peers)]])
                self._update_separators(panel, len(peers), target_len)

            self._update_tx_regions(port, board)

    def _update_separators(self, panel, n_peers, block_h):
        """Horizontal rules between per-transmitter blocks, so the stacked strips read as
        separate links rather than one continuous subcarrier axis."""
        while len(panel['separators']) < n_peers - 1:
            line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen((160, 160, 160), width=1))
            panel['widget'].addItem(line)
            panel['separators'].append(line)
        for i, line in enumerate(panel['separators']):
            line.setPos((i + 1) * block_h)

    def _to_rgba(self, grid):
        """Map amplitudes through the colormap on the fixed 0-180 scale, with NaN cells
        (no packet captured in that time slot) rendered fully transparent so real gaps
        stay visibly empty rather than being filled by neighbouring samples."""
        norm = np.clip(grid / 180.0, 0.0, 1.0)
        idx = np.where(np.isnan(norm), 0, (np.nan_to_num(norm) * 255)).astype(np.uint8)
        rgb = self.lut[idx]                       # (rows, cols, 3)
        alpha = np.where(np.isnan(grid), 0, 255).astype(np.uint8)[:, :, None]
        return np.concatenate([rgb, alpha], axis=2)

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
            # Report the mismatch signed, in both directions. Clamping at zero hid extra
            # intervals (stale buffered markers) behind a reassuring "0 missing".
            delta = observed - expected
            pct = (abs(delta) / expected * 100) if expected else 0.0
            if delta < 0:
                note = f'({-delta} ROLE_TX lost, {pct:.0f}%)  <-- markers lost'
            elif delta > 0:
                note = f'({delta} extra, {pct:.0f}%)  <-- unexpected extra markers'
            else:
                note = '(exact match)'
            print(f'  [{board.label}] TX turns commanded {expected}, intervals seen {observed} '
                  f'{note}; {implicit} closed by inference')

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

        # Per-link, because the aggregate hides which specific pairs are weak, and because
        # the refresh rate that matters for interpreting the waterfall is per-link, not
        # per-board: with N boards only one transmits at a time, so each link is refreshed
        # once per full cycle (N * round_duration), not once per round.
        cycle_s = self.round_s * len(self.ports)
        print(f'\nPer-link capture (cycle {cycle_s * 1000:.0f}ms -> {1 / cycle_s:.1f} Hz per link, '
              f'binning at {1 / self.col_dt:.0f} Hz):')
        for port, board in self.boards.items():
            with board.lock:
                records = list(board.records)
            counts = {}
            for _t, mac, _amp in records:
                counts[mac] = counts.get(mac, 0) + 1
            parts = []
            for mac, label in self.panels[port]['peers']:
                n = counts.get(mac, 0)
                parts.append(f'{label} {n / self.duration_s:5.1f}/s')
            print(f'  [{board.label}] <- ' + '  '.join(parts))
        # Time-resolved, because a single average over the whole run hides exactly the
        # thing a moving-board experiment is trying to show.
        win = 10.0
        n_win = max(int(np.ceil(self.duration_s / win)), 1)
        if n_win > 1:
            print(f'\nPer-link rate in {win:.0f}s windows (rec/s):')
            header = '  ' + ' ' * 22 + ''.join(f'{i * win:>7.0f}s' for i in range(n_win))
            print(header)
            for port, board in self.boards.items():
                with board.lock:
                    records = list(board.records)
                for mac, label in self.panels[port]['peers']:
                    buckets = [0] * n_win
                    for t, m, _amp in records:
                        if m == mac:
                            w = int(t / win)
                            if 0 <= w < n_win:
                                buckets[w] += 1
                    cells = ''.join(f'{c / win:>8.1f}' for c in buckets)
                    print(f'  {board.label} <- {label:<10}' + cells)

        if 1 / self.col_dt > 1 / cycle_s:
            duty = self.round_s / cycle_s
            print(f'  NOTE: binning ({1 / self.col_dt:.0f} Hz) is faster than the per-link refresh '
                  f'({1 / cycle_s:.1f} Hz). Each link only receives during its transmitter\'s turn, '
                  f'so ~{duty * 100:.0f}% of bins hold that burst and the rest are empty -- the strips '
                  f'will look striped. For continuous strips use --bin-hz {1 / cycle_s:.2g} or lower '
                  f'(one bin per full cycle), or shorten --round-duration to cycle faster.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--duration', type=float, default=10.0, help='Total run length in seconds (default: 10)')
    parser.add_argument('-d', '--round-duration', type=float, default=1.0, help='Seconds per TX turn (default: 1.0)')
    parser.add_argument('--no-tx-bands', dest='show_tx_bands', action='store_false',
                         help='Hide the blue TX-span overlays (they are still tracked, and the '
                              'self-capture leak check still runs) -- useful for viewing the raw amplitude data')
    parser.add_argument('-b', '--bin-hz', type=float, default=10.0,
                         help='Waterfall integration rate in Hz (default: 10 = 100ms bins). Each bin '
                              'averages the packets that landed in it; lower = smoother but coarser '
                              'in time, higher = more detail but more single-packet dropouts')
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
    window = RoundRobinViewer(boards, args.duration, args.round_duration, args.warmup,
                              args.show_tx_bands, args.bin_hz)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
