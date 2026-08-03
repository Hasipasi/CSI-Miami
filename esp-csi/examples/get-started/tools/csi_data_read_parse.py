#!/usr/bin/env python3
# -*-coding:utf-8-*-

# SPDX-FileCopyrightText: 2021-2025 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
#

# WARNING: we don't check for Python build-time dependencies until
# check_environment() function below. If possible, avoid importing
# any external libraries here - put in external script, or import in
# their specific function instead.

import sys
import csv
import json
import argparse
import collections
import pandas as pd
import numpy as np

import serial
import serial.tools.list_ports
from os import path
from io import StringIO

from PyQt5.Qt import *
from pyqtgraph import PlotWidget
from PyQt5 import QtCore
import pyqtgraph as pg

# array axis 0 (rows) maps to image Y, axis 1 (columns) maps to image X
pg.setConfigOptions(imageAxisOrder='row-major')
from pyqtgraph import ScatterPlotItem
from PyQt5.QtCore import pyqtSignal, QThread
import threading
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import linregress
import statsmodels.api as sm

# Reduce displayed waveforms to avoid display freezes
CSI_VAID_SUBCARRIER_INTERVAL = 1
csi_vaid_subcarrier_len =0

CSI_DATA_INDEX = 200  # buffer size
CSI_DATA_COLUMNS = 490
DATA_COLUMNS_NAMES_C5C6 = ['type', 'id', 'mac', 'rssi', 'rate','noise_floor','fft_gain','agc_gain', 'channel', 'local_timestamp',  'sig_len', 'rx_state', 'len', 'first_word', 'data']
DATA_COLUMNS_NAMES = ['type', 'id', 'mac', 'rssi', 'rate', 'sig_mode', 'mcs', 'bandwidth', 'smoothing', 'not_sounding', 'aggregation', 'stbc', 'fec_coding',
                      'sgi', 'noise_floor', 'ampdu_cnt', 'channel', 'secondary_channel', 'local_timestamp', 'ant', 'sig_len', 'rx_state', 'len', 'first_word', 'data']

WATERFALL_WINDOW_US = 3_000_000  # 3 seconds, measured using the device's own local_timestamp

# WCH/QinHeng (the CH34x USB-serial chip on our boards). Used to filter which serial
# ports are even worth probing as CSI boards.
WCH_VID = 0x1A86

SCAN_INTERVAL_MS = 1500
PROBE_WINDOW_S = 1.5

# Matches the firmware's status LED palette (app_main.c init_status_led): bright display
# equivalents of the dim RGB values actually driven onto the physical WS2812.
LED_COLOR_HEX = {
    'red': '#ff4d4d',
    'orange': '#ffa64d',
    'lime': '#ccff4d',
    'green': '#4dff4d',
    'teal': '#4dffb3',
    'cyan': '#4dffff',
    'azure': '#4db3ff',
    'blue': '#4d94ff',
    'violet': '#b34dff',
    'pink': '#ff4d94',
    'white': '#e6e6e6',
}
DEFAULT_PANEL_COLOR = '#aaaaaa'  # used until a board's boot-time LED line is seen


class ReceiverBuffers:
    """Per-receiver state. One instance per connected csi_recv board so multiple
    receivers can be read concurrently without stepping on each other's data."""

    def __init__(self):
        self.csi_data_complex = np.zeros([CSI_DATA_INDEX, CSI_DATA_COLUMNS], dtype=np.complex64)
        self.agc_gain_data = np.zeros([CSI_DATA_INDEX], dtype=np.float64)
        self.fft_gain_data = np.zeros([CSI_DATA_INDEX], dtype=np.float64)
        self.fft_gains = []
        self.agc_gains = []
        # (local_timestamp_us, amplitude[valid_subcarriers]) per packet, for the waterfall panel.
        # Kept separate from csi_data_complex so the waterfall's time span isn't tied to
        # the fixed CSI_DATA_INDEX buffer used by the amplitude/phase panels.
        self.waterfall_records = collections.deque(maxlen=20000)


class csi_data_graphical_window(QWidget):
    def __init__(self, store_file: str, log_file: str):
        """Starts with zero receivers. A QTimer periodically scans for WCH-vendor
        serial ports, probes new ones to tell csi_recv boards apart from the sender
        (or anything unflashed), and adds/removes a receiver's panel column live as
        it's plugged in or unplugged -- no ports need to be specified up front."""
        super().__init__()
        self.store_file = store_file
        self.log_file = log_file

        self.resize(1280, 1200)
        self.setWindowTitle('CSI Viewer')

        self.panels = []              # ordered list of panel dicts, one per active receiver
        self.threads = {}             # port -> SubThread
        self.probing = set()          # ports currently being identified
        self.classified = {}          # port -> 'send' | 'unknown' (skipped, not re-probed until replugged)
        self._probe_threads = []      # keepalive refs so QThreads aren't GC'd mid-run

        self.placeholder = QLabel('No CSI receivers detected yet...', self)
        self.placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.placeholder.setStyleSheet('color: #888; font-size: 18px;')
        self.placeholder.setGeometry(QtCore.QRect(0, 0, 1280, 1200))

        self.data_timer = pg.QtCore.QTimer()
        self.data_timer.timeout.connect(self.update_data)
        self.data_timer.start(100)

        self.scan_timer = pg.QtCore.QTimer()
        self.scan_timer.timeout.connect(self.scan_ports)
        self.scan_timer.start(SCAN_INTERVAL_MS)
        self.scan_ports()  # initial scan so already-connected boards show up immediately

    # ---- hot-plug scanning -------------------------------------------------

    def scan_ports(self):
        current = {p.device for p in serial.tools.list_ports.comports() if p.vid == WCH_VID}

        # Forget classifications for ports that vanished, so a future replug (possibly
        # a different physical board reusing the same /dev path) gets re-probed fresh.
        for port in list(self.classified):
            if port not in current:
                del self.classified[port]

        # Fallback removal: normally a receiver's own SubThread notices the disconnect
        # and calls remove_receiver itself; this just catches anything that slipped by.
        for port in list(self.threads):
            if port not in current:
                self.remove_receiver(port)

        new_ports = current - set(self.threads) - self.probing - set(self.classified)
        for port in new_ports:
            self.probing.add(port)
            probe = ProbeThread(port)
            probe.probe_done.connect(self.on_probe_done)
            probe.finished.connect(lambda p=probe: self._probe_threads.remove(p) if p in self._probe_threads else None)
            self._probe_threads.append(probe)
            probe.start()

    def on_probe_done(self, port, role):
        self.probing.discard(port)
        if role == 'recv':
            self.add_receiver(port)
        else:
            self.classified[port] = role
            print(f'[{port}] identified as "{role}", not a CSI receiver -- skipping')

    # ---- receiver lifecycle -------------------------------------------------

    def add_receiver(self, port):
        if port in self.threads:
            return

        buffers = ReceiverBuffers()
        label = path.basename(port)
        panel = self._build_panel(label, buffers)
        self.panels.append(panel)

        store_file = per_receiver_filename(self.store_file, port)
        log_file = per_receiver_filename(self.log_file, port)
        thread = SubThread(port, store_file, log_file, buffers)
        on_colors_ready, on_led_ready = self._make_slots(panel)
        thread.data_ready.connect(on_colors_ready)
        thread.led_ready.connect(on_led_ready)
        thread.disconnected.connect(self.remove_receiver)
        self.threads[port] = thread
        thread.start()

        print(f'[{port}] receiver connected')
        self.relayout()

    def remove_receiver(self, port):
        thread = self.threads.pop(port, None)
        if thread is None:
            return
        print(f'[{port}] receiver disconnected')

        panel = next((p for p in self.panels if p['label'] == path.basename(port)), None)
        if panel is not None:
            self.panels.remove(panel)
            for widget in (panel['amp_widget'], panel['phase_widget'], panel['waterfall_widget']):
                widget.hide()
                widget.deleteLater()

        self.relayout()

    def relayout(self):
        n = len(self.panels)
        self.placeholder.setVisible(n == 0)
        if n == 0:
            return

        panel_width = 1280 // n
        row_height = 400
        for idx, panel in enumerate(self.panels):
            x = idx * panel_width
            panel['amp_widget'].setGeometry(QtCore.QRect(x, 0, panel_width, row_height))
            panel['phase_widget'].setGeometry(QtCore.QRect(x, row_height, panel_width, row_height))
            panel['waterfall_widget'].setGeometry(QtCore.QRect(x, row_height * 2, panel_width, row_height))

    # ---- panel construction -------------------------------------------------

    def _build_panel(self, label, buffers):
        amp_widget = PlotWidget(self)
        amp_widget.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis)
        amp_widget.setTitle(f'CSI Amplitude - {label}')
        amp_widget.setLabel('left', 'Amplitude')
        amp_widget.setLabel('bottom', 'Time (packet count)')
        amp_curves = []
        amp_curves.append(amp_widget.plot(buffers.agc_gain_data, name='AGC Gain', pen=[255, 255, 0]))
        amp_curves.append(amp_widget.plot(buffers.fft_gain_data, name='FFT Gain', pen=[255, 255, 0]))
        amplitude0 = np.abs(buffers.csi_data_complex)
        for i in range(CSI_DATA_COLUMNS):
            amp_curves.append(amp_widget.plot(amplitude0[:, i], pen=(255, 255, 255)))
        amp_widget.show()

        phase_widget = PlotWidget(self)
        phase_widget.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis)
        phase_widget.setLabel('left', 'Phase (rad)')
        phase_widget.setLabel('bottom', 'Time (packet count)')
        phase_curves = []
        phase0 = np.angle(buffers.csi_data_complex)
        for i in range(CSI_DATA_COLUMNS):
            phase_curves.append(phase_widget.plot(phase0[:, i], pen=(255, 255, 255)))
        phase_widget.show()

        waterfall_widget = PlotWidget(self)
        waterfall_widget.setLabel('left', 'Subcarrier Index')
        waterfall_widget.setLabel('bottom', 'Time (s ago)')
        waterfall_img = pg.ImageItem()
        waterfall_widget.addItem(waterfall_img)
        waterfall_cmap = pg.colormap.get('viridis')
        waterfall_img.setColorMap(waterfall_cmap)
        # Fixed scale (not renormalized per-frame): raw CSI I/Q values are signed 8-bit
        # (+/-128), so amplitude = sqrt(I^2+Q^2) tops out around 181; 0-180 covers the
        # full practical range without needing to rescale as the signal changes.
        waterfall_colorbar = pg.ColorBarItem(colorMap=waterfall_cmap, label='Amplitude', values=(0, 180))
        waterfall_colorbar.setImageItem(waterfall_img, insert_in=waterfall_widget.getPlotItem())
        waterfall_widget.show()

        panel = {
            'label': label,
            'buffers': buffers,
            'deta_len': 0,
            'amp_widget': amp_widget,
            'amp_curves': amp_curves,
            'phase_widget': phase_widget,
            'phase_curves': phase_curves,
            'waterfall_widget': waterfall_widget,
            'waterfall_img': waterfall_img,
        }
        self._set_led_color(panel, None)  # neutral titles until the boot line arrives
        return panel

    def _set_led_color(self, panel, color_name):
        """Colors the phase and waterfall panel titles to match a board's physical
        status LED, so a receiver's panels can be matched to the board by eye."""
        hexcolor = LED_COLOR_HEX.get(color_name, DEFAULT_PANEL_COLOR)
        suffix = f' [{color_name}]' if color_name else ''
        panel['phase_widget'].setTitle(f'CSI Phase - {panel["label"]}{suffix}', color=hexcolor)
        panel['waterfall_widget'].setTitle(f'CSI Waterfall - {panel["label"]}{suffix} (last 3s)', color=hexcolor)

    def _make_slots(self, panel):
        """Per-receiver Qt slots, bound to one panel, for that receiver's SubThread signals."""
        def on_colors_ready(color_list):
            panel['deta_len'] = len(color_list)
            for i in range(panel['deta_len']):
                panel['amp_curves'][i + 2].setPen(color_list[i])  # +2 skips the AGC/FFT gain curves
                panel['phase_curves'][i].setPen(color_list[i])

        def on_led_ready(color_name):
            self._set_led_color(panel, color_name)

        return on_colors_ready, on_led_ready

    # ---- live redraw -------------------------------------------------

    def update_data(self):
        for panel in self.panels:
            buffers = panel['buffers']
            csi_data_complex = buffers.csi_data_complex
            amplitude = np.abs(csi_data_complex)
            phase = np.angle(csi_data_complex)

            panel['amp_curves'][CSI_DATA_COLUMNS].setData(buffers.agc_gain_data)
            panel['amp_curves'][CSI_DATA_COLUMNS + 1].setData(buffers.fft_gain_data)
            for i in range(CSI_DATA_COLUMNS):
                panel['amp_curves'][i + 2].setData(amplitude[:, i])
                panel['phase_curves'][i].setData(phase[:, i])

            self.update_waterfall(panel)

    def update_waterfall(self, panel):
        records = panel['buffers'].waterfall_records
        if not records:
            return

        latest_ts, target_len = records[-1][0], records[-1][1].shape[0]

        # Walk newest-to-oldest, stop once we leave the 3s window (avoids scanning the whole buffer)
        windowed = []
        for ts, amp in reversed(records):
            if latest_ts - ts > WATERFALL_WINDOW_US:
                break
            if amp.shape[0] == target_len:
                windowed.append((ts, amp))
        windowed.reverse()

        if len(windowed) < 2:
            return

        waterfall_data = np.array([amp for ts, amp in windowed]).T  # [subcarrier, time]
        panel['waterfall_img'].setImage(waterfall_data, autoLevels=False)  # fixed 0-180 scale, set once at init

        # Map image pixels to real axes: Y = subcarrier index, X = seconds ago (0 = now)
        span_s = (latest_ts - windowed[0][0]) / 1e6
        if span_s > 0:
            panel['waterfall_img'].setRect(QtCore.QRectF(-span_s, 0, span_s, target_len))
        panel['waterfall_widget'].setLabel('left', f'Subcarrier Index (0-{target_len - 1})')

def generate_subcarrier_colors(red_range, green_range, yellow_range, total_num,interval=1):
    colors = []
    for i in range(total_num):
        if red_range and red_range[0] <= i <= red_range[1]:
            intensity = int(255 * (i - red_range[0]) / (red_range[1] - red_range[0]))
            colors.append((intensity, 0, 0))
        elif green_range and green_range[0] <= i <= green_range[1]:
            intensity = int(255 * (i - green_range[0]) / (green_range[1] - green_range[0]))
            colors.append((0, intensity, 0))
        elif yellow_range and yellow_range[0] <= i <= yellow_range[1]:
            intensity = int(255 * (i - yellow_range[0]) / (yellow_range[1] - yellow_range[0]))
            colors.append((0, intensity, intensity))
        else:
            colors.append((200, 200, 200))

    return colors


class ProbeThread(QThread):
    """Briefly connects to a newly-seen serial port to tell a csi_recv board apart
    from the csi_send sender (or an unflashed/unrelated device), so the viewer only
    ever creates panels for actual CSI receivers."""
    probe_done = pyqtSignal(str, str)  # port, role in {'recv', 'send', 'unknown'}

    def __init__(self, port):
        super().__init__()
        self.port = port

    def run(self):
        role = 'unknown'
        try:
            ser = serial.Serial(self.port, 921600, timeout=0.5)
            ser.setDTR(False)
            ser.setRTS(True)
            time.sleep(0.1)
            ser.setRTS(False)
            start = time.time()
            while time.time() - start < PROBE_WINDOW_S:
                line = ser.readline().decode(errors='ignore')
                if 'csi_recv:' in line:
                    role = 'recv'
                    break
                if 'csi_send:' in line:
                    role = 'send'
                    break
            ser.close()
        except (serial.SerialException, OSError):
            role = 'unknown'
        self.probe_done.emit(self.port, role)


def csi_data_read_parse(port: str, csv_writer, log_file_fd, buffers: ReceiverBuffers,
                         callback=None, led_callback=None):
    set = serial.Serial(port=port, baudrate=921600,bytesize=8, parity='N', stopbits=1)
    # Force a clean reset so we always capture the boot-time "status LED" line, regardless
    # of whether the board was already running before this script attached to it. Read
    # starts immediately after releasing reset (no flush/settle delay) so nothing the
    # firmware prints in the first few milliseconds of boot gets discarded.
    set.setDTR(False)
    set.setRTS(True)
    time.sleep(0.1)
    set.setRTS(False)
    count =0
    if set.isOpen():
        print(f'[{port}] open success')
    else:
        print(f'[{port}] open failed')
        return
    try:
        while True:
            strings = str(set.readline())
            if not strings:
                break
            strings = strings.lstrip('b\'').rstrip('\\r\\n\'')
            index = strings.find('CSI_DATA')

            if index == -1:
                if led_callback and 'status LED:' in strings:
                    led_callback(strings.rsplit('status LED:', 1)[-1].strip())
                log_file_fd.write(strings + '\n')
                log_file_fd.flush()
                continue

            csv_reader = csv.reader(StringIO(strings))
            csi_data = next(csv_reader)
            if len(csi_data) != len(DATA_COLUMNS_NAMES) and len(csi_data) != len(DATA_COLUMNS_NAMES_C5C6):
                print(f'[{port}] element number is not equal',len(csi_data),len(DATA_COLUMNS_NAMES) )
                # print(csi_data)
                log_file_fd.write('element number is not equal\n')
                log_file_fd.write(strings + '\n')
                log_file_fd.flush()
                continue

            try:
                csi_data_len = int(csi_data[-3])
            except ValueError:
                print(f'[{port}] csi_data_len is not a number')
                log_file_fd.write('csi_data_len is not a number\n')
                log_file_fd.write(strings + '\n')
                log_file_fd.flush()
                continue

            try:
                csi_raw_data = json.loads(csi_data[-1])
            except json.JSONDecodeError:
                print(f'[{port}] data is incomplete')
                log_file_fd.write('data is incomplete\n')
                log_file_fd.write(strings + '\n')
                log_file_fd.flush()
                continue
            if csi_data_len != len(csi_raw_data):
                print(f'[{port}] csi_data_len is not equal',csi_data_len,len(csi_raw_data))
                log_file_fd.write('csi_data_len is not equal\n')
                log_file_fd.write(strings + '\n')
                log_file_fd.flush()
                continue

            fft_gain = int(csi_data[6])
            agc_gain = int(csi_data[7])

            buffers.fft_gains.append(fft_gain)
            buffers.agc_gains.append(agc_gain)

            csv_writer.writerow(csi_data)

            # Rotate data to the left
            csi_data_complex = buffers.csi_data_complex
            csi_data_complex[:-1] = csi_data_complex[1:]
            buffers.agc_gain_data[:-1] = buffers.agc_gain_data[1:]
            buffers.fft_gain_data[:-1] = buffers.fft_gain_data[1:]
            buffers.agc_gain_data[-1] = agc_gain
            buffers.fft_gain_data[-1] = fft_gain

            if count ==0:
                count = 1
                print(f'[{port}] CSI frame length {csi_data_len} -> {csi_data_len // 2} valid subcarriers')
                if callback:
                    if csi_data_len == 106:
                        colors = generate_subcarrier_colors((0,25), (27,53), None, len(csi_raw_data))
                    elif  csi_data_len == 114:
                        colors = generate_subcarrier_colors((0,27), (29,56), None, len(csi_raw_data))
                    elif  csi_data_len == 52:
                        colors = generate_subcarrier_colors((0,12), (13,26), None, len(csi_raw_data))
                    elif  csi_data_len == 234 :
                        colors = generate_subcarrier_colors((0,28), (29,56), (60,116), len(csi_raw_data))
                    elif  csi_data_len == 228 :
                        colors = generate_subcarrier_colors((0,28), (29,57), (57,113), len(csi_raw_data))
                    elif  csi_data_len == 490 :
                        colors = generate_subcarrier_colors((0,61), (62,122), (123,245), len(csi_raw_data))
                    elif  csi_data_len == 128 :
                        colors = generate_subcarrier_colors((0,31), (32,63), None, len(csi_raw_data))
                    elif  csi_data_len == 256 :
                        colors = generate_subcarrier_colors((0,32), (32,63), (64,128), len(csi_raw_data))
                    elif  csi_data_len == 512 :
                        colors = generate_subcarrier_colors((0,63), (64,127), (128,256), len(csi_raw_data))
                    elif  csi_data_len == 384 :
                        colors = generate_subcarrier_colors((0,63), (64,127), (128,192), len(csi_raw_data))
                    elif csi_data_len > 0 and csi_data_len <= 612:
                        raw_len = len(csi_raw_data)
                        colors = generate_subcarrier_colors((0,raw_len//2), (raw_len//2+1,raw_len-1), None, raw_len)
                    callback(colors)

            for i in range(csi_data_len // 2):
                csi_data_complex[-1][i] = complex(csi_raw_data[i * 2 + 1],
                                                csi_raw_data[i * 2])

            columns = DATA_COLUMNS_NAMES if len(csi_data) == len(DATA_COLUMNS_NAMES) else DATA_COLUMNS_NAMES_C5C6
            local_timestamp = float(csi_data[columns.index('local_timestamp')])
            valid_subcarriers = csi_data_len // 2
            buffers.waterfall_records.append(
                (local_timestamp, np.abs(csi_data_complex[-1][:valid_subcarriers]).copy()))
    except (serial.SerialException, OSError) as e:
        print(f'[{port}] disconnected ({e})')
    finally:
        try:
            set.close()
        except Exception:
            pass


class SubThread (QThread):
    data_ready = pyqtSignal(object)
    led_ready = pyqtSignal(str)
    disconnected = pyqtSignal(str)

    def __init__(self, serial_port, save_file_name, log_file_name, buffers: ReceiverBuffers):
        super().__init__()
        self.serial_port = serial_port
        self.buffers = buffers

        save_file_fd = open(save_file_name, 'w')
        self.log_file_fd = open(log_file_name, 'w')
        self.csv_writer = csv.writer(save_file_fd)
        self.csv_writer.writerow(DATA_COLUMNS_NAMES)

    def run(self):
        try:
            csi_data_read_parse(self.serial_port, self.csv_writer, self.log_file_fd,
                                 self.buffers, callback=self.data_ready.emit,
                                 led_callback=self.led_ready.emit)
        finally:
            self.disconnected.emit(self.serial_port)

    def __del__(self):
        self.wait()
        self.log_file_fd.close()


def per_receiver_filename(base: str, port: str) -> str:
    root, ext = path.splitext(base)
    tag = path.basename(port)
    return f'{root}_{tag}{ext}'


if __name__ == '__main__':
    if sys.version_info < (3, 6):
        print(' Python version should >= 3.6')
        exit()

    parser = argparse.ArgumentParser(
        description='Auto-detects connected csi_recv boards over USB and displays their '
                     'CSI data graphically. Receivers are added/removed live as they are '
                     'plugged in or unplugged -- no ports need to be specified.')
    parser.add_argument('-s', '--store', dest='store_file', action='store', default='./csi_data.csv',
                        help='Save the data printed by the serial port to a file (per-receiver suffix added automatically)')
    parser.add_argument('-l', '--log', dest='log_file', action='store', default='./csi_data_log.txt',
                        help='Save other serial data the bad CSI data to a log file (per-receiver suffix added automatically)')

    args = parser.parse_args()

    app = QApplication(sys.argv)

    window = csi_data_graphical_window(args.store_file, args.log_file)
    window.show()

    sys.exit(app.exec())
