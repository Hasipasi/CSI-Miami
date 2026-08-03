#!/usr/bin/env python3
"""Measure the full directed link matrix between all connected csi_roundrobin boards.

Each board takes a turn transmitting while every other board listens, giving a
capture rate and RSSI for all N*(N-1) directed links. Results are keyed by MAC,
not by /dev/ttyACM path, so they stay comparable across USB re-plugs and hub
reshuffles -- which is what makes this usable as a before/after test when moving
hardware around.

  python3 pairwise_matrix.py --save baseline.json
  python3 pairwise_matrix.py --compare baseline.json
"""

import argparse
import json
import re
import sys
import threading
import time

import serial
import serial.tools.list_ports

WCH_VID = 0x1A86
BAUD = 921600
BOOT_MAC_RE = re.compile(r'Board MAC ([0-9a-fA-F:]{17})')


def discover():
    ports = sorted(p.device for p in serial.tools.list_ports.comports() if p.vid == WCH_VID)
    boards = {}
    for port in ports:
        ser = serial.Serial(port, BAUD, timeout=0.5)
        ser.setDTR(False)
        ser.setRTS(True)
        time.sleep(0.1)
        ser.setRTS(False)
        mac = None
        start = time.time()
        while time.time() - start < 3:
            m = BOOT_MAC_RE.search(ser.readline().decode(errors='ignore'))
            if m:
                mac = m.group(1).lower()
                break
        if mac is None:
            print(f'[{port}] no boot line -- skipping')
            ser.close()
            continue
        print(f'[{port}] {mac}')
        boards[mac] = {'ser': ser, 'port': port}
    return boards


def measure(boards, dwell_s):
    macs = list(boards)
    rate, rssi = {}, {}
    for tx in macs:
        for b in boards.values():
            b['ser'].write(b'RX 000000000000\n')
        time.sleep(0.2)
        boards[tx]['ser'].write(b'TX\n')
        for rx in macs:
            if rx != tx:
                boards[rx]['ser'].write(f'RX {tx.replace(":", "")}\n'.encode())
        time.sleep(0.5)

        res = {}

        def count(mac):
            ser = boards[mac]['ser']
            ser.reset_input_buffer()
            n, rs = 0, []
            st = time.time()
            while time.time() - st < dwell_s:
                raw = ser.readline()
                if raw.startswith(b'CSI_AMP'):
                    try:
                        rs.append(int(raw.split(b',')[3]))
                        n += 1
                    except (ValueError, IndexError):
                        pass
            res[mac] = (n / dwell_s, sum(rs) / len(rs) if rs else None)

        ts = [threading.Thread(target=count, args=(m,)) for m in macs if m != tx]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        for rx, (r, s) in res.items():
            rate[f'{tx}->{rx}'] = r
            rssi[f'{tx}->{rx}'] = s
        boards[tx]['ser'].write(b'RX 000000000000\n')
        print(f'  {tx} transmitted')
    return rate, rssi


def show(macs, rate, rssi, short):
    print('\nCapture rate (rec/s)   rows=TX, cols=RX')
    print('        ' + ''.join(f'{short[m]:>9}' for m in macs))
    for tx in macs:
        cells = ''.join('      ---' if tx == rx else f'{rate.get(f"{tx}->{rx}", 0):9.1f}' for rx in macs)
        print(f'  {short[tx]:>5} ' + cells)
    print('\nRSSI (dBm)')
    print('        ' + ''.join(f'{short[m]:>9}' for m in macs))
    for tx in macs:
        cells = ''
        for rx in macs:
            if tx == rx:
                cells += '      ---'
            else:
                v = rssi.get(f'{tx}->{rx}')
                cells += '      n/a' if v is None else f'{v:9.1f}'
        print(f'  {short[tx]:>5} ' + cells)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dwell', type=float, default=3.0, help='seconds measured per transmitter (default 3)')
    ap.add_argument('--save', help='write results to this JSON file')
    ap.add_argument('--compare', help='diff against a previously saved JSON file')
    args = ap.parse_args()

    boards = discover()
    if len(boards) < 2:
        print('need at least 2 boards')
        sys.exit(1)

    macs = list(boards)
    short = {m: m[-5:] for m in macs}  # last 2 octets are enough to tell them apart
    rate, rssi = measure(boards, args.dwell)
    show(macs, rate, rssi, short)

    if args.save:
        with open(args.save, 'w') as f:
            json.dump({'rate': rate, 'rssi': rssi, 'macs': macs}, f, indent=2)
        print(f'\nsaved -> {args.save}')

    if args.compare:
        with open(args.compare) as f:
            old = json.load(f)
        print(f'\nChange vs {args.compare} (links keyed by MAC, so hub/port changes do not affect pairing):')
        for link in sorted(set(rate) | set(old['rate'])):
            new_r = rate.get(link)
            old_r = old['rate'].get(link)
            if new_r is None or old_r is None:
                print(f'  {link}: only present in one run')
                continue
            d = new_r - old_r
            tag = '  <-- big change' if abs(d) > 5 else ''
            print(f'  {link}: {old_r:5.1f} -> {new_r:5.1f}/s  ({d:+5.1f}){tag}')

    for b in boards.values():
        b['ser'].close()


if __name__ == '__main__':
    main()
