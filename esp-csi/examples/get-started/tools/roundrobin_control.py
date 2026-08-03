#!/usr/bin/env python3
"""PC-driven round-robin controller for the csi_roundrobin firmware.

Every connected board runs identical, purely reactive firmware: it does whatever
the last command it received over UART says. This script owns the schedule --
every ROUND_DURATION_S, it tells the next board "TX" and tells every other board
"RX <tx_mac_hex>" so their CSI filters follow along. Wired UART instead of
ESP-NOW broadcast for coordination, since ESP-NOW broadcasts aren't acknowledged
and testing showed most were getting dropped under CSI-processing load.
"""

import argparse
import re
import sys
import time

import serial
import serial.tools.list_ports

WCH_VID = 0x1A86
BAUD = 921600
BOOT_MAC_RE = re.compile(r'Board MAC ([0-9a-fA-F:]{17})')


def discover_boards():
    ports = [p.device for p in serial.tools.list_ports.comports() if p.vid == WCH_VID]
    boards = {}
    for port in sorted(ports):
        ser = serial.Serial(port, BAUD, timeout=1)
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
            print(f'[{port}] no "Board MAC" boot line seen -- skipping (wrong firmware?)')
            ser.close()
            continue

        print(f'[{port}] MAC {mac}')
        boards[port] = {'serial': ser, 'mac': mac}
    return boards


def mac_hex(mac: str) -> str:
    return mac.replace(':', '')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--round-duration', type=float, default=1.0,
                         help='Seconds each board holds the TX token before handoff (default: 1.0)')
    args = parser.parse_args()

    boards = discover_boards()
    if len(boards) < 2:
        print(f'Found {len(boards)} board(s) -- need at least 2 (one TX, one RX). Exiting.')
        sys.exit(1)

    ports = list(boards.keys())
    print(f'\n{len(ports)} boards in the ring: {ports}')
    print(f'Round duration: {args.round_duration}s. Ctrl+C to stop.\n')

    current_idx = -1
    try:
        while True:
            current_idx = (current_idx + 1) % len(ports)
            tx_port = ports[current_idx]
            tx_mac_hex = mac_hex(boards[tx_port]['mac'])

            boards[tx_port]['serial'].write(b'TX\n')
            for port, board in boards.items():
                if port != tx_port:
                    board['serial'].write(f'RX {tx_mac_hex}\n'.encode())

            print(f'{time.strftime("%H:%M:%S")}  TX -> {tx_port} ({boards[tx_port]["mac"]})')
            time.sleep(args.round_duration)
    except KeyboardInterrupt:
        print('\nStopping.')
    finally:
        for board in boards.values():
            board['serial'].close()


if __name__ == '__main__':
    main()
