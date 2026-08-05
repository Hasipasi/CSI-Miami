#!/usr/bin/env python3
"""Capture CSI and assess whether it is usable for person pose estimation.

Two capture modes, because they trade the same budget differently:

  roundrobin  every board takes a turn transmitting. Many links (N*(N-1)) but
              each is only refreshed once per cycle, so per-link rate is low.
  fixedtx     one board transmits continuously, the rest listen. Fewer links
              (N-1) but each runs at the full ping rate.

Pose estimation needs temporal resolution to track limbs, so which trade is
right is an empirical question -- capture both and compare.

  python3 pose_suitability.py capture --mode fixedtx  --duration 20 --out still.npz
  python3 pose_suitability.py analyze still.npz moving.npz
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

WCH_VID = 0x1A86
BAUD = 921600
BOOT_MAC_RE = re.compile(r'Board MAC ([0-9a-fA-F:]{17})')
N_META = 25  # metadata columns before the amplitude array


def discover():
    ports = sorted(p.device for p in serial.tools.list_ports.comports() if p.vid == WCH_VID)
    boards = {}
    for port in ports:
        ser = serial.Serial(port, BAUD, timeout=0.3)
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
            ser.close()
            continue
        boards[mac] = ser
        print(f'  {port} -> {mac}')
    return boards


def parse(line):
    """-> (tx_mac, rssi, amplitude array) or None."""
    try:
        row = next(csv.reader(StringIO(line)))
        if len(row) != N_META:
            return None
        n_sub = int(row[-3])
        vals = json.loads(row[-1])
        if n_sub != len(vals) or n_sub < 1:
            return None
        return row[2].lower(), int(row[3]), np.array(vals, dtype=np.float32)
    except (ValueError, json.JSONDecodeError, IndexError, StopIteration):
        return None


def capture(mode, duration, round_s, tx_mac=None):
    boards = discover()
    # Sorted, not dict-insertion order: discovery runs in threads, so insertion order
    # varies run to run. A different transmitter between the static and moving captures
    # would silently invalidate the comparison between them.
    macs = sorted(boards)
    if len(macs) < 2:
        print('need >= 2 boards')
        sys.exit(1)
    if tx_mac:
        tx_mac = tx_mac.lower()
        if tx_mac not in boards:
            print(f'requested TX {tx_mac} not connected; have {macs}')
            sys.exit(1)

    recs = {m: [] for m in macs}   # rx_mac -> list of (t, tx_mac, amp)
    stop = threading.Event()
    t0 = time.time()

    def reader(rx_mac):
        ser = boards[rx_mac]
        while not stop.is_set():
            try:
                raw = ser.readline()
            except (serial.SerialException, OSError):
                break
            if not raw.startswith(b'CSI_AMP'):
                continue
            t = time.time() - t0
            p = parse(raw.decode(errors='ignore').strip())
            if p:
                recs[rx_mac].append((t, p[0], p[2]))

    for m in macs:
        boards[m].reset_input_buffer()
    threads = [threading.Thread(target=reader, args=(m,), daemon=True) for m in macs]
    for t in threads:
        t.start()

    if mode == 'fixedtx':
        tx = tx_mac or macs[0]
        boards[tx].write(b'TX\n')
        for m in macs:
            if m != tx:
                boards[m].write(f'RX {tx.replace(":", "")}\n'.encode())
        print(f'  fixed TX = {tx}')
        time.sleep(duration)
    else:
        idx = 0
        end = time.time() + duration
        while time.time() < end:
            tx = macs[idx % len(macs)]
            idx += 1
            boards[tx].write(b'TX\n')
            for m in macs:
                if m != tx:
                    boards[m].write(f'RX {tx.replace(":", "")}\n'.encode())
            time.sleep(round_s)

    stop.set()
    time.sleep(0.4)
    for m in macs:
        boards[m].write(b'RX 000000000000\n')
        boards[m].close()
    return recs


def save(recs, path, mode):
    out = {'mode': mode}
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
            key = f'{tx}|{rx}'
            out[f'{key}|t'] = np.array([t for t, _ in seq], dtype=np.float32)
            out[f'{key}|a'] = np.stack([a for _, a in seq])
    np.savez_compressed(path, **out)
    n_links = sum(1 for k in out if k.endswith('|t'))
    print(f'saved {path}: {n_links} links')


def links(d):
    return sorted({k.rsplit('|', 1)[0] for k in d.files if k.endswith('|t')})


def analyze(paths):
    data = {p: np.load(p, allow_pickle=False) for p in paths}

    print('=' * 78)
    print('1. SAMPLING RATE  (pose estimation needs to out-sample limb motion)')
    print('=' * 78)
    for p, d in data.items():
        rates, jit = [], []
        for lk in links(d):
            t = d[f'{lk}|t']
            if len(t) < 3:
                continue
            dt = np.diff(t)
            rates.append(1.0 / np.mean(dt))
            jit.append(np.std(dt) / np.mean(dt))
        if rates:
            print(f'  {p}: per-link {np.mean(rates):6.1f} Hz  '
                  f'(min {np.min(rates):.1f}, max {np.max(rates):.1f})  '
                  f'jitter CV {np.mean(jit):.2f}  Nyquist {np.mean(rates) / 2:.1f} Hz')

    print()
    print('=' * 78)
    print('2. TEMPORAL VARIATION  (static = noise floor, moving = signal)')
    print('=' * 78)
    stats = {}
    for p, d in data.items():
        per_link = []
        for lk in links(d):
            a = d[f'{lk}|a'].astype(np.float64)
            if a.shape[0] < 5:
                continue
            # std over time per subcarrier, averaged across subcarriers
            per_link.append(np.mean(np.std(a, axis=0)))
        stats[p] = np.array(per_link)
        if len(per_link):
            print(f'  {p}: mean per-subcarrier temporal std = {np.mean(per_link):6.2f} '
                  f'(across {len(per_link)} links, spread {np.min(per_link):.2f}-{np.max(per_link):.2f})')

    if len(paths) >= 2:
        base, other = paths[0], paths[1]
        if len(stats[base]) and len(stats[other]):
            ratio = np.mean(stats[other]) / max(np.mean(stats[base]), 1e-9)
            print(f'\n  motion/static variation ratio = {ratio:.2f}x')
            print('  (>3x = motion clearly detectable; ~1x = motion indistinguishable from noise)')

    print()
    print('=' * 78)
    print('3. CHANNEL COHERENCE  (how fast the channel decorrelates -> required rate)')
    print('=' * 78)
    for p, d in data.items():
        halftimes = []
        for lk in links(d):
            a = d[f'{lk}|a'].astype(np.float64)
            t = d[f'{lk}|t']
            if a.shape[0] < 20:
                continue
            a = a - a.mean(axis=0)
            norm = np.linalg.norm(a, axis=1)
            good = norm > 1e-9
            if good.sum() < 20:
                continue
            a, tt, norm = a[good], t[good], norm[good]
            ref = a[0] / norm[0]
            corr = (a @ ref) / norm
            below = np.where(corr < 0.5)[0]
            if len(below):
                halftimes.append(tt[below[0]] - tt[0])
        if halftimes:
            ht = np.median(halftimes)
            print(f'  {p}: median time to decorrelate below 0.5 = {ht * 1000:6.0f} ms '
                  f'-> implies >= {2 / ht:.1f} Hz sampling to track it')
        else:
            print(f'  {p}: channel never decorrelated below 0.5 (very static, or too few samples)')

    print()
    print('=' * 78)
    print('4. SPATIAL INFORMATION  (effective dimensionality of the subcarrier space)')
    print('=' * 78)
    for p, d in data.items():
        ranks = []
        for lk in links(d):
            a = d[f'{lk}|a'].astype(np.float64)
            if a.shape[0] < 20:
                continue
            a = a - a.mean(axis=0)
            s = np.linalg.svd(a, compute_uv=False)
            if s[0] < 1e-9:
                continue
            var = s ** 2 / np.sum(s ** 2)
            ranks.append(int(np.searchsorted(np.cumsum(var), 0.95) + 1))
        if ranks:
            print(f'  {p}: {np.median(ranks):.0f} components explain 95% of variance '
                  f'(of {d[f"{links(d)[0]}|a"].shape[1]} subcarriers)')

    print()
    print('=' * 78)
    print('5. LINK DIVERSITY  (independent views help; redundant links do not)')
    print('=' * 78)
    for p, d in data.items():
        lks = links(d)
        series = {}
        for lk in lks:
            a = d[f'{lk}|a'].astype(np.float64)
            if a.shape[0] < 20:
                continue
            series[lk] = np.mean(a, axis=1)  # mean amplitude per packet
        keys = list(series)
        if len(keys) < 2:
            continue
        n = min(len(series[k]) for k in keys)
        M = np.stack([series[k][:n] for k in keys])
        M = M - M.mean(axis=1, keepdims=True)
        sd = M.std(axis=1)
        ok = sd > 1e-9
        if ok.sum() < 2:
            continue
        C = np.corrcoef(M[ok])
        off = C[np.triu_indices_from(C, k=1)]
        print(f'  {p}: {ok.sum()} usable links, mean |correlation| between links = {np.mean(np.abs(off)):.2f} '
              f'(low = diverse views, high = redundant)')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    c = sub.add_parser('capture')
    c.add_argument('--mode', choices=['roundrobin', 'fixedtx'], default='fixedtx')
    c.add_argument('--duration', type=float, default=20.0)
    c.add_argument('--round-duration', type=float, default=0.05)
    c.add_argument('--tx', help='MAC of the transmitter for fixedtx mode (default: lowest MAC)')
    c.add_argument('--out', required=True)

    a = sub.add_parser('analyze')
    a.add_argument('files', nargs='+')

    args = ap.parse_args()
    if args.cmd == 'capture':
        recs = capture(args.mode, args.duration, args.round_duration, args.tx)
        save(recs, args.out, args.mode)
    else:
        analyze(args.files)


if __name__ == '__main__':
    main()
