#!/usr/bin/env python3
"""Identify boards physically (via their LEDs) and reconstruct their layout from
hand-measured pairwise distances.

Boards are labelled A, B, C, ... by sorted MAC, so the labels are stable across
runs and replugs. Light one board (or a pair) to work out which physical unit is
which, measure the distances by hand, then solve for coordinates.

  python3 measure_geometry.py list
  python3 measure_geometry.py ident --board A
  python3 measure_geometry.py ident --pair AB
  python3 measure_geometry.py ident --off
  python3 measure_geometry.py solve AB=1.20 AC=2.05 AD=1.80 BC=1.55 BD=2.10 CD=1.35

With 4 boards there are 6 distances. A planar layout has 5 degrees of freedom
(8 coordinates minus 3 for translation/rotation), so the 6th measurement is a
consistency check rather than redundancy -- the solver reports per-pair residuals
and flags whether the layout is really planar.
"""

import argparse
import itertools
import re
import sys
import threading
import time

import numpy as np
import serial
import serial.tools.list_ports

WCH_VID = 0x1A86
BAUD = 921600
BOOT_MAC_RE = re.compile(r'Board MAC ([0-9a-fA-F:]{17})')


def discover():
    """MAC -> port, probing all boards in parallel (serial probing is ~3s each)."""
    ports = sorted(p.device for p in serial.tools.list_ports.comports() if p.vid == WCH_VID)
    found = {}
    lock = threading.Lock()

    def probe(port):
        try:
            ser = serial.Serial(port, BAUD, timeout=0.3)
        except (serial.SerialException, OSError):
            return
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
        ser.close()
        if mac:
            with lock:
                found[mac] = port

    threads = [threading.Thread(target=probe, args=(p,)) for p in ports]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return found


def labelled(found):
    """Stable A, B, C, ... assignment by sorted MAC."""
    macs = sorted(found)
    return {chr(ord('A') + i): (m, found[m]) for i, m in enumerate(macs)}


def send(port, cmd):
    ser = serial.Serial(port, BAUD, timeout=0.5)
    time.sleep(0.2)
    ser.write(cmd)
    time.sleep(0.3)
    ser.close()


def do_ident(args):
    boards = labelled(discover())
    if not boards:
        print('no boards found')
        sys.exit(1)

    targets = []
    if args.off:
        targets = list(boards)
    elif args.board:
        targets = [args.board.upper()]
    elif args.pair:
        targets = list(args.pair.upper())

    unknown = [t for t in targets if t not in boards]
    if unknown:
        print(f'unknown label(s): {unknown}. Known: {sorted(boards)}')
        sys.exit(1)

    for label, (mac, port) in sorted(boards.items()):
        if args.off:
            send(port, b'IDENT OFF\n')
        elif label in targets:
            send(port, b'IDENT\n')
        else:
            send(port, b'IDENT OFF\n')

    if args.off:
        print('all LEDs back to role colour')
    else:
        for t in targets:
            print(f'  {t} = {boards[t][0]}  ({boards[t][1]})  -> ORANGE')
        if len(targets) == 2:
            print(f'\nMeasure the distance between the two lit boards ({targets[0]} and {targets[1]}).')


def do_list(args):
    boards = labelled(discover())
    if not boards:
        print('no boards found')
        sys.exit(1)
    print('label  MAC                port')
    for label, (mac, port) in sorted(boards.items()):
        print(f'  {label}    {mac}  {port}')
    n = len(boards)
    pairs = [''.join(p) for p in itertools.combinations(sorted(boards), 2)]
    print(f'\n{n} boards -> {len(pairs)} pairs to measure: {" ".join(pairs)}')


def solve_layout(labels, dist):
    """Classical MDS for a starting guess, then least-squares refine in 2D."""
    n = len(labels)
    idx = {l: i for i, l in enumerate(labels)}
    D = np.zeros((n, n))
    for (a, b), d in dist.items():
        D[idx[a], idx[b]] = D[idx[b], idx[a]] = d

    # Torgerson MDS
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    w, v = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    X0 = v[:, :2] * np.sqrt(np.maximum(w[:2], 0))

    # Refine: minimise squared error on the distances themselves, not on B.
    from scipy.optimize import least_squares

    def resid(flat):
        X = flat.reshape(n, 2)
        return [np.linalg.norm(X[idx[a]] - X[idx[b]]) - d for (a, b), d in dist.items()]

    out = least_squares(resid, X0.ravel())
    X = out.x.reshape(n, 2)

    # Canonical pose: first board at origin, second on +x axis, third y>0.
    X = X - X[0]
    ang = np.arctan2(X[1, 1], X[1, 0])
    R = np.array([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
    X = X @ R.T
    if n > 2 and X[2, 1] < 0:
        X[:, 1] *= -1
    return X, w, dict(zip(dist.keys(), resid(X.ravel())))


def do_solve(args):
    dist = {}
    for item in args.distances:
        if '=' not in item:
            print(f'bad argument "{item}", expected e.g. AB=1.20')
            sys.exit(1)
        pair, val = item.split('=', 1)
        pair = pair.strip().upper()
        if len(pair) != 2:
            print(f'bad pair "{pair}", expected two labels e.g. AB')
            sys.exit(1)
        dist[(pair[0], pair[1])] = float(val)

    labels = sorted({l for p in dist for l in p})
    n = len(labels)
    need = n * (n - 1) // 2
    if len(dist) != need:
        print(f'have {len(dist)} distances for {n} boards, need {need}: '
              f'{" ".join("".join(p) for p in itertools.combinations(labels, 2))}')
        sys.exit(1)

    X, eig, resid = solve_layout(labels, dist)

    print(f'Reconstructed layout ({n} boards, {len(dist)} measured distances), metres:')
    print('  label        x        y')
    for l, (x, y) in zip(labels, X):
        print(f'    {l}    {x:8.3f} {y:8.3f}')

    print('\nFit residuals (measured - reconstructed):')
    worst = 0.0
    for (a, b), r in sorted(resid.items()):
        worst = max(worst, abs(r))
        flag = '  <-- check this measurement' if abs(r) > 0.05 else ''
        print(f'  {a}{b}: measured {dist[(a, b)]:6.3f}  error {r:+7.4f} m{flag}')
    rms = np.sqrt(np.mean([r ** 2 for r in resid.values()]))
    print(f'  RMS residual {rms:.4f} m, worst {worst:.4f} m')

    # Third eigenvalue of the MDS Gram matrix measures out-of-plane extent: it is
    # zero for a genuinely planar, self-consistent set of distances.
    if n >= 4:
        spread = max(np.max(eig), 1e-12)
        oop = max(eig[2], 0.0) / spread
        print(f'\nPlanarity: out-of-plane eigenvalue is {oop * 100:.1f}% of the largest.')
        if oop < 0.02:
            print('  -> consistent with a flat layout at equal height.')
        else:
            print('  -> boards are NOT coplanar, or a distance is mismeasured.')
            print('     With 4 points, 6 distances always fit exactly in 3D, so a 3D fit')
            print('     could not distinguish the two -- re-measure the flagged pair.')

    span = np.max(np.linalg.norm(X[:, None] - X[None, :], axis=-1))
    print(f'\nLargest separation {span:.2f} m.')
    if span < 1.5:
        print('  NOTE: for pose work the subject should stand between the boards;')
        print('  a span under ~1.5 m leaves little room to stand inside the array.')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    sub.add_parser('list')

    i = sub.add_parser('ident')
    g = i.add_mutually_exclusive_group(required=True)
    g.add_argument('--board', help='light a single board, e.g. A')
    g.add_argument('--pair', help='light two boards, e.g. AB')
    g.add_argument('--off', action='store_true', help='clear all IDENT LEDs')

    s = sub.add_parser('solve')
    s.add_argument('distances', nargs='+', help='e.g. AB=1.20 AC=2.05 ...')

    args = ap.parse_args()
    {'list': do_list, 'ident': do_ident, 'solve': do_solve}[args.cmd](args)


if __name__ == '__main__':
    main()
