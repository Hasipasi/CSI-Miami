#!/usr/bin/env python3
"""Validate a recorded session before anyone trains on it.

Every check here exists because the failure it looks for is silent: the capture
still produces a file, the file still loads, and the damage only shows up as a
model that will not learn or a result that will not reproduce. Cheap to run now,
expensive to discover later.

  python3 check_session.py ~/csi_test/data/balazs_train
"""

import glob
import json
import os
import sys

import numpy as np

# Widths the firmware is known to emit: 192 amplitudes (frame v1) or the 30-subcarrier
# subset (v2 I/Q). Pinning one number here would fail every capture the moment the
# encoding changed, so what is enforced is that every link within a take agrees --
# a take that mixes widths is broken in a way no single expected value would catch.
KNOWN_SUB = (192, 166, 30)


def links(d):
    return sorted({k.rsplit('|', 1)[0] for k in d.files if k.endswith('|t')})


def check(path):
    """Return (row, [problems]) for one take."""
    name = os.path.basename(path)[:-4]
    probs = []
    try:
        d = np.load(path)
    except Exception as e:                                   # noqa: BLE001
        return {'name': name}, [f'unreadable: {e}']
    meta = json.loads(str(d['meta'])) if 'meta' in d.files else {}
    lab = meta.get('boards', {})

    ft = d['frame_t'] if 'frame_t' in d.files else np.array([])
    dur = float(ft[-1] - ft[0]) if len(ft) > 1 else 0.0
    fps = len(ft) / dur if dur > 0 else 0.0

    # frames on disk must match the index, or frame k in the npz is not the image
    # sitting at 000k.jpg and every label lines up with the wrong picture
    fdir = os.path.join(os.path.dirname(path), f'{name}_frames')
    njpg = len(glob.glob(os.path.join(fdir, '*.jpg')))
    if njpg != len(ft):
        probs.append(f'{len(ft)} frame timestamps but {njpg} jpgs on disk')
    if meta.get('dropped_encode'):
        probs.append(f'{meta["dropped_encode"]} frames never encoded')
    if 'frame_seq' in d.files and len(ft) > 1:
        gaps = int(np.sum(np.diff(d['frame_seq']) - 1))
        if gaps > 0.02 * len(ft):
            probs.append(f'{gaps} frames dropped by the driver ({100 * gaps / len(ft):.0f}%)')
    if len(ft) > 1 and not np.all(np.diff(ft) > 0):
        probs.append('frame timestamps not monotonic')

    lks = links(d)
    counts, worst_dt, widths = {}, 0.0, {}
    for lk in lks:
        tx, rx = lk.split('|')
        nm = f'{lab.get(tx, tx[-5:])}->{lab.get(rx, rx[-5:])}'
        a, t = d[f'{lk}|a'], d[f'{lk}|t'].astype(np.float64)
        counts[nm] = len(t)
        widths.setdefault(a.shape[1], []).append(nm)
        if len(t) > 1 and not np.all(np.diff(t) >= 0):
            probs.append(f'{nm}: packet times not monotonic')
        # a dead or stuck link still writes a well-formed array
        if not np.isfinite(a).all():
            probs.append(f'{nm}: non-finite amplitudes')
        if a.size and float(a.std()) == 0.0:
            probs.append(f'{nm}: amplitudes constant (stuck)')
        if a.size and float(np.mean(a.sum(axis=1) == 0)) > 0.05:
            probs.append(f'{nm}: {100 * np.mean(a.sum(axis=1) == 0):.0f}% all-zero packets')
        # Phase, when the boards sent I/Q. Raw phase is carrier/sampling offset and
        # looks like noise by design, so the check is on the detrended residual --
        # that is the part that carries channel information, and it being frozen or
        # non-finite means the complex payload is not usable even though |iq| is.
        if f'{lk}|iq' in d.files:
            q = d[f'{lk}|iq']
            if not np.isfinite(q).all():
                probs.append(f'{nm}: non-finite iq')
            elif len(q) > 2 and q.shape[1] > 3:
                ph = np.unwrap(np.angle(q), axis=1)
                idx = np.arange(q.shape[1])
                c = np.polyfit(idx, ph.T, 1)
                res = ph - (np.outer(idx, c[0]) + c[1]).T
                if float(res.std()) == 0.0:
                    probs.append(f'{nm}: detrended phase constant (stuck)')
        if f'{lk}|dt' in d.files and len(d[f'{lk}|dt']):
            worst_dt = max(worst_dt, float(np.max(np.abs(d[f'{lk}|dt']))) * 1e3)

    if len(widths) > 1:
        probs.append('links disagree on subcarrier count: '
                     + '; '.join(f'{w} ({len(v)} links)' for w, v in sorted(widths.items())))
    for w in widths:
        if w not in KNOWN_SUB:
            probs.append(f'unexpected subcarrier count {w}, known are {KNOWN_SUB}')

    # A round-robin run should yield every ordered pair. Missing links are the
    # failure this whole script exists for: the files load, the counts look sane,
    # and a board that produced nothing is invisible unless you count the pairs.
    # Reporting the inventory without judging it (as this first did) is no check.
    boards = sorted(set(lab.values())) if lab else []
    mode = str(meta.get('mode', ''))
    # A pinned transmitter is expected to yield exactly its own outgoing links, so
    # "12 links missing" is correct there and "one of 3 missing" is the real fault.
    # Checking only the round-robin case left single-TX sessions unvalidated, which
    # is precisely where a silent board is hardest to notice: fewer rows to miss.
    pinned = mode if mode in boards else None
    if pinned is None and mode.startswith('fixed'):
        txs = {n.split('->')[0] for n in counts}
        pinned = next(iter(txs)) if len(txs) == 1 else None

    if boards and pinned:
        want = {f'{pinned}->{b}' for b in boards if b != pinned}
        got = set(counts)
        for miss in sorted(want - got):
            probs.append(f'MISSING link {miss} (transmitter pinned to {pinned})')
        extra = sorted(got - want)
        if extra:
            probs.append(f'unexpected links for a {pinned}-only capture: {extra}')
    elif boards and mode.startswith('round'):
        want = {f'{a}->{b}' for a in boards for b in boards if a != b}
        got = set(counts)
        for miss in sorted(want - got):
            probs.append(f'MISSING link {miss}')
        silent = [b for b in boards if not any(nm.endswith(f'->{b}') for nm in got)]
        for b in silent:
            probs.append(f'board {b} received nothing at all (its reader produced no CSI)')
        deaf = [b for b in boards if not any(nm.startswith(f'{b}->') for nm in got)]
        for b in deaf:
            probs.append(f'board {b} never transmitted')

    if not lks:
        probs.append('no links at all')
    return ({'name': name, 'frames': len(ft), 'dur': dur, 'fps': fps,
             'links': len(lks), 'pkts': sum(counts.values()),
             'counts': counts, 'worst_dt': worst_dt, 'mode': meta.get('mode', '?')},
            probs)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    # Skeletons live beside the captures as <take>_pose.npz and are a different
    # schema; globbing them in made every take look linkless.
    paths = sorted(p for p in glob.glob(os.path.join(root, '*.npz'))
                   if not p.endswith('_pose.npz'))
    if not paths:
        print(f'no .npz in {root}')
        sys.exit(1)

    rows, allprobs = [], []
    for p in paths:
        row, probs = check(p)
        rows.append(row)
        for pr in probs:
            allprobs.append((row['name'], pr))

    good = [r for r in rows if 'frames' in r]
    print(f'{len(paths)} takes in {root}\n')
    print(f'{"take":18s} {"mode":5s} {"frames":>6s} {"dur":>6s} {"fps":>6s} '
          f'{"links":>5s} {"pkts":>6s} {"dt ms":>6s}')
    for r in good:
        flag = '  <--' if any(n == r['name'] for n, _ in allprobs) else ''
        print(f'{r["name"]:18s} {str(r["mode"])[:5]:5s} {r["frames"]:6d} {r["dur"]:6.2f} '
              f'{r["fps"]:6.2f} {r["links"]:5d} {r["pkts"]:6d} {r["worst_dt"]:6.1f}{flag}')

    if good:
        print('\n--- consistency across takes ---')
        for key, fmt in (('frames', '%.1f'), ('dur', '%.2f'), ('fps', '%.2f'),
                         ('pkts', '%.1f'), ('links', '%.1f')):
            v = np.array([r[key] for r in good], dtype=float)
            print(f'  {key:7s} mean {fmt % v.mean():>8s}  min {fmt % v.min():>8s}  '
                  f'max {fmt % v.max():>8s}  spread {100 * v.std() / max(v.mean(), 1e-9):5.1f}%')

        # A link that vanishes in some takes but not others is worse than one that is
        # always absent: the feature matrix silently changes width between samples.
        seen = {}
        for r in good:
            for nm in r['counts']:
                seen.setdefault(nm, 0)
                seen[nm] += 1
        missing = {nm: c for nm, c in seen.items() if c != len(good)}
        print(f'\n  links present in every take: '
              f'{sorted(nm for nm, c in seen.items() if c == len(good))}')
        if missing:
            print('  INCONSISTENT links (present in some takes only):')
            for nm, c in sorted(missing.items()):
                print(f'    {nm}: {c}/{len(good)} takes')

        # per-class counts, since a protocol run is meant to be balanced
        klass = {}
        for r in good:
            klass.setdefault(r['name'].rstrip('0123456789'), []).append(r['pkts'])
        print('\n  per-pose takes and packet counts:')
        for k, v in sorted(klass.items()):
            print(f'    {k:16s} {len(v):2d} takes, packets mean {np.mean(v):6.1f} '
                  f'min {min(v):5d} max {max(v):5d}')

    print(f'\n--- problems: {len(allprobs)} ---')
    if not allprobs:
        print('  none found')
    else:
        for n, pr in allprobs:
            print(f'  {n:18s} {pr}')


if __name__ == '__main__':
    main()
