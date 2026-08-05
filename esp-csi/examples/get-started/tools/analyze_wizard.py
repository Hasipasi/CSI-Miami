#!/usr/bin/env python3
"""Analyse a three-condition capture from capture_wizard.py.

  empty   nobody in the room  -> the reference
  still   person present, motionless
  move    person moving

The three-way split exists because a two-way (static vs moving) comparison was
previously spoiled by a "static" reference that was not actually static. So the
first thing this does is test whether the reference is trustworthy; every later
number is meaningless if it is not.

  python3 analyze_wizard.py ft_empty.npz ft_still.npz ft_move.npz
"""

import sys

import numpy as np

LABEL = {'ab:d4': 'C', '2d:3c': 'A', '2d:a8': 'B', '6b:5c': 'D'}


def links(d):
    return sorted({k.rsplit('|', 1)[0] for k in d.files if k.endswith('|t')})


def name(lk):
    tx, rx = lk.split('|')
    return f'{LABEL.get(tx[-5:], tx[-5:])}->{LABEL.get(rx[-5:], rx[-5:])}'


def series(d, lk):
    """(t, per-packet mean amplitude, full matrix)."""
    a = d[f'{lk}|a'].astype(np.float64)
    return d[f'{lk}|t'], a.mean(axis=1), a


def check_reference(d, tag):
    """A trustworthy reference is flat (no drift) and smooth (high lag-1 autocorr).
    Drift means the room was changing; low autocorr means noise dominates."""
    print(f'--- reference quality: {tag} ---')
    ok = True
    for lk in links(d):
        t, m, a = series(d, lk)
        if len(t) < 30:
            continue
        # linear trend across the window, expressed against the within-window scatter
        slope, _ = np.polyfit(t, m, 1)
        drift = abs(slope) * (t[-1] - t[0])
        resid = np.std(m - np.polyval(np.polyfit(t, m, 1), t))
        ratio = drift / max(resid, 1e-9)
        ac = np.corrcoef(m[:-1], m[1:])[0, 1]
        flag = ''
        if ratio > 1.5:
            flag, ok = '  <-- DRIFTING, room was not static', False
        print(f'  {name(lk):8s} drift {drift:6.2f} over window vs scatter {resid:5.2f} '
              f'(ratio {ratio:4.1f})   lag-1 autocorr {ac:+.2f}{flag}')
    print(f'  => reference {"looks usable" if ok else "is CONTAMINATED"}\n')
    return ok


def variation(d):
    """Per-link temporal std of the mean-amplitude series, and of the full matrix."""
    out = {}
    for lk in links(d):
        t, m, a = series(d, lk)
        if len(t) < 20:
            continue
        out[name(lk)] = (float(np.std(m)), float(np.mean(np.std(a, axis=0))))
    return out


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    paths = sys.argv[1:4]
    tags = ['empty', 'still', 'move']
    data = {tag: np.load(p) for tag, p in zip(tags, paths)}

    print('=' * 74)
    print('STEP 1  is the empty-room reference trustworthy?')
    print('=' * 74)
    good = check_reference(data['empty'], 'empty')

    print('=' * 74)
    print('STEP 2  per-link variation in each condition')
    print('=' * 74)
    v = {tag: variation(d) for tag, d in data.items()}
    common = sorted(set(v['empty']) & set(v['still']) & set(v['move']))
    print(f'  {"link":9s} {"empty":>16s} {"still":>16s} {"move":>16s}')
    print(f'  {"":9s} {"mean-amp / subc":>16s} {"mean-amp / subc":>16s} {"mean-amp / subc":>16s}')
    for lk in common:
        row = ''.join(f'{v[t][lk][0]:7.2f} /{v[t][lk][1]:6.2f}' for t in tags)
        print(f'  {lk:9s} {row}')

    print()
    print('=' * 74)
    print('STEP 3  detectability against the empty room')
    print('=' * 74)
    for tag in ('still', 'move'):
        r_mean = np.mean([v[tag][l][0] for l in common]) / max(np.mean([v['empty'][l][0] for l in common]), 1e-9)
        r_sub = np.mean([v[tag][l][1] for l in common]) / max(np.mean([v['empty'][l][1] for l in common]), 1e-9)
        print(f'  {tag:5s} vs empty:  mean-amplitude {r_mean:5.2f}x   per-subcarrier {r_sub:5.2f}x')
    r = np.mean([v['move'][l][0] for l in common]) / max(np.mean([v['still'][l][0] for l in common]), 1e-9)
    print(f'  move  vs still:  mean-amplitude {r:5.2f}x')
    print('   >3x = clearly separable;  ~1x = indistinguishable')

    print()
    print('=' * 74)
    print('STEP 4  is the difference in the motion band (0.3-5 Hz)?')
    print('=' * 74)
    print('  Human limb motion is slow. Excess power there (and not at high')
    print('  frequency) is the signature of real motion rather than noise.')
    for tag in tags:
        d = data[tag]
        lo = hi = 0.0
        for lk in links(d):
            t, m, a = series(d, lk)
            if len(t) < 64:
                continue
            fs = 1.0 / np.mean(np.diff(t))
            x = m - m.mean()
            P = np.abs(np.fft.rfft(x)) ** 2
            fr = np.fft.rfftfreq(len(x), 1 / fs)
            lo += P[(fr > 0.3) & (fr < 5)].sum()
            hi += P[fr >= 8].sum()
        print(f'  {tag:5s}: 0.3-5Hz power {lo:10.3e}   >=8Hz {hi:10.3e}   lo/hi {lo / max(hi, 1e-9):5.2f}')

    if not good:
        print()
        print('!! The empty-room reference drifted, so the ratios above understate')
        print('!! detectability. Re-capture with the room genuinely undisturbed.')


if __name__ == '__main__':
    main()
