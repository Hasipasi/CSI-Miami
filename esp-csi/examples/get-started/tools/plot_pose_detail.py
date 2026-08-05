#!/usr/bin/env python3
"""Dense single-figure view of a pose capture: link summary, subcarrier
structure, and the raw data underneath both.

Three panels, three levels of aggregation:

  A  link x pose      -- each link's mean change from the empty room (dB)
  B  subcarrier x pose -- the same quantity resolved across frequency for the
                          single most responsive link, which is where the
                          per-link average hides the real structure
  C  raw waterfall     -- the actual per-packet amplitudes the other two panels
                          are computed from, so the reader can see the data
                          rather than only summaries of it

A and B are signed changes, so they take a diverging ramp centred on zero. C is
an absolute magnitude, so it takes a one-hue sequential ramp. No categorical
palette is used anywhere, which is deliberate -- pose identity is carried by
axis position and dividers instead of by hue.

  python3 plot_pose_detail.py --prefix ps --out pose_detail_rr.png
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

LABEL = {'ab:d4': 'C', '2d:3c': 'A', '2d:a8': 'B', '6b:5c': 'D'}
# measured by hand, reconstructed to 1.6 mm RMS -- see NOTES.md
COORDS = {'A': (0.000, 0.000), 'B': (4.698, 0.000),
          'C': (1.986, 1.164), 'D': (1.307, -2.867)}
POSES = ['neutral', 'tpose', 'up', 'split', 'crouch', 'turned']
PRETTY = ['arms\ndown', 'T-pose', 'arms\nup', 'split\narms', 'crouch', 'turned\n90°']
SHORT = ['arms down', 'T-pose', 'arms up', 'split arms', 'crouch', 'turned 90°']

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
DIVERGING = LinearSegmentedColormap.from_list('loss_gain', [
    '#d03b3b', '#e34948', '#e66767', '#f0efec', '#86b6ef', '#3987e5', '#1c5cab'])
CAT = ['#2a78d6', '#eb6834', '#1baf7a']   # palette slots 1-3, ties an axis to its waterfall
SEQ = LinearSegmentedColormap.from_list('amp', [
    '#f0efec', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#104281', '#0d366b'])


def links(d):
    return sorted({k.rsplit('|', 1)[0] for k in d.files if k.endswith('|t')})


def name(lk):
    tx, rx = lk.split('|')
    return f'{LABEL.get(tx[-5:], tx[-5:])}→{LABEL.get(rx[-5:], rx[-5:])}'


def angle_of(lk):
    """Link orientation in degrees, folded to 0-180 (a link has no direction)."""
    tx, rx = lk.split('|')
    a, b = LABEL.get(tx[-5:]), LABEL.get(rx[-5:])
    if a not in COORDS or b not in COORDS:
        return float('nan')
    (x1, y1), (x2, y2) = COORDS[a], COORDS[b]
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)


def pick_axes(lk_list, n=3):
    """Links spanning the widest range of orientations. One waterfall per axis
    shows what a single link cannot: the same pose seen from different angles.
    Reciprocal pairs (A->B and B->A) share an orientation, so keep only one."""
    seen, uniq = set(), []
    for lk in lk_list:
        key = frozenset(name(lk).split('→'))
        if key not in seen:
            seen.add(key)
            uniq.append(lk)
    uniq = [lk for lk in uniq if np.isfinite(angle_of(lk))]
    uniq.sort(key=angle_of)
    if len(uniq) <= n:
        return uniq
    # greedy: widest angular spread
    chosen = [uniq[0], uniq[-1]]
    while len(chosen) < n:
        best, bestgap = None, -1
        for lk in uniq:
            if lk in chosen:
                continue
            gap = min(abs(angle_of(lk) - angle_of(c)) for c in chosen)
            if gap > bestgap:
                best, bestgap = lk, gap
        chosen.append(best)
    return sorted(chosen, key=angle_of)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix', default='ps')
    ap.add_argument('--out', default='pose_detail.png')
    ap.add_argument('--title', default='Round-robin, session 1')
    ap.add_argument('--axes', type=int, default=6,
                    help='how many distinct link orientations to show (default: all unique links)')
    args = ap.parse_args()

    E = np.load(f'{args.prefix}_empty.npz')
    lks = links(E)
    P = {p: np.load(f'{args.prefix}_{p}.npz') for p in POSES}

    # ---- panel A data: link x pose, mean over subcarriers -----------------
    rows, names = [], []
    for lk in lks:
        base = E[f'{lk}|a'].astype(float).mean()
        rows.append([20 * np.log10(max(P[p][f'{lk}|a'].astype(float).mean(), 1e-9) / max(base, 1e-9))
                     if f'{lk}|a' in P[p].files else np.nan for p in POSES])
        names.append(name(lk))
    M = np.array(rows)
    order = np.argsort(np.nanmean(M, axis=1))[::-1]
    M, names = M[order], [names[i] for i in order]
    lks_sorted = [lks[i] for i in order]

    # most responsive link (largest spread across poses) drives panels B and C
    key_i = int(np.nanargmax(np.nanmax(M, axis=1) - np.nanmin(M, axis=1)))
    key_lk, key_name = lks_sorted[key_i], names[key_i]

    # ---- panel B data: subcarrier x pose for that link --------------------
    base_sc = E[f'{key_lk}|a'].astype(float).mean(axis=0)
    S = np.array([20 * np.log10(np.maximum(P[p][f'{key_lk}|a'].astype(float).mean(axis=0), 1e-9)
                                / np.maximum(base_sc, 1e-9)) for p in POSES])
    valid = base_sc > 1.0          # guard-band subcarriers are ~0 and would divide to noise
    S = S[:, valid]

    # ---- panel C data: one waterfall per major axis -----------------------
    # A single link only shows one viewing angle. Selecting links by orientation
    # and stacking them shows the same six poses seen from across the room.
    axes_lks = pick_axes(lks_sorted, n=args.axes)
    waterfalls = []
    for lk in axes_lks:
        base = E[f'{lk}|a'].astype(float)
        v = base.mean(axis=0) > 1.0
        # Standardise EACH subcarrier against its own empty-room distribution:
        # subtract that subcarrier's empty mean and divide by its empty spread.
        # Dividing by the mean alone (a plain dB ratio) removes the offset but
        # leaves each subcarrier's dynamic range intact, so the few loud
        # subcarriers keep dominating the image and the quiet ones stay invisible.
        # In these units every subcarrier is on the same footing and a cell reads
        # directly as "sigma away from what an empty room does here".
        mu = base[:, v].mean(axis=0)
        sd = np.maximum(base[:, v].std(axis=0), 0.5)   # floor: dead subcarriers must not blow up
        segs, bounds, seg_lab = [], [], []
        for tag, d in [('empty', E)] + [(SHORT[i], P[p]) for i, p in enumerate(POSES)]:
            a = d[f'{lk}|a'].astype(float)[:, v]
            segs.append((a - mu[None, :]) / sd[None, :])
            bounds.append(sum(len(x) for x in segs))
            seg_lab.append(tag)
        waterfalls.append((np.vstack(segs).T, bounds, seg_lab, name(lk), angle_of(lk)))

    def sym_limit(arr, floor=1.0):
        """Symmetric colour limit about zero. Guarded: an all-NaN or all-zero
        panel would otherwise produce vmin==vcenter==vmax, which TwoSlopeNorm
        rejects."""
        m = np.nanmax(np.abs(arr)) if np.isfinite(arr).any() else 0.0
        if not np.isfinite(m):
            m = 0.0
        return max(float(np.ceil(m / 2) * 2), floor)

    lim = sym_limit(M)
    limS = sym_limit(S)
    print(f'  link-average limit +/-{lim:.1f} dB')

    nw = len(waterfalls)
    row_h = 2.5 if nw <= 3 else 1.85       # keep the sheet manageable with six rows
    fig = plt.figure(figsize=(16.5, 7.4 + row_h * nw), facecolor=SURFACE)
    # Hue only carries link identity while it stays within the categorical gate;
    # past three simultaneously-comparable series, identity moves to the angle
    # label and axis position instead of inventing more hues.
    tint = CAT if nw <= len(CAT) else [INK] * nw
    gs = fig.add_gridspec(1 + nw, 2, height_ratios=[1.25] + [0.62] * nw,
                          width_ratios=[1, 1.12],
                          hspace=0.52 if nw <= 3 else 0.62, wspace=0.20,
                          left=0.062, right=0.945, top=0.815, bottom=0.075)

    # ================= panel A =================
    ax = fig.add_subplot(gs[0, 0], facecolor=SURFACE)
    im = ax.imshow(M, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect='auto')
    ax.set_xticks(range(len(POSES)), PRETTY, color=INK_2, fontsize=9.5)
    ax.set_yticks(range(len(names)), names, color=INK_2, fontsize=9.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f'{M[i, j]:+.1f}', ha='center', va='center', fontsize=8,
                        color='#ffffff' if abs(M[i, j]) > lim * 0.55 else INK_2)
    ax.set_title(f'A · Each link, averaged over subcarriers',
                 color=INK, fontsize=12, pad=8, loc='left', fontweight='bold')
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('change vs empty (dB)', color=INK_2, fontsize=9)
    cb.ax.tick_params(colors=INK_2, length=0, labelsize=8)
    cb.outline.set_visible(False)

    # ================= panel B: the array =================
    # The geometry earns this slot more than another subcarrier heatmap did: the
    # waterfalls below are labelled by orientation, and this is what makes those
    # angles mean something.
    axg = fig.add_subplot(gs[0, 1], facecolor=SURFACE)
    for i, a in enumerate('ABCD'):
        for b in list('ABCD')[i + 1:]:
            if a in COORDS and b in COORDS:
                axg.plot(*zip(COORDS[a], COORDS[b]), color=GRID, lw=1.4, zorder=1)
    for (W, _bounds, _lab, lname, ang), col in zip(waterfalls, tint):
        a, b = lname.split('→')
        if a in COORDS and b in COORDS:
            axg.plot(*zip(COORDS[a], COORDS[b]),
                     color=col if nw <= len(CAT) else INK_2,
                     lw=2.8 if nw <= len(CAT) else 2.0, zorder=2)
            mx = (COORDS[a][0] + COORDS[b][0]) / 2
            my = (COORDS[a][1] + COORDS[b][1]) / 2
            axg.text(mx, my, f'{ang:.0f}°',
                     color=col if nw <= len(CAT) else INK_2,
                     fontsize=11 if nw <= len(CAT) else 9.5, fontweight='bold',
                     ha='center', va='center',
                     bbox=dict(fc=SURFACE, ec='none', pad=1.2), zorder=4)
    cx = np.mean([c[0] for c in COORDS.values()])
    cy = np.mean([c[1] for c in COORDS.values()])
    axg.plot(cx, cy, marker='*', ms=18, color=MUTED, zorder=3)
    axg.text(cx + 0.28, cy - 0.10, 'subject', color=MUTED, fontsize=9.5,
             ha='left', va='top')
    for k, (x, y) in COORDS.items():
        axg.plot(x, y, 'o', ms=13, color=INK, zorder=5)
        axg.text(x, y + 0.30, k, color=INK, fontsize=12.5, fontweight='bold',
                 ha='center', va='bottom', zorder=5)
    axg.set_aspect('equal')
    axg.set_xlabel('metres', color=INK_2, fontsize=9.5)
    axg.set_title('B · The array — every link below, labelled by orientation',
                  color=INK, fontsize=12, pad=8, loc='left', fontweight='bold')
    axg.tick_params(colors=INK_2, labelsize=8.5, length=0)
    for sp in axg.spines.values():
        sp.set_visible(False)
    axg.grid(color=GRID, lw=0.8)
    axg.set_axisbelow(True)
    axg.margins(0.14)

    # ================= panel C: one row per axis =================
    for r, (W, bounds, seg_lab, lname, ang) in enumerate(waterfalls):
        axr = fig.add_subplot(gs[1 + r, :], facecolor=SURFACE)
        # own scale per axis: a shared one flattens the quieter links to blank
        limR = max(float(np.ceil(np.nanpercentile(np.abs(W), 99) / 2) * 2), 1.0)
        im3 = axr.imshow(W, cmap=DIVERGING, aspect='auto', interpolation='nearest',
                         vmin=-limR, vmax=limR)
        for b in bounds[:-1]:
            axr.axvline(b, color=SURFACE, lw=2.5)
            axr.axvline(b, color=INK_2, lw=1.0, ls=(0, (3, 2)))
        mid = [(([0] + bounds)[i] + bounds[i]) / 2 for i in range(len(bounds))]
        last = (r == len(waterfalls) - 1)
        axr.set_xticks(mid, seg_lab if last else [''] * len(seg_lab),
                       color=INK_2, fontsize=9.5)
        axr.set_ylabel('subcarrier', color=INK_2, fontsize=9)
        axr.set_title(f'C{r + 1} · {lname}  —  {ang:.0f}° across the room, '
                      f'{W.shape[1]} packets, own scale ±{limR:.0f}σ',
                      color=tint[r], fontsize=11.5, pad=6, loc='left',
                      fontweight='bold')
        axr.tick_params(colors=INK_2, length=0, labelsize=9)
        for sp in axr.spines.values():
            sp.set_visible(False)
        cb3 = fig.colorbar(im3, ax=axr, fraction=0.02, pad=0.012)
        cb3.set_label('σ' if r == 0 else '', color=INK_2, fontsize=9)
        cb3.ax.tick_params(colors=INK_2, length=0, labelsize=8)
        cb3.outline.set_visible(False)

    net = np.nanmean(M)
    fig.text(0.062, 0.955, 'A person changes each Wi-Fi link differently — and each subcarrier differently again',
             fontsize=17.5, color=INK, fontweight='bold')
    fig.text(0.062, 0.912,
             f'{args.title} · 4 ESP32-S3 boards · {len(lks)} links · net mean {net:+.2f} dB, '
             f'individual links {np.nanmin(M):+.1f} to {np.nanmax(M):+.1f} dB, '
             f'axes below reach up to '
             f'{max(max(abs(np.nanpercentile(W, 1)), abs(np.nanpercentile(W, 99))) for W, *_ in waterfalls):.0f}σ '
             'from the empty room',
             fontsize=10.5, color=INK_2)
    fig.text(0.062, 0.876,
             'Rows C1-C' + str(len(waterfalls)) + ': every captured packet, on links spanning the widest range of '
             'orientations. Each subcarrier is standardised against its own empty-room mean and spread, so every\n'
             'subcarrier contributes equally instead of the few loudest ones dominating the picture.',
             fontsize=10, color=MUTED)

    fig.savefig(args.out, dpi=155, facecolor=SURFACE)
    print(f'wrote {args.out}  (panel B link {key_name}; '
          f'{len(waterfalls)} axes: ' +
          ', '.join(f'{n} @{a:.0f}°' for _W, _b, _l, n, a in waterfalls) + ')')
    print(f'  link range {np.nanmin(M):+.2f}..{np.nanmax(M):+.2f} dB, '
          f'subcarrier range {np.nanmin(S):+.2f}..{np.nanmax(S):+.2f} dB')


if __name__ == '__main__':
    main()
