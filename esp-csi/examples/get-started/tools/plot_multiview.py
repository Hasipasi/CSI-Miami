#!/usr/bin/env python3
"""What round-robin buys: the same person viewed along several link angles at once.

A fixed transmitter only ever probes the room along links radiating from one
board. Round-robin gives every ordered pair, so the subject is crossed from
several directions in the same capture -- and because the array is not
collinear, those directions differ a lot: A-B runs horizontally across the
room while C-D runs almost perpendicular to it.

This figure puts the array geometry next to the per-link response, then resolves
three links of differing angle across frequency. A fixed-TX capture cannot
produce the bottom row at all, which is the point.

Colours for the highlighted links are the reference palette's first three
categorical slots, documented there as validated all-pairs (the skill caps
all-pairs forms at three regardless).

  python3 plot_multiview.py --prefix ps --out multiview_rr.png
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
SHORT = ['arms down', 'T-pose', 'arms up', 'split arms', 'crouch', 'turned 90°']

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
CAT = ['#2a78d6', '#eb6834', '#1baf7a']          # palette slots 1-3
DIVERGING = LinearSegmentedColormap.from_list('loss_gain', [
    '#d03b3b', '#e34948', '#e66767', '#f0efec', '#86b6ef', '#3987e5', '#1c5cab'])


def links(d):
    return sorted({k.rsplit('|', 1)[0] for k in d.files if k.endswith('|t')})


def ends(lk):
    tx, rx = lk.split('|')
    return LABEL.get(tx[-5:], '?'), LABEL.get(rx[-5:], '?')


def angle_of(a, b):
    """Link orientation in degrees, folded to 0-180 (a link has no direction)."""
    (x1, y1), (x2, y2) = COORDS[a], COORDS[b]
    return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix', default='ps')
    ap.add_argument('--out', default='multiview.png')
    ap.add_argument('--title', default='Round-robin, session 1')
    args = ap.parse_args()

    E = np.load(f'{args.prefix}_empty.npz')
    P = {p: np.load(f'{args.prefix}_{p}.npz') for p in POSES}
    lks = links(E)

    rows, names, lk_keep = [], [], []
    for lk in lks:
        base = E[f'{lk}|a'].astype(float).mean()
        a, b = ends(lk)
        rows.append([20 * np.log10(max(P[p][f'{lk}|a'].astype(float).mean(), 1e-9) / max(base, 1e-9))
                     for p in POSES])
        names.append(f'{a}→{b}')
        lk_keep.append(lk)
    M = np.array(rows)
    order = np.argsort(np.nanmean(M, axis=1))[::-1]
    M, names, lk_keep = M[order], [names[i] for i in order], [lk_keep[i] for i in order]

    # three links of differing orientation. A-C is the most responsive, C-D is
    # near-perpendicular to A-B, and A-B is the horizontal reference.
    want = [('A', 'C'), ('C', 'D'), ('A', 'B')]
    chosen = []
    for a, b in want:
        for i, lk in enumerate(lk_keep):
            if set(ends(lk)) == {a, b}:
                chosen.append((i, lk, f'{a}↔{b}', angle_of(a, b)))
                break
    if len(chosen) < 2:
        print('this figure needs several link directions; a fixed-TX capture '
              'only has links from one board. Use --prefix of a round-robin run.')
        return

    fig = plt.figure(figsize=(16.2, 10.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95], width_ratios=[0.82, 1, 1],
                          hspace=0.34, wspace=0.26,
                          left=0.055, right=0.945, top=0.815, bottom=0.125)

    # ---------- geometry ----------
    axg = fig.add_subplot(gs[0, 0], facecolor=SURFACE)
    for i, a in enumerate('ABCD'):
        for b in list('ABCD')[i + 1:]:
            axg.plot(*zip(COORDS[a], COORDS[b]), color=GRID, lw=1.4, zorder=1)
    for (idx, lk, lab, ang), col in zip(chosen, CAT):
        a, b = lab.split('↔')
        axg.plot(*zip(COORDS[a], COORDS[b]), color=col, lw=2.6, zorder=2)
        mx = (COORDS[a][0] + COORDS[b][0]) / 2
        my = (COORDS[a][1] + COORDS[b][1]) / 2
        axg.text(mx, my, f' {ang:.0f}°', color=col, fontsize=10.5,
                 fontweight='bold', ha='center', va='center',
                 bbox=dict(fc=SURFACE, ec='none', pad=1.2), zorder=4)
    cx = np.mean([c[0] for c in COORDS.values()])
    cy = np.mean([c[1] for c in COORDS.values()])
    axg.plot(cx, cy, marker='*', ms=17, color=MUTED, zorder=3)
    axg.text(cx + 0.30, cy - 0.10, 'subject', color=MUTED, fontsize=9.5,
             ha='left', va='top')
    for k, (x, y) in COORDS.items():
        axg.plot(x, y, 'o', ms=13, color=INK, zorder=5)
        axg.text(x, y + 0.34, k, color=INK, fontsize=12, fontweight='bold',
                 ha='center', va='bottom', zorder=5)
    axg.set_aspect('equal')
    axg.set_xlabel('metres', color=INK_2, fontsize=9.5)
    axg.set_title('The array', color=INK, fontsize=12, pad=8, loc='left', fontweight='bold')
    axg.tick_params(colors=INK_2, labelsize=8.5, length=0)
    for s in axg.spines.values():
        s.set_visible(False)
    axg.grid(color=GRID, lw=0.8)
    axg.set_axisbelow(True)
    axg.margins(0.16)

    # ---------- link x pose ----------
    axh = fig.add_subplot(gs[0, 1:], facecolor=SURFACE)
    lim = max(float(np.ceil(np.nanmax(np.abs(M)) / 2) * 2), 1.0)
    im = axh.imshow(M, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect='auto')
    axh.set_xticks(range(len(POSES)), SHORT, color=INK_2, fontsize=9.5)
    axh.set_yticks(range(len(names)), names, color=INK_2, fontsize=9.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            axh.text(j, i, f'{M[i, j]:+.1f}', ha='center', va='center', fontsize=8,
                     color='#ffffff' if abs(M[i, j]) > lim * 0.55 else INK_2)
    # mark the selected links by colouring their tick label. An outline over the
    # cells is invisible when the row's own colour matches it, and a bar in the
    # margin covers the label it is meant to identify.
    sel = {idx: col for (idx, _lk, _lab, _ang), col in zip(chosen, CAT)}
    for i, t in enumerate(axh.get_yticklabels()):
        if i in sel:
            t.set_color(sel[i])
            t.set_fontweight('bold')
            t.set_fontsize(11)
    axh.set_title(f'All {len(names)} links · a fixed transmitter would give only 3 of these rows',
                  color=INK, fontsize=12, pad=8, loc='left', fontweight='bold')
    for s in axh.spines.values():
        s.set_visible(False)
    axh.tick_params(length=0)
    cb = fig.colorbar(im, ax=axh, fraction=0.028, pad=0.015)
    cb.set_label('change vs empty (dB)', color=INK_2, fontsize=9)
    cb.ax.tick_params(colors=INK_2, length=0, labelsize=8)
    cb.outline.set_visible(False)

    # ---------- per-angle subcarrier structure ----------
    panels = []
    for idx, lk, lab, ang in chosen:
        base_sc = E[f'{lk}|a'].astype(float).mean(axis=0)
        valid = base_sc > 1.0
        S = np.array([20 * np.log10(np.maximum(P[p][f'{lk}|a'].astype(float).mean(axis=0), 1e-9)
                                    / np.maximum(base_sc, 1e-9)) for p in POSES])[:, valid]
        panels.append((S, lab, ang))
    limS = max(float(np.ceil(max(np.nanmax(np.abs(S)) for S, _, _ in panels) / 2) * 2), 1.0)

    for k, ((S, lab, ang), col) in enumerate(zip(panels, CAT)):
        axs = fig.add_subplot(gs[1, k], facecolor=SURFACE)
        im2 = axs.imshow(S, cmap=DIVERGING, vmin=-limS, vmax=limS,
                         aspect='auto', interpolation='nearest')
        axs.set_yticks(range(len(POSES)), SHORT if k == 0 else [''] * len(POSES),
                       color=INK_2, fontsize=9.5)
        axs.set_xlabel('subcarrier index', color=INK_2, fontsize=9.5)
        axs.set_title(f'{lab}   {ang:.0f}° across the room', color=col,
                      fontsize=12, pad=28, loc='left', fontweight='bold')
        axs.text(0.0, 1.018, f'spans {np.nanmin(S):+.1f} to {np.nanmax(S):+.1f} dB',
                 transform=axs.transAxes, fontsize=9, color=MUTED, va='bottom')
        axs.tick_params(colors=INK_2, length=0, labelsize=8.5)
        for s in axs.spines.values():
            s.set_visible(False)
        for sp in ('bottom', 'left', 'top', 'right'):
            axs.spines[sp].set_visible(False)
        # a coloured rule ties the panel back to its link in the geometry
        axs.add_patch(plt.Rectangle((0, -0.6), S.shape[1], 0.16, color=col,
                                    clip_on=False, zorder=6))
        if k == len(panels) - 1:
            cb2 = fig.colorbar(im2, ax=axs, fraction=0.03, pad=0.015)
            cb2.set_label('change vs empty (dB)', color=INK_2, fontsize=9)
            cb2.ax.tick_params(colors=INK_2, length=0, labelsize=8)
            cb2.outline.set_visible(False)

    fig.text(0.055, 0.945,
             'Round-robin sees the same person from several angles at once',
             fontsize=17.5, color=INK, fontweight='bold')
    fig.text(0.055, 0.897,
             f'{args.title} · the three highlighted links cross the room at '
             f'{chosen[0][3]:.0f}°, {chosen[1][3]:.0f}° and {chosen[2][3]:.0f}° — '
             'each resolves the same pose into a different subcarrier pattern',
             fontsize=10.5, color=INK_2)
    fig.text(0.055, 0.020,
             'Bottom row: same six poses, same colour scale, three viewing angles. The shared scale is deliberate — a pose that\n'
             'dominates one angle can barely register on another, which is the information a single transmitter cannot reach.\n'
             'Each panel also states its own range, since the shared scale flattens the quieter links.',
             fontsize=9.5, color=MUTED, linespacing=1.5)

    fig.savefig(args.out, dpi=155, facecolor=SURFACE)
    print(f'wrote {args.out}')
    for _, lk, lab, ang in chosen:
        print(f'  {lab} at {ang:.1f}°')


if __name__ == '__main__':
    main()
