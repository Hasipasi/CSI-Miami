#!/usr/bin/env python3
"""Plot how a person changes each link's CSI amplitude, per pose, in dB.

The quantity is signed (a body blocks some paths and creates reflections on
others), so the colour job is polarity, not magnitude: a diverging blue/red
ramp with a neutral midpoint and limits kept symmetric about 0 dB, so "gain"
and "loss" of equal size read equally strongly.

  python3 plot_pose_db.py --prefix ps --out pose_db.png
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

LABEL = {'ab:d4': 'C', '2d:3c': 'A', '2d:a8': 'B', '6b:5c': 'D'}
POSES = ['neutral', 'tpose', 'up', 'split', 'crouch', 'turned']
PRETTY = ['arms\ndown', 'T-pose', 'arms\nup', 'split\narms', 'crouch', 'turned\n90°']

# design tokens
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
# diverging pair: red (loss) <- neutral gray -> blue (gain), equal steps per arm
DIVERGING = LinearSegmentedColormap.from_list('loss_gain', [
    '#d03b3b', '#e34948', '#e66767', '#f0efec', '#86b6ef', '#3987e5', '#1c5cab'])
EMPH_GAIN = '#2a78d6'
EMPH_LOSS = '#d03b3b'


def links(d):
    return sorted({k.rsplit('|', 1)[0] for k in d.files if k.endswith('|t')})


def name(lk):
    tx, rx = lk.split('|')
    return f'{LABEL.get(tx[-5:], tx[-5:])}→{LABEL.get(rx[-5:], rx[-5:])}'


def load(prefix):
    E = np.load(f'{prefix}_empty.npz')
    lks = links(E)
    rows, names = [], []
    for lk in lks:
        base = E[f'{lk}|a'].astype(float).mean()
        vals = []
        for p in POSES:
            d = np.load(f'{prefix}_{p}.npz')
            if f'{lk}|a' not in d.files:
                vals.append(np.nan)
                continue
            vals.append(20 * np.log10(max(d[f'{lk}|a'].astype(float).mean(), 1e-9) / max(base, 1e-9)))
        rows.append(vals)
        names.append(name(lk))
    M = np.array(rows)
    order = np.argsort(np.nanmean(M, axis=1))[::-1]   # gainers top, losers bottom
    return M[order], [names[i] for i in order]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix', default='ps')
    ap.add_argument('--out', default='pose_db.png')
    ap.add_argument('--title', default='Round-robin, session 1')
    args = ap.parse_args()

    M, names = load(args.prefix)
    lim = np.nanmax(np.abs(M))
    lim = np.ceil(lim / 2) * 2                       # symmetric about zero

    fig = plt.figure(figsize=(13.5, 6.6), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.30,
                          left=0.07, right=0.90, top=0.78, bottom=0.13)

    # ---- panel A: the full grid -------------------------------------------
    ax = fig.add_subplot(gs[0, 0], facecolor=SURFACE)
    im = ax.imshow(M, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect='auto')
    ax.set_xticks(range(len(POSES)), PRETTY, color=INK_2, fontsize=10)
    ax.set_yticks(range(len(names)), names, color=INK_2, fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            # value ink stays neutral; the cell colour carries the sign
            ax.text(j, i, f'{v:+.1f}', ha='center', va='center', fontsize=8.5,
                    color='#ffffff' if abs(v) > lim * 0.55 else INK_2)
    ax.set_title('Every link, every pose', color=INK, fontsize=12.5,
                 pad=10, loc='left', fontweight='bold')
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label('change vs empty room (dB)', color=INK_2, fontsize=10)
    cb.ax.tick_params(colors=INK_2, length=0, labelsize=9)
    cb.outline.set_visible(False)

    # ---- panel B: the two links that carry the story ----------------------
    ax2 = fig.add_subplot(gs[0, 1], facecolor=SURFACE)
    gain_i = int(np.argmax(np.nanmean(M, axis=1)))
    loss_i = int(np.argmin(np.nanmean(M, axis=1)))
    x = np.arange(len(POSES))

    for i in range(M.shape[0]):
        if i in (gain_i, loss_i):
            continue
        ax2.plot(x, M[i], color=MUTED, lw=1.2, alpha=0.45, zorder=1)
    ax2.axhline(0, color=INK_2, lw=1.2, ls=(0, (4, 3)), zorder=2)

    for i, col in ((gain_i, EMPH_GAIN), (loss_i, EMPH_LOSS)):
        ax2.plot(x, M[i], color=col, lw=2, marker='o', ms=8,
                 mec=SURFACE, mew=2, zorder=3, label=names[i])

    # Labels go in reserved space to the RIGHT of the last point: putting them at
    # the extremes collided with the tick labels and with each other.
    ax2.set_xlim(-0.35, len(POSES) - 1 + 1.35)
    ax2.annotate(f'{names[gain_i]}  weak link\nbody adds\na reflection',
                 xy=(x[-1] + 0.15, M[gain_i, -1]), ha='left', va='center',
                 fontsize=9.5, color=EMPH_GAIN, linespacing=1.5)
    ax2.annotate(f'{names[loss_i]}  strong link\nbody blocks it',
                 xy=(x[-1] + 0.15, M[loss_i, -1]), ha='left', va='center',
                 fontsize=9.5, color=EMPH_LOSS, linespacing=1.5)
    # caption sits on the baseline it names
    ax2.text(-0.25, 0.5, 'empty room', fontsize=9, color=MUTED,
             va='bottom', ha='left')

    ax2.set_xticks(x, PRETTY, color=INK_2, fontsize=10)
    ax2.set_ylabel('change vs empty room (dB)', color=INK_2, fontsize=10)
    ax2.set_title('The body redistributes, it does not just absorb',
                  color=INK, fontsize=12.5, pad=26, loc='left', fontweight='bold')
    ax2.tick_params(colors=INK_2, length=0, labelsize=9)
    ax2.grid(axis='y', color=GRID, lw=1)
    ax2.set_axisbelow(True)
    for s in ('top', 'right', 'bottom', 'left'):
        ax2.spines[s].set_visible(False)
    # legend above the plot area, so it cannot land on a mark
    leg = ax2.legend(frameon=False, fontsize=9.5, ncol=2,
                     loc='lower left', bbox_to_anchor=(0, 1.0))
    for t in leg.get_texts():
        t.set_color(INK_2)

    net = np.nanmean(M)
    fig.text(0.07, 0.935, 'A person changes each Wi-Fi link differently',
             fontsize=17, color=INK, fontweight='bold')
    fig.text(0.07, 0.875,
             f'{args.title} · 4 ESP32-S3 boards · net mean {net:+.2f} dB, '
             f'but individual links swing {np.nanmin(M):+.1f} to {np.nanmax(M):+.1f} dB',
             fontsize=10.5, color=INK_2)

    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    print(f'wrote {args.out}')
    print(f'  net mean {net:+.2f} dB, range {np.nanmin(M):+.2f} to {np.nanmax(M):+.2f} dB')


if __name__ == '__main__':
    main()
