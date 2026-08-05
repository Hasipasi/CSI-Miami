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
SEQ = LinearSegmentedColormap.from_list('amp', [
    '#f0efec', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#104281', '#0d366b'])


def links(d):
    return sorted({k.rsplit('|', 1)[0] for k in d.files if k.endswith('|t')})


def name(lk):
    tx, rx = lk.split('|')
    return f'{LABEL.get(tx[-5:], tx[-5:])}→{LABEL.get(rx[-5:], rx[-5:])}'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix', default='ps')
    ap.add_argument('--out', default='pose_detail.png')
    ap.add_argument('--title', default='Round-robin, session 1')
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

    # ---- panel C data: per-packet, normalised per subcarrier --------------
    # Each subcarrier is divided by its OWN empty-room level before plotting.
    # Raw amplitude is dominated by static frequency-selective fading -- strong
    # horizontal banding that is a property of the room, not of the person --
    # and that banding swamps the change we actually want to see. Normalising
    # per subcarrier removes it and puts every subcarrier on the same dB scale.
    ref_sc = E[f'{key_lk}|a'].astype(float)[:, valid].mean(axis=0)
    segs, bounds, seg_lab = [], [], []
    for tag, d in [('empty', E)] + [(SHORT[i], P[p]) for i, p in enumerate(POSES)]:
        a = d[f'{key_lk}|a'].astype(float)[:, valid]
        segs.append(20 * np.log10(np.maximum(a, 1e-9) / np.maximum(ref_sc, 1e-9)[None, :]))
        bounds.append(sum(len(s) for s in segs))
        seg_lab.append(tag)
    RAW = np.vstack(segs).T          # [subcarrier x packet], dB vs own empty level

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
    print(f'  panel A limit +/-{lim:.1f} dB, panel B limit +/-{limS:.1f} dB, '
          f'S finite {np.isfinite(S).sum()}/{S.size}')

    fig = plt.figure(figsize=(16.5, 9.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.82], width_ratios=[1, 1.12],
                          hspace=0.60, wspace=0.20,
                          left=0.062, right=0.945, top=0.815, bottom=0.085)

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

    # ================= panel B =================
    ax2 = fig.add_subplot(gs[0, 1], facecolor=SURFACE)
    im2 = ax2.imshow(S, cmap=DIVERGING, vmin=-limS, vmax=limS,
                     aspect='auto', interpolation='nearest')
    ax2.set_yticks(range(len(POSES)), SHORT, color=INK_2, fontsize=9.5)
    ax2.set_xlabel('subcarrier index', color=INK_2, fontsize=9.5)
    ax2.set_title(f'B · The same link ({key_name}) resolved across frequency',
                  color=INK, fontsize=12, pad=8, loc='left', fontweight='bold')
    ax2.tick_params(colors=INK_2, length=0, labelsize=8.5)
    for s in ax2.spines.values():
        s.set_visible(False)
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.04, pad=0.02)
    cb2.set_label('change vs empty (dB)', color=INK_2, fontsize=9)
    cb2.ax.tick_params(colors=INK_2, length=0, labelsize=8)
    cb2.outline.set_visible(False)
    ax2.text(0.0, -0.235, 'the per-link average in panel A hides this: the response is '
                          'strongly frequency-selective, and\ndifferent poses light up different '
                          f'subcarrier bands — note this scale runs to ±{limS:.0f} dB, not ±{lim:.0f}',
             transform=ax2.transAxes, fontsize=9, color=MUTED, va='top')

    # ================= panel C =================
    ax3 = fig.add_subplot(gs[1, :], facecolor=SURFACE)
    limR = max(float(np.ceil(np.nanpercentile(np.abs(RAW), 99) / 2) * 2), 1.0)
    im3 = ax3.imshow(RAW, cmap=DIVERGING, aspect='auto', interpolation='nearest',
                     vmin=-limR, vmax=limR)
    for b in bounds[:-1]:
        ax3.axvline(b, color=SURFACE, lw=2.5)
        ax3.axvline(b, color=INK_2, lw=1.0, ls=(0, (3, 2)))
    mid = [(0 if i == 0 else bounds[i - 1] + bounds[i]) / 2 if i else bounds[0] / 2
           for i in range(len(bounds))]
    mid = [(([0] + bounds)[i] + bounds[i]) / 2 for i in range(len(bounds))]
    ax3.set_xticks(mid, seg_lab, color=INK_2, fontsize=9.5)
    ax3.set_ylabel('subcarrier index', color=INK_2, fontsize=9.5)
    ax3.set_title(f'C · Every captured packet — {key_name}, each subcarrier '
                  'normalised against its own empty-room level',
                  color=INK, fontsize=12, pad=8, loc='left', fontweight='bold')
    ax3.tick_params(colors=INK_2, length=0, labelsize=9)
    for s in ax3.spines.values():
        s.set_visible(False)
    cb3 = fig.colorbar(im3, ax=ax3, fraction=0.018, pad=0.012)
    cb3.set_label('change vs empty, per subcarrier (dB)', color=INK_2, fontsize=9)
    cb3.ax.tick_params(colors=INK_2, length=0, labelsize=8)
    cb3.outline.set_visible(False)
    ax3.text(0.0, -0.155, f'{RAW.shape[1]} packets × {RAW.shape[0]} subcarriers. Each block is one '
                          'held pose; the empty block at left is flat by construction (it is the reference). '
                          'Without this normalisation the panel is dominated by static fading — a property of '
                          'the room, not the person.',
             transform=ax3.transAxes, fontsize=9, color=MUTED, va='top')

    net = np.nanmean(M)
    fig.text(0.062, 0.945, 'A person changes each Wi-Fi link differently — and each subcarrier differently again',
             fontsize=17.5, color=INK, fontweight='bold')
    fig.text(0.062, 0.895,
             f'{args.title} · 4 ESP32-S3 boards · {len(lks)} links · net mean {net:+.2f} dB, '
             f'individual links {np.nanmin(M):+.1f} to {np.nanmax(M):+.1f} dB, '
             f'individual subcarriers {np.nanmin(S):+.1f} to {np.nanmax(S):+.1f} dB',
             fontsize=10.5, color=INK_2)

    fig.savefig(args.out, dpi=155, facecolor=SURFACE)
    print(f'wrote {args.out}  (key link {key_name}, {RAW.shape[1]} packets, '
          f'{RAW.shape[0]} usable subcarriers)')
    print(f'  link range {np.nanmin(M):+.2f}..{np.nanmax(M):+.2f} dB, '
          f'subcarrier range {np.nanmin(S):+.2f}..{np.nanmax(S):+.2f} dB')


if __name__ == '__main__':
    main()
