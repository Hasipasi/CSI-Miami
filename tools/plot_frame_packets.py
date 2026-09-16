#!/usr/bin/env python3
"""Which transmitter reached which receiver inside one camera frame, drawn.

Round-robin exists to put every link into every frame; this is the picture that
says whether it did. For a chosen frame of a capture:

  * the camera frame itself (the JPEG or depth PNG embedded in the NPZ)
  * a transmitter x receiver grid of packet counts for that frame's interval --
    a board never hears itself, so the diagonal is 0
  * the packets on a timeline inside the frame, one row per directed link grouped
    by transmitter, so the token going round is visible as diagonal stripes
  * every frame of the capture as a strip (links x frames, packets per cell), the
    chosen frame outlined, so one frame can be judged against the rest

  python3 plot_frame_packets.py data/session/take3.npz            # middle frame
  python3 plot_frame_packets.py take3.npz --frame 45 --out f45.png
  python3 plot_frame_packets.py take3.npz --window 16.7            # +-half a frame

A frame's packets are those within --window ms of its timestamp on either side
(default: half the frame period, so consecutive frames' windows are disjoint).
Magnitude is one blue ramp, light to dark; an empty cell is the page colour.
"""

import argparse
import io
import json
import os
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap   # noqa: E402
from matplotlib.patches import Rectangle              # noqa: E402
import numpy as np                                    # noqa: E402
from PIL import Image                                 # noqa: E402

from capture import frame_half_window, frame_window_counts   # noqa: E402

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#e3e2dd'
BLUE = '#2a78d6'
# Sequential blue, light -> dark (steps 100..700 of the reference palette).
RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
        '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']


def count_cmap(vmax):
    """0 is the page colour; 1..vmax step up the ramp."""
    steps = [SURFACE] + [RAMP[int(round(i * (len(RAMP) - 1) / max(vmax - 1, 1)))]
                         for i in range(vmax)]
    return ListedColormap(steps), BoundaryNorm(np.arange(-0.5, vmax + 1), len(steps))


def load(path):
    d = np.load(path)
    meta = json.loads(str(d['meta']))
    lab = meta.get('boards', {})
    ft = d['frame_t']
    links = sorted({k.rsplit('|', 1)[0] for k in d.files if k.endswith('|t')})
    boards = sorted(set(lab.values())) or sorted({m for lk in links for m in lk.split('|')})
    return d, meta, lab, ft, links, boards


def frame_image(path, idx, folder='frames'):
    with zipfile.ZipFile(path) as zf:
        for ext in ('jpg', 'png'):
            name = f'{folder}/{idx:06d}.{ext}'
            if name in zf.namelist():
                img = Image.open(io.BytesIO(zf.read(name)))
                if img.mode == 'I;16':            # depth: show as a grey ramp
                    a = np.asarray(img, dtype=np.float32)
                    a = np.clip(a / max(np.percentile(a[a > 0], 99), 1), 0, 1)
                    return (255 * (1 - a)).astype(np.uint8), 'depth'
                return np.asarray(img.convert('RGB')), 'colour'
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('capture')
    ap.add_argument('--frame', type=int, default=None, help='frame index (default: middle)')
    ap.add_argument('--out', default=None, help='PNG path (default: beside the capture)')
    ap.add_argument('--window', type=float, default=None,
                    help='ms either side of the frame time that count as its CSI '
                         '(default: half the frame period -- disjoint windows)')
    args = ap.parse_args()

    d, meta, lab, ft, links, boards = load(args.capture)
    half = args.window / 1000.0 if args.window else frame_half_window(ft)
    args.window = 1000 * half
    n = len(ft)
    if n < 1:
        raise SystemExit('no frames in the capture')
    i = n // 2 if args.frame is None else int(args.frame)
    if not 0 <= i < n:
        raise SystemExit(f'--frame must be 0..{n - 1}')
    tf = float(ft[i])
    t0, t1 = tf - half, tf + half
    name = lambda mac: lab.get(mac, mac[-5:])       # noqa: E731

    # packets per link per frame window, for the strip and the chosen frame
    per = {lk: frame_window_counts(ft, d[f'{lk}|t'], half) for lk in links}
    order = [lk for tx in boards for lk in links if name(lk.split('|')[0]) == tx]
    mat = np.zeros((len(boards), len(boards)), dtype=int)
    for lk in links:
        tx, rx = lk.split('|')
        mat[boards.index(name(tx)), boards.index(name(rx))] = per[lk][i]
    covered = np.all([per[lk] > 0 for lk in links], axis=0)

    fig = plt.figure(figsize=(16, 9.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], width_ratios=[1.25, 0.8, 1.3],
                          hspace=0.42, wspace=0.28, left=0.05, right=0.985,
                          top=0.88, bottom=0.08)
    fidx = int(d['frame_idx'][i]) if 'frame_idx' in d.files else i
    take = os.path.basename(args.capture)
    fig.suptitle(f'{take} · frame {fidx} at {tf * 1000:.0f} ms after record start · '
                 f'its CSI window is ±{args.window:.1f} ms · {mat.sum()} packets on '
                 f'{int((mat > 0).sum())} of {len(links)} links',
                 color=INK, fontsize=13, x=0.05, ha='left')

    # ---- the frame, and the depth frame nearest to it when the take has depth
    has_depth = 'depth_t' in d.files and 'frame_depth_idx' in d.files
    sub = gs[0, 0].subgridspec(2 if has_depth else 1, 1, hspace=0.12)
    ax = fig.add_subplot(sub[0, 0])
    img, kind = frame_image(args.capture, fidx)
    if img is not None:
        ax.imshow(img, cmap='gray' if kind == 'depth' else None)
        ax.set_title(f'camera frame ({kind})', color=INK2, fontsize=10, loc='left')
    else:
        ax.text(0.5, 0.5, 'no image in the capture', ha='center', color=INK2)
    ax.set_axis_off()
    if has_depth:
        ax = fig.add_subplot(sub[1, 0])
        didx = int(d['frame_depth_idx'][i])
        ddt = float(d['frame_depth_dt'][i]) * 1000
        dimg, _ = frame_image(args.capture, didx, folder='depth')
        if dimg is not None:
            ax.imshow(dimg, cmap='gray')
            ax.set_title(f'depth frame {didx}, {ddt:+.1f} ms from the colour frame '
                         f'(near dark, far light)', color=INK2, fontsize=10, loc='left')
        else:
            ax.text(0.5, 0.5, 'no depth image', ha='center', color=INK2)
        ax.set_axis_off()

    # ---- TX x RX grid for this frame
    ax = fig.add_subplot(gs[0, 1])
    vmax = max(int(mat.max()), 1)
    cmap, norm = count_cmap(vmax)
    ax.imshow(mat, cmap=cmap, norm=norm, aspect='equal')
    for r in range(len(boards)):
        for c in range(len(boards)):
            v = int(mat[r, c])
            dark = v > 0.6 * vmax
            ax.text(c, r, str(v), ha='center', va='center', fontsize=15,
                    color=(SURFACE if dark else INK) if r != c else INK2,
                    fontweight='bold' if r != c else 'normal')
    ax.set_xticks(range(len(boards)), boards)
    ax.set_yticks(range(len(boards)), boards)
    ax.tick_params(length=0, colors=INK2, labelsize=11)
    ax.xaxis.set_ticks_position('top')
    ax.set_xlabel('received by', color=INK2, fontsize=10)
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('sent by', color=INK2, fontsize=10)
    ax.set_xticks(np.arange(-0.5, len(boards)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(boards)), minor=True)
    ax.grid(which='minor', color=SURFACE, linewidth=2)
    ax.tick_params(which='minor', length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f'packets within ±{args.window:.1f} ms of the frame\n(a board never hears itself: 0)',
                 color=INK2, fontsize=10, loc='left', pad=34)

    # ---- timeline inside the frame
    ax = fig.add_subplot(gs[0, 2])
    ylabels = []
    for k, lk in enumerate(order):
        tx, rx = lk.split('|')
        t = d[f'{lk}|t'].astype(np.float64)
        m = (t >= t0) & (t <= t1)
        ax.plot((t[m] - tf) * 1000, np.full(m.sum(), k), 'o', ms=7, color=BLUE,
                markeredgecolor=SURFACE, markeredgewidth=1)
        ylabels.append(f'{name(tx)} → {name(rx)}')
    for g in range(1, len(boards)):
        ax.axhline(g * (len(boards) - 1) - 0.5, color=GRID, linewidth=1)
    # the frame itself, and its neighbours, as time marks
    ax.axvline(0, color=INK, linewidth=1.2)
    for k in (i - 1, i + 1):
        if 0 <= k < n and t0 <= ft[k] <= t1:
            ax.axvline((ft[k] - tf) * 1000, color=INK2, linewidth=1, linestyle=(0, (1, 2)))
    ax.set_yticks(range(len(order)), ylabels)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(-args.window, args.window)
    ax.set_xlabel(f'ms relative to frame {fidx} (the line at 0)', color=INK2, fontsize=10)
    ax.tick_params(length=0, colors=INK2, labelsize=9)
    ax.grid(axis='x', color=GRID, linewidth=1)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title('each packet in the window, by link (rows grouped by transmitter)',
                 color=INK2, fontsize=10, loc='left')

    # ---- every frame of the capture
    ax = fig.add_subplot(gs[1, :])
    strip = np.array([per[lk] for lk in order])
    vmax2 = max(int(strip.max()), 1)
    cmap2, norm2 = count_cmap(vmax2)
    ax.imshow(strip, cmap=cmap2, norm=norm2, aspect='auto', interpolation='nearest')
    ax.add_patch(Rectangle((i - 0.5, -0.5), 1, len(order), fill=False, edgecolor=INK,
                           linewidth=1.5))
    ax.set_yticks(range(len(order)), ylabels)
    for g in range(1, len(boards)):
        ax.axhline(g * (len(boards) - 1) - 0.5, color=SURFACE, linewidth=2)
    missing = np.flatnonzero(~covered)
    for k in missing:
        ax.plot(k, -0.9, 'v', color='#d03b3b', ms=6, clip_on=False)
    ax.set_xlabel('frame', color=INK2, fontsize=10)
    ax.tick_params(length=0, colors=INK2, labelsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f'packets per link in every frame\'s ±{args.window:.1f} ms window · '
                 f'{100 * covered.mean():.1f}% of {n} frames have every link'
                 + (f' · ▼ marks the {len(missing)} that miss one' if len(missing) else ''),
                 color=INK2, fontsize=10, loc='left', pad=14)
    # a small key for the ramp: 0 is the page, then light -> dark
    kx = 0.86
    for v in range(vmax2 + 1):
        fig.patches.append(Rectangle((kx + v * 0.012, 0.035), 0.011, 0.018,
                                     transform=fig.transFigure, facecolor=cmap2(norm2(v)),
                                     edgecolor=GRID, linewidth=0.5))
        fig.text(kx + v * 0.012 + 0.0055, 0.022, str(v), ha='center', va='top',
                 fontsize=8, color=INK2)
    fig.text(kx - 0.005, 0.044, 'packets', ha='right', va='center', fontsize=9, color=INK2)

    out = args.out or os.path.splitext(args.capture)[0] + f'_frame{fidx}.png'
    fig.savefig(out, dpi=120, facecolor=SURFACE)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
