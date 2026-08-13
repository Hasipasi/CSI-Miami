#!/usr/bin/env python3
"""Contact sheet of extracted skeletons, drawn from the stored .npz keypoints.

Deliberately *not* re-running the pose model: the point is to check the ground
truth that will actually be trained on, so any mistake in extraction, indexing or
frame alignment shows up here rather than surviving into the dataset.

  python3 plot_pose_gt.py --out pose_gt.png
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (0, 1), (0, 2), (1, 3), (2, 4)]
# limb colouring: legs / torso / arms / head, so a swapped left-right is visible
COL = {'leg': '#ffb02e', 'torso': '#f062c0', 'arm': '#3aa6ff', 'head': '#42d97a'}
PART = {(15, 13): 'leg', (13, 11): 'leg', (16, 14): 'leg', (14, 12): 'leg',
        (11, 12): 'torso', (5, 11): 'torso', (6, 12): 'torso', (5, 6): 'torso',
        (5, 7): 'arm', (6, 8): 'arm', (7, 9): 'arm', (8, 10): 'arm',
        (0, 1): 'head', (0, 2): 'head', (1, 3): 'head', (2, 4): 'head'}

SURFACE, INK, MUTED = '#101014', '#f0f0ee', '#8b8a86'


def panel(ax, frame_path, kp, conf, title, sub, thr=0.3):
    im = np.asarray(Image.open(frame_path))
    ok = np.isfinite(kp[:, 0]) & (kp[:, 2] >= thr)
    if ok.any():
        x0, x1 = np.min(kp[ok, 0]), np.max(kp[ok, 0])
        y0, y1 = np.min(kp[ok, 1]), np.max(kp[ok, 1])
        mx, my = 0.35 * (x1 - x0) + 30, 0.22 * (y1 - y0) + 30
        x0, x1 = max(x0 - mx, 0), min(x1 + mx, im.shape[1])
        y0, y1 = max(y0 - my, 0), min(y1 + my, im.shape[0])
    else:
        x0, y0, x1, y1 = 0, 0, im.shape[1], im.shape[0]
    ax.imshow(im)
    for a, b in SKELETON:
        if ok[a] and ok[b]:
            ax.plot([kp[a, 0], kp[b, 0]], [kp[a, 1], kp[b, 1]],
                    color=COL[PART[(a, b)]], lw=2.4, solid_capstyle='round', zorder=2)
    ax.scatter(kp[ok, 0], kp[ok, 1], s=16, c='#ffffff', edgecolors='#101014',
               linewidths=0.7, zorder=3)
    faint = np.isfinite(kp[:, 0]) & (kp[:, 2] < thr)
    if faint.any():                       # low-confidence joints, shown but marked
        ax.scatter(kp[faint, 0], kp[faint, 1], s=14, facecolors='none',
                   edgecolors='#ff5555', linewidths=1.2, zorder=3)
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('#2a2a30')
    ax.set_title(title, color=INK, fontsize=11, pad=5, fontweight='bold')
    ax.text(0.5, -0.035, sub, transform=ax.transAxes, ha='center', va='top',
            color=MUTED, fontsize=8.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', default='data')
    ap.add_argument('--out', default='pose_gt.png')
    ap.add_argument('--frame', type=int, default=75)
    args = ap.parse_args()

    subjects = ['gergo', 'balazs', 'yahya']
    acts = ['salute', '2handwave', 'squat', 'stretch-horizontal', 'stretch-vertical']
    fig, axes = plt.subplots(len(subjects), len(acts),
                             figsize=(4.0 * len(acts), 4.3 * len(subjects)),
                             facecolor=SURFACE)
    for r, subj in enumerate(subjects):
        for c, act in enumerate(acts):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            hit = None
            for sess in (f'{subj}_train', f'{subj}_test'):
                for rnd in range(10):
                    p = f'{args.src}/{sess}/{act}{rnd}_pose.npz'
                    if os.path.exists(p):
                        hit = (sess, f'{act}{rnd}', p)
                        break
                if hit:
                    break
            if hit is None:
                ax.axis('off')
                continue
            sess, take, ppath = hit
            P = np.load(ppath, allow_pickle=True)
            frames = sorted(glob.glob(f'{args.src}/{sess}/{take}_frames/*.jpg'))
            i = min(args.frame, len(frames) - 1)
            kp = P['keypoints'][i]
            panel(ax, frames[i], kp, float(P['det_conf'][i]),
                  f'{subj} · {act}',
                  f'{take} frame {i} · det {P["det_conf"][i]:.2f} · '
                  f'{int((kp[:, 2] >= 0.3).sum())}/17 kp')

    fig.suptitle('Extracted pose ground truth  ·  COCO-17 from yolo11x-pose, '
                 'drawn from the stored dataset keypoints',
                 color=INK, fontsize=15, fontweight='bold', y=0.985)
    fig.text(0.5, 0.005,
             'white = keypoint conf ≥ 0.3   ·   red ring = below 0.3 (still stored, flagged)   '
             '·   limb colours: arms blue, legs orange, torso pink, head green',
             ha='center', color=MUTED, fontsize=9.5)
    fig.tight_layout(rect=[0, 0.022, 1, 0.965])
    fig.savefig(args.out, dpi=125, facecolor=SURFACE)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
