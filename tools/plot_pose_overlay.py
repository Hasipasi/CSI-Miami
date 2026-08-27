#!/usr/bin/env python3
"""Draw stored YOLO COCO-17 poses over frames embedded in a CSI capture.

This reads the saved pose arrays rather than running inference again, making it a
visual check of the exact labels and timestamp alignment that downstream training
will consume.
"""

import argparse
import io
import os
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
            (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
            (8, 10), (0, 1), (0, 2), (1, 3), (2, 4)]
LIMB_COLOR = ['#ffb02e'] * 4 + ['#f062c0'] * 4 + ['#3aa6ff'] * 4 + ['#42d97a'] * 4


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('capture', help='self-contained CSI .npz containing frames/')
    ap.add_argument('--pose', help='pose NPZ (default: <capture>_pose.npz)')
    ap.add_argument('-o', '--output', default='pose_overlay.png')
    ap.add_argument('--indices', default='0,30,60,90,120,150',
                    help='comma-separated pose-row indices')
    ap.add_argument('--threshold', type=float, default=0.3)
    args = ap.parse_args()

    pose_path = args.pose or os.path.splitext(args.capture)[0] + '_pose.npz'
    requested = [int(x) for x in args.indices.split(',') if x.strip()]
    with np.load(pose_path) as p:
        total = len(p['keypoints'])
        rows = [min(max(i, 0), total - 1) for i in requested]
        kp = p['keypoints'][rows]
        box = p['bbox'][rows]
        det = p['det_conf'][rows]
        frame_idx = p['frame_idx'][rows] if 'frame_idx' in p.files else np.array(rows)
        t_ns = p['frame_t_ns'][rows] if 'frame_t_ns' in p.files else None

    cols = min(3, len(rows))
    nrows = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(6.2 * cols, 4.1 * nrows),
                             facecolor='#101014', squeeze=False)
    with zipfile.ZipFile(args.capture) as zf:
        for ax, row, fid, points, bbox, confidence, ts in zip(
                axes.flat, rows, frame_idx, kp, box, det,
                t_ns if t_ns is not None else [None] * len(rows)):
            member = f'frames/{int(fid):06d}.jpg'
            image = np.asarray(Image.open(io.BytesIO(zf.read(member))).convert('RGB'))
            ax.imshow(image)
            good = np.isfinite(points[:, 0]) & (points[:, 2] >= args.threshold)
            faint = np.isfinite(points[:, 0]) & ~good
            for (a, b), color in zip(SKELETON, LIMB_COLOR):
                if good[a] and good[b]:
                    ax.plot(points[[a, b], 0], points[[a, b], 1], color=color,
                            linewidth=2.6, solid_capstyle='round')
            ax.scatter(points[good, 0], points[good, 1], s=23, c='white',
                       edgecolors='#101014', linewidths=0.8, zorder=3)
            if faint.any():
                ax.scatter(points[faint, 0], points[faint, 1], s=23,
                           facecolors='none', edgecolors='#ff4040', linewidths=1.3)
            if np.isfinite(bbox).all():
                x1, y1, x2, y2 = bbox
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                           fill=False, color='#00e5ff', linewidth=1.3))
            stamp = f' · t={int(ts) / 1e9:.3f}s' if ts is not None else ''
            ax.set_title(f'frame {int(fid)}{stamp} · det {confidence:.2f} · '
                         f'{int(good.sum())}/17 joints', color='white', fontsize=11)
            ax.set_axis_off()
        for ax in axes.flat[len(rows):]:
            ax.set_axis_off()

    fig.suptitle(f'{os.path.basename(args.capture)} — saved YOLO11x-pose overlays',
                 color='white', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.output, dpi=150, facecolor='#101014')
    print(args.output)


if __name__ == '__main__':
    main()
