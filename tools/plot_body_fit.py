#!/usr/bin/env python3
"""Contact sheet of a fitted body over its take: the fitted COCO-17 skeleton on the
colour frame, the body vertices on the depth frame, and per-frame residuals.

Drawn from the stored *_body.npz and the capture, never by re-running the fit,
so what is shown is exactly what a dataset built from this take would train on.
The depth panel is the one to look at: if the projected vertices sit on the
person's silhouette in the DEPTH image the metric placement is right; the colour
overlay alone cannot tell a body 30 cm too far from one that fits.

  python3 tools/plot_body_fit.py data/<session>/<take>.npz [-o out.png] [--indices 0,50,100]
"""

import argparse
import json
import os
import sys
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from body_common import (COCO_SKELETON, K_of, SMPL, SMPL_NPZ, depth_to_points,   # noqa: E402
                         frame_members, load_calib, project, read_depth_png, read_jpeg)

LIMB_COLOR = ['#ffb02e'] * 4 + ['#f062c0'] * 4 + ['#3aa6ff'] * 4 + ['#42d97a'] * 4


def plot_take(capture, body_path=None, out=None, indices=None, smpl_path=SMPL_NPZ):
    body_path = body_path or capture[:-4] + '_body.npz'
    out = out or body_path[:-4] + '.png'
    B = np.load(body_path)
    bmeta = json.loads(str(B['meta']))
    F = len(B['pose'])
    rows = indices or [int(x) for x in np.linspace(0, F - 1, 4).round()]
    with np.load(capture) as d, zipfile.ZipFile(capture) as zf:
        meta = json.loads(str(d['meta'])) if 'meta' in d.files else {}
        colour, depth_members, _ = frame_members(d, meta)
        calib = load_calib(meta, None, verbose=False)
        calib['colour_intrinsics'] = bmeta.get('colour_intrinsics', calib['colour_intrinsics'])
        K = K_of(calib['colour_intrinsics'])
        di = calib['depth_intrinsics']
        smpl = SMPL(smpl_path, device='cpu')
        import torch
        with torch.no_grad():
            verts, _ = smpl.forward(torch.as_tensor(B['pose'][rows]), torch.as_tensor(B['betas']),
                                    torch.as_tensor(B['trans'][rows]))
        verts = verts.numpy()
        n = len(rows)
        fig, axes = plt.subplots(n, 2, figsize=(13, 3.9 * n), facecolor='#101014', squeeze=False)
        for r, (i, ax_c, ax_d) in enumerate(zip(rows, axes[:, 0], axes[:, 1])):
            if colour[i] is not None:
                ax_c.imshow(read_jpeg(zf, colour[i]))
            kp = B['keypoints2d'][i]
            for (a, b), col in zip(COCO_SKELETON, LIMB_COLOR):
                ax_c.plot(kp[[a, b], 0], kp[[a, b], 1], color=col, lw=2.4, solid_capstyle='round')
            ax_c.scatter(kp[:, 0], kp[:, 1], s=18, c='white', edgecolors='#101014', lw=0.7, zorder=3)
            ax_c.set_xlim(0, calib['colour_intrinsics'].get('width', 1280))
            ax_c.set_ylim(calib['colour_intrinsics'].get('height', 720), 0)
            ok = 'valid' if B['valid'][i] else ('found' if B['found'][i] else 'NOT found')
            ax_c.set_title(f'frame {int(B["frame_idx"][i])} · {ok} · depth med '
                           f'{1e3 * B["depth_med"][i]:.0f} mm · reproj {B["reproj_px"][i]:.0f} px · '
                           f'z {B["trans"][i, 2]:.2f} m (RGB-only {B["trans_nlf"][i, 2]:.2f})',
                           color='white', fontsize=10)
            ax_c.set_axis_off()
            if depth_members[i] is not None:
                depth = read_depth_png(zf, depth_members[i])
                z = depth.astype(np.float32) * calib['depth_scale_m']
                zc = float(B['trans'][i, 2])
                ax_d.imshow(np.where(z > 0, z, np.nan), cmap='turbo', vmin=zc - 1.5, vmax=zc + 1.5)
                # body vertices into the DEPTH image: colour frame -> depth frame
                ex = calib['depth_to_colour']
                R = np.asarray(ex['rotation'], np.float64).reshape(3, 3)
                t = np.asarray(ex['translation'], np.float64)
                vd = (verts[r] - t) @ R.T
                uv = project(K_of(di), vd)
                ax_d.scatter(uv[::4, 0], uv[::4, 1], s=1.2, c='white', alpha=0.55, lw=0)
                ax_d.set_xlim(0, di.get('width', 640))
                ax_d.set_ylim(di.get('height', 480), 0)
            ax_d.set_title('depth + fitted body vertices', color='white', fontsize=10)
            ax_d.set_axis_off()
        fig.suptitle(f'{os.path.basename(capture)} — depth-grounded SMPL fit '
                     f'(valid {100 * B["valid"].mean():.0f}% of {F} frames, '
                     f'distance x{bmeta.get("distance_scale_median", 1):.3f} vs RGB-only)',
                     color='white', fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(out, dpi=110, facecolor='#101014')
        plt.close(fig)
    return out


def plot_residuals(body_path, out=None):
    """Per-frame residual traces for one take: where the fit is loose."""
    B = np.load(body_path)
    out = out or body_path[:-4] + '_residuals.png'
    t = B['frame_t']
    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True, facecolor='white')
    ax[0].plot(t, 1e3 * B['depth_med'], lw=1.2)
    ax[0].set_ylabel('depth median (mm)')
    ax[1].plot(t, B['reproj_px'], lw=1.2)
    ax[1].set_ylabel('reproj (px)')
    ax[2].plot(t, B['trans'][:, 2], label='fitted z', lw=1.2)
    ax[2].plot(t, B['trans_nlf'][:, 2], label='RGB-only z', lw=1.0, alpha=0.7)
    ax[2].set_ylabel('z (m)')
    ax[2].legend()
    ax[2].set_xlabel('t (s)')
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('capture')
    ap.add_argument('--body', default=None)
    ap.add_argument('-o', '--output', default=None)
    ap.add_argument('--indices', default=None, help='comma-separated frame rows')
    ap.add_argument('--residuals', action='store_true', help='also plot per-frame residuals')
    args = ap.parse_args()
    idx = [int(x) for x in args.indices.split(',')] if args.indices else None
    print(plot_take(args.capture, args.body, args.output, idx))
    if args.residuals:
        print(plot_residuals(args.body or args.capture[:-4] + '_body.npz'))


if __name__ == '__main__':
    main()
