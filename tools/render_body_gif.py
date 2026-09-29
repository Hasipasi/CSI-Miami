#!/usr/bin/env python3
"""Animate a fitted take: the colour frame with the fitted skeleton, and the metric
3-D keypoints in the room frame, seen from the front, the side and above.

The three orthographic views are what a CSI model is asked to predict, so this is
the honest picture of the target. Coordinates are the room frame from
body_common (origin on the floor at the centre of the board square, 2.12 m in
front of the camera), and the rig is drawn into every view: the four antennas on
their rods at 1.20 m, the camera 20 cm below the one it shares with board A. The
front view should look like the camera picture, the side view shows the depth
ordering an RGB network could only guess, and the top view shows where in the
array the person actually stands -- which is the geometry the CSI sees.

  python3 tools/render_body_gif.py data/<session>/<take>.npz [-o take.gif] [--step 2]
  ... [--pitch 3]      # if the fitted feet do not sit on the floor line

The camera's tilt is not measured anywhere, so --pitch (degrees, positive = nose
down) is the one knob that moves the whole skeleton against the rig.
"""

import argparse
import io
import json
import os
import sys
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from PIL import Image                    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from body_common import (BOARD_H, BOARDS, CAM_H, CAMERA_XZ, COCO_SKELETON,  # noqa: E402
                         HALF_DIAG, frame_members, read_jpeg, to_room)

LIMB_COLOR = ['#ffb02e'] * 4 + ['#f062c0'] * 4 + ['#3aa6ff'] * 4 + ['#42d97a'] * 4
BG = '#101014'
ROD, NODE, CAM, FLOOR = '#3f4450', '#7dd3fc', '#e8710a', '#4a4a57'


def draw_skeleton(ax, pts, lw=2.2, ms=14):
    """pts [17, 2]; NaN joints are skipped."""
    for (a, b), col in zip(COCO_SKELETON, LIMB_COLOR):
        if np.isfinite(pts[[a, b]]).all():
            ax.plot(pts[[a, b], 0], pts[[a, b], 1], color=col, lw=lw, solid_capstyle='round')
    ok = np.isfinite(pts[:, 0])
    ax.scatter(pts[ok, 0], pts[ok, 1], s=ms, c='white', edgecolors=BG, lw=0.6, zorder=3)


def draw_rig(ax, view):
    """The rods, the antennas and the camera, behind the skeleton (zorder 0)."""
    if view == 'top':
        ring = [BOARDS[k] for k in 'ABCD']
        xs, zs = zip(*(ring + ring[:1]))
        ax.plot(xs, zs, color=ROD, lw=1.4, ls='--', zorder=0)
        for k, (x, z) in BOARDS.items():
            ax.scatter([x], [z], s=70, c=NODE, edgecolors=BG, lw=0.8, zorder=2)
            ax.annotate(k, (x, z), textcoords='offset points', xytext=(7, 5),
                        color=NODE, fontsize=9, fontweight='bold')
        cx, cz = CAMERA_XZ
        # the camera looks along the A->C diagonal, i.e. up the page
        ax.plot([cx - 0.22, cx, cx + 0.22], [cz - 0.02, cz + 0.3, cz - 0.02],
                color=CAM, lw=1.6, zorder=2)
        ax.scatter([0], [0], marker='+', s=90, c=FLOOR, lw=1.2, zorder=1)
        return
    # front looks along Z (rods stand at their x), side looks along X (at their z)
    axis = 0 if view == 'front' else 1
    lo, hi = ax.get_xlim()
    ax.axhline(0.0, color=FLOOR, lw=1.2, zorder=0)
    for k, xz in BOARDS.items():
        u = xz[axis]
        if not lo < u < hi:
            continue
        ax.plot([u, u], [0, BOARD_H], color=ROD, lw=2.0, zorder=0)
        ax.scatter([u], [BOARD_H], s=60, c=NODE, edgecolors=BG, lw=0.8, zorder=1)
        ax.annotate(k, (u, BOARD_H), textcoords='offset points', xytext=(6, 2),
                    color=NODE, fontsize=9, fontweight='bold')
    u = CAMERA_XZ[axis]
    if lo < u < hi:
        ax.scatter([u], [CAM_H], s=55, marker='s', c=CAM, edgecolors=BG, lw=0.8, zorder=1)


def render(capture, body_path=None, out=None, step=2, fps=15, max_frames=None,
           pitch=0.0, cam_height=CAM_H, centre_dist=HALF_DIAG):
    body_path = body_path or capture[:-4] + '_body.npz'
    out = out or capture[:-4] + '_body.gif'
    B = np.load(body_path)
    K3 = to_room(B['keypoints3d'], pitch, cam_height, centre_dist)
    K2 = B['keypoints2d']
    valid = B['valid'].astype(bool)
    F = len(K3)
    rows = list(range(0, F, step))
    if max_frames:
        rows = rows[:max_frames]
    # fixed axes over the whole take so the person does not swim as the limits move
    P = K3[valid].reshape(-1, 3)
    cx, cz = np.median(P[:, 0]), np.median(P[:, 2])
    r = 1.15                                  # half-width of the front/side views, m
    y_hi = max(np.nanmax(P[:, 1]) + 0.2, BOARD_H + 0.35)
    arena = HALF_DIAG + 0.5                   # the top view holds the whole square

    with np.load(capture) as d, zipfile.ZipFile(capture) as zf:
        meta = json.loads(str(d['meta'])) if 'meta' in d.files else {}
        colour, _, _ = frame_members(d, meta)
        frames = []
        fig = plt.figure(figsize=(14, 4.2), facecolor=BG)
        gs = fig.add_gridspec(1, 4, width_ratios=[1.78, 1, 1, 1], wspace=0.08,
                              left=0.01, right=0.99, top=0.88, bottom=0.05)
        axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
        for n, i in enumerate(rows):
            for ax in axes:
                ax.cla()
                ax.set_facecolor(BG)
            ax_c, ax_f, ax_s, ax_t = axes
            if colour[i] is not None:
                ax_c.imshow(read_jpeg(zf, colour[i]))
            draw_skeleton(ax_c, K2[i], lw=2.4, ms=16)
            ax_c.set_xlim(0, 1280)
            ax_c.set_ylim(720, 0)
            ax_c.set_title(f'camera · frame {int(B["frame_idx"][i])} · t = {B["frame_t"][i]:.2f} s'
                           + ('' if valid[i] else ' · NOT VALID'),
                           color='white', fontsize=10)
            p = K3[i] if valid[i] else np.full((17, 3), np.nan)
            # front: X right, Y up, floor at 0 -- the camera's own view of the arena
            ax_f.set_xlim(cx - r, cx + r)
            ax_f.set_ylim(-0.1, y_hi)
            draw_rig(ax_f, 'front')
            draw_skeleton(ax_f, p[:, [0, 1]])
            ax_f.set_title('front (X, Y)', color='white', fontsize=10)
            # side: Z away from the camera, Y up; the camera stands at Z = -2.12
            ax_s.set_xlim(cz - r, cz + r)
            ax_s.set_ylim(-0.1, y_hi)
            draw_rig(ax_s, 'side')
            draw_skeleton(ax_s, p[:, [2, 1]])
            ax_s.set_title('side (Z, Y)', color='white', fontsize=10)
            # top: the whole 3 m square, camera at the near corner looking up the page
            ax_t.set_xlim(-arena, arena)
            ax_t.set_ylim(-arena, arena)
            draw_rig(ax_t, 'top')
            draw_skeleton(ax_t, p[:, [0, 2]], lw=1.6, ms=8)
            ax_t.set_title('top (X, Z) · whole array', color='white', fontsize=10)
            for ax in (ax_f, ax_s, ax_t):
                ax.set_aspect('equal')
                ax.tick_params(colors='#9a9aa5', labelsize=7)
                for sp in ax.spines.values():
                    sp.set_color('#3a3a44')
                ax.grid(color='#2a2a33', lw=0.6)
                ax.set_xlabel('m from the arena centre', color='#9a9aa5', fontsize=8)
            ax_c.set_axis_off()
            foot = np.nanmin(K3[i, :, 1]) if valid[i] else np.nan
            fig.suptitle(f'{os.path.basename(capture)} — depth-grounded SMPL fit, COCO-17 in '
                         f'metres, room frame (origin on the floor at the array centre; '
                         f'lowest joint {foot:+.2f} m)',
                         color='white', fontsize=11, fontweight='bold')
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=72, facecolor=BG)
            buf.seek(0)
            frames.append(Image.open(buf).convert('P', palette=Image.ADAPTIVE, colors=128))
            if n % 20 == 0:
                print(f'  {n + 1}/{len(rows)} frames', flush=True)
        plt.close(fig)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps),
                   loop=0, optimize=False)
    print(f'{out}: {len(frames)} frames at {fps} fps, {os.path.getsize(out) / 1e6:.1f} MB')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('capture')
    ap.add_argument('--body', default=None)
    ap.add_argument('-o', '--output', default=None)
    ap.add_argument('--step', type=int, default=2, help='use every Nth frame')
    ap.add_argument('--fps', type=float, default=15.0)
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--pitch', type=float, default=0.0,
                    help='camera tilt, degrees nose down (default: %(default)s)')
    ap.add_argument('--cam-height', type=float, default=CAM_H)
    ap.add_argument('--centre-dist', type=float, default=HALF_DIAG,
                    help='camera to arena centre, m (default: %(default).2f)')
    args = ap.parse_args()
    render(args.capture, args.body, args.output, args.step, args.fps, args.max_frames,
           args.pitch, args.cam_height, args.centre_dist)


if __name__ == '__main__':
    main()
