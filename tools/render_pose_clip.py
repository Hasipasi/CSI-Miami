#!/usr/bin/env python3
"""Render a timestamp-sampled MP4 from embedded frames and stored YOLO poses."""

import argparse
import io
import os
import zipfile

import cv2
import numpy as np
from PIL import Image


SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
            (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
            (8, 10), (0, 1), (0, 2), (1, 3), (2, 4)]
# BGR: legs orange, torso pink, arms blue, head green.
COLORS = [(46, 176, 255)] * 4 + [(192, 98, 240)] * 4 \
         + [(255, 166, 58)] * 4 + [(122, 217, 66)] * 4


def nearest_indices(source_t, targets):
    right = np.clip(np.searchsorted(source_t, targets), 1, len(source_t) - 1)
    left = right - 1
    return np.where(np.abs(source_t[left] - targets) <=
                    np.abs(source_t[right] - targets), left, right)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('capture')
    ap.add_argument('--pose', help='default: <capture>_pose.npz')
    ap.add_argument('-o', '--output', default='pose_clip.mp4')
    ap.add_argument('--seconds', type=float, default=3.0)
    ap.add_argument('--fps', type=float, default=5.0)
    ap.add_argument('--threshold', type=float, default=0.3)
    args = ap.parse_args()

    pose_path = args.pose or os.path.splitext(args.capture)[0] + '_pose.npz'
    with np.load(pose_path) as p:
        t_ns = p['frame_t_ns'].astype(np.int64)
        kp = p['keypoints']
        bbox = p['bbox']
        det = p['det_conf']
        frame_idx = p['frame_idx']
        found = p['found']

    end_ns = int(t_ns[-1])
    start_ns = max(int(t_ns[0]), end_ns - int(round(args.seconds * 1e9)))
    count = max(1, int(round(args.seconds * args.fps)))
    target_ns = start_ns + np.rint(np.arange(count) * 1e9 / args.fps).astype(np.int64)
    rows = nearest_indices(t_ns, target_ns)

    writer = None
    with zipfile.ZipFile(args.capture) as zf:
        for target, row in zip(target_ns, rows):
            fid = int(frame_idx[row])
            raw = zf.read(f'frames/{fid:06d}.jpg')
            rgb = np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'))
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    args.output, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
                if not writer.isOpened():
                    raise OSError(f'could not create {args.output}')

            points = kp[row]
            good = np.isfinite(points[:, 0]) & (points[:, 2] >= args.threshold)
            for (a, b), color in zip(SKELETON, COLORS):
                if good[a] and good[b]:
                    pa = tuple(np.rint(points[a, :2]).astype(int))
                    pb = tuple(np.rint(points[b, :2]).astype(int))
                    cv2.line(frame, pa, pb, color, 4, cv2.LINE_AA)
            for x, y in points[good, :2]:
                cv2.circle(frame, tuple(np.rint([x, y]).astype(int)), 5,
                           (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, tuple(np.rint([x, y]).astype(int)), 5,
                           (16, 16, 20), 1, cv2.LINE_AA)
            if np.isfinite(bbox[row]).all():
                x1, y1, x2, y2 = np.rint(bbox[row]).astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 229, 0), 2)

            rel_s = (int(target) - start_ns) / 1e9
            rec_s = int(t_ns[row]) / 1e9
            status = (f't={rec_s:5.2f}s  clip={rel_s:4.1f}s  frame={fid}  '
                      f'det={float(det[row]):.2f}  joints={int(good.sum())}/17')
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 58), (12, 12, 16), -1)
            cv2.putText(frame, status, (18, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.85, (245, 245, 245), 2, cv2.LINE_AA)
            if not bool(found[row]):
                cv2.putText(frame, 'NO PERSON DETECTED', (18, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (55, 55, 255),
                            2, cv2.LINE_AA)
            writer.write(frame)

    if writer is not None:
        writer.release()
    print(f'{args.output}: {count} frames at {args.fps:g} FPS, '
          f'{args.seconds:g} seconds from recording end')


if __name__ == '__main__':
    main()
