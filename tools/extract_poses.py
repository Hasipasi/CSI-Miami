#!/usr/bin/env python3
"""Extract 2D skeletons from the recorded video frames with YOLO11x-pose.

These become the supervision target for the CSI->pose task, so the priority is
knowing when a skeleton is untrustworthy rather than always producing one. Every
frame therefore carries its detection confidence and a `found` flag, and frames
with no person are written as NaN rather than zeros -- zeros are a valid pixel
coordinate (top-left) and would train the model towards the corner of the image.

One person is expected. When several are detected the highest-confidence box wins,
and the runner-up's score is recorded so an ambiguous frame can be found later.

Writes <session>/<take>_pose.npz beside the source capture.

  .venv_pose/bin/python esp-csi/examples/get-started/tools/extract_poses.py --src data
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

COCO17 = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
          'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
          'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
          'left_knee', 'right_knee', 'left_ankle', 'right_ankle']


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default='data')
    ap.add_argument('--model', default='yolo11x-pose.pt')
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', default='0')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)

    dirs = sorted(glob.glob(f'{args.src}/*/*_frames'))
    if not dirs:
        sys.exit(f'no *_frames directories under {args.src}/')
    print(f'{len(dirs)} takes, model {args.model} @ imgsz {args.imgsz}', flush=True)

    t0 = time.time()
    tot_f = tot_found = 0
    worst = []
    for n, fdir in enumerate(dirs, 1):
        out = fdir[:-len('_frames')] + '_pose.npz'
        if os.path.exists(out) and not args.overwrite:
            continue
        paths = sorted(glob.glob(os.path.join(fdir, '*.jpg')))
        F = len(paths)
        kp = np.full((F, 17, 3), np.nan, dtype=np.float32)
        box = np.full((F, 4), np.nan, dtype=np.float32)
        conf = np.zeros(F, dtype=np.float32)
        second = np.zeros(F, dtype=np.float32)
        npers = np.zeros(F, dtype=np.int16)

        i = 0
        for s in range(0, F, args.batch):
            for r in model(paths[s:s + args.batch], verbose=False,
                           device=args.device, imgsz=args.imgsz):
                b = r.boxes
                if b is not None and len(b):
                    c = b.conf.cpu().numpy()
                    npers[i] = len(c)
                    k = int(np.argmax(c))
                    conf[i] = c[k]
                    if len(c) > 1:
                        second[i] = float(np.sort(c)[-2])
                    box[i] = b.xyxy.cpu().numpy()[k]
                    xy = r.keypoints.xy.cpu().numpy()[k]
                    kc = r.keypoints.conf.cpu().numpy()[k]
                    kp[i, :, :2] = xy
                    kp[i, :, 2] = kc
                i += 1

        found = np.isfinite(kp[:, 0, 0])
        np.savez_compressed(
            out, keypoints=kp, bbox=box, det_conf=conf, second_conf=second,
            n_persons=npers, found=found,
            meta=np.array(json.dumps(dict(
                model=args.model, imgsz=args.imgsz, layout='COCO-17',
                keypoint_names=COCO17, image_size=[1280, 720],
                coords='pixels, [x, y, keypoint_confidence]; NaN where no person',
                selection='highest-confidence box per frame'))))
        tot_f += F
        tot_found += int(found.sum())
        rate = found.mean()
        if rate < 0.98:
            worst.append((os.path.basename(out)[:-9], rate, F))
        if n % 25 == 0 or n == len(dirs):
            el = time.time() - t0
            print(f'  {n}/{len(dirs)} takes  {tot_f} frames  '
                  f'{el:.0f}s  ({tot_f / max(el, 1e-9):.0f} fps)', flush=True)

    print(f'\n{tot_f} frames, person found in {tot_found} '
          f'({100 * tot_found / max(tot_f, 1):.2f}%)')
    if worst:
        print('takes with <98% detection:')
        for name, rate, F in sorted(worst, key=lambda x: x[1])[:20]:
            print(f'  {name:28s} {100 * rate:5.1f}%  ({F} frames)')
    else:
        print('every take had a person in >=98% of frames')


if __name__ == '__main__':
    main()
