#!/usr/bin/env python3
"""Extract 2D skeletons from the recorded video frames with YOLO11x-pose.

These become the supervision target for the CSI->pose task, so the priority is
knowing when a skeleton is untrustworthy rather than always producing one. Every
frame therefore carries its detection confidence and a `found` flag, and frames
with no person are written as NaN rather than zeros -- zeros are a valid pixel
coordinate (top-left) and would train the model towards the corner of the image.

One person is expected. When several are detected the highest-confidence box wins,
and the runner-up's score is recorded so an ambiguous frame can be found later.

Writes <session>/<take>_pose.npz beside the source capture. The source camera
timestamps and frame indices are copied into the pose archive, so every skeleton
can be joined to CSI by the same integer nanosecond recording clock.

  .venv_pose/bin/python esp-csi/examples/get-started/tools/extract_poses.py --src data
"""

import argparse
import glob
import json
import os
import sys
import tempfile
import time
import zipfile

import numpy as np

COCO17 = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
          'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
          'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
          'left_knee', 'right_knee', 'left_ankle', 'right_ankle']


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', nargs='+', default=['data'],
                    help='capture file, session directory, or parent of sessions')
    ap.add_argument('--model', default='yolo11x-pose.pt')
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', default='0')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)

    dirs = []
    archives = []
    for src in args.src:
        if os.path.isfile(src):
            if src.endswith('.npz') and not src.endswith('_pose.npz'):
                archives.append(src)
            continue
        # Accept either one session directory or a parent containing sessions.
        dirs.extend(glob.glob(os.path.join(src, '*_frames')))
        dirs.extend(glob.glob(os.path.join(src, '*', '*_frames')))
        archives.extend(glob.glob(os.path.join(src, '*.npz')))
        archives.extend(glob.glob(os.path.join(src, '*', '*.npz')))
    dirs = sorted(set(dirs))
    archives = sorted(set(archives))
    sources = [('dir', d, d[:-len('_frames')]) for d in dirs]
    legacy_bases = {base for _kind, _src, base in sources}
    for archive in archives:
        if archive.endswith('_pose.npz') or archive[:-4] in legacy_bases:
            continue
        try:
            with zipfile.ZipFile(archive) as zf:
                has_frames = any(n.startswith('frames/') and n.endswith('.jpg')
                                 for n in zf.namelist())
        except zipfile.BadZipFile:
            has_frames = False
        if has_frames:
            sources.append(('npz', archive, archive[:-4]))
    if not sources:
        sys.exit(f'no captures with video frames under {args.src}/')
    print(f'{len(sources)} takes, model {args.model} @ imgsz {args.imgsz}', flush=True)

    t0 = time.time()
    tot_f = tot_found = 0
    worst = []
    for n, (kind, src, base) in enumerate(sources, 1):
        out = base + '_pose.npz'
        if os.path.exists(out) and not args.overwrite:
            continue
        tmp = None
        if kind == 'npz':
            tmp = tempfile.TemporaryDirectory(prefix='csi_frames_')
            with zipfile.ZipFile(src) as zf:
                for member in zf.namelist():
                    if member.startswith('frames/') and member.endswith('.jpg'):
                        zf.extract(member, tmp.name)
            fdir = os.path.join(tmp.name, 'frames')
        else:
            fdir = src
        paths = sorted(glob.glob(os.path.join(fdir, '*.jpg')))
        F = len(paths)
        capture = src if kind == 'npz' else base + '.npz'
        with np.load(capture) as d:
            source_meta = json.loads(str(d['meta'])) if 'meta' in d.files else {}
            source_idx = (d['frame_idx'].astype(np.int64)
                          if 'frame_idx' in d.files else np.arange(len(d['frame_t'])))
            image_idx = np.array([int(os.path.splitext(os.path.basename(p))[0])
                                  for p in paths], dtype=np.int64)
            lookup = {int(idx): pos for pos, idx in enumerate(source_idx)}
            try:
                take = np.array([lookup[int(idx)] for idx in image_idx], dtype=np.int64)
            except KeyError as e:
                raise ValueError(f'{capture}: image frame {e.args[0]} has no timestamp')
            timing = {}
            for key in ('frame_t', 'frame_t_ns', 'frame_epoch', 'frame_seq'):
                if key in d.files:
                    timing[key] = d[key][take]
            timing['frame_idx'] = image_idx
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
        pose_meta = dict(
                model=args.model, imgsz=args.imgsz, layout='COCO-17',
                keypoint_names=COCO17, image_size=[1280, 720],
                coords='pixels, [x, y, keypoint_confidence]; NaN where no person',
                selection='highest-confidence box per frame',
                source_capture=os.path.basename(capture),
                timestamp_origin=source_meta.get('timestamp_origin', 'record_start'),
                timestamp_unit=source_meta.get('timestamp_unit', 'nanoseconds'))
        tmp_out = out + '.tmp'
        with open(tmp_out, 'wb') as fh:
            np.savez_compressed(
                fh, keypoints=kp, bbox=box, det_conf=conf, second_conf=second,
                n_persons=npers, found=found, meta=np.array(json.dumps(pose_meta)),
                **timing)
        os.replace(tmp_out, out)
        if tmp is not None:
            tmp.cleanup()
        tot_f += F
        tot_found += int(found.sum())
        rate = found.mean()
        if rate < 0.98:
            worst.append((os.path.basename(out)[:-9], rate, F))
        el = time.time() - t0
        done_rate = tot_f / max(el, 1e-9)
        eta = ((len(sources) - n) * (el / n)) if n else 0.0
        print(f'  {n}/{len(sources)} takes  {tot_f} frames  '
              f'{el:.0f}s  ({done_rate:.1f} fps, ETA {eta / 60:.1f} min)', flush=True)

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
