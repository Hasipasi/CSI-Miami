#!/usr/bin/env python3
"""Package timestamp-aligned pose NPZs to overlay an existing data archive."""

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import zipfile

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default='data')
    ap.add_argument('--output', default='data/2026_08_25_poses.zip')
    ap.add_argument('--sessions', nargs='+', required=True)
    args = ap.parse_args()

    files = []
    session_summary = {}
    total_frames = total_found = 0
    checksums = {}
    for session in args.sessions:
        paths = sorted(glob.glob(os.path.join(args.data, session, '*_pose.npz')))
        if not paths:
            raise SystemExit(f'no pose files for {session}')
        frames = found = 0
        for path in paths:
            with np.load(path) as p:
                required = {'keypoints', 'found', 'frame_t_ns', 'frame_idx'}
                missing = required - set(p.files)
                if missing:
                    raise ValueError(f'{path}: missing {sorted(missing)}')
                n = len(p['keypoints'])
                if not (len(p['found']) == len(p['frame_t_ns']) ==
                        len(p['frame_idx']) == n):
                    raise ValueError(f'{path}: pose/timestamp lengths disagree')
                frames += n
                found += int(p['found'].sum())
            arcname = os.path.relpath(path, args.data)
            checksums[arcname] = sha256(path)
            files.append((path, arcname))
        session_summary[session] = dict(
            takes=len(paths), frames=frames, poses_found=found,
            detection_percent=100.0 * found / max(frames, 1))
        total_frames += frames
        total_found += found

    manifest = dict(
        format='csi-rig timestamp-aligned COCO-17 poses',
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        extract_into='data/',
        overlays_archive='2026_08_25.zip',
        model='yolo11x-pose.pt',
        imgsz=960,
        takes=len(files),
        frames=total_frames,
        poses_found=total_found,
        detection_percent=100.0 * total_found / max(total_frames, 1),
        arrays=dict(
            keypoints='float32 [frame,17,3]: x,y,confidence; NaN when absent',
            bbox='float32 [frame,4]: x1,y1,x2,y2',
            det_conf='float32 [frame]', second_conf='float32 [frame]',
            n_persons='int16 [frame]', found='bool [frame]',
            frame_t_ns='int64 [frame], relative to recording start',
            frame_idx='int64 [frame]', frame_seq='int64 [frame]'),
        sessions=session_summary,
        sha256=checksums)

    tmp = args.output + '.tmp'
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_STORED,
                         allowZip64=True) as zf:
        zf.writestr('pose_manifest.json', json.dumps(manifest, indent=2) + '\n')
        for path, arcname in files:
            zf.write(path, arcname)
    with zipfile.ZipFile(tmp) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise OSError(f'ZIP CRC validation failed at {bad}')
        if len(zf.namelist()) != len(files) + 1:
            raise OSError('ZIP member count does not match pose file count')
    os.replace(tmp, args.output)
    print(f'{args.output}: {len(files)} poses, {total_frames} frames, '
          f'{100 * total_found / total_frames:.3f}% detected')


if __name__ == '__main__':
    main()
