#!/usr/bin/env python3
"""Package the recordings + extracted skeletons into a CSI->2D-pose dataset.

Same layout and conventions as the activity dataset, so one adapter pattern covers
both. The extra problem here is pairing: video runs at 30 fps but each CSI link is
only sampled ~5 Hz, so there is no honest way to emit a pose target per video frame
with all 12 links behind it. Instead the pose is resampled onto the *CSI* grid --
the signal decides the rate, not the camera -- and each grid step keeps the time
offset to the video frame it came from so a consumer can reject loose pairings.

Frames with no detected person are NaN in the source and stay NaN here; they are
counted, never filled. A `pose_valid` flag marks steps that are safe to train on.

  python3 build_pose_dataset.py --src data --out dataset/csi5pose_v1
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

COCO17 = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
          'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
          'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
          'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
SKELETON = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
            [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
            [1, 3], [2, 4], [3, 5], [4, 6]]


def links_of(d):
    return sorted({k.rsplit('|', 1)[0] for k in d.files if k.endswith('|t')})


def link_name(lk, lab):
    tx, rx = lk.split('|')
    return f'{lab.get(tx, tx[-5:])}->{lab.get(rx, rx[-5:])}'


def resample_csi(d, lk, grid, half):
    t = d[f'{lk}|t'].astype(np.float64)
    a = d[f'{lk}|a'].astype(np.float32)
    j = np.clip(np.searchsorted(t, grid), 1, max(len(t) - 1, 1))
    lo, hi = np.abs(grid - t[j - 1]), np.abs(grid - t[np.minimum(j, len(t) - 1)])
    pick = np.where(lo <= hi, j - 1, np.minimum(j, len(t) - 1))
    return a[pick], (np.minimum(lo, hi) <= half)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default='data')
    ap.add_argument('--sessions', default='*',
                    help='comma-separated globs over session directory names. '
                         'Round-robin and pinned-TX captures have different link '
                         'counts and cannot share a dataset, so they are selected '
                         'separately (e.g. "*_train,*_test" vs "*_txB").')
    ap.add_argument('--out', default='dataset/csi5pose_v1')
    ap.add_argument('--rate', type=float, default=5.0)
    ap.add_argument('--subcarriers', default=None,
                    help='comma-separated subcarrier indices to keep (of the 192 the '
                         'radio reports). Measured default keeps 30 evenly spaced over '
                         'the live band, which reconstructs all 166 live subcarriers at '
                         'R^2 = 0.973 -- the frequency axis carries ~2.6 effective '
                         'dimensions, so 30 is far above what the channel supports. '
                         'Evenly spaced beats picking the highest-variance ones (0.831), '
                         'which cluster and re-measure the same thing.')
    ap.add_argument('--duration', type=float, default=5.0)
    ap.add_argument('--context', type=int, default=20,
                    help='CSI steps of context each side of a pose target. The sample '
                         'is 2*context+1 steps wide (41 by default); a loader may use '
                         'fewer by trimming the centre, never more.')
    ap.add_argument('--kp-conf', type=float, default=0.3,
                    help='keypoint confidence below which a joint is marked invisible')
    args = ap.parse_args()

    import fnmatch
    pats = [x.strip() for x in args.sessions.split(',') if x.strip()]
    sessions = sorted(n for n in (os.path.basename(p.rstrip('/'))
                                  for p in glob.glob(f'{args.src}/*/'))
                      if not n.startswith('_')            # archives, scratch
                      and any(fnmatch.fnmatch(n, q) for q in pats))
    KEEP = (np.array([int(x) for x in args.subcarriers.split(',')])
            if args.subcarriers else None)

    T = int(round(args.duration * args.rate))
    grid = (np.arange(T) + 0.5) / args.rate
    half = 0.5 / args.rate
    clips_dir = os.path.join(args.out, 'clips')
    os.makedirs(clips_dir, exist_ok=True)

    rows, link_order, n_sub = [], None, None
    split_of = {}
    cov, pose_ok, gaps = [], [], []
    for sess in sessions:
        parts = sess.split('_')
        subject = parts[0]
        split = 'train' if 'train' in parts else ('test' if 'test' in parts else parts[-1])
        split_of[sess] = split
        for path in sorted(glob.glob(f'{args.src}/{sess}/*.npz')):
            if path.endswith('_pose.npz'):
                continue
            take = os.path.basename(path)[:-4]
            ppath = f'{args.src}/{sess}/{take}_pose.npz'
            if not os.path.exists(ppath):
                sys.exit(f'missing skeletons: {ppath}. Run extract_poses.py first.')

            d = np.load(path)
            P = np.load(ppath)
            meta = json.loads(str(d['meta']))
            lab = meta['boards']

            lks = links_of(d)
            names = [link_name(lk, lab) for lk in lks]
            o = np.argsort(names)
            lks, names = [lks[i] for i in o], [names[i] for i in o]
            if link_order is None:
                link_order = names
            elif names != link_order:
                sys.exit(f'{path}: link set differs from {link_order}')

            n_all = d[f'{lks[0]}|a'].shape[1]
            X = np.zeros((T, len(lks), n_all), dtype=np.float32)
            M = np.zeros((T, len(lks)), dtype=bool)
            for i, lk in enumerate(lks):
                X[:, i, :], M[:, i] = resample_csi(d, lk, grid, half)
            if KEEP is not None:
                X = X[:, :, KEEP]
            n_sub = X.shape[2] if n_sub is None else n_sub

            # pose onto the CSI grid: nearest video frame, keeping how far it was
            ft = d['frame_t'].astype(np.float64)
            kp = P['keypoints'].astype(np.float32)          # [F, 17, 3] NaN if absent
            F = min(len(ft), len(kp))
            ft, kp = ft[:F], kp[:F]
            j = np.clip(np.searchsorted(ft, grid), 1, max(F - 1, 1))
            lo, hi = np.abs(grid - ft[j - 1]), np.abs(grid - ft[np.minimum(j, F - 1)])
            src = np.where(lo <= hi, j - 1, np.minimum(j, F - 1))
            dt = np.minimum(lo, hi).astype(np.float32)
            pose = kp[src]                                   # [T, 17, 3]
            # A step is trainable only if a person was actually found in that frame.
            valid = np.isfinite(pose[:, 0, 0])
            vis = (pose[:, :, 2] >= args.kp_conf) & np.isfinite(pose[:, :, 0])

            activity = take.rstrip('0123456789')
            rnd = int(take[len(activity):] or 0)
            clip_id = f'{subject}_{split}_{take}'
            np.savez_compressed(
                os.path.join(clips_dir, f'{clip_id}.npz'),
                csi=X, mask=M, t=grid.astype(np.float32), links=np.array(link_order),
                pose=pose, pose_valid=valid, kp_visible=vis, pose_dt=dt,
                pose_frame=src.astype(np.int32),
                pose_full=kp, frame_t=ft.astype(np.float32),
                det_conf=P['det_conf'].astype(np.float32)[:F],
                n_persons=P['n_persons'][:F])
            cov.append(float(M.mean()))
            pose_ok.append(float(valid.mean()))
            gaps.append(float(dt.max()))
            rows.append(dict(clip_id=clip_id, subject=subject, session=sess, split=split,
                             activity=activity, round=rnd, n_frames=int(F),
                             pose_valid_frac=round(float(valid.mean()), 4),
                             mean_det_conf=round(float(np.nanmean(P['det_conf'][:F])), 4),
                             mask_coverage=round(float(M.mean()), 4),
                             max_pose_dt_ms=round(float(dt.max()) * 1e3, 1),
                             frames_dir=os.path.relpath(
                                 os.path.join(args.src, sess, f'{take}_frames'), args.out)))

    acts = sorted({r['activity'] for r in rows})
    label_map = {a: i for i, a in enumerate(acts)}
    for r in rows:
        r['label'] = label_map[r['activity']]

    # The training unit is a pose target plus its CSI context, not a whole clip. The
    # index below names every valid target; windows are sliced from the stored
    # sequence at load time rather than materialised here, because writing 41 steps
    # per target would duplicate the CSI ~41x for no information gained.
    samples = []
    for r in rows:
        z = np.load(os.path.join(clips_dir, f'{r["clip_id"]}.npz'))
        for t in np.where(z['pose_valid'])[0]:
            lo, hi = int(t) - args.context, int(t) + args.context
            samples.append((r['clip_id'], int(t), r['label'], r['subject'], r['split'],
                            r['session'], int(max(0, -lo) + max(0, hi - (T - 1)))))
    with open(os.path.join(args.out, 'samples.csv'), 'w') as fh:
        fh.write('clip_id,t,label,subject,split,session,pad_steps\n')
        for row in samples:
            fh.write(','.join(str(x) for x in row) + '\n')
    npad = sum(1 for x in samples if x[6])

    tr = [r for r in rows if r['split'] == 'train']
    acc = np.zeros((len(link_order), n_sub)); acc2 = np.zeros_like(acc)
    pk = []
    for r in tr:
        z = np.load(os.path.join(clips_dir, f'{r["clip_id"]}.npz'))
        x = z['csi'].astype(np.float64)
        acc += x.sum(0); acc2 += (x ** 2).sum(0)
        p = z['pose'][z['pose_valid']]
        if len(p):
            pk.append(p[:, :, :2].reshape(-1, 2))
    n = len(tr) * T
    mean = acc / n
    std = np.sqrt(np.maximum(acc2 / n - mean ** 2, 1e-12))
    pk = np.concatenate(pk) if pk else np.zeros((1, 2))
    np.savez_compressed(os.path.join(args.out, 'stats.npz'),
                        csi_mean=mean.astype(np.float32), csi_std=std.astype(np.float32),
                        pose_mean=pk.mean(0).astype(np.float32),
                        pose_std=pk.std(0).astype(np.float32), split='train', n=n)
    valid_sub = ~((std < 1.0).all(axis=0))

    subjects = sorted({r['subject'] for r in rows})
    manifest = dict(
        name='csi5pose', version=1,
        description='WiFi-CSI -> 2D human pose, 4x ESP32-S3 round-robin, 12 links.',
        n_clips=len(rows), subjects=subjects, sessions=sessions,
        activities=acts, label_map=label_map,
        links=link_order, n_links=len(link_order), n_subcarriers=n_sub,
        clip_seconds=args.duration, grid_hz=args.rate, n_timesteps=T,
        csi_layout='[T, L, S] float32 amplitude (NO phase)',
        mask_layout='[T, L] bool, True where a real packet fell within half a grid cell',
        pose_layout='[T, 17, 3] float32 = x, y, keypoint_confidence; pixels in a 1280x720 image',
        context_half=args.context, context_total=2 * args.context + 1,
        context_pad='edge-replicate; samples.csv gives pad_steps per target',
        n_samples=len(samples), n_samples_padded=npad,
        sample_index='samples.csv -- one row per (clip, centre step); slice '
                     'csi[t-context : t+context+1] from the clip file',
        pose_format='COCO-17', keypoint_names=COCO17, skeleton=SKELETON,
        image_size=[1280, 720], pose_model='yolo11x-pose (ultralytics), imgsz 960',
        pose_is_2d=True,
        mean_mask_coverage=round(float(np.mean(cov)), 4),
        mean_pose_valid=round(float(np.mean(pose_ok)), 4),
        max_pose_pairing_gap_ms=round(float(np.max(gaps)) * 1e3, 1),
        valid_subcarriers=[int(i) for i in np.where(valid_sub)[0]],
        n_valid_subcarriers=int(valid_sub.sum()),
        protocols={
            'within_subject': {'train': [s for s in sessions if split_of.get(s) == 'train'],
                               'test': [s for s in sessions if split_of.get(s) == 'test']},
            **{f'cross_subject_{a}2{b}': {'train': [s for s in sessions if s.startswith(a)],
                                          'test': [s for s in sessions if s.startswith(b)]}
               for a in subjects for b in subjects if a != b}},
        clips=rows)
    with open(os.path.join(args.out, 'manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=2)

    cols = ['clip_id', 'subject', 'session', 'split', 'activity', 'label', 'round',
            'n_frames', 'pose_valid_frac', 'mean_det_conf', 'mask_coverage',
            'max_pose_dt_ms', 'frames_dir']
    with open(os.path.join(args.out, 'clips.csv'), 'w') as fh:
        fh.write(','.join(cols) + '\n')
        for r in rows:
            fh.write(','.join(str(r[c]) for c in cols) + '\n')

    print(f'wrote {args.out}/')
    print(f'  {len(rows)} clips · csi [T={T}, L={len(link_order)}, S={n_sub}] @ {args.rate:g} Hz')
    print(f'  pose [T={T}, 17, 3] COCO-17 pixels · valid steps {100*np.mean(pose_ok):.2f}%')
    print(f'  mask coverage {100*np.mean(cov):.1f}% · worst pose pairing gap '
          f'{1e3*np.max(gaps):.0f} ms')
    print(f'  {int(valid_sub.sum())}/{n_sub} subcarriers carry signal')
    print(f'  {len(samples)} pose targets, each with +/-{args.context} CSI steps '
          f'({2 * args.context + 1} wide = {(2 * args.context + 1) / args.rate:.2f} s)')
    print(f'  {npad} targets ({100 * npad / max(len(samples), 1):.0f}%) sit near a clip '
          f'edge and need padding')


if __name__ == '__main__':
    main()
