#!/usr/bin/env python3
"""Package the recorded sessions into a self-contained activity-recognition dataset.

The raw captures are irregularly sampled: round-robin gives each of the 12 links a
packet only while its transmitter holds the token, so a link lands ~5 Hz with gaps.
Models want a dense [T, L, S] tensor. Resampling to a uniform grid is therefore
unavoidable, and the honest way to do it is nearest-neighbour plus an explicit
validity mask: nearest reuses a value the radio actually measured (linear
interpolation would invent one), and the mask says which cells are real rather than
held, so a consumer can weight or drop them instead of silently trusting all of it.

Video frames are referenced, not copied -- they are ~1.8 GB and only needed once
skeletons are extracted for the pose task.

  python3 build_activity_dataset.py --src data --out dataset/csi5act_v1
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict

import numpy as np


def links_of(d):
    return sorted({k.rsplit('|', 1)[0] for k in d.files if k.endswith('|t')})


def link_name(lk, lab):
    tx, rx = lk.split('|')
    return f'{lab.get(tx, tx[-5:])}->{lab.get(rx, rx[-5:])}'


def resample(d, lk, grid, half):
    """Nearest real packet per grid point, plus whether it was actually near.

    Returns (amps [T, S], valid [T]). A grid point with no packet within `half`
    still gets its nearest value -- dropping to zero would look like a dead link,
    which is a different and much louder claim than "held from just before".
    """
    t = d[f'{lk}|t'].astype(np.float64)
    a = d[f'{lk}|a'].astype(np.float32)
    j = np.clip(np.searchsorted(t, grid), 1, max(len(t) - 1, 1))
    lo = np.abs(grid - t[j - 1])
    hi = np.abs(grid - t[np.minimum(j, len(t) - 1)])
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
    ap.add_argument('--out', default='dataset/csi5act_v1')
    ap.add_argument('--rate', type=float, default=5.0,
                    help='resample grid rate (Hz). 5 Hz matches the 200 ms '
                         'round-robin cycle; higher rates mostly duplicate values.')
    ap.add_argument('--subcarriers', default=None,
                    help='comma-separated subcarrier indices to keep (of the 192 the '
                         'radio reports). Measured default keeps 30 evenly spaced over '
                         'the live band, which reconstructs all 166 live subcarriers at '
                         'R^2 = 0.973 -- the frequency axis carries ~2.6 effective '
                         'dimensions, so 30 is far above what the channel supports. '
                         'Evenly spaced beats picking the highest-variance ones (0.831), '
                         'which cluster and re-measure the same thing.')
    ap.add_argument('--duration', type=float, default=5.0, help='clip length (s)')
    args = ap.parse_args()

    import fnmatch
    pats = [x.strip() for x in args.sessions.split(',') if x.strip()]
    sessions = sorted(n for n in (os.path.basename(p.rstrip('/'))
                                  for p in glob.glob(f'{args.src}/*/'))
                      if not n.startswith('_')            # archives, scratch
                      and any(fnmatch.fnmatch(n, q) for q in pats))
    if not sessions:
        sys.exit(f'no sessions under {args.src}/')

    KEEP = (np.array([int(x) for x in args.subcarriers.split(',')])
            if args.subcarriers else None)

    T = int(round(args.duration * args.rate))
    grid = (np.arange(T) + 0.5) / args.rate          # cell centres
    half = 0.5 / args.rate

    clips_dir = os.path.join(args.out, 'clips')
    os.makedirs(clips_dir, exist_ok=True)

    rows, link_order, n_sub = [], None, None
    split_of = {}
    cover = []
    for sess in sessions:
        parts = sess.split('_')
        subject = parts[0]
        split = 'train' if 'train' in parts else ('test' if 'test' in parts else parts[-1])
        split_of[sess] = split
        for path in sorted(glob.glob(f'{args.src}/{sess}/*.npz')):
            if path.endswith('_pose.npz'):
                continue            # skeletons live beside the captures, not in them
            d = np.load(path)
            meta = json.loads(str(d['meta']))
            lab = meta['boards']
            lks = links_of(d)
            names = [link_name(lk, lab) for lk in lks]
            order = np.argsort(names)
            lks = [lks[i] for i in order]
            names = [names[i] for i in order]

            if link_order is None:
                link_order = names
            elif names != link_order:
                sys.exit(f'{path}: link set {names} != {link_order}. '
                         'A dataset with varying link sets cannot be batched.')

            n_all = d[f'{lks[0]}|a'].shape[1]
            X = np.zeros((T, len(lks), n_all), dtype=np.float32)
            M = np.zeros((T, len(lks)), dtype=bool)
            for i, lk in enumerate(lks):
                X[:, i, :], M[:, i] = resample(d, lk, grid, half)
            if KEEP is not None:
                X = X[:, :, KEEP]
            if n_sub is None:
                n_sub = X.shape[2]
            elif X.shape[2] != n_sub:
                sys.exit(f'{path}: {X.shape[2]} subcarriers != {n_sub}')

            # Ship the raw irregular packets too: the grid above is one defensible
            # choice, not the only one, and without this a consumer who wants a
            # different rate or an event-based model has to come back to the source.
            rt, rl, ra, rs = [], [], [], []
            for i, lk in enumerate(lks):
                tt = d[f'{lk}|t'].astype(np.float32)
                rt.append(tt)
                rl.append(np.full(len(tt), i, dtype=np.int16))
                ra.append(d[f'{lk}|a'].astype(np.float32))
                rs.append(d[f'{lk}|lts'].astype(np.float64))
            rt = np.concatenate(rt); rl = np.concatenate(rl)
            ra = np.concatenate(ra); rs = np.concatenate(rs)
            srt = np.argsort(rt, kind='stable')

            take = os.path.basename(path)[:-4]
            activity = take.rstrip('0123456789')
            rnd = int(take[len(activity):] or 0)
            clip_id = f'{subject}_{split}_{take}'
            np.savez_compressed(
                os.path.join(clips_dir, f'{clip_id}.npz'),
                csi=X, mask=M, t=grid.astype(np.float32),
                links=np.array(link_order), frame_t=d['frame_t'].astype(np.float32),
                frame_idx=d['frame_idx'],
                raw_t=rt[srt], raw_link=rl[srt], raw_amp=ra[srt], raw_board_t=rs[srt])
            cover.append(float(M.mean()))
            rows.append(dict(clip_id=clip_id, subject=subject, session=sess,
                             split=split, activity=activity, round=rnd,
                             n_frames=int(len(d['frame_t'])),
                             n_packets=int(sum(len(d[f'{lk}|t']) for lk in lks)),
                             mask_coverage=round(float(M.mean()), 4),
                             frames_dir=os.path.relpath(
                                 os.path.join(args.src, sess, f'{take}_frames'), args.out)))

    acts = sorted({r['activity'] for r in rows})
    label_map = {a: i for i, a in enumerate(acts)}
    for r in rows:
        r['label'] = label_map[r['activity']]

    # Normalisation stats from the train split only. Computed here for convenience,
    # but leaking across a cross-subject protocol is a real risk, so this is scoped
    # and the guide says to recompute when the protocol changes.
    tr = [r for r in rows if r['split'] == 'train']
    acc = np.zeros((len(link_order), n_sub), dtype=np.float64)
    acc2 = np.zeros_like(acc)
    for r in tr:
        x = np.load(os.path.join(clips_dir, f'{r["clip_id"]}.npz'))['csi'].astype(np.float64)
        acc += x.sum(axis=0)
        acc2 += (x ** 2).sum(axis=0)
    n = len(tr) * T
    mean = acc / n
    std = np.sqrt(np.maximum(acc2 / n - mean ** 2, 1e-12))
    np.savez_compressed(os.path.join(args.out, 'stats.npz'), mean=mean.astype(np.float32),
                        std=std.astype(np.float32), split='train', n=n)

    # Guard bands and DC carry no signal; flagged, not removed, so the consumer decides.
    dead = (std < 1.0)
    valid_sub = ~dead.all(axis=0)

    subjects = sorted({r['subject'] for r in rows})
    manifest = dict(
        name='csi5act', version=1,
        description='WiFi-CSI activity recognition, 4x ESP32-S3 round-robin, 12 links.',
        n_clips=len(rows), n_classes=len(acts), classes=acts, label_map=label_map,
        subjects=subjects, sessions=sessions,
        links=link_order, n_links=len(link_order), n_subcarriers=n_sub,
        clip_seconds=args.duration, grid_hz=args.rate, n_timesteps=T,
        csi_layout='[T, L, S] float32 amplitude (NO phase)',
        mask_layout='[T, L] bool, True where a real packet fell within half a grid cell',
        raw_layout=('raw_t [N] s, raw_link [N] index into links, raw_amp [N, S], '
                    'raw_board_t [N] board hardware clock (s) -- the lossless '
                    'irregular packets, sorted by time'),
        mean_mask_coverage=round(float(np.mean(cover)), 4),
        valid_subcarriers=[int(i) for i in np.where(valid_sub)[0]],
        n_valid_subcarriers=int(valid_sub.sum()),
        protocols={
            'within_subject': {'train': [s for s in sessions if split_of.get(s) == 'train'],
                               'test': [s for s in sessions if split_of.get(s) == 'test']},
            **{f'cross_subject_{a}2{b}': {'train': [s for s in sessions if s.startswith(a)],
                                          'test': [s for s in sessions if s.startswith(b)]}
               for a in subjects for b in subjects if a != b},
        },
        clips=rows)
    with open(os.path.join(args.out, 'manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=2)

    cols = ['clip_id', 'subject', 'session', 'split', 'activity', 'label', 'round',
            'n_frames', 'n_packets', 'mask_coverage', 'frames_dir']
    with open(os.path.join(args.out, 'clips.csv'), 'w') as fh:
        fh.write(','.join(cols) + '\n')
        for r in rows:
            fh.write(','.join(str(r[c]) for c in cols) + '\n')

    print(f'wrote {args.out}/')
    print(f'  {len(rows)} clips, {len(acts)} classes {acts}')
    print(f'  csi [T={T}, L={len(link_order)}, S={n_sub}] @ {args.rate:g} Hz, '
          f'mask coverage {100 * np.mean(cover):.1f}%')
    print(f'  {int(valid_sub.sum())}/{n_sub} subcarriers carry signal')
    for p, v in manifest['protocols'].items():
        ntr = sum(1 for r in rows if r['session'] in v['train'])
        nte = sum(1 for r in rows if r['session'] in v['test'])
        print(f'  protocol {p:28s} train {ntr:3d} / test {nte:3d}')


if __name__ == '__main__':
    main()
