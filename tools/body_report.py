#!/usr/bin/env python3
"""Summarise the depth-grounded body fits under a data tree: per session, how many
takes have a fit, the share of valid frames, the point-to-body residual, and how
far the RGB-only distance was off; then the takes worth looking at.

Every number here is read back from the *_body.npz files, so this is also the
check that the batch actually wrote what fit_body.py reported.

  python3 tools/body_report.py data              # all sessions
  python3 tools/body_report.py data/260917_RR_gergo --worst 10
"""

import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', nargs='+')
    ap.add_argument('--worst', type=int, default=8, help='takes to list')
    args = ap.parse_args()

    sessions = {}
    for src in args.src:
        for path in sorted(glob.glob(os.path.join(src, '*.npz')) + glob.glob(os.path.join(src, '*', '*.npz'))):
            if any(path.endswith(s) for s in ('_pose.npz', '_nlf.npz', '_body.npz')):
                continue
            sess = os.path.basename(os.path.dirname(path))
            s = sessions.setdefault(sess, dict(takes=0, nlf=0, body=0, rows=[]))
            s['takes'] += 1
            if os.path.exists(path[:-4] + '_nlf.npz'):
                s['nlf'] += 1
            bp = path[:-4] + '_body.npz'
            if not os.path.exists(bp):
                continue
            s['body'] += 1
            with np.load(bp) as B:
                dz = B['trans'][:, 2] - B['trans_nlf'][:, 2]
                s['rows'].append(dict(
                    take=os.path.basename(path)[:-4], F=len(B['found']),
                    found=float(B['found'].mean()), valid=float(B['valid'].mean()),
                    depth=float(np.nanmedian(B['depth_med'])),
                    depth_max=float(np.nanmax(B['depth_med'])),
                    reproj=float(np.nanmean(B['reproj_px'])),
                    dz=float(np.median(dz)), dz_sd=float(np.std(dz)),
                    z=float(np.median(B['trans'][:, 2])),
                    jump=float(np.max(np.linalg.norm(np.diff(B['keypoints3d'], axis=0), axis=-1)))
                    if len(B['found']) > 1 else 0.0))
    if not sessions:
        raise SystemExit('no captures found')

    print(f'{"session":28s} {"takes":>5s} {"nlf":>4s} {"body":>4s} {"found%":>7s} {"valid%":>7s} '
          f'{"depth mm":>9s} {"reproj px":>9s} {"z m":>5s} {"dz cm":>6s} {"max jump cm":>11s}')
    allrows = []
    for sess, s in sorted(sessions.items()):
        r = s['rows']
        if not r:
            print(f'{sess:28s} {s["takes"]:5d} {s["nlf"]:4d} {s["body"]:4d}')
            continue
        allrows += [(sess, x) for x in r]
        print(f'{sess:28s} {s["takes"]:5d} {s["nlf"]:4d} {s["body"]:4d} '
              f'{100 * np.mean([x["found"] for x in r]):7.1f} {100 * np.mean([x["valid"] for x in r]):7.1f} '
              f'{1e3 * np.median([x["depth"] for x in r]):9.1f} {np.mean([x["reproj"] for x in r]):9.1f} '
              f'{np.median([x["z"] for x in r]):5.2f} {100 * np.median([x["dz"] for x in r]):+6.1f} '
              f'{100 * np.max([x["jump"] for x in r]):11.1f}')
    if allrows:
        print(f'\n{len(allrows)} fitted takes. Worst by valid share, then by residual:')
        for sess, x in sorted(allrows, key=lambda t: (t[1]['valid'], -t[1]['depth']))[:args.worst]:
            print(f'  {sess}/{x["take"]:22s} valid {100 * x["valid"]:5.1f}%  found {100 * x["found"]:5.1f}%  '
                  f'depth {1e3 * x["depth"]:5.1f} (max {1e3 * x["depth_max"]:5.1f}) mm  '
                  f'reproj {x["reproj"]:4.1f} px  max joint jump {100 * x["jump"]:4.1f} cm/frame')
        big = sorted(allrows, key=lambda t: -t[1]['jump'])[:args.worst]
        print('Largest frame-to-frame joint jumps (a cut, a missed frame, or a bad fit):')
        for sess, x in big:
            print(f'  {sess}/{x["take"]:22s} {100 * x["jump"]:5.1f} cm/frame  valid {100 * x["valid"]:5.1f}%')


if __name__ == '__main__':
    main()
