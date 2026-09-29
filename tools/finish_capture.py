#!/usr/bin/env python3
"""Add the per-frame CSI windows to takes recorded with `--raw`.

A raw take holds everything the rig measured -- the packets per link, the colour
frames, the depth frames -- and none of what is computed from them. This is the
computation, run wherever the data ended up rather than on the PC someone is
waiting at. It feeds the packets back through capture.write_capture, the same
single writer the recorder uses, so a finished take is byte-for-byte what the
recorder would have written on the spot.

  python3 tools/finish_capture.py data/<session>            # every raw take in it
  python3 tools/finish_capture.py data/<session>/take.npz --out finished/
  python3 tools/finish_capture.py data/<session> --keep     # keep the raw files

Without --out a finished take replaces its raw file (via a temporary, so an
interrupted run never destroys the original). Takes that already carry windows are
skipped, so running it twice is safe.
"""
import argparse
import glob
import json
import os
import pathlib
import shutil
import sys
import tempfile
import time
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture import WINDOW_SLOTS, WINDOW_SLOTS_PINNED, write_capture   # noqa: E402


def raw_takes(srcs):
    out = []
    for src in srcs:
        paths = [src] if os.path.isfile(src) else sorted(
            glob.glob(os.path.join(src, '*.npz')) + glob.glob(os.path.join(src, '*', '*.npz')))
        for p in paths:
            if any(os.path.basename(p).endswith(s) for s in
                   ('_pose.npz', '_nlf.npz', '_body.npz')):
                continue
            out.append(p)
    return out


def unpack(path, workdir):
    """The arrays, the meta, and the frame/depth images written back out as the
    <prefix>_frames / <prefix>_depth directories write_capture expects."""
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d['meta']))
    prefix = os.path.join(workdir, 'take')
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            for member, folder in (('frames/', f'{prefix}_frames'),
                                   ('depth/', f'{prefix}_depth')):
                if name.startswith(member) and not name.endswith('/'):
                    os.makedirs(folder, exist_ok=True)
                    with zf.open(name) as fh, open(
                            os.path.join(folder, os.path.basename(name)), 'wb') as out:
                        shutil.copyfileobj(fh, out)
    return d, meta, prefix


def rebuild_recs(d, meta):
    """The packet lists write_capture takes, back from the per-link arrays.

    t0 is the take's own zero, so the times handed back are absolute again; the
    writer subtracts t0 itself. `lts` goes back to the board's microseconds, which
    is what the clock mapping and the windows are built from.
    """
    t0 = float(meta.get('t0_epoch', 0.0))
    links = sorted({k.rsplit('|', 1)[0] for k in d.files if k.endswith('|t')})
    recs = {}
    for lk in links:
        tx, rx = lk.split('|')
        t = d[f'{lk}|t'].astype(np.float64) + t0
        lts_us = np.rint(d[f'{lk}|lts'].astype(np.float64) * 1e6).astype(np.int64)
        a = d[f'{lk}|a']
        iq = d[f'{lk}|iq'] if f'{lk}|iq' in d.files else None
        rssi = d[f'{lk}|rssi']
        gains = None
        if f'{lk}|gain' in d.files:
            gains = (d[f'{lk}|gain'], d[f'{lk}|agc'], d[f'{lk}|fft'])
        items = recs.setdefault(rx, [])
        for i in range(len(t)):
            items.append((float(t[i]), tx, int(lts_us[i]), int(rssi[i]), a[i],
                          None if iq is None else iq[i],
                          None if gains is None else (float(gains[0][i]), int(gains[1][i]),
                                                      int(gains[2][i]))))
    for rx in recs:
        recs[rx].sort(key=lambda r: r[0])
    return recs, t0


def finish(path, out_dir=None, keep=False, verbose=False):
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d['meta']))
    if not meta.get('raw'):
        return None, 'already finished'
    t_start = time.time()
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(path))) as work:
        d, meta, prefix = unpack(path, work)
        recs, t0 = rebuild_recs(d, meta)
        frames = [(int(i), int(s), float(e)) for i, s, e in
                  zip(d['frame_idx'], d['frame_seq'], d['frame_epoch'])]
        depth_frames = ([(int(i), float(t) + t0) for i, t in
                         zip(d['depth_idx'], d['depth_t'])]
                        if 'depth_idx' in d.files else None)
        meta = {k: v for k, v in meta.items() if k != 'raw'}
        slots = WINDOW_SLOTS_PINNED if meta.get('mode') == 'fixedtx' else WINDOW_SLOTS
        new, report, ft = write_capture(prefix, recs, frames, t0, meta,
                                        depth_frames=depth_frames, window_slots=slots)
        dest = (os.path.join(out_dir, os.path.basename(path)) if out_dir else
                path if not keep else path.replace('.npz', '_finished.npz'))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        tmp = dest + '.tmp'
        shutil.move(new, tmp)
        os.replace(tmp, dest)
    if verbose:
        print(report)
    return dest, (f'{len(ft)} frames, {sum(len(v) for v in recs.values())} packets, '
                  f'{time.time() - t_start:.1f}s')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', nargs='+', help='raw take(s), or a session/parent directory')
    ap.add_argument('--out', default=None, help='write finished takes here instead of '
                                                'replacing the raw ones')
    ap.add_argument('--keep', action='store_true',
                    help='keep the raw file, writing <take>_finished.npz beside it')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    takes = raw_takes(args.src)
    if not takes:
        sys.exit('no takes found')
    print(f'{len(takes)} takes', flush=True)
    done = skipped = 0
    for i, p in enumerate(takes, 1):
        dest, note = finish(p, args.out, args.keep, args.verbose)
        if dest is None:
            skipped += 1
            print(f'  {i}/{len(takes)} {os.path.basename(p):28s} skipped ({note})', flush=True)
        else:
            done += 1
            print(f'  {i}/{len(takes)} {os.path.basename(p):28s} {note}', flush=True)
    print(f'{done} finished, {skipped} skipped')


if __name__ == '__main__':
    main()
