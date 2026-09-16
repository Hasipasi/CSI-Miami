#!/usr/bin/env python3
"""Apply the recorder's frame pruning to takes already on disk.

Since 2026-09-16 every take is written with only its complete frames: a colour
image, a depth frame within half a frame period, and a CSI window with at least one
packet on every link. Takes recorded before that (or with an older writer) can be
pruned the same way here; the file is rewritten in place, image members of dropped
frames removed, the per-link nearest-frame indices recomputed, and the counts put
in meta. Idempotent.

  python3 prune_frames.py /Volumes/GergoDisk/data/260916_train_RR.gergo
  python3 prune_frames.py <session> --rewindow   # rebuild the windows from the packets
                                                 # on the boards' clocks (exact under host
                                                 # stalls), then prune again
"""

import glob
import io
import json
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture import (apply_frame_mask, assign_frames, board_clock_times,   # noqa: E402
                     build_windows, prune_frames, receiver_tensor)


def prune(path, rewindow=False):
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
    d = np.load(path)
    out = {k: d[k] for k in d.files if k != 'meta'}
    meta = json.loads(str(d['meta']))
    if 'frame_t' not in out:
        return None
    n = len(out['frame_t'])
    links = sorted({k.rsplit('|', 1)[0] for k in out if k.endswith('|t') and '|' in k})
    if rewindow:
        # Board-clock packet times and fresh windows from the raw packet arrays:
        # takes written with host-arrival windows get exact ones.
        meta['clock_offsets'] = board_clock_times(out, links)
        for k in list(out):
            if k.startswith('win_') or k.startswith('csi'):
                del out[k]
        build_windows(out, links, meta, meta.get('window_slots', 16)
                      if str(meta.get('mode', '')) not in meta.get('boards', {}).values()
                      else 16)
    has_img = [f'frames/{i:06d}.jpg' in names or f'frames/{i:06d}.png' in names
               for i in out['frame_idx']]
    has_png = ([f'depth/{i:06d}.png' in names for i in out['depth_idx']]
               if 'depth_idx' in out else None)
    boards = [str(b) for b in out['win_boards']] if 'win_boards' in out else []
    mode = str(meta.get('mode', ''))
    pinned = mode if mode in boards else None
    keep, dropped = prune_frames(out, has_img, has_png, pinned)
    # Takes written before win_a existed get it from win_iq and win_gain.
    added = False
    if 'win_iq' in out and 'win_a' not in out:
        iq = out['win_iq'].astype(np.float32)
        out['win_a'] = (np.hypot(iq[..., 0], iq[..., 1])
                        * out['win_gain'][..., None]).astype(np.float16)
        added = True
    if 'win_a' in out and 'csi' not in out:
        out.update(receiver_tensor(out, boards, meta.get('window_slots', 16), pinned))
        added = True
    if keep.all() and meta.get('frames_kept') == n and not added and not rewindow:
        return n, 0, dropped
    apply_frame_mask(out, keep)
    assign_frames(out, links)
    prev = meta.get('frames_dropped') or {}
    meta.update(frames_recorded=meta.get('frames_recorded', n), frames_kept=int(keep.sum()),
                frames_dropped={k: prev.get(k, 0) + v for k, v in dropped.items()},
                depth_frames=int(len(out.get('depth_t', []))))
    out['meta'] = np.array(json.dumps(meta))
    keep_members = {f'frames/{i:06d}.jpg' for i in out['frame_idx']} | \
        {f'frames/{i:06d}.png' for i in out['frame_idx']} | \
        ({f'depth/{i:06d}.png' for i in out['depth_idx']} if 'depth_idx' in out else set())
    tmp = path + '.tmp'
    with open(tmp, 'wb') as fh:
        np.savez_compressed(fh, **out)
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(tmp, 'a', compression=zipfile.ZIP_STORED) as dst:
        for name in src.namelist():
            if name in keep_members:
                dst.writestr(name, src.read(name))
    with zipfile.ZipFile(tmp) as zf:
        if zf.testzip() is not None:
            raise OSError(f'pruned archive failed CRC validation: {tmp}')
    os.replace(tmp, path)
    return n, int((~keep).sum()), dropped


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    rewindow = '--rewindow' in sys.argv
    root = args[0] if args else '.'
    paths = sorted(p for p in glob.glob(os.path.join(root, '*.npz')) if not p.endswith('_pose.npz'))
    if not paths:
        print(f'no .npz in {root}')
        sys.exit(1)
    total = 0
    for p in paths:
        r = prune(p, rewindow)
        if r is None:
            print(f'{os.path.basename(p):24s} no frames, skipped')
            continue
        n, gone, dropped = r
        total += gone
        print(f'{os.path.basename(p):24s} {n:4d} frames, dropped {gone:3d} '
              f'(no image {dropped["no_image"]}, no depth {dropped["no_depth"]}, '
              f'bad csi {dropped["bad_csi"]})')
    print(f'{len(paths)} takes, {total} frames dropped')


if __name__ == '__main__':
    main()
