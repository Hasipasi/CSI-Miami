#!/usr/bin/env python3
"""Validate a recorded session before anyone trains on it.

Every check here exists because the failure it looks for is silent: the capture
still produces a file, the file still loads, and the damage only shows up as a
model that will not learn or a result that will not reproduce. Cheap to run now,
expensive to discover later.

  python3 check_session.py ~/csi_test/data/balazs_train
"""

import glob
import json
import os
import sys
import zipfile

import numpy as np

# Widths the firmware is known to emit: 192 amplitudes (frame v1), the 166 / 114 / 30
# subsets on the S3 boards, and on the C5 boards every HT-LTF subcarrier the radio
# returns: 117 at HT40, 57 at HT20. Pinning one number here would fail every capture
# the moment the encoding changed, so what is enforced is that every link within a
# take agrees -- a take that mixes widths is broken in a way no single expected
# value would catch.
KNOWN_SUB = (192, 166, 117, 114, 57, 30)


def links(d):
    return sorted({k.rsplit('|', 1)[0] for k in d.files if k.endswith('|t')})


# A frame's CSI is everything within half a frame period of its timestamp, either
# side: windows centred on consecutive frames tile the timeline without overlap.
# Same definition as frame_coverage in capture.py -- duplicated so this script
# stays importable without pyserial and the camera stack.
FRAME_WINDOW_S = 1 / 60          # the 30 fps value, used with fewer than 2 frames


def frame_coverage(ft, link_t, want, half=None):
    """Share of camera frames that carry >= 1 packet on EVERY link in `want`
    within +-half seconds (None: half the median frame interval). A link with
    no packets scores 0 and takes the total with it."""
    ft = np.asarray(ft, dtype=np.float64)
    if ft.size < 1 or not want:
        return float('nan')
    if half is None:
        half = 0.5 * float(np.median(np.diff(ft))) if ft.size > 1 else FRAME_WINDOW_S
    every = np.ones(ft.size, dtype=bool)
    for lk in want:
        t = np.sort(np.asarray(link_t.get(lk, ()), dtype=np.float64))
        lo = np.searchsorted(t, ft - half, side='left')
        hi = np.searchsorted(t, ft + half, side='right')
        every &= (hi - lo) > 0
    return float(every.mean())


def check(path):
    """Return (row, [problems]) for one take."""
    name = os.path.basename(path)[:-4]
    probs = []
    try:
        d = np.load(path)
    except Exception as e:                                   # noqa: BLE001
        return {'name': name}, [f'unreadable: {e}']
    meta = json.loads(str(d['meta'])) if 'meta' in d.files else {}
    lab = meta.get('boards', {})

    ft = d['frame_t'] if 'frame_t' in d.files else np.array([])
    dur = float(ft[-1] - ft[0]) if len(ft) > 1 else 0.0
    fps = len(ft) / dur if dur > 0 else 0.0

    # Frames are embedded in new captures and live in a sibling directory in legacy
    # captures. Either way, frame k must resolve to its zero-padded JPEG name.
    fdir = os.path.join(os.path.dirname(path), f'{name}_frames')
    if os.path.isdir(fdir):
        njpg = len(glob.glob(os.path.join(fdir, '*.jpg'))) + \
            len(glob.glob(os.path.join(fdir, '*.png')))
    else:
        with zipfile.ZipFile(path) as zf:
            njpg = sum(n.startswith('frames/') and n.endswith(('.jpg', '.png'))
                       for n in zf.namelist())
    if njpg != len(ft):
        probs.append(f'{len(ft)} frame timestamps but {njpg} frame images on disk')
    # The depth stream, when a take carries one: its PNGs must match its timestamps
    # and it must run at the colour frame rate, or the depth ground truth is thin.
    ndepth = len(d['depth_t']) if 'depth_t' in d.files else 0
    if ndepth:
        with zipfile.ZipFile(path) as zf:
            npng = sum(n.startswith('depth/') and n.endswith('.png') for n in zf.namelist())
        if npng != ndepth:
            probs.append(f'{ndepth} depth timestamps but {npng} depth images')
        if len(ft) > 1 and ndepth < 0.9 * len(ft):
            probs.append(f'only {ndepth} depth frames for {len(ft)} colour frames')
        if meta.get('dropped_depth_encode'):
            probs.append(f'{meta["dropped_depth_encode"]} depth frames never encoded')
    if meta.get('dropped_encode'):
        probs.append(f'{meta["dropped_encode"]} frames never encoded')
    dropped = meta.get('frames_dropped') or {}
    if sum(dropped.values()) > 0.1 * max(meta.get('frames_recorded', len(ft)), 1):
        probs.append(f'{sum(dropped.values())} of {meta.get("frames_recorded")} frames '
                     f'pruned: {dropped}')
    if 'frame_seq' in d.files and len(ft) > 1:
        # Gaps in the camera's frame counter are frames the driver dropped --
        # minus the frames the writer pruned on purpose, which leave the same gaps.
        gaps = int(np.sum(np.maximum(np.diff(d['frame_seq']) - 1, 0)))
        pruned = sum((meta.get('frames_dropped') or {}).values())
        gaps = max(gaps - pruned, 0)
        if gaps > 0.02 * len(ft):
            probs.append(f'{gaps} frames dropped by the driver ({100 * gaps / len(ft):.0f}%)')
    if len(ft) > 1 and not np.all(np.diff(ft) > 0):
        probs.append('frame timestamps not monotonic')
    want_fps = meta.get('fps_requested')
    if want_fps and len(ft) > 10:
        # Median step, so dropped frames do not count: a camera clock mapped onto the
        # host at the wrong slope once stepped 36.7 ms for 2.4 s with no gap in the
        # frame counter -- only this and the jump back at its end gave it away.
        step_fps = 1.0 / float(np.median(np.diff(ft)))
        if abs(step_fps / float(want_fps) - 1) > 0.05:
            probs.append(f'frame timestamps step at {step_fps:.1f} fps, requested {want_fps}')

    lks = links(d)
    counts, worst_dt, widths = {}, 0.0, {}
    for lk in lks:
        tx, rx = lk.split('|')
        nm = f'{lab.get(tx, tx[-5:])}->{lab.get(rx, rx[-5:])}'
        a, t = d[f'{lk}|a'], d[f'{lk}|t'].astype(np.float64)
        counts[nm] = len(t)
        widths.setdefault(a.shape[1], []).append(nm)
        if len(t) > 1 and not np.all(np.diff(t) >= 0):
            probs.append(f'{nm}: packet times not monotonic')
        # a dead or stuck link still writes a well-formed array
        if not np.isfinite(a).all():
            probs.append(f'{nm}: non-finite amplitudes')
        if a.size and float(a.std()) == 0.0:
            probs.append(f'{nm}: amplitudes constant (stuck)')
        if a.size and float(np.mean(a.sum(axis=1) == 0)) > 0.05:
            probs.append(f'{nm}: {100 * np.mean(a.sum(axis=1) == 0):.0f}% all-zero packets')
        # Phase, when the boards sent I/Q. Raw phase is carrier/sampling offset and
        # looks like noise by design, so the check is on the detrended residual --
        # that is the part that carries channel information, and it being frozen or
        # non-finite means the complex payload is not usable even though |iq| is.
        if f'{lk}|iq' in d.files:
            q = d[f'{lk}|iq']
            if not np.isfinite(q).all():
                probs.append(f'{nm}: non-finite iq')
            elif len(q) > 2 and q.shape[1] > 3:
                ph = np.unwrap(np.angle(q), axis=1)
                idx = np.arange(q.shape[1])
                c = np.polyfit(idx, ph.T, 1)
                res = ph - (np.outer(idx, c[0]) + c[1]).T
                if float(res.std()) == 0.0:
                    probs.append(f'{nm}: detrended phase constant (stuck)')
        if f'{lk}|dt' in d.files and len(d[f'{lk}|dt']):
            worst_dt = max(worst_dt, float(np.max(np.abs(d[f'{lk}|dt']))) * 1e3)

    if len(widths) > 1:
        probs.append('links disagree on subcarrier count: '
                     + '; '.join(f'{w} ({len(v)} links)' for w, v in sorted(widths.items())))
    for w in widths:
        if w not in KNOWN_SUB:
            probs.append(f'unexpected subcarrier count {w}, known are {KNOWN_SUB}')

    # A round-robin run should yield every ordered pair. Missing links are the
    # failure this whole script exists for: the files load, the counts look sane,
    # and a board that produced nothing is invisible unless you count the pairs.
    # Reporting the inventory without judging it (as this first did) is no check.
    boards = sorted(set(lab.values())) if lab else []
    mode = str(meta.get('mode', ''))
    # A pinned transmitter is expected to yield exactly its own outgoing links, so
    # "12 links missing" is correct there and "one of 3 missing" is the real fault.
    # Checking only the round-robin case left single-TX sessions unvalidated, which
    # is precisely where a silent board is hardest to notice: fewer rows to miss.
    pinned = mode if mode in boards else None
    if pinned is None and mode.startswith('fixed'):
        txs = {n.split('->')[0] for n in counts}
        pinned = next(iter(txs)) if len(txs) == 1 else None

    want = set()
    if boards and pinned:
        want = {f'{pinned}->{b}' for b in boards if b != pinned}
        got = set(counts)
        for miss in sorted(want - got):
            probs.append(f'MISSING link {miss} (transmitter pinned to {pinned})')
        extra = sorted(got - want)
        if extra:
            probs.append(f'unexpected links for a {pinned}-only capture: {extra}')
    elif boards and mode.startswith('round'):
        want = {f'{a}->{b}' for a in boards for b in boards if a != b}
        got = set(counts)
        for miss in sorted(want - got):
            probs.append(f'MISSING link {miss}')
        silent = [b for b in boards if not any(nm.endswith(f'->{b}') for nm in got)]
        for b in silent:
            probs.append(f'board {b} received nothing at all (its reader produced no CSI)')
        deaf = [b for b in boards if not any(nm.startswith(f'{b}->') for nm in got)]
        for b in deaf:
            probs.append(f'board {b} never transmitted')

    # Frame coverage: the share of camera frames that saw every expected link. A
    # take can have every link present at a healthy average rate and still leave
    # most frames without most links, because a link is sampled in bursts; this is
    # the number the count-based round-robin (meta 'burst') exists to raise, so a
    # burst-mode take that misses it is reported as the failure it is.
    link_t = {}
    for lk in lks:
        tx, rx = lk.split('|')
        # Board clock mapped to host time (tc) where recorded: it is what the
        # recorder builds the windows from; host arrival (t) lags up to ~36 ms under
        # load and reports empty windows the saved data does not have.
        link_t[f'{lab.get(tx, tx[-5:])}->{lab.get(rx, rx[-5:])}'] = \
            d[f'{lk}|tc'] if f'{lk}|tc' in d.files else d[f'{lk}|t']
    cover = frame_coverage(ft, link_t, sorted(want)) if want else float('nan')
    # and how many packets each link has per window: the minimum is the guarantee
    win_min = win_med = float('nan')
    if want and len(ft) > 1:
        half = 0.5 * float(np.median(np.diff(ft)))
        w = []
        for lk in sorted(want):
            t = np.sort(np.asarray(link_t.get(lk, ()), dtype=np.float64))
            w.append(np.searchsorted(t, ft + half, side='right')
                     - np.searchsorted(t, ft - half, side='left'))
        w = np.array(w)
        win_min, win_med = int(w.min()), float(np.median(w))
    if meta.get('burst') and cover == cover and cover < 0.9:
        probs.append(f'only {100 * cover:.0f}% of frames saw every link in their window (burst '
                     f'{meta["burst"]}, token cycle {meta.get("ring_cycle_ms")} ms, '
                     f'{meta.get("ring_timeouts", 0)} turns without TX_DONE)')

    # The saved per-frame windows: present, the right shape, and consistent with
    # the packet arrays they were cut from.
    if 'win_iq' in d.files:
        w = d['win_iq']
        if w.shape[0] != len(ft):
            probs.append(f'win_iq has {w.shape[0]} frames for {len(ft)} frame timestamps')
        if meta.get('window_overflow'):
            probs.append(f'{meta["window_overflow"]} packets did not fit the '
                         f'{meta.get("window_slots")} window slots')
        wc = d['win_count']
        wb = [str(b) for b in d['win_boards']] if 'win_boards' in d.files else []
        if wc.size and wb and want:
            cells = np.zeros((len(wb), len(wb)), dtype=bool)
            for nm in want:
                a, b = nm.split('->')
                if a in wb and b in wb:
                    cells[wb.index(a), wb.index(b)] = True
            n_empty = int(np.sum((wc == 0) & cells[None]))
            if n_empty:
                probs.append(f'{n_empty} frame-link windows are empty (all-zero padding)')
    elif 'frame_t' in d.files and any(k.endswith('|iq') for k in d.files) \
            and not meta.get('raw'):
        probs.append('no saved CSI windows (win_iq) in an I/Q take')

    if not lks:
        probs.append('no links at all')
    return ({'name': name, 'frames': len(ft), 'depth': ndepth, 'dur': dur, 'fps': fps,
             'links': len(lks), 'pkts': sum(counts.values()), 'cover': cover,
             'win_min': win_min, 'win_med': win_med,
             'counts': counts, 'worst_dt': worst_dt, 'mode': meta.get('mode', '?')},
            probs)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    # Skeletons live beside the captures as <take>_pose.npz and are a different
    # schema; globbing them in made every take look linkless.
    paths = sorted(p for p in glob.glob(os.path.join(root, '*.npz'))
                   if not p.endswith('_pose.npz'))
    if not paths:
        print(f'no .npz in {root}')
        sys.exit(1)

    rows, allprobs = [], []
    for p in paths:
        row, probs = check(p)
        rows.append(row)
        for pr in probs:
            allprobs.append((row['name'], pr))

    good = [r for r in rows if 'frames' in r]
    print(f'{len(paths)} takes in {root}\n')
    print(f'{"take":18s} {"mode":5s} {"frames":>6s} {"depth":>5s} {"dur":>6s} {"fps":>6s} '
          f'{"links":>5s} {"pkts":>6s} {"cov%":>5s} {"win min/med":>11s} {"dt ms":>6s}')
    for r in good:
        flag = '  <--' if any(n == r['name'] for n, _ in allprobs) else ''
        cov = f'{100 * r["cover"]:5.1f}' if r['cover'] == r['cover'] else '    -'
        win = (f'{r["win_min"]:5d} {r["win_med"]:5.1f}' if r['win_med'] == r['win_med']
               else '          -')
        print(f'{r["name"]:18s} {str(r["mode"])[:5]:5s} {r["frames"]:6d} {r["depth"]:5d} '
              f'{r["dur"]:6.2f} {r["fps"]:6.2f} {r["links"]:5d} {r["pkts"]:6d} {cov} {win} '
              f'{r["worst_dt"]:6.1f}{flag}')

    if good:
        print('\n--- consistency across takes ---')
        for key, fmt in (('frames', '%.1f'), ('dur', '%.2f'), ('fps', '%.2f'),
                         ('pkts', '%.1f'), ('links', '%.1f'), ('cover', '%.3f')):
            v = np.array([r[key] for r in good], dtype=float)
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            print(f'  {key:7s} mean {fmt % v.mean():>8s}  min {fmt % v.min():>8s}  '
                  f'max {fmt % v.max():>8s}  spread {100 * v.std() / max(v.mean(), 1e-9):5.1f}%')

        # A link that vanishes in some takes but not others is worse than one that is
        # always absent: the feature matrix silently changes width between samples.
        seen = {}
        for r in good:
            for nm in r['counts']:
                seen.setdefault(nm, 0)
                seen[nm] += 1
        missing = {nm: c for nm, c in seen.items() if c != len(good)}
        print(f'\n  links present in every take: '
              f'{sorted(nm for nm, c in seen.items() if c == len(good))}')
        if missing:
            print('  INCONSISTENT links (present in some takes only):')
            for nm, c in sorted(missing.items()):
                print(f'    {nm}: {c}/{len(good)} takes')

        # per-class counts, since a protocol run is meant to be balanced
        klass = {}
        for r in good:
            klass.setdefault(r['name'].rstrip('0123456789'), []).append(r['pkts'])
        print('\n  per-pose takes and packet counts:')
        for k, v in sorted(klass.items()):
            print(f'    {k:16s} {len(v):2d} takes, packets mean {np.mean(v):6.1f} '
                  f'min {min(v):5d} max {max(v):5d}')

    print(f'\n--- problems: {len(allprobs)} ---')
    if not allprobs:
        print('  none found')
    else:
        for n, pr in allprobs:
            print(f'  {n:18s} {pr}')


if __name__ == '__main__':
    main()
