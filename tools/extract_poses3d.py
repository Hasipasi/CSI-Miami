#!/usr/bin/env python3
"""Per-frame 3-D body estimates from the colour frames with NLF (Neural Localizer
Fields, Sárándi & Pons-Moll 2024): SMPL pose, shape and camera-frame translation
for the one person in the array, plus its 2-D joints and per-joint uncertainty.

This is the FIRST of two steps and its output is an initialisation, not the
target. A monocular network is stable in pose but its distance from the camera is
a guess from body size (it cannot know whether a person is small and near or large
and far); fit_body.py then anchors that guess to the RealSense depth and smooths
the take as one sequence. Keep the two apart: this step is the slow one and never
needs re-running when the fit's weights change.

Why NLF over lifting the 2-D YOLO skeleton with the depth at each joint: a depth
pixel at a joint is the body SURFACE, not the joint inside it, and it is missing
at every occluded, edge, or too-dark joint -- what came out of that approach was a
skeleton whose depth jumped by centimetres frame to frame and whose joints were
biased towards the camera. NLF gives a full parametric body at once, with the
joint positions where a joint actually is, and the depth is then used for what it
is good at: the metric position of the surface.

Writes <take>_nlf.npz beside each capture, with per frame:
  pose [F, 72] axis-angle · betas [F, 10] · trans [F, 3] metres, camera frame
  joints3d [F, 24, 3] (from the SMPL fit) · joints3d_nonparam (the network's direct
  estimate) · joints2d [F, 24, 2] pixels (direct estimate) · joint_unc [F, 24] mm
  bbox [F, 5] x y w h score · found · n_persons · second_conf
  frame_t / frame_t_ns / frame_idx copied from the capture.
NaN where no person was found; the fit interpolates those from the neighbours.

The colour intrinsics come from the capture meta (recorded since 2026-09-17), a
calib JSON (--calib, from tools/rs_calib.py), or nominal D435i values, in that
order. NLF uses them to place the body in metres, so the nominal fallback costs
accuracy in exactly the quantity the depth is meant to fix -- acceptable here
because fit_body.py re-derives the distance, but not for the intrinsics the fit
itself projects with; run rs_calib.py.

  .venv_pose/bin/python tools/extract_poses3d.py --src data
  .venv_pose/bin/python tools/extract_poses3d.py --src data/260916_train_RR.gergo --overwrite
  .venv_pose/bin/python tools/extract_poses3d.py --check-smpl     # verify the SMPL layer

The model file: models/nlf_l_multi_0.3.2.torchscript (493 MB, noncommercial
research licence) from https://github.com/isarandi/nlf/releases.
"""

import argparse
import glob
import json
import os
import sys
import time
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from body_common import (K_of, MODELS_DIR, SMPL, SMPL_NPZ, export_smpl_npz,   # noqa: E402
                         frame_members, load_calib, read_jpeg, smpl_from_nlf)

DEFAULT_MODEL = MODELS_DIR / 'nlf_l_multi_0.3.2.torchscript'


def find_takes(srcs):
    """Capture archives with colour frames under the given files / dirs / parents."""
    archives = []
    for src in srcs:
        if os.path.isfile(src):
            archives.append(src)
            continue
        archives.extend(glob.glob(os.path.join(src, '*.npz')))
        archives.extend(glob.glob(os.path.join(src, '*', '*.npz')))
    out = []
    for a in sorted(set(archives)):
        base = os.path.basename(a)
        if any(base.endswith(s) for s in ('_pose.npz', '_nlf.npz', '_body.npz')):
            continue
        out.append(a)
    return out


def load_model(path, device='cuda'):
    import torch
    import torchvision  # noqa: F401  registers torchvision::nms, which the NLF detector calls
    import warnings
    warnings.filterwarnings('ignore', message='.*torch.jit.load.*')
    t0 = time.time()
    model = torch.jit.load(str(path), map_location=device).eval()
    print(f'loaded {os.path.basename(str(path))} in {time.time() - t0:.1f}s', flush=True)
    return model


def ensure_smpl_npz(model, force=False):
    """models/smpl_neutral.npz from the body model inside the NLF torchscript."""
    if SMPL_NPZ.exists() and not force:
        return SMPL_NPZ
    arrays = smpl_from_nlf(model)
    p = export_smpl_npz(arrays, SMPL_NPZ, source='NLF torchscript body_models.smpl',
                        pose_feature='R')
    print(f'wrote {p} (SMPL arrays extracted from the NLF model)', flush=True)
    return p


def check_smpl(model, device='cuda'):
    """The hand-written SMPL layer must reproduce NLF's own vertices for the same
    parameters; if it does not, every fitted joint would be silently wrong."""
    import torch
    ensure_smpl_npz(model, force=True)
    smpl = SMPL(SMPL_NPZ, device=device)
    bm = model.body_models.smpl
    g = torch.Generator(device='cpu').manual_seed(0)
    pose = (torch.randn(4, 72, generator=g) * 0.4).to(device)
    betas = (torch.randn(4, 10, generator=g) * 1.5).to(device)
    trans = (torch.randn(4, 3, generator=g)).to(device)
    ref = bm(pose, betas, trans)
    rv, rj = ref['vertices'], ref['joints']
    v, j = smpl.forward(pose, betas, trans)
    dv = (v - rv).norm(dim=-1).max().item()
    dj = (j - rj[:, :24]).norm(dim=-1).max().item()
    print(f'SMPL layer vs NLF body model: max vertex diff {1e3 * dv:.3f} mm, '
          f'max joint diff {1e3 * dj:.3f} mm')
    if dv > 1e-3 or dj > 1e-3:
        sys.exit('SMPL layer does NOT match NLF: fix body_common.SMPL before fitting')
    print('OK')


def detect(model, imgs, K, args):
    """One batch of RGB uint8 [B, H, W, 3] -> NLF's parametric result dict."""
    import torch
    x = torch.from_numpy(np.ascontiguousarray(imgs)).permute(0, 3, 1, 2).to('cuda')
    Kt = torch.as_tensor(K, dtype=torch.float32, device='cuda')[None]
    fn = getattr(model, 'detect_smpl_batched', None)
    kw = dict(intrinsic_matrix=Kt, detector_threshold=args.det_thresh,
              num_aug=args.num_aug, internal_batch_size=args.internal_batch)
    with torch.inference_mode():
        if fn is not None:
            return fn(x, **kw)
        return model.detect_parametric_batched(x, model_name='smpl', **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', nargs='+', default=['data'],
                    help='capture file, session directory, or parent of sessions')
    ap.add_argument('--model', default=str(DEFAULT_MODEL))
    ap.add_argument('--calib', default=None, help='calib JSON from tools/rs_calib.py')
    ap.add_argument('--batch', type=int, default=8, help='frames per forward pass')
    ap.add_argument('--internal-batch', type=int, default=32)
    ap.add_argument('--num-aug', type=int, default=1,
                    help='test-time augmentations averaged per person (slower, steadier)')
    ap.add_argument('--det-thresh', type=float, default=0.3)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--check-smpl', action='store_true',
                    help='verify body_common.SMPL against the NLF body model and exit')
    args = ap.parse_args()

    model = load_model(args.model)
    if args.check_smpl:
        check_smpl(model)
        return
    ensure_smpl_npz(model)

    takes = find_takes(args.src)
    if not takes:
        sys.exit(f'no captures under {args.src}')
    print(f'{len(takes)} takes', flush=True)
    t0 = time.time()
    tot_f = tot_found = 0
    worst = []
    skipped = []
    for n, path in enumerate(takes, 1):
        out = path[:-4] + '_nlf.npz'
        if os.path.exists(out) and not args.overwrite:
            continue
        with np.load(path) as d, zipfile.ZipFile(path) as zf:
            meta = json.loads(str(d['meta'])) if 'meta' in d.files else {}
            colour, _depth, _ext = frame_members(d, meta)
            if not any(colour):
                skipped.append((os.path.basename(path), 'depth-only take, no colour frames'))
                continue
            timing = {k: d[k] for k in ('frame_t', 'frame_t_ns', 'frame_idx', 'frame_seq')
                      if k in d.files}
            calib = load_calib(meta, args.calib)
            K = K_of(calib['colour_intrinsics'])
            F = len(colour)
            pose = np.full((F, 72), np.nan, np.float32)
            betas = np.full((F, 10), np.nan, np.float32)
            trans = np.full((F, 3), np.nan, np.float32)
            j3 = np.full((F, 24, 3), np.nan, np.float32)
            j3n = np.full((F, 24, 3), np.nan, np.float32)
            j2 = np.full((F, 24, 2), np.nan, np.float32)
            unc = np.full((F, 24), np.nan, np.float32)
            box = np.full((F, 5), np.nan, np.float32)
            npers = np.zeros(F, np.int16)
            second = np.zeros(F, np.float32)
            for s in range(0, F, args.batch):
                members = colour[s:s + args.batch]
                imgs = np.stack([read_jpeg(zf, m) for m in members])
                if len(members) < args.batch:
                    # TorchScript re-profiles for 10+ s on every new input shape, so
                    # the last batch is padded to the same size rather than shrunk.
                    pad = np.repeat(imgs[-1:], args.batch - len(members), axis=0)
                    imgs = np.concatenate([imgs, pad])
                pred = detect(model, imgs, K, args)
                for b in range(len(members)):
                    i = s + b
                    bx = pred['boxes'][b].cpu().numpy()
                    if len(bx) == 0:
                        continue
                    scores = bx[:, 4]
                    k = int(np.argmax(scores))
                    npers[i] = len(bx)
                    if len(bx) > 1:
                        second[i] = float(np.sort(scores)[-2])
                    box[i] = bx[k]
                    pose[i] = pred['pose'][b][k].cpu().numpy()
                    betas[i] = pred['betas'][b][k].cpu().numpy()[:10]
                    trans[i] = pred['trans'][b][k].cpu().numpy()
                    j3[i] = pred['joints3d'][b][k].cpu().numpy()[:24]
                    j3n[i] = pred['joints3d_nonparam'][b][k].cpu().numpy()[:24]
                    # the network's direct 2-D estimate is the observation the fit
                    # reprojects against; the parametric one is derivable from pose
                    j2[i] = pred['joints2d_nonparam'][b][k].cpu().numpy()[:24]
                    unc[i] = pred['joint_uncertainties'][b][k].cpu().numpy()[:24]
        found = np.isfinite(trans[:, 0])
        nlf_meta = dict(
            model=os.path.basename(args.model), layout='SMPL-24 joints, axis-angle pose',
            units='metres, colour camera optical frame (x right, y down, z forward)',
            image_size=[calib['colour_intrinsics'].get('width', 1280),
                        calib['colour_intrinsics'].get('height', 720)],
            colour_intrinsics=calib['colour_intrinsics'],
            calib_source=calib['source'], num_aug=args.num_aug,
            detector_threshold=args.det_thresh,
            selection='highest-scoring box per frame',
            source_capture=os.path.basename(path),
            timestamp_origin=meta.get('timestamp_origin', 'record_start'),
            timestamp_unit=meta.get('timestamp_unit', 'nanoseconds'))
        tmp = out + '.tmp'
        with open(tmp, 'wb') as fh:
            np.savez_compressed(fh, pose=pose, betas=betas, trans=trans, joints3d=j3,
                                joints3d_nonparam=j3n,
                                joints2d=j2, joint_unc=unc, bbox=box, found=found,
                                n_persons=npers, second_conf=second,
                                meta=np.array(json.dumps(nlf_meta)), **timing)
        os.replace(tmp, out)
        tot_f += F
        tot_found += int(found.sum())
        if found.mean() < 0.98:
            worst.append((os.path.basename(out)[:-8], float(found.mean()), F))
        el = time.time() - t0
        print(f'  {n}/{len(takes)} {os.path.basename(path)[:-4]:24s} {F} frames, '
              f'{100 * found.mean():.1f}% found, z {np.nanmedian(trans[:, 2]):.2f} m  '
              f'({tot_f / max(el, 1e-9):.1f} fps, {el:.0f}s)', flush=True)

    print(f'\n{tot_f} frames, person found in {tot_found} '
          f'({100 * tot_found / max(tot_f, 1):.2f}%)')
    for name, why in skipped:
        print(f'  skipped {name}: {why}')
    if worst:
        print('takes with <98% detection:')
        for name, rate, F in sorted(worst, key=lambda x: x[1])[:20]:
            print(f'  {name:28s} {100 * rate:5.1f}%  ({F} frames)')


if __name__ == '__main__':
    main()
