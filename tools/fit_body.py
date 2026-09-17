#!/usr/bin/env python3
"""Fit one SMPL body per take to the RealSense depth, starting from NLF's per-frame
RGB estimate, and smooth it as one sequence (temporal bundle adjustment).

The division of labour: the RGB network is trusted for POSE (joint angles and the
body's proportions) because it has seen millions of people and a depth camera
tells you nothing about a limb it cannot see. The depth is trusted for WHERE the
body is (metric distance and lateral position) because a monocular network only
guesses that from apparent size, and that guess drifts frame to frame by more than
the movements being recorded. And the whole take is solved at once rather than
frame by frame, because a person's shape does not change within seven seconds and
their joints do not teleport between frames at 30 fps -- both are constraints the
per-frame estimate throws away.

Per take, the unknowns are one shape vector, and per frame a pose (72) and a
translation (3). The residuals, all robust (Geman-McClure), are:

  depth    every depth point on the person -> its nearest SMPL vertex, in metres.
           One-directional on purpose: the depth sees the front surface only, and a
           term pulling unseen back-side vertices to the points would flatten the
           body. Points are the ones inside the person's box, in a depth band
           around the initial body, and (after a first pass) within 15 cm of it,
           so floor and wall never enter the fit.
  2-D      the 24 SMPL joints projected with the colour intrinsics against the
           network's 2-D joint estimate, weighted by its per-joint uncertainty.
  prior    pose stays near the network's per-frame pose; shape near its median.
  temporal joint velocity and acceleration over the sequence, in metres per
           frame, scaled by the true frame spacing so a dropped frame is not
           mistaken for a jump; and pose velocity.

The distance is first read off the depth directly (torso pixels vs the initial
body surface) so the optimiser starts within a few cm rather than tens of cm from
the answer, then translation alone is fitted, then everything. Adam on the whole
sequence with the depth term accumulated in chunks of frames, so a 200-frame take
fits in a few GB of GPU memory.

Writes <take>_body.npz beside the capture:
  pose [F, 72] · betas [10] · trans [F, 3]
  joints3d [F, 24, 3] SMPL joints, metres, colour camera frame
  keypoints3d [F, 17, 3] COCO-17 in the same frame -- the training target
  keypoints2d [F, 17, 2] pixels (the projection of keypoints3d)
  keypoints [F, 17, 3] x, y, 1.0 -- COCO-17 2-D in the layout of *_pose.npz
  found [F] (the network saw a person) · valid [F] (found and the fit closed)
  depth_med [F] median point-to-body distance, m · n_points [F] · reproj_px [F]
  trans_nlf [F, 3] the network's translation before grounding (a diagnostic:
           how far monocular was off)
  frame_t / frame_t_ns / frame_idx as in the capture · meta

  docker compose run --rm pose python3 tools/fit_body.py --src data
  docker compose run --rm pose python3 tools/fit_body.py --src data/<session>/<take>.npz --plot
  docker compose run --rm pose python3 tools/fit_body.py --src data --elaborate   # slower, 2x iterations

The default is the fast setting (20 + 50 iterations, 1200 vertices, ~15 s a take);
--elaborate is 30 + 80 over 2300 (~25 s), which is what the 2026-09-17 corpus was
fitted with. Measured on one take the two give the same point-to-body residual
(31 vs 32 mm) and joints 8 mm apart on average, up to ~18 mm at the knees and
ankles, where the depth cloud is sparsest and the extra iterations still move
things. Use --elaborate for a dataset; the fast setting is for checking a take.
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
from body_common import (COCO17, K_of, SMPL, SMPL_NPZ, TORSO_JOINTS, depth_to_points,   # noqa: E402
                         frame_members, lerp_rotvec, load_calib, project, read_depth_png)


# ------------------------------------------------------------------ loading

def find_takes(srcs):
    archives = []
    for src in srcs:
        if os.path.isfile(src):
            archives.append(src)
            continue
        archives.extend(glob.glob(os.path.join(src, '*.npz')))
        archives.extend(glob.glob(os.path.join(src, '*', '*.npz')))
    return [a for a in sorted(set(archives))
            if not any(a.endswith(s) for s in ('_pose.npz', '_nlf.npz', '_body.npz'))]


def fill_gaps(N):
    """Per-frame initial parameters with undetected frames interpolated from the
    nearest detected neighbours (pose through the shortest rotation, translation
    and box linearly); shape is the median over detected frames."""
    found = N['found'].astype(bool)
    F = len(found)
    pose, trans, box = N['pose'].copy(), N['trans'].copy(), N['bbox'].copy()
    j2, unc = N['joints2d'].copy(), N['joint_unc'].copy()
    idx = np.flatnonzero(found)
    if len(idx) == 0:
        raise ValueError('no frame with a detected person')
    betas = np.nanmedian(N['betas'][found], axis=0).astype(np.float32)
    for i in np.flatnonzero(~found):
        k = np.searchsorted(idx, i)
        lo = idx[k - 1] if k > 0 else None
        hi = idx[k] if k < len(idx) else None
        if lo is None or hi is None:
            src = hi if lo is None else lo
            pose[i], trans[i], box[i] = pose[src], trans[src], box[src]
        else:
            w = (i - lo) / (hi - lo)
            pose[i] = lerp_rotvec(pose[lo], pose[hi], w)
            trans[i] = (1 - w) * trans[lo] + w * trans[hi]
            box[i] = (1 - w) * box[lo] + w * box[hi]
    return dict(found=found, pose=pose, trans=trans, betas=betas, box=box,
                joints2d=j2, joint_unc=unc)


def person_points(depth_u16, calib, K, box, z_lo, z_hi, y_floor, max_points, rng):
    """Depth points that can belong to the person: inside the (expanded) box,
    within the depth band, above the floor line, subsampled to max_points."""
    P, _ = depth_to_points(depth_u16, calib, z_min=max(0.3, z_lo), z_max=z_hi)
    if len(P) == 0:
        return P
    uv = project(K, P)
    x1, y1, bw, bh = box[:4]                 # NLF boxes are x, y, width, height, score
    x2, y2 = x1 + bw, y1 + bh
    mx, my = 0.15 * bw, 0.10 * bh
    keep = ((uv[:, 0] >= x1 - mx) & (uv[:, 0] <= x2 + mx)
            & (uv[:, 1] >= y1 - my) & (uv[:, 1] <= y2 + my) & (P[:, 1] <= y_floor))
    P = P[keep]
    if len(P) > max_points:
        P = P[rng.choice(len(P), max_points, replace=False)]
    return P


def coarse_distance(P, verts, joints, K, r_px=10.0):
    """Scale to apply to the body's camera-space position so its torso surface sits
    at the depth the camera measured: median over torso joints of (depth z of the
    points around the joint's pixel) / (front-most body vertex z there)."""
    if len(P) < 50:
        return 1.0
    uv_p = project(K, P)
    uv_v = project(K, verts)
    uv_j = project(K, joints[TORSO_JOINTS])
    ratios = []
    for j in range(len(uv_j)):
        dp = np.linalg.norm(uv_p - uv_j[j], axis=1) <= r_px
        dv = np.linalg.norm(uv_v - uv_j[j], axis=1) <= r_px
        if dp.sum() >= 5 and dv.sum() >= 3:
            ratios.append(np.median(P[dp, 2]) / verts[dv, 2].min())
    if len(ratios) < 3:
        return 1.0
    return float(np.clip(np.median(ratios), 0.5, 2.0))


# ------------------------------------------------------------------ the fit

def gm(x, sigma):
    """Geman-McClure: quadratic near zero, saturating at 1 beyond ~sigma."""
    x2 = x * x
    return x2 / (sigma * sigma + x2)


class SequenceFit:
    def __init__(self, smpl, init, clouds, j2d, w2d, K, frame_t, args, device):
        import torch
        self.torch, self.smpl, self.args, self.dev = torch, smpl, args, device
        F = len(init['pose'])
        self.F = F
        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=device)
        self.pose0 = t(init['pose'])
        self.betas0 = t(init['betas'])
        self.found = t(init['found'], torch.bool)
        self.pose = self.pose0.clone().requires_grad_(True)
        self.betas = self.betas0.clone().requires_grad_(True)
        self.trans = t(init['trans']).clone().requires_grad_(True)
        self.K = t(K)
        self.j2d = t(j2d)
        self.w2d = t(w2d)
        # per-frame point clouds padded to one tensor + mask, for batched cdist
        n = max(1, max(len(c) for c in clouds))
        P = np.zeros((F, n, 3), np.float32)
        M = np.zeros((F, n), bool)
        for i, c in enumerate(clouds):
            P[i, :len(c)] = c
            M[i, :len(c)] = True
        self.P, self.M = t(P), t(M, torch.bool)
        self.n_points = M.sum(1)
        # frame spacing, relative to nominal, for the temporal terms
        dt = np.diff(np.asarray(frame_t, dtype=np.float64))
        nom = np.median(dt) if len(dt) else 1 / 30
        self.dt_rel = t(np.clip(dt / nom, 0.5, 10.0))
        self.chunk = args.chunk
        # A fixed subset of vertices for the distance transform: the full 6890 at
        # 2000 points a frame is 55 MB per frame of pairwise distances, and the
        # nearest of every third vertex is within a centimetre of the true surface.
        g = torch.Generator(device='cpu').manual_seed(0)
        perm = torch.randperm(smpl.V, generator=g)
        self.vsub = perm[:args.vertices].to(device)
        # The reported residual is point-to-VERTEX, so it grows with sparser vertices
        # whatever the fit did; it is always measured against the same 2300 so the
        # fast and elaborate settings, and different --vertices, are comparable.
        self.vsub_eval = perm[:2300].to(device)

    def data_terms(self, sl, need_depth=True, need_2d=True):
        torch, a = self.torch, self.args
        verts, joints = self.smpl.forward(self.pose[sl], self.betas, self.trans[sl])
        loss = torch.zeros((), device=self.dev)
        B = verts.shape[0]
        if need_depth and a.w_depth > 0:
            P, M = self.P[sl], self.M[sl]
            d = torch.cdist(P, verts[:, self.vsub]).min(dim=2).values     # [B, n]
            per = (gm(d, a.sigma_depth) * M).sum(1) / M.sum(1).clamp_min(1)
            loss = loss + a.w_depth * per.sum()
        if need_2d and a.w_2d > 0:
            uv = joints[:, :, :2] / joints[:, :, 2:3].clamp_min(0.1)
            uv = uv * torch.stack([self.K[0, 0], self.K[1, 1]]) + torch.stack([self.K[0, 2], self.K[1, 2]])
            e = torch.linalg.norm(uv - self.j2d[sl], dim=-1)
            w = self.w2d[sl]
            per = (gm(e, a.sigma_2d) * w).sum(1) / w.sum(1).clamp_min(1e-6)
            loss = loss + a.w_2d * per.sum()
        if a.w_prior > 0:
            dp = ((self.pose[sl] - self.pose0[sl]) ** 2).sum(1)
            wp = torch.where(self.found[sl], 1.0, a.gap_prior)
            loss = loss + a.w_prior * (dp * wp).sum()
        return loss / self.F, verts.detach(), joints.detach()

    def sequence_terms(self):
        torch, a = self.torch, self.args
        _, J = self.smpl.forward(self.pose, self.betas, self.trans, return_vertices=False)
        loss = torch.zeros((), device=self.dev)
        if self.F > 1:
            v = (J[1:] - J[:-1]) / self.dt_rel[:, None, None]
            loss = loss + a.w_vel * (v ** 2).sum(-1).mean()
            pv = (self.pose[1:] - self.pose[:-1]) / self.dt_rel[:, None]
            loss = loss + a.w_pose_vel * (pv ** 2).sum(-1).mean()
        if self.F > 2:
            acc = v[1:] - v[:-1]
            loss = loss + a.w_acc * (acc ** 2).sum(-1).mean()
        loss = loss + a.w_beta * ((self.betas - self.betas0) ** 2).sum()
        return loss

    def run(self, params, iters, lrs, log=None):
        torch = self.torch
        groups = [dict(params=[p], lr=lr) for p, lr in zip(params, lrs)]
        opt = torch.optim.Adam(groups)
        last = None
        for it in range(iters):
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            for s in range(0, self.F, self.chunk):
                sl = slice(s, min(s + self.chunk, self.F))
                l, _, _ = self.data_terms(sl)
                l.backward()
                tot += l.item()
            l = self.sequence_terms()
            l.backward()
            tot += l.item()
            opt.step()
            last = tot
            if log and (it % 25 == 0 or it == iters - 1):
                log(f'    iter {it:4d}  loss {tot:.5f}')
        return last

    @property
    def no_grad(self):
        return self.torch.no_grad

    def residuals(self):
        """Per-frame diagnostics after the fit: median point-to-body distance (m),
        mean 2-D reprojection error of the observed joints (px), and the point
        cloud's nearest-vertex distances (for re-segmentation)."""
        torch = self.torch
        med, rep, near = [], [], []
        with torch.no_grad():
            for s in range(0, self.F, self.chunk):
                sl = slice(s, min(s + self.chunk, self.F))
                verts, joints = self.smpl.forward(self.pose[sl], self.betas, self.trans[sl])
                d = torch.cdist(self.P[sl], verts[:, self.vsub_eval]).min(dim=2).values
                d = torch.where(self.M[sl], d, torch.nan)
                med.append(torch.nanmedian(d, dim=1).values)
                near.append(d)
                uv = joints[:, :, :2] / joints[:, :, 2:3].clamp_min(0.1)
                uv = uv * torch.stack([self.K[0, 0], self.K[1, 1]]) + torch.stack([self.K[0, 2], self.K[1, 2]])
                e = torch.linalg.norm(uv - self.j2d[sl], dim=-1)
                w = (self.w2d[sl] > 0).float()
                rep.append((e * w).sum(1) / w.sum(1).clamp_min(1))
        return (torch.cat(med).cpu().numpy(), torch.cat(rep).cpu().numpy(),
                torch.cat(near))

    def resegment(self, near, radius):
        """Keep only the points within `radius` of the current body."""
        keep = self.M & (near <= radius)
        self.M = keep
        self.n_points = keep.sum(1).cpu().numpy()


# ------------------------------------------------------------------ per take

def fit_take(path, args, smpl, device, log=print):
    import torch
    nlf_path = path[:-4] + '_nlf.npz'
    if not os.path.exists(nlf_path):
        raise FileNotFoundError(f'{nlf_path} missing: run extract_poses3d.py first')
    with np.load(path) as d, np.load(nlf_path) as N, zipfile.ZipFile(path) as zf:
        meta = json.loads(str(d['meta'])) if 'meta' in d.files else {}
        colour, depth_members, _ = frame_members(d, meta)
        timing = {k: d[k] for k in ('frame_t', 'frame_t_ns', 'frame_idx', 'frame_seq')
                  if k in d.files}
        frame_t = d['frame_t'].astype(np.float64)
        calib = load_calib(meta, args.calib, verbose=False)
        K = K_of(calib['colour_intrinsics'])
        init = fill_gaps({k: N[k] for k in N.files if k != 'meta'})
        F = len(init['pose'])
        if len(frame_t) != F:
            raise ValueError(f'{path}: {len(frame_t)} frames but {F} NLF rows')
        trans_nlf = init['trans'].copy()

        # initial body per frame (for the depth band, the floor line and the
        # coarse distance), on the GPU in chunks
        with torch.no_grad():
            V0, J0 = [], []
            for s in range(0, F, args.chunk):
                v, j = smpl.forward(torch.as_tensor(init['pose'][s:s + args.chunk], device=device),
                                    torch.as_tensor(init['betas'], device=device),
                                    torch.as_tensor(init['trans'][s:s + args.chunk], device=device))
                V0.append(v.cpu().numpy()); J0.append(j.cpu().numpy())
            V0, J0 = np.concatenate(V0), np.concatenate(J0)

        rng = np.random.default_rng(0)
        clouds, scales = [], []
        n_depth_missing = 0
        for i in range(F):
            m = depth_members[i]
            if m is None:
                clouds.append(np.zeros((0, 3), np.float32))
                scales.append(1.0)
                n_depth_missing += 1
                continue
            depth = read_depth_png(zf, m)
            zc = V0[i][:, 2]
            y_floor = V0[i][:, 1].max() + args.floor_margin
            P = person_points(depth, calib, K, init['box'][i], zc.min() - args.band,
                              zc.max() + args.band, y_floor, args.points, rng)
            s = coarse_distance(P, V0[i], J0[i], K)
            scales.append(s)
            clouds.append(P)
        # smooth the per-frame distance correction (it is a property of the network's
        # size guess, which varies slowly) and apply it along the pelvis ray
        scales = np.asarray(scales)
        if F >= 5:
            k = min(15, F if F % 2 else F - 1)
            pad = np.pad(scales, k // 2, mode='edge')
            scales = np.array([np.median(pad[i:i + k]) for i in range(F)])
        pelvis = J0[:, 0]
        init['trans'] = (init['trans'] + (scales[:, None] - 1.0) * pelvis).astype(np.float32)
        # re-collect the clouds with the band around the corrected body
        if np.abs(scales - 1).max() > 0.05:
            V0c = V0 + ((scales[:, None] - 1.0) * pelvis)[:, None, :]
            for i in range(F):
                if depth_members[i] is None:
                    continue
                zc = V0c[i][:, 2]
                clouds[i] = person_points(read_depth_png(zf, depth_members[i]), calib, K,
                                          init['box'][i], zc.min() - args.band,
                                          zc.max() + args.band,
                                          V0c[i][:, 1].max() + args.floor_margin,
                                          args.points, rng)

    # 2-D observation weights from the network's uncertainty (mm): full weight up
    # to 30 mm, falling as 30/u beyond; undetected frames carry no 2-D term.
    unc = np.nan_to_num(init['joint_unc'], nan=1e6)
    w2d = np.clip(30.0 / np.maximum(unc, 30.0), 0.0, 1.0) * init['found'][:, None]
    j2d = np.nan_to_num(init['joints2d'], nan=0.0)

    fit = SequenceFit(smpl, init, clouds, j2d, w2d, K, frame_t, args, device)
    t0 = time.time()
    log(f'  stage 1: translation ({args.iters1} iters)')
    fit.run([fit.trans], args.iters1, [args.lr_trans], log if args.verbose else None)
    med, rep, near = fit.residuals()
    log(f'    depth median {1e3 * np.nanmedian(med):.1f} mm, reproj {np.nanmean(rep):.1f} px')
    fit.resegment(near, args.reseg_radius)
    log(f'  stage 2: pose + shape + translation ({args.iters2} iters), points within '
        f'{100 * args.reseg_radius:.0f} cm: median {int(np.median(fit.n_points))}/frame')
    fit.run([fit.pose, fit.betas, fit.trans], args.iters2,
            [args.lr_pose, args.lr_betas, args.lr_trans], log if args.verbose else None)
    med, rep, near = fit.residuals()
    log(f'    depth median {1e3 * np.nanmedian(med):.1f} mm, reproj {np.nanmean(rep):.1f} px, '
        f'{time.time() - t0:.0f}s')

    with torch.no_grad():
        pose = fit.pose.detach()
        betas = fit.betas.detach()
        trans = fit.trans.detach()
        J3, C3 = [], []
        for s in range(0, F, args.chunk):
            v, j = smpl.forward(pose[s:s + args.chunk], betas, trans[s:s + args.chunk])
            J3.append(j.cpu().numpy())
            C3.append(smpl.coco17(v, j).cpu().numpy())
        J3, C3 = np.concatenate(J3), np.concatenate(C3)
    C2 = project(K, C3).astype(np.float32)
    n_points = fit.n_points if isinstance(fit.n_points, np.ndarray) else fit.n_points
    n_points = np.asarray(n_points)
    valid = (init['found'] & (n_points >= args.min_points)
             & (np.nan_to_num(med, nan=1.0) <= args.max_depth_med))
    body_meta = dict(
        source_capture=os.path.basename(path), nlf=os.path.basename(nlf_path),
        units='metres, colour camera optical frame (x right, y down, z forward)',
        keypoint_names=COCO17, smpl_joints=24,
        calib_source=calib['source'], colour_intrinsics=calib['colour_intrinsics'],
        depth_to_colour=calib['depth_to_colour'],
        distance_scale_median=float(np.median(scales)),
        depth_frames_missing=int(n_depth_missing),
        weights={k: getattr(args, k) for k in ('w_depth', 'w_2d', 'w_prior', 'w_vel', 'w_acc',
                                               'w_pose_vel', 'w_beta', 'sigma_depth',
                                               'sigma_2d', 'reseg_radius')},
        iters=[args.iters1, args.iters2], vertices=args.vertices,
        mode='elaborate' if args.elaborate else 'fast',
        timestamp_origin=meta.get('timestamp_origin', 'record_start'),
        timestamp_unit=meta.get('timestamp_unit', 'nanoseconds'))
    out = path[:-4] + '_body.npz'
    tmp = out + '.tmp'
    kp = np.concatenate([C2, np.ones((F, 17, 1), np.float32)], axis=2)
    kp[~init['found']] = np.nan
    with open(tmp, 'wb') as fh:
        np.savez_compressed(
            fh, pose=pose.cpu().numpy(), betas=betas.cpu().numpy(), trans=trans.cpu().numpy(),
            joints3d=J3.astype(np.float32), keypoints3d=C3.astype(np.float32),
            keypoints2d=C2, keypoints=kp, found=init['found'], valid=valid,
            depth_med=np.nan_to_num(med, nan=np.nan).astype(np.float32),
            n_points=n_points.astype(np.int32), reproj_px=rep.astype(np.float32),
            trans_nlf=trans_nlf, distance_scale=scales.astype(np.float32),
            meta=np.array(json.dumps(body_meta)), **timing)
    os.replace(tmp, out)
    return dict(out=out, F=F, valid=float(valid.mean()), depth_med=float(np.nanmedian(med)),
                reproj=float(np.nanmean(rep)), scale=float(np.median(scales)),
                dz=float(np.median(trans.cpu().numpy()[:, 2] - trans_nlf[:, 2])))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', nargs='+', default=['data'])
    ap.add_argument('--calib', default=None, help='calib JSON from tools/rs_calib.py')
    ap.add_argument('--smpl', default=str(SMPL_NPZ))
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--plot', action='store_true', help='also write <take>_body.png')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--elaborate', action='store_true',
                    help='the slower, more thorough fit: 30 + 80 iterations over 2300 body '
                         'vertices instead of 20 + 50 over 1200; ~25 s a take instead of ~15. '
                         'Measured on one take: same residual (31 vs 32 mm), joints 8 mm apart '
                         'on average, up to ~18 mm at the knees and ankles where depth points '
                         'are sparse. The 2026-09-17 corpus was fitted with this setting.')
    g = ap.add_argument_group('point cloud')
    g.add_argument('--points', type=int, default=1500, help='depth points per frame')
    g.add_argument('--vertices', type=int, default=None,
                   help='body vertices in the distance term (default 1200, --elaborate 2300)')
    g.add_argument('--band', type=float, default=0.45,
                   help='m either side of the initial body depth range that points may lie in')
    g.add_argument('--floor-margin', type=float, default=0.04,
                   help='m below the lowest body vertex (camera y) beyond which points are floor')
    g.add_argument('--reseg-radius', type=float, default=0.15,
                   help='m from the stage-1 body within which points are kept for stage 2')
    g = ap.add_argument_group('weights')
    g.add_argument('--w-depth', type=float, default=1.0)
    g.add_argument('--w-2d', type=float, default=1.0)
    g.add_argument('--w-prior', type=float, default=0.05, help='per rad^2 of pose change')
    g.add_argument('--gap-prior', type=float, default=0.2,
                   help='prior weight factor on undetected (interpolated) frames')
    g.add_argument('--w-vel', type=float, default=20.0, help='per m^2/frame^2 joint velocity')
    g.add_argument('--w-acc', type=float, default=40.0, help='per m^2/frame^2 joint acceleration')
    g.add_argument('--w-pose-vel', type=float, default=0.5)
    g.add_argument('--w-beta', type=float, default=0.02)
    g.add_argument('--sigma-depth', type=float, default=0.03, help='m; GM scale of the depth term')
    g.add_argument('--sigma-2d', type=float, default=15.0, help='px; GM scale of the 2-D term')
    g = ap.add_argument_group('optimiser')
    g.add_argument('--iters1', type=int, default=None, help='default 20, --elaborate 30')
    g.add_argument('--iters2', type=int, default=None, help='default 50, --elaborate 80')
    g.add_argument('--lr-trans', type=float, default=0.01)
    g.add_argument('--lr-pose', type=float, default=0.004)
    g.add_argument('--lr-betas', type=float, default=0.002)
    g.add_argument('--chunk', type=int, default=48, help='frames per GPU chunk')
    g = ap.add_argument_group('validity')
    g.add_argument('--min-points', type=int, default=200)
    g.add_argument('--max-depth-med', type=float, default=0.05, help='m')
    args = ap.parse_args()
    # explicit --iters1/--iters2/--vertices win over either mode
    full = dict(iters1=30, iters2=80, vertices=2300)
    fast = dict(iters1=20, iters2=50, vertices=1200)
    for k, v in (full if args.elaborate else fast).items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    smpl = SMPL(args.smpl, device=device)
    takes = find_takes(args.src)
    if not takes:
        sys.exit(f'no captures under {args.src}')
    print(f'{len(takes)} takes on {device}, {"elaborate" if args.elaborate else "fast"} mode '
          f'({args.iters1} + {args.iters2} iterations, {args.vertices} vertices)', flush=True)
    t0 = time.time()
    rows = []
    for n, path in enumerate(takes, 1):
        out = path[:-4] + '_body.npz'
        if os.path.exists(out) and not args.overwrite:
            continue
        print(f'[{n}/{len(takes)}] {os.path.relpath(path)}', flush=True)
        try:
            r = fit_take(path, args, smpl, device)
        except FileNotFoundError as e:
            print(f'  skipped: {e}', flush=True)
            continue
        rows.append((os.path.basename(path)[:-4], r))
        print(f'  -> valid {100 * r["valid"]:.1f}%  depth {1e3 * r["depth_med"]:.1f} mm  '
              f'reproj {r["reproj"]:.1f} px  distance x{r["scale"]:.3f} '
              f'(dz {100 * r["dz"]:+.1f} cm vs RGB-only)  [{time.time() - t0:.0f}s]', flush=True)
        if args.plot:
            from plot_body_fit import plot_take
            plot_take(path, out, out[:-4] + '.png')
    if rows:
        v = np.array([r['valid'] for _, r in rows])
        dm = np.array([r['depth_med'] for _, r in rows])
        dz = np.array([r['dz'] for _, r in rows])
        print(f'\n{len(rows)} takes fitted: valid frames {100 * v.mean():.1f}%, depth median '
              f'{1e3 * np.median(dm):.1f} mm, RGB-only distance was off by '
              f'{100 * np.median(np.abs(dz)):.1f} cm (median |dz|), worst take '
              f'{1e3 * dm.max():.1f} mm ({rows[int(dm.argmax())][0]})')


if __name__ == '__main__':
    main()
