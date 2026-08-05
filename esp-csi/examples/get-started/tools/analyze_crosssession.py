#!/usr/bin/env python3
"""Train pose classification on one session, test on another.

Within-session accuracy is easy to obtain and easy to over-read: a model can key
on the multipath of one particular standing spot rather than on the pose itself.
Training and testing on separate recordings -- ideally with the subject standing
slightly differently -- is what separates "recognises poses" from "recognises
this session".

Everything (feature scaling, PCA basis, class centroids) is fitted on the
training session only and then frozen, so nothing about the test session leaks
into the model.

Two variants are reported:

  train-stats      test data scaled with the TRAINING session's statistics.
                   Strict, but a constant session offset alone will hurt it.
  per-session      each session centred on its own mean first. Still no labels
                   used, and it is what a deployed system could legitimately do
                   after a short unlabelled warm-up. The gap between the two
                   isolates a simple offset from a real change in signature.

  python3 analyze_crosssession.py --train ps --test ps2
"""
import argparse
import glob
import os

import numpy as np

POSE_ORDER = ['empty', 'neutral', 'tpose', 'up', 'split', 'crouch', 'turned']


def links(d):
    return sorted({k.rsplit('|', 1)[0] for k in d.files if k.endswith('|t')})


def windows(d, keep, win=0.5):
    lks = [l for l in keep if f'{l}|t' in d.files and len(d[f'{l}|t']) >= 8]
    if len(lks) != len(keep):
        return None
    n = int(min(d[f'{l}|t'][-1] for l in lks) / win)
    if n < 3:
        return None
    cols = []
    for l in lks:                       # keep order fixed across sessions
        t, a = d[f'{l}|t'], d[f'{l}|a'].astype(float)
        blk = np.zeros((n, a.shape[1]))
        for i in range(n):
            m = (t >= i * win) & (t < (i + 1) * win)
            blk[i] = a[m].mean(0) if m.sum() else (blk[i - 1] if i else a[:5].mean(0))
        cols.append(blk)
    return np.hstack(cols)


def build(prefix, keep, classes):
    X, y = [], []
    for ci, c in enumerate(classes):
        p = f'{prefix}_{c}.npz'
        if not os.path.exists(p):
            return None, None
        w = windows(np.load(p), keep)
        if w is None:
            return None, None
        X.append(w)
        y += [ci] * len(w)
    k = min(x.shape[1] for x in X)
    return np.vstack([x[:, :k] for x in X]), np.array(y)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--npc', type=int, default=20)
    args = ap.parse_args()

    classes = [c for c in POSE_ORDER
               if os.path.exists(f'{args.train}_{c}.npz') and os.path.exists(f'{args.test}_{c}.npz')]
    ltr = set(links(np.load(f'{args.train}_{classes[0]}.npz')))
    lte = set(links(np.load(f'{args.test}_{classes[0]}.npz')))
    keep = sorted(ltr & lte)
    print(f'classes: {classes}')
    print(f'links common to both sessions: {len(keep)}')

    Xtr, ytr = build(args.train, keep, classes)
    Xte, yte = build(args.test, keep, classes)
    if Xtr is None or Xte is None:
        print('could not build features (missing file or too-short capture)')
        return
    k = min(Xtr.shape[1], Xte.shape[1])
    Xtr, Xte = Xtr[:, :k], Xte[:, :k]
    print(f'train {Xtr.shape[0]} windows, test {Xte.shape[0]} windows, {k} features\n')

    chance = 100.0 / len(classes)

    def run(tag, A, B):
        mu, sd = A.mean(0), A.std(0) + 1e-9      # fitted on TRAIN only
        Az, Bz = (A - mu) / sd, (B - mu) / sd
        Vt = np.linalg.svd(Az - Az.mean(0), full_matrices=False)[2]
        npc = min(args.npc, len(Az) - 1)
        P = Vt[:npc].T
        Ztr, Zte = (Az - Az.mean(0)) @ P, (Bz - Az.mean(0)) @ P
        cents = np.stack([Ztr[ytr == c].mean(0) for c in range(len(classes))])
        pred = ((Zte[:, None, :] - cents[None]) ** 2).sum(-1).argmin(1)
        acc = (pred == yte).mean()
        print(f'{tag:16s} {acc * 100:5.1f}%   (chance {chance:.1f}%)')
        return pred

    print('CROSS-SESSION  (model fitted on training session only)')
    pred = run('train-stats', Xtr, Xte)
    # per-session centring: remove each session's own mean before anything else
    pred2 = run('per-session', Xtr - Xtr.mean(0), Xte - Xte.mean(0))

    print('\nconfusion, per-session variant (rows = true):')
    C = np.zeros((len(classes), len(classes)), int)
    for t, p_ in zip(yte, pred2):
        C[t, p_] += 1
    print('            ' + ''.join(f'{c[:7]:>8}' for c in classes))
    for i, c in enumerate(classes):
        print(f'  {c[:10]:10s}' + ''.join(f'{v:8d}' for v in C[i]))

    print('\nfor reference, within-TEST-session accuracy (temporal split):')
    mu, sd = Xte.mean(0), Xte.std(0) + 1e-9
    Bz = (Xte - mu) / sd
    Vt = np.linalg.svd(Bz - Bz.mean(0), full_matrices=False)[2]
    Z = (Bz - Bz.mean(0)) @ Vt[:min(args.npc, len(Bz) - 1)].T
    tr, te = [], []
    for c in range(len(classes)):
        idx = np.where(yte == c)[0]
        h = len(idx) // 2
        tr += list(idx[:h])
        te += list(idx[h:])
    tr, te = np.array(tr), np.array(te)
    cents = np.stack([Z[tr][yte[tr] == c].mean(0) for c in range(len(classes))])
    pred3 = ((Z[te][:, None, :] - cents[None]) ** 2).sum(-1).argmin(1)
    print(f'  {(pred3 == yte[te]).mean() * 100:5.1f}%  -- the gap against cross-session is'
          ' the generalisation cost')


if __name__ == '__main__':
    main()
