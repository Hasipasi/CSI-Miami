#!/usr/bin/env python3
"""Can distinct held poses be told apart from CSI alone?

This is the question that a single motion sequence cannot answer. Rank measured
on one movement reflects how complex that movement was, not what the sensor can
resolve. Held poses give a direct answer: build a feature vector per time window
and see whether a classifier separates them.

Two accuracies are reported deliberately:

  leave-one-out   optimistic. Adjacent windows within a pose are correlated, so
                  a held-out window usually has a near-twin in the training set.
  temporal split  honest. Train on the first half of each hold, test on the
                  second half, so train and test never share a moment in time.

  python3 analyze_poses.py ps_empty.npz ps_neutral.npz ...
"""
import sys
import numpy as np


def links(d):
    return sorted({k.rsplit('|', 1)[0] for k in d.files if k.endswith('|t')})


def windows(d, win=0.5):
    """[n_windows x (links*subcarriers)] of mean amplitude, on a common time grid."""
    lks = [l for l in links(d) if len(d[f'{l}|t']) >= 10]
    if not lks:
        return None
    t_end = min(d[f'{l}|t'][-1] for l in lks)
    n = int(t_end / win)
    if n < 4:
        return None
    cols = []
    for l in lks:
        t, a = d[f'{l}|t'], d[f'{l}|a'].astype(float)
        blk = np.zeros((n, a.shape[1]))
        for i in range(n):
            m = (t >= i * win) & (t < (i + 1) * win)
            blk[i] = a[m].mean(0) if m.sum() else (blk[i - 1] if i else a[:5].mean(0))
        cols.append(blk)
    return np.hstack(cols)


def main():
    paths = sys.argv[1:]
    if len(paths) < 3:
        print(__doc__)
        sys.exit(1)
    names = [p.split('_', 1)[1].replace('.npz', '') for p in paths]

    X, y, order = [], [], []
    for ci, (p, nm) in enumerate(zip(paths, names)):
        w = windows(np.load(p))
        if w is None:
            print(f'  skip {nm}: too few samples')
            continue
        X.append(w)
        y += [len(order)] * len(w)
        order.append(nm)
    k = min(x.shape[1] for x in X)
    X = np.vstack([x[:, :k] for x in X])
    y = np.array(y)
    print(f'{len(order)} classes {order}')
    print(f'{X.shape[0]} windows x {X.shape[1]} features\n')

    # standardise, then project to a modest subspace: with ~20 windows per class
    # and thousands of features the raw space is hopelessly under-determined.
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    U, S, Vt = np.linalg.svd(Xz - Xz.mean(0), full_matrices=False)
    npc = min(20, len(Xz) - 1)
    Z = (Xz - Xz.mean(0)) @ Vt[:npc].T

    def nearest_centroid(tr_idx, te_idx):
        cents = np.stack([Z[tr_idx][y[tr_idx] == c].mean(0) for c in range(len(order))])
        d = ((Z[te_idx][:, None, :] - cents[None]) ** 2).sum(-1)
        return d.argmin(1)

    # leave-one-out
    correct = 0
    for i in range(len(Z)):
        tr = np.setdiff1d(np.arange(len(Z)), [i])
        correct += int(nearest_centroid(tr, [i])[0] == y[i])
    loo = correct / len(Z)

    # temporal split: first half of each hold trains, second half tests
    tr, te = [], []
    for c in range(len(order)):
        idx = np.where(y == c)[0]
        h = len(idx) // 2
        tr += list(idx[:h])
        te += list(idx[h:])
    pred = nearest_centroid(np.array(tr), np.array(te))
    tmp = (pred == y[te]).mean()

    chance = 1.0 / len(order)
    print(f'chance                 {chance * 100:5.1f}%')
    print(f'leave-one-out          {loo * 100:5.1f}%   (optimistic: neighbouring windows correlate)')
    print(f'temporal split         {tmp * 100:5.1f}%   (honest: train and test disjoint in time)')

    print('\nconfusion (temporal split), rows = true:')
    C = np.zeros((len(order), len(order)), int)
    for t, p_ in zip(y[te], pred):
        C[t, p_] += 1
    print('            ' + ''.join(f'{o[:7]:>8}' for o in order))
    for i, o in enumerate(order):
        print(f'  {o[:10]:10s}' + ''.join(f'{v:8d}' for v in C[i]))

    print('\npairwise separability (centroid distance / within-class scatter):')
    cents = np.stack([Z[y == c].mean(0) for c in range(len(order))])
    sd = np.mean([Z[y == c].std(0).mean() for c in range(len(order))])
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            r = np.linalg.norm(cents[i] - cents[j]) / max(sd, 1e-9)
            if r < 3:
                print(f'  {order[i]:9s} vs {order[j]:9s} {r:6.1f}  <-- weak')
    print(f'  (median over all pairs: '
          f'{np.median([np.linalg.norm(cents[i] - cents[j]) / max(sd, 1e-9) for i in range(len(order)) for j in range(i + 1, len(order))]):.1f})')


if __name__ == '__main__':
    main()
