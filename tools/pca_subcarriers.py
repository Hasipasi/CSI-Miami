#!/usr/bin/env python3
"""How many independent degrees of freedom are there across the subcarriers?

192 subcarriers is not 192 numbers of information: neighbouring subcarriers see a
correlated channel, and the ESP32 reports three partly redundant estimates of the
same band. This measures the effective dimensionality by eigendecomposing the
subcarrier covariance, which bounds how much a model can possibly extract from the
frequency axis and says how far the input could be compressed for free.

Two decisions worth stating, because they change the answer:

* Only mask-true samples are used. A held repeat is the same measurement twice; in
  round-robin (32% coverage) counting them would triple-weight whichever moments
  happened to be sampled and understate the true spread.
* Correlation, not covariance, is the default. The three 64-wide CSI fields differ
  ~5x in gain, so a covariance PCA mostly reports which field a subcarrier is in.

  python3 pca_subcarriers.py dataset/csi5act_rr_v2 dataset/csi5act_txB_v1
"""

import argparse
import glob
import json
import os
import sys

import numpy as np


def accumulate(root, use_mask=True, per_link=False):
    """Streamed mean/covariance over subcarriers, so nothing large is held at once."""
    man = json.loads(open(os.path.join(root, 'manifest.json')).read())
    V = np.array(man['valid_subcarriers'])
    S, L = len(V), man['n_links']
    n = 0
    s1 = np.zeros(S)
    s2 = np.zeros((S, S))
    per = [dict(n=0, s1=np.zeros(S), s2=np.zeros((S, S))) for _ in range(L)]

    for p in sorted(glob.glob(os.path.join(root, 'clips', '*.npz'))):
        z = np.load(p)
        x = z['csi'][:, :, V].astype(np.float64)          # [T, L, S]
        m = z['mask'] if use_mask else np.ones(x.shape[:2], bool)
        for li in range(L):
            xi = x[m[:, li], li, :]
            if not len(xi):
                continue
            n += len(xi); s1 += xi.sum(0); s2 += xi.T @ xi
            if per_link:
                per[li]['n'] += len(xi)
                per[li]['s1'] += xi.sum(0)
                per[li]['s2'] += xi.T @ xi
    return man, V, n, s1, s2, per


def eig(n, s1, s2, standardise=True):
    mu = s1 / n
    cov = s2 / n - np.outer(mu, mu)
    if standardise:
        sd = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        cov = cov / np.outer(sd, sd)
    w = np.linalg.eigvalsh(cov)[::-1]
    w = np.maximum(w, 0)
    return w / w.sum()


def report(name, ev):
    cum = np.cumsum(ev)
    ks = [int(np.searchsorted(cum, q) + 1) for q in (0.90, 0.95, 0.99)]
    # participation ratio: a basis-free "effective number of dimensions"
    pr = 1.0 / np.sum(ev ** 2)
    print(f'{name:26s} {100*ev[0]:6.1f}% {100*ev[1]:6.1f}% {100*ev[2]:6.1f}% '
          f'{100*cum[4]:7.1f}% {100*cum[9]:7.1f}% {ks[0]:4d} {ks[1]:4d} {ks[2]:4d} {pr:7.1f}')
    return ev


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('roots', nargs='+')
    ap.add_argument('--no-mask', action='store_true',
                    help='include held repeats (shows how much the mask matters)')
    ap.add_argument('--raw', action='store_true', help='covariance instead of correlation')
    ap.add_argument('--per-link', action='store_true')
    ap.add_argument('--plot', default=None)
    args = ap.parse_args()

    print(f'PCA over subcarriers  ({"covariance" if args.raw else "correlation"}, '
          f'{"all samples" if args.no_mask else "mask-true samples only"})\n')
    print(f'{"dataset":26s} {"PC1":>6s} {"PC2":>6s} {"PC3":>6s} {"cum@5":>7s} {"cum@10":>7s} '
          f'{"n90":>4s} {"n95":>4s} {"n99":>4s} {"eff-d":>7s}')
    curves = {}
    for root in args.roots:
        man, V, n, s1, s2, per = accumulate(root, use_mask=not args.no_mask,
                                            per_link=args.per_link)
        ev = eig(n, s1, s2, standardise=not args.raw)
        curves[os.path.basename(root)] = (ev, man, n, len(V))
        report(os.path.basename(root), ev)

    for name, (ev, man, n, S) in curves.items():
        print(f'\n{name}: {n:,} samples over {S} valid subcarriers, '
              f'{man["n_links"]} links @ {man["grid_hz"]:.0f} Hz')

    if args.per_link:
        print('\nper-link effective dimensionality (participation ratio):')
        for root in args.roots:
            man, V, n, s1, s2, per = accumulate(root, use_mask=not args.no_mask,
                                                per_link=True)
            prs = []
            for li, d in enumerate(per):
                if d['n'] < 100:
                    continue
                e = eig(d['n'], d['s1'], d['s2'], standardise=not args.raw)
                prs.append((man['links'][li], 1.0 / np.sum(e ** 2)))
            print(f'  {os.path.basename(root)}: ' +
                  '  '.join(f'{k} {v:.1f}' for k, v in prs))

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        SURFACE, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#8b8a86', '#e1e0d9'
        COL = ['#2a78d6', '#eb6834']
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 4.9), facecolor=SURFACE)
        for (name, (ev, man, n, S)), c in zip(curves.items(), COL):
            lab = f'{name}  ({man["n_links"]} links)'
            a1.plot(np.arange(1, 21), 100 * ev[:20], 'o-', color=c, lw=2, ms=5, label=lab)
            a2.plot(np.arange(1, S + 1), 100 * np.cumsum(ev), color=c, lw=2.2, label=lab)
        for ax in (a1, a2):
            ax.set_facecolor(SURFACE)
            ax.grid(color=GRID, lw=1)
            ax.set_axisbelow(True)
            ax.tick_params(colors=MUTED, length=0, labelsize=9)
            for s in ax.spines.values():
                s.set_visible(False)
            leg = ax.legend(frameon=False, fontsize=9.5)
            for t in leg.get_texts():
                t.set_color(INK)
        a1.set_yscale('log')
        a1.set_xlabel('component', color=MUTED, fontsize=10)
        a1.set_ylabel('variance explained (%, log)', color=MUTED, fontsize=10)
        a1.set_title('Scree — how fast the subcarrier axis collapses',
                     color=INK, fontsize=12, loc='left', fontweight='bold')
        for q, ls in ((90, (0, (4, 3))), (95, (0, (1, 2)))):
            a2.axhline(q, color=MUTED, lw=1, ls=ls)
            a2.text(S, q, f' {q}%', color=MUTED, fontsize=8.5, va='center')
        a2.set_xlabel('components kept', color=MUTED, fontsize=10)
        a2.set_ylabel('cumulative variance (%)', color=MUTED, fontsize=10)
        a2.set_xlim(1, S)
        a2.set_title('Cumulative — how few carry the signal',
                     color=INK, fontsize=12, loc='left', fontweight='bold')
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150, facecolor=SURFACE)
        print(f'\nwrote {args.plot}')


if __name__ == '__main__':
    main()
