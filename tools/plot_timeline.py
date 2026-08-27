#!/usr/bin/env python3
"""Plot camera-frame and CSI packet overlap from one capture.

The upper raster draws every timestamp as a tick on the shared recording clock.
The lower map counts CSI packets in each interval between consecutive camera
timestamps.  A combined row shows overall density across all receiver links,
which is especially important for round-robin captures where each individual link
is intentionally sparse.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capture', help='self-contained .npz capture')
    ap.add_argument('-o', '--output', default=None)
    args = ap.parse_args()

    with np.load(args.capture) as d:
        meta = json.loads(str(d['meta']))
        boards = meta.get('boards', {})
        if 'frame_t_ns' in d.files:
            frame_t = d['frame_t_ns'].astype(np.float64) / 1e9
            suffix, scale = '|t_ns', 1e9
        else:
            frame_t = d['frame_t'].astype(np.float64)
            suffix, scale = '|t', 1.0
        links = sorted({k[:-len(suffix)] for k in d.files if k.endswith(suffix)},
                       key=lambda lk: tuple(boards.get(m, m[-5:])
                                            for m in lk.split('|')))
        packet_t = {lk: d[lk + suffix].astype(np.float64) / scale for lk in links}

    link_labels = []
    link_counts = np.zeros((len(links), max(len(frame_t) - 1, 0)), dtype=np.int32)
    for row, lk in enumerate(links):
        tx, rx = lk.split('|')
        label = f'{boards.get(tx, tx[-5:])}→{boards.get(rx, rx[-5:])}'
        if len(frame_t) > 1:
            pos = np.searchsorted(frame_t, packet_t[lk], side='right') - 1
            pos = pos[(pos >= 0) & (pos < len(frame_t) - 1)]
            link_counts[row] = np.bincount(pos, minlength=len(frame_t) - 1)
            cov = 100.0 * np.count_nonzero(link_counts[row]) / len(link_counts[row])
        else:
            cov = 0.0
        link_labels.append(f'{label}  ({cov:.0f}% covered)')

    all_packet_t = (np.sort(np.concatenate(list(packet_t.values())))
                    if packet_t else np.array([], dtype=np.float64))
    combined_counts = (link_counts.sum(axis=0) if len(links)
                       else np.zeros(max(len(frame_t) - 1, 0), dtype=np.int32))
    combined_cov = (100.0 * np.count_nonzero(combined_counts) / len(combined_counts)
                    if len(combined_counts) else 0.0)
    labels = [f'ALL CSI combined  ({combined_cov:.0f}% covered)'] + link_labels
    counts = np.vstack([combined_counts, link_counts])

    n_packets = sum(len(v) for v in packet_t.values())
    fig, (ax, heat) = plt.subplots(
        2, 1, figsize=(18, 10), sharex=True,
        gridspec_kw={'height_ratios': [3, 2], 'hspace': 0.12})

    # Full-height camera lines expose exactly which packets land between frames.
    for t in frame_t:
        ax.axvline(t, color='#a7adb8', linewidth=0.35, alpha=0.38, zorder=0)
    ax.eventplot([frame_t], lineoffsets=[0], linelengths=0.72,
                 linewidths=0.9, colors=['#111827'])
    ax.eventplot([all_packet_t], lineoffsets=[1], linelengths=0.78,
                 linewidths=0.75, colors=['#1d4ed8'])
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(links), 1)))
    for row, lk in enumerate(links, 2):
        ax.eventplot([packet_t[lk]], lineoffsets=[row], linelengths=0.72,
                     linewidths=0.65, colors=[colors[row - 2]])
    ax.set_yticks(np.arange(len(links) + 2))
    ax.set_yticklabels(['camera frames'] + labels)
    ax.set_ylim(len(links) + 1.7, -0.7)
    ax.set_ylabel('received stream')
    ax.grid(axis='x', alpha=0.15)
    ax.set_title(
        f'{os.path.basename(args.capture)} — {len(frame_t)} camera frames, '
        f'{n_packets:,} received CSI packets\n'
        f'Combined CSI covers {combined_cov:.1f}% of camera intervals; '
        'timestamps share recording-start zero')

    if len(frame_t) > 1 and links:
        mesh = heat.pcolormesh(frame_t, np.arange(len(labels) + 1), counts,
                               shading='flat', cmap='viridis', vmin=0)
        heat.set_yticks(np.arange(len(labels)) + 0.5)
        heat.set_yticklabels(labels)
        heat.set_ylim(len(labels), 0)
        cb = fig.colorbar(mesh, ax=heat, pad=0.012)
        cb.set_label('CSI packets between consecutive camera frames')
    heat.set_xlabel('seconds from recording start')
    heat.set_ylabel('packets per camera interval')

    output = args.output or os.path.splitext(args.capture)[0] + '_timeline.png'
    fig.subplots_adjust(left=0.18, right=0.96, top=0.91, bottom=0.08)
    fig.savefig(output, dpi=170)
    print(output)


if __name__ == '__main__':
    main()
