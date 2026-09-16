#!/usr/bin/env python3
"""Write takes the viewer had to keep as raw pickles (<take>_raw.pkl beside their
frame folders) because writing them failed. Run after the cause is fixed:

  python3 rewrite_raw.py /Volumes/GergoDisk/data/<session>

Each pickle holds exactly what the writer would have been given; the frame
folders must still be there. The pickle is removed once the NPZ is written.
"""

import glob
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture import write_capture   # noqa: E402


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    raws = sorted(glob.glob(os.path.join(root, '*_raw.pkl')))
    if not raws:
        print(f'no *_raw.pkl in {root}')
        return
    for raw in raws:
        with open(raw, 'rb') as fh:
            r = pickle.load(fh)
        path, report, ft = write_capture(r['prefix'], r['recs'], r['frames'], r['t0'],
                                         r['meta'], r.get('own'),
                                         depth_frames=r.get('depth_frames'))
        os.unlink(raw)
        print(f'wrote {path} ({len(ft)} frames)')


if __name__ == '__main__':
    main()
