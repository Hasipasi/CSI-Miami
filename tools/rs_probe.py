#!/usr/bin/env python3
"""Open the RealSense through librealsense one step at a time and say where it
breaks. Run it the way the viewer will run (on a Mac: with sudo), and paste the
output; -X faulthandler makes a crash print the Python stack that led to it, and
the library's own debug log shows the last backend call before it.

  sudo .venv_mac/bin/python -X faulthandler tools/rs_probe.py
  sudo .venv_mac/bin/python -X faulthandler tools/rs_probe.py --quiet   # no debug log
  sudo .venv_mac/bin/python -X faulthandler tools/rs_probe.py --w 640 --h 480
"""

import argparse
import faulthandler
import os
import sys
import time

faulthandler.enable()


def say(*a):
    print(*a, flush=True)


def stream(rs, ctx, streams, seconds=2.0):
    pipe = rs.pipeline(ctx)
    cfg = rs.config()
    for kind, w, h, fmt, fps in streams:
        cfg.enable_stream(kind, w, h, fmt, fps)
    say(f'  start {[s[0] for s in streams]} ...')
    prof = pipe.start(cfg)
    say('  started')
    n = 0
    t0 = time.time()
    while time.time() - t0 < seconds:
        fs = pipe.wait_for_frames(3000)
        n += 1
        if n == 1:
            for f in fs:
                p = f.get_profile()
                say(f'  first frame: {p.stream_name()} {f.get_frame_number()} '
                    f'ts {f.get_timestamp():.1f} domain {f.get_frame_timestamp_domain()}')
    say(f'  {n} framesets in {seconds:g} s')
    pipe.stop()
    say('  stopped')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--w', type=int, default=1280)
    ap.add_argument('--h', type=int, default=720)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()
    say(f'python {sys.version.split()[0]}, euid {os.geteuid()}, HOME {os.environ.get("HOME")}')
    import pyrealsense2 as rs
    say('pyrealsense2 imported')
    rs.log_to_console(rs.log_severity.warn if args.quiet else rs.log_severity.debug)
    say('step: context')
    ctx = rs.context()
    say('step: query_devices')
    devs = ctx.query_devices()
    say(f'step: {len(devs)} device(s) listed')
    if not devs:
        return
    dev = devs[0]
    for key in ('name', 'serial_number', 'firmware_version', 'physical_port',
                'product_id', 'usb_type_descriptor'):
        say(f'step: get_info {key}')
        try:
            say(f'  {key} = {dev.get_info(getattr(rs.camera_info, key))}')
        except RuntimeError as e:
            say(f'  {key}: not available ({e})')
    say('step: query_sensors')
    for s in dev.query_sensors():
        say('  sensor', s.get_info(rs.camera_info.name))
    steps = [('depth only', [(rs.stream.depth, 640, 480, rs.format.z16, args.fps)])]
    if sys.platform != 'darwin':
        # On macOS the kernel camera driver holds the colour interface and nothing
        # can take it; asking makes the camera re-enumerate badly. Depth only there.
        steps += [
            ('colour only', [(rs.stream.color, args.w, args.h, rs.format.bgr8, args.fps)]),
            ('colour + depth', [(rs.stream.color, args.w, args.h, rs.format.bgr8, args.fps),
                                (rs.stream.depth, 640, 480, rs.format.z16, args.fps)]),
        ]
    for label, streams in steps:
        say(f'== {label}')
        try:
            stream(rs, ctx, streams)
        except RuntimeError as e:
            say(f'  failed: {e}')
    say('done')


if __name__ == '__main__':
    main()
