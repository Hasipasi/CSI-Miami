#!/usr/bin/env python3
"""Dump the RealSense's factory calibration for the stream modes the rig records
(1280x720 colour, 640x480 depth): colour intrinsics, depth intrinsics, the
depth-to-colour extrinsics, and the depth scale, into calib/realsense_<serial>.json.

Why it exists: the 3-D pose fit projects the body with the COLOUR intrinsics and
moves depth points into the colour frame with the extrinsics, and takes recorded
before 2026-09-17 carry neither in their meta (only the depth intrinsics). Without
this file those takes fall back to nominal D435i numbers, which are close but not
measured. Run it once per camera, on the rig, with the camera plugged in and
nothing else using it (stop the viewer / depth_server first); on a Mac run it
with sudo like depth_server.py. The file is small and belongs in git.

  python3 tools/rs_calib.py                 # writes calib/realsense_<serial>.json
  python3 tools/rs_calib.py --print         # show without writing
"""

import argparse
import json
import os
import pathlib
import sys


def intr_dict(i, w, h):
    return dict(width=int(w), height=int(h), fx=float(i.fx), fy=float(i.fy),
                ppx=float(i.ppx), ppy=float(i.ppy), model=str(i.model),
                coeffs=[float(c) for c in i.coeffs])


def read_calib(w=1280, h=720, dw=640, dh=480, fps=30):
    import pyrealsense2 as rs
    ctx = rs.context()
    if len(ctx.query_devices()) == 0:
        raise RuntimeError('no RealSense device found')
    pipe = rs.pipeline(ctx)
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, fps)
    prof = pipe.start(cfg)
    try:
        dev = prof.get_device()
        cs = prof.get_stream(rs.stream.color).as_video_stream_profile()
        ds = prof.get_stream(rs.stream.depth).as_video_stream_profile()
        ex = ds.get_extrinsics_to(cs)
        out = dict(
            serial=dev.get_info(rs.camera_info.serial_number),
            name=dev.get_info(rs.camera_info.name),
            firmware=dev.get_info(rs.camera_info.firmware_version),
            source='librealsense get_intrinsics / get_extrinsics_to',
            colour_intrinsics=intr_dict(cs.get_intrinsics(), cs.width(), cs.height()),
            depth_intrinsics=intr_dict(ds.get_intrinsics(), ds.width(), ds.height()),
            # rotation as librealsense stores it (column-major, 9 floats) and the
            # translation in metres; body_common.depth_to_points applies them the
            # way rs2_transform_point_to_point does.
            depth_to_colour=dict(rotation=[float(r) for r in ex.rotation],
                                 translation=[float(t) for t in ex.translation]),
            depth_scale_m=float(dev.first_depth_sensor().get_depth_scale()))
    finally:
        pipe.stop()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--print', action='store_true', help='print only, do not write')
    ap.add_argument('--out-dir', default=str(pathlib.Path(__file__).resolve().parents[1] / 'calib'))
    args = ap.parse_args()
    c = read_calib()
    print(json.dumps(c, indent=2))
    if args.print:
        return
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f'realsense_{c["serial"]}.json')
    with open(path, 'w') as fh:
        json.dump(c, fh, indent=2)
        fh.write('\n')
    print(f'\nwrote {path}', file=sys.stderr)


if __name__ == '__main__':
    main()
