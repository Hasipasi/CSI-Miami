#!/usr/bin/env python3
"""Serve the RealSense colour AND depth streams to the viewer and the recorder over a
local socket, so that only this small process has to run as root.

On macOS librealsense can only open the camera as root, and the moment it does the
OS camera stack loses the camera entirely (its colour interface disappears from
AVFoundation until a replug), so colour has to come through librealsense as well.
A GUI run under sudo loses the camera permission of its terminal and writes
root-owned files, hence this split: start this once in its own terminal, from a
freshly plugged-in camera, and leave it running,

  sudo .venv_mac/bin/python tools/depth_server.py

then run the viewer or capture.py normally in another; they find the socket
(/tmp/csi-depth.sock) and take colour and depth from it, no flag needed.

Protocol, one connection per client: a 4-byte length and a JSON header (name,
colour w/h/fps, depth w/h/scale/intrinsics), then per frame a 17-byte header --
timestamp as a double (librealsense global time, i.e. host wall time), payload
length, sequence number, stream kind (0 colour BGR8, 1 depth uint16) -- followed
by the raw frame. Only the newest frame of each kind is ever sent; a slow client
misses frames rather than delaying the camera.

  --fake   serve synthetic colour and depth instead of the camera, for testing
"""

import argparse
import json
import os
import signal
import socket
import struct
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture import DEPTH_SOCKET, RealSenseCamera   # noqa: E402


class FakeCamera:
    """Synthetic colour (moving bars) and depth (ramp with a hole), the shape of
    RealSenseCamera: read() -> (seq, wall time, BGR) and .depth = (uint16, time)."""

    def __init__(self, w=1280, h=720, fps=30):
        self.w, self.h, self.fps = w, h, float(fps)
        self.depth_scale, self.depth_w, self.depth_h = 0.001, 640, 480
        self.depth_intrinsics = dict(fx=380.0, fy=380.0, ppx=320.0, ppy=240.0,
                                     model='fake', coeffs=[0.0] * 5)
        self.name = 'fake RealSense'
        self.wall_ts, self.monotonic, self.has_depth = True, False, True
        self.depth, self.seq = None, 0
        self._yy, self._xx = np.mgrid[0:480, 0:640]

    def start(self):
        pass

    def read(self, timeout=1.0):
        time.sleep(1.0 / self.fps)
        t = time.time()
        col = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        x = int((t * 200) % self.w)
        col[:, :, 0] = 40
        col[:, x:x + 80, 2] = 220
        col[:, (x + 300) % self.w:(x + 380) % self.w, 1] = 220
        d = (1000 + 3000 * self._xx / 640 + 200 * np.sin(t * 3 + self._yy / 40)).astype(np.uint16)
        d[200:280, 300:340] = 0
        self.depth = (d, t)
        self.seq += 1
        return self.seq, t, col

    def close(self):
        pass


class Source:
    """Pulls frames from the camera in one thread; clients read the newest."""

    def __init__(self, cam):
        self.cam = cam
        self.color = None          # (seq, ts, bgr)
        self.depth = None          # (array, ts)
        self.colour_frames = self.depth_frames = 0
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.is_set():
            got = self.cam.read()
            if got is None:
                continue
            self.color = got
            self.colour_frames += 1
            d = self.cam.depth
            if d is not None and (self.depth is None or d[1] != self.depth[1]):
                self.depth = d
                self.depth_frames += 1

    def close(self):
        self._stop.set()
        self.cam.close()


def serve_client(conn, src, stop):
    cam = src.cam
    header = json.dumps(dict(
        name=cam.name, w=cam.w, h=cam.h, fps=cam.fps,
        serial=getattr(cam, 'serial', None),
        colour_intrinsics=getattr(cam, 'colour_intrinsics', None),
        depth_to_colour=getattr(cam, 'depth_to_colour', None),
        depth=dict(w=cam.depth_w, h=cam.depth_h, scale=cam.depth_scale,
                   intrinsics=cam.depth_intrinsics))).encode()
    try:
        conn.sendall(struct.pack('!I', len(header)) + header)
        last_c = last_d = None
        while not stop.is_set():
            sent = False
            c = src.color
            if c is not None and c[0] != last_c:
                last_c = c[0]
                payload = np.ascontiguousarray(c[2], dtype=np.uint8).tobytes()
                conn.sendall(struct.pack('!dIIB', c[1], len(payload), c[0] & 0xffffffff, 0) + payload)
                sent = True
            d = src.depth
            if d is not None and d[1] != last_d:
                last_d = d[1]
                payload = np.ascontiguousarray(d[0], dtype=np.uint16).tobytes()
                dseq = (src.color[0] & 0xffffffff) if src.color is not None else 0
                conn.sendall(struct.pack('!dIIB', d[1], len(payload), dseq, 1) + payload)
                sent = True
            if not sent:
                time.sleep(0.002)
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fake', action='store_true', help='synthetic frames, no camera')
    ap.add_argument('--socket', default=DEPTH_SOCKET)
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--fps', type=int, default=30)
    args = ap.parse_args()

    cam = (FakeCamera(args.width, args.height, args.fps) if args.fake
           else RealSenseCamera(args.width, args.height, args.fps))
    cam.start()
    src = Source(cam)
    print(f'camera: {cam.name}: colour {cam.w}x{cam.h} @ {cam.fps:g} fps, depth '
          f'{cam.depth_w}x{cam.depth_h}, {1000 * cam.depth_scale:g} mm/unit', flush=True)
    if os.path.exists(args.socket):
        os.unlink(args.socket)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.socket)
    os.chmod(args.socket, 0o666)        # created by root; the GUI is not root
    srv.listen(4)
    srv.settimeout(0.5)
    stop = threading.Event()
    # A plain kill must clean up like Ctrl-C does, or a stale socket file makes the
    # next viewer believe a server is running.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    print(f'serving on {args.socket} -- start the viewer or capture.py now; Ctrl-C stops',
          flush=True)
    last_report = time.time()
    try:
        while True:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                if time.time() - last_report > 10:
                    last_report = time.time()
                    print(f'  {src.colour_frames} colour / {src.depth_frames} depth frames so far',
                          flush=True)
                continue
            print('client connected', flush=True)
            threading.Thread(target=serve_client, args=(conn, src, stop), daemon=True).start()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        srv.close()
        try:
            os.unlink(args.socket)
        except OSError:
            pass
        src.close()
        print('stopped', flush=True)


if __name__ == '__main__':
    main()
