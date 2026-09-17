#!/usr/bin/env python3
"""Record CSI amplitudes and RealSense colour frames on one clock, so every CSI
packet is assigned to the video frame it belongs with.

The point is supervised pose work: frames give 2D pose ground truth, CSI gives the
input, and they are useless apart unless the alignment is trustworthy. So the
timebase is handled explicitly rather than assumed:

  * frames carry the kernel's CLOCK_MONOTONIC timestamp taken at DMA completion,
    converted once to the wall clock -- not the time Python got round to looking
  * CSI packets carry BOTH the host arrival time (the project's existing
    convention, and what `tx|rx|t` means everywhere else) AND the receiving
    board's own hardware receive timestamp in microseconds

That second one matters. Host arrival time includes UART buffering jitter, which
at 30 fps is a meaningful fraction of a frame. The board timestamp is taken in the
Wi-Fi driver at reception and is jitter-free, but sits on a free-running per-board
clock. Recording both means the jitter can be regressed out offline (fit host
arrival against board time per board, then re-assign) with the host clock still
the anchor -- and crucially, without re-capturing anything. The assignment written
here is the plain nearest-frame one on arrival time; the refinement is left to
analysis, which is why `|lts` is on disk.

Output is one self-contained NPZ: CSI arrays plus JPEG entries under `frames/`.
It keeps `tx|rx|t` and `tx|rx|a` byte-compatible with older captures, and adds
shared-origin int64 nanosecond fields for exact integer time arithmetic.

  python3 capture_synced.py --prefix take1 --seconds 60
  python3 capture_synced.py --prefix take1 --seconds 60 --mode fixedtx
"""

import argparse
import csv
import ctypes
import fcntl
import io
import json
import mmap
import os
import pathlib
import queue
import re
import select
import signal
import sys
import threading
import time
import zipfile
from collections import deque
from io import StringIO

import numpy as np
import serial
import serial.tools.list_ports
from PIL import Image

WCH_VID = 0x1A86
BAUD = 3_000_000
BOOT_MAC_RE = re.compile(r'Board MAC ([0-9a-fA-F:]{17})')
# What a board pings at from power-on until the host sends RATE: CONFIG_SEND_FREQUENCY
# in firmware/main/app_main.c. The host resets every board on discovery, so this is
# the rate a recording runs at unless --rate says otherwise.
FW_DEFAULT_RATE = 243
TX_DONE_RE = re.compile(r'^TX_DONE,(\d+),(\d+)')
N_META = 25
TS_FIELD = 18                      # local_timestamp, microseconds, per the firmware header
RSSI_FIELD = 3
TS_WRAP = 1 << 32                  # the board counter is 32-bit microseconds
C5_MACS = {
    '10:bd:a3:e6:62:3c', '10:bd:a3:e6:37:f4',
    '10:bd:a3:e6:38:14', '10:bd:a3:e6:38:24',
}

# Labels name POSITIONS in the room (seen from the camera: near row B A with A
# beside the camera, far row C D -- see README), not hardware. The rig is the four ESP32-C5-DevKitC-1 boards; the original ESP32-S3
# boards (A-D, and the returned fifth board E) were retired on 2026-09-16 and their
# MACs dropped from here. Recordings made with them carry their own labels in meta.
# Assigned 2026-09-16 by lighting each board and asking where it stood; if the
# boards are ever rearranged, this is the only place to change.
LABEL = {
    '38:24': 'A',      # 10:bd:a3:e6:38:24 / USB 5C94096766
    '38:14': 'B',      # 10:bd:a3:e6:38:14 / USB 5C94096765
    '62:3c': 'C',      # 10:bd:a3:e6:62:3c / USB 5C94096576
    '37:f4': 'D',      # 10:bd:a3:e6:37:f4 / USB 5C94096770
}

# ------------------------------------------------------------ subcarrier layout
# The radio reports 192 subcarriers as three 64-wide fields (LLTF | HT-LTF |
# STBC-HT-LTF) whose gains differ by roughly 5x, and 26 of the 192 are guard bands
# or DC that read ~0 in every capture.
FIELD_WIDTH = 64
DEAD_SUB = (set(range(0, 6)) | {32} | set(range(59, 66))
            | set(range(123, 134)) | {191})

# Mirrors SUB_INDEX_* in firmware/main/app_main.c. The firmware *compacts* a frame
# to only the subcarriers it sends, so row i of a 166-wide frame is original
# subcarrier SUB_INDEX[166][i] -- the field boundaries are then no longer every 64
# rows. Anything normalising per field must map through this; see field_bounds.
SUB_INDEX = {
    192: list(range(192)),
    166: [i for i in range(192) if i not in DEAD_SUB],
    # every HT-LTF subcarrier: 166 minus the 52 5x-weaker legacy-LLTF duplicates
    114: [i for i in range(66, 191) if i not in DEAD_SUB],
    30: [66, 70, 74, 78, 82, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121,
         135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 174, 178, 182, 186, 190],
}


def derive_fields(mean_amp, min_ratio=3.0, win=8, min_sep=12):
    """Field boundaries measured from the data instead of assumed from a table.

    Worth measuring rather than tabulating, because the tabulated answer is wrong.
    The layout is documented here as three 64-wide fields, but the radio's own output
    says otherwise: on a live HT40 link the second and third 64-blocks have the same
    mean amplitude to within 3% (43.1 against 44.3) while the first is 4.8x lower.
    64-191 is one 128-wide HT-LTF whose centre null is the dead band at 123-133, so
    there is exactly one gain step, at 64. A hardcoded three-way split draws a
    boundary where no step exists -- and would be wrong differently again at HT20,
    where the whole layout changes.

    min_ratio is 3 rather than 2 because a deep multipath fade can hold a 2x level
    difference across a whole window and register as a phantom boundary -- observed
    live at row 106 of a 166-wide link. The real hardware step measures 4.8x, so 3x
    separates the two cleanly.

    A gain step is directly observable, so this observes it: walk the live
    subcarriers and compare the median level of the window before each position with
    the window after. Positions whose ratio clears `min_ratio` are candidates; the
    strongest in each cluster wins, and `min_sep` keeps one step from registering
    twice. Dead subcarriers sit near zero and would dominate any window they fell in,
    so they are excluded from the statistics rather than from the row numbering.

    Returns [(lo, hi), ...] covering 0..len(mean_amp), so a caller can normalise per
    field without knowing the bandwidth or the subcarrier count.
    """
    a = np.asarray(mean_amp, dtype=float)
    n = len(a)
    if n < 4 * win:
        return [(0, n)]
    pos = a[a > 0]
    if pos.size == 0:
        return [(0, n)]
    live = a > 0.05 * np.median(pos)
    if live.sum() < 2 * win:
        return [(0, n)]

    scores = np.zeros(n)
    for i in range(win, n - win):
        lo = a[i - win:i][live[i - win:i]]
        hi = a[i:i + win][live[i:i + win]]
        if lo.size < max(2, win // 2) or hi.size < max(2, win // 2):
            continue
        r = np.median(hi) / max(np.median(lo), 1e-9)
        if r >= min_ratio or r <= 1.0 / min_ratio:
            scores[i] = abs(np.log(r))

    cuts = []
    order = np.argsort(scores)[::-1]
    for i in order:
        if scores[i] <= 0:
            break
        if all(abs(i - c) >= min_sep for c in cuts):
            cuts.append(int(i))
    cuts.sort()

    edges = [0] + cuts + [n]
    return [(edges[k], edges[k + 1]) for k in range(len(edges) - 1)]


def field_bounds(n_sub):
    """Fallback boundaries for a frame width, used only before any data has arrived.

    derive_fields is the real answer; this exists so the first repaint has something
    sane. It splits at the one boundary the hardware actually has -- the LLTF/HT-LTF
    step at original subcarrier 64 -- rather than at every 64th row.
    """
    idx = SUB_INDEX.get(n_sub)
    if idx is None:
        return [(0, n_sub)]
    rows = [i for i, o in enumerate(idx) if o >= FIELD_WIDTH]
    if not rows or rows[0] == 0:
        return [(0, n_sub)]
    return [(0, rows[0]), (rows[0], n_sub)]

YELLOW, GREEN, WHITE = (40, 30, 0), (0, 40, 0), (30, 30, 30)

# ---------------------------------------------------------------- V4L2 capture

VIDIOC_S_FMT = 0xc0d05605
VIDIOC_REQBUFS = 0xc0145608
VIDIOC_QUERYBUF = 0xc0585609
VIDIOC_QBUF = 0xc058560f
VIDIOC_DQBUF = 0xc0585611
VIDIOC_STREAMON = 0x40045612
VIDIOC_STREAMOFF = 0x40045613
VIDIOC_S_PARM = 0xc0cc5616
CAPTURE, MMAP = 1, 1
TS_MASK, TS_MONOTONIC = 0xe000, 0x2000


VIDIOC_ENUM_FMT = 0xc0405602


def fourcc(s):
    return sum(ord(c) << (8 * i) for i, c in enumerate(s))


class FmtDesc(ctypes.Structure):
    _fields_ = [('index', ctypes.c_uint32), ('type', ctypes.c_uint32),
                ('flags', ctypes.c_uint32), ('description', ctypes.c_char * 32),
                ('pixelformat', ctypes.c_uint32), ('mbus_code', ctypes.c_uint32),
                ('reserved', ctypes.c_uint32 * 3)]


def find_colour_node():
    """The RealSense colour node, found by capability rather than by number.

    Node numbering is not stable: it depends on what else is plugged in and in
    what order (the built-in webcam takes video0/1 on this machine, pushing the
    camera to video2-7). The camera exposes several nodes and only the colour one
    offers YUYV -- depth is Z16 and the infrared pair is Y8I/Y12I -- so that is
    what identifies it.
    """
    for path in sorted(f'/dev/video{i}' for i in range(64)):
        if not os.path.exists(path):
            continue
        try:
            with open(f'/sys/class/video4linux/{os.path.basename(path)}/name') as fh:
                if 'RealSense' not in fh.read():
                    continue
        except OSError:
            continue
        try:
            fd = os.open(path, os.O_RDWR)
        except OSError:
            continue
        try:
            for i in range(16):
                f = FmtDesc(index=i, type=CAPTURE)
                try:
                    fcntl.ioctl(fd, VIDIOC_ENUM_FMT, f)
                except OSError:
                    break
                if f.pixelformat == fourcc('YUYV'):
                    return path
        finally:
            os.close(fd)
    return None


class PixFormat(ctypes.Structure):
    _fields_ = [('width', ctypes.c_uint32), ('height', ctypes.c_uint32),
                ('pixelformat', ctypes.c_uint32), ('field', ctypes.c_uint32),
                ('bytesperline', ctypes.c_uint32), ('sizeimage', ctypes.c_uint32),
                ('colorspace', ctypes.c_uint32), ('priv', ctypes.c_uint32),
                ('flags', ctypes.c_uint32), ('enc', ctypes.c_uint32),
                ('quantization', ctypes.c_uint32), ('xfer_func', ctypes.c_uint32)]


class Format(ctypes.Structure):
    # the fmt union holds a pointer, so it is 8-aligned and starts at offset 8
    _fields_ = [('type', ctypes.c_uint32), ('_pad', ctypes.c_uint32),
                ('pix', PixFormat), ('_rest', ctypes.c_uint8 * (200 - 48))]


class StreamParm(ctypes.Structure):
    _fields_ = [('type', ctypes.c_uint32), ('capability', ctypes.c_uint32),
                ('capturemode', ctypes.c_uint32), ('num', ctypes.c_uint32),
                ('den', ctypes.c_uint32), ('extendedmode', ctypes.c_uint32),
                ('readbuffers', ctypes.c_uint32), ('_rest', ctypes.c_uint8 * 176)]


class ReqBufs(ctypes.Structure):
    _fields_ = [('count', ctypes.c_uint32), ('type', ctypes.c_uint32),
                ('memory', ctypes.c_uint32), ('capabilities', ctypes.c_uint32),
                ('flags', ctypes.c_uint8), ('reserved', ctypes.c_uint8 * 3)]


class TimeVal(ctypes.Structure):
    _fields_ = [('sec', ctypes.c_long), ('usec', ctypes.c_long)]


class VBuffer(ctypes.Structure):
    _fields_ = [('index', ctypes.c_uint32), ('type', ctypes.c_uint32),
                ('bytesused', ctypes.c_uint32), ('flags', ctypes.c_uint32),
                ('field', ctypes.c_uint32), ('_pad', ctypes.c_uint32),
                ('timestamp', TimeVal), ('timecode', ctypes.c_uint8 * 16),
                ('sequence', ctypes.c_uint32), ('memory', ctypes.c_uint32),
                ('offset', ctypes.c_uint32), ('_pad2', ctypes.c_uint32),
                ('length', ctypes.c_uint32), ('reserved2', ctypes.c_uint32),
                ('request_fd', ctypes.c_int32), ('_tail', ctypes.c_uint32)]


class Camera:
    """Minimal V4L2 mmap capture. Deliberately not librealsense: the SDK is not
    installed, and for a colour stream it would add a dependency without adding
    anything this needs."""

    def __init__(self, dev, w, h, fps, nbuf=8):
        self.name = str(dev)
        self.fd = os.open(dev, os.O_RDWR)
        f = Format(type=CAPTURE)
        f.pix.width, f.pix.height, f.pix.pixelformat, f.pix.field = w, h, fourcc('YUYV'), 1
        fcntl.ioctl(self.fd, VIDIOC_S_FMT, f)
        self.w, self.h = f.pix.width, f.pix.height
        got = ''.join(chr((f.pix.pixelformat >> (8 * k)) & 0xff) for k in range(4)).strip()
        if got != 'YUYV':
            raise RuntimeError(f'{dev} gave {got}, not YUYV')

        p = StreamParm(type=CAPTURE, num=1, den=int(fps))
        fcntl.ioctl(self.fd, VIDIOC_S_PARM, p)
        self.fps = p.den / max(p.num, 1)

        r = ReqBufs(count=nbuf, type=CAPTURE, memory=MMAP)
        fcntl.ioctl(self.fd, VIDIOC_REQBUFS, r)
        self.maps = []
        for i in range(r.count):
            b = VBuffer(index=i, type=CAPTURE, memory=MMAP)
            fcntl.ioctl(self.fd, VIDIOC_QUERYBUF, b)
            self.maps.append(mmap.mmap(self.fd, b.length, mmap.MAP_SHARED,
                                       mmap.PROT_READ | mmap.PROT_WRITE, offset=b.offset))
            fcntl.ioctl(self.fd, VIDIOC_QBUF, b)
        self.monotonic = None

    def start(self):
        fcntl.ioctl(self.fd, VIDIOC_STREAMON, ctypes.c_int(CAPTURE))

    def read(self, timeout=1.0):
        """(sequence, timestamp, frame bytes) or None on timeout."""
        if not select.select([self.fd], [], [], timeout)[0]:
            return None
        b = VBuffer(type=CAPTURE, memory=MMAP)
        fcntl.ioctl(self.fd, VIDIOC_DQBUF, b)
        if self.monotonic is None:
            self.monotonic = (b.flags & TS_MASK) == TS_MONOTONIC
        frame = bytes(self.maps[b.index][:b.bytesused])
        ts = b.timestamp.sec + b.timestamp.usec / 1e6
        seq = b.sequence
        fcntl.ioctl(self.fd, VIDIOC_QBUF, b)
        return seq, ts, frame

    def close(self):
        try:
            fcntl.ioctl(self.fd, VIDIOC_STREAMOFF, ctypes.c_int(CAPTURE))
        except OSError:
            pass
        for m in self.maps:
            m.close()
        os.close(self.fd)

    def to_rgb(self, buf, w, h):
        """Frames from this camera, as RGB. Lives on the camera because it is a
        property of the device, not of the caller: a second backend (see MacCamera)
        hands back a different pixel format, and every consumer that hard-codes
        yuyv_to_rgb would then decode it as garbage rather than fail."""
        return yuyv_to_rgb(buf, w, h)


def yuyv_to_rgb(buf, w, h):
    """Packed 4:2:2 to RGB, BT.601 limited range (what UVC cameras emit).

    int32, not int16: the luma coefficient alone reaches 298*(255-16) = 71222,
    which wraps in int16 and turns anything bright -- a window, a lit subject --
    black with colour fringing, while leaving mid-tones looking perfectly fine.
    """
    d = np.frombuffer(buf, np.uint8).reshape(h, w // 2, 4).astype(np.int32)
    y = np.empty((h, w), np.int32)
    y[:, 0::2], y[:, 1::2] = d[:, :, 0], d[:, :, 2]
    u = np.repeat(d[:, :, 1], 2, axis=1) - 128
    v = np.repeat(d[:, :, 3], 2, axis=1) - 128
    c = y - 16
    r = (298 * c + 409 * v + 128) >> 8
    g = (298 * c - 100 * u - 208 * v + 128) >> 8
    b = (298 * c + 516 * u + 128) >> 8
    return np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8)


class MacCamera:
    """AVFoundation capture through OpenCV, for running the rig on a Mac.

    The V4L2 path above cannot be made to work here: macOS has no /dev/video*
    and no video4linux sysfs, so there is nothing to enumerate and no ioctl to
    call. This is deliberately the same shape as Camera -- start/read/close plus
    to_rgb -- so the viewer and the recorder consume either without knowing which
    one they are holding.

    Two things are genuinely worse on this backend, and the recorded metadata
    says so rather than quietly implying otherwise:

      * **No driver timestamp.** AVFoundation's presentation time is not exposed
        through VideoCapture, so a frame is stamped when it reaches us and
        `monotonic` is False -- which routes callers to their time.time() branch.
        The extra error is grab-loop scheduling jitter, ~1 frame at 30 fps.
      * **No driver sequence number.** `seq` is a local counter, so it cannot
        gap. On V4L2 a gap in `sequence` is exactly what makes an OS-dropped
        frame countable; here such a frame is invisible, and the frame-drop
        figures in RECORDING_2026-08-12.md have no equivalent for macOS sessions.

    Neither disturbs CSI/video alignment more than the grab jitter already does,
    because alignment is done offline from these same timestamps. Do not use this
    backend to make claims about dropped video frames.
    """

    def __init__(self, index, w, h, fps, name=None):
        import cv2
        self.cv2 = cv2
        self.index = int(index)
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            raise RuntimeError(
                f'could not open camera index {self.index}. Another process may hold '
                f'it, or this terminal has not been granted camera access in System '
                f'Settings > Privacy & Security > Camera.')
        # Identify before sizing: the fingerprint is the largest mode this device
        # offers, which is only readable while nothing narrower has been requested.
        if name is None:
            name = _identify_capture(self.cap)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        # Read back rather than trust: AVFoundation silently substitutes the nearest
        # supported mode, and metadata claiming a geometry the frames do not have is
        # worse than metadata admitting the substitution.
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or w
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or h
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or fps
        self.name = (f'{name} (index {self.index})' if name
                     else f'AVFoundation index {self.index}')
        self.monotonic = False
        self.seq = 0

    def start(self):
        pass                      # VideoCapture streams from the moment it opens

    def read(self, timeout=1.0):
        """(sequence, timestamp, BGR frame) or None. `timeout` is accepted for
        interface parity and ignored: VideoCapture.read blocks until the next
        frame and offers no way to bound that wait."""
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        self.seq += 1
        return self.seq - 1, time.time(), frame

    def to_rgb(self, buf, w, h):
        # cvtColor, not buf[:, :, ::-1]: the reversed view is not contiguous, and
        # both consumers -- QImage and PIL -- read the buffer directly and would
        # render the stride as tearing rather than raise.
        return self.cv2.cvtColor(buf, self.cv2.COLOR_BGR2RGB)

    def close(self):
        self.cap.release()


class RealSenseCamera:
    """Colour AND depth from an Intel RealSense through librealsense (pyrealsense2).

    Same start/read/close/to_rgb shape as Camera and MacCamera, plus:

      depth        (uint16 array in device units, wall time) of the depth frame
                   that arrived with the last colour frame read() returned
      depth_scale  metres per unit (0.001 on a D435i)
      depth_rgb()  a colourised 8-bit RGB view of that frame for the GUI
      has_depth    True

    Timestamps: librealsense maps the sensor clock onto the host clock ("global
    time", milliseconds since the epoch), so read() returns wall time directly and
    `wall_ts` is True -- the recorders use it as is instead of stamping arrival.
    `seq` is the camera's own frame counter, so a dropped frame is a visible gap
    again, which the AVFoundation path could never say.

    On macOS librealsense must seize the USB interfaces from the kernel UVC driver,
    which needs root: run the tool with sudo, or accept colour-only through
    AVFoundation (open_camera falls back to that by itself and says so).
    """

    def __init__(self, w, h, fps, depth_w=640, depth_h=480):
        import pyrealsense2 as rs
        self.rs = rs
        ctx = rs.context()
        if len(ctx.query_devices()) == 0:
            raise RuntimeError('no RealSense device found by librealsense')
        self.pipe = rs.pipeline(ctx)
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, int(w), int(h), rs.format.bgr8, int(round(fps)))
        cfg.enable_stream(rs.stream.depth, int(depth_w), int(depth_h), rs.format.z16,
                          int(round(fps)))
        prof = self.pipe.start(cfg)
        dev = prof.get_device()
        serial = dev.get_info(rs.camera_info.serial_number)
        self.name = f'{dev.get_info(rs.camera_info.name)} {serial} (librealsense)'
        self.depth_scale = float(dev.first_depth_sensor().get_depth_scale())
        cs = prof.get_stream(rs.stream.color).as_video_stream_profile()
        self.w, self.h, self.fps = cs.width(), cs.height(), float(cs.fps())
        ds = prof.get_stream(rs.stream.depth).as_video_stream_profile()
        self.depth_w, self.depth_h = ds.width(), ds.height()
        i = ds.get_intrinsics()
        self.depth_intrinsics = dict(fx=i.fx, fy=i.fy, ppx=i.ppx, ppy=i.ppy,
                                     model=str(i.model), coeffs=list(i.coeffs))
        # The 3-D pose fit projects the body with the COLOUR intrinsics and moves
        # depth points into the colour frame with these extrinsics; recorded in
        # meta so a take is self-describing (takes before 2026-09-17 need
        # calib/realsense_<serial>.json from tools/rs_calib.py instead).
        self.serial = serial
        self.colour_intrinsics, self.depth_to_colour = rs_colour_geometry(cs, ds)
        self.monotonic = False
        self.wall_ts = True
        self.has_depth = True
        self.depth = None
        self.seq = 0

    def start(self):
        pass                      # the pipeline streams from start()

    def read(self, timeout=1.0):
        """(camera frame number, wall time, BGR colour frame) or None; the matching
        depth frame lands in self.depth."""
        try:
            fs = self.pipe.wait_for_frames(int(timeout * 1000))
        except RuntimeError:
            return None           # timed out: the caller loops
        c = fs.get_color_frame()
        if not c:
            return None
        ts = c.get_timestamp() / 1000.0
        if not 1e9 < ts < 4e9:    # not global time after all: stamp arrival
            ts = time.time()
        # Copy: librealsense recycles the frame buffer once the frameset dies.
        buf = np.asanyarray(c.get_data()).copy()
        d = fs.get_depth_frame()
        if d:
            self.depth = (np.asanyarray(d.get_data()).copy(), ts)
            hook = getattr(self, 'on_depth', None)
            if hook is not None:
                hook(self.depth[0], ts, int(d.get_frame_number()))
        self.seq = int(c.get_frame_number())
        return self.seq, ts, buf

    def to_rgb(self, buf, w, h):
        return np.ascontiguousarray(buf[:, :, ::-1])

    def depth_rgb(self, max_m=4.0):
        """The last depth frame as 8-bit RGB: near is red, far is blue, no return
        is black. For looking at, not for measuring; the uint16 frame is what gets
        recorded."""
        if self.depth is None:
            return None
        import cv2
        d = self.depth[0]
        v = np.clip(d.astype(np.float32) * self.depth_scale / max_m, 0.0, 1.0)
        img = cv2.applyColorMap((255 * (1.0 - v)).astype(np.uint8), cv2.COLORMAP_JET)
        img[d == 0] = 0
        return np.ascontiguousarray(img[:, :, ::-1])

    def close(self):
        try:
            self.pipe.stop()
        except RuntimeError:
            pass


# Where depth_server.py (run as root on a Mac) offers the depth stream to the
# unprivileged viewer and recorder.
DEPTH_SOCKET = '/tmp/csi-depth.sock'


def rs_colour_geometry(cs, ds):
    """(colour intrinsics dict, depth->colour extrinsics dict) for two librealsense
    video stream profiles, in the layout body_common.load_calib reads."""
    i = cs.get_intrinsics()
    intr = dict(width=cs.width(), height=cs.height(), fx=i.fx, fy=i.fy, ppx=i.ppx,
                ppy=i.ppy, model=str(i.model), coeffs=list(i.coeffs))
    ex = ds.get_extrinsics_to(cs)
    return intr, dict(rotation=[float(r) for r in ex.rotation],
                      translation=[float(t) for t in ex.translation])


def colourise_depth(d, scale, max_m=4.0):
    """A depth frame as 8-bit RGB: near is red, far is blue, no return is black.
    For looking at, not for measuring; the uint16 frame is what gets recorded."""
    import cv2
    v = np.clip(d.astype(np.float32) * scale / max_m, 0.0, 1.0)
    img = cv2.applyColorMap((255 * (1.0 - v)).astype(np.uint8), cv2.COLORMAP_JET)
    img[d == 0] = 0
    return np.ascontiguousarray(img[:, :, ::-1])


class ServedCamera:
    """Colour and depth as served by tools/depth_server.py over its local socket:
    the same shape as RealSenseCamera (read() -> (seq, wall time, BGR), .depth,
    depth_rgb(), has_depth, wall_ts), for a viewer or recorder that must not run
    as root. Frames carry the server's timestamps (librealsense global time, i.e.
    host wall time), so alignment is as good as in-process."""

    def __init__(self, path=DEPTH_SOCKET, timeout=3.0):
        import socket
        import struct
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect(path)
        self.f = self.sock.makefile('rb')
        n = struct.unpack('!I', self._read(4))[0]
        hdr = json.loads(self._read(n))
        self.sock.settimeout(None)
        if 'depth' not in hdr:
            self.sock.close()
            raise OSError('an older depth_server (depth only) is running; restart it')
        self.w, self.h, self.fps = int(hdr['w']), int(hdr['h']), float(hdr['fps'])
        d = hdr['depth']
        self.depth_w, self.depth_h = int(d['w']), int(d['h'])
        self.depth_scale = float(d['scale'])
        self.depth_intrinsics = d['intrinsics']
        self.colour_intrinsics = hdr.get('colour_intrinsics')
        self.depth_to_colour = hdr.get('depth_to_colour')
        self.serial = hdr.get('serial')
        self.name = f"{hdr['name']} via depth_server"
        self.monotonic = False
        self.wall_ts = True
        self.has_depth = True
        self.depth = None
        self.on_depth = None                 # recorder hook: called with every depth frame
        self.colour = None                   # (seq, ts, bgr)
        self.frames = 0
        self._cv = threading.Condition()
        self._struct = struct
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _read(self, n):
        buf = self.f.read(n)
        if len(buf) < n:
            raise ConnectionError('depth_server closed the connection')
        return buf

    def _run(self):
        while not self._stop.is_set():
            try:
                ts, nbytes, seq, kind = self._struct.unpack('!dIIB', self._read(17))
                payload = self._read(nbytes)
            except (ConnectionError, OSError, ValueError):
                with self._cv:
                    self._cv.notify_all()
                return                       # the server went away
            if kind == 1:
                self.depth = (np.frombuffer(payload, dtype=np.uint16)
                              .reshape(self.depth_h, self.depth_w).copy(), ts)
                hook = self.on_depth
                if hook is not None:
                    hook(self.depth[0], ts, seq)
            else:
                bgr = np.frombuffer(payload, dtype=np.uint8).reshape(self.h, self.w, 3).copy()
                with self._cv:
                    self.colour = (seq, ts, bgr)
                    self.frames += 1
                    self._cv.notify_all()

    def start(self):
        pass

    def read(self, timeout=1.0):
        """The next colour frame, or None after `timeout` without one."""
        with self._cv:
            before = self.frames
            self._cv.wait_for(lambda: self.frames != before or self._stop.is_set(), timeout)
            if self.frames == before:
                return None
            return self.colour

    def to_rgb(self, buf, w, h):
        return np.ascontiguousarray(buf[:, :, ::-1])

    def depth_rgb(self, max_m=4.0):
        return None if self.depth is None else colourise_depth(self.depth[0], self.depth_scale, max_m)

    def close(self):
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass


class RealSenseDepth:
    """The RealSense depth stream alone, through librealsense, for a colour camera
    that comes from somewhere else (the OS camera stack on a Mac).

    Why split: on macOS 26 the kernel's own UVC driver claims one of the camera's
    two video interfaces and nothing -- root included -- can take it back
    (RS2_USB_STATUS_ACCESS). Freshly plugged in it claims the colour interface and
    leaves the depth one free, so colour goes through AVFoundation and depth
    through librealsense, each on the interface it can have. Ask for colour from
    librealsense and it not only fails, the failed grab makes macOS re-enumerate
    the camera with the driver on the depth interface instead, after which only a
    replug restores the colour camera.

    A thread keeps `depth` = (uint16 array in device units, host arrival time) up
    to date; the colour grabber pairs each colour frame with the latest one.
    """

    def __init__(self, w=640, h=480, fps=30):
        import pyrealsense2 as rs
        self.rs = rs
        ctx = rs.context()
        if len(ctx.query_devices()) == 0:
            raise RuntimeError('no RealSense device found by librealsense')
        self.pipe = rs.pipeline(ctx)
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, int(w), int(h), rs.format.z16, int(round(fps)))
        prof = self.pipe.start(cfg)
        dev = prof.get_device()
        self.depth_scale = float(dev.first_depth_sensor().get_depth_scale())
        ds = prof.get_stream(rs.stream.depth).as_video_stream_profile()
        self.depth_w, self.depth_h, self.fps = ds.width(), ds.height(), float(ds.fps())
        i = ds.get_intrinsics()
        self.depth_intrinsics = dict(fx=i.fx, fy=i.fy, ppx=i.ppx, ppy=i.ppy,
                                     model=str(i.model), coeffs=list(i.coeffs))
        self.name = f'{dev.get_info(rs.camera_info.name)} depth (librealsense)'
        self.depth = None
        self.frames = 0
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                fs = self.pipe.wait_for_frames(1000)
            except RuntimeError:
                continue
            d = fs.get_depth_frame()
            if d:
                # Copy: librealsense recycles the buffer once the frameset dies.
                self.depth = (np.asanyarray(d.get_data()).copy(), time.time())
                self.frames += 1
                hook = getattr(self, 'on_depth', None)
                if hook is not None:
                    hook(self.depth[0], self.depth[1], self.frames)

    def depth_rgb(self, max_m=4.0):
        return None if self.depth is None else colourise_depth(self.depth[0], self.depth_scale, max_m)

    def close(self):
        self._stop.set()
        self._th.join(timeout=2)
        try:
            self.pipe.stop()
        except RuntimeError:
            pass


def attach_depth(cam, source):
    """Give a colour-only camera object the depth attributes the recorders and the
    viewer look for (has_depth, depth, depth_rgb, depth_scale, depth_w/h,
    depth_intrinsics, on_depth) from a RealSenseDepth, and close both together."""
    cam.has_depth = True
    cam.depth_source = source
    cam.depth_scale = source.depth_scale
    cam.depth_w, cam.depth_h = source.depth_w, source.depth_h
    cam.depth_intrinsics = source.depth_intrinsics
    cam.name = f'{cam.name} + {source.name}'
    cls = cam.__class__
    cam.__class__ = type(cls.__name__ + 'WithDepth', (cls,), {
        'depth': property(lambda self: self.depth_source.depth),
        'on_depth': property(lambda self: self.depth_source.on_depth,
                             lambda self, f: setattr(self.depth_source, 'on_depth', f)),
        'depth_rgb': lambda self, max_m=4.0: self.depth_source.depth_rgb(max_m),
        'close': lambda self: (self.depth_source.close(), cls.close(self)),
    })
    return cam


def frame_wall(cam, ts, mono_offset):
    """The wall-clock time of a frame, from whatever the backend could give: its own
    wall stamp (librealsense), a CLOCK_MONOTONIC driver stamp (V4L2), or arrival."""
    if getattr(cam, 'wall_ts', False):
        return ts
    return ts + mono_offset if cam.monotonic else time.time()


def camera_gt_meta(cam, gt):
    """Meta fields describing what the frames are: 'gt' names the stream, and a depth
    recording also carries the scale and intrinsics needed to read it as metres."""
    out = dict(gt=gt)
    if getattr(cam, 'has_depth', False):
        out.update(depth_scale_m=cam.depth_scale, depth_width=cam.depth_w,
                   depth_height=cam.depth_h, depth_intrinsics=cam.depth_intrinsics)
        for k in ('colour_intrinsics', 'depth_to_colour'):
            if getattr(cam, k, None):
                out[k] = getattr(cam, k)
    if getattr(cam, 'serial', None):
        out['camera_serial'] = cam.serial
    if gt == 'depth':
        out.update(frame_dtype='uint16', frame_unit='depth units, x depth_scale_m = metres')
    return out


def mac_video_devices():
    """Every AVFoundation camera: name, type, and its largest-area mode.

    Needs pyobjc; returns [] without it, which callers treat as "cannot identify"
    rather than "no cameras".
    """
    try:
        import AVFoundation as AV
        import CoreMedia
    except ImportError:
        return []
    out = []
    for d in AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo):
        dims = []
        for f in d.formats():
            m = CoreMedia.CMVideoFormatDescriptionGetDimensions(f.formatDescription())
            dims.append((int(m.width), int(m.height)))
        if not dims:
            continue
        kind = str(d.deviceType()).replace('AVCaptureDeviceType', '')
        out.append(dict(name=str(d.localizedName()), kind=kind,
                        builtin=kind.startswith('BuiltIn'),
                        best=max(dims, key=lambda wh: wh[0] * wh[1])))
    return out


def _opencv_max_mode(index):
    """The largest-area mode OpenCV will give for this index, or None if it won't open.

    Asking for an absurd size and reading back what survives is how the index is
    identified -- see mac_camera_index.
    """
    import cv2
    cap = cv2.VideoCapture(int(index), cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap.release()
        return None
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 100000)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 100000)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h) if w and h else None
    finally:
        cap.release()


def _identify_capture(cap):
    """The name of the device this open capture is actually reading, or None.

    Same fingerprint as mac_camera_index, but run on a capture we already hold, so
    naming the camera costs nothing extra and happens even when an index was given
    explicitly. A viewer that prints the device it truly opened is the check that
    catches a mis-set --device before a session is recorded, not after.
    """
    import cv2
    devs = mac_video_devices()
    if not devs:
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 100000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 100000)
    best = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    hits = [d for d in devs if d['best'] == best]
    return hits[0]['name'] if len(hits) == 1 else None


def mac_camera_index(match=None, max_probe=8):
    """(OpenCV index, device name) for the camera `match` names, by identity not order.

    **OpenCV's index order is not AVFoundation's list order.** Measured on this
    machine with a built-in camera and an iPhone attached over Continuity: pyobjc
    lists [FaceTime, iPhone] while OpenCV's indices are [iPhone, FaceTime] -- exactly
    reversed. So taking a device's position in the AVFoundation list and passing it to
    VideoCapture silently opens the *other* camera, which is how a session gets
    recorded through a phone lying face-down on the desk.

    The two are matched by fingerprint instead. OpenCV selects a device's
    largest-area mode when asked for an impossible size, and that mode differs per
    camera (here 1552x1552 built-in against 1920x1440 for the phone), so it
    identifies which device an index actually opened.

    `match` is a case-insensitive substring of the camera name, or None for the
    built-in one. Returns (None, None) when the answer cannot be established --
    pyobjc missing, no match, or an ambiguous fingerprint -- so the caller can fall
    back loudly rather than open an arbitrary camera and claim it was the right one.
    """
    devs = mac_video_devices()
    if not devs:
        return None, None
    if match is None:
        wanted = [d for d in devs if d['builtin']]
    else:
        wanted = [d for d in devs if match.lower() in d['name'].lower()]
    if len(wanted) != 1:
        return None, None
    want = wanted[0]
    # Ambiguous fingerprint: two cameras whose largest mode is the same size cannot be
    # told apart this way, and guessing between them is the failure this exists to stop.
    if sum(d['best'] == want['best'] for d in devs) != 1:
        return None, None
    # Probe the LIKELY index first, not index 0 upward: opening a Continuity iPhone
    # (merely to fingerprint it) wakes the phone on the desk, every launch. OpenCV's
    # index order measured on this machine is AVFoundation's list order reversed, so
    # that guess is tried first and, when right -- the normal case -- the only camera
    # ever opened is the one asked for.
    guess = (len(devs) - 1) - devs.index(want)
    order = [guess] + [i for i in range(max_probe) if i != guess]
    for i in order:
        if _opencv_max_mode(i) == want['best']:
            return i, want['name']
    return None, None


def default_camera_device():
    """What `--device auto` means on this machine.

    Linux is the rig proper: find the RealSense colour node by capability. On macOS
    it is the *built-in* camera, resolved by identity -- never simply index 0, which
    is the attached phone as often as not.
    """
    if sys.platform != 'darwin':
        return find_colour_node()
    # A RealSense, if one is attached: through depth_server.py when that is running
    # (it holds the camera, so AVFoundation no longer lists it), else its colour
    # stream through AVFoundation.
    if os.path.exists(DEPTH_SOCKET):
        return 'realsense'
    if any('realsense' in d['name'].lower() for d in mac_video_devices()):
        return 'realsense'
    idx, name = mac_camera_index()
    if idx is None:
        print('warning: could not identify the built-in camera; falling back to '
              'index 0. Pass --device <index or name> to choose explicitly.',
              file=sys.stderr, flush=True)
        return '0'
    return str(idx)


def open_camera(dev, w, h, fps, depth=False):
    """A camera for `dev`, choosing the backend the device identifies.

    A path is a V4L2 node. A bare integer is an AVFoundation index, used as given so
    an explicit `--device 1` is never second-guessed. Anything else is a camera name
    to match on macOS, which is the stable way to ask for one: indices move when a
    Continuity camera comes and goes, names do not. 'realsense' is the attached
    RealSense: its colour stream through the OS camera stack, or, with `depth`,
    colour and depth through librealsense -- opt-in because librealsense crashes
    the interpreter rather than failing when it cannot open the device (seen on
    macOS 26 unprivileged at stream start and as root during enumeration).
    """
    dev = str(dev)
    if dev.isdigit():
        return MacCamera(dev, w, h, fps)
    if dev == 'realsense':
        if sys.platform != 'darwin':
            # Linux: librealsense owns the whole camera (V4L2 nodes, udev rules).
            if depth:
                try:
                    return RealSenseCamera(w, h, fps)
                except (ImportError, RuntimeError) as e:
                    print(f'RealSense depth unavailable: {e} -- check the udev rules '
                          f'and that nothing else holds the camera. Colour only.',
                          file=sys.stderr, flush=True)
            cam = Camera(find_colour_node(), w, h, fps)
            cam.has_depth = False
            cam.depth_reason = 'started without --depth' if not depth else 'librealsense failed'
            return cam
        # macOS. With depth_server.py running (as root, holding the camera through
        # librealsense), colour and depth both come from it: once librealsense has
        # the device the OS camera stack loses the colour interface anyway.
        if os.path.exists(DEPTH_SOCKET):
            try:
                return ServedCamera()
            except OSError as e:
                print(f'depth_server socket present but not answering ({e}); '
                      f'restart it. Trying the OS camera stack.', file=sys.stderr, flush=True)
        # Otherwise colour through AVFoundation, and depth -- only with --depth --
        # through librealsense in this process (root). The colour interface is the
        # one whose name ends in "RGB". After librealsense has touched the camera
        # macOS lists only the "... Depth" interface: opening that as a camera
        # yields a grey 8-bit rendering of the infrared stream, not a picture, so
        # it is refused rather than recorded as if it were colour.
        names = [d['name'] for d in mac_video_devices() if 'realsense' in d['name'].lower()]
        rgb = [n for n in names if 'rgb' in n.lower()]
        if not rgb:
            raise SystemExit(
                f'the RealSense colour camera is not visible to macOS (it lists '
                f'{names or "nothing"} for it). Unplug and replug the camera, then '
                f'start again.')
        idx, name = mac_camera_index(rgb[0])
        if idx is None:
            raise SystemExit('could not identify the RealSense colour camera among '
                             'the attached cameras')
        cam = MacCamera(idx, w, h, fps, name=name)
        cam.has_depth = False
        if not depth:
            cam.depth_reason = ('no depth_server running: replug the camera, then '
                                '"sudo .venv_mac/bin/python tools/depth_server.py" in '
                                'another terminal, then start again')
            return cam
        try:
            return attach_depth(cam, RealSenseDepth(fps=fps))
        except (ImportError, RuntimeError, OSError) as e:
            reason = ('pyrealsense2 is not installed' if isinstance(e, ImportError)
                      else str(e).strip().replace(chr(10), ' '))
            hint = ('librealsense needs root on a Mac: run "sudo .venv_mac/bin/'
                    'python tools/depth_server.py" in another terminal. If the '
                    'error says ACCESS or power state, the kernel camera driver '
                    'holds the depth interface: unplug and replug the camera first')
            print(f'RealSense depth unavailable: {reason} -- {hint}. Colour only.',
                  file=sys.stderr, flush=True)
            cam.depth_reason = f'{reason}; {hint}'
            return cam
    if sys.platform == 'darwin':
        idx, name = mac_camera_index(dev)
        if idx is None:
            names = [d['name'] for d in mac_video_devices()]
            raise SystemExit(f'no single camera matches {dev!r}. Available: {names}')
        return MacCamera(idx, w, h, fps, name=name)
    return Camera(dev, w, h, fps)


# --------------------------------------------------------------- channel choice

def ht40_span(primary, ht40=True):
    """The 20 MHz channels an HT40 block on `primary` overlaps.

    This is the whole reason a per-channel survey cannot be read off directly. The
    firmware places the secondary *below* a primary of 5 or more and above a lower
    one (apply_channel in app_main.c), so the 40 MHz block is centred two channels
    away from the primary rather than on it -- "channel 11" is really a block
    centred on 9. Channels sit 5 MHz apart and are 20 MHz wide, so channel c
    overlaps when its centre is within 30 MHz of the block centre.
    """
    if not ht40:
        return [c for c in range(1, 14) if abs(c - primary) * 5 < 20]
    sec = primary - 4 if primary >= 5 else primary + 4
    centre = (primary + sec) / 2.0
    return [c for c in range(1, 14) if abs(c - centre) * 5 < 30]


def rank_channels(stats, ht40=True):
    """[(primary, bytes, worst_rssi, span)] for every primary, quietest first.

    Ranked on bytes seen -- an airtime proxy, and airtime is what actually collides
    with ESP-NOW -- with the strongest interferer in the span as the tie-break and
    as something the caller should show, because a close AP desenses the receiver
    even when it is not talking much. Deliberately not a single blended score: the
    two effects have no honest common unit, and inventing a weighting would hide
    which one drove the answer.
    """
    out = []
    for p in range(1, 14):
        span = ht40_span(p, ht40)
        if not span:
            continue
        nbytes = sum(stats.get(c, {}).get('bytes', 0) for c in span)
        rssi = max((stats[c]['rssi_max'] for c in span if c in stats), default=-128)
        out.append((p, nbytes, rssi, span))
    out.sort(key=lambda t: (t[1], t[2]))
    return out


def parse_scan_line(line):
    """('ch', ch, pkts, bytes, rssi_mean, rssi_max) | ('done', ch) | None."""
    if line.startswith('SCAN_CH,'):
        f = line.split(',')
        if len(f) == 6:
            try:
                return ('ch',) + tuple(int(x) for x in f[1:])
            except ValueError:
                return None
    elif line.startswith('SCAN_DONE'):
        f = line.split(',')
        try:
            return ('done', int(f[1])) if len(f) > 1 else ('done', 0)
        except ValueError:
            return ('done', 0)
    return None


# ------------------------------------------------------------------ CSI boards

def discover():
    ports = sorted(p.device for p in serial.tools.list_ports.comports() if p.vid == WCH_VID)
    boards, lock = {}, threading.Lock()

    def probe(port):
        try:
            ser = serial.Serial(port, BAUD, timeout=0.3)
        except (serial.SerialException, OSError):
            return
        ser.setDTR(False)
        ser.setRTS(True)
        time.sleep(0.1)
        ser.setRTS(False)
        mac, start = None, time.time()
        st = CsiStream()          # a board already receiving is emitting binary frames
        while time.time() - start < 3 and mac is None:
            data = ser.read(ser.in_waiting or 1)
            if not data:
                continue
            _recs, lines = st.feed(data)
            for ln in lines:
                m = BOOT_MAC_RE.search(ln.decode(errors='ignore'))
                if m:
                    mac = m.group(1).lower()
                    break
        if mac:
            with lock:
                boards[mac] = ser
        else:
            ser.close()

    ts = [threading.Thread(target=probe, args=(p,)) for p in ports]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return boards


def parse(line):
    """(tx mac, board timestamp us, rssi, amplitudes) or None."""
    try:
        row = next(csv.reader(StringIO(line)))
        if len(row) != N_META:
            return None
        n = int(row[-3])
        vals = json.loads(row[-1])
        if n != len(vals) or n < 1:
            return None
        return (row[2].lower(), int(row[TS_FIELD]), int(row[RSSI_FIELD]),
                np.array(vals, dtype=np.float32))
    except Exception:
        # Deliberately broad. A corrupted UART line raised _csv.Error here, which is
        # not a ValueError, so it escaped the handler and killed the reader thread
        # outright -- the board then looked dead for the rest of the session while
        # being perfectly healthy. A parser for untrusted bytes must never be able to
        # take down its caller; an unparseable line is data loss, not a crash.
        return None


def removable_volume():
    """The recording drive: on macOS, the first writable external volume under
    /Volumes (the boot volume appears there as a symlink to / and is skipped).
    None when nothing is plugged in, or on Linux, where the rig writes into the
    bind-mounted repo."""
    if sys.platform != 'darwin' or not os.path.isdir('/Volumes'):
        return None
    for name in sorted(os.listdir('/Volumes')):
        path = os.path.join('/Volumes', name)
        try:
            if os.path.islink(path) or os.path.realpath(path) == '/':
                continue
            if not os.path.isdir(path) or not os.access(path, os.W_OK):
                continue
        except OSError:
            continue
        return path
    return None


def default_outdir():
    """Where captures go, in order: $CSI_DATA if set; a plugged-in flash drive on a
    Mac (<volume>/data -- recordings are made on the stick, not the laptop);
    /workspace/data inside the container; else <repo>/data.

    The container check comes before the __file__ walk because __file__ cannot
    identify the repo root inside the container: the repo is bind-mounted at
    /workspace, so that path is checked first and the walk is only the fallback
    for running directly on the host.
    """
    env = os.environ.get('CSI_DATA')
    if env:
        return env
    vol = removable_volume()
    if vol:
        return os.path.join(vol, 'data')
    if os.path.isdir('/workspace/data'):
        return '/workspace/data'
    return str(pathlib.Path(__file__).resolve().parents[1] / 'data')


def resolve_prefix(prefix, outdir):
    """Bare names (and subpaths) land under outdir; absolute paths are left alone."""
    if os.path.isabs(prefix):
        return prefix
    out = os.path.join(outdir, prefix)
    parent = os.path.dirname(out)
    if parent and not os.path.isdir(parent):
        # Ownership comes from outdir, not from `parent`: makedirs would have just
        # created parent as root, so asking who owns it always answers "root" and
        # the chown is skipped, leaving the user unable to delete their own data.
        own = owner_of(os.path.join(outdir, '_'))
        os.makedirs(parent, exist_ok=True)
        rel = os.path.relpath(parent, outdir)
        node = outdir
        for part in rel.split(os.sep):        # chown every level we just made
            node = os.path.join(node, part)
            give(node, own)
    return out


def owner_of(path):
    """(uid, gid) to give new files, or None when that is not our problem.

    The container runs as root because ESP-IDF and the serial ports need it, so
    everything it writes into the bind-mounted workspace lands root-owned and the
    user cannot delete their own captures. Match the directory being written into
    instead. Returns None when not root, or when the target is already ours.
    """
    if os.geteuid() != 0:
        return None
    try:
        st = os.stat(os.path.dirname(os.path.abspath(path)) or '.')
    except OSError:
        return None
    return None if st.st_uid == 0 else (st.st_uid, st.st_gid)


def give(path, own):
    if own:
        try:
            os.chown(path, *own)
        except OSError:
            pass


MAGIC = b'\xa5\x5a'
FRAME_TAIL = 2          # sum16
# (header bytes, bytes per subcarrier) by frame version. v1 is uint8 amplitude, v2 is
# raw int8 I/Q with the AGC gain in the two header bytes v1 spends on a clip counter.
# Both are parsed here rather than behind a mode flag: the 450 recorded takes are v1
# and must stay readable with the same code that reads whatever we record next.
# v3 widens the header to 24: raw (agc, fft) gain pair at 18-19 beside the Q8.8
# factor, first_word_invalid at 20. Same payload encoding as v2.
FRAME_FMT = {1: (18, 1), 2: (20, 2), 3: (24, 2)}


class CsiStream:
    """Splits one board's UART into binary CSI frames and ASCII lines.

    The link carries both: CSI is a binary frame (uint8 amplitudes, ~3x denser than
    the CSV it replaced) while ROLE_* markers and IDF logs stay text. Framing is
    magic + length + checksum precisely because the two are interleaved and a magic
    byte pair can occur inside a payload -- the checksum is what makes a false sync
    detectable, and a failed frame resyncs one byte on rather than discarding the
    buffer, so one bad byte costs one frame instead of everything after it.

    feed() returns (csi_records, text_lines). It never raises on malformed input:
    a parser for untrusted bytes that can throw takes its reader thread with it.

    A record is (tx_mac, board_us, rssi, amplitude, clipped, iq). `iq` is None for
    version 1 frames and a complex64 array for version 2; `amplitude` is filled in
    either way -- computed from I/Q when that is what arrived -- so every consumer
    that only wants magnitude works unchanged against both encodings.
    """

    def __init__(self):
        self.buf = bytearray()
        self.bad = 0

    def feed(self, data):
        self.buf += data
        recs, lines = [], []
        while True:
            i = self.buf.find(MAGIC)
            if i < 0:
                # no frame pending: everything complete up to the last newline is text
                nl = self.buf.rfind(b'\n')
                if nl >= 0:
                    lines += [ln for ln in self.buf[:nl].split(b'\n') if ln]
                    del self.buf[:nl + 1]
                # keep a byte back in case a magic pair straddles this read
                if len(self.buf) > 4096:
                    del self.buf[:-1]
                return recs, lines
            if i:
                head = self.buf[:i]
                nl = head.rfind(b'\n')
                if nl >= 0:
                    lines += [ln for ln in head[:nl].split(b'\n') if ln]
                del self.buf[:i]
                continue
            if len(self.buf) < 4:       # version and n_sub decide the frame's length
                return recs, lines
            fmt = FRAME_FMT.get(self.buf[2])
            n_sub = self.buf[3]
            if fmt is None or n_sub < 1:
                self.bad += 1
                del self.buf[:1]
                continue
            hdr, bps = fmt
            end = hdr + bps * n_sub
            total = end + FRAME_TAIL
            if len(self.buf) < total:
                return recs, lines
            frame = bytes(self.buf[:total])
            got = frame[end] | (frame[end + 1] << 8)
            want = sum(frame[2:end]) & 0xffff
            if got != want:
                self.bad += 1
                del self.buf[:1]        # resync past this magic, not past the payload
                continue
            mac = ':'.join(f'{b:02x}' for b in frame[4:10])
            rssi = int.from_bytes(frame[10:11], 'little', signed=True)
            lts = int.from_bytes(frame[12:16], 'little')
            if bps == 1:
                amp = np.frombuffer(frame, dtype=np.uint8, count=n_sub,
                                    offset=hdr).astype(np.float32)
                recs.append((mac, lts, rssi, amp, frame[16], None, None))
            else:
                # The board sends I/Q uncompensated and passes the AGC factor here as
                # Q8.8, because scaling on-board would round back into int8 and lose
                # the low bits phase depends on. Applying it host-side is free.
                gain_q8 = (frame[16] | (frame[17] << 8)) / 256.0
                # v3 reserves 0 as "AGC not yet calibrated" (the first ~100 frames
                # after boot or a retune). Ship those raw rather than scaled by a
                # made-up factor -- v2 shipped exactly that lie as 1.0x. The raw
                # field value travels in the record's metadata, sentinel intact.
                gain = gain_q8 if gain_q8 > 0 else 1.0
                pay = np.frombuffer(frame, dtype=np.int8, count=2 * n_sub, offset=hdr)
                iq = (pay[1::2].astype(np.float32)             # real
                      + 1j * pay[0::2].astype(np.float32))     # imag
                iq = (iq * gain).astype(np.complex64)
                # Saturated int8 samples: with AGC locked a loud scene clips here
                # silently, so the count rides in the slot v1 used for clipping.
                sat = int((np.abs(pay.astype(np.int16)) >= 127).sum())
                # (gain_q8, raw agc, raw fft): what a recording needs to recover the
                # exact wire int8s (iq / gain) and to segment at gain steps. v2
                # frames have no raw pair; zeros mark that honestly.
                if hdr == 24:
                    fft = frame[19] - 256 if frame[19] > 127 else frame[19]
                    gmeta = (gain_q8, frame[18], fft)
                else:
                    gmeta = (gain_q8, 0, 0)
                recs.append((mac, lts, rssi, np.abs(iq), sat, iq, gmeta))
            del self.buf[:total]


def unwrap_us(raw):
    """The board's microsecond counter is 32-bit and wraps every ~72 minutes.
    Untreated, a wrap mid-capture looks like a 71-minute jump backwards."""
    t = np.asarray(raw, dtype=np.float64)
    if t.size < 2:
        return t
    return t + np.concatenate([[0], np.cumsum(np.diff(t) < -TS_WRAP / 2)]) * TS_WRAP


# ----------------------------------------------------------------- the capture

class JpegWriter:
    """Background JPEG encoder pool writing <dir>/000000.jpg.

    Encoding runs off the grab loop on purpose: a stalled dequeue makes the V4L2
    driver drop frames silently, whereas a dropped JPEG is counted and the frame's
    timestamp still lands in the index. Shared by capture_synced and live_viewer so
    both produce identical frame directories.
    """

    def __init__(self, frame_dir, quality=85, threads=3, maxsize=120, own=None,
                 to_rgb=None, kind='rgb'):
        # kind 'rgb': JPEG of the colour frame. 'depth': the uint16 depth frame as a
        # lossless 16-bit PNG (000000.png) -- JPEG would invent depth values.
        self.dir, self.quality, self.own, self.kind = frame_dir, quality, own, kind
        # Default keeps every existing caller byte-identical; the viewer passes the
        # camera's own converter so a macOS session encodes BGR as BGR.
        self.to_rgb = to_rgb or yuyv_to_rgb
        os.makedirs(frame_dir, exist_ok=True)
        give(frame_dir, own)
        self.q = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self.threads = [threading.Thread(target=self._work, daemon=True)
                        for _ in range(threads)]
        for t in self.threads:
            t.start()

    def _work(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            idx, buf, w, h = item
            bio = io.BytesIO()
            if self.kind == 'depth':
                Image.fromarray(np.ascontiguousarray(buf, dtype=np.uint16)
                                ).save(bio, 'PNG', compress_level=1)
                path = os.path.join(self.dir, f'{idx:06d}.png')
            else:
                Image.fromarray(self.to_rgb(buf, w, h)).save(bio, 'JPEG', quality=self.quality)
                path = os.path.join(self.dir, f'{idx:06d}.jpg')
            with open(path, 'wb') as fh:
                fh.write(bio.getvalue())
            give(path, self.own)

    def submit(self, idx, buf, w, h):
        try:
            self.q.put_nowait((idx, buf, w, h))
        except queue.Full:
            self.dropped += 1

    def close(self, timeout=30):
        for _ in self.threads:
            self.q.put(None)
        for t in self.threads:
            t.join(timeout=timeout)


# Packet slots per link in a frame's saved CSI window. At the tuned round-robin
# schedule a 33 ms window holds 10-13 packets per link; 16 leaves room and keeps
# every take the same shape. A pinned transmitter delivers ~33 per link per
# window (every ping goes to every receiver), hence 48 there. Overflow is counted
# in meta, never silently dropped.
WINDOW_SLOTS = 16
WINDOW_SLOTS_PINNED = 48


def json_default(o):
    """NumPy scalars and arrays in meta become plain JSON; a NumPy int in a
    count field once made json.dumps raise and cost a session's writes."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'{o.__class__.__name__} is not JSON serializable')


def window_tensor(ft, half, boards, link_arrays, slots=WINDOW_SLOTS, pinned=None):
    """The per-frame CSI windows as fixed-shape, zero-padded arrays.

    For each frame i and each transmitter -> receiver pair (b, c) in `boards`
    order, the packets of that link within +-half of the frame time, oldest first,
    up to `slots` of them; everything else is 0 -- empty slots, links with no
    packet in the window, and the diagonal (a board never hears itself). Every
    packet keeps its metadata beside its samples:

      win_iq     int8   [F, B, B, slots, S, 2]  wire I/Q values (re, im); x win_gain
      win_a      f16    [F, B, B, slots, S]     amplitude |I + jQ| x gain, the same
                                                units as tx|rx|a (what the models eat)
      win_gain   f32    [F, B, B, slots]        Q8.8 AGC factor per packet (0 = pad)
      win_t      f32    [F, B, B, slots]        packet time minus frame time, seconds
      win_ts     f64    [F, B, B, slots]        packet host time from record start, s
      win_lts    f64    [F, B, B, slots]        receiving board's own clock, seconds
      win_rssi   int8   [F, B, B, slots]        RSSI, dBm
      win_agc    int16  [F, B, B, slots]        raw AGC gain index (v3 frames)
      win_fft    int16  [F, B, B, slots]        raw FFT gain (v3 frames)
      win_idx    int32  [F, B, B, slots]        index into that link's tx|rx|* arrays,
                                                -1 = padding
      win_count  u8     [F, B, B]               real packets in the window per link
      win_boards str    [B]                      the board labels in grid order

    And the same window flattened per RECEIVER, the model's input layout:

      csi        f16    [F, R, S, T]   amplitude; for receiver r every packet it
                                       heard in the window, from every transmitter,
                                       in time order along T, zero-padded
      csi_t      f32    [F, R, T]      packet time minus frame time, seconds (0 = pad)
      csi_tx     int8   [F, R, T]      index into win_boards of the transmitter, -1 = pad
      csi_count  u8     [F, R]         real packets per receiver
      csi_boards str    [R]            the receivers in order

    R and T follow the mode: round-robin, every board receives (R = B) from B-1
    transmitters (T = (B-1) x slots); a pinned transmitter, the other boards
    receive (R = B-1) from it alone (T = slots).

    `link_arrays` is {(tx_label, rx_label): dict of the link's per-packet arrays}
    with t relative to the same zero as ft; the int8 wire values are recovered
    exactly from the stored complex64 by dividing the gain back out.
    Returns (arrays dict, packets that did not fit).
    """
    ft = np.asarray(ft, dtype=np.float64)
    F, B = ft.size, len(boards)
    S = max((v['iq'].shape[1] for v in link_arrays.values()), default=0)
    idx = {b: i for i, b in enumerate(boards)}
    shape = (F, B, B, slots)
    out = dict(win_iq=np.zeros(shape + (S, 2), dtype=np.int8),
               win_a=np.zeros(shape + (S,), dtype=np.float16),
               win_gain=np.zeros(shape, dtype=np.float32),
               win_t=np.zeros(shape, dtype=np.float32),
               win_ts=np.zeros(shape, dtype=np.float64),
               win_lts=np.zeros(shape, dtype=np.float64),
               win_rssi=np.zeros(shape, dtype=np.int8),
               win_agc=np.zeros(shape, dtype=np.int16),
               win_fft=np.zeros(shape, dtype=np.int16),
               win_idx=np.full(shape, -1, dtype=np.int32),
               win_count=np.zeros((F, B, B), dtype=np.uint8),
               win_boards=np.array(boards))
    overflow = 0
    for (tx, rx), a in link_arrays.items():
        if tx not in idx or rx not in idx or a['iq'].shape[1] != S:
            continue
        bi, bj = idx[tx], idx[rx]
        t = np.asarray(a['t'], dtype=np.float64)
        n_pk = len(t)
        g = np.asarray(a.get('gain', np.ones(n_pk, dtype=np.float32)), dtype=np.float32)
        scale = np.where(g > 0, g, 1.0).astype(np.float32)
        raw = a['iq'] / scale[:, None]
        re = np.clip(np.rint(raw.real), -128, 127).astype(np.int8)
        im = np.clip(np.rint(raw.imag), -128, 127).astype(np.int8)
        rssi = np.asarray(a.get('rssi', np.zeros(n_pk)), dtype=np.int8)
        lts = np.asarray(a.get('lts', np.zeros(n_pk)), dtype=np.float64)
        agc = np.asarray(a.get('agc', np.zeros(n_pk)), dtype=np.int16)
        fft = np.asarray(a.get('fft', np.zeros(n_pk)), dtype=np.int16)
        lo = np.searchsorted(t, ft - half, side='left')
        hi = np.searchsorted(t, ft + half, side='right')
        for i in range(F):
            n = hi[i] - lo[i]
            if n <= 0:
                continue
            if n > slots:
                overflow += int(n - slots)
                n = slots
            sl = slice(lo[i], lo[i] + n)
            out['win_iq'][i, bi, bj, :n, :, 0] = re[sl]
            out['win_iq'][i, bi, bj, :n, :, 1] = im[sl]
            out['win_a'][i, bi, bj, :n] = np.abs(a['iq'][sl]).astype(np.float16)
            out['win_gain'][i, bi, bj, :n] = g[sl]
            out['win_t'][i, bi, bj, :n] = (t[sl] - ft[i]).astype(np.float32)
            out['win_ts'][i, bi, bj, :n] = t[sl]
            out['win_lts'][i, bi, bj, :n] = lts[sl]
            out['win_rssi'][i, bi, bj, :n] = rssi[sl]
            out['win_agc'][i, bi, bj, :n] = agc[sl]
            out['win_fft'][i, bi, bj, :n] = fft[sl]
            out['win_idx'][i, bi, bj, :n] = np.arange(lo[i], lo[i] + n, dtype=np.int32)
            out['win_count'][i, bi, bj] = n
    out.update(receiver_tensor(out, boards, slots, pinned))
    return out, overflow


def receiver_tensor(win, boards, slots=WINDOW_SLOTS, pinned=None):
    """The per-receiver layout (csi, csi_t, csi_tx, csi_count, csi_boards) from
    the tx x rx window arrays: for each receiver, its packets from every
    transmitter in the window, sorted by time, flattened along T."""
    F, B = win['win_count'].shape[:2]
    S = win['win_a'].shape[-1]
    if pinned is not None and pinned in boards:
        ti = boards.index(pinned)
        rxs = [j for j in range(B) if j != ti]
        txs_of = {j: [ti] for j in rxs}
        T = slots
    else:
        rxs = list(range(B))
        txs_of = {j: [i for i in range(B) if i != j] for j in rxs}
        T = (B - 1) * slots
    R = len(rxs)
    csi = np.zeros((F, R, S, T), dtype=np.float16)
    csi_t = np.zeros((F, R, T), dtype=np.float32)
    csi_tx = np.full((F, R, T), -1, dtype=np.int8)
    csi_count = np.zeros((F, R), dtype=np.uint8)
    for r, j in enumerate(rxs):
        for i in range(F):
            parts_t, parts_tx, parts_a = [], [], []
            for ti in txs_of[j]:
                n = int(win['win_count'][i, ti, j])
                if n:
                    parts_t.append(win['win_t'][i, ti, j, :n])
                    parts_tx.append(np.full(n, ti, dtype=np.int8))
                    parts_a.append(win['win_a'][i, ti, j, :n])
            if not parts_t:
                continue
            t = np.concatenate(parts_t)
            order = np.argsort(t, kind='stable')[:T]
            n = len(order)
            csi_t[i, r, :n] = t[order]
            csi_tx[i, r, :n] = np.concatenate(parts_tx)[order]
            csi[i, r, :, :n] = np.concatenate(parts_a)[order].T
            csi_count[i, r] = n
    return dict(csi=csi, csi_t=csi_t, csi_tx=csi_tx, csi_count=csi_count,
                csi_boards=np.array([boards[j] for j in rxs]))


def expected_cells(boards, pinned=None):
    """Which tx x rx cells of a window must hold packets: every off-diagonal cell
    in round-robin, only the pinned transmitter's row with a pinned one."""
    B = len(boards)
    cells = ~np.eye(B, dtype=bool)
    if pinned is not None and pinned in list(boards):
        row = np.zeros((B, B), dtype=bool)
        row[list(boards).index(pinned)] = True
        cells &= row
    return cells


def prune_frames(out, has_image, has_depth_image=None, pinned=None):
    """Which colour frames to keep: those with an image, with a depth frame (when
    the take has depth) within half a frame period that itself has an image, and
    whose CSI window has at least one packet on every link the mode has (all
    twelve in round-robin, the pinned transmitter's three otherwise). Returns
    (keep mask, counts of frames dropped for each reason). Works on the arrays
    `write_capture` builds, so prune_frames.py can apply the same rule to a take
    already on disk.
    """
    n = len(out['frame_t'])
    keep = np.asarray(has_image, dtype=bool).copy()
    dropped = dict(no_image=int((~keep).sum()), no_depth=0, bad_csi=0)
    if 'depth_t' in out and len(out['depth_t']) and 'frame_depth_dt' in out:
        half = frame_half_window(out['frame_t'])
        ok = np.abs(out['frame_depth_dt']) <= half
        if has_depth_image is not None:
            ok &= np.asarray(has_depth_image, dtype=bool)[np.searchsorted(
                out['depth_idx'], out['frame_depth_idx'])]
        dropped['no_depth'] = int((keep & ~ok).sum())
        keep &= ok
    elif 'depth_t' in out:
        dropped['no_depth'] = int(keep.sum())
        keep[:] = False
    if 'win_count' in out and n:
        wc = out['win_count']
        cells = expected_cells([str(b) for b in out['win_boards']], pinned)[None]
        ok = ~np.any((wc == 0) & cells, axis=(1, 2))
        dropped['bad_csi'] = int((keep & ~ok).sum())
        keep &= ok
    return keep, dropped


def apply_frame_mask(out, keep):
    """Filter every per-frame array in `out` (frame_*, win_*) to `keep`, and the
    depth arrays to the depth frames the kept colour frames refer to."""
    keep = np.asarray(keep, dtype=bool)
    for k in list(out):
        if (k.startswith('frame_') or k.startswith('win_') or k.startswith('csi')) \
                and k not in ('win_boards', 'csi_boards') \
                and hasattr(out[k], 'shape') and out[k].shape[:1] == keep.shape:
            out[k] = out[k][keep]
    if 'depth_idx' in out and 'frame_depth_idx' in out:
        used = np.isin(out['depth_idx'], np.unique(out['frame_depth_idx']))
        for k in ('depth_t', 'depth_t_ns', 'depth_idx'):
            if k in out:
                out[k] = out[k][used]
    return out


def assign_frames(out, links):
    """Per link, the nearest kept frame of every packet and the signed gap to it
    (`tx|rx|f`, `tx|rx|dt`, `tx|rx|dt_ns`); returns [(link, packets, worst gap ms)]."""
    ft = out['frame_t']
    ft_ns = out['frame_t_ns']
    report = []
    for k in links:
        t = packet_times(out, k)
        t_ns = out[f'{k}|tc_ns'] if f'{k}|tc_ns' in out else out[f'{k}|t_ns']
        worst = None
        if len(ft) >= 2:
            j = np.clip(np.searchsorted(ft, t), 1, len(ft) - 1)
            pick = np.where(np.abs(t - ft[j - 1]) <= np.abs(t - ft[j]), j - 1, j)
            out[f'{k}|f'] = pick.astype(np.int32)
            out[f'{k}|dt'] = (t - ft[pick]).astype(np.float32)
            out[f'{k}|dt_ns'] = t_ns - ft_ns[pick]
            worst = float(np.max(np.abs(t - ft[pick]))) * 1e3 if len(t) else None
        elif len(ft) == 1:
            out[f'{k}|f'] = np.zeros(len(t), dtype=np.int32)
            out[f'{k}|dt'] = (t - ft[0]).astype(np.float32)
            out[f'{k}|dt_ns'] = t_ns - ft_ns[0]
            worst = float(np.max(np.abs(t - ft[0]))) * 1e3 if len(t) else None
        report.append((k, len(t), worst))
    return report


def board_clock_times(out, links):
    """Packet times on the receiving board's own clock, mapped onto host time:
    for each receiver, offset = the 1st percentile of (host arrival - board time)
    over every packet it received -- the packets that arrived with the least
    delay define the mapping, and a host stall can only add delay, never remove
    it. Writes `tx|rx|tc` (float32 seconds from record start) and `tx|rx|tc_ns`
    beside the arrival times and returns {rx: offset}. Drift between the two
    clocks is ppm-level, nothing over a take."""
    by_rx = {}
    for k in links:
        rx = k.split('|')[1]
        by_rx.setdefault(rx, []).append(k)
    offsets = {}
    for rx, ks in by_rx.items():
        t = np.concatenate([out[f'{k}|t'].astype(np.float64) for k in ks])
        lts = np.concatenate([out[f'{k}|lts'] for k in ks])
        if len(t) < 10:
            continue
        off = float(np.percentile(t - lts, 1))
        offsets[rx] = off
        for k in ks:
            tc = out[f'{k}|lts'] + off
            out[f'{k}|tc'] = tc.astype(np.float32)
            out[f'{k}|tc_ns'] = np.rint(tc * 1_000_000_000).astype(np.int64)
    return offsets


def packet_times(out, k):
    """The packet times to window and assign by: board-clock times when the take
    has them, host arrival otherwise (takes written before 2026-09-16 evening)."""
    return out[f'{k}|tc'].astype(np.float64) if f'{k}|tc' in out \
        else out[f'{k}|t'].astype(np.float64)


def build_windows(out, links, meta, window_slots=WINDOW_SLOTS):
    """The per-frame windows and per-receiver layout from the packet arrays in
    `out`, for the frames in out['frame_t']. Returns (overflow, pinned label,
    half-width) and updates `out` and the window fields of `meta`."""
    ft = out['frame_t']
    lab = meta.get('boards', {})
    link_arrays = {}
    for k in links:
        if f'{k}|iq' not in out:
            continue
        tx, rx = k.split('|')
        link_arrays[(lab.get(tx, tx), lab.get(rx, rx))] = dict(
            t=packet_times(out, k), iq=out[f'{k}|iq'],
            gain=out.get(f'{k}|gain'), rssi=out.get(f'{k}|rssi'),
            lts=out.get(f'{k}|lts'), agc=out.get(f'{k}|agc'), fft=out.get(f'{k}|fft'))
    boards = sorted(set(lab.values())) if lab else sorted({b for pair in link_arrays for b in pair})
    half = frame_half_window(ft)
    mode = str(meta.get('mode', ''))
    pinned = mode if mode in boards else None
    if pinned is None and mode.startswith('fixed'):
        txs = {tx for tx, _rx in link_arrays}
        pinned = next(iter(txs)) if len(txs) == 1 else None
    if pinned is not None and window_slots == WINDOW_SLOTS:
        window_slots = WINDOW_SLOTS_PINNED
    overflow = 0
    if link_arrays and len(ft):
        wins, overflow = window_tensor(ft, half, boards, link_arrays, window_slots, pinned)
        out.update(wins)
    meta.update(window_half_s=half, window_slots=window_slots, window_overflow=overflow,
                packet_time='tc: receiving board clock mapped onto host time by its '
                            'least arrival delay (what windows and frame assignment '
                            'use); t: host arrival',
                window_layout='win_iq [frame, tx, rx, slot, subcarrier, (re, im)] int8 '
                              'wire values x win_gain; win_a [frame, tx, rx, slot, '
                              'subcarrier] float16 amplitude in tx|rx|a units; csi '
                              '[frame, receiver, subcarrier, T] the same amplitude per '
                              'receiver, every transmitter flattened in time order along '
                              'T with csi_t / csi_tx / csi_count beside it; per slot '
                              'also win_t (s from the frame), win_ts, win_lts, win_rssi, '
                              'win_agc, win_fft, win_idx (index into tx|rx|* arrays, -1 = '
                              'pad); win_count real packets per link; zeros are padding '
                              'and the diagonal')
    return overflow, pinned, half


def write_capture(prefix, recs, frames, t0, meta, own=None, depth_frames=None,
                  window_slots=WINDOW_SLOTS):
    """Write one self-contained capture and return (path, report, frame times).

    Beside the per-link packet arrays, every take carries the per-frame CSI
    windows (`win_*`, see window_tensor): each colour frame's +-half-period
    window as a transmitter x receiver grid of up to `window_slots` packets per
    link with their metadata, zero-padded, so a frame's CSI is one fixed-shape
    block to index.

    `depth_frames` [(index, wall_time), ...] is the depth stream recorded beside the
    colour one, its PNGs in <prefix>_depth/; it lands as `depth_t` / `depth_t_ns` /
    `depth_idx` plus `depth/000000.png` entries, and every colour frame gets the
    index of its nearest depth frame (`frame_depth_idx`, -1 if none) and the signed
    gap to it (`frame_depth_dt`, seconds) so the two streams pair without guessing.

    Kept module-level and shared so the live viewer and the headless recorder cannot
    drift into two different on-disk formats.

      recs   {rx_mac: [(host_time, tx_mac, board_us, rssi, amplitudes, iq,
              (gain_q8, agc, fft) | None), ...]}
      frames [(index, driver_sequence, wall_time), ...]

    `tx|rx|t` and `tx|rx|a` stay byte-compatible with capture_wizard.py. When the
    boards are sending I/Q (frame version 2) an extra complex64 `tx|rx|iq` is written
    alongside; `a` remains its magnitude, so a reader that only knows about amplitude
    sees exactly what it saw before and phase is additive rather than a new format.

    With v2/v3 frames three more per-packet arrays land beside `iq`: `gain` (the
    Q8.8 factor already multiplied into `iq` and `a`; 0.0 = the board's AGC was not
    yet calibrated and nothing was applied), and the raw `agc` / `fft` gain indices
    (v3 only; zeros under v2). Recordings are thereby complete: the exact wire int8
    I/Q is `iq / gain`, absolute amplitude can be re-referenced against any
    baseline, and phase can be segmented at gain steps.
    """
    out, links_written = {}, []
    ft = np.array([f[2] for f in frames], dtype=np.float64) - t0
    ft_ns = np.rint(ft * 1_000_000_000).astype(np.int64)
    out['frame_t'] = ft
    out['frame_t_ns'] = ft_ns
    out['frame_epoch'] = np.array([f[2] for f in frames], dtype=np.float64)
    out['frame_seq'] = np.array([f[1] for f in frames], dtype=np.int64)
    out['frame_idx'] = np.array([f[0] for f in frames], dtype=np.int32)
    if depth_frames:
        dt = np.array([f[1] for f in depth_frames], dtype=np.float64) - t0
        out['depth_t'] = dt
        out['depth_t_ns'] = np.rint(dt * 1_000_000_000).astype(np.int64)
        out['depth_idx'] = np.array([f[0] for f in depth_frames], dtype=np.int32)
        if len(ft):
            j = np.clip(np.searchsorted(dt, ft), 1, max(len(dt) - 1, 1))
            pick = np.where(np.abs(ft - dt[j - 1]) <= np.abs(ft - dt[np.minimum(j, len(dt) - 1)]),
                            j - 1, np.minimum(j, len(dt) - 1))
            out['frame_depth_idx'] = out['depth_idx'][pick]
            out['frame_depth_dt'] = (ft - dt[pick]).astype(np.float32)

    for rx, items in recs.items():
        by_tx = {}
        for t, tx, lts, rssi, amp, iq, gm in items:
            by_tx.setdefault(tx, []).append((t, lts, rssi, amp, iq, gm))
        for tx, seq in by_tx.items():
            if len(seq) < 5:
                continue
            # Truncated UART lines show up as a minority subcarrier count; keep
            # the modal width so the stack is rectangular.
            lens = {}
            for _t, _l, _r, x, _q, _g in seq:
                lens[len(x)] = lens.get(len(x), 0) + 1
            n = max(lens, key=lens.get)
            seq = [s for s in seq if len(s[3]) == n]
            t = np.array([s[0] for s in seq], dtype=np.float64) - t0
            t_ns = np.rint(t * 1_000_000_000).astype(np.int64)
            k = f'{tx}|{rx}'
            out[f'{k}|t'] = t.astype(np.float32)
            out[f'{k}|t_ns'] = t_ns
            out[f'{k}|a'] = np.stack([s[3] for s in seq])
            if all(s[4] is not None for s in seq):
                out[f'{k}|iq'] = np.stack([s[4] for s in seq]).astype(np.complex64)
            if all(s[5] is not None for s in seq):
                out[f'{k}|gain'] = np.array([s[5][0] for s in seq], dtype=np.float32)
                out[f'{k}|agc'] = np.array([s[5][1] for s in seq], dtype=np.int16)
                out[f'{k}|fft'] = np.array([s[5][2] for s in seq], dtype=np.int16)
            out[f'{k}|rssi'] = np.array([s[2] for s in seq], dtype=np.int16)
            lts_us = unwrap_us([s[1] for s in seq])
            out[f'{k}|lts'] = lts_us / 1e6
            out[f'{k}|lts_ns'] = np.rint(lts_us * 1000).astype(np.int64)
            links_written.append(k)

    meta = dict(meta)
    # Packet times on the boards' clocks, then the per-frame windows.
    meta['clock_offsets'] = board_clock_times(out, links_written)
    overflow, pinned, half = build_windows(out, links_written, meta, window_slots)
    # Drop the frames that are not a complete sample: no image on disk (the
    # encoder queue was full), no depth frame of their own, or a link missing from
    # their CSI window. The dropped images are removed with the rest.
    frame_dir = f'{prefix}_frames'
    depth_dir = f'{prefix}_depth'
    # Only our own zero-padded names: macOS drops AppleDouble "._000001.png" files
    # into folders on an exFAT stick, and one of those once failed a take's write.
    def frame_files(folder, pattern):
        return sorted(f for f in pathlib.Path(folder).glob(pattern) if f.stem.isdigit())
    all_jpgs = frame_files(frame_dir, '*.jpg') + frame_files(frame_dir, '*.png')
    ext = all_jpgs[0].suffix if all_jpgs else '.jpg'
    all_pngs = frame_files(depth_dir, '*.png') if depth_frames else []
    jpg_by_idx = {int(j.stem): j for j in all_jpgs}
    png_by_idx = {int(j.stem): j for j in all_pngs}
    n_recorded = len(ft)
    if n_recorded:
        keep, dropped = prune_frames(
            out, [i in jpg_by_idx for i in out['frame_idx']],
            [i in png_by_idx for i in out['depth_idx']] if 'depth_idx' in out else None,
            pinned)
        apply_frame_mask(out, keep)
    else:
        dropped = dict(no_image=0, no_depth=0, bad_csi=0)
    ft, ft_ns = out['frame_t'], out['frame_t_ns']
    report = assign_frames(out, links_written)
    jpgs = [jpg_by_idx[i] for i in out['frame_idx'] if i in jpg_by_idx]
    pngs = ([png_by_idx[i] for i in out['depth_idx'] if i in png_by_idx]
            if 'depth_idx' in out else [])
    meta.update(timestamp_origin='record_start', timestamp_unit='nanoseconds',
                timestamp_dtype='int64',
                frame_storage='npz:frames/{frame_idx:06d}' + ext,
                frames_recorded=n_recorded, frames_kept=int(len(ft)),
                frames_dropped=dropped)
    if depth_frames:
        meta.update(depth_storage='npz:depth/{depth_idx:06d}.png',
                    depth_frames=int(len(out.get('depth_t', []))), depth_dtype='uint16',
                    depth_unit='depth units, x depth_scale_m = metres')
    meta['frame_dir'] = None
    out['meta'] = np.array(json.dumps(meta, default=json_default))
    path = f'{prefix}.npz'
    tmp = f'{path}.tmp'
    # NPZ is a ZIP container. NumPy arrays remain ordinary .npy members, while the
    # already-compressed JPEGs are appended verbatim: one portable file without
    # wasting time or quality recompressing images.
    with open(tmp, 'wb') as fh:
        np.savez_compressed(fh, **out)
    with zipfile.ZipFile(tmp, 'a', compression=zipfile.ZIP_STORED) as zf:
        for jpg in jpgs:
            zf.write(jpg, f'frames/{jpg.name}')
        for png in pngs:
            zf.write(png, f'depth/{png.name}')
    # Validate the complete archive before it replaces the destination or any
    # temporary JPEG is removed. A failed stop therefore leaves recoverable files.
    with zipfile.ZipFile(tmp) as zf:
        if zf.testzip() is not None:
            raise OSError(f'capture archive failed CRC validation: {tmp}')
    os.replace(tmp, path)
    for jpg in all_jpgs:
        jpg.unlink()
    for png in all_pngs:
        png.unlink()
    for d in (frame_dir, depth_dir):
        try:
            for stray in pathlib.Path(d).glob('._*'):     # macOS AppleDouble files
                stray.unlink()
            pathlib.Path(d).rmdir()
        except OSError:
            pass
    give(path, own)
    return path, report, ft


# ------------------------------------------------------------------ the token

# The CSI that belongs to a camera frame: every packet within half a frame period
# of the frame's timestamp, on either side. Centred on the frame because the
# picture was taken at an instant and the channel around that instant is what
# explains it; half a period wide on each side so consecutive frames' windows tile
# the timeline without overlapping -- a packet belongs to exactly one frame. The
# half-width is derived from the frame timestamps themselves (frame_half_window),
# so it stays disjoint at whatever rate the camera actually ran; this constant is
# the 30 fps value, used only when there are too few frames to measure.
# Shared by the recorder's report, the viewer's coverage tile, check_session's cov%
# column and plot_frame_packets.
FRAME_WINDOW_S = 1 / 60


def frame_half_window(frame_t, fallback=FRAME_WINDOW_S):
    """Half the median frame interval: the widest centred window that keeps
    neighbouring frames' windows disjoint."""
    ft = np.asarray(frame_t, dtype=np.float64)
    if ft.size < 2:
        return fallback
    return 0.5 * float(np.median(np.diff(ft)))


def frame_window_counts(frame_t, t, half=None):
    """Packets of one link inside each frame's window: [n_frames] ints. `half` is
    the window's half-width in seconds; None derives the disjoint one."""
    ft = np.asarray(frame_t, dtype=np.float64)
    t = np.sort(np.asarray(t, dtype=np.float64))
    if half is None:
        half = frame_half_window(ft)
    lo = np.searchsorted(t, ft - half, side='left')
    hi = np.searchsorted(t, ft + half, side='right')
    return hi - lo


def frame_coverage(frame_t, link_t, links, half=None):
    """How many camera frames carry at least one packet on every link.

    A frame's packets are those within `half` seconds of its timestamp on either
    side (None: half the frame period, so windows are disjoint), on the same clock
    as the packet times (host arrival, relative to the same zero). Returns
    (fraction of frames in which EVERY link in `links` has a packet,
    {link: fraction of frames with a packet on that link}); NaN with no frames.

    This is the number the round-robin schedule is tuned for. Per-link rate is an
    average over a burst and a blind gap and says nothing about whether a given
    frame saw a given transmitter; this does. A link absent from `link_t` scores 0
    and drags the overall figure to 0, which is the honest answer.
    """
    ft = np.asarray(frame_t, dtype=np.float64)
    if ft.size < 1:
        return float('nan'), {lk: float('nan') for lk in links}
    if half is None:
        half = frame_half_window(ft)
    every = np.ones(ft.size, dtype=bool)
    per = {}
    for lk in links:
        hit = frame_window_counts(ft, link_t.get(lk, ()), half) > 0
        per[lk] = float(hit.mean())
        every &= hit
    return float(every.mean()), per


class TokenRing:
    """Host side of the round-robin transmit token.

    One board pings while every other board reports CSI for it; the token then
    moves on. Two ways to decide when it moves:

      burst > 0   count-based. Each turn is "TX <burst> <tag>" to one board, which
                  sends exactly that many pings and answers TX_DONE,<n>,<tag>; the
                  next turn is issued the moment that line arrives (or a timeout
                  passes). Every turn carries the same number of pings, the token
                  goes round in a few milliseconds, and every camera frame sees
                  every transmitter -- the property the dwell-based rotation lacked:
                  at 25 ms a board, four boards took ~100 ms to go round, against a
                  33 ms frame, so a frame saw one or two transmitters.
      burst == 0  the old timed dwell: "TX" to one board, wait `dwell`, stop it with
                  its RX line and start the next. Kept so the two can be measured
                  against each other on real hardware.

    Receivers are armed ONCE with "RX <every other board>", not re-told on every
    handoff. A handoff is therefore a single line to a single board, and the first
    ping of a turn cannot land on a receiver still filtering for the previous
    holder. A board that reboots mid-session comes up with an empty filter; each
    board gets its RX line again once a second, right after its own burst, so that
    heals within a second without a command ever landing mid-burst.

    A board that stops answering (three timeouts in a row) is tried once a second
    instead of every turn: otherwise its timeout would stretch every cycle and cost
    the other links their frame coverage too. A TX_DONE lifts the suspension.

    reader threads call on_line() with every text line from their board; the radio
    thread calls turn() in a loop. `completed`, `cycles` and `timeouts` are for the
    health display and the end-of-run report.
    """
    # On top of the burst's own duration. A turn on real hardware takes 2-4 ms
    # (p99 under 7 ms while recording), so this is the cost of a lost or late
    # TX_DONE: the ring stays silent for it, and every frame window inside that
    # silence loses its links. 50 ms cost two frames per stall; 20 ms keeps a stall
    # inside one frame. Simulated runs with spinning fake readers needed more, but
    # they are not the rig.
    TIMEOUT_MARGIN = 0.020
    SUSPEND_AFTER = 3           # consecutive timeouts before a board is backed off
    RETRY_S = 1.0               # how often a suspended board is tried again
    REARM_S = 1.0               # how often a board gets its RX list re-sent
    # Pipelined schedule: the last ping's airtime plus the firmware's own latency
    # inside a turn, and how many cycles ahead the boards are told their turns.
    # Five cycles is ~60 ms of slack for the host at the default settings: a
    # stall shorter than that never reaches the air (the firmware queues eight).
    AIR_S = 0.0004
    LOOKAHEAD_CYCLES = 5
    # The wire: each receiver's UART carries (boards - 1) x pings frames per cycle,
    # and the schedule must not outrun it. The gated mode never could -- waiting
    # for every TX_DONE throttled it to whatever the wire drained -- but a
    # pipelined schedule keeps its own time, and at 8% over the wire the boards'
    # 24 KB TX rings filled in a second, every TX_DONE queued ~75 ms behind the
    # frames and half were dropped. 85% of the raw byte rate is the measured knee.
    WIRE_UTIL = 0.85

    def __init__(self, boards, macs, stop, burst=2, dwell=0.025,
                 rate_hz=FW_DEFAULT_RATE, lock=None):
        self.boards = boards
        self.macs = list(macs)
        self.stop = stop
        self.burst = int(burst)
        self.dwell = float(dwell)
        self.rate_hz = float(rate_hz)
        # Seconds of silence between one board's last ping and the next board's
        # first. Gated: waited after the TX_DONE. Pipelined: built into the schedule.
        self.guard = 0.0
        # Pipelined (default for counted turns): every board is told its next
        # turns ahead of time as "TX n tag delay_us", and starts them on its own
        # timer; the host only keeps the schedule topped up LOOKAHEAD_CYCLES ahead
        # and reads the TX_DONE lines for the books. Gated: each turn is issued
        # when the previous board's TX_DONE arrives -- one host round trip per
        # turn, which a busy recording process cannot always make in time.
        self.pipelined = True
        self.frame_bytes = 26 + 2 * 117      # v3 header + I/Q pairs; updated from the wire
        self.next_cycle = None
        self.expect_multi = {m: {} for m in self.macs}   # mac -> {tag: due time}
        self.last_done = {}
        self.late_cycles = 0            # cycles the host issued after their start time
        self.lateness = deque(maxlen=4096)   # TX_DONE arrival minus scheduled turn start
        # Serial writes from more than one thread interleave at the byte level only
        # if a write is split; a line is one write here, but the viewer has other
        # writers (LED cues, STATS polls) and hands in its command lock.
        self.lock = lock if lock is not None else threading.Lock()
        self.done = {m: threading.Event() for m in self.macs}
        self.done_n = {}            # rx mac -> pings the board says it sent
        self.expect = {}            # rx mac -> tag its TX_DONE must carry to count
        self.armed = False
        self.armed_at = {m: 0.0 for m in self.macs}
        self.i = 0
        self.tag = 0
        self.prev = None            # timed mode: the board currently holding "TX"
        self.timeouts = 0
        self.misses = {m: 0 for m in self.macs}
        self.retry_at = {}          # suspended mac -> when to try it again
        self.completed = deque(maxlen=4096)     # (time, pings, tx) per finished burst
        self.cycles = deque(maxlen=20)          # seconds per full rotation
        self.cycle_started = None
        self.turns = deque(maxlen=4096)         # (issued, finished, tx, got_done)

    @property
    def suspended(self):
        return set(self.retry_at)

    def rx_command(self, mac, peers_only=False):
        """The RX line that makes `mac` a receiver for every other board; with
        peers_only the PEERS line, which sets the same list without cancelling
        bursts the board has already been scheduled."""
        others = [m.replace(':', '') for m in self.macs if m != mac]
        return (('PEERS ' if peers_only else 'RX ') + ','.join(others) + '\n').encode()

    def turn_time(self):
        """Seconds from one board's first ping to the next board's, pipelined:
        the burst plus the guard, or longer if the wire cannot drain that many
        frames per receiver."""
        nominal = max(self.burst - 1, 0) / max(self.rate_hz, 1.0) + self.AIR_S + self.guard
        n = len(self.macs)
        per_cycle = (n - 1) * max(self.burst, 1) * self.frame_bytes      # bytes per receiver
        wire = per_cycle / (self.WIRE_UTIL * BAUD / 10) / n
        return max(nominal, wire)

    def timeout(self):
        """How long a burst may take before the turn moves on without TX_DONE."""
        return max(self.burst - 1, 0) / max(self.rate_hz, 1.0) + self.TIMEOUT_MARGIN

    def _write(self, mac, data):
        with self.lock:
            self.boards[mac].write(data)

    def arm(self):
        """Tell every board who it may hear. False on a serial error."""
        now = time.time()
        try:
            for m in self.macs:
                self._write(m, self.rx_command(m))
                self.armed_at[m] = now
        except (serial.SerialException, OSError):
            return False
        # Let the lines land before the first TX: the four ports are independent
        # USB devices and the TX line can otherwise beat a receiver's RX line by a
        # fraction of a millisecond, losing the first ping of the first turn (seen
        # in a simulated run). Paid once per arming, not per turn.
        self.stop.wait(0.02)
        self.armed = True
        self.prev = None
        return True

    def disarm(self):
        """Forget the armed state: the next turn re-arms. Call when the boards were
        given other roles in between (pinned transmitter, parked, a survey)."""
        self.armed = False
        self.prev = None
        self.next_cycle = None
        self.expect.clear()
        for d in self.expect_multi.values():
            d.clear()
        for ev in self.done.values():
            ev.clear()

    def on_line(self, rx, text):
        """Every text line a reader sees goes through here; only TX_DONE matters."""
        m = TX_DONE_RE.match(text)
        if m is None:
            return
        tag = int(m.group(2))
        now = time.time()
        due = self.expect_multi.get(rx, {}).pop(tag, None)
        if due is not None:                     # a pipelined turn reporting in
            self.completed.append((now, int(m.group(1)), rx))
            self.lateness.append(now - due)
            self.turns.append((due, now, rx, True))
            self.misses[rx] = 0
            self.retry_at.pop(rx, None)
            last = self.last_done.get(rx)
            if last is not None and rx == self.macs[0]:
                self.cycles.append(now - last)
            self.last_done[rx] = now
            return
        if self.expect.get(rx) == tag:
            self.done_n[rx] = int(m.group(1))
            self.done[rx].set()

    def _next(self):
        """The next board to hand the token to, or None if every board is backed
        off and none is due yet. Also keeps the rotation-time history."""
        now = time.time()
        for _ in range(len(self.macs)):
            if self.i % len(self.macs) == 0:
                if self.cycle_started is not None:
                    self.cycles.append(now - self.cycle_started)
                self.cycle_started = now
            tx = self.macs[self.i % len(self.macs)]
            self.i += 1
            due = self.retry_at.get(tx)
            if due is None or now >= due:
                return tx
        return None

    def turn(self):
        """One handoff. False on a serial error, which ends the radio thread the
        same way it always has; everything else is absorbed and counted."""
        if self.stop.is_set():
            return True
        if not self.armed and not self.arm():
            return False
        if self.burst <= 0:
            return self._timed_turn()
        if self.pipelined:
            return self._pipelined_cycle()
        if self.prev is not None:
            # Switched from the timed dwell live: that holder pings until told to
            # stop, and nothing in the burst path would ever tell it.
            try:
                self._write(self.prev, self.rx_command(self.prev))
            except (serial.SerialException, OSError):
                return False
            self.armed_at[self.prev] = time.time()
            self.prev = None
        tx = self._next()
        if tx is None:
            self.stop.wait(0.05)
            return True
        self.tag = (self.tag + 1) & 0xffffffff
        ev = self.done[tx]
        ev.clear()
        self.expect[tx] = self.tag
        issued = time.time()
        try:
            self._write(tx, f'TX {self.burst} {self.tag}\n'.encode())
        except (serial.SerialException, OSError):
            return False
        ok = ev.wait(self.timeout())
        now = time.time()
        self.expect.pop(tx, None)
        if not ok and self.stop.is_set():
            # Shutting down: the readers have stopped delivering lines, so this is
            # not a lost TX_DONE and must not be saved as one.
            return True
        self.turns.append((issued, now, tx, ok))
        if not ok:
            self.timeouts += 1
            self.misses[tx] += 1
            if self.misses[tx] >= self.SUSPEND_AFTER:
                self.retry_at[tx] = now + self.RETRY_S
            return True
        self.completed.append((now, self.done_n.get(tx, self.burst), tx))
        self.misses[tx] = 0
        self.retry_at.pop(tx, None)
        if self.guard > 0:
            self.stop.wait(self.guard)
        if now - self.armed_at[tx] > self.REARM_S:
            # It has just finished its burst and holds nothing, so this cannot cut
            # a turn short. This is the path that re-arms a rebooted board.
            try:
                self._write(tx, self.rx_command(tx))
            except (serial.SerialException, OSError):
                return False
            self.armed_at[tx] = now
        return True

    def _pipelined_cycle(self):
        """Keep the boards' start queues topped up LOOKAHEAD_CYCLES ahead, then
        sleep until the next cycle is due to be issued. One call per cycle."""
        n = len(self.macs)
        T = self.turn_time()
        C = n * T
        now = time.time()
        if self.prev is not None:               # a timed-mode holder still pinging
            try:
                self._write(self.prev, self.rx_command(self.prev))
            except (serial.SerialException, OSError):
                return False
            self.prev = None
        if self.next_cycle is None:
            self.next_cycle = now + 0.003
        try:
            while self.next_cycle - now < self.LOOKAHEAD_CYCLES * C:
                base = self.next_cycle
                if base < now + 0.002:
                    # The host fell behind the schedule (a stall longer than the
                    # lookahead): resume from now rather than issuing turns in the
                    # past, and count it.
                    self.late_cycles += 1
                    base = now + 0.002
                for k, tx in enumerate(self.macs):
                    due = base + k * T
                    self.tag = (self.tag + 1) & 0xffffffff
                    delay_us = max(int((due - time.time()) * 1e6), 200)
                    self.expect_multi[tx][self.tag] = due
                    self._write(tx, f'TX {self.burst} {self.tag} {delay_us}\n'.encode())
                self.next_cycle = base + C
                now = time.time()
            # Turns that never reported. Counted, not acted on: a board's slot
            # costs the same whether it answers or not, and leaving a board out
            # for a second (the gated mode's back-off) silences its links for
            # far longer than any missed turn would. The slack is generous
            # because under recording load the host's readers, not the boards,
            # are what deliver a TX_DONE late; the schedule itself is unaffected.
            slack = T + self.LOOKAHEAD_CYCLES * C
            for tx in self.macs:
                for tag, due in list(self.expect_multi[tx].items()):
                    if now > due + slack:
                        del self.expect_multi[tx][tag]
                        self.turns.append((due, due + slack, tx, False))
                        self.timeouts += 1
                        self.misses[tx] += 1
            # Re-arm the receiver lists with PEERS, which leaves the queued bursts
            # alone (RX would cancel them).
            for tx in self.macs:
                if now - self.armed_at[tx] > self.REARM_S:
                    self._write(tx, self.rx_command(tx, peers_only=True))
                    self.armed_at[tx] = now
        except (serial.SerialException, OSError):
            return False
        wake = self.next_cycle - self.LOOKAHEAD_CYCLES * C - time.time()
        self.stop.wait(max(wake, 0.0005))
        return True

    def _timed_turn(self):
        tx = self._next()
        if tx is None:
            self.stop.wait(0.05)
            return True
        now = time.time()
        try:
            if self.prev is not None and self.prev != tx:
                # Stop the old holder before starting the new one; its RX line also
                # restores its filter list, so this doubles as its re-arm.
                self._write(self.prev, self.rx_command(self.prev))
                self.armed_at[self.prev] = now
            self._write(tx, b'TX\n')
        except (serial.SerialException, OSError):
            return False
        self.prev = tx
        self.stop.wait(self.dwell)
        return True


class Recorder:
    def __init__(self, args):
        self.args = args
        self.stop = threading.Event()
        self.boards = discover()
        if len(self.boards) < 2:
            raise SystemExit(f'need >= 2 boards, found {len(self.boards)}')
        self.macs = sorted(self.boards)
        self.recs = {m: [] for m in self.macs}
        self.read_errors = 0
        self.clipped = 0
        self.collecting = False
        self.ring = TokenRing(self.boards, self.macs, self.stop, burst=args.burst,
                              dwell=args.round_duration,
                              rate_hz=args.rate or FW_DEFAULT_RATE)
        self.ring.guard = args.guard / 1000.0
        self.ring.pipelined = args.schedule == 'pipelined'

        self.frames = []                      # (index, seq, wall time)
        self.prefix = resolve_prefix(args.prefix, args.outdir)
        self.frame_dir = f'{self.prefix}_frames'
        self.own = owner_of(self.prefix)
        self.cam = open_camera(args.device, args.width, args.height, args.fps,
                               depth=args.depth)
        # What a take's frames are. 'both' (default): colour JPEGs with the depth
        # frames beside them. 'colour': colour alone. 'depth': the depth frames
        # are THE frames (16-bit PNGs under frames/, the CSI windows anchored on
        # their timestamps; no colour at all).
        has_depth = bool(getattr(self.cam, 'has_depth', False))
        self.frame_kind = args.frames
        if self.frame_kind != 'colour' and not has_depth:
            raise SystemExit(f'--frames {args.frames} needs the depth stream '
                             f'({getattr(self.cam, "depth_reason", "none")}); '
                             f'use --frames colour')
        self.depth_only = self.frame_kind == 'depth'
        self.jpeg = JpegWriter(self.frame_dir, args.quality,
                               2 if self.depth_only else args.encoders, args.queue,
                               self.own, self.cam.to_rgb,
                               kind='depth' if self.depth_only else 'rgb')
        self.frames_lock = threading.Lock()
        # Depth beside colour ('both'): its own frame list and PNG writer.
        self.record_depth = self.frame_kind == 'both'
        self.depth_dir = f'{self.prefix}_depth'
        self.depth_frames = []
        self.depth_lock = threading.Lock()
        self.depth_writer = (JpegWriter(self.depth_dir, args.quality, 2, args.queue,
                                        self.own, kind='depth')
                             if self.record_depth else None)
        if self.frame_kind != 'colour':
            # Straight from the camera thread, every depth frame once, on its own
            # timestamp -- not sampled when a colour frame happens to be read.
            self.cam.on_depth = self.on_depth
        # A frame's CSI window reaches half a frame period before it; the first
        # frame is accepted only once every link has packets that far back, so
        # frame 0 never has an empty window.
        self.first_packet = {}
        # One offset, measured once: CLOCK_MONOTONIC and the wall clock drift apart
        # far too slowly to matter across a capture, and re-measuring per frame would
        # inject the very scheduling jitter the driver timestamp exists to avoid.
        self.mono_offset = float(np.median([time.time() - time.monotonic()
                                            for _ in range(9)]))

    def on_depth(self, frame, ts, seq=0):
        if not self.collecting or ts < self.t0 or not self.frames_ready(ts):
            return
        if self.depth_only:
            with self.frames_lock:
                k = len(self.frames)
                self.frames.append((k, seq, ts))
            self.jpeg.submit(k, frame, self.cam.depth_w, self.cam.depth_h)
            return
        with self.depth_lock:
            k = len(self.depth_frames)
            self.depth_frames.append((k, ts))
        self.depth_writer.submit(k, frame, self.cam.depth_w, self.cam.depth_h)

    def frames_ready(self, wall):
        """Whether a frame at `wall` has a full CSI window behind it: packets on
        every expected link at least half a period plus one token cycle earlier."""
        n = len(self.macs)
        want = n * (n - 1) if self.args.mode == 'roundrobin' else n - 1
        if len(self.first_packet) < want:
            return False
        return wall >= max(self.first_packet.values()) + FRAME_WINDOW_S + 0.015

    # ---- boards ----

    def set_leds(self, rgb):
        for m in self.macs:
            try:
                self.boards[m].write(f'LED {rgb[0]},{rgb[1]},{rgb[2]}\n'.encode())
            except (serial.SerialException, OSError):
                pass

    def clear_leds(self):
        for m in self.macs:
            try:
                self.boards[m].write(b'IDENT OFF\n')
            except (serial.SerialException, OSError):
                pass

    def reader(self, rx):
        """One board's byte stream. Only a serial error may end this loop: if it
        exits early the board is silent for the rest of the session while looking
        healthy on the wire, which is exactly how two sessions were lost."""
        ser = self.boards[rx]
        st = CsiStream()
        while not self.stop.is_set():
            try:
                data = ser.read(ser.in_waiting or 1)
            except (serial.SerialException, OSError):
                break
            except Exception:
                self.read_errors += 1
                continue
            if not data:
                continue
            try:
                recs, lines = st.feed(data)
            except Exception:
                self.read_errors += 1
                continue
            # The token's TX_DONE rides the same wire as the frames and must be seen
            # whether or not a take is being collected: the ring turns throughout.
            for ln in lines:
                self.ring.on_line(rx, ln.decode(errors='ignore').strip())
            if not self.collecting:
                continue
            now = time.time()
            for mac, lts, rssi, amp, clipped, iq, gmeta in recs:
                self.clipped += clipped
                self.recs[rx].append((now, mac, lts, rssi, amp, iq, gmeta))
                if (mac, rx) not in self.first_packet:
                    self.first_packet[(mac, rx)] = now
            if recs:
                self.ring.frame_bytes = 26 + 2 * len(recs[-1][3])
        self.read_errors += st.bad

    def radio(self):
        """fixedtx: one board transmits throughout. roundrobin: the TokenRing
        hands the token round until the recording stops."""
        if self.args.mode == 'fixedtx':
            # by label when given: macs[0] is just whichever MAC sorts first, which
            # silently pinned the wrong board when a specific one was wanted
            by_label = {LABEL.get(m[-5:], m): m for m in self.macs}
            tx = by_label.get(self.args.tx, self.macs[0])
            for m in self.macs:
                if m != tx:
                    self.boards[m].write(f'RX {tx.replace(":", "")}\n'.encode())
            self.boards[tx].write(b'TX\n')
            return
        while not self.stop.is_set():
            if not self.ring.turn():
                return

    # ---- camera ----

    def grabber(self):
        idx = 0
        while not self.stop.is_set():
            try:
                got = self.cam.read()
            except OSError:
                return          # camera closed under us during shutdown
            if got is None:
                continue
            try:
                seq, ts, buf = got
                if not self.collecting or self.depth_only:
                    continue          # depth-only takes get their frames from on_depth
                wall = frame_wall(self.cam, ts, self.mono_offset)
                if wall < self.t0 or not self.frames_ready(wall):
                    continue          # before the recording zero, or its window is not yet full
                with self.frames_lock:
                    self.frames.append((idx, seq, wall))
                self.jpeg.submit(idx, buf, self.cam.w, self.cam.h)
                idx += 1
            except Exception:
                # A dying grabber is a take with video silently missing; say so.
                import traceback
                traceback.print_exc()
                self.read_errors += 1

    # ---- run ----

    def run(self):
        a = self.args
        print(f'{len(self.boards)} boards: {[LABEL.get(m[-5:], m) for m in self.macs]}')
        if a.mode != 'roundrobin':
            sched = ''
        elif a.burst > 0:
            sched = f', {a.burst} ping(s) per turn, {a.guard:g} ms guard ({a.schedule})'
        else:
            sched = f', {a.round_duration * 1000:.0f} ms timed dwell'
        print(f'camera {a.device} {self.cam.w}x{self.cam.h} @ {self.cam.fps:g} fps, '
              f'mode {a.mode}{sched}')

        for board in self.boards.values():
            board.write(f'BAND {a.band}\n'.encode())
        time.sleep(0.5)
        if a.band == '5.6' and a.bw == 40:
            # The firmware enters 5 GHz at HT20 and widens as a separate step: a
            # direct band+HT40 transition leaves the ESP-NOW peer behind.
            for board in self.boards.values():
                board.write(b'BW 40\n')
            time.sleep(0.5)
        print(f'band {a.band} GHz (channel {120 if a.band == "5.6" else 13}, '
              f'{a.bw if a.band == "5.6" else 40} MHz)')

        if a.rate is not None:
            for board in self.boards.values():
                board.write(f'RATE {a.rate}\n'.encode())
            time.sleep(0.1)
            print(f'rate {a.rate} Hz')

        readers = []
        for m in self.macs:
            self.boards[m].reset_input_buffer()
            th = threading.Thread(target=self.reader, args=(m,), daemon=True)
            th.start()
            readers.append(th)
        threading.Thread(target=self.radio, daemon=True).start()
        self.cam.start()
        grab = threading.Thread(target=self.grabber, daemon=True)
        grab.start()

        # Let the radio settle and the camera's auto-exposure converge before the
        # clock starts: the first second of any UVC stream is not representative.
        self.set_leds(YELLOW)
        print(f'\nwarming up {a.warmup:.0f}s ...', flush=True)
        time.sleep(a.warmup)

        self.t0_ns = time.time_ns()
        self.t0 = self.t0_ns / 1e9
        self.collecting = True
        self.set_leds(GREEN)
        print(f'RECORDING {a.seconds:.0f}s  (Ctrl-C to stop early)')
        try:
            while time.time() - self.t0 < a.seconds and not self.stop.is_set():
                time.sleep(0.25)
                el = time.time() - self.t0
                n = sum(len(v) for v in self.recs.values())
                print(f'\r  {el:5.1f}s  {len(self.frames):5d} frames  '
                      f'{n:6d} CSI packets  ', end='', flush=True)
        except KeyboardInterrupt:
            print('\n  stopped early')
        print()

        self.collecting = False
        if self.frame_kind != 'colour':
            self.cam.on_depth = None
        self.stop.set()
        # Join before closing: the grabber can be parked in select() for up to its
        # timeout, and closing the fd underneath it would fault mid-ioctl.
        grab.join(timeout=3)
        self.cam.close()
        self.jpeg.close()
        if self.depth_writer is not None:
            self.depth_writer.close()
        self.set_leds(WHITE)
        time.sleep(0.2)
        self.clear_leds()
        # Readers must be out of readline() before the ports shut: pyserial sets its
        # fd to None on close, and a blocked read then dies on it mid-shutdown.
        # The 0.3s port timeout bounds how long that takes.
        for th in readers:
            th.join(timeout=2)
        for m in self.macs:
            try:
                self.boards[m].close()
            except (serial.SerialException, OSError):
                pass
        return self.save()

    # ---- output ----

    def save(self):
        a = self.args
        meta = dict(t0_epoch=self.t0, t0_epoch_ns=self.t0_ns,
                    mode=a.mode, round_duration=a.round_duration,
                    burst=a.burst if a.mode == 'roundrobin' else 0,
                    guard_ms=a.guard if a.mode == 'roundrobin' else 0,
                    ping_rate_hz=a.rate or FW_DEFAULT_RATE,
                    ring_timeouts=self.ring.timeouts,
                    ring_cycle_ms=(round(1000 * float(np.median(self.ring.cycles)), 2)
                                   if self.ring.cycles else None),
                    wifi_band_ghz=a.band,
                    wifi_channel=120 if a.band == '5.6' else 13,
                    wifi_bandwidth_mhz=a.bw if a.band == '5.6' else 40,
                    width=self.cam.w, height=self.cam.h, fps_requested=self.cam.fps,
                    frame_dir=self.frame_dir, jpeg_quality=a.quality,
                    boards={m: LABEL.get(m[-5:], '?') for m in self.macs},
                    driver_monotonic_ts=bool(self.cam.monotonic),
                    camera_wall_ts=bool(getattr(self.cam, 'wall_ts', False)),
                    camera=getattr(self.cam, 'name', str(a.device)),
                    dropped_encode=self.jpeg.dropped,
                    dropped_depth_encode=self.depth_writer.dropped if self.depth_writer else 0,
                    frames=self.frame_kind,
                    **camera_gt_meta(self.cam, 'depth' if self.depth_only else 'rgb'))
        path, report, ft = write_capture(self.prefix, self.recs, self.frames,
                                         self.t0, meta, self.own,
                                         depth_frames=self.depth_frames if self.record_depth else None)
        out = {'frame_seq': np.array([f[1] for f in self.frames], dtype=np.int64)}
        dur = ft[-1] - ft[0] if len(ft) > 1 else 0.0
        # Camera frame numbers, when the frames carry them; a stream without a
        # counter (all zeros) must not read as negative drops.
        gaps = int(np.sum(np.maximum(np.diff(out['frame_seq']) - 1, 0))) if len(ft) > 1 else 0
        print(f'\nwrote self-contained capture {path}')
        print(f'  {len(ft)} frames over {dur:.1f}s = {len(ft) / max(dur, 1e-9):.2f} fps '
              f'achieved (requested {self.cam.fps:g})')
        dropped = meta.get('frames_dropped') or {}
        if any(dropped.values()):
            print(f'  dropped {sum(dropped.values())} of {meta.get("frames_recorded")} '
                  f'frames: {dropped["no_image"]} without an image, {dropped["no_depth"]} '
                  f'without a depth frame, {dropped["bad_csi"]} with a link missing')
        if self.record_depth:
            print(f'  {len(self.depth_frames)} depth frames beside them'
                  + (f', {self.depth_writer.dropped} not encoded' if self.depth_writer.dropped else ''))
        if gaps:
            print(f'  {gaps} frames dropped by the driver (sequence gaps)')
        if self.jpeg.dropped:
            print(f'  {self.jpeg.dropped} frames not encoded (queue full) -- '
                  'timestamps kept, images missing')
        if self.read_errors:
            print(f'  {self.read_errors} unreadable/corrupt CSI frames rejected')
        if self.clipped:
            print(f'  WARNING: {self.clipped} amplitudes hit the uint8 ceiling (255) '
                  f'-- the binary encoding is losing dynamic range')
        if not self.cam.monotonic:
            print('  NOTE: driver did not report monotonic timestamps; frame times '
                  'fall back to userspace arrival and are less precise')
        # Which frames saw which links: the figure the schedule is tuned for.
        link_t = {}
        for rx, items in self.recs.items():
            for it in items:
                link_t.setdefault(f'{it[1]}|{rx}', []).append(it[0] - self.t0)
        if a.mode == 'roundrobin':
            want = [f'{tx}|{rx}' for tx in self.macs for rx in self.macs if tx != rx]
        else:
            txs = {k.split('|')[0] for k in link_t}
            want = [f'{tx}|{rx}' for tx in txs for rx in self.macs if tx != rx]
        half = frame_half_window(ft)
        cover, per_link = frame_coverage(ft, link_t, want, half)

        print(f'  per-link CSI rate, frames with a packet on the link (+-{1000 * half:.1f} ms '
              f'around each frame), worst frame gap:')
        for k, n, worst in sorted(report):
            tx, rx = k.split('|')
            nm = f'{LABEL.get(tx[-5:], tx[-5:])}->{LABEL.get(rx[-5:], rx[-5:])}'
            gap = f'worst gap to its frame {worst:6.1f} ms' if worst is not None \
                else 'no frames to assign to'
            seen = per_link.get(k, float('nan'))
            print(f'    {nm}  {n:5d} pkts  {n / max(dur, 1e-9):6.2f} Hz  '
                  f'{100 * seen:5.1f}% of frames   {gap}')
        if want:
            print(f'  frames with every link ({len(want)}) in their +-{1000 * half:.1f} ms '
                  f'window: {100 * cover:.1f}%')
        if a.mode == 'roundrobin':
            ring = self.ring
            cyc = (f'{1000 * float(np.median(ring.cycles)):.1f} ms median'
                   if ring.cycles else 'not measured')
            print(f'  token cycle {cyc}, {ring.timeouts} turns without TX_DONE'
                  + (f', {ring.late_cycles} cycles issued late by the host'
                     if ring.pipelined else '')
                  + (f', TX_DONE lateness p50/p99 {1000 * float(np.median(ring.lateness)):.1f}/'
                     f'{1000 * float(np.percentile(ring.lateness, 99)):.1f} ms'
                     if ring.pipelined and ring.lateness else ''))
            if a.burst > 0 and ring.timeouts:
                print('    (a turn without TX_DONE means the firmware on that board '
                      'does not know "TX <n>" -- reflash -- or the line was lost)')
            if cover == cover and cover < 0.95:
                print(f'\n  Only {100 * cover:.0f}% of frames saw every link. The token '
                      f'must go round faster than a frame ({1000 / self.cam.fps:.1f} ms):')
                if a.burst > 0:
                    print('  lower --burst (fewer pings per turn) or raise --rate '
                          '(same pings, shorter turn); --mode fixedtx samples 3 links '
                          'continuously.')
                else:
                    print('  use --burst N (count-based turns) instead of the timed '
                          '--round-duration dwell.')
        return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prefix', default='synced')
    ap.add_argument('--outdir', default=None,
                    help='directory for captures (default: the repo data/ folder)')
    ap.add_argument('--seconds', type=float, default=60.0)
    ap.add_argument('--device', default='auto',
                    help='RealSense colour node, or "auto" to find it by capability')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--rate', type=int, default=None,
                    help='firmware ping rate (pings/s while a board holds the token; '
                         'with counted turns this is only the spacing inside a turn). '
                         'Default 2000 at 5.6 GHz, 1200 at 2.4 GHz.')
    ap.add_argument('--band', default='5.6', choices=('2.4', '5.6'),
                    help='radio band; 5.6 requires ESP32-C5 boards')
    ap.add_argument('--bw', type=int, default=40, choices=(20, 40),
                    help='radio bandwidth at 5.6 GHz (the firmware enters 5.6 at 20 '
                         'MHz and is widened after); 117 complex subcarriers at 40')
    ap.add_argument('--quality', type=int, default=85)
    ap.add_argument('--depth', action='store_true',
                    help='open the RealSense through librealsense in this process '
                         '(Linux). On a Mac run tools/depth_server.py instead; its '
                         'colour + depth are taken automatically.')
    ap.add_argument('--frames', choices=('both', 'colour', 'depth'), default='both',
                    help='what a take\'s frames are: colour JPEGs with the depth frames '
                         'beside them (default; 3-D pose needs both), colour alone, or '
                         'the depth frames alone (16-bit PNGs)')
    ap.add_argument('--mode', choices=['roundrobin', 'fixedtx'], default='roundrobin')
    ap.add_argument('--tx', default=None,
                    help='with --mode fixedtx, which discovered board label transmits')
    ap.add_argument('--burst', type=int, default=None,
                    help='pings each board sends per turn before the token moves on '
                         '(firmware "TX <n>", gated on its TX_DONE reply). Measured '
                         '2026-09-16 on four C5 boards at 5.6 GHz / 40 MHz / RATE '
                         '2000 while recording video: 4 pings a turn goes round in '
                         '~12 ms and every 33 ms camera frame holds 5-10 packets on '
                         'every link at ~310 Hz per link with no loss; 6 gets ~330 Hz '
                         'but drops a link from the odd frame, 8 overruns the wire. At '
                         '2.4 GHz the band loses 10-20%% of packets whatever the '
                         'schedule and 2 pings a turn is the most that still covers '
                         'every frame. Default 4 at 5.6 GHz, 2 at 2.4. 0 = the old '
                         'timed dwell (--round-duration), which covers no frame at all.')
    ap.add_argument('--schedule', choices=('pipelined', 'gated'), default='pipelined',
                    help='pipelined: boards are told their turns ahead of time and '
                         'start them on their own timers (a busy host cannot leave '
                         'holes). gated: each turn is issued on the previous TX_DONE.')
    ap.add_argument('--guard', type=float, default=None,
                    help='ms of silence after a TX_DONE before the next board is '
                         'told to send. At 5.6 GHz 0.3-0.5 ms is enough; at 2.4 GHz '
                         'back-to-back transmitters collide and 1 ms is needed '
                         '(delivery 75%% at 0, 90%% at 1, 95%% at 3 ms there). Each ms '
                         'of guard adds 4 ms to the token cycle. Default 0.5 at 5.6 '
                         'GHz, 1.0 at 2.4.')
    ap.add_argument('--round-duration', type=float, default=0.025,
                    help='with --burst 0: seconds each board holds the transmit '
                         'token. Sets the blind gap between a link\'s bursts, not '
                         'its average rate; four boards at 25 ms go round in ~100 '
                         'ms, so a 33 ms frame sees only one or two transmitters.')
    ap.add_argument('--warmup', type=float, default=3.0)
    ap.add_argument('--encoders', type=int, default=3)
    ap.add_argument('--queue', type=int, default=120)
    args = ap.parse_args()
    if args.outdir is None:
        args.outdir = default_outdir()
    # The schedule that measured best on each band (NOTES.md, 2026-09-16); an
    # explicit value always wins.
    fast = args.band == '5.6'
    if args.rate is None:
        args.rate = 2000 if fast else 1200
    if args.burst is None:
        args.burst = 4 if fast else 2
    if args.guard is None:
        args.guard = 0.5 if fast else 1.0

    if args.device == 'auto':
        args.device = default_camera_device()
        if args.device is None:
            raise SystemExit('no RealSense colour node found. Is the camera plugged '
                             'in, and does this container have "c 81:* rmw"?')

    # The ring thread competes with the JPEG/PNG encoders and the camera threads for
    # the interpreter lock; the default 5 ms switch interval let it wait tens of ms
    # for a TX_DONE it had already received. Switch ten times as often.
    sys.setswitchinterval(0.0005)
    rec = Recorder(args)
    signal.signal(signal.SIGINT, lambda *_: rec.stop.set())
    rec.run()


if __name__ == '__main__':
    main()
