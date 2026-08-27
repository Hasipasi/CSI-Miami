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
from io import StringIO

import numpy as np
import serial
import serial.tools.list_ports
from PIL import Image

WCH_VID = 0x1A86
BAUD = 921600
BOOT_MAC_RE = re.compile(r'Board MAC ([0-9a-fA-F:]{17})')
N_META = 25
TS_FIELD = 18                      # local_timestamp, microseconds, per the firmware header
RSSI_FIELD = 3
TS_WRAP = 1 << 32                  # the board counter is 32-bit microseconds

# 'E' is the returned fifth board (ec:da:3b:4c:b8:d0 / 5C39018759). Its documented
# RX fault (11x same-pair asymmetry) did NOT reproduce on the 2026-08-24 matrix
# retest -- 100% as receiver, best of four -- but it keeps its own letter so it can
# never be confused with the original A, and the intermittent-fault suspicion in
# NOTES.md stays attached to this silicon, not to whatever slot it occupies.
LABEL = {'2d:3c': 'A', '6b:5c': 'B', 'ab:d4': 'C', '2d:a8': 'D', 'b8:d0': 'E'}

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
    idx, name = mac_camera_index()
    if idx is None:
        print('warning: could not identify the built-in camera; falling back to '
              'index 0. Pass --device <index or name> to choose explicitly.',
              file=sys.stderr, flush=True)
        return '0'
    return str(idx)


def open_camera(dev, w, h, fps):
    """A camera for `dev`, choosing the backend the device identifies.

    A path is a V4L2 node. A bare integer is an AVFoundation index, used as given so
    an explicit `--device 1` is never second-guessed. Anything else is a camera name
    to match on macOS, which is the stable way to ask for one: indices move when a
    Continuity camera comes and goes, names do not.
    """
    dev = str(dev)
    if dev.isdigit():
        return MacCamera(dev, w, h, fps)
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


def default_outdir():
    """Where captures go: <repo>/data, mounted into the container as /workspace/data.

    Checked in that order because __file__ cannot identify the repo root inside the
    container: the repo is bind-mounted at /workspace, so that path is checked first
    and the __file__ walk is only the fallback for running directly on the host.
    """
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
                 to_rgb=None):
        self.dir, self.quality, self.own = frame_dir, quality, own
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


def write_capture(prefix, recs, frames, t0, meta, own=None):
    """Write one self-contained capture and return (path, report, frame times).

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
    out, report = {}, []
    ft = np.array([f[2] for f in frames], dtype=np.float64) - t0
    ft_ns = np.rint(ft * 1_000_000_000).astype(np.int64)
    out['frame_t'] = ft
    out['frame_t_ns'] = ft_ns
    out['frame_epoch'] = np.array([f[2] for f in frames], dtype=np.float64)
    out['frame_seq'] = np.array([f[1] for f in frames], dtype=np.int64)
    out['frame_idx'] = np.array([f[0] for f in frames], dtype=np.int32)

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
            worst = None
            if len(ft) >= 2:
                # nearest frame in time, and how far off it was -- keeping the
                # signed gap means a consumer can reject packets that fall
                # between frames instead of trusting every assignment equally.
                j = np.clip(np.searchsorted(ft, t), 1, len(ft) - 1)
                pick = np.where(np.abs(t - ft[j - 1]) <= np.abs(t - ft[j]), j - 1, j)
                out[f'{k}|f'] = pick.astype(np.int32)
                out[f'{k}|dt'] = (t - ft[pick]).astype(np.float32)
                out[f'{k}|dt_ns'] = t_ns - ft_ns[pick]
                worst = float(np.max(np.abs(t - ft[pick]))) * 1e3
            elif len(ft) == 1:
                out[f'{k}|f'] = np.zeros(len(t), dtype=np.int32)
                out[f'{k}|dt'] = (t - ft[0]).astype(np.float32)
                out[f'{k}|dt_ns'] = t_ns - ft_ns[0]
                worst = float(np.max(np.abs(t - ft[0]))) * 1e3
            report.append((k, len(t), worst))

    meta = dict(meta)
    meta.update(timestamp_origin='record_start', timestamp_unit='nanoseconds',
                timestamp_dtype='int64',
                frame_storage='npz:frames/{frame_idx:06d}.jpg')
    meta['frame_dir'] = None
    out['meta'] = np.array(json.dumps(meta))
    path = f'{prefix}.npz'
    tmp = f'{path}.tmp'
    # NPZ is a ZIP container. NumPy arrays remain ordinary .npy members, while the
    # already-compressed JPEGs are appended verbatim: one portable file without
    # wasting time or quality recompressing images.
    with open(tmp, 'wb') as fh:
        np.savez_compressed(fh, **out)
    frame_dir = f'{prefix}_frames'
    jpgs = sorted(pathlib.Path(frame_dir).glob('*.jpg'))
    with zipfile.ZipFile(tmp, 'a', compression=zipfile.ZIP_STORED) as zf:
        for jpg in jpgs:
            zf.write(jpg, f'frames/{jpg.name}')
    # Validate the complete archive before it replaces the destination or any
    # temporary JPEG is removed. A failed stop therefore leaves recoverable files.
    with zipfile.ZipFile(tmp) as zf:
        if zf.testzip() is not None:
            raise OSError(f'capture archive failed CRC validation: {tmp}')
    os.replace(tmp, path)
    for jpg in jpgs:
        jpg.unlink()
    try:
        pathlib.Path(frame_dir).rmdir()
    except OSError:
        pass
    give(path, own)
    return path, report, ft


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

        self.frames = []                      # (index, seq, wall time)
        self.prefix = resolve_prefix(args.prefix, args.outdir)
        self.frame_dir = f'{self.prefix}_frames'
        self.own = owner_of(self.prefix)
        self.cam = open_camera(args.device, args.width, args.height, args.fps)
        self.jpeg = JpegWriter(self.frame_dir, args.quality, args.encoders,
                               args.queue, self.own, self.cam.to_rgb)

        # One offset, measured once: CLOCK_MONOTONIC and the wall clock drift apart
        # far too slowly to matter across a capture, and re-measuring per frame would
        # inject the very scheduling jitter the driver timestamp exists to avoid.
        self.mono_offset = float(np.median([time.time() - time.monotonic()
                                            for _ in range(9)]))

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
                recs, _lines = st.feed(data)
            except Exception:
                self.read_errors += 1
                continue
            if not self.collecting:
                continue
            now = time.time()
            for mac, lts, rssi, amp, clipped, iq, gmeta in recs:
                self.clipped += clipped
                self.recs[rx].append((now, mac, lts, rssi, amp, iq, gmeta))
        self.read_errors += st.bad

    def radio(self):
        """fixedtx: one board transmits throughout. roundrobin: rotate the role."""
        if self.args.mode == 'fixedtx':
            # by label when given: macs[0] is just whichever MAC sorts first, which
            # silently pinned the wrong board when a specific one was wanted
            by_label = {LABEL.get(m[-5:], m): m for m in self.macs}
            tx = by_label.get(self.args.tx, self.macs[0])
            self.boards[tx].write(b'TX\n')
            for m in self.macs:
                if m != tx:
                    self.boards[m].write(f'RX {tx.replace(":", "")}\n'.encode())
            return
        i = 0
        while not self.stop.is_set():
            tx = self.macs[i % len(self.macs)]
            i += 1
            try:
                self.boards[tx].write(b'TX\n')
                for m in self.macs:
                    if m != tx:
                        self.boards[m].write(f'RX {tx.replace(":", "")}\n'.encode())
            except (serial.SerialException, OSError):
                return
            self.stop.wait(self.args.round_duration)

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
            seq, ts, buf = got
            if not self.collecting:
                continue
            wall = ts + self.mono_offset if self.cam.monotonic else time.time()
            if wall < self.t0:
                continue              # queued before the shared recording zero
            self.frames.append((idx, seq, wall))
            self.jpeg.submit(idx, buf, self.cam.w, self.cam.h)
            idx += 1

    # ---- run ----

    def run(self):
        a = self.args
        print(f'{len(self.boards)} boards: {[LABEL.get(m[-5:], m) for m in self.macs]}')
        print(f'camera {a.device} {self.cam.w}x{self.cam.h} @ {self.cam.fps:g} fps, '
              f'mode {a.mode}' + (f', {a.round_duration * 1000:.0f} ms dwell'
                                  if a.mode == 'roundrobin' else ''))

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
        self.stop.set()
        # Join before closing: the grabber can be parked in select() for up to its
        # timeout, and closing the fd underneath it would fault mid-ioctl.
        grab.join(timeout=3)
        self.cam.close()
        self.jpeg.close()
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
                    width=self.cam.w, height=self.cam.h, fps_requested=self.cam.fps,
                    frame_dir=self.frame_dir, jpeg_quality=a.quality,
                    boards={m: LABEL.get(m[-5:], '?') for m in self.macs},
                    driver_monotonic_ts=bool(self.cam.monotonic),
                    dropped_encode=self.jpeg.dropped)
        path, report, ft = write_capture(self.prefix, self.recs, self.frames,
                                         self.t0, meta, self.own)
        out = {'frame_seq': np.array([f[1] for f in self.frames], dtype=np.int64)}
        dur = ft[-1] - ft[0] if len(ft) > 1 else 0.0
        gaps = int(np.sum(np.diff(out['frame_seq']) - 1)) if len(ft) > 1 else 0
        print(f'\nwrote self-contained capture {path}')
        print(f'  {len(ft)} frames over {dur:.1f}s = {len(ft) / max(dur, 1e-9):.2f} fps '
              f'achieved (requested {self.cam.fps:g})')
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
        print(f'  per-link CSI rate and worst frame gap:')
        for k, n, worst in sorted(report):
            tx, rx = k.split('|')
            nm = f'{LABEL.get(tx[-5:], tx[-5:])}->{LABEL.get(rx[-5:], rx[-5:])}'
            gap = f'worst gap to its frame {worst:6.1f} ms' if worst is not None \
                else 'no frames to assign to'
            print(f'    {nm}  {n:5d} pkts  {n / max(dur, 1e-9):6.2f} Hz   {gap}')
        if report:
            per = np.mean([n / max(dur, 1e-9) for _k, n, _w in report])
            if per < self.cam.fps:
                print(f'\n  Per-link CSI ({per:.1f} Hz) is slower than the frame rate '
                      f'({self.cam.fps:g} fps), so most frames have no fresh sample on')
                print('  any given link. That is inherent to sharing one channel across '
                      'links, not a fault:')
                print('  shorten --round-duration to raise it, or use --mode fixedtx '
                      'to sample 3 links at full rate.')
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
                    help='set the firmware ping rate before recording')
    ap.add_argument('--quality', type=int, default=85)
    ap.add_argument('--mode', choices=['roundrobin', 'fixedtx'], default='roundrobin')
    ap.add_argument('--tx', default=None,
                    help='with --mode fixedtx, which board transmits (A/B/C/D)')
    ap.add_argument('--round-duration', type=float, default=0.025,
                    help='seconds each board holds the transmit token. This sets the '
                         'blind gap between a link\'s bursts, not its average rate. '
                         'The optimum depends on frame size: 25 ms with 166 '
                         'subcarriers (354 B = 3.84 ms of UART each, so a shorter '
                         'dwell fits too few frames), 12.5 ms with 30. Measured.')
    ap.add_argument('--warmup', type=float, default=3.0)
    ap.add_argument('--encoders', type=int, default=3)
    ap.add_argument('--queue', type=int, default=120)
    args = ap.parse_args()
    if args.outdir is None:
        args.outdir = default_outdir()

    if args.device == 'auto':
        args.device = default_camera_device()
        if args.device is None:
            raise SystemExit('no RealSense colour node found. Is the camera plugged '
                             'in, and does this container have "c 81:* rmw"?')

    rec = Recorder(args)
    signal.signal(signal.SIGINT, lambda *_: rec.stop.set())
    rec.run()


if __name__ == '__main__':
    main()
