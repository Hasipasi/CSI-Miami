#!/usr/bin/env python3
"""Grab frames straight off a V4L2 node with no SDK, to prove the camera works.

librealsense is not installed anywhere yet, so this deliberately bypasses it:
if raw V4L2 delivers frames, the camera, cable, USB link and permissions are all
good, and anything that fails later is an SDK/plumbing problem rather than hardware.
"""
import ctypes
import fcntl
import mmap
import os
import sys
import time

VIDIOC_S_FMT = 0xc0d05605
VIDIOC_REQBUFS = 0xc0145608
VIDIOC_QUERYBUF = 0xc0585609
VIDIOC_QBUF = 0xc058560f
VIDIOC_DQBUF = 0xc0585611
VIDIOC_STREAMON = 0x40045612
VIDIOC_STREAMOFF = 0x40045613
CAPTURE, MMAP = 1, 1


def fourcc(s):
    return sum(ord(c) << (8 * i) for i, c in enumerate(s))


class PixFormat(ctypes.Structure):
    _fields_ = [('width', ctypes.c_uint32), ('height', ctypes.c_uint32),
                ('pixelformat', ctypes.c_uint32), ('field', ctypes.c_uint32),
                ('bytesperline', ctypes.c_uint32), ('sizeimage', ctypes.c_uint32),
                ('colorspace', ctypes.c_uint32), ('priv', ctypes.c_uint32),
                ('flags', ctypes.c_uint32), ('enc', ctypes.c_uint32),
                ('quantization', ctypes.c_uint32), ('xfer_func', ctypes.c_uint32)]


class Format(ctypes.Structure):
    # the fmt union is 8-aligned (it holds a pointer), so it starts at offset 8
    _fields_ = [('type', ctypes.c_uint32), ('_pad', ctypes.c_uint32),
                ('pix', PixFormat), ('_rest', ctypes.c_uint8 * (200 - 48))]


class ReqBufs(ctypes.Structure):
    _fields_ = [('count', ctypes.c_uint32), ('type', ctypes.c_uint32),
                ('memory', ctypes.c_uint32), ('capabilities', ctypes.c_uint32),
                ('flags', ctypes.c_uint8), ('reserved', ctypes.c_uint8 * 3)]


class TimeVal(ctypes.Structure):
    _fields_ = [('sec', ctypes.c_long), ('usec', ctypes.c_long)]


class Buffer(ctypes.Structure):
    _fields_ = [('index', ctypes.c_uint32), ('type', ctypes.c_uint32),
                ('bytesused', ctypes.c_uint32), ('flags', ctypes.c_uint32),
                ('field', ctypes.c_uint32), ('_pad', ctypes.c_uint32),
                ('timestamp', TimeVal), ('timecode', ctypes.c_uint8 * 16),
                ('sequence', ctypes.c_uint32), ('memory', ctypes.c_uint32),
                ('offset', ctypes.c_uint32), ('_pad2', ctypes.c_uint32),
                ('length', ctypes.c_uint32), ('reserved2', ctypes.c_uint32),
                ('request_fd', ctypes.c_int32), ('_tail', ctypes.c_uint32)]


def grab(dev, cc, w, h, n=30):
    fd = os.open(dev, os.O_RDWR)
    f = Format(type=CAPTURE)
    f.pix.width, f.pix.height, f.pix.pixelformat, f.pix.field = w, h, fourcc(cc), 1
    fcntl.ioctl(fd, VIDIOC_S_FMT, f)
    # fourccs are space-padded to 4 chars ('Z16 '), so strip before comparing
    got = (f.pix.width, f.pix.height,
           ''.join(chr((f.pix.pixelformat >> (8 * k)) & 0xff) for k in range(4)).strip())

    r = ReqBufs(count=4, type=CAPTURE, memory=MMAP)
    fcntl.ioctl(fd, VIDIOC_REQBUFS, r)
    maps = []
    for i in range(r.count):
        b = Buffer(index=i, type=CAPTURE, memory=MMAP)
        fcntl.ioctl(fd, VIDIOC_QUERYBUF, b)
        maps.append(mmap.mmap(fd, b.length, mmap.MAP_SHARED,
                              mmap.PROT_READ | mmap.PROT_WRITE, offset=b.offset))
        fcntl.ioctl(fd, VIDIOC_QBUF, b)

    fcntl.ioctl(fd, VIDIOC_STREAMON, ctypes.c_int(CAPTURE))
    stamps, last = [], None
    t0 = time.time()
    for _ in range(n):
        b = Buffer(type=CAPTURE, memory=MMAP)
        fcntl.ioctl(fd, VIDIOC_DQBUF, b)
        stamps.append(b.timestamp.sec + b.timestamp.usec / 1e6)
        last = bytes(maps[b.index][:b.bytesused])
        fcntl.ioctl(fd, VIDIOC_QBUF, b)
    wall = time.time() - t0
    fcntl.ioctl(fd, VIDIOC_STREAMOFF, ctypes.c_int(CAPTURE))
    for m in maps:
        m.close()
    os.close(fd)

    d = [stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)]
    return got, last, n / wall, (min(d), max(d))


def main():
    dev, cc, w, h, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    (gw, gh, gcc), frame, fps, (dmin, dmax) = grab(dev, cc, w, h)
    print(f'{dev}: negotiated {gw}x{gh} {gcc}, {len(frame)} B/frame, '
          f'{fps:.1f} fps, inter-frame {dmin * 1e3:.1f}-{dmax * 1e3:.1f} ms')

    import numpy as np
    if gcc == 'YUYV':                       # luma is every other byte
        img = np.frombuffer(frame, np.uint8).reshape(gh, gw, 2)[:, :, 0]
    elif gcc == 'Z16':                      # depth in mm
        img = np.frombuffer(frame, np.uint16).reshape(gh, gw)
        mid = img[gh // 2 - 20:gh // 2 + 20, gw // 2 - 20:gw // 2 + 20]
        valid = mid[mid > 0]
        print(f'  centre patch: {valid.size * 100 // mid.size}% valid, '
              f'median range {np.median(valid) / 1000 if valid.size else float("nan"):.2f} m')
        img = (np.clip(img, 0, 4000) / 4000 * 255).astype(np.uint8)
    else:
        img = np.frombuffer(frame, np.uint8)[:gh * gw].reshape(gh, gw)
    print(f'  luma mean {img.mean():.1f}, std {img.std():.1f} '
          f'({"real image" if img.std() > 3 else "FLAT - lens capped?"})')
    with open(out, 'wb') as fh:
        fh.write(b'P5\n%d %d\n255\n' % (gw, gh))
        fh.write(img.tobytes())
    print(f'  wrote {out}')


main()
