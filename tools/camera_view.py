#!/usr/bin/env python3
"""Colour + depth, live, and nothing else -- for aiming the camera.

The full viewer needs the boards and half a screen of CSI panels; placing a camera
only needs the picture. Same RealSense path as the recorder (capture.open_camera),
so what this shows is what a capture would record.

  docker compose run --rm -e QT_X11_NO_MITSHM=1 --name csi_cam csi \
    bash -lc "cd /workspace/tools && exec python3 camera_view.py"

Press G for a rule-of-thirds grid with a centre cross (the alignment aid), Q to quit.
"""
import argparse
import sys
import time

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from capture import default_camera_device, open_camera


class View(QtWidgets.QWidget):
    def __init__(self, cam, args):
        super().__init__()
        self.cam, self.grid, self.times = cam, False, []
        self.setWindowTitle('camera + depth')
        self.colour, self.depth = QtWidgets.QLabel(), QtWidgets.QLabel()
        self.status = QtWidgets.QLabel()
        self.status.setStyleSheet('color:#ddd; font-size:15px;')
        lay = QtWidgets.QVBoxLayout(self)
        pics = QtWidgets.QHBoxLayout()
        for w in (self.colour, self.depth):
            w.setAlignment(QtCore.Qt.AlignCenter)
            w.setMinimumSize(320, 240)
            pics.addWidget(w, 1)
        lay.addLayout(pics, 1)
        lay.addWidget(self.status)
        self.setStyleSheet('background:#1b1b1f;')
        cam.start()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(int(1000 / max(args.fps, 1)))

    def draw(self, label, rgb):
        h, w, _ = rgb.shape
        img = QtGui.QImage(np.ascontiguousarray(rgb).data, w, h, 3 * w,
                           QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img).scaled(
            label.width(), label.height(), QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation)
        if self.grid:
            p = QtGui.QPainter(pix)
            p.setPen(QtGui.QPen(QtGui.QColor('#ffcc00'), 1))
            pw, ph = pix.width(), pix.height()
            for i in (1, 2):
                p.drawLine(pw * i // 3, 0, pw * i // 3, ph)
                p.drawLine(0, ph * i // 3, pw, ph * i // 3)
            p.setPen(QtGui.QPen(QtGui.QColor('#ff3b3b'), 2))
            p.drawLine(pw // 2 - 12, ph // 2, pw // 2 + 12, ph // 2)
            p.drawLine(pw // 2, ph // 2 - 12, pw // 2, ph // 2 + 12)
            p.end()
        label.setPixmap(pix)

    def tick(self):
        got = self.cam.read()
        if got is None:
            return
        _seq, _ts, buf = got
        self.times = [t for t in self.times if t > time.time() - 2] + [time.time()]
        self.draw(self.colour, self.cam.to_rgb(buf, self.cam.w, self.cam.h))
        d = getattr(self.cam, 'depth', None)
        if d is None:
            self.depth.setText('no depth stream\n(start with --depth)')
            self.depth.setStyleSheet('color:#888; font-size:16px;')
        else:
            self.draw(self.depth, self.cam.depth_rgb())
        fps = len(self.times) / 2.0
        near = f' · nearest {np.min(d[0][d[0] > 0]) * self.cam.depth_scale:.2f} m' if d is not None and (d[0] > 0).any() else ''
        self.status.setText(f'{self.cam.name} · {self.cam.w}x{self.cam.h} · {fps:.0f} fps{near}'
                            f'   ·   G: grid   Q: quit')

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_G:
            self.grid = not self.grid
        elif e.key() in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
            self.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)   # 16:9, same frame as depth
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--depth-size', default='1280x720',
                    help='depth WxH. 1280x720 and 848x480 both give the stereo '
                         'module its full 89.0 x 57.9 deg; 640x480 is a crop, only '
                         '78.6 deg wide. There is no 1280x800 depth (colour/IR only)')
    ap.add_argument('--no-depth', dest='depth', action='store_false',
                    help='colour only (librealsense not needed)')
    args = ap.parse_args()
    dev = default_camera_device() if args.device == 'auto' else args.device
    dw, dh = (int(x) for x in args.depth_size.lower().split('x'))
    cam = open_camera(dev, args.width, args.height, args.fps, depth=args.depth,
                      depth_size=(dw, dh))
    print(f'camera {cam.name} {cam.w}x{cam.h} @ {cam.fps:g} fps', flush=True)
    app = QtWidgets.QApplication(sys.argv)
    v = View(cam, args)
    v.resize(1500, 560)
    v.show()
    try:
        app.exec_()
    finally:
        cam.close()


if __name__ == '__main__':
    main()
