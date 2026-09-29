#!/usr/bin/env python3
"""NLF on the live camera, a short recording, and the fitted 3-D result, in one go.

  preview     NLF on the camera as it runs: the skeleton on the picture, and a
              top-down panel putting the person inside the board square
  REC (or R)  record a segment (10 s by default, or press again to stop early)
  processing  the recorded take straight through the normal pipeline, progress
              at the bottom: extract_poses3d.py -> fit_body.py -> render_body_gif.py
  result      a second window looping the fitted body -- the camera frame with its
              skeleton, and the metric joints from the front, the side and above,
              with the rig drawn in.
              Close it and the preview carries on.

The three tools run as subprocesses rather than reimplemented here, so what you
watch is exactly what a batch run produces. The live model is dropped while they
run: two copies of NLF do not fit in 6 GB of VRAM. No boards are needed -- the
take has no CSI. Everything 3-D is in the room frame (body_common): floor at
Y = 0, origin at the centre of the array, 2.12 m in front of the camera. The
camera's tilt is not measured -- pass --pitch if the feet float or sink.

  docker compose run --rm -e QT_X11_NO_MITSHM=1 pose python3 tools/live_body.py
"""
import argparse
import os
import re
import subprocess
import sys
import threading
import time

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from body_common import (BOARD_H, BOARDS, CAM_H, CAMERA_XZ, HALF_DIAG,   # noqa: E402
                         K_of, load_calib, to_room)
from capture import (JpegWriter, camera_gt_meta, default_camera_device,  # noqa: E402
                     default_outdir, frame_wall, open_camera, resolve_prefix,
                     set_depth_range, write_capture)
from extract_poses3d import DEFAULT_MODEL, detect, load_model           # noqa: E402

TOOLS = os.path.dirname(os.path.abspath(__file__))
# SMPL's kinematic tree: joint -> its parent. Drawing bones needs nothing else.
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17,
                18, 19, 20, 21]
BONE, JOINT = QtGui.QColor('#39d353'), QtGui.QColor('#ffcc00')
ROD, NODE, CAM = QtGui.QColor('#4a5160'), QtGui.QColor('#7dd3fc'), QtGui.QColor('#e8710a')
IDLE, RECORDING, PROCESSING = 'idle', 'recording', 'processing'
ARENA = HALF_DIAG + 0.5          # half-width of the top-down panel, metres


class TopDown(QtWidgets.QWidget):
    """Where the person stands inside the array, looking straight down.

    The camera picture cannot show this -- it is the geometry the CSI links
    actually see, so it is worth watching while someone places themselves. Room
    frame from body_common: origin on the floor at the array centre, X right,
    Z away from the camera (up the panel).
    """

    def __init__(self):
        super().__init__()
        self.pts = None                      # [24, 3] room-frame joints, or None
        self.setMinimumWidth(320)

    def update_points(self, pts):
        self.pts = pts
        self.update()

    def paintEvent(self, _e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor('#101014'))
        side = min(self.width(), self.height())
        s = side / (2 * ARENA)
        ox, oy = self.width() / 2, self.height() / 2

        def xy(x, z):                        # room metres -> panel pixels, Z up
            return QtCore.QPointF(ox + x * s, oy - z * s)

        p.setPen(QtGui.QPen(ROD, 1.4, QtCore.Qt.DashLine))
        ring = [BOARDS[k] for k in 'ABCD']
        p.drawPolygon(QtGui.QPolygonF([xy(*c) for c in ring]))
        p.setPen(QtGui.QPen(QtGui.QColor('#2a2a33'), 1))
        p.drawLine(xy(-0.15, 0), xy(0.15, 0))
        p.drawLine(xy(0, -0.15), xy(0, 0.15))
        f = p.font()
        f.setPointSize(9)
        f.setBold(True)
        p.setFont(f)
        for k, (x, z) in BOARDS.items():
            p.setPen(QtGui.QPen(NODE, 1))
            p.setBrush(NODE)
            p.drawEllipse(xy(x, z), 4.5, 4.5)
            p.drawText(xy(x, z) + QtCore.QPointF(7, -6), k)
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(CAM, 1.8))
        cx, cz = CAMERA_XZ
        p.drawPolyline(QtGui.QPolygonF([xy(cx - 0.22, cz), xy(cx, cz + 0.3),
                                        xy(cx + 0.22, cz)]))
        p.setPen(QtGui.QPen(QtGui.QColor('#9a9aa5'), 1))
        f.setBold(False)
        p.setFont(f)
        p.drawText(6, self.height() - 8, f'{BOARD_H:.2f} m antennas · camera {CAM_H:.2f} m')
        if self.pts is None:
            p.setPen(QtGui.QPen(QtGui.QColor('#6a6a78'), 1))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, 'no 3-D pose')
            return
        q = self.pts
        # the footprint first: seen from above a skeleton is a knot of dots, and
        # where the person stands is the whole point of this panel
        pelvis = q[0]
        if np.isfinite(pelvis).all():
            p.setPen(QtGui.QPen(QtGui.QColor('#39d353'), 1))
            p.setBrush(QtGui.QColor(57, 211, 83, 40))
            p.drawEllipse(xy(pelvis[0], pelvis[2]), 0.25 * s, 0.25 * s)
        p.setPen(QtGui.QPen(BONE, 2.5))
        for j, parent in enumerate(SMPL_PARENTS):
            if parent >= 0 and np.isfinite(q[[j, parent]]).all():
                p.drawLine(xy(q[j][0], q[j][2]), xy(q[parent][0], q[parent][2]))
        p.setPen(QtGui.QPen(JOINT, 1))
        p.setBrush(JOINT)
        for j in q[np.isfinite(q).all(1)]:
            p.drawEllipse(xy(j[0], j[2]), 2.8, 2.8)


class Pipeline(QtCore.QThread):
    """extract -> fit -> render on one take, reporting progress within each stage.

    Progress comes out of the tools' own stdout; the only stage that prints
    nothing per frame is NLF, which the bar shows as busy rather than faking a
    number for it.
    """
    progress = QtCore.pyqtSignal(float, str, bool)      # 0..1, label, determinate
    done = QtCore.pyqtSignal(str, str)                  # gif path ('' = failed), note

    def __init__(self, take, pitch=0.0, parent=None):
        super().__init__(parent)
        self.take, self.pitch = take, pitch

    def run(self):
        gif = self.take[:-4] + '_body.gif'
        stages = (
            ('NLF over the frames', 0.0, 0.5,
             [sys.executable, f'{TOOLS}/extract_poses3d.py', '--src', self.take, '--overwrite']),
            ('depth-grounded fit', 0.5, 0.88,
             [sys.executable, f'{TOOLS}/fit_body.py', '--src', self.take, '--overwrite',
              '--verbose']),
            ('rendering', 0.88, 1.0,
             [sys.executable, f'{TOOLS}/render_body_gif.py', self.take, '-o', gif,
              '--pitch', str(self.pitch)]),
        )
        note = ''
        for label, lo, hi, cmd in stages:
            self.progress.emit(lo, label, False)
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1, cwd=os.path.dirname(TOOLS))
            tail = ''
            for line in p.stdout:
                line = line.rstrip()
                if line:
                    tail = line
                if line.startswith('  -> valid'):
                    note = line.strip()[3:]
                f = self.frac(line)
                if f is not None:
                    self.progress.emit(lo + (hi - lo) * f, f'{label}  {100 * f:.0f}%', True)
            if p.wait() != 0:
                self.done.emit('', f'{label} failed: {tail[-120:]}')
                return
        self.done.emit(gif if os.path.exists(gif) else '', note or 'fitted')

    @staticmethod
    def frac(line):
        """How far through the current stage the tool says it is, or None."""
        m = re.search(r'(\d+)/(\d+) frames', line)              # renderer
        if m:
            return min(1.0, int(m.group(1)) / max(int(m.group(2)), 1))
        m = re.search(r'iter\s+(\d+)', line)                    # fit_body --verbose
        if m:                                                   # 20 + 50 iterations
            return min(1.0, int(m.group(1)) / 70.0)
        return None


class Result(QtWidgets.QWidget):
    """The rendered GIF, looping until closed -- QMovie loops it by itself."""

    def __init__(self, gif, note=''):
        super().__init__()
        self.setWindowTitle('fitted 3-D body')
        self.setStyleSheet('background:#101014;')
        pic = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.movie = QtGui.QMovie(gif)
        self.movie.setCacheMode(QtGui.QMovie.CacheAll)
        pic.setMovie(self.movie)
        cap = QtWidgets.QLabel(f'{note}   ·   {os.path.basename(gif)}   ·   Esc closes',
                               alignment=QtCore.Qt.AlignCenter)
        cap.setStyleSheet('color:#ddd; font-size:14px;')
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(pic, 1)
        lay.addWidget(cap)
        self.movie.start()
        self.resize(1300, 780)

    def keyPressEvent(self, e):
        if e.key() in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
            self.close()

    def closeEvent(self, e):
        self.movie.stop()
        e.accept()


class Live(QtWidgets.QWidget):
    def __init__(self, cam, args, K):
        super().__init__()
        self.cam, self.args, self.K = cam, args, K
        self.model = load_model(args.model)
        self.latest = None          # newest frame for the GPU
        self.shown = None           # newest frame for the window, skeleton or not
        self.overlay = None         # newest result to draw
        self.lock = threading.Lock()
        self.infer_times, self.cam_times = [], []
        self.state, self.rec, self.result, self.note = IDLE, None, None, ''
        self.stop = threading.Event()
        self.busy = threading.Event()       # held while the pipeline owns the GPU

        self.setWindowTitle('NLF live')
        self.pic = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.top = TopDown()
        self.status = QtWidgets.QLabel()
        self.status.setStyleSheet('color:#ddd; font-size:15px;')
        self.btn = QtWidgets.QPushButton(f'● REC  {args.segment:.0f} s')
        self.btn.setStyleSheet('font-size:16px; padding:7px 18px;')
        self.btn.clicked.connect(self.toggle)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setTextVisible(True)
        self.bar.hide()
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.btn)
        row.addWidget(self.status, 1)
        panes = QtWidgets.QHBoxLayout()
        panes.addWidget(self.pic, 3)
        panes.addWidget(self.top, 1)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(panes, 1)
        lay.addLayout(row)
        lay.addWidget(self.bar)
        self.setStyleSheet('background:#1b1b1f;')

        cam.start()
        threading.Thread(target=self.grabber, daemon=True).start()
        threading.Thread(target=self.infer, daemon=True).start()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.repaint_frame)
        self.timer.start(33)

    # ---- camera thread: newest frame to the GPU, every frame to disk while recording
    def grabber(self):
        while not self.stop.is_set():
            got = self.cam.read()
            if got is None:
                continue
            seq, ts, buf = got
            rgb = self.cam.to_rgb(buf, self.cam.w, self.cam.h)
            now = time.time()
            with self.lock:
                self.latest = self.shown = rgb
                self.cam_times = [t for t in self.cam_times if t > now - 2] + [now]
            rec = self.rec
            if rec is None:
                continue
            wall = frame_wall(self.cam, ts, 0.0)
            k = rec['idx']
            rec['idx'] = k + 1
            rec['frames'].append((k, seq, wall))
            rec['jpeg'].submit(k, buf, self.cam.w, self.cam.h)
            d = getattr(self.cam, 'depth', None)
            if d is not None:
                rec['depth'].append((k, d[1]))
                rec['depth_jpeg'].submit(k, d[0], self.cam.depth_w, self.cam.depth_h)
            if now - rec['t0'] >= self.args.segment:
                QtCore.QTimer.singleShot(0, self.stop_record)

    # ---- GPU thread: newest frame only, and nothing while the pipeline runs ----
    def infer(self):
        while not self.stop.is_set():
            if self.busy.is_set() or self.model is None:
                time.sleep(0.05)
                continue
            with self.lock:
                rgb, self.latest = self.latest, None
            if rgb is None:
                time.sleep(0.005)
                continue
            t0 = time.time()
            try:
                pred = detect(self.model, rgb[None], self.K, self.args)
                # NLF leaves 'boxes' out entirely when it detected nobody, rather
                # than returning an empty one (seen 2026-09-29 with an empty room).
                boxes = pred.get('boxes', [[]])[0]
                if len(boxes):
                    k = int(np.argmax(boxes[:, 4].cpu().numpy()))
                    # NLF returns joints in millimetres even though `trans` is in
                    # metres; everything else here, and the room frame, is metres.
                    j3 = 1e-3 * pred['joints3d_nonparam'][0][k].cpu().numpy()[:24]
                    out = (rgb, pred['joints2d_nonparam'][0][k].cpu().numpy()[:24],
                           to_room(j3, self.args.pitch, self.args.cam_height,
                                   self.args.centre_dist), True)
                else:
                    out = (rgb, None, None, False)
            except Exception as e:                      # noqa: BLE001 - a dead preview
                self.errors = getattr(self, 'errors', 0) + 1   # thread is worse than a
                if self.errors in (1, 100):                    # noisy one
                    print(f'inference failed ({self.errors}x): {e!r}', file=sys.stderr, flush=True)
                self.fail = repr(e)[:80]
                continue
            with self.lock:
                self.overlay = out
                self.infer_times = [t for t in self.infer_times if t > time.time() - 2] + [time.time()]
            self.last_ms = 1e3 * (time.time() - t0)

    # ---- record -> process -> show ----
    def toggle(self):
        if self.state == IDLE:
            self.start_record()
        elif self.state == RECORDING:
            self.stop_record()

    def start_record(self):
        prefix = resolve_prefix(f'{self.args.out}_{time.strftime("%H%M%S")}', default_outdir())
        self.state = RECORDING
        self.btn.setText('■ STOP')
        self.rec = dict(prefix=prefix, idx=0, frames=[], depth=[], t0=time.time(),
                        jpeg=JpegWriter(f'{prefix}_frames', self.args.quality, 3, 120,
                                        to_rgb=self.cam.to_rgb),
                        depth_jpeg=JpegWriter(f'{prefix}_depth', self.args.quality, 2, 120,
                                              kind='depth'))

    def stop_record(self):
        if self.state != RECORDING:
            return
        rec, self.rec = self.rec, None
        self.state = PROCESSING
        self.btn.setEnabled(False)
        self.btn.setText('● REC')
        self.bar.setRange(0, 0)                 # busy until a stage reports a number
        self.bar.setFormat('writing the take …')
        self.bar.show()
        QtWidgets.QApplication.processEvents()
        rec['jpeg'].close()
        rec['depth_jpeg'].close()
        meta = dict(camera=self.cam.name, width=self.cam.w, height=self.cam.h,
                    fps_requested=self.cam.fps, mode='camera-only (live_body.py)',
                    camera_wall_ts=bool(getattr(self.cam, 'wall_ts', False)),
                    **camera_gt_meta(self.cam, 'both'))
        path, _report, ft = write_capture(rec['prefix'], {}, rec['frames'], rec['t0'],
                                          meta, depth_frames=rec['depth'])
        print(f'wrote {path} ({len(ft)} frames)', flush=True)
        # Hand the GPU over: NLF twice over does not fit in 6 GB of VRAM.
        self.busy.set()
        self.model = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:                                       # noqa: BLE001
            pass
        self.job = Pipeline(path, self.args.pitch, self)
        self.job.progress.connect(self.on_progress)
        self.job.done.connect(self.on_done)
        self.job.start()

    def on_progress(self, frac, label, determinate):
        self.bar.setRange(0, 1000 if determinate else 0)
        self.bar.setValue(int(1000 * frac))
        self.bar.setFormat(label if not determinate else f'{label}   (overall %p%)')

    def on_done(self, gif, note):
        self.bar.hide()
        self.btn.setEnabled(True)
        self.state, self.note = IDLE, note
        print(note, flush=True)
        if gif:
            self.result = Result(gif, note)
            self.result.show()
            self.result.raise_()
        self.model = load_model(self.args.model)        # ~20 s, then the preview is back
        self.busy.clear()

    # ---- GUI ----
    def repaint_frame(self):
        with self.lock:
            over, shown = self.overlay, self.shown
            ct, it = list(self.cam_times), list(self.infer_times)
        # The first NLF call pays for the CUDA context and TorchScript's profiling
        # run -- tens of seconds -- and the model load before it takes another ~20.
        # The camera is drawn regardless so the window is live from the first frame.
        if over is not None or shown is not None:
            rgb, j2, j3, found = over if over is not None else (shown, None, None, False)
            self.top.update_points(j3)
            h, w, _ = rgb.shape
            img = QtGui.QImage(np.ascontiguousarray(rgb).data, w, h, 3 * w,
                               QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(img).scaled(
                self.pic.width(), self.pic.height(), QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation)
            if j2 is not None:
                s = pix.width() / w
                p = QtGui.QPainter(pix)
                p.setRenderHint(QtGui.QPainter.Antialiasing)
                p.setPen(QtGui.QPen(BONE, 3))
                for j, parent in enumerate(SMPL_PARENTS):
                    if parent < 0 or not np.isfinite(j2[[j, parent]]).all():
                        continue
                    p.drawLine(int(j2[j][0] * s), int(j2[j][1] * s),
                               int(j2[parent][0] * s), int(j2[parent][1] * s))
                p.setPen(QtGui.QPen(JOINT, 2))
                p.setBrush(JOINT)
                for x, y in j2[np.isfinite(j2).all(1)]:
                    p.drawEllipse(QtCore.QPointF(x * s, y * s), 3.5, 3.5)
                p.end()
            self.pic.setPixmap(pix)
        if self.state == PROCESSING:
            self.status.setText('processing the segment …')
            return
        if over is not None and over[3]:
            q = over[2][0]          # pelvis, room frame
            who = f'person at X {q[0]:+.2f}  Z {q[2]:+.2f} m from the centre, hip {q[1]:.2f} m'
        else:
            who = 'no person' if over is not None else 'warming up the model …'
        if getattr(self, 'errors', 0):
            who = f'inference failing ({self.errors}x): {self.fail}'
        left = ''
        if self.state == RECORDING and self.rec is not None:
            left = (f'   ·   ● REC  {self.args.segment - (time.time() - self.rec["t0"]):.0f} s left'
                    f'  ({self.rec["idx"]} frames)')
        self.status.setText(
            f'{who}   ·   camera {len(ct) / 2:.0f} fps · NLF {len(it) / 2:.1f} fps '
            f'({getattr(self, "last_ms", 0):.0f} ms){left}'
            + (f'   ·   last fit: {self.note}' if self.note and not left else ''))

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_R:
            self.toggle()
        elif e.key() in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
            self.close()

    def closeEvent(self, e):
        self.stop.set()
        self.timer.stop()
        if self.result is not None:
            self.result.close()
        e.accept()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--segment', type=float, default=10.0,
                    help='longest recorded segment, seconds (default: %(default)s)')
    ap.add_argument('--out', default='live/seg', help='capture prefix under the data dir')
    ap.add_argument('--model', default=str(DEFAULT_MODEL))
    ap.add_argument('--device', default='auto')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--depth-size', default='1280x720')
    ap.add_argument('--depth-range', default='0.5,6.0')
    ap.add_argument('--quality', type=int, default=85)
    ap.add_argument('--pitch', type=float, default=0.0,
                    help='camera tilt, degrees nose down; the one knob between the '
                         'camera frame and the room frame (default: %(default)s)')
    ap.add_argument('--cam-height', type=float, default=CAM_H)
    ap.add_argument('--centre-dist', type=float, default=HALF_DIAG,
                    help='camera to arena centre, m (default: %(default).2f)')
    ap.add_argument('--det-thresh', type=float, default=0.2)
    ap.add_argument('--num-aug', type=int, default=1,
                    help='NLF test-time augmentations; 1 is the live setting')
    ap.add_argument('--internal-batch', type=int, default=0)
    args = ap.parse_args()

    dev = default_camera_device() if args.device == 'auto' else args.device
    dw, dh = (int(x) for x in args.depth_size.lower().split('x'))
    cam = open_camera(dev, args.width, args.height, args.fps, depth=True,
                      depth_size=(dw, dh))
    set_depth_range(cam, args.depth_range)
    print(f'camera {cam.name} {cam.w}x{cam.h} @ {cam.fps:g} fps', flush=True)
    K = K_of(load_calib(camera_gt_meta(cam, 'both'))['colour_intrinsics'])
    app = QtWidgets.QApplication(sys.argv)
    v = Live(cam, args, K)
    v.resize(1280, 860)
    v.show()
    try:
        app.exec_()
    finally:
        cam.close()


if __name__ == '__main__':
    main()
