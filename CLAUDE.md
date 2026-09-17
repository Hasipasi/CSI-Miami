# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A four-node WiFi-CSI sensing rig: four ESP32-C5 boards pass a transmit token around a room while a RealSense D435i records colour/depth, producing time-aligned CSI + video for activity recognition and pose estimation. `README.md` is the authoritative description of the hardware, the capture format and the pipeline; `NOTES.md` is the dated lab log (findings, failures, measurements behind every claim); `RECORDING_2026-08-12.md` documents the existing corpus. Read the relevant README section before touching anything; almost every non-obvious design choice is explained there with the measurement that justifies it.

There is no test suite and no linter config. Verification is done on hardware (`ring_sweep.py`, `check_session.py`) or, for firmware, by `idf.py build` for both targets.

## Commands

Everything host-side runs in Docker (repo bind-mounted at `/workspace`): service `csi` (ESP-IDF + capture tools) and service `pose` (`Dockerfile.pose`: torch/CUDA for NLF, the SMPL fit, YOLO and the dataset builders; needs the NVIDIA container toolkit, `runtime: nvidia`). Do not create venvs for the pose side; the user wants Docker. On a Mac the capture tools run natively from `.venv_mac` (untracked).

```bash
docker compose build                                   # once

# GUI: live camera + 12 per-link CSI waterfalls, REC, scripted protocols
docker compose run --rm -e QT_X11_NO_MITSHM=1 --name csi_live csi \
  bash -lc "cd /workspace/tools && exec python3 viewer.py --protocol /workspace/protocols/<dir>/<file>.yaml"

# headless recording, same writer as the GUI
python3 tools/capture.py --prefix take1 --seconds 60 [--mode fixedtx --tx A]

# measure the round-robin schedule on the boards, no camera needed
python3 tools/ring_sweep.py --points 1:400,2:400,0:400 --guard 1 [--band 5.6 --bw 40]

# validate a session — always, before anyone trains on it
python3 tools/check_session.py data/<session>

# 3-D pose targets (GPU service; CSI_DATA=<dir> mounts recordings kept elsewhere at /workspace/data)
docker compose build pose
docker compose run --rm pose python3 tools/extract_poses3d.py --src data      # NLF per frame -> <take>_nlf.npz
docker compose run --rm pose python3 tools/fit_body.py --src data             # depth-grounded fit -> <take>_body.npz
docker compose run --rm pose python3 tools/body_report.py data                # per-session summary
docker compose run --rm pose python3 tools/extract_poses3d.py --check-smpl    # verify the SMPL layer vs NLF
# legacy 2-D targets: tools/extract_poses.py (YOLO) + build_pose.py --target 2d
python3 tools/build_activity.py --src data --sessions '*_train,*_test' --rate 30 --subcarriers $SUB --out dataset/<name>
python3 tools/build_pose.py     --src data --sessions '*_train,*_test' --rate 30 --context 20 --subcarriers $SUB --out dataset/<name>
```

Firmware (ESP-IDF 5.5, all boards run the same image):

```bash
docker compose run --rm csi bash -lc "cd /workspace/firmware && idf.py build"
docker compose run --rm csi bash -lc "cd /workspace/firmware && idf.py -p /dev/ttyACM0 -b 460800 flash"
idf.py set-target esp32s3      # only for the retired S3 boards; target is pinned in sdkconfig.defaults
```

`sdkconfig` and `managed_components/` are generated and gitignored; `sdkconfig.defaults` and `dependencies.lock` are tracked and sufficient. Firmware changes should build clean for esp32c5 and esp32s3 (the S3 paths are kept so old recordings and the old boards stay usable).

Macs only: run `sudo .venv_mac/bin/python tools/depth_server.py` in its own terminal first; the viewer and recorder pick colour and depth up from its socket automatically.

## Architecture

**`tools/capture.py` is the shared library, not just a script.** `viewer.py`, `ring_sweep.py`, `depth_server.py`, `prune_frames.py`, `plot_frame_packets.py` and `rewrite_raw.py` all import from it. The pieces that matter:

- `discover()` finds boards by USB VID, toggles DTR/RTS to **reset every board**, and reads the boot MAC. Consequently any runtime firmware setting (`RATE`, `SUB`, `BAND`, ...) is lost each time a tool starts; persistent changes go into the `CONFIG_*` defines in `firmware/main/app_main.c`.
- `CsiStream` splits one board's UART into binary CSI frames (magic + version + checksum, v1/v2/v3 all parsed) and interleaved text lines (`TX_DONE`, `ROLE_*`, `BAND_OK`, IDF logs).
- `TokenRing` is the host side of the round-robin: count-based turns (`TX <n> <tag> <delay_us>`) scheduled ahead so boards keep the ring turning on their own timers; `--burst 0` is the old timed dwell; `--guard` is the inter-turn silence that prevents air collisions. Both the GUI and the headless recorder use this one class.
- Camera classes: `Camera` (raw V4L2, Linux), `MacCamera` (AVFoundation), `RealSenseCamera` (librealsense, colour + depth), `ServedCamera` (client of `depth_server.py`). `open_camera()` picks one.
- `write_capture()` is the **single writer** for every take. It computes the per-frame CSI windows (`win_*`, `csi [frame, receiver, subcarrier, T]`), prunes incomplete frames, and writes one self-contained NPZ with JPEG/PNG frames embedded. Packet-to-frame assignment uses `tx|rx|tc` (board clock mapped to host time), not `tx|rx|t` (host arrival).
- `LABEL` maps MAC suffix to position letter A–D. **Labels are room positions, not hardware.** The same map is duplicated in `tools/geometry.py`; change both, plus the README table, if boards are rearranged. Old recordings carry their own labels in `meta['boards']`.
- `SUB_INDEX` mirrors the `SUB_INDEX_*` tables in the firmware; the firmware compacts frames to only the subcarriers it sends, so any per-field normalisation must map through it.

**`tools/viewer.py`** is the PyQt5/pyqtgraph GUI (`class Live` holds nearly everything). It is a monitor, not a measurement: display normalisation (Floor, Norm) never touches recordings. Protocol YAMLs (`protocols/`) drive scripted sessions; the take set is cycled, not repeated back-to-back, and output lands in `<outdir>/<yaml name>/<take><round>.npz`. Takes are written in a background thread after the protocol ends; a failed write is kept as `<take>_raw.pkl` for `rewrite_raw.py`.

**Firmware** is a single `firmware/main/app_main.c`. Boards are purely reactive over UART (the PC schedules the ring; boards never coordinate over the air). The header comment lists every serial command and reply. The three knobs `CONFIG_IQ_MODE`, `CONFIG_SEND_FREQUENCY`, `CONFIG_SUB_COUNT` are coupled with each other and with the host's turn settings; changing one without re-measuring the others has produced slower configurations before.

**Pipeline:** firmware → `capture.py`/`viewer.py` → `data/<session>/*.npz` → `check_session.py` → `extract_poses3d.py` (NLF, `<take>_nlf.npz`) → `fit_body.py` (`<take>_body.npz`, COCO-17 in metres, colour camera frame) → `build_activity.py` / `build_pose.py` → `dataset/<name>/` (`manifest.json`, `clips.csv`, `clips/*.npz`). `data/`, `dataset/`, `archive/`, `models/` are gitignored.

**3-D pose side** (`tools/body_common.py` is the shared module): `SMPL` is a hand-written LBS layer reading `models/smpl_neutral.npz`, which `extract_poses3d.py` extracts from the body model embedded in the NLF torchscript (`models/nlf_l_multi_0.3.2.torchscript`, downloaded from github.com/isarandi/nlf/releases, not in git). `load_calib()` resolves camera geometry as meta → `calib/realsense_<serial>.json` (from `rs_calib.py`) → nominal D435i values with a warning. `fit_body.py` optimises one shape + per-frame pose/translation per take against depth points, 2-D joints, a pose prior and temporal smoothness; `plot_body_fit.py` is the visual check (vertices must sit on the silhouette in the *depth* panel). NLF boxes are x, y, w, h, score. TorchScript re-profiles for 10+ s on every new input shape, so batches are padded to a fixed size. Capture output dir resolves as `$CSI_DATA` → Mac flash drive `/Volumes/<stick>/data` → `/workspace/data` → `<repo>/data`.

## Constraints worth knowing before changing anything

- **Only one process can own the serial ports.** Stop the viewer before running `ring_sweep.py`, `capture.py` or `geometry.py`.
- **Boards are identified by MAC, never by `/dev/ttyACM*` or `/dev/cu.*`.** Paths change on every replug.
- `BAND`, `CHAN`, `BW` must be issued to every board together; 5.6 GHz is refused unless every connected board is a known C5 (`C5_MACS` in `capture.py`). The firmware goes to 5 GHz via HT20 before widening to HT40; a direct band+HT40 change loses the ESP-NOW peer.
- Round-robin (12 links) and pinned-transmitter (3 links) takes have different shapes and cannot share a dataset; select them separately with `--sessions`.
- The dataset builders and `extract_poses.py` read the amplitude arrays and colour JPEGs only. `tx|rx|iq` (phase) and depth-only takes (`meta['gt'] == 'depth'`) are silently ignored downstream. Raw phase is unusable until a per-packet linear detrend across subcarrier index is removed.
- Amplitudes are uncalibrated; normalise per link. `win_gain` Q8.8 == 0 means AGC not yet calibrated, do not scale.
- The number to optimise in a schedule is "frames w/ all links" (`cov%` in `check_session.py`), not per-link rate.
- Firmware flashing is ordinary app-partition writes: no eFuse, no secure boot, no flash encryption. Keep it that way.
- `AGC LOCK` kills transmission on current firmware; it stays in the source but must not be used.
- 3-D pose targets are in the colour camera's optical frame (x right, y down, z forward, metres); nothing yet relates that frame to the board positions.
- Takes recorded before 2026-09-17 carry only the depth intrinsics in meta; colour intrinsics and depth→colour extrinsics come from `calib/` or the nominal fallback.

## Documentation conventions

Measurements drive this repo. When a change alters rig behaviour, a default, or a number the README quotes, add a dated entry to `NOTES.md` (headings like `## 2026-09-16 — …`, most recent at the bottom) with what was measured and how, and update the README where it states the affected figure. Comments in the code explain *why* a choice was made and what failed before; keep that style rather than describing what the code does.
