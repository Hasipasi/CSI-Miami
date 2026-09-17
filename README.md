# csi-rig

A four-node WiFi-CSI sensing rig: four ESP32-C5 boards pass a transmit token
around a room while a RealSense watches, producing time-aligned **channel state
information + video (colour or depth)** for human activity recognition and pose
estimation.

Everything runs in Docker: one image for the firmware and the capture tools, a
second (`Dockerfile.pose`, service `pose`) with torch for the 3-D pose side. The host
needs Docker, an X server for the GUI, and the NVIDIA container toolkit for `pose`.

---

## The hardware

The rig is **four dual-band ESP32-C5-DevKitC-1 boards** (since 2026-09-16; the
original four ESP32-S3 N16R8 boards A–D are retired, and the S3 code paths remain
only so old recordings and the `esp32s3` build keep working). Boards are identified
by MAC, never by `/dev/ttyACM*` or `/dev/cu.*` — paths change on every replug and
the tooling auto-discovers.

| label | position (seen from the camera) | MAC | USB serial |
|---|---|---|---|
| A | near row, right, beside the camera | `10:bd:a3:e6:38:24` | 5C94096766 |
| B | near row, left | `10:bd:a3:e6:38:14` | 5C94096765 |
| C | far row, left | `10:bd:a3:e6:62:3c` | 5C94096576 |
| D | far row, right | `10:bd:a3:e6:37:f4` | 5C94096770 |

Retired S3 boards, for reading old `meta['boards']`: A `14:c1:9f:c1:2d:3c`,
B `dc:da:0c:77:6b:5c`, C `30:30:f9:1d:ab:d4`, D `14:c1:9f:c1:2d:a8`, E (the returned
fifth board) `ec:da:3b:4c:b8:d0`. The C5 boards carried the interim labels F/G/H/I
(F=`62:3c`, G=`37:f4`, H=`38:14`, I=`38:24`) in sessions before 2026-09-16.

An all-C5 rig can switch live between 2.4 GHz channel 13
and 5.600 GHz channel 120 from the GUI. Both HT20 (57 complex subcarriers) and HT40
(117) are verified on C5 at 5.6 GHz. The firmware enters 5 GHz through HT20 before
widening because a direct band+HT40 transition leaves the ESP-NOW peer behind. S3 and
C5 boards can coexist at 2.4 GHz, but 5.6 GHz is disabled unless every connected board
is a known C5 because an S3 cannot follow the band change.

**Labels name positions, not hardware.** If the boards are physically rearranged,
update `LABEL` in `tools/capture.py` and the coordinates in `NOTES.md`. Getting this
wrong silently mislabels every downstream figure; it has already happened once.

**Layout** (setup 3): a 3 m square. Seen from the camera, `C D` on the far row and
`B A` on the near row (A beside the camera), diagonals 4.24 m. Measured link quality does *not* follow this geometry — see
`NOTES.md`.

**Intel RealSense D435i** on USB 3.0 for ground truth: 1280×720 colour at 30 fps,
and 640×480 depth (uint16, 1 mm units) alongside it. The GUI shows depth under the
colour picture and a `GT` toggle picks which stream a take records: colour as JPEGs
or depth as lossless 16-bit PNGs (`frames/000000.png`, `meta['gt'] == 'depth'`,
with `depth_scale_m` and the intrinsics in meta). Depth comes through librealsense
(`pyrealsense2`), which also gives the colour frames their hardware timestamp and
frame counter.

**On a Mac, run `tools/depth_server.py` as root and everything else normally.**
librealsense can only open the camera as root there, and once it has the camera
the OS camera stack loses it (the colour interface vanishes from AVFoundation
until a replug), so the one root process serves both colour and depth over a local
socket (`/tmp/csi-depth.sock`) and the viewer and recorder, unprivileged, take
both from it automatically. Plug the camera in fresh, then in its own terminal:

```bash
sudo .venv_mac/bin/python tools/depth_server.py      # leave it running
.venv_mac/bin/python tools/viewer.py                 # in another terminal, as usual
```

Without the server the tools use the RealSense colour stream through AVFoundation,
colour only, and the depth panel says so. The `pyrealsense2` in `.venv_mac` is
librealsense 2.58.4 built from source on 2026-09-16 (the `pyrealsense2-macosx`
wheel crashes on macOS 26); `tools/rs_probe.py` is the step-by-step diagnostic. On
the Linux rig the normal udev rules suffice and the tools open the camera through
librealsense directly with `--depth`.

### How a capture works

One board holds a "transmit token" and pings at `CONFIG_SEND_FREQUENCY`; every other
board receives and reports CSI for that packet. The PC rotates the token over UART —
**round-robin**, giving all 12 ordered board pairs ("links") — or pins one board as
the sole transmitter, giving 3 links at ~4× the per-link rate.

Round-robin turns are **count-based and scheduled ahead** (`--burst N`, live in
the GUI as "pkts/turn"): the host tells every board its next turns several cycles
in advance (`TX <n> <tag> <delay_us>`), each board starts its bursts on its own
timer and answers `TX_DONE`, and the host only keeps the schedule topped up and
reads the replies for the books — so a busy recording process cannot leave holes
in the ring. Receivers are armed once with the list of every other board (`RX
a,b,c`, refreshed with `PEERS`). The turn spacing is floored by what the serial
wire drains (`--schedule gated` is the older one-turn-per-round-trip mode). The point is that the token goes
round all four boards in well under one 33 ms camera frame, so **every video frame
carries CSI from every transmitter**. Measured on 2026-09-16 with the four C5
boards while recording video, at the default **5.6 GHz / 40 MHz / RATE 2000 / 4
pings a turn / 0.5 ms guard**: the token goes round in **12 ms**, every 30 fps
frame window holds at least 5 and typically 10 packets on every one of the 12
links, ~310 Hz per link, no radio loss. The previous timed dwell (25 ms a board,
still available as `--burst 0`) took 113 ms to go round and **0%** of frames saw
every link. At 2.4 GHz the band itself loses 10–20% of packets and back-to-back
transmitters collide, so the defaults there are 2 pings a turn with a 1 ms guard
(~95 Hz per link, every frame covered). `tools/ring_sweep.py` measures all of
this without a camera; the viewer's "frames w/ all links" and "token cycle" tiles
and `check_session.py`'s `cov%` column report it on real takes, and
`tools/plot_frame_packets.py` draws one frame's packets per link. A frame's CSI is
the window half a frame period either side of its timestamp (±16.7 ms at 30 fps):
centred on the picture, and disjoint from the neighbouring frames' windows.

Each link is a distinct path across the room. There is no antenna array here: `L=12`
means twelve room-spanning paths, not twelve antennas.

The firmware has two payload encodings, chosen by `CONFIG_IQ_MODE` and tagged by a
version byte so one parser reads both:

| version | payload | frame | subcarriers | ping knee | ping | dwell | per-link | p99 gap |
|---|---|---|---|---|---|---|---|---|
| 1 | uint8 amplitude | 212 B | 192 | — | 125 Hz | 50 ms | 25.8 Hz | 192 ms |
| 2 | int8 I/Q + gain | 82 B | 30 | ~750 Hz | 675 Hz | 12.5 ms | 118.4 Hz | 71 ms |
| **2** | **int8 I/Q + gain** | **354 B** | **166** | **~270 Hz** | **243 Hz** | **25 ms** | **43.5 Hz** | **100 ms** |

The bold row is what the firmware ships. It is *not* the fastest — 30 subcarriers
gives 2.7× the per-link rate and a shorter blind gap. 166 is chosen because the
R² = 0.973 that justifies subsetting to 30 was measured on recorded **amplitude**, and
nothing has yet shown it holds for phase. Subsetting stays available in analysis;
discarding at the boards does not. `SUB 30` + 675 Hz + 12.5 ms dwell switches.

Version 1 recorded the campaign in `RECORDING_2026-08-12.md`. Version 2 is current:
dropping to 30 subcarriers (justified by the PCA below) makes a frame small enough to
carry *both* complex values and ~5× the rate. The shipped rate is the measured knee
minus 10%, and both knees were found by pushing past them until delivery collapsed.

**The two subcarrier sets are limited by different things.** 166 SC is bandwidth-bound
— it saturates at ~97% UART, exactly as 354 B a frame predicts. 30 SC is *not*: it
degrades at only ~65% UART, because small frames hit per-frame cost (mutex, per-byte
ROM writes, ESP-NOW send rate) long before they fill the wire. Do not expect a
byte-count argument to predict the 30 SC ceiling; it doesn't.

Switch either at runtime with `SUB 30` / `SUB 166` / `SUB 0`, then set
`CONFIG_SUB_COUNT` and reflash — runtime settings do not survive the reset the host
issues when it discovers boards.

**Per-link rate is an average over a burst and a blind gap, not a sampling rate.** A
link is sampled densely while its transmitter holds the token, then not at all. The
number that matters for aligning CSI to 30 fps video is the gap, not the rate:

| | rate | median gap | p99 gap |
|---|---|---|---|
| 30 SC, 400 Hz, 50 ms dwell | 92.3 Hz | 1.86 ms | 165.8 ms |
| 30 SC, 675 Hz, 12.5 ms dwell | 118.4 Hz | 1.39 ms | 71.4 ms |
| **166 SC, 243 Hz, 25 ms dwell** (shipped) | **43.5 Hz** | 4.19 ms | **99.9 ms** |

Those rows are the timed dwell (`--burst 0`). Raising the ping rate makes bursts
denser and leaves the gap untouched — only a shorter turn closes it. Count-based
turns (`--burst N`) replace that: the gap becomes the token cycle,
`boards × ((N-1)/RATE + guard + round trip)`, and the USB round trip for the `TX`
line out and the `TX_DONE` line back measures only ~0.5 ms on this Mac. Measured
2026-09-16, four C5 boards, 2.4 GHz, against a 30 fps clock:

| pings/turn | guard | cycle | frames with all 12 links | per link | delivery |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 ms | 3.6 ms | 95% | 152 Hz | 75% |
| **1** | **1 ms** | **8 ms** | **98–100%** | **105 Hz** | **90%** |
| 1 | 2 ms | 13 ms | 99% | 68 Hz | 93% |
| 2 | 1 ms | 19 ms | 94% | 96 Hz | 93% |
| 4 (RATE 1200) | 1 ms | 18 ms | 92% | 188 Hz | 90% |
| timed 25 ms dwell | — | 113 ms | 0% | 98 Hz | 98.5% |

That table is 2.4 GHz, where the band loses packets. At 5.6 GHz / 40 MHz delivery
is 100% at every setting and the wire is the limit (~340 Hz per link fills the
receivers' UARTs to ~89%): while recording, 4 pings a turn at RATE 2000 covers
every frame with 5–10 packets per link, 6 gets a little more but drops a link
from the odd frame, 8 overruns the wire. More pings a turn buy per-link rate at
the cost of coverage; the guard buys delivery (at 2.4 GHz) at the cost of cycle.
The number to watch is "frames w/ all links", not the per-link rate.

**Raw phase is unusable as it arrives, and usable after detrending.** Measured on a
static scene: raw phase has sd 1.84 rad across packets, against 1.814 for uniform
noise — it is the carrier and sampling offsets, redrawn every packet, not the
channel. Fit and remove a line across subcarrier index per packet and the spread
falls to **0.07 rad**. Any consumer of `|iq|`-plus-phase must do that first.

Measured: **round-robin beats single-TX on activity recognition despite 4× less
temporal resolution.** Spatial diversity is worth more than sample rate here.

---

## Layout

```
csi-rig/
├── firmware/          ESP-IDF project for the boards (esp32s3)
├── tools/             every script; flat, one job each
├── protocols/         YAML capture scripts (which takes, how long, how many rounds)
├── calib/             RealSense calibration JSON per camera (tools/rs_calib.py), tracked
├── models/            NLF torchscript + the SMPL arrays extracted from it (not tracked)
├── data/              raw recordings, one directory per session
├── dataset/           packaged datasets built from data/
├── archive/           superseded datasets, kept for reference
├── Dockerfile         ESP-IDF + capture tools (service `csi`)
├── Dockerfile.pose    torch/CUDA: NLF, the SMPL fit, YOLO, dataset builders (service `pose`)
├── docker-compose.yml
├── NOTES.md           lab log: findings, failures, and why things are the way they are
└── RECORDING_2026-08-12.md   what is in data/: sessions, config, validation, caveats
```

The repo is bind-mounted at `/workspace` in the container, so a path is the same
inside and out apart from that prefix.

---

## Quick start

```bash
docker compose build                 # once

# check the rig is alive
docker compose run --rm csi bash -lc "cd /workspace/tools && python3 geometry.py list"
docker compose run --rm csi bash -lc "cd /workspace/tools && python3 camera_probe.py /dev/video6 YUYV 640 480 /tmp/x.pgm"

# the GUI: live camera + 12 per-link CSI waterfalls, recording, scripted protocols
docker compose run --rm -e QT_X11_NO_MITSHM=1 --name csi_live csi \
  bash -lc "cd /workspace/tools && exec python3 viewer.py --protocol /workspace/protocols/gergo_train.yaml"

# macOS native GUI (uses the checked-out .venv_mac); for depth, first start
# `sudo .venv_mac/bin/python tools/depth_server.py` in another terminal
.venv_mac/bin/python tools/viewer.py

# measure the round-robin schedule on the boards, no camera needed
.venv_mac/bin/python tools/ring_sweep.py --points 1:400,2:400,0:400 --guard 1
```

Captures go to `$CSI_DATA` if set, else `<flash drive>/data` on a Mac with a stick
plugged in (the viewer prints where at startup), else `/workspace/data` in the
container, else `<repo>/data`.

Only one process can own the serial ports at a time — stop the viewer before running
anything else that talks to the boards.

---

## The pipeline

```
firmware ──UART──▶ tools/capture.py ──▶ data/<session>/*.npz
                        │                        │
                   tools/viewer.py          tools/check_session.py   (validate)
                   (GUI + protocols)             │
                                          tools/extract_poses3d.py   (NLF: RGB -> SMPL init)
                                                 │
                                          tools/fit_body.py          (depth-grounded SMPL fit
                                                 │                    + temporal bundle adjustment)
                        ┌────────────────────────┴───────────────┐
              tools/build_activity.py                   tools/build_pose.py
                        └────────────▶ dataset/ ◀───────────────┘
```

### 1. Record

`tools/viewer.py` is the normal way in: live camera beside per-link CSI waterfalls,
a REC button, and scripted protocols with a big cue card for whoever is standing in
the array. `tools/capture.py` does the same headless.

Every take is one NPZ holding the CSI packets per directed link (`tx|rx|t`,
`tx|rx|iq`, `tx|rx|a`, gain and RSSI per packet), the frames, and **the per-frame
CSI windows**. What the frames are is chosen per session (`--frames`, the "frames"
buttons in the GUI): **by default colour JPEGs with the depth frames beside them**
(`frames/000000.jpg`, `depth/000000.png`, `depth_t`, `frame_depth_idx` pairing
each colour frame with its depth frame — 3-D pose needs both), or colour alone, or
the depth frames alone (16-bit PNGs under `frames/`, `meta['gt'] == 'depth'`).
Only complete frames are kept: a frame with no image (encoder queue full), no depth
frame within half a period (in colour+depth mode) or a link missing from its CSI
window is dropped at write time and counted in `meta['frames_dropped']`;
`tools/prune_frames.py` applies the same rule to takes already on disk. The
windows: for each frame, the packets within half a frame period of it as a fixed-shape
transmitter × receiver grid of up to 16 packet slots per link, zero-padded. **The
model input is `csi [frame, receiver, subcarrier, T]`**, float16 amplitude: for each
of the 4 receivers, every packet it heard in the window from every transmitter,
flattened in time order along T (T = 3 × 16 = 48 in round-robin; with a pinned
transmitter the 3 receivers × T = 16), with `csi_t` (seconds from the frame),
`csi_tx` (transmitter index into `win_boards`, −1 = padding) and `csi_count` (real
packets per receiver) beside it. The same amplitude is also kept per link as
`win_a [frame, tx, rx, slot, subcarrier]` (uncalibrated |I+jQ|×gain, the units of
`tx|rx|a`) and
`win_iq [frame, tx, rx, slot, subcarrier, (re, im)]` as the int8 wire values with
`win_gain` to scale them, and beside every slot its time from the frame (`win_t`),
absolute time (`win_ts`), the receiving board's clock (`win_lts`), `win_rssi`,
`win_agc`, `win_fft`, and `win_idx`, the packet's index into the link arrays.
`win_count` says how many slots are real; zeros are padding, and the diagonal is
always zero because a board never hears itself. Packet times for the windows and
the nearest-frame assignment are `tx|rx|tc`, the receiving board's own clock
mapped onto host time (a busy host stamps packets late; the board never does);
`tx|rx|t` remains the host arrival time. `check_session.py` reports the
per-window minimum and median and flags any empty link-window.

A protocol YAML defines the takes:

```yaml
lead_in: 3        # seconds of "get ready"
duration: 5       # seconds recorded
gap: 1            # rest afterwards
repeats: 10       # run the whole set N times, cycling
takes:
  - name: salute
    instruction: SALUTE — right hand to your forehead, hold
```

The set is **cycled**, not repeated take-by-take, so repeats of one activity land
minutes apart and are far more independent samples. Output goes to
`data/<yaml name>/<take><round>.npz`. Each capture is self-contained: NumPy arrays
and JPEGs (`frames/000000.jpg`, etc.) share one archive. `frame_t_ns` and every
`<tx>|<rx>|t_ns` use signed int64 nanoseconds from the same recording-start zero;
`frame_idx` maps each timestamp to its embedded JPEG name.

**A board can go silent mid-session** — the viewer refuses to start a protocol if one
is quiet and aborts the moment one drops, because a silent board produces takes
missing every link into it and that has cost 100 takes before.

### 2. Validate — always

```bash
python3 tools/check_session.py data/gergo_train
```

Checks link completeness, class balance, frame/JPEG agreement, monotonic clocks,
stuck or all-zero amplitudes. Every check exists because that failure happened and
was silent.

### 3. 3-D pose (pose task only)

```bash
docker compose build pose                                   # once; ~5 GB image
# put models/nlf_l_multi_0.3.2.torchscript in place (github.com/isarandi/nlf/releases)
docker compose run --rm pose python3 tools/extract_poses3d.py --src data   # RGB -> SMPL per frame
docker compose run --rm pose python3 tools/fit_body.py --src data          # depth-grounded fit
docker compose run --rm pose python3 tools/body_report.py data             # summary per session
docker compose run --rm pose python3 tools/plot_body_fit.py data/<session>/<take>.npz
```

`CSI_DATA=<dir>` mounts a recordings directory elsewhere at `/workspace/data`.

Two steps, on purpose. **`extract_poses3d.py`** runs NLF (Neural Localizer Fields,
Sárándi & Pons-Moll 2024) over the colour frames: a full SMPL body per frame —
pose, shape, and where it stands in metres — with the 2-D joints and a per-joint
uncertainty. Writes `<take>_nlf.npz`. ~50 fps on the GPU once TorchScript has
warmed up (the first two calls of a shape take 10–15 s, which is why batches are
padded to one size). A monocular network is stable in *pose* but only guesses
*distance* from apparent body size; on the first take checked it was 4 cm off and
drifting by several cm within a 7 s take.

**`fit_body.py`** then anchors that body to the RealSense depth and solves the
take as one sequence: one shape, a pose and a translation per frame, minimising
(all robustly) the distance from every depth point on the person to the body
surface, the reprojection of the joints against the network's 2-D estimate, a
prior towards the network's pose, and joint velocity and acceleration across
frames. The depth points are the ones in the person's box, in a band around the
initial body, then within 15 cm of the first-pass fit — floor and wall never
enter. ~15 s per take, or ~25 s with `--elaborate` (more iterations and vertices;
same residual, joints ~8 mm apart on average and up to ~18 mm at the knees and
ankles — use it for a dataset, the fast default for checking a take). Writes
`<take>_body.npz`: SMPL parameters, the 24 SMPL
joints and **`keypoints3d [F, 17, 3]` — COCO-17 in metres in the colour camera's
frame** (x right, y down, z forward), the training target, with `valid` per
frame and per-frame residuals (`depth_med`, `reproj_px`). Point-to-body residuals
of ~35 mm median are the sensor: a plane fitted to a flat patch of the same
scene at 3 m has 15–17 mm median residual, and the point-to-vertex distance adds
the vertex spacing on top.

The camera geometry the fit needs — colour intrinsics and the depth→colour
extrinsics — is written into meta by the recorder since 2026-09-17. Older takes
(everything up to and including the 2026-09-17 sessions) have only the depth
intrinsics; run `tools/rs_calib.py` once on the rig with the camera plugged in
and commit `calib/realsense_<serial>.json`, which the tools pick up by the
serial in meta. Until then they fall back to nominal D435i values and say so
loudly. Measured on one take, the fallback costs nothing visible (the residual
does not move when the extrinsic offset is flipped or zeroed), but nominal is
nominal.

The 2-D route is still there: `tools/extract_poses.py` (YOLO11x-pose,
`<take>_pose.npz`, pixels) and `build_pose.py --target 2d`.

### 4. Build datasets

```bash
SUB=6,11,17,23,28,35,41,46,52,58,70,76,82,87,93,99,105,110,116,122,138,144,150,155,161,167,172,178,184,190

python3 tools/build_activity.py --src data --sessions '*_train,*_test' \
        --rate 30 --subcarriers $SUB --out dataset/csi5act_rr_v3
python3 tools/build_pose.py     --src data --sessions '*_train,*_test' \
        --rate 30 --context 20 --subcarriers $SUB --out dataset/csi5pose_rr_v3
```

`build_pose.py` takes the 3-D target (`--target 3d`, the default: `pose [T, 17, 3]`
in metres from `<take>_body.npz`, with `pose2d` beside it) or the old 2-D one
(`--target 2d`, pixels from `<take>_pose.npz`).

Round-robin and single-TX captures have different link counts and cannot share a
dataset, so `--sessions` selects them separately (`'*_train,*_test'` vs `'*_txB'`).

Each dataset gets a `manifest.json` (read it first), `clips.csv`, `stats.npz`,
`clips/*.npz`, and for pose a `samples.csv` indexing every target.

---

## Things worth knowing before you change anything

**CSI is irregularly sampled.** The grid in a dataset is a resampling choice, and
`mask` says which cells are a fresh packet versus the nearest one held. Round-robin
at 30 Hz was ~32% fresh under the timed dwell; ignoring the mask means treating stale
values as observations. Count-based turns exist to push that towards 100% — check the
`cov%` column of `check_session.py` on a new session rather than assuming either
figure. The raw irregular packets ship in the activity clips so a different
grid can be built without re-recording.

**192 subcarriers carry ~2.6 effective dimensions.** Measured by PCA
(`tools/pca_subcarriers.py`). 30 evenly-spaced subcarriers reconstruct all 166 live
ones at R² = 0.973, which is why the datasets ship 30, and why the firmware now sends
only those 30. Picking the *highest-variance* 30 instead scores 0.831 — they cluster
and re-measure the same thing.

**3-D pose targets are in the colour camera's frame, not the room's.** The camera
does not move within a session, so this is a rigid room frame up to one unknown
transform; but nothing has been measured between the camera and the boards yet,
so a link's geometry cannot be related to a joint position without that. The
frame is x right, y down, z forward, metres.

**The dataset builders do not read phase yet.** `build_activity.py` and
`build_pose.py` consume the `a` (magnitude) arrays. Captures made with v2 firmware
carry a `tx|rx|iq` array alongside, and it is currently ignored — silently, so this
is worth knowing before concluding that phase did not help.

**UART bandwidth is the binding constraint on sample rate**, not the radio. The
current C5 rig runs at 3 Mbaud 8N1, carrying 300 KB/s per board; the original S3
measurements used 921600 baud (92.2 KB/s). Bandwidth arithmetic is necessary but not
sufficient: 100 Hz on the older CSV encoding fit on paper and still corrupted lines,
because `ets_printf` blocks on a full TX FIFO and starves the UART command task.
At 5.6 GHz HT40, four-board round-robin is verified through 1200 ping/s: 300 Hz per
directed link and 77.5% on the busiest UART. The GUI rate buttons are per-link targets
(`100`, `200`, `300`); round-robin multiplies by the board count before commanding the
firmware, while fixed-TX sends the displayed rate directly.

The binary encoding removed that failure rather than just moving it. Swept on real
hardware, **corruption was zero at every rate tried, for both subcarrier sets, right
through saturation** — including 166 SC at 98% UART, where the CSV encoding broke at
68%. The difference is that `ets_printf` had to *format* each line, holding the FIFO
far longer than writing prepared bytes does.

So the limit is no longer corruption; it is delivery falling behind what you ask for.
Past the knee, the boards keep sending valid frames and simply send fewer — which is
the failure mode you want, since nothing silently corrupts. Baseline delivery is
94–97% at any rate (that shortfall is ESP-NOW broadcast loss, flat in rate); the knee
is where it drops more than ~3 points below that.

Use `RATE <hz>` and `SUB <n>` on the serial line to re-measure rather than guessing —
those commands exist so the ceiling never has to be assumed again. It has already
been wrong once in each direction.

**Amplitudes are uncalibrated** ADC magnitude with per-board AGC compensation. Not
comparable across links or boards in absolute terms; normalise per link.

`NOTES.md` has the full log — including the failures, which are the useful part.
`RECORDING_2026-08-12.md` records exactly what `data/` contains and under what rig
configuration, so a result can always be traced back to the campaign that produced it.

---

## Firmware

```bash
docker compose run --rm csi bash -lc "cd /workspace/firmware && idf.py build"
docker compose run --rm csi bash -lc "cd /workspace/firmware && idf.py -p /dev/ttyACM0 -b 460800 flash"
```

Flash every board; they all run the same image. Port numbering shifts on replug, so
enumerate `/dev/ttyACM*` rather than assuming 0–3.

**These are ordinary app-partition writes — reversible, no eFuse, no secure boot,
no flash encryption.** Keep it that way.

The active target is pinned in `sdkconfig.defaults` (`CONFIG_IDF_TARGET="esp32c5"`),
not in a generated `sdkconfig`; without it the project silently builds for plain
esp32 and fails at link. Use `idf.py set-target esp32s3` before building for the
original boards; the source keeps target-specific CSI and status-LED paths for both.

Knobs at the top of `firmware/main/app_main.c`:

| symbol | now | what it does |
|---|---|---|
| `CONFIG_IQ_MODE` | 1 | 1 = raw int8 I/Q (frame v2), 0 = uint8 amplitude (frame v1) |
| `CONFIG_SEND_FREQUENCY` | 243 | pings/sec while holding the token; power-on default only |
| `CONFIG_SUB_COUNT` | 166 | 0 = all 192; 30 or 166 = the matching `SUB_INDEX_*` table |

**These three are coupled, plus the host's `--round-duration`.** Frame size sets the
ping ceiling; frame size and ping rate together set the best dwell. Changing one and
leaving the others is how you end up slower than before — 166 SC at the 30 SC dwell
measures *worse* than at its own. Re-measure with `RATE <hz>` / `SUB <n>` after any
change; the table in `NOTES.md` records what each combination gave.

`dependencies.lock` is tracked and `managed_components/` is not — `idf.py build`
restores the components from the lock, which also pins their content hashes.

Boards accept `TX`, `TX <n> [tag]`, `RX <mac>[,<mac>...]`, `RATE <hz>`, `SUB <n>`,
`BAND <2.4|5.6>`, `CHAN <n>`, `BW <20|40>`, `SCAN [ms]`, `IDENT`, `IDENT OFF`,
`LED r,g,b` on the serial line and reply with `ROLE_TX` / `TX_DONE,<n>,<tag>` /
`ROLE_RX,<count>,<macs>` / `RATE_OK,<hz>` / `BAND_OK,<band>,<ch>,<bw>` /
`CHAN_OK,<ch>,<bw>` / `BW_OK,<bw>,<ch>` / `SCAN_CH,...` lines. `TX <n>` sends `n`
pings and then returns to receiving by itself, announcing `TX_DONE`; plain `TX` pings
until an `RX` line arrives. `RX` takes up to 8 MACs and accepts CSI from any of
them. `BAND`, `CHAN`, and `BW` must be issued to every board together. CSI arrives
as binary frames interleaved with those text lines on the same UART:

```
        v1 (amplitude)              v2 (I/Q)
  0   2 magic A5 5A            0   2 magic A5 5A
  2   1 version = 1            2   1 version = 2
  3   1 n_sub                  3   1 n_sub
  4   6 transmitter MAC        4   6 transmitter MAC
 10   2 rssi, noise (int8)    10   2 rssi, noise (int8)
 12   4 timestamp µs LE       12   4 timestamp µs LE
 16   1 clipped count         16   2 AGC gain ×256, uint16 LE
 17   1 first_word_invalid    18   1 first_word_invalid
 18  ns amplitude uint8       19   1 reserved
                              20 2ns (imag, real) int8 pairs
 +2 sum16 over [2, end)        +2 sum16 over [2, end)
```

Version 3 (current) widens the header to 24 B: the raw `(agc_gain, fft_gain)` pair
travels at bytes 18-19 beside the Q8.8 factor at 16-17, `first_word_invalid` moves
to 20, and Q8.8 = 0 means "AGC not yet calibrated — do not scale" (the first ~100
frames after boot or a retune). `SUB 114` selects every HT-LTF subcarrier, skipping
the 52 five-fold-weaker legacy-LLTF duplicates; `STATS` reports on-board drop
counters that make wire loss distinguishable from radio loss.

I/Q is sent **uncompensated**, with the AGC factor in the header for the host to
apply — scaling on the board would round back into int8 and cost exactly the low
bits the phase estimate rests on. `tools/capture.py` reads both versions; `|iq|`
equals the v1 amplitude, so amplitude-only consumers need no changes.

---

## Provenance

Started from Espressif's `esp-csi` examples; the vendored tree has been dropped and
only the firmware project remains, so nothing here inherits assumptions from it any
more. `archive/` holds superseded datasets. The lab log in `NOTES.md` carries the
measurements behind the claims above.
