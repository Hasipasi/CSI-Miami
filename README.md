# csi-rig

A four-node WiFi-CSI sensing rig: four ESP32-S3 boards pass a transmit token
around a room while a depth camera watches, producing time-aligned **channel state
information + video** for human activity recognition and pose estimation.

Everything runs in one Docker container. The host needs Docker, an X server for the
GUI, and a separate Python venv only for the pose model (which wants CUDA).

---

## The hardware

**Four ESP32-S3 (N16R8) boards**, USB to one PC, all running identical firmware.
They are identified by MAC, never by `/dev/ttyACM*` — the path changes on every
replug and the tooling auto-discovers.

| label | MAC | USB serial |
|---|---|---|
| A | `14:c1:9f:c1:2d:3c` | 5C37261255 |
| B | `dc:da:0c:77:6b:5c` | 5C37262351 |
| C | `30:30:f9:1d:ab:d4` | 5C39018763 |
| D | `14:c1:9f:c1:2d:a8` | 5C39020696 |

**Labels name positions, not hardware.** If the boards are physically rearranged,
update `LABEL` in `tools/capture.py` and the coordinates in `NOTES.md`. Getting this
wrong silently mislabels every downstream figure; it has already happened once.

**Layout** (setup 3): a 3 m square, `D C` on the far row, `A B` on the near row,
diagonals 4.24 m. Measured link quality does *not* follow this geometry — see
`NOTES.md`.

**Intel RealSense D435i** on USB 3.0 for ground truth: 1280×720 colour at 30 fps.
Depth and IR work but are not currently recorded.

### How a capture works

One board holds a "transmit token" and pings at `CONFIG_SEND_FREQUENCY`; every other
board receives and reports CSI for that packet. The PC rotates the token over UART —
**round-robin**, giving all 12 ordered board pairs ("links") — or pins one board as
the sole transmitter, giving 3 links at ~4× the per-link rate.

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

Raising the ping rate makes bursts denser and leaves the gap untouched — only
`--round-duration` closes it. Its optimum depends on frame size, so it is not a
constant: 12.5 ms with 30 subcarriers, 25 ms with 166 (where 354 B is 3.84 ms of UART
and a shorter dwell fits too few frames to be worth the handoff). Change the
subcarrier count and this needs re-measuring.

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
├── data/              raw recordings, one directory per session
├── dataset/           packaged datasets built from data/
├── archive/           superseded datasets, kept for reference
├── Dockerfile
├── docker-compose.yml
├── NOTES.md           lab log: findings, failures, and why things are the way they are
├── RECORDING_2026-08-12.md   what is in data/: sessions, config, validation, caveats
└── .venv_pose/        CUDA venv, only for the pose model (not needed to record)
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
```

Only one process can own the serial ports at a time — stop the viewer before running
anything else that talks to the boards.

---

## The pipeline

```
firmware ──UART──▶ tools/capture.py ──▶ data/<session>/*.npz + *_frames/
                        │                        │
                   tools/viewer.py          tools/check_session.py   (validate)
                   (GUI + protocols)             │
                                          tools/extract_poses.py     (skeletons)
                                                 │
                        ┌────────────────────────┴───────────────┐
              tools/build_activity.py                   tools/build_pose.py
                        └────────────▶ dataset/ ◀───────────────┘
```

### 1. Record

`tools/viewer.py` is the normal way in: live camera beside per-link CSI waterfalls,
a REC button, and scripted protocols with a big cue card for whoever is standing in
the array. `tools/capture.py` does the same headless.

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
`data/<yaml name>/<take><round>.npz` plus a matching `_frames/` directory.

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

### 3. Extract skeletons (pose task only)

```bash
.venv_pose/bin/python tools/extract_poses.py --src data
```

`yolo11x-pose` over the recorded video, ~14 fps on an RTX 4050. Writes
`<take>_pose.npz` beside each capture. `tools/plot_pose_gt.py` renders a contact
sheet to eyeball the targets.

### 4. Build datasets

```bash
SUB=6,11,17,23,28,35,41,46,52,58,70,76,82,87,93,99,105,110,116,122,138,144,150,155,161,167,172,178,184,190

python3 tools/build_activity.py --src data --sessions '*_train,*_test' \
        --rate 30 --subcarriers $SUB --out dataset/csi5act_rr_v3
python3 tools/build_pose.py     --src data --sessions '*_train,*_test' \
        --rate 30 --context 20 --subcarriers $SUB --out dataset/csi5pose_rr_v3
```

Round-robin and single-TX captures have different link counts and cannot share a
dataset, so `--sessions` selects them separately (`'*_train,*_test'` vs `'*_txB'`).

Each dataset gets a `manifest.json` (read it first), `clips.csv`, `stats.npz`,
`clips/*.npz`, and for pose a `samples.csv` indexing every target.

---

## Things worth knowing before you change anything

**CSI is irregularly sampled.** The grid in a dataset is a resampling choice, and
`mask` says which cells are a fresh packet versus the nearest one held. Round-robin
at 30 Hz is ~32% fresh; ignoring the mask means treating stale values as
observations. The raw irregular packets ship in the activity clips so a different
grid can be built without re-recording.

**192 subcarriers carry ~2.6 effective dimensions.** Measured by PCA
(`tools/pca_subcarriers.py`). 30 evenly-spaced subcarriers reconstruct all 166 live
ones at R² = 0.973, which is why the datasets ship 30, and why the firmware now sends
only those 30. Picking the *highest-variance* 30 instead scores 0.831 — they cluster
and re-measure the same thing.

**The dataset builders do not read phase yet.** `build_activity.py` and
`build_pose.py` consume the `a` (magnitude) arrays. Captures made with v2 firmware
carry a `tx|rx|iq` array alongside, and it is currently ignored — silently, so this
is worth knowing before concluding that phase did not help.

**UART bandwidth is the binding constraint on sample rate**, not the radio. At
921600 8N1 the link carries 92.2 KB/s. Bandwidth arithmetic is necessary but not
sufficient: 100 Hz on the older CSV encoding fit on paper and still corrupted lines,
because `ets_printf` blocks on a full TX FIFO and starves the UART command task.

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

The target is pinned in `sdkconfig.defaults` (`CONFIG_IDF_TARGET="esp32s3"`), not in
a generated `sdkconfig`; without it the project silently builds for plain esp32 and
fails at link.

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

Boards accept `TX`, `RX <mac>`, `RATE <hz>`, `IDENT`, `IDENT OFF`, `LED r,g,b` on the
serial line and reply with `ROLE_TX` / `ROLE_RX,<mac>` / `RATE_OK,<hz>`. CSI arrives
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
