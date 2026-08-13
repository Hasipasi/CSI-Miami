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
means twelve room-spanning paths, not twelve antennas. There is no phase either —
the firmware sends amplitude only.

| mode | links | per-link rate |
|---|---|---|
| round-robin, 4 boards | 12 | ~26 Hz |
| single transmitter | 3 | ~110 Hz |

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
ones at R² = 0.973, which is why the datasets ship 30. Picking the *highest-variance*
30 instead scores 0.831 — they cluster and re-measure the same thing.

**UART bandwidth is the binding constraint on sample rate**, not the radio. A CSI
record is a 212 B binary frame; at 921600 8N1 the link carries 92.2 KB/s. Bandwidth
arithmetic is necessary but not sufficient: 100 Hz on the older CSV encoding fit on
paper and still corrupted lines, because `ets_printf` blocks on a full TX FIFO and
starves the UART command task. Measure after any change.

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
| `CONFIG_SEND_FREQUENCY` | 125 | pings/sec while holding the token |
| `CONFIG_SUB_COUNT` | 0 | 0 = send all 192 subcarriers; 30 = send the table below it |

Boards accept `TX`, `RX <mac>`, `IDENT`, `IDENT OFF`, `LED r,g,b` on the serial line
and reply with `ROLE_TX` / `ROLE_RX,<mac>`. CSI arrives as binary frames — magic
`A5 5A`, version, subcarrier count, MAC, RSSI, noise floor, µs timestamp, clip
count, payload, sum16 — interleaved with those text lines on the same UART.

---

## Provenance

Started from Espressif's `esp-csi` examples; the vendored tree has been dropped and
only the firmware project remains, so nothing here inherits assumptions from it any
more. `archive/` holds superseded datasets. The lab log in `NOTES.md` carries the
measurements behind the claims above.
