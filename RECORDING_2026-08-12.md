# 2026-08-12-recording

Snapshot record of the data collection this repository was built around. Written
2026-08-13 against the data as it stands; nothing here was inferred
from memory — every number was recomputed from the files.

**Recording ID:** `2026-08-12-recording`
**Location:** `data/` (16 GB) · **Derived:** `dataset/` (164 MB)

---

## What was recorded

**450 takes · 67,347 video frames · 3 subjects · 5 activities**, in two
radio configurations covering the same protocol.

Subjects: `gergo`, `balazs`, `yahya`
Activities: `salute`, `2handwave`, `squat`, `stretch-horizontal`, `stretch-vertical`
Structure: 10 takes per activity in `train`, 5 in `test`; 5.0 s each, 3 s lead-in,
1 s gap. The set is cycled, so repeats of one activity land minutes apart.

### Round-robin — all 12 ordered board pairs

| session | takes | links | per-link | frames | frame drops | recorded |
|---|---|---|---|---|---|---|
| `balazs_test` | 25 | 12 | 25.4 Hz | 3752 | 0.32% | 15:08–15:12 |
| `balazs_train` | 50 | 12 | 21.3 Hz | 7474 | 0.71% | 14:42–14:49 |
| `gergo_test` | 25 | 12 | 26.6 Hz | 3717 | 1.13% | 14:59–15:03 |
| `gergo_train` | 50 | 12 | 24.2 Hz | 7510 | 0.35% | 14:33–14:41 |
| `yahya_test` | 25 | 12 | 26.1 Hz | 3705 | 1.48% | 15:04–15:08 |
| `yahya_train` | 50 | 12 | 25.2 Hz | 7509 | 0.15% | 14:21–14:29 |

### Single transmitter (B pinned) — 3 links

| session | takes | links | per-link | frames | frame drops | recorded |
|---|---|---|---|---|---|---|
| `balazs_test_txB` | 25 | 3 | 111.0 Hz | 3762 | 0.27% | 15:42–15:46 |
| `balazs_train_txB` | 50 | 3 | 104.5 Hz | 7471 | 0.88% | 15:27–15:34 |
| `gergo_test_txB` | 25 | 3 | 109.9 Hz | 3762 | 0.11% | 15:46–15:50 |
| `gergo_train_txB` | 50 | 3 | 112.8 Hz | 7456 | 1.06% | 15:18–15:26 |
| `yahya_test_txB` | 25 | 3 | 108.8 Hz | 3760 | 0.29% | 15:51–15:54 |
| `yahya_train_txB` | 50 | 3 | 112.5 Hz | 7469 | 0.94% | 15:35–15:42 |

---

## Rig configuration at the time

| | |
|---|---|
| boards | 4 × ESP32-S3 (N16R8), labels A/B/C/D by MAC |
| firmware | binary CSI frames, `CONFIG_SEND_FREQUENCY` **125 Hz**, `CONFIG_SUB_COUNT` 0 (all 192) |
| CSI record | 212 B binary frame, uint8 amplitude, **no phase** |
| subcarriers | 192 reported, 166 live (26 are guard bands / DC, structurally fixed) |
| UART | 921600 8N1, ~15–17% utilised |
| camera | Intel RealSense D435i, 1280×720 colour @ 30 fps (depth not recorded) |
| geometry | 3 m square — `D C` far row, `A B` near row, diagonals 4.24 m |
| room | lab, subject standing in the middle of the array |

Skeletons were extracted afterwards with `yolo11x-pose` (imgsz 960): a person was
found in **100.00%** of all 67,347 frames.

---

## Validation

All 12 sessions pass `tools/check_session.py`: correct link set in every take, exact
class balance, monotonic host and board clocks, JPEG counts matching frame indices,
no silent boards, no corrupt frames, no stuck or all-zero amplitudes.

The only finding across the whole corpus is **dropped video frames: 240 of 33,680
(0.71%) in the txB half, 199 of 33,667 (0.59%) in the round-robin half**, appearing
as 3–6 frames in individual takes. Alignment is done from timestamps rather than
frame-index arithmetic, so this does not shift anything.

---

## Derived datasets

Built from this recording, all on a 30 Hz grid with the same 30 subcarriers:

| dataset | task | shape | clips |
|---|---|---|---|
| `dataset/csi5act_rr_v3` | activity | `[150, 12, 30]` | 225 |
| `dataset/csi5act_txB_v2` | activity | `[150, 3, 30]` | 225 |
| `dataset/csi5pose_rr_v3` | 2D pose | `[150, 12, 30]` + 33,750 targets | 225 |
| `dataset/csi5pose_txB_v2` | 2D pose | `[150, 3, 30]` + 33,750 targets | 225 |

Clip IDs match across activity and pose, so the two tasks join one-to-one.

---

## Known caveats

1. **Board labels were not re-verified with `IDENT` after the layout changed.** The
   letters are self-consistent throughout the data, so every model result holds; but
   any *spatial* claim (which link is which path, the geometry above) is unconfirmed.
2. **No empty-room reference in any session.** Prior work on this rig only succeeded
   when expressing effects as a ratio against an empty room.
3. **`balazs_*` runs ~13% fewer packets** than the other two subjects with 3× the
   spread. Not attenuation — RSSI is within 0.06 dB and per-link amplitude ratios are
   0.91–1.07×. Packets go missing while the ones that arrive are normal strength.
   Cause unknown.
4. **One subject position per session.** Earlier cross-validation on this rig fell
   from ~100% to ~50% under a position shift, so held-out scores here are optimistic
   about deployment.
5. **Amplitude only** — phase is discarded in firmware and is not recoverable from
   these files.

See `NOTES.md` for the measurements behind these and `README.md` for how the
pipeline works.
