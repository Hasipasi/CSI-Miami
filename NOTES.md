# CSI round-robin — working notes

State as of the end of the first session, so this can be picked up on another machine.

## Hardware

Four ESP32-S3 (N16R8) boards, all running identical `csi_roundrobin` firmware,
connected by USB to one PC. Boards are identified by MAC, not by `/dev/ttyACM*`
path — the path changes on every replug and the tooling auto-discovers.

| MAC | USB serial |
|---|---|
| `14:c1:9f:c1:2d:3c` | 5C37261255 |
| `dc:da:0c:77:6b:5c` | 5C37262351 |
| `30:30:f9:1d:ab:d4` | 5C39018763 |
| `14:c1:9f:c1:2d:a8` | 5C39020696 |

A fifth board (`ec:da:3b:4c:b8:d0` / 5C39018759) was **returned as faulty**. Its
transmit path was perfect (~50 pkt/s to every peer) but its receive path dropped
most packets, including signals arriving at -4 dBm. Ruled out by measurement:
distance/geometry, antenna orientation, noise floor (it had the *lowest* of the
five), per-board TX/RX bias (only ±1 dB across the set), and the USB cable (fault
did not follow a cable swap). The clincher was same-pair/opposite-direction at
matched path loss: 49.5 pkt/s one way vs 4.5 the other, ~1 dB apart.

The remaining four are well matched, all 12 directed links run 46.8-50.0 pkt/s,
and five of six pairs agree within ~1 dB in both directions.

## Things learned the hard way

- **Orientation dominates.** One board lying flat instead of upright dropped it
  from ~49 to ~5.5 pkt/s as a transmitter — polarisation mismatch, worth far more
  than any bench-scale distance change. Keep every board upright, same height,
  antenna end clear, cables routed away from the antenna.
- **UART sets the ceiling, but 100 Hz fits — the old 50 Hz limit was stale.**
  921600 8N1 carries 92.2 KB/s; a *raw I/Q* line was ~1215 B, which is what made
  100 Hz saturate. Dropping phase halved it: measured over real captures a line is
  626 B (603-642). A receiver sees the full ping rate during a turn and is RX for
  3 of every 4 turns, so 100 Hz costs 62.6 KB/s burst (68%) and 46.9 KB/s average
  (51%). 150 Hz would need 93.9 KB/s burst — over the link — so 100 is the ceiling
  for this encoding. Raised 2026-08-12; no truncated lines observed (every capture
  came back at the full 192 subcarriers).
- **A restarted periodic timer loses its first interval.** `become_tx()` called
  `esp_timer_start_periodic`, whose first callback fires one whole period later, so
  the opening 20 ms of every 50 ms dwell was silent — each turn yielded ~1.3 of the
  2.5 packets it should. Firing one ping directly after the start fixed it. Combined
  with the 100 Hz change, per-link rate went **5.2 -> 14.4 Hz** and, because short
  dwells are no longer penalised, a 50 ms dwell now beats 100 ms on *both* rate and
  revisit interval (14.4 Hz / 170 ms vs 14.6 Hz / 314 ms).
- **`ets_printf` is not atomic across tasks.** CSI prints from the WiFi task and
  role markers from the UART command task; concurrent calls interleave mid-line
  and destroyed ~23% of role markers until a mutex was added.
- **Diagnostics need to fail loudly in both directions.** A marker-loss check
  that only detected one failure mode reported "0% loss" while a quarter of the
  markers were missing.

## Tools

- `roundrobin_viewer.py` — combined scheduler + live viewer. Per-(receiver,
  transmitter) binned waterfalls; each link binned separately so a bin spanning a
  handoff cannot blend two links. `--bin-hz` sets integration rate.
- `pairwise_matrix.py` — full directed link matrix (rate + RSSI), `--save` /
  `--compare` for before/after hardware changes. Keyed by MAC.
- `pose_suitability.py` — capture in `fixedtx` or `roundrobin` mode, then
  `analyze` for sampling rate, motion-vs-static variation, channel coherence,
  effective dimensionality and link diversity.

## Ground-truth camera

Intel RealSense **D435i**, serial `039223051974`, on USB 3.0 at 5000M (full link
speed -- at 480M the firmware silently drops the higher stream modes).

| node | stream | verified |
|---|---|---|
| `/dev/video2` | Z16 depth | 640x480 @ 90 fps, centre patch 99% valid |
| `/dev/video4` | Y8I/Y12I infrared (both imagers) | enumerated |
| `/dev/video6` | YUYV colour | 640x480 @ 60 fps |

`video3/5/7` are the matching metadata nodes. Verified by raw V4L2 capture with
no SDK (`rs_probe.py`), so the camera, cable, link and permissions are known-good
independently of librealsense -- which is **not installed** on the host or in the
container yet. A `cp312` manylinux wheel of `pyrealsense2` 2.58.3 exists, so
`pip install pyrealsense2` is all that is needed.

Container access needed `c 81:* rmw` and `c 234:* rmw` in `device_cgroup_rules`
(see `docker-compose.yml`); the `/dev` bind mount alone makes the nodes visible
but not openable, exactly as with the serial ports.

## Room geometry (setup 3, 2026-08-12, 4 boards)

A 3 m square, two rows: **D C** on the far row, **A B** on the near row.

| board | MAC | x (m) | y (m) |
|---|---|---|---|
| A | `14:c1:9f:c1:2d:3c` | 0.00 | 0.00 |
| B | `dc:da:0c:77:6b:5c` | 3.00 | 0.00 |
| C | `30:30:f9:1d:ab:d4` | 3.00 | 3.00 |
| D | `14:c1:9f:c1:2d:a8` | 0.00 | 3.00 |

Neighbours 3.00 m, diagonals 4.243 m (measured as 4.25). Four link orientations --
0 deg (A-B, C-D), 45 deg (A-C), 90 deg (A-D, B-C), 135 deg (B-D) -- so the array is
angularly well spread, unlike setup 2 where two pairs were near-parallel.

**Geometry does not predict link quality here.** Measured single-link activity
accuracy over the 12-08 corpus spans 18.7% to 38.7% (chance 20%), and it does not
follow the layout at all:

| pair | length | angle | distance from centre | accuracy |
|---|---|---|---|---|
| A-D | 3.00 m | 90 deg | 1.50 m | **38.7%** |
| C-D | 3.00 m | 0 deg | 1.50 m | 28.0% |
| B-D | 4.24 m | 135 deg | **0 m** | 28.0% |
| A-C | 4.24 m | 45 deg | **0 m** | 28.0% |
| A-B | 3.00 m | 0 deg | 1.50 m | 21.3% |
| B-C | 3.00 m | 90 deg | 1.50 m | **18.7%** |

The four edges are geometrically identical -- same length, same 1.50 m from the
centre -- yet span 18.7% to 38.7%. The two diagonals pass *through* the centre where
the subject stands and are only mid-table. Correlation between accuracy and
distance-from-centre is **-0.10**, i.e. none. So link quality is set by multipath,
board orientation and surroundings, not by the array layout -- the same conclusion
the RSSI-vs-distance result below reached by another route. Do not explain link
strength geometrically without measuring it.

Setup 2 (2026-08-10, a 3.9 x 3.25 m rectangle) applies only to the archived corpus
in `data/_archive_2026-08-10_poses/`.

## RSSI is useless for distance here -- localization by multilateration is dead

With true distances known, measured RSSI correlates **positively** with distance
(+0.59), giving a fitted path-loss exponent of **-2.92** (free space is +2.0,
indoor 2-4; negative is unphysical). Concretely CD spans 4.09 m at -39.3 dBm
while BC spans 2.95 m at -55.6 dBm -- the shorter link is 16 dB weaker.
Reciprocity is fine (asymmetry 0.0-2.8 dB), so this is the propagation
environment, not the hardware. Multipath and orientation dominate distance
completely at room scale. This confirms, with ground truth, the earlier failure
where MDS produced a non-Euclidean distance matrix.

CSI fingerprinting against surveyed positions remains plausible; RSSI
trilateration does not.

## Validation status

**Hardware: validated.** Four boards matched to +/-1 dB TX/RX bias, all 12
directed links 46.8-50 pkt/s, reciprocity 0.0-2.8 dB. The one defective board
was identified and characterised by these measurements.

**Round-robin method: validated.** 0% role-marker loss over 289 handoffs at 50 ms
rounds, per-link data cleanly attributed by transmitter, 12 links at 9.5 Hz.
Presence detectable at z~27; held poses separable at 22.3 sigma median.

Validated as *sufficient*. Whether round-robin is *optimal* versus fixed-TX for
pose work is still open -- see the link ablation and the missing fixed-TX pose
capture below.

## Pose-estimation assessment

Measured with `capture_wizard.py` (three conditions: empty room / person still /
person moving) and `analyze_wizard.py`. The empty-room reference is what an
earlier attempt lacked, and it passed its drift check this time (ratio 0.0-0.4,
lag-1 autocorr ~0, i.e. a genuine white-noise floor).

**Presence is trivially detectable.** A *motionless* person shifts the channel
mean by z = 16-66 sigma (mean 27.5 across 12 links). An earlier read of
"undetectable" was wrong: it compared variances, which are blind to a constant
offset. This matters because it means body configuration is encoded in the
static channel signature, not only in motion -- the precondition for reading a
held pose.

**Motion is clearly detectable.** Round-robin 3.45x vs empty, fixed-TX 2.14x.

**Round-robin beats fixed-TX for this task**, reversing an earlier verdict that
was based on sampling rate alone:

| mode | links | rate | motion vs empty | effective rank |
|---|---|---|---|---|
| fixed-TX | 3 | 46 Hz | 2.14x | 6 |
| round-robin | 12 | 5 Hz | 3.45x | 9 |

**The real limit is dimensionality, not rate.** After removing the empty-room
mean, the moving data has an effective rank of only 6-9 (95% of variance) out of
576-2304 raw features. A human skeleton is 30-60 DOF, so full skeletal pose is
underdetermined by roughly 5x. Feasible: presence, occupancy, activity
recognition, coarse pose classification. Not feasible with this setup: dense
skeletal pose.

Caveat on that rank figure: it was measured on a single motion sequence (arms,
turn, squat). A low rank may partly reflect limited motion diversity rather than
a sensor limit -- the same confound that made the earlier line-geometry numbers
useless.

### Held-pose discrimination -- the decisive test

Six held poses plus an empty room, round-robin, `analyze_poses.py`:

| | |
|---|---|
| chance | 14.3% |
| leave-one-out | 97.9% (optimistic: adjacent windows correlate) |
| **temporal split** | **88.7%** (train first half of each hold, test second half) |
| median pairwise separability | 22.3 sigma, no pair below 3 |

With four boards, **amplitude only**, 12 s per pose and a nearest-centroid
classifier on 20 PCs. The only real confusion is T-pose vs arms-up (both put the
arms away from the torso), which is semantically coherent rather than arbitrary
-- mild evidence the classifier keys on body configuration, not session noise.

This overturns an earlier pessimistic reading here that compared "effective rank
6-9" against 30-60 skeleton DOF. That was wrong twice over: pose lives on a
low-dimensional manifold so raw DOF is the wrong denominator, and the rank
itself was unmeasurable from that data (round-robin's "120 components above
noise" was exactly its 120 time samples -- observation-limited, not
channel-limited).

### Link-count ablation (same data, same classifier)

| links | accuracy |
|---|---|
| all 12 | 88.7% |
| 6 (two TX) | 85.6-88.7% |
| 3 (one TX) | 82.5-84.5% |

4x the links buys ~5 points -- strongly sublinear, matching the measured link
redundancy. Note these 3-link figures come from round-robin data at 9.5 Hz; a
real fixed-TX capture supplies the same 3 links at ~46 Hz, ~2.2x less
per-window noise, so fixed-TX may match or beat round-robin on held poses.
**Fixed-TX poses were never captured** -- that comparison is open.

![Per-link dB change by pose](esp-csi/examples/get-started/tools/pose_db_rr.png)

Regenerate with `plot_pose_db.py --prefix ps`. The body **redistributes** signal
rather than absorbing it: net mean +0.34 dB while individual links swing -9.6 to
+13.5 dB. A->B (4.70 m, strong, direct path through the middle) loses on every
pose; A<->C (2.30 m but weak, a destructive-interference link) gains 2-13 dB
because the body acts as a scatterer filling a null. The ~11 dB spread across
poses on that one link is where most of the classification power sits -- the
weakest link is the most informative one.

### Cross-session generalization -- a data-scale limit, not a hardware one

Session 2: same six poses, subject deliberately standing in a slightly shifted
spot. Model (scaling, PCA basis, centroids) fitted on session 1 only, frozen,
applied to session 2 (`analyze_crosssession.py`).

| | |
|---|---|
| within session 2 (temporal split) | 100.0% |
| **cross-session** | **47.5% (train stats) / 51.1% (per-session centring)** |
| chance | 14.3% |

So a model trained at one standing spot does not transfer to another. This is a
statement about training data, not about the instrument: 6 poses x 12 s from a
single position is far too little to expect position invariance. Only `empty`
(49/49) and `up` (15/15) transferred.

But it is *not* a total failure, and the reason matters. Cosine similarity
between session-1 and session-2 pose signatures (each session mean-removed):

- same pose across sessions: **+0.57** mean
- different poses: **-0.08** mean

Transferable pose information clearly exists. What fails is the *margin*: four
of seven diagonals are strong (0.63-0.75) but several off-diagonals are just as
high (turned/turned 0.73 vs turned/crouch 0.76; split/split 0.63 vs split/tpose
0.74), so nearest-centroid picks wrong by a hair. This also rules out two
alternative explanations -- a systematic label/timing shift would make one
off-diagonal dominate consistently, and position scrambling the signatures
outright would leave the matrix unstructured. Neither is what we see.

Confound: position and time both changed between sessions, so their
contributions cannot be separated here. Position is very likely dominant.

### Fixed-TX poses, and a controlled rate test

Fixed-TX pose capture (board A transmitting, 3 links at 41.5 Hz), same six
poses: **97.9%** temporal split, median separability 24.2, two errors in 97 test
windows.

Decimating that same capture -- same session, same position, same links, only
the rate reduced -- isolates the effect of sample rate:

| per-link rate | accuracy |
|---|---|
| 41.5 Hz | 97.9% |
| 20.8 Hz | 96.9% |
| 10.4 Hz | 96.9% |
| 6.9 Hz | 93.8% |

**Sample rate barely matters for held poses.** A 4x decimation costs one point.
This makes sense in retrospect: a held pose has no temporal content to resolve,
so what is needed is a clean spatial signature rather than a fast one. It also
refutes the earlier prediction here that fixed-TX would beat round-robin *via*
its higher rate.

**The mode comparison is therefore still not resolved, and probably cannot be
from this data.** At matched rate and link count fixed-TX scores 96.9% against
the round-robin 3-link subset's 82.5-84.5%, but those come from different
sessions and standing positions, and round-robin's own two sessions spanned
88.7% to 100%. Session-to-session variation is as large as the gap that would
be attributed to mode. Settling it needs both modes captured back-to-back from
one position.

Practical consequence: round-robin's ~9.5 Hz per-link rate, flagged early in
this project as disqualifying for pose work, is **not** a handicap for static
pose. Rate would only matter for tracking motion, which is a different task.

### Next steps

1. **Train across multiple standing positions** (3-5 spots). Still the single
   most valuable next dataset -- see the cross-session result above.
2. **A better classifier than nearest-centroid**, once multi-position data
   exists.
3. **Mode comparison back-to-back from one position**, if it matters; both
   modes already work well enough that this is low priority.
4. ~~**Restore phase** (stripped for UART bandwidth).~~ Done -- see below.

## 2026-08-13 -- I/Q firmware, and the UART ceiling was never where we thought

Everything below in one table. All three columns are **measured from real captures**
of the same kind -- the first from `data/gergo_train/salute0.npz` (the recorded
campaign), the other two from 20 s test captures -- and all use the same 12 links and
the same statistics, so the columns are comparable rather than quoted from different
kinds of run.

| | campaign (v1) | I/Q, old dwell | I/Q 30 SC, tuned | **shipped: 166 SC** |
|---|---|---|---|---|
| payload | uint8 amplitude | int8 I/Q | int8 I/Q | int8 I/Q |
| phase | no | yes | yes | **yes** |
| subcarriers | 192 (166 live) | 30 | 30 | **166, all live** |
| frame | 212 B | 82 B | 82 B | **354 B** |
| ping rate | 125 Hz | 400 Hz | 675 Hz | **243 Hz** |
| dwell | 50 ms | 50 ms | 12.5 ms | **25 ms** |
| cycle (4 boards) | 200 ms | 200 ms | 50 ms | **100 ms** |
| per-link rate | 25.8 Hz | 92.3 Hz | *118.4 Hz* | **43.5 Hz** (1.7x campaign) |
| median gap | 8.19 ms | 1.86 ms | *1.39 ms* | **4.19 ms** |
| p99 gap (blind window) | 192.5 ms | 165.8 ms | *71.4 ms* | **99.9 ms** (1.9x better) |
| duty cycle | 14.0% | 12.0% | 10.7% | **14.6%** |
| UART burst load | 26.5 KB/s (29%) | 32.8 KB/s (36%) | 55.4 KB/s (60%) | **86.0 KB/s (93%)** |
| checksum failures | 0 | 0 | 0 | **0** |

All four columns are **measured from real captures** with identical statistics over the
same 12 links -- the first from `data/gergo_train/salute0.npz`, the rest from 20 s test
captures -- so they compare like with like.

**The shipped column is not the best on rate or gap, and that is the deliberate
choice.** 30 SC wins on both (118 vs 43 Hz, 71 vs 100 ms). It was chosen anyway
because the R^2 = 0.973 result that justifies 30 subcarriers was measured on recorded
*amplitude*, and nothing has yet checked whether it holds for phase. Subsetting stays
possible in analysis; discarding at the boards does not. Switch with `SUB 30` plus
675 Hz and a 12.5 ms dwell if the rate turns out to matter more.

**The three knobs are not independent.** Frame size sets the ping ceiling, and ping
rate and frame size together set the best dwell: 166 SC is 3.84 ms of UART per frame,
so a 12.5 ms dwell fits too few frames and measures *worse* than 25 ms (p99 106.6 vs
101.2 ms). Change one, re-measure all three.

Read the two gap rows together; they answer different questions. **Median** is spacing
*inside* a burst and follows ping rate. **p99** is the blind window between bursts and
follows dwell. Raising rate alone (column 2) barely moved p99: 192.5 -> 165.8 ms for
3.2x the pings. Dwell is what moves it. Tune dwell first.

Duty cycle sits at 11-15% throughout, which is arithmetic rather than a shortfall:
duty is roughly dwell/cycle, and shortening dwell shortens both. What improves is how
*often* a link is revisited.

At 93% UART the shipped config has little headroom left, which is why it is the knee
minus 10% rather than the knee. It sustained 20 s of real capture and a 15 s sweep
with zero checksum failures and all boards responsive, but it is the closest to the
edge anything here has run -- watch `check_session` on the first real session.

Phase is back. `CONFIG_IQ_MODE` emits raw int8 I/Q as frame version 2 instead of
computed uint8 amplitude, and `CONFIG_SUB_COUNT` is now 30. Those two go together:
30 complex subcarriers is an 82 B frame against 212 B for 192 amplitudes, so phase
costs *nothing* in bandwidth and buys rate at the same time. 192 complex would have
been 404 B and not worth it.

The version byte carries the format, so `capture.py` reads both encodings and the
450 v1 takes stay readable through the same code path. `|iq|` reproduces the v1
amplitude exactly, so amplitude-only consumers needed no changes at all.

**I/Q is sent uncompensated with the AGC factor in the header.** Applying gain on
the board means rounding a scaled value back into int8, which destroys exactly the
low-order bits phase is estimated from. The host has floats; there it is free.

### The 68% UART limit was an artefact of ets_printf, not of the UART

Swept 400-1000 Hz with the settle window excluded: **zero checksum failures at every
rate, up to 84% of the link**, every board still accepting commands afterwards. The
CSV encoding corrupted at 68%. The difference is not bandwidth -- it is that
`ets_printf` *formatted* each line while holding the FIFO, where the binary path
writes bytes that are already prepared. The old note "bandwidth is necessary, not
sufficient" was right about the mechanism and wrong about the number.

An earlier sweep of mine reported 3-6 corrupt frames at the higher rates. That was
my own measurement flushing the serial buffer mid-frame and charging the resync to
the rate; the count was flat in rate, which is what gave it away. Excluding a 1.5 s
settle window it is zero everywhere.

Delivery sits at 93-95% at every rate. That shortfall is ESP-NOW broadcast loss and
is flat in rate, so it is not a symptom of pushing too hard.

`CONFIG_SEND_FREQUENCY` is 400 (~100 Hz per link in round-robin, against 26 before),
which is a headroom choice, not a ceiling. A new `RATE <hz>` serial command retunes
it live, so nobody has to reflash four boards to re-measure this again.

### Raw phase is uniform noise; detrended phase is worth having

Measured on a static scene, and worth stating plainly because it would be easy to
plot raw phase, see structure that isn't there, and build on it:

| | sd across packets |
|---|---|
| raw phase | 1.840 rad |
| uniform random on [-pi, pi] | 1.814 rad |
| after per-packet linear detrend across subcarrier index | **0.068 rad** |

Absolute phase is carrier and sampling offset, redrawn every packet -- statistically
indistinguishable from noise. Fit a line across subcarrier index within each packet,
subtract it, and what remains is stable to ~0.07 rad. That 0.07 is the *static* noise
floor, so motion must move phase more than that to be visible. **Anything consuming
phase must detrend first**; the raw values are not a usable signal.

Confirmed independently on a 20 s round-robin capture: 0.072 rad mean over all 12
links, against 0.068 from the pinned-TX measurement.

**Two different standard deviations live here, and they are easy to confuse** -- I
briefly misread one as contradicting the other. Take the detrended residual `res`
with shape [packets, subcarriers]:

* `np.std(res, axis=0).mean()` = **0.07**. Temporal stability per subcarrier. This is
  the noise floor, and the number quoted above.
* `res.std()` = **0.6**. Pools subcarriers together, so it also contains the static
  differences *between* subcarriers -- which is the channel's frequency response, i.e.
  signal, not noise.

The second is not a worse version of the first; it answers a different question.

### Endurance, since throughput alone never was the failure mode

6 min round-robin at 400 Hz with the real 50 ms dwell: **387,045 frames, 0 checksum
failures, no link ever silent, all four boards still accepting commands afterwards**,
across 23,672 role changes. Role changes under print pressure are exactly what used to
starve the UART command task, so that count is the part that matters. Repeated at the
final 675 Hz for 3 min: 274,339 frames, 0 failures, all responsive, 127 Hz per link
against 26 Hz on the amplitude firmware.

### Where each subcarrier set actually breaks

Pushed past the knee on purpose -- a ceiling nobody has watched fail is the top of
someone's sweep, not a measurement. Criterion is delivery falling >3 points below that
configuration's own low-rate baseline, since baseline is ~95%, not 100%, from ESP-NOW
loss.

| set | frame | knee | UART at knee | limited by | -10% | per link (RR) |
|---|---|---|---|---|---|---|
| 30 SC | 82 B | ~750 Hz | ~65% | per-frame cost | **675** | **127 Hz** |
| 166 SC | 354 B | ~270 Hz | ~97% | bandwidth | 243 | ~57 Hz |

**These are two different bottlenecks and it matters.** 166 SC saturates the wire
exactly where 354 B a frame says it should. 30 SC gives up at *two-thirds* of the
link, so bytes are not what stops it -- per-frame overhead is (mutex, per-byte ROM
writes, ESP-NOW send rate). Predicting the 30 SC ceiling from a byte count will
overestimate it by ~50%.

Corruption was **zero everywhere**, in both sets, right through saturation. Past the
knee the boards send valid frames and simply send fewer. The old "corrupt lines"
failure mode is gone with `ets_printf`; what remains is honest under-delivery.

Note the 30 SC knee is soft and moves run to run: 800 Hz passed one sweep at 95.1%
and failed two others at 88-93%. 750 passed everywhere, hence 675 rather than 720.

### Per-link rate is an average across a burst and a blind gap

The single most misleading number in this project. Measured on a real 20 s capture at
50 ms dwell:

| | |
|---|---|
| median gap between packets on a link | **2.25 ms** |
| p99 gap | **165 ms** |
| duty cycle | **16%** |

A link is sampled densely while its transmitter holds the token and then not at all
for the rest of the cycle. "127 Hz per link" is a rate averaged over both states; no
link is ever sampled at 127 Hz. At 30 fps video that gap is ~5 consecutive frames with
no fresh CSI on that link, which is where the datasets' 32%-fresh mask comes from.

**Raising the ping rate does not touch this.** It makes bursts denser and leaves the
gap exactly as it was. The lever for temporal coverage is the host's
`--round-duration`.

### A 10 ms poll was costing 5 ms of every dwell

Sweeping dwell at first said shorter was worse below ~12.5 ms: the p99 gap bottomed
out and then climbed again, and yield fell off a cliff (36% at 5 ms). Backing out the
loss per handoff gave ~5 ms, constant across dwells -- a suspiciously round number for
something that ought to be physics.

It was not physics. `uart_command_task` polls a non-blocking stdin and slept
`pdMS_TO_TICKS(10)` between looks, so a role change waited on average 5 ms just to be
*read*. The tick is already 1 kHz, so that delay was ten ticks for no reason. One
line, 10 ms -> 1 ms:

| dwell | p99 gap before | after | per-link rate before | after |
|---|---|---|---|---|
| 12.5 ms | 82 ms | **63 ms** | 103 Hz | 117 Hz |
| 8 ms | 93 ms | 66 ms | 84 Hz | 108 Hz |
| 5 ms | 148 ms | 68 ms | 61 Hz | 99 Hz |

Short dwells stopped collapsing, and 5 ms went from unusable to merely pointless.

### Dwell: 12.5 ms, and shorter stops helping

`--round-duration` now defaults to 0.0125 in both `capture.py` and `viewer.py`.
Measured on real 20 s captures, before and after everything above:

| | rate | median gap | p99 gap | duty |
|---|---|---|---|---|
| 400 Hz, 50 ms dwell | 92.3 Hz | 1.86 ms | 165.8 ms | 12.0% |
| **675 Hz, 12.5 ms dwell** | **118.4 Hz** | 1.39 ms | **71.4 ms** | 10.7% |

The blind gap more than halved while the rate went up. Below 12.5 ms the gap stops
improving and rate falls, so that is the floor -- with the handoff cost fixed, the
remaining limit is that a link cannot be revisited faster than the token can go round.

Duty cycle barely moved (12.0% -> 10.7%), and that is expected rather than
disappointing: duty is roughly dwell/cycle, which shortening dwell does not change.
What changed is how *often* a link is visited, which is the part that matters for
aligning CSI to 30 fps video.

### Still open

- The dataset builders (`build_activity.py`, `build_pose.py`) do **not** read the new
  `tx|rx|iq` array. They consume `a` and will silently use magnitude only.
- Clipping cannot occur in v2 (nothing is scaled into a uint8), so the clip counter is
  absent by construction rather than always zero.
- Rates were measured with boards on a desk. Nothing about the UART depends on
  geometry, but delivery percentage will change once they are spread out.

## 2026-08-18 — macOS port, the channel was never measured, and the field layout was wrong

Session on gergo's Mac with only boards A and D attached. Three findings that
change how the rig should be read, each measured on live hardware.

**The rig runs natively on macOS; the container cannot.** Docker Desktop runs
containers in a Linux VM, so `-v /dev:/dev` exposes the VM's empty `/dev` — no USB
passthrough exists, verified with `--privileged`. The tools now run from a repo-local
`.venv_mac` instead. `capture.py` grew a `MacCamera` backend (AVFoundation via
OpenCV) behind the same interface as the V4L2 `Camera`; `--device` takes a node
path (Linux), an index, or a camera *name* on macOS. Names matter: OpenCV's index
order is the *reverse* of AVFoundation's list order on this machine, and index 0 was
silently the iPhone Continuity Camera, not the built-in one. Cameras are matched by
fingerprint (their largest-area mode) instead of by position. Two honest losses on
this backend, both recorded in metadata: no driver timestamps (`monotonic=False`,
grab-loop jitter ~1 frame) and no driver sequence numbers (OS-dropped frames are
invisible — do not quote frame-drop figures from macOS sessions).

**Transport re-verified from scratch, 2 boards.** Zero corrupt frames at every rate
tried (100–400 Hz at SUB 166, 243–900 Hz at SUB 30), delivery 90–96% below the
knee, knee at ~260 Hz / ~99% UART for 166 — the Linux numbers reproduce on a Mac
to within a few percent. Round-robin handoff costs nothing measurable: 221–226 Hz
total at every dwell from 25–200 ms against 220 Hz pinned, median gap 4.13 ms =
one ping period. The transport is not why any recording was weak.

**The documented "three 64-wide fields" is wrong; the data says two.** Live capture
of all 192 subcarriers: blocks 64–127 and 128–191 have the same mean amplitude
within 3% (43.1 vs 44.3) while 0–63 sits 4.8× lower (9.2). 64–191 is one 128-wide
HT-LTF whose centre null is the dead band at 123–133; the only gain step is at 64.
Worse, the viewer z-scored per 64 rows *of the compacted frame*, where the real
boundary lands at row 52 — so two of its three blocks straddled a 5× gain step and
the banding buried the channel shape (this is the "display looks unordered/noisy"
complaint; display-only, recordings were always raw). The viewer now *derives* the
field split from a running mean (`derive_fields` in capture.py) instead of trusting
any table, and draws each field as its own sub-plot per link (the seam is the
boundary; heights proportional to subcarrier count; waterfalls default to jet,
`--cmap diverge` restores the dark-midpoint map). Threshold is 3× because a real
multipath fade held a 2× step across a whole window and produced a phantom
boundary at row 106 before the margin was raised. Subcarrier rows are in ascending
frequency order throughout — adjacent-subcarrier correlation median 0.78 with no
wrap discontinuity — but not evenly spaced once compacted: row 51→52 jumps
subcarrier 58→66, row 108→109 jumps 122→134.

**`CONFIG_LESS_INTERFERENCE_CHANNEL 11` was a hope, not a measurement.** The
firmware now has `SCAN [ms]` (parks the radio, counts promiscuous traffic per
channel 1–13 at HT20 — a ranking of 802.11 airtime, blind to non-WiFi noise),
`CHAN <n>` (moves the rig live; at HT40 the secondary goes below a primary ≥5,
above otherwise, and the ESP-NOW peer channel is updated with it), and `BW <20|40>`
(runtime bandwidth switch; channel must be re-applied *before* the peer rate config
or the boards keep transmitting HT20 while claiming HT40 — found the hard way, RX
kept reporting 128 subcarriers). Survey of this room: classic 1/6/11 congestion
with a −25 dBm emitter on ch6; every HT40 placement overlaps it (all 13 primaries
within 11.9–16.8 KB of contending airtime per 6.5 s survey), so at 40 MHz there is
nothing to dodge. At 20 MHz there is: **HT20 on ch3 delivered 242.3/243 Hz =
99.7%, against 91–95% for HT40 on ch11.** Even HT40 improves on ch3 (98.2%).
HT20 costs half the span: 128 subcarriers (64 LLTF + 64 HT-LTF, boundary at ~64)
instead of 166/192. Whether 3-dB-quieter narrowband beats 2× frequency diversity
for recognition is an open question the BW button in the viewer exists to answer —
a recording's own frames say which was active via n_sub.

Viewer additions: STOP/START CSI (parks every board via `RX 000000000000`, the
firmware's own match-nothing filter; refuses mid-take), FIND BEST CHANNEL (scans
all boards, pools bytes and worst RSSI per channel, scores HT40 *spans* not
primaries, moves the rig if a better block exists), BW toggle, field-boundary
markers, and link counts that follow however many boards are attached. Recordings
this session go to `/Volumes/GergoDisk/csi-data` (exFAT stick, 147 MB/s).

**The boot channel is now 13** (`CONFIG_LESS_INTERFERENCE_CHANNEL`), not the
inherited 11: four hand-run surveys picked ch13 unanimously (11–18% less contending
airtime than ch11), and delivery measured after the reflash confirms it — pinned
A→D at 243 Hz/SUB 166 lost 5.8% on ch13 against 10–15% on ch11 the same day. That
is back inside the 4–9% ESP-NOW no-ACK baseline; the remaining loss is occupied
air, which no 40 MHz placement escapes in this room.

Boards A and D are running this firmware (built in the stock `espressif/idf:release-v5.5`
image — the repo's own Dockerfile currently fails at the pip layer and needs fixing).
**B and C still run the previous build: old boot channel 11, no SCAN/CHAN/BW. Flash
them before the next full-rig session or they will sit deaf on a different channel
and look like dead serial links.**

## 2026-08-18 (later) — the firmware audit, and loss drops 3× at every operating point

A five-finding hostile audit (fifteen agents, every finding confirmed by two
independent skeptics, one of whom disassembled `libesp_csi_gain_ctrl.a` to check the
gain-path claim) found that the transport's own architecture was masquerading as
radio loss. All fixes are in and measured, boards A and D flashed.

**The root defect: every CSI frame was shifted into the UART FIFO byte-by-byte from
inside `wifi_csi_rx_cb` — which runs in the WiFi task.** 3.84 ms of busy-wait per
166-SC frame, ~93% of the radio task's time at 243 Hz. That single mechanism *was*
the measured 166-SC knee (354 B = 3.84 ms = 260 Hz exactly), most of the per-frame
cost capping 30 SC at ~750 Hz, the 10% operating derate, and a slice of what was
booked as "ESP-NOW baseline loss". ESP_LOG additionally bypassed `print_mux` from
the other core (mid-frame interleave — the old ~23%-of-markers bug, still half-open),
and console RX was the bare 128 B hardware FIFO, which is how a SCAN used to eat
role commands ("dead board" incidents).

**The fix**: `uart_driver_install` (24 KB interrupt-driven TX ring, 2 KB RX ring),
everything — frames, markers, acks, and via `uart_vfs_dev_use_driver` the logs too —
enqueued through one atomic path; a full ring drops whole frames and counts them
(`STATS,framedrops,…`), so the wire saturates visibly instead of stalling the radio.
The 30 s task-WDT override in sdkconfig.defaults (which papered over the busy-wait)
is removed. Measured, same channel, same afternoon:

| point | loss before | loss after |
|---|---|---|
| 166 SC @ 243 Hz | 2.66% | **0.84%** |
| 30 SC @ 675 Hz | 7.85% | **2.49%** |
| 30 SC @ 900 Hz | 5.33% | **2.99%** |
| 30 SC @ 1300 Hz | collapsed at 86.6% | delivers 1072.8 Hz — the v3 wire limit (1071.6) to 0.1% |

Per-frame cost is gone: the UART's byte rate is now the only ceiling, and the
loss cliff above the knee is replaced by counted, attributable drops.

**Frame version 3** (24 B header): raw `(agc_gain, fft_gain)` pair at bytes 18-19
beside the Q8.8 factor, and Q8.8 = 0 reserved as an explicit "AGC not calibrated"
sentinel. v2 shipped gain=1.0× for the first ~100 frames of every session — the
component returns INVALID_STATE until its baseline latches and the return was
ignored — so the start of every recording, where reference windows live, carried
uncalibrated AGC presented as calibrated. The AGC baseline also now restarts on
every CHAN/BW retune (it was boot-time-stale before). Host parses v1/v2/v3.

**SUB 114** (new table): the audit confirmed what the band plot showed — 52 of the
"166 live" subcarriers are the legacy LLTF at ~5× lower gain (~2.9° int8 phase
quantisation noise vs ~0.6° in the HT-LTF), duplicating spectrum the HT-LTF already
covers. 114 = every HT-LTF subcarrier: full frequency resolution, 254 B frame,
362 Hz wire limit; measured 321 Hz delivered at 330. Boot default stays 166 @ 243
until 114's derate point is swept properly.

**Ping send failures** are a counter in STATS now, not an ESP_LOGW from the esp_timer
task (which blocked ~0.76 ms per failure, jittering the surviving pings).

Deferred from the audit: unicast-with-retries (would repair most collision loss;
needs the ping seq carried in the frame first, and the current `payload+15` read is
mid-MAC-header garbage — probe the real ESP-NOW body offset empirically before
trusting any seq), and a higher console baud (2-3 Mbaud would raise every wire
ceiling ~2-3×, gated on what the CH343 bridge sustains).

## 2026-08-18 (later still) — forcing the AGC makes amplitude jitter WORSE, not better

The ~12% per-packet common-mode amplitude wobble is not the receiver's AGC failing
to be compensated — pinning the gain proves it. `AGC LOCK` / `AGC FREE` commands
were added (the gain-ctrl component's `set_rx_force_gain`, reversible with (0,0)),
and measured twice on the pinned A→D link with the test order reversed between
runs: locked showed 19.9% and 34.2% common-mode against auto's 5.5% and 16.4% in
the same minutes — locked is 2–3× worse in both orders, zero saturated samples
either way.

Reading: the AGC is *tracking* a real per-packet level variation (TX-side power
jitter being the prime suspect — no RX setting can remove that) and the Q8.8
compensation restores it accurately; a pinned gain forfeits the tracking and adds
quantisation noise on packets that land low in the ADC. The esp-csi guidance to
force gain for amplitude stability fails on this rig. Consequence: keep AGC auto;
treat the common-mode as the informationless nuisance it is — per-packet level
normalisation in preprocessing (and the viewer's Level-lock for display). The
commands stay in the firmware for re-testing elsewhere; the GUI button was removed
so the everyday path cannot degrade a recording by accident. The host now counts
saturated int8 samples per packet (v2/v3 parse) as the clipping telltale either way.

## 2026-08-24 — round-robin tuned: the handoff is cheap and the ceiling was imaginary

Two-board sweep on the audited firmware (SUB 30, host-driven rotation):

* **Dwell**: 100 ms delivers 98.4% but touches only 67% of camera frames per link
  (bursts miss frames); 25 ms and below touch 100%. The frame-locked dwell
  (cycle = one 33 ms camera frame) costs ~3% handoff tax; even 8 ms costs only ~8%.
* **Rate**: round-robin sustains RATE 1600 at ~93% delivery flat from 1000 up —
  each receiver's UART carries only its share, so the pinned-mode ceiling (1071 at
  SUB 30) does not apply. Two boards at 1600 = ~745 Hz per link, both links,
  100% frame coverage.
* **Four-board projection** (unverified until B/C are flashed and attached): each
  RX carries 3/4 of the rate → R≈1300 fits the wire → ~325 Hz × 12 links, ~10
  packets per camera frame per link at 8.3 ms dwell — ~13× the per-link density of
  the 2026-08-12 campaign, with phase.

The viewer's rate clamp is now mode-aware (round-robin ceiling = pinned × n/(n-1),
firmware cap 2000).

## 2026-08-24 (later) — the "faulty" fifth board returned, and passed

Original board A (14:c1:9f:c1:2d:3c / 5C37261255) left the rig; the board swapped
into its place is the fifth board this log records as returned-faulty
(ec:da:3b:4c:b8:d0 / 5C39018759, broken receive path, 49.5 vs 4.5 pkt/s). Re-tested
on the audited firmware before trusting it: the decisive same-pair both-directions
test showed only 1.6× asymmetry immediately after flashing (a boot-race transient —
its RX command likely landed before the command task was up), and the steady-state
test — D transmitting to all three receivers simultaneously — delivered **98.4% to
this board with the strongest RSSI of the three** (−20 dBm; B −21, C −35). The
documented fault did not reproduce. It now carries the label A (`LABEL` in
capture.py maps `b8:d0` → A). Treat the old fault as possibly intermittent: if A's
links ever show asymmetric loss, this board is the first suspect.

### Side-by-side matrix retest, 2026-08-24

All four boards adjacent (matched path loss), full 12-link matrix at 243 Hz:
every link 98.4–100%, every same-pair asymmetry 1.0× (the fault this board was
returned for measured 11×), and as a receiver the returned board scored **100.0%
— the best of the four**. The documented fault is gone: repaired on return, a
reseated antenna contact, or intermittent. The suspicion note above stands, but
as of today this is the healthiest RX in the rig.

### A/B comparison of the two "A-slot" boards, 2026-08-24

Same slot, same side-by-side layout, same 12-link matrix, minutes apart: the
returned board (now labelled E) scored 98.7% TX / 100.0% RX; the original A
scored 99.6% TX / 99.2% RX. All same-pair asymmetries 1.0x for both. The two are
statistically indistinguishable — the rig has a verified healthy spare for the
first time. E keeps its own label so the NOTES suspicion stays attached to the
silicon rather than the slot.
