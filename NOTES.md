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
- **UART is the bottleneck, not the radio.** At 921600 baud the link carries
  ~90 KB/s. Amplitude-only lines are ~627 bytes, so ~140 lines/s is the ceiling.
  Pushing the ping rate to 100 Hz saturated it and *reduced* throughput via
  truncated lines.
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

## Room geometry (home setup, 4 boards)

Measured by hand, reconstructed by `measure_geometry.py`. RMS residual 1.6 mm,
out-of-plane component 0.0% -- a flat layout at equal height.

| board | MAC | x (m) | y (m) |
|---|---|---|---|
| A | `14:c1:9f:c1:2d:3c` | 0.00 | 0.00 |
| B | `14:c1:9f:c1:2d:a8` | 4.70 | 0.00 |
| C | `30:30:f9:1d:ab:d4` | 1.99 | 1.16 |
| D | `dc:da:0c:77:6b:5c` | 1.31 | -2.87 |

Layout is unique only up to reflection; C is placed on +y by convention. Hull
area ~9.5 m², centroid ~(2.0, -0.4), and both the A-B and C-D links pass within
~0.45 m of it, so that is where a subject should stand.

Caveat: 4 points in a plane have 5 degrees of freedom and 6 distances, so there
is exactly one redundant check. It passing rules out a gross single-measurement
error but not a systematic bias (e.g. always measuring to the USB end).

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

### Next steps

- **Static pose discrimination** is the right next experiment: capture several
  distinct held poses and measure separability. Since a motionless person is
  detectable at z~27, this tests pose-reading capacity directly instead of
  inferring it from one motion sequence.
- **Restore phase** (stripped for UART bandwidth). Likely raises effective rank
  more than any sampling change.
- More boards would add links, but note 12 links only bought rank 9 -- returns
  are sublinear because links and subcarriers are highly redundant.

## Localization idea -- not started

Grid + permutation plan still open. Note the RSSI finding above: use CSI
fingerprinting, not RSSI multilateration.
