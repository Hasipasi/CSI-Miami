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

## Pose-estimation assessment — INCOMPLETE

First pass measured, but with a setup that was **not** suited to the question:
boards in a line, stimulus was a hand waved between them.

Conclusion that survives that flaw (it is purely temporal):

- Channel decorrelates below 0.5 in **~39 ms** during motion, implying **>=51 Hz**
  per link. Fixed-TX gives 49.8 Hz (marginal). Round-robin gives ~5 Hz effective
  (bursty, jitter CV 0.87) — roughly 10x too slow. **Round-robin as configured is
  not suitable for pose estimation**; it trades exactly the resource pose needs.

Conclusions that are confounded and need re-testing:

- "Only 2 principal components explain 95% of variance" — a single hand blocking
  line-of-sight is an inherently low-dimensional stimulus, so this probably
  reflects the stimulus, not an information limit of the hardware.
- "3.85x motion-vs-static variation" — blocking LoS directly is the strongest
  possible perturbation, so this is an optimistic ceiling, not typical.

### Next step

Re-run with geometry appropriate to the task: **four boards at the corners of a
~2-3 m square**, upright, at torso height; subject in the middle doing full-body
motion. That gives crossing paths through the subject from several directions
instead of one axis. Then repeat static + moving captures in fixed-TX mode and
compare against the numbers above.

Also worth trying:
- Raise fixed-TX ping rate toward ~140 Hz (headroom exists; currently 50 Hz).
- Restore phase, which was stripped for bandwidth — likely matters more for
  effective dimensionality than raw sample rate does.

## Localization idea — not started

Plan discussed: place boards on a known grid, permute which board sits at which
point, and fit position from the link data. Note RSSI multilateration was tried
on the 5-board data and **failed** — the distance matrix came out non-Euclidean
(negative eigenvalues), because orientation and multipath dominate RSSI at this
scale. CSI fingerprinting against surveyed positions is the more promising route.
