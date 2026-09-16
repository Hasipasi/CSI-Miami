#!/usr/bin/env python3
"""Measure the round-robin token schedule on the real rig. No camera needed.

For each point (pings per turn, firmware ping rate) it arms the ring, lets it turn
for --seconds, and reports what the schedule actually delivered:

  cycle     median time for the token to go round every board (ms)
  cover     share of synthetic 30 fps frames whose window (half a frame period
            either side, so windows are disjoint) holds a packet on EVERY directed
            link -- the figure the count-based schedule exists to raise
  win       packets per link per window: the minimum over all links and windows
            (the guarantee), then the 10th percentile and the median
  per-link  mean packets per second per directed link
  deliv     packets received / (pings commanded x receivers): radio + wire loss
  p99 gap   worst per-link 99th-percentile inter-packet gap (ms)
  t/o       turns that never reported TX_DONE

  python3 ring_sweep.py                              # default sweep, 2.4 GHz
  python3 ring_sweep.py --points 2:400,4:1200,0:400  # burst:rate; 0 = timed dwell
  python3 ring_sweep.py --band 5.6 --seconds 8

Stop the viewer first: only one process can own the serial ports.
"""

import argparse
import threading
import time

import numpy as np
import serial

from capture import CsiStream, LABEL, TokenRing, discover, frame_coverage, frame_window_counts


class Rig:
    def __init__(self, boards):
        self.boards = boards
        self.macs = sorted(boards)
        self.stop = threading.Event()
        self.lock = threading.Lock()            # serial writes
        self.pk_lock = threading.Lock()
        self.packets = {}                       # (tx, rx) -> [(host time, board us)]
        self.collect = False
        self.ring = None
        self.bad = 0
        self.warnings = []                      # (rx, line) the firmware complained
        self.board_stats = {}                   # rx -> {framedrops, textdrops, sendfail}
        self.reboots = {m: 0 for m in self.macs}   # boot banners seen per board
        self.acks = []                          # (rx, line) BAND_OK / BW_OK replies
        for m in self.macs:
            boards[m].reset_input_buffer()
        for m in self.macs:
            threading.Thread(target=self.reader, args=(m,), daemon=True).start()

    def reader(self, rx):
        ser = self.boards[rx]
        st = CsiStream()
        while not self.stop.is_set():
            try:
                data = ser.read(ser.in_waiting or 1)
            except (serial.SerialException, OSError):
                return
            if not data:
                continue
            try:
                recs, lines = st.feed(data)
            except Exception:
                self.bad += 1
                continue
            for ln in lines:
                t = ln.decode(errors='ignore').strip()
                ring = self.ring
                if ring is not None:
                    ring.on_line(rx, t)
                if 'bad TX' in t or 'bad RX' in t:
                    with self.pk_lock:
                        self.warnings.append((rx, t))
                elif 'Board MAC' in t:
                    # the boot banner: this board just (re)started
                    with self.pk_lock:
                        self.reboots[rx] += 1
                elif t.startswith(('BAND_OK', 'BW_OK', 'CHAN_OK')):
                    with self.pk_lock:
                        self.acks.append((rx, t))
                elif t.startswith('STATS,'):
                    d = {}
                    for part in t.split(',')[1:]:
                        k, _, v = part.partition('=')
                        try:
                            d[k] = int(v)
                        except ValueError:
                            pass
                    with self.pk_lock:
                        self.board_stats[rx] = d
            if recs and self.ring is not None:
                self.ring.frame_bytes = 26 + 2 * len(recs[-1][3])
            if self.collect and recs:
                now = time.time()
                with self.pk_lock:
                    for rec in recs:
                        self.packets.setdefault((rec[0], rx), []).append((now, rec[1]))

    def write_all(self, line):
        with self.lock:
            for m in self.macs:
                self.boards[m].write(line)

    def park(self):
        self.write_all(b'RX 000000000000\n')

    def stats(self):
        """The boards' own drop counters: the only place wire loss (framedrops) and
        radio-side send failures are distinguishable from air loss."""
        self.write_all(b'STATS\n')
        time.sleep(0.3)
        with self.pk_lock:
            return {m: dict(d) for m, d in self.board_stats.items()}

    def point(self, burst, rate, seconds, dwell, fps, guard=0.0, pipelined=True):
        self.write_all(f'RATE {rate}\n'.encode())
        time.sleep(0.1)
        stop_point = threading.Event()
        ring = TokenRing(self.boards, self.macs, stop_point, burst=burst, dwell=dwell,
                         rate_hz=rate, lock=self.lock)
        ring.guard = guard
        ring.pipelined = pipelined
        self.ring = ring

        def radio():
            while not stop_point.is_set():
                if not ring.turn():
                    return
        th = threading.Thread(target=radio, daemon=True)
        th.start()
        time.sleep(0.5)                          # let the ring settle
        st0 = self.stats()
        with self.pk_lock:
            self.packets = {}
            self.warnings = []
        n_before = len(ring.completed)
        with self.pk_lock:
            reboots0 = dict(self.reboots)
        t_start = time.time()
        self.collect = True
        time.sleep(seconds)
        self.collect = False
        t_end = time.time()
        stop_point.set()
        th.join(2)
        self.ring = None
        st1 = self.stats()
        self.park()
        time.sleep(0.2)

        with self.pk_lock:
            packets = {k: np.array(v) for k, v in self.packets.items()}
            warnings = list(self.warnings)
        want = [(tx, rx) for tx in self.macs for rx in self.macs if tx != rx]
        dur = t_end - t_start
        # Windows are judged only where the whole window lies inside the collection
        # span, so an edge window cannot look empty for lack of data.
        frames = np.arange(t_start + 0.5 / fps, t_end - 0.5 / fps, 1.0 / fps)
        host_t = {k: v[:, 0] for k, v in packets.items()}
        cover, per_link_cover = frame_coverage(frames, host_t, want)
        win = np.array([frame_window_counts(frames, host_t.get(k, ())) for k in want])
        counts = {k: len(packets.get(k, ())) for k in want}
        rates = [c / dur for c in counts.values()]
        gaps = [np.percentile(np.diff(host_t[k]) * 1000, 99)
                for k in want if len(packets.get(k, ())) > 10]
        completed = [c for c in list(ring.completed)[n_before:]
                     if t_start <= c[0] <= t_end]
        if burst > 0:
            commanded = sum(n for _t, n, _tx in completed)
        else:
            commanded = rate * dur          # continuous pinging, minus handoffs
        expected = commanded * (len(self.macs) - 1)
        deliv = 100.0 * sum(counts.values()) / expected if expected else float('nan')
        cyc = 1000 * float(np.median(list(ring.cycles))) if ring.cycles else float('nan')

        # Where did the missing packets go? Group each link's packets into bursts
        # on the receiving board's own clock (a gap over 1.5 ping periods starts a
        # new burst) and compare with the bursts the ring commanded from that
        # transmitter: whole bursts missing is a different fault from short ones.
        period_us = 1e6 / rate
        sizes = {}
        bursts_seen = {k: 0 for k in want}
        short = {k: 0 for k in want}            # bursts that arrived with < burst packets
        for k in want:
            v = packets.get(k)
            if v is None or not len(v):
                continue
            lts = np.sort(v[:, 1])
            new = np.concatenate([[True], np.diff(lts) > 1.5 * period_us])
            ids = np.cumsum(new)
            bc = np.bincount(ids)[1:]
            for s in bc:
                sizes[int(s)] = sizes.get(int(s), 0) + 1
            bursts_seen[k] = int(ids[-1])
            short[k] = int(np.sum(bc < burst)) if burst > 0 else 0
        by_tx = {}
        for _t, _n, tx in completed:
            by_tx[tx] = by_tx.get(tx, 0) + 1
        expected_bursts = sum(by_tx.get(tx, 0) for tx, _rx in want) if burst > 0 else 0
        per_link_bursts = {k: (by_tx.get(k[0], 0), bursts_seen[k], short[k]) for k in want}
        turns = [(b - a) * 1000 for a, b, _tx, ok in list(ring.turns) if t_start <= a <= t_end]
        timed_out = [(a - t_start, LABEL.get(tx[-5:], tx[-5:]))
                     for a, b, tx, ok in list(ring.turns) if t_start <= a <= t_end and not ok]
        with self.pk_lock:
            reboots = {LABEL.get(m[-5:], m): self.reboots[m] - reboots0[m] for m in self.macs}
        drops = {}
        for m in self.macs:
            a, b = st0.get(m, {}), st1.get(m, {})
            drops[m] = {k: b.get(k, 0) - a.get(k, 0) for k in ('framedrops', 'textdrops', 'sendfail')}
        return dict(burst=burst, rate=rate, cycle_ms=cyc, cover=cover,
                    win_min=int(win.min()) if win.size else 0,
                    win_p10=int(np.percentile(win, 10)) if win.size else 0,
                    win_med=float(np.median(win)) if win.size else 0.0,
                    per_link=float(np.mean(rates)) if rates else 0.0,
                    min_link=float(np.min(rates)) if rates else 0.0,
                    deliv=deliv, p99=max(gaps) if gaps else float('nan'),
                    timeouts=ring.timeouts, links=sum(1 for c in counts.values() if c),
                    warnings=warnings, per_link_cover=per_link_cover, counts=counts,
                    sizes=sizes, bursts_seen=sum(bursts_seen.values()),
                    bursts_expected=expected_bursts, drops=drops,
                    per_link_bursts=per_link_bursts, timed_out=timed_out,
                    reboots={k: v for k, v in reboots.items() if v},
                    turn_p50=float(np.percentile(turns, 50)) if turns else float('nan'),
                    turn_p99=float(np.percentile(turns, 99)) if turns else float('nan'),
                    turn_max=float(np.max(turns)) if turns else float('nan'))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--points', default='2:400,1:400,4:400,2:1200,4:1200,8:1200,0:400',
                    help='comma-separated burst:rate pairs; burst 0 = timed dwell')
    ap.add_argument('--seconds', type=float, default=4.0, help='per point')
    ap.add_argument('--dwell', type=float, default=0.025, help='for burst 0')
    ap.add_argument('--band', default='2.4', choices=('2.4', '5.6'))
    ap.add_argument('--bw', type=int, default=None, choices=(20, 40),
                    help='radio bandwidth to set after the band (the firmware enters '
                         '5.6 GHz at HT20; pass 40 for HT40 there)')
    ap.add_argument('--fps', type=float, default=30.0,
                    help='the synthetic camera clock the coverage figure uses')
    ap.add_argument('--verbose', action='store_true', help='per-link coverage too')
    ap.add_argument('--schedule', choices=('pipelined', 'gated'), default='pipelined')
    ap.add_argument('--guard', type=float, default=0.0,
                    help='ms to wait after each TX_DONE before the next board is '
                         'told to send (TokenRing.guard); 0 = as fast as the wire')
    args = ap.parse_args()

    boards = discover()
    if len(boards) < 2:
        raise SystemExit(f'need >= 2 boards, found {len(boards)}')
    rig = Rig(boards)
    labels = [LABEL.get(m[-5:], m) for m in rig.macs]
    print(f'{len(boards)} boards {labels}, {len(rig.macs) * (len(rig.macs) - 1)} links, '
          f'{args.band} GHz, {args.seconds:g} s per point, coverage against {args.fps:g} fps')
    rig.write_all(f'BAND {args.band}\n'.encode())
    time.sleep(1.0)
    if args.bw is not None:
        rig.write_all(f'BW {args.bw}\n'.encode())
        time.sleep(1.0)
    with rig.pk_lock:
        acks = list(rig.acks)
    print('radio:', ' '.join(f'{LABEL.get(rx[-5:], rx)}:{ln}' for rx, ln in acks) or 'no BAND/BW replies')

    points = []
    for p in args.points.split(','):
        b, r = p.split(':')
        points.append((int(b), int(r)))

    print(f'\n{"burst":>5s} {"rate":>5s} {"guard":>5s} {"cycle ms":>8s} {"cover":>6s} '
          f'{"win min/p10/med":>15s} {"per-link":>8s} {"deliv":>6s} {"t/o":>4s}  '
          f'{"turn p50/p99/max ms":>20s}')
    print(f'schedule {args.schedule}, guard {args.guard:g} ms')
    for burst, rate in points:
        r = rig.point(burst, rate, args.seconds, args.dwell, args.fps,
                      guard=args.guard / 1000.0, pipelined=args.schedule == 'pipelined')
        print(f'{burst:5d} {rate:5d} {args.guard:5.1f} {r["cycle_ms"]:8.1f} {100 * r["cover"]:5.1f}% '
              f'{r["win_min"]:5d} {r["win_p10"]:4d} {r["win_med"]:4.1f} '
              f'{r["per_link"]:6.1f} Hz {r["deliv"]:5.1f}% {r["timeouts"]:4d}  '
              f'{r["turn_p50"]:5.1f} / {r["turn_p99"]:5.1f} / {r["turn_max"]:5.1f}',
              flush=True)
        if r['timeouts'] or r['reboots']:
            to = ', '.join(f'{t:.2f}s {b}' for t, b in r['timed_out'][:12])
            more = f' … (+{len(r["timed_out"]) - 12})' if len(r['timed_out']) > 12 else ''
            print(f'      turns without TX_DONE at: {to}{more}')
            if r['reboots']:
                print(f'      BOARD REBOOTS during the point: {r["reboots"]}')
        if r['warnings']:
            w = r['warnings'][0]
            print(f'      firmware rejected a command ({len(r["warnings"])}x), e.g. '
                  f'{LABEL.get(w[0][-5:], w[0])}: {w[1]} -- old firmware? reflash')
        if args.verbose and burst > 0 and r['bursts_expected']:
            sz = ', '.join(f'{s} pkts: {n}' for s, n in sorted(r['sizes'].items()))
            print(f'      bursts seen {r["bursts_seen"]} of {r["bursts_expected"]} '
                  f'expected across links ({100 * r["bursts_seen"] / r["bursts_expected"]:.1f}%); '
                  f'sizes {{{sz}}}')
        dl = ' '.join(f'{LABEL.get(m[-5:], m)}:{d["framedrops"]}/{d["textdrops"]}/{d["sendfail"]}'
                      for m, d in r['drops'].items())
        if args.verbose or any(v for d in r['drops'].values() for v in d.values()):
            print(f'      board framedrops/textdrops/sendfail during the point: {dl}')
        if args.verbose:
            order = ' '.join(LABEL.get(m[-5:], m) for m in rig.macs)
            print(f'      ring order {order}; per link: frames covered, packets, '
                  f'bursts seen/commanded, bursts short')
            for (tx, rx), c in sorted(r['per_link_cover'].items()):
                nm = f'{LABEL.get(tx[-5:], tx[-5:])}->{LABEL.get(rx[-5:], rx[-5:])}'
                exp, seen, sh = r['per_link_bursts'][(tx, rx)]
                print(f'      {nm} {100 * c:5.1f}%  {r["counts"][(tx, rx)]:5d} pkts  '
                      f'{seen:4d}/{exp:4d} bursts  {sh:4d} short')
    if rig.bad:
        print(f'\n{rig.bad} unreadable frames')
    rig.stop.set()
    time.sleep(0.4)
    for m in rig.macs:
        try:
            boards[m].close()
        except (serial.SerialException, OSError):
            pass


if __name__ == '__main__':
    main()
