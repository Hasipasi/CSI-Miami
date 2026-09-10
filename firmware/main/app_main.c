/*
 * SPDX-FileCopyrightText: 2025-2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */
/* CSI Round-Robin Example

   Every board runs this same firmware and is purely reactive: the PC (which
   already holds a UART connection open to every board for CSI capture) decides
   the round-robin schedule and tells each board what to do over that same UART
   link, rather than boards self-coordinating over ESP-NOW broadcast. That's a
   deliberate change from an earlier version of this example, which had boards
   hand the TX token to each other via broadcast packets -- testing showed the
   receiving board's WiFi stack was dropping the vast majority of incoming
   ESP-NOW frames while it was also doing per-packet CSI processing at any
   real ping rate, and unlike UART, ESP-NOW broadcasts aren't acknowledged, so
   a dropped handoff could stall the whole ring. Reliable wired UART sidesteps
   that entirely.

   Two commands, one line each over the console UART:
     TX                 -- this board becomes the token holder: broadcasts
                            CSI-trigger pings and lights its LED blue.
     RX <mac_hex>        -- this board becomes/stays a receiver, filtering CSI
                            for the given MAC (the current token holder), LED red.
   <mac_hex> is 12 hex chars, no separators, e.g. ecda3b4cb8d0.

   Also:
     RATE <hz>          -- retune the ping rate live.
     SUB <n>            -- 0 (all 192), 30, 114 (every HT-LTF subcarrier: full
                            frequency resolution without the 5x-weaker legacy LLTF
                            duplicates) or 166 subcarriers per frame.
     AGC LOCK / FREE    -- pin the receive gain to the latched baseline (kills the
                            ~12% per-packet amplitude wobble; risks clipping if the
                            scene gets much louder) / return to automatic gain.
                            Retunes auto-FREE. Needs 100 frames of baseline first.
     STATS              -- STATS,framedrops,textdrops,sendfail,heap. framedrops is
                            whole records dropped because the UART TX ring was full,
                            i.e. wire loss, distinguishable from radio loss.
     BAND <2.4|5.6>     -- move every ESP32-C5 together between the 2.4 GHz boot
                            channel and 5.6 GHz (channel 120). Older ESP32-S3 boards
                            reject 5.6 because their radio is 2.4 GHz-only.
     CHAN <n>           -- move to a channel in the current band. Every board must
                            be moved together, and at HT40 the primary/secondary
                            pair is what is occupied.
     BW <20|40>         -- radio bandwidth in MHz. 40 gives more subcarriers, 20
                            gives a span narrow enough to dodge a busy band. Move
                            every board together.
     SCAN [ms]          -- survey channels 1-13 for `ms` each (default 250) and
                            report SCAN_CH,<ch>,<pkts>,<bytes>,<rssi_mean>,<rssi_max>
                            per channel, then SCAN_DONE,<channel restored>.
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "nvs_flash.h"

#include "esp_mac.h"
#include "rom/ets_sys.h"
#include <stdarg.h>
#include "driver/uart.h"
#include "driver/uart_vfs.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_now.h"
#include "esp_csi_gain_ctrl.h"
#include "esp_timer.h"
#include "led_strip.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#if CONFIG_IDF_TARGET_ESP32C5
#define STATUS_LED_GPIO 27
#else
#define STATUS_LED_GPIO 48
#endif

// 13, not the 11 this shipped with: 11 was inherited from Espressif's example and
// never checked. Surveyed 2026-08-18 (SCAN, both boards pooled, three runs): ch13's
// HT40 span carried 11-18% less contending airtime than ch11's, and the ranking was
// identical every run. At HT40 no placement clears the band entirely -- a -30 dBm AP
// overlaps every span -- so this is the least-bad 40 MHz block, not a quiet one.
// HT20 on a genuinely quiet channel measured 0.3% loss against 5-15% here; that
// remains a runtime experiment via "BW 20" + "CHAN <n>" until the dataset builders
// handle 128-wide frames.
#define CONFIG_LESS_INTERFERENCE_CHANNEL   13
#define CONFIG_5G6_CHANNEL                120
#define CONFIG_WIFI_BANDWIDTH               WIFI_BW_HT40
#if CONFIG_IDF_TARGET_ESP32C5
#define CONFIG_WIFI_BAND_MODE               WIFI_BAND_MODE_2G_ONLY
// esp_wifi_set_protocols() treats 11N as a requested maximum and expands it to
// b/g/n, but the single-band esp_wifi_set_protocol() used during a BAND transition
// requires the complete valid 2.4 GHz chain. N-only is rejected with INVALID_ARG.
#define CONFIG_WIFI_2G_PROTOCOL             (WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G \
                                             | WIFI_PROTOCOL_11N)
#define CONFIG_WIFI_5G_PROTOCOL             (WIFI_PROTOCOL_11A | WIFI_PROTOCOL_11N)
#endif
#define CONFIG_ESP_NOW_PHYMODE              WIFI_PHY_MODE_HT40
#define CONFIG_ESP_NOW_RATE                 WIFI_PHY_RATE_MCS0_LGI
#define CONFIG_FORCE_GAIN                   0
// CSI-trigger pings/sec while holding the token. UART bandwidth is the limit, and the
// binary frame is what moved it: 82 B for 30 complex subcarriers, against 626 B for
// the CSV this replaced.
//
// The number that matters is burst load -- a receiver sees the full ping rate while
// its transmitter holds the token. 243 Hz x 354 B = 86.0 KB/s of the 92.2 KB/s link.
//
// 243 is the measured knee minus 10%. The knee differs by subcarrier set, and for two
// different reasons, which is worth knowing before changing either:
//
//   SUB 30   knee ~750 Hz at only ~65% UART. NOT bandwidth -- it is per-frame cost
//            (mutex, per-byte ROM writes, ESP-NOW send rate). Small frames hit this
//            long before they fill the wire. Pair with 675 Hz and a 12.5 ms dwell.
//   SUB 166  knee ~270 Hz at ~97% UART. This one really is bandwidth: 354 B a frame
//            is 3.84 ms of wire, so the link is ~93% busy at 243 Hz. Currently in use.
//
// Changing CONFIG_SUB_COUNT means changing this too, and the dwell with it -- they are
// not independent knobs. See the table in NOTES.md before touching any of the three.
//
// Note this is the *ping* rate, not the per-link rate. In round-robin each board holds
// the token a quarter of the time, so a link averages a quarter of this.
//
// And "averages" is doing real work in that sentence. A link is sampled in a dense
// burst for its transmitter's whole dwell and then not at all for the rest of the
// cycle -- measured at 50 ms dwell: 2.25 ms median spacing inside the burst, but a
// 165 ms p99 gap, 16% duty. Raising this number makes bursts denser; it does not
// shrink the gap. Shorten the host's --round-duration for that.
//
// The old warning here said not to fill the arithmetic headroom, because 100 Hz of
// CSV (62.6 KB/s, "68%, comfortable") corrupted within minutes. That was measured
// again against this encoding and the limit is not where it looked: swept 400-1000 Hz,
// corruption was zero at every rate, up to 84% of the link, with boards still
// accepting commands. The culprit was never the byte count -- ets_printf *formatted*
// each line while holding the TX FIFO, where this path writes prepared bytes. 400 is
// chosen for headroom, not because it is the ceiling.
//
// Still: this is a power-on default, and "RATE <hz>" retunes it at runtime, so the
// ceiling can be re-measured on real hardware instead of argued about -- which is how
// the number above was corrected in the first place. RATE does not persist: the host
// resets every board when it discovers them, so a recording always runs at whatever
// is compiled in here. Sweep with RATE, then set this and reflash.
#define CONFIG_SEND_FREQUENCY              243
// Send raw I/Q instead of computed amplitude (frame version 2 rather than 1).
//
// Amplitude throws away phase, and phase is where path-length change lives -- a
// target moving a fraction of a wavelength shifts phase long before it shows in
// magnitude. The cost is two bytes per subcarrier instead of one, which is why this
// is paired with the 30-subcarrier subset below: 30 complex subcarriers is a 82 B
// frame against 212 B for 192 amplitudes, so it buys rate *and* phase at once.
//
// I/Q is sent raw, uncompensated, with the AGC compensation factor carried in the
// header for the host to apply. Scaling on the board would have to round back into
// int8 and would throw away the low bits the phase estimate rests on.
#define CONFIG_IQ_MODE                        1

#if CONFIG_IQ_MODE
// Version 3 header (24 B): adds the raw (agc_gain, fft_gain) pair beside the Q8.8
// compensation factor, and reserves 0 in the Q8.8 field as an explicit "AGC not yet
// calibrated" sentinel. Version 2 shipped gain=1.0x for the first ~100 frames of
// every boot -- the component returns INVALID_STATE until its baseline latches, the
// return value was ignored, and the host scaled uncalibrated AGC as if it were truth.
//
//   16  2  gain x256 Q8.8, uint16 LE; 0 = not calibrated, do not scale
//   18  1  raw AGC gain (uint8)   -- host can rebuild any compensation against
//   19  1  raw FFT gain (int8)       any reference, and detect gain steps
//   20  1  first_word_invalid
//   21  3  reserved (0)
#define CSI_FRAME_VERSION                     3
#define CSI_FRAME_HDR                        24
#define CSI_BYTES_PER_SUB                     2
#else
#define CSI_FRAME_VERSION                     1
#define CSI_FRAME_HDR                        18
#define CSI_BYTES_PER_SUB                     1
#endif
// Which subcarriers to transmit. The ESP32 reports 192 for HT40: 0-63 is the legacy
// LLTF (measured 5x weaker, mean 8.3 vs ~39), and 64-191 is the real 40 MHz estimate
// as two 20 MHz halves with the DC/guard gap at 123-133. These 30 are spaced evenly
// across that usable span, skipping every subcarrier measured dead over 480 real
// link-captures, so they cover the full band rather than half of it. Measured mean
// amplitude at these indices is 29-67 -- no weak picks.
//
// Subsetting here bakes the choice into the recordings irreversibly, which is why the
// amplitude campaign sent all 192 and let analysis decide. It is the right trade for
// I/Q, where every subcarrier costs two bytes: 30 evenly-spaced ones reconstruct all
// 166 live at R^2 = 0.973, and the bytes saved buy rate.
//
// Three sets, selectable at runtime with "SUB <n>" so the trade can be measured on
// hardware rather than argued:
//   0    all 192, whatever the radio reports, dead subcarriers included
//   30   evenly spaced across the usable span -- 118 Hz per link, 71 ms blind gap
//   166  every live subcarrier, 192 minus the 26 structurally dead. Currently in use:
//        full frequency resolution, at 48 Hz per link and a 101 ms blind gap.
//
// 166 is the deliberate choice here even though 30 measures better on rate and gap:
// R^2 = 0.973 is a reconstruction of the *recorded amplitude* band, and nothing has
// yet checked that it holds for phase. Sending everything keeps that question open
// rather than answering it by assumption -- subsetting is reversible in analysis,
// discarding at the boards is not.
#define CONFIG_SUB_COUNT                    166
// HT-LTF only: SUB_INDEX_166 minus the 52 legacy-LLTF entries (original 6-58). The
// LLTF duplicates spectrum the HT-LTF already covers at ~5x less gain -- int8 phase
// quantisation noise ~2.9 deg/SC there against ~0.6 deg in the HT-LTF -- so on a
// bandwidth-bound wire those 104 bytes bought near-noise. 114 is "every subcarrier
// worth sending": full frequency resolution at a ~360 Hz wire limit instead of 260.
static const uint8_t SUB_INDEX_114[114] = {
    66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
    96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 134, 135, 136,
    137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151,
    152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166,
    167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
    182, 183, 184, 185, 186, 187, 188, 189, 190,
};
static const uint8_t SUB_INDEX_30[30] = {
    66, 70, 74, 78, 82, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121,
    135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 174, 178, 182, 186, 190,
};
// The 26 excluded are 0-5, 32, 59-65, 123-133 and 191: guard bands and DC. Measured
// over 144 link-captures of the recorded campaign, identical in every one -- this is
// the hardware's band plan, not a property of any particular session.
static const uint8_t SUB_INDEX_166[166] = {
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
    52, 53, 54, 55, 56, 57, 58, 66, 67, 68, 69, 70, 71, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
    89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
    104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
    119, 120, 121, 122, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
    145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
    160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174,
    175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189,
    190,
};
// Live selection, settable with "SUB <n>". Same discipline as ping rate: this does not
// persist across the reset the host issues on discovery, so a recording runs at
// CONFIG_SUB_COUNT. Sweep with the command, then set the define and reflash.
static volatile int sub_count = CONFIG_SUB_COUNT;

static const uint8_t *sub_table(int n)
{
    if (n == 30) {
        return SUB_INDEX_30;
    }
    if (n == 114) {
        return SUB_INDEX_114;
    }
    if (n == 166) {
        return SUB_INDEX_166;
    }
    return NULL;                      // 0 or anything unrecognised: send everything
}

#define CONFIG_GAIN_CONTROL                  1     // supported by ESP32-S3 and ESP32-C5

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
#define ESP_IF_WIFI_STA ESP_MAC_WIFI_STA
#endif

static const char *TAG = "csi_rr";

static const uint8_t BROADCAST_MAC[6] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};

static volatile bool is_tx = false;
// When set (via the IDENT command), the LED is pinned to a fixed identify color and
// role changes stop touching it -- lets a specific board be picked out physically on
// the bench without pulling it out of the ring or reflashing it.
static volatile bool ident_mode = false;
// All output -- binary frames, role markers, acks -- goes through the UART driver's
// interrupt-driven TX ring via uart_write_bytes, which is atomic per call: order on
// the wire is enqueue order, the property the old print_mux existed to provide. The
// old path was worse than unserialised: every byte was shifted into the FIFO from
// the emitting task itself, so the WiFi task busy-waited ~3.8 ms per 166-SC frame
// (the measured rate knee), and ESP_LOG bypassed the mutex entirely, interleaving
// mid-frame from the other core. Now ESP_LOG rides the same driver (see app_main).
//
// A full ring means the wire is saturated: drop whole records and count, never
// block the radio. STATS reports the counters.
static volatile uint32_t frame_drops = 0;
static volatile uint32_t text_drops = 0;
static volatile uint32_t send_fails = 0;

static void emit(const void *buf, size_t len)
{
    size_t room = 0;
    if (uart_get_tx_buffer_free_size(UART_NUM_0, &room) != ESP_OK || room < len) {
        frame_drops++;
        return;
    }
    uart_write_bytes(UART_NUM_0, buf, len);
}

static void emit_textf(const char *fmt, ...)
{
    char line[160];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    if (n <= 0) {
        return;
    }
    if (n >= (int)sizeof(line)) {
        n = (int)sizeof(line) - 1;
    }
    size_t room = 0;
    if (uart_get_tx_buffer_free_size(UART_NUM_0, &room) != ESP_OK || room < (size_t)n) {
        text_drops++;
        return;
    }
    uart_write_bytes(UART_NUM_0, line, n);
}

// Frames received since boot or the last retune; also gates the AGC baseline
// collection (first 100 frames), so a retune restarts calibration -- the old code
// kept compensating against a stale boot-time RF reference after CHAN/BW moved.
static volatile int csi_count = 0;
// The latched AGC baseline, file-scope so "AGC LOCK" can pin the radio to it from
// the command task. Locking stops the per-packet gain re-selection that puts a
// measured ~12% common-mode wobble on every amplitude; the cost is dynamic range
// (a much louder signal than the baseline scene clips at int8), so it is a command,
// not the default, and the host counts saturated samples to catch it.
static uint8_t agc_base = 0;
static int8_t fft_base = 0;
static volatile bool base_ready = false;
static volatile bool agc_locked = false;
static uint8_t tx_filter_mac[6] = {0}; // whichever MAC we should currently accept CSI from
static led_strip_handle_t led_strip;
static esp_timer_handle_t ping_timer;
static void ping_timer_cb(void *arg);   // become_tx() fires one ping directly
// Ping period, settable at runtime with "RATE <hz>". The UART ceiling is an empirical
// number -- bandwidth arithmetic has already been shown to be necessary but not
// sufficient here -- so it has to be swept against real corruption counts, and
// reflashing every board per data point made that too slow to bother doing.
static volatile uint32_t ping_period_us = 1000000 / CONFIG_SEND_FREQUENCY;

static void set_led(uint8_t r, uint8_t g, uint8_t b)
{
    led_strip_set_pixel(led_strip, 0, r, g, b);
    led_strip_refresh(led_strip);
}

static void init_led(void)
{
    led_strip_config_t strip_config = {
        .strip_gpio_num = STATUS_LED_GPIO,
        .max_leds = 1,
        .led_pixel_format = LED_PIXEL_FORMAT_GRB,
        .led_model = LED_MODEL_WS2812,
    };
    led_strip_rmt_config_t rmt_config = {
        .clk_src = RMT_CLK_SRC_DEFAULT,
        .resolution_hz = 10 * 1000 * 1000,
    };
    ESP_ERROR_CHECK(led_strip_new_rmt_device(&strip_config, &rmt_config, &led_strip));
}

static void become_tx(void)
{
    is_tx = true;
    // Invalidate the CSI filter so stray/late packets from whoever we were
    // previously told to receive from (which may not have processed its own
    // "become RX" command yet) don't get recorded as if captured during our
    // own TX span. No real board has an all-zero MAC, so nothing can match.
    memset(tx_filter_mac, 0, sizeof(tx_filter_mac));
    if (!ident_mode) {
        set_led(0, 0, 40); // blue = holding the TX token
    }
    esp_timer_stop(ping_timer); // ignored if not running
    ESP_ERROR_CHECK(esp_timer_start_periodic(ping_timer, ping_period_us));
    // Fire once immediately: a periodic timer's first callback lands one full period
    // after the start, so restarting it on every handoff left the first interval of
    // every dwell silent. At a 50 ms dwell that was 40% of the turn, and it is why
    // each turn yielded ~1.3 of the 2.5 packets it should.
    ping_timer_cb(NULL);
    // Same driver ring as the CSI frames, so a role marker orders exactly with the
    // records around it -- the host decides which dwell a packet belongs to from this.
    emit_textf("ROLE_TX\n");
}

static void become_rx(const uint8_t *peer_mac)
{
    is_tx = false;
    memcpy(tx_filter_mac, peer_mac, 6);
    if (!ident_mode) {
        set_led(40, 0, 0); // red = receiving
    }
    esp_timer_stop(ping_timer); // ignored if not running
    emit_textf("ROLE_RX," MACSTR "\n", MAC2STR(tx_filter_mac));
}

static void ping_timer_cb(void *arg)
{
    static uint32_t seq = 0;
    esp_err_t ret = esp_now_send(BROADCAST_MAC, (const uint8_t *)&seq, sizeof(seq));
    seq++;
    if (ret != ESP_OK) {
        // A counter, not a log: this fires from the esp_timer task at the ping rate,
        // and formatting a log line there blocked ~0.76 ms per failure -- jittering
        // the very pings that were already failing. STATS reports it.
        send_fails++;
    }
}

static bool parse_hex_mac(const char *hex, uint8_t *out)
{
    if (strlen(hex) < 12) {
        return false;
    }
    for (int i = 0; i < 6; i++) {
        unsigned byte;
        if (sscanf(hex + i * 2, "%2x", &byte) != 1) {
            return false;
        }
        out[i] = (uint8_t)byte;
    }
    return true;
}

// ------------------------------------------------------------- channel control
// Which channel the rig runs on was, until now, a compile-time constant inherited
// from Espressif's example, and CONFIG_LESS_INTERFERENCE_CHANNEL was a hope rather
// than a measurement. It matters more than the name suggests: at HT40 the secondary
// sits *below* the primary, so "channel 11" actually occupies 7-11 -- straight
// across the two most crowded channels in a typical building.
static volatile uint8_t cur_channel = CONFIG_LESS_INTERFERENCE_CHANNEL;
static volatile uint8_t channel_2g = CONFIG_LESS_INTERFERENCE_CHANNEL;
#if CONFIG_IDF_TARGET_ESP32C5
static volatile uint8_t channel_5g = CONFIG_5G6_CHANNEL;
#endif
// Bandwidth is runtime state, not the compile-time macro it used to be. 40 MHz cannot
// dodge a congested 2.4 GHz band -- 1/6/11 are 25 MHz apart, so every HT40 placement
// overlaps at least one -- while 20 MHz can, at the cost of frequency diversity. That
// is a trade to measure on the rig, and measuring it meant being able to switch
// without a reflash between every data point.
static volatile uint8_t cur_bw = (CONFIG_WIFI_BANDWIDTH == WIFI_BW_HT40) ? 40 : 20;
static volatile uint8_t bandwidth_2g = (CONFIG_WIFI_BANDWIDTH == WIFI_BW_HT40) ? 40 : 20;
#if CONFIG_IDF_TARGET_ESP32C5
// The current C5/ESP-NOW driver produces valid 5 GHz CSI at HT20. Asking it to
// change bands directly into HT40 leaves the peer on its old 2.4 GHz channel, so
// make the verified mode explicit instead of claiming a width the radio did not use.
static volatile uint8_t bandwidth_5g = 20;
#endif

static bool channel_is_2g(uint8_t ch)
{
    return ch >= 1 && ch <= 13;
}

#if CONFIG_IDF_TARGET_ESP32C5
static bool channel_is_5g(uint8_t ch)
{
    return ((ch >= 36 && ch <= 64 && (ch - 36) % 4 == 0)
            || (ch >= 100 && ch <= 144 && (ch - 100) % 4 == 0)
            || (ch >= 149 && ch <= 177 && (ch - 149) % 4 == 0));
}
#endif

static bool apply_channel(uint8_t ch)
{
    bool on_2g = channel_is_2g(ch);
#if CONFIG_IDF_TARGET_ESP32C5
    if (!on_2g && !channel_is_5g(ch)) {
#else
    if (!on_2g) {
#endif
        return false;
    }
    wifi_second_chan_t sec = WIFI_SECOND_CHAN_NONE;
    if (cur_bw == 40 && on_2g) {
        // HT40 needs its second 20 MHz beside the primary and inside 1-13: below for
        // a high primary, above for a low one. The *pair* is what the rig occupies,
        // so it is the pair a survey has to be scored against.
        sec = (ch >= 5) ? WIFI_SECOND_CHAN_BELOW : WIFI_SECOND_CHAN_ABOVE;
    }
    if (esp_wifi_set_channel(ch, sec) != ESP_OK) {
        return false;
    }
    cur_channel = ch;
    if (on_2g) {
        channel_2g = ch;
#if CONFIG_IDF_TARGET_ESP32C5
    } else {
        channel_5g = ch;
#endif
    }
    // The RF reference just moved: a gain baseline recorded on the old channel is
    // stale, so restart the 100-frame collection (no-op at boot when nothing has
    // been recorded yet).
    if (csi_count > 0) {
        // A lock pinned against the old channel's RF is meaningless on the new one,
        // and re-baselining under forced gain would just measure the forced value.
        if (agc_locked) {
            esp_csi_gain_ctrl_set_rx_force_gain(0, 0);
            agc_locked = false;
            emit_textf("AGC_OK,FREE,retune\n");
        }
        esp_csi_gain_ctrl_reset_rx_gain_baseline();
        base_ready = false;
        csi_count = 0;
    }
    // The broadcast peer was registered against a fixed channel. Leaving it stale
    // points esp_now_send at a channel the radio is no longer on, which fails
    // silently -- pings stop and every link goes quiet with nothing logged.
    esp_now_peer_info_t peer = {0};
    if (esp_now_get_peer(BROADCAST_MAC, &peer) == ESP_OK) {
        peer.channel = ch;
        if (esp_now_mod_peer(&peer) != ESP_OK) {
            return false;
        }
    }
    return true;
}

static esp_err_t set_radio_bandwidth(wifi_bandwidth_t bw)
{
#if CONFIG_IDF_TARGET_ESP32C5
    wifi_bandwidths_t bandwidths = {
        .ghz_2g = bw,
        .ghz_5g = bw,
    };
    return esp_wifi_set_bandwidths(ESP_IF_WIFI_STA, &bandwidths);
#else
    return esp_wifi_set_bandwidth(ESP_IF_WIFI_STA, bw);
#endif
}

static bool apply_peer_rate(int mhz)
{
    esp_now_rate_config_t rc = {
        .phymode = (mhz == 40) ? WIFI_PHY_MODE_HT40 : WIFI_PHY_MODE_HT20,
        .rate    = CONFIG_ESP_NOW_RATE,
        .ersu    = false,
        .dcm     = false,
    };
    esp_err_t err = esp_now_set_peer_rate_config(BROADCAST_MAC, &rc);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "peer rate config (%d MHz): %s", mhz, esp_err_to_name(err));
        return false;
    }
    return true;
}

#if CONFIG_IDF_TARGET_ESP32C5
static bool band_step_ok(const char *step, esp_err_t err)
{
    if (err == ESP_OK) {
        return true;
    }
    ESP_LOGW(TAG, "band switch failed at %s: %s (0x%x)", step,
             esp_err_to_name(err), (unsigned)err);
    return false;
}
#endif

static bool apply_band(int band)
{
    if (band == 24) {
#if CONFIG_IDF_TARGET_ESP32C5
        if (!band_step_ok("2g band mode",
                          esp_wifi_set_band_mode(WIFI_BAND_MODE_2G_ONLY))
            || !band_step_ok("2g protocol",
                             esp_wifi_set_protocol(ESP_IF_WIFI_STA,
                                                   CONFIG_WIFI_2G_PROTOCOL))
            || !band_step_ok("2g bandwidth",
                             set_radio_bandwidth(bandwidth_2g == 40
                                                 ? WIFI_BW_HT40 : WIFI_BW_HT20))) {
            return false;
        }
        cur_bw = bandwidth_2g;
#endif
        return apply_channel(channel_2g) && apply_peer_rate(cur_bw);
    }
#if CONFIG_IDF_TARGET_ESP32C5
    if (band == 56) {
        // This path deliberately selects the verified 20 MHz mode (see
        // bandwidth_5g above) before moving the peer to the 5.6 GHz channel.
        if (!band_step_ok("5g band mode",
                          esp_wifi_set_band_mode(WIFI_BAND_MODE_5G_ONLY))
            || !band_step_ok("5g protocol",
                             esp_wifi_set_protocol(ESP_IF_WIFI_STA,
                                                   CONFIG_WIFI_5G_PROTOCOL))
            || !band_step_ok("5g bandwidth", set_radio_bandwidth(WIFI_BW_HT20))) {
            return false;
        }
        cur_bw = bandwidth_5g;
        return apply_channel(channel_5g) && apply_peer_rate(cur_bw);
    }
#endif
    return false;
}

static bool apply_bandwidth(int mhz)
{
    if (mhz != 20 && mhz != 40) {
        return false;
    }
#if CONFIG_IDF_TARGET_ESP32C5
    if (!channel_is_2g(cur_channel) && mhz != 20) {
        return false;
    }
#endif
    wifi_bandwidth_t bw = (mhz == 40) ? WIFI_BW_HT40 : WIFI_BW_HT20;
    if (set_radio_bandwidth(bw) != ESP_OK) {
        return false;
    }
    cur_bw = (uint8_t)mhz;
    // Channel first, rate config second. The order matters going *up*: HT40 phymode
    // needs the secondary channel to exist, and right after set_bandwidth the radio
    // is still on the HT20 channel with no secondary -- the rate config call then
    // fails and the board keeps transmitting HT20 frames while claiming HT40, which
    // is exactly what happened (RX reported 128 subcarriers after BW 40).
    if (!apply_channel(cur_channel)) {
        return false;
    }
    // ESP-NOW carries its own PHY mode. Left at HT20 while the radio is HT40, every
    // transmission stays 20 MHz wide and the CSI silently loses half its span.
    if (!apply_peer_rate(mhz)) {
        return false;
    }
    if (channel_is_2g(cur_channel)) {
        bandwidth_2g = (uint8_t)mhz;
#if CONFIG_IDF_TARGET_ESP32C5
    } else {
        bandwidth_5g = (uint8_t)mhz;
#endif
    }
    return true;
}

// ---- channel survey
// SCAN parks the radio on each channel and counts what the PHY can hear.
// Deliberately a received-frame census rather than an energy/CCA reading: the
// promiscuous callback is the hook the driver actually gives us, and it counts real
// competing traffic rather than a number that needs calibrating. The limit is that
// it cannot see non-802.11 energy -- microwaves, BLE, analogue video senders -- so a
// channel this calls quiet can still be noisy. It is a ranking, not a noise floor.
static volatile bool scan_mode = false;
static volatile uint32_t scan_pkts;
static volatile uint32_t scan_bytes;
static volatile int32_t  scan_rssi_sum;
static volatile int32_t  scan_rssi_max;

static void promisc_rx_cb(void *buf, wifi_promiscuous_pkt_type_t type)
{
    if (!scan_mode) {
        return;
    }
    const wifi_promiscuous_pkt_t *pkt = (const wifi_promiscuous_pkt_t *)buf;
    scan_pkts++;
    scan_bytes += pkt->rx_ctrl.sig_len;
    scan_rssi_sum += pkt->rx_ctrl.rssi;
    if (pkt->rx_ctrl.rssi > scan_rssi_max) {
        scan_rssi_max = pkt->rx_ctrl.rssi;
    }
}

static void do_scan(uint32_t dwell_ms)
{
    // Stop being a transmitter for the duration: pinging on channels the rest of the
    // rig is not listening on is pure interference, and a board that resumed pinging
    // mid-survey would measure itself.
    is_tx = false;
    esp_timer_stop(ping_timer);
    esp_wifi_set_csi(false);
    esp_wifi_set_promiscuous_rx_cb(promisc_rx_cb);
    // HT20 for the survey: a 40 MHz capture smears two channels into one number, and
    // comparing channels is the entire point.
    set_radio_bandwidth(WIFI_BW_HT20);

    emit_textf("SCAN_BEGIN,%u\n", (unsigned)dwell_ms);

    for (int ch = 1; ch <= 13; ch++) {
        if (esp_wifi_set_channel((uint8_t)ch, WIFI_SECOND_CHAN_NONE) != ESP_OK) {
            continue;
        }
        scan_pkts = 0; scan_bytes = 0; scan_rssi_sum = 0; scan_rssi_max = -128;
        scan_mode = true;
        vTaskDelay(pdMS_TO_TICKS(dwell_ms));
        scan_mode = false;
        uint32_t n = scan_pkts;
        int rssi_mean = n ? (int)(scan_rssi_sum / (int32_t)n) : -128;
        emit_textf("SCAN_CH,%d,%u,%u,%d,%d\n", ch, (unsigned)n,
                   (unsigned)scan_bytes, rssi_mean, (int)scan_rssi_max);
    }

    esp_wifi_set_promiscuous_rx_cb(NULL);
    apply_bandwidth(cur_bw);          // also re-applies the channel

    esp_wifi_set_csi(true);
    // Left as a receiver of nobody on purpose: a survey takes seconds, the host's
    // schedule has moved on, and it issues the next TX/RX itself.
    memset(tx_filter_mac, 0, sizeof(tx_filter_mac));
    emit_textf("SCAN_DONE,%u\n", (unsigned)cur_channel);
}

static void uart_command_task(void *arg)
{
    char line[64];
    while (1) {
        if (fgets(line, sizeof(line), stdin) == NULL) {
            // A UART RX overrun latches the stream's error flag, after which fgets
            // returns NULL forever: the board stops accepting role commands, keeps
            // whatever role it had, and goes silent as a receiver while still
            // transmitting. That is exactly the "dead board" seen twice in these
            // sessions, and it needed a replug because nothing ever cleared the flag.
            // Far less likely now that RX has a 2 KB driver ring instead of the
            // bare 128 B hardware FIFO, but kept: an unclearable error latch is
            // still a dead board.
            clearerr(stdin);
            // With the console on the UART driver, fgets blocks until a full line
            // arrives, so this branch only fires on EOF/error and the delay is no
            // longer the command-latency path it was on the driverless console.
            vTaskDelay(pdMS_TO_TICKS(1));
            continue;
        }
        line[strcspn(line, "\r\n")] = '\0';

        if (strcmp(line, "TX") == 0) {
            become_tx();
        } else if (strncmp(line, "RX ", 3) == 0) {
            uint8_t mac[6];
            if (parse_hex_mac(line + 3, mac)) {
                become_rx(mac);
            } else {
                ESP_LOGW(TAG, "bad RX command: '%s'", line);
            }
        } else if (strncmp(line, "LED ", 4) == 0) {
            // Arbitrary colour, so the boards can act as an in-room status indicator
            // during a capture -- the operator is in the room with them and usually
            // cannot see the PC screen. Deliberately silent: this runs while CSI is
            // streaming and a log line per colour change is just noise on the wire.
            unsigned r, g, b;
            if (sscanf(line + 4, "%u,%u,%u", &r, &g, &b) == 3) {
                ident_mode = true;  // pin it: role changes must not overwrite the cue
                set_led(r & 0xff, g & 0xff, b & 0xff);
            } else {
                ESP_LOGW(TAG, "bad LED command: '%s'", line);
            }
        } else if (strncmp(line, "RATE ", 5) == 0) {
            // Retune the ping rate live. Takes effect on the current dwell if this
            // board is transmitting, otherwise on its next turn with the token.
            unsigned hz;
            if (sscanf(line + 5, "%u", &hz) == 1 && hz >= 1 && hz <= 2000) {
                ping_period_us = 1000000 / hz;
                if (is_tx) {
                    esp_timer_stop(ping_timer);
                    esp_timer_start_periodic(ping_timer, ping_period_us);
                }
                emit_textf("RATE_OK,%u\n", hz);
            } else {
                ESP_LOGW(TAG, "bad RATE command: '%s'", line);
            }
        } else if (strncmp(line, "SUB ", 4) == 0) {
            // Switch subcarrier set live: 0 (all), 30, or 166. Takes effect on the
            // next CSI callback, and the frame carries its own n_sub, so the host
            // resizes from the wire rather than being told separately.
            unsigned nsc;
            if (sscanf(line + 4, "%u", &nsc) == 1
                && (nsc == 0 || nsc == 30 || nsc == 114 || nsc == 166)) {
                sub_count = (int)nsc;
                emit_textf("SUB_OK,%u\n", nsc);
            } else {
                ESP_LOGW(TAG, "bad SUB command: '%s'", line);
            }
        } else if (strncmp(line, "BAND ", 5) == 0) {
            // BAND is intentionally one atomic firmware operation. If the host had
            // to send a band change and a channel change separately, even a brief
            // role-command handoff between them could leave part of the rig deaf.
            int band = 0;
            if (strcmp(line + 5, "2.4") == 0 || strcmp(line + 5, "2") == 0) {
                band = 24;
            } else if (strcmp(line + 5, "5.6") == 0 || strcmp(line + 5, "5") == 0) {
                band = 56;
            }
            if (band && apply_band(band)) {
                emit_textf("BAND_OK,%s,%u,%u\n", band == 24 ? "2.4" : "5.6",
                           (unsigned)cur_channel, (unsigned)cur_bw);
            } else {
                ESP_LOGW(TAG, "unsupported BAND command: '%s'", line);
            }
        } else if (strncmp(line, "BW ", 3) == 0) {
            // Every board must move together, exactly like CHAN: a board left at the
            // other bandwidth cannot decode the others at all.
            unsigned mhz;
            if (sscanf(line + 3, "%u", &mhz) == 1 && apply_bandwidth((int)mhz)) {
                emit_textf("BW_OK,%u,%u\n", mhz, (unsigned)cur_channel);
            } else {
                ESP_LOGW(TAG, "bad BW command: '%s'", line);
            }
        } else if (strncmp(line, "CHAN ", 5) == 0) {
            // Every board must move together: a board left on the old channel or
            // band hears nothing and looks like a dead serial link.
            unsigned ch;
            if (sscanf(line + 5, "%u", &ch) == 1 && apply_channel((uint8_t)ch)) {
                emit_textf("CHAN_OK,%u,%u\n", ch, (unsigned)cur_bw);
            } else {
                ESP_LOGW(TAG, "bad CHAN command: '%s'", line);
            }
        } else if (strncmp(line, "SCAN", 4) == 0) {
            // Blocks this task for 13 * dwell. That is deliberate: commands arriving
            // mid-survey would retune the radio underneath it and silently corrupt
            // the numbers, so they wait in the UART buffer instead.
            unsigned ms = 250;
            if (sscanf(line + 4, "%u", &ms) != 1) {
                ms = 250;                 // bare "SCAN"
            }
            if (ms < 20)   { ms = 20; }   // below a beacon interval measures nothing
            if (ms > 2000) { ms = 2000; } // 13 channels, so this is already 26 s
            do_scan(ms);
        } else if (strcmp(line, "AGC LOCK") == 0) {
            if (base_ready) {
                esp_csi_gain_ctrl_set_rx_force_gain(agc_base, fft_base);
                agc_locked = true;
                emit_textf("AGC_OK,LOCK,%u,%d\n", (unsigned)agc_base, (int)fft_base);
            } else {
                // Nothing sensible to pin to yet: the baseline needs 100 received
                // frames after boot or the last retune.
                emit_textf("AGC_OK,WAIT\n");
            }
        } else if (strcmp(line, "AGC FREE") == 0) {
            esp_csi_gain_ctrl_set_rx_force_gain(0, 0);
            agc_locked = false;
            emit_textf("AGC_OK,FREE\n");
        } else if (strcmp(line, "STATS") == 0) {
            emit_textf("STATS,framedrops=%u,textdrops=%u,sendfail=%u,heap=%lu\n",
                       (unsigned)frame_drops, (unsigned)text_drops,
                       (unsigned)send_fails, (unsigned long)esp_get_free_heap_size());
        } else if (strcmp(line, "IDENT") == 0) {
            ident_mode = true;
            set_led(40, 20, 0); // orange -- physically identifies this board on the bench
            ESP_LOGI(TAG, "IDENT on (LED pinned orange)");
        } else if (strcmp(line, "IDENT OFF") == 0) {
            ident_mode = false;
            set_led(is_tx ? 0 : 40, 0, is_tx ? 40 : 0); // back to role color
            ESP_LOGI(TAG, "IDENT off");
        } else if (line[0] != '\0') {
            ESP_LOGW(TAG, "unknown command: '%s'", line);
        }
    }
}

static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf) {
        ESP_LOGW(TAG, "<%s> wifi_csi_cb", esp_err_to_name(ESP_ERR_INVALID_ARG));
        return;
    }

    // Never record while we ourselves hold the TX token. This is deliberately not just
    // "tx_filter_mac is zeroed in become_tx()": promiscuous mode can loop a station's own
    // outgoing frames back through this callback, and if that self-loopback happens to
    // report info->mac as all-zero too, a zeroed filter would start matching it instead
    // of blocking it. Checking is_tx directly has no such edge case.
    if (is_tx) {
        return;
    }

    // Only keep CSI from whoever we were last told currently holds the TX token.
    if (memcmp(info->mac, tx_filter_mac, 6)) {
        return;
    }

    const wifi_pkt_rx_ctrl_t *rx_ctrl = &info->rx_ctrl;
#if CONFIG_IDF_TARGET_ESP32C5
    if (!rx_ctrl->rx_channel_estimate_info_vld) {
        return;
    }
#endif
    float compensate_gain = 1.0f;
    esp_err_t comp_state = ESP_ERR_INVALID_STATE;
    static uint8_t agc_gain = 0;
    static int8_t fft_gain = 0;
#if CONFIG_GAIN_CONTROL
    esp_csi_gain_ctrl_get_rx_gain(rx_ctrl, &agc_gain, &fft_gain);
    if (csi_count < 100) {
        esp_csi_gain_ctrl_record_rx_gain(agc_gain, fft_gain);
    } else if (csi_count == 100) {
        esp_csi_gain_ctrl_get_rx_gain_baseline(&agc_base, &fft_base);
        base_ready = true;
    }
    // The return value matters: INVALID_STATE until the 100-frame baseline latches,
    // during which compensate_gain keeps its 1.0f init. Ignoring it shipped that
    // 1.0x as a real factor for the first ~0.4 s of every session.
    comp_state = esp_csi_gain_ctrl_get_gain_compensation(&compensate_gain, agc_gain, fft_gain);
#endif

    if (!csi_count) {
        ESP_LOGI(TAG, "========== CSI RECV (binary CSI%d frames) ==========", CSI_FRAME_VERSION);
    }

    // Binary frame instead of a CSV line. ASCII decimal spent ~3.5 bytes encoding
    // one byte of information -- 626 B per record where the payload is 192 bytes --
    // and UART bandwidth is the hard limit on ping rate.
    //
    // Two payload encodings, distinguished by the version byte so one host parser
    // reads both and the amplitude recordings stay readable forever:
    //
    //   version 1, amplitude      version 2, raw I/Q
    //   0    2  magic 0xA5 0x5A    0    2  magic 0xA5 0x5A
    //   2    1  version (1)        2    1  version (2)
    //   3    1  n_sub              3    1  n_sub
    //   4    6  transmitter MAC    4    6  transmitter MAC
    //  10    1  rssi        int8  10    1  rssi        int8
    //  11    1  noise_floor int8  11    1  noise_floor int8
    //  12    4  timestamp us LE   12    4  timestamp us LE
    //  16    1  clipped count     16    2  gain x256, uint16 LE
    //  17    1  first_word_inval  18    1  first_word_invalid
    //  18   ns  amplitude uint8   19    1  reserved (0)
    //                             20  2ns  (imag, real) int8 pairs
    //  18+ns 2  sum16             20+2ns 2 sum16
    //
    // sum16 always covers [2, payload_end) little-endian.
    //
    // Version 2 sends I/Q *uncompensated* and passes the AGC compensation factor in
    // the header. Applying it on the board means rounding a scaled value back into
    // int8, which costs exactly the low-order bits the phase estimate is built on;
    // the host has floats and can apply it for free. Clipping cannot occur, so that
    // counter is what the gain field replaces.
    int n_avail = info->len / 2;
    if (n_avail > 255) {
        n_avail = 255;                     // n_sub is one byte
    }
    // Emit a fixed subset when configured, falling back to everything if the radio
    // reported fewer subcarriers than the table indexes into. Read the selection once:
    // a "SUB <n>" arriving mid-frame must not change the width between the length
    // field and the payload loop.
    int want = sub_count;
    const uint8_t *tbl = sub_table(want);
    int n_sub = (tbl && n_avail > tbl[want - 1]) ? want : n_avail;
    uint8_t frame[CSI_FRAME_HDR + CSI_BYTES_PER_SUB * 255 + 2];
    frame[0] = 0xA5;
    frame[1] = 0x5A;
    frame[2] = CSI_FRAME_VERSION;
    frame[3] = (uint8_t)n_sub;
    memcpy(&frame[4], info->mac, 6);
    frame[10] = (uint8_t)(int8_t)rx_ctrl->rssi;
    frame[11] = (uint8_t)(int8_t)rx_ctrl->noise_floor;
    uint32_t ts = (uint32_t)rx_ctrl->timestamp;
    frame[12] = (uint8_t)(ts & 0xff);
    frame[13] = (uint8_t)((ts >> 8) & 0xff);
    frame[14] = (uint8_t)((ts >> 16) & 0xff);
    frame[15] = (uint8_t)((ts >> 24) & 0xff);

#if CONFIG_IQ_MODE
    // Q8.8 fixed point. Saturating rather than wrapping: a wrapped gain would look
    // like a plausible small number and silently rescale that packet alone.
    // 0 is reserved as the "not calibrated" sentinel (see comp_state above), so a
    // valid factor is clamped to [1, 65535]. The raw pair travels beside it: from
    // (agc, fft) the host can rebuild compensation against any reference, compare
    // absolute amplitude across sessions, and segment phase at gain steps.
    int gain_q8 = 0;
    if (comp_state == ESP_OK) {
        gain_q8 = (int)(compensate_gain * 256.0f + 0.5f);
        if (gain_q8 < 1) {
            gain_q8 = 1;
        } else if (gain_q8 > 65535) {
            gain_q8 = 65535;
        }
    }
    frame[16] = (uint8_t)(gain_q8 & 0xff);
    frame[17] = (uint8_t)(gain_q8 >> 8);
    frame[18] = agc_gain;
    frame[19] = (uint8_t)fft_gain;
    frame[20] = (uint8_t)info->first_word_invalid;
    frame[21] = 0;
    frame[22] = 0;
    frame[23] = 0;
    for (int i = 0; i < n_sub; i++) {
        int k = (n_sub == n_avail) ? i : tbl[i];
        frame[CSI_FRAME_HDR + 2 * i]     = (uint8_t)info->buf[k * 2];      // imag
        frame[CSI_FRAME_HDR + 2 * i + 1] = (uint8_t)info->buf[k * 2 + 1];  // real
    }
#else
    frame[17] = (uint8_t)info->first_word_invalid;
    unsigned clipped = 0;
    for (int i = 0; i < n_sub; i++) {
        int k = (n_sub == n_avail) ? i : tbl[i];
        float imag = compensate_gain * (int8_t)info->buf[k * 2];
        float real = compensate_gain * (int8_t)info->buf[k * 2 + 1];
        int amp = (int)(sqrtf(imag * imag + real * real) + 0.5f);
        if (amp > 255) {
            // Counted, not silently swallowed: if this is ever non-zero the uint8
            // encoding is losing dynamic range and the host needs to know.
            amp = 255;
            clipped++;
        }
        frame[CSI_FRAME_HDR + i] = (uint8_t)amp;
    }
    frame[16] = (uint8_t)(clipped > 255 ? 255 : clipped);
#endif

    int n_end = CSI_FRAME_HDR + CSI_BYTES_PER_SUB * n_sub;
    uint16_t sum = 0;
    for (int i = 2; i < n_end; i++) {
        sum = (uint16_t)(sum + frame[i]);
    }
    frame[n_end] = (uint8_t)(sum & 0xff);
    frame[n_end + 1] = (uint8_t)(sum >> 8);

    // One atomic copy into the driver's interrupt-driven TX ring, then return. The
    // old path shifted every byte into the FIFO from THIS task -- the WiFi task --
    // busy-waiting ~3.8 ms per 166-SC frame: that was the measured rate knee, the
    // above-knee loss cliff, and most of the per-frame cost that capped 30 SC at
    // ~750 Hz. A full ring drops the whole frame, counted in STATS, so saturation
    // degrades visibly instead of stalling the radio.
    emit(frame, n_end + 2);
    csi_count++;
}

static void wifi_init(void)
{
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(esp_netif_init());
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));

#if CONFIG_IDF_TARGET_ESP32C5
    // The C5 is not simultaneously dual-band. Configure both sides up front, then
    // BAND switches the one radio's active band and channel together at runtime. HU
    // is explicit because channel 120 (5600 MHz) is outside the world-safe default
    // mask; the driver remains responsible for the country's DFS restrictions.
    ESP_ERROR_CHECK(esp_wifi_set_country_code("HU", false));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_band_mode(CONFIG_WIFI_BAND_MODE));
    wifi_protocols_t protocols = {
        .ghz_2g = CONFIG_WIFI_2G_PROTOCOL,
        .ghz_5g = CONFIG_WIFI_5G_PROTOCOL,
    };
    ESP_ERROR_CHECK(esp_wifi_set_protocols(ESP_IF_WIFI_STA, &protocols));
    ESP_ERROR_CHECK(set_radio_bandwidth(CONFIG_WIFI_BANDWIDTH));
#else
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(ESP_IF_WIFI_STA, CONFIG_WIFI_BANDWIDTH));
    ESP_ERROR_CHECK(esp_wifi_start());
#endif

    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    // Through apply_channel so boot and "CHAN <n>" cannot drift apart on how the
    // HT40 secondary is placed.
    if (!apply_channel(CONFIG_LESS_INTERFERENCE_CHANNEL)) {
        ESP_LOGE(TAG, "could not set channel %d", CONFIG_LESS_INTERFERENCE_CHANNEL);
    }

    // Unlike csi_send/csi_recv, we deliberately do NOT override the STA MAC here:
    // every board's real factory MAC is what the PC uses to identify it.
}

static void wifi_esp_now_init(void)
{
    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_set_pmk((uint8_t *)"pmk1234567890123"));

    esp_now_peer_info_t peer = {
        .channel   = cur_channel,
        .ifidx     = WIFI_IF_STA,
        .encrypt   = false,
        .peer_addr = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
    };
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));
    esp_now_rate_config_t rate_config = {
        .phymode = CONFIG_ESP_NOW_PHYMODE,
        .rate = CONFIG_ESP_NOW_RATE,
        .ersu = false,
        .dcm = false
    };
    ESP_ERROR_CHECK(esp_now_set_peer_rate_config(peer.peer_addr, &rate_config));
    // No recv callback needed: role/schedule now comes from the PC over UART, not ESP-NOW.
}

static void wifi_csi_init(void)
{
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));

#if CONFIG_IDF_TARGET_ESP32C5
    // C5's HE-capable Wi-Fi driver uses the acquisition bitfield. ESP-NOW is kept
    // on 11n/HT so one HT-LTF is returned: 57 values at HT20 or 117 at HT40. The
    // older S3 driver can return LLTF + HT-LTF + STBC-HT-LTF in one 192-value block,
    // hence its separate legacy structure below.
    wifi_csi_config_t csi_config = {
        .enable                 = true,
        .acquire_csi_legacy     = true,
        .acquire_csi_force_lltf = false,
        .acquire_csi_ht20       = true,
        .acquire_csi_ht40       = true,
        .acquire_csi_vht        = false,
        .acquire_csi_su         = false,
        .acquire_csi_mu         = false,
        .acquire_csi_dcm        = false,
        .acquire_csi_beamformed = false,
        .acquire_csi_he_stbc_mode = ESP_CSI_ACQUIRE_STBC_HELTF1,
        .val_scale_cfg          = 0,
        .dump_ack_en            = false,
        .lltf_bit_mode          = 1,
    };
#else
    wifi_csi_config_t csi_config = {
        .lltf_en           = true,
        .htltf_en          = true,
        .stbc_htltf2_en    = true,
        .ltf_merge_en      = true,
        .channel_filter_en = true,
        .manu_scale        = false,
        .shift             = false,
    };
#endif
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // The UART driver before anything can print: a 24 KB interrupt-driven TX ring
    // so every emitter enqueues and returns (the WiFi task included -- see emit()),
    // and a 2 KB RX ring so commands survive a busy wire (the bare 128 B hardware
    // FIFO is how SCAN used to eat role commands: everything past 128 B vanished).
    // Routing the VFS console through the driver also serialises ESP_LOG onto the
    // same ring, closing the interleave-mid-frame hole print_mux never covered.
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM_0, 2048, 24576, 0, NULL, 0));
    uart_vfs_dev_use_driver(UART_NUM_0);

    init_led();
    set_led(40, 0, 0); // default to red/RX until the PC says otherwise

    uint8_t mac[6];
    ESP_ERROR_CHECK(esp_read_mac(mac, ESP_MAC_WIFI_STA));

    wifi_init();
    wifi_esp_now_init();
    wifi_csi_init();

    const esp_timer_create_args_t ping_timer_args = {
        .callback = &ping_timer_cb,
        .name = "rr_ping",
    };
    ESP_ERROR_CHECK(esp_timer_create(&ping_timer_args, &ping_timer));

    xTaskCreate(uart_command_task, "uart_cmd", 4096, NULL, 5, NULL);

    ESP_LOGI(TAG, "Board MAC " MACSTR ", waiting for PC-driven TX/RX commands", MAC2STR(mac));
}
