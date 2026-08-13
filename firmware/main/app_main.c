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
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "nvs_flash.h"

#include "esp_mac.h"
#include "rom/ets_sys.h"
#include "esp_rom_uart.h"   // esp_rom_uart_tx_one_char: same FIFO path as ets_printf
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

#define STATUS_LED_GPIO 48

#define CONFIG_LESS_INTERFERENCE_CHANNEL   11
#define CONFIG_WIFI_BANDWIDTH               WIFI_BW_HT40
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
#define CSI_FRAME_VERSION                     2
#define CSI_FRAME_HDR                        20     // 2 extra bytes carry the AGC gain
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
    if (n == 166) {
        return SUB_INDEX_166;
    }
    return NULL;                      // 0 or anything unrecognised: send everything
}

#define CONFIG_GAIN_CONTROL                  1     // all our boards are ESP32-S3

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
// ets_printf is not atomic across tasks: CSI lines are printed from the WiFi task
// while role markers are printed from the UART command task, and concurrent calls
// interleave mid-line, corrupting both. That silently ate ~23% of ROLE_TX markers.
// Every multi-line/multi-call print below must hold this.
static SemaphoreHandle_t print_mux;
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
    // ets_printf, not ESP_LOGI: CSI records are emitted with ets_printf (straight to the
    // UART FIFO) while ESP_LOGI goes through the VFS/driver buffer. Mixing the two lets
    // them reorder under buffer pressure, which makes the host misjudge which side of a
    // role change a record falls on. Same path == guaranteed ordering.
    xSemaphoreTake(print_mux, portMAX_DELAY);
    ets_printf("ROLE_TX\n");
    xSemaphoreGive(print_mux);
}

static void become_rx(const uint8_t *peer_mac)
{
    is_tx = false;
    memcpy(tx_filter_mac, peer_mac, 6);
    if (!ident_mode) {
        set_led(40, 0, 0); // red = receiving
    }
    esp_timer_stop(ping_timer); // ignored if not running
    xSemaphoreTake(print_mux, portMAX_DELAY);
    ets_printf("ROLE_RX," MACSTR "\n", MAC2STR(tx_filter_mac)); // ets_printf: see become_tx()
    xSemaphoreGive(print_mux);
}

static void ping_timer_cb(void *arg)
{
    static uint32_t seq = 0;
    esp_err_t ret = esp_now_send(BROADCAST_MAC, (const uint8_t *)&seq, sizeof(seq));
    seq++;
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "free_heap: %ld <%s> ESP-NOW send error", esp_get_free_heap_size(), esp_err_to_name(ret));
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
            // Overruns get likelier as print pressure rises, since ets_printf blocks
            // on a full TX FIFO and starves this task.
            clearerr(stdin);
            // 1 ms, not 10. stdin is non-blocking, so this delay *is* the command
            // latency, and a role change cannot take effect until this task next
            // looks. At 10 ms it cost ~5 ms of every dwell on average -- 17% of a
            // 50 ms dwell and 39% of a 12.5 ms one, which is what made short dwells
            // lose more than they gained. The tick is already 1 kHz, so 1 ms is one
            // tick: the shortest wait that still yields the CPU.
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
                xSemaphoreTake(print_mux, portMAX_DELAY);
                ets_printf("RATE_OK,%u\n", hz);  // ets_printf: see become_tx()
                xSemaphoreGive(print_mux);
            } else {
                ESP_LOGW(TAG, "bad RATE command: '%s'", line);
            }
        } else if (strncmp(line, "SUB ", 4) == 0) {
            // Switch subcarrier set live: 0 (all), 30, or 166. Takes effect on the
            // next CSI callback, and the frame carries its own n_sub, so the host
            // resizes from the wire rather than being told separately.
            unsigned nsc;
            if (sscanf(line + 4, "%u", &nsc) == 1
                && (nsc == 0 || nsc == 30 || nsc == 166)) {
                sub_count = (int)nsc;
                xSemaphoreTake(print_mux, portMAX_DELAY);
                ets_printf("SUB_OK,%u\n", nsc);  // ets_printf: see become_tx()
                xSemaphoreGive(print_mux);
            } else {
                ESP_LOGW(TAG, "bad SUB command: '%s'", line);
            }
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
    static int s_count = 0;
    float compensate_gain = 1.0f;
    static uint8_t agc_gain = 0;
    static int8_t fft_gain = 0;
#if CONFIG_GAIN_CONTROL
    static uint8_t agc_gain_baseline = 0;
    static int8_t fft_gain_baseline = 0;
    esp_csi_gain_ctrl_get_rx_gain(rx_ctrl, &agc_gain, &fft_gain);
    if (s_count < 100) {
        esp_csi_gain_ctrl_record_rx_gain(agc_gain, fft_gain);
    } else if (s_count == 100) {
        esp_csi_gain_ctrl_get_rx_gain_baseline(&agc_gain_baseline, &fft_gain_baseline);
#if CONFIG_FORCE_GAIN
        esp_csi_gain_ctrl_set_rx_force_gain(agc_gain_baseline, fft_gain_baseline);
#endif
    }
    esp_csi_gain_ctrl_get_gain_compensation(&compensate_gain, agc_gain, fft_gain);
#endif

    uint32_t rx_id = *(uint32_t *)(info->payload + 15);
    (void)rx_id;
    if (!s_count) {
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
    int gain_q8 = (int)(compensate_gain * 256.0f + 0.5f);
    if (gain_q8 < 0) {
        gain_q8 = 0;
    } else if (gain_q8 > 65535) {
        gain_q8 = 65535;
    }
    frame[16] = (uint8_t)(gain_q8 & 0xff);
    frame[17] = (uint8_t)(gain_q8 >> 8);
    frame[18] = (uint8_t)info->first_word_invalid;
    frame[19] = 0;
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

    // Same UART path as ets_printf (direct ROM writes to the FIFO), so binary frames
    // and ROLE_* text stay strictly ordered. The mutex keeps a role marker from the
    // command task out of the middle of a frame.
    xSemaphoreTake(print_mux, portMAX_DELAY);
    for (int i = 0; i < n_end + 2; i++) {
        esp_rom_uart_tx_one_char(frame[i]);
    }
    xSemaphoreGive(print_mux);
    s_count++;
}

static void wifi_init(void)
{
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(esp_netif_init());
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));

    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(ESP_IF_WIFI_STA, CONFIG_WIFI_BANDWIDTH));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    if (CONFIG_WIFI_BANDWIDTH == WIFI_BW_HT20) {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_NONE));
    } else {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_BELOW));
    }

    // Unlike csi_send/csi_recv, we deliberately do NOT override the STA MAC here:
    // every board's real factory MAC is what the PC uses to identify it.
}

static void wifi_esp_now_init(void)
{
    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_set_pmk((uint8_t *)"pmk1234567890123"));

    esp_now_peer_info_t peer = {
        .channel   = CONFIG_LESS_INTERFERENCE_CHANNEL,
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

    wifi_csi_config_t csi_config = {
        .lltf_en           = true,
        .htltf_en          = true,
        .stbc_htltf2_en    = true,
        .ltf_merge_en      = true,
        .channel_filter_en = true,
        .manu_scale        = false,
        .shift             = false,
    };
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

    // Before wifi_csi_init() or the command task, so nothing can print unguarded.
    print_mux = xSemaphoreCreateMutex();
    ESP_ERROR_CHECK(print_mux ? ESP_OK : ESP_ERR_NO_MEM);

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
