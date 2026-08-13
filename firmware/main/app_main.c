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
// CSI-trigger pings/sec while holding the token. UART bandwidth is the limit, and
// the binary frame below is what moved it: a record is 212 B (18 hdr + 192 + 2 sum)
// against 626 B for the CSV it replaced, with no subcarrier discarded.
//
// The number that matters is burst load -- a receiver sees the full ping rate while
// its transmitter holds the token. 50 Hz x 626 B = 31.3 KB/s of the 92.2 KB/s link
// ran for a whole session with zero corrupt lines, so that is the proven-safe load.
// At 30 subcarriers a frame is 50 B and 250 Hz costs only 12.5 KB/s, but that subset
// is not currently sent -- see CONFIG_SUB_COUNT.
//
// Do not simply raise this to fill the arithmetic headroom: 100 Hz on the old CSV
// encoding (62.6 KB/s burst, "68%, comfortable") produced corrupt lines within
// minutes, because ets_printf blocks on a full TX FIFO and starves the UART command
// task until its input overruns. Bandwidth is necessary, not sufficient.
#define CONFIG_SEND_FREQUENCY              125
// Which subcarriers to transmit. The ESP32 reports 192 for HT40: 0-63 is the legacy
// LLTF (measured 5x weaker, mean 8.3 vs ~39), and 64-191 is the real 40 MHz estimate
// as two 20 MHz halves with the DC/guard gap at 123-133. These 30 are spaced evenly
// across that usable span, skipping every subcarrier measured dead over 480 real
// link-captures, so they cover the full band rather than half of it. Measured mean
// amplitude at these indices is 29-67 -- no weak picks.
//
// Currently 0: send all 192 and let analysis decide what to keep. Subsetting is a
// preprocessing choice, and doing it here would bake it into the recordings
// irreversibly. Set to 30 to emit the table below instead (frame 212 B -> 50 B, which
// is what buys ping rate); nothing downstream is hardcoded either way, since the frame
// carries its own n_sub and the host sizes arrays from it.
#define CONFIG_SUB_COUNT                      0
static const uint8_t SUB_INDEX[CONFIG_SUB_COUNT] = {
    66, 70, 74, 78, 82, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121,
    135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 174, 178, 182, 186, 190,
};

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
    ESP_ERROR_CHECK(esp_timer_start_periodic(ping_timer, 1000000 / CONFIG_SEND_FREQUENCY));
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
            vTaskDelay(pdMS_TO_TICKS(10));
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
        ESP_LOGI(TAG, "================ CSI RECV (binary CSI1 frames) ================");
    }

    // Binary frame instead of a CSV line. ASCII decimal spent ~3.5 bytes encoding
    // one byte of information -- 626 B per record where the payload is 192 bytes --
    // and UART bandwidth is the hard limit on ping rate. uint8 amplitudes cut a
    // record to 212 B with no subcarrier discarded, so the rate rises ~3x without
    // trading away frequency resolution.
    //
    //   0    2  magic 0xA5 0x5A
    //   2    1  version (1)
    //   3    1  n_sub
    //   4    6  transmitter MAC
    //  10    1  rssi          (int8)
    //  11    1  noise_floor   (int8)
    //  12    4  local_timestamp, microseconds, little-endian
    //  16    1  clipped count (amplitudes above 255, saturating)
    //  17    1  first_word_invalid
    //  18   ns  amplitudes, uint8
    //  18+ns 2  sum16 of bytes [2, 18+ns), little-endian
    int n_avail = info->len / 2;
    if (n_avail > 255) {
        n_avail = 255;                     // n_sub is one byte
    }
    // Emit a fixed subset when configured. Frame drops 212 B -> 50 B, which is what
    // buys the ping rate; the trade is frequency resolution, not band coverage.
    int n_sub = (CONFIG_SUB_COUNT > 0 && n_avail > SUB_INDEX[CONFIG_SUB_COUNT - 1])
                ? CONFIG_SUB_COUNT : n_avail;
    uint8_t frame[18 + 255 + 2];
    frame[0] = 0xA5;
    frame[1] = 0x5A;
    frame[2] = 1;
    frame[3] = (uint8_t)n_sub;
    memcpy(&frame[4], info->mac, 6);
    frame[10] = (uint8_t)(int8_t)rx_ctrl->rssi;
    frame[11] = (uint8_t)(int8_t)rx_ctrl->noise_floor;
    uint32_t ts = (uint32_t)rx_ctrl->timestamp;
    frame[12] = (uint8_t)(ts & 0xff);
    frame[13] = (uint8_t)((ts >> 8) & 0xff);
    frame[14] = (uint8_t)((ts >> 16) & 0xff);
    frame[15] = (uint8_t)((ts >> 24) & 0xff);
    frame[17] = (uint8_t)info->first_word_invalid;

    unsigned clipped = 0;
    for (int i = 0; i < n_sub; i++) {
        int k = (n_sub == n_avail) ? i : SUB_INDEX[i];
        float imag = compensate_gain * (int8_t)info->buf[k * 2];
        float real = compensate_gain * (int8_t)info->buf[k * 2 + 1];
        int amp = (int)(sqrtf(imag * imag + real * real) + 0.5f);
        if (amp > 255) {
            // Counted, not silently swallowed: if this is ever non-zero the uint8
            // encoding is losing dynamic range and the host needs to know.
            amp = 255;
            clipped++;
        }
        frame[18 + i] = (uint8_t)amp;
    }
    frame[16] = (uint8_t)(clipped > 255 ? 255 : clipped);

    uint16_t sum = 0;
    for (int i = 2; i < 18 + n_sub; i++) {
        sum = (uint16_t)(sum + frame[i]);
    }
    frame[18 + n_sub] = (uint8_t)(sum & 0xff);
    frame[19 + n_sub] = (uint8_t)(sum >> 8);

    // Same UART path as ets_printf (direct ROM writes to the FIFO), so binary frames
    // and ROLE_* text stay strictly ordered. The mutex keeps a role marker from the
    // command task out of the middle of a frame.
    xSemaphoreTake(print_mux, portMAX_DELAY);
    for (int i = 0; i < 20 + n_sub; i++) {
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
