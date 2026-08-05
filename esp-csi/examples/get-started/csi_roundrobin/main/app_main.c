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
// CSI-trigger pings/sec while holding the token. Capped by UART bandwidth, not radio:
// each CSI line is ~1215 bytes, and 921600 baud carries ~76 lines/sec. A board is RX
// ~80% of the time, so ping rate R needs 0.8*R lines/sec -- R=100 demands ~97KB/s
// against a ~90KB/s link, which saturates it and truncates/drops lines. 50 leaves ~40%
// headroom. Raising this further needs a denser encoding (binary, or fewer subcarriers).
#define CONFIG_SEND_FREQUENCY               50
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
    if (!s_count) {
        ESP_LOGI(TAG, "================ CSI RECV ================");
        ets_printf("type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_format,n_subcarriers,first_word,amplitude\n");
    }

    // Held across the whole record: the line is emitted by many ets_printf calls, and
    // without this a role marker from the command task lands in the middle of it.
    xSemaphoreTake(print_mux, portMAX_DELAY);
    ets_printf("CSI_AMP,%d," MACSTR ",%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
               rx_id, MAC2STR(info->mac), rx_ctrl->rssi, rx_ctrl->rate, rx_ctrl->sig_mode,
               rx_ctrl->mcs, rx_ctrl->cwb, rx_ctrl->smoothing, rx_ctrl->not_sounding,
               rx_ctrl->aggregation, rx_ctrl->stbc, rx_ctrl->fec_coding, rx_ctrl->sgi,
               rx_ctrl->noise_floor, rx_ctrl->ampdu_cnt, rx_ctrl->channel, rx_ctrl->secondary_channel,
               rx_ctrl->timestamp, rx_ctrl->ant, rx_ctrl->sig_len, rx_ctrl->sig_mode);

    // Emit per-subcarrier amplitude rather than raw I/Q. This use case doesn't need
    // phase, and one value per subcarrier instead of two cuts the line roughly in half
    // -- the bottleneck here is UART bandwidth (a raw-I/Q line is ~1215 bytes against a
    // ~90KB/s link), so halving it directly buys back achievable ping rate and cuts the
    // truncated/corrupted lines that come with running the link near saturation.
    // Line type is CSI_AMP, not CSI_DATA, so a parser expecting I/Q can't misread it.
    int n_sub = info->len / 2;
    ets_printf(",%d,%d,\"[", n_sub, info->first_word_invalid);
    for (int i = 0; i < n_sub; i++) {
        float imag = compensate_gain * (int8_t)info->buf[i * 2];
        float real = compensate_gain * (int8_t)info->buf[i * 2 + 1];
        int amp = (int)(sqrtf(imag * imag + real * real) + 0.5f);
        ets_printf(i ? ",%d" : "%d", amp);
    }
    ets_printf("]\"\n");
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
