/*
 * DHT11 reference reader for Linux GPIO chardev v1 (Linux 4.8+, e.g. 4.19).
 * Build ON THE ATLAS: gcc -O2 -std=c11 -Wall -Wextra dht11_read.c -o dht11_read
 * Run: sudo ./dht11_read --gpio 504 --attempts 5 --interval 3
 * Explicit mapping: sudo ./dht11_read --chip /dev/gpiochipN --line OFFSET
 * Decode-only test (also works on macOS):
 *   cc -O2 -std=c11 -DDHT11_DECODE_SELFTEST dht11_read.c -o decode_test
 *   ./decode_test
 *
 * Hardware UNTESTED. Requires native GPIO, bidirectional direction control,
 * BOTH-edge IRQ support and suitable electrical level conversion/pull-ups.
 * Never substitute an I2C GPIO expander. There is a small unavoidable gap when
 * switching output to input/IRQ on the v1 API; some kernels will miss the
 * response. GPIO events are kernel IRQ timestamps, not hardware input capture;
 * interrupts can still be delayed or lost. Errors never produce dummy samples.
 * No GPIO sysfs export is performed. Stop any other reader of the same line.
 */
#define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 700
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RISE 1U
#define FALL 2U
#define MAX_EDGES 100
struct edge { uint64_t ns; unsigned id; };
struct sample { double temperature_c, humidity_percent; unsigned char raw[5]; };

static int between_us(uint64_t ns, unsigned low, unsigned high)
{
    return ns >= (uint64_t)low * 1000 && ns <= (uint64_t)high * 1000;
}

/* Validate handshake + 40 bits, timing, checksum and the DHT11 byte format. */
static int decode(const struct edge *e, size_t n, struct sample *s)
{
    for (size_t start = 0; start + 81 < n; ++start) {
        if (e[start].id != RISE || e[start + 1].id != FALL ||
            e[start + 1].ns <= e[start].ns ||
            !between_us(e[start + 1].ns - e[start].ns, 60, 115)) continue;
        if (start && e[start - 1].id == FALL &&
            (e[start].ns <= e[start - 1].ns ||
             !between_us(e[start].ns - e[start - 1].ns, 60, 115))) continue;
        unsigned char b[5] = {0};
        int ok = 1;
        for (unsigned bit = 0; bit < 40; ++bit) {
            size_t r = start + 2 + 2 * bit, f = r + 1;
            if (e[r].id != RISE || e[f].id != FALL ||
                e[r].ns <= e[r - 1].ns || e[f].ns <= e[r].ns ||
                !between_us(e[r].ns - e[r - 1].ns, 30, 85)) {
                ok = 0; break;
            }
            uint64_t high = e[f].ns - e[r].ns;
            int value;
            if (between_us(high, 10, 45)) value = 0;
            else if (between_us(high, 55, 105)) value = 1;
            else { ok = 0; break; } /* Reject ambiguous pulse lengths. */
            b[bit / 8] = (unsigned char)((b[bit / 8] << 1) | value);
        }
        if (!ok || (unsigned char)(b[0] + b[1] + b[2] + b[3]) != b[4]) continue;
        if (!(b[0] | b[1] | b[2] | b[3] | b[4])) continue;
        /* DHT11 decimal bytes contain decimal digits, not DHT22's 16-bit value. */
        if (b[1] > 9 || (b[3] & 0x7f) > 9) continue;
        double humidity = b[0] + b[1] / 10.0;
        double temperature = b[2] + (b[3] & 0x7f) / 10.0;
        if (b[3] & 0x80) temperature = -temperature;
        if (humidity < 0 || humidity > 100 || temperature < -20 || temperature > 60)
            continue;
        s->temperature_c = temperature;
        s->humidity_percent = humidity;
        memcpy(s->raw, b, sizeof(b));
        return 0;
    }
    return -1;
}

#ifdef DHT11_DECODE_SELFTEST
static size_t fake_edges(struct edge *e, const unsigned char b[5])
{
    size_t n = 0;
    uint64_t t = 1000000;
    e[n++] = (struct edge){t, FALL};
    t += 80000; e[n++] = (struct edge){t, RISE};
    t += 80000; e[n++] = (struct edge){t, FALL};
    for (unsigned bit = 0; bit < 40; ++bit) {
        t += 50000; e[n++] = (struct edge){t, RISE};
        t += ((b[bit / 8] >> (7 - bit % 8)) & 1) ? 70000 : 27000;
        e[n++] = (struct edge){t, FALL};
    }
    return n;
}
int main(void)
{
    struct edge e[MAX_EDGES]; struct sample s;
    unsigned char b[5] = {55, 0, 24, 0, 79};
    size_t n = fake_edges(e, b);
    if (decode(e, n, &s) || s.temperature_c != 24 || s.humidity_percent != 55) return 1;
    if (decode(e + 1, n - 1, &s)) return 2; /* First response falling edge may be missed. */
    if (!decode(e, n - 2, &s)) return 3;   /* Incomplete frame. */
    b[4] ^= 1; n = fake_edges(e, b);
    if (!decode(e, n, &s)) return 4;       /* Bad checksum. */
    memset(b, 0, sizeof(b)); n = fake_edges(e, b);
    if (!decode(e, n, &s)) return 5;       /* All-zero bus artifact. */
    b[0] = 55; b[2] = 24; b[4] = 79; n = fake_edges(e, b);
    e[4].ns = e[3].ns + 50000;
    if (!decode(e, n, &s)) return 6;       /* Ambiguous high duration. */
    puts("PASS: valid, partial-handshake, incomplete, checksum, zero and timing cases");
    return 0;
}
#else
#ifndef __linux__
#error "Hardware reader must be compiled on Linux; use DHT11_DECODE_SELFTEST on other OSes."
#endif
#include <errno.h>
#include <fcntl.h>
#include <glob.h>
#include <limits.h>
#include <linux/gpio.h>
#include <poll.h>
#include <sched.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t stopped;
static void stop(int signo) { (void)signo; stopped = 1; }
static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
static void pause_ns(uint64_t ns)
{
    struct timespec req = {(time_t)(ns / 1000000000ULL), (long)(ns % 1000000000ULL)};
    while (!stopped && nanosleep(&req, &req) && errno == EINTR) {}
}
static int read_uint(const char *directory, const char *name, unsigned *out)
{
    char path[PATH_MAX];
    if (snprintf(path, sizeof(path), "%s/%s", directory, name) >= (int)sizeof(path)) return -1;
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int result = fscanf(f, "%u", out) == 1 ? 0 : -1;
    fclose(f); return result;
}

/*
 * Sysfs gpiochip<BASENUMBER> and /dev/gpiochip<DEVICEID> use different numbers.
 * Match their actual sysfs parent identities + line counts. Refuse ambiguity.
 */
static int resolve_global(unsigned gpio, char *device, size_t cap, unsigned *offset)
{
    glob_t classes = {0}, chars = {0};
    unsigned matches = 0;
    if (glob("/sys/class/gpio/gpiochip*", 0, NULL, &classes) ||
        glob("/dev/gpiochip*", 0, NULL, &chars)) goto done;
    for (size_t i = 0; i < classes.gl_pathc; ++i) {
        unsigned base, count;
        if (read_uint(classes.gl_pathv[i], "base", &base) ||
            read_uint(classes.gl_pathv[i], "ngpio", &count) ||
            gpio < base || gpio - base >= count) continue;
        char link[PATH_MAX], parent[PATH_MAX];
        if (snprintf(link, sizeof(link), "%s/device", classes.gl_pathv[i]) >= (int)sizeof(link) ||
            !realpath(link, parent)) continue;
        for (size_t j = 0; j < chars.gl_pathc; ++j) {
            int fd = open(chars.gl_pathv[j], O_RDONLY | O_CLOEXEC);
            struct gpiochip_info info = {0};
            if (fd < 0) continue;
            int rc = ioctl(fd, GPIO_GET_CHIPINFO_IOCTL, &info);
            close(fd);
            if (rc || info.lines != count) continue;
            char identity[PATH_MAX], actual[PATH_MAX];
            if (snprintf(identity, sizeof(identity), "/sys/bus/gpio/devices/%.*s",
                         (int)sizeof(info.name), info.name) >= (int)sizeof(identity) ||
                !realpath(identity, actual)) continue;
            int same = strcmp(actual, parent) == 0;
            char *slash = strrchr(actual, '/');
            if (slash) { *slash = '\0'; same |= strcmp(actual, parent) == 0; }
            if (!same) continue;
            if (strlen(chars.gl_pathv[j]) >= cap) continue;
            strcpy(device, chars.gl_pathv[j]);
            *offset = gpio - base;
            ++matches;
        }
    }
done:
    globfree(&classes); globfree(&chars);
    if (matches != 1) {
        fprintf(stderr, "Cannot uniquely map global GPIO %u to a character device (%u matches).\n"
                "Check /sys/class/gpio/gpiochip*/{base,ngpio}, gpioinfo and board driver;\n"
                "then use --chip /dev/gpiochipN --line OFFSET. Do not guess N.\n", gpio, matches);
        return -1;
    }
    return 0;
}

static int request_input(int chip, unsigned offset)
{
    struct gpiohandle_request r = {0};
    r.lineoffsets[0] = offset; r.flags = GPIOHANDLE_REQUEST_INPUT; r.lines = 1;
    strcpy(r.consumer_label, "dht11-idle");
    if (ioctl(chip, GPIO_GET_LINEHANDLE_IOCTL, &r)) return -1;
    close(r.fd); return 0;
}

static int capture(int chip, unsigned offset, struct edge *edges, size_t *count)
{
    struct gpiohandle_request out = {0};
    struct gpioevent_request event = {0};
    *count = 0;
    out.lineoffsets[0] = offset; out.lines = 1;
    out.flags = GPIOHANDLE_REQUEST_OUTPUT; out.default_values[0] = 0;
    strcpy(out.consumer_label, "dht11-start");
    event.lineoffset = offset; event.handleflags = GPIOHANDLE_REQUEST_INPUT;
    event.eventflags = GPIOEVENT_REQUEST_BOTH_EDGES;
    strcpy(event.consumer_label, "dht11-edges");
    if (ioctl(chip, GPIO_GET_LINEHANDLE_IOCTL, &out)) { perror("request output-low"); return -1; }
    pause_ns(20000000); /* Start condition: >=18 ms low. Never actively drive high. */
    close(out.fd);
    if (stopped) { request_input(chip, offset); return -1; }
    if (ioctl(chip, GPIO_GET_LINEEVENT_IOCTL, &event)) {
        int saved = errno;
        request_input(chip, offset);
        errno = saved; perror("request BOTH-edge input events");
        fprintf(stderr, "This driver may lack GPIO IRQ/chardev support; repeated polling in Python is not a fix.\n");
        return -1;
    }
    if (fcntl(event.fd, F_SETFL, O_NONBLOCK) < 0) {
        perror("nonblocking events"); close(event.fd); return -1;
    }
    uint64_t deadline = now_ns() + 20000000; /* Whole frame normally takes <6 ms. */
    while (!stopped && *count < MAX_EDGES) {
        uint64_t now = now_ns();
        if (now >= deadline) break;
        struct pollfd pfd = {event.fd, POLLIN, 0};
        int timeout_ms = (int)((deadline - now + 999999) / 1000000);
        int ready = poll(&pfd, 1, timeout_ms);
        if (ready < 0) { if (errno == EINTR) continue; perror("poll"); break; }
        if (!ready) break;
        if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) break;
        if (!(pfd.revents & POLLIN)) continue;
        for (;;) {
            struct gpioevent_data ev;
            ssize_t got = read(event.fd, &ev, sizeof(ev));
            if (got < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
            if (got < 0 && errno == EINTR) continue;
            if (got != (ssize_t)sizeof(ev)) { close(event.fd); return -1; }
            edges[(*count)++] = (struct edge){ev.timestamp, ev.id};
            if (*count == MAX_EDGES) break;
        }
    }
    /* Release ownership. A driver may restore its default direction on close. */
    close(event.fd);
    return stopped ? -1 : 0;
}

static int number(const char *text, unsigned *result)
{
    char *end; errno = 0;
    unsigned long v = strtoul(text, &end, 10);
    if (!*text || *text == '-' || *end || errno || v > UINT_MAX) return -1;
    *result = (unsigned)v; return 0;
}
static void usage(const char *program)
{
    fprintf(stderr, "Usage: %s [--gpio 504 | --chip /dev/gpiochipN --line OFFSET]\n"
            "       [--attempts 5] [--interval 3]\n"
            "Returns one valid JSON sample; attempts are >=3 seconds apart.\n", program);
}
int main(int argc, char **argv)
{
    unsigned gpio = 504, offset = 0, attempts = 5, interval = 3;
    int explicit_chip = 0, explicit_line = 0, explicit_gpio = 0;
    char device[PATH_MAX] = {0};
    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--help")) { usage(argv[0]); return 0; }
        if (i + 1 >= argc) { usage(argv[0]); return 2; }
        const char *value = argv[++i];
        if (!strcmp(arg, "--chip")) {
            if (strlen(value) >= sizeof(device)) return 2;
            strcpy(device, value); explicit_chip = 1;
        } else if (!strcmp(arg, "--line")) {
            if (number(value, &offset)) return 2;
            explicit_line = 1;
        } else if (!strcmp(arg, "--gpio")) {
            if (number(value, &gpio)) return 2;
            explicit_gpio = 1;
        } else if (!strcmp(arg, "--attempts")) {
            if (number(value, &attempts) || !attempts || attempts > 100) return 2;
        } else if (!strcmp(arg, "--interval")) {
            if (number(value, &interval) || interval < 3 || interval > 3600) return 2;
        } else { usage(argv[0]); return 2; }
    }
    if (explicit_chip != explicit_line || (explicit_chip && explicit_gpio)) {
        usage(argv[0]); return 2;
    }
    if (!explicit_chip && resolve_global(gpio, device, sizeof(device), &offset)) return 2;
    int chip = open(device, O_RDONLY | O_CLOEXEC);
    if (chip < 0) { perror(device); return 2; }
    struct gpiochip_info info = {0};
    if (ioctl(chip, GPIO_GET_CHIPINFO_IOCTL, &info) || offset >= info.lines) {
        fprintf(stderr, "Invalid GPIO chip/line: %s offset %u\n", device, offset);
        close(chip); return 2;
    }
    fprintf(stderr, "Using %s offset %u, label '%.*s' (hardware not verified).\n",
            device, offset, (int)sizeof(info.label), info.label);
    struct sigaction sa = {0}; sa.sa_handler = stop; sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL); sigaction(SIGTERM, &sa, NULL);
    /* A small real-time priority helps drain the v1 FIFO; IRQ threads keep priority. */
    struct sched_param sp = {.sched_priority = 20};
    if (sched_setscheduler(0, SCHED_FIFO, &sp))
        fprintf(stderr, "FIFO scheduling unavailable; edge loss is more likely under load.\n");
    if (request_input(chip, offset)) {
        perror("request idle input (line may be exported/in use)"); close(chip); return 2;
    }
    pause_ns(1000000000); /* Allow a newly powered sensor to settle. */
    for (unsigned attempt = 1; !stopped && attempt <= attempts; ++attempt) {
        uint64_t started = now_ns();
        struct edge edges[MAX_EDGES]; size_t count = 0; struct sample s;
        if (!capture(chip, offset, edges, &count) && !decode(edges, count, &s)) {
            printf("{\"temperature_c\":%.1f,\"humidity_percent\":%.1f,\"raw\":[%u,%u,%u,%u,%u]}\n",
                   s.temperature_c, s.humidity_percent,
                   (unsigned)s.raw[0], (unsigned)s.raw[1], (unsigned)s.raw[2],
                   (unsigned)s.raw[3], (unsigned)s.raw[4]);
            close(chip); return 0;
        }
        fprintf(stderr, "Attempt %u/%u: no valid DHT11 frame (%zu edges).\n", attempt, attempts, count);
        if (attempt < attempts && !stopped) {
            uint64_t elapsed = now_ns() - started, period = (uint64_t)interval * 1000000000ULL;
            if (elapsed < period) pause_ns(period - elapsed);
        }
    }
    close(chip);
    return stopped ? 130 : 1;
}
#endif
