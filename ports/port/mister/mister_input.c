/*
 * mister_input.c  The controls: from the MVC2 wrapper's pads (MVC2_JOY_SHM, its shared page: a mask a player in the
 * core's order), else from Linux's input devices (/dev/input/event*), looked for again every two seconds: a keyboard
 * as the PC's keys, and pads by place (the first player 1's, the second player 2's): West and North punch, South and
 * East kick, the shoulders assist 1, the triggers assist 2, Start, Select a coin
 */

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/input.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "mister_front.h"
#include "naomi_host.h"
#include "naomi_input.h"

#define MAX_DEVS 8

typedef struct {
    int fd;
    char path[64];
    int pad;                        /* 0 a keyboard */
    int player;
    uint32_t held;                  /* JVS bits from its buttons */
    uint32_t axes;                  /* and from its stick and hat */
    int coin, select, quit_held;
    int abs_lo[ABS_HAT0Y + 1], abs_hi[ABS_HAT0Y + 1];
} DEV;

static DEV devs[MAX_DEVS];
static int ndevs;
static uint64_t next_scan;
static int quit_req;

static int bit(const unsigned long* bits, int n) {
    return (bits[n / (8 * sizeof(long))] >> (n % (8 * sizeof(long)))) & 1;
}

static void add(const char* path) {
    unsigned long keys[KEY_MAX / (8 * sizeof(long)) + 1], abs[ABS_MAX / (8 * sizeof(long)) + 1];
    char name[128] = "";
    DEV* d;
    int fd, i, pads = 0;
    for (i = 0; i < ndevs; i++) {
        if (strcmp(devs[i].path, path) == 0) {
            return;
        }
    }
    if (ndevs == MAX_DEVS || (fd = open(path, O_RDONLY | O_NONBLOCK)) < 0) {
        return;
    }
    memset(keys, 0, sizeof keys);
    memset(abs, 0, sizeof abs);
    ioctl(fd, EVIOCGBIT(EV_KEY, sizeof keys), keys);
    ioctl(fd, EVIOCGBIT(EV_ABS, sizeof abs), abs);
    ioctl(fd, EVIOCGNAME(sizeof name), name);
    d = &devs[ndevs];
    memset(d, 0, sizeof *d);
    d->fd = fd;
    snprintf(d->path, sizeof d->path, "%s", path);
    if (bit(keys, BTN_SOUTH) || bit(keys, BTN_JOYSTICK) || bit(keys, BTN_TRIGGER_HAPPY1)) {
        d->pad = 1;
    } else if (!bit(keys, KEY_A) || !bit(keys, KEY_ENTER)) {
        close(fd);      /* neither a pad nor a keyboard (a mouse, the power button) */
        return;
    }
    for (i = 0; i <= ABS_HAT0Y; i++) {
        struct input_absinfo a;
        if (bit(abs, i) && ioctl(fd, EVIOCGABS(i), &a) == 0) {
            int third = (a.maximum - a.minimum) / 3;
            d->abs_lo[i] = a.minimum + third, d->abs_hi[i] = a.maximum - third;
        }
    }
    for (i = 0; i < ndevs; i++) {
        pads += devs[i].pad;
    }
    d->player = d->pad ? pads % 2 : 0;
    ndevs++;
    host_log("input: %s (%s), %s", path, name, d->pad ? (d->player ? "player 2's pad" : "player 1's pad") : "keyboard");
}

static void scan(void) {
    DIR* dir = opendir("/dev/input");
    struct dirent* e;
    char path[64];
    if (dir == NULL) {
        return;
    }
    while ((e = readdir(dir)) != NULL) {
        if (strncmp(e->d_name, "event", 5) == 0) {
            snprintf(path, sizeof path, "/dev/input/%.40s", e->d_name);
            add(path);
        }
    }
    closedir(dir);
}

static void drop(int i) {
    host_log("input: %s gone", devs[i].path);
    close(devs[i].fd);
    devs[i] = devs[--ndevs];
}

/* a key's JVS bit for player side (0, 1), or 0 */
static uint32_t key_bit(int code, int* side) {
    static const struct {
        int code, side;
        uint32_t bit;
    } keys[] = {
        { KEY_UP, 0, JVS_UP }, { KEY_DOWN, 0, JVS_DOWN }, { KEY_LEFT, 0, JVS_LEFT }, { KEY_RIGHT, 0, JVS_RIGHT },
        { KEY_A, 0, JVS_B1 }, { KEY_S, 0, JVS_B2 }, { KEY_D, 0, JVS_B3 }, { KEY_Z, 0, JVS_B4 }, { KEY_X, 0, JVS_B5 },
        { KEY_C, 0, JVS_B6 }, { KEY_1, 0, JVS_START }, { KEY_9, 0, JVS_SERVICE },
        { KEY_I, 1, JVS_UP }, { KEY_K, 1, JVS_DOWN }, { KEY_J, 1, JVS_LEFT }, { KEY_L, 1, JVS_RIGHT },
        { KEY_R, 1, JVS_B1 }, { KEY_T, 1, JVS_B2 }, { KEY_Y, 1, JVS_B3 }, { KEY_F, 1, JVS_B4 }, { KEY_G, 1, JVS_B5 },
        { KEY_H, 1, JVS_B6 }, { KEY_2, 1, JVS_START },
    };
    size_t i;
    for (i = 0; i < sizeof keys / sizeof keys[0]; i++) {
        if (keys[i].code == code) {
            *side = keys[i].side;
            return keys[i].bit;
        }
    }
    return 0;
}

/* a pad's button: by place where it says (BTN_SOUTH...), else a generic stick's first eight in order */
static uint32_t pad_bit(int code) {
    switch (code) {
    case BTN_WEST: return JVS_B1;
    case BTN_NORTH: return JVS_B2;
    case BTN_TL: case BTN_TR: return JVS_B3;
    case BTN_SOUTH: return JVS_B4;
    case BTN_EAST: return JVS_B5;
    case BTN_TL2: case BTN_TR2: return JVS_B6;
    case BTN_START: return JVS_START;
    case BTN_DPAD_UP: return JVS_UP;
    case BTN_DPAD_DOWN: return JVS_DOWN;
    case BTN_DPAD_LEFT: return JVS_LEFT;
    case BTN_DPAD_RIGHT: return JVS_RIGHT;
    case BTN_TRIGGER: return JVS_B1;
    case BTN_THUMB: return JVS_B2;
    case BTN_THUMB2: return JVS_B3;
    case BTN_TOP: return JVS_B4;
    case BTN_TOP2: return JVS_B5;
    case BTN_PINKIE: return JVS_B6;
    case BTN_BASE4: return JVS_START;
    default: return 0;
    }
}

static void axis(DEV* d, int code, int value) {
    uint32_t lo, hi;
    if (code == ABS_X || code == ABS_HAT0X) {
        lo = JVS_LEFT, hi = JVS_RIGHT;
    } else if (code == ABS_Y || code == ABS_HAT0Y) {
        lo = JVS_UP, hi = JVS_DOWN;
    } else {
        return;
    }
    d->axes &= ~(lo | hi);
    d->axes |= value <= d->abs_lo[code] ? lo : value >= d->abs_hi[code] ? hi : 0;
}

/* a pad's bits are its player's; a keyboard's player 2 keys go in the high half */
static void read_dev(int i) {
    DEV* d = &devs[i];
    struct input_event ev[32];
    for (;;) {
        ssize_t n = read(d->fd, ev, sizeof ev);
        int k;
        if (n <= 0) {
            if (n < 0 && errno == ENODEV) {
                drop(i);
            }
            return;
        }
        for (k = 0; k < (int)(n / sizeof ev[0]); k++) {
            const struct input_event* e = &ev[k];
            int side = 0;
            uint32_t b;
            if (e->type == EV_ABS && d->pad) {
                axis(d, e->code, e->value);
                continue;
            }
            if (e->type != EV_KEY || e->value == 2) {
                continue;
            }
            if (e->code == (d->pad ? BTN_SELECT : KEY_5)) {
                d->coin |= e->value == 1;
                d->select = e->value;
            }
            if (!d->pad && e->code == KEY_ESC && e->value == 1) {
                quit_req = 1;
            }
            b = d->pad ? pad_bit(e->code) : key_bit(e->code, &side);
            b <<= side ? 16 : 0;
            d->held = e->value ? d->held | b : d->held & ~b;
        }
    }
}

/* the wrapper's page (/dev/shm/mvc2-joy): the reference's MisterJoyShm, its own magic; the unpack's progress, for
   its OSD, where the page is that long */
#define JOY_MAGIC 0x3243564Du        /* "MVC2" */
typedef struct {
    uint32_t magic, version;
    uint32_t joy_mask[2];
    int8_t left_x[2], left_y[2], right_x[2], right_y[2];
    uint32_t unpack_kb, unpack_kb_total;
    char unpack_name[32];
} JOY_SHM;

static volatile JOY_SHM* joy;
static int joy_tried, joy_unpack;
static uint32_t joy_last[2];

static int joy_open(void) {
    const char* path = getenv("MVC2_JOY_SHM");
    struct stat st;
    int fd;
    if (joy_tried) {
        return joy != NULL;
    }
    joy_tried = 1;
    if (path == NULL || (fd = open(path, O_RDWR)) < 0) {
        return 0;
    }
    joy_unpack = fstat(fd, &st) == 0 && st.st_size >= (off_t)sizeof *joy;
    joy = mmap(NULL, sizeof *joy, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (joy == MAP_FAILED || joy->magic != JOY_MAGIC || joy->version != 1) {
        host_log("input: %s is not the wrapper's pads", path);
        joy = NULL;
        return 0;
    }
    host_log("input: the wrapper's pads, from %s", path);
    return 1;
}

void mister_unpacking(const char* name, uint32_t done_kb, uint32_t total_kb) {
    size_t i;
    if (!joy_open() || !joy_unpack) {
        return;
    }
    for (i = 0; i + 1 < sizeof joy->unpack_name && name[i] != 0; i++) {
        joy->unpack_name[i] = name[i];
    }
    joy->unpack_name[i] = 0;
    joy->unpack_kb = done_kb;
    __sync_synchronize();
    joy->unpack_kb_total = total_kb;
}

/* the core's order: right, left, down, up, LP, HP, A1, LK, HK, A2, Start, Coin, Test, Service */
static void joy_read(HOST_INPUT* in) {
    static const uint32_t bits[] = { JVS_RIGHT, JVS_LEFT, JVS_DOWN, JVS_UP, JVS_B1, JVS_B2, JVS_B3, JVS_B4, JVS_B5,
                                     JVS_B6, JVS_START, 0, 0, JVS_SERVICE };
    static int held[2];
    int p, i;
    for (p = 0; p < 2; p++) {
        uint32_t m = joy->joy_mask[p];
        /* Start and Coin held together for two seconds: quit */
        held[p] = (m & 0xC00) == 0xC00 ? held[p] + 1 : 0;
        if (held[p] == 120) {
            quit_req = 1;
        }
        for (i = 0; i < (int)(sizeof bits / sizeof bits[0]); i++) {
            if (m >> i & 1) {
                in->player[p] |= bits[i];
            }
        }
        in->coin |= (m & ~joy_last[p]) >> 11 & 1;
        in->test |= m >> 12 & 1;
        joy_last[p] = m;
    }
}

void mister_input_read(HOST_INPUT* in) {
    int i;
    uint64_t now = mister_us();
    if (joy_open()) {
        joy_read(in);
        input_opposites(in);
        return;
    }
    if (now >= next_scan) {
        scan();
        next_scan = now + 2000000;
    }
    for (i = ndevs - 1; i >= 0; i--) {
        read_dev(i);
    }
    for (i = 0; i < ndevs; i++) {
        DEV* d = &devs[i];
        if (d->pad) {
            in->player[d->player] |= d->held | d->axes;
            /* Start and Select held together for two seconds: quit */
            d->quit_held = (d->held & JVS_START) && d->select ? d->quit_held + 1 : 0;
            if (d->quit_held == 120) {
                quit_req = 1;
            }
        } else {
            in->player[0] |= d->held & 0xFFFF;
            in->player[1] |= d->held >> 16;
        }
        in->coin |= d->coin;
        d->coin = 0;
    }
    input_opposites(in);
}

int mister_input_quit(void) {
    return quit_req;
}
