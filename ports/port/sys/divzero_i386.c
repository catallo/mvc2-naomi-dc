/*
 * divzero_i386.c  Division by zero gives 0 quotient and remainder, as SHC's code did, on 32-bit x86 Linux: the
 * SIGFPE handler zeroes the faulting DIV/IDIV's results and skips it
 */

#define _GNU_SOURCE
#include <signal.h>
#include <stdint.h>
#include <string.h>
#include <ucontext.h>

extern const unsigned char __executable_start[];
extern const unsigned char etext[];

/* a ModRM byte's register numbers, as the saved registers' indexes */
static const int gpr[8] = { REG_EAX, REG_ECX, REG_EDX, REG_EBX, REG_ESP, REG_EBP, REG_ESI, REG_EDI };

static uint32_t disp32(const unsigned char* p) {
    uint32_t d;
    memcpy(&d, p, 4);
    return d;
}

static void divzero_handler(int sig, siginfo_t* si, void* context) {
    greg_t* g = ((ucontext_t*)context)->uc_mcontext.gregs;
    const unsigned char* p = (const unsigned char*)g[REG_EIP];
    int op16 = 0;
    int byte_op;
    int len = 0;
    unsigned char modrm;
    int mod, rm;
    uint32_t divisor;

    /* only the program's own divisions; FS and GS prefixes (another segment) are left alone */
    if (si->si_code != FPE_INTDIV || p < __executable_start || p >= etext) {
        goto pass;
    }
    while (p[len] == 0x66 || p[len] == 0x2E || p[len] == 0x3E || p[len] == 0x26 || p[len] == 0x36) {
        if (p[len] == 0x66) {
            op16 = 1;
        }
        len++;
    }
    if (p[len] != 0xF6 && p[len] != 0xF7) {
        goto pass;
    }
    byte_op = p[len] == 0xF6;
    modrm = p[len + 1];
    if (((modrm >> 3) & 7) < 6) {
        goto pass;
    }
    mod = modrm >> 6;
    rm = modrm & 7;
    len += 2;
    if (mod == 3) {
        if (byte_op) {
            divisor = rm < 4 ? (uint32_t)g[gpr[rm]] : (uint32_t)g[gpr[rm - 4]] >> 8;
        } else {
            divisor = (uint32_t)g[gpr[rm]];
        }
    } else {
        uint32_t addr;
        if (rm == 4) {
            unsigned char sib = p[len++];
            int base = sib & 7, index = (sib >> 3) & 7;
            if (mod == 0 && base == 5) {
                addr = disp32(p + len);
                len += 4;
            } else {
                addr = (uint32_t)g[gpr[base]];
            }
            if (index != 4) {
                addr += (uint32_t)g[gpr[index]] << (sib >> 6);
            }
        } else if (mod == 0 && rm == 5) {
            addr = disp32(p + len);
            len += 4;
        } else {
            addr = (uint32_t)g[gpr[rm]];
        }
        if (mod == 1) {
            addr += (uint32_t)(int8_t)p[len];
            len += 1;
        } else if (mod == 2) {
            addr += disp32(p + len);
            len += 4;
        }
        divisor = byte_op ? *(const uint8_t*)(uintptr_t)addr : disp32((const unsigned char*)(uintptr_t)addr);
    }
    if (byte_op) {
        divisor &= 0xFF;
    } else if (op16) {
        divisor &= 0xFFFF;
    }
    /* a quotient too big for its register faults too: that stays a fault, as on Windows */
    if (divisor != 0) {
        goto pass;
    }
    if (byte_op) {
        g[REG_EAX] &= 0xFFFF0000;           /* AL quotient, AH remainder */
    } else if (op16) {
        g[REG_EAX] &= 0xFFFF0000;
        g[REG_EDX] &= 0xFFFF0000;
    } else {
        g[REG_EAX] = 0;
        g[REG_EDX] = 0;
    }
    g[REG_EIP] += len;
    return;
pass:
    /* run again, it ends the program */
    (void)sig;
    signal(SIGFPE, SIG_DFL);
}

void host_divzero_install(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = divzero_handler;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGFPE, &sa, NULL);
}
