/*
 * coro_i386.c  Coroutines for 32-bit x86 Linux: coro_i386_swap saves ebp, ebx, esi and edi and swaps stacks
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "coro.h"

struct coro {
    void* sp;
    void* stack;
    unsigned stack_size;
    void (*fn)(void*);
    void* arg;
};

static coro main_coro;
static coro* current;

/* outside any C function, so that gcc takes it as an ordinary call, which may change eax, ecx and edx */
void coro_i386_swap(void** save_sp, void* load_sp) __attribute__((visibility("hidden")));
__asm__(".text\n"
        ".globl coro_i386_swap\n"
        ".type coro_i386_swap, @function\n"
        "coro_i386_swap:\n"
        "    movl 4(%esp), %eax\n"
        "    movl 8(%esp), %edx\n"
        "    pushl %ebp\n"
        "    pushl %ebx\n"
        "    pushl %esi\n"
        "    pushl %edi\n"
        "    movl %esp, (%eax)\n"
        "    movl %edx, %esp\n"
        "    popl %edi\n"
        "    popl %esi\n"
        "    popl %ebx\n"
        "    popl %ebp\n"
        "    ret\n"
        ".size coro_i386_swap, . - coro_i386_swap\n");

static void coro_start(void) {
    current->fn(current->arg);
    abort();                                            /* a coroutine never returns */
}

coro* coro_self(void) {
    if (current == NULL) {
        current = &main_coro;
    }
    return current;
}

/* as if it had switched away at the start of coro_start, entered with the stack 16-byte aligned */
static void start_on_stack(coro* c, void (*fn)(void*), void* arg) {
    uint32_t* top = (uint32_t*)(((uintptr_t)c->stack + c->stack_size) & ~(uintptr_t)15);
    c->fn = fn;
    c->arg = arg;
    /* what coro_i386_swap pops: edi, esi, ebx, ebp, then coro_start, whose return address is never used */
    top -= 1;
    *top = 0;
    top -= 1;
    *top = (uint32_t)(uintptr_t)coro_start;
    top -= 4;
    memset(top, 0, 4 * 4);
    c->sp = top;
}

coro* coro_create(void (*fn)(void*), void* arg, unsigned stack_size) {
    coro* c = calloc(1, sizeof(coro));
    coro_self();
    if (c != NULL) {
        c->stack = malloc(stack_size);
    }
    if (c == NULL || c->stack == NULL) {
        fprintf(stderr, "coro_create: no memory for a %u byte stack\n", stack_size);
        abort();
    }
    c->stack_size = stack_size;
    start_on_stack(c, fn, arg);
    return c;
}

void coro_restart(coro* c, void (*fn)(void*), void* arg) {
    if (c != current && c != &main_coro) {
        start_on_stack(c, fn, arg);
    }
}

void coro_delete(coro* c) {
    if (c != NULL && c != current && c != &main_coro) {
        free(c->stack);
        free(c);
    }
}

void coro_switch(coro* to) {
    coro* from = coro_self();
    if (to != from) {
        current = to;
        coro_i386_swap(&from->sp, to->sp);
    }
}
