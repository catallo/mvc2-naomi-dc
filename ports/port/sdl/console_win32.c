/*
 * console_win32.c  The game's text output on Windows. It is a GUI program, so starting it opens no console: what it
 * writes goes where it was sent (a file or a pipe), else to the terminal it was started from, else to a console of
 * its own when mvc2.cfg says console = yes.
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>

#include "console_win32.h"

static int out_given, err_given, attached;

static int given(DWORD which) {
    HANDLE h = GetStdHandle(which);
    return h != NULL && h != INVALID_HANDLE_VALUE && GetFileType(h) != FILE_TYPE_UNKNOWN;
}

/* stdout and stderr, where not sent anywhere, to the console */
static void to_console(void) {
    if (!out_given) {
        freopen("CONOUT$", "w", stdout);
    }
    if (!err_given) {
        freopen("CONOUT$", "w", stderr);
        setvbuf(stderr, NULL, _IONBF, 0);
    }
}

int console_start(void) {
    out_given = given(STD_OUTPUT_HANDLE), err_given = given(STD_ERROR_HANDLE);
    if ((!out_given || !err_given) && AttachConsole(ATTACH_PARENT_PROCESS)) {
        to_console();
        attached = 1;
    }
    return err_given || attached;
}

void console_open(void) {
    if (!err_given && !attached && AllocConsole()) {
        SetConsoleTitleA("Marvel vs. Capcom 2");
        to_console();
        attached = 1;
    }
}
