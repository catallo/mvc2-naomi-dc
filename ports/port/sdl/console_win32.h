/*
 * console_win32.h  The game's text output on Windows (console_win32.c)
 */
#ifndef CONSOLE_WIN32_H
#define CONSOLE_WIN32_H

int console_start(void);    /* stdout and stderr to the terminal the game was started from; 1 when stderr is read */
void console_open(void);    /* a console of its own (console = yes), unless stderr already goes somewhere */

#endif
