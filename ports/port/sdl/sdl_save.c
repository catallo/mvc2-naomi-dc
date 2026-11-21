/*
 * sdl_save.c  The game's saves (backup.bin, eeprom.bin) written by a thread of their own, as on the 3DS and the
 * MiSTer: the arcade updates its book-keeping as it is played, and a write held the game's frame for up to 23 ms
 */

#include <string.h>

#include "sdl_front.h"
#include "naomi_host.h"

#define SLOTS 4
#define MOST 4096

static struct {
    char name[1100];
    char data[MOST];
    int size, due;
} slot[SLOTS];
static SDL_Mutex* lock;
static SDL_Condition* wake;
static SDL_Thread* writer;
static int quit;

static int writer_main(void* arg) {
    static char name[sizeof slot[0].name], data[MOST];
    (void)arg;
    for (;;) {
        int i, size = 0;
        SDL_LockMutex(lock);
        for (;;) {
            for (i = 0; i < SLOTS && !slot[i].due; i++) {
            }
            if (i < SLOTS || quit) {
                break;
            }
            SDL_WaitCondition(wake, lock);
        }
        if (i < SLOTS) {
            memcpy(name, slot[i].name, sizeof name);
            memcpy(data, slot[i].data, (size_t)slot[i].size);
            size = slot[i].size;
            slot[i].due = 0;
        }
        SDL_UnlockMutex(lock);
        if (i == SLOTS) {
            return 0;
        }
        host_save_file(name, data, size);
    }
}

/* a copy of the file's bytes given to the writer, the newest for each file; 0 when it can't take them */
int sdl_save_later(const char* name, const void* data, int size) {
    int i, free_at = -1;
    if (size <= 0 || size > MOST || strlen(name) >= sizeof slot[0].name) {
        return 0;
    }
    if (lock == NULL) {
        lock = SDL_CreateMutex();
        wake = SDL_CreateCondition();
        if (lock == NULL || wake == NULL) {
            return 0;
        }
    }
    if (writer == NULL) {
        quit = 0;
        writer = SDL_CreateThread(writer_main, "save", NULL);
        if (writer == NULL) {
            return 0;
        }
    }
    SDL_LockMutex(lock);
    for (i = 0; i < SLOTS; i++) {
        if (strcmp(slot[i].name, name) == 0) {
            break;
        }
        if (slot[i].name[0] == '\0' && free_at < 0) {
            free_at = i;
        }
    }
    if (i == SLOTS) {
        i = free_at;
    }
    if (i >= 0) {
        strcpy(slot[i].name, name);
        memcpy(slot[i].data, data, (size_t)size);
        slot[i].size = size;
        slot[i].due = 1;
        SDL_SignalCondition(wake);
    }
    SDL_UnlockMutex(lock);
    return i >= 0;
}

/* every save written and the writer stopped, before a restart reads them again and at the exit */
void sdl_save_flush(void) {
    if (writer == NULL) {
        return;
    }
    SDL_LockMutex(lock);
    quit = 1;
    SDL_SignalCondition(wake);
    SDL_UnlockMutex(lock);
    SDL_WaitThread(writer, NULL);
    writer = NULL;
}
