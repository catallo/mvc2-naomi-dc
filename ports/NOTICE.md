# Notices

## No game data

This repository holds source code only. The game's program, graphics and sound come from the user's own copy,
read at start-up: the arcade's ROM set, or the files of the Dreamcast disc.

## port/naomi/naomi_m1.c

The M1 cartridge's packed-file scheme is as worked out in MAME's `naomim1.cpp` (BSD-3-Clause,
copyright-holders: Olivier Galibert).

## port/render/rd_gpu_shaders.h

Compiled from `rd_gpu.hlsl` and `rd_gpu_resolve.hlsl` by Microsoft's fxc and dxc (`port/tools/gpu_shaders.sh`); nothing else.

## Libraries

The PC build links [SDL](https://libsdl.org), and the Windows and Linux releases carry it (`SDL3.dll`,
`libSDL3.so.0` in the AppImage). The 3DS build links devkitPro's [libctru](https://github.com/devkitPro/libctru) and
[citro3d](https://github.com/devkitPro/citro3d). All three are under the zlib license.

The Linux AppImage starts with [AppImage's type-2 runtime](https://github.com/AppImage/type2-runtime) (release
20251108, unchanged; MIT), which carries libfuse (LGPL-2.1) and squashfuse (BSD-2-Clause); their sources are there.
It also carries Debian bullseye's [libxkbcommon](https://xkbcommon.org) (MIT), for SDL's Wayland.

`port/ext/libchdr` is [libchdr](https://github.com/rtissera/libchdr) (BSD-3-Clause), for reading a disc image
in a CHD, with the decoders it carries: the LZMA SDK's (public domain), miniz (Unlicense), zstd's
(BSD-3-Clause, or GPLv2) and dr_flac (public domain, or MIT-0). Their licenses are in their files there.

Marvel vs. Capcom 2 and Capcom are trademarks of Capcom; Marvel characters are trademarks of Marvel. This
project is not affiliated with or endorsed by Capcom, Marvel or Sega.
