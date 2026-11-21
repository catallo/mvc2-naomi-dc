#!/bin/sh
# usage: tools/linux/build.sh VERSION OUTDIR   (Linux, or Git Bash with Docker Desktop)
# The Linux release, as CI makes it: the game for 32-bit x86 in Dockerfile's toolchain (build-linux), then
# package.sh into OUTDIR. MVC2_DECOMP: the decompilation (default: the tree around ports/).
set -eu
v=${1:?usage: build.sh VERSION OUTDIR} out=${2:?usage: build.sh VERSION OUTDIR}
port=$(cd "$(dirname "$0")/../.." && pwd)
dec=$(cd "${MVC2_DECOMP:-$port/..}" && pwd)
mkdir -p "$out"
out=$(cd "$out" && pwd)
host() { (cd "$1" && { pwd -W 2>/dev/null || pwd; }); }

docker build --platform linux/386 -q -t mvc2-linux-i386 "$(host "$port/tools/linux")" > /dev/null
# setarch: cmake takes the processor from uname, which says x86_64 in a 32-bit container
MSYS_NO_PATHCONV=1 docker run --rm --platform linux/386 -v "$(host "$port"):/port" -v "$(host "$dec"):/decomp:ro" \
    -v "$(host "$out"):/out" -w /port mvc2-linux-i386 setarch i686 sh -c \
    "cmake -S . -B build-linux -G Ninja -DCMAKE_PREFIX_PATH=/opt/sdl3 -DMVC2_DECOMP=/decomp -DMVC2_VERSION=$v > /dev/null &&
     ninja -C build-linux mvc2 mvc2dc && sh tools/linux/package.sh $v build-linux /out"
