# The Linux build's toolchain: Debian bullseye's i386 gcc (glibc 2.31, which SteamOS and most of today's systems
# run) and SDL 3.4.2, the Windows release's, built with its video and sound back ends loaded at run time, from its
# own folder first (RUNPATH $ORIGIN: the AppImage carries libxkbcommon); ninja built with 64-bit file offsets, to
# read a Docker Desktop folder from Windows (whose inode numbers pass 32 bits); the AppImage's i686 type-2 runtime
# (static, FUSE 3) and squashfs-tools; fuse3 runs an AppImage in the gates
#   docker build --platform linux/386 -t mvc2-linux-i386 tools/linux
FROM python:3.11-slim-bullseye@sha256:9e25f400253a5fa3191813d6a67eb801ca1e6f012b3bd2588fa6920b59e3eba6
RUN printf 'deb http://archive.debian.org/debian bullseye main\ndeb http://archive.debian.org/debian-security bullseye-security main\ndeb http://archive.debian.org/debian bullseye-backports main\n' > /etc/apt/sources.list \
    && apt-get -o Acquire::Check-Valid-Until=false update \
    && apt-get install -y --no-install-recommends gcc g++ libc6-dev make pkg-config wget ca-certificates file \
        squashfs-tools fuse3 \
        libx11-dev libxext-dev libxrandr-dev libxcursor-dev libxi-dev libxfixes-dev libxss-dev libxtst-dev \
        libxkbcommon-dev libwayland-dev libegl-dev libgl-dev libgles-dev libdrm-dev libgbm-dev \
        libasound2-dev libpulse-dev libudev-dev libdbus-1-dev \
    && apt-get install -y --no-install-recommends -t bullseye-backports cmake \
    && rm -rf /var/lib/apt/lists/*
RUN wget -q -O /tmp/ninja.tar.gz https://github.com/ninja-build/ninja/archive/refs/tags/v1.12.1.tar.gz \
    && echo "821bdff48a3f683bc4bb3b6f0b5fe7b2d647cf65d52aeb63328c91a6c6df285a  /tmp/ninja.tar.gz" | sha256sum -c \
    && mkdir /tmp/ninja && tar -xzf /tmp/ninja.tar.gz -C /tmp/ninja --strip-components=1 \
    && cd /tmp/ninja && CFLAGS=-D_FILE_OFFSET_BITS=64 python3 configure.py --bootstrap > /dev/null \
    && install -m 755 ninja /usr/local/bin/ninja && cd / && rm -rf /tmp/ninja /tmp/ninja.tar.gz
RUN wget -q -O /tmp/sdl3.tar.gz https://github.com/libsdl-org/SDL/releases/download/release-3.4.2/SDL3-3.4.2.tar.gz \
    && echo "ef39a2e3f9a8a78296c40da701967dd1b0d0d6e267e483863ce70f8a03b4050c  /tmp/sdl3.tar.gz" | sha256sum -c \
    && mkdir /tmp/sdl3 && tar -xzf /tmp/sdl3.tar.gz -C /tmp/sdl3 --strip-components=1 \
    && setarch i686 cmake -S /tmp/sdl3 -B /tmp/sdl3/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/sdl3 -DCMAKE_INSTALL_RPATH='$ORIGIN' -DSDL_SHARED=ON -DSDL_STATIC=OFF \
        -DSDL_TESTS=OFF -DSDL_EXAMPLES=OFF \
    && ninja -C /tmp/sdl3/build install \
    && rm -rf /tmp/sdl3 /tmp/sdl3.tar.gz
RUN wget -q -O /opt/runtime-i686 https://github.com/AppImage/type2-runtime/releases/download/20251108/runtime-i686 \
    && echo "e72ea0b140a0a16e680713238a6f30aad278b62c4ca17919c554864124515498  /opt/runtime-i686" | sha256sum -c
WORKDIR /work
