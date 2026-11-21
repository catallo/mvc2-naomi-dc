#!/bin/sh
# usage: tools/linux/package.sh VERSION BUILD OUTDIR   (in Dockerfile's toolchain)
# The Linux package: OUTDIR/mvc2-linux-VERSION.tar.gz, as the Windows zip with MvC2-VERSION-i686.AppImage in place of
# the executables (mvc2, libSDL3.so.0 and libxkbcommon.so.0 in it, with mvc2.desktop and the icon), and its .sha256. The game keeps its
# data, mvc2.cfg and saves beside the AppImage ($APPIMAGE, port/sdl/start.c).
set -eu
v=${1:?usage: package.sh VERSION BUILD OUTDIR} build=${2:?} out=${3:?}
port=$(cd "$(dirname "$0")/../.." && pwd)
name=mvc2-linux-$v
stage=$(mktemp -d)
s=$stage/$name
cmake --install "$build" --prefix "$s" > /dev/null
app=$stage/AppDir
mkdir "$app"
mv "$s/mvc2" "$s/libSDL3.so.0" "$app/"
rm "$s/mvc2dc"
# SDL's Wayland needs libxkbcommon, which SteamOS lacks in 32 bits; SDL loads it from its own folder first
cp -L /usr/lib/i386-linux-gnu/libxkbcommon.so.0 "$app/libxkbcommon.so.0"
ln -s mvc2 "$app/AppRun"
cp "$port/tools/linux/mvc2.desktop" "$app/mvc2.desktop"
# the icon: mvc2.ico's largest picture, which is a PNG
python3 - "$port/port/sdl/mvc2.ico" "$app/mvc2.png" <<'EOF'
import struct, sys
b = open(sys.argv[1], "rb").read()
pics = [struct.unpack_from("<BBBBHHII", b, 6 + 16 * i) for i in range(struct.unpack_from("<H", b, 4)[0])]
w, h, _, _, _, _, size, at = max(pics, key=lambda p: p[0] or 256)
assert b[at:at + 4] == b"\x89PNG"
open(sys.argv[2], "wb").write(b[at:at + size])
EOF
ln -s mvc2.png "$app/.DirIcon"
chmod 644 "$app/mvc2.desktop" "$app/mvc2.png" "$app/libSDL3.so.0" "$app/libxkbcommon.so.0"
mksquashfs "$app" "$stage/app.squashfs" -root-owned -noappend -comp zstd -all-time 0 -quiet > /dev/null
cat /opt/runtime-i686 "$stage/app.squashfs" > "$s/MvC2-$v-i686.AppImage"
chmod 755 "$s/MvC2-$v-i686.AppImage"
mkdir -p "$s/roms/mvsc2" "$s/disc"
touch "$s/roms/mvsc2/PUT mvsc2.zip HERE.txt" "$s/disc/PUT THE USA DISC'S FILES HERE.txt"
lic=$port/LICENSE; [ -f "$lic" ] || lic=$port/../LICENSE    # the public tree has it at its root
cp "$lic" "$s/LICENSE.txt"
cp "$port/NOTICE.md" "$s/NOTICE.md"
chmod 644 "$s"/*.md "$s/LICENSE.txt" "$s/mvc2.cfg" "$s"/*/*/*.txt "$s"/disc/*.txt
tar --sort=name --owner=0 --group=0 --numeric-owner -C "$stage" -czf "$out/$name.tar.gz" "$name"
(cd "$out" && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256")
rm -rf "$stage"
ls -l "$out/$name.tar.gz"
