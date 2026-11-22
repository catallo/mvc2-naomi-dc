"""List the places where the game's C may run differently on a host than it did under SHC on the SH-4.

usage: hostdiff.py DECOMP OUTDIR [--program=arcade|dc] [--arm]

Compiles the program's C with the host build's options, gcc's warnings of the kinds that matter and -flto,
then links it with -Wlto-type-mismatch:
- a declaration that disagrees with the definition (a float or struct where the SH-4 used another register,
  a narrower parameter than the routine takes, which the caller cuts on x86 and not on the SH-4);
- a value-returning routine that ends without one (the SH-4 returned whatever r0 held);
- a local read before it is set (the SH-4 read a register or its stack);
- an operand changed twice between sequence points, an implicit declaration, a table read past its end
  that gcc can see.
With --arm it also compiles for the 3DS and lists each float converted to an unsigned type there that
port/data/arm_u32.tsv does not make through sh_u32 (VFP's vcvt.u32 makes a negative 0; SHC's code wraps it). Needs mingw32's gcc in PATH, and devkitARM for --arm.
"""
import concurrent.futures
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fmac  # noqa: E402
from gen_host import values_used  # noqa: E402

PORT = Path(__file__).resolve().parent.parent
WARN = ["-Wall", "-Wreturn-type", "-Wreturn-mismatch", "-Wuninitialized", "-Wmaybe-uninitialized", "-Wsequence-point",
        "-Wimplicit-function-declaration", "-Warray-bounds", "-Wshift-count-overflow", "-Wshift-count-negative",
        "-Wdiv-by-zero", "-Waggressive-loop-optimizations"]
KINDS = [w[2:] for w in WARN[1:]] + ["array-bounds="]
BASE = ["-ffp-contract=off", "-std=gnu89", "-fwrapv", "-fno-strict-aliasing", "-fsigned-char", "-O1",
        "-fno-delete-null-pointer-checks", "-fno-toplevel-reorder", "-fno-zero-initialized-in-bss"]
X86 = ["-malign-data=abi", "-mfpmath=sse", "-msse2"]
ARM = ["-march=armv6k", "-mtune=mpcore", "-mfloat-abi=hard", "-mtp=soft", "-D__3DS__", "-fno-short-enums", "-g"]
WRAPPED = {"shl23_00.c", "at02_scene.c", "pl_char.c", "set05_05.c"}


def sources(game, program):
    """The program's modules as CMakeLists.txt takes them, wrappers (port/game/*_host.c) in their place."""
    srcs = sorted(game.glob("src/*.c"))
    if program == "dc":
        dc = [f for f in sorted(game.glob("src/dc/*.c")) if f.name != "strt2.c"]
        names = {f.name for f in dc}
        srcs = [f for f in srcs if f.name not in names] + dc
    return [f for f in srcs if f.name not in WRAPPED] + [PORT / "game" / (n[:-2] + "_host.c") for n in sorted(WRAPPED)]


def defines(program):
    return ["-DBUILD_DATE=20000529", "-DTARGET_DC"] if program == "dc" else ["-DBUILD_DATE=20000225"]


def common(game, program):
    return defines(program) + BASE + ["-include", str(PORT / "include" / "port_prefix.h"), "-I" + str(game / "include"),
                                      "-I" + str(PORT / "include"), "-iquote", str(game / "src")]


def warnings(game, out, program):
    """gcc's warnings of WARN's kinds, and LTO's mismatches, as 'file:line: kind: text' lines."""
    opts = common(game, program) + X86

    def one(src):
        obj = out / (src.name + ".o")
        r = subprocess.run(["gcc"] + opts + WARN + ["-flto", "-ffat-lto-objects", "-c", str(src), "-o", str(obj)],
                           capture_output=True, text=True)
        return obj, r.stderr

    used = values_used(sources(game, program))
    objs, lines = [], set()
    with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as pool:
        for obj, err in pool.map(one, sources(game, program)):
            objs.append(obj)
            func = None
            for line in err.splitlines():
                f = re.match(r"\S+: In function '(\w+)'", line)
                if f:
                    func = f.group(1)
                m = re.match(r"(\S+?):(\d+):\d+: warning: (.*) \[-W([\w=-]+)\]$", line)
                if m and m.group(4) in KINDS:
                    text = m.group(3)
                    if m.group(4) in ("return-type", "return-mismatch") and func:
                        text = "%s: %s%s" % (func, text, ", its value used" if func in used else "")
                    lines.add((Path(m.group(1)).name, int(m.group(2)), m.group(4), text))
    rsp = out / "objs.rsp"
    rsp.write_text("\n".join(str(o).replace("\\", "/") for o in objs))
    r = subprocess.run(["gcc", "-flto", "-O1", "-r", "-nostdlib", "-flinker-output=nolto-rel",
                        "-Wlto-type-mismatch", "@" + str(rsp), "-o", str(out / "all.o")] + opts,
                       capture_output=True, text=True)
    err = r.stderr.splitlines()
    for i, line in enumerate(err):
        m = re.match(r"(\S+?):(\d+):\d+: warning: type of '(\w+)' does not match original declaration", line)
        if m:
            notes = []
            for nxt in err[i + 1:]:
                if "warning:" in nxt:
                    break
                n = re.match(r"(\S+?):(\d+):\d+: note: (.*)", nxt)
                if n and "previously declared" not in n.group(3) and "misoptimized" not in n.group(3):
                    notes.append(n.group(3))
                elif n and "previously declared" in n.group(3):
                    notes.append("defined at %s:%s" % (Path(n.group(1)).name, n.group(2)))
            lines.add((Path(m.group(1)).name, int(m.group(2)), "lto-type-mismatch", m.group(3) + ": " + "; ".join(notes)))
    return sorted(lines)


def arm_unsigned(game, out, program):
    """Each source line where devkitARM still converts a float to an unsigned type."""
    bin_ = next(p for p in (Path(os.environ.get("DEVKITARM", "")), Path("C:/devkitPro/devkitARM"))
                if (p / "bin").is_dir()) / "bin"
    opts = common(game, program) + ARM + ["-w"]

    def one(src):
        obj = out / (src.name + ".arm.o")
        if src.parent != PORT / "game":   # the host's copy, with the uses arm_u32.tsv lists made through sh_u32
            copy = out / (src.stem + ".arm.c")
            fmac.rewrite(PORT / "data" / ("fmac_%s.tsv" % program), src, copy,
                         [PORT / "data" / e for e in ("arg_order.tsv", "board_reads.tsv", "arm_u32.tsv")])
            src, near = copy, ["-iquote", str(src.parent)]
        else:
            near = []
        subprocess.run([str(bin_ / "arm-none-eabi-gcc")] + opts + near + ["-c", str(src), "-o", str(obj)])
        d = subprocess.run([str(bin_ / "arm-none-eabi-objdump"), "-dl", "--no-show-raw-insn", str(obj)],
                           capture_output=True, text=True).stdout
        found, loc = set(), None
        for line in d.splitlines():
            m = re.match(r"^([A-Za-z]:)?[^:\s]*\.[ch]:(\d+)", line)
            if m:
                loc = (Path(line.split(":")[0] if not m.group(1) else line[:line.rindex(":")]).name, int(m.group(2)))
            elif "vcvt.u32.f32" in line and loc:
                found.add(loc)
        return found

    with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as pool:
        return sorted(set().union(*pool.map(one, sources(game, program))))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    program = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--program=")), "arcade")
    game, out = Path(args[0]), Path(args[1])
    out.mkdir(parents=True, exist_ok=True)
    for name, line, kind, text in warnings(game, out, program):
        print("%s:%d: %s: %s" % (name, line, kind, text))
    if "--arm" in sys.argv:
        for name, line in arm_unsigned(game, out, program):
            print("%s:%d: arm-float-to-unsigned" % (name, line))


if __name__ == "__main__":
    main()
