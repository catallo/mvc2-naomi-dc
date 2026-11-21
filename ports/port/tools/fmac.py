"""SHC's FMACs in the game's C, for a host to round them once as the SH-4 does.

SHC turns some float `c + a * b` into one FMAC (a single rounding); gcc on a host rounds the
product and the sum apart, a ULP off now and then. This finds each `+` or `+=` with a product on
one side, asks SHC which of them it fused (by recompiling the module with one product hidden in a
call: a fused site's module has one FMAC fewer), and writes the fused sites to a list: module,
routine, the sum's place among the routine's sums with a product (n/total) and the product's side.
At build time `rewrite` gives a host copy of a module with those sites as FMAC( a, b, c ), and with
the statements the EDITS.tsv files list changed (every host's calls whose arguments SHC's code takes
in another order than gcc's, and the ARM hosts' float-to-unsigned conversions, through sh_u32). A routine whose sums or statements changed stops it: run `find` on that module again.

usage: fmac.py find DECOMP SITES.tsv [MODULE.c ...]     (SHC set in the environment)
       fmac.py rewrite SITES.tsv SRC.c OUT.c [EDITS.tsv ...]
"""
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PUNCT = sorted("""-> ++ -- <<= >>= <= >= == != && || += -= *= /= %= &= |= ^= << >> ...
               + - * / % & | ^ ! ~ < > = ? : ; , . ( ) [ ] { }""".split(), key=len, reverse=True)
TOKEN = re.compile(r"\s*(?:(?P<num>(?:0[xX][0-9a-fA-F]+|\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?)[fFuUlL]*)"
                   r"|(?P<id>[A-Za-z_]\w*)|(?P<str>\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')|(?P<p>"
                   + "|".join(re.escape(p) for p in PUNCT) + "))")
TYPE_WORDS = {"void", "char", "short", "int", "long", "float", "double", "signed", "unsigned", "const",
              "volatile", "struct", "union", "enum", "register", "static", "extern"}
STOP_WORDS = {"return", "case", "default", "else", "do", "goto", "if", "while", "for", "switch", "sizeof"}
CONTROL = {"if", "while", "for", "switch"}
ASSIGN = {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="}
BINARY = {"||": 4, "&&": 5, "|": 6, "^": 7, "&": 8, "==": 9, "!=": 9, "<": 10, ">": 10, "<=": 10, ">=": 10,
          "<<": 11, ">>": 11, "+": 12, "-": 12, "*": 13, "/": 13, "%": 13}


def blank(text):
    """The text with comments and preprocessor lines as spaces, so offsets stay."""
    out = list(text)
    for m in re.finditer(r"/\*.*?\*/|//[^\n]*|^[ \t]*#(?:[^\n]*\\\n)*[^\n]*|\"(?:\\.|[^\"\\\n])*\"", text,
                         re.S | re.M):
        if m.group(0)[0] in "/#":
            for i in range(m.start(), m.end()):
                if out[i] != "\n":
                    out[i] = " "
    return "".join(out)


def tokens(text):
    toks, pos = [], 0
    while True:
        m = TOKEN.match(text, pos)
        if not m or m.end() == pos:
            if text[pos:].strip():
                raise SyntaxError("cannot read at %d: %r" % (pos, text[pos:pos + 20]))
            return toks
        kind = m.lastgroup
        s = m.start(kind)
        toks.append((kind, m.group(kind), s, m.end()))
        pos = m.end()


class Node:
    def __init__(self, op, lo, hi, kids=(), tok=None):
        self.op, self.lo, self.hi, self.kids, self.tok = op, lo, hi, list(kids), tok


class Parser:
    def __init__(self, toks, lo, hi, types):
        self.t, self.i, self.end, self.types = toks, lo, hi, types

    def peek(self, k=0):
        j = self.i + k
        return self.t[j][1] if j < self.end else None

    def kind(self, k=0):
        j = self.i + k
        return self.t[j][0] if j < self.end else None

    def take(self, want=None):
        if self.i >= self.end or (want and self.t[self.i][1] != want):
            raise SyntaxError("expected %r" % want)
        self.i += 1
        return self.i - 1

    def is_type(self, k):
        v = self.peek(k)
        return v in TYPE_WORDS or v in self.types

    def expr(self, comma=True):
        n = self.assign()
        while comma and self.peek() == ",":
            op = self.take()
            r = self.assign()
            n = Node(",", n.lo, r.hi, [n, r], op)
        return n

    def assign(self):
        n = self.cond()
        if self.peek() in ASSIGN:
            op = self.take()
            r = self.assign()
            n = Node(self.t[op][1], n.lo, r.hi, [n, r], op)
        return n

    def cond(self):
        n = self.binary(4)
        if self.peek() == "?":
            q = self.take()
            a = self.expr()
            self.take(":")
            b = self.cond()
            n = Node("?", n.lo, b.hi, [n, a, b], q)
        return n

    def binary(self, level):
        n = self.unary()
        while self.peek() in BINARY and self.kind() == "p" and BINARY[self.peek()] >= level:
            p = BINARY[self.peek()]
            op = self.take()
            r = self.binary(p + 1)
            n = Node(self.t[op][1], n.lo, r.hi, [n, r], op)
        return n

    def unary(self):
        v = self.peek()
        if v in ("-", "+", "!", "~", "*", "&", "++", "--") and self.kind() == "p":
            op = self.take()
            n = self.unary()
            return Node("u" + v, op, n.hi, [n], op)
        if v == "sizeof":
            op = self.take()
            if self.peek() == "(" and self.is_type(1):
                self.take("(")
                depth = 1
                while depth:
                    depth += {"(": 1, ")": -1}.get(self.peek(), 0)
                    self.take()
                return Node("sizeof", op, self.i - 1)
            n = self.unary()
            return Node("sizeof", op, n.hi, [n])
        if v == "(" and self.is_type(1):
            lo = self.take()
            depth = 1
            while depth:
                depth += {"(": 1, ")": -1}.get(self.peek(), 0)
                self.take()
            n = self.unary()
            return Node("cast", lo, n.hi, [n])
        return self.postfix()

    def postfix(self):
        k, v = self.kind(), self.peek()
        if k in ("id", "num", "str"):
            n = Node(k, self.i, self.i)
            self.take()
            while k == "str" and self.kind() == "str":
                n.hi = self.take()
        elif v == "(":
            lo = self.take()
            e = self.expr()
            hi = self.take(")")
            n = Node("()", lo, hi, [e])
        else:
            raise SyntaxError("unexpected %r" % v)
        while True:
            v = self.peek()
            if v == "(":
                self.take()
                args = []
                while self.peek() != ")":
                    args.append(self.assign())
                    if self.peek() == ",":
                        self.take()
                hi = self.take(")")
                n = Node("call", n.lo, hi, [n] + args)
            elif v == "[":
                self.take()
                e = self.expr()
                hi = self.take("]")
                n = Node("[]", n.lo, hi, [n, e])
            elif v in (".", "->"):
                self.take()
                hi = self.take()
                n = Node(v, n.lo, hi, [n])
            elif v in ("++", "--"):
                hi = self.take()
                n = Node("p" + v, n.lo, hi, [n])
            else:
                return n


def first_arms(text):
    """The text with the #else and #elif arms of a conditional blanked where its arms' braces don't balance,
    so that they count as in one build; arms that balance (whole routines for each program) all stay."""
    out, stack = list(text), []
    clean = blank(text)
    for m in re.finditer(r"^[ \t]*#[ \t]*(if|ifdef|ifndef|elif|else|endif)\b[^\n]*", text, re.M):
        d = m.group(1)
        if d.startswith("if"):
            stack.append([m.end()])
        elif not stack:
            continue
        elif d in ("elif", "else"):
            stack[-1] += [m.start(), m.end()]
        else:
            arms = stack.pop() + [m.start()]
            spans = list(zip(arms[0::2], arms[1::2]))
            if len(spans) > 1 and any(clean[a:b].count("{") != clean[a:b].count("}") for a, b in spans):
                for a, b in spans[1:]:
                    for i in range(a, b):
                        if out[i] != "\n":
                            out[i] = " "
    return "".join(out)


def routines(text):
    """(name, start, end) of each routine body, as offsets in text."""
    toks = tokens(blank(first_arms(text)))
    out, depth, start, name = [], 0, None, None
    for i, (k, v, s, e) in enumerate(toks):
        if v == "{":
            if depth == 0 and i and toks[i - 1][1] == ")":
                j = matching(toks, i - 1, -1)
                start, name = s, toks[j - 1][1]
            depth += 1
        elif v == "}":
            depth -= 1
            if depth == 0 and start is not None:
                out.append((name, start, e))
                start = None
    return out


def routine_of(rts, at):
    for name, s, e in rts:
        if s <= at < e:
            return name
    return "-"


def key_text(key, side):
    return "%s\t%d/%d\t%s" % (key[0], key[1], key[2], side)


def matching(toks, i, step):
    """The index of the bracket matching toks[i]."""
    pairs = {"(": ")", "[": "]", "{": "}", ")": "(", "]": "[", "}": "{"}
    open_, close = toks[i][1], pairs[toks[i][1]]
    depth = 0
    while True:
        v = toks[i][1]
        depth += 1 if v == open_ else -1 if v == close else 0
        if depth == 0:
            return i
        i += step


def region(toks, i):
    """The token range of the expression statement or operand list around token i."""
    lo = i
    while lo > 0:
        v = toks[lo - 1][1]
        if v in (")", "]"):
            j = matching(toks, lo - 1, -1)
            if v == ")" and j > 0 and toks[j - 1][1] in CONTROL:
                break
            lo = j
            continue
        if v in (";", "{", "}", ",", "(", "[", ":") or v in STOP_WORDS:
            break
        lo -= 1
    hi, asks = i, 0
    while hi < len(toks):
        v = toks[hi][1]
        if v in ("(", "["):
            hi = matching(toks, hi, 1) + 1
            continue
        if v == ":" and asks:
            asks -= 1
        elif v in (";", "}", ",", ")", "]", ":", "{"):
            break
        asks += v == "?"
        hi += 1
    return lo, hi


def macros(texts):
    """Function-like macros: name -> (parameters, body); one defined twice (on each side of an #if) is
    left out, as which one holds is not known here."""
    out, twice = {}, set()
    for text in texts:
        for m in re.finditer(r"^[ \t]*#[ \t]*define[ \t]+(\w+)\(([^)]*)\)((?:[^\n]*\\\n)*[^\n]*)", text, re.M):
            body = re.sub(r"/\*.*?\*/", " ", m.group(3).replace("\\\n", " "))
            if m.group(1) in out and out[m.group(1)][1] != body.strip():
                twice.add(m.group(1))
            out[m.group(1)] = ([p.strip() for p in m.group(2).split(",")], body.strip())
    return {n: d for n, d in out.items() if n not in twice}


def typedefs(texts):
    """The names typedefs give, read up to each one's ; with its braces dropped."""
    names = set()
    for text in texts:
        text = blank(text)
        for m in re.finditer(r"\btypedef\b", text):
            i, depth, out = m.end(), 0, []
            while i < len(text) and (text[i] != ";" or depth):
                depth += {"{": 1, "}": -1}.get(text[i], 0)
                if depth == 0 and text[i] != "}":
                    out.append(text[i])
                i += 1
            decl = "".join(out)
            for name in re.findall(r"\(\s*\*\s*(\w+)\s*\)", decl):
                names.add(name)
            decl = re.sub(r"\([^()]*\)", "", decl)
            for part in decl.split(","):
                ids = re.findall(r"[A-Za-z_]\w*", re.sub(r"\[[^\]]*\]", "", part))
                if ids:
                    names.add(ids[-1])
    return names


def strip(n):
    while n.op == "()":
        n = n.kids[0]
    return n


def expand(text, macs):
    """The text with each use of a macro that adds a product written out, as the preprocessor would
    (a line for each line the use took, so the lines stay where they were)."""
    sums = {n for n, (params, body) in macs.items() if "#" not in body and "+" in body
            and ("*" in body or any(re.search(r"\b%s\b" % m, body) for m in macs if m != n))}
    for _ in range(8):
        toks = tokens(blank(text))
        uses = []
        for i, (k, v, s, e) in enumerate(toks):
            if k == "id" and v in sums and i + 1 < len(toks) and toks[i + 1][1] == "(":
                j = matching(toks, i + 1, 1)
                args, depth, start = [], 0, i + 2
                for t in range(i + 2, j + 1):
                    w = toks[t][1]
                    if w in "([{" and toks[t][0] == "p":
                        depth += 1
                    elif w in ")]}" and toks[t][0] == "p" and t < j:
                        depth -= 1
                    if (w == "," and depth == 0) or t == j:
                        args.append(text[toks[start][2]:toks[t - 1][3]] if t > start else "")
                        start = t + 1
                uses.append((s, toks[j][3], v, args))
        if not uses:
            return text
        top, end = [], -1
        for u in uses:
            if u[0] >= end:
                top.append(u)
                end = u[1]
        for s, e, name, args in reversed(top):
            params, body = macs[name]
            if params == [""]:
                params, args = [], []
            if len(args) != len(params):
                continue
            new = re.sub(r"\b(%s)\b" % "|".join(map(re.escape, params)), lambda m: args[params.index(m.group(1))],
                         body) if params else body
            text = text[:s] + new + "\n" * text.count("\n", s, e) + text[e:]
    return text


class Module:
    def __init__(self, path, types, macs, text=None):
        self.path = Path(path)
        self.text = expand(self.path.read_text(errors="replace") if text is None else text, macs)
        self.toks = tokens(blank(self.text))
        self.types, self.macs = types, macs
        self.routines = routines(self.text)

    def keys(self, cands):
        """{operator token: (routine, n, total)}: each site's + or += is the routine's nth of total that have
        a product on a side, so that the lists keep through edits elsewhere and a changed routine is seen."""
        ops = sorted({c[0] for c in cands if c[1]})
        names = {i: routine_of(self.routines, self.toks[i][2]) for i in ops}
        total = {}
        for name in names.values():
            total[name] = total.get(name, 0) + 1
        out, seen = {}, {}
        for i in ops:
            seen[names[i]] = seen.get(names[i], 0) + 1
            out[i] = (names[i], seen[names[i]], total[names[i]])
        return out

    def span(self, n):
        return self.toks[n.lo][2], self.toks[n.hi][3]

    def src(self, n):
        s, e = self.span(n)
        return self.text[s:e]

    def product(self, n):
        """(a, b) as text when n is a product, or a macro whose body is one."""
        n = strip(n)
        if n.op == "*":
            return self.src(n.kids[0]), self.src(n.kids[1])
        if n.op == "call" and self.toks[n.kids[0].lo][1] in self.macs and n.kids[0].lo == n.kids[0].hi:
            params, body = self.macs[self.toks[n.kids[0].lo][1]]
            args = [self.src(a) for a in n.kids[1:]]
            if len(args) != len(params):
                return None
            text = re.sub(r"\b(%s)\b" % "|".join(map(re.escape, params)),
                          lambda m: "( " + args[params.index(m.group(1))] + " )", body) if params != [""] else body
            try:
                sub = Module.__new__(Module)
                sub.text, sub.toks, sub.types, sub.macs = text, tokens(text), self.types, self.macs
                p = Parser(sub.toks, 0, len(sub.toks), self.types)
                return sub.product(p.expr())
            except SyntaxError:
                return None
        return None

    def candidates(self):
        """Each + or += with a product on one side: (op token, side, a, b, c, node)."""
        out = []
        for i, (k, v, s, e) in enumerate(self.toks):
            if k != "p" or v not in ("+", "+="):
                continue
            lo, hi = region(self.toks, i)
            # a declaration's initialiser
            j = lo
            while j < i and (self.toks[j][1] in TYPE_WORDS or self.toks[j][1] in self.types or self.toks[j][1] == "*"):
                j += 1
            if j > lo and j < i and self.toks[j][0] == "id" and self.toks[j + 1][1] == "=":
                lo = j + 2
            node = None
            try:
                p = Parser(self.toks, lo, hi, self.types)
                root = p.expr()
                if p.i == hi:
                    node = find(root, i)
            except SyntaxError:
                pass
            if node is None:
                out.append((i, None, None, None, None, None))
                continue
            left, right = node.kids
            for side, prod, other in (("r", right, left), ("l", left, right)):
                if v == "+=" and side == "l":
                    continue
                ab = self.product(prod)
                if ab:
                    out.append((i, side, ab[0], ab[1], self.src(other), node))
        return out

    def line_col(self, i):
        s = self.toks[i][2]
        return self.text.count("\n", 0, s) + 1, s - self.text.rfind("\n", 0, s)

    def render(self, edits):
        """The text with nodes replaced: edits maps a node to a function of (render of a span) -> text."""
        spans = sorted(((self.span(n), k, f) for k, (n, f) in enumerate(edits)), key=lambda x: (x[0][0], -x[0][1]))

        def go(s, e, above=()):
            out, pos = [], s
            top = []
            for x in spans:
                if s <= x[0][0] and x[0][1] <= e and x[1] not in above and (not top or x[0][0] >= top[-1][0][1]):
                    top.append(x)
            for (a, b), k, f in top:
                out.append(self.text[pos:a])
                out.append(f(lambda s, e, k=k: go(s, e, above + (k,))))
                pos = b
            out.append(self.text[pos:e])
            return "".join(out)

        return go(0, len(self.text))


def find(n, i):
    if n.tok == i:
        return n
    for k in n.kids:
        if k.lo <= i <= k.hi:
            r = find(k, i)
            if r:
                return r
    return None


def pairs(cands):
    """The sites with a product on each side: op token -> (left site, right site)."""
    by = {}
    for site in cands:
        by.setdefault(site[0], {})[site[1]] = site
    return {i: (d["l"], d["r"]) for i, d in by.items() if len(d) == 2}


def depth(n):
    """How deep the arithmetic in a node goes (loads, members and casts not counted)."""
    n = strip(n)
    below = max((depth(k) for k in n.kids), default=0)
    return below + (1 if n.op in BINARY or n.op in ("u-", "u~", "call", "?") else 0)


def fmac_edit(mod, site):
    """The edit that writes one site as FMAC( a, b, c )."""
    i, side, a, b, c, node = site
    left, right = node.kids
    prod, other = (right, left) if side == "r" else (left, right)

    def text(go):
        pn = strip(prod)
        if pn.op == "*":
            ta, tb = go(*mod.span(pn.kids[0])), go(*mod.span(pn.kids[1]))
        else:
            ta, tb = a, b
        tc = go(*mod.span(other))
        if node.op == "+=":
            return "%s = FMAC( %s, %s, %s )" % (tc, ta, tb, tc)
        return "FMAC( %s, %s, %s )" % (ta, tb, tc)

    return node, text


def block_edit(mod, site):
    """The edit that hides one site's product in a call, so SHC cannot fuse it."""
    i, side, a, b, c, node = site
    prod = node.kids[1] if side == "r" else node.kids[0]
    return prod, lambda go: "fmac_block( %s )" % go(*mod.span(prod))


# --- finding the sites with SHC --------------------------------------------------------------

def shc_count(decomp, text, name, where):
    """FMACs in a module compiled from text, beside the module, by the decomp's SHC wrapper (its options
    and headers, and the program PROGRAM names)."""
    src = where / (name + ".c")
    src.write_text(text)
    try:
        r = subprocess.run([sys.executable, "-I", str(decomp / "tools/shc.py"), str(src)], capture_output=True,
                           text=True, cwd=decomp)
        obj = decomp / "build/obj" / os.environ.get("PROGRAM", "")
        lis = obj / (name + ".lis")
        if r.returncode or not lis.exists():
            return None
        n = sum(1 for l in lis.read_text(errors="replace").splitlines() if re.search(r"\bFMAC\b", l))
        for ext in (".lis", ".obj", ".o"):
            (obj / (name + ext)).unlink(missing_ok=True)
        return n
    finally:
        src.unlink(missing_ok=True)


def find_sites(decomp, srcs, out):
    decomp = Path(decomp)
    inc = [p.read_text(errors="replace") for p in (decomp / "include").glob("*.h")]
    types, macs = typedefs(inc), macros(inc)
    jobs, mods, unread = [], {}, []
    decl = "extern float fmac_block( float ) ;\n"
    for path in srcs:
        own = Path(path).read_text(errors="replace")
        mod = Module(path, types | typedefs([own]), {**macs, **macros([own])})
        allc = mod.candidates()
        for c in allc:
            if not c[1] and c[2] is None and c[5] is None:
                line, col = mod.line_col(c[0])
                unread.append("%s:%d:%d" % (mod.path.stem, line, col))
        cands = [c for c in allc if c[1]]
        if not cands:
            jobs.append((mod.path.stem, -1, mod.text))
            mods[mod.path.stem] = (mod, [])
            continue
        mods[mod.path.stem] = (mod, cands)
        jobs.append((mod.path.stem, -1, decl + mod.text))
        for k, site in enumerate(cands):
            jobs.append((mod.path.stem, k, decl + mod.render([block_edit(mod, site)])))
        for i, two in pairs(cands).items():
            jobs.append((mod.path.stem, ("b", i), decl + mod.render([block_edit(mod, s) for s in two])))

    def run(job):
        stem, k, text = job
        tag = "%s%d" % (k[0], k[1]) if isinstance(k, tuple) else "%d" % (k + 1)
        return stem, k, shc_count(decomp, text, "zzfm_%s_%s" % (stem, tag), mods[stem][0].path.parent)

    counts = {}
    with ThreadPoolExecutor(int(os.environ.get("JOBS", "16"))) as ex:
        for stem, k, n in ex.map(run, jobs):
            counts[stem, k] = n
        # sites SHC merged (the same sum at the end of two arms): one still fuses while the other is
        # hidden, so hide all such sites of the module, then all but one
        jobs, still = [], {}
        for stem, (mod, cands) in mods.items():
            base, two = counts[stem, -1], pairs(cands)
            if base is None:
                continue
            same = [k for k, s in enumerate(cands) if counts[stem, k] == base and s[0] not in two]
            got = sum(1 for k in range(len(cands)) if counts[stem, k] == base - 1)
            if same and got < base:
                still[stem] = same
                jobs.append((stem, ("a", 0), decl + mod.render([block_edit(mod, cands[k]) for k in same])))
                for k in same:
                    jobs.append((stem, ("a", k + 1), decl + mod.render([block_edit(mod, cands[j])
                                                                        for j in same if j != k])))
        for stem, k, n in ex.map(run, jobs):
            counts[stem, k] = n
    rows, odd = [], []
    for stem, (mod, cands) in sorted(mods.items()):
        base = counts[stem, -1]
        got, two = 0, pairs(cands)
        keys = mod.keys(cands)
        for k in still.get(stem, []):
            n = counts[stem, ("a", k + 1)]
            if n is not None and counts[stem, ("a", 0)] is not None and n == counts[stem, ("a", 0)] + 1:
                rows.append("%s\t%s\tmerged" % (stem, key_text(keys[cands[k][0]], cands[k][1])))
                got = base - counts[stem, ("a", 0)]
        for k, site in enumerate(cands):
            n = counts[stem, k]
            line, col = mod.line_col(site[0])
            if n is None or base is None:
                odd.append("%s:%d:%d does not compile" % (stem, line, col))
            elif n == base - 1:
                rows.append("%s\t%s" % (stem, key_text(keys[site[0]], site[1])))
                got += 1
            elif n != base and site[0] not in two:
                odd.append("%s:%d:%d changes the count %d -> %d" % (stem, line, col, base, n))
        for i, (l, r) in two.items():
            n = counts[stem, ("b", i)]
            singles = [counts[stem, cands.index(s)] for s in (l, r)]
            if base is not None and n == base - 1 and base - 1 not in singles:
                # a product on each side: SHC fuses the one with more arithmetic in it, the left on a tie
                node = l[5]
                side = "r" if depth(node.kids[1]) > depth(node.kids[0]) else "l"
                rows.append("%s\t%s\tboth" % (stem, key_text(keys[i], side)))
                got += 1
        if got != base:
            odd.append("%s: SHC makes %s FMACs, %d sites found" % (stem, base, got))
    Path(out).write_text("".join(r + "\n" for r in rows), newline="\n")
    for o in odd:
        print(o)
    print("%d + or += the parser could not read: %s" % (len(unread), " ".join(unread[:20])))
    print("%d sites fused in %d modules" % (len(rows), len({r.split()[0] for r in rows})))


def arm_edits(text, stem, edits):
    """The text with each statement the edits files list (module, routine, the statement, what it becomes, how
    many times the routine has it) changed; an edit that no longer finds its statements stops."""
    rows = [l.split("\t") for e in edits for l in Path(e).read_text().splitlines()]
    for _, name, old, new, count in (r for r in rows if r[0] == stem):
        spans = [(s, e) for n, s, e in routines(text) if n == name]   # one, or one in each program's arm
        got = sum(text[s:e].count(old) for s, e in spans)
        if got != int(count):
            sys.exit("%s: %s has %r %d times, not %s" % (stem, name, old, got, count))
        for s, e in reversed(spans):
            text = text[:s] + text[s:e].replace(old, new) + text[e:]
    return text


def rewrite(sites, src, out, edits=()):
    """A host copy of src with its fused sums as FMAC( a, b, c ), and the statements edits lists changed
    (such as, on an ARM host, each float made unsigned through sh_u32)."""
    want = {}
    for l in Path(sites).read_text().splitlines():
        stem, name, nth, side = l.split("\t")[:4]
        n, total = map(int, nth.split("/"))
        want.setdefault(stem, set()).add((name, n, total, side))
    path = Path(src)
    mine = want.get(path.stem, set())
    text = arm_edits(path.read_text(errors="replace"), path.stem, edits)
    if mine:
        root = next(d for d in path.resolve().parents if (d / "include").is_dir())
        inc = [p.read_text(errors="replace") for p in (root / "include").glob("*.h")]
        mod = Module(path, typedefs(inc + [text]), {**macros(inc), **macros([text])}, text)
        cands = mod.candidates()
        keys = mod.keys(cands)
        found, seen = [], set()
        for site in cands:
            if site[1] and keys[site[0]] + (site[1],) in mine:
                found.append(fmac_edit(mod, site))
                seen.add(keys[site[0]] + (site[1],))
        if seen != mine:
            sys.exit("%s: these sums are not where SHC fused them (a changed routine: run fmac.py find on it): %s"
                     % (src, sorted(mine - seen)))
        text = mod.render(found)
    Path(out).write_text('#line 1 "%s"\n' % path.as_posix() + text)


if __name__ == "__main__":
    if sys.argv[1] == "find":
        decomp = Path(sys.argv[2])
        srcs = sorted((decomp / "src").glob("*.c"))
        if os.environ.get("PROGRAM") == "dc":   # src/dc's modules in place of their namesakes, but not its start-up
            dc = [p for p in sorted((decomp / "src/dc").glob("*.c")) if p.name != "strt2.c"]
            srcs = [p for p in srcs if p.name not in {d.name for d in dc}] + dc
        find_sites(decomp, sys.argv[4:] or srcs, sys.argv[3])
    elif sys.argv[1] == "rewrite":
        rewrite(*sys.argv[2:5], sys.argv[5:])
