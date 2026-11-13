# Copyright (C) 2026 the Typhon authors
# Ported from pandas 3.1.1 (BSD-3-Clause)

# Column kernels behind Series.str. Missing strings (NA_STR) propagate as
# missing values; boolean predicates give False for missing entries, as the
# pandas 3 default string dtype does.

import re
from .scalar import K_FLOAT, K_INT, K_BOOL, K_STR, NA_STR
from .core import Column

NAN = float("nan")

def _isna(c: Column, k: int) -> bool:
    return c.s[k] == NA_STR

def smap(c: Column, f) -> Column:
    out = List[str](len(c.s))
    for k in range(len(c.s)):
        out.append(NA_STR if c.s[k] == NA_STR else f(c.s[k]))
    return Column.strs(out)

def smap_int(c: Column, f) -> Column:
    # integer result, float64 with NaN when any entry is missing
    has_na = False
    for k in range(len(c.s)):
        if c.s[k] == NA_STR:
            has_na = True
    if has_na:
        out = List[float](len(c.s))
        for k in range(len(c.s)):
            out.append(NAN if c.s[k] == NA_STR else float(f(c.s[k])))
        return Column.floats(out)
    return Column.ints([f(x) for x in c.s])

def smap_bool(c: Column, f) -> Column:
    out = List[bool](len(c.s))
    for k in range(len(c.s)):
        out.append(False if c.s[k] == NA_STR else f(c.s[k]))
    return Column.bools(out)

def _strip_chars(s: str, chars: Optional[str], left: bool, right: bool) -> str:
    if chars is None:
        if left and right:
            return s.strip()
        return s.lstrip() if left else s.rstrip()
    ch = chars.__val__()
    if left and right:
        return s.strip(ch)
    return s.lstrip(ch) if left else s.rstrip(ch)

def strip(c: Column, chars: Optional[str], left: bool, right: bool) -> Column:
    out = List[str](len(c.s))
    for k in range(len(c.s)):
        out.append(NA_STR if c.s[k] == NA_STR else _strip_chars(c.s[k], chars, left, right))
    return Column.strs(out)

def _pat(pat: str, case: bool, regex: bool):
    p = pat if regex else re.escape(pat)
    return re.compile(p, 0 if case else re.IGNORECASE)

def contains(c: Column, pat: str, case: bool, regex: bool, na: Optional[bool]) -> Column:
    rx = _pat(pat, case, regex)
    out = List[bool](len(c.s))
    for k in range(len(c.s)):
        if c.s[k] == NA_STR:
            out.append(False if na is None else na.__val__())
        else:
            out.append(rx.search(c.s[k]) is not None)
    return Column.bools(out)

def match(c: Column, pat: str, case: bool, full: bool) -> Column:
    rx = _pat(pat, case, True)
    out = List[bool](len(c.s))
    for k in range(len(c.s)):
        if c.s[k] == NA_STR:
            out.append(False)
        elif full:
            out.append(rx.fullmatch(c.s[k]) is not None)
        else:
            out.append(rx.match(c.s[k]) is not None)
    return Column.bools(out)

def startswith(c: Column, pat: str, end: bool) -> Column:
    out = List[bool](len(c.s))
    for k in range(len(c.s)):
        if c.s[k] == NA_STR:
            out.append(False)
        else:
            out.append(c.s[k].endswith(pat) if end else c.s[k].startswith(pat))
    return Column.bools(out)

def replace(c: Column, pat: str, repl: str, n: int, case: bool, regex: bool) -> Column:
    out = List[str](len(c.s))
    if regex:
        rx = _pat(pat, case, True)
        for k in range(len(c.s)):
            if c.s[k] == NA_STR:
                out.append(NA_STR)
            else:
                out.append(rx.sub(repl, c.s[k], 0 if n < 0 else n))
    elif case:
        for k in range(len(c.s)):
            if c.s[k] == NA_STR:
                out.append(NA_STR)
            elif n < 0:
                out.append(c.s[k].replace(pat, repl))
            else:
                out.append(c.s[k].replace(pat, repl, n))
    else:
        rx = _pat(pat, False, False)
        for k in range(len(c.s)):
            if c.s[k] == NA_STR:
                out.append(NA_STR)
            else:
                out.append(rx.sub(repl.replace("\\", "\\\\"), c.s[k], 0 if n < 0 else n))
    return Column.strs(out)

def find(c: Column, sub: str, right: bool) -> Column:
    return smap_int(c, lambda x: x.rfind(sub) if right else x.find(sub))

def count(c: Column, pat: str, regex_flags: int) -> Column:
    rx = re.compile(pat, regex_flags)
    return smap_int(c, lambda x: len(rx.findall(x)))

def slice_(c: Column, start: Optional[int], stop: Optional[int], step: Optional[int]) -> Column:
    out = List[str](len(c.s))
    for k in range(len(c.s)):
        if c.s[k] == NA_STR:
            out.append(NA_STR)
        else:
            out.append(c.s[k][start:stop:step] if step is not None else c.s[k][start:stop])
    return Column.strs(out)

def get(c: Column, i: int) -> Column:
    out = List[str](len(c.s))
    for k in range(len(c.s)):
        x = c.s[k]
        n = len(x)
        if x == NA_STR:
            out.append(NA_STR)
        elif i < 0 and -i <= n:
            out.append(x[n + i])
        elif i >= 0 and i < n:
            out.append(x[i])
        else:
            out.append(NA_STR)
    return Column.strs(out)

def pad(c: Column, width: int, side: str, fill: str) -> Column:
    out = List[str](len(c.s))
    for k in range(len(c.s)):
        x = c.s[k]
        if x == NA_STR:
            out.append(NA_STR)
        elif side == "left":
            out.append(x.rjust(width, fill))
        elif side == "right":
            out.append(x.ljust(width, fill))
        elif len(x) >= width:
            out.append(x)
        else:
            # same split as CPython: the extra pad goes left when width is odd
            marg = width - len(x)
            left = marg // 2 + (marg & width & 1)
            out.append(fill * left + x + fill * (marg - left))
    return Column.strs(out)

def zfill(c: Column, width: int) -> Column:
    return smap(c, lambda x: x.zfill(width))

def repeat(c: Column, n: int) -> Column:
    return smap(c, lambda x: x * n)

def cat_all(c: Column, sep: str, na_rep: Optional[str]) -> str:
    parts = List[str]()
    for k in range(len(c.s)):
        if c.s[k] == NA_STR:
            if na_rep is not None:
                parts.append(na_rep.__val__())
        else:
            parts.append(c.s[k])
    return sep.join(parts)

def cat_other(c: Column, o: Column, sep: str, na_rep: Optional[str]) -> Column:
    out = List[str](len(c.s))
    for k in range(len(c.s)):
        a = c.s[k]
        b = o.s[k]
        if a == NA_STR and na_rep is not None:
            a = na_rep.__val__()
        if b == NA_STR and na_rep is not None:
            b = na_rep.__val__()
        if a == NA_STR or b == NA_STR:
            out.append(NA_STR)
        else:
            out.append(a + sep + b)
    return Column.strs(out)

def _ws_split(x: str, n: int, right: bool) -> List[str]:
    allp = x.split()
    if n >= len(allp) - 1:
        return allp
    out = List[str]()
    if not right:
        # first n pieces, remainder (left-stripped) as the last piece
        rest = x.lstrip()
        for _ in range(n):
            i = 0
            while i < len(rest) and not rest[i].isspace():
                i += 1
            out.append(rest[:i])
            rest = rest[i:].lstrip()
        out.append(rest)
    else:
        rest = x.rstrip()
        tail = List[str]()
        for _ in range(n):
            i = len(rest)
            while i > 0 and not rest[i - 1].isspace():
                i -= 1
            tail.append(rest[i:])
            rest = rest[:i].rstrip()
        out.append(rest)
        for k in range(len(tail) - 1, -1, -1):
            out.append(tail[k])
    return out

def _split_one(x: str, pat: Optional[str], n: int, right: bool, regex: bool) -> List[str]:
    if pat is None:
        if n < 0:
            return x.split()
        return _ws_split(x, n, right)
    p = pat.__val__()
    if regex and len(p) > 1:
        rx = re.compile(p)
        return [(q.__val__() if q is not None else "") for q in rx.split(x, 0 if n < 0 else n)]
    if n < 0:
        return x.split(p)
    return x.rsplit(p, n) if right else x.split(p, n)

def split_lists(c: Column, pat: Optional[str], n: int, right: bool, regex: bool) -> List[Optional[List[str]]]:
    out = List[Optional[List[str]]](len(c.s))
    for k in range(len(c.s)):
        if c.s[k] == NA_STR:
            out.append(None)
        else:
            out.append(_split_one(c.s[k], pat, n, right, regex))
    return out

def split_columns(parts: List[Optional[List[str]]]) -> List[Column]:
    w = 0
    for p in parts:
        if p is not None:
            w = max(w, len(p.__val__()))
    cols = List[Column]()
    for j in range(w):
        v = List[str](len(parts))
        for p in parts:
            if p is None:
                v.append(NA_STR)
            else:
                q = p.__val__()
                v.append(q[j] if j < len(q) else NA_STR)
        cols.append(Column.strs(v))
    return cols

def extract(c: Column, pat: str) -> List[Column]:
    rx = re.compile(pat)
    g = rx.groups
    cols = List[List[str]]()
    for j in range(g):
        cols.append(List[str](len(c.s)))
    for k in range(len(c.s)):
        m = None if c.s[k] == NA_STR else rx.search(c.s[k])
        for j in range(g):
            if m is None:
                cols[j].append(NA_STR)
            else:
                v = m.__val__().group(j + 1)
                cols[j].append(NA_STR if v is None else v.__val__())
    return [Column.strs(x) for x in cols]

def removeprefix(c: Column, p: str, suffix: bool) -> Column:
    return smap(c, lambda x: x.removesuffix(p) if suffix else x.removeprefix(p))
