# Copyright (C) 2026 the Typhon authors
# Ported from pandas 3.1.1 (BSD-3-Clause)

# Numeric kernels on List[float]: skew/kurt (pandas/_libs/include/pandas/simd/
# moments_common.hpp + algos.pxd), quantile interpolation modes (numpy
# percentile methods as used by nanops.nanpercentile), mode, linear
# interpolation (missing.py), and the cut/qcut bin helpers (reshape/tile.py).

import math
from .core import npsum

NAN = float("nan")

def _acc_mean(v: List[float], lo: int, hi: int, skipna: bool):
    # returns (sum, count, ok) with the leaf-128 recursive halving of the scalar kernel
    if hi - lo <= 128:
        s = 0.0
        c = 0
        for i in range(lo, hi):
            x = v[i]
            if x != x:
                if not skipna:
                    return 0.0, 0, False
                continue
            s += x
            c += 1
        return s, c, True
    mid = lo + (hi - lo) // 2
    s1, c1, o1 = _acc_mean(v, lo, mid, skipna)
    if not o1:
        return 0.0, 0, False
    s2, c2, o2 = _acc_mean(v, mid, hi, skipna)
    if not o2:
        return 0.0, 0, False
    return s1 + s2, c1 + c2, True

def _acc_diffs(v: List[float], lo: int, hi: int, mean: float, mm: int):
    if hi - lo <= 128:
        m1 = 0.0
        m2 = 0.0
        m3 = 0.0
        m4 = 0.0
        for i in range(lo, hi):
            x = v[i]
            if x != x:
                continue
            d = x - mean
            d2 = d * d
            m1 += d
            m2 += d2
            if mm >= 3:
                m3 += d2 * d
            if mm >= 4:
                m4 += d2 * d2
        return m1, m2, m3, m4
    mid = lo + (hi - lo) // 2
    a = _acc_diffs(v, lo, mid, mean, mm)
    b = _acc_diffs(v, mid, hi, mean, mm)
    return a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]

def _moments(v: List[float], skipna: bool, mm: int):
    # returns (n, m2, m3, m4, nan_flag)
    s, c, ok = _acc_mean(v, 0, len(v), skipna)
    if not ok:
        return len(v), NAN, NAN, NAN, True
    if c == 0:
        return 0, 0.0, 0.0, 0.0, False
    cd = float(c)
    trial = s / cd
    m1, d2, d3, d4 = _acc_diffs(v, 0, len(v), trial, mm)
    corr = m1 / cd
    corr2 = corr * corr
    term1 = m1 * corr
    m2 = max(d2 - term1, 0.0)
    m3 = 0.0
    m4 = 0.0
    if mm >= 3:
        m3 = d3 - (3.0 * d2 * corr) + (2.0 * corr * term1)
    if mm >= 4:
        m4 = d4 - (4.0 * d3 * corr) + (6.0 * d2 * corr2) - (3.0 * term1 * corr2)
        m4 = max(m4, 0.0)
    return c, m2, m3, m4, False

def nanskew_list(v: List[float], skipna: bool = True) -> float:
    n, m2, m3, m4, bad = _moments(v, skipna, 3)
    if bad or n < 3:
        return NAN
    dn = float(n)
    if m2 == 0.0:
        return NAN
    ratio = m3 / (m2 * math.sqrt(m2))
    return ratio * ((dn * math.sqrt(dn - 1.0)) / (dn - 2.0))

def nankurt_list(v: List[float], skipna: bool = True) -> float:
    n, m2, m3, m4, bad = _moments(v, skipna, 4)
    if bad or n < 4:
        return NAN
    dn = float(n)
    if m2 == 0.0:
        return NAN
    ratio = m4 / (m2 * m2)
    inner = dn * (dn + 1.0) * ratio - 3.0 * (dn - 1.0)
    return ((dn - 1.0) / ((dn - 2.0) * (dn - 3.0))) * inner

def _rint(x: float) -> float:
    # round half to even
    f = math.floor(x)
    d = x - f
    if d < 0.5:
        return f
    if d > 0.5:
        return f + 1.0
    return f if math.fmod(f, 2.0) == 0.0 else f + 1.0

def quantile_sorted(a: List[float], q: float, interpolation: str = "linear") -> float:
    n = len(a)
    if n == 0:
        return NAN
    vi = float(n - 1) * q
    if interpolation == "nearest":
        i = int(_rint(vi))
        if i < 0:
            i = 0
        if i > n - 1:
            i = n - 1
        return a[i]
    p = int(math.floor(vi))
    if p < 0:
        p = 0
    nx = p + 1
    if nx > n - 1:
        nx = n - 1
    if p > n - 1:
        p = n - 1
    g = vi - float(p)
    A = a[p]
    B = a[nx]
    if interpolation == "lower":
        return A
    if interpolation == "higher":
        return A if g == 0.0 else B
    if interpolation == "midpoint":
        return A if g == 0.0 else (A + B) / 2.0
    if interpolation != "linear":
        raise ValueError(f"interpolation must be one of linear, lower, higher, midpoint, nearest, got {interpolation}")
    d = B - A
    if g >= 0.5:
        return B - d * (1.0 - g)
    return A + d * g

def mode_values(v: List[float]) -> List[float]:
    # sorted modes, NaN dropped
    vals = [x for x in v if x == x]
    vals.sort()
    out = List[float]()
    best = 0
    i = 0
    n = len(vals)
    while i < n:
        j = i
        while j + 1 < n and vals[j + 1] == vals[i]:
            j += 1
        c = j - i + 1
        if c > best:
            best = c
            out = [vals[i]]
        elif c == best:
            out.append(vals[i])
        i = j + 1
    return out

def interpolate_linear(v: List[float], limit_direction: str = "forward",
                       limit: int = -1, limit_area: str = "") -> List[float]:
    # pandas _interpolate_1d(method='linear') on a positional index
    n = len(v)
    out = [x for x in v]
    valid = [i for i in range(n) if v[i] == v[i]]
    if len(valid) == 0:
        return out
    first = valid[0]
    last = valid[len(valid) - 1]
    for i in range(n):
        if v[i] == v[i]:
            continue
        if i < first:
            # leading NaNs are only filled with limit_direction backward/both
            if limit_direction == "forward" or limit_area == "inside":
                continue
            out[i] = v[first]
            continue
        if i > last:
            if limit_direction == "backward" or limit_area == "inside":
                continue
            out[i] = v[last]
            continue
        # bracket
        lo = i - 1
        while lo >= 0 and v[lo] != v[lo]:
            lo -= 1
        hi = i + 1
        while hi < n and v[hi] != v[hi]:
            hi += 1
        if limit_direction == "backward":
            pass
        x0 = float(lo)
        x1 = float(hi)
        y0 = v[lo]
        y1 = v[hi]
        slope = (y1 - y0) / (x1 - x0)
        out[i] = y0 + slope * (float(i) - x0)
    if limit >= 0:
        # enforce limit on runs filled from the forward direction
        run = 0
        for i in range(n):
            if v[i] != v[i]:
                run += 1
                if run > limit and (limit_direction == "forward" or limit_direction == "both"):
                    out[i] = NAN
            else:
                run = 0
    return out

# -- cut / qcut helpers -------------------------------------------------

def _around(x: float, digits: int) -> float:
    if digits >= 0:
        p = 10.0 ** digits
        return _rint(x * p) / p
    p = 10.0 ** (-digits)
    return _rint(x / p) * p

def round_frac(x: float, precision: int) -> float:
    if x != x or x == float("inf") or x == -float("inf") or x == 0.0:
        return x
    whole = float(int(x)) if x >= 0 else -float(int(-x))
    frac = x - whole
    if whole == 0.0:
        digits = -int(math.floor(math.log10(abs(frac)))) - 1 + precision
    else:
        digits = precision
    return _around(x, digits)

def infer_precision(base: int, bins: List[float]) -> int:
    for precision in range(base, 20):
        lv = [round_frac(b, precision) for b in bins]
        u = Set[float]()
        for z in lv:
            u.add(z)
        if len(u) == len(bins):
            return precision
    return base

def linspace_list(lo: float, hi: float, num: int) -> List[float]:
    # numpy.linspace(endpoint=True)
    out = List[float](num)
    if num == 1:
        out.append(lo)
        return out
    div = float(num - 1)
    delta = hi - lo
    step = delta / div
    for i in range(num):
        out.append(float(i) * step + lo if step != 0.0 else (float(i) / div) * delta + lo)
    out[num - 1] = hi
    return out
