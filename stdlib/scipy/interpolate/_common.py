# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# Shared numeric kernels for scipy.interpolate: interval search, de Boor
# recursion, banded LU with partial pivoting and output reshaping.

import numpy as np
import internal.static as static

NAN = float('nan')

def _ext_mode(e, default: int) -> int:
    # 0: no extrapolation, 1: extrapolate, 2: periodic
    if e is None:
        return default
    elif isinstance(e, str):
        if e == 'periodic':
            return 2
        raise ValueError(f"extrapolate={e} is not allowed")
    else:
        return 1 if bool(e) else 0

def _fmod_py(a: float, b: float) -> float:
    # numpy.remainder semantics (sign of divisor)
    m = _fmod(a, b)
    if m != 0.0 and ((b < 0.0) != (m < 0.0)):
        m += b
    return m

def _fmod(a: float, b: float) -> float:
    import math
    return math.fmod(a, b)

def _searchsorted(a, v: float, right: bool = False) -> int:
    n = len(a)
    if v != v:
        return n
    lo = 0
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        am = a[mid]
        if (am <= v) if right else (am < v):
            lo = mid + 1
        else:
            hi = mid
    return lo

def _pp_find(x, xval: float, ext: bool) -> int:
    # interval index i with x[i] <= xval < x[i+1]; -1 when out of range
    nx = len(x)
    a = x[0]
    b = x[nx - 1]
    if not (a <= xval and xval <= b):
        if xval < a and ext:
            return 0
        elif xval > b and ext:
            return nx - 2
        return -1
    if xval == b:
        return nx - 2
    lo = 0
    hi = nx - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xval < x[mid]:
            hi = mid
        else:
            lo = mid
    return lo

def _bspl_find(t, k: int, xval: float, prev_l: int, ext: bool) -> int:
    n = len(t) - k - 1
    tb = t[k]
    te = t[n]
    if xval != xval:
        return -1
    if (xval < tb or xval > te) and not ext:
        return -1
    l = prev_l if (k < prev_l and prev_l < n) else k
    while xval < t[l] and l != k:
        l -= 1
    l += 1
    while xval >= t[l] and l != n:
        l += 1
    return l - 1

def _deboor(t, x: float, k: int, ell: int, m: int) -> List[float]:
    # values (or m-th derivatives) of the k+1 nonzero B-splines on [t[ell], t[ell+1])
    h = [0.0] * (k + 1)
    hh = [0.0] * (k + 1)
    h[0] = 1.0
    for j in range(1, k - m + 1):
        for q in range(j):
            hh[q] = h[q]
        h[0] = 0.0
        for n in range(1, j + 1):
            ind = ell + n
            xb = t[ind]
            xa = t[ind - j]
            if xb == xa:
                h[n] = 0.0
                continue
            w = hh[n - 1] / (xb - xa)
            h[n - 1] += w * (xb - x)
            h[n] = w * (x - xa)
    for j in range(k - m + 1, k + 1):
        if j < 1:
            continue
        for q in range(j):
            hh[q] = h[q]
        h[0] = 0.0
        for n in range(1, j + 1):
            ind = ell + n
            xb = t[ind]
            xa = t[ind - j]
            if xb == xa:
                h[m] = 0.0
                continue
            w = j * hh[n - 1] / (xb - xa)
            h[n - 1] -= w
            h[n] = w
    return h

def _band_new(n: int, kl: int, ku: int):
    return np.zeros((n, 2 * kl + ku + 1))

def _band_set(ab, kl: int, i: int, c: int, v: float):
    ab[i, c - i + kl] = v

def _band_solve(ab, n: int, kl: int, ku: int, B):
    # Solves A X = B in place (B: (n, R)); A in row-band storage, gbsv-like pivoting.
    R = B.shape[1]
    for j in range(n):
        p = j
        best = abs(ab[j, kl])
        for i in range(j + 1, min(n, j + kl + 1)):
            v = abs(ab[i, j - i + kl])
            if v > best:
                best = v
                p = i
        if best == 0.0:
            raise ValueError("Colocation matrix is singular.")
        cend = min(n, j + kl + ku + 1)
        if p != j:
            for c in range(j, cend):
                tmp = ab[j, c - j + kl]
                ab[j, c - j + kl] = ab[p, c - p + kl]
                ab[p, c - p + kl] = tmp
            for r in range(R):
                tmp = B[j, r]
                B[j, r] = B[p, r]
                B[p, r] = tmp
        piv = ab[j, kl]
        for i in range(j + 1, min(n, j + kl + 1)):
            f = ab[i, j - i + kl] / piv
            if f != 0.0:
                ab[i, j - i + kl] = 0.0
                for c in range(j + 1, cend):
                    ab[i, c - i + kl] -= f * ab[j, c - j + kl]
                for r in range(R):
                    B[i, r] -= f * B[j, r]
    for j in range(n - 1, -1, -1):
        cend = min(n, j + kl + ku + 1)
        for r in range(R):
            s = B[j, r]
            for c in range(j + 1, cend):
                s -= ab[j, c - j + kl] * B[c, r]
            B[j, r] = s / ab[j, kl]
    return B

def _finish(out, xshape, tsh, axis: int):
    # out: (N, R) -> xshape + tsh, with x dims moved to position `axis`
    r = out.reshape(xshape + tsh)
    if static.len(r.shape) == 0:
        return r.item()
    else:
        if axis != 0:
            xnd = static.len(xshape)
            l = [i for i in range(r.ndim)]
            l = l[xnd:xnd + axis] + l[:xnd] + l[xnd + axis:]
            return r.transpose(l)
        return r

def _vals(v, R: int) -> List[float]:
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.size == 1:
        return [float(a[0])] * R
    if a.size != R:
        raise ValueError("shape mismatch: value array could not be broadcast")
    return [float(a[i]) for i in range(R)]

def _poch(a: int, m: int) -> float:
    r = 1.0
    for i in range(m):
        r *= float(a + i)
    return r
