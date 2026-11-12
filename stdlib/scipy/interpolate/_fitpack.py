# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# make_interp_spline and the FITPACK-style tck interface (splrep with s=0
# interpolation, splev, splint, splder, splantider).

import numpy as np
import internal.static as static
from _common import NAN, _bspl_find, _deboor, _band_solve, _searchsorted, _vals
from _bspl import BSpline, _bspl_eval, _splder, _splantider, _pad

def _not_a_knot(x, k: int):
    n = x.size
    if k % 2 == 1:
        k2 = (k + 1) // 2
        inner = [x[i] for i in range(k2, n - k2)]
    else:
        k2 = k // 2
        mid = [(x[i + 1] + x[i]) / 2.0 for i in range(n - 1)]
        inner = [mid[i] for i in range(k2, len(mid) - k2)]
    t = np.empty(2 * (k + 1) + len(inner))
    for i in range(k + 1):
        t[i] = x[0]
        t[t.size - 1 - i] = x[n - 1]
    for i in range(len(inner)):
        t[k + 1 + i] = inner[i]
    return t

def _augknt(x, k: int):
    n = x.size
    t = np.empty(n + 2 * k)
    for i in range(k):
        t[i] = x[0]
        t[n + k + i] = x[n - 1]
    for i in range(n):
        t[k + i] = x[i]
    return t

def _periodic_knots(x, k: int):
    n = x.size
    xc = x.copy()
    if k % 2 == 0:
        dx0 = [xc[i + 1] - xc[i] for i in range(n - 1)]
        for i in range(1, n - 1):
            xc[i] -= dx0[i - 1] / 2.0
    dx = [xc[i + 1] - xc[i] for i in range(n - 1)]
    t = np.zeros(n + 2 * k)
    for i in range(n):
        t[k + i] = xc[i]
    m = n - 1
    for i in range(k):
        t[k - i - 1] = t[k - i] - dx[m - 1 - (i % m)]
        t[n + k + i] = t[n + k + i - 1] + dx[i % m]
    return t

def _parse_derivs(d, R: int) -> List[Tuple[int, List[float]]]:
    out = List[Tuple[int, List[float]]]()
    if d is None:
        return out
    elif isinstance(d, str):
        if d == "clamped":
            out.append((1, [0.0] * R))
        elif d == "natural":
            out.append((2, [0.0] * R))
        else:
            raise ValueError(f"Unknown boundary condition : {d}")
    else:
        for item in d:
            out.append((int(item[0]), _vals(item[1], R)))
    return out

def _periodic_coeffs(x, y2, t, k: int):
    # dense solve of the periodic interpolation system
    n = x.size
    R = y2.shape[1]
    N = n + k - 1
    A = np.zeros((N, N))
    for i in range(k - 1):
        bb = _deboor(t, x[0], k, k, i + 1)
        for a in range(k + 1):
            A[i, a] += bb[a]
        bb = _deboor(t, x[n - 1], k, n + k - 1, i + 1)
        for a in range(k):
            A[i, N - k + a] -= bb[a]
    for i in range(n):
        xv = x[i]
        left = k if xv == t[k] else _searchsorted(t, xv) - 1
        bb = _deboor(t, xv, k, left, 0)
        for a in range(k + 1):
            A[i + k - 1, left - k + a] = bb[a]
    b = np.zeros((N, R))
    for i in range(n):
        for j in range(R):
            b[i + k - 1, j] = y2[i, j]
    return np.linalg.solve(A, b)

def make_interp_spline(x, y, k: int = 3, t = None, bc_type = None, axis: int = 0,
                       check_finite: bool = True):
    xa = np.ascontiguousarray(np.asarray(x, dtype=float))
    ya = np.asarray(y, dtype=float)
    if static.len(xa.shape) != 1:
        compile_error("Expect x to be a 1D strictly increasing sequence.")
    ax = axis % ya.ndim
    ym = np.moveaxis(ya, ax, 0)
    tsh = ym.shape[1:]
    n = xa.size
    if n != ym.shape[0]:
        raise ValueError(f'Shapes of x {xa.shape} and y {ya.shape} are incompatible')
    y2 = np.ascontiguousarray(ym).reshape((n, -1)).copy()
    R = y2.shape[1]
    if check_finite:
        for v in xa:
            if not np.isfinite(v):
                raise ValueError("array must not contain infs or NaNs")
        for v in y2.flat:
            if not np.isfinite(v):
                raise ValueError("array must not contain infs or NaNs")
    periodic = False
    dl = List[Tuple[int, List[float]]]()
    dr = List[Tuple[int, List[float]]]()
    if bc_type is None:
        pass
    elif isinstance(bc_type, str):
        if bc_type == 'periodic':
            periodic = True
        elif bc_type != 'not-a-knot':
            dl = _parse_derivs(bc_type, R)
            dr = _parse_derivs(bc_type, R)
    else:
        dl = _parse_derivs(bc_type[0], R)
        dr = _parse_derivs(bc_type[1], R)
    if periodic:
        for j in range(R):
            if abs(y2[0, j] - y2[n - 1, j]) > 1e-15 + 1e-5 * abs(y2[n - 1, j]):
                raise ValueError("First and last points does not match while periodic case expected")
    for i in range(n - 1):
        if xa[i + 1] == xa[i]:
            raise ValueError("Expect x to not have duplicates")
        if xa[i + 1] < xa[i]:
            raise ValueError("Expect x to be a 1D strictly increasing sequence.")
    has_bc = len(dl) > 0 or len(dr) > 0
    if k == 0:
        if (t is not None) or has_bc:
            raise ValueError("Too much info for k=0: t and bc_type can only be None.")
        tt = np.empty(n + 1)
        for i in range(n):
            tt[i] = xa[i]
        tt[n] = xa[n - 1]
        return BSpline(tt, y2.reshape((n,) + tsh), 0, True, 0, _fast=True)._setax(ax)
    if k == 1 and t is None:
        if has_bc:
            raise ValueError("Too much info for k=1: bc_type can only be None.")
        tt = np.empty(n + 2)
        tt[0] = xa[0]
        for i in range(n):
            tt[i + 1] = xa[i]
        tt[n + 1] = xa[n - 1]
        return BSpline(tt, y2.reshape((n,) + tsh), 1, True, 0, _fast=True)._setax(ax)
    if k < 0:
        raise ValueError("Expect non-negative k.")
    if t is None:
        if not has_bc:
            tt = _periodic_knots(xa, k) if periodic else _not_a_knot(xa, k)
        else:
            tt = _augknt(xa, k)
    else:
        if periodic:
            raise NotImplementedError("For periodic case t is constructed automatically and can not be passed manually")
        tt = np.ascontiguousarray(np.asarray(t, dtype=float)).copy()
    for i in range(tt.size - 1):
        if tt[i + 1] < tt[i]:
            raise ValueError("Expect t to be a 1-D sorted array_like.")
    if tt.size < n + k + 1:
        raise ValueError(f"Got {tt.size} knots, need at least {n + k + 1}.")
    if xa[0] < tt[k] or xa[n - 1] > tt[tt.size - k]:
        raise ValueError(f'Out of bounds w/ x = {xa}.')
    if periodic:
        c = _periodic_coeffs(xa, y2, tt, k)
        b = BSpline(tt, c.reshape((c.shape[0],) + tsh), k, True, 0, _fast=True)._setax(ax)
        b._ext = 2
        return b
    nleft = len(dl)
    nright = len(dr)
    for d in dl:
        if not (0 <= d[0] and d[0] <= k):
            raise ValueError(f"Bad boundary conditions at {xa[0]}.")
    for d in dr:
        if not (0 <= d[0] and d[0] <= k):
            raise ValueError(f"Bad boundary conditions at {xa[n - 1]}.")
    nt = tt.size - k - 1
    if nt - n != nleft + nright:
        raise ValueError(f"The number of derivatives at boundaries does not match: expected {nt - n}, got {nleft}+{nright}")
    kl = k
    ku = k
    ab = np.zeros((nt, 2 * kl + ku + 1))
    rhs = np.zeros((nt, R))
    left = k
    for j in range(n):
        xv = xa[j]
        left = _bspl_find(tt, k, xv, left, False)
        if left < 0:
            raise ValueError(f'Out of bounds w/ x = {xa}.')
        bb = _deboor(tt, xv, k, left, 0)
        row = nleft + j
        for a in range(k + 1):
            col = left - k + a
            ab[row, col - row + kl] = bb[a]
        for r in range(R):
            rhs[row, r] = y2[j, r]
    for side in range(2):
        ders = dl if side == 0 else dr
        xv = xa[0] if side == 0 else xa[n - 1]
        offset = 0 if side == 0 else nt - nright
        lf = _bspl_find(tt, k, xv, k, False)
        for q in range(len(ders)):
            nu = ders[q][0]
            wrk = _deboor(tt, xv, k, lf, nu)
            row = offset + q
            for a in range(k + 1):
                col = lf - k + a
                ab[row, col - row + kl] = wrk[a]
            for r in range(R):
                rhs[row, r] = ders[q][1][r]
    c = _band_solve(ab, nt, kl, ku, rhs)
    return BSpline(tt, c.reshape((nt,) + tsh), k, True, 0, _fast=True)._setax(ax)

# ------------------------------------------------------------------ FITPACK API

def splrep(x, y, w = None, xb = None, xe = None, k: int = 3, task: int = 0,
           s = None, t = None, full_output: int = 0, per: int = 0, quiet: int = 1):
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    m = xa.size
    if k < 1 or k > 5:
        raise TypeError(f"Given degree of the spline (k={k}) is not supported. (1<=k<=5)")
    if m <= k:
        raise TypeError("m > k must hold")
    if ya.size != m:
        raise TypeError("Lengths of the first two arguments (x,y) must be equal")
    sv = 0.0
    if s is None:
        sv = 0.0 if w is None else float(m) - (2.0 * float(m)) ** 0.5
    else:
        sv = float(s)
    if sv != 0.0 or per != 0 or t is not None:
        raise NotImplementedError("splrep: only s=0 non-periodic interpolation without user knots is supported")
    for i in range(m - 1):
        if not (xa[i + 1] > xa[i]):
            raise ValueError("Error on input data")
    b = make_interp_spline(xa, ya, k=k)
    tt = b.t
    cc = np.zeros(tt.size)
    for i in range(b.c.size):
        cc[i] = b.c[i]
    return tt, cc, k

def _tck(tck):
    t = np.ascontiguousarray(np.asarray(tck[0], dtype=float))
    c = np.asarray(tck[1], dtype=float)
    k = int(tck[2])
    return t, c.reshape((c.shape[0], 1)), k

def splev(x, tck, der: int = 0, ext: int = 0):
    t, c2, k = _tck(tck)
    if not (0 <= der and der <= k):
        raise ValueError(f"0<=der={der}<=k={k} must hold")
    if ext < 0 or ext > 3:
        raise ValueError(f"ext = {ext} not in (0, 1, 2, 3) ")
    xa = np.asarray(x, dtype=float)
    xs = np.ascontiguousarray(xa.reshape(-1)).copy()
    n = t.size
    tb = t[k]
    te = t[n - k - 1]
    out = np.empty(xs.size)
    for i in range(xs.size):
        xv = xs[i]
        if xv < tb or xv > te:
            if ext == 1:
                out[i] = 0.0
                continue
            elif ext == 2:
                raise ValueError("Found x value not in the domain")
            elif ext == 3:
                xv = tb if xv < tb else te
        # FITPACK splev/splder: interval search with l such that t[l] <= x < t[l+1], k <= l <= n-k-2
        l = k
        while l < n - k - 2 and xv >= t[l + 1]:
            l += 1
        while l > k and xv < t[l]:
            l -= 1
        if der == 0:
            h = _deboor(t, xv, k, l, 0)
        else:
            h = _deboor(t, xv, k, l, der)
        s = 0.0
        for a in range(k + 1):
            s += c2[l + a - k, 0] * h[a]
        out[i] = s
    r = out.reshape(xa.shape)
    if static.len(r.shape) == 0:
        return r.item()
    else:
        return r

def splint(a: float, b: float, tck, full_output: int = 0) -> float:
    t, c2, k = _tck(tck)
    n = t.size
    lo = t[k]
    hi = t[n - k - 1]
    sign = 1.0
    if b < a:
        a, b = b, a
        sign = -1.0
    a = max(a, lo)
    b = min(b, hi)
    if b <= a:
        return 0.0
    ta, ca, ka = _splantider(t, _pad(t, c2), k, 1)
    o = _bspl_eval(ta, ca, ka, np.array([a, b]), 0, False)
    return sign * (o[1, 0] - o[0, 0])

def splder(tck, n: int = 1):
    t, c2, k = _tck(tck)
    if n < 0:
        return splantider(tck, -n)
    t2, cc, k2 = _splder(t, _pad(t, c2), k, n)
    return t2, cc.reshape(-1), k2

def splantider(tck, n: int = 1):
    t, c2, k = _tck(tck)
    if n < 0:
        return splder(tck, -n)
    t2, cc, k2 = _splantider(t, _pad(t, c2), k, n)
    return t2, cc.reshape(-1), k2
