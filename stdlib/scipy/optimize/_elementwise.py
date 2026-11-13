# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)
# Elementwise scalar root / minimum finding and bracketing:
#   find_root (Chandrupatla), find_minimum (Chandrupatla), bracket_root,
#   bracket_minimum (scipy/optimize/_chandrupatla.py, _bracket.py, _elementwise.py).
# Deviations: `f` always receives 1-D float arrays (the active elements) and
# `args` are float arrays/scalars broadcast to a common length. All-scalar
# inputs give scalar results; otherwise 1-D arrays (multi-dimensional inputs
# are flattened). `kwargs` and `callback` are not supported. Integer `args`
# arrive as float64.

import math
import numpy as np

_EPS = 2.220446049250313e-16
_TINY = 2.2250738585072014e-308
_NAN = float('nan')
_INF = float('inf')

def _isarr(x) -> Literal[bool]:
    return isinstance(x, np.ndarray) or isinstance(x, List) or isinstance(x, Tuple)

def _all_scalar(*xs) -> Literal[bool]:
    if static.len(xs) == 0:
        return True
    elif _isarr(xs[0]):
        return False
    else:
        return _all_scalar(*xs[1:])

def _any_arr(args) -> Literal[bool]:
    if static.len(args) == 0:
        return False
    elif _isarr(args[0]):
        return True
    else:
        return _any_arr(args[1:])

def _bflat(x, n: int) -> np.ndarray[float, 1]:
    a = np.asarray(x).astype(float)
    v = a.reshape(a.size)
    if v.shape[0] == n:
        return v.copy()
    if v.shape[0] == 1:
        out = np.empty(n)
        for i in range(n):
            out[i] = v[0]
        return out
    raise ValueError("The shapes of the inputs are not broadcastable.")

def _bargs(args, n: int):
    if static.len(args) == 0:
        return ()
    return (_bflat(args[0], n),) + _bargs(args[1:], n)

def _take(args, idx: List[int]):
    if static.len(args) == 0:
        return ()
    a = args[0]
    out = np.empty(len(idx))
    for k in range(len(idx)):
        out[k] = a[idx[k]]
    return (out,) + _take(args[1:], idx)

def _size(x) -> int:
    if isinstance(x, NoneType):
        return 1
    return int(np.asarray(x).size)

def _sizes(args) -> int:
    if static.len(args) == 0:
        return 1
    return max(_size(args[0]), _sizes(args[1:]))

def _evalf(f, xs: List[float], args, idx: List[int]) -> List[float]:
    m = len(idx)
    xa = np.empty(m)
    for k in range(m):
        xa[k] = xs[idx[k]]
    r = np.asarray(f(xa, *_take(args, idx))).astype(float)
    if r.size != m:
        raise ValueError("The shape of the array returned by `func` must be the same as the broadcasted shape of `x` and all other `args`.")
    rv = r.reshape(m)
    out = [0.0] * m
    for k in range(m):
        out[k] = rv[k]
    return out

def _evalx(f, xv: List[float], args, idx: List[int]) -> List[float]:
    # xv is aligned with idx
    m = len(idx)
    xa = np.empty(m)
    for k in range(m):
        xa[k] = xv[k]
    r = np.asarray(f(xa, *_take(args, idx))).astype(float)
    if r.size != m:
        raise ValueError("The shape of the array returned by `func` must be the same as the broadcasted shape of `x` and all other `args`.")
    rv = r.reshape(m)
    out = [0.0] * m
    for k in range(m):
        out[k] = rv[k]
    return out

def _div(a: float, b: float) -> float:
    if b == 0.0:
        if a == 0.0 or a != a:
            return _NAN
        neg = (a < 0.0) != (math.copysign(1.0, b) < 0.0)
        return -_INF if neg else _INF
    return a / b

def _sqrt(a: float) -> float:
    if a < 0.0 or a != a:
        return _NAN
    return math.sqrt(a)

def _sgn(a: float) -> float:
    if a != a:
        return _NAN
    if a > 0.0:
        return 1.0
    if a < 0.0:
        return -1.0
    return 0.0

def _isnan(a: float) -> bool:
    return a != a

def _fin(a: float) -> bool:
    return a == a and a != _INF and a != -_INF

class _Raw:
    n: int
    status: List[int]
    x: List[float]
    fun: List[float]
    nit: List[int]
    nfev: List[int]
    xl: List[float]
    xm: List[float]
    xr: List[float]
    fl: List[float]
    fm: List[float]
    fr: List[float]

    def __init__(self, n: int):
        self.n = n
        self.status = [1] * n
        self.x = [_NAN] * n
        self.fun = [_NAN] * n
        self.nit = [0] * n
        self.nfev = [0] * n
        self.xl = [_NAN] * n
        self.xm = [_NAN] * n
        self.xr = [_NAN] * n
        self.fl = [_NAN] * n
        self.fm = [_NAN] * n
        self.fr = [_NAN] * n

def _check_tols(xatol: Optional[float], xrtol: Optional[float], fatol: Optional[float], frtol: Optional[float]):
    if xatol is not None and (xatol < 0 or xatol != xatol):
        raise ValueError('Tolerances must be non-negative scalars.')
    if xrtol is not None and (xrtol < 0 or xrtol != xrtol):
        raise ValueError('Tolerances must be non-negative scalars.')
    if fatol is not None and (fatol < 0 or fatol != fatol):
        raise ValueError('Tolerances must be non-negative scalars.')
    if frtol is not None and (frtol < 0 or frtol != frtol):
        raise ValueError('Tolerances must be non-negative scalars.')

# ------------------------------------------------------------------ find_root
class _RootState:
    x1: List[float]
    f1: List[float]
    x2: List[float]
    f2: List[float]
    x3: List[float]
    f3: List[float]
    t: List[float]
    xmin: List[float]
    fmin: List[float]
    dx: List[float]
    tol: List[float]
    frt: List[float]
    status: List[int]
    xatol: float
    xrtol: float
    fatol: float

    def __init__(self):
        self.x1 = List[float]()
        self.f1 = List[float]()
        self.x2 = List[float]()
        self.f2 = List[float]()
        self.x3 = List[float]()
        self.f3 = List[float]()
        self.t = List[float]()
        self.xmin = List[float]()
        self.fmin = List[float]()
        self.dx = List[float]()
        self.tol = List[float]()
        self.frt = List[float]()
        self.status = List[int]()
        self.xatol = 0.0
        self.xrtol = 0.0
        self.fatol = 0.0

    def check(self, active: List[int], nit: int, nfev: int, res: _Raw) -> List[int]:
        keep = List[int]()
        for i in active:
            stop = False
            if abs(self.f1[i]) < abs(self.f2[i]):
                self.xmin[i] = self.x1[i]
                self.fmin[i] = self.f1[i]
            else:
                self.xmin[i] = self.x2[i]
                self.fmin[i] = self.f2[i]
            if abs(self.fmin[i]) <= self.fatol + self.frt[i]:
                self.status[i] = 0
                stop = True
            if _sgn(self.f1[i]) == _sgn(self.f2[i]) and not stop:
                self.xmin[i] = _NAN
                self.fmin[i] = _NAN
                self.status[i] = -1
                stop = True
            xnf = not (_fin(self.x1[i]) and _fin(self.x2[i]))
            fnan = _isnan(self.f1[i]) and _isnan(self.f2[i])
            if (xnf or fnan) and not stop:
                self.xmin[i] = _NAN
                self.fmin[i] = _NAN
                self.status[i] = -3
                stop = True
            self.dx[i] = abs(self.x2[i] - self.x1[i])
            self.tol[i] = abs(self.xmin[i]) * self.xrtol + self.xatol
            if self.dx[i] < self.tol[i]:
                self.status[i] = 0
                stop = True
            if stop:
                self.record(i, nit, nfev, res)
            else:
                keep.append(i)
        return keep

    def record(self, i: int, nit: int, nfev: int, res: _Raw):
        res.status[i] = self.status[i]
        res.x[i] = self.xmin[i]
        res.fun[i] = self.fmin[i]
        res.nit[i] = nit
        res.nfev[i] = nfev
        res.xl[i] = self.x1[i]
        res.fl[i] = self.f1[i]
        res.xr[i] = self.x2[i]
        res.fr[i] = self.f2[i]

    def update_t(self, i: int):
        xi1 = _div(self.x1[i] - self.x2[i], self.x3[i] - self.x2[i])
        phi1 = _div(self.f1[i] - self.f2[i], self.f3[i] - self.f2[i])
        alpha = _div(self.x3[i] - self.x1[i], self.x2[i] - self.x1[i])
        t = 0.5
        if (1.0 - _sqrt(1.0 - xi1)) < phi1 and phi1 < _sqrt(xi1):
            f1 = self.f1[i]
            f2 = self.f2[i]
            f3 = self.f3[i]
            t = _div(_div(f1, f1 - f2) * f3, f3 - f2) - _div(_div(alpha * f1, f3 - f1) * f2, f2 - f3)
        tl = _div(0.5 * self.tol[i], self.dx[i])
        if t != t or tl != tl:
            self.t[i] = _NAN
        else:
            lo = t if t > tl else tl
            hi = 1.0 - tl
            self.t[i] = lo if lo < hi else hi

def _find_root_raw(f, args, a: np.ndarray[float, 1], b: np.ndarray[float, 1], xatol: Optional[float],
                   xrtol: Optional[float], fatol: Optional[float], frtol: Optional[float], maxiter: Optional[float]) -> _Raw:
    _check_tols(xatol, xrtol, fatol, frtol)
    n = a.shape[0]
    res = _Raw(n)
    st = _RootState()
    st.x1 = [a[i] for i in range(n)]
    st.x2 = [b[i] for i in range(n)]
    allidx = [i for i in range(n)]
    st.f1 = _evalf(f, st.x1, args, allidx)
    st.f2 = _evalf(f, st.x2, args, allidx)
    st.x3 = [0.0] * n
    st.f3 = [0.0] * n
    st.t = [0.5] * n
    st.xmin = [0.0] * n
    st.fmin = [0.0] * n
    st.dx = [0.0] * n
    st.tol = [0.0] * n
    st.status = [1] * n
    st.xatol = 4.0 * _TINY if xatol is None else xatol
    st.xrtol = 4.0 * _EPS if xrtol is None else xrtol
    st.fatol = _TINY if fatol is None else fatol
    frt = 0.0 if frtol is None else frtol
    st.frt = [frt * min(abs(st.f1[i]), abs(st.f2[i])) for i in range(n)]
    mi = 2046.0 if maxiter is None else maxiter
    if mi < 0 or mi != math.floor(mi):
        raise ValueError('`maxiter` must be a non-negative integer.')
    nit = 0
    nfev = 2
    active = st.check(allidx, nit, nfev, res)
    while float(nit) < mi and len(active) > 0:
        xv = [st.x1[i] + st.t[i] * (st.x2[i] - st.x1[i]) for i in active]
        fv = _evalx(f, xv, args, active)
        for k in range(len(active)):
            i = active[k]
            st.x3[i] = st.x2[i]
            st.f3[i] = st.f2[i]
            if _sgn(fv[k]) == _sgn(st.f1[i]):
                st.x3[i] = st.x1[i]
                st.f3[i] = st.f1[i]
            else:
                st.x2[i] = st.x1[i]
                st.f2[i] = st.f1[i]
            st.x1[i] = xv[k]
            st.f1[i] = fv[k]
        nfev += 1
        nit += 1
        active = st.check(active, nit, nfev, res)
        if len(active) == 0:
            break
        for i in active:
            st.update_t(i)
    for i in active:
        st.status[i] = -2
        st.record(i, nit, nfev, res)
    for i in range(n):
        if res.xl[i] >= res.xr[i]:
            t1 = res.xl[i]
            res.xl[i] = res.xr[i]
            res.xr[i] = t1
            t2 = res.fl[i]
            res.fl[i] = res.fr[i]
            res.fr[i] = t2
    return res

# ------------------------------------------------------------------ find_minimum
class _MinState:
    x1: List[float]
    f1: List[float]
    x2: List[float]
    f2: List[float]
    x3: List[float]
    f3: List[float]
    q0: List[float]
    xtol: List[float]
    status: List[int]
    xatol: float
    xrtol: float
    fatol: float
    frtol: float

    def __init__(self):
        self.x1 = List[float]()
        self.f1 = List[float]()
        self.x2 = List[float]()
        self.f2 = List[float]()
        self.x3 = List[float]()
        self.f3 = List[float]()
        self.q0 = List[float]()
        self.xtol = List[float]()
        self.status = List[int]()
        self.xatol = 0.0
        self.xrtol = 0.0
        self.fatol = 0.0
        self.frtol = 0.0

    def check(self, active: List[int], nit: int, nfev: int, res: _Raw) -> List[int]:
        keep = List[int]()
        for i in active:
            stop = False
            if self.f2[i] > self.f1[i] or self.f2[i] > self.f3[i]:
                self.x2[i] = _NAN
                self.f2[i] = _NAN
                stop = True
                self.status[i] = -1
            tot = self.x1[i] + self.x2[i] + self.x3[i] + self.f1[i] + self.f2[i] + self.f3[i]
            if not _fin(tot) and not stop:
                self.x2[i] = _NAN
                self.f2[i] = _NAN
                stop = True
                self.status[i] = -3
            if abs(self.x3[i] - self.x2[i]) < abs(self.x2[i] - self.x1[i]):
                t = self.x1[i]
                self.x1[i] = self.x3[i]
                self.x3[i] = t
                t = self.f1[i]
                self.f1[i] = self.f3[i]
                self.f3[i] = t
            self.xtol[i] = abs(self.x2[i]) * self.xrtol + self.xatol
            conv = abs(self.x3[i] - self.x2[i]) <= 2.0 * self.xtol[i]
            ftol = abs(self.f2[i]) * self.frtol + self.fatol
            if (self.f1[i] - 2.0 * self.f2[i] + self.f3[i]) <= 2.0 * ftol:
                conv = True
            if conv and not stop:
                stop = True
                self.status[i] = 0
            if stop:
                res.status[i] = self.status[i]
                res.x[i] = self.x2[i]
                res.fun[i] = self.f2[i]
                res.nit[i] = nit
                res.nfev[i] = nfev
                res.xl[i] = self.x1[i]
                res.xm[i] = self.x2[i]
                res.xr[i] = self.x3[i]
                res.fl[i] = self.f1[i]
                res.fm[i] = self.f2[i]
                res.fr[i] = self.f3[i]
            else:
                keep.append(i)
        return keep

def _find_min_raw(f, args, a: np.ndarray[float, 1], b: np.ndarray[float, 1], c: np.ndarray[float, 1],
                  xatol: Optional[float], xrtol: Optional[float], fatol: Optional[float], frtol: Optional[float],
                  maxiter: float) -> _Raw:
    _check_tols(xatol, xrtol, fatol, frtol)
    if maxiter < 0 or maxiter != math.floor(maxiter):
        raise ValueError('`maxiter` must be a non-negative integer.')
    n = a.shape[0]
    res = _Raw(n)
    st = _MinState()
    xa = [a[i] for i in range(n)]
    xb = [b[i] for i in range(n)]
    xc = [c[i] for i in range(n)]
    allidx = [i for i in range(n)]
    fa = _evalf(f, xa, args, allidx)
    fb = _evalf(f, xb, args, allidx)
    fc = _evalf(f, xc, args, allidx)
    st.x1 = [0.0] * n
    st.x2 = [0.0] * n
    st.x3 = [0.0] * n
    st.f1 = [0.0] * n
    st.f2 = [0.0] * n
    st.f3 = [0.0] * n
    for i in range(n):
        xs = [xa[i], xb[i], xc[i]]
        fs = [fa[i], fb[i], fc[i]]
        order = [0, 1, 2]
        # stable argsort (NaN last, as numpy)
        for p in range(1, 3):
            q = p
            while q > 0 and (xs[order[q]] < xs[order[q - 1]] or (xs[order[q - 1]] != xs[order[q - 1]] and xs[order[q]] == xs[order[q]])):
                tmp = order[q]
                order[q] = order[q - 1]
                order[q - 1] = tmp
                q -= 1
        st.x1[i] = xs[order[0]]
        st.x2[i] = xs[order[1]]
        st.x3[i] = xs[order[2]]
        st.f1[i] = fs[order[0]]
        st.f2[i] = fs[order[1]]
        st.f3[i] = fs[order[2]]
    st.q0 = [st.x3[i] for i in range(n)]
    st.xtol = [0.0] * n
    st.status = [1] * n
    st.fatol = _TINY if fatol is None else fatol
    st.frtol = _TINY if frtol is None else frtol
    st.xatol = _TINY if xatol is None else xatol
    st.xrtol = math.sqrt(_EPS) if xrtol is None else xrtol
    phi = 0.5 + 0.5 * math.sqrt(5.0)
    nit = 0
    nfev = 3
    active = st.check(allidx, nit, nfev, res)
    while float(nit) < maxiter and len(active) > 0:
        xv = [0.0] * len(active)
        for k in range(len(active)):
            i = active[k]
            x21 = st.x2[i] - st.x1[i]
            x32 = st.x3[i] - st.x2[i]
            A = x21 * (st.f3[i] - st.f2[i])
            B = x32 * (st.f1[i] - st.f2[i])
            C = _div(A, A + B)
            q1 = 0.5 * (C * (st.x1[i] - st.x3[i]) + st.x2[i] + st.x3[i])
            x = st.x2[i] + (2.0 - phi) * x32
            if abs(q1 - st.q0[i]) < 0.5 * abs(x21):
                x = q1
                if abs(q1 - st.x2[i]) <= st.xtol[i]:
                    x = st.x2[i] + _sgn(x32) * st.xtol[i]
            st.q0[i] = q1
            xv[k] = x
        fv = _evalx(f, xv, args, active)
        for k in range(len(active)):
            i = active[k]
            x = xv[k]
            fx = fv[k]
            if _sgn(x - st.x2[i]) == _sgn(st.x3[i] - st.x2[i]):
                if fx > st.f2[i]:
                    st.x3[i] = x
                    st.f3[i] = fx
                else:
                    st.x1[i] = st.x2[i]
                    st.f1[i] = st.f2[i]
                    st.x2[i] = x
                    st.f2[i] = fx
            else:
                if fx > st.f2[i]:
                    st.x1[i] = x
                    st.f1[i] = fx
                else:
                    st.x3[i] = st.x2[i]
                    st.f3[i] = st.f2[i]
                    st.x2[i] = x
                    st.f2[i] = fx
        nfev += 1
        nit += 1
        active = st.check(active, nit, nfev, res)
    for i in active:
        res.status[i] = -2
        res.x[i] = st.x2[i]
        res.fun[i] = st.f2[i]
        res.nit[i] = nit
        res.nfev[i] = nfev
        res.xl[i] = st.x1[i]
        res.xm[i] = st.x2[i]
        res.xr[i] = st.x3[i]
        res.fl[i] = st.f1[i]
        res.fm[i] = st.f2[i]
        res.fr[i] = st.f3[i]
    for i in range(n):
        if res.xl[i] >= res.xr[i]:
            t = res.xl[i]
            res.xl[i] = res.xr[i]
            res.xr[i] = t
            t = res.fl[i]
            res.fl[i] = res.fr[i]
            res.fr[i] = t
    return res

# ------------------------------------------------------------------ bracket_root
_ELIMITS = -1
_ESTOPONESIDE = 2

def _bracket_root_raw(f, args, xl0: np.ndarray[float, 1], xr0: np.ndarray[float, 1], xmin: np.ndarray[float, 1],
                      xmax: np.ndarray[float, 1], factor: np.ndarray[float, 1], maxiter: float) -> _Raw:
    n = xl0.shape[0]
    for i in range(n):
        if not factor[i] > 1.0:
            raise ValueError('All elements of `factor` must be greater than 1.')
    if maxiter < 0 or maxiter != math.floor(maxiter):
        raise ValueError('`maxiter` must be a non-negative integer.')
    m = 2 * n
    # search arrays of length 2n: k < n leftward (moving end starts at xl0), k >= n rightward
    allidx = [i for i in range(n)]
    xs0 = [xl0[i] for i in range(n)]
    xs1 = [xr0[i] for i in range(n)]
    fl0 = _evalf(f, xs0, args, allidx)
    fr0 = _evalf(f, xs1, args, allidx)
    x = [0.0] * m
    f0 = [0.0] * m
    x_last = [0.0] * m
    f_last = [0.0] * m
    x0 = [0.0] * m
    limit = [0.0] * m
    fac = [0.0] * m
    d = [0.0] * m
    status = [1] * m
    for i in range(n):
        x[i] = xs0[i]
        x[n + i] = xs1[i]
        f0[i] = fl0[i]
        f0[n + i] = fr0[i]
        x_last[i] = xs1[i]
        x_last[n + i] = xs0[i]
        f_last[i] = fr0[i]
        f_last[n + i] = fl0[i]
        x0[i] = x_last[i]
        x0[n + i] = x_last[n + i]
        limit[i] = xmin[i]
        limit[n + i] = xmax[i]
        fac[i] = factor[i]
        fac[n + i] = factor[i]
        bad = not (xmin[i] <= xl0[i] and xl0[i] < xr0[i] and xr0[i] <= xmax[i])
        if bad:
            status[i] = -5
            status[n + i] = -5
    for k in range(m):
        if limit[k] == _INF or limit[k] == -_INF:
            d[k] = x[k] - x0[k]
        else:
            d[k] = limit[k] - x[k]
    r_status = [1] * m
    r_x = [0.0] * m
    r_f = [0.0] * m
    r_xl = [0.0] * m
    r_fl = [0.0] * m
    r_nit = [0] * m
    r_nfev = [0] * m
    nit = 0
    nfev = 1
    active = [k for k in range(m)]

    def check(act: List[int]) -> List[int]:
        stop = [False] * m
        conv = List[int]()
        for k in act:
            if status[k] == -5:
                stop[k] = True
        for k in act:
            if stop[k]:
                continue
            sf = _sgn(f0[k])
            sl = _sgn(f_last[k])
            if sl == -sf or sl == 0.0 or sf == 0.0:
                status[k] = 0
                stop[k] = True
                conv.append(k)
        also = List[int]()
        for k in conv:
            also.append((k + n) % m)
        for o in also:
            isact = False
            for k in act:
                if k == o:
                    isact = True
            if isact and not stop[o]:
                status[o] = _ESTOPONESIDE
                stop[o] = True
        for k in act:
            if stop[k]:
                continue
            if x[k] == limit[k]:
                status[k] = _ELIMITS
                stop[k] = True
            elif not (_fin(x[k]) and _fin(f0[k])):
                status[k] = -3
                stop[k] = True
        keep = List[int]()
        for k in act:
            if stop[k]:
                r_status[k] = status[k]
                r_x[k] = x[k]
                r_f[k] = f0[k]
                r_xl[k] = x_last[k]
                r_fl[k] = f_last[k]
                r_nit[k] = nit
                r_nfev[k] = nfev
            else:
                keep.append(k)
        return keep

    active = check(active)
    while float(nit) < maxiter and len(active) > 0:
        xn = [0.0] * len(active)
        for j in range(len(active)):
            k = active[j]
            if limit[k] == _INF or limit[k] == -_INF:
                d[k] *= fac[k]
                xn[j] = x0[k] + d[k]
            else:
                d[k] /= fac[k]
                xn[j] = limit[k] - d[k]
        # f is evaluated on the original (per-element) args: map search id -> element id
        eidx = [k % n for k in active]
        fv = _evalx(f, xn, args, eidx)
        for j in range(len(active)):
            k = active[j]
            x_last[k] = x[k]
            f_last[k] = f0[k]
            x[k] = xn[j]
            f0[k] = fv[j]
        nfev += 1
        nit += 1
        active = check(active)
    for k in active:
        r_status[k] = -2
        r_x[k] = x[k]
        r_f[k] = f0[k]
        r_xl[k] = x_last[k]
        r_fl[k] = f_last[k]
        r_nit[k] = nit
        r_nfev[k] = nfev
    res = _Raw(n)
    for i in range(n):
        xal = r_x[i]
        xar = r_xl[i]
        xbl = r_xl[n + i]
        xbr = r_x[n + i]
        fal = r_f[i]
        far = r_fl[i]
        fbl = r_fl[n + i]
        fbr = r_f[n + i]
        xlo = xal
        flo = fal
        xro = xbr
        fro = fbr
        sa = r_status[i]
        sb = r_status[n + i]
        da = xar - xal
        db = xbr - xbl
        i1 = (da <= db and sa == 0) or (sa == 0 and sb != 0)
        i2 = (db <= da and sb == 0) or (sb == 0 and sa != 0)
        if i1:
            xro = xar
            fro = far
        if i2:
            xlo = xbl
            flo = fbl
        res.xl[i] = xlo
        res.fl[i] = flo
        res.xr[i] = xro
        res.fr[i] = fro
        res.nit[i] = max(r_nit[i], r_nit[n + i])
        res.nfev[i] = r_nfev[i] + r_nfev[n + i]
        res.status[i] = sa if sa == 0 else sb
    return res

# ------------------------------------------------------------------ bracket_minimum
def _bracket_min_raw(f, args, xm0: np.ndarray[float, 1], xl0a: np.ndarray[float, 1], xr0a: np.ndarray[float, 1],
                     xmin: np.ndarray[float, 1], xmax: np.ndarray[float, 1], factor0: np.ndarray[float, 1],
                     maxiter: float) -> _Raw:
    n = xm0.shape[0]
    for i in range(n):
        if not factor0[i] > 1.0:
            raise ValueError('All elements of `factor` must be greater than 1.')
    if maxiter < 0 or maxiter != math.floor(maxiter):
        raise ValueError('`maxiter` must be a non-negative integer.')
    allidx = [i for i in range(n)]
    xl = [xl0a[i] for i in range(n)]
    xm = [xm0[i] for i in range(n)]
    xr = [xr0a[i] for i in range(n)]
    fl = _evalf(f, xl, args, allidx)
    fm = _evalf(f, xm, args, allidx)
    fr = _evalf(f, xr, args, allidx)
    status = [1] * n
    limit = [0.0] * n
    limited = [False] * n
    step = [0.0] * n
    factor = [0.0] * n
    xr_0 = [0.0] * n
    for i in range(n):
        if not (xmin[i] <= xl[i] and xl[i] < xm[i] and xm[i] < xr[i] and xr[i] <= xmax[i]):
            status[i] = -5
        factor[i] = factor0[i]
        if fl[i] < fr[i]:
            t = xl[i]
            xl[i] = xr[i]
            xr[i] = t
            t = fl[i]
            fl[i] = fr[i]
            fr[i] = t
            limit[i] = xmin[i]
        else:
            limit[i] = xmax[i]
        limited[i] = not (limit[i] == _INF or limit[i] == -_INF)
        if not limited[i]:
            step[i] = xr[i] - xm[i]
        else:
            step[i] = limit[i] - xr[i]
            factor[i] = 1.0 / factor[i]
        xr_0[i] = xr[i]
    res = _Raw(n)
    nit = 0
    nfev = 3

    def check(act: List[int]) -> List[int]:
        keep = List[int]()
        for i in act:
            stop = status[i] == -5
            if not stop:
                if (fl[i] >= fm[i] and fr[i] > fm[i]) or (fl[i] > fm[i] and fr[i] >= fm[i]):
                    status[i] = 0
                    stop = True
            if not stop and xr[i] == limit[i]:
                status[i] = _ELIMITS
                stop = True
            if not stop and not (_fin(xr[i]) and _fin(fr[i])):
                status[i] = -3
                stop = True
            if stop:
                res.status[i] = status[i]
                res.xl[i] = xl[i]
                res.xm[i] = xm[i]
                res.xr[i] = xr[i]
                res.fl[i] = fl[i]
                res.fm[i] = fm[i]
                res.fr[i] = fr[i]
                res.nit[i] = nit
                res.nfev[i] = nfev
            else:
                keep.append(i)
        return keep

    active = check(allidx)
    while float(nit) < maxiter and len(active) > 0:
        xn = [0.0] * len(active)
        for j in range(len(active)):
            i = active[j]
            step[i] *= factor[i]
            if not limited[i]:
                xn[j] = xr_0[i] + step[i]
            else:
                v = limit[i] - step[i]
                if v == xr[i]:
                    v = limit[i]
                xn[j] = v
        fv = _evalx(f, xn, args, active)
        for j in range(len(active)):
            i = active[j]
            xl[i] = xm[i]
            xm[i] = xr[i]
            xr[i] = xn[j]
            fl[i] = fm[i]
            fm[i] = fr[i]
            fr[i] = fv[j]
        nfev += 1
        nit += 1
        active = check(active)
    for i in active:
        res.status[i] = -2
        res.xl[i] = xl[i]
        res.xm[i] = xm[i]
        res.xr[i] = xr[i]
        res.fl[i] = fl[i]
        res.fm[i] = fm[i]
        res.fr[i] = fr[i]
        res.nit[i] = nit
        res.nfev[i] = nfev
    for i in range(n):
        if res.xl[i] > res.xr[i]:
            t = res.xl[i]
            res.xl[i] = res.xr[i]
            res.xr[i] = t
            t = res.fl[i]
            res.fl[i] = res.fr[i]
            res.fr[i] = t
    return res

# ------------------------------------------------------------------ public API
class ElementwiseResult[B, S, X, FX, N, BR, FBR]:
    success: B
    status: S
    x: X
    f_x: FX
    nfev: N
    nit: N
    bracket: BR
    f_bracket: FBR

    def __init__(self, success: B, status: S, x: X, f_x: FX, nfev: N, nit: N, bracket: BR, f_bracket: FBR):
        self.success = success
        self.status = status
        self.x = x
        self.f_x = f_x
        self.nfev = nfev
        self.nit = nit
        self.bracket = bracket
        self.f_bracket = f_bracket

def _arr_f(v: List[float], scalar: Literal[bool]):
    if scalar:
        return v[0]
    out = np.empty(len(v))
    for i in range(len(v)):
        out[i] = v[i]
    return out

def _arr_i(v: List[int], scalar: Literal[bool]):
    if scalar:
        return v[0]
    out = np.empty(len(v), dtype=int)
    for i in range(len(v)):
        out[i] = v[i]
    return out

def _arr_b(v: List[int], scalar: Literal[bool]):
    if scalar:
        return v[0] == 0
    out = np.empty(len(v), dtype=bool)
    for i in range(len(v)):
        out[i] = v[i] == 0
    return out

def _optf(x, default: float, n: int) -> np.ndarray[float, 1]:
    if isinstance(x, NoneType):
        return np.full(n, default)
    return _bflat(x, n)

def _opt_size(x) -> int:
    if isinstance(x, NoneType):
        return 1
    return int(np.asarray(x).size)

def find_root(f, init, /, *, args=(), kwargs=None, tolerances=None, maxiter=None, callback=None):
    if not callable(f):
        raise ValueError('`func` must be callable.')
    if kwargs is not None or callback is not None:
        raise ValueError("`kwargs` and `callback` are not supported")
    if not isinstance(args, Tuple):
        raise ValueError("`args` must be a tuple")
    xl = init[0]
    xr = init[1]
    n = max(max(_size(xl), _size(xr)), _sizes(args))
    SC: Literal[bool] = _all_scalar(xl, xr, *args)
    a = _bflat(xl, n)
    b = _bflat(xr, n)
    xa: Optional[float] = None
    xrt: Optional[float] = None
    fa: Optional[float] = None
    fr: Optional[float] = 0.0
    if tolerances is not None:
        if 'xatol' in tolerances:
            xa = float(tolerances['xatol'])
        if 'xrtol' in tolerances:
            xrt = float(tolerances['xrtol'])
        if 'fatol' in tolerances:
            fa = float(tolerances['fatol'])
        if 'frtol' in tolerances:
            fr = float(tolerances['frtol'])
    mi: Optional[float] = None
    if maxiter is not None:
        mi = float(maxiter)
    r = _find_root_raw(f, _bargs(args, n), a, b, xa, xrt, fa, fr, mi)
    return ElementwiseResult(_arr_b(r.status, SC), _arr_i(r.status, SC), _arr_f(r.x, SC), _arr_f(r.fun, SC),
                             _arr_i(r.nfev, SC), _arr_i(r.nit, SC), (_arr_f(r.xl, SC), _arr_f(r.xr, SC)),
                             (_arr_f(r.fl, SC), _arr_f(r.fr, SC)))

def find_minimum(f, init, /, *, args=(), kwargs=None, tolerances=None, maxiter=100, callback=None):
    if not callable(f):
        raise ValueError('`func` must be callable.')
    if kwargs is not None or callback is not None:
        raise ValueError("`kwargs` and `callback` are not supported")
    if not isinstance(args, Tuple):
        raise ValueError("`args` must be a tuple")
    x1 = init[0]
    x2 = init[1]
    x3 = init[2]
    n = max(max(_size(x1), _size(x2)), max(_size(x3), _sizes(args)))
    SC: Literal[bool] = _all_scalar(x1, x2, x3, *args)
    xa: Optional[float] = None
    xrt: Optional[float] = None
    fa: Optional[float] = None
    fr: Optional[float] = None
    if tolerances is not None:
        if 'xatol' in tolerances:
            xa = float(tolerances['xatol'])
        if 'xrtol' in tolerances:
            xrt = float(tolerances['xrtol'])
        if 'fatol' in tolerances:
            fa = float(tolerances['fatol'])
        if 'frtol' in tolerances:
            fr = float(tolerances['frtol'])
    r = _find_min_raw(f, _bargs(args, n), _bflat(x1, n), _bflat(x2, n), _bflat(x3, n), xa, xrt, fa, fr, float(maxiter))
    return ElementwiseResult(_arr_b(r.status, SC), _arr_i(r.status, SC), _arr_f(r.x, SC), _arr_f(r.fun, SC),
                             _arr_i(r.nfev, SC), _arr_i(r.nit, SC),
                             (_arr_f(r.xl, SC), _arr_f(r.xm, SC), _arr_f(r.xr, SC)),
                             (_arr_f(r.fl, SC), _arr_f(r.fm, SC), _arr_f(r.fr, SC)))

def bracket_root(f, xl0, xr0=None, *, xmin=None, xmax=None, factor=None, args=(), kwargs=None, maxiter=1000):
    if not callable(f):
        raise ValueError('`func` must be callable.')
    if kwargs is not None:
        raise ValueError("`kwargs` are not supported")
    if not isinstance(args, Tuple):
        raise ValueError("`args` must be a tuple")
    n = max(max(_size(xl0), _opt_size(xr0)), max(max(_opt_size(xmin), _opt_size(xmax)), max(_opt_size(factor), _sizes(args))))
    SC: Literal[bool] = _all_scalar(xl0, xr0, xmin, xmax, factor, *args)
    a = _bflat(xl0, n)
    lo = _optf(xmin, -_INF, n)
    hi = _optf(xmax, _INF, n)
    b = np.empty(n)
    if isinstance(xr0, NoneType):
        for i in range(n):
            b[i] = a[i] + min((hi[i] - a[i]) / 8.0, 1.0)
    else:
        b = _bflat(xr0, n)
    r = _bracket_root_raw(f, _bargs(args, n), a, b, lo, hi, _optf(factor, 2.0, n), float(maxiter))
    return ElementwiseResult(_arr_b(r.status, SC), _arr_i(r.status, SC), None, None, _arr_i(r.nfev, SC),
                             _arr_i(r.nit, SC), (_arr_f(r.xl, SC), _arr_f(r.xr, SC)),
                             (_arr_f(r.fl, SC), _arr_f(r.fr, SC)))

def bracket_minimum(f, xm0, *, xl0=None, xr0=None, xmin=None, xmax=None, factor=None, args=(), kwargs=None, maxiter=1000):
    if not callable(f):
        raise ValueError('`func` must be callable.')
    if kwargs is not None:
        raise ValueError("`kwargs` are not supported")
    if not isinstance(args, Tuple):
        raise ValueError("`args` must be a tuple")
    n = max(max(_size(xm0), max(_opt_size(xl0), _opt_size(xr0))), max(max(_opt_size(xmin), _opt_size(xmax)), max(_opt_size(factor), _sizes(args))))
    SC: Literal[bool] = _all_scalar(xm0, xl0, xr0, xmin, xmax, factor, *args)
    m = _bflat(xm0, n)
    lo = _optf(xmin, -_INF, n)
    hi = _optf(xmax, _INF, n)
    l = np.empty(n)
    r_ = np.empty(n)
    if isinstance(xl0, NoneType):
        for i in range(n):
            l[i] = m[i] - min((m[i] - lo[i]) / 16.0, 0.5)
    else:
        l = _bflat(xl0, n)
    if isinstance(xr0, NoneType):
        for i in range(n):
            r_[i] = m[i] + min((hi[i] - m[i]) / 16.0, 0.5)
    else:
        r_ = _bflat(xr0, n)
    r = _bracket_min_raw(f, _bargs(args, n), m, l, r_, lo, hi, _optf(factor, 2.0, n), float(maxiter))
    return ElementwiseResult(_arr_b(r.status, SC), _arr_i(r.status, SC), None, None, _arr_i(r.nfev, SC),
                             _arr_i(r.nit, SC), (_arr_f(r.xl, SC), _arr_f(r.xm, SC), _arr_f(r.xr, SC)),
                             (_arr_f(r.fl, SC), _arr_f(r.fm, SC), _arr_f(r.fr, SC)))
