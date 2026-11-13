# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)
# scipy.integrate.tanhsinh (adaptive tanh-sinh quadrature) and nsum
# (scipy/integrate/_tanhsinh.py). The algorithm is elementwise, so each
# element is processed independently; `f` is called with 2-D arrays of shape
# (1, n_nodes) (and (1, 1) args) per element after one initial call with
# 1-D arrays of length 1, matching scipy's per-element call semantics.
# Deviations: real integrands and float results only; `log=True`,
# `preserve_shape`, `kwargs` and `callback` are not supported; results are
# scalars for scalar inputs and 1-D arrays otherwise (inputs are flattened).

import math
import numpy as np

_EPS = 2.220446049250313e-16
_TINY = 2.2250738585072014e-308
_NAN = float('nan')
_INF = float('inf')
_N_BASE_STEPS = 8

def _isarr(x) -> Literal[bool]:
    return isinstance(x, np.ndarray) or isinstance(x, List) or isinstance(x, Tuple)

def _all_scalar(*xs) -> Literal[bool]:
    if static.len(xs) == 0:
        return True
    elif _isarr(xs[0]):
        return False
    else:
        return _all_scalar(*xs[1:])

def _size(x) -> int:
    return int(np.asarray(x).size)

def _sizes(args) -> int:
    if static.len(args) == 0:
        return 1
    return max(_size(args[0]), _sizes(args[1:]))

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

def _shp2(args, k: int, c: int):
    # eim._loop reshapes the active-element args to (n_active, 1)
    if static.len(args) == 0:
        return ()
    o = np.empty((1, 1))
    o[0, 0] = args[0][k]
    return (o,) + _shp2(args[1:], k, c)

def _div(a: float, b: float) -> float:
    if b == 0.0:
        if a == 0.0 or a != a:
            return _NAN
        neg = (a < 0.0) != (math.copysign(1.0, b) < 0.0)
        return -_INF if neg else _INF
    return a / b

def _fnan(a: float, b: float) -> float:
    if a != a or b != b:
        return _NAN
    return a if a > b else b

def _fin(a: float) -> bool:
    return a == a and a != _INF and a != -_INF

# ------------------------------------------------------------------ nodes
def _base_step() -> float:
    fmin = 4.0 * _TINY
    tmax = math.asinh(math.log(2.0 / fmin - 1.0) / math.pi)
    return tmax / _N_BASE_STEPS

def _compute_pair(k: int, h0: float):
    h = h0 / (2.0 ** k)
    mx = _N_BASE_STEPS * (2 ** k)
    js = List[int]()
    if k == 0:
        for j in range(mx + 1):
            js.append(j)
    else:
        j = 1
        while j <= mx:
            js.append(j)
            j += 2
    n = len(js)
    xjc = [0.0] * n
    wj = [0.0] * n
    for i in range(n):
        jh = js[i] * h
        u1 = (math.pi / 2.0) * math.cosh(jh)
        u2 = (math.pi / 2.0) * math.sinh(jh)
        c2 = math.cosh(u2)
        wj[i] = u1 / (c2 * c2)
        xjc[i] = 1.0 / (math.exp(u2) * c2)
    if k == 0:
        wj[0] = wj[0] / 2.0
    return xjc, wj

class _Pairs:
    xjc: List[float]
    wj: List[float]
    indices: List[int]
    h0: float
    def __init__(self, h0: float):
        self.xjc = List[float]()
        self.wj = List[float]()
        self.indices = [0]
        self.h0 = h0

    def ensure(self, k: int):
        i = len(self.indices) - 1
        while i < k + 1:
            p = _compute_pair(i, self.h0)
            for v in p[0]:
                self.xjc.append(v)
            for v in p[1]:
                self.wj.append(v)
            self.indices.append(self.indices[len(self.indices) - 1] + len(p[0]))
            i += 1

# ------------------------------------------------------------------ one element
class _TSRes:
    integral: float
    error: float
    status: int
    nfev: int
    maxlevel: int
    def __init__(self, integral: float, error: float, status: int, nfev: int, maxlevel: int):
        self.integral = integral
        self.error = error
        self.status = status
        self.nfev = nfev
        self.maxlevel = maxlevel

def _ts_one(f, a0: float, b0: float, args, k: int, pairs: _Pairs, maxlevel: int, minlevel: int,
            atol: float, rtol: float) -> _TSRes:
    h0 = pairs.h0
    a = a0
    b = b0
    # integration point for the initial evaluation
    c = (a + b) / 2.0
    if a == _INF or a == -_INF or b == _INF or b == -_INF:
        if a == -_INF:
            c = b - 1.0
        if b == _INF:
            c = a + 1.0
        if a == -_INF and b == _INF:
            c = 0.0
    xin = np.empty((1, 1))
    xin[0, 0] = c
    f0r = np.asarray(f(xin, *_shp2(args, k, 1))).astype(float)
    if f0r.size != 1:
        raise ValueError("The shape of the array returned by `func` must be the same as the broadcasted shape of `x` and all other `args`.")
    f0 = f0r.reshape(1)[0]
    # transform integrals
    if a == b:
        a = 1.0
        b = 1.0
    ab_same = a0 == b0
    negative = b < a
    if negative:
        t = a
        a = b
        b = t
    abinf = (a == _INF or a == -_INF) and (b == _INF or b == -_INF)
    if abinf:
        a = -1.0
        b = 1.0
    ainf = a == -_INF or a == _INF
    if ainf:
        t = a
        a = -b
        b = -t
    binf = b == _INF or b == -_INF
    a_init = a
    if binf:
        a = 0.0
        b = 1.0
    Sn = 0.0
    if a != a or b != b or f0 != f0:
        Sn = _NAN
    aerr = _NAN
    status = 1
    xr0 = -_INF
    fr0 = _NAN
    wr0 = 0.0
    xl0 = _INF
    fl0 = _NAN
    wl0 = 0.0
    d4 = 0.0
    Sk = List[float]()
    maxiter = maxlevel - minlevel + 1
    n = minlevel
    nit = 0
    nfev = 1
    fjwj = List[float]()
    # initial termination check (nit == 0)
    stopped = False
    if ab_same:
        Sn = 0.0 if Sn == Sn else _NAN
        aerr = Sn
        status = 0
        stopped = True
    if not stopped and not _fin(Sn):
        status = -3
        stopped = True
    while not stopped and nit < maxiter:
        h = h0 / (2.0 ** n)
        inclusive = n == minlevel
        pairs.ensure(n)
        start = 0 if inclusive else pairs.indices[n]
        end = pairs.indices[n + 1]
        m = end - start
        alpha = (b - a) / 2.0
        nx = 2 * m
        xj = [0.0] * nx
        wjv = [0.0] * nx
        for i in range(m):
            xjc = pairs.xjc[start + i]
            w = pairs.wj[start + i]
            xj[i] = -alpha * xjc + b
            xj[m + i] = alpha * xjc + a
            wjv[i] = w * alpha
            wjv[m + i] = w * alpha
        for i in range(nx):
            if xj[i] <= a or xj[i] >= b:
                wjv[i] = 0.0
        xin2 = np.empty((1, nx))
        for i in range(nx):
            v = xj[i]
            if abinf:
                v = _div(v, 1.0 - v * v)
            if binf:
                v = _div(1.0, v) - 1.0 + a_init
            if ainf:
                v = -v
            xin2[0, i] = v
        fr_ = np.asarray(f(xin2, *_shp2(args, k, nx))).astype(float)
        if fr_.size != nx:
            raise ValueError("The shape of the array returned by `func` must be the same as the broadcasted shape of `x` and all other `args`.")
        fv = fr_.reshape(nx)
        fj = [fv[i] for i in range(nx)]
        for i in range(nx):
            if abinf:
                dd = 1.0 - xj[i] * xj[i]
                fj[i] = fj[i] * _div(1.0 + xj[i] * xj[i], dd * dd)
            elif binf:
                fj[i] = fj[i] * _div(1.0, xj[i] * xj[i])
        nfev += nx
        # Euler-Maclaurin bookkeeping (right half r = [0, m), left half l = [m, 2m))
        inv_r = [False] * m
        inv_l = [False] * m
        for i in range(m):
            inv_r[i] = (not _fin(fj[i])) or wjv[i] == 0.0
            inv_l[i] = (not _fin(fj[m + i])) or wjv[m + i] == 0.0
        ir = 0
        best = -_INF
        for i in range(m):
            v = -_INF if inv_r[i] else xj[i]
            if i == 0 or v > best:
                best = v
                ir = i
        if best > xr0:
            xr0 = best
            fr0 = fj[ir]
            wr0 = wjv[ir]
        il = 0
        bestl = _INF
        for i in range(m):
            v = _INF if inv_l[i] else xj[m + i]
            if i == 0 or v < bestl:
                bestl = v
                il = i
        if bestl < xl0:
            xl0 = bestl
            fl0 = fj[m + il]
            wl0 = wjv[m + il]
        d4 = _fnan(abs(fl0 * wl0), abs(fr0 * wr0))
        for i in range(m):
            if inv_r[i]:
                fj[i] = fr0
            if inv_l[i]:
                fj[m + i] = fl0
        fjwj = [fj[i] * wjv[i] for i in range(nx)]
        s = 0.0
        for i in range(nx):
            s += fjwj[i]
        Sn = s * h
        if len(Sk) > 0:
            Sn = Sk[len(Sk) - 1] / 2.0 + Sn
        nit += 1
        # termination check
        stop = False
        # error estimate
        rerr = _NAN
        aerr_ = _NAN
        if not (n == 0 or nit == 0):
            if len(Sk) == 0:
                hh = 2.0 * h
                nxx = pairs.indices[n]
                sacc = 0.0
                per = nx // 2
                for i in range(nxx):
                    sacc += fjwj[i]
                    sacc += fjwj[per + i]
                Sk.append(sacc * hh)
            if n != 1:
                if len(Sk) < 2:
                    hh = 4.0 * h
                    nxx = pairs.indices[n - 1]
                    sacc = 0.0
                    per = nx // 2
                    for i in range(nxx):
                        sacc += fjwj[i]
                        sacc += fjwj[per + i]
                    Sk.insert(0, sacc * hh)
                Snm2 = Sk[len(Sk) - 2]
                Snm1 = Sk[len(Sk) - 1]
                d1 = abs(Sn - Snm1)
                d2 = abs(Sn - Snm2)
                mx = 0.0
                for i in range(nx):
                    v = abs(fjwj[i])
                    if v != v:
                        mx = _NAN
                        break
                    if v > mx:
                        mx = v
                d3 = _EPS * mx
                d5 = _EPS * abs(Sn)
                temp = 0.0
                if d1 > 0:
                    temp = d1 ** _div(math.log(d1), math.log(d2)) if d2 > 0 else _NAN
                mxd = _fnan(_fnan(temp, d1 * d1), _fnan(d3, d4))
                ae = mxd
                if ae != ae:
                    ae = _NAN
                else:
                    if ae < d5:
                        ae = d5
                    if ae > d1:
                        ae = d1
                aerr_ = ae
                rerr = _div(ae, abs(Sn))
        aerr = aerr_
        if rerr < rtol or aerr_ < atol:
            status = 0
            stop = True
        if not _fin(Sn) and not stop:
            status = -3
            stop = True
        if stop:
            stopped = True
            break
        n += 1
        Sk.append(Sn)
    if not stopped:
        status = -2
    if negative:
        Sn = -Sn
    ml = minlevel + nit - 1
    if nit == 0:
        ml = -1
    return _TSRes(Sn, aerr, status, nfev, ml)

class TSResult[S, I, E, N]:
    success: S
    status: I
    integral: E
    error: E
    nfev: N
    maxlevel: I

    def __init__(self, success: S, status: I, integral: E, error: E, nfev: N, maxlevel: I):
        self.success = success
        self.status = status
        self.integral = integral
        self.error = error
        self.nfev = nfev
        self.maxlevel = maxlevel

def _out_f(v: List[float], sc: Literal[bool]):
    if sc:
        return v[0]
    o = np.empty(len(v))
    for i in range(len(v)):
        o[i] = v[i]
    return o

def _out_i(v: List[int], sc: Literal[bool]):
    if sc:
        return v[0]
    o = np.empty(len(v), dtype=int)
    for i in range(len(v)):
        o[i] = v[i]
    return o

def _out_b(v: List[int], sc: Literal[bool]):
    if sc:
        return v[0] == 0
    o = np.empty(len(v), dtype=bool)
    for i in range(len(v)):
        o[i] = v[i] == 0
    return o

def tanhsinh(f, a, b, *, args=(), kwargs=None, log: bool = False, maxlevel: Optional[int] = None, minlevel: int = 2,
             atol: Optional[float] = None, rtol: Optional[float] = None, preserve_shape: bool = False, callback=None):
    if not callable(f):
        raise ValueError('`f` must be callable.')
    if kwargs is not None or callback is not None or preserve_shape:
        raise ValueError("`kwargs`, `callback` and `preserve_shape` are not supported")
    if log:
        raise ValueError("`log=True` is not supported")
    if not isinstance(args, Tuple):
        raise ValueError("`args` must be a tuple")
    at = 0.0 if atol is None else atol
    if at < 0 or at != at or at == _INF:
        raise ValueError('`atol` and `rtol` must be non-negative and finite.')
    rt = _EPS ** 0.75 if rtol is None else rtol
    if rt < 0 or rt != rt or rt == _INF:
        raise ValueError('`atol` and `rtol` must be non-negative and finite.')
    ml = 10 if maxlevel is None else maxlevel
    if ml < 0 or minlevel < 0:
        raise ValueError('`maxfun`, `maxlevel`, and `minlevel` must be non-negative.')
    mnl = min(minlevel, ml)
    n = max(max(_size(a), _size(b)), _sizes(args))
    SC: Literal[bool] = _all_scalar(a, b, *args)
    av = _bflat(a, n)
    bv = _bflat(b, n)
    ba = _bargs(args, n)
    pairs = _Pairs(_base_step())
    integ = List[float]()
    err = List[float]()
    st = List[int]()
    nf = List[int]()
    mls = List[int]()
    for k in range(n):
        r = _ts_one(f, av[k], bv[k], ba, k, pairs, ml, mnl, at, rt)
        integ.append(r.integral)
        err.append(r.error)
        st.append(r.status)
        nf.append(r.nfev)
        mls.append(r.maxlevel)
    return TSResult(_out_b(st, SC), _out_i(st, SC), _out_f(integ, SC), _out_f(err, SC), _out_i(nf, SC), _out_i(mls, SC))

# ------------------------------------------------------------------ nsum
class _NegF[F]:
    f: F
    def __init__(self, f: F):
        self.f = f
    def __call__(self, x, *args):
        return self.f(-x, *args)

class _SymF[F]:
    f: F
    def __init__(self, f: F):
        self.f = f
    def __call__(self, x, *args):
        a = np.asarray(self.f(x, *args)).astype(float)
        b = np.asarray(self.f(-x, *args)).astype(float)
        fac = np.where(x == 0, 0.5, 1.0)
        return (a + b) * fac

def _call_arr(f, xs: List[float], args, k: int):
    # f is always called with x of shape (1, m) and args of shape (1, 1)
    m = len(xs)
    x = np.empty((1, m))
    for i in range(m):
        x[0, i] = xs[i]
    r = np.asarray(f(x, *_shp2(args, k, m))).astype(float)
    if r.size != m:
        raise ValueError("The shape of the array returned by `func` must be the same as the broadcasted shape of `x` and all other `args`.")
    rv = r.reshape(m)
    out = [0.0] * m
    for i in range(m):
        out[i] = rv[i]
    return out

def _direct_one(f, a: float, b: float, step: float, args, k: int, inclusive: bool, eps: float):
    adj = 1 if inclusive else 0
    steps = int(round((b - a) / step)) + adj
    xs = List[float]()
    zero_mask = List[bool]()
    for i in range(steps):
        kk = a + i * step
        nanned = kk >= (b + adj * step / 2.0)
        xs.append(_NAN if nanned else kk)
        zero_mask.append(nanned)
    nfev = 0
    S = 0.0
    if steps > 0:
        fs = _call_arr(f, xs, args, k)
        for i in range(steps):
            if zero_mask[i]:
                fs[i] = 0.0
            else:
                nfev += 1
        for i in range(steps):
            S += fs[i]
    return S, eps * abs(S), nfev

def _integral_bound_one(f, a: float, b: float, step: float, args, k: int, pairs: _Pairs, atol: float, rtol: float,
                        maxterms: int, eps: float):
    lb = _ts_one(f, a, b, args, k, pairs, 10, 2, atol, rtol)
    tol = atol + rtol * lb.integral
    i_skip = lb.status == -3
    if i_skip:
        tol = _NAN
    status = lb.status
    log2m = int(math.floor(math.log2(float(maxterms)))) if maxterms > 0 else 0
    n_steps = List[float]()
    for i in range(log2m):
        n_steps.append(2.0 ** i)
    n_steps.append(float(maxterms))
    L = len(n_steps)
    nfev = L * 2
    ks = [a + n_steps[i] * step for i in range(L)]
    ksp = [ks[i] + step for i in range(L)]
    fks = _call_arr(f, ks, args, k)
    fksp = _call_arr(f, ksp, args, k)
    n_ins = 0
    for i in range(L):
        if fks[i] > tol or fksp[i] > fks[i]:
            n_ins += 1
    nt = min(n_ins, L - 1)
    ns = n_steps[nt]
    i_fk_ins = n_ins == nfev // 2
    kk = a + ns * step
    d = _direct_one(f, a, kk, step, args, k, False, eps)
    left = d[0]
    left_err = d[1]
    left_nfev = d[2]
    if left == _INF:
        i_skip = True
        status = -3
    if i_skip:
        kk = _NAN
    right = _ts_one(f, kk, b, args, k, pairs, 10, 2, atol, rtol)
    fk = fks[nt]
    fb = 0.0
    if _fin(b):
        bx = List[float]()
        bx.append(b)
        fb = _call_arr(f, bx, args, k)[0]
        nfev += 1
    S = left + right.integral / step + fk / 2.0 + fb / 2.0
    E = left_err + right.error / step + fk / 2.0 - fb / 2.0
    if not i_skip:
        status = right.status
    if status == 0 and i_fk_ins:
        status = -4
    return S, E, status, left_nfev + right.nfev + nfev + lb.nfev

class NSumResult[S, I, E, N]:
    success: S
    status: I
    sum: E
    error: E
    nfev: N

    def __init__(self, success: S, status: I, sum: E, error: E, nfev: N):
        self.success = success
        self.status = status
        self.sum = sum
        self.error = error
        self.nfev = nfev

def nsum(f, a, b, *, step=1, args=(), kwargs=None, log: bool = False, maxterms: int = 1048576, tolerances=None):
    if not callable(f):
        raise ValueError('`f` must be callable.')
    if kwargs is not None:
        raise ValueError("`kwargs` are not supported")
    if log:
        raise ValueError("`log=True` is not supported")
    if not isinstance(args, Tuple):
        raise ValueError("`args` must be a tuple")
    if maxterms < 0:
        raise ValueError("`maxterms` must be a non-negative integer.")
    atol = 0.0
    rtol = _EPS ** 0.5
    if tolerances is not None:
        if 'atol' in tolerances:
            atol = float(tolerances['atol'])
        if 'rtol' in tolerances:
            rtol = float(tolerances['rtol'])
    if atol < 0 or rtol < 0 or atol != atol or rtol != rtol or atol == _INF or rtol == _INF:
        raise ValueError('`atol`, and `rtol` must be non-negative and finite.')
    n = max(max(_size(a), _size(b)), max(_size(step), _sizes(args)))
    SC: Literal[bool] = _all_scalar(a, b, step, *args)
    av = _bflat(a, n)
    bv = _bflat(b, n)
    sv = _bflat(step, n)
    ba = _bargs(args, n)
    pairs = _Pairs(_base_step())
    Sv = List[float]()
    Ev = List[float]()
    St = List[int]()
    Nf = List[int]()
    for k in range(n):
        ak = av[k]
        bk = bv[k]
        sk = sv[k]
        # initial evaluation f(a) (shape / dtype check)
        a1 = List[float]()
        a1.append(ak)
        _call_arr(f, a1, ba, k)
        nfev = 1
        valid = (bk >= ak) and _fin(sk) and sk > 0
        S = 0.0
        E = 0.0
        status = 0
        if not valid:
            S = _NAN
            E = _NAN
            status = -1
            if bk < ak:
                S = 0.0
                E = 0.0
                status = 0
        else:
            nterms = math.floor((bk - ak) / sk) if _fin((bk - ak) / sk) else _INF
            bb = bk
            if _fin(nterms):
                bb = ak + nterms * sk
            if nterms + 1 <= float(maxterms):
                d = _direct_one(f, ak, bb, sk, ba, k, True, _EPS)
                S = d[0]
                E = d[1]
                nfev += d[2]
                status = 0 if _fin(S) else -3
            elif _fin(ak):
                r = _integral_bound_one(f, ak, bb, sk, ba, k, pairs, atol, rtol, maxterms, _EPS)
                S = r[0]
                E = r[1]
                status = r[2]
                nfev += r[3]
            elif _fin(bb):
                r = _integral_bound_one(_NegF(f), -bb, -ak, sk, ba, k, pairs, atol, rtol, maxterms, _EPS)
                S = r[0]
                E = r[1]
                status = r[2]
                nfev += r[3]
            else:
                r = _integral_bound_one(_SymF(f), 0.0, bb, sk, ba, k, pairs, atol, rtol, maxterms, _EPS)
                S = r[0]
                E = r[1]
                status = r[2]
                nfev += 2 * r[3]
        Sv.append(S)
        Ev.append(E)
        St.append(status)
        Nf.append(nfev)
    return NSumResult(_out_b(St, SC), _out_i(St, SC), _out_f(Sv, SC), _out_f(Ev, SC), _out_i(Nf, SC))
