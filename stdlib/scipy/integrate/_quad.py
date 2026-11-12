# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# quad / dblquad / tplquad / fixed_quad. `quad` returns (y, abserr) only;
# full_output, weight and complex_func are not supported. QUADPACK warnings
# (ier 1..5) are not emitted.

import math
import numpy as np
from _quadpack import qagse, qagie, qagpe

_INF = float('inf')

def _call_args(func, x: float, args):
    if isinstance(args, Tuple):
        return float(func(x, *args))
    else:
        return float(func(x, args))

def quad(func, a, b, args=(), full_output: int = 0, epsabs: float = 1.49e-8,
         epsrel: float = 1.49e-8, limit: int = 50, points=None):
    if full_output != 0:
        raise ValueError("full_output is not supported")
    af = float(a)
    bf = float(b)
    if af == bf:
        return (0.0, 0.0)
    flip = bf < af
    lo = min(af, bf)
    hi = max(af, bf)
    f = lambda x: _call_args(func, x, args)
    res = 0.0
    err = 0.0
    if points is None:
        if hi != _INF and lo != -_INF:
            res, err, _n, _ier = qagse(f, lo, hi, epsabs, epsrel, limit)
        elif hi == _INF and lo != -_INF:
            res, err, _n, _ier = qagie(f, lo, 1, epsabs, epsrel, limit)
        elif hi == _INF and lo == -_INF:
            res, err, _n, _ier = qagie(f, 0.0, 2, epsabs, epsrel, limit)
        else:
            res, err, _n, _ier = qagie(f, hi, -1, epsabs, epsrel, limit)
    else:
        if hi == _INF or lo == -_INF:
            raise ValueError("Infinity inputs cannot be used with break points.")
        pl = sorted(set([float(p) for p in points]))
        pts = [p for p in pl if lo < p and p < hi]
        res, err, _n, _ier = qagpe(f, lo, hi, pts, epsabs, epsrel, limit)
    if flip:
        res = -res
    return (res, err)

def _bound1(g, x: float) -> float:
    if isinstance(g, int) or isinstance(g, float):
        return float(g)
    else:
        return float(g(x))

def _bound2(g, y: float, x: float) -> float:
    if isinstance(g, int) or isinstance(g, float):
        return float(g)
    else:
        return float(g(x, y))

def dblquad(func, a, b, gfun, hfun, args=(), epsabs: float = 1.49e-8,
            epsrel: float = 1.49e-8):
    # nquad semantics: inner integral over y for each x; abserr is the max.
    st = [0.0]
    def inner(x: float) -> float:
        lo = _bound1(gfun, x)
        hi = _bound1(hfun, x)
        r, e = quad(lambda y: float(func(y, x, *args)), lo, hi,
                    epsabs=epsabs, epsrel=epsrel)
        st[0] = max(st[0], e)
        return r
    r, e = quad(inner, a, b, epsabs=epsabs, epsrel=epsrel)
    return (r, max(st[0], e))

def tplquad(func, a, b, gfun, hfun, qfun, rfun, args=(), epsabs: float = 1.49e-8,
            epsrel: float = 1.49e-8):
    st = [0.0]
    def mid(x: float) -> float:
        def innermost(y: float) -> float:
            lo = _bound2(qfun, y, x)
            hi = _bound2(rfun, y, x)
            r, e = quad(lambda z: float(func(z, y, x, *args)), lo, hi,
                        epsabs=epsabs, epsrel=epsrel)
            st[0] = max(st[0], e)
            return r
        r, e = quad(innermost, _bound1(gfun, x), _bound1(hfun, x),
                    epsabs=epsabs, epsrel=epsrel)
        st[0] = max(st[0], e)
        return r
    r, e = quad(mid, a, b, epsabs=epsabs, epsrel=epsrel)
    return (r, max(st[0], e))

def _roots_legendre(n: int):
    # Golub-Welsch style Newton iteration on P_n (matches scipy to ~1 ulp).
    x = [0.0] * n
    w = [0.0] * n
    m = (n + 1) // 2
    for i in range(m):
        z = math.cos(math.pi * (i + 0.75) / (n + 0.5))
        pp = 0.0
        for _ in range(100):
            p1 = 1.0
            p2 = 0.0
            for j in range(n):
                p3 = p2
                p2 = p1
                p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1.0)
            pp = n * (z * p1 - p2) / (z * z - 1.0)
            z1 = z
            z = z1 - p1 / pp
            if abs(z - z1) <= 1e-16:
                break
        p1 = 1.0
        p2 = 0.0
        for j in range(n):
            p3 = p2
            p2 = p1
            p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1.0)
        pp = n * (z * p1 - p2) / (z * z - 1.0)
        x[i] = -z
        x[n - 1 - i] = z
        w[i] = 2.0 / ((1.0 - z * z) * pp * pp)
        w[n - 1 - i] = w[i]
    if n % 2 == 1:
        x[n // 2] = 0.0
    return np.array(x), np.array(w)

def fixed_quad(func, a, b, args=(), n: int = 5):
    af = float(a)
    bf = float(b)
    if math.isinf(af) or math.isinf(bf):
        raise ValueError("Gaussian quadrature is only available for finite limits.")
    x, w = _roots_legendre(n)
    y = (bf - af) * (x + 1.0) / 2.0 + af
    fy = np.asarray(func(y, *args))
    return ((bf - af) / 2.0 * float(np.sum(w * fy)), None)
