# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# quad / dblquad / tplquad / nquad / fixed_quad / newton_cotes.
# quad supports points, weight ('cos', 'sin', 'alg', 'alg-loga', 'alg-logb',
# 'alg-log', 'cauchy'), wvar, maxp1, limlst, complex_func and full_output
# (pass a literal 1). Deviations: with full_output the result is always
# (y, abserr, info); scipy appends a 4th `message` element when ier != 0, here
# it is `info['message']` ('' on success); `info` is a QuadInfo object with
# dict-style access using literal keys. wopts (precomputed moments) is not
# supported. IntegrationWarning messages are printed (warnings filters do not
# apply).

import math
import numpy as np
from _quadpack import qagse, qagie, qagpe, _LAST
from _quadpack_w import qawce, qawse, qawoe, qawfe, _Moments

_INF = float('inf')

def _iwarn(msg: str):
    # IntegrationWarning (printed; the warnings module is not used so that
    # filters do not apply)
    print(f"IntegrationWarning: {msg}")

class IntegrationWarning(Exception):
    def __init__(self, message: str = ""):
        super().__init__(message)

class QuadInfo:
    neval: int
    last: int
    alist: np.ndarray[float, 1]
    blist: np.ndarray[float, 1]
    rlist: np.ndarray[float, 1]
    elist: np.ndarray[float, 1]
    iord: np.ndarray[int, 1]
    level: np.ndarray[int, 1]
    ndin: np.ndarray[int, 1]
    pts: np.ndarray[float, 1]
    lst: int
    rslst: np.ndarray[float, 1]
    erlst: np.ndarray[float, 1]
    ierlst: np.ndarray[int, 1]
    momcom: int
    chebmo: np.ndarray[float, 2]
    message: str
    ier: int

    def __init__(self, limit: int):
        self.neval = 0
        self.last = 0
        self.alist = np.zeros(limit)
        self.blist = np.zeros(limit)
        self.rlist = np.zeros(limit)
        self.elist = np.zeros(limit)
        self.iord = np.zeros(limit, dtype=int)
        self.level = np.zeros(0, dtype=int)
        self.ndin = np.zeros(0, dtype=int)
        self.pts = np.zeros(0)
        self.lst = 0
        self.rslst = np.zeros(0)
        self.erlst = np.zeros(0)
        self.ierlst = np.zeros(0, dtype=int)
        self.momcom = 0
        self.chebmo = np.zeros((0, 0))
        self.message = ''
        self.ier = 0

    def __getitem__(self, key: Literal[str]):
        if key == 'neval':
            return self.neval
        elif key == 'last':
            return self.last
        elif key == 'alist':
            return self.alist
        elif key == 'blist':
            return self.blist
        elif key == 'rlist':
            return self.rlist
        elif key == 'elist':
            return self.elist
        elif key == 'iord':
            return self.iord
        elif key == 'level':
            return self.level
        elif key == 'ndin':
            return self.ndin
        elif key == 'pts':
            return self.pts
        elif key == 'lst':
            return self.lst
        elif key == 'rslst':
            return self.rslst
        elif key == 'erlst':
            return self.erlst
        elif key == 'ierlst':
            return self.ierlst
        elif key == 'momcom':
            return self.momcom
        elif key == 'chebmo':
            return self.chebmo
        elif key == 'message':
            return self.message
        else:
            compile_error("unknown quad infodict key")

def _call_args(func, x: float, args):
    if isinstance(args, Tuple):
        return float(func(x, *args))
    else:
        return float(func(x, args))

def _call_c(func, x: float, args) -> complex:
    if isinstance(args, Tuple):
        return complex(func(x, *args))
    else:
        return complex(func(x, args))
_MSGS = {1: "The maximum number of subdivisions ({limit}) has been achieved.\n  If increasing the limit yields no improvement it is advised to analyze \n  the integrand in order to determine the difficulties.  If the position of a \n  local difficulty can be determined (singularity, discontinuity) one will \n  probably gain from splitting up the interval and calling the integrator \n  on the subranges.  Perhaps a special-purpose integrator should be used.",
         2: "The occurrence of roundoff error is detected, which prevents \n  the requested tolerance from being achieved.  The error may be \n  underestimated.",
         3: "Extremely bad integrand behavior occurs at some points of the\n  integration interval.",
         4: "The algorithm does not converge.  Roundoff error is detected\n  in the extrapolation table.  It is assumed that the requested tolerance\n  cannot be achieved, and that the returned result (if full_output = 1) is \n  the best which can be obtained.",
         5: "The integral is probably divergent, or slowly convergent.",
         6: "The input is invalid.",
         7: "Abnormal termination of the routine.  The estimates for result\n  and error are less reliable.  It is assumed that the requested accuracy\n  has not been achieved."}
_MSGS_F = {1: "The maximum number of cycles allowed has been achieved., e.e.\n  of subintervals (a+(k-1)c, a+kc) where c = (2*int(abs(omega)+1))\n  *pi/abs(omega), for k = 1, 2, ..., lst.  One can allow more cycles by increasing the value of limlst.  Look at info['ierlst'] with full_output=1.",
           4: "The extrapolation table constructed for convergence acceleration\n  of the series formed by the integral contributions over the cycles, \n  does not converge to within the requested accuracy.  Look at \n  info['ierlst'] with full_output=1.",
           7: "Bad integrand behavior occurs within one or more of the cycles.\n  Location and type of the difficulty involved can be determined from \n  the vector info['ierlist'] obtained with full_output=1."}

class _Neg[F]:
    # x -> sg * f(-x) (a class instead of a lambda capturing a lambda)
    f: F
    sg: float

    def __init__(self, f: F, sg: float):
        self.f = f
        self.sg = sg

    def __call__(self, x: float) -> float:
        return self.sg * self.f(-x)

class _Arg[F, A]:
    # x -> float(func(x, *args))
    func: F
    args: A

    def __init__(self, func: F, args: A):
        self.func = func
        self.args = args

    def __call__(self, x: float) -> float:
        return _call_args(self.func, x, self.args)

class _ArgC[F, A]:
    # x -> complex(func(x, *args)).real / .imag
    func: F
    args: A
    imag: bool

    def __init__(self, func: F, args: A, imag: bool):
        self.func = func
        self.args = args
        self.imag = imag

    def __call__(self, x: float) -> float:
        v = _call_c(self.func, x, self.args)
        return v.imag if self.imag else v.real

def _wv1(wvar) -> float:
    if wvar is None:
        raise ValueError("wvar must be given for this weight")
        return 0.0
    elif isinstance(wvar, Tuple) or isinstance(wvar, List):
        raise ValueError("wvar must be a scalar for this weight")
        return 0.0
    else:
        return float(wvar)

def _wv2(wvar) -> Tuple[float, float]:
    if isinstance(wvar, Tuple) or isinstance(wvar, List):
        return (float(wvar[0]), float(wvar[1]))
    else:
        raise ValueError("wvar must be (alpha, beta) for 'alg' weights")
        return (0.0, 0.0)

def _fill_std(info: QuadInfo, limit: int, ndin_too: bool):
    li = _LAST[0]
    s = li.s
    info.last = li.last
    m = len(s.alist) - 1
    for k in range(limit):
        if k + 1 <= m:
            info.alist[k] = s.alist[k + 1]
            info.blist[k] = s.blist[k + 1]
            info.rlist[k] = s.rlist[k + 1]
            info.elist[k] = s.elist[k + 1]
            v = s.iord[k + 1]
            info.iord[k] = v - 1 if v > 0 else 0
    if ndin_too:
        info.level = np.zeros(limit, dtype=int)
        for k in range(limit):
            if k + 1 <= m:
                info.level[k] = s.level[k + 1]

def _fill_w(info: QuadInfo, W, limit: int):
    for k in range(limit):
        info.alist[k] = W.alist[k]
        info.blist[k] = W.blist[k]
        info.rlist[k] = W.rlist[k]
        info.elist[k] = W.elist[k]
        info.iord[k] = W.iord[k]

def _quad_real(f, a: float, b: float, full_output: int, epsabs: float, epsrel: float,
               limit: int, points, weight: Optional[str], wvar, maxp1: int, limlst: int,
               info: QuadInfo):
    # a < b; returns (result, abserr, ier, fourier)
    res = 0.0
    err = 0.0
    ier = 0
    neval = 0
    fourier = False
    if weight is None:
        if points is None:
            if b != _INF and a != -_INF:
                res, err, neval, ier = qagse(f, a, b, epsabs, epsrel, limit)
            elif b == _INF and a != -_INF:
                res, err, neval, ier = qagie(f, a, 1, epsabs, epsrel, limit)
            elif b == _INF and a == -_INF:
                res, err, neval, ier = qagie(f, 0.0, 2, epsabs, epsrel, limit)
            else:
                res, err, neval, ier = qagie(f, b, -1, epsabs, epsrel, limit)
            if full_output:
                _fill_std(info, limit, False)
        else:
            if b == _INF or a == -_INF:
                raise ValueError("Infinity inputs cannot be used with break points.")
            pl = sorted(set([float(p) for p in points]))
            pts = [p for p in pl if a < p and p < b]
            res, err, neval, ier = qagpe(f, a, b, pts, epsabs, epsrel, limit)
            if full_output:
                _fill_std(info, limit, True)
                npts2 = len(pts) + 2
                info.pts = np.zeros(npts2)
                info.pts[0] = a
                for k in range(len(pts)):
                    info.pts[k + 1] = pts[k]
                info.pts[npts2 - 1] = b
                nd = _LAST[0].ndin
                info.ndin = np.zeros(npts2, dtype=int)
                for k in range(npts2):
                    if k + 1 < len(nd):
                        info.ndin[k] = nd[k + 1]
        info.neval = neval
        return res, err, ier, fourier
    w: str = weight if weight is not None else ''
    if w not in ('cos', 'sin', 'alg', 'alg-loga', 'alg-logb', 'alg-log', 'cauchy'):
        raise ValueError(f"{w} not a recognized weighting function.")
    if points is not None:
        _iwarn("Break points cannot be specified when using weighted integrand.\nContinuing, ignoring specified points.")
    if w == 'cos' or w == 'sin':
        integr = 1 if w == 'cos' else 2
        omega = _wv1(wvar)
        if b != _INF and a != -_INF:
            M = _Moments(maxp1)
            res, err, neval, ier, last, W = qawoe(f, a, b, omega, integr, epsabs, epsrel, limit, 1, maxp1, M)
            if ier == 6:
                raise ValueError("Chebyshev moment limit maxp1 must be >=1." if maxp1 < 1 else _MSGS[6])
            if full_output:
                _fill_w(info, W, limit)
                info.last = last
                info.momcom = M.momcom
                info.chebmo = np.zeros((25, maxp1))
                for i in range(maxp1):
                    for j in range(25):
                        info.chebmo[j, i] = M.chebmo[i + maxp1 * j]
        elif a == -_INF and b == _INF:
            raise ValueError("Cannot integrate with this weight from -Inf to +Inf.")
        else:
            fourier = True
            if b == _INF:
                res, err, neval, ier, O = qawfe(f, a, omega, integr, epsabs, limlst, limit, maxp1)
            else:
                g = _Neg(f, 1.0 if integr == 1 else -1.0)
                res, err, neval, ier, O = qawfe(g, -b, omega, integr, epsabs, limlst, limit, maxp1)
            if ier == 6:
                if epsabs <= 0:
                    raise ValueError("Sine or cosine weighted integrals with infinite domain must have 'epsabs'>0.")
                raise ValueError("Cycle limit limlst must be >=3." if limlst < 3 else _MSGS[6])
            if full_output:
                info.lst = O.lst
                info.rslst = np.zeros(limlst)
                info.erlst = np.zeros(limlst)
                info.ierlst = np.zeros(limlst, dtype=int)
                for k in range(min(limlst, len(O.rslst))):
                    info.rslst[k] = O.rslst[k]
                    info.erlst[k] = O.erlst[k]
                    info.ierlst[k] = O.ierlst[k]
        info.neval = neval
        return res, err, ier, fourier
    if a == -_INF or a == _INF or b == -_INF or b == _INF:
        raise ValueError("Cannot integrate with this weight over an infinite interval.")
    if w == 'cauchy':
        c = _wv1(wvar)
        res, err, neval, ier, last, W = qawce(f, a, b, c, epsabs, epsrel, limit)
        if ier == 6:
            if c == a or c == b:
                raise ValueError("Parameter 'wvar' must not equal integration limits 'a' or 'b'.")
            raise ValueError(_MSGS[6])
    else:
        alfa, beta = _wv2(wvar)
        integr = 1 if w == 'alg' else (2 if w == 'alg-loga' else (3 if w == 'alg-logb' else 4))
        res, err, neval, ier, last, W = qawse(f, a, b, alfa, beta, integr, epsabs, epsrel, limit)
        if ier == 6:
            if min(alfa, beta) < -1:
                raise ValueError("wvar parameters (alpha, beta) must both be >= -1.")
            raise ValueError(_MSGS[6])
    if full_output:
        _fill_w(info, W, limit)
        info.last = last
    info.neval = neval
    return res, err, ier, fourier

def _finish(ier: int, limit: int, fourier: bool, full_output: int, info: QuadInfo):
    if ier == 0:
        return
    msg = ''
    if fourier and ier in _MSGS_F:
        msg = _MSGS_F[ier]
    elif ier in _MSGS:
        msg = _MSGS[ier].replace("{limit}", str(limit))
    else:
        msg = "Unknown error."
    if ier == 6:
        raise ValueError(msg)
    info.ier = ier
    info.message = msg
    if not full_output:
        _iwarn(msg)

def _quad1(f, af: float, bf: float, full_output: int, epsabs: float, epsrel: float, limit: int,
           points, weight: Optional[str], wvar, maxp1: int, limlst: int):
    info = QuadInfo(limit)
    if af == bf:
        for k in range(limit):
            info.alist[k] = float('nan')
            info.blist[k] = float('nan')
        return 0.0, 0.0, info
    flip = bf < af
    lo = min(af, bf)
    hi = max(af, bf)
    res, err, ier, fourier = _quad_real(f, lo, hi, full_output, epsabs, epsrel, limit, points,
                                        weight, wvar, maxp1, limlst, info)
    if flip:
        res = -res
    _finish(ier, limit, fourier, full_output, info)
    return res, err, info

def quad(func, a, b, args=(), full_output: Literal[int] = 0, epsabs: float = 1.49e-8,
         epsrel: float = 1.49e-8, limit: int = 50, points=None, weight: Optional[str] = None,
         wvar=None, wopts=None, maxp1: int = 50, limlst: int = 50,
         complex_func: Literal[bool] = False):
    if wopts is not None:
        raise ValueError("wopts is not supported")
    af = float(a)
    bf = float(b)
    if complex_func:
        g_re = _ArgC(func, args, False)
        g_im = _ArgC(func, args, True)
        r0, e0, i0 = _quad1(g_re, af, bf, full_output, epsabs, epsrel, limit, points, weight, wvar, maxp1, limlst)
        r1, e1, i1 = _quad1(g_im, af, bf, full_output, epsabs, epsrel, limit, points, weight, wvar, maxp1, limlst)
        if full_output:
            return (complex(r0, r1), complex(e0, e1), (i0, i1))
        else:
            return (complex(r0, r1), complex(e0, e1))
    else:
        f = _Arg(func, args)
        res, err, info = _quad1(f, af, bf, full_output, epsabs, epsrel, limit, points, weight, wvar, maxp1, limlst)
        if full_output:
            return (res, err, info)
        else:
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

# ---------------------------------------------------------------- nquad
# Deviations: up to 4 dimensions, `ranges` must have a static length (tuple,
# or a list of (lo, hi) lists); `opts` is a dict (or list/tuple of dicts,
# innermost first) with float values for 'epsabs', 'epsrel', 'limit'
# (points/weight per level are not supported); ranges are (lo, hi) pairs or
# callables of the outer variables (and args). full_output adds {'neval'} as
# an int in a 3rd tuple element.

def _rng(r, outer, args) -> Tuple[float, float]:
    if isinstance(r, Tuple) or isinstance(r, List):
        return (float(r[0]), float(r[1]))
    else:
        lh = r(*outer, *args)
        return (float(lh[0]), float(lh[1]))

def _opt(opts, level: int, key: str, default: float) -> float:
    if opts is None:
        return default
    elif isinstance(opts, Dict):
        return float(opts.get(key, default))
    else:
        o = opts[level]
        return float(o.get(key, default))

class _NQ[F, R, O, A]:
    # nquad state. Each level integrates an _NQLevel callable (bound methods
    # and lambdas capturing lambdas do not compile as quad integrands).
    func: F
    ranges: R
    opts: O
    ag: A
    n: int
    abserr: float
    neval: int
    x1: float
    x2: float
    x3: float

    def __init__(self, func: F, ranges: R, opts: O, ag: A, n: int):
        self.func = func
        self.ranges = ranges
        self.opts = opts
        self.ag = ag
        self.n = n
        self.abserr = 0.0
        self.neval = 0
        self.x1 = 0.0
        self.x2 = 0.0
        self.x3 = 0.0

    def f0(self, x0: float) -> float:
        n: Literal[int] = static.len(static.function.args(self.func)) - static.len(self.ag)
        if n == 1:
            return float(self.func(x0, *self.ag))
        elif n == 2:
            return float(self.func(x0, self.x1, *self.ag))
        elif n == 3:
            return float(self.func(x0, self.x1, self.x2, *self.ag))
        else:
            return float(self.func(x0, self.x1, self.x2, self.x3, *self.ag))

    def rng(self, k: Literal[int]) -> Tuple[float, float]:
        m: Literal[int] = static.len(static.function.args(self.func)) - static.len(self.ag) - 1 - k
        if m == 0:
            return _rng(self.ranges[k], (), self.ag)
        elif m == 1:
            if k == 0:
                return _rng(self.ranges[k], (self.x1,), self.ag)
            elif k == 1:
                return _rng(self.ranges[k], (self.x2,), self.ag)
            return _rng(self.ranges[k], (self.x3,), self.ag)
        elif m == 2:
            if k == 0:
                return _rng(self.ranges[k], (self.x1, self.x2), self.ag)
            return _rng(self.ranges[k], (self.x2, self.x3), self.ag)
        else:
            return _rng(self.ranges[k], (self.x1, self.x2, self.x3), self.ag)

    def _q(self, g, lv: Literal[int]) -> float:
        lh = self.rng(lv)
        opts = self.opts
        r = quad(g, lh[0], lh[1], (), 1, _opt(opts, lv, 'epsabs', 1.49e-8),
                 _opt(opts, lv, 'epsrel', 1.49e-8), int(_opt(opts, lv, 'limit', 50.0)))
        if lv == 0:
            self.neval += r[2].neval
        self.abserr = max(self.abserr, r[1])
        return r[0]

    def i0(self) -> float:
        return self._q(_NQL0(self), 0)

    def i1(self) -> float:
        return self._q(_NQL1(self), 1)

    def i2(self) -> float:
        return self._q(_NQL2(self), 2)

    def i3(self) -> float:
        return self._q(_NQL3(self), 3)

class _NQL0[S]:
    st: S
    def __init__(self, st: S):
        self.st = st
    def __call__(self, x: float) -> float:
        return self.st.f0(x)

class _NQL1[S]:
    st: S
    def __init__(self, st: S):
        self.st = st
    def __call__(self, x: float) -> float:
        self.st.x1 = x
        return self.st.i0()

class _NQL2[S]:
    st: S
    def __init__(self, st: S):
        self.st = st
    def __call__(self, x: float) -> float:
        self.st.x2 = x
        return self.st.i1()

class _NQL3[S]:
    st: S
    def __init__(self, st: S):
        self.st = st
    def __call__(self, x: float) -> float:
        self.st.x3 = x
        return self.st.i2()

def nquad(func, ranges, args=None, opts=None, full_output: Literal[bool] = False):
    ag = () if args is None else args
    # depth from the static arity of func (len(ranges) may be a runtime list length)
    depth: Literal[int] = static.len(static.function.args(func)) - static.len(ag)
    if depth < 1 or depth > 4:
        compile_error("nquad supports 1 to 4 dimensions")
    if len(ranges) != depth:
        raise ValueError("number of ranges must match the number of integration variables")
    st = _NQ(func, ranges, opts, ag, depth)
    val = 0.0
    if depth == 1:
        val = st.i0()
    elif depth == 2:
        val = st.i1()
    elif depth == 3:
        val = st.i2()
    else:
        val = st.i3()
    if full_output:
        return val, st.abserr, {'neval': st.neval}
    else:
        return val, st.abserr

# ---------------------------------------------------------------- newton_cotes
_BUILTINCOEFFS = {
    1: (1, 2, [1, 1], -1, 12),
    2: (1, 3, [1, 4, 1], -1, 90),
    3: (3, 8, [1, 3, 3, 1], -3, 80),
    4: (2, 45, [7, 32, 12, 32, 7], -8, 945),
    5: (5, 288, [19, 75, 50, 50, 75, 19], -275, 12096),
    6: (1, 140, [41, 216, 27, 272, 27, 216, 41], -9, 1400),
    7: (7, 17280, [751, 3577, 1323, 2989, 2989, 1323, 3577, 751], -8183, 518400),
    8: (4, 14175, [989, 5888, -928, 10496, -4540, 10496, -928, 5888, 989], -2368, 467775),
    9: (9, 89600, [2857, 15741, 1080, 19344, 5778, 5778, 19344, 1080, 15741, 2857], -4671, 394240),
    10: (5, 299376, [16067, 106300, -48525, 272400, -260550, 427368, -260550, 272400, -48525, 106300, 16067],
         -673175, 163459296),
    11: (11, 87091200, [2171465, 13486539, -3237113, 25226685, -9595542, 15493566, 15493566, -9595542,
                        25226685, -3237113, 13486539, 2171465], -2224234463, 237758976000),
    12: (1, 5255250, [1364651, 9903168, -7587864, 35725120, -51491295, 87516288, -87797136, 87516288,
                      -51491295, 35725120, -7587864, 9903168, 1364651], -3012, 875875),
    13: (13, 402361344000, [8181904909, 56280729661, -31268252574, 156074417954, -151659573325,
                            206683437987, -43111992612, -43111992612, 206683437987, -151659573325,
                            156074417954, -31268252574, 56280729661, 8181904909], -2639651053,
         344881152000),
    14: (7, 2501928000, [90241897, 710986864, -770720657, 3501442784, -6625093363, 12630121616,
                         -16802270373, 19534438464, -16802270373, 12630121616, -6625093363,
                         3501442784, -770720657, 710986864, 90241897], -3740727473, 1275983280000)}

def _ncinv(C: np.ndarray[float, 2], n: int) -> np.ndarray[float, 2]:
    # Gauss-Jordan inverse with partial pivoting
    A = C.copy()
    I = np.zeros((n, n))
    for i in range(n):
        I[i, i] = 1.0
    for k in range(n):
        p = k
        for i in range(k + 1, n):
            if abs(A[i, k]) > abs(A[p, k]):
                p = i
        if p != k:
            for j in range(n):
                t = A[k, j]
                A[k, j] = A[p, j]
                A[p, j] = t
                t = I[k, j]
                I[k, j] = I[p, j]
                I[p, j] = t
        d = A[k, k]
        for j in range(n):
            A[k, j] /= d
            I[k, j] /= d
        for i in range(n):
            if i != k:
                m = A[i, k]
                if m != 0.0:
                    for j in range(n):
                        A[i, j] -= m * A[k, j]
                        I[i, j] -= m * I[k, j]
    return I

def newton_cotes(rn, equal: int = 0):
    N = 0
    rv: List[float] = []
    if isinstance(rn, int):
        N = rn
        rv = [float(i) for i in range(N + 1)]
        equal = 1
    else:
        ra = np.asarray(rn).astype(float)
        N = ra.size - 1
        rv = [ra.reshape(ra.size)[i] for i in range(ra.size)]
        if equal:
            rv = [float(i) for i in range(N + 1)]
        else:
            ok = True
            for i in range(N):
                if rv[i + 1] - rv[i] != 1.0:
                    ok = False
            if ok:
                equal = 1
    if equal and N in _BUILTINCOEFFS:
        na, da, vi, nb, db = _BUILTINCOEFFS[N]
        an = np.array([na * float(v) / da for v in vi])
        return an, float(nb) / db
    if rv[0] != 0 or rv[N] != N:
        raise ValueError("The sample positions must start at 0 and end at N")
    n1 = N + 1
    yi = [r / float(N) for r in rv]
    ti = [2 * y - 1 for y in yi]
    C = np.empty((n1, n1))
    for i in range(n1):
        for j in range(n1):
            C[i, j] = ti[j] ** float(i)
    Cinv = _ncinv(C, n1)
    for _ in range(2):
        Cinv = 2 * Cinv - (Cinv @ C) @ Cinv
    ne = (n1 + 1) // 2
    ai = np.zeros(n1)
    for i in range(n1):
        s = 0.0
        for k in range(ne):
            s += Cinv[i, 2 * k] * (2.0 / (2 * k + 1))
        ai[i] = s * (N / 2.0)
    BN = 0.0
    power = 0
    if N % 2 == 0 and equal:
        BN = N / (N + 3.0)
        power = N + 2
    else:
        BN = N / (N + 2.0)
        power = N + 1
    for i in range(n1):
        BN -= (yi[i] ** float(power)) * ai[i]
    p1 = power + 1
    fac = power * math.log(N) - math.lgamma(p1)
    fac = math.exp(fac)
    return ai, BN * fac
