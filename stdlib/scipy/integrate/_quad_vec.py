# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# quad_vec (scipy/integrate/_quad_vec.py): adaptive Gauss-Kronrod (gk21,
# gk15) or trapezoid quadrature of vector-valued functions. The integrand
# must return a float or a 1-D float array (fixed size). norm in
# ('2', 'max'); `workers` is accepted but evaluation is sequential (scipy's
# order of interval processing is reproduced, so neval matches). The
# interval cache is unbounded (scipy evicts beyond cache_size bytes).
# full_output returns a QuadVecInfo with neval, success, status, message,
# intervals, integrals, errors.

import math
import numpy as np
import heapq

_FMIN = 2.2250738585072014e-308
_XEPS = 2.220446049250313e-16

_GK21_X = [0.995657163025808080735527280689003, 0.973906528517171720077964012084452,
           0.930157491355708226001207180059508, 0.865063366688984510732096688423493,
           0.780817726586416897063717578345042, 0.679409568299024406234327365114874,
           0.562757134668604683339000099272694, 0.433395394129247190799265943165784,
           0.294392862701460198131126603103866, 0.148874338981631210884826001129720, 0.0,
           -0.148874338981631210884826001129720, -0.294392862701460198131126603103866,
           -0.433395394129247190799265943165784, -0.562757134668604683339000099272694,
           -0.679409568299024406234327365114874, -0.780817726586416897063717578345042,
           -0.865063366688984510732096688423493, -0.930157491355708226001207180059508,
           -0.973906528517171720077964012084452, -0.995657163025808080735527280689003]
_GK21_W = [0.066671344308688137593568809893332, 0.149451349150580593145776339657697,
           0.219086362515982043995534934228163, 0.269266719309996355091226921569469,
           0.295524224714752870173892994651338, 0.295524224714752870173892994651338,
           0.269266719309996355091226921569469, 0.219086362515982043995534934228163,
           0.149451349150580593145776339657697, 0.066671344308688137593568809893332]
_GK21_V = [0.011694638867371874278064396062192, 0.032558162307964727478818972459390,
           0.054755896574351996031381300244580, 0.075039674810919952767043140916190,
           0.093125454583697605535065465083366, 0.109387158802297641899210590325805,
           0.123491976262065851077958109831074, 0.134709217311473325928054001771707,
           0.142775938577060080797094273138717, 0.147739104901338491374841515972068,
           0.149445554002916905664936468389821, 0.147739104901338491374841515972068,
           0.142775938577060080797094273138717, 0.134709217311473325928054001771707,
           0.123491976262065851077958109831074, 0.109387158802297641899210590325805,
           0.093125454583697605535065465083366, 0.075039674810919952767043140916190,
           0.054755896574351996031381300244580, 0.032558162307964727478818972459390,
           0.011694638867371874278064396062192]
_GK15_X = [0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
           0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
           0.586087235467691130294144838258730, 0.405845151377397166906606412076961,
           0.207784955007898467600689403773245, 0.0,
           -0.207784955007898467600689403773245, -0.405845151377397166906606412076961,
           -0.586087235467691130294144838258730, -0.741531185599394439863864773280788,
           -0.864864423359769072789712788640926, -0.949107912342758524526189684047851,
           -0.991455371120812639206854697526329]
_GK15_W = [0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
           0.381830050505118944950369775488975, 0.417959183673469387755102040816327,
           0.381830050505118944950369775488975, 0.279705391489276667901467771423780,
           0.129484966168869693270611432679082]
_GK15_V = [0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
           0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
           0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
           0.204432940075298892414161999234649, 0.209482141084727828012999174891714,
           0.204432940075298892414161999234649, 0.190350578064785409913256402421014,
           0.169004726639267902826583426598550, 0.140653259715525918745189590510238,
           0.104790010322250183839876322541518, 0.063092092629978553290700663189204,
           0.022935322010529224963732008058970]

class QuadVecInfo:
    neval: int
    success: bool
    status: int
    message: str
    intervals: np.ndarray[float, 2]
    integrals: np.ndarray[float, 2]
    errors: np.ndarray[float, 1]

    def __init__(self, neval: int, status: int, message: str, intervals: np.ndarray[float, 2],
                 integrals: np.ndarray[float, 2], errors: np.ndarray[float, 1]):
        self.neval = neval
        self.success = status == 0
        self.status = status
        self.message = message
        self.intervals = intervals
        self.integrals = integrals
        self.errors = errors

def _vnorm(v: np.ndarray[float, 1], kind: int) -> float:
    if v.shape[0] == 1 or kind == 0:
        if kind == 1:
            s = 0.0
            for i in range(v.shape[0]):
                s += v[i] * v[i]
            return math.sqrt(s)
        m = 0.0
        for i in range(v.shape[0]):
            if abs(v[i]) > m or (v[i] != v[i]):
                m = abs(v[i])
        return m
    s = 0.0
    # np.linalg.norm: scaled 2-norm (LAPACK dnrm2-free, plain sum of squares)
    for i in range(v.shape[0]):
        s += v[i] * v[i]
    return math.sqrt(s)

def _gk(a: float, b: float, f, kind: int, x: List[float], w: List[float], v: List[float]):
    n = len(x)
    c = 0.5 * (a + b)
    h = 0.5 * (b - a)
    fv: List[np.ndarray[float, 1]] = []
    s_k = f(c + h * x[0]) * 0.0
    s_k_abs = s_k.copy()
    for i in range(n):
        ff = f(c + h * x[i])
        fv.append(ff)
        s_k = s_k + v[i] * ff
        s_k_abs = s_k_abs + v[i] * np.abs(ff)
    s_g = s_k * 0.0
    for i in range(len(w)):
        s_g = s_g + w[i] * fv[2 * i + 1]
    s_k_dabs = s_k * 0.0
    y0 = s_k / 2.0
    for i in range(n):
        s_k_dabs = s_k_dabs + v[i] * np.abs(fv[i] - y0)
    err = _vnorm((s_k - s_g) * h, kind)
    dabs = _vnorm(s_k_dabs * h, kind)
    if dabs != 0 and err != 0:
        err = dabs * min(1.0, (200 * err / dabs) ** 1.5)
    round_err = _vnorm(50 * _XEPS * h * s_k_abs, kind)
    if round_err > _FMIN:
        err = max(err, round_err)
    return h * s_k, err, round_err

def _rule(q: int, a: float, b: float, f, kind: int):
    if q == 0:
        return _gk(a, b, f, kind, _GK21_X, _GK21_W, _GK21_V)
    elif q == 1:
        return _gk(a, b, f, kind, _GK15_X, _GK15_W, _GK15_V)
    x3 = 0.5 * (a + b)
    f1 = f(a)
    f2 = f(b)
    f3 = f(x3)
    s2 = 0.25 * (b - a) * (f1 + 2 * f3 + f2)
    round_err = 0.25 * abs(b - a) * (_vnorm(f1, kind) + 2 * _vnorm(f3, kind) + _vnorm(f2, kind)) * 2e-16
    s1 = 0.5 * (b - a) * (f1 + f2)
    err = 1 / 3 * _vnorm(s1 - s2, kind)
    return s2, err, round_err

_STATUS_MSG = ["Target precision reached.", "Target precision not reached.",
               "Target precision could not be reached due to rounding error.",
               "Non-finite values encountered."]

def _as_vec(r) -> np.ndarray[float, 1]:
    if isinstance(r, float) or isinstance(r, int):
        out = np.empty(1)
        out[0] = float(r)
        return out
    else:
        a = np.asarray(r).astype(float)
        return a.reshape(a.size)

def _semi(f, start: float, sgn: float, t: float) -> np.ndarray[float, 1]:
    tmin = math.sqrt(_FMIN)
    if t < tmin:
        return f(start) * 0.0
    x = start + sgn * (1 - t) / t
    return sgn * (f(x) / t) / t

def _dbl(f, t: float) -> np.ndarray[float, 1]:
    tmin = math.sqrt(_FMIN)
    if abs(t) < tmin:
        return f(0.0) * 0.0
    x = (1 - abs(t)) / t
    return (f(x) / t) / t

def _semi_t(start: float, sgn: float, x: float) -> float:
    z = sgn * (x - start) + 1
    if z == 0:
        return float('inf')
    return 1 / z

class _QVF[F, A]:
    # integrand wrapper with the infinite-interval remapping (no nested lambdas)
    f: F
    args: A
    mode: int
    start: float
    sgn: float

    def __init__(self, f: F, args: A, mode: int, start: float, sgn: float):
        self.f = f
        self.args = args
        self.mode = mode
        self.start = start
        self.sgn = sgn

    def raw(self, x: float) -> np.ndarray[float, 1]:
        return _as_vec(self.f(x, *self.args))

    def __call__(self, t: float) -> np.ndarray[float, 1]:
        if self.mode == 0:
            return self.raw(t)
        tmin = math.sqrt(_FMIN)
        if self.mode == 3:
            if abs(t) < tmin:
                return self.raw(0.0) * 0.0
            x = (1 - abs(t)) / t
            return (self.raw(x) / t) / t
        if t < tmin:
            return self.raw(self.start) * 0.0
        x = self.start + self.sgn * (1 - t) / t
        return self.sgn * (self.raw(x) / t) / t

def _core(fv, a: float, b: float, epsabs: float, epsrel: float, kind: int, limit: int,
          pts: List[float], q: int):
    nev = 21 if q == 0 else (15 if q == 1 else 3)
    initial: List[Tuple[float, float]] = []
    if len(pts) == 0:
        initial.append((a, b))
    else:
        prev = a
        for p in sorted(pts):
            if not (a < p and p < b) or p == prev:
                continue
            initial.append((prev, p))
            prev = p
        initial.append((prev, b))
    cache: Dict[Tuple[float, float], np.ndarray[float, 1]] = {}
    intervals: List[Tuple[float, float, float]] = []
    neval = 0
    first = True
    gi = np.zeros(0)
    ge = 0.0
    re = 0.0
    for ivi in initial:
        x1 = ivi[0]
        x2 = ivi[1]
        ig, err, rnd = _rule(q, x1, x2, fv, kind)
        neval += nev
        if first:
            gi = ig.copy()
            ge = float(err)
            re = float(rnd)
            first = False
        else:
            gi = gi + ig
            ge += err
            re += rnd
        cache[(x1, x2)] = ig.copy()
        intervals.append((-err, x1, x2))
    heapq.heapify(intervals)
    ier = 1
    while len(intervals) > 0 and len(intervals) < limit:
        tol = max(epsabs, epsrel * _vnorm(gi, kind))
        to_process: List[Tuple[float, float, float]] = []
        err_sum = 0.0
        for j in range(128):
            if len(intervals) == 0:
                break
            if j > 0 and err_sum > ge - tol / 8:
                break
            ivp = heapq.heappop(intervals)
            to_process.append(ivp)
            err_sum += -ivp[0]
        for ivq in to_process:
            old_err = -ivq[0]
            y1 = ivq[1]
            y2 = ivq[2]
            dneval = 0
            has_old = (y1, y2) in cache
            old_int = gi
            if has_old:
                old_int = cache.pop((y1, y2))
            c = 0.5 * (y1 + y2)
            s1, err1, round1 = _rule(q, y1, c, fv, kind)
            dneval += nev
            s2, err2, round2 = _rule(q, c, y2, fv, kind)
            dneval += nev
            if not has_old:
                old_int, _e2, _r2 = _rule(q, y1, y2, fv, kind)
                dneval += nev
            if q == 2:
                dneval = 5 if has_old else 5
            neval += dneval
            gi = gi + (s1 + s2 - old_int)
            ge += err1 + err2 - old_err
            re += round1 + round2
            cache[(y1, c)] = s1
            heapq.heappush(intervals, (-err1, y1, c))
            cache[(c, y2)] = s2
            heapq.heappush(intervals, (-err2, c, y2))
        if len(intervals) >= 2:
            tol = max(epsabs, epsrel * _vnorm(gi, kind))
            if ge < tol / 8:
                ier = 0
                break
            if ge < re:
                ier = 2
                break
        if not (math.isfinite(ge) and math.isfinite(re)):
            ier = 3
            break
    m = gi.shape[0]
    k = len(intervals)
    ivs = np.empty((k, 2))
    igs = np.empty((k, m))
    ers = np.empty(k)
    for i in range(k):
        z = intervals[i]
        ivs[i, 0] = z[1]
        ivs[i, 1] = z[2]
        ers[i] = -z[0]
        if (z[1], z[2]) in cache:
            v = cache[(z[1], z[2])]
            for j in range(m):
                igs[i, j] = v[j]
        else:
            for j in range(m):
                igs[i, j] = float('nan')
    info = QuadVecInfo(neval, ier, _STATUS_MSG[ier], ivs, igs, ers)
    return gi, ge + re, info

def _norm_kind(norm: str) -> int:
    if norm == '2':
        return 1
    elif norm == 'max':
        return 0
    raise ValueError(f"unknown norm {norm!r}")

def _quad_kind(quadrature: Optional[str]) -> int:
    if quadrature is None or quadrature == 'gk21':
        return 0
    elif quadrature == 'gk15':
        return 1
    elif quadrature == 'trapezoid':
        return 2
    raise ValueError(f"unknown quadrature {quadrature!r}")

def _pts_list(points) -> List[float]:
    out: List[float] = []
    if points is not None:
        for p in points:
            out.append(float(p))
    return out

def _qv_scalar(r) -> bool:
    if isinstance(r, float) or isinstance(r, int):
        return True
    else:
        return False

def quad_vec(f, a, b, epsabs: float = 1e-200, epsrel: float = 1e-8, norm: str = '2',
             cache_size: float = 100e6, limit: int = 10000, workers: int = 1, points=None,
             quadrature: Optional[str] = None, full_output: Literal[bool] = False, args=()):
    af = float(a)
    bf = float(b)
    kind = _norm_kind(norm)
    pts = _pts_list(points)
    q = _quad_kind(quadrature)
    if math.isinf(af) or math.isinf(bf):
        q = _quad_kind('gk15' if quadrature is None else quadrature)
    sgn_out = 1.0
    mode = 0
    if math.isfinite(af) and math.isinf(bf):
        mode = 1
    elif math.isfinite(bf) and math.isinf(af):
        mode = 2
        sgn_out = -1.0
    elif math.isinf(af) and math.isinf(bf):
        mode = 3
        sgn_out = -1.0 if bf < af else 1.0
    elif not (math.isfinite(af) and math.isfinite(bf)):
        raise ValueError(f"invalid integration bounds a={af}, b={bf}")
    st = af if mode == 1 else bf
    sg = 1.0
    if mode == 1:
        sg = -1.0 if bf < 0 else 1.0
    elif mode == 2:
        sg = -1.0 if af < 0 else 1.0
    g = _QVF(f, args, mode, st, sg)
    lo = af
    hi = bf
    tp = pts
    if mode == 1 or mode == 2:
        lo = 0.0
        hi = 1.0
        tp = [_semi_t(st, sg, p) for p in pts]
    elif mode == 3:
        lo = -1.0
        hi = 1.0 if af != bf else 1.0
        if af == bf:
            lo = 1.0
        tp = [0.0] + [((-1.0 if p < 0 else 1.0) / (abs(p) + 1)) for p in pts]
    res, err, info = _core(g, lo, hi, epsabs, epsrel, kind, limit, tp, q)
    res = res * sgn_out
    if isinstance(f(0.5, *args), float) or isinstance(f(0.5, *args), int):
        # scalar integrand (static dispatch on the return type)
        if full_output:
            return (res[0], err, info)
        else:
            return (res[0], err)
    else:
        if full_output:
            return (res, err, info)
        else:
            return (res, err)
