# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Generic distribution machinery (scipy.stats._distn_infrastructure).
#
# A distribution is a "kernel": a @tuple class of static methods acting on the
# standardized variable and a tuple ``sh`` of float shape parameters. Every
# method except ``nshape`` is optional; missing ones fall back to scipy's
# generic algorithms (quad for cdf/moments/entropy, brentq for ppf, ...):
#
#   nshape() -> Literal[int]     check(sh) -> bool (default: all shapes > 0)
#   a(sh), b(sh) -> float        support of the standardized distribution
#   open_support() -> Literal[bool]   pdf/logpdf use the open support mask
#   pdf, logpdf, cdf, sf, logcdf, logsf (z, sh); ppf, isf (q, sh)
#   stats(sh) -> (mu, mu2, g1, g2), ``_NA`` meaning "None" (use munp)
#   munp(n: float, sh), entropy(sh), momtype0() (pdf-based generic moments)
#   rvs(rs, sh, size)            default: ppf(rs.uniform(size=size))
#   fitstart(data) -> List[float] shape guesses
#   fitc(data, fixed: List[float]) -> List[float]  closed-form fit (empty: MLE)
#   xtol() -> float              brentq tolerance of the generic ppf

import numpy as np
import internal.static as static
import scipy.special._kernels as K
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.optimize._minimize import _minimize_neldermead, _no_bounds
from numpy.random.legacy import RandomState, _rand
from ._common import _is_arr, _anyarr
from C import fabs(float) -> float

NAN = K.NAN
INF = K.INF
_NA = -8.98846567431158e307
_LOGXMAX = 709.782712893384

# ------------------------------------------------------------ kernel access

def _allpos(sh) -> bool:
    for s in sh:
        if not (s > 0):
            return False
    return True

def _k_check(Kr: type, sh) -> bool:
    if hasattr(Kr, 'check'):
        return Kr.check(sh)
    else:
        return _allpos(sh)

def _k_a(Kr: type, sh) -> float:
    if hasattr(Kr, 'a'):
        return float(Kr.a(sh))
    elif hasattr(Kr, 'is_discrete'):
        return 0.0
    else:
        return -INF

def _k_b(Kr: type, sh) -> float:
    if hasattr(Kr, 'b'):
        return float(Kr.b(sh))
    else:
        return INF

def _k_open(Kr: type) -> Literal[bool]:
    if hasattr(Kr, 'open_support'):
        return Kr.open_support()
    else:
        return False

def _k_xtol(Kr: type) -> float:
    if hasattr(Kr, 'xtol'):
        return Kr.xtol()
    else:
        return 1e-14

class _CdfF[Kr, S]:
    sh: S
    def __init__(self, sh: S):
        self.sh = sh
    def __call__(self, x: float) -> float:
        return _k_cdf(Kr, x, self.sh)

class _PdfF[Kr, S]:
    sh: S
    def __init__(self, sh: S):
        self.sh = sh
    def __call__(self, x: float) -> float:
        return _k_pdf(Kr, x, self.sh)

def _k_pdf(Kr: type, z: float, sh) -> float:
    if hasattr(Kr, 'pdf'):
        return float(Kr.pdf(z, sh))
    elif hasattr(Kr, 'logpdf'):
        return K.exp(float(Kr.logpdf(z, sh)))
    elif hasattr(Kr, 'is_discrete'):
        return _k_cdf(Kr, z, sh) - _k_cdf(Kr, z - 1, sh)
    else:
        # _derivative(self._cdf, x, dx=1e-5, order=5)
        dx = 1e-5
        f = _CdfF[Kr, type(sh)](sh)
        return (f(z - 2 * dx) - 8.0 * f(z - dx) + 8.0 * f(z + dx) - f(z + 2 * dx)) / (12.0 * dx)

def _k_logpdf(Kr: type, z: float, sh) -> float:
    if hasattr(Kr, 'logpdf'):
        return float(Kr.logpdf(z, sh))
    else:
        return K.log(_k_pdf(Kr, z, sh))

def _k_cdf(Kr: type, z: float, sh) -> float:
    if hasattr(Kr, 'cdf'):
        return float(Kr.cdf(z, sh))
    elif hasattr(Kr, 'is_discrete'):
        # rv_discrete._cdf_single: sum pmf over arange(int(a), floor(z)+1)
        k = K.floor(z)
        tot = 0.0
        m = float(int(_k_a(Kr, sh)))
        while m <= k:
            tot += _k_pdf(Kr, m, sh)
            m += 1.0
        return tot
    else:
        return quad(_PdfF[Kr, type(sh)](sh), _k_a(Kr, sh), z)[0]

def _k_sf(Kr: type, z: float, sh) -> float:
    if hasattr(Kr, 'sf'):
        return float(Kr.sf(z, sh))
    else:
        return 1.0 - _k_cdf(Kr, z, sh)

def _k_logcdf(Kr: type, z: float, sh) -> float:
    if hasattr(Kr, 'logcdf'):
        return float(Kr.logcdf(z, sh))
    elif hasattr(Kr, 'is_discrete'):
        return K.log(_k_cdf(Kr, z, sh))
    else:
        median = _k_ppf(Kr, 0.5, sh)
        if z < median:
            return K.log(_k_cdf(Kr, z, sh))
        return K.log1p(-_k_sf(Kr, z, sh))

def _k_logsf(Kr: type, z: float, sh) -> float:
    if hasattr(Kr, 'logsf'):
        return float(Kr.logsf(z, sh))
    elif hasattr(Kr, 'is_discrete'):
        return K.log(_k_sf(Kr, z, sh))
    else:
        median = _k_ppf(Kr, 0.5, sh)
        if z > median:
            return K.log(_k_sf(Kr, z, sh))
        return K.log1p(-_k_cdf(Kr, z, sh))

class _PpfSolve[Kr, S]:
    sh: S
    q: float
    def __init__(self, sh: S, q: float):
        self.sh = sh
        self.q = q
    def __call__(self, x: float) -> float:
        return _k_cdf(Kr, x, self.sh) - self.q

def _ppf_single(Kr: type, q: float, sh) -> float:
    factor = 10.
    left = _k_a(Kr, sh)
    right = _k_b(Kr, sh)
    f = _PpfSolve[Kr, type(sh)](sh, q)
    if K.isinf(left):
        left = min(-factor, right)
        while f(left) > 0.:
            left, right = left * factor, left
    if K.isinf(right):
        right = max(factor, left)
        while f(right) < 0.:
            left, right = right, right * factor
    return brentq(f, left, right, xtol=_k_xtol(Kr))

def _k_ppf(Kr: type, q: float, sh) -> float:
    if hasattr(Kr, 'ppf'):
        return float(Kr.ppf(q, sh))
    elif hasattr(Kr, 'is_discrete'):
        return _drv2_ppfsingle(Kr, q, sh)
    else:
        return _ppf_single(Kr, q, sh)

def _k_isf(Kr: type, q: float, sh) -> float:
    if hasattr(Kr, 'isf'):
        return float(Kr.isf(q, sh))
    else:
        return _k_ppf(Kr, 1.0 - q, sh)

def _k_stats(Kr: type, sh):
    if hasattr(Kr, 'stats'):
        r = Kr.stats(sh)
        return (float(r[0]), float(r[1]), float(r[2]), float(r[3]))
    else:
        return (_NA, _NA, _NA, _NA)

class _Mom0[Kr, S]:
    sh: S
    m: float
    def __init__(self, sh: S, m: float):
        self.sh = sh
        self.m = m
    def __call__(self, x: float) -> float:
        return x ** self.m * _k_pdf(Kr, x, self.sh)

class _Mom1[Kr, S]:
    sh: S
    m: float
    def __init__(self, sh: S, m: float):
        self.sh = sh
        self.m = m
    def __call__(self, q: float) -> float:
        return _k_ppf(Kr, q, self.sh) ** self.m

def _k_munp(Kr: type, n: float, sh) -> float:
    if hasattr(Kr, 'munp'):
        return float(Kr.munp(n, sh))
    elif hasattr(Kr, 'is_discrete'):
        return _dexpect(_DMom[Kr, type(sh)](sh, n), _k_a(Kr, sh), _k_b(Kr, sh),
                        _k_ppf(Kr, 0.5, sh))
    elif hasattr(Kr, 'momtype0'):
        return quad(_Mom0[Kr, type(sh)](sh, n), _k_a(Kr, sh), _k_b(Kr, sh))[0]
    else:
        return quad(_Mom1[Kr, type(sh)](sh, n), 0.0, 1.0)[0]

class _EntrF[Kr, S]:
    sh: S
    def __init__(self, sh: S):
        self.sh = sh
    def __call__(self, x: float) -> float:
        return K.entr(_k_pdf(Kr, x, self.sh))

def _k_entropy(Kr: type, sh) -> float:
    if hasattr(Kr, 'entropy'):
        return float(Kr.entropy(sh))
    elif hasattr(Kr, 'is_discrete'):
        return _dexpect(_DEntr[Kr, type(sh)](sh), _k_a(Kr, sh), _k_b(Kr, sh),
                        _k_ppf(Kr, 0.5, sh))
    else:
        a = _k_a(Kr, sh)
        b = _k_b(Kr, sh)
        f = _EntrF[Kr, type(sh)](sh)
        h = quad(f, a, b)[0]
        if not K.isnan(h):
            return h
        low = _k_ppf(Kr, 1e-10, sh)
        upp = _k_ppf(Kr, 1. - 1e-10, sh)
        upper = upp if K.isinf(b) else b
        lower = low if K.isinf(a) else a
        return quad(f, lower, upper)[0]

# ------------------------------------------------------- scalar evaluation

def _in_support(Kr: type, z: float, a: float, b: float) -> bool:
    if _k_open(Kr):
        return a < z and z < b
    else:
        return a <= z and z <= b

def _eval(Kr: type, op: Literal[str], x: float, loc: float, scale: float, sh) -> float:
    if not (scale > 0) or not _k_check(Kr, sh):
        return NAN
    if op == 'ppf' or op == 'isf':
        q = x
        if not (loc == loc) or not (q >= 0 and q <= 1):
            return NAN
        lo = _k_a(Kr, sh) * scale + loc
        hi = _k_b(Kr, sh) * scale + loc
        if op == 'ppf':
            if q == 0:
                return lo
            if q == 1:
                return hi
            return _k_ppf(Kr, q, sh) * scale + loc
        else:
            if q == 0:
                return hi
            if q == 1:
                return lo
            return _k_isf(Kr, q, sh) * scale + loc
    if x != x:
        return NAN
    z = (x - loc) / scale
    if z != z:
        return NAN
    a = _k_a(Kr, sh)
    b = _k_b(Kr, sh)
    if op == 'pdf':
        if not _in_support(Kr, z, a, b):
            return 0.0
        return _k_pdf(Kr, z, sh) / scale
    elif op == 'logpdf':
        if not _in_support(Kr, z, a, b):
            return -INF
        return _k_logpdf(Kr, z, sh) - K.log(scale)
    elif op == 'cdf':
        if z >= b:
            return 1.0
        if z <= a:
            return 0.0
        return _k_cdf(Kr, z, sh)
    elif op == 'sf':
        if z <= a:
            return 1.0
        if z >= b:
            return 0.0
        return _k_sf(Kr, z, sh)
    elif op == 'logcdf':
        if z >= b:
            return 0.0
        if z <= a:
            return -INF
        return _k_logcdf(Kr, z, sh)
    elif op == 'logsf':
        if z <= a:
            return 0.0
        if z >= b:
            return -INF
        return _k_logsf(Kr, z, sh)
    else:
        compile_error("unknown distribution method")

def _bcast_eval(Kr: type, op: Literal[str], x, loc, scale, sh):
    if _anyarr(x, loc, scale, *sh):
        arrs = (np.asarray(x), np.asarray(loc), np.asarray(scale)) + tuple(np.asarray(s) for s in sh)
        shp = np.broadcast_shapes(*tuple(a.shape for a in arrs))
        bs = tuple(np.broadcast_to(a, shp) for a in arrs)
        out = np.empty(shp, dtype=float)
        for idx in np.ndindex(shp):
            vals = tuple(float(b[idx]) for b in bs)
            out[idx] = _eval(Kr, op, vals[0], vals[1], vals[2], vals[3:])
        return out
    else:
        return _eval(Kr, op, float(x), float(loc), float(scale), tuple(float(s) for s in sh))

def _split(Kr: type, args, loc, scale):
    NS: Literal[int] = Kr.nshape()
    if static.len(args) < NS:
        compile_error("missing shape parameter(s) for distribution")
    sh = args[:NS]
    extra = args[NS:]
    if static.len(extra) > 2:
        compile_error("too many positional arguments")
    if static.len(extra) >= 1:
        l = extra[0]
    else:
        l = loc
    if static.len(extra) == 2:
        s = extra[1]
    else:
        s = scale
    return sh, l, s

# ---------------------------------------------------------------- moments

def _mi(c: Literal[str]) -> Literal[int]:
    if c == 'm':
        return 0
    elif c == 'v':
        return 1
    elif c == 's':
        return 2
    else:
        return 3

def _has(moments: Literal[str], c: Literal[str]) -> Literal[bool]:
    if static.len(moments) == 0:
        return False
    else:
        return moments[0] == c or _has(moments[1:], c)

def _std_stats(Kr: type, sh, want_m: bool, want_v: bool, want_s: bool, want_k: bool):
    # scipy rv_generic.stats on the standardized distribution
    mu, mu2, g1, g2 = _k_stats(Kr, sh)
    rm = NAN
    rv = NAN
    rs = NAN
    rk = NAN
    if want_m:
        if mu == _NA:
            mu = _k_munp(Kr, 1.0, sh)
        rm = mu
    if want_v:
        if mu2 == _NA:
            mu2p = _k_munp(Kr, 2.0, sh)
            if mu == _NA:
                mu = _k_munp(Kr, 1.0, sh)
            mu2 = mu2p - mu * mu if not K.isinf(mu) else INF
        rv = mu2
    if want_s:
        if g1 == _NA:
            mu3p = _k_munp(Kr, 3.0, sh)
            if mu == _NA:
                mu = _k_munp(Kr, 1.0, sh)
            if mu2 == _NA:
                mu2p = _k_munp(Kr, 2.0, sh)
                mu2 = mu2p - mu * mu
            mu3 = (-mu * mu - 3 * mu2) * mu + mu3p
            g1 = mu3 / mu2 ** 1.5
        rs = g1
    if want_k:
        if g2 == _NA:
            mu4p = _k_munp(Kr, 4.0, sh)
            if mu == _NA:
                mu = _k_munp(Kr, 1.0, sh)
            if mu2 == _NA:
                mu2p = _k_munp(Kr, 2.0, sh)
                mu2 = mu2p - mu * mu
            mu3 = 0.0
            if g1 == _NA:
                mu3p = _k_munp(Kr, 3.0, sh)
                mu3 = (-mu * mu - 3 * mu2) * mu + mu3p
            else:
                mu3 = g1 * mu2 ** 1.5
            mu4 = ((-mu ** 2 - 6 * mu2) * mu - 4 * mu3) * mu + mu4p
            g2 = mu4 / mu2 ** 2.0 - 3.0
        rk = g2
    return (rm, rv, rs, rk)

def _stats1(Kr: type, moments: Literal[str], l: float, s: float, sh):
    if not (s > 0) or not (l == l) or not _k_check(Kr, sh):
        return (NAN, NAN, NAN, NAN)
    m, v, sk, ku = _std_stats(Kr, sh, _has(moments, 'm'), _has(moments, 'v'),
                              _has(moments, 's'), _has(moments, 'k'))
    return (m * s + l, v * s * s, sk, ku)

def _stats_eval(Kr: type, moments: Literal[str], loc, scale, sh):
    if _anyarr(loc, scale, *sh):
        arrs = (np.asarray(loc), np.asarray(scale)) + tuple(np.asarray(v) for v in sh)
        shp = np.broadcast_shapes(*tuple(a.shape for a in arrs))
        bs = tuple(np.broadcast_to(a, shp) for a in arrs)
        outs = (np.empty(shp, dtype=float), np.empty(shp, dtype=float),
                np.empty(shp, dtype=float), np.empty(shp, dtype=float))
        for idx in np.ndindex(shp):
            vals = tuple(float(b[idx]) for b in bs)
            r = _stats1(Kr, moments, vals[0], vals[1], vals[2:])
            for j in static.range(4):
                outs[j][idx] = r[j]
        res = tuple(outs[_mi(moments[i])] for i in static.range(static.len(moments)))
    else:
        r = _stats1(Kr, moments, float(loc), float(scale), tuple(float(v) for v in sh))
        res = tuple(r[_mi(moments[i])] for i in static.range(static.len(moments)))
    if static.len(moments) == 1:
        return res[0]
    else:
        return res

def _moment_from_stats(Kr: type, n: int, mu: float, mu2: float, g1: float, g2: float, sh) -> float:
    if n == 0:
        return 1.0
    elif n == 1:
        return mu if mu != _NA else _k_munp(Kr, 1.0, sh)
    elif n == 2:
        if mu2 == _NA or mu == _NA:
            return _k_munp(Kr, 2.0, sh)
        return mu2 + mu * mu
    elif n == 3:
        if g1 == _NA or mu2 == _NA or mu == _NA:
            return _k_munp(Kr, 3.0, sh)
        mu3 = g1 * mu2 ** 1.5
        return mu3 + 3 * mu * mu2 + mu * mu * mu
    elif n == 4:
        if g1 == _NA or g2 == _NA or mu2 == _NA or mu == _NA:
            return _k_munp(Kr, 4.0, sh)
        mu4 = (g2 + 3.0) * (mu2 ** 2.0)
        mu3 = g1 * mu2 ** 1.5
        return mu4 + 4 * mu * mu3 + 6 * mu * mu * mu2 + mu * mu * mu * mu
    else:
        return _k_munp(Kr, float(n), sh)

def _moment1(Kr: type, n: int, l: float, s: float, sh) -> float:
    if n < 0:
        raise ValueError("Moment must be positive.")
    if not (s > 0) or not _k_check(Kr, sh):
        return NAN
    mu, mu2, g1, g2 = _NA, _NA, _NA, _NA
    if n > 0 and n < 5:
        mu, mu2, g1, g2 = _k_stats(Kr, sh)
    val = _moment_from_stats(Kr, n, mu, mu2, g1, g2, sh)
    if l == 0:
        return s ** float(n) * val
    fac = s / l
    res2 = 0.0
    for k in range(n):
        valk = _moment_from_stats(Kr, k, mu, mu2, g1, g2, sh)
        res2 += _comb(n, k) * fac ** float(k) * valk
    res2 += fac ** float(n) * val
    res2 *= l ** float(n)
    return res2

def _comb(n: int, k: int) -> float:
    r = 1
    for i in range(k):
        r = r * (n - i) // (i + 1)
    return float(r)

def _entropy1(Kr: type, l: float, s: float, sh) -> float:
    if not (s > 0) or not (l == l) or not _k_check(Kr, sh):
        return NAN
    return _k_entropy(Kr, sh) + K.log(s)

def _vec_ls(f, Kr: type, loc, scale, sh):
    # broadcast a scalar f(Kr, loc, scale, sh) -> float over loc/scale/shapes
    if _anyarr(loc, scale, *sh):
        arrs = (np.asarray(loc), np.asarray(scale)) + tuple(np.asarray(v) for v in sh)
        shp = np.broadcast_shapes(*tuple(a.shape for a in arrs))
        bs = tuple(np.broadcast_to(a, shp) for a in arrs)
        out = np.empty(shp, dtype=float)
        for idx in np.ndindex(shp):
            vals = tuple(float(b[idx]) for b in bs)
            out[idx] = f(Kr, vals[0], vals[1], vals[2:])
        return out
    else:
        return f(Kr, float(loc), float(scale), tuple(float(v) for v in sh))

# -------------------------------------------------------------------- rvs

def check_random_state(seed = None):
    """scipy._lib._util.check_random_state."""
    if seed is None:
        return _rand
    elif isinstance(seed, int):
        return RandomState(seed)
    elif isinstance(seed, Optional):
        return check_random_state(seed.__val__())
    else:
        return seed

def _map_ppf(Kr: type, u, sh):
    if isinstance(u, np.ndarray):
        out = np.empty(u.shape, dtype=float)
        for idx in np.ndindex(u.shape):
            out[idx] = _k_ppf(Kr, float(u[idx]), sh)
        return out
    else:
        return _k_ppf(Kr, float(u), sh)

def _k_rvs(Kr: type, rs, sh, size):
    if hasattr(Kr, 'rvs'):
        return Kr.rvs(rs, sh, size)
    else:
        return _map_ppf(Kr, rs.uniform(size=size), sh)

def _rvs(Kr: type, sh, loc, scale, size, random_state):
    shv = tuple(float(v) for v in sh)
    if not _k_check(Kr, shv) or not (float(scale) >= 0):
        raise ValueError("Domain error in arguments. The `scale` parameter must be "
                         "positive for all distributions, and many distributions "
                         "have restrictions on shape parameters.")
    rs = check_random_state(random_state)
    vals = _k_rvs(Kr, rs, shv, size)
    return vals * float(scale) + float(loc)

# -------------------------------------------------------------------- fit

def _sum_finite(x: np.ndarray[float, 1]):
    fin = np.isfinite(x)
    nbad = x.size - int(np.count_nonzero(fin))
    return float(np.sum(x[fin])), nbad

class _NNLF[Kr]:
    # penalized negative log-likelihood (MLE) or moment error (MM) in the
    # free parameters; ``full`` holds [shapes..., loc, scale], NaN = free
    full: List[float]
    data: np.ndarray[float, 1]
    mm: bool
    dmom: np.ndarray[float, 1]

    def __init__(self, full: List[float], data: np.ndarray[float, 1], mm: bool,
                 dmom: np.ndarray[float, 1]):
        self.full = full
        self.data = data
        self.mm = mm
        self.dmom = dmom

    def restore(self, theta) -> List[float]:
        r = self.full.copy()
        i = 0
        for n in range(len(r)):
            if r[n] != r[n]:
                r[n] = float(theta[i])
                i += 1
        return r

    def __call__(self, theta) -> float:
        NS: Literal[int] = Kr.nshape()
        r = self.restore(theta)
        sh = tuple(r[i] for i in static.range(NS))
        loc = r[NS]
        scale = r[NS + 1]
        if not _k_check(Kr, sh) or scale <= 0:
            return INF
        if self.mm:
            err = 0.0
            for i in range(len(self.dmom)):
                dm = _moment1(Kr, i + 1, loc, scale, sh)
                if dm != dm:
                    raise ValueError("Method of moments encountered a non-finite "
                                     "distribution moment and cannot continue. "
                                     "Consider trying method='MLE'.")
                e = (self.dmom[i] - dm) / max(abs(self.dmom[i]), 1e-8)
                err += e * e
            return err
        a = _k_a(Kr, sh)
        b = _k_b(Kr, sh)
        x = (self.data - loc) / scale
        good = 0
        for v in x:
            if _in_support(Kr, v, a, b):
                good += 1
        n_bad = len(x) - good
        lp = np.empty(good, dtype=float)
        j = 0
        for v in x:
            if _in_support(Kr, v, a, b):
                lp[j] = _k_logpdf(Kr, v, sh)
                j += 1
        tot, nb = _sum_finite(lp)
        n_bad += nb
        return -tot + n_bad * _LOGXMAX * 100 + len(x) * K.log(scale)

def _fit_loc_scale(Kr: type, data: np.ndarray[float, 1], sh):
    mu, mu2, _s, _k = _std_stats(Kr, sh, True, True, False, False)
    muhat = float(data.mean())
    mu2hat = float(data.var())
    Shat = K.sqrt(mu2hat / mu2)
    Lhat = muhat - Shat * mu
    if not np.isfinite(Lhat):
        Lhat = 0.0
    if not (np.isfinite(Shat) and 0 < Shat):
        Shat = 1.0
    return Lhat, Shat

def _fit_loc_scale_support(Kr: type, data: np.ndarray[float, 1], sh):
    loc_hat, scale_hat = _fit_loc_scale(Kr, data, sh)
    a = _k_a(Kr, sh)
    b = _k_b(Kr, sh)
    support_width = b - a
    if support_width <= 0:
        return loc_hat, scale_hat
    a_hat = loc_hat + a * scale_hat
    b_hat = loc_hat + b * scale_hat
    data_a = float(data.min())
    data_b = float(data.max())
    if a_hat < data_a and data_b < b_hat:
        return loc_hat, scale_hat
    data_width = data_b - data_a
    margin = data_width * 0.1
    if support_width < INF:
        return (data_a - a) - margin, (data_width + 2 * margin) / support_width
    if a > -INF:
        return (data_a - a) - margin, 1.0
    else:
        return (data_b - b) + margin, 1.0

def _opt(v) -> float:
    if isinstance(v, NoneType):
        return NAN
    elif isinstance(v, Optional):
        return NAN if v is None else float(v.__val__())
    else:
        return float(v)

def _fit(Kr: type, data, args, f0, f1, f2, f3, floc, fscale, loc, scale, method: str):
    NS: Literal[int] = Kr.nshape()
    x = np.asarray(data).astype(float).ravel()
    if not bool(np.isfinite(x).all()):
        raise ValueError("The data contains non-finite values.")
    fixed = [_opt(f0), _opt(f1), _opt(f2), _opt(f3)][:NS] + [_opt(floc), _opt(fscale)]
    meth = method.lower()
    if meth != 'mle' and meth != 'mm':
        raise ValueError(f"Method '{method}' not available; must be one of {{'mle', 'mm'}}")
    nfree = 0
    for v in fixed:
        if v != v:
            nfree += 1
    if nfree == 0:
        raise ValueError("All parameters fixed. There is nothing to optimize.")
    if hasattr(Kr, 'fitc') and meth == 'mle':
        r = Kr.fitc(x, fixed)
        if len(r) > 0:
            return tuple(r[i] for i in static.range(NS + 2))
    # starting values
    start = [1.0 for _ in range(NS)]
    if hasattr(Kr, 'fitstart'):
        start = Kr.fitstart(x)
    for i in static.range(static.len(args)):
        start[i] = float(args[i])
    sh0 = tuple(start[i] for i in static.range(NS))
    l0, s0 = _fit_loc_scale_support(Kr, x, sh0)
    lg = _opt(loc)
    sg = _opt(scale)
    if lg == lg:
        l0 = lg
    if sg == sg:
        s0 = sg
    allv = start + [l0, s0]
    x0 = np.empty(nfree, dtype=float)
    j = 0
    for i in range(NS + 2):
        if fixed[i] != fixed[i]:
            x0[j] = allv[i]
            j += 1
    dmom = np.empty(0, dtype=float)
    if meth == 'mm':
        dmom = np.empty(nfree, dtype=float)
        for k in range(nfree):
            dmom[k] = float(np.sum(x ** float(k + 1) / len(x)))
    fn = _NNLF[Kr](fixed, x, meth == 'mm', dmom)
    res = _minimize_neldermead(fn, x0, (), None, _no_bounds())
    vals = fn.restore(res.x)
    sh = tuple(vals[i] for i in static.range(NS))
    if not (_k_check(Kr, sh) and vals[NS + 1] > 0):
        raise ValueError("Optimization converged to parameters that are outside "
                         "the range allowed by the distribution.")
    return tuple(vals[i] for i in static.range(NS + 2))

# ----------------------------------------------------------------- expect

class _ExpF[Kr, S, F]:
    sh: S
    func: F
    loc: float
    scale: float
    def __init__(self, sh: S, func: F, loc: float, scale: float):
        self.sh = sh
        self.func = func
        self.loc = loc
        self.scale = scale
    def __call__(self, x: float) -> float:
        p = _eval(Kr, 'pdf', x, self.loc, self.scale, self.sh)
        if self.func is None:
            return x * p
        else:
            return float(self.func(x)) * p

def _expect(Kr: type, func, sh, loc: float, scale: float, lb, ub, conditional: bool) -> float:
    a = _k_a(Kr, sh)
    b = _k_b(Kr, sh)
    lbv = loc + a * scale if lb is None else float(lb)
    ubv = loc + b * scale if ub is None else float(ub)
    c0 = _eval(Kr, 'cdf', lbv, loc, scale, sh)
    c1 = _eval(Kr, 'cdf', ubv, loc, scale, sh)
    invfac = c1 - c0
    alpha = 0.05
    c = loc + _k_ppf(Kr, c0 + invfac * alpha, sh) * scale
    d = loc + _k_ppf(Kr, c0 + invfac * (1 - alpha), sh) * scale
    f = _ExpF[Kr, type(sh), type(func)](sh, func, loc, scale)
    vals = quad(f, lbv, c)[0] + quad(f, c, d)[0] + quad(f, d, ubv)[0]
    if conditional:
        vals /= invfac
    return vals

# ---------------------------------------------------------------- classes

def _support1(Kr: type, l: float, s: float, sh):
    if not (s > 0) or not _k_check(Kr, sh):
        return (NAN, NAN)
    return (_k_a(Kr, sh) * s + l, _k_b(Kr, sh) * s + l)

class rv_frozen[Kr, S, L, C]:
    """Frozen distribution: ``norm(loc, scale)``, ``gamma(a)``, ..."""
    sh: S
    loc: L
    scale: C

    def __init__(self, sh: S, loc: L, scale: C):
        self.sh = sh
        self.loc = loc
        self.scale = scale

    @property
    def args(self):
        return self.sh

    def pdf(self, x): return _bcast_eval(Kr, 'pdf', x, self.loc, self.scale, self.sh)
    def logpdf(self, x): return _bcast_eval(Kr, 'logpdf', x, self.loc, self.scale, self.sh)
    def cdf(self, x): return _bcast_eval(Kr, 'cdf', x, self.loc, self.scale, self.sh)
    def logcdf(self, x): return _bcast_eval(Kr, 'logcdf', x, self.loc, self.scale, self.sh)
    def sf(self, x): return _bcast_eval(Kr, 'sf', x, self.loc, self.scale, self.sh)
    def logsf(self, x): return _bcast_eval(Kr, 'logsf', x, self.loc, self.scale, self.sh)
    def ppf(self, q): return _bcast_eval(Kr, 'ppf', q, self.loc, self.scale, self.sh)
    def isf(self, q): return _bcast_eval(Kr, 'isf', q, self.loc, self.scale, self.sh)
    def stats(self, moments: Literal[str] = 'mv'):
        return _stats_eval(Kr, moments, self.loc, self.scale, self.sh)
    def mean(self): return _stats_eval(Kr, 'm', self.loc, self.scale, self.sh)
    def var(self): return _stats_eval(Kr, 'v', self.loc, self.scale, self.sh)
    def std(self): return np.sqrt(_stats_eval(Kr, 'v', self.loc, self.scale, self.sh))
    def median(self): return self.ppf(0.5)
    def moment(self, order: int):
        return _moment1(Kr, order, float(self.loc), float(self.scale),
                        tuple(float(v) for v in self.sh))
    def entropy(self): return _vec_ls(_entropy1, Kr, self.loc, self.scale, self.sh)
    def interval(self, confidence):
        c = confidence
        return (self.ppf((1.0 - c) / 2.0), self.ppf((1.0 + c) / 2.0))
    def support(self):
        return _support1(Kr, float(self.loc), float(self.scale), tuple(float(v) for v in self.sh))
    def rvs(self, size = None, random_state = None):
        return _rvs(Kr, self.sh, self.loc, self.scale, size, random_state)
    def expect(self, func = None, lb = None, ub = None, conditional: bool = False):
        return _expect(Kr, func, tuple(float(v) for v in self.sh), float(self.loc),
                       float(self.scale), lb, ub, conditional)

class rv_continuous[Kr]:
    name: str

    def __init__(self, name: str):
        self.name = name

    @property
    def numargs(self) -> int:
        return Kr.nshape()

    def __call__(self, *args, loc = 0.0, scale = 1.0):
        sh, l, s = _split(Kr, args, loc, scale)
        return rv_frozen[Kr, type(sh), type(l), type(s)](sh, l, s)
    def freeze(self, *args, loc = 0.0, scale = 1.0):
        return self(*args, loc=loc, scale=scale)

    def pdf(self, x, *args, loc = 0.0, scale = 1.0):
        sh, l, s = _split(Kr, args, loc, scale)
        return _bcast_eval(Kr, 'pdf', x, l, s, sh)
    def logpdf(self, x, *args, loc = 0.0, scale = 1.0):
        sh, l, s = _split(Kr, args, loc, scale)
        return _bcast_eval(Kr, 'logpdf', x, l, s, sh)
    def cdf(self, x, *args, loc = 0.0, scale = 1.0):
        sh, l, s = _split(Kr, args, loc, scale)
        return _bcast_eval(Kr, 'cdf', x, l, s, sh)
    def logcdf(self, x, *args, loc = 0.0, scale = 1.0):
        sh, l, s = _split(Kr, args, loc, scale)
        return _bcast_eval(Kr, 'logcdf', x, l, s, sh)
    def sf(self, x, *args, loc = 0.0, scale = 1.0):
        sh, l, s = _split(Kr, args, loc, scale)
        return _bcast_eval(Kr, 'sf', x, l, s, sh)
    def logsf(self, x, *args, loc = 0.0, scale = 1.0):
        sh, l, s = _split(Kr, args, loc, scale)
        return _bcast_eval(Kr, 'logsf', x, l, s, sh)
    def ppf(self, q, *args, loc = 0.0, scale = 1.0):
        sh, l, s = _split(Kr, args, loc, scale)
        return _bcast_eval(Kr, 'ppf', q, l, s, sh)
    def isf(self, q, *args, loc = 0.0, scale = 1.0):
        sh, l, s = _split(Kr, args, loc, scale)
        return _bcast_eval(Kr, 'isf', q, l, s, sh)
    def stats(self, *args, loc = 0.0, scale = 1.0, moments: Literal[str] = 'mv'):
        sh, l, s = _split(Kr, args, loc, scale)
        return _stats_eval(Kr, moments, l, s, sh)
    def mean(self, *args, loc = 0.0, scale = 1.0):
        sh, l, s = _split(Kr, args, loc, scale)
        return _stats_eval(Kr, 'm', l, s, sh)
    def var(self, *args, loc = 0.0, scale = 1.0):
        sh, l, s = _split(Kr, args, loc, scale)
        return _stats_eval(Kr, 'v', l, s, sh)
    def std(self, *args, loc = 0.0, scale = 1.0):
        sh, l, s = _split(Kr, args, loc, scale)
        return np.sqrt(_stats_eval(Kr, 'v', l, s, sh))
    def moment(self, order: int, *args, loc = 0.0, scale = 1.0):
        sh, l, s = _split(Kr, args, loc, scale)
        return _moment1(Kr, order, float(l), float(s), tuple(float(v) for v in sh))
    def entropy(self, *args, loc = 0.0, scale = 1.0):
        sh, l, s = _split(Kr, args, loc, scale)
        return _vec_ls(_entropy1, Kr, l, s, sh)
    def median(self, *args, loc = 0.0, scale = 1.0):
        return self.ppf(0.5, *args, loc=loc, scale=scale)
    def interval(self, confidence, *args, loc = 0.0, scale = 1.0):
        c = confidence
        return (self.ppf((1.0 - c) / 2.0, *args, loc=loc, scale=scale),
                self.ppf((1.0 + c) / 2.0, *args, loc=loc, scale=scale))
    def support(self, *args, loc = 0.0, scale = 1.0):
        sh, l, s = _split(Kr, args, loc, scale)
        return _support1(Kr, float(l), float(s), tuple(float(v) for v in sh))
    def rvs(self, *args, loc = 0.0, scale = 1.0, size = None, random_state = None):
        sh, l, s = _split(Kr, args, loc, scale)
        return _rvs(Kr, sh, l, s, size, random_state)
    def expect(self, func = None, args = (), loc = 0.0, scale = 1.0, lb = None, ub = None,
               conditional: bool = False):
        return _expect(Kr, func, tuple(float(v) for v in args), float(loc), float(scale),
                       lb, ub, conditional)
    def nnlf(self, theta, x):
        NS: Literal[int] = Kr.nshape()
        th = [float(v) for v in theta]
        sh = tuple(th[i] for i in static.range(NS))
        loc = th[NS]
        scale = th[NS + 1]
        if not _k_check(Kr, sh) or scale <= 0:
            return INF
        xs = (np.asarray(x).astype(float).ravel() - loc) / scale
        tot = 0.0
        a = _k_a(Kr, sh)
        b = _k_b(Kr, sh)
        for v in xs:
            if not _in_support(Kr, v, a, b):
                return INF
            tot += _k_logpdf(Kr, v, sh)
        return -tot + len(xs) * K.log(scale)
    def fit(self, data, *args, f0 = None, f1 = None, f2 = None, f3 = None,
            floc = None, fscale = None, loc = None, scale = None, method: str = 'MLE'):
        return _fit(Kr, data, args, f0, f1, f2, f3, floc, fscale, loc, scale, method)

def _generic_munp(Kr: type, n: float, sh) -> float:
    # rv_continuous.generic_moment (momtype=1: integrate ppf**n over (0, 1))
    return quad(_Mom1[Kr, type(sh)](sh, n), 0.0, 1.0)[0]

# ================================================================ discrete
#
# A discrete kernel declares ``is_discrete() -> Literal[bool]`` and uses the
# same protocol with ``pdf``/``logpdf`` meaning pmf/logpmf. Defaults follow
# rv_discrete: a = 0, pmf = cdf(k) - cdf(k-1), cdf = sum of pmf, ppf by
# bisection (_drv2_ppfsingle), moments/entropy by the chunked _expect sum.

def _npsum(v: List[float]) -> float:
    # numpy pairwise summation (identical rounding for n <= 128)
    n = len(v)
    if n < 8:
        res = 0.0
        for x in v:
            res += x
        return res
    r0, r1, r2, r3 = v[0], v[1], v[2], v[3]
    r4, r5, r6, r7 = v[4], v[5], v[6], v[7]
    i = 8
    while i < n - (n % 8):
        r0 += v[i]
        r1 += v[i + 1]
        r2 += v[i + 2]
        r3 += v[i + 3]
        r4 += v[i + 4]
        r5 += v[i + 5]
        r6 += v[i + 6]
        r7 += v[i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += v[i]
        i += 1
    return res

class _DMom[Kr, S]:
    sh: S
    m: float
    def __init__(self, sh: S, m: float):
        self.sh = sh
        self.m = m
    def __call__(self, x: float) -> float:
        return K.pow(x, self.m) * _k_pdf(Kr, x, self.sh)

class _DEntr[Kr, S]:
    sh: S
    def __init__(self, sh: S):
        self.sh = sh
    def __call__(self, x: float) -> float:
        return K.entr(_k_pdf(Kr, x, self.sh))

def _chunk_sum(fun, x: float, n: int, inc: float) -> float:
    v = List[float](n)
    for i in range(n):
        v.append(float(fun(x + i * inc)))
    return _npsum(v)

def _dexpect(fun, lb: float, ub: float, x0: float, maxcount: int = 1000,
             tolerance: float = 1e-10, chunksize: int = 32) -> float:
    # scipy.stats._distn_infrastructure._expect (inc = 1)
    if (ub - lb) <= chunksize:
        return _chunk_sum(fun, lb, int(K.ceil(ub + 1 - lb)), 1.0)
    if x0 < lb:
        x0 = lb
    if x0 > ub:
        x0 = ub
    count = 0
    tot = 0.0
    x = x0
    x1 = ub + 1
    conv = False
    while (x - x1) < 0:
        delta = min(float(chunksize), fabs(x - x1))
        n = int(K.ceil(delta))
        d = _chunk_sum(fun, x, n, 1.0)
        x += delta
        count += n
        tot += d
        if fabs(d) < tolerance * n:
            break
        if count > maxcount:
            return tot
    x = x0 - 1
    x1 = lb - 1
    while (x - x1) * -1 < 0:
        delta = min(float(chunksize), fabs(x - x1))
        n = int(K.ceil(delta))
        d = _chunk_sum(fun, x, n, -1.0)
        x -= delta
        count += n
        tot += d
        if fabs(d) < tolerance * n:
            break
        if count > maxcount:
            break
    return tot

def _drv2_ppfsingle(Kr: type, q: float, sh) -> float:
    _a = _k_a(Kr, sh)
    _b = _k_b(Kr, sh)
    b = _b
    a = _a
    step = 10.0
    qb = 1.0
    qa = 0.0
    if K.isinf(b):
        b = float(max(100 * q, 10.0))
        while True:
            if b >= _b:
                qb = 1.0
                break
            qb = _k_cdf(Kr, b, sh)
            if qb < q:
                b += step
                step *= 2
            else:
                break
    step = 10.0
    if K.isinf(a):
        a = float(min(-100 * q, -10.0))
        while True:
            if a <= _a:
                qb = 0.0
                break
            qa = _k_cdf(Kr, a, sh)
            if qa > q:
                a -= step
                step *= 2
            else:
                break
    else:
        qa = _k_cdf(Kr, a, sh)
    for i in range(2046):
        if qa == q:
            return a
        if qb == q:
            return b
        if b <= a + 1:
            return a if qa > q else b
        c = float(int((a + b) / 2.0))
        qc = _k_cdf(Kr, c, sh)
        if qc < q:
            if a != c:
                a = c
            else:
                raise RuntimeError('updating stopped, endless loop')
            qa = qc
        elif qc > q:
            if b != c:
                b = c
            else:
                raise RuntimeError('updating stopped, endless loop')
            qb = qc
        else:
            return c
    return NAN

def _clip01(v: float) -> float:
    return v if v != v else min(max(v, 0.0), 1.0)

def _deval(Kr: type, op: Literal[str], x: float, loc: float, sh) -> float:
    if not _k_check(Kr, sh):
        return NAN
    a = _k_a(Kr, sh)
    b = _k_b(Kr, sh)
    if op == 'ppf' or op == 'isf':
        q = x
        if not (loc == loc) or not (q >= 0 and q <= 1):
            return NAN
        if op == 'ppf':
            if q == 0:
                return a - 1 + loc
            if q == 1:
                return b + loc
            return _k_ppf(Kr, q, sh) + loc
        else:
            if q == 1:
                return a - 1 + loc
            if q == 0:
                return b + loc
            if hasattr(Kr, 'isf'):
                return float(Kr.isf(q, sh)) + loc
            return _k_ppf(Kr, 1.0 - q, sh) + loc
    if x != x:
        return NAN
    k = x - loc
    if op == 'pmf' or op == 'logpmf':
        ok = k >= a and k <= b and K.floor(k) == k
        if op == 'pmf':
            return _clip01(_k_pdf(Kr, k, sh)) if ok else 0.0
        return _k_logpdf(Kr, k, sh) if ok else -INF
    elif op == 'cdf':
        if k >= b:
            return 1.0
        if not (k >= a) or K.isinf(k):
            return 0.0
        return _clip01(_k_cdf(Kr, k, sh))
    elif op == 'sf':
        if k < a:
            return 1.0
        if not (k < b) or K.isinf(k):
            return 0.0
        return _clip01(_k_sf(Kr, k, sh))
    elif op == 'logcdf':
        if k >= b:
            return 0.0
        if not (k >= a):
            return -INF
        return _k_logcdf(Kr, k, sh)
    elif op == 'logsf':
        if k < a:
            return 0.0
        if not (k < b):
            return -INF
        return _k_logsf(Kr, k, sh)
    else:
        compile_error("unknown discrete distribution method")

def _dbcast(Kr: type, op: Literal[str], x, loc, sh):
    if _anyarr(x, loc, *sh):
        arrs = (np.asarray(x), np.asarray(loc)) + tuple(np.asarray(s) for s in sh)
        shp = np.broadcast_shapes(*tuple(a.shape for a in arrs))
        bs = tuple(np.broadcast_to(a, shp) for a in arrs)
        out = np.empty(shp, dtype=float)
        for idx in np.ndindex(shp):
            vals = tuple(float(b[idx]) for b in bs)
            out[idx] = _deval(Kr, op, vals[0], vals[1], vals[2:])
        return out
    else:
        return _deval(Kr, op, float(x), float(loc), tuple(float(s) for s in sh))

def _dsplit(Kr: type, args, loc):
    NS: Literal[int] = Kr.nshape()
    if static.len(args) < NS:
        compile_error("missing shape parameter(s) for distribution")
    sh = args[:NS]
    extra = args[NS:]
    if static.len(extra) > 1:
        compile_error("too many positional arguments")
    if static.len(extra) == 1:
        l = extra[0]
    else:
        l = loc
    return sh, l

def _dentropy1(Kr: type, l: float, s: float, sh) -> float:
    if not (l == l) or not _k_check(Kr, sh):
        return NAN
    return _k_entropy(Kr, sh)

def _drvs(Kr: type, sh, loc, size, random_state):
    vals = _rvs(Kr, sh, loc, 1.0, size, random_state)
    if isinstance(vals, np.ndarray):
        return vals.astype(int)
    else:
        return int(vals)

class _DExpF[Kr, S, F]:
    sh: S
    func: F
    loc: float
    def __init__(self, sh: S, func: F, loc: float):
        self.sh = sh
        self.func = func
        self.loc = loc
    def __call__(self, x: float) -> float:
        p = _k_pdf(Kr, x, self.sh)
        if self.func is None:
            return (x + self.loc) * p
        else:
            return float(self.func(x + self.loc)) * p

def _dexpect_api(Kr: type, func, sh, loc: float, lb, ub, conditional: bool) -> float:
    a = _k_a(Kr, sh)
    b = _k_b(Kr, sh)
    lbv = a if lb is None else float(lb) - loc
    ubv = b if ub is None else float(ub) - loc
    invfac = 1.0
    if conditional:
        invfac = _deval(Kr, 'sf', lbv - 1, 0.0, sh) - _deval(Kr, 'sf', ubv, 0.0, sh)
    f = _DExpF[Kr, type(sh), type(func)](sh, func, loc)
    return _dexpect(f, lbv, ubv, _k_ppf(Kr, 0.5, sh)) / invfac

class rv_discrete_frozen[Kr, S, L]:
    sh: S
    loc: L

    def __init__(self, sh: S, loc: L):
        self.sh = sh
        self.loc = loc

    @property
    def args(self):
        return self.sh

    def pmf(self, k): return _dbcast(Kr, 'pmf', k, self.loc, self.sh)
    def logpmf(self, k): return _dbcast(Kr, 'logpmf', k, self.loc, self.sh)
    def cdf(self, k): return _dbcast(Kr, 'cdf', k, self.loc, self.sh)
    def logcdf(self, k): return _dbcast(Kr, 'logcdf', k, self.loc, self.sh)
    def sf(self, k): return _dbcast(Kr, 'sf', k, self.loc, self.sh)
    def logsf(self, k): return _dbcast(Kr, 'logsf', k, self.loc, self.sh)
    def ppf(self, q): return _dbcast(Kr, 'ppf', q, self.loc, self.sh)
    def isf(self, q): return _dbcast(Kr, 'isf', q, self.loc, self.sh)
    def stats(self, moments: Literal[str] = 'mv'):
        return _stats_eval(Kr, moments, self.loc, 1.0, self.sh)
    def mean(self): return _stats_eval(Kr, 'm', self.loc, 1.0, self.sh)
    def var(self): return _stats_eval(Kr, 'v', self.loc, 1.0, self.sh)
    def std(self): return np.sqrt(_stats_eval(Kr, 'v', self.loc, 1.0, self.sh))
    def median(self): return self.ppf(0.5)
    def moment(self, order: int):
        return _moment1(Kr, order, float(self.loc), 1.0, tuple(float(v) for v in self.sh))
    def entropy(self): return _vec_ls(_dentropy1, Kr, self.loc, 1.0, self.sh)
    def interval(self, confidence):
        c = confidence
        return (self.ppf((1.0 - c) / 2.0), self.ppf((1.0 + c) / 2.0))
    def support(self):
        shv = tuple(float(v) for v in self.sh)
        l = float(self.loc)
        return (_k_a(Kr, shv) + l, _k_b(Kr, shv) + l)
    def rvs(self, size = None, random_state = None):
        return _drvs(Kr, self.sh, self.loc, size, random_state)
    def expect(self, func = None, lb = None, ub = None, conditional: bool = False):
        return _dexpect_api(Kr, func, tuple(float(v) for v in self.sh), float(self.loc),
                            lb, ub, conditional)

class rv_discrete[Kr]:
    name: str

    def __init__(self, name: str):
        self.name = name

    @property
    def numargs(self) -> int:
        return Kr.nshape()

    def __call__(self, *args, loc = 0.0):
        sh, l = _dsplit(Kr, args, loc)
        return rv_discrete_frozen[Kr, type(sh), type(l)](sh, l)
    def freeze(self, *args, loc = 0.0):
        return self(*args, loc=loc)

    def pmf(self, k, *args, loc = 0.0):
        sh, l = _dsplit(Kr, args, loc)
        return _dbcast(Kr, 'pmf', k, l, sh)
    def logpmf(self, k, *args, loc = 0.0):
        sh, l = _dsplit(Kr, args, loc)
        return _dbcast(Kr, 'logpmf', k, l, sh)
    def cdf(self, k, *args, loc = 0.0):
        sh, l = _dsplit(Kr, args, loc)
        return _dbcast(Kr, 'cdf', k, l, sh)
    def logcdf(self, k, *args, loc = 0.0):
        sh, l = _dsplit(Kr, args, loc)
        return _dbcast(Kr, 'logcdf', k, l, sh)
    def sf(self, k, *args, loc = 0.0):
        sh, l = _dsplit(Kr, args, loc)
        return _dbcast(Kr, 'sf', k, l, sh)
    def logsf(self, k, *args, loc = 0.0):
        sh, l = _dsplit(Kr, args, loc)
        return _dbcast(Kr, 'logsf', k, l, sh)
    def ppf(self, q, *args, loc = 0.0):
        sh, l = _dsplit(Kr, args, loc)
        return _dbcast(Kr, 'ppf', q, l, sh)
    def isf(self, q, *args, loc = 0.0):
        sh, l = _dsplit(Kr, args, loc)
        return _dbcast(Kr, 'isf', q, l, sh)
    def stats(self, *args, loc = 0.0, moments: Literal[str] = 'mv'):
        sh, l = _dsplit(Kr, args, loc)
        return _stats_eval(Kr, moments, l, 1.0, sh)
    def mean(self, *args, loc = 0.0):
        sh, l = _dsplit(Kr, args, loc)
        return _stats_eval(Kr, 'm', l, 1.0, sh)
    def var(self, *args, loc = 0.0):
        sh, l = _dsplit(Kr, args, loc)
        return _stats_eval(Kr, 'v', l, 1.0, sh)
    def std(self, *args, loc = 0.0):
        sh, l = _dsplit(Kr, args, loc)
        return np.sqrt(_stats_eval(Kr, 'v', l, 1.0, sh))
    def moment(self, order: int, *args, loc = 0.0):
        sh, l = _dsplit(Kr, args, loc)
        return _moment1(Kr, order, float(l), 1.0, tuple(float(v) for v in sh))
    def entropy(self, *args, loc = 0.0):
        sh, l = _dsplit(Kr, args, loc)
        return _vec_ls(_dentropy1, Kr, l, 1.0, sh)
    def median(self, *args, loc = 0.0):
        return self.ppf(0.5, *args, loc=loc)
    def interval(self, confidence, *args, loc = 0.0):
        c = confidence
        return (self.ppf((1.0 - c) / 2.0, *args, loc=loc),
                self.ppf((1.0 + c) / 2.0, *args, loc=loc))
    def support(self, *args, loc = 0.0):
        sh, l = _dsplit(Kr, args, loc)
        shv = tuple(float(v) for v in sh)
        return (_k_a(Kr, shv) + float(l), _k_b(Kr, shv) + float(l))
    def rvs(self, *args, loc = 0.0, size = None, random_state = None):
        sh, l = _dsplit(Kr, args, loc)
        return _drvs(Kr, sh, l, size, random_state)
    def expect(self, func = None, args = (), loc = 0.0, lb = None, ub = None,
               conditional: bool = False):
        return _dexpect_api(Kr, func, tuple(float(v) for v in args), float(loc),
                            lb, ub, conditional)
