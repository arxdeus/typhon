# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# sigmaclip, obrientransform, gzscore, directional_stats, expectile, lmoment,
# chatterjeexi, quantile_test.

import numpy as np
import internal.static as static
import scipy.special._kernels as K
from ._common import _check_policy, _omit1, _f64, _slices, _reduce
from ._stats import SignificanceResult, zscore
from ._morestats import ConfidenceInterval
from ._discrete_distns import _binom_cdf, _binom_sf, _search_ppf, _search_isf, _BinomK

NAN = K.NAN
INF = K.INF

# ------------------------------------------------------------------ sigmaclip

class SigmaclipResult[C]:
    clipped: C
    lower: float
    upper: float
    def __init__(self, clipped: C, lower: float, upper: float):
        self.clipped = clipped
        self.lower = lower
        self.upper = upper
    def __getitem__(self, i: Literal[int]):
        if i == 0 or i == -3:
            return self.clipped
        elif i == 1 or i == -2:
            return self.lower
        elif i == 2 or i == -1:
            return self.upper
        else:
            compile_error("index out of range")
    def __len__(self):
        return 3
    def __iter__(self):
        yield self.clipped
        yield self.lower
        yield self.upper

def sigmaclip(a, low: float = 4., high: float = 4., *, nan_policy: str = 'propagate'):
    _check_policy(a, nan_policy)
    c = _f64(a).ravel()
    has_nan = bool(np.any(np.isnan(c)))
    if has_nan:
        if nan_policy == 'propagate':
            return SigmaclipResult(np.empty(0, dtype=float), NAN, NAN)
        c = _omit1(c)
    delta = 1
    critlower = NAN
    critupper = NAN
    while delta != 0:
        c_std = float(np.std(c))
        c_mean = float(np.mean(c))
        size = c.shape[0]
        critlower = c_mean - c_std * low
        critupper = c_mean + c_std * high
        keep = []
        for i in range(size):
            v = c[i]
            if v >= critlower and v <= critupper:
                keep.append(v)
        c = np.array(keep, dtype=float)
        delta = size - c.shape[0]
    return SigmaclipResult(c, critlower, critupper)

# ------------------------------------------------------------------ obrientransform

def _obrien1(x0, omit: bool):
    x = _omit1(x0) if omit else x0
    n = float(x.shape[0])
    mu = float(np.mean(x))
    sq = (x0 - mu) ** 2
    sumsq = float(np.mean(sq)) * n
    return ((n - 1.5) * n * sq - 0.5 * sumsq) / ((n - 1) * (n - 2))

class _Obrien:
    omit: bool
    def __call__(self, a):
        x = _f64(a).ravel()
        if self.omit:
            xo = _omit1(x)
            n = float(xo.shape[0])
            mu = float(np.mean(xo))
            sq = (x - mu) ** 2
            sumsq = float(np.mean(_omit1(sq))) * n
            return ((n - 1.5) * n * sq - 0.5 * sumsq) / ((n - 1) * (n - 2))
        n = float(x.shape[0])
        mu = float(np.mean(x))
        sq = (x - mu) ** 2
        sumsq = float(np.mean(sq)) * n
        return ((n - 1.5) * n * sq - 0.5 * sumsq) / ((n - 1) * (n - 2))

def _tmap1(f, d):
    if static.len(d) == 0:
        return ()
    else:
        return (f(d[0]),) + _tmap1(f, d[1:])

def obrientransform(*samples, nan_policy: str = 'propagate'):
    if nan_policy != 'propagate' and nan_policy != 'omit' and nan_policy != 'raise':
        raise ValueError("nan_policy must be one of {'propagate', 'raise', 'omit'}")
    return _tmap1(_Obrien(nan_policy == 'omit'), samples)

# ------------------------------------------------------------------ gzscore

def gzscore(a, *, axis = 0, ddof: int = 0, nan_policy: str = 'propagate'):
    return zscore(np.log(_f64(a)), axis=axis, ddof=ddof, nan_policy=nan_policy)

# ------------------------------------------------------------------ directional_stats

class DirectionalStats[M, R]:
    mean_direction: M
    mean_resultant_length: R
    def __init__(self, mean_direction: M, mean_resultant_length: R):
        self.mean_direction = mean_direction
        self.mean_resultant_length = mean_resultant_length
    def __getitem__(self, i: Literal[int]):
        if i == 0 or i == -2:
            return self.mean_direction
        elif i == 1 or i == -1:
            return self.mean_resultant_length
        else:
            compile_error("index out of range")
    def __len__(self):
        return 2
    def __iter__(self):
        yield self.mean_direction
        yield self.mean_resultant_length

def directional_stats(samples, *, axis: int = 0, normalize: bool = True):
    s = _f64(samples)
    if s.ndim < 2:
        raise ValueError("samples must at least be two-dimensional. Instead samples has shape: " + str(s.shape))
    s = np.moveaxis(s, axis, 0)
    if normalize:
        nrm = np.sqrt(np.sum(s * s, axis=-1, keepdims=True))
        s = s / nrm
    mean = np.mean(s, axis=0)
    mrl = np.sqrt(np.sum(mean * mean, axis=-1, keepdims=True))
    md = mean / mrl
    r = np.squeeze(mrl, axis=-1)
    if r.ndim == 0:
        return DirectionalStats(md, float(r))
    else:
        return DirectionalStats(md, r)

# ------------------------------------------------------------------ expectile

class _Expectile:
    alpha: float
    w: np.ndarray[float, 1]
    hasw: bool
    def __call__(self, x):
        n = x.shape[0]
        if n == 0:
            return NAN
        sw = 0.0
        t = 0.0
        for i in range(n):
            wi = self.w[i] if self.hasw else 1.0
            sw += wi
            t += wi * x[i]
        t /= sw
        for _ in range(1000):
            num = 0.0
            den = 0.0
            for i in range(n):
                wi = self.w[i] if self.hasw else 1.0
                c = abs((1.0 if x[i] <= t else 0.0) - self.alpha) * wi
                num += c * x[i]
                den += c
            nt = num / den
            if nt == t:
                break
            t = nt
        return t

def expectile(a, alpha: float = 0.5, *, weights = None, axis = None):
    if alpha < 0 or alpha > 1:
        raise ValueError("The expectile level alpha must be in the range [0, 1].")
    if weights is None:
        return _reduce(a, axis, _Expectile(alpha, np.ones(1), False))
    else:
        if axis is None:
            return _reduce(a, axis, _Expectile(alpha, _f64(weights).ravel(), True))
        else:
            raise ValueError("`weights` with `axis` is not supported; use axis=None")

# ------------------------------------------------------------------ lmoment

def _binom_f(n: float, k: float) -> float:
    if k < 0 or k > n:
        return 0.0
    r = 1.0
    kk = min(k, n - k)
    i = 1.0
    while i <= kk:
        r = r * (n - kk + i) / i
        i += 1.0
    return r

def _lmom1(x0, orders: List[int], standardize: bool, is_sorted: bool):
    x = x0 if is_sorted else np.sort(x0)
    n = x.shape[0]
    nm = 0
    for o in orders:
        if o > nm:
            nm = o
    bk = np.zeros(nm)
    for r in range(nm):
        if r >= n:
            bk[r] = 0.0
            continue
        s = 0.0
        for j in range(r, n):
            s += _binom_f(float(j), float(r)) * x[j]
        bk[r] = s / _binom_f(float(n - 1), float(r)) / n
    lm = np.zeros(nm)
    for r in range(nm):
        s = 0.0
        for k in range(r + 1):
            sign = 1.0 if (r - k) % 2 == 0 else -1.0
            s += sign * _binom_f(float(r), float(k)) * _binom_f(float(r + k), float(k)) * bk[k]
        lm[r] = s
    if standardize and nm > 2:
        for r in range(2, nm):
            lm[r] = lm[r] / lm[1]
    for r in range(nm):
        if r >= n:
            lm[r] = NAN
    out = np.empty(len(orders))
    for i in range(len(orders)):
        out[i] = lm[orders[i] - 1]
    return out

def lmoment(sample, order = None, *, axis = 0, sorted: bool = False, standardize: bool = True):
    if isinstance(order, NoneType):
        return _lmoment_many(sample, [1, 2, 3, 4], axis, sorted, standardize)
    elif isinstance(order, int):
        if order <= 0:
            raise ValueError("`order` must be a scalar or a non-empty array of positive integers.")
        r = _lmoment_many(sample, [order], axis, sorted, standardize)
        if r.ndim == 1:
            return float(r[0])
        return r[0]
    else:
        ol = [int(o) for o in order]
        if len(ol) == 0:
            raise ValueError("`order` must be a scalar or a non-empty array of positive integers.")
        for o in ol:
            if o <= 0:
                raise ValueError("`order` must be a scalar or a non-empty array of positive integers.")
        return _lmoment_many(sample, ol, axis, sorted, standardize)

def _lmoment_many(sample, orders: List[int], axis, is_sorted: bool, standardize: bool):
    m, shp = _slices(sample, axis)
    k = len(orders)
    if static.len(shp) == 0:
        return _lmom1(np.ascontiguousarray(m), orders, standardize, is_sorted)
    else:
        out = np.empty((k,) + shp, dtype=float)
        for idx in np.ndindex(shp):
            r = _lmom1(np.ascontiguousarray(m[idx]), orders, standardize, is_sorted)
            for i in range(k):
                out[(i,) + idx] = r[i]
        return out

# ------------------------------------------------------------------ chatterjeexi

def _rank_max(y):
    n = y.shape[0]
    order = np.argsort(y, kind='mergesort')
    out = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and y[order[j + 1]] == y[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = float(j + 1)
        i = j + 1
    return out

class _Xi:
    y_continuous: bool
    def __call__(self, x, y0):
        n = x.shape[0]
        j = np.argsort(x)
        y = np.empty(n, dtype=float)
        for i in range(n):
            y[i] = y0[j[i]]
        r = _rank_max(y)
        l = _rank_max(-y)
        num = 0.0
        for i in range(n - 1):
            num += abs(r[i + 1] - r[i])
        nf = float(n)
        xi = 0.0
        if self.y_continuous:
            xi = 1 - 3 * num / (nf * nf - 1)
        else:
            den = 0.0
            for i in range(n):
                den += (nf - l[i]) * l[i]
            den *= 2
            xi = 1 - nf * num / den
        std = 0.0
        if self.y_continuous:
            std = K.sqrt(2.0 / 5.0) / K.sqrt(nf)
        else:
            u = np.sort(r)
            v = np.cumsum(u)
            an = 0.0
            bn = 0.0
            cn = 0.0
            dn = 0.0
            for i in range(n):
                ii = float(i + 1)
                an += (2 * nf - 2 * ii + 1) * u[i] ** 2
                bn += (v[i] + (nf - ii) * u[i]) ** 2
                cn += (2 * nf - 2 * ii + 1) * u[i]
                dn += l[i] * (nf - l[i])
            an = an / nf ** 4
            bn = bn / nf ** 5
            cn = cn / nf ** 3
            dn = dn / nf ** 3
            tau2 = (an - 2 * bn + cn ** 2) / dn ** 2
            std = K.sqrt(tau2) / K.sqrt(nf)
        return (xi, K.ndtr(-(xi / std)))

def chatterjeexi(x, y, *, axis = 0, y_continuous: bool = False, method: str = 'asymptotic'):
    if method.lower() != 'asymptotic':
        raise ValueError("`method` must be 'asymptotic' or a `PermutationMethod` instance.")
    xx = _f64(x)
    yy = _f64(y)
    if axis is None:
        xr = xx.ravel()
        yr = yy.ravel()
        r = _Xi(y_continuous)(xr, yr)
        return SignificanceResult(r[0], r[1])
    ax = int(axis)
    if xx.ndim == 1 and yy.ndim == 1:
        r = _Xi(y_continuous)(xx, yy)
        return SignificanceResult(r[0], r[1])
    mx = np.moveaxis(xx, ax, -1)
    my = np.moveaxis(yy, ax, -1)
    shp = np.broadcast_shapes(mx.shape[:-1], my.shape[:-1])
    bx = np.broadcast_to(mx, shp + (mx.shape[-1],))
    by = np.broadcast_to(my, shp + (my.shape[-1],))
    s = np.empty(shp, dtype=float)
    p = np.empty(shp, dtype=float)
    f = _Xi(y_continuous)
    for idx in np.ndindex(shp):
        r = f(np.ascontiguousarray(bx[idx]), np.ascontiguousarray(by[idx]))
        s[idx] = r[0]
        p[idx] = r[1]
    return SignificanceResult(s, p)

# ------------------------------------------------------------------ quantile_test

class QuantileTestResult[S, T, P]:
    statistic: S
    statistic_type: T
    pvalue: P
    _x: np.ndarray[float, 2]
    _p: float
    _alt: str
    _lead: Tuple[int, int]
    def __init__(self, statistic: S, statistic_type: T, pvalue: P, x: np.ndarray[float, 2],
                 p: float, alt: str, lead: Tuple[int, int]):
        self.statistic = statistic
        self.statistic_type = statistic_type
        self.pvalue = pvalue
        self._x = x
        self._p = p
        self._alt = alt
        self._lead = lead

    def confidence_interval(self, confidence_level: float = 0.95):
        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError("`confidence_level` must be a number between 0 and 1.")
        L = self._x.shape[0]
        n = self._x.shape[1]
        lo = np.empty(L)
        hi = np.empty(L)
        sh = (float(n), self._p)
        for l in range(L):
            lo[l] = NAN
            hi[l] = NAN
            if n == 0:
                lo[l] = 0.0
                hi[l] = 0.0
                continue
            if self._alt == 'less':
                p = 1 - confidence_level
                lo[l] = -INF
                hix = int(_search_isf(_BinomK, p, sh, n * self._p, 0.0, float(n)))
                if hix < n:
                    hi[l] = self._x[l, hix]
            elif self._alt == 'greater':
                p = 1 - confidence_level
                lox = int(_search_ppf(_BinomK, p, sh, n * self._p, 0.0, float(n))) - 1
                if lox >= 0:
                    lo[l] = self._x[l, lox]
                hi[l] = INF
            else:
                p = (1 - confidence_level) / 2
                lox = int(_search_ppf(_BinomK, p, sh, n * self._p, 0.0, float(n))) - 1
                if lox >= 0:
                    lo[l] = self._x[l, lox]
                hix = int(_search_isf(_BinomK, p, sh, n * self._p, 0.0, float(n)))
                if hix < n:
                    hi[l] = self._x[l, hix]
        return ConfidenceInterval(float(lo[0]), float(hi[0]))

def _bin_cdf(t: float, n: float, p: float) -> float:
    if t < 0:
        return 0.0
    return _binom_cdf(K.floor(t), n, p)

def _bin_sf(t: float, n: float, p: float) -> float:
    if t < 0:
        return 1.0
    return _binom_sf(K.floor(t), n, p)

def quantile_test(x, *, q: float = 0.0, p: float = 0.5, alternative: str = 'two-sided', axis: int = 0):
    """Quantile test for 1-D samples (scalar ``q`` and ``p``)."""
    if alternative != 'two-sided' and alternative != 'less' and alternative != 'greater':
        raise ValueError("`alternative` must be one of {'two-sided', 'less', 'greater'}")
    xs = np.sort(_f64(x).ravel())
    n = xs.shape[0]
    nf = float(n)
    nan_out = p >= 1 or p <= 0 or p != p or q != q
    for v in xs:
        if v != v:
            nan_out = True
    pp = 0.5 if nan_out else p
    qq = 0.0 if nan_out else q
    T1 = 0.0
    T2 = 0.0
    if n > 0:
        T1 = float(np.searchsorted(xs, qq, side='right'))
        T2 = float(np.searchsorted(xs, qq, side='left'))
    else:
        nan_out = True
    stat = 0.0
    stype = 0.0
    pv = 0.0
    if alternative == 'less':
        pv = _bin_sf(T2 - 1, nf, pp)
        stat = T2
        stype = 2.0
    elif alternative == 'greater':
        pv = _bin_cdf(T1, nf, pp)
        stat = T1
        stype = 1.0
    else:
        c1 = _bin_cdf(T1, nf, pp)
        c2 = _bin_sf(T2 - 1, nf, pp)
        pv = min(max(2 * min(c1, c2), 0.0), 1.0)
        if c1 > c2:
            stat = T2
            stype = 2.0
        else:
            stat = T1
            stype = 1.0
    if nan_out:
        stat = NAN
        stype = NAN
        pv = NAN
    X = np.empty((1, n), dtype=float)
    for i in range(n):
        X[0, i] = xs[i]
    return QuantileTestResult(stat, stype, pv, X, pp, alternative, (1, n))
