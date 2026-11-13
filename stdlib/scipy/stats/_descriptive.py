# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Descriptive statistics: means, trimmed statistics, dispersion, entropy,
# frequency tables, circular statistics, Box-Cox / Yeo-Johnson.

import numpy as np
import internal.static as static
import scipy.special._kernels as K
from C import atan2(float, float) -> float
from scipy.optimize import brent, brentq, fminbound
from ._stats import pearsonr
from ._common import _f64, _slices, _reduce, _reduce2, _mean, _var

NAN = K.NAN
INF = K.INF
_PI = 3.141592653589793

# ------------------------------------------------------------------ results

@tuple
class ModeResult[M, C]:
    mode: M
    count: C

@tuple
class CumfreqResult[H]:
    cumcount: H
    lowerlimit: float
    binsize: float
    extrapoints: int

@tuple
class RelfreqResult[H]:
    frequency: H
    lowerlimit: float
    binsize: float
    extrapoints: int

@tuple
class RepeatedResults:
    values: np.ndarray[float, 1]
    counts: np.ndarray[int, 1]

# ------------------------------------------------------------------ means

class _Gmean:
    def __call__(self, x) -> float:
        return K.exp(float(np.mean(np.log(x))))

class _GmeanW:
    def __call__(self, x, w) -> float:
        return K.exp(float(np.sum(np.log(x) * w)) / float(np.sum(w)))

def gmean(a, axis = 0, dtype = None, weights = None):
    if weights is None:
        return _reduce(a, axis, _Gmean())
    else:
        aa = _f64(a)
        ww = np.broadcast_to(_f64(weights), aa.shape)
        return _reduce2(aa, ww, axis, _GmeanW())

def _neg_to_nan(x):
    return np.where(x < 0, NAN, x)

class _Hmean:
    def __call__(self, x) -> float:
        return 1.0 / float(np.mean(1.0 / _neg_to_nan(x)))

class _HmeanW:
    def __call__(self, x, w) -> float:
        return 1.0 / (float(np.sum((1.0 / _neg_to_nan(x)) * w)) / float(np.sum(w)))

def hmean(a, axis = 0, dtype = None, *, weights = None):
    if weights is None:
        return _reduce(a, axis, _Hmean())
    else:
        aa = _f64(a)
        ww = np.broadcast_to(_f64(weights), aa.shape)
        return _reduce2(aa, ww, axis, _HmeanW())

class _Pmean:
    p: float
    def __call__(self, x) -> float:
        p = self.p
        y = _neg_to_nan(x)
        if abs(p) <= 2e-6:
            # linearized: exp(mean(log x) + p/2 * var(log x))
            lx = np.log(y)
            m = float(np.mean(lx))
            v = float(np.mean((lx - m) ** 2))
            return K.exp(m + p / 2.0 * v)
        return float(np.mean(y ** p)) ** (1.0 / p)

class _PmeanW:
    p: float
    def __call__(self, x, w) -> float:
        p = self.p
        y = _neg_to_nan(x)
        sw = float(np.sum(w))
        if abs(p) <= 2e-6:
            lx = np.log(y)
            m = float(np.sum(lx * w)) / sw
            v = float(np.sum((lx - m) ** 2 * w)) / sw
            return K.exp(m + p / 2.0 * v)
        return (float(np.sum(y ** p * w)) / sw) ** (1.0 / p)

def pmean(a, p: float, *, axis = 0, dtype = None, weights = None):
    if p == 0.0:
        return gmean(a, axis=axis, weights=weights)
    if p == INF or p == -INF:
        raise NotImplementedError("Power mean only implemented for finite `p`")
    if weights is None:
        return _reduce(a, axis, _Pmean(p))
    else:
        aa = _f64(a)
        ww = np.broadcast_to(_f64(weights), aa.shape)
        return _reduce2(aa, ww, axis, _PmeanW(p))

# ------------------------------------------------------------------ mode

def _mode1(x):
    n = x.shape[0]
    if n == 0:
        return (NAN, 0)
    y = np.sort(x)
    best = y[0]
    bestc = 0
    nnan = 0
    i = 0
    while i < n:
        if y[i] != y[i]:
            nnan += 1
            i += 1
            continue
        j = i
        while j + 1 < n and y[j + 1] == y[i]:
            j += 1
        c = j - i + 1
        if c > bestc:
            bestc = c
            best = y[i]
        i = j + 1
    if nnan > bestc:
        return (NAN, nnan)
    return (best, bestc)

def mode(a, axis = 0, nan_policy: str = 'propagate', keepdims: bool = False):
    x = np.asarray(a)
    m, shp = _slices(x, axis)
    if static.len(shp) == 0:
        r = _mode1(np.ascontiguousarray(m))
        return ModeResult(r[0], r[1])
    else:
        mo = np.empty(shp, dtype=float)
        co = np.empty(shp, dtype=int)
        for idx in np.ndindex(shp):
            r = _mode1(np.ascontiguousarray(m[idx]))
            mo[idx] = r[0]
            co[idx] = r[1]
        return ModeResult(mo, co)

# ------------------------------------------------------------------ trimmed

def _limits(limits):
    """-> (has_lo, lo, has_hi, hi)"""
    if limits is None:
        return (False, 0.0, False, 0.0)
    else:
        lo = limits[0]
        hi = limits[1]
        hl = False
        lv = 0.0
        hh = False
        hv = 0.0
        if lo is not None:
            hl = True
            lv = float(lo)
        if hi is not None:
            hh = True
            hv = float(hi)
        return (hl, lv, hh, hv)

class _TMask:
    hl: bool
    lo: float
    hh: bool
    hi: float
    il: bool
    ih: bool
    def keep(self, v: float) -> bool:
        if self.hl:
            if self.il:
                if v < self.lo: return False
            else:
                if v <= self.lo: return False
        if self.hh:
            if self.ih:
                if v > self.hi: return False
            else:
                if v >= self.hi: return False
        return True

def _tmask(limits, inclusive) -> _TMask:
    l = _limits(limits)
    return _TMask(l[0], l[1], l[2], l[3], bool(inclusive[0]), bool(inclusive[1]))

def _tkept(x, mk: _TMask):
    out = []
    for v in x:
        if mk.keep(v):
            out.append(v)
    return np.array(out, dtype=float)

class _TMean:
    mk: _TMask
    def __call__(self, x) -> float:
        # scipy: sum with masked values set to 0 / count
        y = x.copy()
        n = 0.0
        for i in range(x.shape[0]):
            if self.mk.keep(x[i]):
                n += 1.0
            else:
                y[i] = 0.0
        s = float(np.sum(y))
        return s / n if n != 0 else NAN

def _check_all_masked(a, mk: _TMask):
    if mk.hl or mk.hh:
        anyk = False
        for v in _f64(a).flat:
            if mk.keep(v):
                anyk = True
                break
        if not anyk:
            raise ValueError("No array values within given limits")

def tmean(a, limits = None, inclusive = (True, True), axis = None):
    mk = _tmask(limits, inclusive)
    _check_all_masked(a, mk)
    return _reduce(a, axis, _TMean(mk))

class _TVar:
    mk: _TMask
    ddof: int
    def __call__(self, x) -> float:
        # nan-omitting variance of the kept values (weights 0 at masked)
        y = x.copy()
        n = 0.0
        for i in range(x.shape[0]):
            if self.mk.keep(x[i]) and x[i] == x[i]:
                n += 1.0
            else:
                y[i] = 0.0
        if n == 0.0:
            return NAN
        m = float(np.sum(y)) / n
        d = y - m
        for i in range(x.shape[0]):
            if not (self.mk.keep(x[i]) and x[i] == x[i]):
                d[i] = 0.0
        v = float(np.sum(d * d)) / n
        nc = n - self.ddof
        if self.ddof != 0:
            v *= (n / nc) if nc > 0 else NAN
        return v

class _TSem:
    mk: _TMask
    ddof: int
    def __call__(self, x) -> float:
        v = _TVar(self.mk, self.ddof)(x)
        n = 0.0
        for i in range(x.shape[0]):
            if self.mk.keep(x[i]) and x[i] == x[i]:
                n += 1.0
        return v ** 0.5 / n ** 0.5

def tvar(a, limits = None, inclusive = (True, True), axis = 0, ddof: int = 1):
    mk = _tmask(limits, inclusive)
    _check_all_masked(a, mk)
    return _reduce(a, axis, _TVar(mk, ddof))

def tstd(a, limits = None, inclusive = (True, True), axis = 0, ddof: int = 1):
    return tvar(a, limits, inclusive, axis, ddof) ** 0.5

def tsem(a, limits = None, inclusive = (True, True), axis = 0, ddof: int = 1):
    mk = _tmask(limits, inclusive)
    _check_all_masked(a, mk)
    return _reduce(a, axis, _TSem(mk, ddof))

class _TMin:
    lo: float
    has: bool
    inc: bool
    def __call__(self, x) -> float:
        r = INF
        anyk = False
        for v in x:
            if self.has and ((v < self.lo) if self.inc else (v <= self.lo)):
                continue
            anyk = True
            if v != v:
                return NAN
            if v < r:
                r = v
        return r if anyk else NAN

class _TMax:
    hi: float
    has: bool
    inc: bool
    def __call__(self, x) -> float:
        r = -INF
        anyk = False
        for v in x:
            if self.has and ((v > self.hi) if self.inc else (v >= self.hi)):
                continue
            anyk = True
            if v != v:
                return NAN
            if v > r:
                r = v
        return r if anyk else NAN

def tmin(a, lowerlimit = None, axis = 0, inclusive: bool = True, nan_policy: str = 'propagate'):
    if lowerlimit is None:
        return _reduce(a, axis, _TMin(0.0, False, inclusive))
    else:
        return _reduce(a, axis, _TMin(float(lowerlimit), True, inclusive))

def tmax(a, upperlimit = None, axis = 0, inclusive: bool = True, nan_policy: str = 'propagate'):
    if upperlimit is None:
        return _reduce(a, axis, _TMax(0.0, False, inclusive))
    else:
        return _reduce(a, axis, _TMax(float(upperlimit), True, inclusive))

def _cut(a, lowercut: int, uppercut: int, axis):
    if axis is None:
        x = np.sort(np.asarray(a).ravel())
        return x[lowercut:uppercut]
    else:
        x = np.sort(np.asarray(a), axis=axis)
        return np.moveaxis(np.moveaxis(x, axis, -1)[..., lowercut:uppercut], -1, axis)

def trimboth(a, proportiontocut: float, axis = 0):
    """Slices off ``proportiontocut`` of both tails. The kept values are
    returned sorted along ``axis`` (scipy returns them partitioned)."""
    x = np.asarray(a)
    nobs = x.size if axis is None else x.shape[int(axis)]
    lowercut = int(proportiontocut * nobs)
    uppercut = nobs - lowercut
    if lowercut >= uppercut:
        raise ValueError("Proportion too big.")
    return _cut(x, lowercut, uppercut, axis)

def trim1(a, proportiontocut: float, tail: str = 'right', axis = 0):
    x = np.asarray(a)
    nobs = x.size if axis is None else x.shape[int(axis)]
    if proportiontocut >= 1:
        return _cut(x, 0, 0, axis)
    if tail.lower() == 'right':
        lowercut = 0
        uppercut = nobs - int(proportiontocut * nobs)
    elif tail.lower() == 'left':
        lowercut = int(proportiontocut * nobs)
        uppercut = nobs
    else:
        raise ValueError("`tail` must be 'left' or 'right'")
    return _cut(x, lowercut, uppercut, axis)

class _TrimMean:
    p: float
    def __call__(self, x) -> float:
        nobs = x.shape[0]
        if nobs == 0:
            return NAN
        lowercut = int(self.p * nobs)
        uppercut = nobs - lowercut
        if lowercut > uppercut:
            raise ValueError("Proportion too big.")
        y = np.sort(x)
        return float(np.mean(y[lowercut:uppercut]))

def trim_mean(a, proportiontocut: float, axis = 0):
    return _reduce(a, axis, _TrimMean(proportiontocut))

# ------------------------------------------------------------------ dispersion

class _Variation:
    ddof: int
    def __call__(self, x) -> float:
        n = float(x.shape[0])
        m = float(np.mean(x))
        s = float(np.std(x))
        if float(self.ddof) == n:
            if s > 0:
                return INF if m >= 0 else -INF
            return NAN
        corr = (n / (n - self.ddof)) ** 0.5
        return s * corr / m

def variation(a, axis = 0, nan_policy: str = 'propagate', ddof: int = 0, *, keepdims: bool = False):
    return _reduce(a, axis, _Variation(ddof))

def _iqr_scale(scale) -> float:
    if isinstance(scale, str):
        if scale.lower() == 'normal':
            return 1.3489795003921634
        raise ValueError(f"{scale} not a valid scale for `iqr`")
    else:
        return float(scale)

def _quantile_sorted(y, q: float, method: str) -> float:
    n = y.shape[0]
    if n == 0:
        return NAN
    v = q * (n - 1)
    lo = int(K.floor(v))
    hi = lo + 1 if lo + 1 < n else n - 1
    g = v - lo
    if method == 'linear':
        # numpy's lerp formula
        a = y[lo]
        b = y[hi]
        d = b - a
        r = a + d * g
        if g >= 0.5:
            r = b - d * (1.0 - g)
        return r
    elif method == 'lower':
        return y[lo]
    elif method == 'higher':
        return y[hi] if g > 0 else y[lo]
    elif method == 'midpoint':
        return (y[lo] + y[hi]) / 2.0 if g > 0 else y[lo]
    elif method == 'nearest':
        if g > 0.5 or (g == 0.5 and lo % 2 == 1):
            return y[hi]
        return y[lo]
    else:
        raise ValueError("unsupported interpolation")

class _Iqr:
    lo: float
    hi: float
    scale: float
    method: str
    def __call__(self, x) -> float:
        for v in x:
            if v != v:
                return NAN
        y = np.sort(x)
        r = _quantile_sorted(y, self.hi, self.method) - _quantile_sorted(y, self.lo, self.method)
        if self.scale != 1.0:
            r /= self.scale
        return r

def iqr(x, axis = None, rng = (25, 75), scale = 1.0, nan_policy: str = 'propagate',
        interpolation: str = 'linear', keepdims: bool = False):
    r0 = float(rng[0])
    r1 = float(rng[1])
    if r0 > r1:
        r0, r1 = r1, r0
    lo = r0 / 100.0
    hi = r1 / 100.0
    if lo < 0 or hi > 1:
        raise ValueError("Elements of `rng` must be in the range [0, 100].")
    return _reduce(x, axis, _Iqr(lo, hi, _iqr_scale(scale), interpolation))

def _median1(x) -> float:
    for v in x:
        if v != v:
            return NAN
    n = x.shape[0]
    y = np.sort(x)
    if n % 2 == 1:
        return y[n // 2]
    return float(np.mean(y[n // 2 - 1:n // 2 + 1]))

class _Mad:
    scale: float
    def __call__(self, x) -> float:
        med = _median1(x)
        return _median1(np.abs(x - med)) / self.scale

def median_abs_deviation(x, axis = 0, center = None, scale = 1.0, nan_policy: str = 'propagate'):
    sc = 1.0
    if isinstance(scale, str):
        if scale.lower() == 'normal':
            sc = 0.6744897501960817
        else:
            raise ValueError(f"{scale} is not a valid scale value.")
    else:
        sc = float(scale)
    return _reduce(x, axis, _Mad(sc))

# ------------------------------------------------------------------ percentiles

def _pos1(a, s: float, kind: str) -> float:
    n = a.shape[0]
    for v in a:
        if v != v:
            return NAN
    if s != s:
        return NAN
    if n == 0:
        return NAN
    left = 0
    right = 0
    for v in a:
        if v < s:
            left += 1
        if v <= s:
            right += 1
    if kind == 'rank':
        plus1 = 1 if left < right else 0
        return (left + right + plus1) * (50.0 / n)
    elif kind == 'strict':
        return left * (100.0 / n)
    elif kind == 'weak':
        return right * (100.0 / n)
    elif kind == 'mean':
        return (left + right) * (50.0 / n)
    else:
        raise ValueError("kind can only be 'rank', 'strict', 'weak' or 'mean'")

def percentileofscore(a, score, kind: str = 'rank', nan_policy: str = 'propagate'):
    x = _f64(a)
    if x.ndim != 1:
        raise ValueError("`a` must be 1-dimensional.")
    if isinstance(score, np.ndarray) or isinstance(score, List) or isinstance(score, Tuple):
        sc = _f64(score)
        out = np.empty(sc.shape, dtype=float)
        for idx in np.ndindex(sc.shape):
            out[idx] = _pos1(x, float(sc[idx]), kind)
        return out
    else:
        return _pos1(x, float(score), kind)

def _qth(y, per: float, method: str) -> float:
    if not (0 <= per <= 100):
        raise ValueError("percentile must be in the range [0, 100]")
    idx = per / 100. * (y.shape[0] - 1)
    if float(int(idx)) != idx:
        if method == 'lower':
            idx = K.floor(idx)
        elif method == 'higher':
            idx = K.ceil(idx)
        elif method == 'fraction':
            pass
        else:
            raise ValueError("interpolation_method can only be 'fraction', 'lower' or 'higher'")
    i = int(idx)
    if float(i) == idx:
        return y[i] * 1.0 / 1.0
    j = i + 1
    w0 = j - idx
    w1 = idx - i
    return (y[i] * w0 + y[i + 1] * w1) / (w0 + w1)

def scoreatpercentile(a, per, limit = (), interpolation_method: str = 'fraction', axis = None):
    x = _f64(a).ravel()
    if static.len(limit) == 2:
        keep = []
        for v in x:
            if float(limit[0]) <= v and v <= float(limit[1]):
                keep.append(v)
        x = np.array(keep, dtype=float)
    y = np.sort(x)
    if isinstance(per, np.ndarray) or isinstance(per, List) or isinstance(per, Tuple):
        pp = _f64(per)
        out = np.empty(pp.shape, dtype=float)
        for idx in np.ndindex(pp.shape):
            out[idx] = NAN if y.shape[0] == 0 else _qth(y, float(pp[idx]), interpolation_method)
        return out
    else:
        if y.shape[0] == 0:
            return NAN
        return _qth(y, float(per), interpolation_method)

# ------------------------------------------------------------------ entropy

class _Entropy:
    base: float
    def __call__(self, pk) -> float:
        s = float(np.sum(pk))
        p = pk / s
        v = np.empty(p.shape[0], dtype=float)
        for i in range(p.shape[0]):
            v[i] = K.entr(p[i])
        S = float(np.sum(v))
        if self.base > 0:
            S /= K.log(self.base)
        return S

class _RelEntropy:
    base: float
    def __call__(self, pk, qk) -> float:
        p = pk / float(np.sum(pk))
        q = qk / float(np.sum(qk))
        v = np.empty(p.shape[0], dtype=float)
        for i in range(p.shape[0]):
            v[i] = K.rel_entr(p[i], q[i])
        S = float(np.sum(v))
        if self.base > 0:
            S /= K.log(self.base)
        return S

def entropy(pk, qk = None, base = None, axis = 0):
    b = -1.0
    if base is not None:
        b = float(base)
        if b <= 0:
            raise ValueError("`base` must be a positive number or `None`.")
    if qk is None:
        return _reduce(pk, axis, _Entropy(b))
    else:
        p = _f64(pk)
        q = _f64(qk)
        shp = np.broadcast_shapes(p.shape, q.shape)
        return _reduce2(np.broadcast_to(p, shp), np.broadcast_to(q, shp), axis, _RelEntropy(b))

def _padded(X, m: int):
    n = X.shape[0]
    out = np.empty(n + 2 * m, dtype=float)
    for i in range(m):
        out[i] = X[0]
        out[n + m + i] = X[n - 1]
    for i in range(n):
        out[m + i] = X[i]
    return out

class _DiffEntropy:
    wl: int
    has_wl: bool
    base: float
    method: str
    def __call__(self, values) -> float:
        n = values.shape[0]
        m = self.wl if self.has_wl else int(K.floor(K.sqrt(float(n)) + 0.5))
        if not (2 <= 2 * m < n):
            raise ValueError(f"Window length ({m}) must be positive and less than half the sample size ({n}).")
        X = np.sort(values)
        method = self.method.lower()
        if method == 'auto':
            if n <= 10:
                method = 'van es'
            elif n <= 1000:
                method = 'ebrahimi'
            else:
                method = 'vasicek'
        res = 0.0
        if method == 'vasicek':
            Xp = _padded(X, m)
            logs = np.log(n / (2 * m) * (Xp[2 * m:] - Xp[:n]))
            res = float(np.mean(logs))
        elif method == 'van es':
            difference = X[m:] - X[:n - m]
            term1 = 1 / (n - m) * float(np.sum(np.log((n + 1) / m * difference)))
            k = np.arange(m, n + 1).astype(float)
            res = term1 + float(np.sum(1 / k)) + K.log(float(m)) - K.log(float(n + 1))
        elif method == 'ebrahimi':
            Xp = _padded(X, m)
            differences = Xp[2 * m:] - Xp[:n]
            ci = np.empty(n, dtype=float)
            for ii in range(n):
                i = float(ii + 1)
                c = 1 + (i - 1) / m if i <= m else 2.
                if i >= n - m + 1:
                    c = 1 + (n - i) / m
                ci[ii] = c
            logs = np.log(n * differences / (ci * m))
            res = float(np.mean(logs))
        elif method == 'correa':
            Xp = _padded(X, m)
            lg = np.empty(n, dtype=float)
            for ii in range(n):
                i = ii + 1
                s = 0.0
                for dj in range(-m, m + 1):
                    s += Xp[i + dj + m - 1]
                xbar = s / (2 * m + 1)
                num = 0.0
                den = 0.0
                for dj in range(-m, m + 1):
                    d = Xp[i + dj + m - 1] - xbar
                    num += d * dj
                    den += d * d
                lg[ii] = K.log(num / (n * den))
            res = -float(np.mean(lg))
        else:
            raise ValueError("`method` must be one of {'vasicek', 'van es', 'correa', 'ebrahimi', 'auto'}")
        if self.base > 0:
            res /= K.log(self.base)
        return res

def differential_entropy(values, *, window_length = None, base = None, axis = 0,
                         method: str = "auto"):
    b = -1.0
    if base is not None:
        b = float(base)
        if b <= 0:
            raise ValueError("`base` must be a positive number or `None`.")
    if window_length is None:
        return _reduce(values, axis, _DiffEntropy(0, False, b, method))
    else:
        return _reduce(values, axis, _DiffEntropy(int(window_length), True, b, method))

# ------------------------------------------------------------------ frequency

def _histogram(a, numbins: int, defaultlimits, weights, density: bool, cumulative: bool):
    x = _f64(a).ravel()
    n = x.shape[0]
    lo = 0.0
    hi = 1.0
    if defaultlimits is None:
        if n > 0:
            dmin = float(np.min(x))
            dmax = float(np.max(x))
            s = (dmax - dmin) / (2. * (numbins - 1.))
            lo = dmin - s
            hi = dmax + s
    else:
        lo = float(defaultlimits[0])
        hi = float(defaultlimits[1])
        if not (lo < hi):
            raise ValueError("If specified, `defaultreallimits` must be given as an iterable in the order (lower limit, upper limit).")
    edges = np.linspace(lo, hi, numbins + 1)
    hist = np.zeros(numbins, dtype=float)
    total_w = 0.0
    binned = 0
    if weights is None:
        y = np.sort(x)
        idx = np.searchsorted(y, edges, side='left')
        for i in range(numbins):
            hist[i] = float(idx[i + 1] - idx[i])
        c = 0
        for v in x:
            if v == edges[numbins]:
                c += 1
        hist[numbins - 1] += float(c)
        binned = int(np.sum(hist))
    else:
        w = _f64(weights).ravel()
        order = np.argsort(x)
        xs = x[order]
        ws = w[order]
        cw = np.empty(n + 1, dtype=float)
        cw[0] = 0.0
        acc = 0.0
        for i in range(n):
            acc += ws[i]
            cw[i + 1] = acc
        total_w = cw[n]
        idx = np.searchsorted(xs, edges, side='left')
        for i in range(numbins):
            hist[i] = cw[idx[i + 1]] - cw[idx[i]]
        ex = 0.0
        for i in range(n):
            if xs[i] == edges[numbins]:
                ex += ws[i]
        hist[numbins - 1] += ex
        for v in x:
            if edges[0] <= v and v <= edges[numbins]:
                binned += 1
    binsize = edges[1] - edges[0]
    extrapoints = n - binned
    if density:
        hist = hist / (float(n) if weights is None else total_w)
    if cumulative:
        hist = np.cumsum(hist)
    return hist, lo, binsize, extrapoints

def cumfreq(a, numbins: int = 10, defaultreallimits = None, weights = None):
    h, l, b, e = _histogram(a, numbins, defaultreallimits, weights, False, True)
    return CumfreqResult(h, l, b, e)

def relfreq(a, numbins: int = 10, defaultreallimits = None, weights = None):
    h, l, b, e = _histogram(a, numbins, defaultreallimits, weights, True, False)
    return RelfreqResult(h, l, b, e)

def find_repeats(arr):
    x = np.sort(_f64(arr).ravel())
    vals = []
    cnts = []
    n = x.shape[0]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[j + 1] == x[i]:
            j += 1
        if j > i:
            vals.append(x[i])
            cnts.append(j - i + 1)
        i = j + 1
    return RepeatedResults(np.array(vals, dtype=float), np.array(cnts, dtype=int))

# ------------------------------------------------------------------ distances

def _cdf_distance(p: int, u_values, v_values, u_weights, v_weights) -> float:
    u = _f64(u_values)
    v = _f64(v_values)
    if u.shape[0] == 0 or v.shape[0] == 0:
        raise ValueError("Distribution can't be empty.")
    us = np.argsort(u)
    vs = np.argsort(v)
    allv = np.sort(np.concatenate((u, v)), kind='mergesort')
    na = allv.shape[0]
    deltas = np.diff(allv)
    uu = u[us]
    vv = v[vs]
    ui = np.searchsorted(uu, allv[:na - 1], side='right')
    vi = np.searchsorted(vv, allv[:na - 1], side='right')
    ucdf = np.empty(na - 1, dtype=float)
    vcdf = np.empty(na - 1, dtype=float)
    if u_weights is None:
        for i in range(na - 1):
            ucdf[i] = ui[i] / u.shape[0]
    else:
        uw = _f64(u_weights)[us]
        cw = np.concatenate((np.zeros(1), np.cumsum(uw)))
        for i in range(na - 1):
            ucdf[i] = cw[ui[i]] / cw[cw.shape[0] - 1]
    if v_weights is None:
        for i in range(na - 1):
            vcdf[i] = vi[i] / v.shape[0]
    else:
        vw = _f64(v_weights)[vs]
        cw = np.concatenate((np.zeros(1), np.cumsum(vw)))
        for i in range(na - 1):
            vcdf[i] = cw[vi[i]] / cw[cw.shape[0] - 1]
    if p == 1:
        return float(np.dot(np.abs(ucdf - vcdf), deltas))
    d = ucdf - vcdf
    return K.sqrt(float(np.dot(d * d, deltas)))

def wasserstein_distance(u_values, v_values, u_weights = None, v_weights = None) -> float:
    return _cdf_distance(1, u_values, v_values, u_weights, v_weights)

def energy_distance(u_values, v_values, u_weights = None, v_weights = None) -> float:
    return 1.4142135623730951 * _cdf_distance(2, u_values, v_values, u_weights, v_weights)

# ------------------------------------------------------------------ circular

class _CircMean:
    high: float
    low: float
    def __call__(self, x) -> float:
        period = self.high - self.low
        sc = x * ((2.0 * _PI) / period)
        ss = float(np.sum(np.sin(sc)))
        cs = float(np.sum(np.cos(sc)))
        res = _atan2(ss, cs)
        r = res * (period / (2.0 * _PI)) - self.low
        r = r % period
        return r + self.low

def _atan2(y: float, x: float) -> float:
    return atan2(y, x)

class _CircR:
    high: float
    low: float
    def __call__(self, x) -> float:
        period = self.high - self.low
        sc = x * ((2.0 * _PI) / period)
        sm = float(np.mean(np.sin(sc)))
        cm = float(np.mean(np.cos(sc)))
        h = (sm ** 2. + cm ** 2.) ** 0.5
        return h if h < 1. else 1.

class _CircVar:
    r: _CircR
    def __call__(self, x) -> float:
        return 1. - self.r(x)

class _CircStd:
    r: _CircR
    normalize: bool
    def __call__(self, x) -> float:
        R = self.r(x)
        res = (-2 * K.log(R)) ** 0.5 + 0.0
        if not self.normalize:
            res *= (self.r.high - self.r.low) / (2. * _PI)
        return res

def circmean(samples, high = 2 * _PI, low = 0, axis = None, nan_policy: str = 'propagate'):
    return _reduce(samples, axis, _CircMean(float(high), float(low)))

def circvar(samples, high = 2 * _PI, low = 0, axis = None, nan_policy: str = 'propagate'):
    return _reduce(samples, axis, _CircVar(_CircR(float(high), float(low))))

def circstd(samples, high = 2 * _PI, low = 0, axis = None, nan_policy: str = 'propagate', *,
            normalize: bool = False):
    return _reduce(samples, axis, _CircStd(_CircR(float(high), float(low)), normalize))

# ------------------------------------------------------------------ multiple testing

def false_discovery_control(ps, *, axis = 0, method: str = 'bh'):
    p = _f64(ps)
    for v in p.flat:
        if not (0. <= v <= 1.):
            raise ValueError("All values in `ps` must lie between 0. and 1.")
    meth = method.lower()
    if meth != 'bh' and meth != 'by':
        raise ValueError(f"Unrecognized `method` '{method}'.Method must be one of {{'bh', 'by'}}.")
    if axis is None:
        return _fdr1(p.ravel(), meth)
    else:
        ax = int(axis)
        if ax < 0:
            ax += p.ndim
        m = np.moveaxis(p, ax, -1)
        out = np.empty(m.shape, dtype=float)
        for idx in np.ndindex(m.shape[:-1]):
            r = _fdr1(np.ascontiguousarray(m[idx]), meth)
            for k in range(r.shape[0]):
                out[idx + (k,)] = r[k]
        return np.moveaxis(out, -1, ax)

def _fdr1(p, method: str):
    m = p.shape[0]
    if m <= 1:
        return p.copy()
    order = np.argsort(p)
    ps = p[order]
    sinv = 0.0
    for k in range(m):
        sinv += 1.0 / float(k + 1)
    for k in range(m):
        ps[k] *= m / float(k + 1)
        if method == 'by':
            ps[k] *= sinv
    for k in range(m - 2, -1, -1):
        if ps[k + 1] < ps[k]:
            ps[k] = ps[k + 1]
    out = np.empty(m, dtype=float)
    for k in range(m):
        v = ps[k]
        out[order[k]] = 0. if v < 0. else (1. if v > 1. else v)
    return out

# ------------------------------------------------------------------ Box-Cox / Yeo-Johnson

def _log_var(logx) -> float:
    # logsumexp-based log variance (scipy _log_var)
    n = logx.shape[0]
    logmean = _lse(logx) - K.log(float(n))
    # log|x - mean| per element via logsumexp of (logx, logmean) with b=(1,-1)
    l2 = np.empty(n, dtype=float)
    for i in range(n):
        a = logx[i]
        b = logmean
        mm = a if a >= b else b
        # scipy _logsumexp: max term weight m (=1 or -1), the other added
        t = 0.0
        msum = 0.0
        if a > b:
            t = -K.exp(b - mm)
            msum = 1.0
        elif b > a:
            t = K.exp(a - mm)
            msum = -1.0
        else:
            t = 0.0
            msum = 0.0
        if msum == 0.0:
            l2[i] = -INF
            continue
        ss = t / msum if t != 0 else t
        if ss < -1:
            ss = -ss - 2
        l2[i] = 2.0 * (K.log1p(ss) + K.log(abs(msum)) + mm)
    return _lse(l2) - K.log(float(n))

def _lse(a) -> float:
    # scipy special.logsumexp (real, b=None) with max-masking
    n = a.shape[0]
    m2 = float(np.max(a))
    if not (m2 > -INF and m2 < INF):
        return K.log(float(np.sum(np.exp(a))))
    cnt = 0.0
    for v in a:
        if v == m2:
            cnt += 1.0
    acc = np.empty(n, dtype=float)
    for i in range(n):
        acc[i] = 0.0 if a[i] == m2 else K.exp(a[i] - m2)
    s2 = float(np.sum(acc)) / cnt
    return K.log1p(s2) + K.log(cnt) + m2

def _np_var(x) -> float:
    return float(np.var(x))

def _boxcox_llf1(data, lmb: float) -> float:
    N = data.shape[0]
    if N == 0:
        return NAN
    logdata = np.log(data)
    logvar = 0.0
    if lmb == 0:
        logvar = K.log(_np_var(logdata))
    else:
        logvar = _log_var(lmb * logdata) - 2 * K.log(abs(lmb))
    return (lmb - 1) * float(np.sum(logdata)) - N / 2 * logvar

class _BoxcoxLlf:
    lmb: float
    def __call__(self, x) -> float:
        return _boxcox_llf1(x, self.lmb)

def boxcox_llf(lmb: float, data, *, axis = 0, keepdims: bool = False, nan_policy: str = 'propagate'):
    return _reduce(data, axis, _BoxcoxLlf(lmb))

def _yeojohnson_transform(x, lmbda: float):
    eps = 2.220446049250313e-16
    out = np.zeros(x.shape, dtype=float)
    for idx in np.ndindex(x.shape):
        v = float(x[idx])
        if v >= 0:
            if abs(lmbda) < eps:
                out[idx] = K.log1p(v)
            else:
                out[idx] = K.expm1(lmbda * K.log1p(v)) / lmbda
        else:
            if abs(lmbda - 2) > eps:
                out[idx] = -K.expm1((2 - lmbda) * K.log1p(-v)) / (2 - lmbda)
            else:
                out[idx] = -K.log1p(-v)
    return out

def _yeojohnson_llf1(data, lmb: float) -> float:
    n = data.shape[0]
    if n == 0:
        return NAN
    eps = 2.220446049250313e-16
    allpos = True
    allneg = True
    for v in data:
        if v >= 0:
            allneg = False
        else:
            allpos = False
    logvar = 0.0
    if allpos:
        if abs(lmb) < eps:
            logvar = K.log(_np_var(np.log1p(data)))
        else:
            logvar = _log_var(lmb * np.log1p(data)) - 2 * K.log(abs(lmb))
    elif allneg:
        if abs(lmb - 2) < eps:
            logvar = K.log(_np_var(np.log1p(-data)))
        else:
            logvar = _log_var((2 - lmb) * np.log1p(-data)) - 2 * K.log(abs(2 - lmb))
    else:
        y = _yeojohnson_transform(data, lmb)
        sigma = _np_var(y)
        logvar = K.log(sigma) if sigma >= 2.2250738585072014e-308 else -INF
    return -n / 2 * logvar + (lmb - 1) * float(np.sum(np.sign(data) * np.log1p(np.abs(data))))

class _YJLlf:
    lmb: float
    def __call__(self, x) -> float:
        return _yeojohnson_llf1(x, self.lmb)

def yeojohnson_llf(lmb: float, data, *, axis = 0, nan_policy: str = 'propagate', keepdims: bool = False):
    return _reduce(data, axis, _YJLlf(lmb))

class _NegBoxcoxLlf:
    def __call__(self, lmb: float, data) -> float:
        return -_boxcox_llf1(data, lmb)

class _BoxcoxPearson:
    def __call__(self, lmb: float, xvals, samps) -> float:
        y = np.empty(samps.shape[0], dtype=float)
        for i in range(samps.shape[0]):
            y[i] = K.boxcox(samps[i], lmb)
        r = pearsonr(xvals, np.sort(y))
        return 1 - r.statistic

def _calc_uniform_order_statistic_medians(n: int):
    v = np.empty(n, dtype=float)
    v[n - 1] = 0.5 ** (1.0 / n)
    v[0] = 1 - v[n - 1]
    for i in range(2, n):
        v[i - 1] = (i - 0.3175) / (n + 0.365)
    return v

def boxcox_normmax(x, brack = None, method: str = 'pearsonr', optimizer = None, *,
                   nan_policy: str = 'propagate'):
    xx = _f64(x)
    for v in xx:
        if not (v >= 0 and v < INF):
            raise ValueError("The `x` argument of `boxcox_normmax` must contain only positive, finite, real numbers.")
    br = (-2.0, 2.0)
    if brack is not None:
        br = (float(brack[0]), float(brack[1]))
    if method == 'mle':
        return brent(_NegBoxcoxLlf(), args=(xx,), brack=[br[0], br[1]])
    elif method == 'pearsonr':
        osm = _calc_uniform_order_statistic_medians(xx.shape[0])
        xvals = np.empty(osm.shape[0], dtype=float)
        for i in range(osm.shape[0]):
            xvals[i] = K.ndtri(osm[i])
        return brent(_BoxcoxPearson(), args=(xvals, xx), brack=[br[0], br[1]])
    else:
        raise ValueError(f"Method {method} not recognized.")

def _boxcox_arr(x, lmbda: float):
    xx = _f64(x)
    out = np.empty(xx.shape, dtype=float)
    for idx in np.ndindex(xx.shape):
        out[idx] = K.boxcox(float(xx[idx]), lmbda)
    return out

def boxcox(x, lmbda = None, alpha = None, optimizer = None, *, nan_policy: str = 'propagate'):
    """``boxcox(x, lmbda)`` -> transformed array; ``boxcox(x)`` -> (y, lmax);
    ``boxcox(x, alpha=a)`` -> (y, lmax, (lo, hi))."""
    if lmbda is not None:
        return _boxcox_arr(x, float(lmbda))
    else:
        xx = _f64(x)
        if xx.ndim != 1:
            raise ValueError("Data must be 1-dimensional.")
        if np.all(xx == xx[0]):
            raise ValueError("Data must not be constant.")
        if np.any(xx <= 0):
            raise ValueError("Data must be positive.")
        lmax = boxcox_normmax(xx, method='mle')
        y = _boxcox_arr(xx, lmax)
        if alpha is None:
            return y, lmax
        else:
            return y, lmax, _boxcox_conf_interval(xx, lmax, float(alpha))

class _BoxcoxRoot:
    def __call__(self, lmbda: float, data, target: float) -> float:
        return _boxcox_llf1(data, lmbda) - target

def _boxcox_conf_interval(x, lmax: float, alpha: float):
    fac = 0.5 * K.chdtri(1.0, alpha)
    target = _boxcox_llf1(x, lmax) - fac
    rf = _BoxcoxRoot()
    newlm = lmax + 0.5
    N = 0
    while rf(newlm, x, target) > 0.0 and N < 500:
        newlm += 0.1
        N += 1
    if N == 500:
        raise RuntimeError("Could not find endpoint.")
    lmplus = brentq(rf, lmax, newlm, args=(x, target))
    newlm = lmax - 0.5
    N = 0
    while rf(newlm, x, target) > 0.0 and N < 500:
        newlm -= 0.1
        N += 1
    if N == 500:
        raise RuntimeError("Could not find endpoint.")
    lmminus = brentq(rf, newlm, lmax, args=(x, target))
    return (lmminus, lmplus)

class _NegYJLlf:
    def __call__(self, lmb: float, data) -> float:
        v = _yeojohnson_llf1(data, lmb)
        if v == INF or v == -INF:
            v = -INF
        return -v

def yeojohnson_normmax(x, brack = None, *, nan_policy: str = 'propagate') -> float:
    xx = _f64(x)
    for v in xx:
        if not (v > -INF and v < INF):
            raise ValueError('Yeo-Johnson input must be finite.')
    if np.all(xx == 0):
        return 1.0
    if brack is not None:
        return brent(_NegYJLlf(), brack=[float(brack[0]), float(brack[1])], args=(xx,))
    log1p_max_x = K.log1p(20 * float(np.max(np.abs(xx))))
    log_eps = K.log(2.220446049250313e-16)
    log_tiny_float = (K.log(2.2250738585072014e-308) - log_eps) / 2
    log_max_float = (K.log(1.7976931348623157e308) + log_eps) / 2
    lb = log_tiny_float / log1p_max_x
    ub = log_max_float / log1p_max_x
    if np.all(xx < 0):
        lb, ub = 2 - ub, 2 - lb
    elif np.any(xx < 0):
        lb, ub = max(2 - ub, lb), min(2 - lb, ub)
    return fminbound(_NegYJLlf(), lb, ub, args=(xx,), xtol=1.48e-08)

def yeojohnson(x, lmbda = None, *, nan_policy: str = 'propagate'):
    xx = _f64(x)
    if lmbda is not None:
        return _yeojohnson_transform(xx, float(lmbda))
    else:
        lmax = yeojohnson_normmax(xx)
        return _yeojohnson_transform(xx, lmax), lmax
