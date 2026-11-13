# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# CensoredData, ecdf (with Greenwood confidence intervals), logrank,
# spearmanrho, bws_test, ppcc_max, wasserstein_distance_nd.

import numpy as np
import internal.static as static
import scipy.special._kernels as K
from ._common import _f64
from ._resampling import permutation_test
from ._stats import rankdata, pearsonr, SignificanceResult
from ._morestats import ConfidenceInterval
from ._descriptive import wasserstein_distance
from ._continuous_distns3 import tukeylambda
from ._final_misc import PermutationMethod
from scipy.optimize import brent, linprog
from scipy.spatial.distance import cdist

NAN = K.NAN
INF = K.INF

# ------------------------------------------------------------ CensoredData

class CensoredData:
    _uncensored: np.ndarray[float, 1]
    _left: np.ndarray[float, 1]
    _right: np.ndarray[float, 1]
    _interval: np.ndarray[float, 2]

    def __init__(self, uncensored = None, *, left = None, right = None, interval = None):
        self._uncensored = CensoredData._arr(uncensored)
        self._left = CensoredData._arr(left)
        self._right = CensoredData._arr(right)
        if interval is None:
            self._interval = np.empty((0, 2), dtype=float)
        else:
            iv = _f64(interval)
            if iv.ndim != 2 or iv.shape[1] != 2:
                raise ValueError("`interval` must be a two-dimensional array with shape (m, 2), where m is the number of interval-censored values, but got shape " + str(iv.shape))
            self._interval = iv
        # intervals with equal endpoints are uncensored; (-inf, b] left; [a, inf) right
        nu = []
        nl = []
        nr = []
        ni = []
        for i in range(self._interval.shape[0]):
            lo = self._interval[i, 0]
            hi = self._interval[i, 1]
            if lo > hi:
                raise ValueError("In the `interval` array, each row must have the first value <= the second value.")
            if lo == hi:
                nu.append(lo)
            elif lo == -INF:
                nl.append(hi)
            elif hi == INF:
                nr.append(lo)
            else:
                ni.append(i)
        if len(nu) + len(nl) + len(nr) > 0:
            self._uncensored = np.concatenate((self._uncensored, np.array(nu, dtype=float)))
            self._left = np.concatenate((self._left, np.array(nl, dtype=float)))
            self._right = np.concatenate((self._right, np.array(nr, dtype=float)))
            keep = np.empty((len(ni), 2), dtype=float)
            for k in range(len(ni)):
                keep[k, 0] = self._interval[ni[k], 0]
                keep[k, 1] = self._interval[ni[k], 1]
            self._interval = keep
        if bool(np.any(np.isnan(self._uncensored))) or bool(np.any(np.isnan(self._left))) or \
           bool(np.any(np.isnan(self._right))) or bool(np.any(np.isnan(self._interval))):
            raise ValueError("Input data must not contain NaN.")

    @staticmethod
    def _arr(a) -> np.ndarray[float, 1]:
        if a is None:
            return np.empty(0, dtype=float)
        x = _f64(a)
        if x.ndim != 1:
            raise ValueError("Input must be a one-dimensional array.")
        return x

    @staticmethod
    def left_censored(x, censored):
        xa = _f64(x).ravel()
        ca = np.asarray(censored).astype(bool).ravel()
        if xa.shape[0] != ca.shape[0]:
            raise ValueError("`x` and `censored` must have the same length.")
        return CensoredData(uncensored=xa[~ca], left=xa[ca])

    @staticmethod
    def right_censored(x, censored):
        xa = _f64(x).ravel()
        ca = np.asarray(censored).astype(bool).ravel()
        if xa.shape[0] != ca.shape[0]:
            raise ValueError("`x` and `censored` must have the same length.")
        return CensoredData(uncensored=xa[~ca], right=xa[ca])

    @staticmethod
    def interval_censored(low, high):
        lo = _f64(low).ravel()
        hi = _f64(high).ravel()
        if lo.shape[0] != hi.shape[0]:
            raise ValueError("`low` and `high` must have the same length.")
        iv = np.empty((lo.shape[0], 2), dtype=float)
        for i in range(lo.shape[0]):
            iv[i, 0] = lo[i]
            iv[i, 1] = hi[i]
        return CensoredData(interval=iv)

    def __len__(self) -> int:
        return (self._uncensored.shape[0] + self._left.shape[0] + self._right.shape[0]
                + self._interval.shape[0])

    def num_censored(self) -> int:
        return self._left.shape[0] + self._right.shape[0] + self._interval.shape[0]

    def _uncensor(self) -> np.ndarray[float, 1]:
        parts = [self._uncensored, self._left, self._right]
        out = np.concatenate((self._uncensored, self._left, self._right))
        if self._interval.shape[0] > 0:
            mid = np.empty(self._interval.shape[0], dtype=float)
            for i in range(mid.shape[0]):
                lo = self._interval[i, 0]
                hi = self._interval[i, 1]
                mid[i] = 0.5 * (lo + hi)
            out = np.concatenate((out, mid))
        return out

    def _supported(self, a: float, b: float):
        us = self._uncensored
        keep = (us > a) & (us < b)
        return CensoredData(uncensored=us[keep], left=self._left[self._left > a],
                            right=self._right[self._right < b], interval=self._interval)

# -------------------------------------------------------------------- ecdf

class EmpiricalDistributionFunction:
    quantiles: np.ndarray[float, 1]
    probabilities: np.ndarray[float, 1]
    _n: np.ndarray[float, 1]
    _d: np.ndarray[float, 1]
    _sf: np.ndarray[float, 1]
    _kind: str
    _has_n: bool

    def __init__(self, q, p, n, d, kind: str, has_n: bool = True):
        self.quantiles = q
        self.probabilities = p
        self._n = n
        self._d = d
        self._kind = kind
        self._has_n = has_n
        self._sf = p if kind == 'sf' else 1 - p

    def evaluate(self, x):
        init = 0.0 if self._kind == 'cdf' else 1.0
        q = self.quantiles
        def one(t: float) -> float:
            if t != t:
                return NAN
            lo = 0
            hi = q.shape[0]
            while lo < hi:
                mid = (lo + hi) // 2
                if q[mid] <= t:
                    lo = mid + 1
                else:
                    hi = mid
            return init if lo == 0 else self.probabilities[lo - 1]
        if isinstance(x, np.ndarray) or isinstance(x, List):
            a = _f64(x)
            out = np.empty(a.shape, dtype=float)
            for idx in np.ndindex(a.shape):
                out[idx] = one(float(a[idx]))
            return out
        else:
            return one(float(x))

    def confidence_interval(self, confidence_level: float = 0.95, *, method: str = 'linear'):
        if not self._has_n:
            raise NotImplementedError("Confidence interval bounds do not implement a `confidence_interval` method.")
        m = method.lower()
        if m != 'linear' and m != 'log-log':
            raise ValueError("`method` must be one of {'linear', 'log-log'}.")
        if not (0 <= confidence_level and confidence_level <= 1):
            raise ValueError("`confidence_level` must be a scalar between 0 and 1.")
        z = K.ndtri(0.5 + confidence_level / 2)
        k = self.quantiles.shape[0]
        low = np.empty(k, dtype=float)
        high = np.empty(k, dtype=float)
        cs = 0.0
        for i in range(k):
            n = self._n[i]
            d = self._d[i]
            cs += d / (n * (n - d)) if n * (n - d) != 0 else (INF if d > 0 else NAN)
            sf = self._sf[i]
            if m == 'linear':
                se = K.sqrt(sf ** 2 * cs)
                low[i] = self.probabilities[i] - z * se
                high[i] = self.probabilities[i] + z * se
            else:
                lg = K.log(sf)
                se = K.sqrt(1 / lg ** 2 * cs)
                lnl = K.log(-lg) if -lg > 0 else -INF
                lo_sf = K.exp(-K.exp(lnl + z * se))
                hi_sf = K.exp(-K.exp(lnl - z * se))
                if self._kind == 'sf':
                    low[i] = lo_sf
                    high[i] = hi_sf
                else:
                    low[i] = 1 - hi_sf
                    high[i] = 1 - lo_sf
        for i in range(k):
            low[i] = min(max(low[i], 0.0), 1.0) if low[i] == low[i] else NAN
            high[i] = min(max(high[i], 0.0), 1.0) if high[i] == high[i] else NAN
        return EDFInterval(
            EmpiricalDistributionFunction(self.quantiles, low, self._n, self._d, self._kind, False),
            EmpiricalDistributionFunction(self.quantiles, high, self._n, self._d, self._kind, False))

class EDFInterval:
    low: EmpiricalDistributionFunction
    high: EmpiricalDistributionFunction
    def __init__(self, low: EmpiricalDistributionFunction, high: EmpiricalDistributionFunction):
        self.low = low
        self.high = high

class ECDFResult:
    cdf: EmpiricalDistributionFunction
    sf: EmpiricalDistributionFunction
    def __init__(self, q, cdf, sf, n, d):
        self.cdf = EmpiricalDistributionFunction(q, cdf, n, d, 'cdf')
        self.sf = EmpiricalDistributionFunction(q, sf, n, d, 'sf')

def _ecdf_uncensored(sample: np.ndarray[float, 1]):
    s = np.sort(sample)
    n = s.shape[0]
    xs = []
    counts = []
    i = 0
    while i < n:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        xs.append(s[i])
        counts.append(float(j - i))
        i = j
    k = len(xs)
    x = np.array(xs, dtype=float)
    cnt = np.array(counts, dtype=float)
    cdf = np.empty(k, dtype=float)
    sf = np.empty(k, dtype=float)
    at_risk = np.empty(k, dtype=float)
    ev = 0.0
    for t in range(k):
        at_risk[t] = n - ev
        ev += cnt[t]
        cdf[t] = ev / n
        sf[t] = 1 - cdf[t]
    return ECDFResult(x, cdf, sf, at_risk, cnt)

def _ecdf_right_censored(sample: CensoredData):
    tod = sample._uncensored
    tol = sample._right
    nt = tod.shape[0] + tol.shape[0]
    times = np.concatenate((tod, tol))
    died = np.concatenate((np.ones(tod.shape[0], dtype=float), np.zeros(tol.shape[0], dtype=float)))
    order = np.argsort(times)
    times = times[order]
    died = died[order]
    ts = []
    ns = []
    ds = []
    i = 0
    while i < nt:
        j = i
        dd = 0.0
        while j < nt and times[j] == times[i]:
            dd += died[j]
            j += 1
        ts.append(times[i])
        ns.append(float(nt - i))
        ds.append(dd)
        i = j
    k = len(ts)
    t = np.array(ts, dtype=float)
    n = np.array(ns, dtype=float)
    d = np.array(ds, dtype=float)
    sf = np.empty(k, dtype=float)
    cdf = np.empty(k, dtype=float)
    cp = 1.0
    for q in range(k):
        cp *= (n[q] - d[q]) / n[q]
        sf[q] = cp
        cdf[q] = 1 - cp
    return ECDFResult(t, cdf, sf, n, d)

def ecdf(sample):
    if isinstance(sample, CensoredData):
        cd = sample
    else:
        cd = CensoredData(uncensored=sample)
    if cd.num_censored() == 0:
        return _ecdf_uncensored(cd._uncensor())
    elif cd.num_censored() == cd._right.shape[0]:
        return _ecdf_right_censored(cd)
    raise NotImplementedError("Currently, only uncensored and right-censored data is supported.")

class LogRankResult:
    statistic: float
    pvalue: float
    def __init__(self, statistic: float, pvalue: float):
        self.statistic = statistic
        self.pvalue = pvalue

def logrank(x, y, alternative: str = 'two-sided'):
    xc = x if isinstance(x, CensoredData) else CensoredData(uncensored=x)
    yc = y if isinstance(y, CensoredData) else CensoredData(uncensored=y)
    xy = CensoredData(uncensored=np.concatenate((xc._uncensored, yc._uncensored)),
                      right=np.concatenate((xc._right, yc._right)))
    res = ecdf(xy)
    rx = ecdf(xc)
    qx = rx.sf.quantiles
    sum_var = 0.0
    sum_exp = 0.0
    for i in range(res.sf.quantiles.shape[0]):
        d = res.sf._d[i]
        if d == 0:
            continue
        t = res.sf.quantiles[i]
        nxy = res.sf._n[i]
        j = 0
        while j < qx.shape[0] and qx[j] < t:
            j += 1
        nx = rx.sf._n[j] if j < qx.shape[0] else 0.0
        ny = nxy - nx
        if nxy > 1:
            sum_var += nx * ny * d * (nxy - d) / (nxy ** 2 * (nxy - 1))
        sum_exp += nx * (d / nxy)
    statistic = (xc._uncensored.shape[0] - sum_exp) / K.sqrt(sum_var)
    alt = alternative.lower()
    pv = 0.0
    if alt == 'less':
        pv = K.ndtr(statistic)
    elif alt == 'greater':
        pv = K.ndtr(-statistic)
    elif alt == 'two-sided':
        pv = 2 * K.ndtr(-abs(statistic))
    else:
        raise ValueError("`alternative` must be one of {'two-sided', 'less', 'greater'}")
    return LogRankResult(statistic, pv)

# -------------------------------------------------------------- spearmanrho

def spearmanrho(x, y, *, alternative: str = 'two-sided', method = None, axis = 0):
    rx = rankdata(x, axis=axis)
    ry = rankdata(y, axis=axis)
    r = pearsonr(rx, ry, alternative=alternative, axis=axis)
    if isinstance(method, PermutationMethod):
        a = _f64(x).ravel() if axis is None else _f64(x)
        b = _f64(y).ravel() if axis is None else _f64(y)
        res = permutation_test((a, b), _spearman_stat, permutation_type='pairings',
                               vectorized=False, n_resamples=method.n_resamples,
                               alternative=alternative, axis=0 if axis is None else axis,
                               rng=method.rng)
        return SignificanceResult(r.statistic, res.pvalue)
    return SignificanceResult(r.statistic, r.pvalue)

def _spearman_stat(u, v):
    return pearsonr(rankdata(u), rankdata(v)).statistic

# ----------------------------------------------------------------- bws_test

def _bws_statistic(x: np.ndarray[float, 1], y: np.ndarray[float, 1], two_sided: bool) -> float:
    Ri = np.sort(x)
    Hj = np.sort(y)
    n = Ri.shape[0]
    m = Hj.shape[0]
    bx = 0.0
    by = 0.0
    for a in range(n):
        i = a + 1.0
        num = Ri[a] - (m + n) / n * i
        num = num * num if two_sided else num * abs(num)
        den = i / (n + 1) * (1 - i / (n + 1)) * m * (m + n) / n
        bx += num / den
    for a in range(m):
        j = a + 1.0
        num = Hj[a] - (m + n) / m * j
        num = num * num if two_sided else num * abs(num)
        den = j / (m + 1) * (1 - j / (m + 1)) * n * (m + n) / m
        by += num / den
    bx /= n
    by /= m
    return (bx + by) / 2 if two_sided else (bx - by) / 2

class _Bws:
    two_sided: bool
    def __init__(self, two_sided: bool):
        self.two_sided = two_sided
    def __call__(self, x, y):
        return _bws_statistic(x, y, self.two_sided)

def bws_test(x, y, *, alternative: str = 'two-sided', method = None):
    alt = alternative.lower()
    if alt != 'two-sided' and alt != 'less' and alt != 'greater':
        raise ValueError("`alternative` must be one of {'two-sided', 'less', 'greater'}.")
    nres = 9999
    rng = None
    if isinstance(method, PermutationMethod):
        nres = method.n_resamples
        rng = method.rng
    palt = 'less' if alt == 'less' else 'greater'
    return permutation_test((_f64(x).ravel(), _f64(y).ravel()), _Bws(alt == 'two-sided'),
                            vectorized=False, n_resamples=nres, alternative=palt, rng=rng)

# ----------------------------------------------------------------- ppcc_max

def _uniform_order_stat_medians(n: int) -> np.ndarray[float, 1]:
    v = np.empty(n, dtype=float)
    v[n - 1] = 0.5 ** (1.0 / n)
    v[0] = 1 - v[n - 1]
    for i in range(2, n):
        v[i - 1] = (i - 0.3175) / (n + 0.365)
    return v

class _Ppcc:
    mi: np.ndarray[float, 1]
    osr: np.ndarray[float, 1]
    def __init__(self, mi: np.ndarray[float, 1], osr: np.ndarray[float, 1]):
        self.mi = mi
        self.osr = osr
    def __call__(self, shape: float) -> float:
        xv = np.asarray(tukeylambda.ppf(self.mi, shape))
        r = pearsonr(xv, self.osr).statistic
        return 1 - r

def ppcc_max(x, brack = (0.0, 1.0), dist: str = 'tukeylambda'):
    if dist != 'tukeylambda':
        raise ValueError("only dist='tukeylambda' is supported")
    xs = _f64(x).ravel()
    n = xs.shape[0]
    return brent(_Ppcc(_uniform_order_stat_medians(n), np.sort(xs)), brack=brack)

# ------------------------------------------------- wasserstein_distance_nd

def wasserstein_distance_nd(u_values, v_values, u_weights = None, v_weights = None):
    u = np.asarray(u_values).astype(float)
    v = np.asarray(v_values).astype(float)
    if u.ndim > 2 or v.ndim > 2:
        raise ValueError("Invalid input values. The inputs must have either one or two dimensions.")
    if u.ndim != v.ndim:
        raise ValueError("Invalid input values. Dimensions of inputs must be equal.")
    if u.ndim == 1:
        return wasserstein_distance(u, v, u_weights, v_weights)
    if u.shape[1] != v.shape[1]:
        raise ValueError("Invalid input values. If two-dimensional, `u_values` and `v_values` must have the same number of columns.")
    if bool(np.any(np.isinf(u))) != bool(np.any(np.isinf(v))):
        return INF
    if bool(np.any(np.isinf(u))) and bool(np.any(np.isinf(v))):
        return NAN
    m = u.shape[0]
    n = v.shape[0]
    pu = np.full(m, 1.0 / m)
    pv = np.full(n, 1.0 / n)
    if u_weights is not None:
        w = _f64(u_weights)
        pu = w / float(np.sum(w))
    if v_weights is not None:
        w = _f64(v_weights)
        pv = w / float(np.sum(w))
    D = cdist(u, v)
    cost = D.ravel()
    Aeq = np.zeros((m + n, m * n))
    for i in range(m):
        for j in range(n):
            Aeq[i, i * n + j] = 1.0
            Aeq[m + j, i * n + j] = 1.0
    beq = np.concatenate((pu, pv))
    res = linprog(cost, A_eq=Aeq, b_eq=beq, bounds=(0, None))
    return res.fun

# ---------------------------------------------------- censored MLE fit

class _CensNnlf[D]:
    dist: D
    data: CensoredData
    nshape: int
    def __init__(self, dist: D, data: CensoredData, nshape: int):
        self.dist = dist
        self.data = data
        self.nshape = nshape
    def __call__(self, th: np.ndarray[float, 1]) -> float:
        ns = self.nshape
        loc = th[ns]
        scale = th[ns + 1]
        if not (scale > 0):
            return INF
        sh = [th[i] for i in range(ns)]
        d = self.data
        tot = 0.0
        if d._uncensored.shape[0] > 0:
            tot -= float(np.sum(self.dist.logpdf(d._uncensored, *sh, loc=loc, scale=scale)))
        if d._left.shape[0] > 0:
            tot -= float(np.sum(self.dist.logcdf(d._left, *sh, loc=loc, scale=scale)))
        if d._right.shape[0] > 0:
            tot -= float(np.sum(self.dist.logsf(d._right, *sh, loc=loc, scale=scale)))
        for i in range(d._interval.shape[0]):
            lo = float(d._interval[i, 0])
            hi = float(d._interval[i, 1])
            p = float(self.dist.cdf(hi, *sh, loc=loc, scale=scale)) - float(self.dist.cdf(lo, *sh, loc=loc, scale=scale))
            tot -= K.log(p) if p > 0 else -INF
        if tot != tot:
            return INF
        return tot

def fit_censored(dist, data: CensoredData, x0):
    """Maximum likelihood fit of ``dist`` (shape params..., loc, scale) to
    censored data by Nelder-Mead from the starting vector ``x0``; the
    counterpart of ``dist.fit(CensoredData)``."""
    from scipy.optimize import minimize
    start = np.asarray(x0).astype(float).ravel()
    ns = start.shape[0] - 2
    res = minimize(_CensNnlf(dist, data, ns), start, method='Nelder-Mead')
    return res.x
