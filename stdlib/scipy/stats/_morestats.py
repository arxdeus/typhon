# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# scipy.stats._morestats / _hypotests / _ksstats: Shapiro-Wilk, Anderson-Darling,
# Ansari-Bradley, Mood, Levene, Bartlett, Fligner-Killeen, median test,
# binomial and Fisher exact tests, Kolmogorov-Smirnov, Cramer-von Mises,
# Poisson means and Epps-Singleton tests.

import numpy as np
import internal.static as static
import scipy.special._kernels as K
import scipy.special._kolmogorov as KS
import scipy.special._bessel as BS
import scipy.special._misc as MS
import scipy.special._cdf as CD
from C import acos(float) -> float
from C import rint(float) -> float
from C import ldexp(float, i32) -> float
from ._common import _f64, _slices, _var, _rank_avg, _pvalue_norm, _pvalue_t
from ._stats import _Result2, SignificanceResult, _map_slices, _map_slices2, \
                    _ksample, chi2_contingency
from scipy.optimize import fsolve, brentq
from ._continuous_distns import gumbel_r, gumbel_l, norm, expon, uniform, logistic, laplace, cauchy, t, chi2, gamma, beta, lognorm
from ._descriptive import _median1, _TrimMean, _Iqr

NAN = K.NAN
INF = K.INF
_PI = 3.141592653589793

class ShapiroResult[S, P](_Result2[S, P]):
    def __init__(self, statistic: S, pvalue: P):
        super().__init__(statistic, pvalue)

class AnsariResult[S, P](_Result2[S, P]):
    def __init__(self, statistic: S, pvalue: P):
        super().__init__(statistic, pvalue)

class LeveneResult[S, P](_Result2[S, P]):
    def __init__(self, statistic: S, pvalue: P):
        super().__init__(statistic, pvalue)

class BartlettResult[S, P](_Result2[S, P]):
    def __init__(self, statistic: S, pvalue: P):
        super().__init__(statistic, pvalue)

class FlignerResult[S, P](_Result2[S, P]):
    def __init__(self, statistic: S, pvalue: P):
        super().__init__(statistic, pvalue)

class CramerVonMisesResult[S, P](_Result2[S, P]):
    def __init__(self, statistic: S, pvalue: P):
        super().__init__(statistic, pvalue)

class Epps_Singleton_2sampResult[S, P](_Result2[S, P]):
    def __init__(self, statistic: S, pvalue: P):
        super().__init__(statistic, pvalue)

class KstestResult[S, P, L, G](_Result2[S, P]):
    statistic_location: L
    statistic_sign: G
    def __init__(self, statistic: S, pvalue: P, statistic_location: L, statistic_sign: G):
        super().__init__(statistic, pvalue)
        self.statistic_location = statistic_location
        self.statistic_sign = statistic_sign

class AndersonResult:
    statistic: float
    critical_values: np.ndarray[float, 1]
    significance_level: np.ndarray[float, 1]
    def __init__(self, statistic: float, critical_values: np.ndarray[float, 1],
                 significance_level: np.ndarray[float, 1]):
        self.statistic = statistic
        self.critical_values = critical_values
        self.significance_level = significance_level

class Anderson_ksampResult:
    statistic: float
    critical_values: np.ndarray[float, 1]
    pvalue: float
    def __init__(self, statistic: float, critical_values: np.ndarray[float, 1], pvalue: float):
        self.statistic = statistic
        self.critical_values = critical_values
        self.pvalue = pvalue
    @property
    def significance_level(self):
        return self.pvalue

class MedianTestResult:
    statistic: float
    pvalue: float
    median: float
    table: np.ndarray[int, 2]
    def __init__(self, statistic: float, pvalue: float, median: float, table: np.ndarray[int, 2]):
        self.statistic = statistic
        self.pvalue = pvalue
        self.median = median
        self.table = table

class ConfidenceInterval:
    low: float
    high: float
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

# ------------------------------------------------------------------ Shapiro-Wilk

def _swilk(y) -> Tuple[float, float]:
    # scipy 1.18 _morestats._swilk (Royston approximation in NumPy form)
    n = y.shape[0]
    if n == 3:
        c = K.sqrt(2.0) / 2
        a3 = np.array([-c, 0., c])
        W = _swilk_w(y, a3)
        W = min(max(W, 0.75), 1.)
        pv = 1. - 6 / _PI * acos(K.sqrt(W))
        return (W, min(max(pv, 0.), 1.))
    m = np.empty(n, dtype=float)
    for k in range(n):
        m[k] = K.ndtri((float(k + 1) - 3 / 8) / (n + 1 / 4))
    u = float(n) ** (-0.5)
    mTm = float(np.dot(m, m))
    cc = mTm ** (-0.5) * m
    mnm1 = m[n - 2]
    mn = m[n - 1]
    cnm1 = cc[n - 2]
    cn = cc[n - 1]
    an = (cn + 0.221157 * u - 0.147981 * u ** 2
          - 2.071190 * u ** 3 + 4.434685 * u ** 4 - 2.706056 * u ** 5)
    anm1 = (cnm1 + 0.042981 * u - 0.293762 * u ** 2
            - 1.752461 * u ** 3 + 5.682633 * u ** 4 - 3.582633 * u ** 5)
    phi = 0.0
    if n <= 5:
        phi = (mTm - 2 * mn ** 2) / (1 - 2 * an ** 2)
    else:
        phi = (mTm - 2 * mn ** 2 - 2 * mnm1 ** 2) / (1 - 2 * an ** 2 - 2 * anm1 ** 2)
    a = phi ** (-0.5) * m
    if n > 5:
        a[n - 2] = anm1
        a[1] = -anm1
    a[n - 1] = an
    a[0] = -an
    W = _swilk_w(y, a)
    gW = 0.0
    mu = 0.0
    sigma = 0.0
    if n <= 11:
        uu = float(n)
        gamma = -2.273 + 0.459 * uu
        mu = 0.5440 - 0.39978 * uu + 0.025054 * uu ** 2 - 0.0006714 * uu ** 3
        sigma = K.exp(1.3822 - 0.77857 * uu + 0.062767 * uu ** 2 - 0.0020322 * uu ** 3)
        gW = -K.log(gamma - K.log(1 - W))
    else:
        uu = K.log(float(n))
        mu = -1.5861 - 0.31082 * uu - 0.083751 * uu ** 2 + 0.0038915 * uu ** 3
        sigma = K.exp(-0.4803 - 0.082676 * uu + 0.0030302 * uu ** 2)
        gW = K.log(1 - W)
    z = (gW - mu) / sigma
    return (W, K.ndtr(-z))

def _swilk_w(y, a) -> float:
    num = float(np.dot(a, y)) ** 2
    y_ = y - float(np.mean(y))
    den = float(np.dot(y_, y_))
    return num / den

def shapiro(x, *, axis = None):
    """Shapiro-Wilk test (scipy 1.18 uses the Royston swilk routine on the
    sorted, median-shifted sample)."""
    v = _f64(x).ravel()
    N = v.shape[0]
    if N < 3:
        raise ValueError("Data must be at least length 3.")
    y = np.sort(v)
    y = y - v[N // 2]
    r = _swilk(y)
    return ShapiroResult(r[0], r[1])

# ------------------------------------------------------------------ Anderson-Darling

def _around3(v: float) -> float:
    return rint(v * 1000.0) / 1000.0

class _LogisticRoot:
    def __call__(self, ab, xj, N: float):
        a = ab[0]
        b = ab[1]
        tmp = (xj - a) / b
        tmp2 = np.exp(tmp)
        out = np.empty(2, dtype=float)
        out[0] = float(np.sum(1.0 / (1 + tmp2))) - 0.5 * N
        out[1] = float(np.sum(tmp * (1.0 - tmp2) / (1 + tmp2))) + N
        return out

def _anderson_impl(x, dist0: str):
    dist = dist0.lower()
    if dist == 'extreme1' or dist == 'gumbel':
        dist = 'gumbel_l'
    xx = _f64(x).ravel()
    y = np.sort(xx)
    xbar = float(np.mean(xx))
    N = y.shape[0]
    Nf = float(N)
    logcdf = np.empty(N, dtype=float)
    logsf = np.empty(N, dtype=float)
    sig = np.empty(0, dtype=float)
    critical = np.empty(0, dtype=float)
    if dist == 'norm':
        s = float(np.std(xx, ddof=1))
        for i in range(N):
            w = (y[i] - xbar) / s
            logcdf[i] = K.log_ndtr(w)
            logsf[i] = K.log_ndtr(-w)
        sig = np.array([15., 10., 5., 2.5, 1.])
        av = [0.561, 0.631, 0.752, 0.873, 1.035]
        critical = np.array([_around3(v / (1.0 + 0.75 / Nf + 2.25 / Nf / Nf)) for v in av])
    elif dist == 'expon':
        for i in range(N):
            w = y[i] / xbar
            logcdf[i] = K.log1p(-K.exp(-w)) if w > 0.6931471805599453 else K.log(-K.expm1(-w))
            logsf[i] = -w
        sig = np.array([15., 10., 5., 2.5, 1.])
        av = [0.916, 1.062, 1.321, 1.591, 1.959]
        critical = np.array([_around3(v / (1.0 + 0.6 / Nf)) for v in av])
    elif dist == 'logistic':
        sol0 = np.array([xbar, float(np.std(xx, ddof=1))])
        sol = fsolve(_LogisticRoot(), sol0, args=(xx, Nf), xtol=1e-5)
        for i in range(N):
            w = (y[i] - sol[0]) / sol[1]
            logcdf[i] = K.log_expit(w)
            logsf[i] = K.log_expit(-w)
        sig = np.array([25., 10., 5., 2.5, 1., 0.5])
        av = [0.426, 0.563, 0.66, 0.769, 0.906, 1.01]
        critical = np.array([_around3(v / (1.0 + 0.25 / Nf)) for v in av])
    elif dist == 'gumbel_r' or dist == 'gumbel_l':
        loc = 0.0
        sc = 1.0
        if dist == 'gumbel_r':
            loc, sc = _gumbel_r_fit(xx)
        else:
            loc, sc = _gumbel_r_fit(-xx)
            loc = -loc
        for i in range(N):
            w = (y[i] - loc) / sc
            if dist == 'gumbel_r':
                logcdf[i] = -K.exp(-w)
                logsf[i] = K.log(-K.expm1(-K.exp(-w)))
            else:
                logsf[i] = -K.exp(w)
                logcdf[i] = K.log(-K.expm1(-K.exp(w)))
        sig = np.array([25., 10., 5., 2.5, 1.])
        av = [0.474, 0.637, 0.757, 0.877, 1.038]
        critical = np.array([_around3(v / (1.0 + 0.2 / K.sqrt(Nf))) for v in av])
    else:
        raise ValueError("Invalid distribution; dist must be in {'norm', 'expon', 'gumbel_l', 'gumbel_r', 'logistic'}.")
    terms = np.empty(N, dtype=float)
    for i in range(N):
        terms[i] = (2 * (i + 1) - 1.0) / Nf * (logcdf[i] + logsf[N - 1 - i])
    A2 = -Nf - float(np.sum(terms))
    return A2, critical, sig

class _GumbelScaleEq:
    def __call__(self, scale: float, data) -> float:
        sdata = -data / scale
        mx = float(np.max(sdata))
        w = np.exp(sdata - mx)
        wavg = float(np.sum(data * w)) / float(np.sum(w))
        return float(np.mean(data)) - wavg - scale

def _gumbel_r_fit(data):
    # scipy gumbel_r_gen.fit (loc and scale free)
    f = _GumbelScaleEq()
    lb = 0.5
    rb = 2.0
    def sgn(v: float) -> float:
        return 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)
    while sgn(f(lb, data)) == sgn(f(rb, data)) and (lb > 0 or rb < INF):
        lb /= 2
        rb *= 2
    scale = brentq(f, lb, rb, args=(data,), xtol=1e-14, rtol=1e-14)
    sd = -data / scale
    mx = float(np.max(sd))
    cnt = 0.0
    for v in sd:
        if v == mx:
            cnt += 1.0
    acc = np.empty(sd.shape[0], dtype=float)
    for i in range(sd.shape[0]):
        acc[i] = 0.0 if sd[i] == mx else K.exp(sd[i] - mx)
    lse = K.log1p(float(np.sum(acc)) / cnt) + K.log(cnt) + mx
    loc = -scale * (lse - K.log(float(data.shape[0])))
    return loc, scale

def anderson(x, dist: str = 'norm', *, method = None):
    """``method=None``: AndersonResult(statistic, critical_values,
    significance_level); ``method='interpolate'``: SignificanceResult."""
    A2, critical, sig = _anderson_impl(x, dist)
    if method is None:
        return AndersonResult(A2, critical, sig)
    else:
        if method != 'interpolate':
            raise ValueError("`method` must be either 'interpolate' or an instance of `MonteCarloMethod`.")
        p = float(np.interp(A2, critical, sig / 100))
        return SignificanceResult(A2, p)

def _searchsorted(s, v: float, right: bool) -> int:
    lo = 0
    hi = s.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if (s[mid] <= v) if right else (s[mid] < v):
            lo = mid + 1
        else:
            hi = mid
    return lo

def anderson_ksamp(samples, midrank: bool = True, *, variant: str = '', method = None):
    k = len(samples)
    if k < 2:
        raise ValueError("anderson_ksamp needs at least two samples")
    ss = [_f64(s).ravel() for s in samples]
    Z = np.sort(np.concatenate(ss))
    N = Z.shape[0]
    zl = []
    for i in range(N):
        if i == 0 or Z[i] != Z[i - 1]:
            zl.append(Z[i])
    Zstar = np.array(zl, dtype=float)
    if Zstar.shape[0] < 2:
        raise ValueError("anderson_ksamp needs more than one distinct observation")
    n = np.array([float(s.shape[0]) for s in ss])
    for v in n:
        if v == 0:
            raise ValueError("anderson_ksamp encountered sample without observations")
    var = variant
    if var == '':
        var = 'midrank' if midrank else 'right'
    Nf = float(N)
    A2kN = 0.
    nz = Zstar.shape[0]
    if var == 'midrank':
        for i in range(k):
            s = np.sort(ss[i])
            acc = np.empty(nz, dtype=float)
            for j in range(nz):
                zl_ = float(_searchsorted(Z, Zstar[j], False))
                lj = 1. if N == nz else float(_searchsorted(Z, Zstar[j], True)) - zl_
                Bj = zl_ + lj / 2.
                sr = _searchsorted(s, Zstar[j], True)
                fij = sr - _searchsorted(s, Zstar[j], False)
                Mij = float(sr) - fij / 2.
                acc[j] = lj / Nf * (Nf * Mij - Bj * n[i]) ** 2 / (Bj * (Nf - Bj) - Nf * lj / 4.)
            A2kN += float(np.sum(acc)) / n[i]
        A2kN *= (Nf - 1.) / Nf
    elif var == 'right':
        lj = np.empty(nz - 1, dtype=float)
        for j in range(nz - 1):
            lj[j] = float(_searchsorted(Z, Zstar[j], True) - _searchsorted(Z, Zstar[j], False))
        Bj = np.cumsum(lj)
        for i in range(k):
            s = np.sort(ss[i])
            acc = np.empty(nz - 1, dtype=float)
            for j in range(nz - 1):
                Mij = float(_searchsorted(s, Zstar[j], True))
                acc[j] = lj[j] / Nf * (Nf * Mij - Bj[j] * n[i]) ** 2 / (Bj[j] * (Nf - Bj[j]))
            A2kN += float(np.sum(acc)) / n[i]
    else:
        raise ValueError("`variant` must be one of 'midrank', 'right', or 'continuous'.")
    H = float(np.sum(1. / n))
    hs = np.empty(N - 2, dtype=float)
    for q in range(N - 2):
        hs[q] = 1. / float(N - 1 - q)
    hs_cs = np.cumsum(hs)
    h = hs_cs[N - 3] + 1
    g = 0.0
    gg = np.empty(N - 2, dtype=float)
    for q in range(N - 2):
        gg[q] = hs_cs[q] / float(q + 2)
    g = float(np.sum(gg))
    kf = float(k)
    a = (4 * g - 6) * (kf - 1) + (10 - 6 * g) * H
    b = (2 * g - 4) * kf ** 2 + 8 * h * kf + (2 * g - 14 * h - 4) * H - 8 * h + 4 * g - 6
    c = (6 * h + 2 * g - 2) * kf ** 2 + (4 * h - 4 * g + 6) * kf + (2 * h - 6) * H + 4 * h
    d = (2 * h + 6) * kf ** 2 - 4 * h * kf
    sigmasq = (a * Nf ** 3 + b * Nf ** 2 + c * Nf + d) / ((Nf - 1.) * (Nf - 2.) * (Nf - 3.))
    m = kf - 1
    A2 = (A2kN - m) / K.sqrt(sigmasq)
    b0 = np.array([0.675, 1.281, 1.645, 1.96, 2.326, 2.573, 3.085])
    b1 = np.array([-0.245, 0.25, 0.678, 1.149, 1.822, 2.364, 3.615])
    b2 = np.array([-0.105, -0.305, -0.362, -0.391, -0.396, -0.345, -0.154])
    critical = b0 + b1 / K.sqrt(m) + b2 / m
    sig = np.array([0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001])
    p = 0.0
    if A2 < float(np.min(critical)):
        p = 0.25
    elif A2 > float(np.max(critical)):
        p = 0.001
    else:
        pf = np.polyfit(critical, np.log(sig), 2)
        p = K.exp(float(np.polyval(pf, A2)))
    return Anderson_ksampResult(A2, critical, p)

# ------------------------------------------------------------------ Ansari-Bradley

def _start1(a, n: int):
    lout = 1 + (n // 2)
    for i in range(lout):
        a[i] = float32(2.)
    if n % 2 == 0:
        a[lout - 1] = float32(1.)

def _start2(a, n: int):
    odd = n % 2
    A = 1.
    B = 3.
    C = 2. if odd else 0.
    ndo = (n + 2 + odd) // 2 - odd
    for ind in range(ndo):
        a[ind] = float32(A)
        A += B
        B = 4 - B
    A = 1.
    B = 3.
    for ind in range(n - odd, ndo - 1, -1):
        a[ind] = float32(A + C)
        A += B
        B = 4 - B
    if odd == 1:
        a[(ndo * 2) - 1] = float32(2.)

def _frqadd(a, b, lenb: int, offset: int) -> int:
    for ind in range(lenb):
        a[offset + ind] = a[offset + ind] + float32(2.) * b[ind]
    return lenb + offset

def _imply(a, curlen: int, reslen: int, b, offset: int) -> int:
    i2 = -offset
    j2 = reslen - offset
    j2min = (j2 + 1) // 2 - 1
    nextlenb = j2
    j1 = reslen - 1
    j2 -= 1
    for i1 in range((reslen + 1) // 2):
        summ = float32(0.)
        if i2 < 0:
            summ = a[i1]
        else:
            summ = a[i1] + b[i2]
            a[i1] = summ
        i2 += 1
        if j2 >= j2min:
            diff = summ if j1 > curlen - 1 else summ - a[j1]
            b[i1] = diff
            b[j2] = diff
            j2 -= 1
        a[j1] = summ
        j1 -= 1
    return nextlenb

def _gscale(test: int, other: int):
    m = min(test, other)
    n = max(test, other)
    astart = ((test + 1) // 2) * (1 + (test // 2))
    LL = (test * other) // 2 + 1
    symm = (m + n) % 2 == 0
    odd = n % 2
    # scipy stores the frequency tables in float32
    a1 = np.zeros(LL, dtype=np.float32)
    a2 = np.zeros(LL, dtype=np.float32)
    a3 = np.zeros(LL, dtype=np.float32)
    if m == 0:
        a1[0] = float32(1.)
        return astart, a1
    if m == 1:
        _start1(a1, n)
        if not (symm or (other > test)):
            a1[0] = float32(1.)
            a1[LL - 1] = float32(2.)
        return astart, a1
    if m == 2:
        _start2(a1, n)
        if not (symm or (other > test)):
            for ind in range(LL // 2):
                t = a1[ind]
                a1[ind] = a1[LL - 1 - ind]
                a1[LL - 1 - ind] = t
        return astart, a1
    len1 = 0
    len2 = 0
    len3 = 0
    n2b1 = 0
    n2b2 = 0
    part_no = 0
    if odd:
        _start1(a1, n)
        _start2(a2, n - 1)
        len1 = 1 + (n // 2)
        len2 = n
        n2b1 = 1
        n2b2 = 2
        part_no = 0
    else:
        _start2(a1, n)
        _start1(a2, n - 1)
        _start2(a3, n - 2)
        len1 = n + 1
        len2 = n // 2
        len3 = n - 1
        n2b1 = 2
        n2b2 = 1
        part_no = 1
    loop_m = 3
    while loop_m <= m:
        if part_no == 0:
            l1out = _frqadd(a1, a2, len2, n2b1)
            len1 += n
            len3 = _imply(a1, l1out, len1, a3, loop_m)
            n2b1 += 1
            loop_m += 1
            part_no = 1
        else:
            l2out = _frqadd(a2, a3, len3, n2b2)
            len2 += n - 1
            _imply(a2, l2out, len2, a3, loop_m)
            n2b2 += 1
            loop_m += 1
            part_no = 0
    if not symm:
        ks = (m + 3) // 2 - 1
        for ind in range(len2):
            a1[ks + ind] += a2[ind]
    if other > test:
        for ind in range(LL // 2):
            t = a1[ind]
            a1[ind] = a1[LL - 1 - ind]
            a1[LL - 1 - ind] = t
    return astart, a1

class _Ansari:
    alternative: str
    method: str
    def __call__(self, x, y):
        n = x.shape[0]
        m = y.shape[0]
        if m < 1:
            raise ValueError("Not enough other observations.")
        if n < 1:
            raise ValueError("Not enough test observations.")
        N = m + n
        rank, ties = _rank_avg(np.concatenate((x, y)))
        symrank = np.minimum(rank, N - rank + 1)
        AB = float(np.sum(symrank[:n]))
        repeats = False
        for t in ties:
            if t > 1:
                repeats = True
        meth = self.method
        if meth == 'auto':
            meth = 'exact' if (m < 55 and n < 55 and not repeats) else 'asymptotic'
        alt = self.alternative
        if meth == 'exact':
            astart, freqs = _gscale(n, m)
            # scipy: freqs = a1.astype(float64); sums in float64
            f64 = freqs.astype(float)
            total = float(np.sum(f64))
            ci = int(K.ceil(AB - astart))
            fi = int(K.floor(AB - astart))
            cdf = float(np.sum(f64[:ci + 1])) / total
            sf = float(np.sum(f64[fi:])) / total
            pval = 0.0
            if alt == 'two-sided':
                pval = 2.0 * min(cdf, sf)
            elif alt == 'greater':
                pval = cdf
            else:
                pval = sf
            return (AB, min(pval, 1.0))
        Nf = float(N)
        mnAB = (n * (Nf + 1.0) ** 2 / 4.0 / Nf) if N % 2 else (n * (Nf + 2.0) / 4.0)
        varAB = 0.0
        if repeats:
            fac = float(np.sum(symrank ** 2))
            if N % 2:
                varAB = m * n * (16 * Nf * fac - (Nf + 1) ** 4) / (16.0 * Nf ** 2 * (Nf - 1))
            else:
                varAB = m * n * (16 * fac - Nf * (Nf + 2) ** 2) / (16.0 * Nf * (Nf - 1))
        else:
            if N % 2:
                varAB = n * m * (Nf + 1.0) * (3 + Nf ** 2) / (48.0 * Nf ** 2)
            else:
                varAB = m * n * (Nf + 2) * (Nf - 2.0) / 48 / (Nf - 1.0)
        z = (mnAB - AB) / K.sqrt(varAB)
        return (AB, _pvalue_norm(z, alt))

def ansari(x, y, alternative: str = 'two-sided', *, axis = 0, method: str = 'auto'):
    if alternative != 'two-sided' and alternative != 'greater' and alternative != 'less':
        raise ValueError("'alternative' must be 'two-sided', 'greater', or 'less'.")
    r = _map_slices2(x, y, axis, _Ansari(alternative, method))
    return AnsariResult(r[0], r[1])

# ------------------------------------------------------------------ Mood

class _Mood:
    alternative: str
    def __call__(self, x, y):
        m = x.shape[0]
        n = y.shape[0]
        N = m + n
        if m == 0 or n == 0 or N < 3:
            return (NAN, NAN)
        xy = np.concatenate((x, y))
        r, ties = _rank_avg(xy)
        mf = float(m)
        nf = float(n)
        Nf = float(N)
        anyt = False
        for t in ties:
            if t > 1:
                anyt = True
        z = 0.0
        if anyt:
            # scipy _mood_statistic_with_ties
            E_0_T = mf * (Nf * Nf - 1) / 12
            nt = len(ties)
            S = np.zeros(nt + 1, dtype=float)
            for i in range(nt):
                S[i + 1] = S[i] + ties[i]
            acc = np.empty(nt, dtype=float)
            for i in range(nt):
                t = ties[i]
                acc[i] = t * (t ** 2 - 1) * (t ** 2 - 4 + (15 * (Nf - S[i + 1] - S[i]) ** 2))
            varM = (mf * nf * (Nf + 1.0) * (Nf ** 2 - 4) / 180
                    - mf * nf / (180 * Nf * (Nf - 1)) * float(np.sum(acc)))
            c = (Nf + 1) / 2
            phi = np.empty(nt, dtype=float)
            for i in range(nt):
                a = S[i] + 1
                b = S[i + 1]
                sI2 = (b * (b + 1) * (2 * b + 1) / 6) - (a * (a + 1) * (2 * a + 1) / 6) + a ** 2
                sI = (b * (b + 1) / 2) - (a * (a + 1) / 2) + a
                s1 = (b - a) + 1
                phi[i] = (sI2 - 2 * c * sI + s1 * c ** 2) / ties[i]
            # number of x values in each tie group
            xs = np.sort(x)
            ys = np.sort(y)
            T = 0.0
            # walk the sorted unique values
            allv = np.sort(xy, kind='mergesort')
            gi = 0
            i = 0
            ix = 0
            while i < N:
                j = i
                while j + 1 < N and allv[j + 1] == allv[i]:
                    j += 1
                v = allv[i]
                cx = 0
                while ix < m and xs[ix] == v:
                    cx += 1
                    ix += 1
                T += float(cx) * phi[gi]
                gi += 1
                i = j + 1
            z = (T - E_0_T) / K.sqrt(varM)
        else:
            rx = r[:m]
            M = float(np.sum((rx - (Nf + 1.0) / 2) ** 2))
            E_0_T = mf * (Nf * Nf - 1.0) / 12
            varM = mf * nf * (Nf + 1.0) * (Nf + 2) * (Nf - 2) / 180
            z = (M - E_0_T) / K.sqrt(varM)
        return (z, _pvalue_norm(z, self.alternative))

def mood(x, y, axis = 0, alternative: str = "two-sided"):
    r = _map_slices2(x, y, axis, _Mood(alternative))
    return SignificanceResult(r[0], r[1])

# ------------------------------------------------------------------ Levene / Bartlett / Fligner

class _Center:
    kind: str
    p: float
    def __call__(self, x) -> float:
        if self.kind == 'median':
            return _median1(x)
        elif self.kind == 'mean':
            return float(np.mean(x))
        else:
            return _TrimMean(self.p)(x)

def _chk_center(center: str):
    if center != 'mean' and center != 'median' and center != 'trimmed':
        raise ValueError("center must be 'mean', 'median' or 'trimmed'.")

class _Levene:
    cf: _Center
    def __call__(self, ss):
        k = len(ss)
        Ntot = 0.0
        Zs = []
        Zbars = []
        Nis = []
        for s in ss:
            Ni = float(s.shape[0])
            Z = np.abs(s - self.cf(s))
            Zs.append(Z)
            Zbars.append(float(np.mean(Z)))
            Nis.append(Ni)
            Ntot += Ni
        Zbar = 0.0
        for i in range(k):
            Zbar += Nis[i] * Zbars[i]
        Zbar /= Ntot
        dfd = Ntot - k
        numer = 0.0
        for i in range(k):
            numer += Nis[i] * (Zbars[i] - Zbar) ** 2
        numer *= dfd
        dfn = k - 1.0
        den = 0.0
        for i in range(k):
            den += float(np.sum((Zs[i] - Zbars[i]) ** 2))
        den *= dfn
        W = numer / den
        return (W, K.fdtrc(dfn, dfd, W))

def levene(*samples, center: str = 'median', proportiontocut: float = 0.05, axis = 0):
    _chk_center(center)
    if static.len(samples) < 2:
        compile_error("Must provide at least two samples.")
    arrs = [_f64(s) for s in samples]
    r = _ksample(arrs, axis, _Levene(_Center(center, proportiontocut)))
    return LeveneResult(r[0], r[1])

class _Bartlett:
    def __call__(self, ss):
        k = len(ss)
        Ntot = 0.0
        sp = 0.0
        sl = 0.0
        sinv = 0.0
        for s in ss:
            Ni = float(s.shape[0])
            v = _var(s, 1)
            Ntot += Ni
            sp += (Ni - 1) * v
            sl += (Ni - 1) * K.log(v)
            sinv += 1 / (Ni - 1)
        spsq = sp / (Ntot - k)
        numer = (Ntot - k) * K.log(spsq) - sl
        denom = 1 + 1 / (3 * (k - 1)) * (sinv - 1 / (Ntot - k))
        T = numer / denom
        p = K.chdtrc(float(k - 1), T)
        return (max(T, 0.), p)

def bartlett(*samples, axis = 0):
    if static.len(samples) < 2:
        compile_error("Must enter at least two input sample vectors.")
    arrs = [_f64(s) for s in samples]
    r = _ksample(arrs, axis, _Bartlett())
    return BartlettResult(r[0], r[1])

class _Fligner:
    cf: _Center
    def __call__(self, ss):
        k = len(ss)
        parts = []
        lens = []
        for s in ss:
            parts.append(np.abs(s - self.cf(s)))
            lens.append(s.shape[0])
        allv = np.concatenate(parts)
        N = float(allv.shape[0])
        ranks, _ = _rank_avg(allv)
        a = np.empty(allv.shape[0], dtype=float)
        for i in range(allv.shape[0]):
            a[i] = K.ndtri(ranks[i] / (2 * (N + 1.0)) + 0.5)
        abar = float(np.mean(a))
        V2 = _var(a, 1)
        st = 0.0
        j = 0
        for i in range(k):
            ni = lens[i]
            Ab = float(np.mean(a[j:j + ni]))
            st += float(ni) * (Ab - abar) ** 2
            j += ni
        st /= V2
        return (st, K.chdtrc(float(k - 1), st))

def fligner(*samples, center: str = 'median', proportiontocut: float = 0.05, axis = 0):
    _chk_center(center)
    if static.len(samples) < 2:
        compile_error("Must provide at least two samples.")
    arrs = [_f64(s) for s in samples]
    r = _ksample(arrs, axis, _Fligner(_Center(center, proportiontocut)))
    return FlignerResult(r[0], r[1])

def median_test(*samples, ties: str = 'below', correction: bool = True, lambda_ = 1,
                nan_policy: str = 'propagate'):
    if static.len(samples) < 2:
        compile_error('median_test requires two or more samples.')
    if ties != 'below' and ties != 'above' and ties != 'ignore':
        raise ValueError(f"invalid 'ties' option '{ties}'; 'ties' must be one of: 'below', 'above', 'ignore'")
    data = [_f64(s).ravel() for s in samples]
    gm = _median1(np.concatenate(data))
    table = np.zeros((2, len(data)), dtype=int)
    for k in range(len(data)):
        nab = 0
        nbe = 0
        for v in data[k]:
            if v > gm:
                nab += 1
            elif v < gm:
                nbe += 1
        neq = data[k].shape[0] - nab - nbe
        table[0, k] += nab
        table[1, k] += nbe
        if ties == "below":
            table[1, k] += neq
        elif ties == "above":
            table[0, k] += neq
    if int(np.sum(table[0])) == 0:
        raise ValueError(f"All values are below the grand median ({gm}).")
    if int(np.sum(table[1])) == 0:
        raise ValueError(f"All values are above the grand median ({gm}).")
    r = chi2_contingency(table, correction=correction, lambda_=lambda_)
    return MedianTestResult(r.statistic, r.pvalue, gm, table)

# ------------------------------------------------------------------ discrete helpers

def _lbinom(n: float, k: float) -> float:
    return K.gammaln(n + 1) - K.gammaln(k + 1) - K.gammaln(n - k + 1)

def _hyp_pmf(x: float, M: float, n: float, N: float) -> float:
    lo = max(0., N - (M - n))
    hi = min(N, n)
    if x < lo or x > hi:
        return 0.0
    return K.exp(_lbinom(n, x) + _lbinom(M - n, N - x) - _lbinom(M, N))

def _hyp_cdf(x: float, M: float, n: float, N: float) -> float:
    lo = max(0., N - (M - n))
    s = 0.0
    k = lo
    while k <= x:
        s += _hyp_pmf(k, M, n, N)
        k += 1.
    return min(s, 1.0)

def _hyp_sf(x: float, M: float, n: float, N: float) -> float:
    hi = min(N, n)
    s = 0.0
    k = hi
    while k > x:
        s += _hyp_pmf(k, M, n, N)
        k -= 1.
    return min(s, 1.0)

def _binom_pmf(k: float, n: float, p: float) -> float:
    if k < 0 or k > n:
        return 0.0
    if p == 0.0:
        return 1.0 if k == 0 else 0.0
    if p == 1.0:
        return 1.0 if k == n else 0.0
    return K.exp(_lbinom(n, k) + K.xlogy(k, p) + K.xlog1py(n - k, -p))

def _binom_cdf(k: float, n: float, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return K.betainc(n - k, k + 1, 1 - p)

def _binom_sf(k: float, n: float, p: float) -> float:
    if k < 0:
        return 1.0
    if k >= n:
        return 0.0
    return K.betainc(k + 1, n - k, p)

# ------------------------------------------------------------------ binomtest

class _NegPmfB:
    n: float
    p: float
    neg: bool
    def __call__(self, x: float) -> float:
        v = _binom_pmf(K.floor(x), self.n, self.p)
        return -v if self.neg else v

def _bsearch(f, d: float, lo0: float, hi0: float) -> float:
    lo = lo0
    hi = hi0
    while lo < hi:
        mid = lo + K.floor((hi - lo) / 2)
        midval = f(mid)
        if midval < d:
            lo = mid + 1
        elif midval > d:
            hi = mid - 1
        else:
            lo = mid
            hi = mid
    return lo if f(lo) <= d else lo - 1

class BinomTestResult:
    k: int
    n: int
    alternative: str
    statistic: float
    pvalue: float
    proportion_estimate: float
    def __init__(self, k: int, n: int, alternative: str, statistic: float, pvalue: float):
        self.k = k
        self.n = n
        self.alternative = alternative
        self.statistic = statistic
        self.pvalue = pvalue
        self.proportion_estimate = statistic
    def proportion_ci(self, confidence_level: float = 0.95, method: str = 'exact'):
        k = float(self.k)
        n = float(self.n)
        alt = self.alternative
        if method == 'exact':
            alpha = (1 - confidence_level) / 2 if alt == 'two-sided' else 1 - confidence_level
            plow = 0.0
            phigh = 1.0
            if alt != 'less' and k != 0:
                plow = K.betaincinv(k, n - k + 1, alpha)
            if alt != 'greater' and k != n:
                phigh = K.betaincinv(k + 1, n - k, 1 - alpha)
            return ConfidenceInterval(plow, phigh)
        elif method == 'wilson' or method == 'wilsoncc':
            p = k / n
            z = K.ndtri(0.5 + 0.5 * confidence_level) if alt == 'two-sided' else K.ndtri(confidence_level)
            denom = 2 * (n + z ** 2)
            center = (2 * n * p + z ** 2) / denom
            q = 1 - p
            dlo = 0.0
            dhi = 0.0
            if method == 'wilsoncc':
                dlo = (1 + z * K.sqrt(z ** 2 - 2 - 1 / n + 4 * p * (n * q + 1))) / denom
                dhi = (1 + z * K.sqrt(z ** 2 + 2 - 1 / n + 4 * p * (n * q - 1))) / denom
            else:
                dlo = z / denom * K.sqrt(4 * n * p * q + z ** 2)
                dhi = dlo
            lo = 0.0 if (k == 0 or alt == 'less') else center - dlo
            hi = 1.0 if (k == n or alt == 'greater') else center + dhi
            return ConfidenceInterval(lo, hi)
        else:
            raise ValueError(f"method ('{method}') must be one of 'exact', 'wilson' or 'wilsoncc'.")

def binomtest(k: int, n: int, p: float = 0.5, alternative: str = 'two-sided'):
    if alternative != 'two-sided' and alternative != 'less' and alternative != 'greater':
        raise ValueError(f"alternative ('{alternative}') not recognized; \nmust be 'two-sided', 'less' or 'greater'")
    kf = float(k)
    nf = float(n)
    pval = 0.0
    if alternative == 'less':
        pval = _binom_cdf(kf, nf, p)
    elif alternative == 'greater':
        pval = _binom_sf(kf - 1, nf, p)
    else:
        d = _binom_pmf(kf, nf, p)
        rerr = 1 + 1e-7
        if kf < p * nf:
            ix = _bsearch(_NegPmfB(nf, p, True), -d * rerr, K.ceil(p * nf), nf)
            y = nf - ix + (1. if d * rerr == _binom_pmf(ix, nf, p) else 0.)
            pval = _binom_cdf(kf, nf, p) + _binom_sf(nf - y, nf, p)
        else:
            ix = _bsearch(_NegPmfB(nf, p, False), d * rerr, 0., K.floor(p * nf))
            y = ix + 1
            pval = _binom_cdf(y - 1, nf, p) + _binom_sf(kf - 1, nf, p)
        pval = min(1.0, pval)
    return BinomTestResult(k, n, alternative, kf / nf, pval)

# ------------------------------------------------------------------ Fisher exact (2x2)

class _HPmf:
    M: float
    n: float
    N: float
    neg: bool
    def __call__(self, x: float) -> float:
        v = _hyp_pmf(K.floor(x), self.M, self.n, self.N)
        return -v if self.neg else v

def fisher_exact(table, alternative: str = 'two-sided', *, method = None):
    c = np.asarray(table).astype(int)
    if c.ndim != 2 or c.shape[0] != 2 or c.shape[1] != 2:
        raise ValueError("only 2x2 tables are supported")
    for v in c.flat:
        if v < 0:
            raise ValueError("All values in `table` must be nonnegative.")
    a = float(c[0, 0])
    b = float(c[0, 1])
    cc = float(c[1, 0])
    d = float(c[1, 1])
    if a + cc == 0 or b + d == 0 or a + b == 0 or cc + d == 0:
        return SignificanceResult(NAN, 1.0)
    oddsratio = INF
    if cc > 0 and b > 0:
        oddsratio = a * d / (cc * b)
    n1 = a + b
    n2 = cc + d
    n = a + cc
    M = n1 + n2
    pvalue = 0.0
    if alternative == 'less':
        pvalue = _hyp_cdf(a, M, n1, n)
    elif alternative == 'greater':
        pvalue = _hyp_cdf(b, M, n1, b + d)
    elif alternative == 'two-sided':
        mode = float(int((n + 1) * (n1 + 1) / (n1 + n2 + 2)))
        pexact = _hyp_pmf(a, M, n1, n)
        pmode = _hyp_pmf(mode, M, n1, n)
        epsilon = 1e-14
        gamma = 1 + epsilon
        if abs(pexact - pmode) / max(pexact, pmode) <= epsilon:
            return SignificanceResult(oddsratio, 1.)
        elif a < mode:
            plower = _hyp_cdf(a, M, n1, n)
            if _hyp_pmf(n, M, n1, n) > pexact * gamma:
                return SignificanceResult(oddsratio, plower)
            guess = _bsearch(_HPmf(M, n1, n, True), -pexact * gamma, mode, n)
            pvalue = plower + _hyp_sf(guess, M, n1, n)
        else:
            pupper = _hyp_sf(a - 1, M, n1, n)
            if _hyp_pmf(0., M, n1, n) > pexact * gamma:
                return SignificanceResult(oddsratio, pupper)
            guess = _bsearch(_HPmf(M, n1, n, False), pexact * gamma, 0., mode)
            pvalue = pupper + _hyp_cdf(guess, M, n1, n)
    else:
        raise ValueError("`alternative` should be one of {'two-sided', 'less', 'greater'}")
    return SignificanceResult(oddsratio, min(pvalue, 1.0))

# ------------------------------------------------------------------ Kolmogorov-Smirnov distributions (scipy _ksstats)

_E128 = 128
_EP128 = 3.402823669209385e+38
_EM128 = 2.938735877055719e-39
_MIN_LOG = -708.

def _log_nfactorial_div_n_pow_n(n: float) -> float:
    rn = 1.0 / n
    co = [-2.955065359477124183e-2, 6.4102564102564102564e-3,
          -1.9175269175269175269e-3, 8.4175084175084175084e-4,
          -5.952380952380952381e-4, 7.9365079365079365079e-4,
          -2.7777777777777777778e-3, 8.3333333333333333333e-2]
    x = rn / n
    pv = 0.0
    for c in co:
        pv = pv * x + c
    return K.log(n) / 2 - n + K.log(2 * _PI) / 2 + rn * pv

def _clip01(p: float) -> float:
    return 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)

def _sel(cdfprob: float, sfprob: float, cdf: bool) -> float:
    return _clip01(cdfprob if cdf else sfprob)

def _kolmogn_DMTW(n: int, d: float, cdf: bool) -> float:
    if d >= 1.0:
        return _sel(1.0, 0.0, cdf)
    nd = n * d
    if nd <= 0.5:
        return _sel(0.0, 1.0, cdf)
    k = int(K.ceil(nd))
    h = k - nd
    m = 2 * k - 1
    H = np.zeros((m, m))
    v = np.empty(m, dtype=float)
    w = np.empty(m, dtype=float)
    fac = 1.0
    for j in range(1, m + 1):
        w[j - 1] = fac
        fac /= j
        v[j - 1] = (1.0 - h ** j) * fac
    tt = max(2 * h - 1.0, 0.) ** m - 2 * h ** m
    v[m - 1] = (1.0 + tt) * fac
    for i in range(1, m):
        for r in range(i - 1, m):
            H[r, i] = w[r - (i - 1)]
    for r in range(m):
        H[r, 0] = v[r]
    for c in range(m):
        H[m - 1, c] = v[m - 1 - c]
    Hpwr = np.eye(m)
    nn = n
    expnt = 0
    Hexpnt = 0
    while nn > 0:
        if nn % 2:
            Hpwr = Hpwr @ H
            expnt += Hexpnt
        H = H @ H
        Hexpnt *= 2
        if abs(H[k - 1, k - 1]) > _EP128:
            H = H / _EP128
            Hexpnt += _E128
        nn = nn // 2
    p = Hpwr[k - 1, k - 1]
    for i in range(1, n + 1):
        p = i * p / n
        if abs(p) < _EM128:
            p *= _EP128
            expnt -= _E128
    if expnt != 0:
        p = ldexp(p, i32(expnt))
    return _sel(p, 1.0 - p, cdf)

def _pomeranz_j1j2(i: int, n: int, ll: int, ceilf: int, roundf: int):
    j1 = 0
    j2 = 0
    if i == 0:
        j1 = -ll - ceilf - 1
        j2 = ll + ceilf - 1
    else:
        ip1div2 = (i + 1) // 2
        ip1mod2 = (i + 1) % 2
        if ip1mod2 == 0:
            if ip1div2 == n + 1:
                j1 = n - ll - ceilf - 1
                j2 = n + ll + ceilf - 1
            else:
                j1 = ip1div2 - 1 - ll - roundf - 1
                j2 = ip1div2 + ll - 1 + ceilf - 1
        else:
            j1 = ip1div2 - 1 - ll - 1
            j2 = ip1div2 + ll + roundf - 1
    return max(j1 + 2, 0), min(j2, n)

def _kolmogn_Pomeranz(n: int, x: float, cdf: bool) -> float:
    t = n * x
    ll = int(K.floor(t))
    f = 1.0 * (t - ll)
    g = min(f, 1.0 - f)
    ceilf = 1 if f > 0 else 0
    roundf = 1 if f > 0.5 else 0
    npwrs = 2 * (ll + 1)
    gpower = np.empty(npwrs, dtype=float)
    twogpower = np.empty(npwrs, dtype=float)
    onem2gpower = np.empty(npwrs, dtype=float)
    gpower[0] = 1.0
    twogpower[0] = 1.0
    onem2gpower[0] = 1.0
    expnt = 0
    g_over_n = g / n
    two_g_over_n = 2 * g / n
    omt = (1 - 2 * g) / n
    for m in range(1, npwrs):
        gpower[m] = gpower[m - 1] * g_over_n / m
        twogpower[m] = twogpower[m - 1] * two_g_over_n / m
        onem2gpower[m] = onem2gpower[m - 1] * omt / m
    V0 = np.zeros(npwrs)
    V1 = np.zeros(npwrs)
    V1[0] = 1.
    V0s = 0
    V1s = 0
    j1, j2 = _pomeranz_j1j2(0, n, ll, ceilf, roundf)
    for i in range(1, 2 * n + 2):
        k1 = j1
        V0, V1 = V1, V0
        V0s, V1s = V1s, V0s
        for q in range(npwrs):
            V1[q] = 0.0
        j1, j2 = _pomeranz_j1j2(i, n, ll, ceilf, roundf)
        pwrs = gpower
        if not (i == 1 or i == 2 * n + 1):
            pwrs = twogpower if i % 2 else onem2gpower
        ln2 = j2 - k1 + 1
        if ln2 > 0:
            conv = np.convolve(V0[k1 - V0s:k1 - V0s + ln2], pwrs[:ln2])
            conv_start = j1 - k1
            conv_len = j2 - j1 + 1
            for q in range(conv_len):
                V1[q] = conv[conv_start + q]
            mx = float(np.max(V1))
            if 0 < mx < _EM128:
                V1 = V1 * _EP128
                expnt -= _E128
            V1s = V0s + j1 - k1
    ans = V1[n - V1s]
    for m in range(1, n + 1):
        if abs(ans) > _EP128:
            ans *= _EM128
            expnt += _E128
        ans *= m
    if expnt != 0:
        ans = ldexp(ans, i32(expnt))
    return _sel(ans, 1.0 - ans, cdf)

def _kolmogn_PelzGood(n: int, x: float, cdf: bool) -> float:
    if x <= 0.0:
        return _sel(0.0, 1.0, cdf)
    if x >= 1.0:
        return _sel(1.0, 0.0, cdf)
    PI2 = _PI ** 2
    PI4 = _PI ** 4
    PI6 = _PI ** 6
    z = K.sqrt(float(n)) * x
    zsquared = z ** 2
    zthree = z ** 3
    zfour = z ** 4
    zsix = z ** 6
    qlog = -PI2 / 8 / zsquared
    if qlog < _MIN_LOG:
        return _sel(0.0, 1.0, cdf)
    q = K.exp(qlog)
    k1a = -zsquared
    k1b = PI2 / 4
    k2a = 6 * zsix + 2 * zfour
    k2b = (2 * zfour - 5 * zsquared) * PI2 / 4
    k2c = PI4 * (1 - 2 * zsquared) / 16
    k3d = PI6 * (5 - 30 * zsquared) / 64
    k3c = PI4 * (-60 * zsquared + 212 * zfour) / 16
    k3b = PI2 * (135 * zfour - 96 * zsix) / 4
    k3a = -30 * zsix - 90 * z ** 8
    K0 = [0.0, 0.0, 0.0, 0.0]
    maxk = int(K.ceil(16 * z / _PI))
    for k in range(maxk, 0, -1):
        m = 2 * k - 1
        msq = float(m ** 2)
        mfour = float(m ** 4)
        msix = float(m ** 6)
        qpower = q ** float(8 * k)
        co = [1.0, k1a + k1b * msq, k2a + k2b * msq + k2c * mfour,
              k3a + k3b * msq + k3c * mfour + k3d * msix]
        for i in range(4):
            K0[i] = K0[i] * qpower + co[i]
    sq2pi = K.sqrt(2 * _PI)
    den = [z, 6 * zfour, 72 * z ** 7, 6480 * z ** 10]
    for i in range(4):
        K0[i] = K0[i] * q * sq2pi / den[i]
    q2 = K.exp(-PI2 / 2 / zsquared)
    sqrt3z = K.sqrt(3.0) * z
    nk = maxk
    a2 = np.empty(nk, dtype=float)
    a3 = np.empty(nk, dtype=float)
    for ii in range(nk):
        kk = float(maxk - ii)
        ksq = kk ** 2
        qp = q2 ** ksq
        kspi = _PI * kk
        a2[ii] = ksq * qp
        a3[ii] = (sqrt3z + kspi) * (sqrt3z - kspi) * ksq * qp
    K0[2] += float(np.sum(a2)) * (PI2 * sq2pi / (-36 * zthree))
    K0[3] += float(np.sum(a3)) * (PI2 * sq2pi / (216 * zsix))
    for i in range(4):
        K0[i] /= float(n) ** (i / 2.0)
    if not cdf:
        for i in range(4):
            K0[i] *= -1
        K0[0] += 1
    return K0[0] + K0[1] + K0[2] + K0[3]

def _kolmogn(n: int, x: float, cdf: bool) -> float:
    if n <= 0:
        return NAN
    if x >= 1.0:
        return _sel(1.0, 0.0, cdf)
    if x <= 0.0:
        return _sel(0.0, 1.0, cdf)
    t = n * x
    if t <= 1.0:
        if t <= 0.5:
            return _sel(0.0, 1.0, cdf)
        prob = 0.0
        if n <= 140:
            arr = np.empty(n, dtype=float)
            for i in range(n):
                arr[i] = (i + 1) * (1.0 / n) * (2 * t - 1)
            prob = float(np.prod(arr))
        else:
            prob = K.exp(_log_nfactorial_div_n_pow_n(float(n)) + n * K.log(2 * t - 1))
        return _sel(prob, 1.0 - prob, cdf)
    if t >= n - 1:
        prob = 2 * (1.0 - x) ** n
        return _sel(1 - prob, prob, cdf)
    if x >= 0.5:
        prob = 2 * KS.smirnov(float(n), x)
        return _sel(1.0 - prob, prob, cdf)
    nxsquared = t * x
    if n <= 140:
        if nxsquared <= 0.754693:
            prob = _kolmogn_DMTW(n, x, True)
            return _sel(prob, 1.0 - prob, cdf)
        if nxsquared <= 4:
            prob = _kolmogn_Pomeranz(n, x, True)
            return _sel(prob, 1.0 - prob, cdf)
        prob = 2 * KS.smirnov(float(n), x)
        return _sel(1.0 - prob, prob, cdf)
    if not cdf:
        if nxsquared >= 370.0:
            return 0.0
        if nxsquared >= 2.2:
            return _clip01(2 * KS.smirnov(float(n), x))
    cdfprob = 0.0
    if nxsquared >= 18.0:
        cdfprob = 1.0
    elif n <= 100000 and n * x ** 1.5 <= 1.4:
        cdfprob = _kolmogn_DMTW(n, x, True)
    else:
        cdfprob = _kolmogn_PelzGood(n, x, True)
    return _sel(cdfprob, 1.0 - cdfprob, cdf)

# ------------------------------------------------------------------ KS tests

def _ks1_core(x, cdfvals, alternative: str, mode: str):
    n = x.shape[0]
    N = float(n)
    dp = -INF
    dpl = 0.0
    dm = -INF
    dml = 0.0
    for i in range(n):
        a = (i + 1.0) / N - cdfvals[i]
        b = cdfvals[i] - i / N
        if a > dp:
            dp = a
            dpl = x[i]
        if b > dm:
            dm = b
            dml = x[i]
    if alternative == 'greater':
        return (dp, _clip01(KS.smirnov(N, dp)), dpl, 1)
    if alternative == 'less':
        return (dm, _clip01(KS.smirnov(N, dm)), dml, -1)
    D = dp
    loc = dpl
    sg = 1
    if not (dp > dm):
        D = dm
        loc = dml
        sg = -1
    prob = 0.0
    if mode == 'auto' or mode == 'exact':
        prob = _kolmogn(n, D, False)
    elif mode == 'asymp':
        prob = KS.kolmogorov(D * K.sqrt(N))
    else:
        prob = 2 * KS.smirnov(N, D)
    return (D, _clip01(prob), loc, sg)

def _norm_alt(alternative: str) -> str:
    c = alternative.lower()[0]
    if c == 't':
        return 'two-sided'
    elif c == 'g':
        return 'greater'
    elif c == 'l':
        return 'less'
    raise ValueError(f"Unexpected value alternative={alternative}")

def _cdf_of(cdf, args, xs):
    # cdf: callable or distribution object with .cdf, or a name string
    out = np.empty(xs.shape[0], dtype=float)
    if isinstance(cdf, str):
        vals = np.empty(xs.shape[0], dtype=float)
        if cdf == 'norm':
            vals = norm.cdf(xs, *args)
        elif cdf == 'expon':
            vals = expon.cdf(xs, *args)
        elif cdf == 'uniform':
            vals = uniform.cdf(xs, *args)
        elif cdf == 'logistic':
            vals = logistic.cdf(xs, *args)
        elif cdf == 'laplace':
            vals = laplace.cdf(xs, *args)
        elif cdf == 'cauchy':
            vals = cauchy.cdf(xs, *args)
        else:
            raise ValueError(f"unsupported distribution name {cdf!r}")
        for i in range(xs.shape[0]):
            out[i] = float(vals[i])
    elif hasattr(cdf, 'cdf'):
        vals = cdf.cdf(xs, *args)
        for i in range(xs.shape[0]):
            out[i] = float(vals[i])
    else:
        vals = cdf(xs, *args)
        for i in range(xs.shape[0]):
            out[i] = float(vals[i])
    return out

class _KS1[C, A]:
    cdf: C
    args: A
    alternative: str
    mode: str
    def __call__(self, x0):
        x = np.sort(x0)
        cv = _cdf_of(self.cdf, self.args, x)
        return _ks1_core(x, cv, self.alternative, self.mode)

def _map4(a, axis, f):
    m, shp = _slices(a, axis)
    if static.len(shp) == 0:
        return f(np.ascontiguousarray(m))
    else:
        s = np.empty(shp, dtype=float)
        p = np.empty(shp, dtype=float)
        l = np.empty(shp, dtype=float)
        g = np.empty(shp, dtype=int)
        for idx in np.ndindex(shp):
            r = f(np.ascontiguousarray(m[idx]))
            s[idx] = r[0]
            p[idx] = r[1]
            l[idx] = r[2]
            g[idx] = r[3]
        return (s, p, l, g)

def ks_1samp(x, cdf, args = (), alternative: str = 'two-sided', method: str = 'auto', *, axis = 0):
    if method != 'auto' and method != 'exact' and method != 'approx' and method != 'asymp':
        raise ValueError(f"Invalid value for method: {method!r}. Must be one of 'auto', 'exact', 'approx', 'asymp'.")
    alt = _norm_alt(alternative)
    r = _map4(x, axis, _KS1(cdf, args, alt, method))
    return KstestResult(r[0], r[1], r[2], r[3])

def _prob_outside_square(n: int, h: int) -> float:
    P = 0.0
    k = int(K.floor(n / h))
    while k >= 0:
        p1 = 1.0
        for j in range(h):
            p1 = (n - k * h - j) * p1 / (n + k * h + j + 1)
        P = p1 * (1.0 - P)
        k -= 1
    return 2 * P

def _outer_prob_inside(m0: int, n0: int, g: int, h: int) -> float:
    m = m0
    n = n0
    if m < n:
        m, n = n, m
    mg = m // g
    ng = n // g
    minj = 0
    maxj = min(int(K.ceil(h / mg)), n + 1)
    curlen = maxj - minj
    lenA = min(2 * maxj + 2, n + 1)
    A = np.ones(lenA, dtype=float)
    for q in range(minj, maxj):
        A[q] = 0.0
    for i in range(1, m + 1):
        lastminj = minj
        lastlen = curlen
        minj = max(int(K.floor((ng * i - h) / mg)) + 1, 0)
        minj = min(minj, n)
        maxj = min(int(K.ceil((ng * i + h) / mg)), n + 1)
        if maxj <= minj:
            return 1.0
        val = 0.0 if minj == 0 else 1.0
        for jj in range(maxj - minj):
            j = jj + minj
            val = (A[jj + minj - lastminj] * i + val * j) / (i + j)
            A[jj] = val
        curlen = maxj - minj
        if lastlen > curlen:
            for q in range(maxj - minj, min(maxj - minj + (lastlen - curlen), lenA)):
                A[q] = 1.
    return A[maxj - minj - 1]

def _binomf(n: float, k: float) -> float:
    return MS.binom(n, k)

def _count_paths_outside(m0: int, n0: int, g: int, h: int) -> float:
    m = m0
    n = n0
    if m < n:
        m, n = n, m
    mg = m // g
    ng = n // g
    lxj = n + (mg - h) // mg
    xj = [(h + mg * j + ng - 1) // ng for j in range(lxj)]
    if lxj == 0:
        return _binomf(float(m + n), float(n))
    B = np.zeros(lxj)
    B[0] = 1
    for j in range(1, lxj):
        Bj = _binomf(float(xj[j] + j), float(j))
        for i in range(j):
            Bj -= _binomf(float(xj[j] - xj[i] + j - i), float(j - i)) * B[i]
        B[j] = Bj
    num = 0.0
    for j in range(lxj):
        num += B[j] * _binomf(float((m - xj[j]) + (n - j)), float(n - j))
    return num

def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a

def _ks2_prob(d0: float, n1: int, n2: int, mode0: str, alternative: str) -> float:
    g = _gcd(n1, n2)
    n1g = n1 // g
    n2g = n2 // g
    mode = mode0
    d = d0
    prob = -INF
    if mode == 'auto':
        mode = 'exact' if max(n1, n2) <= 10000 else 'asymp'
    elif mode == 'exact':
        if n1g >= 2147483647 / n2g:
            mode = 'asymp'
    if mode == 'exact':
        lcm = (n1 // g) * n2
        h = int(rint(d * lcm))
        d = h * 1.0 / lcm
        ok = True
        if h == 0:
            prob = 1.0
        else:
            if alternative == 'two-sided':
                if n1 == n2:
                    prob = _prob_outside_square(n1, h)
                else:
                    prob = _outer_prob_inside(n1, n2, g, h)
            else:
                if n1 == n2:
                    pr = 1.0
                    arr = np.empty(h, dtype=float)
                    for j in range(h):
                        arr[j] = (n1 - j) / (n1 + j + 1.0)
                    prob = float(np.prod(arr))
                else:
                    num = _count_paths_outside(n1, n2, g, h)
                    bn = _binomf(float(n1 + n2), float(n1))
                    if num > bn or bn == INF or num != num:
                        ok = False
                    else:
                        prob = num / bn
            if not (0 <= prob <= 1):
                ok = False
        if not ok:
            mode = 'asymp'
    if mode == 'asymp':
        m = float(max(n1, n2))
        n = float(min(n1, n2))
        en = m * n / (m + n)
        if alternative == 'two-sided':
            prob = _kolmogn(int(rint(en)), d, False)
        else:
            z = K.sqrt(en) * d
            expt = -2 * z ** 2 - 2 * z * (m + 2 * n) / K.sqrt(m * n * (m + n)) / 3.0
            prob = K.exp(expt)
    return _clip01(prob)

class _KS2:
    alternative: str
    mode: str
    def __call__(self, a, b):
        d1 = np.sort(a)
        d2 = np.sort(b)
        n1 = d1.shape[0]
        n2 = d2.shape[0]
        if min(n1, n2) == 0:
            raise ValueError('Data passed to ks_2samp must not be empty')
        allv = np.sort(np.concatenate((d1, d2)), kind='mergesort')
        n = allv.shape[0]
        minS = INF
        maxS = -INF
        lmin = 0.0
        lmax = 0.0
        for i in range(n):
            v = allv[i]
            c1 = float(_searchsorted(d1, v, True)) / n1
            c2 = float(_searchsorted(d2, v, True)) / n2
            dd = c1 - c2
            if dd < minS:
                minS = dd
                lmin = v
            if dd > maxS:
                maxS = dd
                lmax = v
        minS = -minS
        minS = _clip01(minS)
        sel = False
        if self.alternative == 'less':
            sel = True
        elif self.alternative == 'two-sided':
            sel = minS > maxS
        d = minS if sel else maxS
        loc = lmin if sel else lmax
        sg = -1 if sel else 1
        return (d, _ks2_prob(d, n1, n2, self.mode, self.alternative), loc, sg)

def _map4_2(a, b, axis, f):
    if axis is None:
        return f(_f64(a).ravel(), _f64(b).ravel())
    ma, shp = _slices(a, axis)
    mb, _ = _slices(b, axis)
    if static.len(shp) == 0:
        return f(np.ascontiguousarray(ma), np.ascontiguousarray(mb))
    else:
        s = np.empty(shp, dtype=float)
        p = np.empty(shp, dtype=float)
        l = np.empty(shp, dtype=float)
        g = np.empty(shp, dtype=int)
        for idx in np.ndindex(shp):
            r = f(np.ascontiguousarray(ma[idx]), np.ascontiguousarray(mb[idx]))
            s[idx] = r[0]
            p[idx] = r[1]
            l[idx] = r[2]
            g[idx] = r[3]
        return (s, p, l, g)

def ks_2samp(data1, data2, alternative: str = 'two-sided', method: str = 'auto', *, axis = 0):
    if method != 'auto' and method != 'exact' and method != 'asymp':
        raise ValueError(f'Invalid value for mode: {method}')
    alt = _norm_alt(alternative)
    r = _map4_2(data1, data2, axis, _KS2(alt, method))
    return KstestResult(r[0], r[1], r[2], r[3])

def kstest(rvs, cdf, args = (), N: int = 20, alternative: str = 'two-sided',
           method: str = 'auto', *, axis = 0):
    alt = 'two-sided' if alternative == 'two_sided' else alternative
    if isinstance(cdf, np.ndarray) or isinstance(cdf, List):
        return ks_2samp(rvs, cdf, alternative=alt, method=method, axis=axis)
    else:
        return ks_1samp(rvs, cdf, args=args, alternative=alt, method=method, axis=axis)

# ------------------------------------------------------------------ Cramer-von Mises

def _cdf_cvm_inf(x: float) -> float:
    tot = 0.0
    k = 0
    while True:
        u = K.exp(K.gammaln(k + 0.5) - K.gammaln(k + 1.)) / (_PI ** 1.5 * K.sqrt(x))
        y = 4 * k + 1
        q = y ** 2 / (16 * x)
        b = BS.kv(0.25, q)
        z = u * K.sqrt(float(y)) * K.exp(-q) * b
        tot += z
        k += 1
        if not (abs(z) >= 1e-7):
            break
    return tot

def _ed2(y: float) -> float:
    z = y ** 2 / 4
    b = BS.kv(1 / 4, z) + BS.kv(3 / 4, z)
    return K.exp(-z) * (y / 2) ** (3 / 2) * b / K.sqrt(_PI)

def _ed3(y: float) -> float:
    z = y ** 2 / 4
    c = K.exp(-z) / K.sqrt(_PI)
    return c * (y / 2) ** (5 / 2) * (2 * BS.kv(1 / 4, z) + 3 * BS.kv(3 / 4, z) - BS.kv(5 / 4, z))

def _psi1_mod(x: float) -> float:
    tot = 0.0
    k = 0
    while True:
        m = 2 * k + 1
        sx = 2 * K.sqrt(x)
        y1 = x ** (3 / 4)
        y2 = x ** (5 / 4)
        g12 = K.gamma(k + 1 / 2)
        g32 = K.gamma(k + 3 / 2)
        e1 = m * g12 * _ed2((4 * k + 3) / sx) / (9 * y1)
        e2 = g12 * _ed3((4 * k + 1) / sx) / (72 * y2)
        e3 = 2 * (m + 2) * g32 * _ed3((4 * k + 5) / sx) / (12 * y2)
        e4 = 7 * m * g12 * _ed2((4 * k + 1) / sx) / (144 * y1)
        e5 = 7 * m * g12 * _ed2((4 * k + 5) / sx) / (144 * y1)
        Ak = e1 + e2 + e3 + e4 + e5
        z = -Ak / (_PI * K.gamma(k + 1.))
        tot += z
        k += 1
        if not (abs(z) >= 1e-7):
            break
    return tot

def _cdf_cvm(x: float, n: float) -> float:
    if 1. / (12 * n) < x and x < n / 3.:
        return _cdf_cvm_inf(x) * (1 + 1. / (12 * n)) + _psi1_mod(x) / n
    if x >= n / 3:
        return 1.
    return 0.

class _CvM[C, A]:
    cdf: C
    args: A
    def __call__(self, rvs):
        n = rvs.shape[0]
        if n <= 1:
            raise ValueError('The sample must contain at least two observations.')
        vals = np.sort(rvs)
        cv = _cdf_of(self.cdf, self.args, vals)
        nf = float(n)
        acc = np.empty(n, dtype=float)
        for i in range(n):
            u = (2 * (i + 1.) - 1) / (2 * nf)
            acc[i] = (u - cv[i]) ** 2
        w = 1 / (12 * nf) + float(np.sum(acc))
        p = max(1. - _cdf_cvm(w, nf), 0.)
        return (w, p)

def cramervonmises(rvs, cdf, args = (), *, axis = 0):
    r = _map_slices(rvs, axis, _CvM(cdf, args))
    return CramerVonMisesResult(r[0], r[1])

def _lcm(a: int, b: int) -> int:
    return a // _gcd(a, b) * b

def _pval_cvm_2samp_exact(s: int, m: int, n: int) -> float:
    lcm = _lcm(m, n)
    a = lcm // m
    b = lcm // n
    mn = m * n
    zeta = lcm ** 2 * (m + n) * (6 * s - mn * (4 * mn - 1)) // (6 * mn ** 2)
    # gs[v]: dict value -> frequency
    gs = [{0: 1}]
    for _ in range(m):
        gs.append(Dict[int, int]())
    for u in range(n + 1):
        next_gs = []
        tmp = Dict[int, int]()
        for v in range(len(gs)):
            g = gs[v]
            merged = Dict[int, int]()
            for key, val in tmp.items():
                merged[key] = val
            for key, val in g.items():
                merged[key] = merged.get(key, 0) + val
            res = (a * v - b * u) ** 2
            tmp = Dict[int, int]()
            for key, val in merged.items():
                tmp[key + res] = val
            next_gs.append(tmp)
        gs = next_gs
    tot = 0
    for key, val in gs[m].items():
        if key >= zeta:
            tot += val
    comb = MS.binom(float(m + n), float(m))
    return float(tot) / comb

class _CvM2:
    method: str
    def __call__(self, x, y):
        nx = x.shape[0]
        ny = y.shape[0]
        if nx <= 1 or ny <= 1:
            raise ValueError('x and y must contain at least two observations.')
        meth = self.method
        if meth == 'auto':
            meth = 'asymptotic' if max(nx, ny) > 20 else 'exact'
        xa = np.sort(x)
        ya = np.sort(y)
        r, _ = _rank_avg(np.concatenate((xa, ya)))
        sx = 0.0
        ax = np.empty(nx, dtype=float)
        for i in range(nx):
            ax[i] = (r[i] - (i + 1.)) ** 2
        ay = np.empty(ny, dtype=float)
        for i in range(ny):
            ay[i] = (r[nx + i] - (i + 1.)) ** 2
        u = nx * float(np.sum(ax)) + ny * float(np.sum(ay))
        k = float(nx * ny)
        N = float(nx + ny)
        t = u / (k * N) - (4 * k - 1) / (6 * N)
        p = 0.0
        if meth == 'exact':
            p = _pval_cvm_2samp_exact(int(u), nx, ny)
        else:
            et = (1 + 1 / N) / 6
            vt = (N + 1) * (4 * k * N - 3 * (float(nx) ** 2 + float(ny) ** 2) - 2 * k)
            vt = vt / (45 * N ** 2 * 4 * k)
            tn = 1 / 6 + (t - et) / K.sqrt(45 * vt)
            p = max(1. - _cdf_cvm_inf(tn), 0.) if tn >= 0.003 else 1.
        return (t, p)

def cramervonmises_2samp(x, y, method: str = 'auto', *, axis = 0):
    if method != 'auto' and method != 'exact' and method != 'asymptotic':
        raise ValueError('method must be either auto, exact or asymptotic.')
    r = _map_slices2(x, y, axis, _CvM2(method))
    return CramerVonMisesResult(r[0], r[1])

# ------------------------------------------------------------------ Poisson means

def _pois_pmf(k: float, mu: float) -> float:
    if k < 0:
        return 0.0
    return K.exp(K.xlogy(k, mu) - mu - K.gammaln(k + 1))

def _pois_ppf(q: float, mu: float) -> float:
    # smallest k with cdf(k) >= q (scipy rv_discrete._ppf via pdtrik + check)
    vals = CD.pdtrik(q, mu)
    vals1 = max(K.ceil(vals) - 1, -1.)
    temp = CD.pdtr(vals1, mu) if vals1 >= 0 else 0.0
    return vals1 if temp >= q else K.ceil(vals)

def poisson_means_test(k1: int, n1: float, k2: int, n2: float, *, diff: float = 0.,
                       alternative: str = 'two-sided'):
    if k1 < 0 or k2 < 0:
        raise ValueError('`k1` and `k2` must be greater than or equal to 0.')
    if n1 <= 0 or n2 <= 0:
        raise ValueError('`n1` and `n2` must be greater than 0.')
    if diff < 0:
        raise ValueError('diff must be greater than or equal to 0.')
    lmbd_hat2 = ((k1 + k2) / (n1 + n2) - diff * n1 / (n1 + n2))
    if lmbd_hat2 <= 0:
        return SignificanceResult(0., 1.)
    var = k1 / (n1 ** 2) + k2 / (n2 ** 2)
    t_k1k2 = (k1 / n1 - k2 / n2 - diff) / K.sqrt(var)
    nl1 = n1 * (lmbd_hat2 + diff)
    nl2 = n2 * lmbd_hat2
    x1_lb = _pois_ppf(1e-10, nl1)
    x1_ub = _pois_ppf(1 - 1e-16, nl1)
    x2_lb = _pois_ppf(1e-10, nl2)
    x2_ub = _pois_ppf(1 - 1e-16, nl2)
    nx1 = int(x1_ub - x1_lb) + 1
    nx2 = int(x2_ub - x2_lb) + 1
    acc = np.zeros((nx2, nx1))
    for j in range(nx2):
        x2 = x2_lb + j
        p2 = _pois_pmf(x2, nl2)
        l2 = x2 / n2
        for i in range(nx1):
            x1 = x1_lb + i
            l1 = x1 / n1
            vv = l1 / n1 + l2 / n2
            t = (l1 - l2 - diff).cdiv(K.sqrt(vv))
            ind = False
            if alternative == 'two-sided':
                ind = abs(t) >= abs(t_k1k2)
            elif alternative == 'less':
                ind = t <= t_k1k2
            else:
                ind = t >= t_k1k2
            if ind:
                acc[j, i] = _pois_pmf(x1, nl1) * p2
    # scipy sums the boolean-masked products in row-major order
    s = 0.0
    flat = []
    for j in range(nx2):
        for i in range(nx1):
            if acc[j, i] != 0.0:
                flat.append(acc[j, i])
    return SignificanceResult(t_k1k2, float(np.sum(np.array(flat, dtype=float))))

# ------------------------------------------------------------------ Epps-Singleton

class _ES:
    t: List[float]
    def __call__(self, x, y):
        nx = x.shape[0]
        ny = y.shape[0]
        if nx < 5 or ny < 5:
            raise ValueError(f'x and y should have at least 5 elements, but len(x) = {nx} and len(y) = {ny}.')
        n = nx + ny
        for v in self.t:
            if v <= 0:
                raise ValueError('t must contain positive elements only.')
        sigma = _Iqr(0.25, 0.75, 1.0, 'linear')(np.concatenate((x, y))) / 2
        nt = len(self.t)
        gx = np.empty((nx, 2 * nt))
        gy = np.empty((ny, 2 * nt))
        for j in range(nt):
            ts = self.t[j] / sigma
            for i in range(nx):
                gx[i, j] = K.cos(ts * x[i])
                gx[i, nt + j] = K.sin(ts * x[i])
            for i in range(ny):
                gy[i, j] = K.cos(ts * y[i])
                gy[i, nt + j] = K.sin(ts * y[i])
        cov_x = np.cov(gx.T) * (nx - 1) / nx
        cov_y = np.cov(gy.T) * (ny - 1) / ny
        est_cov = (n / nx) * cov_x + (n / ny) * cov_y
        inv = np.linalg.pinv(est_cov)
        r = float(np.linalg.matrix_rank(inv))
        gd = np.mean(gx, axis=0) - np.mean(gy, axis=0)
        w = n * float(gd @ (inv @ gd))
        if max(nx, ny) < 25:
            corr = 1.0 / (1.0 + n ** (-0.45) + 10.1 * (nx ** (-1.7) + ny ** (-1.7)))
            w *= corr
        return (w, K.chdtrc(r, w))

def epps_singleton_2samp(x, y, t = (0.4, 0.8), *, axis = 0):
    tl = [float(v) for v in t]
    r = _map_slices2(x, y, axis, _ES(tl))
    return Epps_Singleton_2sampResult(r[0], r[1])
