# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Discrete distribution kernels (scipy.stats._discrete_distns).
# Kernel protocol: see _distn_infrastructure (``is_discrete`` kernels, with
# ``pdf``/``logpdf`` meaning pmf/logpmf).

import numpy as np
import scipy.special._kernels as K
import scipy.special._cdf as C
import scipy.special._gamma2 as G2
import scipy.special._noncentral as NC
import scipy.special._bessel as B
from numpy.random.legacy import RandomState
from ._distn_infrastructure import rv_discrete, rv_discrete_frozen, NAN, INF, _NA, \
    _k_cdf, _k_sf, _npsum, _drv2_ppfsingle

from C import fabs(float) -> float
from C import tanh(float) -> float
from C import cosh(float) -> float
from C import sinh(float) -> float

def _isint(x: float) -> bool: return x == K.floor(x)

def _search_ppf(Kr: type, q: float, sh, guess: float, lo: float, hi: float) -> float:
    # smallest k in [lo, hi] with cdf(k) >= q (discrete quantile, round up)
    k = min(max(K.floor(guess), lo), hi)
    while k < hi and Kr.cdf(k, sh) < q:
        k += 1.0
    while k > lo and Kr.cdf(k - 1.0, sh) >= q:
        k -= 1.0
    return k

def _search_isf(Kr: type, q: float, sh, guess: float, lo: float, hi: float) -> float:
    # smallest k in [lo, hi] with sf(k) <= q
    k = min(max(K.floor(guess), lo), hi)
    while k < hi and Kr.sf(k, sh) > q:
        k += 1.0
    while k > lo and Kr.sf(k - 1.0, sh) <= q:
        k -= 1.0
    return k

def _ibeta_derivative(a: float, b: float, x: float) -> float:
    if x == 0 or x == 1:
        return NC._ibeta_deriv_term(a, b, x, 1.0 - x)
    return K.exp(K.xlogy(a - 1, x) + K.xlog1py(b - 1, -x) - K.betaln(a, b))

# ----------------------------------------------------------------- binom

def _binom_pmf(k: float, n: float, p: float) -> float:
    # boost binomial pdf
    if p == 0:
        return 1.0 if k == 0 else 0.0
    if p == 1:
        return 1.0 if k == n else 0.0
    if n == 0:
        return 1.0
    if k == 0:
        return K.pow(1 - p, n)
    if k == n:
        return K.pow(p, k)
    return _ibeta_derivative(k + 1, n - k + 1, p) / (n + 1)

def _binom_cdf(k: float, n: float, p: float) -> float:
    if k >= n:
        return 1.0
    if k < 0:
        return 0.0
    return K.betaincc(k + 1, n - k, p)

def _binom_sf(k: float, n: float, p: float) -> float:
    if k >= n:
        return 0.0
    if k < 0:
        return 1.0
    return K.betainc(k + 1, n - k, p)

@tuple
class _BinomK:
    @staticmethod
    def nshape() -> Literal[int]: return 2
    @staticmethod
    def is_discrete() -> Literal[bool]: return True
    @staticmethod
    def check(sh) -> bool: return sh[0] >= 0 and _isint(sh[0]) and sh[1] >= 0 and sh[1] <= 1
    @staticmethod
    def b(sh) -> float: return sh[0]
    @staticmethod
    def rvs(rs, sh, size): return rs.binomial(int(sh[0]), sh[1], size)
    @staticmethod
    def logpdf(x: float, sh) -> float:
        n = sh[0]
        p = sh[1]
        k = K.floor(x)
        combiln = K.gammaln(n + 1) - (K.gammaln(k + 1) + K.gammaln(n - k + 1))
        return combiln + K.xlogy(k, p) + K.xlog1py(n - k, -p)
    @staticmethod
    def pdf(x: float, sh) -> float: return _binom_pmf(x, sh[0], sh[1])
    @staticmethod
    def cdf(x: float, sh) -> float: return _binom_cdf(K.floor(x), sh[0], sh[1])
    @staticmethod
    def sf(x: float, sh) -> float: return _binom_sf(K.floor(x), sh[0], sh[1])
    @staticmethod
    def ppf(q: float, sh) -> float:
        n = sh[0]
        p = sh[1]
        return _search_ppf(_BinomK, q, sh, n * p, 0.0, n)
    @staticmethod
    def isf(q: float, sh) -> float:
        n = sh[0]
        p = sh[1]
        return _search_isf(_BinomK, q, sh, n * p, 0.0, n)
    @staticmethod
    def stats(sh):
        n = sh[0]
        p = sh[1]
        mu = n * p
        var = mu - n * p * p
        pq = p - p * p
        npq_sqrt = K.sqrt(n * pq)
        g1 = 1.0 / npq_sqrt - (2.0 * p) / npq_sqrt
        g2 = 1.0 / (n * pq) - 6.0 / n
        return (mu, var, g1, g2)

@tuple
class _BernoulliK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def is_discrete() -> Literal[bool]: return True
    @staticmethod
    def check(sh) -> bool: return sh[0] >= 0 and sh[0] <= 1
    @staticmethod
    def b(sh) -> float: return 1.0
    @staticmethod
    def rvs(rs, sh, size): return rs.binomial(1, sh[0], size)
    @staticmethod
    def logpdf(x: float, sh) -> float: return _BinomK.logpdf(x, (1.0, sh[0]))
    @staticmethod
    def pdf(x: float, sh) -> float: return _binom_pmf(x, 1.0, sh[0])
    @staticmethod
    def cdf(x: float, sh) -> float: return _binom_cdf(K.floor(x), 1.0, sh[0])
    @staticmethod
    def sf(x: float, sh) -> float: return _binom_sf(K.floor(x), 1.0, sh[0])
    @staticmethod
    def ppf(q: float, sh) -> float: return _BinomK.ppf(q, (1.0, sh[0]))
    @staticmethod
    def isf(q: float, sh) -> float: return _BinomK.isf(q, (1.0, sh[0]))
    @staticmethod
    def stats(sh): return _BinomK.stats((1.0, sh[0]))
    @staticmethod
    def entropy(sh) -> float: return K.entr(sh[0]) + K.entr(1 - sh[0])

# ------------------------------------------------------------- betabinom

@tuple
class _BetabinomK:
    @staticmethod
    def nshape() -> Literal[int]: return 3
    @staticmethod
    def is_discrete() -> Literal[bool]: return True
    @staticmethod
    def check(sh) -> bool: return sh[0] >= 0 and _isint(sh[0]) and sh[1] > 0 and sh[2] > 0
    @staticmethod
    def b(sh) -> float: return sh[0]
    @staticmethod
    def rvs(rs, sh, size):
        p = rs.beta(sh[1], sh[2], size)
        return rs.binomial(int(sh[0]), p, size)
    @staticmethod
    def logpdf(x: float, sh) -> float:
        n = sh[0]
        a = sh[1]
        b = sh[2]
        k = K.floor(x)
        combiln = -K.log(n + 1) - K.betaln(n - k + 1, k + 1)
        return combiln + K.betaln(k + a, n - k + b) - K.betaln(a, b)
    @staticmethod
    def pdf(x: float, sh) -> float: return K.exp(_BetabinomK.logpdf(x, sh))
    @staticmethod
    def stats(sh):
        n = sh[0]
        a = sh[1]
        b = sh[2]
        e_p = a / (a + b)
        e_q = 1 - e_p
        mu = n * e_p
        var = n * (a + b + n) * e_p * e_q / (a + b + 1)
        g1 = 1.0 / K.sqrt(var)
        g1 *= (a + b + 2 * n) * (b - a)
        g1 /= (a + b + 2) * (a + b)
        g2 = a + b
        g2 *= (a + b - 1 + 6 * n)
        g2 += 3 * a * b * (n - 2)
        g2 += 6 * n ** 2
        g2 -= 3 * e_p * b * n * (6 - n)
        g2 -= 18 * e_p * e_q * n ** 2
        g2 *= (a + b) ** 2 * (1 + a + b)
        g2 /= (n * a * b * (a + b + 2) * (a + b + 3) * (a + b + n))
        g2 -= 3
        return (mu, var, g1, g2)

# ----------------------------------------------------------------- nbinom

@tuple
class _NbinomK:
    @staticmethod
    def nshape() -> Literal[int]: return 2
    @staticmethod
    def is_discrete() -> Literal[bool]: return True
    @staticmethod
    def check(sh) -> bool: return sh[0] > 0 and sh[1] > 0 and sh[1] <= 1
    @staticmethod
    def rvs(rs, sh, size): return rs.negative_binomial(sh[0], sh[1], size)
    @staticmethod
    def pdf(x: float, sh) -> float:
        # boost negative_binomial pdf
        r = sh[0]
        p = sh[1]
        return (p / (r + x)) * _ibeta_derivative(r, x + 1, p)
    @staticmethod
    def logpdf(x: float, sh) -> float:
        n = sh[0]
        p = sh[1]
        coeff = K.gammaln(n + x) - K.gammaln(x + 1) - K.gammaln(n)
        return coeff + n * K.log(p) + K.xlog1py(x, -p)
    @staticmethod
    def cdf(x: float, sh) -> float: return K.betainc(sh[0], K.floor(x) + 1, sh[1])
    @staticmethod
    def logcdf(x: float, sh) -> float:
        k = K.floor(x)
        c = _NbinomK.cdf(k, sh)
        if c > 0.5:
            return K.log1p(-K.betainc(k + 1, sh[0], 1 - sh[1]))
        return K.log(c)
    @staticmethod
    def sf(x: float, sh) -> float: return K.betaincc(sh[0], K.floor(x) + 1, sh[1])
    @staticmethod
    def ppf(q: float, sh) -> float:
        r = sh[0]
        p = sh[1]
        return _search_ppf(_NbinomK, q, sh, r * (1 - p) / p, 0.0, INF)
    @staticmethod
    def isf(q: float, sh) -> float:
        r = sh[0]
        p = sh[1]
        return _search_isf(_NbinomK, q, sh, r * (1 - p) / p, 0.0, INF)
    @staticmethod
    def stats(sh):
        r = sh[0]
        p = sh[1]
        return (r * (1 - p) / p, r * (1 - p) / (p * p), (2 - p) / K.sqrt(r * (1 - p)),
                (6 - p * (6 - p)) / (r * (1 - p)))

@tuple
class _BetanbinomK:
    @staticmethod
    def nshape() -> Literal[int]: return 3
    @staticmethod
    def is_discrete() -> Literal[bool]: return True
    @staticmethod
    def check(sh) -> bool: return sh[0] >= 0 and _isint(sh[0]) and sh[1] > 0 and sh[2] > 0
    @staticmethod
    def rvs(rs, sh, size):
        p = rs.beta(sh[1], sh[2], size)
        return rs.negative_binomial(sh[0], p, size)
    @staticmethod
    def logpdf(x: float, sh) -> float:
        n = sh[0]
        a = sh[1]
        b = sh[2]
        k = K.floor(x)
        combiln = -K.log(n + k) - K.betaln(n, k + 1)
        return combiln + K.betaln(a + n, b + k) - K.betaln(a, b)
    @staticmethod
    def pdf(x: float, sh) -> float: return K.exp(_BetanbinomK.logpdf(x, sh))
    @staticmethod
    def stats(sh):
        n = sh[0]
        a = sh[1]
        b = sh[2]
        mu = n * b / (a - 1.) if a > 1 else INF
        var = (n * b * (n + a - 1.) * (a + b - 1.) / ((a - 2.) * (a - 1.) ** 2.)) if a > 2 else INF
        g1 = INF
        if a > 3:
            g1 = ((2 * n + a - 1.) * (2 * b + a - 1.)
                  / (a - 3.) / K.sqrt(n * b * (n + a - 1.) * (b + a - 1.) / (a - 2.)))
        g2 = INF
        if a > 4:
            term = (a - 2.)
            term_2 = ((a - 1.) ** 2. * (a ** 2. + a * (6 * b - 1.) + 6. * (b - 1.) * b)
                      + 3. * n ** 2. * ((a + 5.) * b ** 2. + (a + 5.) * (a - 1.) * b
                                        + 2. * (a - 1.) ** 2)
                      + 3 * (a - 1.) * n * ((a + 5.) * b ** 2. + (a + 5.) * (a - 1.) * b
                                            + 2. * (a - 1.) ** 2.))
            den = ((a - 4.) * (a - 3.) * b * n * (a + b - 1.) * (a + n - 1.))
            g2 = term * term_2 / den - 3.
        return (mu, var, g1, g2)

# ------------------------------------------------------------------- geom

@tuple
class _GeomK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def is_discrete() -> Literal[bool]: return True
    @staticmethod
    def check(sh) -> bool: return sh[0] <= 1 and sh[0] > 0
    @staticmethod
    def a(sh) -> float: return 1.0
    @staticmethod
    def rvs(rs, sh, size): return rs.geometric(sh[0], size=size)
    @staticmethod
    def pdf(k: float, sh) -> float: return K.pow(1 - sh[0], k - 1) * sh[0]
    @staticmethod
    def logpdf(k: float, sh) -> float: return K.xlog1py(k - 1, -sh[0]) + K.log(sh[0])
    @staticmethod
    def cdf(x: float, sh) -> float: return -K.expm1(K.log1p(-sh[0]) * K.floor(x))
    @staticmethod
    def logsf(x: float, sh) -> float: return K.floor(x) * K.log1p(-sh[0])
    @staticmethod
    def sf(x: float, sh) -> float: return K.exp(_GeomK.logsf(x, sh))
    @staticmethod
    def ppf(q: float, sh) -> float:
        p = sh[0]
        vals = K.ceil(K.log1p(-q) / K.log1p(-p))
        temp = _GeomK.cdf(vals - 1, sh)
        return vals - 1 if (temp >= q and vals > 0) else vals
    @staticmethod
    def stats(sh):
        p = sh[0]
        qr = 1.0 - p
        return (1.0 / p, qr / p / p, (2.0 - p) / K.sqrt(qr), ((p - 6) * p + 6) / (1.0 - p))
    @staticmethod
    def entropy(sh) -> float:
        p = sh[0]
        return -K.log(p) - K.log1p(-p) * (1.0 - p) / p

# -------------------------------------------------------------- hypergeom

def _hyp_logpmf(k: float, M: float, n: float, N: float) -> float:
    tot = M
    good = n
    bad = tot - good
    return (K.betaln(good + 1, 1) + K.betaln(bad + 1, 1) + K.betaln(tot - N + 1, N + 1)
            - K.betaln(k + 1, good - k + 1) - K.betaln(N - k + 1, bad - N + k + 1)
            - K.betaln(tot + 1, 1))

def _hyp_pmf(k: float, M: float, n: float, N: float) -> float:
    lo = max(N - (M - n), 0.0)
    hi = min(n, N)
    if k < lo or k > hi:
        return 0.0
    return K.exp(_hyp_logpmf(k, M, n, N))

@tuple
class _HypergeomK:
    @staticmethod
    def nshape() -> Literal[int]: return 3
    @staticmethod
    def is_discrete() -> Literal[bool]: return True
    @staticmethod
    def check(sh) -> bool:
        M = sh[0]
        n = sh[1]
        N = sh[2]
        return (M > 0 and n >= 0 and N >= 0 and n <= M and N <= M
                and _isint(M) and _isint(n) and _isint(N))
    @staticmethod
    def a(sh) -> float: return max(sh[2] - (sh[0] - sh[1]), 0.0)
    @staticmethod
    def b(sh) -> float: return min(sh[1], sh[2])
    @staticmethod
    def rvs(rs, sh, size):
        return rs.hypergeometric(int(sh[1]), int(sh[0] - sh[1]), int(sh[2]), size=size)
    @staticmethod
    def logpdf(k: float, sh) -> float: return _hyp_logpmf(k, sh[0], sh[1], sh[2])
    @staticmethod
    def pdf(k: float, sh) -> float: return _hyp_pmf(k, sh[0], sh[1], sh[2])
    @staticmethod
    def cdf(x: float, sh) -> float:
        k = K.floor(x)
        lo = _HypergeomK.a(sh)
        hi = _HypergeomK.b(sh)
        if k >= hi:
            return 1.0
        mode = K.floor((sh[2] + 1) * (sh[1] + 1) / (sh[0] + 2))
        if k <= mode:
            s = 0.0
            j = k
            while j >= lo:
                s += _hyp_pmf(j, sh[0], sh[1], sh[2])
                j -= 1
            return s
        s = 0.0
        j = k + 1
        while j <= hi:
            s += _hyp_pmf(j, sh[0], sh[1], sh[2])
            j += 1
        return 1.0 - s
    @staticmethod
    def sf(x: float, sh) -> float:
        k = K.floor(x)
        lo = _HypergeomK.a(sh)
        hi = _HypergeomK.b(sh)
        if k >= hi:
            return 0.0
        mode = K.floor((sh[2] + 1) * (sh[1] + 1) / (sh[0] + 2))
        if k >= mode:
            s = 0.0
            j = k + 1
            while j <= hi:
                s += _hyp_pmf(j, sh[0], sh[1], sh[2])
                j += 1
            return s
        s = 0.0
        j = k
        while j >= lo:
            s += _hyp_pmf(j, sh[0], sh[1], sh[2])
            j -= 1
        return 1.0 - s
    @staticmethod
    def stats(sh):
        M = sh[0]
        n = sh[1]
        N = sh[2]
        m = M - n
        g2 = M * (M + 1) - 6. * N * (M - N) - 6. * n * m
        g2 *= (M - 1) * M * M
        g2 += 6. * n * N * (M - N) * m * (5. * M - 6)
        g2 /= n * N * (M - N) * m * (M - 2.) * (M - 3.)
        mean = N * n / M
        var = N * n * (M - n) * (M - N) / (M * M * (M - 1))
        skew = ((M - 2 * N) * K.sqrt(M - 1) * (M - 2 * n)
                / (K.sqrt(N * n * (M - n) * (M - N)) * (M - 2)))
        return (mean, var, skew, g2)
    @staticmethod
    def entropy(sh) -> float:
        M = sh[0]
        n = sh[1]
        N = sh[2]
        v = List[float]()
        k = N - (M - n)
        while k <= min(n, N):
            v.append(K.entr(_hyp_pmf(k, M, n, N)))
            k += 1
        return _npsum(v)

@tuple
class _NhypergeomK:
    @staticmethod
    def nshape() -> Literal[int]: return 3
    @staticmethod
    def is_discrete() -> Literal[bool]: return True
    @staticmethod
    def check(sh) -> bool:
        M = sh[0]
        n = sh[1]
        r = sh[2]
        return (n >= 0 and n <= M and r >= 0 and r <= M - n
                and _isint(M) and _isint(n) and _isint(r))
    @staticmethod
    def b(sh) -> float: return sh[1]
    @staticmethod
    def rvs(rs, sh, size):
        ks = List[float]()
        cs = List[float]()
        k = 0.0
        while k <= sh[1]:
            ks.append(k)
            cs.append(_k_cdf(_NhypergeomK, k, sh))
            k += 1
        u = rs.uniform(size=size)
        out = np.empty(u.shape, dtype=int)
        for idx in np.ndindex(u.shape):
            x = float(u[idx])
            j = 0
            while j < len(cs) - 1 and cs[j] < x:
                j += 1
            out[idx] = int(ks[j])
        return out
    @staticmethod
    def logpdf(k: float, sh) -> float:
        M = sh[0]
        n = sh[1]
        r = sh[2]
        if r == 0 and k == 0:
            return 0.0
        return (-K.betaln(k + 1, r) + K.betaln(k + r, 1)
                - K.betaln(n - k + 1, M - r - n + 1) + K.betaln(M - r - k + 1, 1)
                + K.betaln(n + 1, M - n + 1) - K.betaln(M + 1, 1))
    @staticmethod
    def pdf(k: float, sh) -> float: return K.exp(_NhypergeomK.logpdf(k, sh))
    @staticmethod
    def stats(sh):
        M = sh[0]
        n = sh[1]
        r = sh[2]
        mu = r * n / (M - n + 1)
        var = r * (M + 1) * n / ((M - n + 1) * (M - n + 2)) * (1 - r / (M - n + 1))
        return (mu, var, _NA, _NA)

# ----------------------------------------------------------------- logser

@tuple
class _LogserK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def is_discrete() -> Literal[bool]: return True
    @staticmethod
    def check(sh) -> bool: return sh[0] > 0 and sh[0] < 1
    @staticmethod
    def a(sh) -> float: return 1.0
    @staticmethod
    def rvs(rs, sh, size): return rs.logseries(sh[0], size=size)
    @staticmethod
    def pdf(k: float, sh) -> float:
        p = sh[0]
        return -K.pow(p, k) * 1.0 / k / K.log1p(-p)
    @staticmethod
    def sf(k: float, sh) -> float:
        p = sh[0]
        tiny = 1e-100
        return -K.betainc(k + 1, tiny, p) * K.beta(k + 1, tiny) / K.log1p(-p)
    @staticmethod
    def stats(sh):
        p = sh[0]
        r = K.log1p(-p)
        mu = p / (p - 1.0) / r
        mu2p = -p / r / (p - 1.0) ** 2
        var = mu2p - mu * mu
        mu3p = -p / r * (1.0 + p) / (1.0 - p) ** 3
        mu3 = mu3p - 3 * mu * mu2p + 2 * mu ** 3
        g1 = mu3 / K.pow(var, 1.5)
        mu4p = -p / r * (1.0 / (p - 1) ** 2 - 6 * p / (p - 1) ** 3 + 6 * p * p / (p - 1) ** 4)
        mu4 = mu4p - 4 * mu3p * mu + 6 * mu2p * mu * mu - 3 * mu ** 4
        g2 = mu4 / var ** 2 - 3.0
        return (mu, var, g1, g2)

# ---------------------------------------------------------------- poisson

@tuple
class _PoissonK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def is_discrete() -> Literal[bool]: return True
    @staticmethod
    def check(sh) -> bool: return sh[0] >= 0
    @staticmethod
    def rvs(rs, sh, size): return rs.poisson(sh[0], size)
    @staticmethod
    def logpdf(k: float, sh) -> float:
        mu = sh[0]
        return K.xlogy(k, mu) - K.gammaln(k + 1) - mu
    @staticmethod
    def pdf(k: float, sh) -> float: return K.exp(_PoissonK.logpdf(k, sh))
    @staticmethod
    def cdf(x: float, sh) -> float: return C.pdtr(K.floor(x), sh[0])
    @staticmethod
    def sf(x: float, sh) -> float: return C.pdtrc(K.floor(x), sh[0])
    @staticmethod
    def ppf(q: float, sh) -> float:
        mu = sh[0]
        vals = K.ceil(C.pdtrik(q, mu))
        vals1 = max(vals - 1, 0.0)
        temp = C.pdtr(vals1, mu)
        return vals1 if temp >= q else vals
    @staticmethod
    def stats(sh):
        mu = sh[0]
        return (mu, mu, K.sqrt(1.0 / mu) if mu > 0 else INF, 1.0 / mu if mu > 0 else INF)

# ----------------------------------------------------------------- planck

@tuple
class _PlanckK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def is_discrete() -> Literal[bool]: return True
    @staticmethod
    def rvs(rs, sh, size):
        p = -K.expm1(-sh[0])
        return rs.geometric(p, size=size) - 1.0
    @staticmethod
    def pdf(k: float, sh) -> float:
        l = sh[0]
        return -K.expm1(-l) * K.exp(-l * k)
    @staticmethod
    def cdf(x: float, sh) -> float: return -K.expm1(-sh[0] * (K.floor(x) + 1))
    @staticmethod
    def logsf(x: float, sh) -> float: return -sh[0] * (K.floor(x) + 1)
    @staticmethod
    def sf(x: float, sh) -> float: return K.exp(_PlanckK.logsf(x, sh))
    @staticmethod
    def ppf(q: float, sh) -> float:
        l = sh[0]
        vals = K.ceil(-1.0 / l * K.log1p(-q) - 1)
        vals1 = max(vals - 1, 0.0)
        temp = _PlanckK.cdf(vals1, sh)
        return vals1 if temp >= q else vals
    @staticmethod
    def stats(sh):
        l = sh[0]
        return (1 / K.expm1(l), K.exp(-l) / (K.expm1(-l)) ** 2, 2 * cosh(l / 2.0),
                4 + 2 * cosh(l))
    @staticmethod
    def entropy(sh) -> float:
        l = sh[0]
        C_ = -K.expm1(-l)
        return l * K.exp(-l) / C_ - K.log(C_)

# -------------------------------------------------------------- boltzmann

@tuple
class _BoltzmannK:
    @staticmethod
    def nshape() -> Literal[int]: return 2
    @staticmethod
    def is_discrete() -> Literal[bool]: return True
    @staticmethod
    def check(sh) -> bool: return sh[0] > 0 and sh[1] > 0 and _isint(sh[1])
    @staticmethod
    def b(sh) -> float: return sh[1] - 1
    @staticmethod
    def pdf(k: float, sh) -> float:
        l = sh[0]
        N = sh[1]
        fact = (1 - K.exp(-l)) / (1 - K.exp(-l * N))
        return fact * K.exp(-l * k)
    @staticmethod
    def cdf(x: float, sh) -> float:
        l = sh[0]
        N = sh[1]
        return (1 - K.exp(-l * (K.floor(x) + 1))) / (1 - K.exp(-l * N))
    @staticmethod
    def ppf(q: float, sh) -> float:
        l = sh[0]
        N = sh[1]
        qnew = q * (1 - K.exp(-l * N))
        vals = K.ceil(-1.0 / l * K.log(1 - qnew) - 1)
        vals1 = max(vals - 1, 0.0)
        temp = _BoltzmannK.cdf(vals1, sh)
        return vals1 if temp >= q else vals
    @staticmethod
    def stats(sh):
        l = sh[0]
        N = sh[1]
        z = K.exp(-l)
        zN = K.exp(-l * N)
        mu = z / (1.0 - z) - N * zN / (1 - zN)
        var = z / (1.0 - z) ** 2 - N * N * zN / (1 - zN) ** 2
        trm = (1 - zN) / (1 - z)
        trm2 = (z * trm ** 2 - N * N * zN)
        g1 = z * (1 + z) * trm ** 3 - N ** 3 * zN * (1 + zN)
        g1 = g1 / trm2 ** (1.5)
        g2 = z * (1 + 4 * z + z * z) * trm ** 4 - N ** 4 * zN * (1 + 4 * zN + zN * zN)
        g2 = g2 / trm2 / trm2
        return (mu, var, g1, g2)

# ---------------------------------------------------------------- randint

@tuple
class _RandintK:
    @staticmethod
    def nshape() -> Literal[int]: return 2
    @staticmethod
    def is_discrete() -> Literal[bool]: return True
    @staticmethod
    def check(sh) -> bool: return sh[1] > sh[0] and _isint(sh[0]) and _isint(sh[1])
    @staticmethod
    def a(sh) -> float: return sh[0]
    @staticmethod
    def b(sh) -> float: return sh[1] - 1
    @staticmethod
    def rvs(rs, sh, size):
        if isinstance(rs, RandomState):
            return rs.randint(int(sh[0]), int(sh[1]), size)
        else:
            return rs.integers(int(sh[0]), int(sh[1]), size=size)
    @staticmethod
    def pdf(k: float, sh) -> float:
        low = sh[0]
        high = sh[1]
        return 1.0 / (high - low) if (k >= low and k < high) else 0.0
    @staticmethod
    def cdf(x: float, sh) -> float: return (K.floor(x) - sh[0] + 1.) / (sh[1] - sh[0])
    @staticmethod
    def ppf(q: float, sh) -> float:
        low = sh[0]
        high = sh[1]
        vals = K.ceil(q * (high - low) + low) - 1
        vals1 = min(max(vals - 1, low), high)
        temp = _RandintK.cdf(vals1, sh)
        return vals1 if temp >= q else vals
    @staticmethod
    def stats(sh):
        m2 = sh[1]
        m1 = sh[0]
        d = m2 - m1
        return ((m2 + m1 - 1.0) / 2, (d * d - 1) / 12.0, 0.0,
                -6.0 / 5.0 * (d * d + 1.0) / (d * d - 1.0))
    @staticmethod
    def entropy(sh) -> float: return K.log(sh[1] - sh[0])

# ------------------------------------------------------------ zipf/zipfian

@tuple
class _ZipfK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def is_discrete() -> Literal[bool]: return True
    @staticmethod
    def check(sh) -> bool: return sh[0] > 1
    @staticmethod
    def a(sh) -> float: return 1.0
    @staticmethod
    def rvs(rs, sh, size): return rs.zipf(sh[0], size=size)
    @staticmethod
    def pdf(k: float, sh) -> float:
        a = sh[0]
        return 1.0 / G2.zeta(a, 1.0) * K.pow(k, -a)
    @staticmethod
    def munp(n: float, sh) -> float:
        a = sh[0]
        if a > n + 1:
            return G2.zeta(a - n, 1.0) / G2.zeta(a, 1.0)
        return INF

def _gen_harmonic(n: float, a: float) -> float:
    s = 0.0
    k = n
    while k >= 1:
        s += K.pow(k, -a)
        k -= 1
    return s

def _gen_harmonic_range(j: float, k: float, a: float) -> float:
    s = 0.0
    i = k
    while i >= j:
        s += K.pow(i, -a)
        i -= 1
    return s

@tuple
class _ZipfianK:
    @staticmethod
    def nshape() -> Literal[int]: return 2
    @staticmethod
    def is_discrete() -> Literal[bool]: return True
    @staticmethod
    def check(sh) -> bool:
        return sh[0] >= 0 and sh[1] >= 1 and sh[1] <= 2147483647.0 and _isint(sh[1])
    @staticmethod
    def a(sh) -> float: return 1.0
    @staticmethod
    def b(sh) -> float: return K.floor(sh[1])
    @staticmethod
    def pdf(k: float, sh) -> float:
        k = K.floor(k)
        n = K.floor(sh[1])
        return K.pow(k, -sh[0]) / _gen_harmonic(n, sh[0])
    @staticmethod
    def cdf(k: float, sh) -> float:
        k = K.floor(k)
        n = K.floor(sh[1])
        return _gen_harmonic_range(1.0, k, sh[0]) / _gen_harmonic(n, sh[0])
    @staticmethod
    def sf(k: float, sh) -> float:
        k = K.floor(k)
        n = K.floor(sh[1])
        return _gen_harmonic_range(k + 1, n, sh[0]) / _gen_harmonic(n, sh[0])
    @staticmethod
    def stats(sh):
        a = sh[0]
        n = K.floor(sh[1])
        Hna = _gen_harmonic(n, a)
        Hna1 = _gen_harmonic(n, a - 1)
        Hna2 = _gen_harmonic(n, a - 2)
        Hna3 = _gen_harmonic(n, a - 3)
        Hna4 = _gen_harmonic(n, a - 4)
        mu1 = Hna1 / Hna
        mu2n = (Hna2 * Hna - Hna1 ** 2)
        mu2d = Hna ** 2
        mu2 = mu2n / mu2d
        g1 = (Hna3 / Hna - 3 * Hna1 * Hna2 / Hna ** 2 + 2 * Hna1 ** 3 / Hna ** 3) / mu2 ** (3 / 2)
        g2 = (Hna ** 3 * Hna4 - 4 * Hna ** 2 * Hna1 * Hna3 + 6 * Hna * Hna1 ** 2 * Hna2
              - 3 * Hna1 ** 4) / mu2n ** 2
        g2 -= 3
        return (mu1, mu2, g1, g2)

# --------------------------------------------------------------- dlaplace

@tuple
class _DlaplaceK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def is_discrete() -> Literal[bool]: return True
    @staticmethod
    def a(sh) -> float: return -INF
    @staticmethod
    def rvs(rs, sh, size):
        p = -K.expm1(-sh[0])
        x = rs.geometric(p, size=size)
        y = rs.geometric(p, size=size)
        return x - y
    @staticmethod
    def pdf(k: float, sh) -> float:
        a = sh[0]
        return tanh(a / 2.0) * K.exp(-a * fabs(k))
    @staticmethod
    def cdf(x: float, sh) -> float:
        a = sh[0]
        k = K.floor(x)
        if k >= 0:
            return 1.0 - K.exp(-a * k) / (K.exp(a) + 1)
        return K.exp(a * (k + 1)) / (K.exp(a) + 1)
    @staticmethod
    def ppf(q: float, sh) -> float:
        a = sh[0]
        const = 1 + K.exp(a)
        vals = K.ceil(K.log(q * const) / a - 1 if q < 1.0 / (1 + K.exp(-a))
                      else -K.log((1 - q) * const) / a)
        vals1 = vals - 1
        return vals1 if _DlaplaceK.cdf(vals1, sh) >= q else vals
    @staticmethod
    def stats(sh):
        ea = K.exp(sh[0])
        mu2 = 2. * ea / (ea - 1.) ** 2
        mu4 = 2. * ea * (ea ** 2 + 10. * ea + 1.) / (ea - 1.) ** 4
        return (0., mu2, 0., mu4 / mu2 ** 2 - 3.)
    @staticmethod
    def entropy(sh) -> float:
        a = sh[0]
        return a / sinh(a) - K.log(tanh(a / 2.0))

# ---------------------------------------------------------------- skellam

@tuple
class _SkellamK:
    @staticmethod
    def nshape() -> Literal[int]: return 2
    @staticmethod
    def is_discrete() -> Literal[bool]: return True
    @staticmethod
    def a(sh) -> float: return -INF
    @staticmethod
    def rvs(rs, sh, size): return rs.poisson(sh[0], size) - rs.poisson(sh[1], size)
    @staticmethod
    def pdf(x: float, sh) -> float:
        mu1 = sh[0]
        mu2 = sh[1]
        # exp(-(mu1+mu2)) (mu1/mu2)^(k/2) I_|k|(2 sqrt(mu1 mu2))
        z = 2 * K.sqrt(mu1 * mu2)
        d = K.sqrt(mu1) - K.sqrt(mu2)
        return K.exp(-d * d + x / 2 * K.log(mu1 / mu2)) * B.ive(fabs(x), z)
    @staticmethod
    def cdf(x: float, sh) -> float:
        mu1 = sh[0]
        mu2 = sh[1]
        x = K.floor(x)
        if x < 0:
            return NC.chndtr(2 * mu2, -2 * x, 2 * mu1)
        return NC.chndtrc(2 * mu1, 2 * (x + 1), 2 * mu2)
    @staticmethod
    def stats(sh):
        mean = sh[0] - sh[1]
        var = sh[0] + sh[1]
        return (mean, var, mean / K.sqrt(var ** 3), 1 / var)

# -------------------------------------------------------------- yulesimon

@tuple
class _YulesimonK:
    @staticmethod
    def nshape() -> Literal[int]: return 1
    @staticmethod
    def is_discrete() -> Literal[bool]: return True
    @staticmethod
    def a(sh) -> float: return 1.0
    @staticmethod
    def rvs(rs, sh, size):
        E1 = rs.standard_exponential(size)
        E2 = rs.standard_exponential(size)
        return np.ceil(-E1 / np.log1p(-np.exp(-E2 / sh[0])))
    @staticmethod
    def pdf(x: float, sh) -> float: return sh[0] * K.beta(x, sh[0] + 1)
    @staticmethod
    def logpdf(x: float, sh) -> float: return K.log(sh[0]) + K.betaln(x, sh[0] + 1)
    @staticmethod
    def cdf(x: float, sh) -> float: return 1 - x * K.beta(x, sh[0] + 1)
    @staticmethod
    def sf(x: float, sh) -> float: return x * K.beta(x, sh[0] + 1)
    @staticmethod
    def logsf(x: float, sh) -> float: return K.log(x) + K.betaln(x, sh[0] + 1)
    @staticmethod
    def stats(sh):
        alpha = sh[0]
        mu = INF if alpha <= 1 else alpha / (alpha - 1)
        mu2 = alpha ** 2 / ((alpha - 2.0) * (alpha - 1) ** 2) if alpha > 2 else INF
        if alpha <= 1:
            mu2 = NAN
        g1 = K.sqrt(alpha - 2) * (alpha + 1) ** 2 / (alpha * (alpha - 3)) if alpha > 3 else INF
        if alpha <= 2:
            g1 = NAN
        g2 = (alpha + 3 + ((11 * alpha ** 3 - 49 * alpha - 22) / (alpha * (alpha - 4) * (alpha - 3)))
              if alpha > 4 else INF)
        if alpha <= 2:
            g2 = NAN
        return (mu, mu2, g1, g2)

binom = rv_discrete[_BinomK]('binom')
bernoulli = rv_discrete[_BernoulliK]('bernoulli')
betabinom = rv_discrete[_BetabinomK]('betabinom')
nbinom = rv_discrete[_NbinomK]('nbinom')
betanbinom = rv_discrete[_BetanbinomK]('betanbinom')
geom = rv_discrete[_GeomK]('geom')
hypergeom = rv_discrete[_HypergeomK]('hypergeom')
nhypergeom = rv_discrete[_NhypergeomK]('nhypergeom')
logser = rv_discrete[_LogserK]('logser')
poisson = rv_discrete[_PoissonK]('poisson')
planck = rv_discrete[_PlanckK]('planck')
boltzmann = rv_discrete[_BoltzmannK]('boltzmann')
randint = rv_discrete[_RandintK]('randint')
zipf = rv_discrete[_ZipfK]('zipf')
zipfian = rv_discrete[_ZipfianK]('zipfian')
dlaplace = rv_discrete[_DlaplaceK]('dlaplace')
skellam = rv_discrete[_SkellamK]('skellam')
yulesimon = rv_discrete[_YulesimonK]('yulesimon')
