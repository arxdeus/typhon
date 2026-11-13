# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Discrete distributions, part 2 (scipy.stats._discrete_distns):
# nchypergeom_fisher, nchypergeom_wallenius, poisson_binom.

import numpy as np
import internal.static as static
import scipy.special._kernels as K
from ._distn_infrastructure import rv_discrete, check_random_state, NAN, INF, _NA
from ._biasedurn import FishersNCHypergeometric, WalleniusNCHypergeometric, StochasticLib3
from ._continuous_distns5 import _nsamples, _shape_out

def _isint(x: float) -> bool: return x == K.floor(x)

def _nch_check(sh) -> bool:
    M = sh[0]
    n = sh[1]
    N = sh[2]
    odds = sh[3]
    return (M == M and n == n and N == N and _isint(M) and M >= 0 and _isint(n) and n >= 0
            and _isint(N) and N >= 0 and odds > 0 and N <= M and n <= M)

# ------------------------------------------------- nchypergeom_fisher

@tuple
class _NchFisherK:
    @staticmethod
    def nshape() -> Literal[int]: return 4
    @staticmethod
    def is_discrete() -> Literal[bool]: return True
    @staticmethod
    def check(sh) -> bool: return _nch_check(sh)
    @staticmethod
    def a(sh) -> float: return max(0.0, sh[2] - (sh[0] - sh[1]))
    @staticmethod
    def b(sh) -> float: return min(sh[2], sh[1])
    @staticmethod
    def rvs(rs, sh, size):
        M = int(sh[0])
        n = int(sh[1])
        N = int(sh[2])
        odds = sh[3]
        length = _nsamples(size)
        urn = StochasticLib3(rs)
        out = np.empty(length, dtype=float)
        for i in range(length):
            out[i] = float(urn.FishersNCHyp(N, n, M, odds))
        return _shape_out(out, size)
    @staticmethod
    def pdf(x: float, sh) -> float:
        urn = FishersNCHypergeometric(int(sh[2]), int(sh[1]), int(sh[0]), sh[3], 1e-12)
        return urn.probability(int(x))
    @staticmethod
    def stats(sh):
        urn = FishersNCHypergeometric(int(sh[2]), int(sh[1]), int(sh[0]), sh[3], 1e-12)
        r = urn.moments()
        return (r[0], r[1], _NA, _NA)

# ----------------------------------------------- nchypergeom_wallenius

@tuple
class _NchWalleniusK:
    @staticmethod
    def nshape() -> Literal[int]: return 4
    @staticmethod
    def is_discrete() -> Literal[bool]: return True
    @staticmethod
    def check(sh) -> bool: return _nch_check(sh)
    @staticmethod
    def a(sh) -> float: return max(0.0, sh[2] - (sh[0] - sh[1]))
    @staticmethod
    def b(sh) -> float: return min(sh[2], sh[1])
    @staticmethod
    def rvs(rs, sh, size):
        M = int(sh[0])
        n = int(sh[1])
        N = int(sh[2])
        odds = sh[3]
        length = _nsamples(size)
        urn = StochasticLib3(rs)
        out = np.empty(length, dtype=float)
        for i in range(length):
            out[i] = float(urn.WalleniusNCHyp(N, n, M, odds))
        return _shape_out(out, size)
    @staticmethod
    def pdf(x: float, sh) -> float:
        urn = WalleniusNCHypergeometric(int(sh[2]), int(sh[1]), int(sh[0]), sh[3], 1e-12)
        return urn.probability(int(x))
    @staticmethod
    def stats(sh):
        urn = WalleniusNCHypergeometric(int(sh[2]), int(sh[1]), int(sh[0]), sh[3], 1e-12)
        r = urn.moments()
        return (r[0], r[1], _NA, _NA)

nchypergeom_fisher = rv_discrete[_NchFisherK]('nchypergeom_fisher')
nchypergeom_wallenius = rv_discrete[_NchWalleniusK]('nchypergeom_wallenius')

# ----------------------------------------------------- poisson_binom

def _pb_pmf_all(p: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
    # xsf poisson_binom_pmf_all
    n = p.shape[0]
    res = np.zeros(n + 1, dtype=float)
    if n == 0:
        res[0] = 1.0
        return res
    res[0] = 1.0 - p[0]
    res[1] = p[0]
    for i in range(1, n):
        p_i = p[i]
        q_i = 1.0 - p_i
        j = i + 1
        while j >= 1:
            tmp = res[j - 1] * p_i
            res[j - 1] *= q_i
            res[j] += tmp
            j -= 1
    return res

def _pb_cdf_all(p: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
    n = p.shape[0]
    res = _pb_pmf_all(p)
    for i in range(1, n):
        res[i] = min(res[i] + res[i - 1], 1.0)
    res[n] = 1.0
    return res

def _pb_pmf1(res: np.ndarray[float, 1], k: float) -> float:
    if k != k:
        return NAN
    ki = int(k)
    if ki < 0 or ki >= res.shape[0]:
        return 0.0
    return res[ki]

def _pb_cdf1(res: np.ndarray[float, 1], k: float) -> float:
    if k != k:
        return NAN
    ki = int(K.floor(k))
    if ki < 0:
        return 0.0
    if ki >= res.shape[0] - 1:
        return 1.0
    return res[ki]

def _pb_map(f, k):
    if isinstance(k, np.ndarray) or isinstance(k, List) or isinstance(k, Tuple):
        a = np.asarray(k).astype(float)
        out = np.empty(a.shape, dtype=float)
        for idx in np.ndindex(a.shape):
            out[idx] = f(float(a[idx]))
        return out
    else:
        return f(float(k))

class _PbPmf:
    res: np.ndarray[float, 1]
    loc: float
    def __init__(self, res: np.ndarray[float, 1], loc: float):
        self.res = res
        self.loc = loc
    def __call__(self, k: float) -> float:
        kk = k - self.loc
        if K.floor(kk) != kk:
            return 0.0
        return _pb_pmf1(self.res, kk)

class _PbLogPmf(_PbPmf):
    def __call__(self, k: float) -> float:
        return K.log(_PbPmf.__call__(self, k))

class _PbCdf:
    res: np.ndarray[float, 1]
    loc: float
    kind: int
    def __init__(self, res: np.ndarray[float, 1], loc: float, kind: int):
        self.res = res
        self.loc = loc
        self.kind = kind
    def __call__(self, k: float) -> float:
        c = _pb_cdf1(self.res, k - self.loc)
        if self.kind == 0:
            return c
        elif self.kind == 1:
            return K.log(c)
        elif self.kind == 2:
            return min(max(1.0 - c, 0.0), 1.0)
        else:
            return K.log(min(max(1.0 - c, 0.0), 1.0))

class _PbPpf:
    cdf: np.ndarray[float, 1]
    loc: float
    isf: bool
    def __init__(self, cdf: np.ndarray[float, 1], loc: float, isf: bool):
        self.cdf = cdf
        self.loc = loc
        self.isf = isf
    def __call__(self, q: float) -> float:
        n = self.cdf.shape[0] - 1
        if not (q >= 0 and q <= 1):
            return NAN
        if self.isf:
            if q == 1:
                return -1.0 + self.loc
            if q == 0:
                return float(n) + self.loc
            for k in range(n + 1):
                if 1.0 - self.cdf[k] <= q:
                    return float(k) + self.loc
            return float(n) + self.loc
        if q == 0:
            return -1.0 + self.loc
        if q == 1:
            return float(n) + self.loc
        for k in range(n + 1):
            if self.cdf[k] >= q:
                return float(k) + self.loc
        return float(n) + self.loc

class poisson_binomial_frozen:
    p: np.ndarray[float, 1]
    loc: float

    def __init__(self, p: np.ndarray[float, 1], loc: float):
        self.p = p
        self.loc = loc
        for i in range(p.shape[0]):
            if not (p[i] >= 0 and p[i] <= 1):
                raise ValueError("Domain error in arguments. The `scale` parameter must be "
                                 "positive for all distributions, and many distributions "
                                 "have restrictions on shape parameters.")

    def pmf(self, k): return _pb_map(_PbPmf(_pb_pmf_all(self.p), self.loc), k)
    def logpmf(self, k): return _pb_map(_PbLogPmf(_pb_pmf_all(self.p), self.loc), k)
    def cdf(self, k): return _pb_map(_PbCdf(_pb_cdf_all(self.p), self.loc, 0), k)
    def logcdf(self, k): return _pb_map(_PbCdf(_pb_cdf_all(self.p), self.loc, 1), k)
    def sf(self, k): return _pb_map(_PbCdf(_pb_cdf_all(self.p), self.loc, 2), k)
    def logsf(self, k): return _pb_map(_PbCdf(_pb_cdf_all(self.p), self.loc, 3), k)
    def ppf(self, q): return _pb_map(_PbPpf(_pb_cdf_all(self.p), self.loc, False), q)
    def isf(self, q): return _pb_map(_PbPpf(_pb_cdf_all(self.p), self.loc, True), q)

    def mean(self) -> float:
        return float(np.sum(self.p)) + self.loc
    def var(self) -> float:
        return float(np.sum(self.p * (1.0 - self.p)))
    def std(self) -> float:
        return K.sqrt(self.var())
    def median(self): return self.ppf(0.5)

    def _central(self, order: int) -> float:
        pm = _pb_pmf_all(self.p)
        mu = 0.0
        for k in range(pm.shape[0]):
            mu += k * pm[k]
        s = 0.0
        for k in range(pm.shape[0]):
            s += (k - mu) ** order * pm[k]
        return s

    def stats(self, moments: Literal[str] = 'mv'):
        mean = self.mean()
        var = self.var()
        v3 = self._central(3)
        v4 = self._central(4)
        g1 = v3 / var ** 1.5
        g2 = v4 / var ** 2 - 3.0
        if moments == 'm':
            return mean
        elif moments == 'v':
            return var
        elif moments == 's':
            return g1
        elif moments == 'k':
            return g2
        elif moments == 'mv':
            return (mean, var)
        elif moments == 'ms':
            return (mean, g1)
        elif moments == 'mk':
            return (mean, g2)
        elif moments == 'vs':
            return (var, g1)
        elif moments == 'vk':
            return (var, g2)
        elif moments == 'sk':
            return (g1, g2)
        elif moments == 'mvs':
            return (mean, var, g1)
        elif moments == 'mvk':
            return (mean, var, g2)
        elif moments == 'msk':
            return (mean, g1, g2)
        elif moments == 'vsk':
            return (var, g1, g2)
        else:
            return (mean, var, g1, g2)

    def support(self):
        return (self.loc, float(self.p.shape[0]) + self.loc)

    def rvs(self, size = None, random_state = None):
        rs = check_random_state(random_state)
        n = self.p.shape[0]
        cnt = _nsamples(size)
        out = np.empty(cnt, dtype=float)
        pp = self.p
        bern = rs.binomial(1, np.tile(pp, cnt), size=cnt * n)
        for i in range(cnt):
            s = 0.0
            for j in range(n):
                s += float(bern[i * n + j])
            out[i] = s + self.loc
        return _shape_out(out, size)

class poisson_binom_gen:
    def __init__(self):
        pass
    def __call__(self, p, loc = 0.0):
        return poisson_binomial_frozen(np.asarray(p).astype(float).ravel(), float(loc))
    def freeze(self, p, loc = 0.0): return self(p, loc)
    def pmf(self, k, p, loc = 0.0): return self(p, loc).pmf(k)
    def logpmf(self, k, p, loc = 0.0): return self(p, loc).logpmf(k)
    def cdf(self, k, p, loc = 0.0): return self(p, loc).cdf(k)
    def logcdf(self, k, p, loc = 0.0): return self(p, loc).logcdf(k)
    def sf(self, k, p, loc = 0.0): return self(p, loc).sf(k)
    def logsf(self, k, p, loc = 0.0): return self(p, loc).logsf(k)
    def ppf(self, q, p, loc = 0.0): return self(p, loc).ppf(q)
    def isf(self, q, p, loc = 0.0): return self(p, loc).isf(q)
    def mean(self, p, loc = 0.0): return self(p, loc).mean()
    def var(self, p, loc = 0.0): return self(p, loc).var()
    def std(self, p, loc = 0.0): return self(p, loc).std()
    def median(self, p, loc = 0.0): return self(p, loc).median()
    def stats(self, p, loc = 0.0, moments: Literal[str] = 'mv'):
        return self(p, loc).stats(moments)
    def support(self, p, loc = 0.0): return self(p, loc).support()
    def rvs(self, p, loc = 0.0, size = None, random_state = None):
        return self(p, loc).rvs(size, random_state)

poisson_binom = poisson_binom_gen()
