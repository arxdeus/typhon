# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# scipy.stats.contingency: expected_freq, margins, association, crosstab,
# relative_risk, odds_ratio (sample / conditional), plus somersd and
# page_trend_test (contingency/rank based tests).

import numpy as np
import scipy.special._kernels as K
from scipy.optimize import brentq
from ._common import _f64, _pvalue_norm
from ._stats import chi2_contingency, Chi2ContingencyResult, _expected_freq, \
                    SignificanceResult, _Result2, rankdata
from ._morestats import ConfidenceInterval, _hyp_pmf

NAN = K.NAN
INF = K.INF

def expected_freq(observed):
    return _expected_freq(_f64(observed))

def margins(a):
    x = np.asarray(a)
    d = x.ndim
    out = []
    for k in range(d):
        marg = x
        for j in range(d - 1, -1, -1):
            if j != k:
                marg = np.sum(marg, axis=j, keepdims=True)
        out.append(marg)
    return out

def association(observed, method: str = "cramer", correction: bool = False, lambda_ = None) -> float:
    arr = np.asarray(observed)
    if arr.ndim != 2:
        raise ValueError("method only accepts 2d arrays")
    r = chi2_contingency(arr, correction=correction, lambda_=lambda_)
    phi2 = r.statistic / float(np.sum(arr))
    n_rows = arr.shape[0]
    n_cols = arr.shape[1]
    value = 0.0
    if method == "cramer":
        value = phi2 / min(n_cols - 1, n_rows - 1)
    elif method == "tschuprow":
        value = phi2 / K.sqrt(float((n_rows - 1) * (n_cols - 1)))
    elif method == 'pearson':
        value = phi2 / (1 + phi2)
    else:
        raise ValueError("Invalid argument value: 'method' argument must be 'cramer', 'tschuprow', or 'pearson'")
    return K.sqrt(value)

class CrosstabResult[E, C]:
    elements: E
    count: C
    def __init__(self, elements: E, count: C):
        self.elements = elements
        self.count = count
    def __getitem__(self, i: Literal[int]):
        if i == 0: return self.elements
        elif i == 1: return self.count
        else: compile_error("result index out of range")

def crosstab(x, y):
    """Two-way crosstab (levels = sorted unique values of each input)."""
    a = np.asarray(x)
    b = np.asarray(y)
    if a.shape[0] != b.shape[0]:
        raise ValueError("All input sequences must have the same length.")
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    count = np.zeros((ua.shape[0], ub.shape[0]), dtype=int)
    for i in range(a.shape[0]):
        count[ia[i], ib[i]] += 1
    return CrosstabResult((ua, ub), count)

class RelativeRiskResult:
    relative_risk: float
    exposed_cases: int
    exposed_total: int
    control_cases: int
    control_total: int
    def __init__(self, relative_risk: float, exposed_cases: int, exposed_total: int,
                 control_cases: int, control_total: int):
        self.relative_risk = relative_risk
        self.exposed_cases = exposed_cases
        self.exposed_total = exposed_total
        self.control_cases = control_cases
        self.control_total = control_total
    def confidence_interval(self, confidence_level: float = 0.95):
        if not 0 <= confidence_level <= 1:
            raise ValueError('confidence_level must be in the interval [0, 1].')
        if self.exposed_cases == 0 and self.control_cases == 0:
            return ConfidenceInterval(NAN, NAN)
        elif self.exposed_cases == 0:
            return ConfidenceInterval(0.0, NAN)
        elif self.control_cases == 0:
            return ConfidenceInterval(NAN, INF)
        alpha = 1 - confidence_level
        z = K.ndtri(1 - alpha / 2)
        rr = self.relative_risk
        se = K.sqrt(1 / self.exposed_cases - 1 / self.exposed_total +
                    1 / self.control_cases - 1 / self.control_total)
        delta = z * se
        return ConfidenceInterval(rr * K.exp(-delta), rr * K.exp(delta))

def relative_risk(exposed_cases: int, exposed_total: int, control_cases: int, control_total: int):
    if exposed_cases > exposed_total:
        raise ValueError('exposed_cases must not exceed exposed_total.')
    if control_cases > control_total:
        raise ValueError('control_cases must not exceed control_total.')
    rr = 0.0
    if exposed_cases == 0 and control_cases == 0:
        rr = NAN
    elif exposed_cases == 0:
        rr = 0.0
    elif control_cases == 0:
        rr = INF
    else:
        rr = (exposed_cases / exposed_total) / (control_cases / control_total)
    return RelativeRiskResult(rr, exposed_cases, exposed_total, control_cases, control_total)

# ------------------------------------------------------------------ odds ratio

def _nchg_support(M: int, n: int, N: int):
    return max(0, N - (M - n)), min(N, n)

def _nchg_weights(M: int, n: int, N: int, odds: float):
    # unnormalized log-weights of Fisher's noncentral hypergeometric
    lo, hi = _nchg_support(M, n, N)
    lw = []
    for x in range(lo, hi + 1):
        xf = float(x)
        lw.append(K.gammaln(n + 1.) - K.gammaln(xf + 1) - K.gammaln(n - xf + 1)
                  + K.gammaln(M - n + 1.) - K.gammaln(N - xf + 1) - K.gammaln(M - n - N + xf + 1)
                  + xf * K.log(odds))
    mx = max(lw)
    w = [K.exp(v - mx) for v in lw]
    s = 0.0
    for v in w:
        s += v
    return lo, [v / s for v in w]

def _nchg_mean(M: int, n: int, N: int, odds: float) -> float:
    lo, p = _nchg_weights(M, n, N, odds)
    m = 0.0
    for i in range(len(p)):
        m += (lo + i) * p[i]
    return m

def _nchg_cdf(x: int, M: int, n: int, N: int, odds: float) -> float:
    lo, p = _nchg_weights(M, n, N, odds)
    s = 0.0
    for i in range(len(p)):
        if lo + i <= x:
            s += p[i]
    return min(s, 1.0)

def _nchg_sf(x: int, M: int, n: int, N: int, odds: float) -> float:
    lo, p = _nchg_weights(M, n, N, odds)
    s = 0.0
    for i in range(len(p)):
        if lo + i > x:
            s += p[i]
    return min(s, 1.0)

class _MeanEq:
    x: float
    M: int
    n: int
    N: int
    def __call__(self, nc: float) -> float:
        return _nchg_mean(self.M, self.n, self.N, nc) - self.x

class _CdfEq:
    x: int
    M: int
    n: int
    N: int
    alpha: float
    upper: bool
    def __call__(self, nc: float) -> float:
        if self.upper:
            return -_nchg_cdf(self.x, self.M, self.n, self.N, nc) + self.alpha
        return _nchg_sf(self.x - 1, self.M, self.n, self.N, nc) - self.alpha

def _solve(func) -> float:
    nc = 1.0
    value = func(nc)
    if value == 0:
        return nc
    factor = 2.0
    lo = 0.0
    hi = 0.0
    if value > 0:
        nc /= factor
        while func(nc) > 0:
            nc /= factor
        lo = nc
        hi = factor * nc
    else:
        nc *= factor
        while func(nc) < 0:
            nc *= factor
        lo = nc / factor
        hi = nc
    return brentq(func, lo, hi, xtol=1e-13)

def _ci_lower(so: float, x: int, M: int, n: int, N: int, alpha: float) -> float:
    if so == 0:
        return 0.
    return _solve(_CdfEq(x, M, n, N, alpha, False))

def _ci_upper(so: float, x: int, M: int, n: int, N: int, alpha: float) -> float:
    if so == INF:
        return INF
    return _solve(_CdfEq(x, M, n, N, alpha, True))

def _sample_or(t) -> float:
    if t[1, 0] > 0 and t[0, 1] > 0:
        return float(t[0, 0] * t[1, 1]) / float(t[1, 0] * t[0, 1])
    elif t[0, 0] == 0 or t[1, 1] == 0:
        return NAN
    return INF

class OddsRatioResult:
    _table: np.ndarray[int, 2]
    _kind: str
    statistic: float
    def __init__(self, _table: np.ndarray[int, 2], _kind: str, statistic: float):
        self._table = _table
        self._kind = _kind
        self.statistic = statistic

    def confidence_interval(self, confidence_level: float = 0.95, alternative: str = 'two-sided'):
        if alternative != 'two-sided' and alternative != 'less' and alternative != 'greater':
            raise ValueError("`alternative` must be 'two-sided', 'less' or 'greater'.")
        if confidence_level < 0 or confidence_level > 1:
            raise ValueError('confidence_level must be between 0 and 1')
        t = self._table
        if (t[0, 0] + t[1, 0] == 0 or t[0, 1] + t[1, 1] == 0 or
                t[0, 0] + t[0, 1] == 0 or t[1, 0] + t[1, 1] == 0):
            return ConfidenceInterval(0., INF)
        if self._kind == 'conditional':
            x = int(t[0, 0])
            M = int(np.sum(t))
            n = int(t[0, 0] + t[0, 1])
            N = int(t[0, 0] + t[1, 0])
            so = _sample_or(t)
            if alternative == 'two-sided':
                a = 0.5 * (1 - confidence_level)
                return ConfidenceInterval(_ci_lower(so, x, M, n, N, a), _ci_upper(so, x, M, n, N, a))
            elif alternative == 'less':
                return ConfidenceInterval(0.0, _ci_upper(so, x, M, n, N, 1 - confidence_level))
            else:
                return ConfidenceInterval(_ci_lower(so, x, M, n, N, 1 - confidence_level), INF)
        else:
            orr = _sample_or(t)
            log_or = K.log(orr)
            se = K.sqrt(1 / float(t[0, 0]) + 1 / float(t[0, 1]) + 1 / float(t[1, 0]) + 1 / float(t[1, 1]))
            if alternative == 'less':
                z = K.ndtri(confidence_level)
                return ConfidenceInterval(0.0, K.exp(log_or + z * se))
            elif alternative == 'greater':
                z = K.ndtri(confidence_level)
                return ConfidenceInterval(K.exp(log_or - z * se), INF)
            z = K.ndtri(0.5 * confidence_level + 0.5)
            return ConfidenceInterval(K.exp(log_or - z * se), K.exp(log_or + z * se))

def odds_ratio(table, *, kind: str = 'conditional'):
    if kind != 'conditional' and kind != 'sample':
        raise ValueError("`kind` must be 'conditional' or 'sample'.")
    c = np.asarray(table).astype(int)
    if c.ndim != 2 or c.shape[0] != 2 or c.shape[1] != 2:
        raise ValueError("The input `table` must be of shape (2, 2).")
    for v in c.flat:
        if v < 0:
            raise ValueError("All values in `table` must be nonnegative.")
    if (c[0, 0] + c[1, 0] == 0 or c[0, 1] + c[1, 1] == 0 or
            c[0, 0] + c[0, 1] == 0 or c[1, 0] + c[1, 1] == 0):
        return OddsRatioResult(c, kind, NAN)
    if kind == 'sample':
        return OddsRatioResult(c, kind, _sample_or(c))
    x = int(c[0, 0])
    M = int(np.sum(c))
    n = int(c[0, 0] + c[0, 1])
    N = int(c[0, 0] + c[1, 0])
    lo, hi = _nchg_support(M, n, N)
    if x == lo:
        return OddsRatioResult(c, kind, 0.)
    if x == hi:
        return OddsRatioResult(c, kind, INF)
    return OddsRatioResult(c, kind, _solve(_MeanEq(float(x), M, n, N)))

# ------------------------------------------------------------------ Somers' D

def _Aij(A, i: int, j: int) -> float:
    return float(np.sum(A[:i, :j])) + float(np.sum(A[i + 1:, j + 1:]))

def _Dij(A, i: int, j: int) -> float:
    return float(np.sum(A[i + 1:, :j])) + float(np.sum(A[:i, j + 1:]))

class SomersDResult:
    statistic: float
    pvalue: float
    table: np.ndarray[int, 2]
    def __init__(self, statistic: float, pvalue: float, table: np.ndarray[int, 2]):
        self.statistic = statistic
        self.pvalue = pvalue
        self.table = table
    @property
    def correlation(self):
        return self.statistic

def _somers_d(A, alternative: str):
    if A.shape[0] <= 1 or A.shape[1] <= 1:
        return (NAN, NAN)
    NA = float(np.sum(A))
    m = A.shape[0]
    n = A.shape[1]
    PA = 0.0
    QA = 0.0
    S2 = 0.0
    for i in range(m):
        for j in range(n):
            a = _Aij(A, i, j)
            d = _Dij(A, i, j)
            PA += A[i, j] * a
            QA += A[i, j] * d
            S2 += A[i, j] * (a - d) ** 2
    Sri2 = float(np.sum(np.sum(A, axis=1) ** 2))
    dd = (PA - QA) / (NA ** 2 - Sri2)
    S = S2 - (PA - QA) ** 2 / NA
    Z = (PA - QA).cdiv((4 * S) ** 0.5)
    return (dd, _pvalue_norm(Z, alternative))

def somersd(x, y = None, alternative: str = 'two-sided'):
    xa = np.asarray(x)
    if xa.ndim == 1:
        ya = np.asarray(y)
        if xa.shape[0] != ya.shape[0]:
            raise ValueError("Rankings must be of equal length.")
        table = crosstab(xa, ya).count
        r = _somers_d(table.astype(float), alternative)
        return SomersDResult(r[0], r[1], table)
    else:
        table = xa.astype(int)
        r = _somers_d(table.astype(float), alternative)
        return SomersDResult(r[0], r[1], table)

# ------------------------------------------------------------------ Page's L

class PageTrendTestResult:
    statistic: float
    pvalue: float
    method: str
    def __init__(self, statistic: float, pvalue: float, method: str):
        self.statistic = statistic
        self.pvalue = pvalue
        self.method = method

def _tally(pp: List[int], counts: List[float], k: int, a: int):
    s = 0
    for i in range(k):
        s += (i + 1) * pp[i]
    counts[s - a] += 1.0

def _perm_L_counts(k: int, a: int, b: int):
    # histogram of sum(i * perm[i]) over all permutations of 1..k
    counts = [0.0] * (b - a + 1)
    perm = list(range(1, k + 1))
    c = [0] * k
    _tally(perm, counts, k, a)
    i = 0
    while i < k:
        if c[i] < i:
            if i % 2 == 0:
                perm[0], perm[i] = perm[i], perm[0]
            else:
                perm[c[i]], perm[i] = perm[i], perm[c[i]]
            _tally(perm, counts, k, a)
            c[i] += 1
            i = 0
        else:
            c[i] = 0
            i += 1
    f = 1.0
    for q in range(2, k + 1):
        f *= q
    return [v / f for v in counts]

def _page_exact_sf(L: int, m: int, k: int) -> float:
    a = (k * (k + 1) * (k + 2)) // 6
    b = (k * (k + 1) * (2 * k + 1)) // 6
    p1 = _perm_L_counts(k, a, b)
    # convolution of m copies (pmf over L in [m*a, m*b])
    cur = p1
    for _ in range(m - 1):
        nxt = [0.0] * (len(cur) + len(p1) - 1)
        for i in range(len(cur)):
            if cur[i] == 0.0:
                continue
            for j in range(len(p1)):
                nxt[i + j] += cur[i] * p1[j]
        cur = nxt
    lo = m * a
    vals = []
    for l in range(L, m * b + 1):
        idx = l - lo
        vals.append(cur[idx] if 0 <= idx < len(cur) else 0.0)
    return float(np.sum(np.array(vals, dtype=float)))

def page_trend_test(data, ranked: bool = False, predicted_ranks = None, method: str = 'auto'):
    if method != 'auto' and method != 'exact' and method != 'asymptotic':
        raise ValueError("`method` must be in {'auto', 'exact', 'asymptotic'}")
    d = _f64(data)
    if d.ndim != 2:
        raise ValueError("`data` must be a 2d array.")
    m = d.shape[0]
    n = d.shape[1]
    if m < 2 or n < 3:
        raise ValueError("Page's L is only appropriate for data with two or more rows and three or more columns.")
    ranks = d if ranked else rankdata(d, axis=-1)
    pr = np.arange(1, n + 1).astype(float)
    if predicted_ranks is not None:
        pr = _f64(predicted_ranks)
    colsums = np.sum(ranks, axis=0)
    L = float(np.sum(pr * colsums))
    meth = method
    if meth == 'auto':
        meth = "asymptotic" if (n > 8 or (m > 12 and n > 3) or m > 20) else "exact"
    p = 0.0
    if meth == 'asymptotic':
        E0 = (m * n * (n + 1) ** 2) / 4
        V0 = (m * n ** 2 * (n + 1) * (n ** 2 - 1)) / 144
        p = K.ndtr(-(L - E0) / K.sqrt(float(V0)))
    else:
        p = _page_exact_sf(int(L), m, n)
    return PageTrendTestResult(L, p, meth)
