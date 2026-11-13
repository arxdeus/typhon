# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Hypothesis tests (normality, goodness of fit, rank tests, correlation).

import numpy as np
import internal.static as static
import scipy.special._kernels as K
from C import rint(float) -> float
from ._common import _f64, _slices, _reduce, _mean, _moment, _var, _rank_avg, \
                     _tie_term, _pvalue_t, _pvalue_norm
from ._stats import _Result2, SignificanceResult, _map_slices, _map_slices2, \
                    _ksample, _skew1, _kurt1, _pearson1, PearsonRResult, pearsonr, _lambda_value

NAN = K.NAN
INF = K.INF

class SkewtestResult[S, P](_Result2[S, P]):
    def __init__(self, statistic: S, pvalue: P):
        super().__init__(statistic, pvalue)

class KurtosistestResult[S, P](_Result2[S, P]):
    def __init__(self, statistic: S, pvalue: P):
        super().__init__(statistic, pvalue)

class NormaltestResult[S, P](_Result2[S, P]):
    def __init__(self, statistic: S, pvalue: P):
        super().__init__(statistic, pvalue)

class Power_divergenceResult[S, P](_Result2[S, P]):
    def __init__(self, statistic: S, pvalue: P):
        super().__init__(statistic, pvalue)

class Ttest_indResult[S, P](_Result2[S, P]):
    def __init__(self, statistic: S, pvalue: P):
        super().__init__(statistic, pvalue)

class RanksumsResult[S, P](_Result2[S, P]):
    def __init__(self, statistic: S, pvalue: P):
        super().__init__(statistic, pvalue)

class BrunnerMunzelResult[S, P](_Result2[S, P]):
    def __init__(self, statistic: S, pvalue: P):
        super().__init__(statistic, pvalue)

class FriedmanchisquareResult[S, P](_Result2[S, P]):
    def __init__(self, statistic: S, pvalue: P):
        super().__init__(statistic, pvalue)

class AlexanderGovernResult[S, P](_Result2[S, P]):
    def __init__(self, statistic: S, pvalue: P):
        super().__init__(statistic, pvalue)

@tuple
class TheilslopesResult:
    slope: float
    intercept: float
    low_slope: float
    high_slope: float

@tuple
class SiegelslopesResult:
    slope: float
    intercept: float

def _chk_alt(alternative: str):
    if alternative != 'less' and alternative != 'greater' and alternative != 'two-sided':
        raise ValueError("`alternative` must be 'less', 'greater', or 'two-sided'.")

# ------------------------------------------------------------------ normality

class _Skewtest:
    alternative: str
    def __call__(self, a):
        b2 = _skew1(a, True)
        n = float(a.shape[0])
        if n < 8:
            n = NAN
        y = b2 * K.sqrt(((n + 1) * (n + 3)) / (6.0 * (n - 2)))
        beta2 = (3.0 * (n ** 2 + 27 * n - 70) * (n + 1) * (n + 3) /
                 ((n - 2.0) * (n + 5) * (n + 7) * (n + 9)))
        W2 = -1 + K.sqrt(2 * (beta2 - 1))
        delta = 1 / K.sqrt(0.5 * K.log(W2))
        alpha = K.sqrt(2.0 / (W2 - 1))
        if y == 0:
            y = 1.
        Z = delta * K.log(y / alpha + K.sqrt((y / alpha) ** 2 + 1))
        return (Z, _pvalue_norm(Z, self.alternative))

def skewtest(a, axis = 0, nan_policy: str = 'propagate', alternative: str = 'two-sided'):
    _chk_alt(alternative)
    r = _map_slices(a, axis, _Skewtest(alternative))
    return SkewtestResult(r[0], r[1])

def _sign(x: float) -> float:
    return 1.0 if x > 0 else (-1.0 if x < 0 else (0.0 if x == 0 else x))

class _Kurtosistest:
    alternative: str
    def __call__(self, a):
        b2 = _kurt1(a, False, True)
        n = float(a.shape[0])
        if n < 5:
            n = NAN
        E = 3.0 * (n - 1) / (n + 1)
        varb2 = 24.0 * n * (n - 2) * (n - 3) / ((n + 1) * (n + 1.) * (n + 3) * (n + 5))
        x = (b2 - E) / varb2 ** 0.5
        sqrtbeta1 = 6.0 * (n * n - 5 * n + 2) / ((n + 7) * (n + 9)) * ((6.0 * (n + 3) * (n + 5))
                                                                    / (n * (n - 2) * (n - 3))) ** 0.5
        A = 6.0 + 8.0 / sqrtbeta1 * (2.0 / sqrtbeta1 + (1 + 4.0 / (sqrtbeta1 ** 2)) ** 0.5)
        term1 = 1 - 2 / (9.0 * A)
        denom = 1 + x * (2 / (A - 4.0)) ** 0.5
        term2 = _sign(denom) * (NAN if denom == 0.0 else ((1 - 2.0 / A) / abs(denom)) ** (1 / 3))
        Z = (term1 - term2) / (2 / (9.0 * A)) ** 0.5
        return (Z, _pvalue_norm(Z, self.alternative))

def kurtosistest(a, axis = 0, nan_policy: str = 'propagate', alternative: str = 'two-sided'):
    _chk_alt(alternative)
    r = _map_slices(a, axis, _Kurtosistest(alternative))
    return KurtosistestResult(r[0], r[1])

class _Normaltest:
    def __call__(self, a):
        s = _Skewtest('two-sided')(a)[0]
        k = _Kurtosistest('two-sided')(a)[0]
        st = s * s + k * k
        return (st, K.chdtrc(2.0, st))

def normaltest(a, axis = 0, nan_policy: str = 'propagate'):
    r = _map_slices(a, axis, _Normaltest())
    return NormaltestResult(r[0], r[1])

class _JB:
    def __call__(self, x):
        mu = float(np.mean(x))
        d = x - mu
        s = _skew1(d, True)
        k = _kurt1(d, True, True)
        n = float(x.shape[0])
        st = n / 6 * (s ** 2 + k ** 2 / 4)
        return (st, K.chdtrc(2.0, st))

def jarque_bera(x, *, axis = None):
    r = _map_slices(x, axis, _JB())
    return SignificanceResult(r[0], r[1])

# ------------------------------------------------------------------ power divergence

class _PowerDiv:
    lam: float
    ddof: float
    def __call__(self, fo, fe):
        n = fo.shape[0]
        terms = np.empty(n, dtype=float)
        lam = self.lam
        for i in range(n):
            o = fo[i]
            e = fe[i]
            if lam == 1:
                terms[i] = (o - e) ** 2 / e
            elif lam == 0:
                terms[i] = 2.0 * K.xlogy(o, o / e)
            elif lam == -1:
                terms[i] = 2.0 * K.xlogy(e, e / o)
            else:
                t = o * ((o / e) ** lam - 1)
                t /= 0.5 * lam * (lam + 1)
                terms[i] = t
        stat = float(np.sum(terms))
        df = float(n) - 1 - self.ddof
        return (stat, K.chdtrc(df, stat))

class _PowerDivMean:
    pd: _PowerDiv
    def __call__(self, fo):
        fe = np.full(fo.shape[0], float(np.mean(fo)))
        return self.pd(fo, fe)

def _check_sums(fo, fe, axis):
    rtol = 1.4901161193847656e-08
    so = np.sum(fo, axis=axis, keepdims=True)
    se = np.sum(fe, axis=axis, keepdims=True)
    for idx in np.ndindex(so.shape):
        a = float(so[idx])
        b = float(se[idx])
        if abs(a - b) / min(a, b) > rtol:
            raise ValueError(f"For each axis slice, the sum of the observed frequencies must agree with the sum of the expected frequencies to a relative tolerance of {rtol}")

def _power_divergence(f_obs, f_exp, ddof, axis, lambda_, sum_check: bool):
    lam = _lambda_value(lambda_)
    fo = _f64(f_obs)
    pd = _PowerDiv(lam, float(ddof))
    if f_exp is None:
        r = _map_slices(fo, axis, _PowerDivMean(pd))
        return Power_divergenceResult(r[0], r[1])
    else:
        fe = _f64(f_exp)
        shp = np.broadcast_shapes(fo.shape, fe.shape)
        fo2 = np.broadcast_to(fo, shp)
        fe2 = np.broadcast_to(fe, shp)
        if sum_check:
            _check_sums(fo2, fe2, axis)
        r = _map_slices2(fo2, fe2, axis, pd)
        return Power_divergenceResult(r[0], r[1])

def power_divergence(f_obs, f_exp = None, ddof = 0, axis = 0, lambda_ = None):
    return _power_divergence(f_obs, f_exp, ddof, axis, lambda_, True)

def chisquare(f_obs, f_exp = None, ddof = 0, axis = 0, *, sum_check: bool = True):
    return _power_divergence(f_obs, f_exp, ddof, axis, "pearson", sum_check)

# ------------------------------------------------------------------ t-test from stats

def ttest_ind_from_stats(mean1: float, std1: float, nobs1: float, mean2: float,
                         std2: float, nobs2: float, equal_var: bool = True,
                         alternative: str = "two-sided"):
    v1 = std1 ** 2
    v2 = std2 ** 2
    n1 = float(nobs1)
    n2 = float(nobs2)
    df = 0.0
    denom = 0.0
    if equal_var:
        if n1 == 1:
            v1 = 0.
        if n2 == 1:
            v2 = 0.
        df = n1 + n2 - 2.0
        svar = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
        denom = K.sqrt(svar * (1.0 / n1 + 1.0 / n2))
    else:
        vn1 = v1 / n1
        vn2 = v2 / n2
        df = (vn1 + vn2) ** 2 / (vn1 ** 2 / (n1 - 1) + vn2 ** 2 / (n2 - 1))
        if df != df:
            df = 1.
        denom = K.sqrt(vn1 + vn2)
    t = (mean1 - mean2) / denom
    return Ttest_indResult(t, _pvalue_t(t, df, alternative))

def pointbiserialr(x, y, *, axis = 0):
    r = pearsonr(_f64(x), _f64(y), axis=axis)
    return SignificanceResult(r.statistic, r.pvalue)

# ------------------------------------------------------------------ rank tests

def ranksums(x, y, alternative: str = 'two-sided'):
    a = _f64(x).ravel()
    b = _f64(y).ravel()
    n1 = a.shape[0]
    n2 = b.shape[0]
    ranked, _ = _rank_avg(np.concatenate((a, b)))
    s = float(np.sum(ranked[:n1]))
    expected = n1 * (n1 + n2 + 1) / 2.0
    z = (s - expected) / K.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    return RanksumsResult(z, _pvalue_norm(z, alternative))

class _BM:
    alternative: str
    distribution: str
    def __call__(self, x, y):
        nx = float(x.shape[0])
        ny = float(y.shape[0])
        lx = x.shape[0]
        rankc, _ = _rank_avg(np.concatenate((x, y)))
        rankcx = rankc[:lx]
        rankcy = rankc[lx:]
        rcxm = float(np.mean(rankcx))
        rcym = float(np.mean(rankcy))
        rankx, _ = _rank_avg(x)
        ranky, _ = _rank_avg(y)
        rxm = float(np.mean(rankx))
        rym = float(np.mean(ranky))
        tx = rankcx - rankx - rcxm + rxm
        Sx = float(np.dot(tx, tx))
        Sx /= nx - 1
        ty = rankcy - ranky - rcym + rym
        Sy = float(np.dot(ty, ty))
        Sy /= ny - 1
        wbfn = nx * ny * (rcym - rcxm)
        wbfn /= (nx + ny) * K.sqrt(nx * Sx + ny * Sy)
        if self.distribution == "t":
            df_numer = (nx * Sx + ny * Sy) ** 2.0
            df_denom = (nx * Sx) ** 2.0 / (nx - 1)
            df_denom += (ny * Sy) ** 2.0 / (ny - 1)
            df = df_numer / df_denom
            return (wbfn, _pvalue_t(-wbfn, df, self.alternative))
        elif self.distribution == "normal":
            return (wbfn, _pvalue_norm(-wbfn, self.alternative))
        else:
            raise ValueError("distribution should be 't' or 'normal'")

def brunnermunzel(x, y, alternative: str = "two-sided", distribution: str = "t",
                  nan_policy: str = 'propagate', *, axis = 0):
    r = _map_slices2(x, y, axis, _BM(alternative, distribution))
    return BrunnerMunzelResult(r[0], r[1])

class _Friedman:
    def __call__(self, ss):
        k = len(ss)
        n = ss[0].shape[0]
        ssbn_cols = np.zeros(k, dtype=float)
        ties = 0.0
        row = np.empty(k, dtype=float)
        for i in range(n):
            for j in range(k):
                row[j] = ss[j][i]
            r, t = _rank_avg(row)
            for j in range(k):
                ssbn_cols[j] += r[j]
            for tt in t:
                ties += tt * (tt * tt - 1)
        count = float(n)
        c = 1 - ties / (k * (k * k - 1) * count)
        ssbn = float(np.sum(ssbn_cols ** 2))
        st = (12.0 / (k * count * (k + 1)) * ssbn - 3 * count * (k + 1)) / c
        return (st, K.chdtrc(float(k - 1), st))

def friedmanchisquare(*samples, axis = 0):
    if static.len(samples) < 3:
        compile_error("At least 3 samples must be given for Friedman test")
    arrs = [_f64(s) for s in samples]
    r = _ksample(arrs, axis, _Friedman())
    return FriedmanchisquareResult(r[0], r[1])

class _AG:
    def __call__(self, ss):
        k = len(ss)
        means = np.empty(k, dtype=float)
        se2 = np.empty(k, dtype=float)
        lengths = np.empty(k, dtype=float)
        for i in range(k):
            s = ss[i]
            if s.shape[0] <= 1:
                raise ValueError("Input sample size must be greater than one.")
            lengths[i] = float(s.shape[0])
            means[i] = float(np.mean(s))
            se2[i] = _var(s, 1) / lengths[i]
        se = se2 ** 0.5
        eps = 2.220446049250313e-16
        for i in range(k):
            if se[i] <= abs(eps * means[i]):
                se[i] = NAN
        inv = 1 / se2
        weights = inv / float(np.sum(inv))
        var_w = float(np.sum(weights * means))
        z = np.empty(k, dtype=float)
        for i in range(k):
            t = (means[i] - var_w) / se[i]
            v = lengths[i] - 1
            a = v - .5
            b = 48 * a ** 2
            c = (a * K.log(1 + (t ** 2) / v)) ** .5
            z[i] = (c + ((c ** 3 + 3 * c) / b) -
                    ((4 * c ** 7 + 33 * c ** 5 + 240 * c ** 3 + 855 * c) /
                     (b ** 2 * 10 + 8 * b * c ** 4 + 1000 * b)))
        A = float(np.dot(z, z))
        return (A, K.chdtrc(float(k - 1), A))

def alexandergovern(*samples, nan_policy: str = 'propagate', axis = 0):
    if static.len(samples) < 2:
        compile_error("2 or more inputs required")
    arrs = [_f64(s) for s in samples]
    r = _ksample(arrs, axis, _AG())
    return AlexanderGovernResult(r[0], r[1])

# ------------------------------------------------------------------ Kendall tau

def _factorial(n: int) -> float:
    r = 1.0
    for i in range(2, n + 1):
        r *= float(i)
    return r

def _kendall_p_exact(n: int, c0: int, alternative: str) -> float:
    tot = (n * (n - 1)) // 2
    in_right_tail = c0 >= tot - c0
    alternative_greater = alternative == 'greater'
    c = min(c0, tot - c0)
    prob = 0.0
    p_mass_at_c = 0.0
    if n <= 0:
        raise ValueError(f'n ({n}) must be positive')
    elif c < 0 or 4 * c > n * (n - 1):
        raise ValueError(f'c ({c}) must satisfy 0 <= 4c <= n(n-1) = {n*(n-1)}.')
    elif n == 1:
        prob = 1.0
        p_mass_at_c = 1.
    elif n == 2:
        prob = 1.0
        p_mass_at_c = 0.5
    elif c == 0:
        prob = 2.0 / _factorial(n) if n < 171 else 0.0
        p_mass_at_c = prob / 2
    elif c == 1:
        prob = 2.0 / _factorial(n - 1) if n < 172 else 0.0
        p_mass_at_c = (n - 1) / _factorial(n)
    elif 4 * c == n * (n - 1) and alternative == 'two-sided':
        prob = 1.0
    elif n < 171:
        new = np.zeros(c + 1)
        new[0] = 1.0
        new[1] = 1.0
        for j in range(3, n + 1):
            new = np.cumsum(new)
            if j <= c:
                old = new.copy()
                for q in range(j, c + 1):
                    new[q] -= old[q - j]
        f = _factorial(n)
        prob = 2.0 * float(np.sum(new)) / f
        p_mass_at_c = new[c] / f
    else:
        new = np.zeros(c + 1)
        new[0] = 1.0
        new[1] = 1.0
        for j in range(3, n + 1):
            new = np.cumsum(new) / j
            if j <= c:
                old = new.copy()
                for q in range(j, c + 1):
                    new[q] -= old[q - j]
        prob = float(np.sum(new))
        p_mass_at_c = new[c] / 2
    if alternative != 'two-sided':
        if in_right_tail == alternative_greater:
            prob /= 2
        else:
            prob = 1 - prob / 2 + p_mass_at_c
    return min(max(prob, 0.0), 1.0)

def _dense_ranks_sorted(v):
    # v sorted -> dense ranks starting at 1
    n = v.shape[0]
    r = np.empty(n, dtype=int)
    c = 0
    for i in range(n):
        if i == 0 or v[i] != v[i - 1]:
            c += 1
        r[i] = c
    return r

def _count_rank_tie(ranks):
    mx = 0
    for r in ranks:
        if r > mx:
            mx = r
    cnt = np.zeros(mx + 1, dtype=int)
    for r in ranks:
        cnt[r] += 1
    a = 0
    b = 0
    c = 0
    for q in cnt:
        if q > 1:
            a += q * (q - 1) // 2
            b += q * (q - 1) * (q - 2)
            c += q * (q - 1) * (2 * q + 5)
    return a, b, c

def _kendall_dis(x, y) -> int:
    # number of discordant pairs: x sorted (ties broken by y order),
    # count pairs i<j with x[i]<x[j] and y[i]>y[j] (Fenwick tree)
    n = x.shape[0]
    mx = 0
    for v in y:
        if v > mx:
            mx = v
    tree = [0] * (mx + 2)
    dis = 0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[j + 1] == x[i]:
            j += 1
        # query block against previous blocks
        for q in range(i, j + 1):
            # count previous with y > y[q]
            s = 0
            k = int(y[q])
            while k > 0:
                s += tree[k]
                k -= k & -k
            dis += i - s
        for q in range(i, j + 1):
            k = int(y[q])
            while k <= mx:
                tree[k] += 1
                k += k & -k
        i = j + 1
    return dis

def _kendalltau1(x0, y0, method: str, variant: str, alternative: str):
    size = x0.shape[0]
    if size != y0.shape[0]:
        raise ValueError("Array shapes are incompatible for broadcasting.")
    if size == 0:
        return (NAN, NAN)
    perm = np.argsort(y0)
    xs = x0[perm]
    ys = y0[perm]
    yr = _dense_ranks_sorted(ys)
    perm2 = np.argsort(xs, kind='mergesort')
    xs2 = xs[perm2]
    y = yr[perm2]
    x = _dense_ranks_sorted(xs2)
    dis = _kendall_dis(x, y)
    ntie = 0
    i = 0
    while i < size:
        j = i
        while j + 1 < size and x[j + 1] == x[i] and y[j + 1] == y[i]:
            j += 1
        c = j - i + 1
        ntie += c * (c - 1) // 2
        i = j + 1
    xtie, x0t, x1t = _count_rank_tie(x)
    ytie, y0t, y1t = _count_rank_tie(y)
    tot = (size * (size - 1)) // 2
    if xtie == tot or ytie == tot:
        return (NAN, NAN)
    con_minus_dis = tot - xtie - ytie + ntie - 2 * dis
    tau = 0.0
    if variant == 'b':
        tau = con_minus_dis / K.sqrt(float(tot - xtie)) / K.sqrt(float(tot - ytie))
    elif variant == 'c':
        nx = x[size - 1]
        ny = 0
        for v in y:
            if v > ny:
                ny = v
        minclasses = min(nx, ny)
        tau = 2 * con_minus_dis / (size ** 2 * (minclasses - 1) / minclasses)
    else:
        raise ValueError(f"Unknown `variant` '{variant}' specified. `variant` must be 'b' or 'c'.")
    tau = min(1., max(-1., tau))
    m = method
    if m == 'exact' and (xtie != 0 or ytie != 0):
        raise ValueError("Ties found; exact method cannot be used.")
    if m == 'auto':
        if (xtie == 0 and ytie == 0) and (size <= 33 or min(dis, tot - dis) <= 1):
            m = 'exact'
        else:
            m = 'asymptotic'
    pvalue = 0.0
    if xtie == 0 and ytie == 0 and m == 'exact':
        pvalue = _kendall_p_exact(size, tot - dis, alternative)
    elif m == 'asymptotic':
        mm = size * (size - 1.)
        var = ((mm * (2 * size + 5) - x1t - y1t) / 18 +
               (2 * xtie * ytie) / mm + x0t * y0t / (9 * mm * (size - 2)))
        z = con_minus_dis / K.sqrt(var)
        pvalue = _pvalue_norm(z, alternative)
    else:
        raise ValueError(f"Unknown `method` '{method}' specified.  `method` must be'auto', 'exact' or 'asymptotic'.")
    return (tau, pvalue)

class _KT:
    method: str
    variant: str
    alternative: str
    def __call__(self, x, y):
        return _kendalltau1(x, y, self.method, self.variant, self.alternative)

def kendalltau(x, y, *, nan_policy: str = 'propagate', method: str = 'auto',
               variant: str = 'b', alternative: str = 'two-sided', axis = None):
    r = _map_slices2(x, y, axis, _KT(method, variant, alternative))
    return SignificanceResult(r[0], r[1])

# ------------------------------------------------------------------ robust slopes

def _median_sorted(v) -> float:
    n = v.shape[0]
    if n == 0:
        return NAN
    if n % 2 == 1:
        return v[n // 2]
    return float(np.mean(v[n // 2 - 1:n // 2 + 1]))

def _median(v) -> float:
    for q in v:
        if q != q:
            return NAN
    return _median_sorted(np.sort(v))

def _repeat_counts(v):
    s = np.sort(v)
    out = []
    n = s.shape[0]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        if j > i:
            out.append(j - i + 1)
        i = j + 1
    return out

def theilslopes(y, x = None, alpha: float = 0.95, method: str = 'separate', *, axis = None):
    if method != 'joint' and method != 'separate':
        raise ValueError(f"method must be either 'joint' or 'separate'.'{method}' is invalid.")
    yy = _f64(y).ravel()
    xx = np.arange(yy.shape[0]).astype(float)
    if x is not None:
        xx = _f64(x).ravel()
        if xx.shape[0] != yy.shape[0]:
            raise ValueError("Array shapes are incompatible for broadcasting.")
    n = yy.shape[0]
    if n < 2:
        raise ValueError("`x` and `y` must have length at least 2.")
    sl = []
    for i in range(n):
        for j in range(n):
            dx = xx[i] - xx[j]
            if dx > 0:
                sl.append((yy[i] - yy[j]) / dx)
    slopes = np.sort(np.array(sl, dtype=float))
    medslope = _median_sorted(slopes)
    medinter = 0.0
    if method == 'joint':
        medinter = _median(yy - medslope * xx)
    else:
        medinter = _median(yy) - medslope * _median(xx)
    al = alpha
    if al > 0.5:
        al = 1. - al
    z = K.ndtri(al / 2.)
    nt = slopes.shape[0]
    ny = float(n)
    sx = 0.0
    for k in _repeat_counts(xx):
        sx += k * (k - 1) * (2 * k + 5)
    sy = 0.0
    for k in _repeat_counts(yy):
        sy += k * (k - 1) * (2 * k + 5)
    sigsq = 1 / 18. * (ny * (ny - 1) * (2 * ny + 5) - sx - sy)
    lo = NAN
    hi = NAN
    if sigsq >= 0 and nt > 0:
        sigma = K.sqrt(sigsq)
        Ru = min(int(rint((nt - z * sigma) / 2.)), nt - 1)
        Rl = max(int(rint((nt + z * sigma) / 2.)) - 1, 0)
        if 0 <= Rl < nt and 0 <= Ru < nt:
            lo = slopes[Rl]
            hi = slopes[Ru]
    return TheilslopesResult(medslope, medinter, lo, hi)

def siegelslopes(y, x = None, method: str = 'hierarchical', *, axis = None):
    if method != 'hierarchical' and method != 'separate':
        raise ValueError("method can only be 'hierarchical' or 'separate'")
    yy = _f64(y).ravel()
    xx = np.arange(yy.shape[0]).astype(float)
    if x is not None:
        xx = _f64(x).ravel()
        if xx.shape[0] != yy.shape[0]:
            raise ValueError("Array shapes are incompatible for broadcasting.")
    n = yy.shape[0]
    if n < 2:
        raise ValueError("`x` and `y` must have length at least 2.")
    slopes = np.empty(n, dtype=float)
    inters = np.empty(n, dtype=float)
    for j in range(n):
        sj = []
        ij = []
        for i in range(n):
            dx = xx[j] - xx[i]
            if dx != 0:
                sj.append((yy[j] - yy[i]) / dx)
                if method == 'separate':
                    ij.append((yy[i] * xx[j] - yy[j] * xx[i]) / dx)
        slopes[j] = _median(np.array(sj, dtype=float))
        if method == 'separate':
            inters[j] = _median(np.array(ij, dtype=float))
    medslope = _median(slopes)
    medinter = 0.0
    if method == 'separate':
        medinter = _median(inters)
    else:
        medinter = _median(yy - medslope * xx)
    return SiegelslopesResult(medslope, medinter)

# ------------------------------------------------------------------ combine p-values

class _Combine:
    method: str
    def __call__(self, p, w):
        n = float(p.shape[0])
        if self.method == 'fisher':
            st = -2 * float(np.sum(np.log(p)))
            return (st, K.chdtrc(2 * n, st))
        elif self.method == 'pearson':
            st = 2 * float(np.sum(np.log1p(-p)))
            return (st, K.chdtr(2 * n, -st))
        elif self.method == 'mudholkar_george':
            nf = K.sqrt(3 / n) / 3.141592653589793
            st = -float(np.sum(np.log(p))) + float(np.sum(np.log1p(-p)))
            nu = 5 * n + 4
            af = K.sqrt(nu / (nu - 2))
            return (st, K.stdtr(nu, -(st * nf * af)))
        elif self.method == 'tippett':
            st = float(np.min(p))
            return (st, K.betainc(1.0, n, st))
        elif self.method == 'stouffer':
            Zi = np.empty(p.shape[0], dtype=float)
            for i in range(p.shape[0]):
                Zi[i] = -K.ndtri(p[i])
            st = float(np.sum(w * Zi)) / float(np.linalg.norm(w))
            return (st, K.ndtr(-st))
        else:
            raise ValueError(f"Invalid method {self.method!r}. Valid methods are 'fisher', 'pearson', 'mudholkar_george', 'tippett', and 'stouffer'")

def combine_pvalues(pvalues, method: str = 'fisher', weights = None, *, axis = 0):
    p = _f64(pvalues)
    w = np.ones(p.shape, dtype=float)
    if weights is not None:
        w = np.broadcast_to(_f64(weights), p.shape)
    r = _map_slices2(p, w, axis, _Combine(method))
    return SignificanceResult(r[0], r[1])
