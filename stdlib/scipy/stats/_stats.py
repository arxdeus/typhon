# Ported from scipy 1.18.1 (BSD-3-Clause)

# Hypothesis tests and descriptive statistics.
#
# Result objects expose the scipy attribute names and unpack like scipy's
# named tuples (``t, p = ttest_ind(a, b)``). ``statistic``/``pvalue`` are
# floats for 1-D input (or axis=None) and ndarrays otherwise.

import numpy as np
import internal.static as static
import scipy.special._kernels as K
from ._common import _check_policy, _omit1, _omit2, _f64, _slices, _reduce, _reduce2, _mean, _moment, _var, \
                     _rank_avg, _tie_term, _pvalue_t, _pvalue_norm

NAN = K.NAN

class _Result2[S, P]:
    # Two-field result that unpacks to (statistic, pvalue).
    statistic: S
    pvalue: P

    def __init__(self, statistic: S, pvalue: P):
        self.statistic = statistic
        self.pvalue = pvalue

    def __getitem__(self, i: Literal[int]):
        if i == 0 or i == -2:
            return self.statistic
        elif i == 1 or i == -1:
            return self.pvalue
        else:
            compile_error("result index out of range")

    def __len__(self):
        return 2

    def __iter__(self):
        yield self.statistic
        yield self.pvalue

    def __setitem__(self, i: int, v):
        # results are immutable; defined so tuple-unpacking uses __iter__
        compile_error("result objects are read-only")

    def __repr__(self):
        return f"{self.__class__.__name__}(statistic={self.statistic}, pvalue={self.pvalue})"

class TtestResult[S, P, D](_Result2[S, P]):
    df: D
    def __init__(self, statistic: S, pvalue: P, df: D):
        super().__init__(statistic, pvalue)
        self.df = df

class SignificanceResult[S, P](_Result2[S, P]):
    def __init__(self, statistic: S, pvalue: P):
        super().__init__(statistic, pvalue)
    @property
    def correlation(self):
        return self.statistic

class PearsonRResult[S, P](_Result2[S, P]):
    def __init__(self, statistic: S, pvalue: P):
        super().__init__(statistic, pvalue)

class F_onewayResult[S, P](_Result2[S, P]):
    def __init__(self, statistic: S, pvalue: P):
        super().__init__(statistic, pvalue)

class KruskalResult[S, P](_Result2[S, P]):
    def __init__(self, statistic: S, pvalue: P):
        super().__init__(statistic, pvalue)

class MannwhitneyuResult[S, P](_Result2[S, P]):
    def __init__(self, statistic: S, pvalue: P):
        super().__init__(statistic, pvalue)

class WilcoxonResult[S, P](_Result2[S, P]):
    def __init__(self, statistic: S, pvalue: P):
        super().__init__(statistic, pvalue)

@tuple
class LinregressResult:
    slope: float
    intercept: float
    rvalue: float
    pvalue: float
    stderr: float
    intercept_stderr: float

@tuple
class DescribeResult[N, MM, M, V, S, KK]:
    nobs: N
    minmax: MM
    mean: M
    variance: V
    skewness: S
    kurtosis: KK

class Chi2ContingencyResult[E]:
    statistic: float
    pvalue: float
    dof: int
    expected_freq: E
    def __init__(self, statistic: float, pvalue: float, dof: int, expected_freq: E):
        self.statistic = statistic
        self.pvalue = pvalue
        self.dof = dof
        self.expected_freq = expected_freq
    def __getitem__(self, i: Literal[int]):
        if i == 0: return self.statistic
        elif i == 1: return self.pvalue
        elif i == 2: return self.dof
        elif i == 3: return self.expected_freq
        else: compile_error("result index out of range")

# ------------------------------------------------------------------ helpers

def _map_slices(a, axis, f):
    """f: 1-D float array -> (float, float). Returns (stat, p) scalar or arrays."""
    m, shp = _slices(a, axis)
    if static.len(shp) == 0:
        return f(np.ascontiguousarray(m))
    else:
        s = np.empty(shp, dtype=float)
        p = np.empty(shp, dtype=float)
        for idx in np.ndindex(shp):
            r = f(np.ascontiguousarray(m[idx]))
            s[idx] = r[0]
            p[idx] = r[1]
        return s, p

def _map_slices2(a, b, axis, f):
    if axis is None:
        return f(_f64(a).ravel(), _f64(b).ravel())
    else:
        ma, shp = _slices(a, axis)
        mb, shpb = _slices(b, axis)
        if static.len(shp) == 0:
            return f(np.ascontiguousarray(ma), np.ascontiguousarray(mb))
        else:
            s = np.empty(shp, dtype=float)
            p = np.empty(shp, dtype=float)
            for idx in np.ndindex(shp):
                r = f(np.ascontiguousarray(ma[idx]), np.ascontiguousarray(mb[idx]))
                s[idx] = r[0]
                p[idx] = r[1]
            return s, p

# ------------------------------------------------------------------ t-tests

def _trim_var_mean_len(a0, trim: float):
    # Yuen: winsorized variance, trimmed mean and trimmed length
    a = np.sort(a0)
    n = a.shape[0]
    g = int(n * trim)
    if g == 0:
        v = _var(a, 1)
    else:
        w = a.copy()
        for i in range(g):
            w[i] = a[g]
            w[n - 1 - i] = a[n - 1 - g]
        v = _var(w, 2 * g + 1)
    lo = int(trim * n)
    hi = n - lo
    if lo > hi:
        raise ValueError("Proportion too big.")
    m = _mean(np.ascontiguousarray(a[lo:hi]))
    return v, m, float(n - 2 * g)

def _ttest_ind_1d(a0, b0, equal_var: bool, alternative: str, trim: float = 0.0, omit: bool = False):
    a = _omit1(a0) if omit else a0
    b = _omit1(b0) if omit else b0
    n1 = float(a.shape[0])
    n2 = float(b.shape[0])
    if n1 == 0 or n2 == 0:
        return (NAN, NAN, NAN)
    if trim != 0:
        v1, m1, n1 = _trim_var_mean_len(a, trim)
        v2, m2, n2 = _trim_var_mean_len(b, trim)
    else:
        v1 = _var(a, 1) if n1 > 1 else NAN
        v2 = _var(b, 1) if n2 > 1 else NAN
        m1 = _mean(a)
        m2 = _mean(b)
    if equal_var:
        if n1 == 1:
            v1 = 0.0
        if n2 == 1:
            v2 = 0.0
        df = n1 + n2 - 2.0
        svar = ((n1 - 1) * v1 + (n2 - 1) * v2).cdiv(df)
        denom = K.sqrt(svar * (1.0 / n1 + 1.0 / n2))
    else:
        vn1 = v1 / n1
        vn2 = v2 / n2
        df = ((vn1 + vn2) ** 2).cdiv(vn1 ** 2 / (n1 - 1) + vn2 ** 2 / (n2 - 1))
        if df != df:
            df = 1.0
        denom = K.sqrt(vn1 + vn2)
    t = (m1 - m2).cdiv(denom)
    p = _pvalue_t(t, df, alternative) if t == t else NAN
    return (t, p, df)

def ttest_ind(a, b, *, axis = 0, equal_var: bool = True, nan_policy: str = 'propagate',
              permutations = None, random_state = None, alternative: str = 'two-sided',
              trim: float = 0, method = None):
    """Independent two-sample t-test; ``trim`` > 0 gives Yuen's trimmed
    t-test. Permutation/Monte Carlo ``method`` is not supported."""
    if not (0 <= trim < 0.5):
        raise ValueError("Trimming percentage should be 0 <= `trim` < .5.")
    _check_policy(a, nan_policy)
    _check_policy(b, nan_policy)
    om = nan_policy == 'omit'
    if axis is None:
        r = _ttest_ind_1d(_f64(a).ravel(), _f64(b).ravel(), equal_var, alternative, trim, om)
        return TtestResult(r[0], r[1], r[2])
    ma, shp = _slices(a, axis)
    mb, shpb = _slices(b, axis)
    if static.len(shp) == 0:
        r = _ttest_ind_1d(np.ascontiguousarray(ma), np.ascontiguousarray(mb), equal_var, alternative, trim, om)
        return TtestResult(r[0], r[1], r[2])
    else:
        shp2 = np.broadcast_shapes(shp, shpb)
        ba = np.broadcast_to(ma, shp2 + (ma.shape[-1],))
        bb = np.broadcast_to(mb, shp2 + (mb.shape[-1],))
        s = np.empty(shp2, dtype=float)
        p = np.empty(shp2, dtype=float)
        d = np.empty(shp2, dtype=float)
        for idx in np.ndindex(shp2):
            r = _ttest_ind_1d(np.ascontiguousarray(ba[idx]), np.ascontiguousarray(bb[idx]),
                              equal_var, alternative, trim, om)
            s[idx] = r[0]
            p[idx] = r[1]
            d[idx] = r[2]
        return TtestResult(s, p, d)

def _ttest_1samp_1d(a0, popmean: float, alternative: str, omit: bool = False):
    a = _omit1(a0) if omit else a0
    n = float(a.shape[0])
    df = n - 1.0
    m = _mean(a)
    v = _var(a, 1)
    denom = K.sqrt(v / n)
    t = (m - popmean).cdiv(denom)
    p = _pvalue_t(t, df, alternative) if t == t else NAN
    return (t, p, df)

def ttest_1samp(a, popmean, axis = 0, nan_policy: str = 'propagate',
                alternative: str = 'two-sided'):
    _check_policy(a, nan_policy)
    om = nan_policy == 'omit'
    if axis is None:
        r = _ttest_1samp_1d(_f64(a).ravel(), float(popmean), alternative, om)
        return TtestResult(r[0], r[1], r[2])
    m, shp = _slices(a, axis)
    if static.len(shp) == 0:
        r = _ttest_1samp_1d(np.ascontiguousarray(m), float(popmean), alternative, om)
        return TtestResult(r[0], r[1], r[2])
    else:
        pm = np.broadcast_to(np.asarray(popmean).astype(float), shp)
        s = np.empty(shp, dtype=float)
        p = np.empty(shp, dtype=float)
        d = np.empty(shp, dtype=float)
        for idx in np.ndindex(shp):
            r = _ttest_1samp_1d(np.ascontiguousarray(m[idx]), float(pm[idx]), alternative, om)
            s[idx] = r[0]
            p[idx] = r[1]
            d[idx] = r[2]
        return TtestResult(s, p, d)

def ttest_rel(a, b, axis = 0, nan_policy: str = 'propagate', alternative: str = 'two-sided'):
    d = _f64(a) - _f64(b)
    return ttest_1samp(d, 0.0, axis=axis, nan_policy=nan_policy, alternative=alternative)

# ----------------------------------------------------------- descriptive

def sem(a, axis = 0, ddof: int = 1, nan_policy: str = 'propagate'):
    def f(x) -> float:
        n = float(x.shape[0])
        return K.sqrt(_var(x, ddof)) / n ** 0.5
    return _reduce(a, axis, f)

def zscore(a, axis = 0, ddof: int = 0, nan_policy: str = 'propagate'):
    return zmap(a, a, axis=axis, ddof=ddof, nan_policy=nan_policy)

def zmap(scores, compare, axis = 0, ddof: int = 0, nan_policy: str = 'propagate'):
    s = _f64(scores)
    c = _f64(compare)
    _check_policy(c, nan_policy)
    if nan_policy == 'omit':
        m, shp = _slices(c, axis)
        if static.len(shp) == 0:
            o = _omit1(np.ascontiguousarray(m))
            return (s - float(np.mean(o))) / float(np.std(o, ddof=ddof))
        mn = np.empty(shp, dtype=float)
        sd = np.empty(shp, dtype=float)
        for idx in np.ndindex(shp):
            o = _omit1(np.ascontiguousarray(m[idx]))
            mn[idx] = float(np.mean(o))
            sd[idx] = float(np.std(o, ddof=ddof))
        ax = int(axis)
        if ax < 0:
            ax += c.ndim
        return (s - np.expand_dims(mn, ax)) / np.expand_dims(sd, ax)
    if axis is None:
        cf = c.ravel()
        mn = np.mean(cf)
        sd = np.std(cf, ddof=ddof)
        return (s - mn) / sd
    else:
        mn = np.mean(c, axis=axis, keepdims=True)
        sd = np.std(c, axis=axis, ddof=ddof, keepdims=True)
        return (s - mn) / sd

def _skew1(x, bias: bool) -> float:
    n = float(x.shape[0])
    mean = _mean(x)
    m2 = _moment(x, 2)
    m3 = _moment(x, 3)
    eps = 2.220446049250313e-16
    zero = m2 <= (eps * mean) ** 2
    val = NAN if zero else m3 / m2 ** 1.5
    if not bias and not zero and n > 2:
        val = ((n - 1.0) * n) ** 0.5 / (n - 2.0) * m3 / m2 ** 1.5
    return val

def _kurt1(x, fisher: bool, bias: bool) -> float:
    n = float(x.shape[0])
    mean = _mean(x)
    m2 = _moment(x, 2)
    m4 = _moment(x, 4)
    eps = 2.220446049250313e-16
    zero = m2 <= (eps * mean) ** 2
    val = NAN if zero else m4 / m2 ** 2.0
    if not bias and not zero and n > 3:
        val = 1.0 / (n - 2) / (n - 3) * ((n * n - 1.0) * m4 / m2 ** 2.0 - 3 * (n - 1) ** 2.0) + 3.0
    return val - 3 if fisher else val

def skew(a, axis = 0, bias: bool = True, nan_policy: str = 'propagate'):
    return _reduce(a, axis, lambda x: _skew1(x, bias))

def kurtosis(a, axis = 0, fisher: bool = True, bias: bool = True, nan_policy: str = 'propagate'):
    return _reduce(a, axis, lambda x: _kurt1(x, fisher, bias))

def describe(a, axis = 0, ddof: int = 1, bias: bool = True, nan_policy: str = 'propagate'):
    x = _f64(a)
    if axis is None:
        xf = x.ravel()
        n = xf.shape[0]
        if n == 0:
            raise ValueError("The input must not be empty.")
        mm = (float(np.min(xf)), float(np.max(xf)))
        return DescribeResult(n, mm, _mean(xf), _var(xf, ddof), _skew1(xf, bias), _kurt1(xf, True, bias))
    else:
        m, shp = _slices(x, axis)
        n = m.shape[-1]
        if n == 0:
            raise ValueError("The input must not be empty.")
        mm = (np.min(x, axis=axis), np.max(x, axis=axis))
        return DescribeResult(n, mm, _reduce(x, axis, _mean),
                              _reduce(x, axis, lambda v: _var(v, ddof)),
                              skew(x, axis, bias), kurtosis(x, axis, True, bias))

def moment(a, order: int = 1, axis = 0, nan_policy: str = 'propagate', *, center = None):
    def f(x) -> float:
        if order == 0:
            return 1.0
        if center is None:
            if order == 1:
                return 0.0
            return _moment(x, order)
        else:
            return float(np.mean((x - float(center)) ** order))
    return _reduce(a, axis, f)

# ---------------------------------------------------------------- ranking

def rankdata(a, method: str = 'average', *, axis = None, nan_policy: str = 'propagate'):
    """Ranks; methods 'average', 'min', 'max', 'dense', 'ordinal'. Returns
    float64 for 'average' and int (as float64 values) otherwise.
    Deviation: always returns a float64 array."""
    if method not in ('average', 'min', 'max', 'dense', 'ordinal'):
        raise ValueError(f'unknown method "{method}"')
    _check_policy(a, nan_policy)
    def r1(x0):
        has_nan = False
        for v in x0:
            if v != v:
                has_nan = True
        if has_nan and nan_policy == 'propagate':
            return np.full(x0.shape[0], NAN)
        x = _omit1(x0) if has_nan else x0
        n = x.shape[0]
        order = np.argsort(x, kind='mergesort')
        out = np.empty(n, dtype=float)
        i = 0
        dense = 0
        while i < n:
            j = i
            while j + 1 < n and x[order[j + 1]] == x[order[i]]:
                j += 1
            dense += 1
            for k in range(i, j + 1):
                if method == 'average':
                    out[order[k]] = 0.5 * (i + j) + 1.0
                elif method == 'min':
                    out[order[k]] = float(i + 1)
                elif method == 'max':
                    out[order[k]] = float(j + 1)
                elif method == 'dense':
                    out[order[k]] = float(dense)
                else:
                    out[order[k]] = float(k + 1)
            i = j + 1
        if has_nan:
            full = np.empty(x0.shape[0], dtype=float)
            q = 0
            for t in range(x0.shape[0]):
                if x0[t] != x0[t]:
                    full[t] = NAN
                else:
                    full[t] = out[q]
                    q += 1
            return full
        return out
    if axis is None:
        x = _f64(a)
        return r1(x.ravel())
    else:
        x = _f64(a)
        ax = int(axis)
        if ax < 0:
            ax += x.ndim
        m = np.moveaxis(x, ax, -1)
        out = np.empty(m.shape, dtype=float)
        for idx in np.ndindex(m.shape[:-1]):
            out[idx] = r1(np.ascontiguousarray(m[idx]))
        return np.moveaxis(out, -1, ax)

def tiecorrect(rankvals):
    x = _f64(rankvals).ravel()
    r, ties = _rank_avg(x)
    n = float(x.shape[0])
    if n < 2:
        return 1.0
    return 1.0 - _tie_term(ties) / (n ** 3 - n)

# ------------------------------------------------------------ correlation

def _pearson1(x, y, alternative: str):
    n = float(x.shape[0])
    if x.shape[0] != y.shape[0]:
        raise ValueError('`x` and `y` must have the same length along `axis`.')
    if n < 2:
        raise ValueError('`x` and `y` must have length at least 2.')
    const_x = True
    const_y = True
    for i in range(x.shape[0]):
        if x[i] != x[0]:
            const_x = False
        if y[i] != y[0]:
            const_y = False
    if const_x or const_y:
        return (NAN, NAN)
    xm = x - np.mean(x)
    ym = y - np.mean(y)
    xmax = float(np.max(np.abs(xm)))
    ymax = float(np.max(np.abs(ym)))
    normxm = xmax * float(np.linalg.norm(xm / xmax))
    normym = ymax * float(np.linalg.norm(ym / ymax))
    r = float(np.dot(xm / normxm, ym / normym))
    r = min(max(r, -1.0), 1.0)
    if n == 2:
        return (r, 1.0)
    ab = n / 2 - 1
    # beta(ab, ab) on (-1, 1)
    z = (r + 1.0) / 2.0
    if alternative == 'less':
        p = K.betainc(ab, ab, z)
    elif alternative == 'greater':
        p = K.betaincc(ab, ab, z)
    elif alternative == 'two-sided':
        p = 2 * K.betaincc(ab, ab, (abs(r) + 1.0) / 2.0)
    else:
        raise ValueError("`alternative` must be 'less', 'greater', or 'two-sided'.")
    return (r, p)

def _pearson_stat(u, v):
    return _pearson1(_f64(u), _f64(v), 'two-sided')[0]

def pearsonr(x, y, *, alternative: str = 'two-sided', method = None, axis = 0):
    from ._final_misc import PermutationMethod
    if isinstance(method, PermutationMethod):
        from ._resampling import permutation_test
        xa = _f64(x).ravel()
        ya = _f64(y).ravel()
        r0 = _pearson1(xa, ya, alternative)
        pt = permutation_test((xa, ya), _pearson_stat, permutation_type='pairings',
                              vectorized=False, n_resamples=method.n_resamples,
                              alternative=alternative, rng=method.rng)
        return PearsonRResult(r0[0], pt.pvalue)
    if axis is None:
        r = _pearson1(_f64(x).ravel(), _f64(y).ravel(), alternative)
        return PearsonRResult(r[0], r[1])
    res = _map_slices2(x, y, axis, lambda u, v: _pearson1(u, v, alternative))
    return PearsonRResult(res[0], res[1])

def _spearman1(x, y, alternative: str):
    rx, tx = _rank_avg(x)
    ry, ty = _rank_avg(y)
    n = x.shape[0]
    # np.corrcoef of ranks
    mx = np.mean(rx)
    my = np.mean(ry)
    dx = rx - mx
    dy = ry - my
    cxx = float(np.dot(dx, dx)) / (n - 1)
    cyy = float(np.dot(dy, dy)) / (n - 1)
    cxy = float(np.dot(dx, dy)) / (n - 1)
    d = K.sqrt(cxx) * K.sqrt(cyy)
    rs = cxy.cdiv(d)
    rs = min(max(rs, -1.0), 1.0)
    dof = float(n - 2)
    q = dof.cdiv((rs + 1.0) * (1.0 - rs))
    if q < 0:
        q = 0.0
    tt = rs * K.sqrt(q)
    p = _pvalue_t(tt, dof, alternative)
    return (rs, p)

def _spearman_matrix(cols, alternative: str):
    # cols: list of ranked 1-D variables; returns (rs, p) matrices
    nv = len(cols)
    n = cols[0].shape[0]
    dev = [c - np.mean(c) for c in cols]
    cov = np.empty((nv, nv), dtype=float)
    for i in range(nv):
        for j in range(nv):
            cov[i, j] = float(np.dot(dev[i], dev[j])) / (n - 1)
    rs = np.empty((nv, nv), dtype=float)
    pv = np.empty((nv, nv), dtype=float)
    dof = float(n - 2)
    for i in range(nv):
        for j in range(nv):
            r = cov[i, j].cdiv(K.sqrt(cov[i, i]) * K.sqrt(cov[j, j]))
            r = min(max(r, -1.0), 1.0)
            rs[i, j] = r
            q = dof.cdiv((r + 1.0) * (1.0 - r))
            if q < 0:
                q = 0.0
            pv[i, j] = _pvalue_t(r * K.sqrt(q), dof, alternative)
    return rs, pv

def _spearman_vars(x, axisout: int):
    # x 2-D, observations along axisout; returns list of variable columns
    cols = []
    if axisout == 0:
        for j in range(x.shape[1]):
            cols.append(np.ascontiguousarray(x[:, j]))
    else:
        for i in range(x.shape[0]):
            cols.append(np.ascontiguousarray(x[i, :]))
    return cols

def spearmanr(a, b = None, axis = 0, nan_policy: str = 'propagate',
              alternative: str = 'two-sided'):
    """Spearman rank correlation. Two 1-D inputs give scalar
    ``statistic``/``pvalue`` as in scipy. Any 2-D input gives the full
    correlation and p-value matrices. Deviation: scipy returns scalars when a
    2-D input holds exactly two variables, but here the result type has to be
    fixed at compile time, so read ``res.statistic[0, 1]`` in that case."""
    x = _f64(a)
    ax = 0 if axis is None else int(axis)
    if b is None:
        if x.ndim != 2:
            compile_error("spearmanr needs at least 2 variables to compare")
        cols = _spearman_vars(x, ax)
        ranked = [_rank_avg(c)[0] for c in cols]
        rs, pv = _spearman_matrix(ranked, alternative)
        return SignificanceResult(rs, pv)
    else:
        y = _f64(b)
        if x.ndim == 1 and y.ndim == 1:
            r = _spearman1(x, y, alternative)
            return SignificanceResult(r[0], r[1])
        else:
            cols = [x.ravel()] if x.ndim == 1 else _spearman_vars(x, ax)
            if y.ndim == 1:
                cols.append(y.ravel())
            else:
                cols.extend(_spearman_vars(y, ax))
            ranked = [_rank_avg(c)[0] for c in cols]
            rs, pv = _spearman_matrix(ranked, alternative)
            return SignificanceResult(rs, pv)

def linregress(x, y = None, alternative: str = 'two-sided'):
    xx = _f64(x).ravel()
    yy = _f64(y).ravel()
    n = float(xx.shape[0])
    TINY = 1.0e-20
    xmean = float(np.mean(xx))
    ymean = float(np.mean(yy))
    x_ = xx - xmean
    y_ = yy - ymean
    ssxm = float(np.dot(x_, x_)) / n
    ssym = float(np.dot(y_, y_)) / n
    ssxym = float(np.dot(x_, y_)) / n
    if ssxm == 0.0 or ssym == 0.0:
        r = NAN if ssxym == 0 else 0.0
    else:
        r = min(max(ssxym / K.sqrt(ssxm * ssym), -1.0), 1.0)
    slope = ssxym.cdiv(ssxm)
    intercept = ymean - slope * xmean
    df = n - 2
    t = r * K.sqrt(df.cdiv((1.0 - r + TINY) * (1.0 + r + TINY)))
    prob = _pvalue_t(t, df, alternative)
    slope_stderr = K.sqrt(((1 - r * r) * ssym).cdiv(ssxm).cdiv(df))
    intercept_stderr = slope_stderr * K.sqrt(ssxm + xmean * xmean)
    return LinregressResult(slope, intercept, r, prob, slope_stderr, intercept_stderr)

# --------------------------------------------------------- ANOVA / ranks

def _concat1(samples):
    total = 0
    for s in samples:
        total += s.shape[0]
    out = np.empty(total, dtype=float)
    k = 0
    for s in samples:
        for i in range(s.shape[0]):
            out[k] = s[i]
            k += 1
    return out

def _f_oneway1(ss, equal_var: bool):
    num_groups = len(ss)
    alldata = _concat1(ss)
    bign = float(alldata.shape[0])
    if equal_var:
        offset = float(np.mean(alldata))
        ad = alldata - offset
        normalized_ss = float(np.sum(ad)) ** 2. / bign
        sstot = float(np.dot(ad, ad)) - normalized_ss
        ssbn = 0.0
        for s in ss:
            ssbn = ssbn + float(np.sum(s - offset)) ** 2. / float(s.shape[0])
        ssbn = ssbn - normalized_ss
        sswn = sstot - ssbn
        dfbn = float(num_groups - 1)
        dfwn = bign - num_groups
        msb = ssbn / dfbn
        msw = sswn / dfwn
        fs = msb.cdiv(msw)
        dfn = dfbn
        dfd = dfwn
    else:
        k = float(num_groups)
        y_t = [_mean(s) for s in ss]
        n_t = [float(s.shape[0]) for s in ss]
        s_t2 = [_var(s, 1) for s in ss]
        w_t = [n_t[i] / s_t2[i] for i in range(num_groups)]
        s_w_t = 0.0
        for w in w_t:
            s_w_t += w
        y_hat = 0.0
        for i in range(num_groups):
            y_hat += w_t[i] * y_t[i]
        y_hat /= s_w_t
        num = 0.0
        for i in range(num_groups):
            num += w_t[i] * (y_t[i] - y_hat) ** 2
        num /= (k - 1)
        acc = 0.0
        for i in range(num_groups):
            acc += 1.0 / (n_t[i] - 1) * (1 - w_t[i] / s_w_t) ** 2
        den = 1 + 2 * (k - 2) / (k * k - 1) * acc
        fs = num / den
        dfn = k - 1
        dfd = (k * k - 1) / (3 * acc)
    # constant-input conventions
    all_const = True
    for s in ss:
        for i in range(1, s.shape[0]):
            if s[i] != s[0]:
                all_const = False
    if all_const:
        same = True
        for i in range(1, alldata.shape[0]):
            if alldata[i] != alldata[0]:
                same = False
        fs = K.NAN if same else K.INF
    p = K.fdtrc(dfn, dfd, fs)
    return (fs, p)

def _kruskal1(ss):
    alldata = _concat1(ss)
    ranked, t = _rank_avg(alldata)
    totaln = float(alldata.shape[0])
    ties = 1 - _tie_term(t) / (totaln ** 3 - totaln)
    ssbn = 0.0
    j = 0
    for s in ss:
        n = s.shape[0]
        acc = float(np.sum(ranked[j:j + n]))
        ssbn += acc ** 2 / float(n)
        j += n
    h = 12.0 / (totaln * (totaln + 1)) * ssbn - 3 * (totaln + 1)
    df = float(len(ss) - 1)
    h = h.cdiv(ties)
    p = K.chdtrc(df, h)
    return (h, p)

def _ksample(samples, axis, f):
    """Apply ``f(list of 1-D float arrays) -> (stat, p)`` along ``axis`` for
    N-D samples (batch dimensions broadcast)."""
    if axis is None:
        return f([_f64(s).ravel() for s in samples])
    else:
        ms = [_slices(s, axis)[0] for s in samples]
        shp = ms[0].shape[:-1]
        if static.len(shp) == 0:
            return f([np.ascontiguousarray(m) for m in ms])
        else:
            for m in ms:
                shp = np.broadcast_shapes(shp, m.shape[:-1])
            bs = [np.broadcast_to(m, shp + (m.shape[-1],)) for m in ms]
            st = np.empty(shp, dtype=float)
            pv = np.empty(shp, dtype=float)
            for idx in np.ndindex(shp):
                r = f([np.ascontiguousarray(b[idx]) for b in bs])
                st[idx] = r[0]
                pv[idx] = r[1]
            return st, pv

def f_oneway(*samples, axis = 0, equal_var: bool = True):
    """One-way ANOVA; ``equal_var=False`` gives Welch's ANOVA. N-D samples
    are reduced along ``axis``."""
    if static.len(samples) < 2:
        compile_error("at least two inputs are required")
    arrs = [_f64(s) for s in samples]
    r = _ksample(arrs, axis, lambda ss: _f_oneway1(ss, equal_var))
    return F_onewayResult(r[0], r[1])

def kruskal(*samples, nan_policy: str = 'propagate', axis = 0):
    if static.len(samples) < 2:
        compile_error("Need at least two groups in stats.kruskal()")
    arrs = [_f64(s) for s in samples]
    r = _ksample(arrs, axis, _kruskal1)
    return KruskalResult(r[0], r[1])

# MWU exact null distribution (scipy _MWU.build_u_freqs_array)
def _mwu_freqs(n1: int, n2: int, maxu: int):
    s_array = [0] * (maxu + 1)
    for d in range(1, n1 + 1):
        idx = d
        while idx <= maxu:
            s_array[idx] += d
            idx += d
    for d in range(n2 + 1, n2 + n1 + 1):
        idx = d
        while idx <= maxu:
            s_array[idx] -= d
            idx += d
    conf = [0.0] * (maxu + 1)
    conf[0] = 1.0
    for u in range(1, maxu + 1):
        acc = 0.0
        for i in range(u):
            acc += conf[i] * float(s_array[u - i])
        conf[u] = acc / u
    total = K.exp(K.gammaln(n1 + n2 + 1.0) - K.gammaln(n1 + 1.0) - K.gammaln(n2 + 1.0))
    total = float(round(total))
    return [c / total for c in conf]

def _mwu_sf(k: int, n1i: int, n2i: int) -> float:
    n1 = min(n1i, n2i)
    n2 = max(n1i, n2i)
    kc = n1 * n2 - k
    if k < kc:
        pm = _mwu_freqs(n1, n2, k)
        c = 0.0
        for v in pm:
            c += v
        return 1.0 - c + pm[k]
    else:
        pm = _mwu_freqs(n1, n2, kc)
        c = 0.0
        for v in pm:
            c += v
        return c

def _mwu1(x, y, use_continuity: bool, alternative: str, method: str):
    n1 = x.shape[0]
    n2 = y.shape[0]
    xy = np.concatenate((x, y))
    ranks, t = _rank_avg(xy)
    R1 = float(np.sum(ranks[:n1]))
    U1 = R1 - n1 * (n1 + 1) / 2
    U2 = n1 * n2 - U1
    if alternative == 'greater':
        U = U1
        fct = 1.0
    elif alternative == 'less':
        U = U2
        fct = 1.0
    elif alternative == 'two-sided':
        U = max(U1, U2)
        fct = 2.0
    else:
        raise ValueError("`alternative` must be 'less', 'greater', or 'two-sided'.")
    m = method
    if m == 'auto':
        has_ties = False
        for tv in t:
            if tv > 1:
                has_ties = True
        if n1 > 8 and n2 > 8:
            m = 'asymptotic'
        elif has_ties:
            m = 'asymptotic'
        else:
            m = 'exact'
    if m == 'exact':
        p = _mwu_sf(int(U), n1, n2) * fct
    elif m == 'asymptotic':
        mu = n1 * n2 / 2
        n = float(n1 + n2)
        tie_term = _tie_term(t)
        s = K.sqrt(n1 * n2 / 12 * ((n + 1) - tie_term / (n * (n - 1))))
        numerator = U1 - mu
        if use_continuity:
            if alternative == 'greater':
                sign = 1.0
            elif alternative == 'less':
                sign = -1.0
            else:
                sign = 1.0 if numerator > 0 else (-1.0 if numerator < 0 else 0.0)
            numerator -= 0.5 * sign
        z = numerator.cdiv(s)
        p = _pvalue_norm(z, alternative)
    else:
        raise ValueError("`method` must be one of 'auto', 'asymptotic', 'exact'")
    p = min(max(p, 0.0), 1.0)
    return (U1, p)

def mannwhitneyu(x, y, use_continuity: bool = True, alternative: str = 'two-sided',
                 axis = 0, method: str = 'auto', nan_policy: str = 'propagate'):
    _check_policy(x, nan_policy)
    _check_policy(y, nan_policy)
    om = nan_policy == 'omit'
    res = _map_slices2(x, y, axis, lambda u, v: _mwu1(_omit1(u) if om else u, _omit1(v) if om else v, use_continuity, alternative, method))
    return MannwhitneyuResult(res[0], res[1])

def _wilcoxon_pmf(n: int):
    c = [1.0]
    for k in range(1, n + 1):
        prev = c
        c = [0.0] * (k * (k + 1) // 2 + 1)
        m = len(prev)
        for i in range(m):
            c[i] = prev[i] * 0.5
        L = len(c)
        for i in range(m):
            c[L - m + i] += prev[i] * 0.5
    return c

def _wilcoxon1(d0, zero_method: str, correction: bool, alternative: str, method: str):
    n_all = d0.shape[0]
    n_zero = 0
    for v in d0:
        if v == 0:
            n_zero += 1
    if zero_method == 'wilcox':
        dl = [v for v in d0 if v != 0]
    else:
        dl = [v for v in d0]
    d = np.array(dl, dtype=float)
    count = float(d.shape[0])
    r, t = _rank_avg(np.abs(d))
    r_plus = 0.0
    r_minus = 0.0
    r_zero = 0.0
    for i in range(d.shape[0]):
        if d[i] > 0:
            r_plus += r[i]
        elif d[i] < 0:
            r_minus += r[i]
        else:
            r_zero += r[i]
    has_ties = False
    for tv in t:
        if tv > 1:
            has_ties = True
    if zero_method == 'zsplit':
        r_plus += r_zero / 2
        r_minus += r_zero / 2
    mn = count * (count + 1.) * 0.25
    se = count * (count + 1.) * (2. * count + 1.)
    tie_t = t
    if zero_method == 'pratt':
        nz = float(n_zero)
        mn -= nz * (nz + 1.) * 0.25
        se -= nz * (nz + 1.) * (2. * nz + 1.)
        # zeros are excluded from tie correction (they sort first)
        if n_zero > 0:
            tie_t = t[1:]
    se = K.sqrt((se - _tie_term(tie_t) / 2) / 24)
    m = method
    if m == 'auto':
        if n_all > 50:
            m = 'asymptotic'
        elif not (has_ties or n_zero > 0):
            m = 'exact'
        elif n_all <= 13:
            m = 'permutation'
        else:
            m = 'asymptotic'
    if m == 'asymptotic':
        z = (r_plus - mn).cdiv(se)
        if correction:
            if alternative == 'greater':
                sign = 1.0
            elif alternative == 'less':
                sign = -1.0
            else:
                sign = 1.0 if z > 0 else (-1.0 if z < 0 else 0.0)
            z -= sign * 0.5 / se
        p = _pvalue_norm(z, alternative)
    elif m == 'exact':
        pmf = _wilcoxon_pmf(int(count))
        nn = int(count)
        mean = nn * (nn + 1) / 4
        def cdf(k: int) -> float:
            if k <= mean:
                s = 0.0
                for i in range(0, min(k, len(pmf) - 1) + 1):
                    s += pmf[i]
                return s
            else:
                s = 0.0
                for i in range(k + 1, len(pmf)):
                    s += pmf[i]
                return 1 - s
        def sf(k: int) -> float:
            if k <= mean:
                s = 0.0
                for i in range(max(k, 0), len(pmf)):
                    s += pmf[i]
                return s
            else:
                s = 0.0
                for i in range(0, min(k - 1, len(pmf) - 1) + 1):
                    s += pmf[i]
                return 1 - s
        if alternative == 'less':
            p = cdf(int(K.ceil(r_plus)))
        elif alternative == 'greater':
            p = sf(int(K.floor(r_plus)))
        else:
            p = 2 * min(sf(int(K.floor(r_plus))), cdf(int(K.ceil(r_plus))))
            p = min(max(p, 0.0), 1.0)
    elif m == 'permutation':
        # exhaustive sign-flip permutation test (scipy permutation_test with
        # permutation_type='samples', all 2**n arrangements, exact p-value);
        # zeros never contribute to r_plus so only nonzero entries are flipped
        rk = [r[i] for i in range(d.shape[0]) if d[i] != 0]
        nn = len(rk)
        total = 1 << nn
        gamma = abs(2.220446049250313e-14 * r_plus)
        cnt_le = 0
        cnt_ge = 0
        base = r_zero / 2 if zero_method == 'zsplit' else 0.0
        for mask in range(total):
            s = base
            for i in range(nn):
                if (mask >> i) & 1:
                    s += rk[i]
            if s <= r_plus + gamma:
                cnt_le += 1
            if s >= r_plus - gamma:
                cnt_ge += 1
        pl = cnt_le / total
        pg = cnt_ge / total
        if alternative == 'less':
            p = pl
        elif alternative == 'greater':
            p = pg
        else:
            p = min(pl, pg) * 2
        p = min(max(p, 0.0), 1.0)
    else:
        raise ValueError("`method` must be one of 'auto', 'asymptotic', 'exact'")
    stat = min(r_plus, r_minus) if alternative == 'two-sided' else r_plus
    return (stat, p)

def wilcoxon(x, y = None, zero_method: str = 'wilcox', correction: bool = False,
             alternative: str = 'two-sided', method: str = 'auto', *, axis = 0):
    if y is None:
        d = _f64(x)
    else:
        d = _f64(x) - _f64(y)
    res = _map_slices(d, axis, lambda v: _wilcoxon1(v, zero_method, correction, alternative, method))
    return WilcoxonResult(res[0], res[1])

def _lambda_value(lambda_) -> float:
    if lambda_ is None:
        return 1.0
    elif isinstance(lambda_, str):
        if lambda_ == "pearson":
            return 1.0
        elif lambda_ == "log-likelihood":
            return 0.0
        elif lambda_ == "freeman-tukey":
            return -0.5
        elif lambda_ == "mod-log-likelihood":
            return -1.0
        elif lambda_ == "neyman":
            return -2.0
        elif lambda_ == "cressie-read":
            return 2 / 3
        else:
            raise ValueError(f"invalid string for lambda_: {lambda_!r}.")
    else:
        return float(lambda_)

def _expected_freq(obs):
    d = obs.ndim
    total = float(np.sum(obs))
    expected = np.ones(obs.shape, dtype=float)
    for k in range(d):
        axes = [j for j in range(d) if j != k]
        marg = obs
        for j in axes[::-1]:
            marg = np.sum(marg, axis=j, keepdims=True)
        expected = expected * marg
    return expected / total ** (d - 1)

def chi2_contingency(observed, correction: bool = True, lambda_ = None):
    """Chi-square test of independence for an N-D contingency table
    (``lambda_`` selects the Cressie-Read power divergence statistic)."""
    obs = _f64(observed)
    for v in obs.flat:
        if v < 0:
            raise ValueError("All values in `observed` must be nonnegative.")
    expected = _expected_freq(obs)
    for v in expected.flat:
        if v == 0:
            raise ValueError("The internally computed table of expected frequencies has a zero element.")
    ssum = 0
    for v in expected.shape:
        ssum += v
    dof = expected.size - ssum + expected.ndim - 1
    if dof == 0:
        return Chi2ContingencyResult(0.0, 1.0, 0, expected)
    o = obs
    if dof == 1 and correction:
        o = obs.copy()
        for idx in np.ndindex(obs.shape):
            diff = expected[idx] - obs[idx]
            direction = 1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0)
            magnitude = min(0.5, abs(diff))
            o[idx] = obs[idx] + magnitude * direction
    lam = _lambda_value(lambda_)
    fo = o.ravel()
    fe = expected.ravel()
    n = fo.shape[0]
    terms = np.empty(n, dtype=float)
    for i in range(n):
        a = fo[i]
        e = fe[i]
        if lam == 1:
            terms[i] = (a - e) ** 2 / e
        elif lam == 0:
            terms[i] = 2.0 * K.xlogy(a, a / e)
        elif lam == -1:
            terms[i] = 2.0 * K.xlogy(e, e / a)
        else:
            t = a * ((a / e) ** lam - 1)
            t /= 0.5 * lam * (lam + 1)
            terms[i] = t
    chi2 = float(np.sum(terms))
    p = K.chdtrc(float(dof), chi2)
    return Chi2ContingencyResult(chi2, p, dof, expected)
