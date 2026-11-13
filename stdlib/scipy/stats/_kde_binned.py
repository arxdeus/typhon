# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# gaussian_kde, binned_statistic(_2d/_dd), weightedtau.

import numpy as np
import scipy.special._kernels as K
from scipy.linalg import cholesky, solve_triangular, cho_factor, cho_solve
from ._common import _f64
from ._stats import SignificanceResult

NAN = K.NAN
INF = K.INF
_PI = 3.141592653589793

# ------------------------------------------------------------------ gaussian_kde

class gaussian_kde:
    dataset: np.ndarray[float, 2]
    d: int
    n: int
    weights: np.ndarray[float, 1]
    neff: float
    factor: float
    covariance: np.ndarray[float, 2]
    cho_cov: np.ndarray[float, 2]
    log_det: float
    _data_covariance: np.ndarray[float, 2]
    _data_cho_cov: np.ndarray[float, 2]
    _bw_kind: int
    _bw_const: float

    def __init__(self, dataset, bw_method = None, weights = None):
        ds = np.atleast_2d(_f64(dataset))
        self.dataset = ds
        if not ds.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")
        self.d = ds.shape[0]
        self.n = ds.shape[1]
        if weights is not None:
            w = np.atleast_1d(_f64(weights)).astype(float)
            w = w / float(np.sum(w))
            if w.shape[0] != self.n:
                raise ValueError("`weights` input should be of length n")
            self.weights = w
            self.neff = 1 / float(np.dot(w, w))
        else:
            self.weights = np.full(self.n, 1.0 / self.n)
            self.neff = float(self.n)
        if self.d > self.n:
            raise ValueError("Number of dimensions is greater than number of samples.")
        self._bw_kind = 0
        self._bw_const = 0.0
        self._data_covariance = np.zeros((self.d, self.d))
        self._data_cho_cov = np.zeros((self.d, self.d))
        self._set_bw(bw_method)
        self._data_covariance = np.atleast_2d(np.cov(self.dataset, rowvar=True, bias=False,
                                                     aweights=self.weights))
        self._data_cho_cov = cholesky(self._data_covariance, lower=True)
        self._compute_covariance()

    def _set_bw(self, bw_method):
        if bw_method is None:
            self._bw_kind = 0
        elif isinstance(bw_method, str):
            if bw_method == 'scott':
                self._bw_kind = 0
            elif bw_method == 'silverman':
                self._bw_kind = 1
            else:
                raise ValueError("`bw_method` should be 'scott', 'silverman', a scalar or a callable.")
        else:
            self._bw_kind = 2
            self._bw_const = float(bw_method)

    def scotts_factor(self) -> float:
        return self.neff ** (-1. / (self.d + 4))

    def silverman_factor(self) -> float:
        return (self.neff * (self.d + 2.0) / 4.0) ** (-1. / (self.d + 4))

    def covariance_factor(self) -> float:
        if self._bw_kind == 0:
            return self.scotts_factor()
        elif self._bw_kind == 1:
            return self.silverman_factor()
        return self._bw_const

    def _compute_covariance(self):
        self.factor = self.covariance_factor()
        self.covariance = self._data_covariance * self.factor ** 2
        self.cho_cov = self._data_cho_cov * self.factor
        ld = 0.0
        acc = np.empty(self.d, dtype=float)
        for i in range(self.d):
            acc[i] = K.log(self.cho_cov[i, i] * K.sqrt(2 * _PI))
        self.log_det = 2 * float(np.sum(acc))

    def set_bandwidth(self, bw_method = None):
        if bw_method is not None:
            self._set_bw(bw_method)
        self._compute_covariance()

    def _points(self, pts):
        p = np.atleast_2d(_f64(pts))
        if p.shape[0] != self.d:
            if p.shape[0] == 1 and p.shape[1] == self.d:
                return p.reshape(self.d, 1)
            raise ValueError(f"points have dimension {p.shape[0]}, dataset has dimension {self.d}")
        return p

    def evaluate(self, points):
        p = self._points(points)
        m = p.shape[1]
        pts = solve_triangular(self.cho_cov, self.dataset, lower=True).T
        xi = solve_triangular(self.cho_cov, p, lower=True).T
        norm = (2 * _PI) ** (-self.d / 2.)
        for i in range(self.d):
            norm /= self.cho_cov[i, i]
        est = np.zeros(m, dtype=float)
        for i in range(self.n):
            for j in range(m):
                arg = 0.0
                for k in range(self.d):
                    r = pts[i, k] - xi[j, k]
                    arg += r * r
                arg = K.exp(-arg / 2.) * norm
                est[j] += self.weights[i] * arg
        return est

    def __call__(self, points):
        return self.evaluate(points)

    def pdf(self, x):
        return self.evaluate(x)

    def logpdf(self, x):
        p = self._points(x)
        m = p.shape[1]
        pts = solve_triangular(self.cho_cov, self.dataset, lower=True).T
        xi = solve_triangular(self.cho_cov, p, lower=True).T
        log_norm = (-self.d / 2) * K.log(2 * _PI)
        for i in range(self.d):
            log_norm -= K.log(self.cho_cov[i, i])
        est = np.full(m, -INF)
        for i in range(self.n):
            lv = K.log(self.weights[i])
            for j in range(m):
                arg = 0.0
                for k in range(self.d):
                    r = pts[i, k] - xi[j, k]
                    arg += r * r
                b = -arg / 2 + log_norm + lv
                a = est[j]
                c = max(a, b)
                est[j] = c + K.log(K.exp(a - c) + K.exp(b - c))
        return est

    def integrate_box_1d(self, low: float, high: float) -> float:
        if self.d != 1:
            raise ValueError("integrate_box_1d() only handles 1D pdfs")
        stdev = K.sqrt(self.covariance[0, 0])
        acc = np.empty(self.n, dtype=float)
        for i in range(self.n):
            nl = (low - self.dataset[0, i]) / stdev
            nh = (high - self.dataset[0, i]) / stdev
            acc[i] = self.weights[i] * (K.ndtr(nh) - K.ndtr(nl))
        return float(np.sum(acc))

    def integrate_gaussian(self, mean, cov) -> float:
        mu = np.atleast_1d(_f64(mean)).reshape(self.d, 1)
        cv = np.atleast_2d(_f64(cov))
        sum_cov = self.covariance + cv
        ch = cho_factor(sum_cov)
        diff = self.dataset - mu
        tdiff = cho_solve(ch, diff)
        sqrt_det = 1.0
        for i in range(self.d):
            sqrt_det *= ch[0][i, i]
        norm_const = (2 * _PI) ** (sum_cov.shape[0] / 2.0) * sqrt_det
        energies = np.sum(diff * tdiff, axis=0) / 2.0
        return float(np.sum(np.exp(-energies) * self.weights)) / norm_const

# ------------------------------------------------------------------ binned statistics

class BinnedStatisticResult[S]:
    statistic: S
    bin_edges: np.ndarray[float, 1]
    binnumber: np.ndarray[int, 1]
    def __init__(self, statistic: S, bin_edges: np.ndarray[float, 1], binnumber: np.ndarray[int, 1]):
        self.statistic = statistic
        self.bin_edges = bin_edges
        self.binnumber = binnumber

class BinnedStatistic2dResult[S]:
    statistic: S
    x_edge: np.ndarray[float, 1]
    y_edge: np.ndarray[float, 1]
    binnumber: np.ndarray[int, 1]
    def __init__(self, statistic: S, x_edge: np.ndarray[float, 1], y_edge: np.ndarray[float, 1],
                 binnumber: np.ndarray[int, 1]):
        self.statistic = statistic
        self.x_edge = x_edge
        self.y_edge = y_edge
        self.binnumber = binnumber

class BinnedStatisticddResult[S]:
    statistic: S
    bin_edges: List[np.ndarray[float, 1]]
    binnumber: np.ndarray[int, 1]
    def __init__(self, statistic: S, bin_edges: List[np.ndarray[float, 1]], binnumber: np.ndarray[int, 1]):
        self.statistic = statistic
        self.bin_edges = bin_edges
        self.binnumber = binnumber

def _round_dec(v: float, decimal: int) -> float:
    return float(np.around(np.array([v]), decimal)[0])

def _bin_edges_numbers(sample, bins_list: List[np.ndarray[float, 1]]):
    # sample: (D, Ndim); bins_list: per-dim edges -> (nbin, binnumbers)
    Dlen = sample.shape[0]
    Ndim = sample.shape[1]
    nbin = [e.shape[0] + 1 for e in bins_list]
    binnumbers = np.zeros(Dlen, dtype=int)
    strides = [1] * Ndim
    for i in range(Ndim - 2, -1, -1):
        strides[i] = strides[i + 1] * nbin[i + 1]
    for i in range(Ndim):
        e = bins_list[i]
        de = np.diff(e)
        dmin = float(np.min(de))
        if dmin == 0:
            raise ValueError('The smallest edge difference is numerically 0.')
        decimal = int(-float(np.log10(np.array([dmin]))[0])) + 6
        last = e[e.shape[0] - 1]
        rlast = _round_dec(last, decimal)
        for r in range(Dlen):
            v = sample[r, i]
            b = 0
            # np.digitize (right=False): index such that e[b-1] <= v < e[b]
            lo = 0
            hi = e.shape[0]
            while lo < hi:
                mid = (lo + hi) // 2
                if e[mid] <= v:
                    lo = mid + 1
                else:
                    hi = mid
            b = lo
            if v >= last and _round_dec(v, decimal) == rlast:
                b -= 1
            binnumbers[r] += b * strides[i]
    return nbin, binnumbers

def _auto_edges(col, nb: int, has_range: bool, lo: float, hi: float):
    smin = lo
    smax = hi
    if not has_range:
        smin = INF
        smax = -INF
        for v in col:
            if v == v and v != INF and v != -INF:
                if v < smin:
                    smin = v
                if v > smax:
                    smax = v
    if smin == smax:
        smin -= .5
        smax += .5
    return np.linspace(smin, smax, nb + 1)

def _compute_stat(statistic: str, values, binnumbers, nbtot: int):
    D = values.shape[0]
    result = np.full(nbtot, NAN)
    if statistic == 'count':
        result = np.zeros(nbtot)
        for r in range(binnumbers.shape[0]):
            result[binnumbers[r]] += 1.0
        return result
    if statistic == 'sum':
        result = np.zeros(nbtot)
        for r in range(D):
            result[binnumbers[r]] += values[r]
        return result
    groups = [List[float]() for _ in range(nbtot)]
    for r in range(D):
        groups[binnumbers[r]].append(values[r])
    for b in range(nbtot):
        g = groups[b]
        if len(g) == 0:
            continue
        arr = np.array(g, dtype=float)
        if statistic == 'mean':
            # bincount-based: sum / count
            s = 0.0
            for v in g:
                s += v
            result[b] = s / len(g)
        elif statistic == 'std':
            s = 0.0
            for v in g:
                s += v
            m = s / len(g)
            q = 0.0
            for v in g:
                q += (v - m) * (v - m)
            result[b] = K.sqrt(q / len(g))
        elif statistic == 'median':
            sa = np.sort(arr)
            k = sa.shape[0]
            mid = (k - 1) / 2
            result[b] = (sa[int(K.floor(mid))] + sa[int(K.ceil(mid))]) / 2
        elif statistic == 'min':
            result[b] = float(np.min(arr))
        elif statistic == 'max':
            result[b] = float(np.max(arr))
        else:
            raise ValueError(f'invalid statistic {statistic!r}')
    return result

def _core_dd(sample, values, statistic: str, bins_list: List[np.ndarray[float, 1]]):
    nbin, bn = _bin_edges_numbers(sample, bins_list)
    tot = 1
    for v in nbin:
        tot *= v
    res = _compute_stat(statistic, values, bn, tot)
    return nbin, bn, res

def _edges_for(col, bins, rng_i):
    if isinstance(bins, int):
        if rng_i is None:
            return _auto_edges(col, bins, False, 0., 0.)
        else:
            return _auto_edges(col, bins, True, float(rng_i[0]), float(rng_i[1]))
    else:
        return _f64(bins)

def binned_statistic(x, values, statistic: str = 'mean', bins = 10, range = None):
    xs = _f64(x).ravel()
    vs = _f64(values).ravel()
    sample = xs.reshape(xs.shape[0], 1)
    e = _edges_for(xs, bins, range)
    nbin, bn, res = _core_dd(sample, vs, statistic, [e])
    stat = np.ascontiguousarray(res[1:nbin[0] - 1])
    return BinnedStatisticResult(stat, e, bn)

def binned_statistic_2d(x, y, values, statistic: str = 'mean', bins = 10, range = None,
                        expand_binnumbers: bool = False):
    xs = _f64(x).ravel()
    ys = _f64(y).ravel()
    vs = _f64(values).ravel()
    n = xs.shape[0]
    sample = np.empty((n, 2))
    for i in builtins_range(n):
        sample[i, 0] = xs[i]
        sample[i, 1] = ys[i]
    bx = bins
    by = bins
    ex = np.empty(0)
    ey = np.empty(0)
    if range is None:
        ex = _edges_for(xs, bx, None)
        ey = _edges_for(ys, by, None)
    else:
        ex = _edges_for(xs, bx, range[0])
        ey = _edges_for(ys, by, range[1])
    nbin, bn, res = _core_dd(sample, vs, statistic, [ex, ey])
    full = res.reshape(nbin[0], nbin[1])
    stat = np.ascontiguousarray(full[1:nbin[0] - 1, 1:nbin[1] - 1])
    return BinnedStatistic2dResult(stat, ex, ey, bn)

def builtins_range(n: int):
    return range(n)

def binned_statistic_dd(sample, values, statistic: str = 'mean', bins = 10, range = None):
    s = _f64(sample)
    if s.ndim == 1:
        s = s.reshape(s.shape[0], 1)
    Ndim = s.shape[1]
    vs = _f64(values).ravel()
    edges = []
    for i in builtins_range(Ndim):
        col = np.ascontiguousarray(s[:, i])
        if range is None:
            edges.append(_edges_for(col, bins, None))
        else:
            edges.append(_edges_for(col, bins, range[i]))
    nbin, bn, res = _core_dd(s, vs, statistic, edges)
    # reshape to the (nbin - 2) core for up to 3 dims
    if Ndim == 1:
        return BinnedStatisticddResult(np.ascontiguousarray(res[1:nbin[0] - 1]).reshape(-1), edges, bn)
    else:
        cnt = 1
        for v in nbin:
            cnt *= (v - 2)
        out = np.empty(cnt)
        strides = [1] * Ndim
        i = Ndim - 2
        while i >= 0:
            strides[i] = strides[i + 1] * nbin[i + 1]
            i -= 1
        k = 0
        cur = [1] * Ndim
        while True:
            off = 0
            for i in builtins_range(Ndim):
                off += cur[i] * strides[i]
            out[k] = res[off]
            k += 1
            j = Ndim - 1
            while j >= 0:
                cur[j] += 1
                if cur[j] < nbin[j] - 1:
                    break
                cur[j] = 1
                j -= 1
            if j < 0:
                break
        return BinnedStatisticddResult(out, edges, bn)

# ------------------------------------------------------------------ weightedtau

def _weigh_default(r: int) -> float:
    return 1. / (1 + r)

class _WTau:
    x: np.ndarray[float, 1]
    y: np.ndarray[float, 1]
    rank: np.ndarray[int, 1]
    perm: np.ndarray[int, 1]
    temp: np.ndarray[int, 1]
    additive: bool
    exch: float

    def weigh(self, offset: int, length: int) -> float:
        if length == 1:
            return _weigh_default(self.rank[self.perm[offset]])
        length0 = length // 2
        length1 = length - length0
        middle = offset + length0
        residual = self.weigh(offset, length0)
        weight = self.weigh(middle, length1) + residual
        if self.y[self.perm[middle - 1]] < self.y[self.perm[middle]]:
            return weight
        i = 0
        j = 0
        k = 0
        while j < length0 and k < length1:
            if self.y[self.perm[offset + j]] <= self.y[self.perm[middle + k]]:
                self.temp[i] = self.perm[offset + j]
                residual -= _weigh_default(self.rank[self.temp[i]])
                j += 1
            else:
                self.temp[i] = self.perm[middle + k]
                wv = _weigh_default(self.rank[self.temp[i]])
                if self.additive:
                    self.exch += wv * (length0 - j) + residual
                else:
                    self.exch += wv * residual
                k += 1
            i += 1
        # perm[offset+i:offset+i+length0-j] = perm[offset+j:offset+length0]
        tmp = [self.perm[offset + q] for q in range(j, length0)]
        for q in range(len(tmp)):
            self.perm[offset + i + q] = tmp[q]
        for q in range(i):
            self.perm[offset + q] = self.temp[q]
        return weight

def _lexsort_xy(x, y):
    n = x.shape[0]
    idx = np.argsort(y, kind='mergesort')
    xs = x[idx]
    idx2 = np.argsort(xs, kind='mergesort')
    return idx[idx2]

def _weightedrankedtau(x, y, rank_in, has_rank: bool, additive: bool) -> float:
    n = x.shape[0]
    perm = _lexsort_xy(x, y)
    temp = np.empty(n, dtype=int)
    rank = np.empty(n, dtype=int)
    if has_rank:
        rank = rank_in
    else:
        # rank[perm[::-1][i]] = i  (inverse of reversed perm)
        for i in range(n):
            rank[perm[n - 1 - i]] = i
    # joint ties
    first = 0
    t = 0.0
    w = _weigh_default(rank[perm[first]])
    s = w
    sq = w * w
    for i in range(1, n):
        if x[perm[first]] != x[perm[i]] or y[perm[first]] != y[perm[i]]:
            t += s * (i - first - 1) if additive else (s * s - sq) / 2
            first = i
            s = 0.
            sq = 0.
        w = _weigh_default(rank[perm[i]])
        s += w
        sq += w * w
    t += s * (n - first - 1) if additive else (s * s - sq) / 2
    # ties in x
    first = 0
    u = 0.0
    w = _weigh_default(rank[perm[first]])
    s = w
    sq = w * w
    for i in range(1, n):
        if x[perm[first]] != x[perm[i]]:
            u += s * (i - first - 1) if additive else (s * s - sq) / 2
            first = i
            s = 0.
            sq = 0.
        w = _weigh_default(rank[perm[i]])
        s += w
        sq += w * w
    u += s * (n - first - 1) if additive else (s * s - sq) / 2
    if first == 0:
        return NAN
    st = _WTau(x, y, rank, perm, temp, additive, 0.0)
    st.weigh(0, n)
    perm = st.perm
    first = 0
    v = 0.0
    w = _weigh_default(rank[perm[first]])
    s = w
    sq = w * w
    for i in range(1, n):
        if y[perm[first]] != y[perm[i]]:
            v += s * (i - first - 1) if additive else (s * s - sq) / 2
            first = i
            s = 0.
            sq = 0.
        w = _weigh_default(rank[perm[i]])
        s += w
        sq += w * w
    v += s * (n - first - 1) if additive else (s * s - sq) / 2
    if first == 0:
        return NAN
    s = 0.
    sq = 0.
    for i in range(n):
        w = _weigh_default(rank[perm[i]])
        s += w
        sq += w * w
    tot = s * (n - 1) if additive else (s * s - sq) / 2
    tau = ((tot - (v + u - t)) - 2. * st.exch) / K.sqrt(tot - u) / K.sqrt(tot - v)
    return min(1., max(-1., tau))

def weightedtau(x, y, rank = True, weigher = None, additive: bool = True):
    """Weighted Kendall tau with the default hyperbolic weigher 1/(1+r)."""
    xs = _f64(x).ravel()
    ys = _f64(y).ravel()
    if xs.shape[0] != ys.shape[0]:
        raise ValueError("Array shapes are incompatible for broadcasting.")
    if xs.shape[0] == 0:
        return SignificanceResult(NAN, NAN)
    dummy = np.empty(0, dtype=int)
    if isinstance(rank, bool):
        if rank:
            tau = (_weightedrankedtau(xs, ys, dummy, False, additive) +
                   _weightedrankedtau(ys, xs, dummy, False, additive)) / 2
            return SignificanceResult(tau, NAN)
        else:
            return SignificanceResult(_weightedrankedtau(xs, ys, np.arange(xs.shape[0]), True, additive), NAN)
    else:
        r = np.asarray(rank).astype(int)
        return SignificanceResult(_weightedrankedtau(xs, ys, r, True, additive), NAN)
