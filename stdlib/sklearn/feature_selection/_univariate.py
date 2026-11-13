# Ported from scikit-learn 1.9.1 (BSD-3-Clause)
# Univariate score functions and filter selectors.

from ..base import estimator_repr as _estimator_repr
import numpy as np
from scipy.special import fdtrc, chdtrc, digamma

_F64_MIN = -1.7976931348623157e308
_F64_MAX = 1.7976931348623157e308
_F64_EPS = 2.220446049250313e-16

def _classes(y):
    return np.unique(np.asarray(y))

def f_oneway(groups: List[np.ndarray[float, 2]]):
    n_classes = len(groups)
    p = groups[0].shape[1]
    n_samples = 0
    ss_alldata = np.zeros(p)
    sums_all = np.zeros(p)
    sums: List[np.ndarray[float, 1]] = []
    for g in groups:
        n_samples += g.shape[0]
        ss_alldata += (g * g).sum(axis=0)
        s = g.sum(axis=0)
        sums.append(s)
        sums_all += s
    sq_all = sums_all ** 2
    sstot = ss_alldata - sq_all / float(n_samples)
    ssbn = np.zeros(p)
    for k in range(n_classes):
        ssbn += sums[k] ** 2 / float(groups[k].shape[0])
    ssbn -= sq_all / float(n_samples)
    sswn = sstot - ssbn
    dfbn = n_classes - 1
    dfwn = n_samples - n_classes
    msb = ssbn / float(dfbn)
    msw = sswn / float(dfwn)
    f = msb / msw
    prob = fdtrc(float(dfbn), float(dfwn), f)
    return f, prob

def f_classif(X, y):
    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y)
    groups: List[np.ndarray[float, 2]] = []
    for c in _classes(ya):
        groups.append(Xa[ya == c])
    return f_oneway(groups)

def chi2(X, y):
    Xa = np.asarray(X, dtype=float)
    if np.any(Xa < 0):
        raise ValueError("Input X must be non-negative.")
    ya = np.asarray(y)
    cls = _classes(ya)
    n = Xa.shape[0]
    Y = np.zeros((n, len(cls)))
    for i in range(n):
        for j in range(len(cls)):
            if ya[i] == cls[j]:
                Y[i, j] = 1.0
    observed = Y.T @ Xa
    feature_count = Xa.sum(axis=0).reshape((1, -1))
    class_prob = Y.mean(axis=0).reshape((1, -1))
    expected = class_prob.T @ feature_count
    chisq = (observed - expected) ** 2 / expected
    cs = chisq.sum(axis=0)
    k = observed.shape[0]
    return cs, chdtrc(float(k - 1), cs)

def r_regression(X, y, *, center: bool = True, force_finite: bool = True):
    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=float)
    n = Xa.shape[0]
    norms = np.empty(Xa.shape[1])
    if center:
        ya = ya - np.mean(ya)
        mu = Xa.mean(axis=0)
        norms = np.sqrt(np.sum(Xa * Xa, axis=0) - n * mu ** 2)
    else:
        norms = np.sqrt(np.sum(Xa * Xa, axis=0))
    cc = ya @ Xa
    cc /= norms
    cc /= float(np.sqrt(np.dot(ya, ya)))
    if force_finite:
        for i in range(cc.shape[0]):
            if np.isnan(cc[i]):
                cc[i] = 0.0
    return cc

def f_regression(X, y, *, center: bool = True, force_finite: bool = True):
    cc = r_regression(X, y, center=center, force_finite=force_finite)
    dof = np.asarray(y).shape[0] - (2 if center else 1)
    c2 = cc ** 2
    f = c2 / (1 - c2) * dof
    p = fdtrc(1.0, float(dof), f)
    if force_finite:
        for i in range(f.shape[0]):
            if np.isinf(f[i]):
                f[i] = _F64_MAX
            elif np.isnan(f[i]):
                f[i] = 0.0
                p[i] = 1.0
    return f, p

# ------------------------------------------------------------ mutual information

def _kth_cheb_radius(P, k: int):
    """distance to the k-th neighbour (self excluded), max-norm, then
    nextafter(r, 0) as in sklearn."""
    n = P.shape[0]
    d = P.shape[1]
    out = np.empty(n)
    buf = np.empty(n - 1)
    for i in range(n):
        m = 0
        for j in range(n):
            if j != i:
                dist = 0.0
                for a in range(d):
                    t = abs(P[i, a] - P[j, a])
                    if t > dist:
                        dist = t
                buf[m] = dist
                m += 1
        s = np.sort(buf)
        out[i] = np.nextafter(s[k - 1], 0.0)
    return out

def _kth_euclid_radius_1d(v, k: int):
    """NearestNeighbors(n_neighbors=k).fit(v).kneighbors() radius for 1-d
    data. sklearn picks 'brute' when k >= n // 2 (distances via
    -2xy + x^2 + y^2, so rounding differs from |x - y|), else a KD tree."""
    n = v.shape[0]
    brute = k >= n // 2
    out = np.empty(n)
    buf = np.empty(n - 1)
    for i in range(n):
        m = 0
        for j in range(n):
            if j != i:
                dist = 0.0
                if brute:
                    d2 = -2.0 * v[i] * v[j]
                    d2 += v[i] * v[i]
                    d2 += v[j] * v[j]
                    dist = np.sqrt(max(d2, 0.0))
                else:
                    dist = abs(v[i] - v[j])
                buf[m] = dist
                m += 1
        s = np.sort(buf)
        out[i] = np.nextafter(s[k - 1], 0.0)
    return out

def _count_within(x, radius):
    # KDTree(x).query_radius(x, radius, count_only=True) for 1-d data
    n = x.shape[0]
    out = np.empty(n)
    for i in range(n):
        c = 0
        for j in range(n):
            if abs(x[i] - x[j]) <= radius[i]:
                c += 1
        out[i] = float(c)
    return out

def _compute_mi_cc(x, y, n_neighbors: int) -> float:
    n = x.shape[0]
    xy = np.empty((n, 2))
    xy[:, 0] = x
    xy[:, 1] = y
    radius = _kth_cheb_radius(xy, n_neighbors)
    nx = _count_within(x, radius) - 1.0
    ny = _count_within(y, radius) - 1.0
    mi = (float(digamma(float(n))) + float(digamma(float(n_neighbors)))
          - float(np.mean(digamma(nx + 1))) - float(np.mean(digamma(ny + 1))))
    return max(0.0, mi)

def _compute_mi_cd(c, d, n_neighbors: int) -> float:
    n = c.shape[0]
    radius = np.empty(n)
    label_counts = np.empty(n)
    k_all = np.empty(n)
    for lab in np.unique(d):
        mask = d == lab
        count = int(np.sum(mask))
        if count > 1:
            k = min(n_neighbors, count - 1)
            r = _kth_euclid_radius_1d(np.ascontiguousarray(c[mask]), k)
            radius[mask] = r
            k_all[mask] = float(k)
        label_counts[mask] = float(count)
    keep = label_counts > 1
    ns = int(np.sum(keep))
    lc = label_counts[keep]
    ka = k_all[keep]
    cc = c[keep]
    rr = radius[keep]
    m_all = _count_within(cc, rr)
    mi = (float(digamma(float(ns))) + float(np.mean(digamma(ka))) - float(np.mean(digamma(lc)))
          - float(np.mean(digamma(m_all))))
    return max(0.0, mi)

def _mutual_info_score(a, b) -> float:
    ua = np.unique(a)
    ub = np.unique(b)
    if ua.shape[0] == 1 or ub.shape[0] == 1:
        return 0.0
    C = np.zeros((ua.shape[0], ub.shape[0]))
    for i in range(a.shape[0]):
        ia = int(np.searchsorted(ua, a[i]))
        ib = int(np.searchsorted(ub, b[i]))
        C[ia, ib] += 1.0
    N = float(C.sum())
    pi = C.sum(axis=1)
    pj = C.sum(axis=0)
    mi = 0.0
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            v = C[i, j]
            if v > 0:
                nm = v / N
                lo = -np.log(pi[i] * pj[j]) + np.log(float(pi.sum())) + np.log(float(pj.sum()))
                t = nm * (np.log(v) - np.log(N)) + nm * lo
                if abs(t) < _F64_EPS:
                    t = 0.0
                mi += t
    return max(mi, 0.0)

def _scale_no_mean(a):
    s = np.std(a, axis=0)
    s = np.where(s == 0.0, 1.0, s)
    return a / s

def _discrete_mask(discrete_features, n_features: int):
    mask = np.zeros(n_features, dtype=bool)
    if isinstance(discrete_features, str):
        if discrete_features != "auto":
            raise ValueError("Invalid string value for discrete_features.")
    elif isinstance(discrete_features, bool):
        mask.fill(discrete_features)
    else:
        df = np.asarray(discrete_features)
        if df.dtype is bool:
            mask = df.copy()
        else:
            for i in df:
                mask[int(i)] = True
    return mask

def _estimate_mi(X, y, discrete_features, discrete_target: bool, n_neighbors: int,
                 random_state: Optional[int]):
    Xa = np.array(X, dtype=float)
    n, p = Xa.shape
    dmask = _discrete_mask(discrete_features, p)
    cont = np.zeros(p, dtype=int)
    nc = 0
    for j in range(p):
        if not dmask[j]:
            cont[nc] = j
            nc += 1
    rng = np.random.RandomState(random_state)
    if nc > 0:
        cols = cont[:nc]
        Xc = _scale_no_mean(Xa[:, cols])
        means = np.maximum(1.0, np.mean(np.abs(Xc), axis=0))
        Xc += 1e-10 * means * rng.standard_normal(size=(n, nc))
        Xa[:, cols] = Xc
    ya = np.array(y, dtype=float)
    if not discrete_target:
        sd = float(np.std(ya))
        if sd == 0.0:
            sd = 1.0
        ya = ya / sd
        ya += 1e-10 * max(1.0, float(np.mean(np.abs(ya)))) * rng.standard_normal(size=n)
    mi = np.empty(p)
    for j in range(p):
        x = np.ascontiguousarray(Xa[:, j])
        if dmask[j] and discrete_target:
            mi[j] = _mutual_info_score(x, ya)
        elif dmask[j]:
            mi[j] = _compute_mi_cd(ya, x, n_neighbors)
        elif discrete_target:
            mi[j] = _compute_mi_cd(x, ya, n_neighbors)
        else:
            mi[j] = _compute_mi_cc(x, ya, n_neighbors)
    return mi

def mutual_info_regression(X, y, *, discrete_features = "auto", n_neighbors: int = 3,
                           copy: bool = True, random_state: Optional[int] = None, n_jobs = None):
    return _estimate_mi(X, y, discrete_features, False, n_neighbors, random_state)

def mutual_info_classif(X, y, *, discrete_features = "auto", n_neighbors: int = 3,
                        copy: bool = True, random_state: Optional[int] = None, n_jobs = None):
    return _estimate_mi(X, y, discrete_features, True, n_neighbors, random_state)

# ------------------------------------------------------------ selectors

def _clean_nans(scores):
    s = np.array(scores, dtype=float)
    for i in range(s.shape[0]):
        if np.isnan(s[i]):
            s[i] = _F64_MIN
    return s

class _SelectorMixin:
    def get_support(self, indices: Literal[bool] = False):
        m = self._get_support_mask()
        if indices:
            return np.flatnonzero(m)
        else:
            return m

    def transform(self, X):
        return np.asarray(X)[:, self._get_support_mask()]

    def inverse_transform(self, X):
        Xa = np.asarray(X, dtype=float)
        m = self._get_support_mask()
        out = np.zeros((Xa.shape[0], m.shape[0]))
        out[:, m] = Xa
        return out

    def fit_transform(self, X, y = None):
        return self.fit(X, y).transform(X)

def _run_score(func, X, y):
    if func is None:
        return f_classif(X, y)
    else:
        return func(X, y)

class _BaseFilter[F](_SelectorMixin):
    score_func: F
    scores_: np.ndarray[float, 1]
    pvalues_: Optional[np.ndarray[float, 1]]

    def _fit_scores(self, X, y):
        r = _run_score(self.score_func, np.asarray(X, dtype=float), y)
        if isinstance(r, Tuple):
            self.scores_ = np.asarray(r[0], dtype=float)
            self.pvalues_ = np.asarray(r[1], dtype=float)
        else:
            self.scores_ = np.asarray(r, dtype=float)
            self.pvalues_ = None

    def fit(self, X, y = None):
        self._fit_scores(X, y)
        return self

def _mask_percentile(scores_, percentile: float):
    n = scores_.shape[0]
    if percentile == 100:
        return np.ones(n, dtype=bool)
    elif percentile == 0:
        return np.zeros(n, dtype=bool)
    scores = _clean_nans(scores_)
    threshold = float(np.percentile(scores, 100 - percentile))
    mask = scores > threshold
    max_feats = int(n * percentile / 100)
    kept = max_feats - int(mask.sum())
    for i in range(n):
        if kept <= 0:
            break
        if scores[i] == threshold:
            mask[i] = True
            kept -= 1
    return mask

def _mask_kbest(scores_, k: int, all_: bool):
    n = scores_.shape[0]
    if all_:
        return np.ones(n, dtype=bool)
    mask = np.zeros(n, dtype=bool)
    if k == 0:
        return mask
    if k > n:
        raise ValueError(f"k should be <= n_features = {n}; got {k}. Use k='all' to return all features.")
    order = np.argsort(_clean_nans(scores_), kind="mergesort")
    for i in range(n - k, n):
        mask[order[i]] = True
    return mask

def _need_p(p: Optional[np.ndarray[float, 1]]):
    if p is None:
        raise ValueError("score_func must return p-values for this selector")
    return p.__val__()

def _mask_fdr(pv, alpha: float):
    n = pv.shape[0]
    sv = np.sort(pv)
    mx = -1.0
    found = False
    for i in range(n):
        if sv[i] <= alpha / n * (i + 1):
            if not found or sv[i] > mx:
                mx = sv[i]
            found = True
    if not found:
        return np.zeros(n, dtype=bool)
    return pv <= mx

class SelectPercentile[F](_BaseFilter[F]):
    percentile: float

    def __init__(self, score_func: F = None, *, percentile: float = 10):
        self.score_func = score_func
        self.percentile = percentile
        self.scores_ = np.empty(0)
        self.pvalues_ = None

    def _get_support_mask(self):
        return _mask_percentile(self.scores_, self.percentile)

class SelectKBest[F](_BaseFilter[F]):
    k: int
    k_all: bool

    def __init__(self, score_func: F = None, *, k = 10):
        self.score_func = score_func
        if isinstance(k, str):
            if k != "all":
                raise ValueError("k must be an int or 'all'")
            self.k = -1
            self.k_all = True
        else:
            self.k = k
            self.k_all = False
        self.scores_ = np.empty(0)
        self.pvalues_ = None

    def _get_support_mask(self):
        return _mask_kbest(self.scores_, self.k, self.k_all)

class SelectFpr[F](_BaseFilter[F]):
    alpha: float

    def __init__(self, score_func: F = None, *, alpha: float = 5e-2):
        self.score_func = score_func
        self.alpha = alpha
        self.scores_ = np.empty(0)
        self.pvalues_ = None

    def _get_support_mask(self):
        return _need_p(self.pvalues_) < self.alpha

class SelectFdr[F](_BaseFilter[F]):
    alpha: float

    def __init__(self, score_func: F = None, *, alpha: float = 5e-2):
        self.score_func = score_func
        self.alpha = alpha
        self.scores_ = np.empty(0)
        self.pvalues_ = None

    def _get_support_mask(self):
        return _mask_fdr(_need_p(self.pvalues_), self.alpha)

class SelectFwe[F](_BaseFilter[F]):
    alpha: float

    def __init__(self, score_func: F = None, *, alpha: float = 5e-2):
        self.score_func = score_func
        self.alpha = alpha
        self.scores_ = np.empty(0)
        self.pvalues_ = None

    def _get_support_mask(self):
        pv = _need_p(self.pvalues_)
        return pv < self.alpha / pv.shape[0]

class GenericUnivariateSelect[F](_BaseFilter[F]):
    mode: str
    param: float

    def __init__(self, score_func: F = None, *, mode: str = "percentile", param: float = 1e-5):
        if mode not in ("percentile", "k_best", "fpr", "fdr", "fwe"):
            raise ValueError(f"invalid mode {mode}")
        self.score_func = score_func
        self.mode = mode
        self.param = param
        self.scores_ = np.empty(0)
        self.pvalues_ = None

    def _get_support_mask(self):
        if self.mode == "percentile":
            return _mask_percentile(self.scores_, self.param)
        elif self.mode == "k_best":
            return _mask_kbest(self.scores_, int(self.param), False)
        elif self.mode == "fpr":
            return _need_p(self.pvalues_) < self.param
        elif self.mode == "fdr":
            return _mask_fdr(_need_p(self.pvalues_), self.param)
        else:
            pv = _need_p(self.pvalues_)
            return pv < self.param / pv.shape[0]

class VarianceThreshold(_SelectorMixin):
    def __repr__(self) -> str:
        return _estimator_repr(self)

    threshold: float
    variances_: np.ndarray[float, 1]

    def __init__(self, threshold: float = 0.0):
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self.threshold = threshold
        self.variances_ = np.empty(0)

    def fit(self, X, y = None):
        Xa = np.asarray(X, dtype=float)
        v = np.nanvar(Xa, axis=0)
        if self.threshold == 0:
            ptp = np.max(Xa, axis=0) - np.min(Xa, axis=0)
            for j in range(v.shape[0]):
                if np.isnan(v[j]):
                    v[j] = ptp[j]
                elif not np.isnan(ptp[j]) and ptp[j] < v[j]:
                    v[j] = ptp[j]
        ok = False
        for j in range(v.shape[0]):
            if np.isfinite(v[j]) and v[j] > self.threshold:
                ok = True
        if not ok:
            raise ValueError(f"No feature in X meets the variance threshold {self.threshold:.5f}")
        self.variances_ = v
        return self

    def _get_support_mask(self):
        return self.variances_ > self.threshold
