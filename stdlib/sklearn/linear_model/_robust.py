# Ported from scikit-learn 1.9.1 (BSD-3-Clause)
#
# RANSACRegressor (_ransac.py) and TheilSenRegressor (_theil_sen.py).
# random_state: None / int (numpy legacy RandomState streams reproduced).
# RANSAC: estimator defaults to LinearRegression; is_data_valid /
# is_model_valid callables optional; loss 'absolute_error' / 'squared_error'.

import math
import numpy as np
from numpy.random import RandomState
from ._base import LinearRegression, _check_sw, _r2

_EPSILON = 2.220446049250313e-16

def sample_without_replacement(n_population: int, n_samples: int, rs: RandomState) -> np.ndarray[int, 1]:
    """sklearn.utils.random.sample_without_replacement(method='auto')."""
    ratio = float(n_samples) / float(n_population) if n_population != 0 else 1.0
    if ratio > 0.01 and ratio < 0.99:
        return rs.permutation(n_population)[:n_samples].copy()
    out = np.empty(n_samples, dtype=int)
    if ratio < 0.2:
        sel = Set[int]()
        for i in range(n_samples):
            j = int(rs.randint(n_population))
            while j in sel:
                j = int(rs.randint(n_population))
            sel.add(j)
            out[i] = j
    else:
        for i in range(n_samples):
            out[i] = i
        for i in range(n_samples, n_population):
            j = int(rs.randint(0, i + 1))
            if j < n_samples:
                out[j] = i
    return out

def _dynamic_max_trials(n_inliers: int, n_samples: int, min_samples: int, probability: float) -> float:
    inlier_ratio = float(n_inliers) / float(n_samples)
    nom = max(_EPSILON, 1.0 - probability)
    denom = max(_EPSILON, 1.0 - inlier_ratio ** float(min_samples))
    if nom == 1.0:
        return 0.0
    if denom == 1.0:
        return math.inf
    return abs(float(math.ceil(math.log(nom) / math.log(denom))))

class RANSACRegressor[E = LinearRegression, DV = NoneType, MV = NoneType]:
    estimator: E
    min_samples: Optional[float]
    residual_threshold: Optional[float]
    is_data_valid: DV
    is_model_valid: MV
    max_trials: int
    max_skips: float
    stop_n_inliers: float
    stop_score: float
    stop_probability: float
    loss: str
    random_state: Optional[int]
    estimator_: E
    inlier_mask_: np.ndarray[bool, 1]
    n_trials_: int
    n_skips_no_inliers_: int
    n_skips_invalid_data_: int
    n_skips_invalid_model_: int
    n_features_in_: int

    def __init__(self, estimator: E = LinearRegression(), *, min_samples: Optional[float] = None,
                 residual_threshold: Optional[float] = None, is_data_valid: DV = None,
                 is_model_valid: MV = None, max_trials: int = 100, max_skips: float = math.inf,
                 stop_n_inliers: float = math.inf, stop_score: float = math.inf,
                 stop_probability: float = 0.99, loss: str = "absolute_error",
                 random_state: Optional[int] = None):
        self.estimator = estimator
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.is_data_valid = is_data_valid
        self.is_model_valid = is_model_valid
        self.max_trials = max_trials
        self.max_skips = max_skips
        self.stop_n_inliers = stop_n_inliers
        self.stop_score = stop_score
        self.stop_probability = stop_probability
        self.loss = loss
        self.random_state = random_state
        self.estimator_ = estimator
        self.inlier_mask_ = np.zeros(0, dtype=bool)
        self.n_trials_ = 0
        self.n_skips_no_inliers_ = 0
        self.n_skips_invalid_data_ = 0
        self.n_skips_invalid_model_ = 0
        self.n_features_in_ = 0

    def fit(self, X, y, sample_weight = None):
        from ..base import clone
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        ya = np.asarray(y, dtype=float)
        n, p = Xa.shape
        est = clone(self.estimator)
        min_samples = 0
        if self.min_samples is None:
            if not isinstance(est, LinearRegression):
                raise ValueError("`min_samples` needs to be explicitly set when estimator is not a LinearRegression.")
            min_samples = p + 1
        else:
            ms = self.min_samples.__val__()
            if 0.0 < ms and ms < 1.0:
                min_samples = int(math.ceil(ms * float(n)))
            else:
                min_samples = int(ms)
        if min_samples > n:
            raise ValueError(f"`min_samples` may not be larger than number of samples: n_samples = {n}.")
        if self.residual_threshold is None:
            residual_threshold = float(np.median(np.abs(ya - np.median(ya))))
        else:
            residual_threshold = self.residual_threshold.__val__()
        if self.loss != "absolute_error" and self.loss != "squared_error":
            raise ValueError(f"loss should be 'absolute_error' or 'squared_error', got {self.loss!r}.")
        rs = RandomState(self.random_state.__val__()) if self.random_state is not None else RandomState()
        if hasattr(est, "random_state"):
            pass
        n_inliers_best = 1
        score_best = -math.inf
        has_best = False
        mask_best = np.zeros(n, dtype=bool)
        idx_best = np.zeros(0, dtype=int)
        self.n_skips_no_inliers_ = 0
        self.n_skips_invalid_data_ = 0
        self.n_skips_invalid_model_ = 0
        self.n_trials_ = 0
        max_trials = float(self.max_trials)
        sw = _check_sw(sample_weight, n)
        has_sw = sample_weight is not None
        while float(self.n_trials_) < max_trials:
            self.n_trials_ += 1
            if float(self.n_skips_no_inliers_ + self.n_skips_invalid_data_ + self.n_skips_invalid_model_) > self.max_skips:
                break
            sub = sample_without_replacement(n, min_samples, rs)
            Xs = Xa[sub]
            ys = ya[sub]
            if not isinstance(self.is_data_valid, NoneType):
                if not self.is_data_valid(Xs, ys):
                    self.n_skips_invalid_data_ += 1
                    continue
            if has_sw:
                est.fit(Xs, ys, sample_weight=sw[sub])
            else:
                est.fit(Xs, ys)
            if not isinstance(self.is_model_valid, NoneType):
                if not self.is_model_valid(est, Xs, ys):
                    self.n_skips_invalid_model_ += 1
                    continue
            yp = np.asarray(est.predict(Xa), dtype=float)
            res = np.abs(ya - yp) if self.loss == "absolute_error" else (ya - yp) ** 2
            mask = res <= residual_threshold
            nin = int(np.sum(mask))
            if nin < n_inliers_best:
                self.n_skips_no_inliers_ += 1
                continue
            ii = np.nonzero(mask)[0]
            if has_sw:
                score = est.score(Xa[ii], ya[ii], sample_weight=sw[ii])
            else:
                score = est.score(Xa[ii], ya[ii])
            if nin == n_inliers_best and score < score_best:
                continue
            n_inliers_best = nin
            score_best = score
            mask_best = mask
            idx_best = ii
            has_best = True
            max_trials = min(max_trials, _dynamic_max_trials(n_inliers_best, n, min_samples, self.stop_probability))
            if float(n_inliers_best) >= self.stop_n_inliers or score_best >= self.stop_score:
                break
        if not has_best:
            if float(self.n_skips_no_inliers_ + self.n_skips_invalid_data_ + self.n_skips_invalid_model_) > self.max_skips:
                raise ValueError("RANSAC skipped more iterations than `max_skips` without finding a valid consensus set. Iterations were skipped because each randomly chosen sub-sample failed the passing criteria. See estimator attributes for diagnostics (n_skips*).")
            raise ValueError("RANSAC could not find a valid consensus set. All `max_trials` iterations were skipped because each randomly chosen sub-sample failed the passing criteria. See estimator attributes for diagnostics (n_skips*).")
        if has_sw:
            est.fit(Xa[idx_best], ya[idx_best], sample_weight=sw[idx_best])
        else:
            est.fit(Xa[idx_best], ya[idx_best])
        self.estimator_ = est
        self.inlier_mask_ = mask_best
        self.n_features_in_ = p
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def score(self, X, y, sample_weight = None):
        return self.estimator_.score(X, y)

def _modified_weiszfeld_step(X: np.ndarray[float, 2], x_old: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
    diff = X - x_old
    dn = np.sqrt(np.sum(diff ** 2, axis=1))
    keep = List[int]()
    for i in range(X.shape[0]):
        if dn[i] >= _EPSILON:
            keep.append(i)
    ki = np.array(keep, dtype=int)
    is_in = 1.0 if ki.shape[0] < X.shape[0] else 0.0
    d = diff[ki]
    dnk = dn[ki][:, None]
    qn = float(np.linalg.norm(np.sum(d / dnk, axis=0)))
    if qn > _EPSILON:
        nd = np.sum(X[ki, :] / dnk, axis=0) / np.sum(1.0 / dnk, axis=0)
        return max(0.0, 1.0 - is_in / qn) * nd + min(1.0, is_in / qn) * x_old
    qn = 1.0
    return max(0.0, 1.0 - is_in / qn) * 1.0 + min(1.0, is_in / qn) * x_old

def _spatial_median(X: np.ndarray[float, 2], max_iter: int, tol: float):
    if X.shape[1] == 1:
        return 1, np.array([float(np.median(X.ravel()))])
    tol2 = tol * tol
    old = np.mean(X, axis=0)
    med = old
    n_iter = 0
    for it in range(max_iter):
        n_iter = it
        med = _modified_weiszfeld_step(X, old)
        if float(np.sum((old - med) ** 2)) < tol2:
            break
        else:
            old = med
    return n_iter, med

def _breakdown_point(n_samples: int, n_subsamples: int) -> float:
    return 1.0 - (0.5 ** (1.0 / float(n_subsamples)) * float(n_samples - n_subsamples + 1)
                  + float(n_subsamples) - 1.0) / float(n_samples)

def _gelss_x(A: np.ndarray[float, 2], b: np.ndarray[float, 1], nf: int) -> np.ndarray[float, 1]:
    """lapack gelss(A, b)[1][:nf] (default rcond -1, i.e. machine precision)."""
    from scipy.linalg._lapack import dgelss_
    from numpy.linalg.blas import fint
    m, n = A.shape
    ldb = max(m, n, 1)
    Af = Ptr[float](max(m * n, 1))
    for j in range(n):
        for i in range(m):
            Af[i + j * m] = A[i, j]
    B = Ptr[float](ldb)
    for i in range(ldb):
        B[i] = b[i] if i < b.shape[0] else 0.0
    k = min(m, n)
    S = Ptr[float](max(k, 1))
    M = fint(m); N = fint(n); NR = fint(1); LDA = fint(max(m, 1)); LDB = fint(ldb)
    rank = fint(0); info = fint(0)
    rcond = -1.0
    q = 0.0
    LW = fint(-1)
    dgelss_(__ptr__(M).as_byte(), __ptr__(N).as_byte(), __ptr__(NR).as_byte(), Af.as_byte(), __ptr__(LDA).as_byte(), B.as_byte(), __ptr__(LDB).as_byte(), S.as_byte(), __ptr__(rcond).as_byte(), __ptr__(rank).as_byte(), __ptr__(q).as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
    lw = max(int(q), 1)
    W = Ptr[float](lw)
    LW = fint(lw)
    dgelss_(__ptr__(M).as_byte(), __ptr__(N).as_byte(), __ptr__(NR).as_byte(), Af.as_byte(), __ptr__(LDA).as_byte(), B.as_byte(), __ptr__(LDB).as_byte(), S.as_byte(), __ptr__(rcond).as_byte(), __ptr__(rank).as_byte(), W.as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
    out = np.empty(nf)
    for i in range(nf):
        out[i] = B[i]
    return out

def _combinations(n: int, k: int) -> List[List[int]]:
    # itertools.combinations(range(n), k) in lexicographic order
    res = List[List[int]]()
    idx = [i for i in range(k)]
    while True:
        res.append(idx.copy())
        i = k - 1
        while i >= 0 and idx[i] == i + n - k:
            i -= 1
        if i < 0:
            break
        idx[i] += 1
        for j in range(i + 1, k):
            idx[j] = idx[j - 1] + 1
    return res

class TheilSenRegressor:
    fit_intercept: bool
    max_subpopulation: float
    n_subsamples: Optional[int]
    max_iter: int
    tol: float
    random_state: Optional[int]
    coef_: np.ndarray[float, 1]
    intercept_: float
    breakdown_: float
    n_iter_: int
    n_subpopulation_: int
    n_features_in_: int

    def __init__(self, *, fit_intercept: bool = True, copy_X = "deprecated",
                 max_subpopulation: float = 1e4, n_subsamples: Optional[int] = None,
                 max_iter: int = 300, tol: float = 1.0e-3, random_state: Optional[int] = None,
                 n_jobs = None, verbose: bool = False):
        self.fit_intercept = fit_intercept
        self.max_subpopulation = max_subpopulation
        self.n_subsamples = n_subsamples
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.coef_ = np.zeros(0)
        self.intercept_ = 0.0
        self.breakdown_ = 0.0
        self.n_iter_ = 0
        self.n_subpopulation_ = 0
        self.n_features_in_ = 0

    def fit(self, X, y):
        from scipy.special import binom
        rs = RandomState(self.random_state.__val__()) if self.random_state is not None else RandomState()
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        ya = np.asarray(y, dtype=float)
        n, p = Xa.shape
        n_dim = p + 1 if self.fit_intercept else p
        if self.n_subsamples is not None:
            ns = self.n_subsamples.__val__()
            if ns > n:
                raise ValueError(f"Invalid parameter since n_subsamples > n_samples ({ns} > {n}).")
            if n >= p:
                if n_dim > ns:
                    plus = "+1" if self.fit_intercept else ""
                    raise ValueError(f"Invalid parameter since n_features{plus} > n_subsamples ({n_dim} > {ns}).")
            elif ns != n:
                raise ValueError(f"Invalid parameter since n_subsamples != n_samples ({ns} != {n}) while n_samples < n_features.")
        else:
            ns = min(n_dim, n)
        allc = max(1.0, float(np.rint(binom(float(n), float(ns)))))
        self.n_subpopulation_ = int(min(self.max_subpopulation, allc))
        self.breakdown_ = _breakdown_point(n, ns)
        idx = List[List[int]]()
        if float(np.rint(binom(float(n), float(ns)))) <= self.max_subpopulation:
            idx = _combinations(n, ns)
        else:
            for _ in range(self.n_subpopulation_):
                c = rs.choice(n, size=ns, replace=False)
                idx.append([int(v) for v in c])
        fi = 1 if self.fit_intercept else 0
        nf = p + fi
        W = np.empty((len(idx), nf))
        A = np.ones((ns, nf))
        b = np.zeros(max(ns, nf))
        for r in range(len(idx)):
            sub = idx[r]
            for a in range(ns):
                for j in range(p):
                    A[a, fi + j] = Xa[sub[a], j]
                b[a] = ya[sub[a]]
            w = _gelss_x(A, b, nf)
            for j in range(nf):
                W[r, j] = w[j]
        self.n_iter_, coefs = _spatial_median(W, self.max_iter, self.tol)
        if self.fit_intercept:
            self.intercept_ = coefs[0]
            self.coef_ = coefs[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = coefs
        self.n_features_in_ = p
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def score(self, X, y, sample_weight = None) -> float:
        sw = _check_sw(sample_weight, np.asarray(y).shape[0])
        return _r2(np.asarray(y, dtype=float), self.predict(X), sw, sample_weight is not None)
