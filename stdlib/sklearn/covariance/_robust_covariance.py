# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# MinCovDet (FastMCD, Rousseeuw & Van Driessen) and EllipticEnvelope, with
# the same RandomState stream as sklearn.

import numpy as np
import scipy.linalg as sl
from numpy.random.legacy import RandomState, check_random_state
from . import EmpiricalCovariance, empirical_covariance
from ..decomposition._extmath import fast_logdet

class _Est:
    location: np.ndarray[float, 1]
    covariance: np.ndarray[float, 2]
    det: float
    support: np.ndarray[bool, 1]
    dist: np.ndarray[float, 1]

    def __init__(self, location: np.ndarray[float, 1], covariance: np.ndarray[float, 2], det: float,
                 support: np.ndarray[bool, 1], dist: np.ndarray[float, 1]):
        self.location = location
        self.covariance = covariance
        self.det = det
        self.support = support
        self.dist = dist

def _mdist(X, location, precision):
    Xc = X - location
    return np.sum(np.dot(Xc, precision) * Xc, axis=1)

def _c_step(X: np.ndarray[float, 2], n_support: int, rs: RandomState, remaining_iterations: int,
            init_loc: Optional[np.ndarray[float, 1]], init_cov: Optional[np.ndarray[float, 2]]) -> _Est:
    n_samples, n_features = X.shape
    dist = np.full(n_samples, np.inf)
    support_indices = np.empty(0, dtype=int)
    if init_loc is None:
        support_indices = rs.permutation(n_samples)[:n_support].copy()
    else:
        precision0 = sl.pinvh(init_cov.__val__())
        dist = _mdist(X, init_loc.__val__(), precision0)
        support_indices = np.argpartition(dist, n_support - 1)[:n_support].copy()
    X_support = X[support_indices]
    location = X_support.mean(0)
    covariance = empirical_covariance(X_support)
    det = fast_logdet(covariance)
    precision = np.empty((0, 0))
    if np.isinf(det):
        precision = sl.pinvh(covariance)
    previous_det = np.inf
    previous_location = location
    previous_covariance = covariance
    previous_support_indices = support_indices
    while det < previous_det and remaining_iterations > 0 and not np.isinf(det):
        previous_location = location
        previous_covariance = covariance
        previous_det = det
        previous_support_indices = support_indices
        precision = sl.pinvh(covariance)
        dist = _mdist(X, location, precision)
        support_indices = np.argpartition(dist, n_support - 1)[:n_support].copy()
        X_support = X[support_indices]
        location = X_support.mean(axis=0)
        covariance = empirical_covariance(X_support)
        det = fast_logdet(covariance)
        remaining_iterations -= 1
    previous_dist = dist
    dist = _mdist(X, location, precision)
    rl = location; rc = covariance; rd = det; rsi = support_indices; rdist = dist
    if np.isinf(det):
        pass
    if np.allclose(det, previous_det):
        pass
    elif det > previous_det:
        rl = previous_location; rc = previous_covariance; rd = previous_det
        rsi = previous_support_indices; rdist = previous_dist
    if remaining_iterations == 0:
        rl = location; rc = covariance; rd = det; rsi = support_indices; rdist = dist
    support = np.zeros(n_samples, dtype=bool)
    for i in rsi:
        support[i] = True
    return _Est(rl, rc, rd, support, rdist)

def _select_candidates(X: np.ndarray[float, 2], n_support: int, ests: List[_Est], select: int) -> List[_Est]:
    dets = np.array([e.det for e in ests])
    idx = np.argsort(dets)[:select]
    return [ests[int(i)] for i in idx]

def _run_trials(X, n_support: int, n_trials: int, n_iter: int, rs: RandomState) -> List[_Est]:
    return [_c_step(X, n_support, rs, n_iter, None, None) for _ in range(n_trials)]

def _run_from(X, n_support: int, starts: List[_Est], n_iter: int, rs: RandomState) -> List[_Est]:
    return [_c_step(X, n_support, rs, n_iter, e.location, e.covariance) for e in starts]

def fast_mcd(X, support_fraction: Optional[float] = None, random_state = None):
    rs = check_random_state(random_state)
    return _fast_mcd(np.ascontiguousarray(np.asarray(X, dtype=float)), support_fraction, rs)

def _fast_mcd(X: np.ndarray[float, 2], support_fraction: Optional[float], rs: RandomState):
    n_samples, n_features = X.shape
    n_support = 0
    if support_fraction is None:
        n_support = min(int(np.ceil(0.5 * (n_samples + n_features + 1))), n_samples)
    else:
        n_support = int(support_fraction.__val__() * n_samples)
    location = np.empty(0)
    covariance = np.empty((0, 0))
    support = np.zeros(n_samples, dtype=bool)
    dist = np.empty(0)
    if n_features == 1:
        x = np.ascontiguousarray(X[:, 0])
        if n_support < n_samples:
            Xs = np.sort(x)
            diff = Xs[n_support:] - Xs[:n_samples - n_support]
            mn = np.min(diff)
            acc = List[float]()
            for k in range(diff.shape[0]):
                if diff[k] == mn:
                    acc.append(0.5 * (Xs[n_support + k] + Xs[k]))
            loc = float(np.mean(np.array(acc)))
            Xc = x - loc
            order = np.argsort(np.abs(Xc))[:n_support]
            for i in order:
                support[i] = True
            sel = List[float]()
            for i in range(n_samples):
                if support[i]:
                    sel.append(x[i])
            covariance = np.full((1, 1), float(np.var(np.array(sel))))
            location = np.array([loc])
        else:
            support = np.ones(n_samples, dtype=bool)
            covariance = np.full((1, 1), float(np.var(x)))
            location = np.array([float(np.mean(x))])
        precision = sl.pinvh(covariance)
        dist = _mdist(X, location, precision)
    elif n_samples > 500:
        n_subsets = n_samples // 300
        n_samples_subsets = n_samples // n_subsets
        samples_shuffle = rs.permutation(n_samples)
        h_subset = int(np.ceil(n_samples_subsets * (n_support / float(n_samples))))
        n_trials_tot = 500
        n_best_sub = 10
        n_trials = max(10, n_trials_tot // n_subsets)
        allbest = List[_Est]()
        for i in range(n_subsets):
            lo = i * n_samples_subsets
            sub = np.ascontiguousarray(X[samples_shuffle[lo:lo + n_samples_subsets]])
            for e in _select_candidates(sub, h_subset, _run_trials(sub, h_subset, n_trials, 2, rs), n_best_sub):
                allbest.append(e)
        n_merged = min(1500, n_samples)
        h_merged = int(np.ceil(n_merged * (n_support / float(n_samples))))
        n_best_merged = 10 if n_samples > 1500 else 1
        selection = rs.permutation(n_samples)[:n_merged].copy()
        Xm = np.ascontiguousarray(X[selection])
        merged = _select_candidates(Xm, h_merged, _run_from(Xm, h_merged, allbest, 30, rs), n_best_merged)
        if n_samples < 1500:
            location = merged[0].location
            covariance = merged[0].covariance
            dist = np.zeros(n_samples)
            for t in range(n_merged):
                support[selection[t]] = merged[0].support[t]
                dist[selection[t]] = merged[0].dist[t]
        else:
            full = _select_candidates(X, n_support, _run_from(X, n_support, merged, 30, rs), 1)
            location = full[0].location
            covariance = full[0].covariance
            support = full[0].support
            dist = full[0].dist
    else:
        best = _select_candidates(X, n_support, _run_trials(X, n_support, 30, 2, rs), 10)
        full = _select_candidates(X, n_support, _run_from(X, n_support, best, 30, rs), 1)
        location = full[0].location
        covariance = full[0].covariance
        support = full[0].support
        dist = full[0].dist
    return location, covariance, support, dist

def _consistency_factor(n_features: int, alpha: float) -> float:
    from scipy.stats import chi2
    q_alpha = float(chi2.ppf(alpha, n_features))
    return alpha / float(chi2.cdf(q_alpha, n_features + 2))

class MinCovDet(EmpiricalCovariance):
    support_fraction: Optional[float]
    random_state: Optional[int]
    raw_location_: np.ndarray[float, 1]
    raw_covariance_: np.ndarray[float, 2]
    raw_support_: np.ndarray[bool, 1]
    support_: np.ndarray[bool, 1]
    dist_: np.ndarray[float, 1]

    def __init__(self, *, store_precision: bool = True, assume_centered: bool = False,
                 support_fraction: Optional[float] = None, random_state: Optional[int] = None):
        super().__init__(store_precision=store_precision, assume_centered=assume_centered)
        self.support_fraction = support_fraction
        self.random_state = random_state
        self.raw_location_ = np.empty(0)
        self.raw_covariance_ = np.empty((0, 0))
        self.raw_support_ = np.empty(0, dtype=bool)
        self.support_ = np.empty(0, dtype=bool)
        self.dist_ = np.empty(0)

    def fit(self, X, y = None):
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        rs = check_random_state(self.random_state)
        n, p = Xa.shape
        loc, cov, sup, dist = _fast_mcd(Xa, self.support_fraction, rs)
        if self.assume_centered:
            loc = np.zeros(p)
            cov = empirical_covariance(Xa[sup], assume_centered=True)
            prec = sl.pinvh(cov)
            dist = np.sum(np.dot(Xa, prec) * Xa, 1)
        self.raw_location_ = loc
        self.raw_covariance_ = cov
        self.raw_support_ = sup
        self.location_ = loc
        self.support_ = sup
        self.dist_ = dist
        self.correct_covariance(Xa)
        self.reweight_covariance(Xa)
        return self

    def correct_covariance(self, data):
        n_samples = self.dist_.shape[0]
        n_support = 0
        for v in self.support_:
            if v:
                n_support += 1
        p = self.raw_covariance_.shape[0]
        if n_support < n_samples and np.allclose(self.raw_covariance_, 0):
            raise ValueError("The covariance matrix of the support data is equal to 0, try to increase support_fraction")
        cf = _consistency_factor(p, n_support / n_samples)
        corrected = self.raw_covariance_ * cf
        self.dist_ = self.dist_ / cf
        return corrected

    def reweight_covariance(self, data):
        from scipy.stats import chi2
        Xa = np.asarray(data, dtype=float)
        n, p = Xa.shape
        qt = 0.025
        thr = float(chi2.isf(qt, p))
        mask = self.dist_ < thr
        loc = np.zeros(p) if self.assume_centered else Xa[mask].mean(0)
        cov = empirical_covariance(Xa[mask], assume_centered=self.assume_centered)
        cf = _consistency_factor(p, 1 - qt)
        self._set_covariance(cov * cf)
        self.location_ = loc
        self.support_ = mask.copy()
        Xc = Xa - self.location_
        self.dist_ = np.sum(np.dot(Xc, self.get_precision()) * Xc, 1)
        return loc, cov, self.support_

class EllipticEnvelope(MinCovDet):
    contamination: float
    offset_: float

    def __init__(self, *, store_precision: bool = True, assume_centered: bool = False,
                 support_fraction: Optional[float] = None, contamination: float = 0.1,
                 random_state: Optional[int] = None):
        super().__init__(store_precision=store_precision, assume_centered=assume_centered,
                         support_fraction=support_fraction, random_state=random_state)
        self.contamination = contamination
        self.offset_ = 0.0

    def fit(self, X, y = None):
        MinCovDet.fit(self, X)
        self.offset_ = float(np.percentile(-self.dist_, 100.0 * self.contamination))
        return self

    def score_samples(self, X):
        return -self.mahalanobis(X)

    def decision_function(self, X):
        return self.score_samples(X) - self.offset_

    def predict(self, X):
        v = self.decision_function(X)
        out = np.full(v.shape[0], -1, dtype=int)
        for i in range(v.shape[0]):
            if v[i] >= 0:
                out[i] = 1
        return out

    def fit_predict(self, X, y = None):
        return self.fit(X).predict(X)

    def score(self, X, y, sample_weight = None) -> float:
        from ..base import _accuracy
        return _accuracy(y, self.predict(X))
