# Ported from pyriemann 0.12 (BSD-3-Clause)

import numpy as np
from .geometry.mean import gmean, mean_power
from .tangentspace import TangentSpace, FGDA
from .geometry.kernel import kernel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC as _SKSVC
from .geometry.distance import distance
from .geometry.base import logm
from scipy.optimize import minimize, LinearConstraint

def _softmax(X):
    # sklearn.utils.extmath.softmax
    m = np.max(X, axis=1, keepdims=True)
    E = np.exp(X - m)
    return E / np.sum(E, axis=1, keepdims=True)

class MDM:
    # Labels are integers (cast with int()).
    metric: str
    n_jobs: int
    classes_: np.ndarray[int, 1]
    covmeans_: np.ndarray[float, 3]

    def __init__(self, metric: str = "riemann", n_jobs: int = 1):
        self.metric = metric
        self.n_jobs = n_jobs
        self.classes_ = np.empty(0, dtype=int)
        self.covmeans_ = np.empty((0, 0, 0))

    def fit(self, X, y, sample_weight = None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        sw = np.ones(X.shape[0]) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        nc = self.classes_.shape[0]
        n = X.shape[1]
        self.covmeans_ = np.empty((nc, n, n))
        for k in range(nc):
            mask = y == self.classes_[k]
            self.covmeans_[k] = gmean(X[mask], metric=self.metric,
                                      sample_weight=sw[mask])
        return self

    def _predict_distances(self, X):
        X = np.asarray(X, dtype=float)
        nc = self.classes_.shape[0]
        dist = np.empty((X.shape[0], nc))
        for k in range(nc):
            dist[:, k] = distance(X, self.covmeans_[k], metric=self.metric)[:, 0]
        return dist

    def predict(self, X):
        dist = self._predict_distances(X)
        return self.classes_[np.argmin(dist, axis=1)]

    def transform(self, X):
        return self._predict_distances(X)

    def fit_transform(self, X, y, sample_weight = None):
        return self.fit(X, y, sample_weight=sample_weight).transform(X)

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)

    def predict_proba(self, X):
        d = self._predict_distances(X)
        return _softmax(-(d * d))

    def score(self, X, y, sample_weight = None):
        pred = self.predict(X)
        y = np.asarray(y, dtype=int)
        correct = np.asarray(pred == y, dtype=float)
        if sample_weight is None:
            return float(np.mean(correct))
        else:
            w = np.asarray(sample_weight, dtype=float)
            return float(np.sum(correct * w) / np.sum(w))

class FgMDM:
    metric: str
    tsupdate: bool
    _mdm: MDM
    _fgda: FGDA
    classes_: np.ndarray[int, 1]

    def __init__(self, metric: str = "riemann", tsupdate: bool = False, n_jobs: int = 1):
        self.metric = metric
        self.tsupdate = tsupdate
        self._mdm = MDM(metric)
        self._fgda = FGDA(metric, tsupdate)
        self.classes_ = np.empty(0, dtype=int)

    def fit(self, X, y, sample_weight = None):
        self._mdm = MDM(self.metric)
        self._fgda = FGDA(self.metric, self.tsupdate)
        cov = self._fgda.fit_transform(X, y, sample_weight=sample_weight)
        self._mdm.fit(cov, y, sample_weight=sample_weight)
        self.classes_ = self._mdm.classes_
        return self

    def predict(self, X):
        return self._mdm.predict(self._fgda.transform(X))

    def predict_proba(self, X):
        return self._mdm.predict_proba(self._fgda.transform(X))

    def transform(self, X):
        return self._mdm.transform(self._fgda.transform(X))

    def fit_transform(self, X, y, sample_weight = None):
        return self.fit(X, y, sample_weight=sample_weight).transform(X)

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)

    def score(self, X, y, sample_weight = None):
        pred = self.predict(X)
        y = np.asarray(y, dtype=int)
        return float(np.mean(np.asarray(pred == y, dtype=float)))

class TSClassifier[CLF = LogisticRegression]:
    # The final estimator is a LogisticRegression (pyriemann default).
    metric: str
    tsupdate: bool
    clf: CLF
    _ts: TangentSpace
    classes_: np.ndarray[int, 1]

    def __init__(self, metric: str = "riemann", tsupdate: bool = False, clf: CLF = LogisticRegression()):
        self.metric = metric
        self.tsupdate = tsupdate
        self.clf = clf
        self._ts = TangentSpace(metric, tsupdate)
        self.classes_ = np.empty(0, dtype=int)

    def fit(self, X, y, sample_weight = None):
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        self._ts = TangentSpace(self.metric, self.tsupdate)
        ts = self._ts.fit_transform(X, sample_weight=sample_weight)
        self.clf.fit(ts, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.clf.predict(self._ts.transform(X))

    def predict_proba(self, X):
        return self.clf.predict_proba(self._ts.transform(X))

    def score(self, X, y, sample_weight = None):
        pred = self.predict(X)
        y = np.asarray(y, dtype=int)
        return float(np.mean(np.asarray(pred == y, dtype=float)))

class KNearestNeighbor(MDM):
    n_neighbors: int
    classmeans_: np.ndarray[int, 1]

    def __init__(self, n_neighbors: int = 5, metric: str = "riemann", n_jobs: int = 1):
        super().__init__(metric, n_jobs)
        self.n_neighbors = n_neighbors
        self.classmeans_ = np.empty(0, dtype=int)

    def fit(self, X, y, sample_weight = None):
        self.covmeans_ = np.array(np.asarray(X, dtype=float))
        self.classmeans_ = np.array(np.asarray(y, dtype=int))
        self.classes_ = np.unique(self.classmeans_)
        return self

    def _knn_dist(self, X):
        X = np.asarray(X, dtype=float)
        nt = self.covmeans_.shape[0]
        dist = np.empty((X.shape[0], nt))
        for k in range(nt):
            dist[:, k] = distance(X, self.covmeans_[k], metric=self.metric)[:, 0]
        return dist

    def predict(self, X):
        dist = self._knn_dist(X)
        nm = dist.shape[0]
        pred = np.empty(nm, dtype=int)
        nc = self.classes_.shape[0]
        for m in range(nm):
            idx = np.argsort(dist[m])
            cnt = np.zeros(nc, dtype=int)
            for j in range(self.n_neighbors):
                lab = self.classmeans_[idx[j]]
                for c in range(nc):
                    if self.classes_[c] == lab:
                        cnt[c] += 1
            # scipy.stats.mode: smallest label among the most frequent
            pred[m] = self.classes_[np.argmax(cnt)]
        return pred

    def predict_proba(self, X):
        dist = self._knn_dist(X)
        nm = dist.shape[0]
        nc = self.classes_.shape[0]
        prob = np.zeros((nm, nc))
        for m in range(nm):
            idx = np.argsort(dist[m])
            d = np.empty(self.n_neighbors)
            for j in range(self.n_neighbors):
                d[j] = -(dist[m, idx[j]] ** 2)
            e = np.exp(d - np.max(d))
            e = e / np.sum(e)
            for j in range(self.n_neighbors):
                lab = self.classmeans_[idx[j]]
                for c in range(nc):
                    if self.classes_[c] == lab:
                        prob[m, c] += e[j]
        return prob

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)

class SVC:
    # Riemannian-kernel SVM: precomputed Gram matrix K(X, X) at Cref.
    metric: str
    C: float
    has_cref: bool
    Cref: np.ndarray[float, 2]
    Cref_: np.ndarray[float, 2]
    svm: _SKSVC
    Xtr_: np.ndarray[float, 3]
    classes_: np.ndarray[int, 1]

    def __init__(self, metric: str = "riemann", Cref = None, C: float = 1.0,
                 shrinking: bool = True, probability: bool = False,
                 tol: float = 1e-3, max_iter: int = -1, random_state = None):
        self.metric = metric
        self.C = C
        self.has_cref = Cref is not None
        self.Cref = np.empty((0, 0)) if Cref is None else np.asarray(Cref, dtype=float)
        self.Cref_ = np.empty((0, 0))
        self.svm = _SKSVC(kernel="precomputed", C=C, shrinking=shrinking,
                          probability=probability, tol=tol, max_iter=max_iter,
                          random_state=random_state)
        self.Xtr_ = np.empty((0, 0, 0))
        self.classes_ = np.empty(0, dtype=int)

    def fit(self, X, y, sample_weight = None):
        X = np.asarray(X, dtype=float)
        if self.has_cref:
            self.Cref_ = self.Cref
        else:
            self.Cref_ = gmean(X, metric=self.metric)
        self.Xtr_ = np.array(X)
        K = kernel(X, X, Cref=self.Cref_, metric=self.metric)
        self.svm.fit(K, np.asarray(y, dtype=int), sample_weight=sample_weight)
        self.classes_ = self.svm.classes_
        return self

    def _gram(self, X):
        return kernel(np.asarray(X, dtype=float), self.Xtr_, Cref=self.Cref_, metric=self.metric)

    def predict(self, X):
        return self.svm.predict(self._gram(X))

    def decision_function(self, X):
        return self.svm.decision_function(self._gram(X))

    def predict_proba(self, X):
        return self.svm.predict_proba(self._gram(X))

    def score(self, X, y, sample_weight = None):
        pred = self.predict(X)
        y = np.asarray(y, dtype=int)
        return float(np.mean(np.asarray(pred == y, dtype=float)))

class MeanField:
    power_list: List[float]
    method_combination: str
    metric: str
    classes_: np.ndarray[int, 1]
    covmeans_: np.ndarray[float, 4]

    def __init__(self, power_list: List[float] = [-1.0, 0.0, 1.0],
                 method_combination: str = "sum_means", metric: str = "riemann",
                 n_jobs: int = 1):
        self.power_list = power_list
        self.method_combination = method_combination
        self.metric = metric
        self.classes_ = np.empty(0, dtype=int)
        self.covmeans_ = np.empty((0, 0, 0, 0))

    def fit(self, X, y, sample_weight = None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        nc = self.classes_.shape[0]
        npw = len(self.power_list)
        n = X.shape[1]
        sw = np.ones(X.shape[0]) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.covmeans_ = np.zeros((nc, npw, n, n))
        for ic in range(nc):
            mask = y == self.classes_[ic]
            for ip in range(npw):
                self.covmeans_[ic, ip] = mean_power(X[mask], self.power_list[ip],
                                                    sample_weight=sw[mask])
        return self

    def _predict_distances(self, X):
        X = np.asarray(X, dtype=float)
        nc = self.classes_.shape[0]
        npw = len(self.power_list)
        d2 = np.zeros((X.shape[0], nc, npw))
        for ic in range(nc):
            for ip in range(npw):
                d2[:, ic, ip] = distance(X, self.covmeans_[ic, ip], metric=self.metric,
                                         squared=True)[:, 0]
        if self.method_combination == "sum_means":
            return np.sqrt(np.sum(d2, axis=2))
        elif self.method_combination == "inf_means":
            return np.sqrt(np.min(d2, axis=2))
        elif self.method_combination == "none":
            return np.sqrt(d2.reshape(X.shape[0], nc * npw))
        raise ValueError("Unsupported method_combination " + self.method_combination)

    def predict(self, X):
        if self.method_combination == "none":
            raise ValueError("Classification by MeanField is not available when method_combination is None")
        return self.classes_[np.argmin(self._predict_distances(X), axis=1)]

    def transform(self, X):
        return self._predict_distances(X)

    def fit_transform(self, X, y, sample_weight = None):
        return self.fit(X, y, sample_weight=sample_weight).transform(X)

    def predict_proba(self, X):
        if self.method_combination == "none":
            raise ValueError("Classification by MeanField is not available when method_combination is None")
        d = self._predict_distances(X)
        return _softmax(-(d * d))

class _NCHFun:
    D1: np.ndarray[float, 2]
    d2: np.ndarray[float, 1]

    def __init__(self, D1: np.ndarray[float, 2], d2: np.ndarray[float, 1]):
        self.D1 = D1
        self.d2 = d2

    def fun(self, w):
        return w @ self.D1 @ w - 2.0 * (self.d2 @ w)

    def jac(self, w):
        return 2.0 * (self.D1 @ w) - 2.0 * self.d2

class NearestConvexHull(MDM):
    method: str
    mats_: np.ndarray[float, 3]
    classmats_: np.ndarray[int, 1]

    def __init__(self, metric: str = "logeuclid", n_jobs: int = 1, method: str = "SLSQP"):
        super().__init__(metric, n_jobs)
        self.method = method
        self.mats_ = np.empty((0, 0, 0))
        self.classmats_ = np.empty(0, dtype=int)

    def fit(self, X, y, sample_weight = None):
        if self.metric != "euclid" and self.metric != "logeuclid":
            raise ValueError(f"NCH does not support metric {self.metric}")
        self.mats_ = np.array(np.asarray(X, dtype=float))
        self.classmats_ = np.array(np.asarray(y, dtype=int))
        self.classes_ = np.unique(self.classmats_)
        return self

    def _find_weights_to_convex_hull(self, D1, d2):
        n = D1.shape[0]
        w0 = np.full(n, 1.0 / n)
        f = _NCHFun(D1, d2)
        bnds = [(0.0, None) for _ in range(n)]
        cons = LinearConstraint(np.ones((1, n)), 1.0, 1.0)
        res = minimize(f.fun, w0, method=self.method, jac=f.jac, bounds=bnds,
                       constraints=cons,
                       options={"maxiter": 50, "ftol": 1e-6, "disp": False})
        return np.clip(res.x, 0.0, 1.0)

    def _predict_distance(self, A, B):
        na = A.shape[0]
        nb = B.shape[0]
        A_ = A
        B_ = B
        if self.metric == "logeuclid":
            A_ = logm(A)
            B_ = logm(B)
        D1 = np.zeros((na, na))
        for i in range(na):
            for j in range(i, na):
                v = np.trace(A_[i] @ A_[j])
                D1[i, j] = v
                D1[j, i] = v
        dist = np.zeros(nb)
        for i in range(nb):
            d2 = np.zeros(na)
            for j in range(na):
                d2[j] = np.trace(B_[i] @ A_[j])
            weights = self._find_weights_to_convex_hull(D1, d2)
            H = gmean(A, metric=self.metric, sample_weight=weights)
            dist[i] = distance(H, B[i], metric=self.metric)[0, 0]
        return dist

    def _predict_distances(self, X):
        X = np.asarray(X, dtype=float)
        nc = self.classes_.shape[0]
        dist = np.empty((X.shape[0], nc))
        for k in range(nc):
            dist[:, k] = self._predict_distance(self.mats_[self.classmats_ == self.classes_[k]], X)
        return dist
