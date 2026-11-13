# Copyright (C) 2026 the Typhon authors
# Ported from pyriemann 0.12 (BSD-3-Clause)

import numpy as np
from scipy.special import ndtr, ndtri, chdtrc
from .classification import MDM
from .geometry.geodesic import geodesic
from .geometry.mean import gmean

class Potato:
    metric: str
    threshold: float
    n_iter_max: int
    pos_label: int
    neg_label: int
    _mdm: MDM
    _mean: float
    _std: float
    covmean_: np.ndarray[float, 2]
    _fitted: bool

    def __init__(self, metric: str = "riemann", threshold: float = 3.0,
                 n_iter_max: int = 100, pos_label: int = 1, neg_label: int = 0):
        self.metric = metric
        self.threshold = threshold
        self.n_iter_max = n_iter_max
        self.pos_label = pos_label
        self.neg_label = neg_label
        self._mdm = MDM(metric)
        self._mean = 0.0
        self._std = 1.0
        self.covmean_ = np.empty((0, 0))
        self._fitted = False

    def _check_labels(self, X, y):
        n = X.shape[0]
        out = np.ones(n)
        if y is not None:
            ya = np.asarray(y, dtype=int)
            if ya.shape[0] != n:
                raise ValueError("y must be the same length of X")
            has_pos = False
            for i in range(n):
                if ya[i] == self.pos_label:
                    has_pos = True
            if len(np.unique(ya)) > 2:
                raise ValueError("number of classes must be maximum 2")
            if not has_pos:
                raise ValueError("y must contain a positive class")
            for i in range(n):
                out[i] = 1.0 if ya[i] == self.pos_label else 0.0
        return out

    def _get_z_score(self, d):
        return (d - self._mean) / self._std

    def fit(self, X, y = None, sample_weight = None):
        if self.pos_label == self.neg_label:
            raise ValueError("Positive and negative labels must be different")
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        y_old = self._check_labels(X, y)
        sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self._mdm = MDM(self.metric)
        for _ in range(self.n_iter_max):
            ix = y_old == 1.0
            if not np.any(ix):
                raise ValueError("Iterative outlier removal has rejected all "
                                 "matrices. Choose a higher threshold.")
            Xi = X[ix]
            self._mdm.fit(Xi, np.ones(Xi.shape[0], dtype=int), sample_weight=sw[ix])
            d = np.log(self._mdm.transform(Xi)[:, 0])
            self._mean = float(np.mean(d))
            self._std = float(np.std(d))
            z = self._get_z_score(d)
            ynew = np.zeros(n)
            k = 0
            for i in range(n):
                if ix[i]:
                    ynew[i] = 1.0 if z[k] < self.threshold else 0.0
                    k += 1
            same = True
            for i in range(n):
                if ynew[i] != y_old[i]:
                    same = False
            if same:
                break
            y_old = ynew
        self.covmean_ = self._mdm.covmeans_[0]
        self._fitted = True
        return self

    def partial_fit(self, X, y = None, sample_weight = None, alpha: float = 0.1):
        if not self._fitted:
            raise ValueError("partial_fit can be called only on an already fitted potato.")
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        if X.shape[1] != self._mdm.covmeans_[0].shape[0]:
            raise ValueError("X does not have the good number of channels.")
        yl = self._check_labels(X, y)
        sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("Parameter alpha must be in [0, 1]")
        if alpha == 0.0:
            return self
        sel = yl == 1.0
        Xm = gmean(X[sel], metric=self.metric, sample_weight=sw[sel])
        cm = geodesic(self._mdm.covmeans_[0], Xm, alpha, metric=self.metric)
        self._mdm.covmeans_[0] = cm
        d = float(np.log(self._mdm.transform(Xm.reshape(1, Xm.shape[0], Xm.shape[1]))[0, 0]))
        self._mean = (1.0 - alpha) * self._mean + alpha * d
        self._std = float(np.sqrt((1.0 - alpha) * self._std ** 2 + alpha * (d - self._mean) ** 2))
        self.covmean_ = self._mdm.covmeans_[0]
        return self

    def transform(self, X):
        d = np.log(self._mdm.transform(np.asarray(X, dtype=float))[:, 0])
        return self._get_z_score(d)

    def fit_transform(self, X, y = None, sample_weight = None):
        return self.fit(X, y, sample_weight=sample_weight).transform(X)

    def predict(self, X):
        z = self.transform(X)
        out = np.zeros(z.shape[0]) + float(self.neg_label)
        for i in range(z.shape[0]):
            if z[i] < self.threshold:
                out[i] = float(self.pos_label)
        return out

    def predict_proba(self, X):
        z = self.transform(X)
        return 1.0 - ndtr(z)

class PotatoField:
    n_potatoes: int
    p_threshold: float
    z_threshold: float
    metric: str
    n_iter_max: int
    pos_label: int
    neg_label: int
    method_combination: str
    _potatoes: List[Potato]

    def __init__(self, n_potatoes: int = 1, p_threshold: float = 0.01, z_threshold: float = 3.0,
                 metric: str = "riemann", n_iter_max: int = 10, pos_label: int = 1,
                 neg_label: int = 0, method_combination: str = "fisher"):
        self.n_potatoes = n_potatoes
        self.p_threshold = p_threshold
        self.z_threshold = z_threshold
        self.metric = metric
        self.n_iter_max = n_iter_max
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.method_combination = method_combination
        self._potatoes = List[Potato]()

    def _check(self, X: List[np.ndarray[float, 3]]):
        if len(X) != self.n_potatoes:
            raise ValueError("Length of X is not equal to n_potatoes.")
        n = X[0].shape[0]
        for x in X:
            if x.shape[0] != n:
                raise ValueError("Unequal n_matrices between ndarray of X.")

    def fit(self, X: List[np.ndarray[float, 3]], y = None, sample_weight = None):
        if self.n_potatoes < 1:
            raise ValueError("Parameter n_potatoes must be at least 1")
        if not (0.0 < self.p_threshold < 1.0):
            raise ValueError("Parameter p_threshold must be in (0, 1)")
        self._check(X)
        self._potatoes = List[Potato]()
        for i in range(self.n_potatoes):
            pt = Potato(self.metric, self.z_threshold, self.n_iter_max, self.pos_label, self.neg_label)
            pt.fit(X[i], y, sample_weight=sample_weight)
            self._potatoes.append(pt)
        return self

    def partial_fit(self, X: List[np.ndarray[float, 3]], y = None, sample_weight = None, alpha: float = 0.1):
        if len(self._potatoes) == 0:
            raise ValueError("partial_fit can be called only on an already fitted potato field.")
        self._check(X)
        for i in range(self.n_potatoes):
            self._potatoes[i].partial_fit(X[i], y, sample_weight=sample_weight, alpha=alpha)
        return self

    def transform(self, X: List[np.ndarray[float, 3]]):
        self._check(X)
        z = np.zeros((X[0].shape[0], self.n_potatoes))
        for i in range(self.n_potatoes):
            z[:, i] = self._potatoes[i].transform(X[i])
        return z

    def fit_transform(self, X: List[np.ndarray[float, 3]], y = None, sample_weight = None):
        return self.fit(X, y, sample_weight=sample_weight).transform(X)

    def predict_proba(self, X: List[np.ndarray[float, 3]]):
        self._check(X)
        n = X[0].shape[0]
        k = self.n_potatoes
        probas = np.zeros((k, n))
        for i in range(k):
            probas[i] = np.clip(self._potatoes[i].predict_proba(X[i]), 1e-10, 1.0)
        out = np.zeros(n)
        if self.method_combination == "fisher":
            stat = -2.0 * np.sum(np.log(probas), axis=0)
            for j in range(n):
                out[j] = chdtrc(2.0 * k, stat[j])
        elif self.method_combination == "stouffer":
            zz = ndtri(1.0 - probas)
            stat = np.sum(zz, axis=0) / np.sqrt(float(k))
            out = 1.0 - ndtr(stat)
        else:
            raise ValueError("method_combination must be 'fisher' or 'stouffer'")
        return out

    def predict(self, X: List[np.ndarray[float, 3]]):
        p = self.predict_proba(X)
        out = np.zeros(p.shape[0]) + float(self.neg_label)
        for i in range(p.shape[0]):
            if p[i] > self.p_threshold:
                out[i] = float(self.pos_label)
        return out
