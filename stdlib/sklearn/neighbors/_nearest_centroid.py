# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

import numpy as np
from ._dist import _metric_code, _pairwise, _as2d
from ._unsupervised import _class_index
from ._supervised import _acc

class NearestCentroid:
    """metric 'euclidean' (class means) or 'manhattan' (class medians);
    priors 'uniform' (default), 'empirical' or an explicit list."""
    metric: str
    shrink_threshold: Optional[float]
    _priors_kind: str
    _priors_list: List[float]
    classes_: np.ndarray[int, 1]
    centroids_: np.ndarray[float, 2]
    class_prior_: np.ndarray[float, 1]
    within_class_std_dev_: np.ndarray[float, 1]
    deviations_: np.ndarray[float, 2]
    n_features_in_: int
    _estimator_type: Literal[str] = "classifier"

    def __init__(self, metric: str = "euclidean",
                 shrink_threshold: Optional[float] = None, priors = "uniform"):
        if metric != "euclidean" and metric != "manhattan":
            raise ValueError("metric must be 'euclidean' or 'manhattan'")
        self.metric = metric
        self.shrink_threshold = shrink_threshold
        self._priors_list = List[float]()
        if isinstance(priors, str):
            if priors != "uniform" and priors != "empirical":
                raise ValueError("priors must be 'uniform', 'empirical' or array-like")
            self._priors_kind = priors
        else:
            self._priors_kind = "list"
            for v in priors:
                self._priors_list.append(float(v))
        self.classes_ = np.empty(0, dtype=int)
        self.centroids_ = np.empty((0, 0))
        self.class_prior_ = np.empty(0)
        self.within_class_std_dev_ = np.empty(0)
        self.deviations_ = np.empty((0, 0))
        self.n_features_in_ = 0

    def fit(self, X, y):
        Xa = _as2d(X)
        n_samples, n_features = Xa.shape
        self.n_features_in_ = n_features
        classes, yi = _class_index(y)
        self.classes_ = classes
        n_classes = classes.shape[0]
        if n_classes < 2:
            raise ValueError(f"The number of classes has to be greater than one; got {n_classes} class")
        cnt = np.bincount(yi)
        if self._priors_kind == "empirical":
            self.class_prior_ = cnt.astype(float) / float(n_samples)
        elif self._priors_kind == "uniform":
            self.class_prior_ = np.full(n_classes, 1.0 / float(n_classes))
        else:
            self.class_prior_ = np.array(self._priors_list)
        if np.any(self.class_prior_ < 0.0):
            raise ValueError("priors must be non-negative")
        if not np.isclose(self.class_prior_.sum(), 1.0):
            self.class_prior_ = self.class_prior_ / self.class_prior_.sum()
        cents = np.empty((n_classes, n_features))
        nk = np.zeros(n_classes)
        for c in range(n_classes):
            Xc = Xa[yi == c]
            nk[c] = float(Xc.shape[0])
            if self.metric == "manhattan":
                cents[c] = np.median(Xc, axis=0)
            else:
                cents[c] = Xc.mean(axis=0)
        var = np.zeros(n_features)
        for i in range(n_samples):
            for j in range(n_features):
                var[j] += (Xa[i, j] - cents[yi[i], j]) ** 2
        self.within_class_std_dev_ = np.sqrt(var / float(n_samples - n_classes))
        all_const = True
        for j in range(n_features):
            if np.max(Xa[:, j]) - np.min(Xa[:, j]) != 0.0:
                all_const = False
        if all_const:
            raise ValueError("All features have zero variance. Division by zero.")
        dc = Xa.mean(axis=0)
        m = np.sqrt((1.0 / nk) - (1.0 / float(n_samples)))
        s = self.within_class_std_dev_ + np.median(self.within_class_std_dev_)
        ms = m.reshape((n_classes, 1)) * s
        self.deviations_ = (cents - dc) / ms
        if self.shrink_threshold is not None and self.shrink_threshold.__val__() != 0.0:
            th = self.shrink_threshold.__val__()
            signs = np.sign(self.deviations_)
            dev = np.abs(self.deviations_) - th
            dev = np.clip(dev, 0.0, None)
            dev *= signs
            self.deviations_ = dev
            cents = dc + ms * dev
        self.centroids_ = np.ascontiguousarray(cents)
        return self

    def _decision_function(self, X):
        Xn = _as2d(X).copy()
        C = self.centroids_.copy()
        sd = self.within_class_std_dev_
        for j in range(sd.shape[0]):
            if sd[j] != 0.0:
                for i in range(Xn.shape[0]):
                    Xn[i, j] /= sd[j]
                for i in range(C.shape[0]):
                    C[i, j] /= sd[j]
        D = _pairwise(Xn, C, _metric_code(self.metric, 2.0), 2.0)
        out = np.empty(D.shape)
        for i in range(D.shape[0]):
            for c in range(D.shape[1]):
                out[i, c] = -(D[i, c] ** 2) + 2.0 * np.log(self.class_prior_[c])
        return out

    def _uniform(self) -> bool:
        nc = self.classes_.shape[0]
        return bool(np.isclose(self.class_prior_, 1.0 / float(nc)).all())

    def decision_function(self, X):
        """(n_samples, n_classes) scores. Deviation: for two classes
        scikit-learn returns the 1-D difference sc[:, 1] - sc[:, 0]; this
        static port always returns the 2-D matrix."""
        return self._decision_function(X)

    def predict(self, X):
        Xa = _as2d(X)
        out = np.empty(Xa.shape[0], dtype=int)
        if self._uniform():
            D = _pairwise(Xa, self.centroids_, _metric_code(self.metric, 2.0), 2.0)
            idx = np.argmin(D, axis=1)
        else:
            idx = np.argmax(self._decision_function(Xa), axis=1)
        for i in range(Xa.shape[0]):
            out[i] = self.classes_[idx[i]]
        return out

    def predict_log_proba(self, X):
        sc = self._decision_function(X)
        m = np.max(sc, axis=1, keepdims=True)
        lse = np.log(np.sum(np.exp(sc - m), axis=1, keepdims=True)) + m
        return sc - lse

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def score(self, X, y, sample_weight = None):
        return _acc(y, self.predict(X))
