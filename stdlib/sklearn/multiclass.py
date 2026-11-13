# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# OneVsRestClassifier / OneVsOneClassifier / OutputCodeClassifier over a
# statically typed base estimator E (integer class labels; single-label
# targets for OvR). Binary sub-problems with a single class use a constant
# predictor as sklearn's _ConstantPredictor.

import numpy as np
from numpy.random.legacy import check_random_state
from .base import clone, is_regressor, _accuracy

def _predict_binary(est, X) -> np.ndarray[float, 1]:
    if is_regressor(est):
        return np.asarray(est.predict(X), dtype=float).ravel()
    elif hasattr(est, "decision_function"):
        return np.ascontiguousarray(np.asarray(est.decision_function(X), dtype=float).ravel())
    else:
        P = est.predict_proba(X)
        return np.ascontiguousarray(P[:, 1])

def _threshold(est) -> float:
    if hasattr(est, "decision_function") and not is_regressor(est):
        return 0.0
    return 0.5

class _Binary[E]:
    """A fitted binary estimator or a constant label (_ConstantPredictor)."""
    est: E
    const: bool
    value: int

    def __init__(self, est: E, const: bool, value: int):
        self.est = est
        self.const = const
        self.value = value

    def score(self, X) -> np.ndarray[float, 1]:
        if self.const:
            return np.full(np.asarray(X).shape[0], float(self.value))
        return _predict_binary(self.est, X)

    def predict(self, X) -> np.ndarray[int, 1]:
        if self.const:
            return np.full(np.asarray(X).shape[0], self.value, dtype=int)
        return np.asarray(self.est.predict(X), dtype=int).ravel()

    def proba1(self, X) -> np.ndarray[float, 1]:
        if self.const:
            return np.full(np.asarray(X).shape[0], float(self.value))
        return np.ascontiguousarray(self.est.predict_proba(X)[:, 1])

def _fit_binary(estimator: E, X, y: np.ndarray[int, 1], E: type) -> _Binary[E]:
    u = np.unique(y)
    if u.shape[0] == 1:
        return _Binary[E](estimator, True, int(u[0]))
    e = clone(estimator)
    e.fit(X, y)
    return _Binary[E](e, False, 0)

class OneVsRestClassifier[E]:
    estimator: E
    n_jobs: Optional[int]
    classes_: np.ndarray[int, 1]
    estimators_: List[_Binary[E]]
    _estimator_type: ClassVar[str] = "classifier"

    def __init__(self, estimator: E, *, n_jobs: Optional[int] = None, verbose: int = 0):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.classes_ = np.empty(0, dtype=int)
        self.estimators_ = []

    @property
    def n_classes_(self) -> int:
        return self.classes_.shape[0]

    def fit(self, X, y):
        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y, dtype=int)
        self.classes_ = np.unique(ya)
        k = self.classes_.shape[0]
        # LabelBinarizer: binary targets give one column (positive = classes_[1])
        cols = [1] if k == 2 else list(range(k))
        self.estimators_ = []
        for c in cols:
            col = np.empty(ya.shape[0], dtype=int)
            for i in range(ya.shape[0]):
                col[i] = 1 if ya[i] == self.classes_[c] else 0
            self.estimators_.append(_fit_binary(self.estimator, Xa, col))
        return self

    def decision_function(self, X):
        Xa = np.asarray(X, dtype=float)
        m = len(self.estimators_)
        out = np.empty((Xa.shape[0], m))
        for j in range(m):
            s = self.estimators_[j].score(Xa)
            for i in range(Xa.shape[0]):
                out[i, j] = s[i]
        return out

    def predict(self, X):
        Xa = np.asarray(X, dtype=float)
        n = Xa.shape[0]
        out = np.empty(n, dtype=int)
        if self.classes_.shape[0] == 2:
            thr = _threshold(self.estimator)
            s = self.estimators_[0].score(Xa)
            for i in range(n):
                out[i] = self.classes_[1] if s[i] > thr else self.classes_[0]
            return out
        maxima = np.full(n, -np.inf)
        arg = np.zeros(n, dtype=int)
        m = len(self.estimators_)
        for t in range(m):
            e = self.estimators_[m - 1 - t]
            pred = e.score(Xa)
            for i in range(n):
                if pred[i] > maxima[i]:
                    maxima[i] = pred[i]
                if maxima[i] == pred[i]:
                    arg[i] = m - t - 1
        for i in range(n):
            out[i] = self.classes_[arg[i]]
        return out

    def predict_proba(self, X):
        Xa = np.asarray(X, dtype=float)
        n = Xa.shape[0]
        m = len(self.estimators_)
        Y = np.empty((n, m))
        for j in range(m):
            p = self.estimators_[j].proba1(Xa)
            for i in range(n):
                Y[i, j] = p[i]
        if m == 1:
            Z = np.empty((n, 2))
            for i in range(n):
                Z[i, 0] = 1 - Y[i, 0]
                Z[i, 1] = Y[i, 0]
            Y = Z
        for i in range(n):
            s = 0.0
            for j in range(Y.shape[1]):
                s += Y[i, j]
            if s != 0:
                for j in range(Y.shape[1]):
                    Y[i, j] /= s
        return Y

    def score(self, X, y, sample_weight = None) -> float:
        return _accuracy(y, self.predict(X))

    def _sk_clone(self):
        return OneVsRestClassifier[E](clone(self.estimator), n_jobs=self.n_jobs)

class OneVsOneClassifier[E]:
    estimator: E
    n_jobs: Optional[int]
    classes_: np.ndarray[int, 1]
    estimators_: List[_Binary[E]]
    _estimator_type: ClassVar[str] = "classifier"

    def __init__(self, estimator: E, *, n_jobs: Optional[int] = None):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.classes_ = np.empty(0, dtype=int)
        self.estimators_ = []

    @property
    def n_classes_(self) -> int:
        return self.classes_.shape[0]

    def fit(self, X, y):
        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y, dtype=int)
        self.classes_ = np.unique(ya)
        k = self.classes_.shape[0]
        if k == 1:
            raise ValueError("OneVsOneClassifier can not be fit when only one class is present.")
        self.estimators_ = []
        for i in range(k):
            for j in range(i + 1, k):
                idx = List[int]()
                for t in range(ya.shape[0]):
                    if ya[t] == self.classes_[i] or ya[t] == self.classes_[j]:
                        idx.append(t)
                ia = np.array(idx)
                yb = np.empty(len(idx), dtype=int)
                for t in range(len(idx)):
                    yb[t] = 0 if ya[idx[t]] == self.classes_[i] else 1
                self.estimators_.append(_fit_binary(self.estimator, np.ascontiguousarray(Xa[ia]), yb))
        return self

    def _decision(self, X):
        from .utils.multiclass import _ovr_decision_function
        Xa = np.asarray(X, dtype=float)
        n = Xa.shape[0]
        m = len(self.estimators_)
        P = np.empty((n, m))
        C = np.empty((n, m))
        for j in range(m):
            p = self.estimators_[j].predict(Xa)
            s = self.estimators_[j].score(Xa)
            for i in range(n):
                P[i, j] = float(p[i])
                C[i, j] = s[i]
        return _ovr_decision_function(P, C, self.classes_.shape[0])

    def decision_function(self, X):
        # DEVIATION: binary case returned as (n, 1) (column 1 of sklearn's Y)
        Y = self._decision(X)
        if self.classes_.shape[0] == 2:
            return np.ascontiguousarray(Y[:, 1:2])
        return Y

    def predict(self, X):
        Y = self._decision(X)
        n = Y.shape[0]
        out = np.empty(n, dtype=int)
        if self.classes_.shape[0] == 2:
            thr = _threshold(self.estimator)
            for i in range(n):
                out[i] = self.classes_[1 if Y[i, 1] > thr else 0]
            return out
        idx = np.argmax(Y, axis=1)
        for i in range(n):
            out[i] = self.classes_[idx[i]]
        return out

    def score(self, X, y, sample_weight = None) -> float:
        return _accuracy(y, self.predict(X))

    def _sk_clone(self):
        return OneVsOneClassifier[E](clone(self.estimator), n_jobs=self.n_jobs)

class OutputCodeClassifier[E]:
    estimator: E
    code_size: float
    random_state: Optional[int]
    classes_: np.ndarray[int, 1]
    code_book_: np.ndarray[float, 2]
    estimators_: List[_Binary[E]]
    _estimator_type: ClassVar[str] = "classifier"

    def __init__(self, estimator: E, *, code_size: float = 1.5, random_state: Optional[int] = None,
                 n_jobs: Optional[int] = None):
        self.estimator = estimator
        self.code_size = code_size
        self.random_state = random_state
        self.classes_ = np.empty(0, dtype=int)
        self.code_book_ = np.empty((0, 0))
        self.estimators_ = []

    def fit(self, X, y):
        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y, dtype=int)
        rs = check_random_state(self.random_state)
        self.classes_ = np.unique(ya)
        k = self.classes_.shape[0]
        ne = int(k * self.code_size)
        cb = rs.uniform(size=(k, ne))
        neg = -1.0 if hasattr(self.estimator, "decision_function") else 0.0
        for i in range(k):
            for j in range(ne):
                cb[i, j] = 1.0 if cb[i, j] > 0.5 else neg
        self.code_book_ = cb
        self.estimators_ = []
        for j in range(ne):
            col = np.empty(ya.shape[0], dtype=int)
            for t in range(ya.shape[0]):
                col[t] = int(cb[int(np.searchsorted(self.classes_, ya[t])), j])
            self.estimators_.append(_fit_binary(self.estimator, Xa, col))
        return self

    def predict(self, X):
        Xa = np.asarray(X, dtype=float)
        n = Xa.shape[0]
        m = len(self.estimators_)
        Y = np.empty((n, m))
        for j in range(m):
            s = self.estimators_[j].score(Xa)
            for i in range(n):
                Y[i, j] = s[i]
        # pairwise_distances_argmin (euclidean): squared distances via norms
        from .decomposition._extmath import euclidean_distances_sq
        D = euclidean_distances_sq(Y, self.code_book_)
        idx = np.argmin(D, axis=1)
        out = np.empty(n, dtype=int)
        for i in range(n):
            out[i] = self.classes_[idx[i]]
        return out

    def score(self, X, y, sample_weight = None) -> float:
        return _accuracy(y, self.predict(X))

    def _sk_clone(self):
        return OutputCodeClassifier[E](clone(self.estimator), code_size=self.code_size,
                                       random_state=self.random_state)
