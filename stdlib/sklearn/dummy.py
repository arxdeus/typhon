# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# DummyClassifier (single output, integer labels) and DummyRegressor
# (1-D or 2-D targets).

import numpy as np

def _sw_or_ones(sample_weight, n: int) -> np.ndarray[float, 1]:
    if isinstance(sample_weight, NoneType):
        return np.ones(n)
    else:
        return np.asarray(sample_weight, dtype=float) * np.ones(n)
from numpy.random.legacy import RandomState, check_random_state

class DummyClassifier:
    strategy: str
    random_state: Optional[int]
    constant: Optional[int]
    classes_: np.ndarray[int, 1]
    n_classes_: int
    class_prior_: np.ndarray[float, 1]
    n_outputs_: int
    _rs: RandomState

    def __init__(self, *, strategy: str = "prior", random_state: Optional[int] = None,
                 constant: Optional[int] = None):
        if strategy not in ("most_frequent", "prior", "stratified", "uniform", "constant"):
            raise ValueError(f"The 'strategy' parameter of DummyClassifier must be a str among {{'most_frequent', 'prior', 'stratified', 'uniform', 'constant'}}. Got {strategy!r} instead.")
        self.strategy = strategy
        self.random_state = random_state
        self.constant = constant
        self.classes_ = np.empty(0, dtype=int)
        self.n_classes_ = 0
        self.class_prior_ = np.empty(0)
        self.n_outputs_ = 1
        self._rs = RandomState(0)

    def _rng(self) -> RandomState:
        # check_random_state(self.random_state): int -> a fresh stream per call,
        # None -> the global stream
        if self.random_state is None:
            return check_random_state(None)
        return RandomState(self.random_state)

    def fit(self, X, y, sample_weight = None):
        ya = np.asarray(y, dtype=int)
        n = ya.shape[0]
        sw = _sw_or_ones(sample_weight, n)
        cl = np.unique(ya)
        pr = np.zeros(cl.shape[0])
        for i in range(n):
            pr[int(np.searchsorted(cl, ya[i]))] += sw[i]
        self.class_prior_ = pr / np.sum(pr)
        self.classes_ = cl
        self.n_classes_ = cl.shape[0]
        if self.strategy == "constant":
            if self.constant is None:
                raise ValueError("Constant target value has to be specified when the constant strategy is used.")
            found = False
            for c in cl:
                if c == self.constant:
                    found = True
            if not found:
                raise ValueError(f"The constant target value must be present in the training data. You provided constant={self.constant}. Possible values are: {cl.tolist()}.")
        return self

    def _num(self, X) -> int:
        if isinstance(X, List):
            return len(X)
        return np.asarray(X).shape[0]

    def predict_proba(self, X):
        return self._proba(self._num(X), self._rng())

    def _proba(self, n: int, rs: RandomState):
        k = self.n_classes_
        out = np.zeros((n, k))
        if self.strategy == "most_frequent":
            ind = int(np.argmax(self.class_prior_))
            for i in range(n):
                out[i, ind] = 1.0
        elif self.strategy == "prior":
            for i in range(n):
                for j in range(k):
                    out[i, j] = self.class_prior_[j]
        elif self.strategy == "stratified":
            out = rs.multinomial(1, self.class_prior_, size=n).astype(float)
        elif self.strategy == "uniform":
            for i in range(n):
                for j in range(k):
                    out[i, j] = 1.0 / float(k)
        else:
            for j in range(k):
                if self.classes_[j] == self.constant:
                    for i in range(n):
                        out[i, j] = 1.0
        return out

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def predict(self, X):
        n = self._num(X)
        rs = self._rng()
        out = np.empty(n, dtype=int)
        if self.strategy == "most_frequent" or self.strategy == "prior":
            c = self.classes_[int(np.argmax(self.class_prior_))]
            for i in range(n):
                out[i] = c
        elif self.strategy == "stratified":
            P = self._proba(n, rs)
            for i in range(n):
                out[i] = self.classes_[int(np.argmax(P[i]))]
        elif self.strategy == "uniform":
            r = rs.randint(self.n_classes_, size=n)
            for i in range(n):
                out[i] = self.classes_[int(r[i])]
        else:
            c = self.constant
            for i in range(n):
                out[i] = c if c is not None else 0
        return out

    def score(self, X, y, sample_weight = None) -> float:
        ya = np.asarray(y, dtype=int)
        p = self.predict(np.zeros((ya.shape[0], 1)) if X is None else X)
        sw = _sw_or_ones(sample_weight, ya.shape[0])
        s = 0.0; t = 0.0
        for i in range(ya.shape[0]):
            t += sw[i]
            if p[i] == ya[i]:
                s += sw[i]
        return s / t

def _wpct_col(a: np.ndarray[float, 1], w: np.ndarray[float, 1], q: float) -> float:
    # sklearn.utils.stats._weighted_percentile (average=False), one column
    n = a.shape[0]
    idx = np.argsort(a)
    cdf = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc += w[idx[i]]
        cdf[i] = acc
    adj = q / 100.0 * cdf[n - 1]
    if adj == 0:
        adj = np.nextafter(adj, adj + 1)
    pi = int(np.searchsorted(cdf, adj))
    pi = min(max(pi, 0), n - 1)
    return a[idx[pi]]

class DummyRegressor[D: Literal[int] = 1]:
    strategy: str
    constant: Optional[List[float]]
    quantile: Optional[float]
    constant_: np.ndarray[float, 2]
    n_outputs_: int
    _ravel: bool

    def __init__(self, *, strategy: str = "mean", constant = None, quantile: Optional[float] = None):
        if strategy not in ("mean", "median", "quantile", "constant"):
            raise ValueError(f"The 'strategy' parameter of DummyRegressor must be a str among {{'mean', 'median', 'quantile', 'constant'}}. Got {strategy!r} instead.")
        self.strategy = strategy
        if constant is None:
            self.constant = None
        elif isinstance(constant, float) or isinstance(constant, int):
            self.constant = [float(constant)]
        else:
            self.constant = [float(v) for v in constant]
        self.quantile = quantile
        self.constant_ = np.empty((1, 0))
        self.n_outputs_ = 1
        self._ravel = True

    def fit(self, X, y, sample_weight = None):
        ya = np.asarray(y, dtype=float)
        if static.len(ya.shape) != D:
            from .linear_model._base import _retype
            return _retype(self, DummyRegressor[static.len(ya.shape)]).fit(X, ya, sample_weight)
        if ya.shape[0] == 0:
            raise ValueError("y must not be empty.")
        Y = np.empty((0, 0))
        if static.len(ya.shape) == 1:
            Y = ya.reshape(-1, 1)
            self._ravel = True
        else:
            Y = ya
            self._ravel = False
        n, k = Y.shape
        self.n_outputs_ = k
        c = np.empty((1, k))
        for j in range(k):
            col = np.ascontiguousarray(Y[:, j])
            if self.strategy == "mean":
                if isinstance(sample_weight, NoneType):
                    c[0, j] = float(np.average(col))
                else:
                    c[0, j] = float(np.average(col, weights=np.asarray(sample_weight, dtype=float)))
            elif self.strategy == "median" or self.strategy == "quantile":
                q = 50.0
                if self.strategy == "quantile":
                    if self.quantile is None:
                        raise ValueError("When using `strategy='quantile', you have to specify the desired quantile in the range [0, 1].")
                    q = self.quantile * 100.0
                if isinstance(sample_weight, NoneType):
                    c[0, j] = float(np.percentile(col, q))
                else:
                    c[0, j] = _wpct_col(col, np.asarray(sample_weight, dtype=float), q)
            else:
                if self.constant is None:
                    raise TypeError("Constant target value has to be specified when the constant strategy is used.")
                cv = self.constant
                if k != 1 and len(cv) != k:
                    raise ValueError(f"Constant target value should have shape ({k}, 1).")
                c[0, j] = cv[j] if len(cv) > j else cv[0]
        self.constant_ = c
        return self

    def predict(self, X):
        n = len(X) if isinstance(X, List) else np.asarray(X).shape[0]
        k = self.n_outputs_
        out = np.empty((n, k))
        for i in range(n):
            for j in range(k):
                out[i, j] = self.constant_[0, j]
        if D == 1:
            return np.ascontiguousarray(out[:, 0])
        else:
            return out

    def score(self, X, y, sample_weight = None) -> float:
        from .metrics import r2_score
        ya = np.asarray(y, dtype=float)
        p = self.predict(np.zeros((ya.shape[0], 1)) if X is None else X)
        return r2_score(ya, p)
