# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# MultiOutputRegressor / MultiOutputClassifier / RegressorChain /
# ClassifierChain over a statically typed base estimator E (dense X, 2-D Y).
# Chains: chain_method='predict'; cv=None or an int (KFold for regressors,
# StratifiedKFold for classifiers, as check_cv).

import numpy as np
from numpy.random.legacy import check_random_state
from .base import clone

def _col(Y, j: int):
    return np.ascontiguousarray(Y[:, j])

class _MultiOutput[E]:
    estimator: E
    n_jobs: Optional[int]
    estimators_: List[E]

    def _fit_all(self, X, Y, sample_weight):
        Xa = np.asarray(X, dtype=float)
        Ya = np.asarray(Y)
        if Ya.ndim == 1:
            raise ValueError("y must have at least two dimensions for multi-output regression but has only one.")
        self.estimators_ = []
        for j in range(Ya.shape[1]):
            e = clone(self.estimator)
            if isinstance(sample_weight, NoneType):
                e.fit(Xa, _col(Ya, j))
            else:
                e.fit(Xa, _col(Ya, j), sample_weight=sample_weight)
            self.estimators_.append(e)

class MultiOutputRegressor[E](_MultiOutput[E]):
    _estimator_type: Literal[str] = "regressor"

    def __init__(self, estimator: E, *, n_jobs: Optional[int] = None):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.estimators_ = []

    def fit(self, X, y, sample_weight = None):
        self._fit_all(X, np.asarray(y, dtype=float), sample_weight)
        return self

    def predict(self, X):
        Xa = np.asarray(X, dtype=float)
        m = len(self.estimators_)
        out = np.empty((Xa.shape[0], m))
        for j in range(m):
            p = np.asarray(self.estimators_[j].predict(Xa), dtype=float).ravel()
            for i in range(Xa.shape[0]):
                out[i, j] = p[i]
        return out

    def score(self, X, y, sample_weight = None) -> float:
        from .metrics import r2_score
        return float(r2_score(y, self.predict(X)))

    def _sk_clone(self):
        return MultiOutputRegressor[E](clone(self.estimator), n_jobs=self.n_jobs)

class MultiOutputClassifier[E](_MultiOutput[E]):
    _estimator_type: Literal[str] = "classifier"

    def __init__(self, estimator: E, *, n_jobs: Optional[int] = None):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.estimators_ = []

    def fit(self, X, Y, sample_weight = None):
        self._fit_all(X, np.asarray(Y, dtype=int), sample_weight)
        return self

    @property
    def classes_(self):
        return [e.classes_ for e in self.estimators_]

    def predict(self, X):
        Xa = np.asarray(X, dtype=float)
        m = len(self.estimators_)
        out = np.empty((Xa.shape[0], m), dtype=int)
        for j in range(m):
            p = np.asarray(self.estimators_[j].predict(Xa), dtype=int).ravel()
            for i in range(Xa.shape[0]):
                out[i, j] = p[i]
        return out

    def predict_proba(self, X):
        return [e.predict_proba(X) for e in self.estimators_]

    def score(self, X, y) -> float:
        Ya = np.asarray(y, dtype=int)
        P = self.predict(X)
        c = 0
        for i in range(Ya.shape[0]):
            ok = True
            for j in range(Ya.shape[1]):
                if P[i, j] != Ya[i, j]:
                    ok = False
            if ok:
                c += 1
        return float(c) / float(Ya.shape[0])

    def _sk_clone(self):
        return MultiOutputClassifier[E](clone(self.estimator), n_jobs=self.n_jobs)

class _BaseChain[E]:
    estimator: E
    _order_kind: str
    _order_list: List[int]
    cv: int
    random_state: Optional[int]
    order_: np.ndarray[int, 1]
    estimators_: List[E]

    def _init_chain(self, estimator: E, order, cv, random_state: Optional[int]):
        self.estimator = estimator
        self._order_list = []
        if isinstance(order, NoneType):
            self._order_kind = ""
        elif isinstance(order, str):
            self._order_kind = order
        else:
            self._order_kind = "list"
            self._order_list = [int(v) for v in order]
        self.cv = 0 if isinstance(cv, NoneType) else int(cv)
        self.random_state = random_state
        self.order_ = np.empty(0, dtype=int)
        self.estimators_ = []

    def _fit_chain(self, X, Y, classifier: bool):
        Xa = np.asarray(X, dtype=float)
        Ya = np.asarray(Y, dtype=float)
        n, p = Xa.shape
        m = Ya.shape[1]
        rs = check_random_state(self.random_state)
        if self._order_kind == "":
            self.order_ = np.arange(m)
        elif self._order_kind == "random":
            self.order_ = rs.permutation(m)
        else:
            if sorted(self._order_list) != list(range(m)):
                raise ValueError("invalid order")
            self.order_ = np.array(self._order_list)
        Xaug = np.zeros((n, p + m))
        for i in range(n):
            for k in range(p):
                Xaug[i, k] = Xa[i, k]
        if self.cv == 0:
            for i in range(n):
                for c in range(m):
                    Xaug[i, p + c] = Ya[i, self.order_[c]]
        self.estimators_ = []
        for c in range(m):
            y = _col(Ya, int(self.order_[c]))
            e = clone(self.estimator)
            Xc = np.ascontiguousarray(Xaug[:, :p + c])
            if classifier:
                e.fit(Xc, y.astype(int))
            else:
                e.fit(Xc, y)
            self.estimators_.append(e)
            if self.cv > 0 and c < m - 1:
                cvp = self._cv_predict(Xc, y, classifier)
                for i in range(n):
                    Xaug[i, p + c] = cvp[i]

    def _cv_predict(self, Xc, y, classifier: bool) -> np.ndarray[float, 1]:
        from .model_selection import KFold, StratifiedKFold
        n = Xc.shape[0]
        out = np.empty(n)
        folds = List[Tuple[np.ndarray[int, 1], np.ndarray[int, 1]]]()
        if classifier:
            for tr, te in StratifiedKFold(self.cv).split(Xc, y.astype(int)):
                folds.append((tr, te))
        else:
            for tr, te in KFold(self.cv).split(Xc):
                folds.append((tr, te))
        for tr, te in folds:
            e = clone(self.estimator)
            if classifier:
                e.fit(Xc[tr], y[tr].astype(int))
            else:
                e.fit(Xc[tr], y[tr])
            pr = np.asarray(e.predict(Xc[te]), dtype=float).ravel()
            for t in range(te.shape[0]):
                out[te[t]] = pr[t]
        return out

    def _chain(self, X, output: int):
        """output 0: predict, 1: predict_proba[:, 1], 2: decision_function."""
        Xa = np.asarray(X, dtype=float)
        n, p = Xa.shape
        m = len(self.estimators_)
        Xaug = np.zeros((n, p + m))
        for i in range(n):
            for k in range(p):
                Xaug[i, k] = Xa[i, k]
        outc = np.empty((n, m))
        for c in range(m):
            e = self.estimators_[c]
            Xc = np.ascontiguousarray(Xaug[:, :p + c])
            fp = np.asarray(e.predict(Xc), dtype=float).ravel()
            for i in range(n):
                Xaug[i, p + c] = fp[i]
            if output == 0:
                for i in range(n):
                    outc[i, c] = fp[i]
            elif output == 1:
                if hasattr(e, "predict_proba"):
                    pp = e.predict_proba(Xc)
                    for i in range(n):
                        outc[i, c] = pp[i, 1]
            else:
                if hasattr(e, "decision_function"):
                    dd = np.asarray(e.decision_function(Xc), dtype=float).ravel()
                    for i in range(n):
                        outc[i, c] = dd[i]
        out = np.empty((n, m))
        for c in range(m):
            for i in range(n):
                out[i, self.order_[c]] = outc[i, c]
        return out

class RegressorChain[E](_BaseChain[E]):
    _estimator_type: Literal[str] = "regressor"

    def __init__(self, estimator: E, *, order = None, cv = None, random_state: Optional[int] = None,
                 verbose: bool = False):
        self._init_chain(estimator, order, cv, random_state)

    def fit(self, X, Y):
        self._fit_chain(X, Y, False)
        return self

    def predict(self, X):
        return self._chain(X, 0)

    def score(self, X, y, sample_weight = None) -> float:
        from .metrics import r2_score
        return float(r2_score(y, self.predict(X)))

class ClassifierChain[E](_BaseChain[E]):
    _estimator_type: Literal[str] = "classifier"

    def __init__(self, estimator: E, *, order = None, cv = None, chain_method: str = "predict",
                 random_state: Optional[int] = None, verbose: bool = False):
        if chain_method != "predict":
            raise NotImplementedError("ClassifierChain: only chain_method='predict' is supported")
        self._init_chain(estimator, order, cv, random_state)

    def fit(self, X, Y):
        self._fit_chain(X, Y, True)
        return self

    @property
    def classes_(self):
        return [e.classes_ for e in self.estimators_]

    def predict(self, X):
        return self._chain(X, 0).astype(int)

    def predict_proba(self, X):
        return self._chain(X, 1)

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def decision_function(self, X):
        return self._chain(X, 2)

    def score(self, X, y, sample_weight = None) -> float:
        Ya = np.asarray(y, dtype=int)
        P = self.predict(X)
        c = 0
        for i in range(Ya.shape[0]):
            ok = True
            for j in range(Ya.shape[1]):
                if P[i, j] != Ya[i, j]:
                    ok = False
            if ok:
                c += 1
        return float(c) / float(Ya.shape[0])
