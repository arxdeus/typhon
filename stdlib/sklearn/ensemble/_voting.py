# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# VotingClassifier / VotingRegressor (sklearn/ensemble/_voting.py).
# `estimators` is a static tuple of (name, estimator) pairs (heterogeneous
# estimator types are resolved at compile time, like Pipeline steps).

from ..base import estimator_repr as _estimator_repr
import numpy as np
from ..base import clone
from ..tree._classes import _encode_classes, _r2

def _clone_named(ests):
    if static.len(ests) == 0:
        return ()
    else:
        return ((ests[0][0], clone(ests[0][1])),) + _clone_named(ests[1:])

def _fit_all(ests, X, y, sample_weight):
    if static.len(ests) > 0:
        if sample_weight is None:
            ests[0][1].fit(X, y)
        else:
            ests[0][1].fit(X, y, sample_weight=sample_weight)
        _fit_all(ests[1:], X, y, sample_weight)

def _predict_all(ests, X, out: List[np.ndarray[float, 1]]):
    if static.len(ests) > 0:
        out.append(np.asarray(ests[0][1].predict(X), dtype=float))
        _predict_all(ests[1:], X, out)

def _proba_all(ests, X, out: List[np.ndarray[float, 2]]):
    if static.len(ests) > 0:
        out.append(np.asarray(ests[0][1].predict_proba(X), dtype=float))
        _proba_all(ests[1:], X, out)

def _weights(weights, m: int) -> List[float]:
    if weights is None:
        return [1.0] * m
    w = [float(v) for v in weights]
    if len(w) != m:
        raise ValueError(f"Number of `estimators` and weights must be equal; got {len(w)} weights, {m} estimators")
    return w

class VotingClassifier[S]:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    estimators: S
    voting: str
    weights: Optional[List[float]]
    flatten_transform: bool
    estimators_: S
    classes_: np.ndarray[int, 1]

    def __init__(self, estimators: S, *, voting: str = "hard", weights = None,
                 n_jobs: Optional[int] = None, flatten_transform: bool = True,
                 verbose: bool = False):
        self.estimators = estimators
        self.voting = voting
        self.weights = None if weights is None else [float(v) for v in weights]
        self.flatten_transform = flatten_transform
        self.estimators_ = _clone_named(estimators)
        self.classes_ = np.empty(0, dtype=int)

    def _sk_clone(self):
        return VotingClassifier(_clone_named(self.estimators), voting=self.voting,
                                weights=self.weights, flatten_transform=self.flatten_transform)

    @property
    def _estimator_type(self) -> str:
        return "classifier"

    @property
    def named_estimators_(self):
        return self.estimators_

    def fit(self, X, y, sample_weight = None):
        classes, enc = _encode_classes(np.asarray(y))
        self.classes_ = classes
        ye = np.array([int(v) for v in enc])
        self.estimators_ = _clone_named(self.estimators)
        _fit_all(self.estimators_, X, ye, sample_weight)
        return self

    def _collect_probas(self, X) -> List[np.ndarray[float, 2]]:
        out = List[np.ndarray[float, 2]]()
        _proba_all(self.estimators_, X, out)
        return out

    def predict_proba(self, X):
        if self.voting == "hard":
            raise AttributeError("predict_proba is not available when voting='hard'")
        probas = self._collect_probas(X)
        m = len(probas)
        w = _weights(self.weights, m)
        tw = 0.0
        for v in w:
            tw += v
        n = probas[0].shape[0]
        nc = probas[0].shape[1]
        out = np.zeros((n, nc))
        for j in range(m):
            out += probas[j] * w[j]
        return out / tw

    def predict(self, X):
        n = 0
        nc = self.classes_.shape[0]
        maj = List[int]()
        if self.voting == "soft":
            pr = self.predict_proba(X)
            n = pr.shape[0]
            for i in range(n):
                best = 0
                for c in range(1, nc):
                    if pr[i, c] > pr[i, best]:
                        best = c
                maj.append(best)
        else:
            preds = List[np.ndarray[float, 1]]()
            _predict_all(self.estimators_, X, preds)
            m = len(preds)
            w = _weights(self.weights, m)
            n = preds[0].shape[0]
            for i in range(n):
                cnt = [0.0] * nc
                for j in range(m):
                    cnt[int(preds[j][i])] += w[j]
                best = 0
                for c in range(1, nc):
                    if cnt[c] > cnt[best]:
                        best = c
                maj.append(best)
        out = np.empty(n, dtype=int)
        for i in range(n):
            out[i] = self.classes_[maj[i]]
        return out

    def transform(self, X):
        if self.voting == "soft":
            probas = self._collect_probas(X)
            return np.hstack(probas)
        preds = List[np.ndarray[float, 1]]()
        _predict_all(self.estimators_, X, preds)
        n = preds[0].shape[0]
        out = np.empty((n, len(preds)))
        for j in range(len(preds)):
            for i in range(n):
                out[i, j] = preds[j][i]
        return out

    def score(self, X, y, sample_weight = None) -> float:
        from ..base import _accuracy
        return _accuracy(y, self.predict(X))

class VotingRegressor[S]:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    estimators: S
    weights: Optional[List[float]]
    estimators_: S

    def __init__(self, estimators: S, *, weights = None, n_jobs: Optional[int] = None,
                 verbose: bool = False):
        self.estimators = estimators
        self.weights = None if weights is None else [float(v) for v in weights]
        self.estimators_ = _clone_named(estimators)

    def _sk_clone(self):
        return VotingRegressor(_clone_named(self.estimators), weights=self.weights)

    @property
    def _estimator_type(self) -> str:
        return "regressor"

    def fit(self, X, y, sample_weight = None):
        self.estimators_ = _clone_named(self.estimators)
        _fit_all(self.estimators_, X, np.asarray(y, dtype=float), sample_weight)
        return self

    def transform(self, X):
        preds = List[np.ndarray[float, 1]]()
        _predict_all(self.estimators_, X, preds)
        n = preds[0].shape[0]
        out = np.empty((n, len(preds)))
        for j in range(len(preds)):
            for i in range(n):
                out[i, j] = preds[j][i]
        return out

    def predict(self, X):
        P = self.transform(X)
        w = _weights(self.weights, P.shape[1])
        return np.average(P, axis=1, weights=np.array(w))

    def score(self, X, y, sample_weight = None) -> float:
        return _r2(y, self.predict(X))
