# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# BaggingClassifier / BaggingRegressor (sklearn/ensemble/_bagging.py) and
# IsolationForest (sklearn/ensemble/_iforest.py). Estimators exposing
# `sample_weight` in fit get bootstrap counts as weights (like sklearn for
# trees); `bootstrap_features` and fractional max_features select columns.

import math
import numpy as np
from ..base import clone
from ..tree._classes import (DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeRegressor,
                             _check_rs, _r2, _encode_classes)
from ._base import MAX_INT, _parse_max_samples, _get_n_samples_bootstrap, sample_without_replacement

def _set_rs(est, seed: int):
    if hasattr(est, "random_state"):
        est.random_state = seed

def _gen_indices(rs, bootstrap: bool, n_population: int, n_samples: int):
    if bootstrap:
        return rs.randint(0, n_population, n_samples)
    return sample_without_replacement(n_population, n_samples, rs)

def _bagging_indices(seed: int, bootstrap_features: bool, bootstrap: bool, n_features: int,
                     n_samples: int, max_features: int, max_samples: int):
    rs = np.random.RandomState(seed)
    feats = _gen_indices(rs, bootstrap_features, n_features, max_features)
    samples = _gen_indices(rs, bootstrap, n_samples, max_samples)
    return (feats, samples)

def _cols(X, feats):
    n = X.shape[0]
    m = feats.shape[0]
    out = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            out[i, j] = X[i, feats[j]]
    return out

class _BaseBagging[E]:
    estimator: E
    n_estimators: int
    _ms_kind: str
    _ms_value: float
    _mf_kind: str
    _mf_value: float
    bootstrap: bool
    bootstrap_features: bool
    oob_score: bool
    warm_start: bool
    n_jobs: Optional[int]
    random_state: Optional[int]
    verbose: int
    estimators_: List[E]
    estimators_features_: List[np.ndarray[int, 1]]
    _seeds: List[int]
    _max_samples: int
    _max_features: int
    _n_samples: int
    n_features_in_: int
    oob_score_: float
    _max_depth: Optional[int]

    def __init__(self, estimator: E, n_estimators: int, max_samples, max_features,
                 bootstrap: bool, bootstrap_features: bool, oob_score: bool, warm_start: bool,
                 n_jobs: Optional[int], random_state: Optional[int], verbose: int):
        self.estimator = estimator
        self.n_estimators = n_estimators
        k, v = _parse_max_samples(max_samples)
        self._ms_kind = k
        self._ms_value = v
        if isinstance(max_features, int):
            self._mf_kind = "int"
            self._mf_value = float(max_features)
        else:
            self._mf_kind = "float"
            self._mf_value = float(max_features)
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.estimators_ = List[E]()
        self.estimators_features_ = List[np.ndarray[int, 1]]()
        self._seeds = List[int]()
        self._max_samples = 0
        self._max_features = 0
        self._n_samples = 0
        self.n_features_in_ = 0
        self.oob_score_ = 0.0
        self._max_depth = None

    def _fit_bagging(self, X, y, sample_weight, max_samples_override: int):
        Xa = np.asarray(X, dtype=float)
        n = Xa.shape[0]
        p = Xa.shape[1]
        self.n_features_in_ = p
        self._n_samples = n
        rs = _check_rs(self.random_state)
        ms = 0
        if max_samples_override >= 0:
            ms = max_samples_override
        else:
            tot = float(n)
            has_w = False
            if sample_weight is not None:
                has_w = True
                tot = float(np.sum(np.asarray(sample_weight, dtype=float)))
            ms = _get_n_samples_bootstrap(n, self._ms_kind, self._ms_value, tot, has_w)
        if not self.bootstrap and ms > n:
            raise ValueError(f"Effective max_samples={ms} must be <= n_samples={n} to be able to sample without replacement.")
        self._max_samples = ms
        mf = int(self._mf_value) if self._mf_kind == "int" else int(self._mf_value * float(p))
        if mf > p:
            raise ValueError("max_features must be <= n_features")
        mf = max(1, mf)
        self._max_features = mf
        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")
        seeds = [int(v) for v in rs.randint(MAX_INT, size=self.n_estimators)]
        self._seeds = seeds
        self.estimators_ = List[E]()
        self.estimators_features_ = List[np.ndarray[int, 1]]()
        need_cols = self.bootstrap_features or mf != p
        for i in range(self.n_estimators):
            est = clone(self.estimator)
            if self._max_depth is not None:
                est.max_depth = self._max_depth
            # _set_random_states(est, seed): random_state = RandomState(seed).randint(MAX_INT)
            _set_rs(est, int(np.random.RandomState(seeds[i]).randint(MAX_INT)))
            feats, idx = _bagging_indices(seeds[i], self.bootstrap_features, self.bootstrap,
                                          p, n, mf, ms)
            Xs = _cols(Xa, feats) if need_cols else Xa
            if hasattr(est, "_fit_encoded"):
                cnt = np.zeros(n)
                for j in range(idx.shape[0]):
                    cnt[int(idx[j])] += 1.0
                est.fit(Xs, y, sample_weight=cnt)
            else:
                est.fit(Xs[idx], y[idx])
            self.estimators_.append(est)
            self.estimators_features_.append(feats)
        return Xa

    @property
    def estimators_samples_(self):
        out = List[np.ndarray[int, 1]]()
        for s in self._seeds:
            _, idx = _bagging_indices(s, self.bootstrap_features, self.bootstrap,
                                      self.n_features_in_, self._n_samples,
                                      self._max_features, self._max_samples)
            out.append(idx)
        return out

    def _sub(self, Xa, j: int):
        if self.bootstrap_features or self._max_features != self.n_features_in_:
            return _cols(Xa, self.estimators_features_[j])
        return Xa

    def __len__(self) -> int:
        return len(self.estimators_)

class BaggingClassifier[E](_BaseBagging[E]):
    classes_: np.ndarray[int, 1]
    n_classes_: int
    oob_decision_function_: np.ndarray[float, 2]

    def __init__(self, estimator: E = DecisionTreeClassifier(), n_estimators: int = 10, *,
                 max_samples = None, max_features = 1.0, bootstrap: bool = True,
                 bootstrap_features: bool = False, oob_score: bool = False,
                 warm_start: bool = False, n_jobs: Optional[int] = None,
                 random_state: Optional[int] = None, verbose: int = 0):
        _BaseBagging[E].__init__(self, estimator, n_estimators, max_samples, max_features,
                                 bootstrap, bootstrap_features, oob_score, warm_start, n_jobs,
                                 random_state, verbose)
        self.classes_ = np.empty(0, dtype=int)
        self.n_classes_ = 0
        self.oob_decision_function_ = np.empty((0, 0))

    def _sk_clone(self):
        b = BaggingClassifier(clone(self.estimator), self.n_estimators,
                              bootstrap=self.bootstrap, bootstrap_features=self.bootstrap_features,
                              oob_score=self.oob_score, random_state=self.random_state)
        b._ms_kind = self._ms_kind; b._ms_value = self._ms_value
        b._mf_kind = self._mf_kind; b._mf_value = self._mf_value
        return b

    @property
    def _estimator_type(self) -> str:
        return "classifier"

    def fit(self, X, y, sample_weight = None):
        classes, enc = _encode_classes(np.asarray(y))
        self.classes_ = classes
        self.n_classes_ = classes.shape[0]
        ye = np.array([int(v) for v in enc])
        Xa = self._fit_bagging(X, ye, sample_weight, -1)
        if self.oob_score:
            n = Xa.shape[0]
            nc = self.n_classes_
            pred = np.zeros((n, nc))
            for j in range(len(self.estimators_)):
                idx = self.estimators_samples_[j]
                mask = [True] * n
                for t in range(idx.shape[0]):
                    mask[int(idx[t])] = False
                rows = [i for i in range(n) if mask[i]]
                if len(rows) == 0:
                    continue
                Xs = self._sub(Xa, j)[np.array(rows)]
                pr = self._est_proba(j, Xs)
                for r in range(len(rows)):
                    for c in range(nc):
                        pred[rows[r], c] += pr[r, c]
            out = np.empty((n, nc))
            correct = 0
            for i in range(n):
                s = 0.0
                for c in range(nc):
                    s += pred[i, c]
                best = 0
                for c in range(nc):
                    out[i, c] = pred[i, c] / s if s != 0.0 else math.nan
                    if out[i, c] > out[i, best]:
                        best = c
                if best == int(ye[i]):
                    correct += 1
            self.oob_decision_function_ = out
            self.oob_score_ = float(correct) / float(n)
        return self

    def _est_proba(self, j: int, Xs):
        est = self.estimators_[j]
        nc = self.n_classes_
        if hasattr(est, "predict_proba"):
            pr = est.predict_proba(Xs)
            ec = est.classes_
            if ec.shape[0] == nc:
                return pr
            out = np.zeros((Xs.shape[0], nc))
            for i in range(Xs.shape[0]):
                for c in range(ec.shape[0]):
                    out[i, int(ec[c])] += pr[i, c]
            return out
        p = est.predict(Xs)
        out = np.zeros((Xs.shape[0], nc))
        for i in range(Xs.shape[0]):
            out[i, int(p[i])] += 1.0
        return out

    def predict_proba(self, X):
        Xa = np.asarray(X, dtype=float)
        n = Xa.shape[0]
        proba = np.zeros((n, self.n_classes_))
        for j in range(len(self.estimators_)):
            proba += self._est_proba(j, self._sub(Xa, j))
        return proba / float(self.n_estimators)

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def predict(self, X):
        pr = self.predict_proba(X)
        n = pr.shape[0]
        out = np.empty(n, dtype=int)
        for i in range(n):
            best = 0
            for c in range(1, self.n_classes_):
                if pr[i, c] > pr[i, best]:
                    best = c
            out[i] = self.classes_[best]
        return out

    def score(self, X, y, sample_weight = None) -> float:
        from ..base import _accuracy
        return _accuracy(y, self.predict(X))

class BaggingRegressor[E](_BaseBagging[E]):
    oob_prediction_: np.ndarray[float, 1]

    def __init__(self, estimator: E = DecisionTreeRegressor(), n_estimators: int = 10, *,
                 max_samples = None, max_features = 1.0, bootstrap: bool = True,
                 bootstrap_features: bool = False, oob_score: bool = False,
                 warm_start: bool = False, n_jobs: Optional[int] = None,
                 random_state: Optional[int] = None, verbose: int = 0):
        _BaseBagging[E].__init__(self, estimator, n_estimators, max_samples, max_features,
                                 bootstrap, bootstrap_features, oob_score, warm_start, n_jobs,
                                 random_state, verbose)
        self.oob_prediction_ = np.empty(0)

    def _sk_clone(self):
        b = BaggingRegressor(clone(self.estimator), self.n_estimators,
                             bootstrap=self.bootstrap, bootstrap_features=self.bootstrap_features,
                             oob_score=self.oob_score, random_state=self.random_state)
        b._ms_kind = self._ms_kind; b._ms_value = self._ms_value
        b._mf_kind = self._mf_kind; b._mf_value = self._mf_value
        return b

    @property
    def _estimator_type(self) -> str:
        return "regressor"

    def fit(self, X, y, sample_weight = None):
        ya = np.asarray(y, dtype=float)
        Xa = self._fit_bagging(X, ya, sample_weight, -1)
        if self.oob_score:
            n = Xa.shape[0]
            pred = np.zeros(n)
            cnt = np.zeros(n)
            for j in range(len(self.estimators_)):
                idx = self.estimators_samples_[j]
                mask = [True] * n
                for t in range(idx.shape[0]):
                    mask[int(idx[t])] = False
                rows = [i for i in range(n) if mask[i]]
                if len(rows) == 0:
                    continue
                pr = self.estimators_[j].predict(self._sub(Xa, j)[np.array(rows)])
                for r in range(len(rows)):
                    pred[rows[r]] += pr[r]
                    cnt[rows[r]] += 1.0
            for i in range(n):
                if cnt[i] == 0.0:
                    cnt[i] = 1.0
            pred /= cnt
            self.oob_prediction_ = pred
            self.oob_score_ = _r2(ya, pred)
        return self

    def predict(self, X):
        Xa = np.asarray(X, dtype=float)
        out = np.zeros(Xa.shape[0])
        for j in range(len(self.estimators_)):
            out += self.estimators_[j].predict(self._sub(Xa, j))
        return out / float(self.n_estimators)

    def score(self, X, y, sample_weight = None) -> float:
        return _r2(y, self.predict(X))

# ------------------------------------------------------------ IsolationForest

_EULER: float = 0.5772156649015329

def _average_path_length(n: float) -> float:
    if n <= 1.0:
        return 0.0
    if n == 2.0:
        return 1.0
    return 2.0 * (math.log(n - 1.0) + _EULER) - 2.0 * (n - 1.0) / n

class IsolationForest(_BaseBagging[ExtraTreeRegressor]):
    contamination: float
    _contamination_auto: bool
    _ms_auto: bool
    max_samples_: int
    offset_: float
    _path_len: List[List[float]]
    _depths: List[np.ndarray[int, 1]]

    def __init__(self, *, n_estimators: int = 100, max_samples = "auto",
                 contamination = "auto", max_features = 1.0, bootstrap: bool = False,
                 n_jobs: Optional[int] = None, random_state: Optional[int] = None,
                 verbose: int = 0, warm_start: bool = False):
        _BaseBagging[ExtraTreeRegressor].__init__(
            self, ExtraTreeRegressor(max_features=1, splitter="random", random_state=random_state),
            n_estimators, None if isinstance(max_samples, str) else max_samples, max_features,
            bootstrap, False, False, warm_start, n_jobs, random_state, verbose)
        self._ms_auto = isinstance(max_samples, str)
        if isinstance(contamination, str):
            self._contamination_auto = True
            self.contamination = 0.0
        else:
            self._contamination_auto = False
            self.contamination = float(contamination)
        self.max_samples_ = 0
        self.offset_ = -0.5
        self._path_len = List[List[float]]()
        self._depths = List[np.ndarray[int, 1]]()

    def fit(self, X, y = None, sample_weight = None):
        Xa = np.asarray(X, dtype=float)
        n = Xa.shape[0]
        rnd = _check_rs(self.random_state)
        yy = rnd.uniform(size=n)
        ms = 0
        if self._ms_auto:
            ms = min(256, n)
        elif self._ms_kind == "int":
            ms = min(int(self._ms_value), n)
        else:
            ms = int(self._ms_value * float(n))
        self.max_samples_ = ms
        self._max_depth = int(math.ceil(math.log2(float(max(ms, 2)))))
        self._fit_bagging(Xa, yy, sample_weight, ms)
        self._path_len = List[List[float]]()
        self._depths = List[np.ndarray[int, 1]]()
        for t in self.estimators_:
            self._path_len.append([_average_path_length(float(v)) for v in t.tree_.nns])
            self._depths.append(t.tree_.compute_node_depths())
        if self._contamination_auto:
            self.offset_ = -0.5
        else:
            self.offset_ = float(np.percentile(self.score_samples(Xa), 100.0 * self.contamination))
        return self

    def score_samples(self, X):
        Xa = np.asarray(X, dtype=float)
        n = Xa.shape[0]
        depths = np.zeros(n)
        for j in range(len(self.estimators_)):
            leaves = self.estimators_[j].tree_.apply(self._sub(Xa, j))
            for i in range(n):
                l = int(leaves[i])
                depths[i] += float(self._depths[j][l]) + self._path_len[j][l] - 1.0
        den = float(len(self.estimators_)) * _average_path_length(float(self._max_samples))
        out = np.empty(n)
        for i in range(n):
            out[i] = -(2.0 ** (-(depths[i] / den if den != 0.0 else 1.0)))
        return out

    def decision_function(self, X):
        return self.score_samples(X) - self.offset_

    def predict(self, X):
        d = self.decision_function(X)
        out = np.ones(d.shape[0], dtype=int)
        for i in range(d.shape[0]):
            if d[i] < 0.0:
                out[i] = -1
        return out

    def fit_predict(self, X, y = None):
        return self.fit(X).predict(X)
