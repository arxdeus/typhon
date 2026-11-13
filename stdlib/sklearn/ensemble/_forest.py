# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Random forests and extremely randomized trees (sklearn/ensemble/_forest.py).
# Trees are built sequentially (n_jobs is accepted and ignored); the random
# streams (per-tree seeds, bootstrap draws, splitter seeds) are those of
# sklearn, so fitted trees, predictions and importances are identical.

from ..base import estimator_repr as _estimator_repr
import math
import numpy as np
from ..tree._classes import (DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier,
                             ExtraTreeRegressor, _set_tree_params, _copy_tree_params,
                             _to_f32_colmajor, _missing_mask, _encode_classes, _check_rs, _r2)
from ._base import (MAX_INT, _get_n_samples_bootstrap, _parse_max_samples,
                    _generate_sample_indices, _generate_unsampled_indices, _bincount, _mean_rows)

def _balanced_class_weight(yenc: List[float], nc: int, indices: List[int]) -> List[float]:
    """compute_sample_weight('balanced', y, indices=indices) for encoded y."""
    n = len(yenc)
    cnt = [0] * nc
    for i in indices:
        cnt[int(yenc[i])] += 1
    present = 0
    for c in cnt:
        if c > 0:
            present += 1
    m = len(indices)
    cw = [float(m) / (float(present) * float(cnt[k])) if cnt[k] > 0 else 0.0 for k in range(nc)]
    return [cw[int(yenc[i])] for i in range(n)]

class _BaseForest[T]:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    n_estimators: int
    bootstrap: bool
    oob_score: bool
    n_jobs: Optional[int]
    random_state: Optional[int]
    verbose: int
    warm_start: bool
    _ms_kind: str
    _ms_value: float
    # tree parameters
    criterion: str
    splitter: str
    max_depth: Optional[int]
    min_samples_split: float
    _mss_int: bool
    min_samples_leaf: float
    _msl_int: bool
    min_weight_fraction_leaf: float
    max_features: str
    _mf_value: float
    max_leaf_nodes: Optional[int]
    min_impurity_decrease: float
    ccp_alpha: float
    monotonic_cst: List[int]
    class_weight: str
    _class_weight_dict: Dict[int, float]
    _is_clf: bool
    # fitted
    estimators_: List[T]
    _seeds: List[int]
    n_features_in_: int
    n_outputs_: int
    classes_: np.ndarray[int, 1]
    n_classes_: int
    oob_score_: float
    oob_decision_function_: np.ndarray[float, 2]
    oob_prediction_: np.ndarray[float, 1]
    _n_samples: int
    _n_samples_bootstrap: int
    _sample_weight: List[float]
    _has_w: bool
    _fitted: bool

    def __init__(self, is_clf: bool, n_estimators: int, criterion: str, max_depth: Optional[int],
                     min_samples_split, min_samples_leaf, min_weight_fraction_leaf: float,
                     max_features, max_leaf_nodes: Optional[int], min_impurity_decrease: float,
                     bootstrap: bool, oob_score: bool, n_jobs: Optional[int],
                     random_state: Optional[int], verbose: int, warm_start: bool, class_weight,
                     ccp_alpha: float, max_samples, monotonic_cst, splitter: str):
        self._is_clf = is_clf
        self.n_estimators = n_estimators
        _set_tree_params(self, criterion, splitter, max_depth, min_samples_split,
                         min_samples_leaf, min_weight_fraction_leaf, max_features,
                         max_leaf_nodes, min_impurity_decrease, ccp_alpha, monotonic_cst)
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        k, v = _parse_max_samples(max_samples)
        self._ms_kind = k
        self._ms_value = v
        self._class_weight_dict = Dict[int, float]()
        if class_weight is None:
            self.class_weight = ""
        elif isinstance(class_weight, str):
            self.class_weight = class_weight
        else:
            self.class_weight = "dict"
            for kk, vv in class_weight.items():
                self._class_weight_dict[int(kk)] = float(vv)
        self.estimators_ = List[T]()
        self._seeds = List[int]()
        self.n_features_in_ = 0
        self.n_outputs_ = 1
        self.classes_ = np.empty(0, dtype=int)
        self.n_classes_ = 0
        self.oob_score_ = 0.0
        self.oob_decision_function_ = np.empty((0, 0))
        self.oob_prediction_ = np.empty(0)
        self._n_samples = 0
        self._n_samples_bootstrap = 0
        self._sample_weight = List[float]()
        self._has_w = False
        self._fitted = False

    def _make_tree(self, seed: int) -> T:
        t = self._new_tree()
        _copy_tree_params(t, self)
        t.random_state = seed
        return t

    def fit(self, X, y, sample_weight = None):
        Xf, n, p = _to_f32_colmajor(X)
        missing = _missing_mask(Xf, n, p)
        ya = np.asarray(y)
        if ya.shape[0] != n:
            raise ValueError("X and y have inconsistent numbers of samples")
        self.n_features_in_ = p
        self._n_samples = n
        yv = List[float]()
        nc = 1
        if self._is_clf:
            classes, enc = _encode_classes(ya)
            self.classes_ = classes
            nc = classes.shape[0]
            self.n_classes_ = nc
            yv = enc
        else:
            yv = [float(ya[i]) for i in range(n)]
            if self.criterion == "poisson":
                s = 0.0
                for v in yv:
                    if v < 0.0:
                        raise ValueError("Some value(s) of y are negative which is not allowed for Poisson regression.")
                    s += v
                if s <= 0.0:
                    raise ValueError("Sum of y is not strictly positive which is necessary for Poisson regression.")
        has_w = sample_weight is not None
        sw = [1.0] * n
        if sample_weight is not None:
            swa = np.asarray(sample_weight, dtype=float)
            for i in range(n):
                sw[i] = float(swa[i])
        subsample_cw = False
        if self._is_clf and self.class_weight != "":
            if self.class_weight == "balanced_subsample" and self.bootstrap:
                subsample_cw = True
            else:
                cw = List[float]()
                if self.class_weight == "balanced" or self.class_weight == "balanced_subsample":
                    cw = _balanced_class_weight(yv, nc, list(range(n)))
                else:
                    cw = [self._class_weight_dict.get(int(self.classes_[int(v)]), 1.0) for v in yv]
                for i in range(n):
                    sw[i] = sw[i] * cw[i] if has_w else cw[i]
                has_w = True
        self._sample_weight = sw
        self._has_w = has_w
        if not self.bootstrap and self._ms_kind != "none":
            raise ValueError("`max_sample` cannot be set if `bootstrap=False`. Either switch to `bootstrap=True` or set `max_sample=None`.")
        nb = 0
        if self.bootstrap:
            tot = 0.0
            for w in sw:
                tot += w
            nb = _get_n_samples_bootstrap(n, self._ms_kind, self._ms_value, tot, has_w)
        self._n_samples_bootstrap = nb
        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")
        rs = _check_rs(self.random_state)
        trees = List[T]()
        seeds = List[int]()
        for i in range(self.n_estimators):
            seed = int(rs.randint(MAX_INT))
            seeds.append(seed)
            trees.append(self._make_tree(seed))
        for i in range(self.n_estimators):
            t = trees[i]
            if self.bootstrap:
                idx = _generate_sample_indices(seeds[i], n, nb, sw, has_w)
                wt = _bincount(idx, n)
                if subsample_cw:
                    cw = _balanced_class_weight(yv, nc, [int(idx[j]) for j in range(idx.shape[0])])
                    for j in range(n):
                        wt[j] = wt[j] * cw[j]
                t._fit_encoded(Xf, n, p, yv, nc, wt, True, _check_rs(seeds[i]), missing)
            else:
                t._fit_encoded(Xf, n, p, yv, nc, sw, has_w, _check_rs(seeds[i]), missing)
            if self._is_clf:
                t.classes_ = np.arange(nc)
                t.n_classes_ = nc
        self.estimators_ = trees
        self._seeds = seeds
        self._fitted = True
        if self.oob_score:
            self._set_oob(X, yv)
        return self

    def _check_X(self, X):
        if not self._fitted:
            raise ValueError("This estimator is not fitted yet. Call 'fit' first.")
        Xa = np.asarray(X, dtype=float)
        if Xa.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xa.shape[1]} features, but the forest is expecting {self.n_features_in_} features as input.")
        return Xa

    def apply(self, X):
        Xa = self._check_X(X)
        n = Xa.shape[0]
        out = np.empty((n, len(self.estimators_)), dtype=int)
        for j in range(len(self.estimators_)):
            leaves = self.estimators_[j].tree_.apply(Xa)
            for i in range(n):
                out[i, j] = leaves[i]
        return out

    @property
    def feature_importances_(self):
        rows = List[np.ndarray[float, 1]]()
        for t in self.estimators_:
            if t.tree_.node_count > 1:
                rows.append(t.tree_.compute_feature_importances())
        p = self.n_features_in_
        if len(rows) == 0:
            return np.zeros(p)
        m = _mean_rows(rows, p)
        return m / np.sum(m)

    @property
    def estimators_samples_(self):
        out = List[np.ndarray[int, 1]]()
        for s in self._seeds:
            if not self.bootstrap:
                out.append(np.arange(self._n_samples))
            else:
                out.append(_generate_sample_indices(s, self._n_samples, self._n_samples_bootstrap,
                                                    self._sample_weight, self._has_w))
        return out

    def __len__(self) -> int:
        return len(self.estimators_)

    def __getitem__(self, i: int) -> T:
        return self.estimators_[i]

class _ForestClassifier[T](_BaseForest[T]):
    def predict_proba(self, X):
        Xa = self._check_X(X)
        n = Xa.shape[0]
        nc = self.n_classes_
        proba = np.zeros((n, nc))
        for t in self.estimators_:
            proba += t.predict_proba(Xa)
        proba /= float(len(self.estimators_))
        return proba

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def predict(self, X):
        proba = self.predict_proba(X)
        n = proba.shape[0]
        out = np.empty(n, dtype=int)
        for i in range(n):
            best = 0
            for c in range(1, self.n_classes_):
                if proba[i, c] > proba[i, best]:
                    best = c
            out[i] = self.classes_[best]
        return out

    def score(self, X, y, sample_weight = None) -> float:
        from ..base import _accuracy
        return _accuracy(y, self.predict(X))

    def _set_oob(self, X, yv: List[float]):
        Xa = np.asarray(X, dtype=float)
        n = self._n_samples
        nc = self.n_classes_
        oob = np.zeros((n, nc))
        cnt = [0] * n
        for j in range(len(self.estimators_)):
            un = _generate_unsampled_indices(self._seeds[j], n, self._n_samples_bootstrap,
                                             self._sample_weight, self._has_w)
            if len(un) == 0:
                continue
            pr = self.estimators_[j].predict_proba(Xa[np.array(un)])
            for r in range(len(un)):
                i = un[r]
                for c in range(nc):
                    oob[i, c] += pr[r, c]
                cnt[i] += 1
        for i in range(n):
            d = float(cnt[i]) if cnt[i] > 0 else 1.0
            for c in range(nc):
                oob[i, c] /= d
        self.oob_decision_function_ = oob
        correct = 0
        for i in range(n):
            best = 0
            for c in range(1, nc):
                if oob[i, c] > oob[i, best]:
                    best = c
            if float(best) == yv[i]:
                correct += 1
        self.oob_score_ = float(correct) / float(n)

class _ForestRegressor[T](_BaseForest[T]):
    def predict(self, X):
        Xa = self._check_X(X)
        n = Xa.shape[0]
        y_hat = np.zeros(n)
        for t in self.estimators_:
            y_hat += t.predict(Xa)
        y_hat /= float(len(self.estimators_))
        return y_hat

    def score(self, X, y, sample_weight = None) -> float:
        return _r2(y, self.predict(X))

    def _set_oob(self, X, yv: List[float]):
        Xa = np.asarray(X, dtype=float)
        n = self._n_samples
        oob = np.zeros(n)
        cnt = [0] * n
        for j in range(len(self.estimators_)):
            un = _generate_unsampled_indices(self._seeds[j], n, self._n_samples_bootstrap,
                                             self._sample_weight, self._has_w)
            if len(un) == 0:
                continue
            pr = self.estimators_[j].predict(Xa[np.array(un)])
            for r in range(len(un)):
                oob[un[r]] += pr[r]
                cnt[un[r]] += 1
        for i in range(n):
            oob[i] /= float(cnt[i]) if cnt[i] > 0 else 1.0
        self.oob_prediction_ = oob
        self.oob_score_ = _r2(np.array(yv), oob)

class RandomForestClassifier(_ForestClassifier[DecisionTreeClassifier]):
    @property
    def _estimator_type(self) -> str:
        return "classifier"

    def __init__(self, n_estimators: int = 100, *, criterion: str = "gini",
                 max_depth: Optional[int] = None, min_samples_split = 2, min_samples_leaf = 1,
                 min_weight_fraction_leaf: float = 0.0, max_features = "sqrt",
                 max_leaf_nodes: Optional[int] = None, min_impurity_decrease: float = 0.0,
                 bootstrap: bool = True, oob_score: bool = False, n_jobs: Optional[int] = None,
                 random_state: Optional[int] = None, verbose: int = 0, warm_start: bool = False,
                 class_weight = None, ccp_alpha: float = 0.0, max_samples = None,
                 monotonic_cst = None):
        _BaseForest[DecisionTreeClassifier].__init__(self, True, n_estimators, criterion, max_depth, min_samples_split,
                          min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes,
                          min_impurity_decrease, bootstrap, oob_score, n_jobs, random_state,
                          verbose, warm_start, class_weight, ccp_alpha, max_samples,
                          monotonic_cst, "best")

    def _new_tree(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier()

class RandomForestRegressor(_ForestRegressor[DecisionTreeRegressor]):
    @property
    def _estimator_type(self) -> str:
        return "regressor"

    def __init__(self, n_estimators: int = 100, *, criterion: str = "squared_error",
                 max_depth: Optional[int] = None, min_samples_split = 2, min_samples_leaf = 1,
                 min_weight_fraction_leaf: float = 0.0, max_features = 1.0,
                 max_leaf_nodes: Optional[int] = None, min_impurity_decrease: float = 0.0,
                 bootstrap: bool = True, oob_score: bool = False, n_jobs: Optional[int] = None,
                 random_state: Optional[int] = None, verbose: int = 0, warm_start: bool = False,
                 ccp_alpha: float = 0.0, max_samples = None, monotonic_cst = None):
        _BaseForest[DecisionTreeRegressor].__init__(self, False, n_estimators, criterion, max_depth, min_samples_split,
                          min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes,
                          min_impurity_decrease, bootstrap, oob_score, n_jobs, random_state,
                          verbose, warm_start, None, ccp_alpha, max_samples, monotonic_cst, "best")

    def _new_tree(self) -> DecisionTreeRegressor:
        return DecisionTreeRegressor()

class ExtraTreesClassifier(_ForestClassifier[ExtraTreeClassifier]):
    @property
    def _estimator_type(self) -> str:
        return "classifier"

    def __init__(self, n_estimators: int = 100, *, criterion: str = "gini",
                 max_depth: Optional[int] = None, min_samples_split = 2, min_samples_leaf = 1,
                 min_weight_fraction_leaf: float = 0.0, max_features = "sqrt",
                 max_leaf_nodes: Optional[int] = None, min_impurity_decrease: float = 0.0,
                 bootstrap: bool = False, oob_score: bool = False, n_jobs: Optional[int] = None,
                 random_state: Optional[int] = None, verbose: int = 0, warm_start: bool = False,
                 class_weight = None, ccp_alpha: float = 0.0, max_samples = None,
                 monotonic_cst = None):
        _BaseForest[ExtraTreeClassifier].__init__(self, True, n_estimators, criterion, max_depth, min_samples_split,
                          min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes,
                          min_impurity_decrease, bootstrap, oob_score, n_jobs, random_state,
                          verbose, warm_start, class_weight, ccp_alpha, max_samples,
                          monotonic_cst, "random")

    def _new_tree(self) -> ExtraTreeClassifier:
        return ExtraTreeClassifier()

class ExtraTreesRegressor(_ForestRegressor[ExtraTreeRegressor]):
    @property
    def _estimator_type(self) -> str:
        return "regressor"

    def __init__(self, n_estimators: int = 100, *, criterion: str = "squared_error",
                 max_depth: Optional[int] = None, min_samples_split = 2, min_samples_leaf = 1,
                 min_weight_fraction_leaf: float = 0.0, max_features = 1.0,
                 max_leaf_nodes: Optional[int] = None, min_impurity_decrease: float = 0.0,
                 bootstrap: bool = False, oob_score: bool = False, n_jobs: Optional[int] = None,
                 random_state: Optional[int] = None, verbose: int = 0, warm_start: bool = False,
                 ccp_alpha: float = 0.0, max_samples = None, monotonic_cst = None):
        _BaseForest[ExtraTreeRegressor].__init__(self, False, n_estimators, criterion, max_depth, min_samples_split,
                          min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes,
                          min_impurity_decrease, bootstrap, oob_score, n_jobs, random_state,
                          verbose, warm_start, None, ccp_alpha, max_samples, monotonic_cst, "random")

    def _new_tree(self) -> ExtraTreeRegressor:
        return ExtraTreeRegressor()
