# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

import numpy as np
from ..base import clone, _accuracy
from ._split import check_cv

def _take(X, idx):
    return np.asarray(X)[idx]

def _cvp_impl(estimator, X, y, cv):
    cvo = check_cv(5 if cv is None else cv)
    Xa = np.asarray(X)
    ya = np.asarray(y)
    n = Xa.shape[0]
    out = np.zeros(n, dtype=ya.dtype)
    for tr, te in cvo.split(Xa, ya):
        est = clone(estimator)
        est.fit(Xa[tr], ya[tr])
        out[te] = np.asarray(est.predict(Xa[te]))
    return out

def _cvs_impl(estimator, X, y, cv):
    cvo = check_cv(5 if cv is None else cv)
    Xa = np.asarray(X)
    ya = np.asarray(y)
    scores = List[float]()
    for tr, te in cvo.split(Xa, ya):
        est = clone(estimator)
        est.fit(Xa[tr], ya[tr])
        scores.append(_accuracy(ya[te], est.predict(Xa[te])))
    return np.array(scores)

def cross_val_predict(estimator, X, y = None, *, groups = None, cv = None,
                      n_jobs = None, verbose: int = 0, params = None,
                      pre_dispatch: str = "2*n_jobs", method: str = "predict"):
    """Out-of-fold predictions. Folds run sequentially (n_jobs ignored).
    `estimator` may be a Union (e.g. a value of a dict of different pipelines)."""
    if isinstance(estimator, Union):
        for tag, T in static.vars_types(estimator, with_index=True):
            if Union._get_tag(estimator) == tag:
                return _cvp_impl(Union._get_data(estimator, T), X, y, cv)
        raise TypeError("invalid union")
    else:
        return _cvp_impl(estimator, X, y, cv)

def cross_val_score(estimator, X, y = None, *, groups = None, scoring = None,
                    cv = None, n_jobs = None, verbose: int = 0, params = None,
                    pre_dispatch: str = "2*n_jobs", error_score = None):
    """Per-fold accuracy (classifier default scorer). Sequential."""
    if isinstance(estimator, Union):
        for tag, T in static.vars_types(estimator, with_index=True):
            if Union._get_tag(estimator) == tag:
                return _cvs_impl(Union._get_data(estimator, T), X, y, cv)
        raise TypeError("invalid union")
    else:
        return _cvs_impl(estimator, X, y, cv)

# ------------------------------------------------------------ grid search

def _set_param_rt(est, key: str, val: float) -> bool:
    """Set parameter `key` (runtime string, sklearn `step__param` nesting) to a
    numeric value. Returns False when no field matches."""
    if hasattr(est, "_sk_clone") and hasattr(est, "steps"):
        i = key.find("__")
        if i < 0:
            return False
        head = key[:i]
        rest = key[i + 2:]
        ok = False
        for st in est.steps:
            if st[0] == head:
                ok = _set_param_rt(st[1], rest, val)
        return ok
    else:
        ok = False
        for k, v in static.vars(est):
            if k == key:
                if isinstance(v, float):
                    setattr(est, k, float(val)); ok = True
                elif isinstance(v, int):
                    setattr(est, k, int(val)); ok = True
                elif isinstance(v, Optional[float]):
                    setattr(est, k, Optional[float](float(val))); ok = True
                elif isinstance(v, Optional[int]):
                    setattr(est, k, Optional[int](int(val))); ok = True
        return ok

def _param_grid(grid: Dict[str, List[float]]) -> List[List[Tuple[str, float]]]:
    keys = sorted(grid.keys())
    out = [List[Tuple[str, float]]()]
    for k in keys:
        nxt = List[List[Tuple[str, float]]]()
        for base in out:
            for v in grid[k]:
                c = base.copy()
                c.append((k, v))
                nxt.append(c)
        out = nxt
    return out

class ParameterGrid:
    grid: Dict[str, List[float]]
    def __init__(self, param_grid: Dict[str, List[float]]):
        self.grid = param_grid
    def __iter__(self):
        for c in _param_grid(self.grid):
            yield {k: v for k, v in c}
    def __len__(self) -> int:
        return len(_param_grid(self.grid))

class GridSearchCV[E, CV]:
    """Exhaustive search over a numeric parameter grid (dict name -> list of
    floats; ints are cast), accuracy scoring, refit on the full data."""
    estimator: E
    param_grid: Dict[str, List[float]]
    cv: CV
    refit: bool
    cv_results_: Dict[str, np.ndarray[float, 1]]
    params_: List[Dict[str, float]]
    best_index_: int
    best_score_: float
    best_params_: Dict[str, float]
    best_estimator_: E
    n_splits_: int

    def __init__(self, estimator: E, param_grid: Dict[str, List[float]], *, scoring = None,
                 n_jobs = None, refit: bool = True, cv: CV = 5, verbose: int = 0,
                 pre_dispatch = "2*n_jobs", error_score = None, return_train_score: bool = False):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.refit = refit
        self.cv_results_ = {}
        self.params_ = []
        self.best_index_ = -1
        self.best_score_ = 0.0
        self.best_params_ = {}
        self.best_estimator_ = clone(estimator)
        self.n_splits_ = 0

    def _sk_clone(self):
        return GridSearchCV(clone(self.estimator), self.param_grid, refit=self.refit, cv=self.cv)

    def fit(self, X, y):
        cvo = check_cv(self.cv)
        Xa = np.asarray(X)
        ya = np.asarray(y)
        folds = [(tr, te) for tr, te in cvo.split(Xa, ya)]
        combos = _param_grid(self.param_grid)
        nk = len(combos)
        nf = len(folds)
        self.n_splits_ = nf
        means = np.empty(nk)
        stds = np.empty(nk)
        split_scores = [np.empty(nk) for _ in range(nf)]
        self.params_ = []
        for ci in range(nk):
            sc = np.empty(nf)
            for fi in range(nf):
                tr, te = folds[fi]
                est = clone(self.estimator)
                for kv in combos[ci]:
                    if not _set_param_rt(est, kv[0], kv[1]):
                        raise ValueError(f"Invalid parameter {kv[0]!r} for estimator")
                est.fit(Xa[tr], ya[tr])
                sc[fi] = _accuracy(ya[te], est.predict(Xa[te]))
                split_scores[fi][ci] = sc[fi]
            means[ci] = np.mean(sc)
            stds[ci] = np.std(sc)
            self.params_.append({kv[0]: kv[1] for kv in combos[ci]})
        # rank_test_score: rankdata(-mean, method="min"); best = first rank 1
        best = 0
        for ci in range(1, nk):
            if means[ci] > means[best]:
                best = ci
        rank = np.empty(nk)
        for ci in range(nk):
            r = 1
            for cj in range(nk):
                if means[cj] > means[ci]:
                    r += 1
            rank[ci] = float(r)
        res = Dict[str, np.ndarray[float, 1]]()
        res["mean_test_score"] = means
        res["std_test_score"] = stds
        res["rank_test_score"] = rank
        for fi in range(nf):
            res[f"split{fi}_test_score"] = split_scores[fi]
        self.cv_results_ = res
        self.best_index_ = best
        self.best_score_ = means[best]
        self.best_params_ = self.params_[best]
        if self.refit:
            est = clone(self.estimator)
            for kv in combos[best]:
                _set_param_rt(est, kv[0], kv[1])
            est.fit(Xa, ya)
            self.best_estimator_ = est
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def score(self, X, y, sample_weight = None):
        return _accuracy(y, self.predict(X))
