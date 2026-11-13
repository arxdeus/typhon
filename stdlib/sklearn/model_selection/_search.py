# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# GridSearchCV / RandomizedSearchCV (sklearn.model_selection._search).
# Candidates are evaluated sequentially; parameter values are _PV scalars
# (compare equal to plain float/int/str/bool/None). cv_results_ supports
# r["mean_test_score"] (float arrays), r["params"] (list of dicts) and
# r["param_<name>"] (list of values).

from ..base import estimator_repr as _estimator_repr
import numpy as np
import time
from ..base import clone
from ._validation import check_cv, _folds, _is_clf
from ._scorer_shim import _scores_all, _metric_names, _is_multi
from ._params import _PV, _norm_grid, _grid_iter, _sample_params, _apply_params

class _SearchResults:
    arrays: Dict[str, np.ndarray[float, 1]]
    ranks: Dict[str, np.ndarray[int, 1]]
    params: List[Dict[str, _PV]]

    def __init__(self):
        self.arrays = {}
        self.ranks = {}
        self.params = []

    def __getitem__(self, key: str) -> np.ndarray[float, 1]:
        return self.arrays[key]
    def __getitem__(self, key: Literal[str]):
        if key == "params":
            return self.params
        elif key[:5] == "rank_":
            return self.ranks[key]
        elif key[:6] == "param_":
            return self._param_col(key[6:])
        else:
            return self.arrays[key]

    def _param_col(self, name: str) -> List[_PV]:
        out = List[_PV]()
        for p in self.params:
            out.append(p[name] if name in p else _PV(4, 0.0, ""))
        return out

    def __contains__(self, key: str) -> bool:
        if key == "params" or key in self.arrays or key in self.ranks:
            return True
        if key[:6] == "param_":
            for p in self.params:
                if key[6:] in p:
                    return True
        return False

    def keys(self) -> List[str]:
        k = [s for s in self.arrays.keys()]
        for s in self.ranks.keys():
            k.append(s)
        names = Set[str]()
        for p in self.params:
            for s in p.keys():
                names.add("param_" + s)
        for s in names:
            k.append(s)
        k.append("params")
        return sorted(k)

def _rank_min(means: np.ndarray[float, 1]) -> np.ndarray[int, 1]:
    # rankdata(-means, method="min") with NaN -> lowest
    n = means.shape[0]
    allnan = True
    for v in means:
        if v == v:
            allnan = False
    r = np.ones(n, dtype=int)
    if allnan:
        return r
    lo = float(np.nanmin(means)) - 1.0
    m = means.copy()
    for i in range(n):
        if m[i] != m[i]:
            m[i] = lo
    for i in range(n):
        c = 1
        for j in range(n):
            if m[j] > m[i]:
                c += 1
        r[i] = c
    return r

class _BaseSearch[E, P, S, CV, R]:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    estimator: E
    _space: P
    scoring: S
    cv: CV
    refit: R
    return_train_score: bool
    _random: bool
    n_iter: int
    random_state: Optional[int]
    cv_results_: _SearchResults
    best_index_: int
    best_score_: float
    best_params_: Dict[str, _PV]
    best_estimator_: E
    n_splits_: int
    multimetric_: bool
    refit_time_: float
    scorer_names_: List[str]

    def __init__(self, estimator: E, space: P, scoring: S, cv: CV, refit: R,
                 return_train_score: bool, random: bool, n_iter: int, random_state: Optional[int]):
        self.estimator = estimator
        self._space = space
        self.scoring = scoring
        self.cv = cv
        self.refit = refit
        self.return_train_score = return_train_score
        self._random = random
        self.n_iter = n_iter
        self.random_state = random_state
        self.cv_results_ = _SearchResults()
        self.best_index_ = -1
        self.best_score_ = 0.0
        self.best_params_ = {}
        self.best_estimator_ = clone(estimator)
        self.n_splits_ = 0
        self.multimetric_ = _is_multi(scoring)
        self.refit_time_ = 0.0
        self.scorer_names_ = []

    def _candidates(self) -> List[Dict[str, _PV]]:
        if self._random:
            return _sample_params(self._space, self.n_iter, self.random_state)
        return _grid_iter(_norm_grid(self._space))

    def _refit_metric(self, names: List[str]) -> str:
        if not self.multimetric_:
            return "score"
        if isinstance(self.refit, str):
            if self.refit not in names:
                raise ValueError(f"For multi-metric scoring, the parameter refit must be set to a scorer key or a callable to refit an estimator with the best parameter setting on the whole data and make the best_* attributes available for that metric. If this is not needed, refit should be set to False explicitly. {self.refit!r} was passed.")
            return self.refit
        elif isinstance(self.refit, bool):
            if self.refit:
                raise ValueError(f"For multi-metric scoring, the parameter refit must be set to a scorer key or a callable to refit an estimator with the best parameter setting on the whole data and make the best_* attributes available for that metric. If this is not needed, refit should be set to False explicitly. True was passed.")
            return ""
        return ""

    def _do_refit(self) -> bool:
        if isinstance(self.refit, bool):
            return self.refit
        return True

    def fit(self, X, y = None, groups = None):
        Xa = np.asarray(X)
        ya = np.asarray(y)
        cvo = check_cv(self.cv, ya, classifier=_is_clf(self.estimator))
        folds = _folds(cvo, Xa, ya, groups)
        cands = self._candidates()
        names = _metric_names(self.scoring)
        refit_metric = self._refit_metric(names)
        nk = len(cands)
        nf = len(folds)
        m = len(names)
        self.n_splits_ = nf
        self.scorer_names_ = names
        te_sc = [np.empty((nk, nf)) for _ in range(m)]
        tr_sc = [np.empty((nk, nf)) for _ in range(m)]
        fit_t = np.empty((nk, nf))
        sc_t = np.empty((nk, nf))
        # sklearn order: for each candidate, for each split
        for ci in range(nk):
            for fi in range(nf):
                tr, te = folds[fi]
                est = clone(self.estimator)
                _apply_params(est, cands[ci])
                t0 = time.perf_counter()
                est.fit(Xa[tr], ya[tr])
                t1 = time.perf_counter()
                s = _scores_all(self.scoring, est, Xa[te], ya[te])
                t2 = time.perf_counter()
                fit_t[ci, fi] = t1 - t0
                sc_t[ci, fi] = t2 - t1
                for j in range(m):
                    te_sc[j][ci, fi] = s[j]
                if self.return_train_score:
                    st = _scores_all(self.scoring, est, Xa[tr], ya[tr])
                    for j in range(m):
                        tr_sc[j][ci, fi] = st[j]
        res = _SearchResults()
        self._store(res, "fit_time", fit_t, False, False)
        self._store(res, "score_time", sc_t, False, False)
        res.params = cands
        for j in range(m):
            nm = names[j] if self.multimetric_ else "score"
            self._store(res, "test_" + nm, te_sc[j], True, True)
            if self.return_train_score:
                self._store(res, "train_" + nm, tr_sc[j], True, False)
        self.cv_results_ = res
        if refit_metric != "" and (self._do_refit() or not self.multimetric_):
            ranks = res.ranks["rank_test_" + refit_metric]
            best = int(np.argmin(ranks))
            self.best_index_ = best
            self.best_params_ = cands[best]
            self.best_score_ = float(res.arrays["mean_test_" + refit_metric][best])
        if self._do_refit() and refit_metric != "":
            est = clone(self.estimator)
            _apply_params(est, cands[self.best_index_])
            t0 = time.perf_counter()
            est.fit(Xa, ya)
            self.refit_time_ = time.perf_counter() - t0
            self.best_estimator_ = est
        return self

    def _store(self, res: _SearchResults, key: str, a: np.ndarray[float, 2], splits: bool, rank: bool):
        nk, nf = a.shape
        if splits:
            for f in range(nf):
                res.arrays[f"split{f}_{key}"] = np.ascontiguousarray(a[:, f])
        means = np.mean(a, axis=1)
        res.arrays["mean_" + key] = means
        stds = np.empty(nk)
        for i in range(nk):
            s = 0.0
            for f in range(nf):
                d = a[i, f] - means[i]
                s += d * d
            stds[i] = np.sqrt(s / float(nf))
        res.arrays["std_" + key] = stds
        if rank:
            res.ranks["rank_" + key] = _rank_min(means)

    def _check_refit(self):
        if not self._do_refit():
            raise AttributeError("This search estimator was initialized with refit=False; the best estimator is not available.")

    def predict(self, X):
        self._check_refit()
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        self._check_refit()
        return self.best_estimator_.predict_proba(X)

    def predict_log_proba(self, X):
        self._check_refit()
        return self.best_estimator_.predict_log_proba(X)

    def decision_function(self, X):
        self._check_refit()
        return self.best_estimator_.decision_function(X)

    def transform(self, X):
        self._check_refit()
        return self.best_estimator_.transform(X)

    def inverse_transform(self, X):
        self._check_refit()
        return self.best_estimator_.inverse_transform(X)

    def score(self, X, y = None, sample_weight = None) -> float:
        from ._scorer_shim import _score_name, _score
        self._check_refit()
        if self.multimetric_:
            if isinstance(self.refit, str):
                return _score_name(self.refit, self.best_estimator_, X, y)
            raise ValueError("score() needs refit=<metric name> with multi-metric scoring")
        return _score(self.scoring, self.best_estimator_, X, y)

    @property
    def classes_(self):
        return self.best_estimator_.classes_

    @property
    def n_features_in_(self):
        return self.best_estimator_.n_features_in_

def GridSearchCV(estimator, param_grid, *, scoring = None, n_jobs = None, refit = True,
                 cv = 5, verbose: int = 0, pre_dispatch = "2*n_jobs", error_score = None,
                 return_train_score: bool = False):
    """param_grid: dict name -> list (values may be float/int/str/bool/None,
    mixed lists become Unions) or a list of such dicts."""
    return _BaseSearch(estimator, param_grid, scoring, cv, refit, return_train_score,
                       False, 0, None)

def RandomizedSearchCV(estimator, param_distributions, *, n_iter: int = 10, scoring = None,
                       n_jobs = None, refit = True, cv = 5, verbose: int = 0,
                       pre_dispatch = "2*n_jobs", random_state: Optional[int] = None,
                       error_score = None, return_train_score: bool = False):
    """param_distributions values: lists or scipy.stats frozen distributions
    (anything with .rvs(random_state=...)). random_state must be an int."""
    if random_state is None:
        raise ValueError("RandomizedSearchCV needs an integer random_state in Typhon")
    return _BaseSearch(estimator, param_distributions, scoring, cv, refit, return_train_score,
                       True, n_iter, random_state)
