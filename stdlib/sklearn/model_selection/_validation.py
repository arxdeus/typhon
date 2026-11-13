# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# sklearn.model_selection._validation: cross_validate, cross_val_score,
# cross_val_predict, learning_curve, validation_curve, permutation_test_score.
# Folds run sequentially (n_jobs ignored); fit errors propagate
# (error_score="raise" semantics).

import numpy as np
import time
from ..base import clone
from ._split import KFold, StratifiedKFold
from ._scorer_shim import _score, _scores_all, _metric_names, _is_multi
from ._params import _pv, _apply_params, _PV

def _is_clf(est) -> bool:
    if hasattr(est, "_estimator_type"):
        return est._estimator_type == "classifier"
    elif hasattr(est, "steps"):
        return _is_clf(est.steps[-1][1])
    elif hasattr(est, "best_estimator_"):
        return _is_clf(est.estimator)
    else:
        return hasattr(est, "classes_") or hasattr(est, "predict_proba")

class _IntCV:
    """check_cv(int): StratifiedKFold for classifiers with binary/multiclass
    y, KFold otherwise (decided per split() from y)."""
    n_splits: int
    classifier: bool

    def __init__(self, n_splits: int, classifier: bool):
        self.n_splits = n_splits
        self.classifier = classifier

    def get_n_splits(self, X = None, y = None, groups = None) -> int:
        return self.n_splits

    def split(self, X, y = None, groups = None):
        strat = False
        if self.classifier and y is not None:
            from ..utils.multiclass import type_of_target
            t = type_of_target(y)
            strat = t == "binary" or t == "multiclass"
        out = List[Tuple[np.ndarray[int, 1], np.ndarray[int, 1]]]()
        if strat:
            for tr, te in StratifiedKFold(self.n_splits).split(X, y):
                out.append((tr, te))
        else:
            for tr, te in KFold(self.n_splits).split(X):
                out.append((tr, te))
        return out

def check_cv(cv = 5, y = None, *, classifier: bool = False):
    if cv is None:
        return _IntCV(5, classifier)
    elif isinstance(cv, int):
        return _IntCV(cv, classifier)
    else:
        return cv

def _folds(cv, X, y, groups) -> List[Tuple[np.ndarray[int, 1], np.ndarray[int, 1]]]:
    out = List[Tuple[np.ndarray[int, 1], np.ndarray[int, 1]]]()
    for tr, te in cv.split(X, y, groups):
        out.append((np.asarray(tr), np.asarray(te)))
    return out

def _unwrap_union(est, f):
    return f(est)

class _CVResults[E]:
    """Return value of cross_validate (sklearn: dict). r["test_score"] etc.
    give float arrays; r["estimator"] / r["indices"] the fitted estimators /
    fold indices."""
    scores: Dict[str, np.ndarray[float, 1]]
    estimators: List[E]
    train_indices: List[np.ndarray[int, 1]]
    test_indices: List[np.ndarray[int, 1]]

    def __init__(self):
        self.scores = {}
        self.estimators = []
        self.train_indices = []
        self.test_indices = []

    def __getitem__(self, key: str) -> np.ndarray[float, 1]:
        return self.scores[key]
    def __getitem__(self, key: Literal[str]):
        if key == "estimator":
            return self.estimators
        elif key == "indices":
            return {"train": self.train_indices, "test": self.test_indices}
        else:
            return self.scores[key]

    def __contains__(self, key: str) -> bool:
        return key in self.scores or (key == "estimator" and len(self.estimators) > 0) or (key == "indices" and len(self.test_indices) > 0)

    def keys(self) -> List[str]:
        k = sorted(self.scores.keys())
        if len(self.estimators) > 0:
            k.append("estimator")
        if len(self.test_indices) > 0:
            k.append("indices")
        return sorted(k)

def _cross_validate(estimator, X, y, groups, scoring, cv, return_train_score: bool,
                    return_estimator: bool, return_indices: bool):
    Xa = np.asarray(X)
    ya = np.asarray(y)
    cvo = check_cv(cv, ya, classifier=_is_clf(estimator))
    folds = _folds(cvo, Xa, ya, groups)
    names = _metric_names(scoring)
    multi = _is_multi(scoring)
    nf = len(folds)
    m = len(names)
    test = [np.empty(nf) for _ in range(m)]
    train = [np.empty(nf) for _ in range(m)]
    fit_t = np.empty(nf)
    score_t = np.empty(nf)
    res = _CVResults[type(clone(estimator))]()
    for fi in range(nf):
        tr, te = folds[fi]
        est = clone(estimator)
        t0 = time.perf_counter()
        est.fit(Xa[tr], ya[tr])
        t1 = time.perf_counter()
        sc = _scores_all(scoring, est, Xa[te], ya[te])
        t2 = time.perf_counter()
        fit_t[fi] = t1 - t0
        score_t[fi] = t2 - t1
        for j in range(m):
            test[j][fi] = sc[j]
        if return_train_score:
            st = _scores_all(scoring, est, Xa[tr], ya[tr])
            for j in range(m):
                train[j][fi] = st[j]
        if return_estimator:
            res.estimators.append(est)
        if return_indices:
            res.train_indices.append(tr)
            res.test_indices.append(te)
    res.scores["fit_time"] = fit_t
    res.scores["score_time"] = score_t
    for j in range(m):
        nm = names[j] if multi else "score"
        res.scores["test_" + nm] = test[j]
        if return_train_score:
            res.scores["train_" + nm] = train[j]
    return res

def cross_validate(estimator, X, y = None, *, groups = None, scoring = None, cv = None,
                   n_jobs = None, verbose: int = 0, params = None, pre_dispatch = "2*n_jobs",
                   return_train_score: bool = False, return_estimator: bool = False,
                   return_indices: bool = False, error_score = None):
    return _cross_validate(estimator, X, y, groups, scoring, cv, return_train_score,
                           return_estimator, return_indices)

def _cvs_impl(estimator, X, y, groups, scoring, cv):
    Xa = np.asarray(X)
    ya = np.asarray(y)
    cvo = check_cv(cv, ya, classifier=_is_clf(estimator))
    scores = List[float]()
    for tr, te in _folds(cvo, Xa, ya, groups):
        est = clone(estimator)
        est.fit(Xa[tr], ya[tr])
        scores.append(_score(scoring, est, Xa[te], ya[te]))
    return np.array(scores)

def cross_val_score(estimator, X, y = None, *, groups = None, scoring = None,
                    cv = None, n_jobs = None, verbose: int = 0, params = None,
                    pre_dispatch: str = "2*n_jobs", error_score = None):
    """`estimator` may be a Union (e.g. a value of a dict of pipelines)."""
    if isinstance(estimator, Union):
        for tag, T in static.vars_types(estimator, with_index=True):
            if Union._get_tag(estimator) == tag:
                return _cvs_impl(Union._get_data(estimator, T), X, y, groups, scoring, cv)
        raise TypeError("invalid union")
    else:
        return _cvs_impl(estimator, X, y, groups, scoring, cv)

def _call_method(est, X, method: Literal[str]):
    if method == "predict":
        return est.predict(X)
    elif method == "predict_proba":
        return est.predict_proba(X)
    elif method == "predict_log_proba":
        return est.predict_log_proba(X)
    elif method == "decision_function":
        return est.decision_function(X)
    else:
        compile_error("cross_val_predict: unsupported method")

def _cvp_impl(estimator, X, y, groups, cv, method: Literal[str]):
    Xa = np.asarray(X)
    ya = np.asarray(y)
    cvo = check_cv(cv, ya, classifier=_is_clf(estimator))
    folds = _folds(cvo, Xa, ya, groups)
    n = Xa.shape[0]
    if method == "predict":
        out = np.zeros(n, dtype=ya.dtype)
        for tr, te in folds:
            est = clone(estimator)
            est.fit(Xa[tr], ya[tr])
            out[te] = np.asarray(est.predict(Xa[te]))
        return out
    else:
        # label-encode y; fold columns are mapped onto all classes
        classes, ye = np.unique(ya, return_inverse=True)
        tr0, te0 = folds[0]
        est0 = clone(estimator)
        est0.fit(Xa[tr0], ye[tr0])
        P0 = np.asarray(_call_method(est0, Xa[te0], method), dtype=float)
        return _cvp_proba(P0, estimator, Xa, ye, classes.shape[0], folds, method)

def _cvp_proba(P0, estimator, Xa, ye, nc: int, folds, method: Literal[str]):
    n = Xa.shape[0]
    if static.len(P0.shape) == 1:
        # binary decision_function: 1-d like sklearn
        out1 = np.zeros(n)
        for fi in range(len(folds)):
            tr, te = folds[fi]
            if fi == 0:
                out1[te] = P0
            else:
                est = clone(estimator)
                est.fit(Xa[tr], ye[tr])
                out1[te] = np.asarray(_call_method(est, Xa[te], method), dtype=float)
        return out1
    else:
        fill = 0.0 if method == "predict_proba" else -1.7976931348623157e308
        if P0.shape[1] == 1:
            # binary decision_function returned as a column: (n_samples, 1)
            o1 = np.zeros((n, 1))
            for fi in range(len(folds)):
                tr, te = folds[fi]
                if fi == 0:
                    P = P0
                else:
                    est = clone(estimator)
                    est.fit(Xa[tr], ye[tr])
                    P = np.asarray(_call_method(est, Xa[te], method), dtype=float)
                for i in range(te.shape[0]):
                    o1[te[i], 0] = P[i, 0]
            return o1
        out = np.zeros((n, nc))
        for fi in range(len(folds)):
            tr, te = folds[fi]
            est = clone(estimator)
            est.fit(Xa[tr], ye[tr])
            P = P0 if fi == 0 else np.asarray(_call_method(est, Xa[te], method), dtype=float)
            fc = np.asarray(est.classes_)
            if fc.shape[0] != nc:
                if method == "decision_function" and fc.shape[0] <= 2:
                    raise ValueError(f"Only {fc.shape[0]} class/es in training fold, but {nc} in overall dataset. This is not supported for decision_function with imbalanced folds.")
                for i in range(te.shape[0]):
                    for c in range(nc):
                        out[te[i], c] = fill
            for i in range(te.shape[0]):
                for c in range(fc.shape[0]):
                    out[te[i], int(fc[c])] = P[i, c]
        return out

def cross_val_predict(estimator, X, y = None, *, groups = None, cv = None,
                      n_jobs = None, verbose: int = 0, params = None,
                      pre_dispatch: str = "2*n_jobs", method: Literal[str] = "predict"):
    """Out-of-fold predictions. method "predict_proba"/"predict_log_proba"/
    "decision_function" return a 2-d (n_samples, n_classes) array (a binary
    decision_function gives shape (n_samples, 1); use [:, 0])."""
    if isinstance(estimator, Union):
        for tag, T in static.vars_types(estimator, with_index=True):
            if Union._get_tag(estimator) == tag:
                return _cvp_impl(Union._get_data(estimator, T), X, y, groups, cv, method)
        raise TypeError("invalid union")
    else:
        return _cvp_impl(estimator, X, y, groups, cv, method)

def _translate_train_sizes(train_sizes, n_max: int):
    ts = np.asarray(train_sizes)
    if ts.dtype is float:
        lo = float(np.min(ts)); hi = float(np.max(ts))
        if lo <= 0.0 or hi > 1.0:
            raise ValueError(f"train_sizes has been interpreted as fractions of the maximum number of training samples and must be within (0, 1], but is within [{lo}, {hi}].")
        a = (ts * float(n_max)).astype(int)
        a = np.clip(a, 1, n_max)
        return np.unique(a)
    else:
        a = ts.astype(int)
        lo = int(np.min(a)); hi = int(np.max(a))
        if lo <= 0 or hi > n_max:
            raise ValueError(f"train_sizes has been interpreted as absolute numbers of training samples and must be within (0, {n_max}], but is within [{lo}, {hi}].")
        return np.unique(a)

def learning_curve(estimator, X, y, *, groups = None, train_sizes = np.linspace(0.1, 1.0, 5),
                   cv = None, scoring = None, exploit_incremental_learning: bool = False,
                   n_jobs = None, pre_dispatch = "all", verbose: int = 0, shuffle: bool = False,
                   random_state: Optional[int] = None, error_score = None,
                   return_times: bool = False, params = None):
    """Returns (train_sizes_abs, train_scores, test_scores), scores of shape
    (n_ticks, n_folds). return_times / exploit_incremental_learning are not
    supported."""
    Xa = np.asarray(X)
    ya = np.asarray(y)
    cvo = check_cv(cv, ya, classifier=_is_clf(estimator))
    folds = _folds(cvo, Xa, ya, groups)
    nf = len(folds)
    sizes = _translate_train_sizes(train_sizes, folds[0][0].shape[0])
    nt = sizes.shape[0]
    if shuffle:
        rng = np.random.RandomState(random_state)
        for fi in range(nf):
            tr, te = folds[fi]
            folds[fi] = (rng.permutation(tr), te)
    trs = np.empty((nt, nf))
    tes = np.empty((nt, nf))
    for fi in range(nf):
        tr, te = folds[fi]
        for k in range(nt):
            sub = tr[:sizes[k]]
            est = clone(estimator)
            est.fit(Xa[sub], ya[sub])
            tes[k, fi] = _score(scoring, est, Xa[te], ya[te])
            trs[k, fi] = _score(scoring, est, Xa[sub], ya[sub])
    return sizes, trs, tes

def validation_curve(estimator, X, y, *, param_name: str, param_range, groups = None,
                     cv = None, scoring = None, n_jobs = None, pre_dispatch = "all",
                     verbose: int = 0, error_score = None, params = None):
    """Returns (train_scores, test_scores) of shape (n_params, n_folds)."""
    Xa = np.asarray(X)
    ya = np.asarray(y)
    cvo = check_cv(cv, ya, classifier=_is_clf(estimator))
    folds = _folds(cvo, Xa, ya, groups)
    vals = List[_PV]()
    for v in param_range:
        vals.append(_pv(v))
    nf = len(folds)
    npar = len(vals)
    trs = np.empty((npar, nf))
    tes = np.empty((npar, nf))
    for fi in range(nf):
        tr, te = folds[fi]
        for k in range(npar):
            est = clone(estimator)
            _apply_params(est, {param_name: vals[k]})
            est.fit(Xa[tr], ya[tr])
            tes[k, fi] = _score(scoring, est, Xa[te], ya[te])
            trs[k, fi] = _score(scoring, est, Xa[tr], ya[tr])
    return trs, tes

def _perm_score(estimator, Xa, ya, groups, cvo, scoring) -> float:
    s = 0.0
    k = 0
    for tr, te in _folds(cvo, Xa, ya, groups):
        est = clone(estimator)
        est.fit(Xa[tr], ya[tr])
        s += _score(scoring, est, Xa[te], ya[te])
        k += 1
    return s / float(k)

def permutation_test_score(estimator, X, y, *, groups = None, cv = None,
                           n_permutations: int = 100, n_jobs = None, random_state: int = 0,
                           verbose: int = 0, scoring = None, fit_params = None, params = None):
    """Returns (score, permutation_scores, pvalue)."""
    Xa = np.asarray(X)
    ya = np.asarray(y)
    cvo = check_cv(cv, ya, classifier=_is_clf(estimator))
    rng = np.random.RandomState(random_state)
    score = _perm_score(estimator, Xa, ya, groups, cvo, scoring)
    perms = np.empty(n_permutations)
    n = ya.shape[0]
    for p in range(n_permutations):
        if groups is None:
            idx = rng.permutation(n)
        else:
            g = np.asarray(groups)
            idx = np.arange(n)
            for grp in np.unique(g):
                sel = np.nonzero(g == grp)[0]
                idx[sel] = rng.permutation(idx[sel])
        perms[p] = _perm_score(estimator, Xa, ya[idx], groups, cvo, scoring)
    pv = (float(np.sum(perms >= score)) + 1.0) / float(n_permutations + 1)
    return score, perms, pv
