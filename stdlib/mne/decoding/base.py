# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne/decoding/base.py (LinearModel, get_coef, cross_val_multiscore),
# mne/decoding/search_light.py (SlidingEstimator, GeneralizingEstimator),
# mne/decoding/ems.py (EMS, compute_ems).

import numpy as np
from sklearn.base import clone


def _is_classifier(est) -> bool:
    if hasattr(est, "_estimator_type"):
        return est._estimator_type == "classifier"
    return False


# ---------------------------------------------------------------------------
# LinearModel

class LinearModel[M]:
    """Compute and store patterns from linear models (mne.decoding.LinearModel).

    model: a linear sklearn estimator with ``coef_`` (default in mne:
    LogisticRegression(solver='liblinear', random_state=0); pass it explicitly)."""
    model: M
    model_: M
    patterns_: np.ndarray[float, 2]
    _fitted: bool
    _1d: bool

    @property
    def _estimator_type(self) -> str:
        if hasattr(self.model, "_estimator_type"):
            return str(self.model._estimator_type)
        return ""

    def __init__(self, model: M):
        self.model = model
        self.model_ = model
        self.patterns_ = np.zeros((0, 0))
        self._fitted = False
        self._1d = False

    def fit(self, X, y, **fit_params):
        Xa = np.asarray(X, dtype=float)
        if Xa.ndim != 2:
            raise ValueError("X must be 2-D (n_samples, n_features)")
        self.model_ = clone(self.model)
        self.model_.fit(Xa, y)
        Xc = Xa - Xa.mean(0)[None, :]
        n = Xc.shape[0]
        covX = (Xc.T @ Xc) / (n - 1)
        F = self._filters2()
        ya = np.asarray(y)
        if static.len(ya.shape) == 2:
            if ya.shape[1] != 1:
                yc = ya.astype(float) - ya.astype(float).mean(0)[None, :]
                covY = (yc.T @ yc) / (yc.shape[0] - 1)
                inv_Y = np.linalg.pinv(covY)
                self.patterns_ = (covX @ (F.T @ inv_Y)).T.copy()
            else:
                self.patterns_ = (covX @ F.T).T.copy()
        else:
            self.patterns_ = (covX @ F.T).T.copy()
        self._fitted = True
        return self

    def _filters2(self) -> np.ndarray[float, 2]:
        c = np.asarray(self.model_.coef_, dtype=float)
        if static.len(c.shape) == 1:
            self._1d = True
            return c.reshape((1, c.shape[0]))
        else:
            self._1d = c.shape[0] == 1
            return c

    @property
    def filters_(self) -> np.ndarray[float, 2]:
        """(n_targets, n_features); mne squeezes a single target to 1-D (use [0])."""
        return self._filters2()

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=float))

    def decision_function(self, X):
        return self.model_.decision_function(np.asarray(X, dtype=float))

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=float))

    def score(self, X, y) -> float:
        return float(self.model_.score(np.asarray(X, dtype=float), y))

    def transform(self, X):
        return self.model_.transform(np.asarray(X, dtype=float))

    @property
    def classes_(self):
        return self.model_.classes_


def _inv_steps(st, coef, k: Literal[int]):
    if k < 0:
        return coef
    else:
        c = np.asarray(st[k][1].inverse_transform(coef), dtype=float)
        return _inv_steps(st, c, k - 1)


def get_coef(estimator, attr: Literal[str] = "filters_", inverse_transform: bool = False):
    """Retrieve the coefficients of an estimator ending with a LinearModel (mne).

    Supports LinearModel and Pipeline(..., LinearModel) (with
    inverse_transform through invertible preceding steps). Returns
    (n_targets, n_features) with the single-target dimension squeezed."""
    if hasattr(estimator, "steps"):
        st = estimator.steps
        coef = np.asarray(getattr(st[static.len(st) - 1][1], attr), dtype=float)
        if inverse_transform:
            return _inv_steps(st, coef, static.len(st) - 2)
        return coef
    else:
        if inverse_transform:
            raise ValueError("inverse_transform can only be applied onto pipeline estimators.")
        return np.asarray(getattr(estimator, attr), dtype=float)


# ---------------------------------------------------------------------------
# cross_val_multiscore

def cross_val_multiscore(estimator, X, y = None, groups = None, scoring = None, cv = None,
                         n_jobs = None, verbose = None, fit_params = None,
                         pre_dispatch: str = "2*n_jobs"):
    """Evaluate a score by cross-validation (mne): scores may be multi-dimensional.

    Returns (n_splits,) for scalar scores or (n_splits, ...) arrays for
    Sliding/Generalizing estimators."""
    from sklearn.model_selection._validation import check_cv, _folds
    from sklearn.model_selection._scorer_shim import _score
    Xa = np.asarray(X)
    ya = np.asarray(y)
    cvo = check_cv(cv, ya, classifier=_is_classifier(estimator))
    folds = _folds(cvo, Xa, ya, groups)
    first = _fold_score(estimator, Xa, ya, folds[0], scoring)
    if isinstance(first, float):
        out = np.empty(len(folds))
        out[0] = first
        for k in range(1, len(folds)):
            out[k] = _fold_score(estimator, Xa, ya, folds[k], scoring)
        return out
    else:
        out2 = np.empty((len(folds),) + first.shape)
        out2[0] = first
        for k in range(1, len(folds)):
            out2[k] = _fold_score(estimator, Xa, ya, folds[k], scoring)
        return out2


def _fold_score(estimator, Xa, ya, fold, scoring):
    from sklearn.model_selection._scorer_shim import _score
    tr, te = fold
    est = clone(estimator)
    est.fit(Xa[tr], ya[tr])
    if hasattr(est, "_mne_multiscore"):
        return est.score(Xa[te], ya[te])
    else:
        return _score(scoring, est, Xa[te], ya[te])


# ---------------------------------------------------------------------------
# SlidingEstimator / GeneralizingEstimator

def _score_one(scoring: Optional[str], est, X, y) -> float:
    from sklearn.model_selection._scorer_shim import _score
    if scoring is None:
        if _is_classifier(est):
            return _score("accuracy", est, X, y)
        return float(est.score(X, y))
    else:
        return _score(scoring.__val__(), est, X, y)


def _opt_str(x) -> Optional[str]:
    if isinstance(x, NoneType):
        return None
    elif isinstance(x, Optional):
        return x
    else:
        return str(x)


def _move_last(X, axis: int):
    a = np.asarray(X, dtype=float)
    nd = static.len(a.shape)
    ax = axis if axis >= 0 else axis + nd
    if ax == 0:
        raise ValueError("axis must not be the sample axis (0).")
    if ax != nd - 1:
        return np.ascontiguousarray(np.moveaxis(a, ax, nd - 1))
    return np.ascontiguousarray(a)


def _slice_last(X, i: int):
    if static.len(X.shape) == 3:
        return np.ascontiguousarray(X[:, :, i])
    elif static.len(X.shape) == 4:
        return np.ascontiguousarray(X[:, :, :, i])
    else:
        compile_error("X must be 3-D or 4-D")


class SlidingEstimator[E]:
    """Search Light: fit, and predict/score a series of models on each
    time sample (last dimension of X) (mne.decoding.SlidingEstimator).

    scoring: None (accuracy for classifiers / est.score), or a scorer name."""
    base_estimator: E
    scoring: Optional[str]
    axis: int
    estimators_: List[E]
    _mne_multiscore: bool

    def __init__(self, base_estimator: E, scoring = None, n_jobs = None, *, position: int = 0,
                 allow_2d: bool = False, axis: int = -1, verbose = None):
        self.base_estimator = base_estimator
        self.scoring = _opt_str(scoring)
        self.axis = axis
        self.estimators_ = List[E]()
        self._mne_multiscore = True

    def __len__(self) -> int:
        return len(self.estimators_)

    def fit(self, X, y, **fit_params):
        Xa = _move_last(X, self.axis)
        self.estimators_ = List[E]()
        for ii in range(Xa.shape[Xa.ndim - 1]):
            est = clone(self.base_estimator)
            est.fit(_slice_last(Xa, ii), y)
            self.estimators_.append(est)
        return self

    def _check_n(self, Xa):
        if Xa.shape[Xa.ndim - 1] != len(self.estimators_):
            raise ValueError("The number of estimators does not match X.shape[-1]")

    def predict(self, X):
        Xa = _move_last(X, self.axis)
        self._check_n(Xa)
        p0 = np.asarray(self.estimators_[0].predict(_slice_last(Xa, 0)))
        out = np.zeros((Xa.shape[0], len(self.estimators_)), dtype=p0.dtype)
        out[:, 0] = p0
        for ii in range(1, len(self.estimators_)):
            out[:, ii] = np.asarray(self.estimators_[ii].predict(_slice_last(Xa, ii)))
        return out

    def decision_function(self, X):
        Xa = _move_last(X, self.axis)
        self._check_n(Xa)
        p0 = np.asarray(self.estimators_[0].decision_function(_slice_last(Xa, 0)), dtype=float)
        out = np.zeros((Xa.shape[0], len(self.estimators_)) + p0.shape[1:])
        out[:, 0] = p0
        for ii in range(1, len(self.estimators_)):
            out[:, ii] = np.asarray(self.estimators_[ii].decision_function(_slice_last(Xa, ii)), dtype=float)
        return out

    def predict_proba(self, X):
        Xa = _move_last(X, self.axis)
        self._check_n(Xa)
        p0 = np.asarray(self.estimators_[0].predict_proba(_slice_last(Xa, 0)), dtype=float)
        out = np.zeros((Xa.shape[0], len(self.estimators_), p0.shape[1]))
        out[:, 0] = p0
        for ii in range(1, len(self.estimators_)):
            out[:, ii] = np.asarray(self.estimators_[ii].predict_proba(_slice_last(Xa, ii)), dtype=float)
        return out

    def transform(self, X):
        if hasattr(self.base_estimator, "transform"):
            Xa = _move_last(X, self.axis)
            self._check_n(Xa)
            p0 = np.asarray(self.estimators_[0].transform(_slice_last(Xa, 0)), dtype=float)
            out = np.zeros((Xa.shape[0], len(self.estimators_)) + p0.shape[1:])
            out[:, 0] = p0
            for ii in range(1, len(self.estimators_)):
                out[:, ii] = np.asarray(self.estimators_[ii].transform(_slice_last(Xa, ii)), dtype=float)
            return out
        else:
            return np.asarray(self.predict(X), dtype=float)

    def fit_transform(self, X, y, **fit_params):
        return self.fit(X, y).transform(X)

    def score(self, X, y):
        Xa = _move_last(X, self.axis)
        self._check_n(Xa)
        out = np.zeros(len(self.estimators_))
        for ii in range(len(self.estimators_)):
            out[ii] = _score_one(self.scoring, self.estimators_[ii], _slice_last(Xa, ii), y)
        return out

    def _sk_clone(self):
        return SlidingEstimator[E](clone(self.base_estimator), self.scoring, axis=self.axis)

    @property
    def classes_(self):
        return self.estimators_[0].classes_


def _stack_times(Xa):
    """(n_samples, ..., n_times) -> (n_samples * n_times, ...) (sample-major)."""
    n = Xa.shape[0]
    nt = Xa.shape[Xa.ndim - 1]
    if static.len(Xa.shape) == 3:
        return np.ascontiguousarray(Xa.transpose(0, 2, 1)).reshape((n * nt, Xa.shape[1]))
    else:
        return np.ascontiguousarray(Xa.transpose(0, 3, 1, 2)).reshape((n * nt, Xa.shape[1], Xa.shape[2]))


class GeneralizingEstimator[E]:
    """Generalization Light: fit a model on each time point and predict on
    all time points (mne.decoding.GeneralizingEstimator)."""
    base_estimator: E
    scoring: Optional[str]
    axis: int
    estimators_: List[E]
    _mne_multiscore: bool

    def __init__(self, base_estimator: E, scoring = None, n_jobs = None, *, position: int = 0,
                 allow_2d: bool = False, axis: int = -1, verbose = None):
        self.base_estimator = base_estimator
        self.scoring = _opt_str(scoring)
        self.axis = axis
        self.estimators_ = List[E]()
        self._mne_multiscore = True

    def __len__(self) -> int:
        return len(self.estimators_)

    def fit(self, X, y, **fit_params):
        Xa = _move_last(X, self.axis)
        self.estimators_ = List[E]()
        for ii in range(Xa.shape[Xa.ndim - 1]):
            est = clone(self.base_estimator)
            est.fit(_slice_last(Xa, ii), y)
            self.estimators_.append(est)
        return self

    def predict(self, X):
        Xa = _move_last(X, self.axis)
        n = Xa.shape[0]
        nt = Xa.shape[Xa.ndim - 1]
        Xs = _stack_times(Xa)
        p0 = np.asarray(self.estimators_[0].predict(Xs))
        out = np.zeros((n, len(self.estimators_), nt), dtype=p0.dtype)
        out[:, 0, :] = p0.reshape((n, nt))
        for ii in range(1, len(self.estimators_)):
            out[:, ii, :] = np.asarray(self.estimators_[ii].predict(Xs)).reshape((n, nt))
        return out

    def decision_function(self, X):
        Xa = _move_last(X, self.axis)
        n = Xa.shape[0]
        nt = Xa.shape[Xa.ndim - 1]
        Xs = _stack_times(Xa)
        out = np.zeros((n, len(self.estimators_), nt))
        for ii in range(len(self.estimators_)):
            out[:, ii, :] = np.asarray(self.estimators_[ii].decision_function(Xs), dtype=float).reshape((n, nt))
        return out

    def predict_proba(self, X):
        Xa = _move_last(X, self.axis)
        n = Xa.shape[0]
        nt = Xa.shape[Xa.ndim - 1]
        Xs = _stack_times(Xa)
        p0 = np.asarray(self.estimators_[0].predict_proba(Xs), dtype=float)
        nc = p0.shape[1]
        out = np.zeros((n, len(self.estimators_), nt, nc))
        out[:, 0] = p0.reshape((n, nt, nc))
        for ii in range(1, len(self.estimators_)):
            out[:, ii] = np.asarray(self.estimators_[ii].predict_proba(Xs), dtype=float).reshape((n, nt, nc))
        return out

    def score(self, X, y):
        Xa = _move_last(X, self.axis)
        nt = Xa.shape[Xa.ndim - 1]
        out = np.zeros((len(self.estimators_), nt))
        for ii in range(len(self.estimators_)):
            for jj in range(nt):
                out[ii, jj] = _score_one(self.scoring, self.estimators_[ii], _slice_last(Xa, jj), y)
        return out

    def _sk_clone(self):
        return GeneralizingEstimator[E](clone(self.base_estimator), self.scoring, axis=self.axis)

    @property
    def classes_(self):
        return self.estimators_[0].classes_


# ---------------------------------------------------------------------------
# EMS

class EMS:
    """Transformer to compute event-matched spatial filters (mne.decoding.EMS)."""
    classes_: np.ndarray[int, 1]
    filters_: np.ndarray[float, 2]
    _estimator_type: ClassVar[str] = "transformer"

    def __init__(self):
        self.classes_ = np.zeros(0, dtype=int)
        self.filters_ = np.zeros((0, 0))

    def __repr__(self) -> str:
        if self.filters_.shape[0] > 0:
            return f"<EMS: fitted with {self.filters_.shape[0]} filters on {self.classes_.shape[0]} classes.>"
        return "<EMS: not fitted.>"

    def fit(self, X, y):
        Xa = np.asarray(X, dtype=float)
        classes, yi = np.unique(np.asarray(y), return_inverse=True)
        if classes.shape[0] > 2:
            raise ValueError("EMS only works for binary classification.")
        self.classes_ = classes
        i0 = np.flatnonzero(yi == 0)
        i1 = np.flatnonzero(yi == 1)
        f = Xa[i0].mean(0) - Xa[i1].mean(0)
        nrm = np.sqrt(np.sum(f * f, axis=0))
        self.filters_ = f / nrm[None, :]
        return self

    def transform(self, X):
        Xa = np.asarray(X, dtype=float)
        return np.sum(Xa * self.filters_[None, :, :], axis=1)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


def compute_ems(epochs, conditions = None, picks = None, n_jobs = None, cv = None, *, verbose = None):
    """Compute event-matched spatial filter on epochs (mne, LeaveOneOut cv by default).

    Returns (surrogate_trials, spatial_filter, conditions)."""
    from .._info import _picks_to_idx
    pk = _picks_to_idx(epochs.info, picks)
    ev = epochs.events[:, 2].copy()
    keys = List[str]()
    if isinstance(conditions, NoneType):
        for k in epochs.event_id.keys():
            keys.append(k)
    else:
        for k in conditions:
            keys.append(str(k))
    keys.sort()
    if len(keys) != 2:
        raise ValueError(f"Currently this function expects exactly 2 conditions but you gave me {len(keys)}")
    counts = Dict[int, int]()
    for v in ev:
        counts[int(v)] = counts.get(int(v), 0) + 1
    vals = list(counts.values())
    for v in vals:
        if v != vals[0]:
            raise ValueError("The same number of epochs is required by this function. Please consider `epochs.equalize_event_counts`")
    data = epochs.get_data()[:, np.array(pk), :].copy()
    types = [epochs.info.ch_types[p] for p in pk]
    for ct in ("mag", "grad", "eeg"):
        sel = [i for i in range(len(types)) if types[i] == ct]
        if len(sel) > 0:
            sa = np.array(sel)
            sd = float(np.std(data[:, sa]))
            data[:, sa] /= sd
    cond_idx = [np.flatnonzero(ev == epochs.event_id[k]) for k in keys]
    n = ev.shape[0]
    n_ch = data.shape[1]
    n_t = data.shape[2]
    surrogates = np.zeros((n, n_t))
    filt = np.zeros((n_ch, n_t))
    for t in range(n):
        tr = np.array([i for i in range(n) if i != t], dtype=int)
        a = np.intersect1d(cond_idx[0], tr)
        b = np.intersect1d(cond_idx[1], tr)
        d = data[a].mean(0) - data[b].mean(0)
        d /= np.sqrt(np.sum(d ** 2, axis=0))[None, :]
        surrogates[t] = np.sum(data[t] * d, axis=0)
        filt += d
    filt /= n
    return surrogates, filt, ev
