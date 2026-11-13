# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# StackingClassifier / StackingRegressor (sklearn/ensemble/_stacking.py).
# `estimators` is a static tuple of (name, estimator) pairs (as in Voting*);
# an estimator given as the string "drop" is skipped. Out-of-fold meta
# features come from cross_val_predict-style loops on cloned estimators.
# cv: None/int (StratifiedKFold for classifiers, KFold for regressors), a
# splitter object (KFold / StratifiedKFold), or "prefit".

import numpy as np
from ..base import clone
from ..model_selection import KFold, StratifiedKFold
from ..tree._classes import _encode_classes, _r2
from ..linear_model import LogisticRegression, RidgeCV

def _is_drop(est) -> bool:
    if isinstance(est, str):
        if est != "drop":
            raise ValueError("an estimator given as a string must be 'drop'")
        return True
    else:
        return False

def _method_of(est, method: str, is_clf: bool) -> str:
    if isinstance(est, str):
        return ""
    else:
        if method == "auto":
            if is_clf and hasattr(est, "predict_proba"):
                return "predict_proba"
            elif is_clf and hasattr(est, "decision_function"):
                return "decision_function"
            return "predict"
        if method == "predict_proba" and not hasattr(est, "predict_proba"):
            raise ValueError("Underlying estimator does not implement the method predict_proba.")
        if method == "decision_function" and not hasattr(est, "decision_function"):
            raise ValueError("Underlying estimator does not implement the method decision_function.")
        return method

def _call(est, method: str, X) -> np.ndarray[float, 2]:
    if isinstance(est, str):
        return np.empty((0, 0))
    else:
        if method == "predict_proba":
            if hasattr(est, "predict_proba"):
                return np.asarray(est.predict_proba(X), dtype=float)
        elif method == "decision_function":
            if hasattr(est, "decision_function"):
                d = np.asarray(est.decision_function(X), dtype=float)
                if d.ndim == 1:
                    return d.reshape((d.shape[0], 1))
                else:
                    return d.reshape((d.shape[0], d.shape[1]))
        p = np.asarray(est.predict(X), dtype=float)
        return p.reshape((p.shape[0], 1))

def _methods_all(ests, method: str, is_clf: bool, out: List[str]):
    if static.len(ests) > 0:
        out.append(_method_of(ests[0][1], method, is_clf))
        _methods_all(ests[1:], method, is_clf, out)

def _fit_all(ests, X, y, sample_weight):
    if static.len(ests) > 0:
        if not _is_drop(ests[0][1]):
            if sample_weight is None:
                ests[0][1].fit(X, y)
            else:
                ests[0][1].fit(X, y, sample_weight=sample_weight)
        _fit_all(ests[1:], X, y, sample_weight)

def _clone_named(ests):
    if static.len(ests) == 0:
        return ()
    else:
        e = ests[0][1]
        if isinstance(e, str):
            return ((ests[0][0], e),) + _clone_named(ests[1:])
        else:
            return ((ests[0][0], clone(e)),) + _clone_named(ests[1:])

def _predict_all(ests, methods: List[str], X, j: int, out: List[np.ndarray[float, 2]]):
    if static.len(ests) > 0:
        if not _is_drop(ests[0][1]):
            out.append(_call(ests[0][1], methods[j], X))
        _predict_all(ests[1:], methods, X, j + 1, out)

def _oof_one(est, method: str, X, y, folds: List[Tuple[np.ndarray[int, 1], np.ndarray[int, 1]]],
             sample_weight) -> np.ndarray[float, 2]:
    """cross_val_predict(clone(est), X, y, cv=folds, method=method)."""
    if isinstance(est, str):
        return np.empty((0, 0))
    else:
        n = X.shape[0]
        out = np.empty((0, 0))
        first = True
        for tr, te in folds:
            e = clone(est)
            if sample_weight is None:
                e.fit(X[tr], y[tr])
            else:
                e.fit(X[tr], y[tr], sample_weight=np.asarray(sample_weight)[tr])
            pr = _call(e, method, X[te])
            if first:
                out = np.zeros((n, pr.shape[1]))
                first = False
            for t in range(te.shape[0]):
                for c in range(pr.shape[1]):
                    out[te[t], c] = pr[t, c]
        return out

def _oof_all(ests, methods: List[str], X, y, folds, sample_weight, j: int,
             out: List[np.ndarray[float, 2]]):
    if static.len(ests) > 0:
        if not _is_drop(ests[0][1]):
            out.append(_oof_one(ests[0][1], methods[j], X, y, folds, sample_weight))
        _oof_all(ests[1:], methods, X, y, folds, sample_weight, j + 1, out)

def _folds(cv, X, y, is_clf: bool) -> List[Tuple[np.ndarray[int, 1], np.ndarray[int, 1]]]:
    out = List[Tuple[np.ndarray[int, 1], np.ndarray[int, 1]]]()
    if cv is None or isinstance(cv, int):
        k = 5 if cv is None else cv
        if is_clf:
            for tr, te in StratifiedKFold(k).split(X, y):
                out.append((tr, te))
        else:
            for tr, te in KFold(k).split(X, y):
                out.append((tr, te))
    else:
        for tr, te in cv.split(X, y):
            out.append((tr, te))
    return out

def _concat(preds: List[np.ndarray[float, 2]], methods: List[str], n_classes: int, X,
            passthrough: bool) -> np.ndarray[float, 2]:
    cols = List[np.ndarray[float, 2]]()
    live = [m for m in methods if m != ""]
    for j in range(len(preds)):
        pr = preds[j]
        if live[j] == "predict_proba" and n_classes == 2:
            cols.append(pr[:, 1:].copy())
        else:
            cols.append(pr)
    if passthrough:
        cols.append(np.asarray(X, dtype=float))
    n = cols[0].shape[0]
    tot = 0
    for c in cols:
        tot += c.shape[1]
    out = np.empty((n, tot))
    off = 0
    for c in cols:
        for i in range(n):
            for k in range(c.shape[1]):
                out[i, off + k] = c[i, k]
        off += c.shape[1]
    return out

class _BaseStacking[S, F, CV]:
    estimators: S
    final_estimator: F
    cv: CV
    stack_method: str
    passthrough: bool
    verbose: int
    estimators_: S
    final_estimator_: F
    stack_method_: List[str]
    n_features_in_: int
    _is_clf: bool
    _n_classes: int

    def _fit_impl(self, X, y, sample_weight):
        Xa = np.asarray(X, dtype=float)
        self.n_features_in_ = Xa.shape[1]
        is_prefit = False
        if isinstance(self.cv, str):
            if self.cv != "prefit":
                raise ValueError("cv must be None, an int, a CV splitter or 'prefit'")
            is_prefit = True
        if is_prefit:
            self.estimators_ = self.estimators
        else:
            self.estimators_ = _clone_named(self.estimators)
            _fit_all(self.estimators_, Xa, y, sample_weight)
        self.stack_method_ = List[str]()
        _methods_all(self.estimators, self.stack_method, self._is_clf, self.stack_method_)
        preds = List[np.ndarray[float, 2]]()
        if is_prefit:
            _predict_all(self.estimators_, self.stack_method_, Xa, 0, preds)
        else:
            folds = _folds(self.cv, Xa, y, self._is_clf)
            _oof_all(self.estimators, self.stack_method_, Xa, y, folds, sample_weight, 0, preds)
        X_meta = _concat(preds, self.stack_method_, self._n_classes, Xa, self.passthrough)
        self.final_estimator_ = clone(self.final_estimator)
        if sample_weight is None:
            self.final_estimator_.fit(X_meta, y)
        else:
            self.final_estimator_.fit(X_meta, y, sample_weight=sample_weight)

    def transform(self, X):
        Xa = np.asarray(X, dtype=float)
        preds = List[np.ndarray[float, 2]]()
        _predict_all(self.estimators_, self.stack_method_, Xa, 0, preds)
        return _concat(preds, self.stack_method_, self._n_classes, Xa, self.passthrough)

    def fit_transform(self, X, y, sample_weight = None):
        self.fit(X, y, sample_weight=sample_weight)
        return self.transform(X)

    @property
    def named_estimators_(self):
        return self.estimators_

class StackingClassifier[S, F, CV](_BaseStacking[S, F, CV]):
    classes_: np.ndarray[int, 1]

    def __init__(self, estimators: S, final_estimator: F = LogisticRegression(), *,
                 cv: CV = None, stack_method: str = "auto", n_jobs: Optional[int] = None,
                 passthrough: bool = False, verbose: int = 0):
        if stack_method != "auto" and stack_method != "predict_proba" and stack_method != "decision_function" and stack_method != "predict":
            raise ValueError(f"stack_method={stack_method} is not supported")
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.stack_method = stack_method
        self.passthrough = passthrough
        self.verbose = verbose
        self.estimators_ = _clone_named(estimators)
        self.final_estimator_ = clone(final_estimator)
        self.stack_method_ = List[str]()
        self.n_features_in_ = 0
        self._is_clf = True
        self._n_classes = 0
        self.classes_ = np.empty(0, dtype=int)

    def _sk_clone(self):
        return StackingClassifier(_clone_named(self.estimators), clone(self.final_estimator),
                                  cv=self.cv, stack_method=self.stack_method,
                                  passthrough=self.passthrough, verbose=self.verbose)

    @property
    def _estimator_type(self) -> str:
        return "classifier"

    def fit(self, X, y, sample_weight = None):
        classes, enc = _encode_classes(np.asarray(y))
        self.classes_ = classes
        self._n_classes = classes.shape[0]
        ye = np.array([int(v) for v in enc])
        self._fit_impl(X, ye, sample_weight)
        return self

    def predict(self, X):
        p = np.asarray(self.final_estimator_.predict(self.transform(X)))
        n = p.shape[0]
        out = np.empty(n, dtype=int)
        for i in range(n):
            out[i] = self.classes_[int(p[i])]
        return out

    def predict_proba(self, X):
        return np.asarray(self.final_estimator_.predict_proba(self.transform(X)), dtype=float)

    def decision_function(self, X):
        return self.final_estimator_.decision_function(self.transform(X))

    def score(self, X, y, sample_weight = None) -> float:
        from ..base import _accuracy
        return _accuracy(y, self.predict(X))

class StackingRegressor[S, F, CV](_BaseStacking[S, F, CV]):
    def __init__(self, estimators: S, final_estimator: F = RidgeCV(), *, cv: CV = None,
                 n_jobs: Optional[int] = None, passthrough: bool = False, verbose: int = 0):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.stack_method = "predict"
        self.passthrough = passthrough
        self.verbose = verbose
        self.estimators_ = _clone_named(estimators)
        self.final_estimator_ = clone(final_estimator)
        self.stack_method_ = List[str]()
        self.n_features_in_ = 0
        self._is_clf = False
        self._n_classes = 0

    def _sk_clone(self):
        return StackingRegressor(_clone_named(self.estimators), clone(self.final_estimator),
                                 cv=self.cv, passthrough=self.passthrough, verbose=self.verbose)

    @property
    def _estimator_type(self) -> str:
        return "regressor"

    def fit(self, X, y, sample_weight = None):
        self._fit_impl(X, np.asarray(y, dtype=float), sample_weight)
        return self

    def predict(self, X):
        return np.asarray(self.final_estimator_.predict(self.transform(X)), dtype=float)

    def score(self, X, y, sample_weight = None) -> float:
        return _r2(y, self.predict(X))
