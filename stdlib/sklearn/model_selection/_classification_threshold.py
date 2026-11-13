# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# FixedThresholdClassifier and TunedThresholdClassifierCV for binary
# classifiers with integer class labels. Scoring: a metric name
# ("balanced_accuracy", "accuracy", "f1", "precision", "recall", "jaccard")
# or a callable metric(y_true, y_pred) where greater is better.

import numpy as np
from ..base import clone

def _metric_by_name(name: str, yt: np.ndarray[int, 1], yp: np.ndarray[int, 1]) -> float:
    from ..metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score,
                           recall_score, jaccard_score)
    if name == "balanced_accuracy":
        return float(balanced_accuracy_score(yt, yp))
    elif name == "accuracy":
        return float(accuracy_score(yt, yp))
    elif name == "f1":
        return float(f1_score(yt, yp))
    elif name == "precision":
        return float(precision_score(yt, yp))
    elif name == "recall":
        return float(recall_score(yt, yp))
    elif name == "jaccard":
        return float(jaccard_score(yt, yp))
    raise ValueError(f"{name!r} is not a valid scoring value for TunedThresholdClassifierCV")

def _response_values(est, X, method: str, classes: np.ndarray[int, 1], pos_label: int):
    """(scores for the positive class, method used)."""
    pos_idx = 1
    if classes[0] == pos_label:
        pos_idx = 0
    use = method
    if method == "auto":
        use = "predict_proba" if hasattr(est, "predict_proba") else "decision_function"
    if use == "predict_proba":
        pp = np.asarray(est.predict_proba(X), dtype=float)
        return np.ascontiguousarray(pp[:, pos_idx]), use
    elif use == "decision_function":
        d = np.asarray(est.decision_function(X), dtype=float).ravel()
        if pos_idx == 0:
            d = -d
        return d, use
    raise ValueError(f"response_method must be 'auto', 'predict_proba' or 'decision_function', got {method!r}")

def _scores_to_labels(y_score: np.ndarray[float, 1], threshold: float,
                      classes: np.ndarray[int, 1], pos_label: int) -> np.ndarray[int, 1]:
    pos = classes[1]
    neg = classes[0]
    if classes[0] == pos_label:
        pos = classes[0]
        neg = classes[1]
    out = np.empty(y_score.shape[0], dtype=int)
    for i in range(y_score.shape[0]):
        out[i] = pos if y_score[i] >= threshold else neg
    return out

def _check_binary(y) -> np.ndarray[int, 1]:
    ya = np.asarray(y, dtype=int)
    u = np.unique(ya)
    if u.shape[0] != 2:
        raise ValueError(f"Only binary classification is supported. Unknown label type: {'multiclass' if u.shape[0] > 2 else 'unknown'}")
    return ya

class FixedThresholdClassifier[E]:
    _estimator_type: ClassVar[str] = "classifier"
    estimator: E
    threshold: float
    _auto: bool
    pos_label: Optional[int]
    response_method: str
    estimator_: E
    classes_: np.ndarray[int, 1]
    n_features_in_: int

    def __init__(self, estimator: E, *, threshold = "auto", pos_label: Optional[int] = None,
                 response_method: str = "auto"):
        self.estimator = estimator
        self._auto = isinstance(threshold, str)
        self.threshold = 0.0 if isinstance(threshold, str) else float(threshold)
        self.pos_label = pos_label
        self.response_method = response_method
        self.estimator_ = clone(estimator)
        self.classes_ = np.empty(0, dtype=int)
        self.n_features_in_ = 0

    def fit(self, X, y):
        ya = _check_binary(y)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, ya)
        self.classes_ = np.asarray(self.estimator_.classes_, dtype=int)
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def _pl(self) -> int:
        return self.pos_label.__val__() if self.pos_label is not None else int(self.classes_[1])

    def predict(self, X):
        s, used = _response_values(self.estimator_, X, self.response_method, self.classes_, self._pl())
        th = self.threshold
        if self._auto:
            th = 0.5 if used == "predict_proba" else 0.0
        return _scores_to_labels(s, th, self.classes_, self._pl())

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict_log_proba(self, X):
        return self.estimator_.predict_log_proba(X)

    def decision_function(self, X):
        return self.estimator_.decision_function(X)

    def score(self, X, y, sample_weight = None) -> float:
        from ..metrics import accuracy_score
        return float(accuracy_score(y, self.predict(X)))

class TunedThresholdClassifierCV[E, SC = str]:
    _estimator_type: ClassVar[str] = "classifier"
    estimator: E
    scoring: SC
    response_method: str
    thresholds_n: int
    thresholds_arr: np.ndarray[float, 1]
    _use_arr: bool
    cv_n: int
    cv_frac: float
    prefit: bool
    refit: bool
    random_state: Optional[int]
    store_cv_results: bool
    estimator_: E
    classes_: np.ndarray[int, 1]
    best_threshold_: float
    best_score_: float
    n_features_in_: int
    cv_results_thresholds_: np.ndarray[float, 1]
    cv_results_scores_: np.ndarray[float, 1]

    def __init__(self, estimator: E, *, scoring: SC = "balanced_accuracy",
                 response_method: str = "auto", thresholds = 100, cv = None, refit = True,
                 n_jobs = None, random_state: Optional[int] = None,
                 store_cv_results: bool = False):
        self.estimator = estimator
        self.scoring = scoring
        self.response_method = response_method
        self._use_arr = not isinstance(thresholds, int)
        self.thresholds_n = 0
        self.thresholds_arr = np.empty(0)
        if isinstance(thresholds, int):
            self.thresholds_n = thresholds
        else:
            self.thresholds_arr = np.asarray(thresholds, dtype=float)
        self.cv_n = 5
        self.cv_frac = 0.0
        self.prefit = False
        if isinstance(cv, NoneType):
            pass
        elif isinstance(cv, str):
            self.prefit = True
        elif isinstance(cv, float):
            self.cv_frac = cv
        else:
            self.cv_n = int(cv)
        self.refit = bool(refit)
        self.random_state = random_state
        self.store_cv_results = store_cv_results
        self.estimator_ = clone(estimator)
        self.classes_ = np.empty(0, dtype=int)
        self.best_threshold_ = 0.0
        self.best_score_ = 0.0
        self.n_features_in_ = 0
        self.cv_results_thresholds_ = np.empty(0)
        self.cv_results_scores_ = np.empty(0)

    def _score(self, yt: np.ndarray[int, 1], yp: np.ndarray[int, 1]) -> float:
        if isinstance(self.scoring, str):
            return _metric_by_name(self.scoring, yt, yp)
        else:
            return float(self.scoring(yt, yp))

    def _curve(self, est, X, y: np.ndarray[int, 1]):
        classes = np.asarray(est.classes_, dtype=int)
        pl = int(classes[1])
        s, used = _response_values(est, X, self.response_method, classes, pl)
        if self._use_arr:
            th = self.thresholds_arr.copy()
        else:
            th = np.linspace(np.min(s), np.max(s), self.thresholds_n)
        sc = np.empty(th.shape[0])
        for t in range(th.shape[0]):
            sc[t] = self._score(y, _scores_to_labels(s, th[t], classes, pl))
        return sc, th

    def fit(self, X, y):
        from ._split import StratifiedKFold
        from ._shuffle import StratifiedShuffleSplit
        Xa = np.asarray(X, dtype=float)
        ya = _check_binary(y)
        n = ya.shape[0]
        folds = List[Tuple[np.ndarray[int, 1], np.ndarray[int, 1]]]()
        if self.prefit:
            if self.refit:
                raise ValueError("When cv='prefit', refit cannot be True.")
            self.estimator_ = self.estimator
            folds.append((np.empty(0, dtype=int), np.arange(n)))
        else:
            if self.cv_frac > 0.0:
                for tr, te in StratifiedShuffleSplit(1, test_size=self.cv_frac,
                                                     random_state=self.random_state).split(Xa, ya):
                    folds.append((tr, te))
            else:
                for tr, te in StratifiedKFold(self.cv_n).split(Xa, ya):
                    folds.append((tr, te))
            if not self.refit and len(folds) > 1:
                raise ValueError("When cv has several folds, refit cannot be False.")
            self.estimator_ = clone(self.estimator)
            if self.refit:
                self.estimator_.fit(Xa, ya)
            else:
                self.estimator_.fit(Xa[folds[0][0]], ya[folds[0][0]])
        all_sc = List[np.ndarray[float, 1]]()
        all_th = List[np.ndarray[float, 1]]()
        for tr, te in folds:
            if self.prefit:
                sc, th = self._curve(self.estimator_, Xa, ya)
            else:
                e = clone(self.estimator)
                e.fit(Xa[tr], ya[tr])
                sc, th = self._curve(e, Xa[te], ya[te])
            all_sc.append(sc)
            all_th.append(th)
        mn = float("inf")
        mx = -float("inf")
        for th in all_th:
            if np.isclose(th[0], th[th.shape[0] - 1]):
                raise ValueError("The provided estimator makes constant predictions. Therefore, it is impossible to optimize the decision threshold.")
            mn = min(mn, float(np.min(th)))
            mx = max(mx, float(np.max(th)))
        if self._use_arr:
            dec = self.thresholds_arr.copy()
        else:
            dec = np.linspace(mn, mx, self.thresholds_n)
        obj = np.zeros(dec.shape[0])
        for f in range(len(all_th)):
            ip = np.interp(dec, all_th[f], all_sc[f])
            for t in range(dec.shape[0]):
                obj[t] += ip[t]
        for t in range(dec.shape[0]):
            obj[t] /= float(len(all_th))
        best = int(np.argmax(obj))
        self.best_score_ = float(obj[best])
        self.best_threshold_ = float(dec[best])
        self.classes_ = np.asarray(self.estimator_.classes_, dtype=int)
        self.n_features_in_ = Xa.shape[1]
        if self.store_cv_results:
            self.cv_results_thresholds_ = dec
            self.cv_results_scores_ = obj
        return self

    def predict(self, X):
        pl = int(self.classes_[1])
        s, used = _response_values(self.estimator_, X, self.response_method, self.classes_, pl)
        return _scores_to_labels(s, self.best_threshold_, self.classes_, pl)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict_log_proba(self, X):
        return self.estimator_.predict_log_proba(X)

    def decision_function(self, X):
        return self.estimator_.decision_function(X)

    def score(self, X, y, sample_weight = None) -> float:
        from ..metrics import accuracy_score
        return float(accuracy_score(y, self.predict(X)))
