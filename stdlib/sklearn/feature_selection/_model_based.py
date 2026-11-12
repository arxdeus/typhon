# Ported from scikit-learn 1.9.1 (BSD-3-Clause)
# Model-based selectors: SelectFromModel, RFE, RFECV, SequentialFeatureSelector.
# Estimators are duck-typed at compile time: importances come from `coef_`
# or `feature_importances_`; scoring is the estimator's own score().

import numpy as np
from ..base import clone
from ._univariate import _SelectorMixin

def _importances(est, transform_func: str, norm_order: int = 1):
    imp = np.empty(0)
    if hasattr(est, "coef_"):
        c = np.asarray(est.coef_, dtype=float)
        if c.ndim == 1:
            if transform_func == "norm":
                return np.abs(c)
            return c * c
        else:
            if transform_func == "norm":
                if norm_order == 1:
                    return np.sum(np.abs(c), axis=0)
                elif norm_order == 2:
                    return np.sqrt(np.sum(c * c, axis=0))
                else:
                    return np.max(np.abs(c), axis=0)
            return np.sum(c * c, axis=0)
    elif hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=float)
        if transform_func == "norm":
            return np.abs(imp)
        return imp * imp
    else:
        compile_error("estimator must expose coef_ or feature_importances_")
    return imp

def _is_classifier(est) -> bool:
    return hasattr(est, "predict_proba") or hasattr(est, "decision_function")

def _splits(cv, X, y, classifier: bool):
    from ..model_selection import KFold, StratifiedKFold
    out: List[Tuple[np.ndarray[int, 1], np.ndarray[int, 1]]] = []
    if isinstance(cv, int):
        if classifier:
            for tr, te in StratifiedKFold(cv).split(X, y):
                out.append((np.asarray(tr), np.asarray(te)))
        else:
            for tr, te in KFold(cv).split(X, y):
                out.append((np.asarray(tr), np.asarray(te)))
    else:
        for tr, te in cv.split(X, y):
            out.append((np.asarray(tr), np.asarray(te)))
    return out

def _threshold(imp, threshold, l1: bool) -> float:
    if threshold is None:
        if l1:
            return 1e-5
        return float(np.mean(imp))
    elif isinstance(threshold, str):
        t = threshold
        if "*" in t:
            parts = t.split("*")
            scale = float(parts[0].strip())
            ref = parts[1].strip()
            if ref == "median":
                return scale * float(np.median(imp))
            elif ref == "mean":
                return scale * float(np.mean(imp))
            raise ValueError("Unknown reference: " + ref)
        elif t == "median":
            return float(np.median(imp))
        elif t == "mean":
            return float(np.mean(imp))
        raise ValueError(f"Expected threshold='mean' or threshold='median' got {t}")
    else:
        return float(threshold)

def _is_l1(est) -> bool:
    if hasattr(est, "penalty"):
        if isinstance(est.penalty, str):
            return est.penalty == "l1"
    return False

class SelectFromModel[E, TH](_SelectorMixin):
    estimator: E
    estimator_: E
    threshold: TH
    prefit: bool
    norm_order: int
    max_features: Optional[int]
    threshold_: float
    _fitted: bool

    def __init__(self, estimator: E, *, threshold: TH = None, prefit: bool = False,
                 norm_order: int = 1, max_features: Optional[int] = None,
                 importance_getter: str = "auto"):
        self.estimator = estimator
        self.estimator_ = estimator
        self.threshold = threshold
        self.prefit = prefit
        self.norm_order = norm_order
        self.max_features = max_features
        self.threshold_ = 0.0
        self._fitted = prefit

    def fit(self, X, y = None):
        if self.prefit:
            raise ValueError("SelectFromModel: fit() is not available with prefit=True")
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(np.asarray(X, dtype=float), y)
        self._fitted = True
        return self

    def _get_support_mask(self):
        est = self.estimator_
        scores = _importances(est, "norm", self.norm_order)
        th = _threshold(scores, self.threshold, _is_l1(est))
        self.threshold_ = th
        n = scores.shape[0]
        mask = np.ones(n, dtype=bool)
        if self.max_features is not None:
            mask = np.zeros(n, dtype=bool)
            order = np.argsort(-scores, kind="mergesort")
            for i in range(min(self.max_features.__val__(), n)):
                mask[order[i]] = True
        for i in range(n):
            if scores[i] < th:
                mask[i] = False
        return mask

class RFE[E](_SelectorMixin):
    estimator: E
    estimator_: E
    n_features_to_select: Optional[float]
    step: float
    support_: np.ndarray[bool, 1]
    ranking_: np.ndarray[int, 1]
    n_features_: int
    step_n_features_: List[int]
    step_scores_: List[float]

    def __init__(self, estimator: E, *, n_features_to_select = None, step: float = 1,
                 verbose: int = 0, importance_getter: str = "auto"):
        self.estimator = estimator
        self.estimator_ = estimator
        self.n_features_to_select = None
        if n_features_to_select is not None:
            self.n_features_to_select = float(n_features_to_select)
        self.step = float(step)
        self.support_ = np.empty(0, dtype=bool)
        self.ranking_ = np.empty(0, dtype=int)
        self.n_features_ = 0
        self.step_n_features_ = []
        self.step_scores_ = []

    def _n_select(self, n_features: int) -> int:
        if self.n_features_to_select is None:
            return n_features // 2
        v = self.n_features_to_select.__val__()
        if v >= 1.0 and v == float(int(v)):
            return int(v)
        return int(n_features * v)

    def _fit(self, X, y, n_select: int, Xte, yte, record: bool):
        n_features = X.shape[1]
        step = int(max(1.0, self.step * n_features)) if 0.0 < self.step < 1.0 else int(self.step)
        support = np.ones(n_features, dtype=bool)
        ranking = np.ones(n_features, dtype=int)
        self.step_n_features_ = []
        self.step_scores_ = []
        while int(np.sum(support)) > n_select:
            features = np.flatnonzero(support)
            est = clone(self.estimator)
            est.fit(X[:, features], y)
            if record:
                self.step_n_features_.append(features.shape[0])
                self.step_scores_.append(float(est.score(Xte[:, features], yte)))
            imp = _importances(est, "square")
            ranks = np.argsort(imp, kind="stable")
            threshold = min(step, int(np.sum(support)) - n_select)
            for i in range(threshold):
                support[features[ranks[i]]] = False
            for j in range(n_features):
                if not support[j]:
                    ranking[j] += 1
        features = np.flatnonzero(support)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[:, features], y)
        if record:
            self.step_n_features_.append(features.shape[0])
            self.step_scores_.append(float(self.estimator_.score(Xte[:, features], yte)))
        self.n_features_ = int(support.sum())
        self.support_ = support
        self.ranking_ = ranking

    def fit(self, X, y):
        Xa = np.asarray(X, dtype=float)
        self._fit(Xa, y, self._n_select(Xa.shape[1]), Xa, y, False)
        return self

    def _get_support_mask(self):
        return self.support_

    def predict(self, X):
        return self.estimator_.predict(self.transform(X))

    def score(self, X, y) -> float:
        return float(self.estimator_.score(self.transform(X), y))

class RFECV[E, CV](_SelectorMixin):
    estimator: E
    estimator_: E
    step: float
    min_features_to_select: int
    cv: CV
    support_: np.ndarray[bool, 1]
    ranking_: np.ndarray[int, 1]
    n_features_: int
    cv_results_mean_: np.ndarray[float, 1]
    cv_results_std_: np.ndarray[float, 1]
    cv_results_n_features_: np.ndarray[int, 1]

    def __init__(self, estimator: E, *, step: float = 1, min_features_to_select: int = 1,
                 cv: CV = 5, scoring = None, verbose: int = 0, n_jobs = None,
                 importance_getter: str = "auto"):
        self.estimator = estimator
        self.estimator_ = estimator
        self.step = float(step)
        self.min_features_to_select = min_features_to_select
        self.cv = cv
        self.support_ = np.empty(0, dtype=bool)
        self.ranking_ = np.empty(0, dtype=int)
        self.n_features_ = 0
        self.cv_results_mean_ = np.empty(0)
        self.cv_results_std_ = np.empty(0)
        self.cv_results_n_features_ = np.empty(0, dtype=int)

    def fit(self, X, y):
        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y)
        n_features = Xa.shape[1]
        nsel = min(self.min_features_to_select, n_features)
        splits = _splits(self.cv, Xa, ya, _is_classifier(self.estimator))
        all_scores: List[List[float]] = []
        nfeat: List[int] = []
        for tr, te in splits:
            r = RFE(self.estimator, n_features_to_select=nsel, step=self.step)
            r._fit(Xa[tr], ya[tr], nsel, Xa[te], ya[te], True)
            all_scores.append(r.step_scores_)
            nfeat = r.step_n_features_
        m = len(nfeat)
        S = np.empty((len(all_scores), m))
        for i in range(len(all_scores)):
            for j in range(m):
                S[i, j] = all_scores[i][m - 1 - j]
        nrev = np.empty(m, dtype=int)
        for j in range(m):
            nrev[j] = nfeat[m - 1 - j]
        sums = np.sum(S, axis=0)
        best = nrev[int(np.argmax(sums))]
        r = RFE(self.estimator, n_features_to_select=best, step=self.step)
        r._fit(Xa, ya, best, Xa, ya, False)
        self.support_ = r.support_
        self.n_features_ = r.n_features_
        self.ranking_ = r.ranking_
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(Xa[:, self.support_], ya)
        self.cv_results_mean_ = np.mean(S, axis=0)
        self.cv_results_std_ = np.std(S, axis=0)
        self.cv_results_n_features_ = nrev
        return self

    def _get_support_mask(self):
        return self.support_

    def predict(self, X):
        return self.estimator_.predict(self.transform(X))

    def score(self, X, y) -> float:
        return float(self.estimator_.score(self.transform(X), y))

class SequentialFeatureSelector[E, CV](_SelectorMixin):
    estimator: E
    n_features_to_select: float
    nfs_auto: bool
    tol: Optional[float]
    direction: str
    cv: CV
    support_: np.ndarray[bool, 1]
    n_features_to_select_: int

    def __init__(self, estimator: E, *, n_features_to_select = "auto", tol: Optional[float] = None,
                 direction: str = "forward", scoring = None, cv: CV = 5, n_jobs = None):
        self.estimator = estimator
        if isinstance(n_features_to_select, str):
            self.nfs_auto = True
            self.n_features_to_select = 0.0
        else:
            self.nfs_auto = False
            self.n_features_to_select = float(n_features_to_select)
        self.tol = tol
        if direction != "forward" and direction != "backward":
            raise ValueError("direction must be 'forward' or 'backward'")
        self.direction = direction
        self.cv = cv
        self.support_ = np.empty(0, dtype=bool)
        self.n_features_to_select_ = 0

    def _cv_score(self, X, y, splits) -> float:
        s = 0.0
        for tr, te in splits:
            est = clone(self.estimator)
            est.fit(X[tr], y[tr])
            s += float(est.score(X[te], y[te]))
        return s / float(len(splits))

    def fit(self, X, y):
        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y)
        n_features = Xa.shape[1]
        if self.nfs_auto:
            self.n_features_to_select_ = n_features - 1 if self.tol is not None else n_features // 2
        else:
            v = self.n_features_to_select
            if v >= 1.0 and v == float(int(v)):
                if int(v) >= n_features:
                    raise ValueError("n_features_to_select must be < n_features.")
                self.n_features_to_select_ = int(v)
            else:
                self.n_features_to_select_ = int(n_features * v)
        if self.tol is not None and self.tol.__val__() < 0 and self.direction == "forward":
            raise ValueError("tol must be strictly positive when doing forward selection")
        splits = _splits(self.cv, Xa, ya, _is_classifier(self.estimator))
        mask = np.zeros(n_features, dtype=bool)
        n_iter = (self.n_features_to_select_ if self.nfs_auto or self.direction == "forward"
                  else n_features - self.n_features_to_select_)
        old = -np.inf
        auto = self.tol is not None and self.nfs_auto
        for _ in range(n_iter):
            best_idx = -1
            best = -np.inf
            for f in range(n_features):
                if mask[f]:
                    continue
                cand = mask.copy()
                cand[f] = True
                if self.direction == "backward":
                    cand = ~cand
                sc = self._cv_score(Xa[:, cand], ya, splits)
                if best_idx < 0 or sc > best:
                    best = sc
                    best_idx = f
            if auto and best - old < self.tol.__val__():
                break
            old = best
            mask[best_idx] = True
        if self.direction == "backward":
            mask = ~mask
        self.support_ = mask
        self.n_features_to_select_ = int(mask.sum())
        return self

    def _get_support_mask(self):
        return self.support_
