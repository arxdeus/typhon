# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# HistGradientBoostingClassifier / HistGradientBoostingRegressor
# (sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py).
# Single-threaded; dense float input (NaN = missing). Categorical features are
# given as a boolean mask or integer indices (no dataframes, so
# categorical_features="from_dtype" means "none"). Supported scoring for early
# stopping: "loss", None (accuracy / r2), "accuracy", "r2".

import math
import numpy as np
from numpy.random import Generator, PCG64
from .binning import _BinMapper
from .grower import TreeGrower, TreePredictor, make_predictor
from ._loss import _HGBLoss, _average
from ...tree._classes import _check_rs, _encode_classes, _r2
from ...model_selection import _tts_indices

def _norm_cat(cf) -> Tuple[List[int], List[bool], bool]:
    """-> (indices, mask, is_mask); empty means no categorical features."""
    idx = List[int]()
    mask = List[bool]()
    if cf is None:
        return idx, mask, False
    elif isinstance(cf, str):
        if cf != "from_dtype":
            raise ValueError(f"categorical_features must be an array-like of bool, int or str, got: {cf}.")
        return idx, mask, False
    else:
        is_mask = False
        for v in cf:
            if isinstance(v, bool):
                mask.append(v)
                is_mask = True
            else:
                idx.append(int(v))
        return idx, mask, is_mask

def _norm_ic(ic) -> Tuple[str, List[List[int]]]:
    out = List[List[int]]()
    if ic is None:
        return "", out
    elif isinstance(ic, str):
        if ic != "pairwise" and ic != "no_interactions":
            raise ValueError(f"interaction_cst must be 'pairwise', 'no_interactions' or a sequence, got {ic}")
        return ic, out
    else:
        for grp in ic:
            g = List[int]()
            for v in grp:
                g.append(int(v))
            out.append(g)
        return "list", out

def _norm_mono(mc) -> List[int]:
    out = List[int]()
    if mc is not None:
        for v in mc:
            out.append(int(v))
    return out

def _norm_cw(class_weight) -> Tuple[str, Dict[int, float]]:
    d = Dict[int, float]()
    if isinstance(class_weight, NoneType):
        return "", d
    elif isinstance(class_weight, str):
        if class_weight != "balanced":
            raise ValueError("class_weight must be dict, 'balanced', or None")
        return "balanced", d
    else:
        for k, v in class_weight.items():
            d[int(k)] = float(v)
        return "dict", d

def _should_stop(scores: List[float], n_iter_no_change: int, tol: float) -> bool:
    ref = n_iter_no_change + 1
    if len(scores) < ref:
        return False
    reference_score = scores[len(scores) - ref] + tol
    for t in range(len(scores) - ref + 1, len(scores)):
        if scores[t] > reference_score:
            return False
    return True

def _rows(X: np.ndarray[float, 2], idx) -> np.ndarray[float, 2]:
    n = len(idx)
    p = X.shape[1]
    out = np.empty((n, p))
    for t in range(n):
        r = int(idx[t])
        for j in range(p):
            out[t, j] = X[r, j]
    return out

def _sub(v: List[float], idx) -> List[float]:
    return [v[int(idx[t])] for t in range(len(idx))]

class _Preproc:
    """ColumnTransformer([("encoder", OrdinalEncoder, cat), ("numerical", ..., ~cat)]):
    categorical columns first (ordinal-encoded, unknown -> NaN), then numeric."""
    active: bool
    order: List[int]                 # original column of each output column
    n_cat: int
    categories: List[List[float]]    # sorted known raw values per categorical output column

    def __init__(self):
        self.active = False
        self.order = List[int]()
        self.n_cat = 0
        self.categories = List[List[float]]()

    def fit(self, X: np.ndarray[float, 2], is_cat: List[bool], max_bins: int):
        p = X.shape[1]
        n = X.shape[0]
        self.active = True
        self.order = [f for f in range(p) if is_cat[f]] + [f for f in range(p) if not is_cat[f]]
        self.n_cat = 0
        self.categories = List[List[float]]()
        for f in range(p):
            if is_cat[f]:
                self.n_cat += 1
                vals = sorted([X[i, f] for i in range(n) if not math.isnan(X[i, f])])
                cats = List[float]()
                for v in vals:
                    if len(cats) == 0 or cats[len(cats) - 1] != v:
                        cats.append(v)
                if len(cats) > max_bins:
                    raise ValueError(f"Categorical feature at index {f} is expected to have a cardinality <= {max_bins} but actually has a cardinality of {len(cats)}.")
                self.categories.append(cats)

    def transform(self, X: np.ndarray[float, 2]) -> np.ndarray[float, 2]:
        if not self.active:
            return X
        n = X.shape[0]
        p = len(self.order)
        out = np.empty((n, p))
        for j in range(p):
            f = self.order[j]
            if j < self.n_cat:
                cats = self.categories[j]
                for i in range(n):
                    v = X[i, f]
                    code = math.nan
                    if not math.isnan(v):
                        lo = 0
                        hi = len(cats)
                        while lo < hi:
                            mid = (lo + hi) >> 1
                            if cats[mid] < v:
                                lo = mid + 1
                            else:
                                hi = mid
                        if lo < len(cats) and cats[lo] == v:
                            code = float(lo)
                    out[i, j] = code
            else:
                for i in range(n):
                    out[i, j] = X[i, f]
        return out

class _BaseHGB:
    # parameters
    loss: str
    quantile: Optional[float]
    learning_rate: float
    max_iter: int
    max_leaf_nodes: Optional[int]
    max_depth: Optional[int]
    min_samples_leaf: int
    l2_regularization: float
    max_features: float
    max_bins: int
    _cat_idx: List[int]
    _cat_mask: List[bool]
    _cat_is_mask: bool
    monotonic_cst: List[int]
    _ic_kind: str
    _ic_list: List[List[int]]
    warm_start: bool
    early_stopping: str          # "auto", "true", "false"
    scoring: str                 # "loss", "default", "accuracy", "r2"
    validation_fraction: Optional[float]
    n_iter_no_change: int
    tol: float
    verbose: int
    random_state: Optional[int]
    _is_clf: bool
    _class_weight: str           # "", "balanced", "dict"
    _class_weight_dict: Dict[int, float]
    # fitted
    _predictors: List[List[TreePredictor]]
    _bin_mapper: _BinMapper
    _baseline_prediction: List[float]
    _loss: _HGBLoss
    _random_seed: int
    _feature_subsample_rng: Generator[PCG64]
    _pre: _Preproc
    train_score_: np.ndarray[float, 1]
    validation_score_: np.ndarray[float, 1]
    do_early_stopping_: bool
    n_trees_per_iteration_: int
    n_features_in_: int
    _n_features: int
    classes_: np.ndarray[int, 1]
    is_categorical_: List[bool]
    _fitted_with_sw: bool
    _fitted: bool

    def _init_common(self, loss: str, learning_rate: float, max_iter: int,
                     max_leaf_nodes: Optional[int], max_depth: Optional[int],
                     min_samples_leaf: int, l2_regularization: float, max_features: float,
                     max_bins: int, categorical_features, monotonic_cst, interaction_cst,
                     warm_start: bool, early_stopping, scoring, validation_fraction,
                     n_iter_no_change: int, tol: float, verbose: int,
                     random_state: Optional[int], is_clf: bool,
                     cw: Tuple[str, Dict[int, float]], quantile: Optional[float]):
        if is_clf and loss != "log_loss":
            raise ValueError(f"unknown loss '{loss}'")
        self.loss = loss
        self.quantile = quantile
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_features = max_features
        self.max_bins = max_bins
        self._cat_idx, self._cat_mask, self._cat_is_mask = _norm_cat(categorical_features)
        self.monotonic_cst = _norm_mono(monotonic_cst)
        self._ic_kind, self._ic_list = _norm_ic(interaction_cst)
        self.warm_start = warm_start
        if isinstance(early_stopping, str):
            if early_stopping != "auto":
                raise ValueError("early_stopping must be 'auto' or a bool")
            self.early_stopping = "auto"
        else:
            self.early_stopping = "true" if early_stopping else "false"
        if scoring is None:
            self.scoring = "default"
        else:
            self.scoring = scoring
        if self.scoring != "loss" and self.scoring != "default" and self.scoring != "accuracy" and self.scoring != "r2":
            raise ValueError(f"HistGradientBoosting: scoring={self.scoring} is not supported (use 'loss', None, 'accuracy' or 'r2')")
        if validation_fraction is None:
            self.validation_fraction = None
        else:
            self.validation_fraction = float(validation_fraction)
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self._is_clf = is_clf
        self._class_weight = cw[0]
        self._class_weight_dict = cw[1]
        self._predictors = List[List[TreePredictor]]()
        self._bin_mapper = _BinMapper(256, 0)
        self._baseline_prediction = List[float]()
        self._loss = _HGBLoss("squared_error", 0.5, 1, False)
        self._random_seed = 0
        self._feature_subsample_rng = np.random.default_rng(0)
        self._pre = _Preproc()
        self.train_score_ = np.empty(0)
        self.validation_score_ = np.empty(0)
        self.do_early_stopping_ = False
        self.n_trees_per_iteration_ = 1
        self.n_features_in_ = 0
        self._n_features = 0
        self.classes_ = np.empty(0, dtype=int)
        self.is_categorical_ = List[bool]()
        self._fitted_with_sw = False
        self._fitted = False

    @property
    def n_iter_(self) -> int:
        if not self._fitted:
            raise ValueError("This estimator is not fitted yet. Call 'fit' first.")
        return len(self._predictors)

    def _is_fitted(self) -> bool:
        return len(self._predictors) > 0

    def _check_interaction_cst(self, n_features: int) -> Tuple[bool, List[List[int]]]:
        if self._ic_kind == "":
            return False, List[List[int]]()
        cst = List[List[int]]()
        if self._ic_kind == "no_interactions":
            for i in range(n_features):
                cst.append([i])
        elif self._ic_kind == "pairwise":
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    cst.append([i, j])
        else:
            for g in self._ic_list:
                s = sorted(g)
                d = List[int]()
                for v in s:
                    if len(d) == 0 or d[len(d) - 1] != v:
                        d.append(v)
                cst.append(d)
        seen = [False] * n_features
        for g in cst:
            for x in g:
                if not (0 <= x < n_features):
                    raise ValueError(f"Interaction constraints must consist of integer indices in [0, n_features - 1] = [0, {n_features - 1}], specifying the position of features, got invalid indices: {g}")
                seen[x] = True
        rest = [f for f in range(n_features) if not seen[f]]
        if len(rest) > 0:
            cst.append(rest)
        return True, cst

    def _scorer_score(self, raw: List[float], n: int, y: List[float], sw: List[float],
                      has_w: bool) -> float:
        """Default scorer of the estimator (accuracy / r2) from raw predictions."""
        K = self.n_trees_per_iteration_
        if self._is_clf:
            m = List[float](n)
            for i in range(n):
                c = 0
                if K == 1:
                    c = 1 if raw[i] > 0.0 else 0
                else:
                    for k in range(1, K):
                        if raw[k * n + i] > raw[c * n + i]:
                            c = k
                m.append(1.0 if float(c) == y[i] else 0.0)
            return _average(m, sw, has_w)
        yp = [self._loss.inverse_link(raw[i]) for i in range(n)]
        w = sw if has_w else [1.0] * n
        ybar = _average(y, sw, has_w)
        num = np.empty(n)
        den = np.empty(n)
        for i in range(n):
            num[i] = w[i] * ((y[i] - yp[i]) * (y[i] - yp[i]))
            den[i] = w[i] * ((y[i] - ybar) * (y[i] - ybar))
        nu = float(np.sum(num))
        de = float(np.sum(den))
        if de == 0.0:
            return 1.0 if nu == 0.0 else 0.0
        return 1.0 - nu / de

    def _predict_binned_into(self, preds: List[TreePredictor], Xb: List[int], n: int,
                             raw: List[float]):
        mb = self._bin_mapper.missing_values_bin_idx_
        for k in range(len(preds)):
            pr = preds[k]
            for i in range(n):
                raw[k * n + i] += pr.predict_binned_row(Xb, n, i, mb)

    def _check_es(self, y_tr: List[float], raw_tr: List[float], sw_tr: List[float],
                  y_va: List[float], raw_va: List[float], sw_va: List[float], has_w: bool,
                  use_val: bool, tscores: List[float], vscores: List[float]) -> bool:
        n_tr = len(y_tr)
        if self.scoring == "loss":
            tscores.append(-self._loss(y_tr, raw_tr, sw_tr, has_w))
            if use_val:
                vscores.append(-self._loss(y_va, raw_va, sw_va, has_w))
                return _should_stop(vscores, self.n_iter_no_change, self.tol)
            return _should_stop(tscores, self.n_iter_no_change, self.tol)
        if n_tr > 10000:
            raise NotImplementedError("HistGradientBoosting: scorer-based early stopping on more than 10000 training samples (small-trainset resampling) is not supported; use scoring='loss'")
        tscores.append(self._scorer_score(raw_tr, n_tr, y_tr, sw_tr, has_w))
        if use_val:
            vscores.append(self._scorer_score(raw_va, len(y_va), y_va, sw_va, has_w))
            return _should_stop(vscores, self.n_iter_no_change, self.tol)
        return _should_stop(tscores, self.n_iter_no_change, self.tol)

    def _preprocess_fit(self, Xa: np.ndarray[float, 2]) -> Tuple[np.ndarray[float, 2], List[List[float]], List[bool]]:
        p = Xa.shape[1]
        is_cat = [False] * p
        if self._cat_is_mask:
            if len(self._cat_mask) != p:
                raise ValueError(f"categorical_features set as a boolean mask must have shape (n_features,), got: ({len(self._cat_mask)},)")
            is_cat = list(self._cat_mask)
        else:
            for f in self._cat_idx:
                if f >= p or f < 0:
                    raise ValueError("categorical_features set as integer indices must be in [0, n_features - 1]")
                is_cat[f] = True
        any_cat = False
        for v in is_cat:
            if v:
                any_cat = True
        self.n_features_in_ = p
        self._pre = _Preproc()
        known = List[List[float]]()
        remapped = List[bool]()
        if not any_cat:
            self.is_categorical_ = List[bool]()
            return Xa, known, remapped
        self.is_categorical_ = is_cat
        self._pre.fit(Xa, is_cat, self.max_bins)
        Xt = self._pre.transform(Xa)
        for j in range(p):
            if j < self._pre.n_cat:
                known.append([float(c) for c in range(len(self._pre.categories[j]))])
                remapped.append(True)
            else:
                known.append(List[float]())
                remapped.append(False)
        return Xt, known, remapped

    def _check_X(self, X) -> np.ndarray[float, 2]:
        if not self._fitted:
            raise ValueError("This estimator is not fitted yet. Call 'fit' first.")
        Xa = np.asarray(X, dtype=float)
        if Xa.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xa.shape[1]} features, but {self.__class__.__name__} is expecting {self.n_features_in_} features as input.")
        return self._pre.transform(Xa)

    def _raw_predict_iter(self, Xt: np.ndarray[float, 2], start: int, stop: int,
                          raw: np.ndarray[float, 2]):
        kc = self._bin_mapper.known_cat_bitsets()
        n = Xt.shape[0]
        for it in range(start, stop):
            preds = self._predictors[it]
            for k in range(len(preds)):
                pr = preds[k]
                for i in range(n):
                    raw[i, k] += pr.predict_row(Xt, i, kc)

    def _raw_predict(self, X) -> np.ndarray[float, 2]:
        Xt = self._check_X(X)
        n = Xt.shape[0]
        K = self.n_trees_per_iteration_
        raw = np.empty((n, K))
        for i in range(n):
            for k in range(K):
                raw[i, k] = self._baseline_prediction[k]
        self._raw_predict_iter(Xt, 0, len(self._predictors), raw)
        return raw

    def _staged_raw_predict(self, X):
        Xt = self._check_X(X)
        n = Xt.shape[0]
        K = self.n_trees_per_iteration_
        raw = np.empty((n, K))
        for i in range(n):
            for k in range(K):
                raw[i, k] = self._baseline_prediction[k]
        for it in range(len(self._predictors)):
            self._raw_predict_iter(Xt, it, it + 1, raw)
            yield raw.copy()

    def _compute_partial_dependence_recursion(self, grid, target_features: List[int]):
        if self._fitted_with_sw:
            raise NotImplementedError("{} does not support partial dependence plots with the 'recursion' method when sample weights were given during fit time.")
        g = np.asarray(grid, dtype=float)
        K = self.n_trees_per_iteration_
        out = np.zeros((K, g.shape[0]))
        accs = [[0.0] * g.shape[0] for _ in range(K)]
        for preds in self._predictors:
            for k in range(len(preds)):
                preds[k].compute_partial_dependence(g, target_features, accs[k])
        for k in range(K):
            for s in range(g.shape[0]):
                out[k, s] = accs[k][s]
        return out

    def _fit_impl(self, X, yv: List[float], sample_weight, cw: List[float], has_cw: bool):
        Xa0 = np.asarray(X, dtype=float)
        n_all = Xa0.shape[0]
        Xa, known, remapped = self._preprocess_fit(Xa0)
        has_w = sample_weight is not None or has_cw
        self._fitted_with_sw = sample_weight is not None
        sw = [1.0] * n_all
        if sample_weight is not None:
            swa = np.asarray(sample_weight, dtype=float)
            if swa.shape[0] != n_all:
                raise ValueError(f"sample_weight.shape == ({swa.shape[0]},), expected ({n_all},)!")
            for i in range(n_all):
                sw[i] = float(swa[i])
        if has_cw:
            for i in range(n_all):
                sw[i] = sw[i] * cw[i]
        rng = _check_rs(self.random_state)
        if not self.warm_start or not self._is_fitted():
            self._random_seed = int(rng.randint(4294967295))
            fseed = int(rng.randint(4294967295))
            self._feature_subsample_rng = np.random.default_rng(fseed)
        if len(self.monotonic_cst) > 0 and self.n_trees_per_iteration_ != 1:
            raise ValueError("monotonic constraints are not supported for multiclass classification.")
        p = Xa.shape[1]
        mono = [0] * p
        if len(self.monotonic_cst) > 0:
            if len(self.monotonic_cst) != p:
                raise ValueError(f"monotonic_cst has shape ({len(self.monotonic_cst)},) but the input data X has {p} features.")
            for v in self.monotonic_cst:
                if v != -1 and v != 0 and v != 1:
                    raise ValueError("monotonic_cst must be an array-like of -1, 0 or 1.")
            if len(remapped) > 0:
                order = self._pre.order
                mono = [self.monotonic_cst[order[j]] for j in range(p)]
            else:
                mono = list(self.monotonic_cst)
        self._n_features = p
        has_ic, ic = self._check_interaction_cst(p)
        self._loss = self._get_loss(has_w)
        if self.early_stopping == "auto":
            self.do_early_stopping_ = n_all > 10000
        else:
            self.do_early_stopping_ = self.early_stopping == "true"
        use_val = self.validation_fraction is not None
        X_tr = Xa
        y_tr = yv
        sw_tr = sw
        X_va = np.empty((0, p))
        y_va = List[float]()
        sw_va = List[float]()
        if self.do_early_stopping_ and use_val:
            vf = self.validation_fraction
            strat = np.array(yv)
            nt = int(math.ceil(vf * float(n_all))) if vf < 1.0 else int(vf)
            tr = np.empty(0, dtype=int)
            te = np.empty(0, dtype=int)
            if self._is_clf:
                tr, te = _tts_indices(n_all, nt, None, self._random_seed, True, strat)
            else:
                tr, te = _tts_indices(n_all, nt, None, self._random_seed, True, None)
            X_tr = _rows(Xa, tr)
            X_va = _rows(Xa, te)
            y_tr = _sub(yv, tr)
            y_va = _sub(yv, te)
            sw_tr = _sub(sw, tr)
            sw_va = _sub(sw, te)
        n = X_tr.shape[0]
        n_va = X_va.shape[0]
        n_bins = self.max_bins + 1
        self._bin_mapper = _BinMapper(n_bins, self._random_seed)
        self._bin_mapper.fit(X_tr, sw_tr, has_w, remapped, known)
        Xb = self._bin_mapper.transform(X_tr)
        Xb_va = self._bin_mapper.transform(X_va) if n_va > 0 else List[int]()
        mb = self._bin_mapper.missing_values_bin_idx_
        has_missing = [False] * p
        for f in range(p):
            for i in range(n):
                if Xb[f * n + i] == mb:
                    has_missing[f] = True
                    break
        is_cat = remapped if len(remapped) > 0 else [False] * p
        K = self.n_trees_per_iteration_
        raw = [0.0] * (n * K)
        raw_va = [0.0] * (n_va * K)
        tscores = List[float]()
        vscores = List[float]()
        begin = 0
        loss = self._loss
        if not (self._is_fitted() and self.warm_start):
            self._baseline_prediction = loss.fit_intercept_only(y_tr, sw_tr, has_w)
            for k in range(K):
                for i in range(n):
                    raw[k * n + i] = self._baseline_prediction[k]
                for i in range(n_va):
                    raw_va[k * n_va + i] = self._baseline_prediction[k]
            self._predictors = List[List[TreePredictor]]()
            if self.do_early_stopping_:
                self._check_es(y_tr, raw, sw_tr, y_va, raw_va, sw_va, has_w, use_val,
                               tscores, vscores)
        else:
            if self.max_iter < len(self._predictors):
                raise ValueError(f"max_iter={self.max_iter} must be larger than or equal to n_iter_={len(self._predictors)} when warm_start==True")
            tscores = [float(v) for v in self.train_score_]
            vscores = [float(v) for v in self.validation_score_]
            for k in range(K):
                for i in range(n):
                    raw[k * n + i] = self._baseline_prediction[k]
                for i in range(n_va):
                    raw_va[k * n_va + i] = self._baseline_prediction[k]
            for preds in self._predictors:
                self._predict_binned_into(preds, Xb, n, raw)
                if self.do_early_stopping_ and n_va > 0:
                    self._predict_binned_into(preds, Xb_va, n_va, raw_va)
            begin = len(self._predictors)
        self._fit_loop(begin, Xb, n, p, n_bins, y_tr, sw_tr, has_w, raw, Xb_va, n_va, y_va,
                       sw_va, raw_va, use_val, has_missing, is_cat, mono, ic, has_ic,
                       tscores, vscores)
        self.train_score_ = np.array(tscores)
        self.validation_score_ = np.array(vscores)
        self._fitted = True

    def _fit_loop(self, begin: int, Xb: List[int], n: int, p: int, n_bins: int,
                  y_tr: List[float], sw_tr: List[float], has_w: bool, raw: List[float],
                  Xb_va: List[int], n_va: int, y_va: List[float], sw_va: List[float],
                  raw_va: List[float], use_val: bool, has_missing: List[bool],
                  is_cat: List[bool], mono: List[int], ic: List[List[int]], has_ic: bool,
                  tscores: List[float], vscores: List[float]):
        loss = self._loss
        K = self.n_trees_per_iteration_
        const_h = loss.constant_hessian
        g = [float32(0.0)] * (n * K)
        h = [float32(1.0)] * (1 if const_h else n * K)
        for it in range(begin, self.max_iter):
            loss.gradient_hessian(y_tr, raw, sw_tr, has_w, g, h)
            stage = List[TreePredictor]()
            for k in range(K):
                gk = g[k * n:(k + 1) * n] if K > 1 else g
                hk = h if const_h else (h[k * n:(k + 1) * n] if K > 1 else h)
                grower = TreeGrower(Xb, n, p, gk, hk, const_h, self.max_leaf_nodes,
                                    self.max_depth, self.min_samples_leaf, n_bins,
                                    self._bin_mapper.n_bins_non_missing_, has_missing, is_cat,
                                    mono, ic, has_ic, self.l2_regularization,
                                    self.max_features, self._feature_subsample_rng,
                                    self.learning_rate)
                grower.grow()
                part = grower.splitter.partition
                if not loss.differentiable:
                    for leaf in grower.finalized_leaves:
                        idx = part[leaf.start:leaf.stop]
                        diff = [y_tr[i] - raw[k * n + i] for i in idx]
                        wl = [sw_tr[i] for i in idx] if has_w else List[float]()
                        upd = loss.fit_intercept_only(diff, wl, has_w)[0]
                        leaf.value = grower.shrinkage * upd
                pred = make_predictor(grower, self._bin_mapper.bin_thresholds_)
                stage.append(pred)
                for leaf in grower.finalized_leaves:
                    v = leaf.value
                    for pos in range(leaf.start, leaf.stop):
                        raw[k * n + part[pos]] += v
            self._predictors.append(stage)
            stop = False
            if self.do_early_stopping_:
                if n_va > 0:
                    self._predict_binned_into(stage, Xb_va, n_va, raw_va)
                stop = self._check_es(y_tr, raw, sw_tr, y_va, raw_va, sw_va, has_w, use_val,
                                      tscores, vscores)
            if self.verbose >= 2:
                print(f"[{it + 1}/{self.max_iter}]")
            if stop:
                break

class HistGradientBoostingRegressor(_BaseHGB):
    @property
    def _estimator_type(self) -> str:
        return "regressor"

    def __init__(self, loss: str = "squared_error", *, quantile: Optional[float] = None,
                 learning_rate: float = 0.1, max_iter: int = 100,
                 max_leaf_nodes: Optional[int] = 31, max_depth: Optional[int] = None,
                 min_samples_leaf: int = 20, l2_regularization: float = 0.0,
                 max_features: float = 1.0, max_bins: int = 255,
                 categorical_features = "from_dtype", monotonic_cst = None,
                 interaction_cst = None, warm_start: bool = False, early_stopping = "auto",
                 scoring = "loss", validation_fraction = 0.1, n_iter_no_change: int = 10,
                 tol: float = 1e-7, verbose: int = 0, random_state: Optional[int] = None):
        self._init_common(loss, learning_rate, max_iter, max_leaf_nodes, max_depth,
                          min_samples_leaf, l2_regularization, max_features, max_bins,
                          categorical_features, monotonic_cst, interaction_cst, warm_start,
                          early_stopping, scoring, validation_fraction, n_iter_no_change, tol,
                          verbose, random_state, False, _norm_cw(None), quantile)

    def _get_loss(self, has_w: bool) -> _HGBLoss:
        if self.loss == "quantile":
            if self.quantile is None:
                raise ValueError("quantile must be set when loss='quantile'")
            q = self.quantile
            if not (0.0 < q < 1.0):
                raise ValueError(f"quantile == {q}, must be > 0 and < 1.")
            return _HGBLoss("quantile", q, 1, has_w)
        if (self.loss != "squared_error" and self.loss != "absolute_error"
                and self.loss != "gamma" and self.loss != "poisson"):
            raise ValueError(f"unknown loss '{self.loss}'")
        return _HGBLoss(self.loss, 0.5, 1, has_w)

    def fit(self, X, y, sample_weight = None):
        ya = np.asarray(y, dtype=float)
        yv = [float(ya[i]) for i in range(ya.shape[0])]
        self.n_trees_per_iteration_ = 1
        if self.loss == "gamma":
            for v in yv:
                if not (v > 0.0):
                    raise ValueError("loss='gamma' requires strictly positive y.")
        elif self.loss == "poisson":
            s = 0.0
            for v in yv:
                if v < 0.0:
                    raise ValueError("loss='poisson' requires non-negative y and sum(y) > 0.")
                s += v
            if not (s > 0.0):
                raise ValueError("loss='poisson' requires non-negative y and sum(y) > 0.")
        self._fit_impl(X, yv, sample_weight, List[float](), False)
        return self

    def predict(self, X):
        raw = self._raw_predict(X)
        n = raw.shape[0]
        out = np.empty(n)
        for i in range(n):
            out[i] = self._loss.inverse_link(raw[i, 0])
        return out

    def staged_predict(self, X):
        for raw in self._staged_raw_predict(X):
            n = raw.shape[0]
            out = np.empty(n)
            for i in range(n):
                out[i] = self._loss.inverse_link(raw[i, 0])
            yield out

    def score(self, X, y, sample_weight = None) -> float:
        return _r2(y, self.predict(X))

class HistGradientBoostingClassifier(_BaseHGB):
    @property
    def _estimator_type(self) -> str:
        return "classifier"

    def __init__(self, loss: str = "log_loss", *, learning_rate: float = 0.1,
                 max_iter: int = 100, max_leaf_nodes: Optional[int] = 31,
                 max_depth: Optional[int] = None, min_samples_leaf: int = 20,
                 l2_regularization: float = 0.0, max_features: float = 1.0,
                 max_bins: int = 255, categorical_features = "from_dtype",
                 monotonic_cst = None, interaction_cst = None, warm_start: bool = False,
                 early_stopping = "auto", scoring = "loss", validation_fraction = 0.1,
                 n_iter_no_change: int = 10, tol: float = 1e-7, verbose: int = 0,
                 random_state: Optional[int] = None, class_weight = None):
        self._init_common(loss, learning_rate, max_iter, max_leaf_nodes, max_depth,
                          min_samples_leaf, l2_regularization, max_features, max_bins,
                          categorical_features, monotonic_cst, interaction_cst, warm_start,
                          early_stopping, scoring, validation_fraction, n_iter_no_change, tol,
                          verbose, random_state, True, _norm_cw(class_weight), None)

    def _get_loss(self, has_w: bool) -> _HGBLoss:
        if self.n_trees_per_iteration_ == 1:
            return _HGBLoss("binomial", 0.5, 1, has_w)
        return _HGBLoss("multinomial", 0.5, self.n_trees_per_iteration_, has_w)

    def fit(self, X, y, sample_weight = None):
        classes, enc = _encode_classes(np.asarray(y))
        self.classes_ = classes
        nc = classes.shape[0]
        self.n_trees_per_iteration_ = 1 if nc <= 2 else nc
        cw = List[float]()
        if self._class_weight == "balanced":
            cnt = [0] * nc
            for v in enc:
                cnt[int(v)] += 1
            n = len(enc)
            w = [float(n) / (float(nc) * float(cnt[k])) for k in range(nc)]
            cw = [w[int(v)] for v in enc]
        elif self._class_weight == "dict":
            # compute_sample_weight on the label-encoded target (sklearn quirk:
            # keys refer to encoded classes 0..n_classes-1)
            cw = [self._class_weight_dict.get(int(v), 1.0) for v in enc]
        self._fit_impl(X, enc, sample_weight, cw, self._class_weight != "")
        return self

    def _labels(self, raw):
        n = raw.shape[0]
        out = np.empty(n, dtype=int)
        for i in range(n):
            if raw.shape[1] == 1:
                out[i] = self.classes_[1 if raw[i, 0] > 0.0 else 0]
            else:
                best = 0
                for k in range(1, raw.shape[1]):
                    if raw[i, k] > raw[i, best]:
                        best = k
                out[i] = self.classes_[best]
        return out

    def predict(self, X):
        return self._labels(self._raw_predict(X))

    def staged_predict(self, X):
        for raw in self._staged_raw_predict(X):
            yield self._labels(raw)

    def predict_proba(self, X):
        return self._loss.predict_proba(self._raw_predict(X))

    def staged_predict_proba(self, X):
        for raw in self._staged_raw_predict(X):
            yield self._loss.predict_proba(raw)

    def decision_function(self, X):
        """(n, n_trees_per_iteration_). Deviation: binary problems give (n, 1)
        (sklearn ravels to (n,)); the static return rank cannot depend on the
        number of classes. Use decision_function_1d for the (n,) form."""
        return self._raw_predict(X)

    def decision_function_1d(self, X):
        """Binary-only (n,) decision function, like sklearn's binary output."""
        raw = self._raw_predict(X)
        if raw.shape[1] != 1:
            raise ValueError("decision_function_1d is only defined for binary problems")
        return raw.ravel()

    def staged_decision_function(self, X):
        for raw in self._staged_raw_predict(X):
            yield raw

    def score(self, X, y, sample_weight = None) -> float:
        from ...base import _accuracy
        return _accuracy(y, self.predict(X))
