# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Gradient tree boosting (sklearn/ensemble/_gb.py, _gradient_boosting.pyx and
# the losses of sklearn/_loss). The regression trees share the boosting
# RandomState like sklearn (splitter seeds and subsample masks come from the
# same stream), so trees, predictions and importances match sklearn.
# init may be None (prior), "zero" or an estimator (fitted on the float32
# training data like sklearn); n_iter_no_change early stopping and warm_start
# follow sklearn (train_test_split needs an integer random_state).

import math
import numpy as np
from numpy.random import RandomState
from ..tree._classes import (DecisionTreeRegressor, _set_tree_params, _copy_tree_params,
                             _to_f32_colmajor, _encode_classes, _check_rs, _r2)
from ..tree._tree import TREE_LEAF
from ._base import _mean_rows
from ..model_selection._split import _tts_indices

_EPS: float = 2.220446049250313e-16

def _expit(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _logit(x: float) -> float:
    if x < 0.3 or x > 0.65:
        return math.log(x / (1.0 - x))
    s = 2.0 * (x - 0.5)
    return math.log1p(s) - math.log1p(-s)

def _log1pexp(x: float) -> float:
    if x <= -37.0:
        return math.exp(x)
    elif x <= -2.0:
        return math.log1p(math.exp(x))
    elif x <= 18.0:
        return math.log(1.0 + math.exp(x))
    elif x <= 33.3:
        return x + math.exp(-x)
    return x

def _avg(a: List[float], w: List[float]) -> float:
    # np.average(a, weights=w): (a * w).sum() / w.sum() with numpy pairwise sums
    n = len(a)
    aw = np.empty(n)
    wa = np.empty(n)
    for i in range(n):
        aw[i] = a[i] * w[i]
        wa[i] = w[i]
    return float(np.sum(aw)) / float(np.sum(wa))

def _weighted_percentile(a: List[float], w: List[float], q: float) -> float:
    """sklearn.utils.stats._weighted_percentile (1-d, average=False)."""
    n = len(a)
    idx = sorted(range(n), key=lambda i: a[i])
    cdf = [0.0] * n
    acc = 0.0
    for t in range(n):
        acc += w[idx[t]]
        cdf[t] = acc
    adj = q / 100.0 * cdf[n - 1]
    if adj == 0.0:
        adj = 5e-324
    lo = 0
    hi = n
    while lo < hi:
        mid = (lo + hi) >> 1
        if cdf[mid] < adj:
            lo = mid + 1
        else:
            hi = mid
    if lo > n - 1:
        lo = n - 1
    return a[idx[lo]]

def _init_name(init) -> str:
    if init is None:
        return ""
    elif isinstance(init, str):
        if init != "zero":
            raise ValueError(f"The init parameter must be an estimator or 'zero'. Got init={init}")
        return "zero"
    else:
        return "est"

def _init_fit(est, X, y, sw, has_w: bool):
    if isinstance(est, NoneType) or isinstance(est, str):
        pass
    else:
        if has_w:
            est.fit(X, y, sample_weight=sw)
        else:
            est.fit(X, y)

def _init_predict(est, X, is_clf: bool) -> np.ndarray[float, 2]:
    if isinstance(est, NoneType) or isinstance(est, str):
        return np.empty((0, 0))
    elif hasattr(est, "predict_proba"):
        if is_clf:
            return np.asarray(est.predict_proba(X), dtype=float)
        else:
            pr = np.asarray(est.predict(X), dtype=float)
            return pr.reshape((pr.shape[0], 1))
    else:
        if is_clf:
            raise ValueError("The init estimator of a classifier must support predict_proba.")
        pr = np.asarray(est.predict(X), dtype=float)
        return pr.reshape((pr.shape[0], 1))

class _Loss:
    kind: str          # squared_error absolute_error huber quantile log_loss(binomial) multinomial exponential
    quantile: float
    delta: float
    n_classes: int

    def __init__(self, kind: str, quantile: float, n_classes: int):
        self.kind = kind
        self.quantile = quantile
        self.delta = 0.5
        self.n_classes = n_classes

    def factor(self) -> float:
        return 2.0 if self.kind == "squared_error" or self.kind == "binomial" else 1.0

    def neg_gradient(self, y: List[float], raw, k: int) -> List[float]:
        n = len(y)
        g = [0.0] * n
        kind = self.kind
        for i in range(n):
            r = raw[i, k]
            yi = y[i]
            v = 0.0
            if kind == "squared_error":
                v = r - yi
            elif kind == "absolute_error":
                v = 1.0 if r > yi else -1.0
            elif kind == "huber":
                res = r - yi
                if abs(res) <= self.delta:
                    v = res
                else:
                    v = self.delta if res >= 0.0 else -self.delta
            elif kind == "quantile":
                v = -self.quantile if yi >= r else 1.0 - self.quantile
            elif kind == "binomial":
                if r > -37.0:
                    e = math.exp(-r)
                    v = ((1.0 - yi) - yi * e) / (1.0 + e)
                else:
                    v = math.exp(r) - yi
            elif kind == "exponential":
                e = math.exp(r)
                v = -yi / e + (1.0 - yi) * e
            g[i] = -v
        return g

    def neg_gradient_multinomial(self, y: List[float], raw) -> np.ndarray[float, 2]:
        n = raw.shape[0]
        K = raw.shape[1]
        out = np.empty((n, K))
        p = [0.0] * K
        for i in range(n):
            mx = raw[i, 0]
            for k in range(1, K):
                if mx < raw[i, k]:
                    mx = raw[i, k]
            s = 0.0
            for k in range(K):
                p[k] = math.exp(raw[i, k] - mx)
                s += p[k]
            for k in range(K):
                p[k] /= s
                out[i, k] = -(p[k] - (1.0 if int(y[i]) == k else 0.0))
        return out

    def pointwise(self, y: List[float], raw, rows: List[int]) -> List[float]:
        out = List[float](len(rows))
        kind = self.kind
        for i in rows:
            yi = y[i]
            if kind == "multinomial":
                K = raw.shape[1]
                mx = raw[i, 0]
                for k in range(1, K):
                    if mx < raw[i, k]:
                        mx = raw[i, k]
                s = 0.0
                for k in range(K):
                    s += math.exp(raw[i, k] - mx)
                out.append(math.log(s) + mx - raw[i, int(yi)])
                continue
            r = raw[i, 0]
            if kind == "squared_error":
                out.append(0.5 * (r - yi) * (r - yi))
            elif kind == "absolute_error":
                out.append(abs(r - yi))
            elif kind == "huber":
                ae = abs(yi - r)
                out.append(0.5 * ae ** 2 if ae <= self.delta else self.delta * (ae - 0.5 * self.delta))
            elif kind == "quantile":
                out.append(self.quantile * (yi - r) if yi >= r else (1.0 - self.quantile) * (r - yi))
            elif kind == "binomial":
                out.append(_log1pexp(r) - yi * r)
            else:
                e = math.exp(r)
                out.append(yi / e + (1.0 - yi) * e)
        return out

    def __call__(self, y: List[float], raw, rows: List[int], sw: List[float]) -> float:
        return _avg(self.pointwise(y, raw, rows), [sw[i] for i in rows])

    def predict_proba(self, raw):
        n = raw.shape[0]
        if self.kind == "multinomial":
            K = raw.shape[1]
            out = np.empty((n, K))
            for i in range(n):
                mx = raw[i, 0]
                for k in range(1, K):
                    if raw[i, k] > mx:
                        mx = raw[i, k]
                s = 0.0
                for k in range(K):
                    out[i, k] = math.exp(raw[i, k] - mx)
                    s += out[i, k]
                for k in range(K):
                    out[i, k] /= s
            return out
        out = np.empty((n, 2))
        for i in range(n):
            p1 = _expit(2.0 * raw[i, 0]) if self.kind == "exponential" else _expit(raw[i, 0])
            out[i, 1] = p1
            out[i, 0] = 1.0 - p1
        return out

def _random_sample_mask(n: int, n_inbag: int, rs) -> List[bool]:
    rand = rs.uniform(size=n)
    mask = [False] * n
    n_bagged = 0
    for i in range(n):
        if rand[i] * float(n - i) < float(n_inbag - n_bagged):
            mask[i] = True
            n_bagged += 1
    return mask

class _BaseGB[T, I]:
    loss: str
    learning_rate: float
    n_estimators: int
    subsample: float
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
    init_: str           # "" (prior), "zero" or "est"
    init: I
    random_state: Optional[int]
    alpha: float
    verbose: int
    n_iter_no_change: Optional[int]
    validation_fraction: float
    tol: float
    warm_start: bool
    _rng: RandomState
    _is_clf: bool
    # fitted
    estimators_: List[List[T]]
    train_score_: np.ndarray[float, 1]
    oob_improvement_: np.ndarray[float, 1]
    oob_scores_: np.ndarray[float, 1]
    oob_score_: float
    n_estimators_: int
    n_features_in_: int
    n_trees_per_iteration_: int
    classes_: np.ndarray[int, 1]
    n_classes_: int
    max_features_: int
    _loss: _Loss
    _init_raw: List[float]
    _fitted: bool

    def __init__(self, is_clf: bool, loss: str, learning_rate: float, n_estimators: int,
                 subsample: float, min_samples_split, min_samples_leaf,
                 min_weight_fraction_leaf: float, max_depth: Optional[int],
                 min_impurity_decrease: float, init, random_state: Optional[int], max_features,
                 alpha: float, verbose: int, max_leaf_nodes: Optional[int],
                 n_iter_no_change: Optional[int], ccp_alpha: float, warm_start: bool,
                 validation_fraction: float, tol: float):
        self._is_clf = is_clf
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        _set_tree_params(self, "squared_error", "best", max_depth, min_samples_split,
                         min_samples_leaf, min_weight_fraction_leaf, max_features,
                         max_leaf_nodes, min_impurity_decrease, ccp_alpha, None)
        self.init_ = _init_name(init)
        self.init = init
        self.random_state = random_state
        self.alpha = alpha
        self.verbose = verbose
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.tol = tol
        self.warm_start = warm_start
        self._rng = np.random.RandomState(0)
        self.estimators_ = List[List[T]]()
        self.train_score_ = np.empty(0)
        self.oob_improvement_ = np.empty(0)
        self.oob_scores_ = np.empty(0)
        self.oob_score_ = 0.0
        self.n_estimators_ = 0
        self.n_features_in_ = 0
        self.n_trees_per_iteration_ = 1
        self.classes_ = np.empty(0, dtype=int)
        self.n_classes_ = 0
        self.max_features_ = 0
        self._loss = _Loss("squared_error", 0.5, 1)
        self._init_raw = List[float]()
        self._fitted = False

    # ------------------------------------------------------------- init

    def _compute_init(self, y: List[float], sw: List[float], has_w: bool) -> List[float]:
        K = self.n_trees_per_iteration_
        if self.init_ == "zero":
            return [0.0] * K
        kind = self._loss.kind
        n = len(y)
        if self._is_clf:
            nc = self.n_classes_
            prior = [0.0] * nc
            for i in range(n):
                prior[int(y[i])] += sw[i] if has_w else 1.0
            tot = 0.0
            for v in prior:
                tot += v
            for c in range(nc):
                prior[c] = prior[c] / tot
            if kind == "multinomial":
                p = [min(max(prior[c], _EPS), 1.0 - _EPS) for c in range(nc)]
                lg = [math.log(v) for v in p]
                s = 0.0
                for v in lg:
                    s += v
                gm = math.exp(s / float(nc))
                return [math.log(p[c] / gm) for c in range(nc)]
            p1 = min(max(prior[1], _EPS), 1.0 - _EPS)
            if kind == "exponential":
                return [0.5 * _logit(p1)]
            return [_logit(p1)]
        if kind == "squared_error":
            ya = np.array(y)
            if has_w:
                return [_avg(y, sw)]
            return [float(np.mean(ya))]
        q = 0.5
        if kind == "quantile":
            q = self.alpha
        if has_w:
            return [_weighted_percentile(y, sw, q * 100.0)]
        return [float(np.percentile(np.array(y), q * 100.0))]

    # -------------------------------------------------------------- fit

    def _update_terminal_regions(self, tree: DecisionTreeRegressor, X, y: List[float],
                                 neg_g: List[float], raw, sw: List[float],
                                 mask: List[bool], k: int):
        t = tree.tree_
        n = len(y)
        leaves = t.apply(X)
        kind = self._loss.kind
        if kind != "squared_error":
            groups = Dict[int, List[int]]()
            for i in range(n):
                if mask[i]:
                    l = int(leaves[i])
                    if l not in groups:
                        groups[l] = List[int]()
                    groups[l].append(i)
            vs = t.value_stride
            for leaf in range(t.node_count):
                if t.left[leaf] != TREE_LEAF:
                    continue
                ind = groups.get(leaf, List[int]())
                if len(ind) == 0:
                    continue
                w = [sw[i] for i in ind]
                upd = 0.0
                if kind == "binomial" or kind == "multinomial":
                    ng = [neg_g[i] for i in ind]
                    prob = [y[i] - neg_g[i] for i in ind]
                    num = _avg(ng, w)
                    if kind == "multinomial":
                        K = float(self._loss.n_classes)
                        num *= (K - 1.0) / K
                    den = _avg([pp * (1.0 - pp) for pp in prob], w)
                    upd = 0.0 if abs(den) < 1e-150 else num / den
                elif kind == "exponential":
                    ng = [neg_g[i] for i in ind]
                    hess = [(-neg_g[i] if y[i] == 0.0 else neg_g[i]) for i in ind]
                    num = _avg(ng, w)
                    den = _avg(hess, w)
                    upd = 0.0 if abs(den) < 1e-150 else num / den
                else:
                    diff = [y[i] - raw[i, k] for i in ind]
                    if kind == "absolute_error":
                        upd = _weighted_percentile(diff, w, 50.0)
                    elif kind == "quantile":
                        upd = _weighted_percentile(diff, w, 100.0 * self._loss.quantile)
                    else:
                        med = _weighted_percentile(diff, w, 50.0)
                        term = List[float](len(diff))
                        for d0 in diff:
                            d = d0 - med
                            sg = 1.0 if d > 0.0 else (-1.0 if d < 0.0 else 0.0)
                            term.append(sg * min(self._loss.delta, abs(d)))
                        upd = med + _avg(term, w)
                t.val[leaf * vs] = upd
        lr = self.learning_rate
        vs = t.value_stride
        for i in range(n):
            raw[i, k] += lr * t.val[int(leaves[i]) * vs]

    def _init_raw_of(self, X32, n: int):
        """raw predictions of the init estimator (_init_raw_predictions)."""
        K = self.n_trees_per_iteration_
        raw = np.empty((n, K))
        if self.init_ == "est":
            pr = _init_predict(self.init, X32, self._is_clf)
            kind = self._loss.kind
            for i in range(n):
                if not self._is_clf:
                    raw[i, 0] = pr[i, 0]
                elif kind == "multinomial":
                    s = 0.0
                    lg = [0.0] * K
                    for k in range(K):
                        lg[k] = math.log(min(max(pr[i, k], _EPS), 1.0 - _EPS))
                    # MultinomialLogit.link: log(p) - log(gmean(p))
                    a = np.array(lg)
                    gm = math.exp(float(np.mean(a)))
                    for k in range(K):
                        raw[i, k] = math.log(min(max(pr[i, k], _EPS), 1.0 - _EPS) / gm)
                else:
                    p1 = min(max(pr[i, 1], _EPS), 1.0 - _EPS)
                    raw[i, 0] = 0.5 * _logit(p1) if kind == "exponential" else _logit(p1)
        else:
            for i in range(n):
                for k in range(K):
                    raw[i, k] = self._init_raw[k]
        return raw

    def _fit_impl(self, X, yv: List[float], sample_weight):
        Xa_all = np.asarray(X, dtype=float)
        n_all = Xa_all.shape[0]
        self.n_features_in_ = Xa_all.shape[1]
        if not self.warm_start:
            self.estimators_ = List[List[T]]()
        has_w = sample_weight is not None
        sw_all = [1.0] * n_all
        if sample_weight is not None:
            swa = np.asarray(sample_weight, dtype=float)
            for i in range(n_all):
                sw_all[i] = float(swa[i])
        Xa = Xa_all
        yt = yv
        sw = sw_all
        Xv = np.empty((0, Xa_all.shape[1]))
        yval = List[float]()
        swv = List[float]()
        if self.n_iter_no_change is not None:
            tr = np.empty(0, dtype=int)
            te = np.empty(0, dtype=int)
            if self._is_clf:
                tr, te = _tts_indices(n_all, self.validation_fraction, None, self.random_state, True, np.array(yv))
            else:
                tr, te = _tts_indices(n_all, self.validation_fraction, None, self.random_state, True, None)
            Xa = Xa_all[tr]
            Xv = Xa_all[te].astype(np.float32).astype(float)
            yt = [yv[int(i)] for i in tr]
            yval = [yv[int(i)] for i in te]
            sw = [sw_all[int(i)] for i in tr]
            swv = [sw_all[int(i)] for i in te]
        Xf, n, p = _to_f32_colmajor(Xa)
        X32 = Xa.astype(np.float32).astype(float)
        probe = DecisionTreeRegressor()
        _copy_tree_params(probe, self)
        self.max_features_ = probe._resolve_max_features(p)
        K = self.n_trees_per_iteration_
        begin = 0
        raw = np.empty((0, 0))
        if len(self.estimators_) == 0:
            self.train_score_ = np.zeros(self.n_estimators)
            if self.subsample < 1.0:
                self.oob_improvement_ = np.zeros(self.n_estimators)
                self.oob_scores_ = np.zeros(self.n_estimators)
                self.oob_score_ = math.nan
            if self.init_ == "est":
                yi = np.array(yt)
                if self._is_clf:
                    yi_int = np.array([int(v) for v in yt])
                    _init_fit(self.init, X32, yi_int, np.array(sw), has_w)
                else:
                    _init_fit(self.init, X32, yi, np.array(sw), has_w)
            else:
                self._init_raw = self._compute_init(yt, sw, has_w)
            raw = self._init_raw_of(X32, n)
            self._rng = _check_rs(self.random_state)
        else:
            if self.n_estimators < len(self.estimators_):
                raise ValueError(f"n_estimators={self.n_estimators} must be larger or equal to estimators_.shape[0]={len(self.estimators_)} when warm_start==True")
            begin = len(self.estimators_)
            raw = self._raw_predict(X32)
            self.train_score_ = np.resize(self.train_score_, self.n_estimators)
            if self.subsample < 1.0 or self.oob_improvement_.shape[0] > 0:
                if self.oob_improvement_.shape[0] > 0:
                    self.oob_improvement_ = np.resize(self.oob_improvement_, self.n_estimators)
                    self.oob_scores_ = np.resize(self.oob_scores_, self.n_estimators)
                else:
                    self.oob_improvement_ = np.zeros(self.n_estimators)
                    self.oob_scores_ = np.zeros(self.n_estimators)
                self.oob_score_ = math.nan
        n_stages = self._fit_stages(Xf, X32, n, p, yt, sw, raw, Xv, yval, swv, begin)
        if n_stages != len(self.estimators_):
            self.estimators_ = self.estimators_[:n_stages]
        if n_stages != self.train_score_.shape[0]:
            self.train_score_ = self.train_score_[:n_stages].copy()
            if self.oob_improvement_.shape[0] > 0:
                self.oob_improvement_ = self.oob_improvement_[:n_stages].copy()
                self.oob_scores_ = self.oob_scores_[:n_stages].copy()
                self.oob_score_ = self.oob_scores_[n_stages - 1]
        self.n_estimators_ = n_stages
        self._fitted = True

    def _fit_stages(self, Xf: List[float32], X32, n: int, p: int, yv: List[float],
                    sw: List[float], raw, Xv, yval: List[float], swv: List[float],
                    begin: int) -> int:
        rng = self._rng
        do_oob = self.subsample < 1.0
        mask = [True] * n
        n_inbag = max(1, int(self.subsample * float(n)))
        loss = self._loss
        factor = loss.factor()
        K = self.n_trees_per_iteration_
        all_rows = list(range(n))
        initial_loss = 0.0
        es = self.n_iter_no_change is not None
        nv = len(yval)
        loss_history = List[float]()
        raw_val = np.empty((0, 0))
        val_rows = list(range(nv))
        val_stage = 0
        if es:
            loss_history = [math.inf] * self.n_iter_no_change
            raw_val = self._init_raw_of(Xv, nv)
        it = begin
        while it < self.n_estimators:
            in_rows = all_rows
            oob_rows = List[int]()
            if do_oob:
                mask = _random_sample_mask(n, n_inbag, rng)
                in_rows = [i for i in range(n) if mask[i]]
                oob_rows = [i for i in range(n) if not mask[i]]
                if it == 0:
                    initial_loss = factor * loss(yv, raw, oob_rows, sw)
            # _fit_stage
            if loss.kind == "huber":
                abserr = [abs(yv[i] - raw[i, 0]) for i in range(n)]
                loss.delta = _weighted_percentile(abserr, sw, 100.0 * loss.quantile)
            ngm = np.empty((0, 0))
            if loss.kind == "multinomial":
                ngm = loss.neg_gradient_multinomial(yv, raw)
            stage = List[T]()
            tw = list(sw)
            if do_oob:
                for i in range(n):
                    tw[i] = sw[i] * (1.0 if mask[i] else 0.0)
            for k in range(K):
                ng = List[float]()
                yk = yv
                if loss.kind == "multinomial":
                    ng = [ngm[i, k] for i in range(n)]
                    yk = [1.0 if int(yv[i]) == k else 0.0 for i in range(n)]
                else:
                    ng = loss.neg_gradient(yv, raw, 0)
                tree = DecisionTreeRegressor()
                _copy_tree_params(tree, self)
                tree._fit_encoded(Xf, n, p, ng, 1, tw, True, rng, List[bool]())
                self._update_terminal_regions(tree, X32, yk, ng, raw, tw, mask, k)
                stage.append(tree)
            self.estimators_.append(stage)
            if do_oob:
                self.train_score_[it] = factor * loss(yv, raw, in_rows, sw)
                self.oob_scores_[it] = factor * loss(yv, raw, oob_rows, sw)
                prev = initial_loss if it == 0 else self.oob_scores_[it - 1]
                self.oob_improvement_[it] = prev - self.oob_scores_[it]
                self.oob_score_ = self.oob_scores_[self.oob_scores_.shape[0] - 1]
            else:
                self.train_score_[it] = factor * loss(yv, raw, all_rows, sw)
            if es:
                # sklearn's staged validation iterator starts at stage 0 even
                # when warm-starting (it is created lazily at begin_at_stage)
                self._predict_stage(val_stage, Xv, raw_val)
                val_stage += 1
                vl = factor * loss(yval, raw_val, val_rows, swv)
                improved = False
                for v in loss_history:
                    if vl + self.tol < v:
                        improved = True
                if improved:
                    loss_history[it % len(loss_history)] = vl
                else:
                    break
            it += 1
        if it == self.n_estimators:
            return it
        return it + 1

    # ---------------------------------------------------------- predict

    def _raw_predict_init(self, X):
        return self._init_raw_of(X.astype(np.float32).astype(float), X.shape[0])

    def _check_X(self, X):
        if not self._fitted:
            raise ValueError("This estimator is not fitted yet. Call 'fit' first.")
        Xa = np.asarray(X, dtype=float)
        if Xa.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xa.shape[1]} features, but the model is expecting {self.n_features_in_} features as input.")
        return Xa

    def _predict_stage(self, i: int, X, raw):
        lr = self.learning_rate
        n = X.shape[0]
        for k in range(self.n_trees_per_iteration_):
            t = self.estimators_[i][k].tree_
            vs = t.value_stride
            for r in range(n):
                raw[r, k] += lr * t.val[t._leaf_of(X, r) * vs]

    def _raw_predict(self, X):
        Xa = self._check_X(X)
        raw = self._raw_predict_init(Xa)
        for i in range(len(self.estimators_)):
            self._predict_stage(i, Xa, raw)
        return raw

    def _staged_raw_predict(self, X):
        Xa = self._check_X(X)
        raw = self._raw_predict_init(Xa)
        for i in range(len(self.estimators_)):
            self._predict_stage(i, Xa, raw)
            yield raw.copy()

    @property
    def feature_importances_(self):
        rows = List[np.ndarray[float, 1]]()
        for stage in self.estimators_:
            for t in stage:
                if t.tree_.node_count > 1:
                    rows.append(t.tree_.compute_feature_importances(False))
        p = self.n_features_in_
        if len(rows) == 0:
            return np.zeros(p)
        m = _mean_rows(rows, p)
        return m / np.sum(m)

    def _apply3(self, X):
        Xa = self._check_X(X)
        n = Xa.shape[0]
        K = self.n_trees_per_iteration_
        m = len(self.estimators_)
        out = np.empty((n, m, K), dtype=int)
        for i in range(m):
            for k in range(K):
                lv = self.estimators_[i][k].tree_.apply(Xa)
                for r in range(n):
                    out[r, i, k] = lv[r]
        return out

class GradientBoostingClassifier[I](_BaseGB[DecisionTreeRegressor, I]):
    @property
    def _estimator_type(self) -> str:
        return "classifier"

    def __init__(self, *, loss: str = "log_loss", learning_rate: float = 0.1,
                 n_estimators: int = 100, subsample: float = 1.0, criterion: str = "deprecated",
                 min_samples_split = 2, min_samples_leaf = 1,
                 min_weight_fraction_leaf: float = 0.0, max_depth: Optional[int] = 3,
                 min_impurity_decrease: float = 0.0, init: I = None,
                 random_state: Optional[int] = None, max_features = None, verbose: int = 0,
                 max_leaf_nodes: Optional[int] = None, warm_start: bool = False,
                 validation_fraction: float = 0.1, n_iter_no_change: Optional[int] = None,
                 tol: float = 1e-4, ccp_alpha: float = 0.0):
        _BaseGB[DecisionTreeRegressor, I].__init__(self, True, loss, learning_rate, n_estimators, subsample, min_samples_split,
                      min_samples_leaf, min_weight_fraction_leaf, max_depth,
                      min_impurity_decrease, init, random_state, max_features, 0.9, verbose,
                      max_leaf_nodes, n_iter_no_change, ccp_alpha, warm_start,
                      validation_fraction, tol)

    def fit(self, X, y, sample_weight = None, monitor = None):
        classes, enc = _encode_classes(np.asarray(y))
        self.classes_ = classes
        nc = classes.shape[0]
        if nc < 2:
            raise ValueError("y contains 1 class, while a minimum of 2 classes are required.")
        self.n_classes_ = nc
        self.n_trees_per_iteration_ = 1 if nc <= 2 else nc
        if self.loss == "log_loss":
            self._loss = _Loss("binomial" if nc == 2 else "multinomial", 0.5, nc)
        elif self.loss == "exponential":
            if nc > 2:
                raise ValueError(f"loss='exponential' is only suitable for a binary classification problem, you have n_classes={nc}. Please use loss='log_loss' instead.")
            self._loss = _Loss("exponential", 0.5, nc)
        else:
            raise ValueError(f"unknown loss '{self.loss}'")
        self._fit_impl(X, enc, sample_weight)
        return self

    def decision_function(self, X):
        """(n, n_trees_per_iteration_): binary problems give (n, 1) (sklearn
        ravels to (n,); the static return rank cannot depend on n_classes).
        decision_function_1d returns the (n,) binary form."""
        return self._raw_predict(X)

    def decision_function_1d(self, X):
        raw = self._raw_predict(X)
        if raw.shape[1] != 1:
            raise ValueError("decision_function_1d is only defined for binary problems")
        return raw.ravel()

    def _labels(self, raw):
        n = raw.shape[0]
        out = np.empty(n, dtype=int)
        for i in range(n):
            if raw.shape[1] == 1:
                out[i] = self.classes_[1 if raw[i, 0] >= 0.0 else 0]
            else:
                best = 0
                for k in range(1, raw.shape[1]):
                    if raw[i, k] > raw[i, best]:
                        best = k
                out[i] = self.classes_[best]
        return out

    def predict(self, X):
        return self._labels(self._raw_predict(X))

    def apply(self, X):
        return self._apply3(X)

    def predict_proba(self, X):
        return self._loss.predict_proba(self._raw_predict(X))

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def staged_decision_function(self, X):
        for raw in self._staged_raw_predict(X):
            yield raw

    def staged_predict(self, X):
        for raw in self._staged_raw_predict(X):
            yield self._labels(raw)

    def staged_predict_proba(self, X):
        for raw in self._staged_raw_predict(X):
            yield self._loss.predict_proba(raw)

    def score(self, X, y, sample_weight = None) -> float:
        from ..base import _accuracy
        return _accuracy(y, self.predict(X))

class GradientBoostingRegressor[I](_BaseGB[DecisionTreeRegressor, I]):
    @property
    def _estimator_type(self) -> str:
        return "regressor"

    def __init__(self, *, loss: str = "squared_error", learning_rate: float = 0.1,
                 n_estimators: int = 100, subsample: float = 1.0, criterion: str = "deprecated",
                 min_samples_split = 2, min_samples_leaf = 1,
                 min_weight_fraction_leaf: float = 0.0, max_depth: Optional[int] = 3,
                 min_impurity_decrease: float = 0.0, init: I = None,
                 random_state: Optional[int] = None, max_features = None, alpha: float = 0.9,
                 verbose: int = 0, max_leaf_nodes: Optional[int] = None,
                 warm_start: bool = False, validation_fraction: float = 0.1,
                 n_iter_no_change: Optional[int] = None, tol: float = 1e-4,
                 ccp_alpha: float = 0.0):
        _BaseGB[DecisionTreeRegressor, I].__init__(self, False, loss, learning_rate, n_estimators, subsample, min_samples_split,
                      min_samples_leaf, min_weight_fraction_leaf, max_depth,
                      min_impurity_decrease, init, random_state, max_features, alpha, verbose,
                      max_leaf_nodes, n_iter_no_change, ccp_alpha, warm_start,
                      validation_fraction, tol)

    def fit(self, X, y, sample_weight = None, monitor = None):
        ya = np.asarray(y, dtype=float)
        yv = [float(ya[i]) for i in range(ya.shape[0])]
        self.n_trees_per_iteration_ = 1
        if self.loss != "squared_error" and self.loss != "absolute_error" and self.loss != "huber" and self.loss != "quantile":
            raise ValueError(f"unknown loss '{self.loss}'")
        self._loss = _Loss(self.loss, self.alpha, 1)
        self._fit_impl(X, yv, sample_weight)
        return self

    def predict(self, X):
        return self._raw_predict(X).ravel()

    def staged_predict(self, X):
        for raw in self._staged_raw_predict(X):
            yield raw.ravel()

    def score(self, X, y, sample_weight = None) -> float:
        return _r2(y, self.predict(X))

    def apply(self, X):
        a = self._apply3(X)
        return a.reshape((a.shape[0], a.shape[1]))
