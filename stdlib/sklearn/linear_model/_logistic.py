# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)
#
# LogisticRegression / LogisticRegressionCV with all solvers:
#   lbfgs (scipy L-BFGS-B on LinearModelLoss), newton-cg, newton-cholesky,
#   liblinear (sklearn's bundled liblinear: TRON / dual CD / newGLMNET),
#   sag, saga (sag_fast port). Penalties via l1_ratio / C as in 1.9.1
#   (penalty='deprecated' default; explicit penalty still accepted).
# class_weight: None, 'balanced' or dict {label: weight}. Dense X only.
# LogisticRegressionCV: Cs int or list, cv int or splitter, scoring None
# (accuracy), refit True/False, use_legacy_attributes=False layout.

from ..base import estimator_repr as _estimator_repr
import math
import numpy as np
from ._base import _check_sw, _expit
from ._linear_loss import LinearModelLoss, lbfgs_fit, BINOMIAL, MULTINOMIAL
from ._newton import newton_cg, NewtonCholeskySolver
from ..utils import _LegacyMT
from ..base import BaseEstimator

def _check_solver(solver: str, penalty: str, dual: bool) -> str:
    if solver not in ("lbfgs", "newton-cg", "newton-cholesky", "liblinear", "sag", "saga"):
        raise ValueError(f"The 'solver' parameter of LogisticRegression must be a str among {{'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'}}. Got {solver!r} instead.")
    if solver != "liblinear" and solver != "saga" and penalty != "l2" and penalty != "none":
        raise ValueError(f"Solver {solver} supports only 'l2' or None penalties, got {penalty} penalty.")
    if solver != "liblinear" and dual:
        raise ValueError(f"Solver {solver} supports only dual=False, got dual={dual}")
    if penalty == "elasticnet" and solver != "saga":
        raise ValueError(f"Only 'saga' solver supports elasticnet penalty, got solver={solver}.")
    if solver == "liblinear" and penalty == "none":
        raise ValueError("C=np.inf as well as penalty=None is not supported for the liblinear solver")
    return solver

def _class_weight_arr(class_weight, classes, yi: np.ndarray[int, 1], sw: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
    """compute_class_weight(class_weight, classes, y, sample_weight)."""
    K = classes.shape[0]
    cw = np.ones(K)
    if class_weight is None:
        pass
    elif isinstance(class_weight, str):
        if class_weight != "balanced":
            raise ValueError("class_weight must be dict, 'balanced', or None")
        cnt = np.zeros(K)
        for i in range(yi.shape[0]):
            cnt[yi[i]] += sw[i]
        tot = float(np.sum(cnt))
        for k in range(K):
            cw[k] = tot / (float(K) * cnt[k])
    else:
        for k in range(K):
            key = classes[k]
            if key in class_weight:
                cw[k] = float(class_weight[key])
    return cw

def _resolve_penalty(penalty: str, l1_ratio: Optional[float], C: float) -> str:
    if penalty == "deprecated":
        if l1_ratio is None or l1_ratio.__val__() == 0.0:
            p = "l2"
        elif l1_ratio.__val__() == 1.0:
            p = "l1"
        else:
            p = "elasticnet"
        if C == math.inf:
            p = "none"
        return p
    if penalty == "None" or penalty == "none":
        return "none"
    return penalty

class _ImaxStream:
    """A RandomState shared across a path: randint(np.iinfo('i').max) for
    liblinear, randint(1, 2**31 - 1) for the sag dataset seed."""
    mt: _LegacyMT
    has: bool
    def __init__(self, random_state: Optional[int]):
        self.has = random_state is not None
        self.mt = _LegacyMT(random_state.__val__() if random_state is not None else 0)
    def _interval(self, mx: int) -> int:
        rng = u32(mx)
        mask = rng
        mask |= mask >> u32(1); mask |= mask >> u32(2); mask |= mask >> u32(4)
        mask |= mask >> u32(8); mask |= mask >> u32(16)
        v = self.mt.genrand_int32() & mask
        while v > rng:
            v = self.mt.genrand_int32() & mask
        return int(v)
    def next(self) -> int:
        if not self.has:
            return int(np.random.randint(0, 2147483647))
        return self._interval(0x7fffffff - 1)
    def sag_seed(self) -> int:
        if not self.has:
            return int(np.random.randint(1, 2147483647))
        return 1 + self._interval(2147483647 - 1 - 1)

class _PathResult:
    coefs: List[np.ndarray[float, 1]]   # per C: flat (binary: p[+1]; multi: (K, p[+1]) C-order)
    n_iter: List[int]
    def __init__(self):
        self.coefs = List[np.ndarray[float, 1]]()
        self.n_iter = List[int]()

def _lr_path(X: np.ndarray[float, 2], yi: np.ndarray[int, 1], K: int, Cs: List[float],
             fit_intercept: bool, max_iter: int, tol: float, solver: str, coef0: Optional[np.ndarray[float, 2]],
             cw: np.ndarray[float, 1], has_cw: bool, dual: bool, penalty: str, intercept_scaling: float,
             random_state: Optional[int], sw_in: np.ndarray[float, 1], has_sw: bool,
             l1_ratio: float, max_squared_sum: float = -1.0) -> _PathResult:
    """_logistic_regression_path for one (train) set. coef0 shape (nb, p[+1])."""
    n, p = X.shape
    binary = K == 2
    nd = p + (1 if fit_intercept else 0)
    sw = sw_in.copy()
    if has_cw:
        for i in range(n):
            sw[i] *= cw[yi[i]]
    res = _PathResult()
    if solver == "liblinear":
        if not binary:
            raise ValueError("The 'liblinear' solver does not support multiclass classification (n_classes >= 3). Either use another solver or wrap the estimator in a OneVsRestClassifier to keep applying a one-versus-rest scheme.")
        # path: class weights are folded into sample_weight, random_state is a
        # RandomState shared across Cs
        rs = _ImaxStream(random_state)
        for C in Cs:
            coef, inter, nit = _liblinear_fit(X, yi, C, fit_intercept, intercept_scaling, np.ones(2),
                                              penalty, dual, max_iter, tol, rs.next(), sw)
            w = np.empty(nd)
            for j in range(p):
                w[j] = coef[0, j]
            if fit_intercept:
                w[p] = inter[0]
            res.coefs.append(w)
            res.n_iter.append(int(nit[0]))
        return res
    if solver == "sag" or solver == "saga":
        from ._sag import sag_solver
        nc = 1 if binary else K
        ci = np.zeros((p, nc))
        ii = np.zeros(nc)
        if coef0 is not None:
            c0 = coef0.__val__()
            for k in range(nc):
                for j in range(p):
                    ci[j, k] = c0[k, j]
                if fit_intercept:
                    ii[k] = c0[k, p]
        yt = np.empty(n)
        for i in range(n):
            yt[i] = float(yi[i]) if not binary else (1.0 if yi[i] == 1 else 0.0)
        mss = max_squared_sum
        if mss < 0.0:
            mss = 0.0
            for i in range(n):
                s = 0.0
                for j in range(p):
                    s += X[i, j] * X[i, j]
                mss = max(mss, s)
        from ._sag import SagMemory
        mem = SagMemory()
        rs = _ImaxStream(random_state)
        for C in Cs:
            if penalty == "l1":
                alpha = 0.0
                beta = 1.0 / C
            elif penalty == "l2" or penalty == "none":
                alpha = 1.0 / C
                beta = 0.0
            else:
                alpha = (1.0 / C) * (1.0 - l1_ratio)
                beta = (1.0 / C) * l1_ratio
            nit = sag_solver(X, yt, sw, "log" if binary else "multinomial", alpha, beta, max_iter,
                             tol, random_state, ci, ii, fit_intercept, solver == "saga", mss,
                             mem, rs.sag_seed())
            w = np.empty(nc * nd)
            for k in range(nc):
                for j in range(p):
                    w[k * nd + j] = ci[j, k]
                if fit_intercept:
                    w[k * nd + p] = ii[k]
            res.coefs.append(w)
            res.n_iter.append(nit)
        return res
    # lbfgs / newton-cg / newton-cholesky on LinearModelLoss
    if binary:
        yt = np.empty(n)
        for i in range(n):
            yt[i] = 1.0 if yi[i] == 1 else 0.0
        lm = LinearModelLoss(X, yt, sw, BINOMIAL, 0.0, 2, fit_intercept)
        w0 = np.zeros(nd)
        if coef0 is not None:
            c0 = coef0.__val__()
            for j in range(nd):
                w0[j] = c0[0, j]
    else:
        yt = np.empty(n)
        for i in range(n):
            yt[i] = float(yi[i])
        lm = LinearModelLoss(X, yt, sw, MULTINOMIAL, 0.0, K, fit_intercept)
        w0 = np.zeros(K * nd)
        if coef0 is not None:
            c0 = coef0.__val__()
            for k in range(K):
                for j in range(nd):
                    w0[j * K + k] = c0[k, j]
    sw_sum = float(np.sum(sw)) if (has_sw or has_cw) else float(n)
    for C in Cs:
        l2 = 1.0 / (C * sw_sum)
        nit = 0
        if solver == "lbfgs":
            w0, nit, st = lbfgs_fit(lm, w0, l2, max_iter, tol)
        elif solver == "newton-cg":
            w0, nit = newton_cg(lm, w0, l2, tol, max_iter)
        else:
            sol = NewtonCholeskySolver(lm, w0, l2, tol, max_iter)
            w0 = sol.solve()
            nit = sol.iteration
        if binary:
            res.coefs.append(w0.copy())
        else:
            w = np.empty(K * nd)
            for k in range(K):
                for j in range(nd):
                    w[k * nd + j] = w0[j * K + k]
            res.coefs.append(w)
        res.n_iter.append(nit)
    return res

def _liblinear_fit(X: np.ndarray[float, 2], yi: np.ndarray[int, 1], C: float, fit_intercept: bool,
                   intercept_scaling: float, cw: np.ndarray[float, 1], penalty: str, dual: bool,
                   max_iter: int, tol: float, seed: int, sw: np.ndarray[float, 1]):
    """_fit_liblinear for logistic regression (y already label-encoded)."""
    from ..svm._liblinear import train_wrap, solver_type
    if fit_intercept and intercept_scaling <= 0.0:
        raise ValueError(f"Intercept scaling is {intercept_scaling} but needs to be greater than 0. To disable fitting an intercept, set fit_intercept=False.")
    stype = solver_type("ovr", penalty, "logistic_regression", dual)
    bias = intercept_scaling if fit_intercept else -1.0
    n = yi.shape[0]
    yl = np.empty(n)
    for i in range(n):
        yl[i] = float(yi[i])
    raw, nit = train_wrap(X, yl, stype, tol, bias, C, cw, max_iter, seed, 0.1, sw)
    p = X.shape[1]
    coef = np.empty((raw.shape[0], p))
    inter = np.zeros(raw.shape[0])
    for k in range(raw.shape[0]):
        for j in range(p):
            coef[k, j] = raw[k, j]
        if fit_intercept:
            inter[k] = intercept_scaling * raw[k, p]
    return coef, inter, np.asarray(nit, dtype=int)

def _softmax_rows(s: np.ndarray[float, 2]) -> np.ndarray[float, 2]:
    # scipy.special.softmax(x, axis=1)
    m = np.max(s, axis=1, keepdims=True)
    E = np.exp(s - m)
    return E / np.sum(E, axis=1, keepdims=True)

class LogisticRegression[CW = NoneType](BaseEstimator):
    def __repr__(self) -> str:
        return _estimator_repr(self)

    penalty: str
    C: float
    l1_ratio: Optional[float]
    dual: bool
    tol: float
    fit_intercept: bool
    intercept_scaling: float
    class_weight: CW
    random_state: Optional[int]
    solver: str
    max_iter: int
    verbose: int
    warm_start: bool
    n_jobs: Optional[int]
    classes_: np.ndarray[int, 1]
    coef_: np.ndarray[float, 2]
    intercept_: np.ndarray[float, 1]
    n_iter_: np.ndarray[int, 1]
    n_features_in_: int
    _fitted: bool
    _estimator_type: Literal[str] = "classifier"

    def __init__(self, penalty: Optional[str] = "deprecated", *, C: float = 1.0,
                 l1_ratio: Optional[float] = 0.0, dual: bool = False, tol: float = 1e-4,
                 fit_intercept: bool = True, intercept_scaling: float = 1.0,
                 class_weight: CW = None, random_state: Optional[int] = None,
                 solver: str = "lbfgs", max_iter: int = 100, verbose: int = 0,
                 warm_start: bool = False, n_jobs: Optional[int] = None):
        self.penalty = "none" if penalty is None else penalty.__val__()
        self.C = C
        self.l1_ratio = l1_ratio
        self.dual = dual
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.classes_ = np.empty(0, dtype=int)
        self.coef_ = np.empty((0, 0))
        self.intercept_ = np.empty(0)
        self.n_iter_ = np.zeros(1, dtype=int)
        self.n_features_in_ = 0
        self._fitted = False

    def fit(self, X, y, sample_weight = None):
        penalty = _resolve_penalty(self.penalty, self.l1_ratio, self.C)
        solver = _check_solver(self.solver, penalty, self.dual)
        if penalty == "elasticnet" and self.l1_ratio is None:
            raise ValueError("l1_ratio must be specified when penalty is elasticnet.")
        C_ = self.C
        if self.penalty == "none":
            C_ = math.inf
            penalty = "l2"
        elif penalty == "none":
            penalty = "l2"
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        ya = np.asarray(y)
        n, p = Xa.shape
        classes, yi = np.unique(ya, return_inverse=True)
        K = classes.shape[0]
        self.classes_ = classes
        self.n_features_in_ = p
        has_sw = sample_weight is not None
        sw = _check_sw(sample_weight, n)
        has_cw = self.class_weight is not None
        if solver == "liblinear":
            if K > 2:
                raise ValueError("The 'liblinear' solver does not support multiclass classification (n_classes >= 3). Either use another solver or wrap the estimator in a OneVsRestClassifier to keep applying a one-versus-rest scheme.")
            if K < 2:
                raise ValueError(f"This solver needs samples of at least 2 classes in the data, but the data contains only one class: {classes[0]}")
            cw = _class_weight_arr(self.class_weight, classes, yi, sw)
            rs = _ImaxStream(self.random_state)
            coef, inter, nit = _liblinear_fit(Xa, yi, self.C, self.fit_intercept, self.intercept_scaling,
                                              cw, penalty, self.dual, self.max_iter, self.tol, rs.next(), sw)
            self.coef_ = coef
            self.intercept_ = inter
            self.n_iter_ = nit
            self._fitted = True
            return self
        if K < 2:
            raise ValueError(f"This solver needs samples of at least 2 classes in the data, but the data contains only one class: {classes[0]}")
        cw = _class_weight_arr(self.class_weight, classes, yi, sw)
        nd = p + (1 if self.fit_intercept else 0)
        coef0: Optional[np.ndarray[float, 2]] = None
        if self.warm_start and self._fitted:
            nb = self.coef_.shape[0]
            c0 = np.zeros((nb, nd))
            for k in range(nb):
                for j in range(p):
                    c0[k, j] = self.coef_[k, j]
                if self.fit_intercept:
                    c0[k, p] = self.intercept_[k]
            coef0 = c0
        l1r = 0.0 if self.l1_ratio is None else self.l1_ratio.__val__()
        Cs = [C_]
        res = _lr_path(Xa, yi, K, Cs, self.fit_intercept, self.max_iter, self.tol, solver, coef0,
                       cw, has_cw, self.dual, penalty, self.intercept_scaling, self.random_state,
                       sw, has_sw, l1r)
        w = res.coefs[0]
        nb = 1 if K == 2 else K
        coef = np.empty((nb, p))
        inter = np.zeros(nb)
        for k in range(nb):
            for j in range(p):
                coef[k, j] = w[k * nd + j]
            if self.fit_intercept:
                inter[k] = w[k * nd + p]
        self.coef_ = coef
        self.intercept_ = inter
        self.n_iter_ = np.array([res.n_iter[0]])
        self._fitted = True
        return self

    def _scores(self, X) -> np.ndarray[float, 2]:
        Xa = np.asarray(X, dtype=float)
        return Xa @ self.coef_.T + self.intercept_

    def decision_function(self, X):
        # DEVIATION: always 2-d (binary: (n, 1)); see decision_function_1d
        return self._scores(X)

    def decision_function_1d(self, X):
        """Binary decision function with sklearn's (n,) shape."""
        s = self._scores(X)
        if s.shape[1] != 1:
            raise ValueError("decision_function_1d is only defined for binary problems")
        return np.ascontiguousarray(s[:, 0])

    def predict(self, X):
        s = self._scores(X)
        n = s.shape[0]
        idx = np.zeros(n, dtype=int)
        if s.shape[1] == 1:
            for i in range(n):
                idx[i] = 1 if s[i, 0] > 0.0 else 0
        else:
            idx = np.argmax(s, axis=1)
        return self.classes_[idx]

    def predict_proba(self, X):
        s = self._scores(X)
        n = s.shape[0]
        if s.shape[1] == 1:
            pr = np.empty((n, 2))
            for i in range(n):
                e = _expit(s[i, 0])
                pr[i, 1] = e
                pr[i, 0] = 1.0 - e
            return pr
        return _softmax_rows(s)

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def score(self, X, y, sample_weight = None) -> float:
        ya = np.asarray(y)
        pr = self.predict(X)
        n = ya.shape[0]
        sw = _check_sw(sample_weight, n)
        num = 0.0
        den = 0.0
        for i in range(n):
            den += sw[i]
            if ya[i] == pr[i]:
                num += sw[i]
        return num / den

    def densify(self):
        return self

    def sparsify(self):
        return self

class LogisticRegressionCV[CW = NoneType, CV = NoneType](BaseEstimator):
    def __repr__(self) -> str:
        return _estimator_repr(self)

    """Legacy attribute layout (use_legacy_attributes default in 1.9.1):
    C_, l1_ratio_ arrays (n_classes or 1,), scores_/coefs_paths_ dicts keyed by
    class (positive class only when binary), n_iter_ (1, n_folds, n_Cs[, n_l1])."""
    Cs: List[float]
    n_Cs: int
    l1_ratios: Optional[List[float]]
    fit_intercept: bool
    cv: CV
    dual: bool
    penalty: str
    tol: float
    max_iter: int
    class_weight: CW
    solver: str
    refit: bool
    intercept_scaling: float
    random_state: Optional[int]
    classes_: np.ndarray[int, 1]
    coef_: np.ndarray[float, 2]
    intercept_: np.ndarray[float, 1]
    Cs_: np.ndarray[float, 1]
    C_: np.ndarray[float, 1]
    l1_ratio_: np.ndarray[float, 1]
    l1_ratios_: np.ndarray[float, 1]
    scores_: Dict[int, np.ndarray[float, 2]]
    coefs_paths_: Dict[int, np.ndarray[float, 3]]
    _scores3: np.ndarray[float, 3]
    _paths: np.ndarray[float, 5]
    n_iter_: np.ndarray[int, 3]
    n_features_in_: int
    _estimator_type: Literal[str] = "classifier"

    def __init__(self, *, Cs = 10, l1_ratios: Optional[List[float]] = None, fit_intercept: bool = True,
                 cv: CV = None, dual: bool = False, penalty: str = "deprecated", scoring = None,
                 solver: str = "lbfgs", tol: float = 1e-4, max_iter: int = 100,
                 class_weight: CW = None, n_jobs = None, verbose: int = 0, refit: bool = True,
                 intercept_scaling: float = 1.0, random_state: Optional[int] = None,
                 use_legacy_attributes = True):
        if isinstance(Cs, int):
            self.n_Cs = Cs
            self.Cs = List[float]()
        else:
            self.n_Cs = 0
            self.Cs = [float(c) for c in Cs]
        self.l1_ratios = l1_ratios
        self.fit_intercept = fit_intercept
        self.cv = cv
        self.dual = dual
        self.penalty = penalty
        self.solver = solver
        self.tol = tol
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.refit = refit
        self.intercept_scaling = intercept_scaling
        self.random_state = random_state
        self.classes_ = np.empty(0, dtype=int)
        self.coef_ = np.empty((0, 0))
        self.intercept_ = np.empty(0)
        self.Cs_ = np.empty(0)
        self.C_ = np.empty(0)
        self.l1_ratio_ = np.empty(0)
        self.l1_ratios_ = np.empty(0)
        self.scores_ = Dict[int, np.ndarray[float, 2]]()
        self.coefs_paths_ = Dict[int, np.ndarray[float, 3]]()
        self._scores3 = np.empty((0, 0, 0))
        self._paths = np.empty((0, 0, 0, 0, 0))
        self.n_iter_ = np.zeros((0, 0, 0), dtype=int)
        self.n_features_in_ = 0

    def _folds(self, X, y):
        from ..model_selection import StratifiedKFold
        folds = List[Tuple[np.ndarray[int, 1], np.ndarray[int, 1]]]()
        if self.cv is None:
            for tr, te in StratifiedKFold(5).split(X, y):
                folds.append((tr, te))
        elif isinstance(self.cv, int):
            for tr, te in StratifiedKFold(self.cv).split(X, y):
                folds.append((tr, te))
        else:
            for tr, te in self.cv.split(X, y):
                folds.append((tr, te))
        return folds

    def fit(self, X, y, sample_weight = None):
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        ya = np.asarray(y)
        n, p = Xa.shape
        l1s = List[float]()
        if self.l1_ratios is not None:
            for v in self.l1_ratios.__val__():
                l1s.append(float(v))
        if self.penalty == "deprecated":
            if len(l1s) == 0:
                penalty = "l2"
            else:
                all0 = True
                all1 = True
                for v in l1s:
                    if v != 0.0:
                        all0 = False
                    if v != 1.0:
                        all1 = False
                penalty = "l2" if all0 else ("l1" if all1 else "elasticnet")
        else:
            penalty = self.penalty
        solver = _check_solver(self.solver, penalty, self.dual)
        no_l1 = len(l1s) == 0
        if penalty == "elasticnet":
            for v in l1s:
                if v < 0.0 or v > 1.0:
                    raise ValueError(f"l1_ratios must be an array-like of numbers between 0 and 1; got (l1_ratios={l1s})")
        if no_l1:
            l1s.append(0.0)
        classes, yi = np.unique(ya, return_inverse=True)
        K = classes.shape[0]
        if K < 2:
            raise ValueError(f"This solver needs samples of at least 2 classes in the data, but the data contains only one class: {classes[0]}.")
        self.classes_ = classes
        self.n_features_in_ = p
        binary = K == 2
        has_sw = sample_weight is not None
        sw = _check_sw(sample_weight, n)
        has_cw = self.class_weight is not None
        cw = _class_weight_arr(self.class_weight, classes, yi, sw)
        if self.n_Cs > 0:
            Csa = np.logspace(-4.0, 4.0, self.n_Cs)
        else:
            Csa = np.array(self.Cs)
        Cs = List[float]()
        for c in Csa:
            Cs.append(float(c))
        nC = len(Cs)
        nL = len(l1s)
        folds = self._folds(Xa, ya)
        mss = 0.0
        for i in range(n):
            s = 0.0
            for j in range(p):
                s += Xa[i, j] * Xa[i, j]
            mss = max(mss, s)
        nF = len(folds)
        nd = p + (1 if self.fit_intercept else 0)
        nb = 1 if binary else K
        paths = np.zeros((nb, nF, nC, nL, nd))
        scores = np.zeros((nF, nC, nL))
        niter = np.zeros((nF, nC, nL), dtype=int)
        for fi in range(nF):
            tr, te = folds[fi]
            Xtr = np.ascontiguousarray(Xa[tr])
            Xte = Xa[te]
            ytr = yi[tr]
            yte = yi[te]
            swtr = sw[tr]
            swte = sw[te]
            for li in range(nL):
                res = _lr_path(Xtr, ytr, K, Cs, self.fit_intercept, self.max_iter, self.tol, solver,
                               None, cw, has_cw, self.dual, penalty, self.intercept_scaling,
                               self.random_state, swtr, has_sw, l1s[li], mss)
                for ci in range(nC):
                    w = res.coefs[ci]
                    niter[fi, ci, li] = res.n_iter[ci]
                    for k in range(nb):
                        for j in range(nd):
                            paths[k, fi, ci, li, j] = w[k * nd + j]
                    # LogisticRegression.score: weighted accuracy on the test fold
                    S = np.zeros((Xte.shape[0], nb))
                    for k in range(nb):
                        for i in range(Xte.shape[0]):
                            s = w[k * nd + p] if self.fit_intercept else 0.0
                            acc = 0.0
                            for j in range(p):
                                acc += Xte[i, j] * w[k * nd + j]
                            S[i, k] = acc + s
                    num = 0.0
                    den = 0.0
                    for i in range(Xte.shape[0]):
                        if binary:
                            pr = 1 if S[i, 0] > 0.0 else 0
                        else:
                            pr = 0
                            for k in range(1, K):
                                if S[i, k] > S[i, pr]:
                                    pr = k
                        den += swte[i]
                        if pr == yte[i]:
                            num += swte[i]
                    scores[fi, ci, li] = num / den
        self.Cs_ = Csa
        coef = np.zeros((nb, p))
        inter = np.zeros(nb)
        C_list = List[float]()
        l1_list = List[float]()
        if self.refit:
            ssum = np.sum(scores, axis=0)
            bc = 0
            bl = 0
            best = ssum[0, 0]
            for ci in range(nC):
                for li in range(nL):
                    if ssum[ci, li] > best:
                        best = ssum[ci, li]
                        bc = ci
                        bl = li
            C_list.append(Cs[bc])
            l1_list.append(l1s[bl])
            c0 = np.zeros((nb, nd))
            for k in range(nb):
                for j in range(nd):
                    acc = 0.0
                    for fi in range(nF):
                        acc += paths[k, fi, bc, bl, j]
                    c0[k, j] = acc / float(nF)
            res = _lr_path(Xa, yi, K, [Cs[bc]], self.fit_intercept, self.max_iter, self.tol, solver,
                           c0, cw, has_cw, self.dual, penalty, self.intercept_scaling,
                           self.random_state, sw, has_sw, l1s[bl], mss)
            w = res.coefs[0]
        else:
            w = np.zeros(nb * nd)
            bcs = 0.0
            bls = 0.0
            for fi in range(nF):
                bc = 0
                bl = 0
                best = scores[fi, 0, 0]
                for ci in range(nC):
                    for li in range(nL):
                        if scores[fi, ci, li] > best:
                            best = scores[fi, ci, li]
                            bc = ci
                            bl = li
                bcs += Cs[bc]
                bls += l1s[bl]
                for k in range(nb):
                    for j in range(nd):
                        w[k * nd + j] += paths[k, fi, bc, bl, j]
            w /= float(nF)
            C_list.append(bcs / float(nF))
            l1_list.append(bls / float(nF) if penalty == "elasticnet" else 0.0)
        for k in range(nb):
            for j in range(p):
                coef[k, j] = w[k * nd + j]
            if self.fit_intercept:
                inter[k] = w[k * nd + p]
        self.coef_ = coef
        self.intercept_ = inter
        nrep = 1 if binary else K
        self.C_ = np.full(nrep, C_list[0])
        self.l1_ratio_ = np.full(nrep, l1_list[0])
        self.l1_ratios_ = np.array(l1s) if not no_l1 else np.array([math.nan])
        self._scores3 = scores
        self._paths = paths
        self.scores_ = Dict[int, np.ndarray[float, 2]]()
        self.coefs_paths_ = Dict[int, np.ndarray[float, 3]]()
        if no_l1:
            for k in range(nb):
                key = int(classes[1]) if binary else int(classes[k])
                self.scores_[key] = np.ascontiguousarray(scores[:, :, 0])
                self.coefs_paths_[key] = np.ascontiguousarray(paths[k, :, :, 0, :])
            self.n_iter_ = np.ascontiguousarray(niter[:, :, 0])[None, :, :]
        else:
            for k in range(nb):
                key = int(classes[1]) if binary else int(classes[k])
                self.scores_[key] = np.ascontiguousarray(scores.reshape(nF, nC * nL))
                self.coefs_paths_[key] = np.ascontiguousarray(paths[k].reshape(nF, nC * nL, nd))
            self.n_iter_ = niter.reshape(1, nF, nC * nL)
        return self

    def _scores(self, X) -> np.ndarray[float, 2]:
        Xa = np.asarray(X, dtype=float)
        return Xa @ self.coef_.T + self.intercept_

    def decision_function(self, X):
        # DEVIATION: always 2-d (binary: (n, 1)); see decision_function_1d
        return self._scores(X)

    def decision_function_1d(self, X):
        """Binary decision function with sklearn's (n,) shape."""
        s = self._scores(X)
        if s.shape[1] != 1:
            raise ValueError("decision_function_1d is only defined for binary problems")
        return np.ascontiguousarray(s[:, 0])

    def predict(self, X):
        s = self._scores(X)
        n = s.shape[0]
        idx = np.zeros(n, dtype=int)
        if s.shape[1] == 1:
            for i in range(n):
                idx[i] = 1 if s[i, 0] > 0.0 else 0
        else:
            idx = np.argmax(s, axis=1)
        return self.classes_[idx]

    def predict_proba(self, X):
        s = self._scores(X)
        n = s.shape[0]
        if s.shape[1] == 1:
            pr = np.empty((n, 2))
            for i in range(n):
                e = _expit(s[i, 0])
                pr[i, 1] = e
                pr[i, 0] = 1.0 - e
            return pr
        return _softmax_rows(s)

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def score(self, X, y, sample_weight = None) -> float:
        ya = np.asarray(y)
        pr = self.predict(X)
        n = ya.shape[0]
        sw = _check_sw(sample_weight, n)
        num = 0.0
        den = 0.0
        for i in range(n):
            den += sw[i]
            if ya[i] == pr[i]:
                num += sw[i]
        return num / den
