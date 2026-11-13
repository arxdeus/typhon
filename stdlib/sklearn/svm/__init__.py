# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# sklearn.svm: SVC / NuSVC / SVR / NuSVR / OneClassSVM over the libsvm port
# (_libsvm.py) and LinearSVC / LinearSVR over the liblinear port
# (_liblinear.py); l1_min_c. Dense input. Class labels are integers.
# DEVIATION: decision_function of a binary SVC/NuSVC is (n, 1) (static
# return type); ravel it for sklearn's 1-D shape.

import math
import numpy as np
from ..base import _accuracy
from ._newrand import _StdMT, _fma, _f32
from . import _libsvm as L
from ._libsvm import sigmoid_predict as _sigmoid_predict, multiclass_probability as _multiclass_probability
from ._liblinear import train_wrap as _ll_train, solver_type as _ll_solver_type

def _legacy_randint_imax(seed: int) -> int:
    # np.random.RandomState(seed).randint(np.iinfo("i").max): masked rejection
    from ..utils import _LegacyMT
    g = _LegacyMT(seed)
    rng = u32(0x7fffffff - 1)
    mask = rng
    mask |= mask >> u32(1); mask |= mask >> u32(2); mask |= mask >> u32(4)
    mask |= mask >> u32(8); mask |= mask >> u32(16)
    v = g.genrand_int32() & mask
    while v > rng:
        v = g.genrand_int32() & mask
    return int(v)

def _seed_of(random_state: Optional[int]) -> int:
    if random_state is None:
        from numpy.random.legacy import check_random_state
        return int(check_random_state(None).randint(2147483647))
    return _legacy_randint_imax(random_state.__val__())

def _farr(v: List[float]):
    a = np.empty(len(v))
    for i in range(len(v)):
        a[i] = v[i]
    return a

def _ovr_decision_function(predictions, confidences, n_classes: int):
    n = predictions.shape[0]
    votes = np.zeros((n, n_classes))
    sum_conf = np.zeros((n, n_classes))
    k = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            for r in range(n):
                sum_conf[r, i] -= confidences[r, k]
                sum_conf[r, j] += confidences[r, k]
                if predictions[r, k]:
                    votes[r, j] += 1
                else:
                    votes[r, i] += 1
            k += 1
    m = np.max(np.abs(sum_conf))
    if m != 0:
        tc = sum_conf / (3 * (np.abs(sum_conf) + 1))
    else:
        tc = sum_conf
    return votes + tc

def _class_weight_vec(class_weight, classes, y, sample_weight = None) -> List[float]:
    from ..utils.class_weight import compute_class_weight
    if isinstance(class_weight, NoneType):
        return List[float]()
    w = compute_class_weight(class_weight, classes=classes, y=y, sample_weight=sample_weight)
    return [float(v) for v in w]

class _BaseLibSVM:
    _impl: int
    kernel: str
    degree: int
    gamma: str
    gamma_value: float
    coef0: float
    tol: float
    C: float
    nu: float
    epsilon: float
    shrinking: bool
    probability: bool
    cache_size: float
    max_iter: int
    random_state: Optional[int]
    _gamma: float
    support_: np.ndarray[int, 1]
    support_vectors_: np.ndarray[float, 2]
    _n_support: np.ndarray[int, 1]
    dual_coef_: np.ndarray[float, 2]
    _dual_coef_: np.ndarray[float, 2]
    intercept_: np.ndarray[float, 1]
    _intercept_: np.ndarray[float, 1]
    _probA: np.ndarray[float, 1]
    _probB: np.ndarray[float, 1]
    n_iter_: np.ndarray[int, 1]
    fit_status_: int
    shape_fit_: Tuple[int, int]
    n_features_in_: int
    _model: L.Model
    _par: L._Param
    _data: L._Data

    def _init_base(self, impl: int, kernel: str, degree: int, gamma, coef0: float, tol: float,
                   C: float, nu: float, epsilon: float, shrinking: bool, probability: bool,
                   cache_size: float, max_iter: int, random_state: Optional[int]):
        if kernel not in ("linear", "poly", "rbf", "sigmoid", "precomputed"):
            raise ValueError(f"The 'kernel' parameter must be a str among {{'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}} or a callable. Got {kernel!r} instead.")
        self._impl = impl
        self.kernel = kernel
        self.degree = degree
        if isinstance(gamma, str):
            if gamma != "scale" and gamma != "auto":
                raise ValueError("gamma must be 'scale', 'auto' or a float")
            self.gamma = gamma
            self.gamma_value = 0.0
        else:
            self.gamma = ""
            self.gamma_value = float(gamma)
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.nu = nu
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.probability = probability
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.random_state = random_state
        self._gamma = 0.0
        self.support_ = np.empty(0, dtype=int)
        self.support_vectors_ = np.empty((0, 0))
        self._n_support = np.empty(0, dtype=int)
        self.dual_coef_ = np.empty((0, 0))
        self._dual_coef_ = np.empty((0, 0))
        self.intercept_ = np.empty(0)
        self._intercept_ = np.empty(0)
        self._probA = np.empty(0)
        self._probB = np.empty(0)
        self.n_iter_ = np.empty(0, dtype=int)
        self.fit_status_ = 0
        self.shape_fit_ = (0, 0)
        self.n_features_in_ = 0
        self._model = L.Model()
        self._par = L._Param(0, "rbf", 3, 0.0, 0.0, 1.0, 0.5, 0.1, 1e-3, True, False, -1, List[float]())
        self._data = L._Data(np.empty((0, 0)), self._par)

    def _fit_libsvm(self, X, yv: List[float], sample_weight, cw: List[float]):
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        n, p = Xa.shape
        if self._impl != L.ONE_CLASS and n != len(yv):
            raise ValueError(f"X and y have incompatible shapes.\nX has {n} samples, but y has {len(yv)}.")
        if self.kernel == "precomputed":
            if n != p:
                raise ValueError(f"Precomputed matrix must be a square matrix. Input is a {n}x{p} matrix.")
            self._gamma = 0.0
        elif self.gamma == "scale":
            v = float(Xa.var())
            self._gamma = 1.0 / (float(p) * v) if v != 0 else 1.0
        elif self.gamma == "auto":
            self._gamma = 1.0 / float(p)
        else:
            self._gamma = self.gamma_value
        seed = _seed_of(self.random_state)
        sw = np.ones(n) if isinstance(sample_weight, NoneType) else np.asarray(sample_weight, dtype=float)
        if sw.shape[0] != n:
            raise ValueError(f"sample_weight and X have incompatible shapes: {sw.shape} vs {Xa.shape}")
        par = L._Param(self._impl, self.kernel, self.degree, self._gamma, self.coef0, self.C,
                       self.nu, self.epsilon, self.tol, self.shrinking, self.probability,
                       self.max_iter, cw)
        self._check_param(par, yv, sw)
        d = L._Data(Xa, par)
        tr = L.Trainer(d)
        prob = L._Prob(list(range(n)), yv, [float(v) for v in sw])
        wl = list(range(len(cw)))
        m = tr.train(prob, par, wl, cw, seed)
        self._model = m
        self._par = par
        self._data = d
        total = len(m.sv_rows)
        # sklearn's support_ is sv_ind of the zero-weight-filtered problem
        self.support_ = np.array(m.sv_ind, dtype=int) if total > 0 else np.empty(0, dtype=int)
        if self.kernel == "precomputed":
            self.support_vectors_ = np.empty((0, 0))
        else:
            sv = np.empty((total, p))
            for a in range(total):
                for b in range(p):
                    sv[a, b] = Xa[m.sv_rows[a], b]
            self.support_vectors_ = sv
        if self._impl == L.C_SVC or self._impl == L.NU_SVC:
            self._n_support = np.array(m.nSV, dtype=int)
        else:
            self._n_support = np.array([total, total], dtype=int)
        nc = m.nr_class
        dc = np.empty((nc - 1, total))
        for a in range(nc - 1):
            for b in range(total):
                dc[a, b] = m.sv_coef[a][b]
        ic = np.empty(len(m.rho))
        for i in range(len(m.rho)):
            t = m.rho[i]
            ic[i] = -t if t != 0 else 0.0
        self._probA = _farr(m.probA)
        self._probB = _farr(m.probB)
        self.shape_fit_ = (n, p)
        self.n_features_in_ = p
        self._intercept_ = ic.copy()
        self._dual_coef_ = dc
        if (self._impl == L.C_SVC or self._impl == L.NU_SVC) and nc == 2:
            self.intercept_ = -ic
            self.dual_coef_ = -dc
        else:
            self.intercept_ = ic
            self.dual_coef_ = dc
        self.n_iter_ = np.array(m.n_iter, dtype=int)
        self.fit_status_ = 1 if (self.max_iter != -1 and max(m.n_iter) >= self.max_iter) else 0

    def _check_param(self, par: L._Param, yv: List[float], sw):
        st = par.svm_type
        if par.gamma < 0:
            raise ValueError("gamma < 0")
        if par.eps <= 0:
            raise ValueError("eps <= 0")
        if (st == L.C_SVC or st == L.EPSILON_SVR or st == L.NU_SVR) and par.C <= 0:
            raise ValueError("C <= 0")
        if (st == L.NU_SVC or st == L.ONE_CLASS or st == L.NU_SVR) and (par.nu <= 0 or par.nu > 1):
            raise ValueError("nu <= 0 or nu > 1")
        if st == L.EPSILON_SVR and par.p < 0:
            raise ValueError("epsilon < 0")
        if st == L.NU_SVC:
            labs = List[int]()
            cnt = List[float]()
            for i in range(len(yv)):
                lab = int(yv[i])
                if lab in labs:
                    cnt[labs.index(lab)] += sw[i]
                else:
                    labs.append(lab)
                    cnt.append(sw[i])
            for i in range(len(labs)):
                for j in range(i + 1, len(labs)):
                    if par.nu * (cnt[i] + cnt[j]) / 2 > min(cnt[i], cnt[j]):
                        raise ValueError("specified nu is infeasible")
        anypos = False
        for i in range(len(yv)):
            if sw[i] > 0:
                anypos = True
        if not anypos:
            raise ValueError("Invalid input - all samples have zero or negative weights.")

    def _dec_values(self, X):
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        n = Xa.shape[0]
        m = self._model
        npair = len(m.rho)
        out = np.empty((n, npair))
        for r in range(n):
            _, dv = L.predict_values(self._data, m, self._par, Xa.data + r * Xa.shape[1], -1)
            for k in range(npair):
                out[r, k] = dv[k]
        return out

    @property
    def n_support_(self):
        if self._impl == L.C_SVC or self._impl == L.NU_SVC:
            return self._n_support
        return np.array([self._n_support[0]], dtype=int)

    @property
    def coef_(self):
        if self.kernel != "linear":
            raise AttributeError("coef_ is only available when using a linear kernel")
        nc = self._dual_coef_.shape[0] + 1
        if self._dual_coef_.shape[0] == 1:
            return self.dual_coef_ @ self.support_vectors_
        out = np.empty((nc * (nc - 1) // 2, self.support_vectors_.shape[1]))
        loc = [0] * (nc + 1)
        for c in range(nc):
            loc[c + 1] = loc[c] + int(self._n_support[c])
        k = 0
        for c1 in range(nc):
            for c2 in range(c1 + 1, nc):
                for f in range(out.shape[1]):
                    s = 0.0
                    for t in range(loc[c1], loc[c1 + 1]):
                        s += self.dual_coef_[c2 - 1, t] * self.support_vectors_[t, f]
                    s2 = 0.0
                    for t in range(loc[c2], loc[c2 + 1]):
                        s2 += self.dual_coef_[c1, t] * self.support_vectors_[t, f]
                    out[k, f] = s + s2
                k += 1
        return out

class _BaseSVC(_BaseLibSVM):
    decision_function_shape: str
    break_ties: bool
    classes_: np.ndarray[int, 1]
    class_weight_: np.ndarray[float, 1]

    def _init_svc(self, decision_function_shape: str, break_ties: bool):
        if decision_function_shape != "ovr" and decision_function_shape != "ovo":
            raise ValueError("decision_function_shape must be 'ovr' or 'ovo'")
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.classes_ = np.empty(0, dtype=int)
        self.class_weight_ = np.empty(0)

    def _fit_svc(self, X, y, sample_weight, class_weight):
        ya = np.asarray(y, dtype=int)
        classes, yi = np.unique(ya, return_inverse=True)
        if classes.shape[0] < 2:
            raise ValueError(f"The number of classes has to be greater than one; got {classes.shape[0]} class")
        self.classes_ = classes
        cw = _class_weight_vec(class_weight, classes, ya)
        if len(cw) == 0:
            cw = [1.0] * classes.shape[0]
        self.class_weight_ = _farr(cw)
        yi1 = np.asarray(yi).reshape(-1)
        self._fit_libsvm(X, [float(v) for v in yi1], sample_weight, cw)
        return self

    def decision_function(self, X):
        d = self._dec_values(X)
        nc = self.classes_.shape[0]
        if nc == 2:
            return -d
        if self.decision_function_shape == "ovo":
            return d
        return _ovr_decision_function(d < 0, -d, nc)

    def predict(self, X):
        nc = self.classes_.shape[0]
        if self.break_ties and self.decision_function_shape == "ovo":
            raise ValueError("break_ties must be False when decision_function_shape is 'ovo'")
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        n = Xa.shape[0]
        out = np.empty(n, dtype=int)
        if self.break_ties and nc > 2:
            D = self.decision_function(Xa)
            for r in range(n):
                out[r] = self.classes_[int(np.argmax(D[r]))]
            return out
        for r in range(n):
            v, _ = L.predict_values(self._data, self._model, self._par, Xa.data + r * Xa.shape[1], -1)
            out[r] = self.classes_[int(v)]
        return out

    def predict_proba(self, X):
        if not self.probability:
            raise AttributeError("predict_proba is not available when probability=False")
        d = self._dec_values(X)
        nc = self.classes_.shape[0]
        n = d.shape[0]
        out = np.empty((n, nc))
        min_prob = 1e-7
        for r in range(n):
            pw = [[0.0] * nc for _ in range(nc)]
            k = 0
            for i in range(nc):
                for j in range(i + 1, nc):
                    v = min(max(_sigmoid_predict(d[r, k], self._probA[k], self._probB[k]), min_prob), 1 - min_prob)
                    pw[i][j] = v
                    pw[j][i] = 1 - v
                    k += 1
            pe = _multiclass_probability(nc, pw)
            for i in range(nc):
                out[r, i] = pe[i]
        return out

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    @property
    def probA_(self):
        return self._probA

    @property
    def probB_(self):
        return self._probB

    def score(self, X, y, sample_weight = None) -> float:
        return _accuracy(y, self.predict(X))

class SVC(_BaseSVC):
    class_weight: Optional[Dict[int, float]]
    _balanced: bool

    def __init__(self, *, C: float = 1.0, kernel: str = "rbf", degree: int = 3,
                 gamma = "scale", coef0: float = 0.0, shrinking: bool = True,
                 probability: bool = False, tol: float = 1e-3, cache_size: float = 200.0,
                 class_weight = None, verbose: bool = False, max_iter: int = -1,
                 decision_function_shape: str = "ovr", break_ties: bool = False,
                 random_state: Optional[int] = None):
        self._init_base(L.C_SVC, kernel, degree, gamma, coef0, tol, C, 0.0, 0.0, shrinking,
                        probability, cache_size, max_iter, random_state)
        self._init_svc(decision_function_shape, break_ties)
        self._balanced = False
        self.class_weight = None
        if isinstance(class_weight, str):
            self._balanced = True
        elif not isinstance(class_weight, NoneType):
            self.class_weight = {int(k): float(v) for k, v in class_weight.items()}

    def fit(self, X, y, sample_weight = None):
        if self._balanced:
            return self._fit_svc(X, y, sample_weight, "balanced")
        else:
            return self._fit_svc(X, y, sample_weight, self.class_weight)

class NuSVC(_BaseSVC):
    class_weight: Optional[Dict[int, float]]
    _balanced: bool

    def __init__(self, *, nu: float = 0.5, kernel: str = "rbf", degree: int = 3,
                 gamma = "scale", coef0: float = 0.0, shrinking: bool = True,
                 probability: bool = False, tol: float = 1e-3, cache_size: float = 200.0,
                 class_weight = None, verbose: bool = False, max_iter: int = -1,
                 decision_function_shape: str = "ovr", break_ties: bool = False,
                 random_state: Optional[int] = None):
        self._init_base(L.NU_SVC, kernel, degree, gamma, coef0, tol, 0.0, nu, 0.0, shrinking,
                        probability, cache_size, max_iter, random_state)
        self._init_svc(decision_function_shape, break_ties)
        self._balanced = False
        self.class_weight = None
        if isinstance(class_weight, str):
            self._balanced = True
        elif not isinstance(class_weight, NoneType):
            self.class_weight = {int(k): float(v) for k, v in class_weight.items()}

    def fit(self, X, y, sample_weight = None):
        if self._balanced:
            return self._fit_svc(X, y, sample_weight, "balanced")
        else:
            return self._fit_svc(X, y, sample_weight, self.class_weight)

class _BaseSVR(_BaseLibSVM):
    def fit(self, X, y, sample_weight = None):
        ya = np.asarray(y, dtype=float)
        self._fit_libsvm(X, [float(v) for v in ya], sample_weight, List[float]())
        return self

    def predict(self, X):
        return np.ascontiguousarray(self._dec_values(X)[:, 0])

    def score(self, X, y, sample_weight = None) -> float:
        from ..metrics import r2_score
        return r2_score(y, self.predict(X))

class SVR(_BaseSVR):
    def __init__(self, *, kernel: str = "rbf", degree: int = 3, gamma = "scale",
                 coef0: float = 0.0, tol: float = 1e-3, C: float = 1.0, epsilon: float = 0.1,
                 shrinking: bool = True, cache_size: float = 200.0, verbose: bool = False,
                 max_iter: int = -1):
        self._init_base(L.EPSILON_SVR, kernel, degree, gamma, coef0, tol, C, 0.0, epsilon,
                        shrinking, False, cache_size, max_iter, None)

class NuSVR(_BaseSVR):
    def __init__(self, *, nu: float = 0.5, C: float = 1.0, kernel: str = "rbf", degree: int = 3,
                 gamma = "scale", coef0: float = 0.0, shrinking: bool = True,
                 tol: float = 1e-3, cache_size: float = 200.0, verbose: bool = False,
                 max_iter: int = -1):
        self._init_base(L.NU_SVR, kernel, degree, gamma, coef0, tol, C, nu, 0.0,
                        shrinking, False, cache_size, max_iter, None)

class OneClassSVM(_BaseLibSVM):
    offset_: np.ndarray[float, 1]

    def __init__(self, *, kernel: str = "rbf", degree: int = 3, gamma = "scale",
                 coef0: float = 0.0, tol: float = 1e-3, nu: float = 0.5, shrinking: bool = True,
                 cache_size: float = 200.0, verbose: bool = False, max_iter: int = -1):
        self._init_base(L.ONE_CLASS, kernel, degree, gamma, coef0, tol, 0.0, nu, 0.0,
                        shrinking, False, cache_size, max_iter, None)
        self.offset_ = np.empty(0)

    def fit(self, X, y = None, sample_weight = None):
        n = np.asarray(X).shape[0]
        self._fit_libsvm(X, [1.0] * n, sample_weight, List[float]())
        self.offset_ = -self._intercept_
        return self

    def decision_function(self, X):
        return np.ascontiguousarray(self._dec_values(X)[:, 0])

    def score_samples(self, X):
        return self.decision_function(X) + self.offset_[0]

    def predict(self, X):
        d = self.decision_function(X)
        out = np.empty(d.shape[0], dtype=int)
        for i in range(d.shape[0]):
            out[i] = 1 if d[i] > 0 else -1
        return out

    def fit_predict(self, X, y = None):
        return self.fit(X).predict(X)

# ------------------------------------------------------------ liblinear

def _ll_supported(multi_class: str, penalty: str, loss: str, dual: bool) -> bool:
    try:
        _ll_solver_type(multi_class, penalty, loss, dual)
        return True
    except ValueError:
        return False

def _validate_dual(dual, loss: str, penalty: str, multi_class: str, n: int, p: int) -> bool:
    if isinstance(dual, str):
        if n < p:
            return _ll_supported(multi_class, penalty, loss, True)
        else:
            return not _ll_supported(multi_class, penalty, loss, False)
    return bool(dual)

def _fit_liblinear(X, yf: List[float], C: float, fit_intercept: bool, intercept_scaling: float,
                   cw: List[float], penalty: str, dual: bool, max_iter: int, tol: float,
                   random_state: Optional[int], multi_class: str, loss: str, epsilon: float,
                   sample_weight):
    Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
    n = Xa.shape[0]
    bias = -1.0
    if fit_intercept:
        if intercept_scaling <= 0:
            raise ValueError(f"Intercept scaling is {intercept_scaling} but needs to be greater than 0. To disable fitting an intercept, set fit_intercept=False.")
        bias = intercept_scaling
    seed = _seed_of(random_state)
    sw = np.ones(n) if isinstance(sample_weight, NoneType) else np.asarray(sample_weight, dtype=float)
    st = _ll_solver_type(multi_class, penalty, loss, dual)
    if st == 4:
        raise NotImplementedError("multi_class='crammer_singer' is not supported")
    raw, n_iter = _ll_train(Xa, np.array(yf), st, tol, bias, C, _farr(cw), max_iter, seed, epsilon, sw)
    nf = Xa.shape[1]
    if fit_intercept:
        coef = np.ascontiguousarray(raw[:, :nf])
        icpt = intercept_scaling * np.ascontiguousarray(raw[:, nf])
        return coef, icpt, n_iter
    else:
        return raw, np.zeros(raw.shape[0]), n_iter

class LinearSVC:
    penalty: str
    loss: str
    dual: str
    tol: float
    C: float
    multi_class: str
    fit_intercept: bool
    intercept_scaling: float
    class_weight: Optional[Dict[int, float]]
    _balanced: bool
    random_state: Optional[int]
    max_iter: int
    classes_: np.ndarray[int, 1]
    coef_: np.ndarray[float, 2]
    intercept_: np.ndarray[float, 1]
    n_iter_: int
    n_features_in_: int

    def __init__(self, penalty: str = "l2", loss: str = "squared_hinge", *, dual = "auto",
                 tol: float = 1e-4, C: float = 1.0, multi_class: str = "ovr",
                 fit_intercept: bool = True, intercept_scaling: float = 1.0, class_weight = None,
                 verbose: int = 0, random_state: Optional[int] = None, max_iter: int = 1000):
        self.penalty = penalty
        self.loss = loss
        self.dual = dual if isinstance(dual, str) else ("true" if dual else "false")
        self.tol = tol
        self.C = C
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self._balanced = False
        self.class_weight = None
        if isinstance(class_weight, str):
            self._balanced = True
        elif not isinstance(class_weight, NoneType):
            self.class_weight = {int(k): float(v) for k, v in class_weight.items()}
        self.random_state = random_state
        self.max_iter = max_iter
        self.classes_ = np.empty(0, dtype=int)
        self.coef_ = np.empty((0, 0))
        self.intercept_ = np.empty(0)
        self.n_iter_ = 0
        self.n_features_in_ = 0

    def _dual(self, n: int, p: int) -> bool:
        if self.dual == "auto":
            return _validate_dual("auto", self.loss, self.penalty, self.multi_class, n, p)
        return self.dual == "true"

    def fit(self, X, y, sample_weight = None):
        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y, dtype=int)
        classes, yi = np.unique(ya, return_inverse=True)
        if classes.shape[0] < 2:
            raise ValueError(f"This solver needs samples of at least 2 classes in the data, but the data contains only one class: {classes[0]}")
        self.classes_ = classes
        cw = List[float]()
        if self._balanced:
            cw = _class_weight_vec("balanced", classes, ya, sample_weight)
        else:
            cw = _class_weight_vec(self.class_weight, classes, ya)
        if len(cw) == 0:
            cw = [1.0] * classes.shape[0]
        n, p = Xa.shape
        d = self._dual(n, p)
        yi1 = np.asarray(yi).reshape(-1)
        coef, icpt, it = _fit_liblinear(Xa, [float(v) for v in yi1], self.C, self.fit_intercept,
                                        self.intercept_scaling, cw, self.penalty, d, self.max_iter,
                                        self.tol, self.random_state, self.multi_class, self.loss,
                                        0.1, sample_weight)
        self.coef_ = coef
        self.intercept_ = icpt
        self.n_iter_ = int(np.max(it))
        self.n_features_in_ = p
        return self

    def decision_function(self, X):
        D = np.asarray(X, dtype=float) @ self.coef_.T + self.intercept_
        return D

    def predict(self, X):
        D = self.decision_function(X)
        n = D.shape[0]
        out = np.empty(n, dtype=int)
        for i in range(n):
            if D.shape[1] == 1:
                out[i] = self.classes_[1] if D[i, 0] > 0 else self.classes_[0]
            else:
                out[i] = self.classes_[int(np.argmax(D[i]))]
        return out

    def score(self, X, y, sample_weight = None) -> float:
        return _accuracy(y, self.predict(X))

class LinearSVR:
    epsilon: float
    tol: float
    C: float
    loss: str
    fit_intercept: bool
    intercept_scaling: float
    dual: str
    random_state: Optional[int]
    max_iter: int
    coef_: np.ndarray[float, 1]
    intercept_: np.ndarray[float, 1]
    n_iter_: int

    def __init__(self, *, epsilon: float = 0.0, tol: float = 1e-4, C: float = 1.0,
                 loss: str = "epsilon_insensitive", fit_intercept: bool = True,
                 intercept_scaling: float = 1.0, dual = "auto", verbose: int = 0,
                 random_state: Optional[int] = None, max_iter: int = 1000):
        self.epsilon = epsilon
        self.tol = tol
        self.C = C
        self.loss = loss
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.dual = dual if isinstance(dual, str) else ("true" if dual else "false")
        self.random_state = random_state
        self.max_iter = max_iter
        self.coef_ = np.empty(0)
        self.intercept_ = np.empty(0)
        self.n_iter_ = 0

    def fit(self, X, y, sample_weight = None):
        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y, dtype=float)
        n, p = Xa.shape
        d = _validate_dual("auto", self.loss, "l2", "ovr", n, p) if self.dual == "auto" else self.dual == "true"
        coef, icpt, it = _fit_liblinear(Xa, [float(v) for v in ya], self.C, self.fit_intercept,
                                        self.intercept_scaling, List[float](), "l2", d,
                                        self.max_iter, self.tol, self.random_state, "ovr",
                                        self.loss, self.epsilon, sample_weight)
        self.coef_ = np.ascontiguousarray(coef[0])
        self.intercept_ = icpt
        self.n_iter_ = int(np.max(it))
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_[0]

    def score(self, X, y, sample_weight = None) -> float:
        from ..metrics import r2_score
        return r2_score(y, self.predict(X))

def l1_min_c(X, y, *, loss: str = "squared_hinge", fit_intercept: bool = True,
             intercept_scaling: float = 1.0) -> float:
    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=int)
    classes = np.unique(ya)
    n = Xa.shape[0]
    k = classes.shape[0]
    # LabelBinarizer(neg_label=-1): one row for binary, k rows otherwise
    nr = 1 if k == 2 else k
    den = 0.0
    for r in range(nr):
        Yr = np.empty(n)
        for i in range(n):
            pos = ya[i] == classes[1] if k == 2 else ya[i] == classes[r]
            Yr[i] = 1.0 if pos else -1.0
        v = Yr @ Xa
        den = max(den, float(np.max(np.abs(v))))
        if fit_intercept:
            den = max(den, abs(float(np.sum(Yr)) * intercept_scaling))
    if den == 0.0:
        raise ValueError("Ill-posed l1_min_c calculation: l1 will always select zero coefficients for this data")
    if loss == "squared_hinge":
        return 0.5 / den
    else:
        return 2.0 / den
