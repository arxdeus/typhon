# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# sklearn.decomposition._dict_learning: sparse_encode, dict_learning,
# dict_learning_online, DictionaryLearning, MiniBatchDictionaryLearning,
# SparseCoder.  Encoders reuse the LARS / coordinate-descent / OMP solvers of
# sklearn.linear_model (float64 only; n_jobs and callbacks are ignored).

from ..base import estimator_repr as _estimator_repr
import numpy as np
import scipy.linalg as sl
from numpy.random import RandomState
from ._extmath import check_random_state, randomized_svd, svd_flip_u, row_norms
from ..linear_model._least_angle import _lars_path_solver, _EPS
from ..linear_model._coordinate_descent import _enet_path_1d
from ..linear_model._omp import orthogonal_mp_gram

def _opt2(x):
    if isinstance(x, NoneType):
        return np.empty((0, 0)), False
    else:
        return np.ascontiguousarray(np.asarray(x, dtype=float)), True

def _opt_int(x) -> Optional[int]:
    if isinstance(x, NoneType):
        return None
    else:
        return int(x)

def _opt_float(x) -> Optional[float]:
    if isinstance(x, NoneType):
        return None
    else:
        return float(x)

def _check_positive_coding(method: str, positive: bool):
    if positive and (method == "omp" or method == "lars"):
        raise ValueError(f"Positive constraint not supported for '{method}' coding method.")

def _check_algorithm(algorithm: str):
    if (algorithm != "lasso_lars" and algorithm != "lasso_cd" and algorithm != "lars"
            and algorithm != "omp" and algorithm != "threshold"):
        raise ValueError(f"The 'algorithm' parameter of sparse_encode must be a str among lasso_lars, lasso_cd, lars, omp, threshold. Got {algorithm} instead.")

def _se_precomputed(X: np.ndarray[float, 2], D: np.ndarray[float, 2], gram: np.ndarray[float, 2],
                    cov: np.ndarray[float, 2], has_cov: bool, algorithm: str, regularization: float,
                    init: np.ndarray[float, 2], has_init: bool, max_iter: int, positive: bool):
    n_samples, n_features = X.shape
    n_comp = D.shape[0]
    code = np.zeros((n_samples, n_comp))
    if algorithm == "lasso_lars":
        alpha = regularization / float(n_features)
        Dt = np.ascontiguousarray(D.T)
        for t in range(n_samples):
            xyt = np.ascontiguousarray(cov[:, t])
            al, act, cf, it = _lars_path_solver(Dt, np.ascontiguousarray(X[t]), True, xyt, gram,
                                                n_features, max_iter, alpha, "lasso", _EPS,
                                                False, positive)
            code[t] = cf[:, 0]
    elif algorithm == "lasso_cd":
        alpha = regularization / float(n_features)
        Dt = np.ascontiguousarray(D.T)
        Xy = D @ X.T
        rng = RandomState(0)
        al = np.array([alpha])
        for t in range(n_samples):
            c0 = np.zeros(n_comp)
            if has_init:
                c0 = np.ascontiguousarray(init[t]).copy()
            c, g, it = _enet_path_1d(Dt, np.ascontiguousarray(X[t]), 1.0, al, True, gram,
                                     np.ascontiguousarray(Xy[:, t]), c0, 1e-8, max_iter, rng,
                                     False, positive)
            code[t] = c[:, 0]
    elif algorithm == "lars":
        nnz = int(regularization)
        Dt = np.ascontiguousarray(D.T)
        for t in range(n_samples):
            xyt = np.ascontiguousarray(cov[:, t])
            al, act, cf, it = _lars_path_solver(Dt, np.ascontiguousarray(X[t]), True, xyt, gram,
                                                n_features, nnz, 0.0, "lar", _EPS, False, False)
            code[t] = cf[:, 0]
    elif algorithm == "threshold":
        for t in range(n_samples):
            for k in range(n_comp):
                v = cov[k, t]
                s = 0.0
                if v > 0.0:
                    s = 1.0
                elif v < 0.0:
                    s = -1.0
                r = s * max(abs(v) - regularization, 0.0)
                if positive and r < 0.0:
                    r = 0.0
                code[t, k] = r
    else:
        nsq = row_norms(X, True)
        code = np.ascontiguousarray(orthogonal_mp_gram(gram, cov, n_nonzero_coefs=int(regularization),
                                                       tol=None, norms_squared=nsq).T)
    return code

def _sparse_encode(X: np.ndarray[float, 2], D: np.ndarray[float, 2], algorithm: str,
                   n_nonzero_coefs: Optional[int], alpha: Optional[float],
                   gram0: np.ndarray[float, 2], has_gram: bool, cov0: np.ndarray[float, 2],
                   has_cov: bool, init: np.ndarray[float, 2], has_init: bool, max_iter: int,
                   positive: bool):
    n_samples, n_features = X.shape
    n_comp = D.shape[0]
    regularization = 1.0
    if algorithm == "lars" or algorithm == "omp":
        if n_nonzero_coefs is None:
            regularization = min(max(float(n_features) / 10.0, 1.0), float(n_comp))
        else:
            regularization = float(n_nonzero_coefs.__val__())
    else:
        if alpha is not None:
            regularization = alpha.__val__()
    gram = gram0
    if not has_gram and algorithm != "threshold":
        gram = D @ D.T
    cov = cov0
    if not has_cov and algorithm != "lasso_cd":
        cov = D @ X.T
    return _se_precomputed(X, D, gram, cov, algorithm != "lasso_cd", algorithm, regularization,
                           init, has_init, max_iter, positive)

def sparse_encode(X, dictionary, *, gram = None, cov = None, algorithm: str = "lasso_lars",
                  n_nonzero_coefs = None, alpha = None, copy_cov: bool = True, init = None,
                  max_iter: int = 1000, n_jobs = None, check_input: bool = True,
                  verbose: int = 0, positive: bool = False):
    _check_algorithm(algorithm)
    Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
    Da = np.ascontiguousarray(np.asarray(dictionary, dtype=float))
    if Xa.ndim != 2:
        raise ValueError(f"Expected 2D array, got {Xa.ndim}D array instead")
    if Da.shape[1] != Xa.shape[1]:
        raise ValueError(f"Dictionary and X have different numbers of features:dictionary.shape: {Da.shape} X.shape{Xa.shape}")
    _check_positive_coding(algorithm, positive)
    g, hg = _opt2(gram)
    c, hc = _opt2(cov)
    i0, hi = _opt2(init)
    return _sparse_encode(Xa, Da, algorithm, _opt_int(n_nonzero_coefs), _opt_float(alpha),
                          g, hg, c, hc, i0, hi, max_iter, positive)

def _update_dict(dictionary: np.ndarray[float, 2], Y: np.ndarray[float, 2],
                 code: np.ndarray[float, 2], A0: np.ndarray[float, 2], B0: np.ndarray[float, 2],
                 has_AB: bool, rs, positive: bool):
    n_samples, n_components = code.shape
    A = A0
    B = B0
    if not has_AB:
        A = code.T @ code
        B = Y.T @ code
    nf = dictionary.shape[1]
    for k in range(n_components):
        if A[k, k] > 1e-6:
            tmp = A[k] @ dictionary
            for j in range(nf):
                dictionary[k, j] += (B[j, k] - tmp[j]) / A[k, k]
        else:
            newd = np.ascontiguousarray(Y[int(rs.choice(n_samples))])
            sd = float(np.std(newd))
            noise_level = 0.01 * (sd if sd != 0.0 else 1.0)
            noise = rs.normal(0.0, noise_level, size=nf)
            for j in range(nf):
                dictionary[k, j] = newd[j] + noise[j]
            for i in range(n_samples):
                code[i, k] = 0.0
        if positive:
            for j in range(nf):
                if dictionary[k, j] < 0.0:
                    dictionary[k, j] = 0.0
        nrm = 0.0
        for j in range(nf):
            nrm += dictionary[k, j] * dictionary[k, j]
        nrm = max(np.sqrt(nrm), 1.0)
        for j in range(nf):
            dictionary[k, j] /= nrm

def _cost(X: np.ndarray[float, 2], code: np.ndarray[float, 2], dictionary: np.ndarray[float, 2],
          alpha: float) -> float:
    R = X - code @ dictionary
    return 0.5 * float(np.sum(R ** 2)) + alpha * float(np.sum(np.abs(code)))

def _dict_learning(X: np.ndarray[float, 2], n_components: int, alpha: float, max_iter: int,
                   tol: float, method: str, dict_init: np.ndarray[float, 2], has_dict_init: bool,
                   code_init: np.ndarray[float, 2], has_code_init: bool, rs,
                   positive_dict: bool, positive_code: bool, method_max_iter: int):
    code = np.empty((0, 0))
    dictionary = np.empty((0, 0))
    if has_code_init and has_dict_init:
        code = code_init.copy()
        dictionary = dict_init.copy()
    else:
        U, S, Vt = sl.svd(X, full_matrices=False)
        svd_flip_u(U, Vt)
        code = U
        dictionary = S.reshape((-1, 1)) * Vt
    r = dictionary.shape[0]
    if n_components <= r:
        code = np.ascontiguousarray(code[:, :n_components])
        dictionary = np.ascontiguousarray(dictionary[:n_components, :])
    else:
        c2 = np.zeros((code.shape[0], n_components))
        c2[:, :code.shape[1]] = code
        code = c2
        d2 = np.zeros((n_components, dictionary.shape[1]))
        d2[:r, :] = dictionary
        dictionary = d2
    errors = List[float]()
    ii = -1
    dummy = np.empty((0, 0))
    for it in range(max_iter):
        ii = it
        code = _sparse_encode(X, dictionary, method, None, alpha, dummy, False, dummy, False,
                              code, True, method_max_iter, positive_code)
        _update_dict(dictionary, X, code, dummy, dummy, False, rs, positive_dict)
        current_cost = _cost(X, code, dictionary, alpha)
        errors.append(current_cost)
        if ii > 0:
            dE = errors[len(errors) - 2] - errors[len(errors) - 1]
            if dE < tol * errors[len(errors) - 1]:
                break
    return code, dictionary, errors, ii + 1

class _BaseSparseCoding:
    transform_algorithm: str
    transform_n_nonzero_coefs: Optional[int]
    transform_alpha: Optional[float]
    transform_max_iter: int
    split_sign: bool
    positive_code: bool
    has_alpha: bool
    alpha_default: float

    def __init__(self, transform_algorithm: str, transform_n_nonzero_coefs: Optional[int],
                 transform_alpha: Optional[float], split_sign: bool, n_jobs,
                 positive_code: bool, transform_max_iter: int, has_alpha: bool, alpha_default: float):
        _check_algorithm(transform_algorithm)
        self.transform_algorithm = transform_algorithm
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs
        self.transform_alpha = transform_alpha
        self.transform_max_iter = transform_max_iter
        self.split_sign = split_sign
        self.positive_code = positive_code
        self.has_alpha = has_alpha
        self.alpha_default = alpha_default

    def _transform(self, X, dictionary: np.ndarray[float, 2]):
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        if Xa.ndim != 2:
            raise ValueError(f"Expected 2D array, got {Xa.ndim}D array instead")
        if Xa.shape[1] != dictionary.shape[1]:
            raise ValueError(f"X has {Xa.shape[1]} features, but {self.__class__.__name__} is expecting {dictionary.shape[1]} features as input")
        ta = self.transform_alpha
        if self.has_alpha and ta is None:
            ta = self.alpha_default
        _check_positive_coding(self.transform_algorithm, self.positive_code)
        dummy = np.empty((0, 0))
        code = _sparse_encode(Xa, dictionary, self.transform_algorithm, self.transform_n_nonzero_coefs,
                              ta, dummy, False, dummy, False, dummy, False, self.transform_max_iter,
                              self.positive_code)
        if self.split_sign:
            n, k = code.shape
            sc = np.empty((n, 2 * k))
            for i in range(n):
                for j in range(k):
                    sc[i, j] = max(code[i, j], 0.0)
                    sc[i, k + j] = -min(code[i, j], 0.0)
            code = sc
        return code

    def _inverse_transform(self, code, dictionary: np.ndarray[float, 2]):
        C = np.ascontiguousarray(np.asarray(code, dtype=float))
        expected = dictionary.shape[0]
        if self.split_sign:
            expected += expected
        if C.shape[1] != expected:
            raise ValueError(f"The number of components in the code is different from the number of components in the dictionary.Expected {expected}, got {C.shape[1]}.")
        if self.split_sign:
            h = C.shape[1] // 2
            C = np.ascontiguousarray(C[:, :h] - C[:, h:])
        return C @ dictionary

class SparseCoder(_BaseSparseCoding):
    def __repr__(self) -> str:
        return _estimator_repr(self)

    dictionary: np.ndarray[float, 2]
    n_components_: int
    n_features_in_: int

    def __init__(self, dictionary, *, transform_algorithm: str = "omp",
                 transform_n_nonzero_coefs: Optional[int] = None,
                 transform_alpha: Optional[float] = None, split_sign: bool = False,
                 n_jobs = None, positive_code: bool = False, transform_max_iter: int = 1000):
        _BaseSparseCoding.__init__(self, transform_algorithm, transform_n_nonzero_coefs,
                                   transform_alpha, split_sign, n_jobs, positive_code,
                                   transform_max_iter, False, 1.0)
        self.dictionary = np.ascontiguousarray(np.asarray(dictionary, dtype=float))
        self.n_components_ = 0
        self.n_features_in_ = 0

    def fit(self, X, y = None):
        Xa = np.asarray(X, dtype=float)
        self.n_components_ = self.dictionary.shape[0]
        if Xa.shape[1] != self.dictionary.shape[1]:
            raise ValueError(f"Dictionary and X have different numbers of features:dictionary.shape: {self.dictionary.shape} X.shape: {Xa.shape}")
        self.n_features_in_ = Xa.shape[1]
        return self

    def transform(self, X, y = None):
        return self._transform(X, self.dictionary)

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return self._inverse_transform(X, self.dictionary)

class DictionaryLearning(_BaseSparseCoding):
    def __repr__(self) -> str:
        return _estimator_repr(self)

    n_components: Optional[int]
    alpha: float
    max_iter: int
    tol: float
    fit_algorithm: str
    code_init: np.ndarray[float, 2]
    has_code_init: bool
    dict_init: np.ndarray[float, 2]
    has_dict_init: bool
    random_state: Optional[int]
    positive_dict: bool
    components_: np.ndarray[float, 2]
    error_: List[float]
    n_iter_: int
    n_features_in_: int

    def __init__(self, n_components: Optional[int] = None, *, alpha: float = 1.0, max_iter: int = 1000,
                 tol: float = 1e-8, fit_algorithm: str = "lars", transform_algorithm: str = "omp",
                 transform_n_nonzero_coefs: Optional[int] = None,
                 transform_alpha: Optional[float] = None, n_jobs = None, code_init = None,
                 dict_init = None, callback = None, verbose: int = 0, split_sign: bool = False,
                 random_state: Optional[int] = None, positive_code: bool = False,
                 positive_dict: bool = False, transform_max_iter: int = 1000):
        if fit_algorithm != "lars" and fit_algorithm != "cd":
            raise ValueError(f"The 'fit_algorithm' parameter of DictionaryLearning must be a str among cd, lars. Got {fit_algorithm} instead.")
        _BaseSparseCoding.__init__(self, transform_algorithm, transform_n_nonzero_coefs,
                                   transform_alpha, split_sign, n_jobs, positive_code,
                                   transform_max_iter, True, alpha)
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_algorithm = fit_algorithm
        self.code_init, self.has_code_init = _opt2(code_init)
        self.dict_init, self.has_dict_init = _opt2(dict_init)
        self.random_state = random_state
        self.positive_dict = positive_dict
        self.components_ = np.empty((0, 0))
        self.error_ = List[float]()
        self.n_iter_ = 0
        self.n_features_in_ = 0

    def fit_transform(self, X, y = None):
        _check_positive_coding(self.fit_algorithm, self.positive_code)
        method = "lasso_" + self.fit_algorithm
        rs = check_random_state(self.random_state)
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        if Xa.ndim != 2:
            raise ValueError(f"Expected 2D array, got {Xa.ndim}D array instead")
        self.n_features_in_ = Xa.shape[1]
        nc = Xa.shape[1] if self.n_components is None else self.n_components.__val__()
        V, U, E, it = _dict_learning(Xa, nc, self.alpha, self.max_iter, self.tol, method,
                                     self.dict_init, self.has_dict_init, self.code_init,
                                     self.has_code_init, rs, self.positive_dict,
                                     self.positive_code, self.transform_max_iter)
        self.components_ = U
        self.error_ = E
        self.n_iter_ = it
        return V

    def fit(self, X, y = None):
        self.fit_transform(X)
        return self

    def transform(self, X):
        return self._transform(X, self.components_)

    def inverse_transform(self, X):
        return self._inverse_transform(X, self.components_)

class MiniBatchDictionaryLearning(_BaseSparseCoding):
    def __repr__(self) -> str:
        return _estimator_repr(self)

    n_components: Optional[int]
    alpha: float
    max_iter: int
    fit_algorithm: str
    batch_size: int
    shuffle: bool
    dict_init: np.ndarray[float, 2]
    has_dict_init: bool
    random_state: Optional[int]
    positive_dict: bool
    tol: float
    max_no_improvement: Optional[int]
    verbose: int
    components_: np.ndarray[float, 2]
    n_iter_: float
    n_steps_: int
    n_features_in_: int
    _n_components: int
    _fit_algorithm: str
    _batch_size: int
    _A: np.ndarray[float, 2]
    _B: np.ndarray[float, 2]
    _rs: RandomState
    _ewa_cost: float
    _has_ewa: bool
    _ewa_cost_min: float
    _has_ewa_min: bool
    _no_improvement: int
    _has_components: bool

    def __init__(self, n_components: Optional[int] = None, *, alpha: float = 1.0, max_iter: int = 1000,
                 fit_algorithm: str = "lars", n_jobs = None, batch_size: int = 256,
                 shuffle: bool = True, dict_init = None, transform_algorithm: str = "omp",
                 transform_n_nonzero_coefs: Optional[int] = None,
                 transform_alpha: Optional[float] = None, verbose: int = 0, split_sign: bool = False,
                 random_state: Optional[int] = None, positive_code: bool = False,
                 positive_dict: bool = False, transform_max_iter: int = 1000, callback = None,
                 tol: float = 1e-3, max_no_improvement: Optional[int] = 10):
        if fit_algorithm != "lars" and fit_algorithm != "cd":
            raise ValueError(f"The 'fit_algorithm' parameter of MiniBatchDictionaryLearning must be a str among cd, lars. Got {fit_algorithm} instead.")
        _BaseSparseCoding.__init__(self, transform_algorithm, transform_n_nonzero_coefs,
                                   transform_alpha, split_sign, n_jobs, positive_code,
                                   transform_max_iter, True, alpha)
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.fit_algorithm = fit_algorithm
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dict_init, self.has_dict_init = _opt2(dict_init)
        self.random_state = random_state
        self.positive_dict = positive_dict
        self.tol = tol
        self.max_no_improvement = max_no_improvement
        self.verbose = verbose
        self.components_ = np.empty((0, 0))
        self.n_iter_ = 0.0
        self.n_steps_ = 0
        self.n_features_in_ = 0
        self._n_components = 0
        self._fit_algorithm = ""
        self._batch_size = 0
        self._A = np.empty((0, 0))
        self._B = np.empty((0, 0))
        self._rs = RandomState(0)
        self._ewa_cost = 0.0
        self._has_ewa = False
        self._ewa_cost_min = 0.0
        self._has_ewa_min = False
        self._no_improvement = 0
        self._has_components = False

    def _check_params(self, X: np.ndarray[float, 2]):
        self._n_components = X.shape[1] if self.n_components is None else self.n_components.__val__()
        _check_positive_coding(self.fit_algorithm, self.positive_code)
        self._fit_algorithm = "lasso_" + self.fit_algorithm
        self._batch_size = min(self.batch_size, X.shape[0])

    def _initialize_dict(self, X: np.ndarray[float, 2]) -> np.ndarray[float, 2]:
        dictionary = np.empty((0, 0))
        if self.has_dict_init:
            dictionary = self.dict_init.copy()
        else:
            _, S, Vt = randomized_svd(X, self._n_components, self._rs)
            dictionary = S.reshape((-1, 1)) * Vt
        if self._n_components <= dictionary.shape[0]:
            return np.ascontiguousarray(dictionary[:self._n_components, :])
        d2 = np.zeros((self._n_components, dictionary.shape[1]))
        d2[:dictionary.shape[0], :] = dictionary
        return d2

    def _update_inner_stats(self, X: np.ndarray[float, 2], code: np.ndarray[float, 2],
                            batch_size: int, step: int):
        theta = 0.0
        if step < batch_size - 1:
            theta = float((step + 1) * batch_size)
        else:
            theta = float(batch_size * batch_size + step + 1 - batch_size)
        beta = (theta + 1.0 - float(batch_size)) / (theta + 1.0)
        self._A = self._A * beta + (code.T @ code) / float(batch_size)
        self._B = self._B * beta + (X.T @ code) / float(batch_size)

    def _minibatch_step(self, X: np.ndarray[float, 2], dictionary: np.ndarray[float, 2], step: int) -> float:
        batch_size = X.shape[0]
        dummy = np.empty((0, 0))
        code = _sparse_encode(X, dictionary, self._fit_algorithm, None, self.alpha, dummy, False,
                              dummy, False, dummy, False, self.transform_max_iter, self.positive_code)
        R = X - code @ dictionary
        batch_cost = (0.5 * float(np.sum(R ** 2)) + self.alpha * float(np.sum(np.abs(code)))) / float(batch_size)
        self._update_inner_stats(X, code, batch_size, step)
        _update_dict(dictionary, X, code, self._A, self._B, True, self._rs, self.positive_dict)
        return batch_cost

    def _check_convergence(self, batch_size: int, batch_cost: float, new_dict: np.ndarray[float, 2],
                           old_dict: np.ndarray[float, 2], n_samples: int, step0: int) -> bool:
        step = step0 + 1
        if float(step) <= min(100.0, float(n_samples) / float(batch_size)):
            return False
        if not self._has_ewa:
            self._ewa_cost = batch_cost
            self._has_ewa = True
        else:
            a = float(batch_size) / float(n_samples + 1)
            a = min(a, 1.0)
            self._ewa_cost = self._ewa_cost * (1.0 - a) + batch_cost * a
        dd = float(np.sqrt(np.sum((new_dict - old_dict) ** 2))) / float(self._n_components)
        if self.tol > 0.0 and dd <= self.tol:
            return True
        if (not self._has_ewa_min) or self._ewa_cost < self._ewa_cost_min:
            self._no_improvement = 0
            self._ewa_cost_min = self._ewa_cost
            self._has_ewa_min = True
        else:
            self._no_improvement += 1
        if self.max_no_improvement is not None and self._no_improvement >= self.max_no_improvement.__val__():
            return True
        return False

    def fit(self, X, y = None):
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        if Xa.ndim != 2:
            raise ValueError(f"Expected 2D array, got {Xa.ndim}D array instead")
        self.n_features_in_ = Xa.shape[1]
        self._check_params(Xa)
        self._rs = check_random_state(self.random_state)
        dictionary = self._initialize_dict(Xa)
        old_dict = dictionary.copy()
        X_train = Xa
        if self.shuffle:
            X_train = Xa.copy()
            self._rs.shuffle(X_train)
        n_samples, n_features = X_train.shape
        self._A = np.zeros((self._n_components, self._n_components))
        self._B = np.zeros((n_features, self._n_components))
        self._has_ewa = False
        self._has_ewa_min = False
        self._no_improvement = 0
        bs = self._batch_size
        nb = (n_samples + bs - 1) // bs
        n_steps_per_iter = nb
        n_steps = self.max_iter * n_steps_per_iter
        i = -1
        for step in range(n_steps):
            i = step
            b = step % nb
            lo = b * bs
            hi = min(n_samples, lo + bs)
            Xb = np.ascontiguousarray(X_train[lo:hi])
            cost = self._minibatch_step(Xb, dictionary, step)
            if self._check_convergence(Xb.shape[0], cost, dictionary, old_dict, n_samples, step):
                break
            old_dict = dictionary.copy()
        self.n_steps_ = i + 1
        self.n_iter_ = float(-((-self.n_steps_) // n_steps_per_iter))
        self.components_ = dictionary
        self._has_components = True
        return self

    def partial_fit(self, X, y = None):
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        dictionary = np.empty((0, 0))
        if not self._has_components:
            self.n_features_in_ = Xa.shape[1]
            self._check_params(Xa)
            self._rs = check_random_state(self.random_state)
            dictionary = self._initialize_dict(Xa)
            self.n_steps_ = 0
            self._A = np.zeros((self._n_components, self._n_components))
            self._B = np.zeros((Xa.shape[1], self._n_components))
        else:
            dictionary = self.components_
        self._minibatch_step(Xa, dictionary, self.n_steps_)
        self.components_ = dictionary
        self._has_components = True
        self.n_steps_ += 1
        return self

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

    def transform(self, X):
        return self._transform(X, self.components_)

    def inverse_transform(self, X):
        return self._inverse_transform(X, self.components_)

def dict_learning(X, n_components: int, *, alpha: float, max_iter: int = 100, tol: float = 1e-8,
                  method: str = "lars", n_jobs = None, dict_init = None, code_init = None,
                  callback = None, verbose: int = 0, random_state: Optional[int] = None,
                  return_n_iter: Literal[bool] = False, positive_dict: bool = False,
                  positive_code: bool = False, method_max_iter: int = 1000):
    est = DictionaryLearning(n_components=n_components, alpha=alpha, max_iter=max_iter, tol=tol,
                             fit_algorithm=method, code_init=code_init, dict_init=dict_init,
                             random_state=random_state, positive_code=positive_code,
                             positive_dict=positive_dict, transform_max_iter=method_max_iter)
    code = est.fit_transform(X)
    if return_n_iter:
        return code, est.components_, est.error_, est.n_iter_
    else:
        return code, est.components_, est.error_

def dict_learning_online(X, n_components: int = 2, *, alpha: float = 1.0, max_iter: int = 100,
                         return_code: Literal[bool] = True, dict_init = None, callback = None,
                         batch_size: int = 256, verbose: int = 0, shuffle: bool = True,
                         n_jobs = None, method: str = "lars", random_state: Optional[int] = None,
                         positive_dict: bool = False, positive_code: bool = False,
                         method_max_iter: int = 1000, tol: float = 1e-3,
                         max_no_improvement: Optional[int] = 10):
    est = MiniBatchDictionaryLearning(n_components=n_components, alpha=alpha, max_iter=max_iter,
                                      fit_algorithm=method, batch_size=batch_size, shuffle=shuffle,
                                      dict_init=dict_init, random_state=random_state,
                                      transform_algorithm="lasso_" + method, transform_alpha=alpha,
                                      positive_code=positive_code, positive_dict=positive_dict,
                                      transform_max_iter=method_max_iter, tol=tol,
                                      max_no_improvement=max_no_improvement).fit(X)
    if return_code:
        code = est.transform(X)
        return code, est.components_
    else:
        return est.components_
