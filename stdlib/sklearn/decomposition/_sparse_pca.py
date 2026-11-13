# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# sklearn.decomposition.SparsePCA / MiniBatchSparsePCA

import numpy as np
from ._extmath import check_random_state, svd_flip_u
from ._dict_learning import (DictionaryLearning, MiniBatchDictionaryLearning, _opt2,
                             _dict_learning)

def _normalize_rows(C: np.ndarray[float, 2]):
    for i in range(C.shape[0]):
        s = 0.0
        for j in range(C.shape[1]):
            s += C[i, j] * C[i, j]
        s = np.sqrt(s)
        if s == 0.0:
            s = 1.0
        for j in range(C.shape[1]):
            C[i, j] /= s

class _SparsePCABase:
    n_components: Optional[int]
    alpha: float
    ridge_alpha: float
    max_iter: int
    tol: float
    method: str
    random_state: Optional[int]
    components_: np.ndarray[float, 2]
    mean_: np.ndarray[float, 1]
    n_components_: int
    n_features_in_: int

    def transform(self, X):
        from ..linear_model._ridge import ridge_regression
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float)) - self.mean_
        return ridge_regression(np.ascontiguousarray(self.components_.T),
                                np.ascontiguousarray(Xa.T), self.ridge_alpha, solver="cholesky")

    def inverse_transform(self, X):
        return (np.asarray(X, dtype=float) @ self.components_) + self.mean_

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

class SparsePCA(_SparsePCABase):
    U_init: np.ndarray[float, 2]
    has_U: bool
    V_init: np.ndarray[float, 2]
    has_V: bool
    error_: List[float]
    n_iter_: int

    def __init__(self, n_components: Optional[int] = None, *, alpha: float = 1.0,
                 ridge_alpha: float = 0.01, max_iter: int = 1000, tol: float = 1e-8,
                 method: str = "lars", n_jobs = None, U_init = None, V_init = None,
                 verbose: int = 0, random_state: Optional[int] = None):
        if method != "lars" and method != "cd":
            raise ValueError(f"The 'method' parameter of SparsePCA must be a str among cd, lars. Got {method} instead.")
        self.n_components = n_components
        self.alpha = alpha
        self.ridge_alpha = ridge_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.random_state = random_state
        self.U_init, self.has_U = _opt2(U_init)
        self.V_init, self.has_V = _opt2(V_init)
        self.components_ = np.empty((0, 0))
        self.mean_ = np.empty(0)
        self.n_components_ = 0
        self.n_features_in_ = 0
        self.error_ = List[float]()
        self.n_iter_ = 0

    def fit(self, X, y = None):
        rs = check_random_state(self.random_state)
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        self.n_features_in_ = Xa.shape[1]
        self.mean_ = Xa.mean(axis=0)
        Xc = Xa - self.mean_
        nc = Xc.shape[1] if self.n_components is None else self.n_components.__val__()
        code_init = np.empty((0, 0))
        dict_init = np.empty((0, 0))
        use_init = self.has_U and self.has_V
        if use_init:
            code_init = np.ascontiguousarray(self.V_init.T)
            dict_init = np.ascontiguousarray(self.U_init.T)
        Xt = np.ascontiguousarray(Xc.T)
        code, dictionary, E, it = _dict_learning(Xt, nc, self.alpha, self.max_iter, self.tol,
                                                 "lasso_" + self.method, dict_init, use_init,
                                                 code_init, use_init, rs, False, False, 1000)
        svd_flip_u(code, dictionary)
        comp = np.ascontiguousarray(code.T)
        _normalize_rows(comp)
        self.components_ = comp
        self.n_components_ = comp.shape[0]
        self.error_ = E
        self.n_iter_ = it
        return self

class MiniBatchSparsePCA(_SparsePCABase):
    batch_size: int
    shuffle: bool
    max_no_improvement: Optional[int]
    n_iter_: float

    def __init__(self, n_components: Optional[int] = None, *, alpha: float = 1.0,
                 ridge_alpha: float = 0.01, max_iter: int = 1000, callback = None,
                 batch_size: int = 3, verbose: int = 0, shuffle: bool = True, n_jobs = None,
                 method: str = "lars", random_state: Optional[int] = None, tol: float = 1e-3,
                 max_no_improvement: Optional[int] = 10):
        if method != "lars" and method != "cd":
            raise ValueError(f"The 'method' parameter of MiniBatchSparsePCA must be a str among cd, lars. Got {method} instead.")
        self.n_components = n_components
        self.alpha = alpha
        self.ridge_alpha = ridge_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.random_state = random_state
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_no_improvement = max_no_improvement
        self.components_ = np.empty((0, 0))
        self.mean_ = np.empty(0)
        self.n_components_ = 0
        self.n_features_in_ = 0
        self.n_iter_ = 0.0

    def fit(self, X, y = None):
        rs = check_random_state(self.random_state)
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        self.n_features_in_ = Xa.shape[1]
        self.mean_ = Xa.mean(axis=0)
        Xc = Xa - self.mean_
        nc = Xc.shape[1] if self.n_components is None else self.n_components.__val__()
        est = MiniBatchDictionaryLearning(n_components=nc, alpha=self.alpha, max_iter=self.max_iter,
                                          batch_size=self.batch_size, shuffle=self.shuffle,
                                          fit_algorithm=self.method, random_state=self.random_state,
                                          transform_algorithm="lasso_" + self.method,
                                          transform_alpha=self.alpha, tol=self.tol,
                                          max_no_improvement=self.max_no_improvement)
        est._rs = rs
        Xt = np.ascontiguousarray(Xc.T)
        est.fit(Xt)
        comp = np.ascontiguousarray(est.transform(Xt).T)
        self.n_iter_ = est.n_iter_
        _normalize_rows(comp)
        self.components_ = comp
        self.n_components_ = comp.shape[0]
        return self
