# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# sklearn.decomposition.MiniBatchNMF and non_negative_factorization.

import math
import numpy as np
from ._extmath import check_random_state
from ._dict_learning import _opt2
from ._nmf import (NMF, _beta_loss_to_float, _beta_divergence, _initialize_nmf, _mu_w, EPSILON)

_F64_EPS = 2.220446049250313e-16

def _mu_h_batch(X: np.ndarray[float, 2], W: np.ndarray[float, 2], H: np.ndarray[float, 2],
                beta_loss: float, l1_reg_H: float, l2_reg_H: float, gamma: float,
                A: np.ndarray[float, 2], B: np.ndarray[float, 2], rho: float):
    """_multiplicative_update_h with the online (A, B, rho) accumulators; updates H in place."""
    numerator = np.empty((0, 0))
    denominator = np.empty((0, 0))
    if beta_loss == 2.0:
        numerator = W.T @ X
        denominator = (W.T @ W) @ H
    else:
        WHX = W @ H
        WH = WHX.copy()
        if beta_loss - 1.0 < 0:
            WH[WH < EPSILON] = EPSILON
        if beta_loss - 2.0 < 0:
            WHX[WHX < EPSILON] = EPSILON
        if beta_loss == 1.0:
            WHX = X / WHX
        elif beta_loss == 0.0:
            WHX = (WHX ** -1.0) ** 2.0 * X
        else:
            WHX = WHX ** (beta_loss - 2) * X
        numerator = W.T @ WHX
        if beta_loss == 1.0:
            W_sum = np.sum(W, axis=0)
            W_sum[W_sum == 0] = 1.0
            denominator = np.zeros_like(numerator) + W_sum[:, None]
        else:
            WH = WH ** (beta_loss - 1)
            denominator = W.T @ WH
    if l1_reg_H > 0:
        denominator += l1_reg_H
    if l2_reg_H > 0:
        denominator = denominator + l2_reg_H * H
    denominator[denominator == 0] = EPSILON
    Hn = H.copy()
    if gamma != 1:
        Hn = Hn ** (1.0 / gamma)
    numerator = numerator * Hn
    A *= rho
    B *= rho
    A += numerator
    B += denominator
    Hn = A / B
    if gamma != 1:
        Hn = Hn ** gamma
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            H[i, j] = Hn[i, j]

class MiniBatchNMF:
    n_components: Optional[int]
    init: str
    batch_size: int
    beta_loss: float
    tol: float
    max_no_improvement: Optional[int]
    max_iter: int
    alpha_W: float
    alpha_H: float
    alpha_H_same: bool
    l1_ratio: float
    forget_factor: float
    fresh_restarts: bool
    fresh_restarts_max_iter: int
    transform_max_iter: Optional[int]
    random_state: Optional[int]
    components_: np.ndarray[float, 2]
    n_components_: int
    reconstruction_err_: float
    n_iter_: int
    n_steps_: int
    n_features_in_: int
    _n_components: int
    _batch_size: int
    _rho: float
    _gamma: float
    _transform_max_iter: int
    _num: np.ndarray[float, 2]
    _den: np.ndarray[float, 2]
    _ewa_cost: float
    _has_ewa: bool
    _ewa_cost_min: float
    _has_ewa_min: bool
    _no_improvement: int
    _W0: np.ndarray[float, 2]
    _H0: np.ndarray[float, 2]
    _custom: bool

    def __init__(self, n_components: Optional[int] = None, *, init: Optional[str] = None,
                 batch_size: int = 1024, beta_loss = "frobenius", tol: float = 1e-4,
                 max_no_improvement: Optional[int] = 10, max_iter: int = 200,
                 alpha_W: float = 0.0, alpha_H = "same", l1_ratio: float = 0.0,
                 forget_factor: float = 0.7, fresh_restarts: bool = False,
                 fresh_restarts_max_iter: int = 30, transform_max_iter: Optional[int] = None,
                 random_state: Optional[int] = None, verbose: int = 0):
        self.n_components = n_components
        self.init = "" if init is None else init.__val__()
        self.batch_size = batch_size
        self.beta_loss = _beta_loss_to_float(beta_loss)
        self.tol = tol
        self.max_no_improvement = max_no_improvement
        self.max_iter = max_iter
        self.alpha_W = alpha_W
        if isinstance(alpha_H, str):
            self.alpha_H_same = True
            self.alpha_H = alpha_W
        else:
            self.alpha_H_same = False
            self.alpha_H = float(alpha_H)
        self.l1_ratio = l1_ratio
        self.forget_factor = forget_factor
        self.fresh_restarts = fresh_restarts
        self.fresh_restarts_max_iter = fresh_restarts_max_iter
        self.transform_max_iter = transform_max_iter
        self.random_state = random_state
        self.components_ = np.empty((0, 0))
        self.n_components_ = 0
        self.reconstruction_err_ = 0.0
        self.n_iter_ = 0
        self.n_steps_ = 0
        self.n_features_in_ = 0
        self._n_components = 0
        self._batch_size = 0
        self._rho = 0.0
        self._gamma = 1.0
        self._transform_max_iter = 0
        self._num = np.empty((0, 0))
        self._den = np.empty((0, 0))
        self._ewa_cost = 0.0
        self._has_ewa = False
        self._ewa_cost_min = 0.0
        self._has_ewa_min = False
        self._no_improvement = 0
        self._W0 = np.empty((0, 0))
        self._H0 = np.empty((0, 0))
        self._custom = False

    def _check_params(self, X: np.ndarray[float, 2]):
        self._n_components = X.shape[1] if self.n_components is None else self.n_components.__val__()
        self._batch_size = min(self.batch_size, X.shape[0])
        self._rho = self.forget_factor ** (float(self._batch_size) / float(X.shape[0]))
        b = self.beta_loss
        if b < 1:
            self._gamma = 1.0 / (2.0 - b)
        elif b > 2:
            self._gamma = 1.0 / (b - 1.0)
        else:
            self._gamma = 1.0
        self._transform_max_iter = self.max_iter if self.transform_max_iter is None else self.transform_max_iter.__val__()

    def _regularization(self, X: np.ndarray[float, 2]):
        n_samples, n_features = X.shape
        aH = self.alpha_W if self.alpha_H_same else self.alpha_H
        l1W = float(n_features) * self.alpha_W * self.l1_ratio
        l1H = float(n_samples) * aH * self.l1_ratio
        l2W = float(n_features) * self.alpha_W * (1.0 - self.l1_ratio)
        l2H = float(n_samples) * aH * (1.0 - self.l1_ratio)
        return l1W, l1H, l2W, l2H

    def _solve_W(self, X: np.ndarray[float, 2], H: np.ndarray[float, 2], max_iter: int) -> np.ndarray[float, 2]:
        avg = float(np.sqrt(X.mean() / float(self._n_components)))
        W = np.full((X.shape[0], self._n_components), avg)
        Wb = W.copy()
        l1W, _, l2W, _ = self._regularization(X)
        for _i in range(max_iter):
            _mu_w(X, W, H, self.beta_loss, l1W, l2W, self._gamma)
            diff = float(np.sqrt(np.sum((W - Wb) ** 2))) / float(np.sqrt(np.sum(W ** 2)))
            if self.tol > 0 and diff <= self.tol:
                break
            Wb = W.copy()
        return W

    def _minibatch_step(self, X: np.ndarray[float, 2], W0: np.ndarray[float, 2], has_W: bool,
                        H: np.ndarray[float, 2], update_H: bool):
        batch_size = X.shape[0]
        l1W, l1H, l2W, l2H = self._regularization(X)
        W = W0
        if self.fresh_restarts or not has_W:
            W = self._solve_W(X, H, self.fresh_restarts_max_iter)
        else:
            _mu_w(X, W, H, self.beta_loss, l1W, l2W, self._gamma)
        if self.beta_loss < 1:
            W[W < _F64_EPS] = 0.0
        cost = (_beta_divergence(X, W, H, self.beta_loss) + l1W * float(np.sum(W)) + l1H * float(np.sum(H))
                + l2W * float(np.sum(W ** 2)) + l2H * float(np.sum(H ** 2))) / float(batch_size)
        if update_H:
            _mu_h_batch(X, W, H, self.beta_loss, l1H, l2H, self._gamma, self._num, self._den, self._rho)
            if self.beta_loss <= 1:
                H[H < _F64_EPS] = 0.0
        return W, cost

    def _convergence(self, batch_size: int, cost: float, H: np.ndarray[float, 2],
                     Hb: np.ndarray[float, 2], n_samples: int, step0: int) -> bool:
        step = step0 + 1
        if step == 1:
            return False
        if not self._has_ewa:
            self._ewa_cost = cost
            self._has_ewa = True
        else:
            a = min(float(batch_size) / float(n_samples + 1), 1.0)
            self._ewa_cost = self._ewa_cost * (1.0 - a) + cost * a
        hd = float(np.sqrt(np.sum((H - Hb) ** 2))) / float(np.sqrt(np.sum(H ** 2)))
        if self.tol > 0 and hd <= self.tol:
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

    def _fit_transform(self, X: np.ndarray[float, 2], update_H: bool, H_in: np.ndarray[float, 2]):
        if float(X.min()) < 0.0:
            raise ValueError("Negative values in data passed to MiniBatchNMF (input X)")
        self._check_params(X)
        if X.min() == 0 and self.beta_loss <= 0:
            raise ValueError("When beta_loss <= 0 and X contains zeros, the solver may diverge. Please add small values to X, or use a positive beta_loss.")
        n_samples = X.shape[0]
        W = np.empty((0, 0))
        H = np.empty((0, 0))
        if self._custom:
            W = self._W0.copy()
            H = self._H0.copy()
        else:
            W, H = _initialize_nmf(X, self._n_components, self.init, self.random_state)
        Hb = H.copy()
        self._num = H.copy()
        self._den = np.ones(H.shape)
        self._has_ewa = False
        self._has_ewa_min = False
        self._no_improvement = 0
        bs = self._batch_size
        nb = (n_samples + bs - 1) // bs
        n_steps = self.max_iter * nb
        i = -1
        for step in range(n_steps):
            i = step
            b = step % nb
            lo = b * bs
            hi = min(n_samples, lo + bs)
            Xb = np.ascontiguousarray(X[lo:hi])
            Wb = np.ascontiguousarray(W[lo:hi])
            Wn, cost = self._minibatch_step(Xb, Wb, True, H, update_H)
            for r in range(hi - lo):
                for c in range(W.shape[1]):
                    W[lo + r, c] = Wn[r, c]
            if update_H and self._convergence(Xb.shape[0], cost, H, Hb, n_samples, step):
                break
            Hb = H.copy()
        if self.fresh_restarts:
            W = self._solve_W(X, H, self._transform_max_iter)
        n_steps_done = i + 1
        n_iter = -((-n_steps_done) // nb)
        return W, H, n_iter, n_steps_done

    def fit_transform(self, X, y = None):
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        self.n_features_in_ = Xa.shape[1]
        W, H, n_iter, n_steps = self._fit_transform(Xa, True, np.empty((0, 0)))
        self.reconstruction_err_ = _beta_divergence(Xa, W, H, self.beta_loss, square_root=True)
        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_iter_ = n_iter
        self.n_steps_ = n_steps
        return W

    def fit(self, X, y = None):
        self.fit_transform(X)
        return self

    def transform(self, X):
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        if float(Xa.min()) < 0.0:
            raise ValueError("Negative values in data passed to MiniBatchNMF (input X)")
        return self._solve_W(Xa, self.components_, self._transform_max_iter)

    def inverse_transform(self, X):
        return np.asarray(X, dtype=float) @ self.components_

    def partial_fit(self, X, y = None):
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        if float(Xa.min()) < 0.0:
            raise ValueError("Negative values in data passed to MiniBatchNMF (input X)")
        H = np.empty((0, 0))
        if self.n_steps_ == 0 and self.components_.shape[0] == 0:
            self.n_features_in_ = Xa.shape[1]
            self._check_params(Xa)
            _, H = _initialize_nmf(Xa, self._n_components, self.init, self.random_state)
            self._num = H.copy()
            self._den = np.ones(H.shape)
            self._has_ewa = False
            self._has_ewa_min = False
        else:
            H = self.components_
        dummy = np.empty((0, 0))
        self._minibatch_step(Xa, dummy, False, H, True)
        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_steps_ += 1
        return self

def non_negative_factorization(X, W = None, H = None, n_components = None, *, init: Optional[str] = None,
                               update_H: bool = True, solver: str = "cd", beta_loss = "frobenius",
                               tol: float = 1e-4, max_iter: int = 200, alpha_W: float = 0.0,
                               alpha_H = "same", l1_ratio: float = 0.0,
                               random_state: Optional[int] = None, verbose: int = 0,
                               shuffle: bool = False):
    Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
    if float(Xa.min()) < 0.0:
        raise ValueError("Negative values in data passed to NMF (input X)")
    nc: Optional[int] = None
    if n_components is not None and not isinstance(n_components, str):
        nc = int(n_components)
    est = NMF(n_components=nc, init=init, solver=solver, beta_loss=beta_loss, tol=tol,
              max_iter=max_iter, random_state=random_state, alpha_W=alpha_W, alpha_H=alpha_H,
              l1_ratio=l1_ratio, shuffle=shuffle)
    H0, hasH = _opt2(H)
    W0, hasW = _opt2(W)
    if hasH and nc is None:
        est.n_components = H0.shape[0]
    if est.init == "custom" and update_H:
        est._custom = True
        est._W0 = W0
        est._H0 = H0
    if not update_H and not hasH:
        raise ValueError("H must be given when update_H=False")
    Wr, Hr, n_iter = est._fit_transform(Xa, H0, update_H)
    return Wr, Hr, n_iter
