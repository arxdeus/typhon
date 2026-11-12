# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

import numpy as np
from ._extmath import check_random_state, randomized_svd, squared_norm

EPSILON = 1.1920928955078125e-07  # np.finfo(np.float32).eps
_F64_EPS = 2.220446049250313e-16

def _beta_loss_to_float(beta_loss) -> float:
    if isinstance(beta_loss, str):
        if beta_loss == "frobenius":
            return 2.0
        elif beta_loss == "kullback-leibler":
            return 1.0
        elif beta_loss == "itakura-saito":
            return 0.0
        raise ValueError(f"invalid beta_loss {beta_loss}")
    else:
        return float(beta_loss)

def _beta_divergence(X, W, H, beta: float, square_root: bool = False) -> float:
    if beta == 2.0:
        res = squared_norm(X - W @ H) / 2.0
        if square_root:
            return float(np.sqrt(res * 2))
        return res
    WH = W @ H
    Xf = X.ravel()
    WHf = WH.ravel()
    n = 0
    for i in range(Xf.shape[0]):
        if Xf[i] > EPSILON:
            n += 1
    Xd = np.empty(n)
    Wd = np.empty(n)
    j = 0
    for i in range(Xf.shape[0]):
        if Xf[i] > EPSILON:
            Xd[j] = Xf[i]
            Wd[j] = WHf[i] if WHf[i] >= EPSILON else EPSILON
            j += 1
    res = 0.0
    if beta == 1.0:
        sum_WH = float(np.dot(np.sum(W, axis=0), np.sum(H, axis=1)))
        div = Xd / Wd
        res = float(np.dot(Xd, np.log(div)))
        res += sum_WH - float(Xd.sum())
    elif beta == 0.0:
        div = Xd / Wd
        res = float(np.sum(div)) - float(X.shape[0] * X.shape[1]) - float(np.sum(np.log(div)))
    else:
        sum_WH_beta = float(np.sum(WH ** beta))
        sum_X_WH = float(np.dot(Xd, Wd ** (beta - 1)))
        res = float((Xd ** beta).sum()) - beta * sum_X_WH
        res += sum_WH_beta * (beta - 1)
        res /= beta * (beta - 1)
    if square_root:
        return float(np.sqrt(2 * max(res, 0.0)))
    return res

def _initialize_nmf(X, n_components: int, init: str, random_state: Optional[int], eps: float = 1e-6):
    n_samples, n_features = X.shape
    if init != "" and init != "random" and n_components > min(n_samples, n_features):
        raise ValueError(f"init = '{init}' can only be used when n_components <= min(n_samples, n_features)")
    ini = init
    if ini == "":
        ini = "nndsvda" if n_components <= min(n_samples, n_features) else "random"
    if ini == "random":
        avg = np.sqrt(X.mean() / n_components)
        rng = check_random_state(random_state)
        H = avg * rng.standard_normal(size=(n_components, n_features))
        W = avg * rng.standard_normal(size=(n_samples, n_components))
        return np.abs(W), np.abs(H)
    U, S, V = randomized_svd(X, n_components, check_random_state(random_state))
    W = np.zeros_like(U)
    H = np.zeros_like(V)
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])
    for j in range(1, n_components):
        x = U[:, j]
        y = V[j, :]
        x_p = np.maximum(x, 0.0)
        y_p = np.maximum(y, 0.0)
        x_n = np.abs(np.minimum(x, 0.0))
        y_n = np.abs(np.minimum(y, 0.0))
        x_p_nrm = float(np.sqrt(np.dot(x_p, x_p)))
        y_p_nrm = float(np.sqrt(np.dot(y_p, y_p)))
        x_n_nrm = float(np.sqrt(np.dot(x_n, x_n)))
        y_n_nrm = float(np.sqrt(np.dot(y_n, y_n)))
        m_p = x_p_nrm * y_p_nrm
        m_n = x_n_nrm * y_n_nrm
        u = x_p
        v = y_p
        sigma = 0.0
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n
        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v
    W[W < eps] = 0.0
    H[H < eps] = 0.0
    if ini == "nndsvd":
        pass
    elif ini == "nndsvda":
        avg = X.mean()
        W[W == 0.0] = avg
        H[H == 0.0] = avg
    elif ini == "nndsvdar":
        rng = check_random_state(random_state)
        avg = X.mean()
        nw = int(np.sum(W == 0.0))
        W[W == 0.0] = np.abs(avg * rng.standard_normal(size=nw) / 100)
        nh = int(np.sum(H == 0.0))
        H[H == 0.0] = np.abs(avg * rng.standard_normal(size=nh) / 100)
    else:
        raise ValueError(f"Invalid init parameter: got {ini}")
    return W, H

def _update_cdnmf_fast(W, HHt, XHt, permutation) -> float:
    violation = 0.0
    n_components = W.shape[1]
    n_samples = W.shape[0]
    for s in range(n_components):
        t = int(permutation[s])
        for i in range(n_samples):
            grad = -XHt[i, t]
            for r in range(n_components):
                grad += HHt[t, r] * W[i, r]
            pg = min(0.0, grad) if W[i, t] == 0 else grad
            violation += abs(pg)
            hess = HHt[t, t]
            if hess != 0:
                W[i, t] = max(W[i, t] - grad / hess, 0.0)
    return violation

def _update_coordinate_descent(X, W, Ht, l1_reg: float, l2_reg: float, shuffle: bool, rng) -> float:
    n_components = Ht.shape[1]
    HHt = Ht.T @ Ht
    XHt = X @ Ht
    if l2_reg != 0.0:
        for i in range(n_components):
            HHt[i, i] += l2_reg
    if l1_reg != 0.0:
        XHt -= l1_reg
    perm = rng.permutation(n_components) if shuffle else np.arange(n_components)
    return _update_cdnmf_fast(W, HHt, XHt, perm)

def _mu_w(X, W, H, beta_loss: float, l1_reg_W: float, l2_reg_W: float, gamma: float):
    numerator = np.empty((0, 0))
    denominator = np.empty((0, 0))
    if beta_loss == 2.0:
        numerator = X @ H.T
        denominator = W @ (H @ H.T)
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
        numerator = WHX @ H.T
        if beta_loss == 1.0:
            H_sum = np.sum(H, axis=1)
            denominator = np.zeros_like(numerator) + H_sum[None, :]
        else:
            WH = WH ** (beta_loss - 1)
            denominator = WH @ H.T
    if l1_reg_W > 0:
        denominator += l1_reg_W
    if l2_reg_W > 0:
        denominator = denominator + l2_reg_W * W
    denominator[denominator == 0] = EPSILON
    numerator /= denominator
    if gamma != 1:
        numerator = numerator ** gamma
    W *= numerator

def _mu_h(X, W, H, beta_loss: float, l1_reg_H: float, l2_reg_H: float, gamma: float):
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
    numerator /= denominator
    if gamma != 1:
        numerator = numerator ** gamma
    H *= numerator

class NMF:
    """solver 'cd' (frobenius) / 'mu' (any beta_loss), init None/'random'/
    'nndsvd'/'nndsvda'/'nndsvdar', alpha_W/alpha_H/l1_ratio regularization."""
    n_components: Optional[int]
    init: str
    solver: str
    beta_loss: float
    tol: float
    max_iter: int
    random_state: Optional[int]
    alpha_W: float
    alpha_H: float
    alpha_H_same: bool
    l1_ratio: float
    shuffle: bool
    components_: np.ndarray[float, 2]
    n_components_: int
    n_iter_: int
    reconstruction_err_: float

    def __init__(self, n_components: Optional[int] = None, *, init: Optional[str] = None,
                 solver: str = "cd", beta_loss = "frobenius", tol: float = 1e-4,
                 max_iter: int = 200, random_state: Optional[int] = None, alpha_W: float = 0.0,
                 alpha_H = "same", l1_ratio: float = 0.0, verbose: int = 0, shuffle: bool = False):
        self.n_components = n_components
        self.init = "" if init is None else init.__val__()
        self.solver = solver
        self.beta_loss = _beta_loss_to_float(beta_loss)
        if solver == "cd" and self.beta_loss != 2.0:
            raise ValueError("Invalid beta_loss parameter: solver 'cd' does not handle beta_loss != 'frobenius'")
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha_W = alpha_W
        if isinstance(alpha_H, str):
            self.alpha_H_same = True
            self.alpha_H = alpha_W
        else:
            self.alpha_H_same = False
            self.alpha_H = float(alpha_H)
        self.l1_ratio = l1_ratio
        self.shuffle = shuffle
        self.components_ = np.empty((0, 0))
        self.n_components_ = 0
        self.n_iter_ = 0
        self.reconstruction_err_ = 0.0

    def _fit_transform(self, X, H0, update_H: bool):
        n_samples, n_features = X.shape
        k = n_features if self.n_components is None else self.n_components.__val__()
        if X.min() == 0 and self.beta_loss <= 0:
            raise ValueError("When beta_loss <= 0 and X contains zeros, the solver may diverge.")
        W = np.empty((0, 0))
        H = H0
        if not update_H:
            if self.solver == "mu":
                W = np.full((n_samples, k), float(np.sqrt(X.mean() / k)))
            else:
                W = np.zeros((n_samples, k))
        else:
            W, H = _initialize_nmf(X, k, self.init, self.random_state)
        aH = self.alpha_W if self.alpha_H_same else self.alpha_H
        l1W = n_features * self.alpha_W * self.l1_ratio
        l1H = n_samples * aH * self.l1_ratio
        l2W = n_features * self.alpha_W * (1.0 - self.l1_ratio)
        l2H = n_samples * aH * (1.0 - self.l1_ratio)
        n_iter = 0
        if self.solver == "cd":
            W = np.ascontiguousarray(W)
            Ht = np.ascontiguousarray(H.T)
            Xt = np.ascontiguousarray(X.T)
            rng = check_random_state(self.random_state)
            violation_init = 0.0
            for it in range(1, self.max_iter + 1):
                n_iter = it
                violation = _update_coordinate_descent(X, W, Ht, l1W, l2W, self.shuffle, rng)
                if update_H:
                    violation += _update_coordinate_descent(Xt, Ht, W, l1H, l2H, self.shuffle, rng)
                if it == 1:
                    violation_init = violation
                if violation_init == 0:
                    break
                if violation / violation_init <= self.tol:
                    break
            H = np.ascontiguousarray(Ht.T)
        elif self.solver == "mu":
            b = self.beta_loss
            gamma = 1.0
            if b < 1:
                gamma = 1.0 / (2.0 - b)
            elif b > 2:
                gamma = 1.0 / (b - 1.0)
            error_at_init = _beta_divergence(X, W, H, b, square_root=True)
            previous_error = error_at_init
            for it in range(1, self.max_iter + 1):
                n_iter = it
                _mu_w(X, W, H, b, l1W, l2W, gamma)
                if b < 1:
                    W[W < _F64_EPS] = 0.0
                if update_H:
                    _mu_h(X, W, H, b, l1H, l2H, gamma)
                    if b <= 1:
                        H[H < _F64_EPS] = 0.0
                if self.tol > 0 and it % 10 == 0:
                    error = _beta_divergence(X, W, H, b, square_root=True)
                    if (previous_error - error) / error_at_init < self.tol:
                        break
                    previous_error = error
        else:
            raise ValueError(f"Invalid solver parameter '{self.solver}'.")
        return W, H, n_iter

    def fit_transform(self, X, y = None):
        Xa = np.asarray(X, dtype=float)
        if Xa.min() < 0:
            raise ValueError("Negative values in data passed to NMF (input X)")
        W, H, n_iter = self._fit_transform(Xa, np.empty((0, 0)), True)
        self.reconstruction_err_ = _beta_divergence(Xa, W, H, self.beta_loss, square_root=True)
        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_iter_ = n_iter
        return W

    def fit(self, X, y = None):
        self.fit_transform(X)
        return self

    def transform(self, X):
        Xa = np.asarray(X, dtype=float)
        W, H, n = self._fit_transform(Xa, self.components_, False)
        return W

    def inverse_transform(self, X):
        return np.asarray(X, dtype=float) @ self.components_
