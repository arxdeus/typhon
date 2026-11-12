# Ported from pyriemann 0.12 (BSD-3-Clause)

import numpy as np

def ctranspose(X):
    return np.swapaxes(np.conj(X), -1, -2)

def _eigvalsh(A, B):
    L = np.linalg.cholesky(B)
    Y = np.linalg.solve(L, A)
    Z = ctranspose(np.linalg.solve(L, ctranspose(Y)))
    return np.linalg.eigvalsh(Z)

def _check_square(X):
    if X.ndim < 2:
        raise ValueError("Input must be at least a 2D ndarray")
    if X.shape[X.ndim - 2] != X.shape[X.ndim - 1]:
        raise ValueError("Input must contain square matrices")
    if not np.all(np.isfinite(X)):
        raise ValueError(
            "Matrices must be positive definite. "
            "You should add regularization to avoid this error."
        )

def _matrix_operator(X, operator):
    X = np.asarray(X)
    _check_square(X)
    eigvals, eigvecs = np.linalg.eigh(X)
    eigvals = operator(eigvals)
    return eigvecs @ (eigvals[..., None] * ctranspose(eigvecs))

def expm(C):
    return _matrix_operator(C, np.exp)

def invsqrtm(C):
    return _matrix_operator(C, lambda x: 1.0 / np.sqrt(x))

def logm(C):
    return _matrix_operator(C, np.log)

def powm(C, alpha: float):
    return _matrix_operator(C, lambda x: x ** alpha)

def sqrtm(C):
    return _matrix_operator(C, np.sqrt)

def _first_divided_difference(d, fct, fctder, atol: float = 1e-12, rtol: float = 1e-12):
    di = d[..., None]
    dj = d[..., None, :]
    close_ = np.isclose(di, dj, atol=atol, rtol=rtol)
    diff = di - dj
    safe_diff = np.where(close_, np.ones_like(diff), diff)
    return np.where(close_, fctder(di + 0 * dj), (fct(di) - fct(dj)) / safe_diff)

def ddexpm(X, Cref):
    d, V = np.linalg.eigh(Cref)
    Vh = ctranspose(V)
    expfdd = _first_divided_difference(d, np.exp, np.exp)
    return V @ (expfdd * (Vh @ X @ V)) @ Vh

def ddlogm(X, Cref):
    d, V = np.linalg.eigh(Cref)
    Vh = ctranspose(V)
    logfdd = _first_divided_difference(d, np.log, lambda x: 1.0 / x)
    return V @ (logfdd * (Vh @ X @ V)) @ Vh

def nearest_sym_pos_def(X, reg: float = 1e-06):
    # Single-matrix and batched inputs are handled matrix by matrix.
    X = np.asarray(X)
    if X.ndim == 2:
        return _nearest_spd_2d(X, reg)
    else:
        out = np.empty_like(X)
        Xf = X.reshape(-1, X.shape[X.ndim - 2], X.shape[X.ndim - 1])
        Of = out.reshape(-1, X.shape[X.ndim - 2], X.shape[X.ndim - 1])
        for b in range(Xf.shape[0]):
            Of[b] = _nearest_spd_2d(Xf[b], reg)
        return out

def _nearest_spd_2d(X, reg: float):
    n = X.shape[1]
    eps = np.finfo(np.float64).eps
    A = (X + X.T) / 2
    _, s, Vh = np.linalg.svd(A)
    H = Vh.T @ (s[..., None] * Vh)
    B = (A + H) / 2
    P = (B + B.T) / 2
    eigvals = np.linalg.eigvalsh(P)
    neg_ev = bool(np.any(eigvals <= 0))
    if neg_ev:
        spacing = abs(np.linalg.norm(A)) * eps
        k = 1
        while neg_ev and k < 100:
            mineig = np.min(np.linalg.eigvalsh(P))
            P = P + (-mineig * k ** 2 + spacing) * np.eye(n)
            neg_ev = bool(np.any(np.linalg.eigvalsh(P) <= 0))
            k += 1
    ei, ev = np.linalg.eigh(P)
    if np.min(ei) / np.max(ei) < reg:
        ei_reg = ei + reg
        P = ev @ (ei_reg[..., None] * ev.T)
    return P
