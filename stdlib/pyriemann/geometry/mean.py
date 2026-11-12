# Ported from pyriemann 0.12 (BSD-3-Clause)
#
# Means take a stack of matrices X with shape (n_matrices, n, n). Leading
# batch dimensions beyond that (pyriemann's _vectorize_nd) are not supported.

import numpy as np
from .base import ctranspose, sqrtm, invsqrtm, logm, expm, powm
from .geodesic import geodesic_riemann
from .tangentspace import log_map_wasserstein, exp_map_wasserstein

def check_weights(weights, n_weights: int, check_positivity: bool = False):
    if weights is None:
        w = np.ones(n_weights, dtype=float)
        return w / np.sum(w)
    else:
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or w.shape[0] != n_weights:
            raise ValueError(
                f"Weights do not have the good shape. Should be ({n_weights},) "
                f"but got {w.shape}.")
        if check_positivity and np.any(w <= 0):
            raise ValueError("Weights must be strictly positive.")
        return w / np.sum(w)

def _check_init(init, n: int):
    M = np.asarray(init, dtype=float)
    if M.ndim != 2 or M.shape[0] != n or M.shape[1] != n:
        raise ValueError(
            f"Init matrix does not have the good shape. Should be ({n},{n}) "
            f"but got {M.shape}.")
    return M

def _wsum(w, X):
    # Weighted sum over the first axis, equivalent to
    # np.tensordot(w, X, axes=([0], [0])) (avoids a tensordot shape bug).
    out = w[0] * X[0]
    for i in range(1, X.shape[0]):
        out = out + w[i] * X[i]
    return out

def _check3(X):
    X = np.asarray(X)
    if X.ndim != 3:
        compile_error("pyriemann mean: X must have shape (n_matrices, n, n)")
    return X

def _warn():
    # pyriemann emits a UserWarning here. Typhon has no warnings module, so
    # the message is printed to stderr instead.
    print("UserWarning: Convergence not reached", file=sys.stderr)

import sys

def mean_euclid(X, sample_weight = None):
    X = np.asarray(X)
    if sample_weight is None:
        return np.mean(X, axis=-3)
    else:
        if X.ndim != 3:
            compile_error("weighted mean_euclid: X must have shape (n_matrices, n, n)")
        w = check_weights(sample_weight, X.shape[0])
        return _wsum(w, X)

def mean_chol(X, sample_weight = None):
    L = mean_euclid(np.linalg.cholesky(X), sample_weight=sample_weight)
    return L @ ctranspose(L)

def mean_harmonic(X, sample_weight = None):
    X = np.asarray(X)
    eye_n = np.eye(X.shape[X.ndim - 1], dtype=X.dtype)
    X_inv = np.linalg.solve(X, eye_n)
    M_inv = mean_euclid(X_inv, sample_weight=sample_weight)
    return np.linalg.solve(M_inv, eye_n)

def mean_kullback_sym(X, sample_weight = None):
    M_euclid = mean_euclid(X, sample_weight=sample_weight)
    M_harmonic = mean_harmonic(X, sample_weight=sample_weight)
    return geodesic_riemann(M_euclid, M_harmonic, 0.5)

def mean_logeuclid(X, sample_weight = None):
    return expm(mean_euclid(logm(X), sample_weight=sample_weight))

def mean_poweuclid(X, p: float, sample_weight = None):
    if p == 1:
        return mean_euclid(X, sample_weight=sample_weight)
    if p == 0:
        return mean_logeuclid(X, sample_weight=sample_weight)
    if p == -1:
        return mean_harmonic(X, sample_weight=sample_weight)
    return powm(mean_euclid(powm(X, p), sample_weight=sample_weight), 1 / p)

def mean_logdet(X, tol: float = 0.0001, maxiter: int = 50, init = None,
                sample_weight = None):
    X = _check3(X)
    n_matrices, n = X.shape[0], X.shape[2]
    w = check_weights(sample_weight, n_matrices)
    M = mean_euclid(X, sample_weight=w) if init is None else _check_init(init, n)
    eye_n = np.eye(n, dtype=X.dtype)
    converged = False
    for _ in range(maxiter):
        X_inv = np.linalg.solve(0.5 * X + 0.5 * M, eye_n)
        J = _wsum(w, X_inv)
        Mnew = np.linalg.solve(J, eye_n)
        crit = np.linalg.norm(Mnew - M, 'fro')
        M = Mnew
        if crit <= tol:
            converged = True
            break
    if not converged:
        _warn()
    return M

def mean_riemann(X, tol: float = 1e-08, maxiter: int = 50, init = None,
                 sample_weight = None):
    X = _check3(X)
    n_matrices, n = X.shape[0], X.shape[2]
    w = check_weights(sample_weight, n_matrices)
    M = mean_euclid(X, sample_weight=w) if init is None else _check_init(init, n)
    nu = 1.0
    tau = np.finfo(np.float64).max
    converged = False
    for _ in range(maxiter):
        M12 = sqrtm(M)
        Mm12 = invsqrtm(M)
        J = _wsum(w, logm(Mm12 @ X @ Mm12))
        M = M12 @ expm(nu * J) @ M12
        crit = np.linalg.norm(J, 'fro')
        h = nu * crit
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu
        if crit <= tol or nu <= tol:
            converged = True
            break
    if not converged:
        _warn()
    return M

def mean_power(X, p: float, sample_weight = None, zeta: float = 1e-09,
               maxiter: int = 100, init = None):
    X = _check3(X)
    if p < -1 or 1 < p:
        raise ValueError("Exponent p must be in [-1,+1]")
    if p == 1:
        return mean_euclid(X, sample_weight=sample_weight)
    if p == 0:
        return mean_riemann(X, sample_weight=sample_weight, init=init, tol=zeta,
                            maxiter=maxiter)
    if p == -1:
        return mean_harmonic(X, sample_weight=sample_weight)
    n_matrices, n = X.shape[0], X.shape[2]
    w = check_weights(sample_weight, n_matrices)
    phi = 0.375 / abs(p)
    G = powm(_wsum(w, powm(X, p)), 1 / p) if init is None else _check_init(init, n)
    K = invsqrtm(G) if p > 0 else sqrtm(G)
    eye_n = np.eye(n)
    sqrt_n = n ** 0.5
    sgn = 1.0 if p > 0 else -1.0
    Xs = powm(X, sgn)
    converged = False
    for _ in range(maxiter):
        H = _wsum(w, powm(K @ Xs @ ctranspose(K), abs(p)))
        K = powm(H, -phi) @ K
        crit = np.linalg.norm(H - eye_n) / sqrt_n
        if crit <= zeta:
            converged = True
            break
    if not converged:
        _warn()
    M = ctranspose(K) @ K
    if p > 0:
        M = np.linalg.solve(M, eye_n)
    return M

def mean_wasserstein(X, tol: float = 1e-08, maxiter: int = 50, init = None,
                     sample_weight = None):
    X = _check3(X)
    n_matrices, n = X.shape[0], X.shape[2]
    w = check_weights(sample_weight, n_matrices)
    M = mean_euclid(X, sample_weight=w) if init is None else _check_init(init, n)
    converged = False
    for _ in range(maxiter):
        J = _wsum(w, log_map_wasserstein(X, M))
        M = exp_map_wasserstein(J, M)
        crit = np.linalg.norm(J)
        if crit <= tol:
            converged = True
            break
    if not converged:
        _warn()
    return M

def gmean(X, metric: str = "riemann", sample_weight = None):
    if metric == "riemann":
        return mean_riemann(X, sample_weight=sample_weight)
    elif metric == "logeuclid":
        return mean_logeuclid(X, sample_weight=sample_weight)
    elif metric == "euclid":
        return mean_euclid(X, sample_weight=sample_weight)
    elif metric == "chol":
        return mean_chol(X, sample_weight=sample_weight)
    elif metric == "harmonic":
        return mean_harmonic(X, sample_weight=sample_weight)
    elif metric == "kullback_sym":
        return mean_kullback_sym(X, sample_weight=sample_weight)
    elif metric == "logdet":
        return mean_logdet(X, sample_weight=sample_weight)
    elif metric == "wasserstein":
        return mean_wasserstein(X, sample_weight=sample_weight)
    else:
        raise ValueError(
            f"Unknown function name '{metric}'. Must be one of chol euclid harmonic "
            "kullback_sym logdet logeuclid riemann wasserstein")

def mean_covariance(X, metric: str = "riemann", sample_weight = None):
    # Deprecated in pyriemann 0.12. Note: pyriemann 0.12 ignores both
    # `metric` and `sample_weight` here (it always calls the riemann mean
    # unweighted); this behaviour is reproduced.
    return gmean(X, metric="riemann", sample_weight=None)
