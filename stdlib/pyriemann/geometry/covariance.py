# Ported from pyriemann 0.12 (BSD-3-Clause)
# Shrinkage estimators follow scikit-learn's ledoit_wolf / oas (BSD-3-Clause).
#
# Real-valued input only. Estimators 'mcd', 'hub', 'stu', 'tyl' are not
# implemented.

import numpy as np
from .base import ctranspose

def _per_matrix(X, f):
    # Apply f: (n_channels, n_times) -> (n_channels, n_channels) over all
    # leading batch dimensions.
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        return f(X)
    else:
        c = X.shape[X.ndim - 2]
        t = X.shape[X.ndim - 1]
        Xf = X.reshape(-1, c, t)
        nb = Xf.shape[0]
        out = np.empty((nb, c, c))
        for b in range(nb):
            out[b] = f(Xf[b])
        return out.reshape(X.shape[:-2] + (c, c))

def _empirical_covariance(Xs, assume_centered: bool):
    # Xs: (n_samples, n_features), as in sklearn.
    if not assume_centered:
        Xs = Xs - np.mean(Xs, axis=0)
    return (Xs.T @ Xs) / Xs.shape[0]

def _ledoit_wolf_shrinkage(Xs):
    # Xs centered, (n_samples, n_features). Single-block version of sklearn's
    # blocked computation (identical for n_features <= block_size=1000).
    n_samples, n_features = Xs.shape[0], Xs.shape[1]
    X2 = Xs * Xs
    emp_cov_trace = np.sum(X2, axis=0) / n_samples
    mu = np.sum(emp_cov_trace) / n_features
    beta_ = float(np.sum(X2.T @ X2))
    XtX = Xs.T @ Xs
    delta_ = float(np.sum(XtX * XtX)) / n_samples ** 2
    beta = 1.0 / (n_features * n_samples) * (beta_ / n_samples - delta_)
    delta = delta_ - 2.0 * mu * np.sum(emp_cov_trace) + n_features * mu ** 2
    delta /= n_features
    beta = min(beta, delta)
    return 0.0 if beta == 0 else beta / delta

def ledoit_wolf(X, assume_centered: bool = False):
    # sklearn.covariance.ledoit_wolf: X (n_samples, n_features) -> (cov, shrinkage)
    Xs = np.asarray(X, dtype=float)
    if not assume_centered:
        Xs = Xs - np.mean(Xs, axis=0)
    n_features = Xs.shape[1]
    if n_features == 1:
        return np.full((1, 1), float(np.mean(Xs * Xs))), 0.0
    shrinkage = _ledoit_wolf_shrinkage(Xs)
    emp_cov = (Xs.T @ Xs) / Xs.shape[0]
    mu = np.trace(emp_cov) / n_features
    shrunk = (1.0 - shrinkage) * emp_cov
    for i in range(n_features):
        shrunk[i, i] += shrinkage * mu
    return shrunk, shrinkage

def oas(X, assume_centered: bool = False):
    # sklearn.covariance.oas: X (n_samples, n_features) -> (cov, shrinkage)
    Xs = np.asarray(X, dtype=float)
    if not assume_centered:
        Xs = Xs - np.mean(Xs, axis=0)
    n_samples, n_features = Xs.shape[0], Xs.shape[1]
    if n_features == 1:
        return np.full((1, 1), float(np.mean(Xs * Xs))), 0.0
    emp_cov = (Xs.T @ Xs) / n_samples
    alpha = np.mean(emp_cov * emp_cov)
    mu = np.trace(emp_cov) / n_features
    mu_squared = mu ** 2
    num = alpha + mu_squared
    den = (n_samples + 1) * (alpha - mu_squared / n_features)
    shrinkage = 1.0 if den == 0 else min(num / den, 1.0)
    shrunk = (1.0 - shrinkage) * emp_cov
    for i in range(n_features):
        shrunk[i, i] += shrinkage * mu
    return shrunk, shrinkage

def _cov(x):
    return np.atleast_2d(np.cov(x))

def _corr(x):
    return np.atleast_2d(np.corrcoef(x))

def _lwf(x):
    return ledoit_wolf(x.T)[0]

def _oas(x):
    return oas(x.T)[0]

def _scm(x):
    xc = x - np.mean(x, axis=-1, keepdims=True)
    return (xc @ xc.T) / x.shape[1]

def _sch(x):
    n_times = x.shape[1]
    n = x.shape[0]
    X_c = x - np.mean(x, axis=-1, keepdims=True)
    C_scm = X_c @ X_c.T / n_times
    std = np.std(x, axis=-1)
    R = n_times / ((n_times - 1.0) * (std[:, None] * std[None, :])) * C_scm
    X_c2 = X_c * X_c
    var_R = X_c2 @ X_c2.T - n_times * C_scm * C_scm
    var = np.var(x, axis=-1)
    var_R *= n_times / ((n_times - 1) ** 3 * (var[:, None] * var[None, :]))
    for i in range(n):
        R[i, i] = 0.0
        var_R[i, i] = 0.0
    R2_sum = float(np.sum(R * R))
    g = 0.0 if R2_sum == 0 else float(np.sum(var_R)) / R2_sum
    gamma = min(max(g, 0.0), 1.0)
    out = (1.0 - gamma) * (n_times / (n_times - 1.0)) * C_scm
    for i in range(n):
        out[i, i] += gamma * (n_times / (n_times - 1.0)) * C_scm[i, i]
    return out

def covariance_scm(X, assume_centered: bool = False, weights = None):
    X = np.asarray(X, dtype=float)
    n_times = X.shape[X.ndim - 1]
    if weights is None:
        w = np.ones(n_times) / n_times
    else:
        w0 = np.asarray(weights, dtype=float)
        w = w0 / np.sum(w0)
    if not assume_centered:
        X = X - np.sum(X * w, axis=-1, keepdims=True)
    return (X * w) @ ctranspose(X)

def covariance_sch(X):
    return _per_matrix(X, _sch)

def covariances(X, estimator: str = "cov"):
    if estimator == "cov":
        return _per_matrix(X, _cov)
    elif estimator == "scm":
        return covariance_scm(X)
    elif estimator == "lwf":
        return _per_matrix(X, _lwf)
    elif estimator == "oas":
        return _per_matrix(X, _oas)
    elif estimator == "sch":
        return _per_matrix(X, _sch)
    elif estimator == "corr":
        return _per_matrix(X, _corr)
    else:
        raise ValueError(
            f"Unknown function name '{estimator}'. Must be one of corr cov hub lwf "
            "mcd oas sch scm stu tyl (mcd, hub, stu, tyl are not available in Codon)")

def normalize(X, norm: str):
    X = np.asarray(X)
    if X.shape[X.ndim - 1] != X.shape[X.ndim - 2]:
        raise ValueError("Matrices must be square")
    n = X.shape[X.ndim - 1]
    if norm == "corr":
        stddev = np.sqrt(np.abs(np.diagonal(X, axis1=-2, axis2=-1)))
        Xn = X / (stddev[..., None, :] * stddev[..., None])
        return np.clip(Xn, -1.0, 1.0)
    elif norm == "trace":
        denom = np.trace(X, axis1=-2, axis2=-1)
        return X / np.asarray(denom)[..., None, None]
    elif norm == "determinant":
        denom = np.abs(np.linalg.det(X)) ** (1.0 / n)
        return X / np.asarray(denom)[..., None, None]
    else:
        raise ValueError(f"{norm} is not a supported normalization")

def get_nondiag_weight(X):
    X = np.asarray(X)
    if X.shape[X.ndim - 1] != X.shape[X.ndim - 2]:
        raise ValueError("Matrices must be square")
    X2 = X * X
    denom = np.trace(X2, axis1=-2, axis2=-1)
    num = np.sum(X2, axis=(-2, -1)) - denom
    return 1.0 / (X.shape[X.ndim - 1] - 1) * (num / denom)
