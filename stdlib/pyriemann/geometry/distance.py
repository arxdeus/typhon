# Ported from pyriemann 0.12 (BSD-3-Clause)

import numpy as np
from .base import _eigvalsh, invsqrtm, logm, powm, sqrtm, ctranspose

def _fro2(D):
    # Squared Frobenius norm over the last two axes.
    a = np.abs(D)
    return np.sum(a * a, axis=(-2, -1))

def _sqrt_or_sq(d2, squared: bool):
    return d2 if squared else np.sqrt(d2)

def _clip0(x):
    if isinstance(x, float):
        return x if x > 0.0 else 0.0
    else:
        return np.clip(x, 0.0, None)

def _check_pair(A, B):
    if A.ndim < 2 or B.ndim < 2:
        raise ValueError("Inputs must be at least 2D arrays")
    if (A.shape[A.ndim - 2] != B.shape[B.ndim - 2] or
            A.shape[A.ndim - 1] != B.shape[B.ndim - 1]):
        raise ValueError("Inputs must have equal matrix dimensions")

def _trace(X):
    return np.trace(X, axis1=-2, axis2=-1)

def _real(x):
    if isinstance(x, complex):
        return x.real
    elif isinstance(x, float) or isinstance(x, int):
        return float(x)
    else:
        return np.real(x)

def distance_euclid(A, B, squared: bool = False):
    A = np.asarray(A)
    B = np.asarray(B)
    _check_pair(A, B)
    d2 = _fro2(A - B)
    return _sqrt_or_sq(d2, squared)

def distance_chol(A, B, squared: bool = False):
    return distance_euclid(np.linalg.cholesky(A), np.linalg.cholesky(B), squared=squared)

def distance_harmonic(A, B, squared: bool = False):
    A = np.asarray(A)
    B = np.asarray(B)
    eye_n = np.eye(A.shape[A.ndim - 1], dtype=A.dtype)
    return distance_euclid(np.linalg.solve(A, eye_n), np.linalg.solve(B, eye_n),
                           squared=squared)

def distance_kullback(A, B, squared: bool = False):
    A = np.asarray(A)
    B = np.asarray(B)
    _check_pair(A, B)
    n = A.shape[A.ndim - 1]
    tr = _trace(np.linalg.solve(B, A))
    logdet = np.linalg.slogdet(B)[1] - np.linalg.slogdet(A)[1]
    d = 0.5 * _real(tr - n + logdet)
    return d ** 2 if squared else d

def distance_kullback_right(A, B, squared: bool = False):
    return distance_kullback(B, A, squared=squared)

def distance_kullback_sym(A, B, squared: bool = False):
    d = distance_kullback(A, B) + distance_kullback_right(A, B)
    return d ** 2 if squared else d

def distance_logdet(A, B, squared: bool = False):
    A = np.asarray(A)
    B = np.asarray(B)
    _check_pair(A, B)
    logdet_ApB = np.linalg.slogdet((A + B) / 2.0)[1]
    logdet_AxB = np.linalg.slogdet(A @ B)[1]
    d2 = _clip0(logdet_ApB - 0.5 * logdet_AxB)
    return _sqrt_or_sq(d2, squared)

def distance_logeuclid(A, B, squared: bool = False):
    return distance_euclid(logm(A), logm(B), squared=squared)

def distance_poweuclid(A, B, p: float, squared: bool = False):
    if p == 1:
        return distance_euclid(A, B, squared=squared)
    if p == 0:
        return distance_logeuclid(A, B, squared=squared)
    if p == -1:
        return distance_harmonic(A, B, squared=squared)
    return distance_euclid(powm(A, p), powm(B, p), squared=squared) / abs(p)

def distance_riemann(A, B, squared: bool = False):
    A = np.asarray(A)
    B = np.asarray(B)
    _check_pair(A, B)
    lv = np.log(_eigvalsh(A, B))
    d2 = np.sum(lv * lv, axis=-1)
    return _sqrt_or_sq(d2, squared)

def distance_thompson(A, B, squared: bool = False):
    A = np.asarray(A)
    B = np.asarray(B)
    _check_pair(A, B)
    d = np.max(np.abs(np.log(_eigvalsh(A, B))), axis=-1)
    return d ** 2 if squared else d

def distance_wasserstein(A, B, squared: bool = False):
    A = np.asarray(A)
    B = np.asarray(B)
    _check_pair(A, B)
    B12 = sqrtm(B)
    d2 = _clip0(_real(_trace(A + B - 2 * sqrtm(B12 @ A @ B12))))
    return _sqrt_or_sq(d2, squared)

def _distance_dispatch(A, B, metric: str, squared: bool):
    if metric == "riemann":
        return distance_riemann(A, B, squared=squared)
    elif metric == "logeuclid":
        return distance_logeuclid(A, B, squared=squared)
    elif metric == "euclid":
        return distance_euclid(A, B, squared=squared)
    elif metric == "chol":
        return distance_chol(A, B, squared=squared)
    elif metric == "harmonic":
        return distance_harmonic(A, B, squared=squared)
    elif metric == "kullback":
        return distance_kullback(A, B, squared=squared)
    elif metric == "kullback_right":
        return distance_kullback_right(A, B, squared=squared)
    elif metric == "kullback_sym":
        return distance_kullback_sym(A, B, squared=squared)
    elif metric == "logdet":
        return distance_logdet(A, B, squared=squared)
    elif metric == "thompson":
        return distance_thompson(A, B, squared=squared)
    elif metric == "wasserstein":
        return distance_wasserstein(A, B, squared=squared)
    else:
        raise ValueError(
            f"Unknown function name '{metric}'. Must be one of chol euclid harmonic "
            "kullback kullback_right kullback_sym logchol logdet logeuclid riemann "
            "thompson wasserstein")

def distance(A, B, metric: str = "riemann", squared: bool = False):
    A = np.asarray(A)
    B = np.asarray(B)
    if A.ndim == 3 and B.ndim == 2:
        return _distance_dispatch(A, B, metric, squared)[..., None]
    elif A.ndim == B.ndim:
        if A.shape != B.shape:
            raise ValueError("Inputs have incompatible dimensions.")
        return _distance_dispatch(A, B, metric, squared)
    else:
        compile_error("Inputs have incompatible dimensions.")

def _flat(X):
    return X.reshape(X.shape[0], -1)

def _euclid_pairs(X, Y, squared: bool):
    # Direct computation of (squared) Euclidean distances between rows.
    nx = X.shape[0]
    ny = Y.shape[0]
    out = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            s = 0.0
            for k in range(X.shape[1]):
                a = abs(X[i, k] - Y[j, k])
                s += a * a
            out[i, j] = s if squared else s ** 0.5
    return out

def _pairwise_distance_euclid(X, Y, squared: bool):
    if Y is None:
        Xf = _flat(X)
        return _euclid_pairs(Xf, Xf, squared)
    else:
        return _euclid_pairs(_flat(X), _flat(Y), squared)

def _pairwise_distance_riemann(X, Y, squared: bool):
    nx = X.shape[0]
    Xinv12 = invsqrtm(X)
    if Y is None:
        dist = np.zeros((nx, nx))
        for i in range(nx):
            x_ = Xinv12[i]
            ev = np.linalg.eigvalsh(x_ @ X[i:] @ x_)
            lv = np.log(ev)
            dist[i, i:] = np.sum(lv * lv, axis=-1)
        dist = dist + dist.T
        return dist if squared else np.sqrt(dist)
    else:
        ny = Y.shape[0]
        dist = np.zeros((nx, ny))
        for i in range(nx):
            x_ = Xinv12[i]
            ev = np.linalg.eigvalsh(x_ @ Y @ x_)
            lv = np.log(ev)
            dist[i, :] = np.sum(lv * lv, axis=-1)
        return dist if squared else np.sqrt(dist)

def pairwise_distance(X, Y = None, metric: str = "riemann", squared: bool = False):
    X = np.asarray(X)
    if metric == "euclid":
        return _pairwise_distance_euclid(X, Y, squared)
    elif metric == "harmonic":
        eye_n = np.eye(X.shape[X.ndim - 1], dtype=X.dtype)
        if Y is None:
            return _pairwise_distance_euclid(np.linalg.solve(X, eye_n), None, squared)
        else:
            return _pairwise_distance_euclid(np.linalg.solve(X, eye_n),
                                             np.linalg.solve(np.asarray(Y), eye_n), squared)
    elif metric == "logeuclid":
        if Y is None:
            return _pairwise_distance_euclid(logm(X), None, squared)
        else:
            return _pairwise_distance_euclid(logm(X), logm(np.asarray(Y)), squared)
    elif metric == "riemann":
        if Y is None:
            return _pairwise_distance_riemann(X, None, squared)
        else:
            return _pairwise_distance_riemann(X, np.asarray(Y), squared)
    nx = X.shape[0]
    if Y is None:
        dist = np.zeros((nx, nx))
        if metric == "kullback" or metric == "kullback_right":
            for i in range(nx):
                for j in range(nx):
                    dist[i, j] = _distance_dispatch(X[i], X[j], metric, squared)
        else:
            for i in range(nx):
                for j in range(i + 1, nx):
                    dist[i, j] = _distance_dispatch(X[i], X[j], metric, squared)
            dist = dist + dist.T
        return dist
    else:
        Ya = np.asarray(Y)
        ny = Ya.shape[0]
        dist = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                dist[i, j] = _distance_dispatch(X[i], Ya[j], metric, squared)
        return dist

def distance_mahalanobis(X, cov, mean = None, squared: bool = False):
    X = np.asarray(X)
    if mean is not None:
        X = X - np.asarray(mean)
    Xw = invsqrtm(cov) @ X
    a = np.abs(Xw)
    d2 = np.sum(a * a, axis=-2)
    return d2 if squared else np.sqrt(d2)
