# Ported from pyriemann 0.12 (BSD-3-Clause)

import numpy as np
from .base import ctranspose, expm, invsqrtm, logm, sqrtm, ddexpm, ddlogm

def exp_map_euclid(X, Cref):
    return np.asarray(X) + np.asarray(Cref)

def exp_map_logeuclid(X, Cref):
    return expm(logm(Cref) + ddlogm(X, Cref))

def exp_map_riemann(X, Cref, Cm12: bool = False):
    X = np.asarray(X)
    if Cm12:
        C = invsqrtm(Cref)
        X = C @ X @ C
    C12 = sqrtm(Cref)
    return C12 @ expm(X) @ C12

def exp_map_wasserstein(X, Cref):
    X = np.asarray(X)
    Cref = np.asarray(Cref)
    d, V = np.linalg.eigh(Cref)
    Vh = ctranspose(V)
    C = 1 / (d[:, None] + d[None, :])
    X_tmp = C * (Vh @ X @ V)
    X_tmp = X_tmp @ (d[..., None] * X_tmp)
    X_tmp = V @ X_tmp @ Vh
    return Cref + X + X_tmp

def exp_map(X, Cref, metric: str = "riemann"):
    if metric == "riemann":
        return exp_map_riemann(X, Cref)
    elif metric == "logeuclid":
        return exp_map_logeuclid(X, Cref)
    elif metric == "euclid":
        return exp_map_euclid(X, Cref)
    elif metric == "wasserstein":
        return exp_map_wasserstein(X, Cref)
    else:
        raise ValueError(
            f"Unknown function name '{metric}'. Must be one of euclid logchol "
            "logeuclid riemann wasserstein")

def log_map_euclid(X, Cref):
    return np.asarray(X) - np.asarray(Cref)

def log_map_logeuclid(X, Cref):
    logCref = logm(Cref)
    return ddexpm(logm(X) - logCref, logCref)

def log_map_riemann(X, Cref, C12: bool = False):
    Cm12 = invsqrtm(Cref)
    X_new = logm(Cm12 @ np.asarray(X) @ Cm12)
    if C12:
        S = sqrtm(Cref)
        X_new = S @ X_new @ S
    return X_new

def log_map_wasserstein(X, Cref):
    Cref = np.asarray(Cref)
    P12 = sqrtm(Cref)
    P12inv = invsqrtm(Cref)
    sqrt_bracket = sqrtm(P12 @ np.asarray(X) @ P12)
    tmp = P12inv @ sqrt_bracket @ P12
    return tmp + ctranspose(tmp) - 2 * Cref

def log_map(X, Cref, metric: str = "riemann"):
    if metric == "riemann":
        return log_map_riemann(X, Cref)
    elif metric == "logeuclid":
        return log_map_logeuclid(X, Cref)
    elif metric == "euclid":
        return log_map_euclid(X, Cref)
    elif metric == "wasserstein":
        return log_map_wasserstein(X, Cref)
    else:
        raise ValueError(
            f"Unknown function name '{metric}'. Must be one of euclid logchol "
            "logeuclid riemann wasserstein")

def upper(X):
    X = np.asarray(X)
    n = X.shape[X.ndim - 1]
    if X.shape[X.ndim - 2] != n:
        raise ValueError("Matrices must be square")
    m = n * (n + 1) // 2
    Xf = X.reshape(-1, n, n)
    nb = Xf.shape[0]
    Tf = np.zeros((nb, m), dtype=X.dtype)
    s2 = 2.0 ** 0.5
    for b in range(nb):
        k = 0
        for i in range(n):
            Tf[b, k] = Xf[b, i, i]
            k += 1
            for j in range(i + 1, n):
                Tf[b, k] = s2 * Xf[b, i, j]
                k += 1
    # Row-major triu order: (0,0),(0,1),...,(1,1),... as in np.triu_indices.
    return Tf.reshape(X.shape[:-2] + (m,))

def unupper(T):
    T = np.asarray(T)
    m = T.shape[T.ndim - 1]
    n = int(((1 + 8 * m) ** 0.5 - 1) / 2)
    Tf = T.reshape(-1, m)
    nb = Tf.shape[0]
    Xf = np.zeros((nb, n, n), dtype=T.dtype)
    s2 = 2.0 ** 0.5
    for b in range(nb):
        k = 0
        for i in range(n):
            Xf[b, i, i] = Tf[b, k]
            k += 1
            for j in range(i + 1, n):
                v = Tf[b, k] / s2
                Xf[b, i, j] = v
                Xf[b, j, i] = np.conj(v)
                k += 1
    return Xf.reshape(T.shape[:-1] + (n, n))

def tangent_space(X, Cref, metric: str = "riemann"):
    return upper(log_map(X, Cref, metric=metric))

def untangent_space(T, Cref, metric: str = "riemann"):
    return exp_map(unupper(T), Cref, metric=metric)

def _apply_inner_product(X, Y):
    G = np.real(np.sum(np.conj(X) * Y, axis=(-2, -1)))
    return G

def innerproduct_euclid(X, Y, Cref = None):
    X = np.asarray(X)
    if Y is None:
        return _apply_inner_product(X, X)
    else:
        return _apply_inner_product(X, np.asarray(Y))

def innerproduct_logeuclid(X, Y, Cref):
    X_ = ddlogm(X, Cref)
    if Y is None:
        return _apply_inner_product(X_, X_)
    else:
        return _apply_inner_product(X_, ddlogm(Y, Cref))

def innerproduct_riemann(X, Y, Cref):
    Cm12 = invsqrtm(Cref)
    X_ = Cm12 @ np.asarray(X) @ Cm12
    if Y is None:
        return _apply_inner_product(X_, X_)
    else:
        return _apply_inner_product(X_, Cm12 @ np.asarray(Y) @ Cm12)

def innerproduct(X, Y, Cref, metric: str = "riemann"):
    if metric == "riemann":
        return innerproduct_riemann(X, Y, Cref)
    elif metric == "logeuclid":
        return innerproduct_logeuclid(X, Y, Cref)
    elif metric == "euclid":
        return innerproduct_euclid(X, Y, Cref)
    else:
        raise ValueError(
            f"Unknown function name '{metric}'. Must be one of euclid logchol "
            "logeuclid riemann")

def norm(X, Cref, metric: str = "riemann"):
    return np.sqrt(innerproduct(X, None, Cref, metric=metric))

def transport_euclid(X, A = None, B = None):
    return X

def transport_logeuclid(X, A, B):
    return ddexpm(ddlogm(X, A), logm(B))

def transport_riemann(X, A, B):
    A12 = sqrtm(A)
    A12inv = invsqrtm(A)
    E = A12 @ sqrtm(A12inv @ np.asarray(B) @ A12inv) @ A12inv
    return E @ np.asarray(X) @ ctranspose(E)

def transport(X, A, B, metric: str = "riemann"):
    if metric == "riemann":
        return transport_riemann(X, A, B)
    elif metric == "logeuclid":
        return transport_logeuclid(X, A, B)
    elif metric == "euclid":
        return np.asarray(X) * 1.0
    else:
        raise ValueError(
            f"Unknown function name '{metric}'. Must be one of euclid logchol "
            "logeuclid riemann")
