# Ported from pyriemann 0.12 (BSD-3-Clause)

import numpy as np
from .base import ctranspose, _eigvalsh, sqrtm, invsqrtm, powm, logm, expm

def geodesic_euclid(A, B, alpha: float = 0.5):
    return (1 - alpha) * np.asarray(A) + alpha * np.asarray(B)

def geodesic_chol(A, B, alpha: float = 0.5):
    geo = (1 - alpha) * np.linalg.cholesky(A) + alpha * np.linalg.cholesky(B)
    return geo @ ctranspose(geo)

def geodesic_logeuclid(A, B, alpha: float = 0.5):
    return expm((1 - alpha) * logm(A) + alpha * logm(B))

def geodesic_riemann(A, B, alpha: float = 0.5):
    sA = sqrtm(A)
    isA = invsqrtm(A)
    return sA @ powm(isA @ np.asarray(B) @ isA, alpha) @ sA

def geodesic_wasserstein(A, B, alpha: float = 0.5):
    A = np.asarray(A)
    B = np.asarray(B)
    A12 = sqrtm(A)
    A12inv = invsqrtm(A)
    AB12 = A12 @ sqrtm(A12 @ B @ A12) @ A12inv
    return ((1 - alpha) ** 2 * A + alpha ** 2 * B
            + alpha * (1 - alpha) * (AB12 + ctranspose(AB12)))

def geodesic_thompson(A, B, alpha: float = 0.5):
    A = np.asarray(A)
    B = np.asarray(B)
    E = _eigvalsh(B, A)
    Emin = np.asarray(np.min(E, axis=-1))
    Emax = np.asarray(np.max(E, axis=-1))
    Emin_a = Emin ** alpha
    Emax_a = Emax ** alpha
    equal = np.asarray(np.isclose(Emin, Emax))
    den = np.asarray(np.where(equal, 1.0, Emax - Emin))
    b = (Emax_a - Emin_a)[..., None, None]
    a = (Emax * Emin_a - Emin * Emax_a)[..., None, None]
    C = (b * B + a * A) / den[..., None, None]
    return np.where(equal[..., None, None], Emin_a[..., None, None] * A, C)

def geodesic(A, B, alpha: float, metric: str = "riemann"):
    if metric == "riemann":
        return geodesic_riemann(A, B, alpha)
    elif metric == "logeuclid":
        return geodesic_logeuclid(A, B, alpha)
    elif metric == "euclid":
        return geodesic_euclid(A, B, alpha)
    elif metric == "chol":
        return geodesic_chol(A, B, alpha)
    elif metric == "logchol":
        return geodesic_logchol(A, B, alpha)
    elif metric == "thompson":
        return geodesic_thompson(A, B, alpha)
    elif metric == "wasserstein":
        return geodesic_wasserstein(A, B, alpha)
    else:
        raise ValueError(
            f"Unknown function name '{metric}'. Must be one of chol euclid logchol "
            "logeuclid riemann thompson wasserstein")

def geodesic_logchol(A, B, alpha: float = 0.5):
    A = np.asarray(A)
    B = np.asarray(B)
    Ac = np.linalg.cholesky(A)
    Bc = np.linalg.cholesky(B)
    n = A.shape[A.ndim - 1]
    geo = np.zeros(Ac.shape)
    for i in range(n):
        for j in range(i):
            geo[..., i, j] = (1.0 - alpha) * Ac[..., i, j] + alpha * Bc[..., i, j]
        geo[..., i, i] = Ac[..., i, i] ** (1.0 - alpha) * Bc[..., i, i] ** alpha
    return geo @ ctranspose(geo)
