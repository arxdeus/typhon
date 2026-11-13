# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Dense pairwise kernels used by kernel_ridge / kernel_approximation /
# gaussian_process / semi_supervised (sklearn.metrics.pairwise formulas, same
# operation order).

import numpy as np
from .decomposition._extmath import euclidean_distances_sq, row_norms

def _k_gamma(gamma, p: int) -> float:
    if gamma is None:
        return 1.0 / float(p)
    return float(gamma)

def manhattan_distances(X, Y):
    n = X.shape[0]; m = Y.shape[0]; p = X.shape[1]
    D = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            s = 0.0
            for k in range(p):
                s += abs(X[i, k] - Y[j, k])
            D[i, j] = s
    return D

def additive_chi2_kernel(X, Y):
    # _chi2_kernel_fast
    n = X.shape[0]; m = Y.shape[0]; p = X.shape[1]
    R = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            res = 0.0
            for k in range(p):
                den = X[i, k] + Y[j, k]
                nom = X[i, k] - Y[j, k]
                if den != 0:
                    res += -nom * nom / den
            R[i, j] = res
    return R

def kernel_matrix(X, Y, metric: str, gamma = None, degree = 3.0, coef0 = 1.0, same: bool = False):
    """pairwise_kernels(X, Y, metric, filter_params=True, gamma, degree, coef0)."""
    Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
    Ya = np.ascontiguousarray(np.asarray(Y, dtype=float))
    p = Xa.shape[1]
    if metric == "linear":
        return Xa @ Ya.T
    elif metric == "poly" or metric == "polynomial":
        K = Xa @ Ya.T
        K *= _k_gamma(gamma, p)
        K += float(coef0)
        K **= float(degree)
        return K
    elif metric == "rbf":
        K = euclidean_distances_sq(Xa, Ya, same)
        K *= -_k_gamma(gamma, p)
        np.exp(K, out=K)
        return K
    elif metric == "sigmoid":
        K = Xa @ Ya.T
        K *= _k_gamma(gamma, p)
        K += float(coef0)
        np.tanh(K, out=K)
        return K
    elif metric == "laplacian":
        K = -_k_gamma(gamma, p) * manhattan_distances(Xa, Ya)
        np.exp(K, out=K)
        return K
    elif metric == "cosine":
        nx = row_norms(Xa)
        ny = row_norms(Ya)
        Xn = Xa / np.where(nx == 0.0, 1.0, nx)[:, None]
        Yn = Ya / np.where(ny == 0.0, 1.0, ny)[:, None]
        return Xn @ Yn.T
    elif metric == "chi2":
        K = additive_chi2_kernel(Xa, Ya)
        K *= 1.0 if gamma is None else float(gamma)
        np.exp(K, out=K)
        return K
    elif metric == "additive_chi2":
        return additive_chi2_kernel(Xa, Ya)
    else:
        raise ValueError(f"Unknown kernel {metric!r}")
