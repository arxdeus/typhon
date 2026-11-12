# Ported from scikit-learn 1.9.1 (BSD-3-Clause)
# Shared numeric helpers (sklearn.utils.extmath subset) for the
# decomposition / manifold / mixture ports.

import numpy as np
import scipy.linalg as sl

def check_random_state(seed):
    if isinstance(seed, np.random.RandomState):
        return seed
    else:
        return np.random.RandomState(seed)

def svd_flip_v(u, v):
    """svd_flip(u, v, u_based_decision=False): rows of v decide."""
    for r in range(v.shape[0]):
        best = 0
        for c in range(v.shape[1]):
            if abs(v[r, c]) > abs(v[r, best]):
                best = c
        if v[r, best] < 0:
            for c in range(v.shape[1]):
                v[r, c] = -v[r, c]
            for i in range(u.shape[0]):
                u[i, r] = -u[i, r]

def svd_flip_u(u, v):
    """svd_flip(u, v, u_based_decision=True): columns of u decide."""
    for r in range(u.shape[1]):
        best = 0
        for i in range(u.shape[0]):
            if abs(u[i, r]) > abs(u[best, r]):
                best = i
        if u[best, r] < 0:
            for i in range(u.shape[0]):
                u[i, r] = -u[i, r]
            if r < v.shape[0]:
                for c in range(v.shape[1]):
                    v[r, c] = -v[r, c]

def svd_flip_u_only(u):
    e = np.empty((0, 0))
    svd_flip_u(u, e)

def randomized_range_finder(A, size: int, n_iter: int, normalizer: str, rs):
    Q = rs.normal(size=(A.shape[1], size))
    norm = normalizer
    if norm == "auto":
        norm = "none" if n_iter <= 2 else "LU"
    for _ in range(n_iter):
        if norm == "LU":
            Q = sl.lu(A @ Q, permute_l=True)[0]
            Q = sl.lu(A.T @ Q, permute_l=True)[0]
        elif norm == "QR":
            Q = sl.qr(A @ Q, mode='economic')[0]
            Q = sl.qr(A.T @ Q, mode='economic')[0]
        else:
            Q = A @ Q
            Q = A.T @ Q
    Q = sl.qr(A @ Q, mode='economic')[0]
    return Q

def randomized_svd(M0, n_components: int, rs, n_oversamples: int = 10, n_iter: int = -1,
                   normalizer: str = "auto", flip_sign: bool = True):
    """_randomized_svd with transpose='auto'; n_iter=-1 means 'auto'."""
    M = np.asarray(M0, dtype=float)
    n_random = n_components + n_oversamples
    n_samples, n_features = M.shape
    it = n_iter
    if it < 0:
        it = 7 if n_components < 0.1 * min(n_samples, n_features) else 4
    transpose = n_samples < n_features
    if transpose:
        M = np.ascontiguousarray(M.T)
    Q = randomized_range_finder(M, n_random, it, normalizer, rs)
    B = Q.T @ M
    Uhat, s, Vt = sl.svd(B, full_matrices=False)
    U = Q @ Uhat
    if flip_sign:
        if not transpose:
            svd_flip_u(U, Vt)
        else:
            svd_flip_v(U, Vt)
    if transpose:
        return (np.ascontiguousarray(Vt[:n_components, :].T), s[:n_components].copy(),
                np.ascontiguousarray(U[:, :n_components].T))
    return (np.ascontiguousarray(U[:, :n_components]), s[:n_components].copy(),
            np.ascontiguousarray(Vt[:n_components, :]))

def squared_norm(x) -> float:
    a = np.asarray(x, dtype=float).ravel()
    return float(np.dot(a, a))

def _row_sq_einsum(X):
    """np.einsum('ij,ij->i', X, X) for float64, bit-exact with numpy's NEON
    sum-of-products kernel: 2 lanes, 8-element blocks consumed highest pair
    first with fused multiply-adds, zero-padded tail, then lane0 + lane1."""
    n, p = X.shape
    out = np.empty(n)
    for r in range(n):
        a0 = 0.0
        a1 = 0.0
        i = 0
        while p - i >= 8:
            blk = 3
            while blk >= 0:
                v0 = X[r, i + 2 * blk]
                v1 = X[r, i + 2 * blk + 1]
                a0 = float._fma(v0, v0, a0)
                a1 = float._fma(v1, v1, a1)
                blk -= 1
            i += 8
        while i < p:
            v0 = X[r, i]
            v1 = X[r, i + 1] if i + 1 < p else 0.0
            a0 = float._fma(v0, v0, a0)
            a1 = float._fma(v1, v1, a1)
            i += 2
        out[r] = 0.0 + (a0 + a1)
    return out

def row_norms(X, squared: bool = False):
    r = _row_sq_einsum(X)
    if not squared:
        r = np.sqrt(r)
    return r

def fast_logdet(A) -> float:
    s, ld = np.linalg.slogdet(A)
    if not s > 0:
        return -np.inf
    return float(ld)

def logsumexp_rows(a):
    """log(sum(exp(a), axis=1)) as scipy.special.logsumexp."""
    n = a.shape[0]
    out = np.empty(n)
    for i in range(n):
        m = -np.inf
        for j in range(a.shape[1]):
            if a[i, j] > m:
                m = a[i, j]
        if m == -np.inf:
            out[i] = -np.inf
            continue
        s = 0.0
        for j in range(a.shape[1]):
            s += np.exp(a[i, j] - m)
        out[i] = np.log(s) + m
    return out

def euclidean_distances_sq(X, Y, same: bool = False):
    """sklearn euclidean_distances(X, Y, squared=True) for float64
    (same=True: Y is X, diagonal forced to 0)."""
    XX = row_norms(X, squared=True)
    YY = row_norms(Y, squared=True)
    D = -2.0 * (X @ Y.T)
    D += XX[:, None]
    D += YY[None, :]
    np.maximum(D, 0.0, out=D)
    if same:
        for i in range(min(D.shape[0], D.shape[1])):
            D[i, i] = 0.0
    return D

def pairwise_kernel(X, Y, kernel: str, gamma: float, degree: float, coef0: float,
                    same: bool = False):
    if kernel == "linear":
        return X @ Y.T
    elif kernel == "poly":
        K = X @ Y.T
        K *= gamma
        K += coef0
        return K ** degree
    elif kernel == "rbf":
        K = euclidean_distances_sq(X, Y, same)
        K *= -gamma
        return np.exp(K)
    elif kernel == "sigmoid":
        K = X @ Y.T
        K *= gamma
        K += coef0
        return np.tanh(K)
    elif kernel == "cosine":
        nx = row_norms(X)
        ny = row_norms(Y)
        Xn = X / np.where(nx == 0.0, 1.0, nx)[:, None]
        Yn = Y / np.where(ny == 0.0, 1.0, ny)[:, None]
        return Xn @ Yn.T
    else:
        raise ValueError(f"unknown kernel {kernel}")
