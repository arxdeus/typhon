# Copyright (C) 2026 the Typhon authors
# Ported from pyriemann 0.12 (BSD-3-Clause)

# Approximate joint diagonalization (real-valued matrices).

import numpy as np
from .mean import check_weights

def _reshape_input(X):
    n_m = X.shape[0]
    n = X.shape[1]
    A = np.empty((n, n_m * n))
    for k in range(n_m):
        A[:, k * n:(k + 1) * n] = X[k]
    return A

def _reshape_output(A, n: int, n_m: int):
    D = np.empty((n_m, n, n))
    for k in range(n_m):
        D[k] = A[:, k * n:(k + 1) * n]
    return D

def _init_v(init, n: int):
    if init is None:
        return np.eye(n)
    V = np.array(np.asarray(init, dtype=float))
    if V.shape[0] != n or V.shape[1] != n:
        raise ValueError("Init has not the good shape")
    return V

def rjd(X, init = None, eps: float = 1e-8, n_iter_max: int = 100):
    X = np.asarray(X, dtype=float)
    n_m = X.shape[0]
    A = _reshape_input(X)
    n = A.shape[0]
    N = A.shape[1]
    V = _init_v(init, n)
    for _ in range(n_iter_max):
        crit = False
        for p in range(n):
            for q in range(p + 1, n):
                g0 = A[p, p::n] - A[q, q::n]
                g1 = A[p, q::n] + A[q, p::n]
                g00 = np.dot(g0, g0)
                g11 = np.dot(g1, g1)
                g01 = np.dot(g0, g1)
                ton = g00 - g11
                toff = 2.0 * g01
                theta = 0.5 * np.arctan2(toff, ton + np.sqrt(ton ** 2 + toff ** 2))
                c = np.cos(theta)
                s = np.sin(theta)
                if abs(s) > eps:
                    crit = True
                    tmp = A[:, p::n].copy()
                    A[:, p::n] = c * A[:, p::n] + s * A[:, q::n]
                    A[:, q::n] = c * A[:, q::n] - s * tmp
                    tmp2 = A[p, :].copy()
                    A[p, :] = c * A[p, :] + s * A[q, :]
                    A[q, :] = c * A[q, :] - s * tmp2
                    tmp3 = V[:, p].copy()
                    V[:, p] = c * V[:, p] + s * V[:, q]
                    V[:, q] = c * V[:, q] - s * tmp3
        if not crit:
            break
    return V, _reshape_output(A, n, n_m)

def ajd_pham(X, init = None, eps: float = 1e-6, n_iter_max: int = 20,
             sample_weight = None):
    X = np.asarray(X, dtype=float)
    n_m = X.shape[0]
    w = check_weights(sample_weight, n_m, check_positivity=True)
    A = _reshape_input(X)
    n = A.shape[0]
    V = _init_v(init, n)
    epsilon = n * (n - 1) * eps
    for _ in range(n_iter_max):
        crit = 0.0
        for ii in range(1, n):
            for jj in range(ii):
                c1 = A[ii, ii::n]
                c2 = A[jj, jj::n]
                aij = A[ii, jj::n]
                g12 = np.sum(w * (aij / c1))
                g21 = np.sum(w * (aij / c2))
                omega21 = np.sum(w * (c1 / c2))
                omega12 = np.sum(w * (c2 / c1))
                omega = np.sqrt(omega12 * omega21)
                tmp = np.sqrt(omega21 / omega12)
                tmp1 = (tmp * g12 + g21) / (omega + 1.0)
                omega = max(omega - 1.0, 1e-9)
                tmp2 = (tmp * g12 - g21) / omega
                h12 = tmp1 + tmp2
                h21 = (tmp1 - tmp2) / tmp
                crit += n_m * (g12 * h12 + g21 * h21) / 2.0
                t = 1.0 + np.sqrt(1.0 - h12 * h21)
                tau01 = -h12 / t
                tau10 = -h21 / t
                # A[[ii, jj], :] = tau @ A[[ii, jj], :]
                ri = A[ii, :].copy()
                rj = A[jj, :].copy()
                A[ii, :] = ri + tau01 * rj
                A[jj, :] = tau10 * ri + rj
                # columns: [Ai, Aj] @ tau.T
                ci = A[:, ii::n].copy()
                cj = A[:, jj::n].copy()
                A[:, ii::n] = ci + tau01 * cj
                A[:, jj::n] = tau10 * ci + cj
                vi = V[ii, :].copy()
                vj = V[jj, :].copy()
                V[ii, :] = vi + tau01 * vj
                V[jj, :] = tau10 * vi + vj
        if crit < epsilon:
            break
    return V, _reshape_output(A, n, n_m)

def uwedge(X, init = None, eps: float = 1e-7, n_iter_max: int = 100):
    X = np.asarray(X, dtype=float)
    n_m = X.shape[0]
    M = _reshape_input(X)
    n = M.shape[0]
    if init is None:
        E, H = np.linalg.eig(M[:, 0:n])
        V = np.real(H).T / np.sqrt(np.abs(np.real(E)))[:, None]
    else:
        V = _init_v(init, n)
    Ms = M.copy()
    Rs = np.zeros((n, n_m))
    for k in range(n_m):
        blk = M[:, k * n:(k + 1) * n]
        M[:, k * n:(k + 1) * n] = 0.5 * (blk + blk.T)
        Ms[:, k * n:(k + 1) * n] = V @ M[:, k * n:(k + 1) * n] @ V.T
        Rs[:, k] = np.diagonal(Ms[:, k * n:(k + 1) * n])
    crit = np.sum(Ms ** 2) - np.sum(Rs ** 2)
    for _ in range(n_iter_max):
        B = Rs @ Rs.T
        C1 = np.zeros((n, n))
        for i in range(n):
            C1[:, i] = np.sum(Ms[:, i::n] * Rs, axis=1)
        Bd = np.diagonal(B).copy()
        D0 = B * B.T - np.outer(Bd, Bd)
        A0 = (C1 * B - Bd[:, None] * C1.T) / (D0 + np.eye(n))
        for i in range(n):
            A0[i, i] += 1.0
        V = np.linalg.solve(A0, V)
        Raux = V @ M[:, 0:n] @ V.T
        aux = 1.0 / np.sqrt(np.abs(np.diagonal(Raux)))
        V = aux[:, None] * V
        for k in range(n_m):
            Ms[:, k * n:(k + 1) * n] = V @ M[:, k * n:(k + 1) * n] @ V.T
            Rs[:, k] = np.diagonal(Ms[:, k * n:(k + 1) * n])
        crit_new = np.sum(Ms ** 2) - np.sum(Rs ** 2)
        if abs(crit_new - crit) < eps:
            break
        crit = crit_new
    return V, _reshape_output(Ms, n, n_m)

def ajd(X, method: str = "ajd_pham", init = None, eps: float = 1e-6,
        n_iter_max: int = 100):
    if method == "ajd_pham":
        return ajd_pham(X, init=init, eps=eps, n_iter_max=n_iter_max)
    elif method == "rjd":
        return rjd(X, init=init, eps=eps, n_iter_max=n_iter_max)
    elif method == "uwedge":
        return uwedge(X, init=init, eps=eps, n_iter_max=n_iter_max)
    raise ValueError("Unknown AJD method")
