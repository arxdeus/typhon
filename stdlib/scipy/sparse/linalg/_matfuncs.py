# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# norm, onenormest, expm, expm_multiply, matrix_power for sparse inputs.
# onenormest returns the exact 1-norm (scipy's Higham-Tisseur estimate is
# exact for the small block sizes it is usually applied to, and is only an
# estimate otherwise). expm densifies (scipy uses a sparse Pade
# approximation; values agree to rounding) and returns CSC.

import math
import numpy as np
from .._base import _compressed, _coo, _up, _is_cplx
from .._construct import csc_array, csc_matrix, csr_array, eye_array
from ._interface import aslinearoperator, _is_linop

def _abs_sums(x, axis: int):
    c = x.tocoo()
    m, n = c.shape
    r = np.zeros(n if axis == 0 else m)
    for k in range(len(c.data)):
        if axis == 0:
            r[c.col[k]] += abs(c.data[k])
        else:
            r[c.row[k]] += abs(c.data[k])
    return r

def norm(x, ord = None, axis = None):
    if not (isinstance(x, _compressed) or isinstance(x, _coo)):
        compile_error("input is not sparse. use numpy.linalg.norm")
    A = x.tocsr()
    A.sum_duplicates()
    if axis is None:
        if ord is None or isinstance(ord, str):
            if ord is not None:
                if ord != "fro" and ord != "f":
                    raise ValueError("Invalid norm order for matrices.")
            s = 0.0
            for v in A.data:
                s += abs(v) * abs(v)
            return math.sqrt(s)
        else:
            o = float(ord)
            if o == 1.0:
                return float(_abs_sums(A, 0).max())
            elif o == math.inf:
                return float(_abs_sums(A, 1).max())
            elif o == -1.0:
                return float(_abs_sums(A, 0).min())
            elif o == -math.inf:
                return float(_abs_sums(A, 1).min())
            elif o == 2.0:
                from ._eigen import svds
                s = svds(A, k=1, return_singular_vectors=False)
                return float(s[0])
            else:
                raise ValueError("Invalid norm order for matrices.")
                return 0.0
    else:
        a = axis if axis >= 0 else axis + 2
        m, n = A.shape
        c = A.tocoo()
        L = n if a == 0 else m
        o = 2.0 if ord is None else float(ord)
        res = np.zeros(L)
        if o == math.inf or o == -math.inf:
            cnt = np.zeros(L, dtype=int)
            init = np.zeros(L, dtype=bool)
            for k in range(len(c.data)):
                q = c.col[k] if a == 0 else c.row[k]
                v = abs(c.data[k])
                cnt[q] += 1
                if not init[q]:
                    res[q] = v
                    init[q] = True
                elif (v > res[q]) if o > 0 else (v < res[q]):
                    res[q] = v
            full = m if a == 0 else n
            for q in range(L):
                if cnt[q] < full:
                    if o > 0:
                        res[q] = max(res[q], 0.0)
                    else:
                        res[q] = 0.0
            return res
        if o == 0.0:
            for k in range(len(c.data)):
                if c.data[k] != type(c.data[k])():
                    q = c.col[k] if a == 0 else c.row[k]
                    res[q] += 1.0
            return res
        for k in range(len(c.data)):
            q = c.col[k] if a == 0 else c.row[k]
            res[q] += abs(c.data[k]) ** o
        for q in range(L):
            res[q] = res[q] ** (1.0 / o)
        return res

def onenormest(A, t: int = 2, itmax: int = 5, compute_v: bool = False,
               compute_w: bool = False):
    if isinstance(A, _compressed) or isinstance(A, _coo):
        return float(_abs_sums(A, 0).max())
    elif isinstance(A, np.ndarray):
        return float(np.abs(A).sum(axis=0).max())
    else:
        op = aslinearoperator(A)
        n = op.shape[1]
        best = 0.0
        for j in range(n):
            e = np.zeros(n, dtype=op.dtype)
            e[j] = op.dtype(1.0)
            y = op.matvec(e)
            s = 0.0
            for v in y:
                s += abs(v)
            best = max(best, s)
        return best

def expm(A):
    from scipy.linalg import expm as dexpm
    D = A.toarray()
    E = dexpm(D)
    if A._mstatic():
        return csc_matrix(E)
    else:
        return csc_array(E)

def matrix_power(A, power: int):
    m, n = A.shape
    if m != n:
        raise TypeError("sparse matrix is not square")
    if power < 0:
        raise ValueError("exponent must be >= 0")
    if power == 0:
        return eye_array(m, dtype=A.data.dtype).asformat("csr")
    R = A.tocsr()
    result = R.copy()
    base = R.copy()
    p = power - 1
    while p > 0:
        if p & 1:
            result = result @ base
        base = base @ base
        p >>= 1
    return result

_THETA_M = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 35, 40, 45, 50, 55]
_THETA_V = [2.29e-16, 2.58e-8, 1.39e-5, 3.40e-4, 2.40e-3, 9.07e-3, 2.38e-2,
            5.00e-2, 8.96e-2, 1.44e-1, 2.14e-1, 3.00e-1, 4.00e-1, 5.14e-1,
            6.41e-1, 7.81e-1, 9.31e-1, 1.09, 1.26, 1.44, 1.62, 1.82, 2.01,
            2.22, 2.43, 2.64, 2.86, 3.08, 3.31, 3.54, 4.7, 6.0, 7.2, 8.5, 9.9]

def _theta(m: int) -> float:
    for i in range(len(_THETA_M)):
        if _THETA_M[i] == m:
            return _THETA_V[i]
    return -1.0

def _compute_p_max(m_max: int) -> int:
    sq = math.sqrt(m_max)
    p_low = int(math.floor(sq))
    p_high = int(math.ceil(sq + 1))
    best = 0
    for p in range(p_low, p_high + 1):
        if p * (p - 1) <= m_max + 1:
            best = max(best, p)
    return best

def _inf_norm_dense(B) -> float:
    if B.ndim == 1:
        s = 0.0
        for v in B:
            s = max(s, abs(v))
        return s
    else:
        best = 0.0
        for i in range(B.shape[0]):
            s = 0.0
            for j in range(B.shape[1]):
                s += abs(B[i, j])
            best = max(best, s)
        return best

def expm_multiply(A, B, start = None, stop = None, num = None, endpoint = None,
                  traceA = None):
    if start is not None or stop is not None or num is not None:
        raise NotImplementedError("expm_multiply: only the simple form (no time grid) is supported")
    S = A.tocsr()
    n = S.shape[0]
    if S.shape[0] != S.shape[1]:
        raise ValueError("expected A to be like a square matrix")
    Bv = np.asarray(B)
    if Bv.shape[0] != n:
        raise ValueError("shapes of matrices A and B are incompatible")
    T = type(_up(type(_up(S.data.dtype, Bv.dtype)), float))
    n0 = 1 if Bv.ndim == 1 else Bv.shape[1]
    tol = 2.0 ** -53
    t = 1.0
    tr = S.trace() if traceA is None else traceA
    mu = T(tr) / T(float(n))
    I = eye_array(n, dtype=T).tocsr()
    Am = S.astype(T) - I * mu
    A1 = float(_abs_sums(Am, 0).max())
    m_star = 0
    s = 1
    if t * A1 != 0:
        ell = 2
        m_max = 55
        p_max = _compute_p_max(m_max)
        a = 2 * ell * p_max * (p_max + 3)
        b = _theta(m_max) / float(n0 * m_max)
        best_m = -1
        best_s = -1
        if A1 <= a * b:
            for i in range(len(_THETA_M)):
                mm = _THETA_M[i]
                ss = int(math.ceil(A1 / _THETA_V[i]))
                if best_m < 0 or mm * ss < best_m * best_s:
                    best_m = mm
                    best_s = ss
        else:
            dcache = Dict[int, float]()
            def dp(p: int) -> float:
                if p not in dcache:
                    P = matrix_power(Am, p)
                    dcache[p] = float(_abs_sums(P, 0).max()) ** (1.0 / p)
                return dcache[p]
            for p in range(2, p_max + 1):
                for mm in range(p * (p - 1) - 1, m_max + 1):
                    th = _theta(mm)
                    if th > 0:
                        alpha = max(dp(p), dp(p + 1))
                        ss = int(math.ceil(alpha / th))
                        if best_m < 0 or mm * ss < best_m * best_s:
                            best_m = mm
                            best_s = ss
            best_s = max(best_s, 1)
        m_star = best_m
        s = best_s
    Bc = Bv.astype(T)
    F = Bc.copy()
    eta = T(np.exp(t * mu / float(s))) if _is_cplx(T) else math.exp(float(t * mu) / float(s))
    Bk = Bc.copy()
    for i in range(s):
        c1 = _inf_norm_dense(Bk)
        for j in range(m_star):
            coeff = t / float(s * (j + 1))
            Bk = (Am @ Bk) * T(coeff)
            c2 = _inf_norm_dense(Bk)
            F = F + Bk
            if c1 + c2 <= tol * _inf_norm_dense(F):
                break
            c1 = c2
        F = F * T(eta)
        Bk = F.copy()
    return F
