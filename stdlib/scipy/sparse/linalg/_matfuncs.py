# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# norm, onenormest (_onenormest.py), expm (scipy's sparse Pade algorithm
# _expm with _ell / onenormest-based degree selection), expm_multiply and
# matrix_power for sparse inputs.

import math
import numpy as np
from .._base import _compressed, _coo, _up, _is_cplx, _issparse
from .._construct import csc_array, csc_matrix, csr_array, eye_array
from ._interface import aslinearoperator, _is_linop, LinearOperator
from ._onenormest import onenormest

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
    if not (_issparse(x)):
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

def _exact_1_norm(A) -> float:
    # max column abs sum (sparse or dense)
    if _issparse(A):
        return float(_abs_sums(A, 0).max())
    else:
        return float(np.abs(A).sum(axis=0).max())

def _onenorm_matrix_power_nnm(A, p: int) -> float:
    # 1-norm of A^p for non-negative A (scipy: v = (A.T)^p @ ones)
    n = A.shape[0]
    v = np.ones(n, dtype=float)
    M = A.T.tocsr()
    for _ in range(p):
        v = M @ v
    mx = v[0]
    for i in range(n):
        mx = max(mx, v[i])
    return mx

_ELL_C = (100800., 10059033600., 4487938430976000., 5914384781877411840000.,
          113250775606021113483283660800000000.)

def _ell(A, m: int) -> int:
    ci = _ELL_C[0] if m == 3 else (_ELL_C[1] if m == 5 else (
        _ELL_C[2] if m == 7 else (_ELL_C[3] if m == 9 else _ELL_C[4])))
    u = 2.0 ** -53
    nrm = _onenorm_matrix_power_nnm(abs(A), 2 * m + 1)
    if nrm == 0.0:
        return 0
    alpha = nrm / (_exact_1_norm(A) * ci)
    value = int(math.ceil(math.log2(alpha / u) / (2 * m)))
    return max(value, 0)

def _onenormest_power(A, p: int) -> float:
    # scipy _onenormest_matrix_power: onenormest(MatrixPowerOperator(A, p))
    C = A.tocsr()
    CT = A.T.tocsr()
    def mv(x):
        y = x
        for _ in range(p):
            y = C @ y
        return y
    def rmv(x):
        y = x
        for _ in range(p):
            y = CT @ y
        return y
    def mm(X):
        Y = X
        for _ in range(p):
            Y = C @ Y
        return Y
    def rmm(X):
        Y = X
        for _ in range(p):
            Y = CT @ Y
        return Y
    op = LinearOperator(A.shape, mv, rmv, mm, float, rmm)
    return float(onenormest(op))

def _onenormest_prod(A, B) -> float:
    # onenormest(ProductOperator(A, B)) = onenormest of A @ B
    Ac = A.tocsr()
    Bc = B.tocsr()
    AT = A.T.tocsr()
    BT = B.T.tocsr()
    op = LinearOperator(A.shape, lambda x: Ac @ (Bc @ x), lambda x: BT @ (AT @ x),
                        lambda X: Ac @ (Bc @ X), float, lambda X: BT @ (AT @ X))
    return float(onenormest(op))

def _sparse_expm(A):
    # scipy _expm for sparse input (use_exact_onenorm='auto': n < 200),
    # result in the format of A (U, V built from the CSR/CSC products;
    # P/Q solved by spsolve, giving A's compressed format)
    from ._dsolve import spsolve
    n = A.shape[0]
    T = type(_up(A.dtype, float))
    Af = A.astype(T)
    if n == 1:
        c = Af.tocoo()
        v = T(0)
        for k in range(len(c.data)):
            v += c.data[k]
        return csc_array(np.array([[np.exp(v)]], dtype=T)) if not A._mstatic() \
            else csc_array(np.array([[np.exp(v)]], dtype=T))
    exact = n < 200
    upper = True
    c0 = Af.tocoo()
    for k in range(len(c0.data)):
        if c0.row[k] > c0.col[k] and c0.data[k] != T():
            upper = False
    I = eye_array(n, dtype=T).asformat("csr")
    Ac = Af.tocsr()
    A2 = Ac @ Ac
    A4 = A2 @ A2
    A6 = A4 @ A2
    d4t = _exact_1_norm(A4) ** 0.25
    d6t = _exact_1_norm(A6) ** (1. / 6.)
    d4l = d4t if exact else _onenormest_power(A2, 2) ** 0.25
    d6l = d6t if exact else _onenormest_power(A2, 3) ** (1. / 6.)
    eta_1 = max(d4l, d6l)
    if eta_1 < 1.495585217958292e-002 and _ell(Ac, 3) == 0:
        b3 = (120., 60., 12., 1.)
        U = Ac @ (b3[3] * A2 + b3[1] * I)
        V = b3[2] * A2 + b3[0] * I
        return _solve_PQ(U, V, spsolve)
    eta_2 = max(d4t, d6l)
    if eta_2 < 2.539398330063230e-001 and _ell(Ac, 5) == 0:
        b5 = (30240., 15120., 3360., 420., 30., 1.)
        U = Ac @ (b5[5] * A4 + b5[3] * A2 + b5[1] * I)
        V = b5[4] * A4 + b5[2] * A2 + b5[0] * I
        return _solve_PQ(U, V, spsolve)
    A8 = A6 @ A2
    d8t = _exact_1_norm(A8) ** 0.125
    d8l = d8t if exact else _onenormest_power(A4, 2) ** 0.125
    eta_3 = max(d6t, d8l)
    if eta_3 < 9.504178996162932e-001 and _ell(Ac, 7) == 0:
        b7 = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
        U = Ac @ (b7[7] * A6 + b7[5] * A4 + b7[3] * A2 + b7[1] * I)
        V = b7[6] * A6 + b7[4] * A4 + b7[2] * A2 + b7[0] * I
        return _solve_PQ(U, V, spsolve)
    if eta_3 < 2.097847961257068e+000 and _ell(Ac, 9) == 0:
        b9 = (17643225600., 8821612800., 2075673600., 302702400., 30270240.,
              2162160., 110880., 3960., 90., 1.)
        U = Ac @ (b9[9] * A8 + b9[7] * A6 + b9[5] * A4 + b9[3] * A2 + b9[1] * I)
        V = b9[8] * A8 + b9[6] * A6 + b9[4] * A4 + b9[2] * A2 + b9[0] * I
        return _solve_PQ(U, V, spsolve)
    if exact:
        A10 = A4 @ A6
        d10l = _exact_1_norm(A10) ** 0.1
    else:
        d10l = _onenormest_prod(A4, A6) ** 0.1
    eta_4 = max(d8l, d10l)
    eta_5 = min(eta_3, eta_4)
    theta_13 = 4.25
    if eta_5 == 0:
        s = 0
    else:
        s = max(int(math.ceil(math.log2(eta_5 / theta_13))), 0)
    s = s + _ell(Ac * (2.0 ** -s), 13)
    b = (64764752532480000., 32382376266240000., 7771770303897600.,
         1187353796428800., 129060195264000., 10559470521600.,
         670442572800., 33522128640., 1323241920., 40840800., 960960.,
         16380., 182., 1.)
    B = Ac * (2.0 ** -s)
    B2 = A2 * (2.0 ** (-2 * s))
    B4 = A4 * (2.0 ** (-4 * s))
    B6 = A6 * (2.0 ** (-6 * s))
    U2 = B6 @ (b[13] * B6 + b[11] * B4 + b[9] * B2)
    U = B @ (U2 + b[7] * B6 + b[5] * B4 + b[3] * B2 + b[1] * I)
    V2 = B6 @ (b[12] * B6 + b[10] * B4 + b[8] * B2)
    V = V2 + b[6] * B6 + b[4] * B4 + b[2] * B2 + b[0] * I
    X = _solve_PQ(U, V, spsolve)
    if upper:
        X = _fragment_2_1(X, Ac, s)
    else:
        for _ in range(s):
            X = X @ X
    return X

def _solve_PQ(U, V, spsolve):
    P = U + V
    Q = -U + V
    return spsolve(Q.tocsc(), P.tocsc())

def _exp_sinch(a: float, x: float) -> float:
    if abs(x) < 0.0135:
        x2 = x * x
        return math.exp(a) * (1 + (x2 / 6.) * (1 + (x2 / 20.) * (1 + (x2 / 42.))))
    else:
        return (math.exp(a + x) - math.exp(a - x)) / (2 * x)

def _fragment_2_1(X, Tm, s: int):
    # scipy _fragment_2_1 on a sparse X (item assignment as in scipy)
    n = X.shape[0]
    Td = Tm.diagonal()
    Tl = Tm.tolil()
    Xl = X.tolil()
    scale = 2.0 ** -s
    for k in range(n):
        Xl[k, k] = math.exp(scale * Td[k])
    for i in range(s - 1, -1, -1):
        Xc = Xl.tocsr()
        Xl = (Xc @ Xc).tolil()
        scale = 2.0 ** -i
        for k in range(n):
            Xl[k, k] = math.exp(scale * Td[k])
        for k in range(n - 1):
            lam_1 = scale * Td[k]
            lam_2 = scale * Td[k + 1]
            t_12 = scale * Tl[k, k + 1]
            a = 0.5 * (lam_1 + lam_2)
            bb = 0.5 * (lam_1 - lam_2)
            Xl[k, k + 1] = t_12 * _exp_sinch(a, bb)
    return Xl.tocsc()

def expm(A):
    # scipy sparse expm: Pade approximation on sparse matrices; the result
    # has the compressed format of spsolve (CSC), converted like scipy to
    # the input's class for CSR input
    E = _sparse_expm(A)
    if A._spfmt() == "csr":
        return E.tocsr()
    else:
        return E.tocsc()

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
