# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Matrix functions built on expm / Schur: cosm, sinm, tanm, coshm, sinhm,
# tanhm, funm (Schur-Parlett), signm, expm_frechet (Al-Mohy & Higham 2009,
# 'SPS' and 'blockEnlarge'), expm_cond.
#
# Deviation: functions whose scipy result is real-if-possible (_maybe_real)
# return the input's real/complex kind statically (real input -> real output).

import numpy as np
from numpy.linalg import LinAlgError
from ._util import _arr_lapack, _is_complex, _Pair
from ._matfuncs import _expm_2d
from ._basic import _solve_2d, _svd_2d, _inv_2d, _herm_t, lu_factor, lu_solve
from ._decomp_schur import schur, rsf2csf

_eps = 2.220446049250313e-16

def _sq_in(A):
    a = _arr_lapack(A, False)
    if a.ndim != 2:
        compile_error('expected square array_like input')
    if a.shape[0] != a.shape[1]:
        raise ValueError('expected square array_like input')
    return a

def _real_part(X):
    n, m = X.shape
    r = np.empty((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            r[i, j] = X[i, j].real
    return r

def _imag_part(X):
    n, m = X.shape
    r = np.empty((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            r[i, j] = X[i, j].imag
    return r

def _maybe_real(a, X):
    if _is_complex(a.dtype):
        return X
    else:
        if _is_complex(X.dtype):
            return _real_part(X)
        else:
            return X

def cosm(A):
    a = _sq_in(A)
    ja = a.astype(complex) * complex(0.0, 1.0)
    if _is_complex(a.dtype):
        return 0.5 * (_expm_2d(ja) + _expm_2d(-ja))
    else:
        return _real_part(_expm_2d(ja))

def sinm(A):
    a = _sq_in(A)
    ja = a.astype(complex) * complex(0.0, 1.0)
    if _is_complex(a.dtype):
        return complex(0.0, -0.5) * (_expm_2d(ja) - _expm_2d(-ja))
    else:
        return _imag_part(_expm_2d(ja))

def tanm(A):
    a = _sq_in(A)
    return _maybe_real(a, _solve_2d(cosm(a), sinm(a)))

def coshm(A):
    a = _sq_in(A)
    return _maybe_real(a, 0.5 * (_expm_2d(a) + _expm_2d(-a)))

def sinhm(A):
    a = _sq_in(A)
    return _maybe_real(a, 0.5 * (_expm_2d(a) - _expm_2d(-a)))

def tanhm(A):
    a = _sq_in(A)
    return _maybe_real(a, _solve_2d(coshm(a), sinhm(a)))

def _funm_core(a, func):
    T0, Z0 = schur(a)
    T, Z = rsf2csf(T0, Z0)
    n = T.shape[0]
    dg = np.empty(n, dtype=complex)
    for i in range(n):
        dg[i] = T[i, i]
    fd = np.asarray(func(dg))
    F = np.zeros((n, n), dtype=complex)
    for i in range(n):
        F[i, i] = complex(fd[i])
    minden = abs(T[0, 0]) if n > 0 else 0.0
    # _funm_loops (Parlett recurrence)
    for p in range(1, n):
        for i in range(1, n - p + 1):
            j = i + p
            s = T[i - 1, j - 1] * (F[j - 1, j - 1] - F[i - 1, i - 1])
            ksl = slice(i, j - 1)
            val = complex()
            for k in range(i, j - 1):
                val += T[i - 1, k] * F[k, j - 1] - F[i - 1, k] * T[k, j - 1]
            s = s + val
            den = T[j - 1, j - 1] - T[i - 1, i - 1]
            if den != complex():
                s = s / den
            F[i - 1, j - 1] = s
            minden = min(minden, abs(den))
    F = Z @ F @ _herm_t(Z)
    Fr = _maybe_real(a, F)
    tol = _eps
    if minden == 0.0:
        minden = tol
    nrm = 0.0
    for j in range(n):
        cs = 0.0
        for i in range(j):
            cs += abs(T[i, j])
        nrm = max(nrm, cs)
    err = min(1.0, max(tol, tol / minden * nrm))
    allnan = n > 0
    for i in range(n):
        for j in range(n):
            v = F[i, j]
            if np.isfinite(v.real) and np.isfinite(v.imag):
                allnan = False
    if allnan:
        err = float('inf')
    return Fr, err

def funm(A, func, disp: Literal[bool] = True):
    a = _sq_in(A)
    F, err = _funm_core(a, func)
    if disp:
        if err > 1000 * _eps:
            print('funm result may be inaccurate, approximate err =', err)
        return F
    else:
        return (F, err)

class _RoundedSign:
    c: float
    def __call__(self, x):
        n = x.shape[0]
        r = np.empty(n, dtype=float)
        for i in range(n):
            rx = x[i].real
            r[i] = (1.0 if rx > 0 else -1.0) if abs(rx) > self.c else 0.0
        return r

def signm(A, disp: bool = True):
    a = _sq_in(A)
    T0, Z0 = schur(a)
    # c = 1000*eps*amax(x) over the eigenvalues (numpy complex ordering)
    T, Z = rsf2csf(T0, Z0)
    n = T.shape[0]
    best = T[0, 0] if n > 0 else complex()
    for i in range(1, n):
        v = T[i, i]
        if v.real > best.real or (v.real == best.real and v.imag > best.imag):
            best = v
    c = 1000.0 * _eps * best.real
    result, errest = _funm_core(a, _RoundedSign(c))
    errtol = 1000.0 * _eps
    if errest < errtol:
        return result
    vals = _svd_2d(a, compute_uv=False)
    max_sv = 0.0
    for v in vals:
        max_sv = max(max_sv, v)
    cc = 0.5 / max_sv
    S0 = a + cc * np.eye(n, dtype=a.dtype)
    prev = errest
    for _ in range(100):
        iS0 = _inv_2d(S0)
        S0 = 0.5 * (S0 + iS0)
        Pp = 0.5 * (S0 @ S0 + S0)
        D = Pp @ Pp - Pp
        e = 0.0
        for j in range(n):
            cs = 0.0
            for i in range(n):
                cs += abs(D[i, j])
            e = max(e, cs)
        errest = e
        if errest < errtol or prev == errest:
            break
        prev = errest
    if disp and (not np.isfinite(errest) or errest >= errtol):
        print('signm result may be inaccurate, approximate err =', errest)
    return S0

# ---------------------------------------------------------------- Frechet derivative

_ell_table_61 = (0.0, 2.11e-08, 0.000356, 0.0108, 0.0649, 0.2, 0.437, 0.783, 1.23, 1.78,
                 2.42, 3.13, 3.9, 4.74, 5.63, 6.56, 7.52, 8.53, 9.56, 10.6, 11.7)

def _norm1(A) -> float:
    n, m = A.shape
    best = 0.0
    for j in range(m):
        s = 0.0
        for i in range(n):
            s += abs(A[i, j])
        best = max(best, s)
    return best

def _diff_pade(A, E, ident, m: int):
    A2 = A @ A
    M2 = A @ E + E @ A
    if m == 3:
        b = (120.0, 60.0, 12.0, 1.0)
        U = A @ (b[3] * A2 + b[1] * ident)
        V = b[2] * A2 + b[0] * ident
        Lu = A @ (b[3] * M2) + E @ (b[3] * A2 + b[1] * ident)
        Lv = b[2] * M2
        return U, V, Lu, Lv
    A4 = A2 @ A2
    M4 = A2 @ M2 + M2 @ A2
    if m == 5:
        b = (30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0)
        U = A @ (b[5] * A4 + b[3] * A2 + b[1] * ident)
        V = b[4] * A4 + b[2] * A2 + b[0] * ident
        Lu = A @ (b[5] * M4 + b[3] * M2) + E @ (b[5] * A4 + b[3] * A2 + b[1] * ident)
        Lv = b[4] * M4 + b[2] * M2
        return U, V, Lu, Lv
    A6 = A2 @ A4
    M6 = A4 @ M2 + M4 @ A2
    if m == 7:
        b = (17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0)
        U = A @ (b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident)
        V = b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident
        Lu = A @ (b[7] * M6 + b[5] * M4 + b[3] * M2) + E @ (b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident)
        Lv = b[6] * M6 + b[4] * M4 + b[2] * M2
        return U, V, Lu, Lv
    b = (17643225600.0, 8821612800.0, 2075673600.0, 302702400.0, 30270240.0, 2162160.0,
         110880.0, 3960.0, 90.0, 1.0)
    A8 = A4 @ A4
    M8 = A4 @ M4 + M4 @ A4
    U = A @ (b[9] * A8 + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident)
    V = b[8] * A8 + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident
    Lu = A @ (b[9] * M8 + b[7] * M6 + b[5] * M4 + b[3] * M2) + E @ (b[9] * A8 + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident)
    Lv = b[8] * M8 + b[6] * M6 + b[4] * M4 + b[2] * M2
    return U, V, Lu, Lv

def _expm_frechet_algo_64(A0, E0):
    n = A0.shape[0]
    ident = np.eye(n, dtype=A0.dtype)
    A = A0
    E = E0
    A_norm_1 = _norm1(A)
    s = -1
    U = ident; V = ident; Lu = ident; Lv = ident
    for m in (3, 5, 7, 9):
        if A_norm_1 <= _ell_table_61[m]:
            U, V, Lu, Lv = _diff_pade(A, E, ident, m)
            s = 0
            break
    if s < 0:
        s = max(0, int(np.ceil(np.log2(A_norm_1 / _ell_table_61[13]))))
        A = A * 2.0 ** (-s)
        E = E * 2.0 ** (-s)
        A2 = A @ A
        M2 = A @ E + E @ A
        A4 = A2 @ A2
        M4 = A2 @ M2 + M2 @ A2
        A6 = A2 @ A4
        M6 = A4 @ M2 + M4 @ A2
        b = (6.476475253248e+16, 3.238237626624e+16, 7771770303897600.0, 1187353796428800.0,
             129060195264000.0, 10559470521600.0, 670442572800.0, 33522128640.0,
             1323241920.0, 40840800.0, 960960.0, 16380.0, 182.0, 1.0)
        W1 = b[13] * A6 + b[11] * A4 + b[9] * A2
        W2 = b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident
        Z1 = b[12] * A6 + b[10] * A4 + b[8] * A2
        Z2 = b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident
        W = A6 @ W1 + W2
        U = A @ W
        V = A6 @ Z1 + Z2
        Lw1 = b[13] * M6 + b[11] * M4 + b[9] * M2
        Lw2 = b[7] * M6 + b[5] * M4 + b[3] * M2
        Lz1 = b[12] * M6 + b[10] * M4 + b[8] * M2
        Lz2 = b[6] * M6 + b[4] * M4 + b[2] * M2
        Lw = A6 @ Lw1 + M6 @ W1 + Lw2
        Lu = A @ Lw + E @ W
        Lv = A6 @ Lz1 + M6 @ Z1 + Lz2
    lu_piv = lu_factor(-U + V)
    R = lu_solve(lu_piv, U + V)
    L = lu_solve(lu_piv, Lu + Lv + (Lu - Lv) @ R)
    for _ in range(s):
        L = R @ L + L @ R
        R = R @ R
    return R, L

def expm_frechet(A, E, method: Optional[str] = None, compute_expm: Literal[bool] = True,
                 check_finite: bool = True):
    a = _arr_lapack(A, check_finite)
    e = _arr_lapack(E, check_finite)
    if a.ndim != 2 or e.ndim != 2:
        compile_error('expected A and E to be square matrices')
    if a.shape[0] != a.shape[1]:
        raise ValueError('expected A to be a square matrix')
    if e.shape[0] != e.shape[1]:
        raise ValueError('expected E to be a square matrix')
    if a.shape[0] != e.shape[0]:
        raise ValueError('expected A and E to be the same shape')
    if _is_complex(a.dtype) or _is_complex(e.dtype):
        aa = a.astype(complex); ee = e.astype(complex)
    else:
        aa = a; ee = e
    meth = 'SPS' if method is None else method
    if meth == 'SPS':
        R, L = _expm_frechet_algo_64(aa, ee)
    elif meth == 'blockEnlarge':
        n = aa.shape[0]
        M = np.zeros((2 * n, 2 * n), dtype=aa.dtype)
        for i in range(n):
            for j in range(n):
                M[i, j] = aa[i, j]
                M[i, n + j] = ee[i, j]
                M[n + i, n + j] = aa[i, j]
        X = _expm_2d(M)
        R = X[:n, :n].copy()
        L = X[:n, n:].copy()
    else:
        raise ValueError(f'Unknown implementation {meth}')
    if compute_expm:
        return (R, L)
    else:
        return L

def expm_cond(A, check_finite: bool = True):
    a = _arr_lapack(A, check_finite)
    if a.ndim != 2:
        compile_error('expected a square matrix')
    if a.shape[0] != a.shape[1]:
        raise ValueError('expected a square matrix')
    n = a.shape[0]
    X = _expm_2d(a)
    K = np.empty((n * n, n * n), dtype=a.dtype)
    col = 0
    for i in range(n):
        for j in range(n):
            E = np.zeros((n, n), dtype=a.dtype)
            E[i, j] = a.dtype(1)
            _, F = _expm_frechet_algo_64(a, E)
            # vec(F) = F.T.ravel() (column stacking)
            idx = 0
            for c in range(n):
                for r in range(n):
                    K[idx, col] = F[r, c]
                    idx += 1
            col += 1
    def _fro(M) -> float:
        s = 0.0
        for v in M.flat:
            s += abs(v) ** 2
        return np.sqrt(s)
    sv = _svd_2d(K, compute_uv=False)
    K_norm = 0.0
    for v in sv:
        K_norm = max(K_norm, v)
    return K_norm * _fro(a) / _fro(X)
