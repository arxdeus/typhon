# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Matrix equation solvers: solve_sylvester (Bartels-Stewart via ?trsyl),
# solve_continuous_lyapunov / solve_lyapunov, solve_discrete_lyapunov
# ('direct' Kronecker and 'bilinear'), solve_continuous_are and
# solve_discrete_are (Arnold-Laub / ordered QZ on the extended pencil).
# Real input only for the ARE solvers.

import numpy as np
from numpy.linalg import LinAlgError
from numpy.linalg.blas import fint
from ._lapack import *
from ._util import _arr_lapack, _to_f_as, _from_f, _is_complex, _promote
from ._decomp_schur import schur, ordqz, matrix_balance
from ._basic import _solve_2d, _inv_2d, _qr_2d, _lu_2d, solve_triangular, _herm_t

def _ch(c: str):
    return u8(ord(c))

def _ct(a):
    return _herm_t(a)

def _trsyl(R, S, F, tranb: str, isgn: int, T: type):
    m = R.shape[0]
    n = S.shape[0]
    Rp = _to_f_as(R, T); Sp = _to_f_as(S, T); Fp = _to_f_as(F, T)
    ta = _ch('N'); tb = _ch(tranb)
    ISGN = fint(isgn); M = fint(m); N = fint(n)
    LDA = fint(max(m, 1)); LDB = fint(max(n, 1)); LDC = fint(max(m, 1))
    scale = 0.0; info = fint(0)
    if _is_complex(T):
        ztrsyl_(__ptr__(ta).as_byte(), __ptr__(tb).as_byte(), __ptr__(ISGN).as_byte(), __ptr__(M).as_byte(), __ptr__(N).as_byte(), Rp.as_byte(), __ptr__(LDA).as_byte(), Sp.as_byte(), __ptr__(LDB).as_byte(), Fp.as_byte(), __ptr__(LDC).as_byte(), __ptr__(scale).as_byte(), __ptr__(info).as_byte())
    else:
        dtrsyl_(__ptr__(ta).as_byte(), __ptr__(tb).as_byte(), __ptr__(ISGN).as_byte(), __ptr__(M).as_byte(), __ptr__(N).as_byte(), Rp.as_byte(), __ptr__(LDA).as_byte(), Sp.as_byte(), __ptr__(LDB).as_byte(), Fp.as_byte(), __ptr__(LDC).as_byte(), __ptr__(scale).as_byte(), __ptr__(info).as_byte())
    Y = _from_f(Fp, m, n, max(m, 1))
    return Y, scale, int(info)

def solve_sylvester(a, b, q):
    a1 = _arr_lapack(a, True)
    b1 = _arr_lapack(b, True)
    q1 = _arr_lapack(q, True)
    T = type(_promote(type(_promote(a1.dtype, b1.dtype)), q1.dtype))
    A = a1.astype(T); B = b1.astype(T); Q = q1.astype(T)
    r, u = schur(A, output='real')
    s, v = schur(_ct(B), output='real')
    f = _ct(u) @ Q @ v
    y, scale, info = _trsyl(r, s, f, 'C' if _is_complex(T) else 'T', 1, T)
    if info < 0:
        raise LinAlgError(f'Illegal value encountered in the {-info} term')
    y = y * scale
    return u @ y @ _ct(v)

def solve_continuous_lyapunov(a, q):
    a1 = np.atleast_2d(_arr_lapack(a, True))
    q1 = np.atleast_2d(_arr_lapack(q, True))
    T = type(_promote(a1.dtype, q1.dtype))
    if a1.shape[0] != a1.shape[1]:
        raise ValueError("Matrix a should be square.")
    if q1.shape[0] != q1.shape[1]:
        raise ValueError("Matrix q should be square.")
    if a1.shape[0] != q1.shape[0]:
        raise ValueError('Matrix a and q should have the same shape.')
    A = a1.astype(T); Q = q1.astype(T)
    r, u = schur(A, output='real')
    f = _ct(u) @ (Q @ u)
    y, scale, info = _trsyl(r, r, f, 'C' if _is_complex(T) else 'T', 1, T)
    if info < 0:
        raise ValueError(f'?TRSYL exited with the internal error "illegal value in argument number {-info}.". See LAPACK documentation for the ?TRSYL error codes.')
    y = y * scale
    return u @ y @ _ct(u)

solve_lyapunov = solve_continuous_lyapunov

def _kron_conj(a):
    n = a.shape[0]
    T = a.dtype
    K = np.empty((n * n, n * n), dtype=T)
    for i in range(n):
        for j in range(n):
            aij = a[i, j]
            for k in range(n):
                for l in range(n):
                    if _is_complex(T):
                        K[i * n + k, j * n + l] = aij * a[k, l].conjugate()
                    else:
                        K[i * n + k, j * n + l] = aij * a[k, l]
    return K

def solve_discrete_lyapunov(a, q, method: Optional[str] = None):
    a1 = np.asarray(a)
    q1 = np.asarray(q)
    T = type(_promote(_arr_lapack(a1, False).dtype, _arr_lapack(q1, False).dtype))
    A = a1.astype(T); Q = q1.astype(T)
    n = A.shape[0]
    meth = ('bilinear' if n >= 10 else 'direct') if method is None else method.lower()
    if meth == 'direct':
        lhs = _kron_conj(A)
        N2 = n * n
        for i in range(N2):
            for j in range(N2):
                lhs[i, j] = (T(1) if i == j else T()) - lhs[i, j]
        x = _solve_2d(lhs, Q.flatten())
        return x.reshape((n, n))
    elif meth == 'bilinear':
        eye = np.eye(n, dtype=T)
        aH = _ct(A)
        aHI_inv = _inv_2d(aH + eye)
        b = (aH - eye) @ aHI_inv
        c = T(2) * (_inv_2d(A + eye) @ Q @ aHI_inv)
        return solve_continuous_lyapunov(_ct(b), -c)
    else:
        raise ValueError(f'Unknown solver {method}')

def _spacing(x: float) -> float:
    return np.nextafter(abs(x), float('inf')) - abs(x)

def _norm1(a) -> float:
    m, n = a.shape
    best = 0.0
    for j in range(n):
        s = 0.0
        for i in range(m):
            s += abs(a[i, j])
        best = max(best, s)
    return best

def _are_validate(a, b, q, r, e, s, eq: str):
    a1 = np.atleast_2d(_arr_lapack(a, True))
    b1 = np.atleast_2d(_arr_lapack(b, True))
    q1 = np.atleast_2d(_arr_lapack(q, True))
    r1 = np.atleast_2d(_arr_lapack(r, True))
    if _is_complex(a1.dtype) or _is_complex(b1.dtype) or _is_complex(q1.dtype) or _is_complex(r1.dtype):
        compile_error("solve_*_are: complex input is not supported")
    for nm, mat in (('a', a1), ('q', q1), ('r', r1)):
        if mat.shape[0] != mat.shape[1]:
            raise ValueError(f"Matrix {nm} should be square.")
    m, n = b1.shape
    if m != a1.shape[0]:
        raise ValueError('Matrix a and b should have the same number of rows.')
    if m != q1.shape[0]:
        raise ValueError('Matrix a and q should have the same shape.')
    if n != r1.shape[0]:
        raise ValueError('Matrix b and r should have the same number of cols.')
    for nm, mat in (('q', q1), ('r', r1)):
        if _norm1(mat - mat.T) > _spacing(_norm1(mat)) * 100:
            raise ValueError(f"Matrix {nm} should be symmetric/hermitian.")
    if eq == 'care':
        sv = np.linalg.svd(r1, compute_uv=False)
        min_sv = sv[sv.shape[0] - 1]
        if min_sv == 0.0 or min_sv < _spacing(1.0) * _norm1(r1):
            raise ValueError('Matrix r is numerically singular.')
    has_e = e is not None
    has_s = s is not None
    E = np.eye(m)
    S = np.zeros((m, n))
    if e is not None:
        E = np.atleast_2d(_arr_lapack(e, True)).astype(float)
        if E.shape[0] != E.shape[1]:
            raise ValueError('Matrix e should be square.')
        if m != E.shape[0]:
            raise ValueError('Matrix a and e should have the same shape.')
        sv = np.linalg.svd(E, compute_uv=False)
        min_sv = sv[sv.shape[0] - 1]
        if min_sv == 0.0 or min_sv < _spacing(1.0) * _norm1(E):
            raise ValueError('Matrix e is numerically singular.')
    if s is not None:
        S = np.atleast_2d(_arr_lapack(s, True)).astype(float)
        if S.shape[0] != m or S.shape[1] != n:
            raise ValueError('Matrix b and s should have the same shape.')
    return a1, b1, q1, r1, E, S, m, n, has_e, has_s

def _balance_pencil(H, J, m: int, balanced: bool):
    N = H.shape[0]
    sca = np.ones(N)
    if balanced:
        M = np.abs(H) + np.abs(J)
        for i in range(N):
            M[i, i] = 0.0
        _, ss = matrix_balance(M, separate=True, permute=False)
        sc = ss[0]
        allone = True
        for i in range(N):
            if not (abs(sc[i] - 1.0) <= 1e-8 + 1e-5):
                allone = False
        if not allone:
            lg = np.log2(sc)
            for i in range(m):
                sv = np.round((lg[m + i] - lg[i]) / 2)
                sca[i] = 2.0 ** sv
                sca[m + i] = 2.0 ** (-sv)
            for i in range(2 * m, N):
                sca[i] = 2.0 ** lg[i]
            for i in range(N):
                for j in range(N):
                    f = sca[i] / sca[j]
                    H[i, j] *= f
                    J[i, j] *= f
            return sca, True
    return sca, False

def _are_finish(u, m: int, sca, scaled: bool, has_e: bool, E, msg: str):
    if has_e:
        st = np.empty((2 * m, m))
        eu = E @ u[:m, :m].copy()
        for i in range(m):
            for j in range(m):
                st[i, j] = eu[i, j]
                st[m + i, j] = u[m + i, j]
        u = _qr_2d(st)[0]
    u00 = u[:m, :m].copy()
    u10 = u[m:2 * m, :m].copy()
    up, ul, uu = _lu_2d(u00)
    if 1.0 / np.linalg.cond(uu) < _spacing(1.0):
        raise LinAlgError('Failed to find a finite solution.')
    inner = solve_triangular(uu.T, u10.T, lower=True)
    x = solve_triangular(ul.T, inner, unit_diagonal=True).T @ up.T
    if scaled:
        for i in range(m):
            for j in range(m):
                x[i, j] *= sca[i] * sca[j]
    u_sym = u00.T @ u10
    n_u_sym = _norm1(u_sym)
    u_sym = u_sym - u_sym.T
    thr = max(_spacing(1000.0), 0.1 * n_u_sym)
    if _norm1(u_sym) > thr:
        raise LinAlgError(msg)
    return (x + x.T) / 2

def solve_continuous_are(a, b, q, r, e = None, s = None, balanced: bool = True):
    a1, b1, q1, r1, E, S, m, n, has_e, has_s = _are_validate(a, b, q, r, e, s, 'care')
    N = 2 * m + n
    H = np.zeros((N, N))
    for i in range(m):
        for j in range(m):
            H[i, j] = a1[i, j]
            H[m + i, j] = -q1[i, j]
            H[m + i, m + j] = -a1[j, i]
        for j in range(n):
            H[i, 2 * m + j] = b1[i, j]
            H[m + i, 2 * m + j] = -S[i, j]
            H[2 * m + j, i] = S[i, j]
            H[2 * m + j, m + i] = b1[i, j]
    for i in range(n):
        for j in range(n):
            H[2 * m + i, 2 * m + j] = r1[i, j]
    J = np.zeros((N, N))
    for i in range(m):
        for j in range(m):
            J[i, j] = E[i, j] if has_e else (1.0 if i == j else 0.0)
            J[m + i, m + j] = E[j, i] if has_e else (1.0 if i == j else 0.0)
    sca, scaled = _balance_pencil(H, J, m, balanced)
    q_, _ = _qr_2d(H[:, 2 * m:].copy())
    qn = q_[:, n:].copy()
    H2 = qn.T @ H[:, :2 * m].copy()
    J2 = q_[:2 * m, n:].copy().T @ J[:2 * m, :2 * m].copy()
    res = ordqz(H2, J2, sort='lhp', output='real')
    u = res[5]
    return _are_finish(u, m, sca, scaled, has_e, E,
                       'The associated Hamiltonian pencil has eigenvalues too close to the imaginary axis')

def solve_discrete_are(a, b, q, r, e = None, s = None, balanced: bool = True):
    a1, b1, q1, r1, E, S, m, n, has_e, has_s = _are_validate(a, b, q, r, e, s, 'dare')
    N = 2 * m + n
    H = np.zeros((N, N))
    J = np.zeros((N, N))
    for i in range(m):
        for j in range(m):
            H[i, j] = a1[i, j]
            H[m + i, j] = -q1[i, j]
            H[m + i, m + j] = E[j, i] if has_e else (1.0 if i == j else 0.0)
            J[i, j] = E[i, j] if has_e else (1.0 if i == j else 0.0)
            J[m + i, m + j] = a1[j, i]
        for j in range(n):
            H[i, 2 * m + j] = b1[i, j]
            H[m + i, 2 * m + j] = -S[i, j]
            H[2 * m + j, i] = S[i, j]
            J[2 * m + j, m + i] = -b1[i, j]
    for i in range(n):
        for j in range(n):
            H[2 * m + i, 2 * m + j] = r1[i, j]
    sca, scaled = _balance_pencil(H, J, m, balanced)
    q_, _ = _qr_2d(H[:, 2 * m:].copy())
    qn = q_[:, n:].copy()
    H2 = qn.T @ H[:, :2 * m].copy()
    J2 = qn.T @ J[:, :2 * m].copy()
    res = ordqz(H2, J2, sort='iuc', output='real')
    u = res[5]
    return _are_finish(u, m, sca, scaled, has_e, E,
                       'The associated symplectic pencil has eigenvalues too close to the unit circle')
