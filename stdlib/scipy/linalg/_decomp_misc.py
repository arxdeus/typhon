# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# ldl (?sytrf/?hetrf), rq (?gerqf + ?orgrq), polar, orthogonal_procrustes,
# subspace_angles, bandwidth, issymmetric/ishermitian.

import numpy as np
from numpy.linalg import LinAlgError
from numpy.linalg.blas import fint
from ._lapack import *
from ._util import _arr, _arr_lapack, _to_f_as, _from_f, _vec, _is_complex, \
                   _work_size, _Pair, _Triple
from ._basic import _svd_2d, _herm_t, orth, _svdvals_2d

def _ch(c: str):
    return u8(ord(c))

# ---------------------------------------------------------------- ldl

def ldl(A, lower: bool = True, hermitian: bool = True, overwrite_a: bool = False,
        check_finite: bool = True):
    a = _arr_lapack(np.atleast_2d(_arr(A)), check_finite)
    if a.shape[0] != a.shape[1]:
        raise ValueError('The input array "a" should be square.')
    n = a.shape[0]
    T = a.dtype
    if n == 0:
        return (np.empty((0, 0), dtype=T), np.empty((0, 0), dtype=T), np.empty(0, dtype=int))
    Ap = _to_f_as(a, T)
    uplo = _ch('L' if lower else 'U')
    N = fint(n); LD = fint(n); info = fint(0)
    piv = Ptr[fint](n)
    herm = _is_complex(T) and hermitian
    q = T()
    LW = fint(-1)
    for it in range(2):
        if it == 1:
            LW = fint(max(_work_size(q), n, 1))
            work = Ptr[T](int(LW))
        else:
            work = __ptr__(q)
        if _is_complex(T):
            if herm:
                zhetrf_(__ptr__(uplo).as_byte(), __ptr__(N).as_byte(), Ap.as_byte(), __ptr__(LD).as_byte(), piv.as_byte(), work.as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
            else:
                zsytrf_(__ptr__(uplo).as_byte(), __ptr__(N).as_byte(), Ap.as_byte(), __ptr__(LD).as_byte(), piv.as_byte(), work.as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
        else:
            dsytrf_(__ptr__(uplo).as_byte(), __ptr__(N).as_byte(), Ap.as_byte(), __ptr__(LD).as_byte(), piv.as_byte(), work.as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
    if int(info) < 0:
        s = 'HETRF' if herm else 'SYTRF'
        raise ValueError(f'{s} exited with the internal error "illegal value in argument number {-int(info)}". See LAPACK documentation for the error codes.')
    ldu = _from_f(Ap, n, n, n)
    ipiv = [int(piv[i]) for i in range(n)]
    # _ldl_sanitize_ipiv
    swap_ = [i for i in range(n)]
    pivots = [0 for _ in range(n)]
    skip = False
    x, y, rs, re, ri = (1, 0, 0, n, 1) if lower else (-1, -1, n - 1, -1, -1)
    for ind in range(rs, re, ri):
        if skip:
            skip = False
            continue
        cur = ipiv[ind]
        if cur > 0:
            if cur != ind + 1:
                swap_[ind] = swap_[cur - 1]
            pivots[ind] = 1
        elif cur < 0 and cur == ipiv[ind + x]:
            if -cur != ind + 2:
                swap_[ind + x] = swap_[-cur - 1]
            pivots[ind + y] = 2
            skip = True
        else:
            raise ValueError('While parsing the permutation array in "scipy.linalg.ldl", invalid entries found. The array syntax is invalid.')
    # _ldl_get_d_and_l
    d = np.zeros((n, n), dtype=T)
    lu = np.zeros((n, n), dtype=T)
    for i in range(n):
        d[i, i] = ldu[i, i]
        for j in range(n):
            if (lower and i > j) or ((not lower) and i < j):
                lu[i, j] = ldu[i, j]
        lu[i, i] = T(1)
    xx, yy = (1, 0) if lower else (0, 1)
    blk_i = 0
    for blk in pivots:
        if blk == 0:
            continue
        if blk == 2:
            d[blk_i + xx, blk_i + yy] = ldu[blk_i + xx, blk_i + yy]
            if _is_complex(T) and hermitian:
                d[blk_i + yy, blk_i + xx] = ldu[blk_i + xx, blk_i + yy].conjugate()
            else:
                d[blk_i + yy, blk_i + xx] = ldu[blk_i + xx, blk_i + yy]
            lu[blk_i + xx, blk_i + yy] = T()
        blk_i += blk
    # _ldl_construct_tri_factor
    perm = [i for i in range(n)]
    rs2, re2, ri2 = (n - 1, -1, -1) if lower else (0, n, 1)
    for ind in range(rs2, re2, ri2):
        s_ind = swap_[ind]
        if s_ind != ind:
            col_s = ind if lower else 0
            col_e = n if lower else ind + 1
            if pivots[ind] == (0 if lower else 2):
                col_s += -1 if lower else 0
                col_e += 0 if lower else 1
            for c in range(col_s, col_e):
                t = lu[s_ind, c]; lu[s_ind, c] = lu[ind, c]; lu[ind, c] = t
            t2 = perm[s_ind]; perm[s_ind] = perm[ind]; perm[ind] = t2
    p = np.empty(n, dtype=int)
    for i in range(n):
        p[perm[i]] = i
    return (lu, d, p)

# ---------------------------------------------------------------- rq

def rq(a, overwrite_a: bool = False, lwork = None, mode: Literal[str] = 'full',
       check_finite: bool = True):
    if mode != 'full' and mode != 'r' and mode != 'economic':
        compile_error("Mode argument should be one of ['full', 'r', 'economic']")
    a1 = _arr_lapack(a, check_finite)
    if a1.ndim != 2:
        compile_error('expected matrix')
    T = a1.dtype
    M, N = a1.shape
    K = min(M, N)
    A = _to_f_as(a1, T)
    tau = Ptr[T](max(K, 1))
    Mi = fint(M); Ni = fint(N); LD = fint(max(M, 1)); info = fint(0)
    q = T()
    LW = fint(-1)
    for it in range(2):
        if it == 1:
            LW = fint(max(_work_size(q), 1))
            work = Ptr[T](int(LW))
        else:
            work = __ptr__(q)
        if _is_complex(T):
            zgerqf_(__ptr__(Mi).as_byte(), __ptr__(Ni).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), tau.as_byte(), work.as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
        else:
            dgerqf_(__ptr__(Mi).as_byte(), __ptr__(Ni).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), tau.as_byte(), work.as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
    if int(info) < 0:
        raise ValueError(f'illegal value in {-int(info)}th argument of internal gerqf')
    rqm = _from_f(A, M, N, max(M, 1))
    if mode != 'economic' or N < M:
        R = np.zeros((M, N), dtype=T)
        for i in range(M):
            for j in range(max(i + N - M, 0), N):
                R[i, j] = rqm[i, j]
    else:
        R = np.zeros((M, M), dtype=T)
        for i in range(M):
            for j in range(i, M):
                R[i, j] = rqm[i, N - M + j]
    if mode == 'r':
        return R
    else:
        # build the input for ?orgrq
        if N < M:
            rows = N
            src_off = M - N
            qr_rows = N
        elif mode == 'economic':
            rows = M
            src_off = 0
            qr_rows = M
        else:
            rows = N
            src_off = 0
            qr_rows = N
        Qp = Ptr[T](max(qr_rows * N, 1))
        for j in range(N):
            for i in range(qr_rows):
                Qp[i + j * qr_rows] = T()
        if N < M:
            for j in range(N):
                for i in range(N):
                    Qp[i + j * N] = rqm[src_off + i, j]
        elif mode == 'economic':
            for j in range(N):
                for i in range(M):
                    Qp[i + j * M] = rqm[i, j]
        else:
            for j in range(N):
                for i in range(M):
                    Qp[(N - M + i) + j * N] = rqm[i, j]
        Mq = fint(qr_rows); Kq = fint(min(qr_rows, K)); LDq = fint(max(qr_rows, 1))
        # tau for the last rows: when N<M only the last N reflectors are used
        tq = tau
        q2 = T()
        LW = fint(-1)
        for it in range(2):
            if it == 1:
                LW = fint(max(_work_size(q2), 1))
                work = Ptr[T](int(LW))
            else:
                work = __ptr__(q2)
            if _is_complex(T):
                zungrq_(__ptr__(Mq).as_byte(), __ptr__(Ni).as_byte(), __ptr__(Kq).as_byte(), Qp.as_byte(), __ptr__(LDq).as_byte(), tq.as_byte(), work.as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
            else:
                dorgrq_(__ptr__(Mq).as_byte(), __ptr__(Ni).as_byte(), __ptr__(Kq).as_byte(), Qp.as_byte(), __ptr__(LDq).as_byte(), tq.as_byte(), work.as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
        if int(info) < 0:
            raise ValueError(f'illegal value in {-int(info)}th argument of internal orgrq')
        return (R, _from_f(Qp, qr_rows, N, max(qr_rows, 1)))

# ---------------------------------------------------------------- polar / procrustes

def polar(a, side: str = 'right'):
    if side != 'right' and side != 'left':
        raise ValueError("`side` must be either 'right' or 'left'")
    a1 = np.asarray(a)
    if a1.ndim != 2:
        compile_error('`a` must be a 2-D array.')
    w, s, vh = _svd_2d(a1, full_matrices=False)
    u = w @ vh
    if side == 'right':
        vhH = _herm_t(vh)
        for i in range(vhH.shape[0]):
            for j in range(vhH.shape[1]):
                vhH[i, j] = vhH[i, j] * s[j]
        p = vhH @ vh
    else:
        ws = w.copy()
        for i in range(ws.shape[0]):
            for j in range(ws.shape[1]):
                ws[i, j] = ws[i, j] * s[j]
        p = ws @ _herm_t(w)
    return (u, p)

def orthogonal_procrustes(A, B, check_finite: bool = True):
    A1 = _arr_lapack(A, check_finite)
    B1 = _arr_lapack(B, check_finite)
    if A1.ndim != 2:
        compile_error('expected ndim to be 2')
    if A1.shape[0] != B1.shape[0] or A1.shape[1] != B1.shape[1]:
        raise ValueError(f'the shapes of A and B differ ({A1.shape} vs {B1.shape})')
    # (B.T @ conj(A)).T == A^H-ish: conj(A).T @ B
    M = (B1.T @ np.conj(A1)).T
    u, w, vt = _svd_2d(M.copy())
    R = u @ vt
    scale = 0.0
    for v in w:
        scale += v
    return (R, scale)

def subspace_angles(A, B):
    A1 = _arr_lapack(A, True)
    if A1.ndim != 2:
        compile_error('expected 2D array')
    QA = orth(A1)
    B1 = _arr_lapack(B, True)
    if B1.ndim != 2:
        compile_error('expected 2D array')
    if B1.shape[0] != QA.shape[0]:
        raise ValueError(f'A and B must have the same number of rows, got {QA.shape[0]} and {B1.shape[0]}')
    QB = orth(B1)
    QA_H_QB = _herm_t(QA) @ QB
    sigma = _svdvals_2d(QA_H_QB)
    if QA.shape[1] >= QB.shape[1]:
        Bm = QB - QA @ QA_H_QB
    else:
        Bm = QA - QB @ _herm_t(QA_H_QB)
    k = sigma.shape[0]
    anymask = False
    for i in range(k):
        if sigma[i] ** 2 >= 0.5:
            anymask = True
    theta = np.empty(k, dtype=float)
    if anymask:
        mu = _svdvals_2d(Bm)
        for i in range(k):
            if sigma[i] ** 2 >= 0.5:
                theta[i] = np.arcsin(min(max(mu[i], -1.0), 1.0))
            else:
                theta[i] = np.arccos(min(max(sigma[k - 1 - i], -1.0), 1.0))
    else:
        for i in range(k):
            theta[i] = np.arccos(min(max(sigma[k - 1 - i], -1.0), 1.0))
    return theta

# ---------------------------------------------------------------- structure checks

def bandwidth(a):
    a1 = np.asarray(a)
    if a1.ndim != 2:
        compile_error('bandwidth: only 2-D input is supported')
    m, n = a1.shape
    lo = 0
    up = 0
    z = a1.dtype()
    for i in range(m):
        for j in range(n):
            if a1[i, j] != z:
                if i > j:
                    lo = max(lo, i - j)
                elif j > i:
                    up = max(up, j - i)
    return (lo, up)

def _is_sym(a, atol, rtol, herm: bool) -> bool:
    a1 = np.asarray(a)
    if a1.ndim != 2:
        compile_error('Input array must be a 2D array.')
    n, m = a1.shape
    if n != m:
        return False
    exact = atol is None and rtol is None
    at = 0.0 if atol is None else float(atol)
    rt = 0.0 if rtol is None else float(rtol)
    for i in range(n):
        for j in range(i, n):
            x = a1[i, j]
            y = a1[j, i]
            if herm:
                if isinstance(y, complex) or isinstance(y, complex64):
                    y = y.conjugate()
            if exact:
                if x != y:
                    return False
            else:
                if not (abs(x - y) <= at + rt * abs(y)):
                    return False
    return True

def issymmetric(a, atol = None, rtol = None) -> bool:
    return _is_sym(a, atol, rtol, False)

def ishermitian(a, atol = None, rtol = None) -> bool:
    return _is_sym(a, atol, rtol, True)
