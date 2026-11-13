# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Banded and tridiagonal routines: solve_banded (?gtsv / ?gbsv),
# solveh_banded (?ptsv / ?pbsv), cholesky_banded / cho_solve_banded
# (?pbtrf / ?pbtrs), eig_banded / eigvals_banded (?sbevd, ?sbevx, ?hbevd),
# eigh_tridiagonal / eigvalsh_tridiagonal (?stevd, ?stev, ?sterf, ?stebz +
# ?stein), solve_toeplitz (Levinson recursion), solve_circulant.

import numpy as np
from numpy.linalg import LinAlgError
from numpy.linalg.blas import fint
from ._lapack import *
from ._util import _arr, _arr_lapack, _to_f_as, _from_f, _vec, _is_complex, \
                   _work_size, _promote, _Pair

def _ch(c: str):
    return u8(ord(c))

def _rhs(b1, T: type):
    if b1.ndim == 1:
        n = b1.shape[0]
        p = Ptr[T](max(n, 1))
        for i in range(n):
            p[i] = T(b1[i])
        return p, n, 1
    else:
        return _to_f_as(b1, T), b1.shape[0], b1.shape[1]

def _rhs_back(p: Ptr[T], b1, n: int, nrhs: int, T: type):
    if b1.ndim == 1:
        return _vec(p, n)
    else:
        return _from_f(p, n, nrhs, max(n, 1))

# ---------------------------------------------------------------- banded solves

def solve_banded(l_and_u, ab, b, overwrite_ab: bool = False, overwrite_b: bool = False,
                 check_finite: bool = True):
    nlower, nupper = int(l_and_u[0]), int(l_and_u[1])
    a1 = _arr_lapack(ab, check_finite)
    b1 = _arr_lapack(b, check_finite)
    T = type(_promote(a1.dtype, b1.dtype))
    n = a1.shape[1]
    if n != b1.shape[0]:
        raise ValueError('shapes of ab and b are not compatible.')
    if nlower + nupper + 1 != a1.shape[0]:
        raise ValueError(f'invalid values for the number of lower and upper diagonals: l+u+1 ({nlower + nupper + 1}) does not equal ab.shape[0] ({a1.shape[0]})')
    B, nb, nrhs = _rhs(b1, T)
    if n == 1:
        for i in range(nrhs):
            B[i] = B[i] / T(a1[nupper, 0])
        return _rhs_back(B, b1, n, nrhs)
    N = fint(n); NR = fint(nrhs); LDB = fint(max(n, 1)); info = fint(0)
    if nlower == 1 and nupper == 1:
        du = Ptr[T](n - 1); d = Ptr[T](n); dl = Ptr[T](n - 1)
        for i in range(n):
            d[i] = T(a1[1, i])
        for i in range(n - 1):
            du[i] = T(a1[0, i + 1])
            dl[i] = T(a1[2, i])
        if _is_complex(T):
            zgtsv_(__ptr__(N).as_byte(), __ptr__(NR).as_byte(), dl.as_byte(), d.as_byte(), du.as_byte(), B.as_byte(), __ptr__(LDB).as_byte(), __ptr__(info).as_byte())
        else:
            dgtsv_(__ptr__(N).as_byte(), __ptr__(NR).as_byte(), dl.as_byte(), d.as_byte(), du.as_byte(), B.as_byte(), __ptr__(LDB).as_byte(), __ptr__(info).as_byte())
    else:
        ld = 2 * nlower + nupper + 1
        A2 = Ptr[T](ld * n)
        for j in range(n):
            for i in range(ld):
                A2[i + j * ld] = T(a1[i - nlower, j]) if i >= nlower else T()
        piv = Ptr[fint](n)
        KL = fint(nlower); KU = fint(nupper); LDAB = fint(ld)
        if _is_complex(T):
            zgbsv_(__ptr__(N).as_byte(), __ptr__(KL).as_byte(), __ptr__(KU).as_byte(), __ptr__(NR).as_byte(), A2.as_byte(), __ptr__(LDAB).as_byte(), piv.as_byte(), B.as_byte(), __ptr__(LDB).as_byte(), __ptr__(info).as_byte())
        else:
            dgbsv_(__ptr__(N).as_byte(), __ptr__(KL).as_byte(), __ptr__(KU).as_byte(), __ptr__(NR).as_byte(), A2.as_byte(), __ptr__(LDAB).as_byte(), piv.as_byte(), B.as_byte(), __ptr__(LDB).as_byte(), __ptr__(info).as_byte())
    i = int(info)
    if i > 0:
        raise LinAlgError('singular matrix')
    if i < 0:
        raise ValueError(f'illegal value in {-i}-th argument of internal gbsv/gtsv')
    return _rhs_back(B, b1, n, nrhs)

def solveh_banded(ab, b, overwrite_ab: bool = False, overwrite_b: bool = False,
                  lower: bool = False, check_finite: bool = True):
    a1 = _arr_lapack(ab, check_finite)
    b1 = _arr_lapack(b, check_finite)
    T = type(_promote(a1.dtype, b1.dtype))
    n = a1.shape[1]
    if n != b1.shape[0]:
        raise ValueError('shapes of ab and b are not compatible.')
    B, nb, nrhs = _rhs(b1, T)
    N = fint(n); NR = fint(nrhs); LDB = fint(max(n, 1)); info = fint(0)
    kd = a1.shape[0] - 1
    if kd == 1:
        d = Ptr[float](n); e = Ptr[T](max(n - 1, 1))
        for i in range(n):
            d[i] = (a1[0, i].real if lower else a1[1, i].real) if _is_complex(a1.dtype) else float(a1[0, i] if lower else a1[1, i])
        for i in range(n - 1):
            if lower:
                e[i] = T(a1[1, i])
            else:
                if _is_complex(a1.dtype):
                    e[i] = T(a1[0, i + 1].conjugate())
                else:
                    e[i] = T(a1[0, i + 1])
        if _is_complex(T):
            zptsv_(__ptr__(N).as_byte(), __ptr__(NR).as_byte(), d.as_byte(), e.as_byte(), B.as_byte(), __ptr__(LDB).as_byte(), __ptr__(info).as_byte())
        else:
            dptsv_(__ptr__(N).as_byte(), __ptr__(NR).as_byte(), d.as_byte(), e.as_byte(), B.as_byte(), __ptr__(LDB).as_byte(), __ptr__(info).as_byte())
    else:
        A = _to_f_as(a1, T)
        uplo = _ch('L' if lower else 'U')
        KD = fint(kd); LDAB = fint(kd + 1)
        if _is_complex(T):
            zpbsv_(__ptr__(uplo).as_byte(), __ptr__(N).as_byte(), __ptr__(KD).as_byte(), __ptr__(NR).as_byte(), A.as_byte(), __ptr__(LDAB).as_byte(), B.as_byte(), __ptr__(LDB).as_byte(), __ptr__(info).as_byte())
        else:
            dpbsv_(__ptr__(uplo).as_byte(), __ptr__(N).as_byte(), __ptr__(KD).as_byte(), __ptr__(NR).as_byte(), A.as_byte(), __ptr__(LDAB).as_byte(), B.as_byte(), __ptr__(LDB).as_byte(), __ptr__(info).as_byte())
    i = int(info)
    if i > 0:
        raise LinAlgError(f'{i}th leading minor not positive definite')
    if i < 0:
        raise ValueError(f'illegal value in {-i}th argument of internal pbsv')
    return _rhs_back(B, b1, n, nrhs)

def cholesky_banded(ab, overwrite_ab: bool = False, lower: bool = False,
                    check_finite: bool = True):
    a1 = _arr_lapack(ab, check_finite)
    T = a1.dtype
    ld, n = a1.shape
    A = _to_f_as(a1, T)
    uplo = _ch('L' if lower else 'U')
    N = fint(n); KD = fint(ld - 1); LDAB = fint(ld); info = fint(0)
    if _is_complex(T):
        zpbtrf_(__ptr__(uplo).as_byte(), __ptr__(N).as_byte(), __ptr__(KD).as_byte(), A.as_byte(), __ptr__(LDAB).as_byte(), __ptr__(info).as_byte())
    else:
        dpbtrf_(__ptr__(uplo).as_byte(), __ptr__(N).as_byte(), __ptr__(KD).as_byte(), A.as_byte(), __ptr__(LDAB).as_byte(), __ptr__(info).as_byte())
    i = int(info)
    if i > 0:
        raise LinAlgError(f'{i}-th leading minor not positive definite')
    if i < 0:
        raise ValueError(f'illegal value in {i}-th argument of internal pbtrf')
    return _from_f(A, ld, n, ld)

def cho_solve_banded(cb_and_lower, b, overwrite_b: bool = False, check_finite: bool = True):
    cb = _arr_lapack(cb_and_lower[0], check_finite)
    lower = bool(cb_and_lower[1])
    b1 = _arr_lapack(b, check_finite)
    T = type(_promote(cb.dtype, b1.dtype))
    ld, n = cb.shape
    if n != b1.shape[0]:
        raise ValueError('shapes of cb and b are not compatible.')
    A = _to_f_as(cb, T)
    B, nb, nrhs = _rhs(b1, T)
    uplo = _ch('L' if lower else 'U')
    N = fint(n); KD = fint(ld - 1); LDAB = fint(ld); NR = fint(nrhs); LDB = fint(max(n, 1)); info = fint(0)
    if _is_complex(T):
        zpbtrs_(__ptr__(uplo).as_byte(), __ptr__(N).as_byte(), __ptr__(KD).as_byte(), __ptr__(NR).as_byte(), A.as_byte(), __ptr__(LDAB).as_byte(), B.as_byte(), __ptr__(LDB).as_byte(), __ptr__(info).as_byte())
    else:
        dpbtrs_(__ptr__(uplo).as_byte(), __ptr__(N).as_byte(), __ptr__(KD).as_byte(), __ptr__(NR).as_byte(), A.as_byte(), __ptr__(LDAB).as_byte(), B.as_byte(), __ptr__(LDB).as_byte(), __ptr__(info).as_byte())
    i = int(info)
    if i > 0:
        raise LinAlgError(f'{i}th leading minor not positive definite')
    if i < 0:
        raise ValueError(f'illegal value in {-i}th argument of internal pbtrs')
    return _rhs_back(B, b1, n, nrhs)

# ---------------------------------------------------------------- selection helper

def _check_select(select, select_range, max_ev: int, max_len: int):
    s = 0
    if isinstance(select, str):
        sl = select.lower()
        if sl == 'a' or sl == 'all':
            s = 0
        elif sl == 'v' or sl == 'value':
            s = 1
        elif sl == 'i' or sl == 'index':
            s = 2
        else:
            raise ValueError('invalid argument for select')
    else:
        s = int(select)
        if s < 0 or s > 2:
            raise ValueError('invalid argument for select')
    vl = 0.0; vu = 1.0; il = 1; iu = 1
    if s != 0:
        if select_range is None:
            raise ValueError('select_range must be a 2-element array-like in nondecreasing order')
        else:
            r0 = select_range[0]
            r1 = select_range[1]
            if r1 < r0:
                raise ValueError('select_range must be a 2-element array-like in nondecreasing order')
            if s == 1:
                vl = float(r0); vu = float(r1)
                if max_ev == 0:
                    max_ev = max_len
            else:
                if isinstance(r0, float) or isinstance(r1, float):
                    raise ValueError('when using select="i", select_range must contain integers')
                il = int(r0) + 1; iu = int(r1) + 1
                if min(il, iu) < 1 or max(il, iu) > max_len:
                    raise ValueError('select_range out of bounds')
                max_ev = iu - il + 1
    return s, vl, vu, il, iu, max_ev

def _check_info(info: int, driver: str):
    if info < 0:
        raise ValueError(f'illegal value in argument {-info} of internal {driver}')
    if info > 0:
        raise LinAlgError(f'{driver} did not converge (LAPACK info={info})')

_safe_min = 2.2250738585072014e-308

# ---------------------------------------------------------------- eig_banded

def _eig_banded_impl(a_band, lower: bool, eigvals_only: Literal[bool], select, select_range,
                     max_ev: int, check_finite: bool):
    a1 = _arr_lapack(a_band, check_finite)
    if a1.ndim != 2:
        compile_error('expected a 2-D array')
    T = a1.dtype
    ld, n = a1.shape
    s, vl, vu, il, iu, mev = _check_select(select, select_range, max_ev, n)
    A = _to_f_as(a1, T)
    uplo = _ch('L' if lower else 'U')
    jobz = _ch('N' if eigvals_only else 'V')
    N = fint(n); KD = fint(ld - 1); LDAB = fint(ld); info = fint(0)
    W = Ptr[float](max(n, 1))
    if s == 0:
        LDZ = fint(max(n, 1))
        Z = Ptr[T](max(n * n, 1))
        if _is_complex(T):
            lw = (2 * n * n) if not eigvals_only else n
            lrw = (1 + 5 * n + 2 * n * n) if not eigvals_only else n
            liw = (3 + 5 * n) if not eigvals_only else 1
            work = Ptr[T](max(lw, 1)); rwork = Ptr[float](max(lrw, 1)); iwork = Ptr[fint](max(liw, 1))
            LW = fint(max(lw, 1)); LRW = fint(max(lrw, 1)); LIW = fint(max(liw, 1))
            zhbevd_(__ptr__(jobz).as_byte(), __ptr__(uplo).as_byte(), __ptr__(N).as_byte(), __ptr__(KD).as_byte(), A.as_byte(), __ptr__(LDAB).as_byte(), W.as_byte(), Z.as_byte(), __ptr__(LDZ).as_byte(), work.as_byte(), __ptr__(LW).as_byte(), rwork.as_byte(), __ptr__(LRW).as_byte(), iwork.as_byte(), __ptr__(LIW).as_byte(), __ptr__(info).as_byte())
        else:
            lw = (1 + 5 * n + 2 * n * n) if not eigvals_only else 2 * n
            liw = (3 + 5 * n) if not eigvals_only else 1
            work = Ptr[T](max(lw, 1)); iwork = Ptr[fint](max(liw, 1))
            LW = fint(max(lw, 1)); LIW = fint(max(liw, 1))
            dsbevd_(__ptr__(jobz).as_byte(), __ptr__(uplo).as_byte(), __ptr__(N).as_byte(), __ptr__(KD).as_byte(), A.as_byte(), __ptr__(LDAB).as_byte(), W.as_byte(), Z.as_byte(), __ptr__(LDZ).as_byte(), work.as_byte(), __ptr__(LW).as_byte(), iwork.as_byte(), __ptr__(LIW).as_byte(), __ptr__(info).as_byte())
        _check_info(int(info), 'hbevd' if _is_complex(T) else 'sbevd')
        m = n
        ldz = max(n, 1)
    else:
        if eigvals_only:
            mev = 1
        mmax = mev if not eigvals_only else 1
        ldz = max(n, 1) if not eigvals_only else 1
        Z = Ptr[T](max(ldz * mmax, 1))
        Q = Ptr[T](max(n * n, 1))
        LDQ = fint(max(n, 1))
        rng = _ch('V' if s == 1 else 'I')
        VL = vl; VU = vu; IL = fint(il); IU = fint(iu); ABSTOL = 2 * _safe_min
        M = fint(0)
        iwork = Ptr[fint](max(5 * n, 1)); ifail = Ptr[fint](max(n, 1))
        LDZ = fint(ldz)
        jz = _ch('N' if eigvals_only else 'V')
        if _is_complex(T):
            workc = Ptr[T](max(n, 1)); rworkc = Ptr[float](max(7 * n, 1))
            zhbevx_(__ptr__(jz).as_byte(), __ptr__(rng).as_byte(), __ptr__(uplo).as_byte(), __ptr__(N).as_byte(), __ptr__(KD).as_byte(), A.as_byte(), __ptr__(LDAB).as_byte(), Q.as_byte(), __ptr__(LDQ).as_byte(), __ptr__(VL).as_byte(), __ptr__(VU).as_byte(), __ptr__(IL).as_byte(), __ptr__(IU).as_byte(), __ptr__(ABSTOL).as_byte(), __ptr__(M).as_byte(), W.as_byte(), Z.as_byte(), __ptr__(LDZ).as_byte(), workc.as_byte(), rworkc.as_byte(), iwork.as_byte(), ifail.as_byte(), __ptr__(info).as_byte())
        else:
            work = Ptr[T](max(7 * n, 1))
            dsbevx_(__ptr__(jz).as_byte(), __ptr__(rng).as_byte(), __ptr__(uplo).as_byte(), __ptr__(N).as_byte(), __ptr__(KD).as_byte(), A.as_byte(), __ptr__(LDAB).as_byte(), Q.as_byte(), __ptr__(LDQ).as_byte(), __ptr__(VL).as_byte(), __ptr__(VU).as_byte(), __ptr__(IL).as_byte(), __ptr__(IU).as_byte(), __ptr__(ABSTOL).as_byte(), __ptr__(M).as_byte(), W.as_byte(), Z.as_byte(), __ptr__(LDZ).as_byte(), work.as_byte(), iwork.as_byte(), ifail.as_byte(), __ptr__(info).as_byte())
        _check_info(int(info), 'hbevx' if _is_complex(T) else 'sbevx')
        m = int(M)
    w = _vec(W, m)
    if eigvals_only:
        return w
    else:
        return (w, _from_f(Z, n, m, ldz))

def eig_banded(a_band, lower: bool = False, eigvals_only: Literal[bool] = False,
               overwrite_a_band: bool = False, select = 'a', select_range = None,
               max_ev: int = 0, check_finite: bool = True):
    return _eig_banded_impl(a_band, lower, eigvals_only, select, select_range, max_ev, check_finite)

def eigvals_banded(a_band, lower: bool = False, overwrite_a_band: bool = False,
                   select = 'a', select_range = None, check_finite: bool = True):
    return _eig_banded_impl(a_band, lower, True, select, select_range, 0, check_finite)

# ---------------------------------------------------------------- tridiagonal

def _eigh_tri_impl(d, e, eigvals_only: Literal[bool], select, select_range,
                   check_finite: bool, tol: float, lapack_driver: str):
    d1 = _arr_lapack(d, check_finite)
    e1 = _arr_lapack(e, check_finite)
    if d1.ndim != 1 or e1.ndim != 1:
        compile_error('expected a 1-D array')
    if _is_complex(d1.dtype) or _is_complex(e1.dtype):
        compile_error('Only real arrays currently supported')
    n = d1.shape[0]
    if n != e1.shape[0] + 1:
        raise ValueError(f'd ({n}) must have one more element than e ({e1.shape[0]})')
    s, vl, vu, il, iu, _ = _check_select(select, select_range, 0, n)
    drv = lapack_driver
    if drv != 'auto' and drv != 'stemr' and drv != 'sterf' and drv != 'stebz' and drv != 'stev' and drv != 'stevd':
        raise ValueError(f"lapack_driver must be one of ('auto', 'stemr', 'sterf', 'stebz', 'stev', 'stevd'), got {lapack_driver}")
    if drv == 'auto':
        drv = 'stevd' if s == 0 else 'stebz'
    if n == 1:
        if s == 1 and not (vl < d1[0] <= vu):
            w0 = np.empty(0, dtype=float)
            v0 = np.empty((1, 0), dtype=float)
        else:
            w0 = np.array([d1[0]])
            v0 = np.array([[1.0]])
        if eigvals_only:
            return w0
        else:
            return (w0, v0)
    D = Ptr[float](n); E = Ptr[float](n)
    for i in range(n):
        D[i] = d1[i]
    for i in range(n - 1):
        E[i] = e1[i]
    E[n - 1] = 0.0
    N = fint(n); info = fint(0)
    if drv == 'stemr':
        # ?STEMR requires e of size n
        jz2 = _ch('N' if eigvals_only else 'V')
        rg2 = _ch('A' if s == 0 else ('V' if s == 1 else 'I'))
        VL2 = vl; VU2 = vu; IL2 = fint(il); IU2 = fint(iu)
        M2 = fint(0); NZC = fint(n); LDZ2 = fint(n if not eigvals_only else 1)
        W2 = Ptr[float](n)
        Z2 = Ptr[float](max(n * n, 1))
        isuppz = Ptr[fint](max(2 * n, 1))
        tryrac = fint(1)
        lw2 = (18 * n) if not eigvals_only else (12 * n)
        liw2 = (10 * n) if not eigvals_only else (8 * n)
        work2 = Ptr[float](max(lw2, 1)); iwork2 = Ptr[fint](max(liw2, 1))
        LW2 = fint(max(lw2, 1)); LIW2 = fint(max(liw2, 1))
        dstemr_(__ptr__(jz2).as_byte(), __ptr__(rg2).as_byte(), __ptr__(N).as_byte(), D.as_byte(), E.as_byte(), __ptr__(VL2).as_byte(), __ptr__(VU2).as_byte(), __ptr__(IL2).as_byte(), __ptr__(IU2).as_byte(), __ptr__(M2).as_byte(), W2.as_byte(), Z2.as_byte(), __ptr__(LDZ2).as_byte(), __ptr__(NZC).as_byte(), isuppz.as_byte(), __ptr__(tryrac).as_byte(), work2.as_byte(), __ptr__(LW2).as_byte(), iwork2.as_byte(), __ptr__(LIW2).as_byte(), __ptr__(info).as_byte())
        _check_info(int(info), 'stemr (eigh_tridiagonal)')
        m2 = int(M2)
        w2v = _vec(W2, m2)
        if eigvals_only:
            return w2v
        else:
            return (w2v, _from_f(Z2, n, m2, n))
    if drv == 'sterf' or drv == 'stev' or drv == 'stevd':
        if s != 0:
            raise ValueError(f'{drv} can only be used when select == "a"')
        if drv == 'sterf' and not eigvals_only:
            raise ValueError('sterf can only be used when eigvals_only is True')
        Z = Ptr[float](n * n)
        LDZ = fint(n)
        jobz = _ch('N' if eigvals_only else 'V')
        if drv == 'sterf':
            dsterf_(__ptr__(N).as_byte(), D.as_byte(), E.as_byte(), __ptr__(info).as_byte())
        elif drv == 'stev':
            work = Ptr[float](max(2 * n - 2, 1))
            dstev_(__ptr__(jobz).as_byte(), __ptr__(N).as_byte(), D.as_byte(), E.as_byte(), Z.as_byte(), __ptr__(LDZ).as_byte(), work.as_byte(), __ptr__(info).as_byte())
        else:
            lw = (1 + 4 * n + n * n) if not eigvals_only else 1
            liw = (3 + 5 * n) if not eigvals_only else 1
            work = Ptr[float](lw); iwork = Ptr[fint](liw)
            LW = fint(lw); LIW = fint(liw)
            dstevd_(__ptr__(jobz).as_byte(), __ptr__(N).as_byte(), D.as_byte(), E.as_byte(), Z.as_byte(), __ptr__(LDZ).as_byte(), work.as_byte(), __ptr__(LW).as_byte(), iwork.as_byte(), __ptr__(LIW).as_byte(), __ptr__(info).as_byte())
        _check_info(int(info), drv + ' (eigh_tridiagonal)')
        w = _vec(D, n)
        if eigvals_only:
            return w
        else:
            return (w, _from_f(Z, n, n, n))
    # stebz (+ stein)
    rng = _ch('A' if s == 0 else ('V' if s == 1 else 'I'))
    order = _ch('E' if eigvals_only else 'B')
    VL = vl; VU = vu; IL = fint(il); IU = fint(iu); TOL = float(tol)
    M = fint(0); NS = fint(0)
    W = Ptr[float](n); IB = Ptr[fint](n); IS = Ptr[fint](n)
    work = Ptr[float](4 * n); iwork = Ptr[fint](3 * n)
    dstebz_(__ptr__(rng).as_byte(), __ptr__(order).as_byte(), __ptr__(N).as_byte(), __ptr__(VL).as_byte(), __ptr__(VU).as_byte(), __ptr__(IL).as_byte(), __ptr__(IU).as_byte(), __ptr__(TOL).as_byte(), D.as_byte(), E.as_byte(), __ptr__(M).as_byte(), __ptr__(NS).as_byte(), W.as_byte(), IB.as_byte(), IS.as_byte(), work.as_byte(), iwork.as_byte(), __ptr__(info).as_byte())
    _check_info(int(info), 'stebz (eigh_tridiagonal)')
    m = int(M)
    w = _vec(W, m)
    if eigvals_only:
        return w
    else:
        Z = Ptr[float](max(n * m, 1))
        LDZ = fint(n)
        work2 = Ptr[float](5 * n); iwork2 = Ptr[fint](n); ifail = Ptr[fint](max(m, 1))
        dstein_(__ptr__(N).as_byte(), D.as_byte(), E.as_byte(), __ptr__(M).as_byte(), W.as_byte(), IB.as_byte(), IS.as_byte(), Z.as_byte(), __ptr__(LDZ).as_byte(), work2.as_byte(), iwork2.as_byte(), ifail.as_byte(), __ptr__(info).as_byte())
        i = int(info)
        if i < 0:
            raise ValueError(f'illegal value in argument {-i} of internal stein (eigh_tridiagonal)')
        if i > 0:
            raise LinAlgError(f'stein (eigh_tridiagonal) {i} eigenvectors failed to converge')
        v = _from_f(Z, n, m, n)
        order_ = np.argsort(w, kind='stable')
        w2 = np.empty(m, dtype=float)
        v2 = np.empty((n, m), dtype=float)
        for k in range(m):
            w2[k] = w[order_[k]]
            for r in range(n):
                v2[r, k] = v[r, order_[k]]
        return (w2, v2)

def eigh_tridiagonal(d, e, eigvals_only: Literal[bool] = False, select = 'a',
                     select_range = None, check_finite: bool = True, tol: float = 0.0,
                     lapack_driver: str = 'auto'):
    return _eigh_tri_impl(d, e, eigvals_only, select, select_range, check_finite, tol, lapack_driver)

def eigvalsh_tridiagonal(d, e, select = 'a', select_range = None, check_finite: bool = True,
                         tol: float = 0.0, lapack_driver: str = 'auto'):
    return _eigh_tri_impl(d, e, True, select, select_range, check_finite, tol, lapack_driver)

# ---------------------------------------------------------------- Toeplitz / circulant

def _levinson(a: Ptr[T], b: Ptr[T], n: int, T: type):
    x = Ptr[T](n); g = Ptr[T](n); h = Ptr[T](n)
    for i in range(n):
        x[i] = T(); g[i] = T(); h[i] = T()
    if a[n - 1] == T():
        raise LinAlgError('Singular principal minor')
    x[0] = b[0] / a[n - 1]
    if n == 1:
        return x
    g[0] = a[n - 2] / a[n - 1]
    h[0] = a[n] / a[n - 1]
    for m in range(1, n):
        x_num = -b[m]
        x_den = -a[n - 1]
        for j in range(m):
            nmj = n + m - (j + 1)
            x_num = x_num + a[nmj] * x[j]
            x_den = x_den + a[nmj] * g[m - j - 1]
        if x_den == T():
            raise LinAlgError('Singular principal minor')
        x[m] = x_num / x_den
        for j in range(m):
            x[j] = x[j] - x[m] * g[m - j - 1]
        if m == n - 1:
            return x
        g_num = -a[n - m - 2]
        h_num = -a[n + m]
        g_den = -a[n - 1]
        for j in range(m):
            g_num = g_num + a[n + j - m - 1] * g[j]
            h_num = h_num + a[n + m - j - 1] * h[j]
            g_den = g_den + a[n + j - m - 1] * h[m - j - 1]
        if g_den == T():
            raise LinAlgError("Singular principal minor")
        g[m] = g_num / g_den
        h[m] = h_num / x_den
        k = m - 1
        m2 = (m + 1) >> 1
        c1 = g[m]
        c2 = h[m]
        for j in range(m2):
            gj = g[j]; gk = g[k]; hj = h[j]; hk = h[k]
            g[j] = gj - (c1 * hk)
            g[k] = gk - (c1 * hj)
            h[j] = hj - (c2 * gk)
            h[k] = hk - (c2 * gj)
            k -= 1
    return x

def _toeplitz_cr(c_or_cr):
    if isinstance(c_or_cr, Tuple):
        return np.asarray(c_or_cr[0]).ravel(), np.asarray(c_or_cr[1]).ravel()
    else:
        c = np.asarray(c_or_cr).ravel()
        return c, np.conj(c)

def solve_toeplitz(c_or_cr, b, check_finite: bool = True):
    c, r = _toeplitz_cr(c_or_cr)
    b1 = np.asarray(b)
    if _is_complex(c.dtype) or _is_complex(r.dtype) or _is_complex(b1.dtype):
        return _solve_toeplitz_t(c, r, b1, complex)
    else:
        return _solve_toeplitz_t(c, r, b1, float)

def _solve_toeplitz_t(c, r, b1, T: type):
    n = r.shape[0]
    if c.shape[0] != n or b1.shape[0] != n:
        raise ValueError('Incompatible dimensions.')
    vals = Ptr[T](2 * n - 1)
    for i in range(n - 1):
        vals[i] = T(r[n - 1 - i])
    for i in range(n):
        vals[n - 1 + i] = T(c[i])
    if b1.ndim == 1:
        bb = Ptr[T](n)
        for i in range(n):
            bb[i] = T(b1[i])
        return _vec(_levinson(vals, bb, n), n)
    else:
        k = b1.shape[1]
        out = np.empty((n, k), dtype=T)
        bb = Ptr[T](n)
        for col in range(k):
            for i in range(n):
                bb[i] = T(b1[i, col])
            x = _levinson(vals, bb, n)
            for i in range(n):
                out[i, col] = x[i]
        return out

def matmul_toeplitz(c_or_cr, x, check_finite: bool = False, workers = None):
    from ._basic import toeplitz
    c, r = _toeplitz_cr(c_or_cr)
    return toeplitz(c, r) @ np.asarray(x)

def solve_circulant(c, b, singular: str = 'raise', tol = None, caxis: int = -1,
                    baxis: int = 0, outaxis: int = 0):
    c1 = np.atleast_1d(np.asarray(c))
    b1 = np.atleast_1d(np.asarray(b))
    if c1.ndim != 1:
        compile_error('solve_circulant: only 1-D c is supported')
    nc = c1.shape[0]
    if b1.shape[0] != nc:
        raise ValueError(f'Shapes of c {c1.shape} and b {b1.shape} are incompatible')
    fc = np.fft.fft(c1.astype(complex))
    mx = 0.0
    for v in fc:
        mx = max(mx, abs(v))
    tl = mx * nc * 2.220446049250313e-16 if tol is None else float(tol)
    near = [abs(fc[i]) <= tl for i in range(nc)]
    any_near = False
    for v in near:
        if v:
            any_near = True
    if any_near:
        if singular == 'raise':
            raise LinAlgError('near singular circulant matrix.')
        for i in range(nc):
            if near[i]:
                fc[i] = complex(1.0, 0.0)
    if b1.ndim == 1:
        fb = np.fft.fft(b1.astype(complex))
        q = np.empty(nc, dtype=complex)
        for i in range(nc):
            q[i] = complex() if near[i] else fb[i] / fc[i]
        x = np.fft.ifft(q)
    else:
        bt = b1.astype(complex)
        x = np.empty(bt.shape, dtype=complex)
        for col in range(bt.shape[1]):
            fb = np.fft.fft(bt[:, col].copy())
            q = np.empty(nc, dtype=complex)
            for i in range(nc):
                q[i] = complex() if near[i] else fb[i] / fc[i]
            xc = np.fft.ifft(q)
            for i in range(nc):
                x[i, col] = xc[i]
    if _is_complex(c1.dtype) or _is_complex(b1.dtype):
        return x
    else:
        return x.real.copy()
