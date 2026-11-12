# Ported from scipy 1.18.1 (BSD-3-Clause)

# scipy.linalg.eigh / eigvalsh with scipy's LAPACK driver selection:
# standard problem -> ?syevr/?heevr, generalized -> ?sygvd/?hegvd, any subset
# with b -> ?sygvx/?hegvx. Explicit drivers ev, evd, evr, evx, gv, gvd, gvx
# are supported. Workspace sizes mirror the f2py wrappers scipy uses.

import numpy as np
from numpy.linalg import LinAlgError
from numpy.linalg.blas import fint
from ._lapack import *
from ._util import _arr_lapack, _square2, _to_f, _to_f_as, _from_f, _vec, \
                   _is_complex, _work_size, _promote, _Pair

def _err(info: int, n: int, driver: str, eigvals_only: bool):
    if info < -1:
        raise LinAlgError(f'Illegal value in argument {-info} of internal {driver}')
    elif info > n:
        raise LinAlgError(f'The leading minor of order {info-n} of B is not '
                          'positive definite. The factorization of B '
                          'could not be completed and no eigenvalues '
                          'or eigenvectors were computed.')
    else:
        if driver == 'ev' or driver == 'gv':
            msg = (f'The algorithm failed to converge; {info} off-diagonal elements '
                   'of an intermediate tridiagonal form did not converge to zero.')
        elif driver == 'evx' or driver == 'gvx':
            msg = f'{info} eigenvectors failed to converge.'
        elif driver == 'evd' or driver == 'gvd':
            if eigvals_only:
                msg = (f'The algorithm failed to converge; {info} off-diagonal elements '
                       'of an intermediate tridiagonal form did not converge to zero.')
            else:
                msg = (f'The algorithm failed to compute an eigenvalue while working '
                       f'on the submatrix lying in rows and columns {info}/{n+1} '
                       f'through mod({info},{n+1}).')
        else:
            msg = 'Internal Error.'
        raise LinAlgError(msg)

def _eigh_core(A: Ptr[T], Bm: Ptr[T], n: int, has_b: bool, lower: bool,
               compute_v: bool, itype: int, driver: str, rng: u8,
               vl: float, vu: float, il: int, iu: int, T: type):
    # Returns (w pointer, z pointer, ldz, m, info). z may alias A.
    cplx: Literal[bool] = _is_complex(T)
    jobz = u8(86) if compute_v else u8(78)   # 'V' / 'N'
    uplo = u8(76) if lower else u8(85)       # 'L' / 'U'
    N = fint(n)
    LDA = fint(max(n, 1))
    W = Ptr[float](max(n, 1))
    info = fint(0)
    M = fint(n)
    IL = fint(il)
    IU = fint(iu)
    VL = vl
    VU = vu
    ABSTOL = 0.0
    ITYPE = fint(itype)
    z = A
    ldz = max(n, 1)

    if driver == 'evr':
        ncols = (iu - il + 1) if rng == u8(73) else n
        z = Ptr[T](max(n * max(ncols, 1), 1))
        LDZ = fint(max(n, 1))
        isuppz = Ptr[fint](max(2 * n, 2))
        qw = T()
        qr = 0.0
        qi = fint(0)
        LW = fint(-1)
        LRW = fint(-1)
        LIW = fint(-1)
        # workspace query (scipy uses ?syevr_lwork, i.e. the LAPACK query)
        if cplx:
            zheevr_(__ptr__(jobz).as_byte(), __ptr__(rng).as_byte(), __ptr__(uplo).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), __ptr__(VL).as_byte(), __ptr__(VU).as_byte(),
                    __ptr__(IL).as_byte(), __ptr__(IU).as_byte(), __ptr__(ABSTOL).as_byte(), __ptr__(M).as_byte(), W.as_byte(), z.as_byte(), __ptr__(LDZ).as_byte(),
                    isuppz.as_byte(), __ptr__(qw).as_byte(), __ptr__(LW).as_byte(), __ptr__(qr).as_byte(), __ptr__(LRW).as_byte(), __ptr__(qi).as_byte(), __ptr__(LIW).as_byte(), __ptr__(info).as_byte())
        else:
            dsyevr_(__ptr__(jobz).as_byte(), __ptr__(rng).as_byte(), __ptr__(uplo).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), __ptr__(VL).as_byte(), __ptr__(VU).as_byte(),
                    __ptr__(IL).as_byte(), __ptr__(IU).as_byte(), __ptr__(ABSTOL).as_byte(), __ptr__(M).as_byte(), W.as_byte(), z.as_byte(), __ptr__(LDZ).as_byte(),
                    isuppz.as_byte(), __ptr__(qw).as_byte(), __ptr__(LW).as_byte(), __ptr__(qi).as_byte(), __ptr__(LIW).as_byte(), __ptr__(info).as_byte())
        LW = fint(max(_work_size(qw), 1))
        LIW = fint(max(int(qi), 1))
        LRW = fint(max(int(qr), 1))
        work = Ptr[T](int(LW))
        iwork = Ptr[fint](int(LIW))
        rwork = Ptr[float](int(LRW))
        info = fint(0)
        if cplx:
            zheevr_(__ptr__(jobz).as_byte(), __ptr__(rng).as_byte(), __ptr__(uplo).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), __ptr__(VL).as_byte(), __ptr__(VU).as_byte(),
                    __ptr__(IL).as_byte(), __ptr__(IU).as_byte(), __ptr__(ABSTOL).as_byte(), __ptr__(M).as_byte(), W.as_byte(), z.as_byte(), __ptr__(LDZ).as_byte(),
                    isuppz.as_byte(), work.as_byte(), __ptr__(LW).as_byte(), rwork.as_byte(), __ptr__(LRW).as_byte(),
                    iwork.as_byte(), __ptr__(LIW).as_byte(), __ptr__(info).as_byte())
        else:
            dsyevr_(__ptr__(jobz).as_byte(), __ptr__(rng).as_byte(), __ptr__(uplo).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), __ptr__(VL).as_byte(), __ptr__(VU).as_byte(),
                    __ptr__(IL).as_byte(), __ptr__(IU).as_byte(), __ptr__(ABSTOL).as_byte(), __ptr__(M).as_byte(), W.as_byte(), z.as_byte(), __ptr__(LDZ).as_byte(),
                    isuppz.as_byte(), work.as_byte(), __ptr__(LW).as_byte(), iwork.as_byte(), __ptr__(LIW).as_byte(), __ptr__(info).as_byte())
    elif driver == 'evx' or driver == 'gvx':
        ncols = (iu - il + 1) if rng == u8(73) else n
        z = Ptr[T](max(n * max(ncols, 1), 1))
        LDZ = fint(max(n, 1))
        ifail = Ptr[fint](max(n, 1))
        iwork = Ptr[fint](max(5 * n, 1))
        rwork = Ptr[float](max(7 * n, 1))
        LDB = LDA
        qw = T()
        LW = fint(-1)
        if driver == 'evx':
            if cplx:
                zheevx_(__ptr__(jobz).as_byte(), __ptr__(rng).as_byte(), __ptr__(uplo).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), __ptr__(VL).as_byte(), __ptr__(VU).as_byte(),
                        __ptr__(IL).as_byte(), __ptr__(IU).as_byte(), __ptr__(ABSTOL).as_byte(), __ptr__(M).as_byte(), W.as_byte(), z.as_byte(), __ptr__(LDZ).as_byte(),
                        __ptr__(qw).as_byte(), __ptr__(LW).as_byte(), rwork.as_byte(), iwork.as_byte(), ifail.as_byte(), __ptr__(info).as_byte())
            else:
                dsyevx_(__ptr__(jobz).as_byte(), __ptr__(rng).as_byte(), __ptr__(uplo).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), __ptr__(VL).as_byte(), __ptr__(VU).as_byte(),
                        __ptr__(IL).as_byte(), __ptr__(IU).as_byte(), __ptr__(ABSTOL).as_byte(), __ptr__(M).as_byte(), W.as_byte(), z.as_byte(), __ptr__(LDZ).as_byte(),
                        __ptr__(qw).as_byte(), __ptr__(LW).as_byte(), iwork.as_byte(), ifail.as_byte(), __ptr__(info).as_byte())
        else:
            if cplx:
                zhegvx_(__ptr__(ITYPE).as_byte(), __ptr__(jobz).as_byte(), __ptr__(rng).as_byte(), __ptr__(uplo).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(),
                        Bm.as_byte(), __ptr__(LDB).as_byte(), __ptr__(VL).as_byte(), __ptr__(VU).as_byte(), __ptr__(IL).as_byte(), __ptr__(IU).as_byte(), __ptr__(ABSTOL).as_byte(), __ptr__(M).as_byte(),
                        W.as_byte(), z.as_byte(), __ptr__(LDZ).as_byte(), __ptr__(qw).as_byte(), __ptr__(LW).as_byte(), rwork.as_byte(),
                        iwork.as_byte(), ifail.as_byte(), __ptr__(info).as_byte())
            else:
                dsygvx_(__ptr__(ITYPE).as_byte(), __ptr__(jobz).as_byte(), __ptr__(rng).as_byte(), __ptr__(uplo).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(),
                        Bm.as_byte(), __ptr__(LDB).as_byte(), __ptr__(VL).as_byte(), __ptr__(VU).as_byte(), __ptr__(IL).as_byte(), __ptr__(IU).as_byte(), __ptr__(ABSTOL).as_byte(), __ptr__(M).as_byte(),
                        W.as_byte(), z.as_byte(), __ptr__(LDZ).as_byte(), __ptr__(qw).as_byte(), __ptr__(LW).as_byte(),
                        iwork.as_byte(), ifail.as_byte(), __ptr__(info).as_byte())
        if driver == 'evx':
            LW = fint(max(_work_size(qw), 1))
        else:
            LW = fint(max(_work_size(qw), 1))
        work = Ptr[T](int(LW))
        info = fint(0)
        if driver == 'evx':
            if cplx:
                zheevx_(__ptr__(jobz).as_byte(), __ptr__(rng).as_byte(), __ptr__(uplo).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), __ptr__(VL).as_byte(), __ptr__(VU).as_byte(),
                        __ptr__(IL).as_byte(), __ptr__(IU).as_byte(), __ptr__(ABSTOL).as_byte(), __ptr__(M).as_byte(), W.as_byte(), z.as_byte(), __ptr__(LDZ).as_byte(),
                        work.as_byte(), __ptr__(LW).as_byte(), rwork.as_byte(), iwork.as_byte(), ifail.as_byte(), __ptr__(info).as_byte())
            else:
                dsyevx_(__ptr__(jobz).as_byte(), __ptr__(rng).as_byte(), __ptr__(uplo).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), __ptr__(VL).as_byte(), __ptr__(VU).as_byte(),
                        __ptr__(IL).as_byte(), __ptr__(IU).as_byte(), __ptr__(ABSTOL).as_byte(), __ptr__(M).as_byte(), W.as_byte(), z.as_byte(), __ptr__(LDZ).as_byte(),
                        work.as_byte(), __ptr__(LW).as_byte(), iwork.as_byte(), ifail.as_byte(), __ptr__(info).as_byte())
        else:
            if cplx:
                zhegvx_(__ptr__(ITYPE).as_byte(), __ptr__(jobz).as_byte(), __ptr__(rng).as_byte(), __ptr__(uplo).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(),
                        Bm.as_byte(), __ptr__(LDB).as_byte(), __ptr__(VL).as_byte(), __ptr__(VU).as_byte(), __ptr__(IL).as_byte(), __ptr__(IU).as_byte(), __ptr__(ABSTOL).as_byte(), __ptr__(M).as_byte(),
                        W.as_byte(), z.as_byte(), __ptr__(LDZ).as_byte(), work.as_byte(), __ptr__(LW).as_byte(), rwork.as_byte(),
                        iwork.as_byte(), ifail.as_byte(), __ptr__(info).as_byte())
            else:
                dsygvx_(__ptr__(ITYPE).as_byte(), __ptr__(jobz).as_byte(), __ptr__(rng).as_byte(), __ptr__(uplo).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(),
                        Bm.as_byte(), __ptr__(LDB).as_byte(), __ptr__(VL).as_byte(), __ptr__(VU).as_byte(), __ptr__(IL).as_byte(), __ptr__(IU).as_byte(), __ptr__(ABSTOL).as_byte(), __ptr__(M).as_byte(),
                        W.as_byte(), z.as_byte(), __ptr__(LDZ).as_byte(), work.as_byte(), __ptr__(LW).as_byte(),
                        iwork.as_byte(), ifail.as_byte(), __ptr__(info).as_byte())
    elif driver == 'evd':
        qw = T()
        qr = 0.0
        qi = fint(0)
        LW = fint(-1)
        LRW = fint(-1)
        LIW = fint(-1)
        if cplx:
            zheevd_(__ptr__(jobz).as_byte(), __ptr__(uplo).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), W.as_byte(), __ptr__(qw).as_byte(), __ptr__(LW).as_byte(),
                    __ptr__(qr).as_byte(), __ptr__(LRW).as_byte(), __ptr__(qi).as_byte(), __ptr__(LIW).as_byte(), __ptr__(info).as_byte())
        else:
            dsyevd_(__ptr__(jobz).as_byte(), __ptr__(uplo).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), W.as_byte(), __ptr__(qw).as_byte(), __ptr__(LW).as_byte(),
                    __ptr__(qi).as_byte(), __ptr__(LIW).as_byte(), __ptr__(info).as_byte())
        LW = fint(max(_work_size(qw), 1))
        LRW = fint(max(int(qr), 1))
        LIW = fint(max(int(qi), 1))
        work = Ptr[T](int(LW))
        rwork = Ptr[float](int(LRW))
        iwork = Ptr[fint](int(LIW))
        info = fint(0)
        if cplx:
            zheevd_(__ptr__(jobz).as_byte(), __ptr__(uplo).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), W.as_byte(), work.as_byte(), __ptr__(LW).as_byte(),
                    rwork.as_byte(), __ptr__(LRW).as_byte(), iwork.as_byte(), __ptr__(LIW).as_byte(), __ptr__(info).as_byte())
        else:
            dsyevd_(__ptr__(jobz).as_byte(), __ptr__(uplo).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), W.as_byte(), work.as_byte(), __ptr__(LW).as_byte(),
                    iwork.as_byte(), __ptr__(LIW).as_byte(), __ptr__(info).as_byte())
    elif driver == 'ev':
        # f2py ?syev / ?heev default lwork
        lw = max(3 * n - 1, 1) if not cplx else max(2 * n - 1, 1)
        LW = fint(lw)
        work = Ptr[T](lw)
        rwork = Ptr[float](max(3 * n - 2, 1))
        if cplx:
            zheev_(__ptr__(jobz).as_byte(), __ptr__(uplo).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), W.as_byte(), work.as_byte(), __ptr__(LW).as_byte(),
                   rwork.as_byte(), __ptr__(info).as_byte())
        else:
            dsyev_(__ptr__(jobz).as_byte(), __ptr__(uplo).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), W.as_byte(), work.as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
    elif driver == 'gv':
        lw = max(3 * n - 1, 1) if not cplx else max(2 * n - 1, 1)
        LW = fint(lw)
        work = Ptr[T](lw)
        rwork = Ptr[float](max(3 * n - 2, 1))
        if cplx:
            zhegv_(__ptr__(ITYPE).as_byte(), __ptr__(jobz).as_byte(), __ptr__(uplo).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), Bm.as_byte(), __ptr__(LDA).as_byte(),
                   W.as_byte(), work.as_byte(), __ptr__(LW).as_byte(), rwork.as_byte(), __ptr__(info).as_byte())
        else:
            dsygv_(__ptr__(ITYPE).as_byte(), __ptr__(jobz).as_byte(), __ptr__(uplo).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), Bm.as_byte(), __ptr__(LDA).as_byte(),
                   W.as_byte(), work.as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
    else:  # gvd, f2py default workspace sizes (no query in scipy)
        if cplx:
            lw = (n + 1) if not compute_v else n * (n + 2)
            lrw = max(n if not compute_v else 2 * n * n + 5 * n + 1, 1)
        else:
            lw = (2 * n + 1) if not compute_v else 1 + 6 * n + 2 * n * n
            lrw = 1
        liw = 1 if not compute_v else 5 * n + 3
        LW = fint(lw)
        LRW = fint(lrw)
        LIW = fint(liw)
        work = Ptr[T](max(lw, 1))
        rwork = Ptr[float](lrw)
        iwork = Ptr[fint](max(liw, 1))
        if cplx:
            zhegvd_(__ptr__(ITYPE).as_byte(), __ptr__(jobz).as_byte(), __ptr__(uplo).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), Bm.as_byte(), __ptr__(LDA).as_byte(),
                    W.as_byte(), work.as_byte(), __ptr__(LW).as_byte(), rwork.as_byte(), __ptr__(LRW).as_byte(), iwork.as_byte(),
                    __ptr__(LIW).as_byte(), __ptr__(info).as_byte())
        else:
            dsygvd_(__ptr__(ITYPE).as_byte(), __ptr__(jobz).as_byte(), __ptr__(uplo).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), Bm.as_byte(), __ptr__(LDA).as_byte(),
                    W.as_byte(), work.as_byte(), __ptr__(LW).as_byte(), iwork.as_byte(), __ptr__(LIW).as_byte(), __ptr__(info).as_byte())
    return W, z, ldz, int(M), int(info)

def _eigh_2d(a, b = None, *, lower: bool = True, eigvals_only: Literal[bool] = False,
         overwrite_a: bool = False, overwrite_b: bool = False, type: int = 1,
         check_finite: bool = True, subset_by_index = None, subset_by_value = None,
         driver: Optional[str] = None):
    """
    Solve a standard or generalized eigenvalue problem for a complex
    Hermitian or real symmetric matrix. Returns ``w`` if ``eigvals_only``
    (must be a compile-time constant) else the pair ``(w, v)``.
    """
    return _eigh_impl(a, b, lower, eigvals_only, type, check_finite,
                      subset_by_index, subset_by_value, driver)

def _eigh_impl(a, b, lower: bool, eigvals_only: Literal[bool], itype: int,
               check_finite: bool, subset_by_index, subset_by_value,
               driver: Optional[str]):
    drv: str = driver if driver is not None else ''
    if drv not in ('', 'ev', 'evd', 'evr', 'evx', 'gv', 'gvd', 'gvx'):
        raise ValueError(f'"{drv}" is unknown. Possible values are "None", "ev", "evd", "evr", "evx", "gv", "gvd", "gvx".')

    a1 = _arr_lapack(a, check_finite)
    if a1.ndim != 2:
        compile_error("expected square \"a\" matrix (batched eigh is not supported)")
    if a1.shape[0] != a1.shape[1]:
        raise ValueError('expected square "a" matrix')
    n = a1.shape[0]

    if b is None:
        T = type(_promote(a1.dtype, a1.dtype))
        has_b = False
    else:
        b1 = _arr_lapack(b, check_finite)
        if b1.ndim != 2:
            compile_error("expected square \"b\" matrix")
        if b1.shape[0] != b1.shape[1]:
            raise ValueError('expected square "b" matrix')
        if b1.shape != a1.shape:
            raise ValueError(f"wrong b dimensions {b1.shape}, should be {a1.shape}")
        if itype not in (1, 2, 3):
            raise ValueError('"type" keyword only accepts 1, 2, and 3.')
        T = type(_promote(a1.dtype, b1.dtype))
        has_b = True

    if n == 0:
        w0 = np.empty(0, dtype=float)
        if eigvals_only:
            return w0
        else:
            return _Pair(w0, np.empty((0, 0), dtype=T))

    rng = u8(65)  # 'A'
    il, iu = 1, n
    vl, vu = 0.0, 1.0
    subset = False
    if subset_by_index is not None:
        if subset_by_value is not None:
            raise ValueError('Either index or value subset can be requested.')
        lo = int(subset_by_index[0])
        hi = int(subset_by_index[1])
        if not (0 <= lo <= hi < n):
            raise ValueError('Requested eigenvalue indices are not valid. '
                             f'Valid range is [0, {n-1}] and start <= end, but '
                             f'start={lo}, end={hi} is given')
        rng = u8(73)  # 'I'
        il, iu = lo + 1, hi + 1
        subset = True
    if subset_by_value is not None:
        lo = float(subset_by_value[0])
        hi = float(subset_by_value[1])
        if not (-np.inf <= lo < hi <= np.inf):
            raise ValueError('Requested eigenvalue bounds are not valid. '
                             'Valid range is (-inf, inf) and low < high, but '
                             f'low={lo}, high={hi} is given')
        rng = u8(86)  # 'V'
        vl, vu = lo, hi
        subset = True

    if drv:
        if not has_b and drv in ('gv', 'gvd', 'gvx'):
            raise ValueError(f'{drv} requires input b array to be supplied '
                             'for generalized eigenvalue problems.')
        if has_b and drv in ('ev', 'evd', 'evr', 'evx'):
            raise ValueError(f'"{drv}" does not accept input b array '
                             'for standard eigenvalue problems.')
        if subset and drv in ('ev', 'evd', 'gv', 'gvd'):
            raise ValueError(f'"{drv}" cannot compute subsets of eigenvalues')
    else:
        drv = 'evr' if not has_b else ('gvx' if subset else 'gvd')

    A = _to_f_as(a1, T)
    if b is None:
        Bm = Ptr[T]()
    else:
        Bm = _to_f_as(b1, T)

    W, Z, ldz, m, info = _eigh_core(A, Bm, n, has_b, lower, not eigvals_only, itype,
                                    drv, rng, vl, vu, il, iu)
    if info != 0:
        _err(info, n, drv, eigvals_only)

    mm = m if subset else n
    w = _vec(W, mm)
    if eigvals_only:
        return w
    else:
        v = _from_f(Z, n, mm, ldz)
        return _Pair(w, v)
