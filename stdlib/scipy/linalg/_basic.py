# Ported from scipy 1.18.1 (BSD-3-Clause)

# scipy.linalg basic routines: solve, solve_triangular, inv, det, cholesky,
# cho_factor/cho_solve, lu/lu_factor/lu_solve, qr, svd/svdvals, lstsq,
# pinv/pinvh, norm, orth, null_space, eig/eigvals and special matrices.
# Only 2-D inputs are supported (scipy's batch dimensions are not).

import numpy as np
from numpy.linalg import LinAlgError
from numpy.linalg.blas import fint
from ._lapack import *
from ._util import _arr, _arr_lapack, _to_f_as, _from_f, _vec, _is_complex, \
                   _work_size, _promote, _Pair, _Triple
from ._decomp import _eigh_2d

def _ch(c: str):
    return u8(ord(c))

def _sq(a, name: str = 'a'):
    if a.ndim != 2:
        compile_error("expected a 2-D matrix")
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"Expected square matrix, got a1.shape={a.shape}")
    return a.shape[0]

def _rhs2(b, T: type):
    # returns (F-ordered buffer, n rows, nrhs)
    if b.ndim == 1:
        n = b.shape[0]
        p = Ptr[T](max(n, 1))
        for i in range(n):
            p[i] = T(b[i])
        return p, n, 1
    elif b.ndim == 2:
        return _to_f_as(b, T), b.shape[0], b.shape[1]
    else:
        compile_error("b must be 1-D or 2-D")

def _rhs_out(p: Ptr[T], b, n: int, nrhs: int, T: type):
    if b.ndim == 1:
        return _vec(p, n)
    else:
        return _from_f(p, n, nrhs, n)

def _is_triu(a) -> bool:
    n, m = a.shape
    for i in range(n):
        for j in range(min(i, m)):
            if a[i, j] != type(a[i, j])():
                return False
    return True

def _is_tril(a) -> bool:
    n, m = a.shape
    for i in range(n):
        for j in range(i + 1, m):
            if a[i, j] != type(a[i, j])():
                return False
    return True

def _getrf(A: Ptr[T], m: int, n: int, T: type):
    M = fint(m); N = fint(n); LDA = fint(max(m, 1))
    piv = Ptr[fint](max(min(m, n), 1))
    info = fint(0)
    if _is_complex(T):
        zgetrf_(__ptr__(M).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), piv.as_byte(), __ptr__(info).as_byte())
    else:
        dgetrf_(__ptr__(M).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), piv.as_byte(), __ptr__(info).as_byte())
    return piv, int(info)

def _getrs(A: Ptr[T], piv: Ptr[fint], n: int, B: Ptr[T], nrhs: int, trans: u8, T: type):
    N = fint(n); NR = fint(nrhs); LD = fint(max(n, 1)); info = fint(0)
    if _is_complex(T):
        zgetrs_(__ptr__(trans).as_byte(), __ptr__(N).as_byte(), __ptr__(NR).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), piv.as_byte(), B.as_byte(), __ptr__(LD).as_byte(), __ptr__(info).as_byte())
    else:
        dgetrs_(__ptr__(trans).as_byte(), __ptr__(N).as_byte(), __ptr__(NR).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), piv.as_byte(), B.as_byte(), __ptr__(LD).as_byte(), __ptr__(info).as_byte())
    return int(info)

def _potrf(A: Ptr[T], n: int, lower: bool, T: type):
    uplo = _ch('L' if lower else 'U')
    N = fint(n); LD = fint(max(n, 1)); info = fint(0)
    if _is_complex(T):
        zpotrf_(__ptr__(uplo).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), __ptr__(info).as_byte())
    else:
        dpotrf_(__ptr__(uplo).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), __ptr__(info).as_byte())
    return int(info)

def _potrs(A: Ptr[T], n: int, lower: bool, B: Ptr[T], nrhs: int, T: type):
    uplo = _ch('L' if lower else 'U')
    N = fint(n); NR = fint(nrhs); LD = fint(max(n, 1)); info = fint(0)
    if _is_complex(T):
        zpotrs_(__ptr__(uplo).as_byte(), __ptr__(N).as_byte(), __ptr__(NR).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), B.as_byte(), __ptr__(LD).as_byte(), __ptr__(info).as_byte())
    else:
        dpotrs_(__ptr__(uplo).as_byte(), __ptr__(N).as_byte(), __ptr__(NR).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), B.as_byte(), __ptr__(LD).as_byte(), __ptr__(info).as_byte())
    return int(info)

def _trtrs(A: Ptr[T], n: int, lower: bool, trans: u8, unit: bool, B: Ptr[T], nrhs: int, T: type):
    uplo = _ch('L' if lower else 'U')
    diag = _ch('U' if unit else 'N')
    N = fint(n); NR = fint(nrhs); LD = fint(max(n, 1)); info = fint(0)
    if _is_complex(T):
        ztrtrs_(__ptr__(uplo).as_byte(), __ptr__(trans).as_byte(), __ptr__(diag).as_byte(), __ptr__(N).as_byte(), __ptr__(NR).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), B.as_byte(), __ptr__(LD).as_byte(), __ptr__(info).as_byte())
    else:
        dtrtrs_(__ptr__(uplo).as_byte(), __ptr__(trans).as_byte(), __ptr__(diag).as_byte(), __ptr__(N).as_byte(), __ptr__(NR).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), B.as_byte(), __ptr__(LD).as_byte(), __ptr__(info).as_byte())
    return int(info)

def _sysv(A: Ptr[T], n: int, lower: bool, herm: bool, B: Ptr[T], nrhs: int, T: type):
    uplo = _ch('L' if lower else 'U')
    N = fint(n); NR = fint(nrhs); LD = fint(max(n, 1)); info = fint(0)
    piv = Ptr[fint](max(n, 1))
    q = T()
    LW = fint(-1)
    for it in range(2):
        if it == 1:
            LW = fint(max(_work_size(q), 1))
            work = Ptr[T](int(LW))
        else:
            work = __ptr__(q)
        if _is_complex(T):
            if herm:
                zhesv_(__ptr__(uplo).as_byte(), __ptr__(N).as_byte(), __ptr__(NR).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), piv.as_byte(), B.as_byte(), __ptr__(LD).as_byte(), work.as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
            else:
                zsysv_(__ptr__(uplo).as_byte(), __ptr__(N).as_byte(), __ptr__(NR).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), piv.as_byte(), B.as_byte(), __ptr__(LD).as_byte(), work.as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
        else:
            dsysv_(__ptr__(uplo).as_byte(), __ptr__(N).as_byte(), __ptr__(NR).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), piv.as_byte(), B.as_byte(), __ptr__(LD).as_byte(), work.as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
    return int(info)

def _structure(assume_a: Optional[str]) -> int:
    if assume_a is None:
        return -1
    s = assume_a
    if s == 'general' or s == 'gen':
        return 0
    if s == 'diagonal':
        return 11
    if s == 'tridiagonal':
        return 0
    if s == 'banded':
        return 0
    if s == 'upper triangular':
        return 21
    if s == 'lower triangular':
        return 22
    if s == 'pos' or s == 'positive definite':
        return 101
    if s == 'sym' or s == 'symmetric':
        return 201
    if s == 'her' or s == 'hermitian':
        return 211
    raise ValueError(f'{s} is not a recognized matrix structure')

def _detect(a) -> int:
    up = _is_triu(a)
    lo = _is_tril(a)
    if up and lo:
        return 11
    if up:
        return 21
    if lo:
        return 22
    return 0

def _solve_2d(a, b, lower: bool = False, overwrite_a: bool = False,
          overwrite_b: bool = False, check_finite: bool = True,
          assume_a: Optional[str] = None, transposed: bool = False):
    """Solve ``a @ x == b`` for ``x`` (2-D ``a``; 1-D or 2-D ``b``)."""
    structure = _structure(assume_a)
    a1 = _arr_lapack(a, check_finite)
    b1 = _arr_lapack(b, check_finite)
    T = type(_promote(a1.dtype, b1.dtype))
    n = _sq(a1)
    if _is_complex(T) and transposed:
        raise NotImplementedError('scipy.linalg.solve can currently not solve a^T x = b or a^H x = b for complex matrices.')
    if b1.shape[0] != n and n != 1:
        raise ValueError(f"incompatible shapes: a1.shape={a1.shape} and b1.shape={b1.shape}")
    if structure == -1:
        structure = _detect(a1)
    B, nb, nrhs = _rhs2(b1, T)
    if n == 0:
        return _rhs_out(B, b1, nb, nrhs)
    A = _to_f_as(a1, T)
    tr = _ch('T' if transposed else 'N')
    if structure == 11:
        for i in range(n):
            d = A[i + i * n]
            if d == T():
                raise LinAlgError("A singular matrix detected.")
            for k in range(nrhs):
                B[i + k * n] = B[i + k * n] / d
    elif structure == 21 or structure == 22:
        info = _trtrs(A, n, structure == 22, tr, False, B, nrhs)
        if info > 0:
            raise LinAlgError("A singular matrix detected.")
    elif structure == 101:
        if transposed:
            # a is symmetric: a.T = a
            pass
        info = _potrf(A, n, lower)
        if info > 0:
            raise LinAlgError("A singular matrix detected.")
        _potrs(A, n, lower, B, nrhs)
    elif structure == 201 or structure == 211:
        info = _sysv(A, n, lower, structure == 211, B, nrhs)
        if info > 0:
            raise LinAlgError("A singular matrix detected.")
    else:
        piv, info = _getrf(A, n, n)
        if info > 0:
            raise LinAlgError("A singular matrix detected.")
        _getrs(A, piv, n, B, nrhs, tr)
    return _rhs_out(B, b1, n, nrhs)

def solve_triangular(a, b, trans = 0, lower: bool = False,
                     unit_diagonal: bool = False, overwrite_b: bool = False,
                     check_finite: bool = True):
    """Solve ``a x = b`` for triangular ``a``. ``trans``: 0/'N', 1/'T', 2/'C'."""
    a1 = _arr_lapack(a, check_finite)
    b1 = _arr_lapack(b, check_finite)
    T = type(_promote(a1.dtype, b1.dtype))
    n = _sq(a1)
    if b1.shape[0] != n:
        raise ValueError(f"shapes of a {a1.shape} and b {b1.shape} are incompatible")
    if isinstance(trans, str):
        t = trans
    else:
        t = ('N', 'T', 'C')[int(trans)]
    A = _to_f_as(a1, T)
    B, nb, nrhs = _rhs2(b1, T)
    info = _trtrs(A, n, lower, _ch(t), unit_diagonal, B, nrhs)
    if info > 0:
        raise LinAlgError(f"singular matrix: resolution failed at diagonal {info-1}")
    if info < 0:
        raise ValueError(f'illegal value in {-info}-th argument of internal trtrs')
    return _rhs_out(B, b1, n, nrhs)

def _inv_2d(a, overwrite_a: bool = False, check_finite: bool = True, *,
        assume_a: Optional[str] = None, lower: bool = False):
    """Compute the inverse of a matrix."""
    a1 = _arr_lapack(a, check_finite)
    T = a1.dtype
    n = _sq(a1)
    structure = _structure(assume_a)
    if structure == -1:
        structure = _detect(a1)
    A = _to_f_as(a1, T)
    if n == 0:
        return np.empty((0, 0), dtype=T)
    N = fint(n); LD = fint(n); info = fint(0)
    if structure == 11:
        r = np.zeros((n, n), dtype=T)
        for i in range(n):
            if A[i + i * n] == T():
                raise LinAlgError("A singular matrix detected.")
            r[i, i] = T(1) / A[i + i * n]
        return r
    elif structure == 21 or structure == 22:
        uplo = _ch('L' if structure == 22 else 'U')
        diag = _ch('N')
        if _is_complex(T):
            ztrtri_(__ptr__(uplo).as_byte(), __ptr__(diag).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), __ptr__(info).as_byte())
        else:
            dtrtri_(__ptr__(uplo).as_byte(), __ptr__(diag).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), __ptr__(info).as_byte())
        if int(info) > 0:
            raise LinAlgError("A singular matrix detected.")
        r = _from_f(A, n, n, n)
        return np.triu(r) if structure == 21 else np.tril(r)
    elif structure == 101:
        if _potrf(A, n, lower) > 0:
            raise LinAlgError("A singular matrix detected.")
        uplo = _ch('L' if lower else 'U')
        if _is_complex(T):
            zpotri_(__ptr__(uplo).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), __ptr__(info).as_byte())
        else:
            dpotri_(__ptr__(uplo).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), __ptr__(info).as_byte())
        r = _from_f(A, n, n, n)
        for i in range(n):
            for j in range(n):
                if (lower and j > i) or (not lower and j < i):
                    v = r[j, i]
                    if _is_complex(T):
                        r[i, j] = v.conjugate()
                    else:
                        r[i, j] = v
        return r
    else:
        piv, inf = _getrf(A, n, n)
        if inf > 0:
            raise LinAlgError("A singular matrix detected.")
        q = T()
        LW = fint(-1)
        if _is_complex(T):
            zgetri_(__ptr__(N).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), piv.as_byte(), __ptr__(q).as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
        else:
            dgetri_(__ptr__(N).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), piv.as_byte(), __ptr__(q).as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
        LW = fint(max(_work_size(q), 1))
        work = Ptr[T](int(LW))
        if _is_complex(T):
            zgetri_(__ptr__(N).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), piv.as_byte(), work.as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
        else:
            dgetri_(__ptr__(N).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), piv.as_byte(), work.as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
        return _from_f(A, n, n, n)

def _det_2d(a, overwrite_a: bool = False, check_finite: bool = True):
    """Determinant via LU (getrf), product of U's diagonal with pivot signs."""
    a1 = _arr_lapack(a, check_finite)
    T = a1.dtype
    n = _sq(a1)
    if n == 0:
        return T(1)
    A = _to_f_as(a1, T)
    piv, info = _getrf(A, n, n)
    d = T(1)
    for i in range(n):
        d *= A[i + i * n]
        if int(piv[i]) != i + 1:
            d = -d
    return d

def _cholesky_2d(a, lower: bool = False, overwrite_a: bool = False, check_finite: bool = True):
    """Cholesky factor; upper triangular ``U`` (``a = U^H U``) unless ``lower``."""
    a1 = _arr_lapack(a, check_finite)
    T = a1.dtype
    n = _sq(a1)
    A = _to_f_as(a1, T)
    info = _potrf(A, n, lower)
    if info > 0:
        raise LinAlgError(f"{info}-th leading minor of the array is not positive definite")
    r = _from_f(A, n, n, n)
    return np.tril(r) if lower else np.triu(r)

def cho_factor(a, lower: bool = False, overwrite_a: bool = False, check_finite: bool = True):
    """Returns ``(c, lower)``; the unused triangle of ``c`` keeps the input values."""
    a1 = _arr_lapack(a, check_finite)
    T = a1.dtype
    n = _sq(a1)
    A = _to_f_as(a1, T)
    info = _potrf(A, n, lower)
    if info > 0:
        raise LinAlgError(f"{info}-th leading minor of the array is not positive definite")
    return _Pair(_from_f(A, n, n, n), lower)

def cho_solve(c_and_lower, b, overwrite_b: bool = False, check_finite: bool = True):
    c = _arr_lapack(c_and_lower[0], check_finite)
    lower = bool(c_and_lower[1])
    b1 = _arr_lapack(b, check_finite)
    T = type(_promote(c.dtype, b1.dtype))
    n = _sq(c)
    if b1.shape[0] != n:
        raise ValueError(f"incompatible dimensions ({c.shape} and {b1.shape})")
    A = _to_f_as(c, T)
    B, nb, nrhs = _rhs2(b1, T)
    _potrs(A, n, lower, B, nrhs)
    return _rhs_out(B, b1, n, nrhs)

def lu_factor(a, overwrite_a: bool = False, check_finite: bool = True):
    """Returns ``(lu, piv)`` with 0-based pivot indices."""
    a1 = _arr_lapack(a, check_finite)
    T = a1.dtype
    m, n = a1.shape
    A = _to_f_as(a1, T)
    piv, info = _getrf(A, m, n)
    k = min(m, n)
    p = np.empty(k, dtype=int)
    for i in range(k):
        p[i] = int(piv[i]) - 1
    return _Pair(_from_f(A, m, n, m), p)

def lu_solve(lu_and_piv, b, trans: int = 0, overwrite_b: bool = False, check_finite: bool = True):
    lu = _arr_lapack(lu_and_piv[0], check_finite)
    pv = lu_and_piv[1]
    b1 = _arr_lapack(b, check_finite)
    T = type(_promote(lu.dtype, b1.dtype))
    n = _sq(lu)
    if b1.shape[0] != n:
        raise ValueError(f"Shapes of lu {lu.shape} and b {b1.shape} are incompatible")
    A = _to_f_as(lu, T)
    piv = Ptr[fint](max(n, 1))
    for i in range(n):
        piv[i] = fint(int(pv[i]) + 1)
    B, nb, nrhs = _rhs2(b1, T)
    _getrs(A, piv, n, B, nrhs, _ch(('N', 'T', 'C')[trans]))
    return _rhs_out(B, b1, n, nrhs)

def _lu_2d(a, permute_l: Literal[bool] = False, overwrite_a: bool = False,
       check_finite: bool = True, p_indices: Literal[bool] = False):
    """``(p, l, u)`` with ``a = p @ l @ u``, or ``(pl, u)`` if ``permute_l``."""
    a1 = _arr_lapack(a, check_finite)
    T = a1.dtype
    m, n = a1.shape
    k = min(m, n)
    A = _to_f_as(a1, T)
    piv, info = _getrf(A, m, n)
    perm = [i for i in range(m)]
    for i in range(k):
        j = int(piv[i]) - 1
        perm[i], perm[j] = perm[j], perm[i]
    # row i of (P^T a) is row perm[i] of a; a = P L U with P[perm[i], i] = 1
    L = np.zeros((m, k), dtype=T)
    U = np.zeros((k, n), dtype=T)
    for i in range(m):
        for j in range(k):
            if i == j:
                L[i, j] = T(1)
            elif i > j:
                L[i, j] = A[i + j * m]
    for i in range(k):
        for j in range(i, n):
            U[i, j] = A[i + j * m]
    if permute_l:
        PL = np.zeros((m, k), dtype=T)
        for i in range(m):
            for j in range(k):
                PL[perm[i], j] = L[i, j]
        return _Pair(PL, U)
    else:
        if p_indices:
            p = np.array(perm)
            return _Triple(p, L, U)
        else:
            P = np.zeros((m, m), dtype=type(_basetype_of(T)))
            for i in range(m):
                P[perm[i], i] = 1.0
            return _Triple(P, L, U)

def _basetype_of(T: type):
    return float()

def _gesdd(a1, full: bool, compute_uv: bool, T: type):
    m, n = a1.shape
    k = min(m, n)
    A = _to_f_as(a1, T)
    S = Ptr[float](max(k, 1))
    jobz = _ch('N') if not compute_uv else (_ch('A') if full else _ch('S'))
    ucol = m if full else k
    vrow = n if full else k
    U = Ptr[T](max(m * ucol, 1))
    VT = Ptr[T](max(vrow * n, 1))
    M = fint(m); N = fint(n); LDA = fint(max(m, 1)); LDU = fint(max(m, 1)); LDVT = fint(max(vrow, 1))
    iwork = Ptr[fint](max(8 * k, 1))
    mx = max(m, n)
    lrw = max(5 * k * k + 5 * k, 2 * mx * k + 2 * k * k + k) if compute_uv else 7 * k
    rwork = Ptr[float](max(lrw, 1))
    info = fint(0)
    q = T()
    LW = fint(-1)
    for it in range(2):
        if it == 1:
            LW = fint(max(_work_size(q), 1))
            work = Ptr[T](int(LW))
        else:
            work = __ptr__(q)
        if _is_complex(T):
            zgesdd_(__ptr__(jobz).as_byte(), __ptr__(M).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), S.as_byte(), U.as_byte(), __ptr__(LDU).as_byte(), VT.as_byte(), __ptr__(LDVT).as_byte(), work.as_byte(), __ptr__(LW).as_byte(), rwork.as_byte(), iwork.as_byte(), __ptr__(info).as_byte())
        else:
            dgesdd_(__ptr__(jobz).as_byte(), __ptr__(M).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), S.as_byte(), U.as_byte(), __ptr__(LDU).as_byte(), VT.as_byte(), __ptr__(LDVT).as_byte(), work.as_byte(), __ptr__(LW).as_byte(), iwork.as_byte(), __ptr__(info).as_byte())
    if int(info) > 0:
        raise LinAlgError("SVD did not converge")
    return _Triple(_from_f(U, m, ucol, m), _vec(S, k), _from_f(VT, vrow, n, vrow))

def _svd_2d(a, full_matrices: bool = True, compute_uv: Literal[bool] = True,
        overwrite_a: bool = False, check_finite: bool = True,
        lapack_driver: str = 'gesdd'):
    """SVD via LAPACK ?gesdd (the only driver implemented)."""
    if lapack_driver != 'gesdd' and lapack_driver != 'gesvd':
        raise ValueError(f'lapack_driver must be "gesdd" or "gesvd", not "{lapack_driver}"')
    a1 = _arr_lapack(a, check_finite)
    if a1.ndim != 2:
        compile_error("expected matrix")
    r = _gesdd(a1, full_matrices, compute_uv, a1.dtype)
    if compute_uv:
        return r
    else:
        return r[1]

def _svdvals_2d(a, overwrite_a: bool = False, check_finite: bool = True):
    return _svd_2d(a, compute_uv=False, check_finite=check_finite)

def _herm_t(a):
    # conjugate transpose copy
    m, n = a.shape
    r = np.empty((n, m), dtype=a.dtype)
    for i in range(m):
        for j in range(n):
            if _is_complex(a.dtype):
                r[j, i] = a[i, j].conjugate()
            else:
                r[j, i] = a[i, j]
    return r

def _pinv_2d(a, *, atol = None, rtol = None, return_rank: Literal[bool] = False,
         check_finite: bool = True):
    a1 = _arr_lapack(a, check_finite)
    u, s, vh = _svd_2d(a1, full_matrices=False, check_finite=False)
    m, n = a1.shape
    at = 0.0 if atol is None else float(atol)
    rt = float(max(m, n)) * 2.220446049250313e-16 if rtol is None else float(rtol)
    if at < 0. or rt < 0.:
        raise ValueError("atol and rtol values must be positive.")
    maxS = 0.0
    for v in s:
        maxS = max(maxS, v)
    val = at + maxS * rt
    k = s.shape[0]
    rank = 0
    for i in range(k):
        if s[i] > val:
            s[i] = 1.0 / s[i]
            rank += 1
        else:
            s[i] = 0.0
    # B = vh^H @ (s[:, None] * u^H)
    uh = _herm_t(u)
    for i in range(k):
        for j in range(m):
            uh[i, j] = uh[i, j] * s[i]
    B = _herm_t(vh) @ uh
    if return_rank:
        return _Pair(B, rank)
    else:
        return B

def pinvh(a, atol = None, rtol = None, lower: bool = True,
          return_rank: Literal[bool] = False, check_finite: bool = True):
    a1 = _arr_lapack(a, check_finite)
    s, u = _eigh_2d(a1, lower=lower, check_finite=False, driver='ev')
    n = a1.shape[0]
    at = 0.0 if atol is None else float(atol)
    rt = float(n) * 2.220446049250313e-16 if rtol is None else float(rtol)
    if at < 0. or rt < 0.:
        raise ValueError("atol and rtol values must be positive.")
    maxS = 0.0
    for v in s:
        maxS = max(maxS, abs(v))
    val = at + maxS * rt
    idx = [i for i in range(n) if abs(s[i]) > val]
    r = len(idx)
    T = a1.dtype
    U = np.empty((n, r), dtype=T)
    for c in range(r):
        for i in range(n):
            U[i, c] = u[i, idx[c]] / s[idx[c]]
    Ub = np.empty((n, r), dtype=T)
    for c in range(r):
        for i in range(n):
            Ub[i, c] = u[i, idx[c]]
    B = U @ _herm_t(Ub)
    if return_rank:
        return _Pair(B, r)
    else:
        return B

def lstsq(a, b, cond = None, overwrite_a: bool = False, overwrite_b: bool = False,
          check_finite: bool = True, lapack_driver: Optional[str] = None):
    """Least squares via ?gelsd. Returns ``(x, residues, rank, s)``. Real only."""
    a1 = _arr_lapack(a, check_finite)
    b1 = _arr_lapack(b, check_finite)
    m, n = a1.shape
    if b1.shape[0] != m:
        raise ValueError(f"Shape mismatch: a and b should have the same number of rows ({m} != {b1.shape[0]}).")
    nrhs = 1 if b1.ndim == 1 else b1.shape[1]
    ldb = max(m, n, 1)
    A = _to_f_as(a1, float)
    B = Ptr[float](ldb * max(nrhs, 1))
    for k in range(nrhs):
        for i in range(m):
            if b1.ndim == 1:
                B[i] = float(b1[i])
            else:
                B[i + k * ldb] = float(b1[i, k])
    k = min(m, n)
    S = Ptr[float](max(k, 1))
    rcond = -1.0 if cond is None else float(cond)
    M = fint(m); N = fint(n); NR = fint(nrhs); LDA = fint(max(m, 1)); LDB = fint(ldb)
    rank = fint(0); info = fint(0)
    q = 0.0
    qi = fint(0)
    LW = fint(-1)
    dgelsd_(__ptr__(M).as_byte(), __ptr__(N).as_byte(), __ptr__(NR).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), B.as_byte(), __ptr__(LDB).as_byte(), S.as_byte(), __ptr__(rcond).as_byte(), __ptr__(rank).as_byte(), __ptr__(q).as_byte(), __ptr__(LW).as_byte(), __ptr__(qi).as_byte(), __ptr__(info).as_byte())
    LW = fint(max(int(q), 1))
    work = Ptr[float](int(LW))
    iwork = Ptr[fint](max(int(qi), 1))
    dgelsd_(__ptr__(M).as_byte(), __ptr__(N).as_byte(), __ptr__(NR).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), B.as_byte(), __ptr__(LDB).as_byte(), S.as_byte(), __ptr__(rcond).as_byte(), __ptr__(rank).as_byte(), work.as_byte(), __ptr__(LW).as_byte(), iwork.as_byte(), __ptr__(info).as_byte())
    if int(info) > 0:
        raise LinAlgError("SVD did not converge in Linear Least Squares")
    resids = np.empty(nrhs if (m > n and int(rank) == n) else 0, dtype=float)
    if m > n and int(rank) == n:
        for c in range(nrhs):
            acc = 0.0
            for i in range(n, m):
                acc += B[i + c * ldb] * B[i + c * ldb]
            resids[c] = acc
    if b1.ndim == 1:
        x = np.empty(n, dtype=float)
        for i in range(n):
            x[i] = B[i]
        return (x, resids, int(rank), _vec(S, k))
    else:
        x = _from_f(B, n, nrhs, ldb)
        return (x, resids, int(rank), _vec(S, k))

def norm(a, ord = None, axis = None, keepdims: Literal[bool] = False, check_finite: bool = True):
    a1 = _arr(a)
    return np.linalg.norm(a1, ord=ord, axis=axis, keepdims=keepdims)

def orth(A, rcond = None):
    a1 = _arr_lapack(A)
    u, s, vh = _svd_2d(a1, full_matrices=False)
    M, N = a1.shape
    rc = 2.220446049250313e-16 * max(M, N) if rcond is None else float(rcond)
    mx = 0.0
    for v in s:
        mx = max(mx, v)
    tol = mx * rc
    num = 0
    for v in s:
        if v > tol:
            num += 1
    return u[:, :num].copy()

def null_space(A, rcond = None, *, overwrite_a: bool = False, check_finite: bool = True,
               lapack_driver: str = 'gesdd'):
    a1 = _arr_lapack(A, check_finite)
    u, s, vh = _svd_2d(a1, full_matrices=True)
    M, N = a1.shape
    rc = 2.220446049250313e-16 * max(M, N) if rcond is None else float(rcond)
    mx = 0.0
    for v in s:
        mx = max(mx, v)
    tol = mx * rc
    num = 0
    for v in s:
        if v > tol:
            num += 1
    Q = _herm_t(vh[num:, :])
    return Q

def _qr_2d(a, overwrite_a: bool = False, lwork = None, mode: Literal[str] = 'full',
       pivoting: Literal[bool] = False, check_finite: bool = True):
    """QR via ?geqrf (or ?geqp3 when ``pivoting``) + ?orgqr/?ungqr. With
    ``pivoting`` the permutation ``P`` (0-based) is returned last."""
    if mode != 'full' and mode != 'r' and mode != 'economic' and mode != 'raw':
        compile_error("Mode argument should be one of ['full', 'r', 'economic', 'raw']")
    a1 = _arr_lapack(a, check_finite)
    T = a1.dtype
    m, n = a1.shape
    k = min(m, n)
    ncq = m if mode == 'full' else k
    lda = m
    A = Ptr[T](max(m * max(n, ncq), 1))
    for j in range(n):
        for i in range(m):
            A[i + j * m] = a1[i, j]
    tau = Ptr[T](max(k, 1))
    M = fint(m); N = fint(n); LDA = fint(max(m, 1)); info = fint(0)
    q = T()
    LW = fint(-1)
    jpvt = Ptr[fint](max(n, 1))
    for i in range(n):
        jpvt[i] = fint(0)
    rw = Ptr[float](max(2 * n, 1))
    for it in range(2):
        if it == 1:
            LW = fint(max(_work_size(q), 1))
            work = Ptr[T](int(LW))
        else:
            work = __ptr__(q)
        if pivoting:
            if _is_complex(T):
                zgeqp3_(__ptr__(M).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), jpvt.as_byte(), tau.as_byte(), work.as_byte(), __ptr__(LW).as_byte(), rw.as_byte(), __ptr__(info).as_byte())
            else:
                dgeqp3_(__ptr__(M).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), jpvt.as_byte(), tau.as_byte(), work.as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
        elif _is_complex(T):
            zgeqrf_(__ptr__(M).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), tau.as_byte(), work.as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
        else:
            dgeqrf_(__ptr__(M).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), tau.as_byte(), work.as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
    rrows = k if (mode == 'economic' or mode == 'raw') else m
    R = np.zeros((rrows, n), dtype=T)
    for i in range(rrows):
        for j in range(i, n):
            R[i, j] = A[i + j * m]
    P = np.empty(n, dtype=int)
    for i in range(n):
        P[i] = int(jpvt[i]) - 1
    if mode == 'r':
        if pivoting:
            return _Pair(R, P)
        else:
            return R
    elif mode == 'raw':
        if pivoting:
            return _Triple(_Pair(_from_f(A, m, n, m), _vec(tau, k)), R, P)
        else:
            return _Pair(_Pair(_from_f(A, m, n, m), _vec(tau, k)), R)
    else:
        NC = fint(ncq); K = fint(k)
        q2 = T()
        LW = fint(-1)
        for it in range(2):
            if it == 1:
                LW = fint(max(_work_size(q2), 1))
                work = Ptr[T](int(LW))
            else:
                work = __ptr__(q2)
            if _is_complex(T):
                zungqr_(__ptr__(M).as_byte(), __ptr__(NC).as_byte(), __ptr__(K).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), tau.as_byte(), work.as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
            else:
                dorgqr_(__ptr__(M).as_byte(), __ptr__(NC).as_byte(), __ptr__(K).as_byte(), A.as_byte(), __ptr__(LDA).as_byte(), tau.as_byte(), work.as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
        if pivoting:
            return _Triple(_from_f(A, m, ncq, m), R, P)
        else:
            return _Pair(_from_f(A, m, ncq, m), R)

def _unpack_real_vecs(V: Ptr[float], wi: Ptr[float], n: int):
    # LAPACK real ?geev/?ggev packing: complex pair j, j+1 stored as re, im columns
    out = np.empty((n, n), dtype=complex)
    j = 0
    while j < n:
        if wi[j] == 0.0:
            for i in range(n):
                out[i, j] = complex(V[i + j * n], 0.0)
            j += 1
        else:
            for i in range(n):
                re = V[i + j * n]
                im = V[i + (j + 1) * n]
                out[i, j] = complex(re, im)
                out[i, j + 1] = complex(re, -im)
            j += 2
    return out

def _geev_core(a1, b1, has_b: bool, left: bool, right: bool, T: type):
    """Returns (alpha, beta, vl, vr) with complex alpha; beta is None-like ones
    for the standard problem."""
    n = a1.shape[0]
    A = _to_f_as(a1, T)
    Bm = _to_f_as(b1, T) if has_b else Ptr[T]()
    jobvl = _ch('V' if left else 'N')
    jobvr = _ch('V' if right else 'N')
    N = fint(n); LD = fint(max(n, 1)); info = fint(0)
    VL = Ptr[T](max(n * n, 1))
    VR = Ptr[T](max(n * n, 1))
    alpha = np.empty(n, dtype=complex)
    beta = np.ones(n, dtype=complex)
    q = T()
    LW = fint(-1)
    if _is_complex(T):
        W = Ptr[complex](max(n, 1))
        BE = Ptr[complex](max(n, 1))
        rwork = Ptr[float](max(8 * n, 1))
        for it in range(2):
            if it == 1:
                LW = fint(max(_work_size(q), 1))
                work = Ptr[T](int(LW))
            else:
                work = __ptr__(q)
            if has_b:
                zggev_(__ptr__(jobvl).as_byte(), __ptr__(jobvr).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), Bm.as_byte(), __ptr__(LD).as_byte(), W.as_byte(), BE.as_byte(), VL.as_byte(), __ptr__(LD).as_byte(), VR.as_byte(), __ptr__(LD).as_byte(), work.as_byte(), __ptr__(LW).as_byte(), rwork.as_byte(), __ptr__(info).as_byte())
            else:
                zgeev_(__ptr__(jobvl).as_byte(), __ptr__(jobvr).as_byte(), __ptr__(N), A.as_byte(), __ptr__(LD), W.as_byte(), VL.as_byte(), __ptr__(LD), VR.as_byte(), __ptr__(LD), work.as_byte(), __ptr__(LW), rwork.as_byte(), __ptr__(info))
        for i in range(n):
            alpha[i] = W[i]
            if has_b:
                beta[i] = BE[i]
        vl = _from_f(VL, n, n, n)
        vr = _from_f(VR, n, n, n)
    else:
        WR = Ptr[float](max(n, 1))
        WI = Ptr[float](max(n, 1))
        BE = Ptr[float](max(n, 1))
        for it in range(2):
            if it == 1:
                LW = fint(max(_work_size(q), 1))
                work = Ptr[T](int(LW))
            else:
                work = __ptr__(q)
            if has_b:
                dggev_(__ptr__(jobvl).as_byte(), __ptr__(jobvr).as_byte(), __ptr__(N).as_byte(), A.as_byte(), __ptr__(LD).as_byte(), Bm.as_byte(), __ptr__(LD).as_byte(), WR.as_byte(), WI.as_byte(), BE.as_byte(), VL.as_byte(), __ptr__(LD).as_byte(), VR.as_byte(), __ptr__(LD).as_byte(), work.as_byte(), __ptr__(LW).as_byte(), __ptr__(info).as_byte())
            else:
                dgeev_(__ptr__(jobvl).as_byte(), __ptr__(jobvr).as_byte(), __ptr__(N), A.as_byte(), __ptr__(LD), WR.as_byte(), WI.as_byte(), VL.as_byte(), __ptr__(LD), VR.as_byte(), __ptr__(LD), work.as_byte(), __ptr__(LW), __ptr__(info))
        for i in range(n):
            alpha[i] = complex(WR[i], WI[i])
            if has_b:
                beta[i] = complex(BE[i], 0.0)
        vl = _unpack_real_vecs(VL, WI, n)
        vr = _unpack_real_vecs(VR, WI, n)
    if int(info) != 0:
        raise LinAlgError("eig algorithm did not converge (only eigenvalues with order >= %d have converged)" % int(info))
    if has_b:
        # ?ggev eigenvectors are not normalized; scipy scales columns to unit 2-norm
        for M in (vl, vr):
            for j in range(n):
                s = 0.0
                for i in range(n):
                    s += abs(M[i, j]) ** 2
                s = K_sqrt(s)
                if s != 0.0:
                    for i in range(n):
                        M[i, j] = M[i, j] / s
    return alpha, beta, vl, vr

def K_sqrt(x: float) -> float:
    from math import sqrt
    return sqrt(x)

def _make_eigvals(alpha, beta, has_b: bool, homogeneous: Literal[bool]):
    n = alpha.shape[0]
    if homogeneous:
        out = np.empty((2, n), dtype=complex)
        for i in range(n):
            out[0, i] = alpha[i]
            out[1, i] = beta[i]
        return out
    else:
        if not has_b:
            return alpha
        w = np.empty(n, dtype=complex)
        allreal = True
        for i in range(n):
            if alpha[i].imag != 0:
                allreal = False
        for i in range(n):
            if beta[i] != complex():
                w[i] = alpha[i] / beta[i]
            elif alpha[i] != complex():
                w[i] = complex(float('inf'), 0.0)
            else:
                w[i] = complex(float('nan'), 0.0) if allreal else complex(float('nan'), float('nan'))
        return w

def _eig_2d(a, b = None, left: Literal[bool] = False, right: Literal[bool] = True,
            overwrite_a: bool = False, overwrite_b: bool = False,
            check_finite: bool = True, homogeneous_eigvals: Literal[bool] = False):
    """Standard (?geev) or generalized (?ggev) eigenproblem. Eigenvectors are
    always complex arrays (scipy returns real ones when all eigenvalues are
    real)."""
    a1 = _arr_lapack(a, check_finite)
    n = _sq(a1)
    if b is None:
        T = a1.dtype
        al, be, vl, vr = _geev_core(a1, a1, False, left, right, T)
        w = _make_eigvals(al, be, False, homogeneous_eigvals)
    else:
        b1 = _arr_lapack(b, check_finite)
        if b1.shape != a1.shape:
            raise ValueError('a and b must have the same shape')
        T = type(_promote(a1.dtype, b1.dtype))
        al, be, vl, vr = _geev_core(a1, b1, True, left, right, T)
        w = _make_eigvals(al, be, True, homogeneous_eigvals)
    if left:
        if right:
            return _Triple(w, vl, vr)
        else:
            return _Pair(w, vl)
    elif right:
        return _Pair(w, vr)
    else:
        return w

def _eigvals_2d(a, b = None, overwrite_a: bool = False, check_finite: bool = True,
                homogeneous_eigvals: Literal[bool] = False):
    return _eig_2d(a, b, right=False, check_finite=check_finite,
                   homogeneous_eigvals=homogeneous_eigvals)

# Special matrices

def toeplitz(c, r = None):
    c1 = np.asarray(c).ravel()
    if r is None:
        r1 = c1.conj() if _is_complex(c1.dtype) else c1
    else:
        r1 = np.asarray(r).ravel()
    T = type(_promote(c1.dtype, r1.dtype)) if (_is_complex(c1.dtype) or _is_complex(r1.dtype)) else type(c1.data[0] + r1.data[0])
    m = c1.shape[0]
    n = r1.shape[0]
    out = np.empty((m, n), dtype=T)
    for i in range(m):
        for j in range(n):
            out[i, j] = T(c1[i - j]) if i >= j else T(r1[j - i])
    return out

def circulant(c):
    c1 = np.asarray(c).ravel()
    n = c1.shape[0]
    out = np.empty((n, n), dtype=c1.dtype)
    for i in range(n):
        for j in range(n):
            out[i, j] = c1[(i - j) % n]
    return out

def hankel(c, r = None):
    c1 = np.asarray(c).ravel()
    m = c1.shape[0]
    if r is None:
        out = np.zeros((m, m), dtype=c1.dtype)
        for i in range(m):
            for j in range(m):
                if i + j < m:
                    out[i, j] = c1[i + j]
        return out
    else:
        r1 = np.asarray(r).ravel()
        n = r1.shape[0]
        T = type(c1.data[0] + r1.data[0])
        out = np.empty((m, n), dtype=T)
        for i in range(m):
            for j in range(n):
                out[i, j] = T(c1[i + j]) if i + j < m else T(r1[i + j - m + 1])
        return out

def block_diag(*arrs):
    if static.len(arrs) == 0:
        return np.empty((1, 0), dtype=float)
    else:
        mats = tuple(np.atleast_2d(np.asarray(x)) for x in arrs)
        T = type(_sum_types(mats))
        R = 0
        C = 0
        for x in mats:
            R += x.shape[0]
            C += x.shape[1]
        out = np.zeros((R, C), dtype=T)
        r = 0
        c = 0
        for x in mats:
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    out[r + i, c + j] = T(x[i, j])
            r += x.shape[0]
            c += x.shape[1]
        return out

def _sum_types(mats):
    if static.len(mats) == 1:
        return mats[0].dtype()
    else:
        return mats[0].dtype() + _sum_types(mats[1:])

import internal.static as static
