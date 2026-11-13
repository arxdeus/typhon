# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Minimal scipy.linalg.get_blas_funcs / get_lapack_funcs. The routine is
# selected by its (prefix-less) name given as a string literal, and a
# float64 implementation is returned (the prefix of the dtype is ignored:
# arrays are processed in double precision). Supported BLAS: copy, scal,
# axpy, dot, nrm2, asum, gemv, gemm, syrk, swap, ger. LAPACK: gesv, posv,
# getrf, getrs, potrf, potrs, geqrf, gelsd (via lstsq), syev (eigh),
# gesdd (svd).

import numpy as np
import scipy.linalg as _la

def _blas_copy(x, n = -1, offx = 0, incx = 1, y = None, offy = 0, incy = 1):
    return np.array(np.asarray(x, dtype=float))

def _blas_scal(a: float, x, n = -1, offx = 0, incx = 1):
    return np.asarray(x, dtype=float) * a

def _blas_axpy(x, y, n = -1, a: float = 1.0, offx = 0, incx = 1, offy = 0, incy = 1):
    return np.asarray(y, dtype=float) + a * np.asarray(x, dtype=float)

def _blas_dot(x, y):
    return float(np.dot(np.asarray(x, dtype=float), np.asarray(y, dtype=float)))

def _blas_nrm2(x, n = -1, offx = 0, incx = 1):
    return float(np.linalg.norm(np.asarray(x, dtype=float)))

def _blas_asum(x, n = -1, offx = 0, incx = 1):
    return float(np.sum(np.abs(np.asarray(x, dtype=float))))

def _blas_gemv(alpha: float, a, x, beta: float = 0.0, y = None, offx = 0, incx = 1,
               offy = 0, incy = 1, trans: int = 0, overwrite_y: int = 0):
    A = np.asarray(a, dtype=float)
    X = np.asarray(x, dtype=float)
    r = alpha * ((A.T @ X) if trans == 1 else (A @ X))
    if y is not None and beta != 0.0:
        r = r + beta * np.asarray(y, dtype=float)
    return r

def _blas_gemm(alpha: float, a, b, beta: float = 0.0, c = None, trans_a: int = 0,
               trans_b: int = 0, overwrite_c: int = 0):
    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)
    if trans_a != 0:
        A = A.T
    if trans_b != 0:
        B = B.T
    r = alpha * (A @ B)
    if c is not None and beta != 0.0:
        r = r + beta * np.asarray(c, dtype=float)
    return r

def _blas_syrk(alpha: float, a, beta: float = 0.0, c = None, trans: int = 0, lower: int = 0,
               overwrite_c: int = 0):
    A = np.asarray(a, dtype=float)
    r = alpha * ((A.T @ A) if trans != 0 else (A @ A.T))
    if c is not None and beta != 0.0:
        r = r + beta * np.asarray(c, dtype=float)
    n = r.shape[0]
    for i in range(n):
        for j in range(n):
            if (lower != 0 and j > i) or (lower == 0 and j < i):
                r[i, j] = 0.0
    return r

def _blas_ger(alpha: float, x, y, incx = 1, incy = 1, a = None, overwrite_x = 1, overwrite_y = 1,
              overwrite_a = 0):
    r = alpha * np.outer(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if a is not None:
        r = r + np.asarray(a, dtype=float)
    return r

def get_blas_funcs(name: Literal[str], arrays = (), dtype = None, ilp64: bool = False):
    """Return the BLAS routine `name` ('gemm', 'dgemm', 'axpy', ...)."""
    nm: Literal[str] = name[1:] if (len(name) > 1 and name[0] in ("s", "d", "c", "z")) and name[1:] in ("copy", "scal", "axpy", "dot", "nrm2", "asum", "gemv", "gemm", "syrk", "ger") else name
    if nm == "copy":
        return _blas_copy
    elif nm == "scal":
        return _blas_scal
    elif nm == "axpy":
        return _blas_axpy
    elif nm == "dot":
        return _blas_dot
    elif nm == "nrm2":
        return _blas_nrm2
    elif nm == "asum":
        return _blas_asum
    elif nm == "gemv":
        return _blas_gemv
    elif nm == "gemm":
        return _blas_gemm
    elif nm == "syrk":
        return _blas_syrk
    elif nm == "ger":
        return _blas_ger
    else:
        compile_error("get_blas_funcs: unsupported routine")

def _lp_gesv(a, b, overwrite_a = 0, overwrite_b = 0):
    A = np.asarray(a, dtype=float)
    x = _la.solve(A, np.asarray(b, dtype=float))
    return A, np.arange(A.shape[0]), x, 0

def _lp_posv(a, b, lower: int = 0, overwrite_a = 0, overwrite_b = 0):
    A = np.asarray(a, dtype=float)
    x = _la.solve(A, np.asarray(b, dtype=float))
    return _la.cholesky(A, lower=lower != 0), x, 0

def _lp_getrf(a, overwrite_a = 0):
    A = np.asarray(a, dtype=float)
    lu, piv = _la.lu_factor(A)
    return lu, piv, 0

def _lp_getrs(lu, piv, b, trans: int = 0, overwrite_b = 0):
    return _la.lu_solve((np.asarray(lu, dtype=float), np.asarray(piv)), np.asarray(b, dtype=float), trans=trans), 0

def _lp_potrf(a, lower: int = 0, clean: int = 1, overwrite_a = 0):
    return _la.cholesky(np.asarray(a, dtype=float), lower=lower != 0), 0

def _lp_potrs(c, b, lower: int = 0, overwrite_b = 0):
    return _la.cho_solve((np.asarray(c, dtype=float), lower != 0), np.asarray(b, dtype=float)), 0

def _lp_geqrf(a, overwrite_a = 0):
    q, r = np.linalg.qr(np.asarray(a, dtype=float), mode='complete')
    return r, q, 0

def _lp_syev(a, compute_v: int = 1, lower: int = 0, overwrite_a = 0):
    w, v = np.linalg.eigh(np.asarray(a, dtype=float), UPLO='L' if lower != 0 else 'U')
    return w, v, 0

def _lp_gesdd(a, compute_uv: int = 1, full_matrices: int = 1, overwrite_a = 0):
    u, s, vt = np.linalg.svd(np.asarray(a, dtype=float), full_matrices=full_matrices != 0)
    return u, s, vt, 0

def get_lapack_funcs(name: Literal[str], arrays = (), dtype = None, ilp64: bool = False):
    """Return the LAPACK routine `name` ('gesv', 'dpotrf', ...)."""
    nm: Literal[str] = name[1:] if (len(name) > 1 and name[0] in ("s", "d", "c", "z")) and name[1:] in ("gesv", "posv", "getrf", "getrs", "potrf", "potrs", "geqrf", "syev", "gesdd") else name
    if nm == "gesv":
        return _lp_gesv
    elif nm == "posv":
        return _lp_posv
    elif nm == "getrf":
        return _lp_getrf
    elif nm == "getrs":
        return _lp_getrs
    elif nm == "potrf":
        return _lp_potrf
    elif nm == "potrs":
        return _lp_potrs
    elif nm == "geqrf":
        return _lp_geqrf
    elif nm == "syev":
        return _lp_syev
    elif nm == "gesdd":
        return _lp_gesdd
    else:
        compile_error("get_lapack_funcs: unsupported routine")
