
from .linalg import cholesky, eig, eigh, eigvals, \
                    eigvalsh, inv, lstsq, qr, solve, \
                    svd, svdvals, inner, outer, \
                    tensordot, matmul, multi_dot, \
                    trace, tensorsolve, tensorinv, \
                    norm, pinv, matrix_rank, cond, \
                    matrix_power, det, slogdet, \
                    _matmul_ufunc, _trace_ufunc, \
                    matrix_transpose, LinAlgError, \
                    dot as _linalg_dot, \
                    vdot as _linalg_vdot, \
                    tensordot as _linalg_tensordot, \
                    inner as _linalg_inner, \
                    outer as _linalg_outer, \
                    kron as _linalg_kron

# numpy 2.x array-API linalg functions
from ..routines import diagonal as _diagonal_impl, asarray as _asarray

def vecdot(x1, x2, axis: int = -1):
    from ..extras import vecdot as _vecdot_impl
    return _vecdot_impl(x1, x2, axis=axis)

def cross(x1, x2, axis: int = -1):
    from ..ndmath import cross as _cross_impl
    a = _asarray(x1)
    b = _asarray(x2)
    if a.shape[axis] != 3 or b.shape[axis] != 3:
        raise ValueError("Both input arrays must be (arrays of) 3-dimensional vectors, but they are " + str(a.shape[axis]) + " and " + str(b.shape[axis]) + " dimensional instead.")
    return _cross_impl(a, b, axis=axis)

def diagonal(x, offset: int = 0):
    return _diagonal_impl(x, offset=offset, axis1=-2, axis2=-1)

def vector_norm(x, axis = None, keepdims: Literal[bool] = False, ord = 2):
    a = _asarray(x)
    if axis is None:
        r = norm(a.ravel(), ord=ord)
        if keepdims:
            return _asarray(r).reshape((1,) * a.ndim)
        return r
    return norm(a, ord=ord, axis=axis, keepdims=keepdims)

def matrix_norm(x, keepdims: Literal[bool] = False, ord = 'fro'):
    return norm(x, ord=ord, axis=(-2, -1), keepdims=keepdims)
