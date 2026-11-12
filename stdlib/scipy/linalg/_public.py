# Ported from scipy 1.18.1 (BSD-3-Clause)

# Public scipy.linalg entry points: 2-D inputs go straight to the kernels,
# inputs with ndim > 2 are processed as batches of matrices.

import numpy as np
from ._batch import batch1, batch2
from ._decomp import _eigh_2d
from ._basic import _solve_2d, _inv_2d, _det_2d, _cholesky_2d, _lu_2d, \
                    _svd_2d, _svdvals_2d, _pinv_2d, _qr_2d, _eig_2d, _eigvals_2d
from ._matfuncs import _expm_2d, _sqrtm_2d, _logm_2d, _fractional_matrix_power_2d

import internal.static as static

def _nd(a) -> Literal[int]:
    # compile-time number of dimensions so that only one branch is realized
    if isinstance(a, np.ndarray):
        return static.len(a.shape)
    else:
        return static.len(np.asarray(a).shape)

def eigh(a, b = None, *, lower: bool = True, eigvals_only: Literal[bool] = False,
         overwrite_a: bool = False, overwrite_b: bool = False, type: int = 1,
         check_finite: bool = True, subset_by_index = None, subset_by_value = None,
         driver: Optional[str] = None):
    """Hermitian/symmetric (generalized) eigenproblem; ``eigvals_only`` must
    be a compile-time constant. Batched input supported."""
    itype = type
    if b is None:
        if _nd(a) > 2:
            return batch1(lambda x: _eigh_2d(x, None, lower=lower, eigvals_only=eigvals_only,
                                             type=itype, check_finite=check_finite,
                                             subset_by_index=subset_by_index,
                                             subset_by_value=subset_by_value, driver=driver), a)
        else:
            return _eigh_2d(a, None, lower=lower, eigvals_only=eigvals_only, type=itype,
                            check_finite=check_finite, subset_by_index=subset_by_index,
                            subset_by_value=subset_by_value, driver=driver)
    else:
        if _nd(a) > 2 or _nd(b) > 2:
            return batch2(lambda x, y: _eigh_2d(x, y, lower=lower, eigvals_only=eigvals_only,
                                                type=itype, check_finite=check_finite,
                                                subset_by_index=subset_by_index,
                                                subset_by_value=subset_by_value, driver=driver), a, b)
        else:
            return _eigh_2d(a, b, lower=lower, eigvals_only=eigvals_only, type=itype,
                            check_finite=check_finite, subset_by_index=subset_by_index,
                            subset_by_value=subset_by_value, driver=driver)

def eigvalsh(a, b = None, *, lower: bool = True, overwrite_a: bool = False,
             overwrite_b: bool = False, type: int = 1, check_finite: bool = True,
             subset_by_index = None, subset_by_value = None,
             driver: Optional[str] = None):
    return eigh(a, b, lower=lower, eigvals_only=True, type=type,
                check_finite=check_finite, subset_by_index=subset_by_index,
                subset_by_value=subset_by_value, driver=driver)

def solve(a, b, lower: bool = False, overwrite_a: bool = False,
          overwrite_b: bool = False, check_finite: bool = True,
          assume_a: Optional[str] = None, transposed: bool = False):
    if _nd(a) > 2 or _nd(b) > 2:
        f = lambda x, y: _solve_2d(x, y, lower=lower, check_finite=check_finite,
                                   assume_a=assume_a, transposed=transposed)
        if _nd(b) == 1:
            return batch2(f, a, b, bcore2=False)
        else:
            return batch2(f, a, b)
    else:
        return _solve_2d(a, b, lower=lower, check_finite=check_finite,
                         assume_a=assume_a, transposed=transposed)

def inv(a, overwrite_a: bool = False, check_finite: bool = True, *,
        assume_a: Optional[str] = None, lower: bool = False):
    if _nd(a) > 2:
        return batch1(lambda x: _inv_2d(x, check_finite=check_finite, assume_a=assume_a, lower=lower), a)
    else:
        return _inv_2d(a, check_finite=check_finite, assume_a=assume_a, lower=lower)

def det(a, overwrite_a: bool = False, check_finite: bool = True):
    if _nd(a) > 2:
        return batch1(lambda x: _det_2d(x, check_finite=check_finite), a)
    else:
        return _det_2d(a, check_finite=check_finite)

def cholesky(a, lower: bool = False, overwrite_a: bool = False, check_finite: bool = True):
    if _nd(a) > 2:
        return batch1(lambda x: _cholesky_2d(x, lower=lower, check_finite=check_finite), a)
    else:
        return _cholesky_2d(a, lower=lower, check_finite=check_finite)

def lu(a, permute_l: Literal[bool] = False, overwrite_a: bool = False,
       check_finite: bool = True, p_indices: Literal[bool] = False):
    if _nd(a) > 2:
        return batch1(lambda x: _lu_2d(x, permute_l=permute_l, check_finite=check_finite,
                                       p_indices=p_indices), a)
    else:
        return _lu_2d(a, permute_l=permute_l, check_finite=check_finite, p_indices=p_indices)

def svd(a, full_matrices: bool = True, compute_uv: Literal[bool] = True,
        overwrite_a: bool = False, check_finite: bool = True,
        lapack_driver: str = 'gesdd'):
    if _nd(a) > 2:
        return batch1(lambda x: _svd_2d(x, full_matrices=full_matrices, compute_uv=compute_uv,
                                        check_finite=check_finite, lapack_driver=lapack_driver), a)
    else:
        return _svd_2d(a, full_matrices=full_matrices, compute_uv=compute_uv,
                       check_finite=check_finite, lapack_driver=lapack_driver)

def svdvals(a, overwrite_a: bool = False, check_finite: bool = True):
    return svd(a, compute_uv=False, check_finite=check_finite)

def pinv(a, *, atol = None, rtol = None, return_rank: Literal[bool] = False,
         check_finite: bool = True):
    if _nd(a) > 2:
        return batch1(lambda x: _pinv_2d(x, atol=atol, rtol=rtol, return_rank=return_rank,
                                         check_finite=check_finite), a)
    else:
        return _pinv_2d(a, atol=atol, rtol=rtol, return_rank=return_rank, check_finite=check_finite)

def qr(a, overwrite_a: bool = False, lwork = None, mode: Literal[str] = 'full',
       pivoting: Literal[bool] = False, check_finite: bool = True):
    if _nd(a) > 2:
        return batch1(lambda x: _qr_2d(x, mode=mode, pivoting=pivoting, check_finite=check_finite), a)
    else:
        return _qr_2d(a, mode=mode, pivoting=pivoting, check_finite=check_finite)

def eig(a, b = None, left: Literal[bool] = False, right: Literal[bool] = True,
        overwrite_a: bool = False, overwrite_b: bool = False,
        check_finite: bool = True, homogeneous_eigvals: Literal[bool] = False):
    if b is None:
        if _nd(a) > 2:
            return batch1(lambda x: _eig_2d(x, None, left=left, right=right, check_finite=check_finite,
                                            homogeneous_eigvals=homogeneous_eigvals), a)
        else:
            return _eig_2d(a, None, left=left, right=right, check_finite=check_finite,
                           homogeneous_eigvals=homogeneous_eigvals)
    else:
        if _nd(a) > 2 or _nd(b) > 2:
            return batch2(lambda x, y: _eig_2d(x, y, left=left, right=right, check_finite=check_finite,
                                               homogeneous_eigvals=homogeneous_eigvals), a, b)
        else:
            return _eig_2d(a, b, left=left, right=right, check_finite=check_finite,
                           homogeneous_eigvals=homogeneous_eigvals)

def eigvals(a, b = None, overwrite_a: bool = False, check_finite: bool = True,
            homogeneous_eigvals: Literal[bool] = False):
    return eig(a, b, right=False, check_finite=check_finite,
               homogeneous_eigvals=homogeneous_eigvals)

def expm(A):
    if _nd(A) > 2:
        return batch1(_expm_2d, A)
    else:
        return _expm_2d(A)

def sqrtm(A):
    if _nd(A) > 2:
        return batch1(_sqrtm_2d, A)
    else:
        return _sqrtm_2d(A)

def logm(A, disp: bool = True):
    if _nd(A) > 2:
        return batch1(lambda x: _logm_2d(x), A)
    else:
        return _logm_2d(A)

def fractional_matrix_power(A, t):
    if _nd(A) > 2:
        return batch1(lambda x: _fractional_matrix_power_2d(x, t), A)
    else:
        return _fractional_matrix_power_2d(A, t)
