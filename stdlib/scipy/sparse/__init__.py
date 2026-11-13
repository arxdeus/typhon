# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# scipy.sparse: CSR / CSC / COO / DIA / BSR / LIL / DOK containers (sparse
# arrays and matrices) and constructors. Submodules linalg and csgraph are
# imported on demand.

from ._base import _compressed, _coo
from ._dia import _dia
from ._bsr import _bsr
from ._lil import _lil
from ._dok import _dok
from ._construct import csr_array, csr_matrix, csc_array, csc_matrix, \
    coo_array, coo_matrix, dia_array, dia_matrix, bsr_array, bsr_matrix, \
    lil_array, lil_matrix, dok_array, dok_matrix, issparse, isspmatrix, eye, \
    eye_array, identity, diags, diags_array, spdiags, kron, kronsum, hstack, \
    vstack, bmat, block_array, block_diag, find, tril, triu, random, \
    random_array, rand

def _isfmt(x, fmt: Literal[str]) -> bool:
    if hasattr(x, "_spfmt"):
        return x._spfmt() == fmt and x._is_matrix
    return False

def isspmatrix_csr(x) -> bool:
    return _isfmt(x, "csr")

def isspmatrix_csc(x) -> bool:
    return _isfmt(x, "csc")

def isspmatrix_coo(x) -> bool:
    return _isfmt(x, "coo")

def isspmatrix_dia(x) -> bool:
    return _isfmt(x, "dia")

def isspmatrix_bsr(x) -> bool:
    return _isfmt(x, "bsr")

def isspmatrix_lil(x) -> bool:
    return _isfmt(x, "lil")

def isspmatrix_dok(x) -> bool:
    return _isfmt(x, "dok")

class SparseWarning(Warning):
    pass

class SparseEfficiencyWarning(SparseWarning):
    pass

def matrix_transpose(x):
    return x.transpose()

def permute_dims(a, axes = (1, 0), copy: bool = True):
    # Only the transposing permutation (1, 0) / (-1, -2) is supported: the
    # identity permutation would return a different static type.
    a0 = int(axes[0]) % 2
    a1 = int(axes[1]) % 2
    if len(axes) != 2 or a0 == a1:
        raise ValueError("axes don't match array")
    if a0 != 1:
        raise ValueError("permute_dims: only axes=(1, 0) is supported")
    return a.transpose()

def swapaxes(a, axis1: int, axis2: int):
    if axis1 < -2 or axis1 > 1 or axis2 < -2 or axis2 > 1:
        raise ValueError("axis out of bounds for 2-D sparse array")
    if axis1 % 2 == axis2 % 2:
        raise ValueError("swapaxes: axis1 == axis2 is not supported")
    return a.transpose()

def get_index_dtype(arrays = (), maxval = None, check_contents: bool = False):
    import numpy as np
    # Indices are stored as 64-bit ints in this port, but scipy reports int32
    # whenever the values fit; maxval beyond int32 is rejected (type would differ).
    if maxval is not None and int(maxval) > 2147483647:
        raise ValueError("get_index_dtype: maxval exceeds int32 range (int64 index dtype is not selectable)")
    return np.int32

from ._matrix_io import save_npz, load_npz

class sparray:
    """Marker for sparse arrays (csr_array, ...). Typhon sparse containers
    are generic classes, so use issparse()/the `_is_matrix` flag instead of
    isinstance checks."""
    pass

class spmatrix:
    """Marker for sparse matrices (csr_matrix, ...); see sparray."""
    pass

def expand_dims(A, axis: int = 0):
    """Insert a unit axis into a 2-D sparse matrix/array. Sparse containers
    here are 2-D only, so the result is the dense equivalent with 3 dims."""
    import numpy as np
    return np.expand_dims(A.toarray(), axis)
