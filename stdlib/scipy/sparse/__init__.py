# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# scipy.sparse: CSR / CSC / COO containers (sparse arrays and matrices) and
# constructors. Submodules linalg and csgraph are imported on demand.

from ._base import _compressed, _coo
from ._construct import csr_array, csr_matrix, csc_array, csc_matrix, \
    coo_array, coo_matrix, issparse, isspmatrix, eye, eye_array, identity, \
    diags, diags_array, spdiags, kron, kronsum, hstack, vstack, bmat, \
    block_diag, find, tril, triu, random, random_array, rand

def isspmatrix_csr(x) -> bool:
    if isinstance(x, _compressed):
        return x.format == "csr" and x._is_matrix
    return False

def isspmatrix_csc(x) -> bool:
    if isinstance(x, _compressed):
        return x.format == "csc" and x._is_matrix
    return False

def isspmatrix_coo(x) -> bool:
    if isinstance(x, _coo):
        return x._is_matrix
    return False
