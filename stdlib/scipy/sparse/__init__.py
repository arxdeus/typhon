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
