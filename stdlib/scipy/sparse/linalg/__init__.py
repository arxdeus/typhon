# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# scipy.sparse.linalg: LinearOperator, Krylov solvers, sparse LU, eigen /
# singular value solvers, norms and matrix functions.

from ._interface import LinearOperator, aslinearoperator, IdentityOperator
from ._isolve import cg, cgs, bicg, bicgstab, gmres, minres, lsqr, lsmr
from ._dsolve import splu, spilu, spsolve, spsolve_triangular, factorized, \
    inv, SuperLU, MatrixRankWarning, is_sptriangular, spbandwidth
from ._eigen import eigsh, eigs, svds, ArpackNoConvergence, ArpackError
from ._lobpcg import lobpcg
from ._matfuncs import norm, onenormest, expm, expm_multiply, matrix_power
