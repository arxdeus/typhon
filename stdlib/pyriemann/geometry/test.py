# Copyright (C) 2026 the Typhon authors
# Ported from pyriemann 0.12 (BSD-3-Clause)

# Matrix property tests (real-valued matrices).

import numpy as np

def _allclose(A, B) -> bool:
    return bool(np.all(np.abs(A - B) <= 1e-8 + 1e-5 * np.abs(B)))

def is_square(X) -> bool:
    X = np.asarray(X)
    return X.ndim >= 2 and X.shape[X.ndim - 2] == X.shape[X.ndim - 1]

def is_sym(X) -> bool:
    X = np.asarray(X)
    return is_square(X) and _allclose(X, np.swapaxes(X, -1, -2))

def is_skew_sym(X) -> bool:
    X = np.asarray(X)
    return is_square(X) and _allclose(X, -np.swapaxes(X, -1, -2))

def is_hermitian(X) -> bool:
    return is_sym(X)

def is_real(X) -> bool:
    return True

def is_real_type(X) -> bool:
    return True

def is_hankel(X) -> bool:
    X = np.asarray(X)
    if not is_square(X) or X.ndim != 2:
        return False
    n = X.shape[0]
    for i in range(n):
        for j in range(n):
            if i + j < n:
                if X[i, j] != X[i + j, 0]:
                    return False
            else:
                if X[i, j] != X[i + j - n + 1, n - 1]:
                    return False
    return True

def _eigs(X):
    X = np.asarray(X)
    n = X.shape[X.ndim - 1]
    X3 = X.reshape(-1, n, n)
    out = np.empty((X3.shape[0], n))
    for b in range(X3.shape[0]):
        out[b] = np.real(np.linalg.eigvals(X3[b]))
    return out

def is_pos_def(X, tol: float = 0.0, fast_mode: bool = False) -> bool:
    X = np.asarray(X)
    if not is_square(X):
        return False
    if fast_mode:
        try:
            np.linalg.cholesky(X)
            return True
        except:
            return False
    return bool(np.all(_eigs(X) > tol))

def is_pos_semi_def(X) -> bool:
    X = np.asarray(X)
    if not is_square(X):
        return False
    return bool(np.all(_eigs(X) >= 0.0))

def is_sym_pos_def(X, tol: float = 0.0) -> bool:
    return is_sym(X) and is_pos_def(X, tol)

def is_sym_pos_semi_def(X) -> bool:
    return is_sym(X) and is_pos_semi_def(X)

def is_herm_pos_def(X, tol: float = 0.0) -> bool:
    return is_sym_pos_def(X, tol)

def is_herm_pos_semi_def(X) -> bool:
    return is_sym_pos_semi_def(X)
