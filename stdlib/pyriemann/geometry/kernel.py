# Copyright (C) 2026 the Typhon authors
# Ported from pyriemann 0.12 (BSD-3-Clause)

import numpy as np
from .base import logm
from .mean import mean_riemann, mean_euclid
from .base import invsqrtm

def _apply_matrix_kernel(Xt, Y, equal: bool, reg: float):
    nx = Xt.shape[0]
    ny = Y.shape[0]
    K = np.empty((nx, ny))
    for i in range(nx):
        for j in range(ny):
            K[i, j] = np.sum(Xt[i] * np.transpose(Y[j]))
    if equal:
        for i in range(nx):
            K[i, i] += reg
    return K

def _check_cref(X, Cref):
    if Cref.shape[0] != X.shape[1] or Cref.shape[1] != X.shape[2]:
        raise ValueError("Dimension of Cref must match dimension of matrices in X.")

def kernel_euclid(X, Y = None, Cref = None, reg: float = 1e-10):
    X = np.asarray(X, dtype=float)
    equal = Y is None
    Y_ = X if Y is None else np.asarray(Y, dtype=float)
    if Cref is None:
        Xt = np.swapaxes(X, -1, -2)
        return _apply_matrix_kernel(Xt, Y_, equal, reg)
    C = np.asarray(Cref, dtype=float)
    _check_cref(X, C)
    return _apply_matrix_kernel(np.swapaxes(X - C, -1, -2), Y_ - C, equal, reg)

def kernel_logeuclid(X, Y = None, Cref = None, reg: float = 1e-10):
    X = np.asarray(X, dtype=float)
    equal = Y is None
    Y_ = X if Y is None else np.asarray(Y, dtype=float)
    if Cref is None:
        return _apply_matrix_kernel(logm(X), logm(Y_), equal, reg)
    C = np.asarray(Cref, dtype=float)
    _check_cref(X, C)
    lc = logm(C)
    return _apply_matrix_kernel(logm(X) - lc, logm(Y_) - lc, equal, reg)

def kernel_riemann(X, Y = None, Cref = None, reg: float = 1e-10):
    X = np.asarray(X, dtype=float)
    equal = Y is None
    Y_ = X if Y is None else np.asarray(Y, dtype=float)
    if Cref is None:
        C = mean_riemann(X)
    else:
        C = np.asarray(Cref, dtype=float)
        _check_cref(X, C)
    isq = invsqrtm(C)
    return _apply_matrix_kernel(logm(isq @ X @ isq), logm(isq @ Y_ @ isq), equal, reg)

def kernel(X, Y = None, Cref = None, metric: str = "riemann", reg: float = 1e-10):
    if metric == "euclid":
        return kernel_euclid(X, Y, Cref=Cref, reg=reg)
    elif metric == "logeuclid":
        return kernel_logeuclid(X, Y, Cref=Cref, reg=reg)
    elif metric == "riemann":
        return kernel_riemann(X, Y, Cref=Cref, reg=reg)
    raise ValueError(f"Unknown function name '{metric}'. Must be one of euclid logeuclid riemann")
