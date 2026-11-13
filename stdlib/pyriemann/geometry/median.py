# Copyright (C) 2026 the Typhon authors
# Ported from pyriemann 0.12 (BSD-3-Clause)

import numpy as np
from .base import sqrtm, invsqrtm, logm, expm
from .mean import mean_euclid, check_weights, _warn
from .distance import distance

def median_euclid(X, tol: float = 10e-6, maxiter: int = 50, init = None, weights = None):
    X = np.asarray(X, dtype=float)
    n_m = X.shape[0]
    w0 = check_weights(weights, n_m)
    M = mean_euclid(X, sample_weight=w0) if init is None else np.array(np.asarray(init, dtype=float))
    converged = False
    for _ in range(maxiter):
        dists = distance(X, M, metric="euclid")[:, 0]
        nz = dists != 0
        w = w0[nz] / dists[nz]
        Mnew = mean_euclid(X[nz], sample_weight=w)
        n_zeros = n_m - int(np.sum(np.asarray(nz, dtype=int)))
        if n_zeros > 0:
            R = np.tensordot(w, X[nz] - M, axes=([0], [0]))
            r = np.linalg.norm(R)
            rinv = 0.0 if r == 0 else float(np.mean(w0[~nz])) / r
            Mnew = max(0.0, 1.0 - rinv) * Mnew + min(1.0, rinv) * M
        crit = np.linalg.norm(Mnew - M)
        M = Mnew
        if crit <= tol:
            converged = True
            break
    if not converged:
        _warn()
    return M

def median_riemann(X, tol: float = 10e-6, maxiter: int = 50, init = None, weights = None,
                   step_size: float = 1.0):
    if not (0 < step_size and step_size <= 2):
        raise ValueError(f"Value step_size must be included in (0, 2] (Got {step_size})")
    X = np.asarray(X, dtype=float)
    n_m = X.shape[0]
    w0 = check_weights(weights, n_m)
    M = mean_euclid(X, sample_weight=w0) if init is None else np.array(np.asarray(init, dtype=float))
    converged = False
    for _ in range(maxiter):
        dists = distance(X, M, metric="riemann")[:, 0]
        nz = dists != 0
        w = w0[nz] / dists[nz]
        M12 = sqrtm(M)
        Mm12 = invsqrtm(M)
        tv = logm(Mm12 @ X[nz] @ Mm12)
        wn = w / np.sum(w)
        J = np.tensordot(wn, tv, axes=([0], [0]))
        M = M12 @ expm(step_size * J) @ M12
        crit = np.linalg.norm(J)
        if crit <= tol:
            converged = True
            break
    if not converged:
        _warn()
    return M
