# Copyright (C) 2026 the Typhon authors
# Ported from pyriemann 0.12 (BSD-3-Clause)
# Real-valued kinds only: real, inv, orth, sym, spd, spsd.

import numpy as np
from numpy.random.legacy import RandomState, check_random_state
from ..geometry.base import sqrtm, powm
from ..geometry.distance import distance_riemann

def _make_eyes(n_matrices: int, n_dim: int):
    out = np.zeros((n_matrices, n_dim, n_dim))
    for i in range(n_matrices):
        out[i] = np.eye(n_dim)
    return out

def _check_kind(kind: str):
    if not (kind == "real" or kind == "inv" or kind == "orth" or kind == "sym"
            or kind == "spd" or kind == "spsd"):
        raise ValueError(f"Unsupported matrix kind: {kind}")

def make_matrices_rect(n_matrices: int, n_dim: List[int], rs = None):
    # kind="real" with explicit (possibly non-square) dimensions.
    r = check_random_state(rs)
    X = r.randn(n_matrices, n_dim[0], n_dim[1])
    return X

def _qr_batch(X):
    out = np.empty(X.shape)
    for i in range(X.shape[0]):
        out[i] = np.linalg.qr(X[i])[0]
    return out

def _spd_like(n_matrices: int, n_dim: int, kind: str, r: RandomState,
              evals_low: float, evals_high: float, eigvecs_same: bool,
              eigvecs_mean: float, eigvecs_std: float):
    X = eigvecs_std * r.randn(n_matrices, n_dim, n_dim) + eigvecs_mean
    if evals_low <= 0.0:
        raise ValueError(f"Lowest value must be strictly positive (Got {evals_low}).")
    if evals_high <= evals_low:
        raise ValueError(f"Highest value must be superior to lowest value (Got {evals_high} and {evals_low}).")
    evals = r.uniform(evals_low, evals_high, size=(n_matrices, n_dim))
    if kind == "spsd":
        evals[:, n_dim - 1] = 1e-10
    mats = np.empty((n_matrices, n_dim, n_dim))
    if eigvecs_same:
        evecs0 = np.linalg.qr(X[0])[0]
        for i in range(n_matrices):
            mats[i] = (evecs0 * evals[i]) @ evecs0.T
    else:
        for i in range(n_matrices):
            evecs = np.linalg.qr(X[i])[0]
            mats[i] = (evecs * evals[i]) @ evecs.T
    return mats, evals

def make_matrices_params(n_matrices: int, n_dim: int, kind: str, rs = None,
                         evals_low: float = 0.5, evals_high: float = 2.0,
                         eigvecs_same: bool = False, eigvecs_mean: float = 0.0,
                         eigvecs_std: float = 1.0):
    # make_matrices(..., return_params=True) for kinds spd and spsd: (mats, evals).
    if kind != "spd" and kind != "spsd":
        raise ValueError("Parameters are only returned for spd and spsd")
    r = check_random_state(rs)
    return _spd_like(n_matrices, n_dim, kind, r, evals_low, evals_high,
                     eigvecs_same, eigvecs_mean, eigvecs_std)

def make_matrices(n_matrices: int, n_dim: int, kind: str, rs = None,
                  evals_low: float = 0.5, evals_high: float = 2.0,
                  eigvecs_same: bool = False, eigvecs_mean: float = 0.0,
                  eigvecs_std: float = 1.0):
    _check_kind(kind)
    r = check_random_state(rs)
    if kind == "spd" or kind == "spsd":
        return _spd_like(n_matrices, n_dim, kind, r, evals_low, evals_high,
                         eigvecs_same, eigvecs_mean, eigvecs_std)[0]
    if kind == "inv":
        while True:
            X = r.randn(n_matrices, n_dim, n_dim)
            ok = True
            for i in range(n_matrices):
                if np.linalg.det(X[i]) == 0.0:
                    ok = False
            if ok:
                return X
    X = eigvecs_std * r.randn(n_matrices, n_dim, n_dim) + eigvecs_mean
    if kind == "orth":
        return _qr_batch(X)
    if kind == "sym":
        return X + X.transpose(0, 2, 1)
    return X

def make_masks(n_masks: int, n_dim0: int, n_dim1_min: int, rs = None):
    r = check_random_state(rs)
    masks = List[np.ndarray[float, 2]]()
    for _ in range(n_masks):
        n_dim1 = int(r.randint(n_dim1_min, n_dim0, size=1)[0])
        q = np.linalg.qr(r.randn(n_dim0, n_dim1))[0]
        masks.append(q)
    return masks

def make_outliers(n_matrices: int, mean, sigma: float, outlier_coeff: float = 10.0,
                  random_state = None):
    mean = np.asarray(mean, dtype=float)
    n_dim = mean.shape[1]
    mean_sqrt = sqrtm(mean)
    outliers = np.zeros((n_matrices, n_dim, n_dim))
    for i in range(n_matrices):
        Oi = make_matrices(1, n_dim, "spd", random_state)[0]
        epsilon_num = outlier_coeff * sigma * n_dim
        epsilon_den = distance_riemann(Oi, np.eye(n_dim), squared=True)
        epsilon = np.sqrt(epsilon_num / epsilon_den)
        outliers[i] = mean_sqrt @ powm(Oi, float(epsilon)) @ mean_sqrt
    return outliers
