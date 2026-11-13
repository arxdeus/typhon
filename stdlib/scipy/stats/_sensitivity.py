# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# sobol_indices for a scalar-output function ``func(x)`` with ``x`` of shape
# (d, n) returning shape (n,) (or (1, n)), and ``method='saltelli_2010'``.
# ``dists`` is a sequence of objects with a ``ppf`` method. The bootstrap
# confidence intervals of the multi-output result are not ported.

import numpy as np
from .qmc import Sobol

class SobolResult:
    first_order: np.ndarray[float, 1]
    total_order: np.ndarray[float, 1]
    def __init__(self, first_order: np.ndarray[float, 1], total_order: np.ndarray[float, 1]):
        self.first_order = first_order
        self.total_order = total_order

def f_ishigami(x):
    xa = np.atleast_2d(np.asarray(x).astype(float))
    return np.sin(xa[0]) + 7 * np.sin(xa[1]) ** 2 + 0.1 * (xa[2] ** 4) * np.sin(xa[0])

def _flat(v, n: int) -> np.ndarray[float, 1]:
    a = np.asarray(v).astype(float).ravel()
    if a.shape[0] != n:
        raise ValueError("'func' output should have a shape ``(s, -1)`` with ``s`` the number of output.")
    return a

def sobol_indices(*, func, n, dists = None, method: str = 'saltelli_2010', rng = None):
    n_ = int(n)
    if not (n_ & (n_ - 1) == 0) or n != n_:
        raise ValueError("The balance properties of Sobol' points require 'n' to be a power of 2.")
    if method.lower() != 'saltelli_2010':
        raise ValueError(f"'{method}' is not a valid 'method'. It must be one of {{'saltelli_2010'}} or a callable.")
    if dists is None:
        raise ValueError("'dists' must be defined when 'func' is a callable.")
    dl = [d for d in dists]
    d = len(dl)
    # A_B = Sobol(d=2d, bits=64).random(n).T reshaped (2, d, n)
    raw = Sobol(2 * d, bits=64, rng=rng).random(n_)
    A = np.empty((d, n_), dtype=float)
    B = np.empty((d, n_), dtype=float)
    for j in range(d):
        colA = np.ascontiguousarray(raw[:, j])
        colB = np.ascontiguousarray(raw[:, d + j])
        A[j] = np.asarray(dl[j].ppf(colA)).astype(float)
        B[j] = np.asarray(dl[j].ppf(colB)).astype(float)
    f_A = _flat(func(A), n_)
    f_B = _flat(func(B), n_)
    f_AB = np.empty((d, n_), dtype=float)
    for i in range(d):
        ABi = A.copy()
        ABi[i] = B[i]
        f_AB[i] = _flat(func(ABi), n_)
    mean = (float(np.sum(f_A)) + float(np.sum(f_B))) / (2 * n_)
    f_A = f_A - mean
    f_B = f_B - mean
    f_AB = f_AB - mean
    allv = np.concatenate((f_A, f_B))
    mu = float(np.mean(allv))
    var = float(np.sum((allv - mu) ** 2)) / allv.shape[0]
    first = np.empty(d, dtype=float)
    total = np.empty(d, dtype=float)
    for i in range(d):
        s = float(np.mean(f_B * (f_AB[i] - f_A))) / var
        t = 0.5 * float(np.mean((f_A - f_AB[i]) ** 2)) / var
        first[i] = s if np.isfinite(s) else 0.0
        total[i] = t if np.isfinite(t) else 0.0
    return SobolResult(first, total)
