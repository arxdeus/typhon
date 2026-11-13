# Copyright (C) 2026 the Typhon authors
# Ported from pyriemann 0.12 (BSD-3-Clause)

import numpy as np
from scipy.linalg import eigh
from .geometry.base import sqrtm, invsqrtm
from .geometry.mean import gmean

class Whitening:
    # dim_red: None, or one of ("n_components", v) / ("expl_var", v) / ("max_cond", v)
    metric: str
    dim_red_key: str
    dim_red_val: float
    n_components_: int
    filters_: np.ndarray[float, 2]
    inv_filters_: np.ndarray[float, 2]
    _mean: np.ndarray[float, 2]

    def __init__(self, metric: str = "euclid", dim_red_key: str = "", dim_red_val: float = 0.0):
        self.metric = metric
        self.dim_red_key = dim_red_key
        self.dim_red_val = dim_red_val
        self.n_components_ = 0
        self.filters_ = np.empty((0, 0))
        self.inv_filters_ = np.empty((0, 0))
        self._mean = np.empty((0, 0))

    def fit(self, X, y = None, sample_weight = None):
        X = np.asarray(X, dtype=float)
        self._mean = gmean(X, metric=self.metric, sample_weight=sample_weight)
        n = X.shape[2]
        if self.dim_red_key == "":
            self.n_components_ = n
            self.filters_ = invsqrtm(self._mean)
            self.inv_filters_ = sqrtm(self._mean)
            return self
        eigvals, eigvecs = eigh(self._mean)
        ev = eigvals[::-1].copy()
        evec = eigvecs[:, ::-1].copy()
        key = self.dim_red_key
        val = self.dim_red_val
        if key == "n_components":
            if val < 1:
                raise ValueError("Value n_components must be superior to 1")
            self.n_components_ = min(int(val), n)
        elif key == "expl_var":
            if not (0 < val and val <= 1):
                raise ValueError("Value expl_var must be included in (0, 1]")
            cum = np.cumsum(ev / np.sum(ev))
            self.n_components_ = int(np.searchsorted(cum, val, side="right")) + 1
        elif key == "max_cond":
            if val <= 1:
                raise ValueError("Value max_cond must be strictly superior to 1")
            conds = ev[0] / ev
            self.n_components_ = int(np.searchsorted(conds, val, side="left"))
        else:
            raise ValueError("Unknown key in parameter dim_red")
        k = self.n_components_
        pf = evec[:, :k]
        ps = np.sqrt(ev[:k])
        self.filters_ = pf * (1.0 / ps)[None, :]
        self.inv_filters_ = ps[:, None] * pf.T
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return self.filters_.T @ X @ self.filters_

    def fit_transform(self, X, y = None, sample_weight = None):
        return self.fit(X, y, sample_weight=sample_weight).transform(X)

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=float)
        return self.inv_filters_.T @ X @ self.inv_filters_
