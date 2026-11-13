# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# SplineTransformer (dense output; sample_weight only for knots='uniform'
# zero-weight masking and unweighted quantiles).

from ..base import estimator_repr as _estimator_repr
import numpy as np
from scipy.interpolate import BSpline

def _spl_eval(t, k: int, nb: int, x, nu: int, extrapolate: bool):
    coef = np.eye(nb)
    spl = BSpline(t, coef, k, extrapolate, 0, _fast=True)
    return spl(np.asarray(x, dtype=float), nu)

class SplineTransformer:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    n_knots: int
    degree: int
    knots: str
    _knots_array: np.ndarray[float, 2]
    extrapolation: str
    include_bias: bool
    order: str
    handle_missing: str
    sparse_output: bool
    _t: List[np.ndarray[float, 1]]
    _n_splines: int
    n_features_in_: int
    n_features_out_: int

    def __init__(self, n_knots: int = 5, degree: int = 3, *, knots = "uniform",
                 extrapolation: str = "constant", include_bias: bool = True, order: str = "C",
                 handle_missing: str = "error", sparse_output: bool = False):
        self.n_knots = n_knots
        self.degree = degree
        if isinstance(knots, str):
            self.knots = knots
            self._knots_array = np.empty((0, 0))
        else:
            self.knots = "array"
            self._knots_array = np.asarray(knots, dtype=float)
        self.extrapolation = extrapolation
        self.include_bias = include_bias
        self.order = order
        self.handle_missing = handle_missing
        self.sparse_output = sparse_output
        self._t = []
        self._n_splines = 0
        self.n_features_in_ = 0
        self.n_features_out_ = 0

    def fit(self, X, y = None, sample_weight = None):
        Xa = np.asarray(X, dtype=float)
        n, p = Xa.shape
        if self.handle_missing != "zeros" and np.any(np.isnan(Xa)):
            raise ValueError("Input X contains NaN values and `SplineTransformer` is configured to error in this case (handle_missing='error'). To avoid this error, set handle_missing='zeros' to encode missing values as splines with value 0 or ensure no missing values in X.")
        base = np.empty((0, 0))
        if self.knots == "quantile":
            ranks = 100.0 * np.linspace(0.0, 1.0, self.n_knots)
            base = np.nanpercentile(Xa, ranks, axis=0)
        elif self.knots == "uniform":
            xmin = np.zeros(p); xmax = np.zeros(p)
            for j in range(p):
                vals = List[float]()
                for i in range(n):
                    ok = sample_weight is None or float(np.asarray(sample_weight)[i]) > 0.0
                    if ok and Xa[i, j] == Xa[i, j]:
                        vals.append(Xa[i, j])
                if len(vals) > 0:
                    xmin[j] = min(vals); xmax[j] = max(vals)
            base = np.empty((self.n_knots, p))
            for j in range(p):
                base[:, j] = np.linspace(xmin[j], xmax[j], self.n_knots)
        else:
            base = self._knots_array.copy()
            if base.shape[0] < 2:
                raise ValueError("Number of knots, knots.shape[0], must be >= 2.")
            elif base.shape[1] != p:
                raise ValueError("knots.shape[1] == n_features is violated.")
            elif not np.all(np.diff(base, axis=0) > 0):
                raise ValueError("knots must be sorted without duplicates.")
        nk = base.shape[0]
        deg = self.degree
        if self.extrapolation == "periodic" and nk <= deg:
            raise ValueError(f"Periodic splines require degree < n_knots. Got n_knots={nk} and degree={deg}.")
        ns = nk + deg - 1 if self.extrapolation != "periodic" else nk - 1
        self._t = []
        for j in range(p):
            b = np.ascontiguousarray(base[:, j])
            t = np.empty(nk + 2 * deg)
            if self.extrapolation == "periodic":
                per = b[nk - 1] - b[0]
                for q in range(deg):
                    t[q] = b[nk - 1 - deg + q] - per
                    t[nk + deg + q] = b[1 + q] + per
            else:
                dmin = b[1] - b[0]
                dmax = b[nk - 1] - b[nk - 2]
                lo = np.linspace(b[0] - deg * dmin, b[0] - dmin, deg)
                hi = np.linspace(b[nk - 1] + dmax, b[nk - 1] + deg * dmax, deg)
                for q in range(deg):
                    t[q] = lo[q]
                    t[nk + deg + q] = hi[q]
            for q in range(nk):
                t[deg + q] = b[q]
            self._t.append(t)
        self._n_splines = ns
        self.n_features_in_ = p
        self.n_features_out_ = p * ns - p * (0 if self.include_bias else 1)
        return self

    def transform(self, X):
        Xa = np.asarray(X, dtype=float)
        n, p = Xa.shape
        if p != self.n_features_in_:
            raise ValueError(f"X has {p} features, but SplineTransformer is expecting {self.n_features_in_} features as input.")
        if self.handle_missing != "zeros" and np.any(np.isnan(Xa)):
            raise ValueError("Input X contains NaN.")
        ns = self._n_splines
        deg = self.degree
        nb = ns + deg if self.extrapolation == "periodic" else ns
        XBS = np.zeros((n, p * ns))
        for f in range(p):
            t = self._t[f]
            col = np.ascontiguousarray(Xa[:, f])
            nanm = np.isnan(col)
            o = f * ns
            if self.extrapolation == "continue" or self.extrapolation == "error" or self.extrapolation == "periodic":
                x = col
                if self.extrapolation == "periodic":
                    nn = t.shape[0] - deg - 1
                    if t[nn] - t[deg] > 0:
                        x = t[deg] + np.mod(col - t[deg], t[nn] - t[deg])
                    else:
                        x = np.zeros(n)
                v = _spl_eval(t, deg, nb, x, 0, self.extrapolation != "error")
                for i in range(n):
                    for j in range(ns):
                        XBS[i, o + j] = v[i, j]
                    if self.extrapolation == "periodic":
                        for j in range(deg):
                            XBS[i, o + j] += v[i, ns + j]
                    if nanm[i]:
                        for j in range(ns):
                            XBS[i, o + j] = 0.0
                if self.extrapolation == "error":
                    for i in range(n):
                        for j in range(ns):
                            if XBS[i, o + j] != XBS[i, o + j]:
                                raise ValueError("`X` contains values beyond the limits of the knots.")
            else:
                xmin = t[deg]
                xmax = t[t.shape[0] - deg - 1]
                ends = _spl_eval(t, deg, nb, np.array([xmin, xmax]), 0, False)
                inside = List[int]()
                for i in range(n):
                    if xmin <= col[i] and col[i] <= xmax:
                        inside.append(i)
                if len(inside) > 0:
                    ia = np.array(inside)
                    v = _spl_eval(t, deg, nb, col[ia], 0, False)
                    for q in range(len(inside)):
                        for j in range(ns):
                            XBS[inside[q], o + j] = v[q, j]
                if self.extrapolation == "constant":
                    for i in range(n):
                        if col[i] < xmin:
                            for j in range(deg):
                                XBS[i, o + j] = ends[0, j]
                        if col[i] > xmax:
                            for j in range(ns - deg, ns):
                                XBS[i, o + j] = ends[1, j]
                elif self.extrapolation == "linear":
                    d = _spl_eval(t, deg, nb, np.array([xmin, xmax]), 1, False)
                    dg = deg + 1 if deg <= 1 else deg
                    for j in range(dg):
                        k = ns - 1 - j
                        for i in range(n):
                            if col[i] < xmin:
                                XBS[i, o + j] = ends[0, j] + (col[i] - xmin) * d[0, j]
                            if col[i] > xmax:
                                XBS[i, o + k] = ends[1, k] + (col[i] - xmax) * d[1, k]
        if self.include_bias:
            return XBS
        keep = List[int]()
        for j in range(p * ns):
            if (j + 1) % ns != 0:
                keep.append(j)
        return np.ascontiguousarray(XBS[:, np.array(keep)])

    def fit_transform(self, X, y = None, sample_weight = None):
        return self.fit(X, y, sample_weight).transform(X)
