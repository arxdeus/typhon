# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# NdBSpline: tensor-product B-spline in N dimensions
# (scipy/interpolate/_ndbspline.py, evaluate_ndbspline from _dierckx.cc).
# Real coefficients only; the knot vectors are a tuple of arrays.

import numpy as np
import internal.static as static
from _common import NAN, _bspl_find, _deboor
from _bspl import BSpline

def _k_list(k, ndim: int) -> List[int]:
    if isinstance(k, int):
        return [k] * ndim
    out = List[int]()
    for v in k:
        out.append(int(v))
    if len(out) != ndim:
        raise ValueError(f"len(t) = {ndim} != len(k) = {len(out)}.")
    return out

class NdBSpline[T]:
    t: T
    _c: np.ndarray[float, 1]
    _cshape: List[int]
    _k: List[int]
    _tl: List[np.ndarray[float, 1]]
    extrapolate: bool

    def __init__(self, t: T, c, k, *, extrapolate = None):
        self.t = t
        nd = static.len(t)
        self._k = _k_list(k, nd)
        self._tl = _knots(t)
        self.extrapolate = True if extrapolate is None else bool(extrapolate)
        ca = np.ascontiguousarray(np.asarray(c, dtype=float))
        self._cshape = [ca.shape[i] for i in range(ca.ndim)]
        if len(self._cshape) < nd:
            raise ValueError(f"Coefficients must be at least {nd}-dimensional.")
        for d in range(nd):
            kd = self._k[d]
            n = self._tl[d].shape[0] - kd - 1
            if kd < 0:
                raise ValueError(f"Spline degree in dimension {d} cannot be negative.")
            if n < kd + 1:
                raise ValueError(f"Need at least {2 * kd + 2} knots for degree {kd} in dimension {d}, got {self._tl[d].shape[0]} knots.")
            ts = self._tl[d]
            for i in range(ts.shape[0] - 1):
                if ts[i] > ts[i + 1]:
                    raise ValueError(f"Knots in dimension {d} must be in a non-decreasing order.")
            if self._cshape[d] != n:
                raise ValueError(f"Knots, coefficients and degree in dimension {d} are inconsistent: got {self._cshape[d]} coefficients for {self._tl[d].shape[0]} knots, need at least {n} for k={kd}.")
        self._c = ca.reshape(-1)

    @property
    def k(self):
        return self._k

    @property
    def c(self):
        return self._c.reshape(self._cshape)

    def __call__(self, xi, *, nu = None, extrapolate = None):
        nd = len(self._k)
        ext = self.extrapolate if extrapolate is None else bool(extrapolate)
        nus = [0] * nd
        if nu is not None:
            nus = [int(v) for v in nu]
            if len(nus) != nd:
                raise ValueError(f"invalid number of derivative orders nu for ndim = {nd}.")
            for v in nus:
                if v < 0:
                    raise ValueError("derivatives must be positive")
        xa = np.asarray(xi, dtype=float)
        if xa.shape[xa.ndim - 1] != nd:
            raise ValueError(f"Shapes: xi.shape={xa.shape} and ndim={nd}")
        X = np.ascontiguousarray(xa).reshape((-1, nd))
        trail = 1
        for i in range(nd, len(self._cshape)):
            trail *= self._cshape[i]
        # strides of the coefficient array over the first nd axes (in units of trail)
        cst = [1] * nd
        for d in range(nd - 2, -1, -1):
            cst[d] = cst[d + 1] * self._cshape[d + 1]
        out = np.zeros((X.shape[0], trail))
        for p in range(X.shape[0]):
            bad = False
            ells = [0] * nd
            hs = List[List[float]]()
            for d in range(nd):
                kd = self._k[d]
                td = self._tl[d]
                xv = X[p, d]
                if xv != xv:
                    bad = True
                    hs.append([0.0])
                    continue
                ell = _bspl_find(td, kd, xv, kd, ext)
                if ell < 0:
                    bad = True
                    hs.append([0.0])
                    continue
                ells[d] = ell
                if nus[d] > kd:
                    hs.append([0.0] * (kd + 1))
                else:
                    hs.append(_deboor(td, xv, kd, ell, nus[d]))
            if bad:
                for r in range(trail):
                    out[p, r] = NAN
                continue
            idx = [0] * nd
            while True:
                w = 1.0
                off = 0
                for d in range(nd):
                    w *= hs[d][idx[d]]
                    off += (ells[d] - self._k[d] + idx[d]) * cst[d]
                for r in range(trail):
                    out[p, r] += w * self._c[off * trail + r]
                d = nd - 1
                while d >= 0:
                    idx[d] += 1
                    if idx[d] <= self._k[d]:
                        break
                    idx[d] = 0
                    d -= 1
                if d < 0:
                    break
        return _out_shape(xa, self._cshape, nd, out)

def _out_shape(xa, cshape: List[int], nd: int, out):
    lead = [xa.shape[i] for i in range(xa.ndim - 1)]
    full = lead + cshape[nd:]
    return _reshape_list(out, full)

def _reshape_list(out, shape: List[int]):
    if len(shape) == 1:
        return out.reshape((shape[0],))
    elif len(shape) == 2:
        return out.reshape((shape[0], shape[1]))
    elif len(shape) == 3:
        return out.reshape((shape[0], shape[1], shape[2]))
    elif len(shape) == 0:
        return out.reshape(())
    raise ValueError("NdBSpline: output with more than 3 dimensions is not supported")

def _knots(t) -> List[np.ndarray[float, 1]]:
    out = List[np.ndarray[float, 1]]()
    for v in t:
        out.append(np.ascontiguousarray(np.asarray(v, dtype=float)))
    return out
