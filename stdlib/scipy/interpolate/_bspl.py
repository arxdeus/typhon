# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# B-splines: BSpline, make_interp_spline and the FITPACK-style tck helpers.

import numpy as np
import math
import internal.static as static
from _common import NAN, _ext_mode, _fmod_py, _bspl_find, _deboor, _band_solve, \
    _finish, _vals

def _bspl_eval(t, c2, k: int, xs, nu: int, ext: bool):
    # c2: (n, R) coefficients; xs: flat float array -> (N, R)
    N = xs.size
    R = c2.shape[1]
    out = np.empty((N, R))
    if nu < 0:
        raise NotImplementedError(f"Cannot do derivative order {nu}.")
    ell = k
    for i in range(N):
        xv = xs[i]
        ell = _bspl_find(t, k, xv, ell, ext)
        if ell < 0:
            for j in range(R):
                out[i, j] = NAN
            continue
        if nu > k:
            for j in range(R):
                out[i, j] = 0.0
            continue
        h = _deboor(t, xv, k, ell, nu)
        for j in range(R):
            s = 0.0
            for a in range(k + 1):
                s += c2[ell + a - k, j] * h[a]
            out[i, j] = s
    return out

def _splder(t, c2, k: int, n: int):
    # t: knots, c2: (len(t), R) padded coefficients
    if n > k:
        raise ValueError(f"Order of derivative (n = {n}) must be <= order of spline (k = {k})")
    R = c2.shape[1]
    for _ in range(n):
        nt = t.size
        m = nt - k - 2
        cn = np.zeros((nt - 2, R))
        for i in range(m):
            dt = t[i + k + 1] - t[i + 1]
            if dt == 0.0:
                raise ValueError(f"The spline has internal repeated knots and is not differentiable {n} times")
            for j in range(R):
                cn[i, j] = (c2[i + 1, j] - c2[i, j]) * k / dt
        t = t[1:nt - 1].copy()
        c2 = cn
        k -= 1
    return t, c2, k

def _splantider(t, c2, k: int, n: int):
    R = c2.shape[1]
    for _ in range(n):
        nt = t.size
        m = nt - k - 1
        cn = np.zeros((nt + 2, R))
        for j in range(R):
            acc = 0.0
            for i in range(m):
                acc += c2[i, j] * (t[i + k + 1] - t[i])
                cn[i + 1, j] = acc / (k + 1)
            for i in range(m + 1, nt + 2):
                cn[i, j] = cn[m, j]
        tn = np.empty(nt + 2)
        tn[0] = t[0]
        for i in range(nt):
            tn[i + 1] = t[i]
        tn[nt + 1] = t[nt - 1]
        t = tn
        c2 = cn
        k += 1
    return t, c2, k

def _pad(t, c2):
    ct = t.size - c2.shape[0]
    if ct > 0:
        R = c2.shape[1]
        cn = np.zeros((t.size, R))
        for i in range(c2.shape[0]):
            for j in range(R):
                cn[i, j] = c2[i, j]
        return cn
    return c2

class BSpline:
    def __init__(self, t, c, k: int, extrapolate = True, axis: int = 0, _fast: bool = False):
        ta = np.ascontiguousarray(np.asarray(t, dtype=float))
        ca = np.asarray(c, dtype=float)
        if static.len(ca.shape) < 1:
            compile_error("Coefficients must be at least 1-dimensional.")
        ax = axis % ca.ndim
        if ax != 0:
            ca = np.moveaxis(ca, ax, 0)
        if not _fast:
            n = ta.size - k - 1
            if k < 0:
                raise ValueError("Spline order cannot be negative.")
            if n < k + 1:
                raise ValueError(f"Need at least {2 * k + 2} knots for degree {k}")
            for i in range(ta.size - 1):
                if ta[i + 1] < ta[i]:
                    raise ValueError("Knots must be in a non-decreasing order.")
            if ta[k] == ta[n]:
                raise ValueError("Need at least two internal knots.")
            for i in range(ta.size):
                if not np.isfinite(ta[i]):
                    raise ValueError("Knots should not have nans or infs.")
            if ca.shape[0] < n:
                raise ValueError("Knots, coefficients and degree are inconsistent.")
        self.t = ta
        self.c = np.ascontiguousarray(ca)
        self.k = k
        self.axis = ax
        self.tsh = ca.shape[1:]
        self._c2 = self.c.reshape((ca.shape[0], -1))
        self._ext = _ext_mode(extrapolate, 1)

    def _setax(self, ax: int):
        self.axis = ax
        return self

    @property
    def extrapolate(self) -> bool:
        return self._ext != 0

    @property
    def tck(self):
        return self.t, self.c, self.k

    @staticmethod
    def construct_fast(t, c, k: int, extrapolate = True, axis: int = 0):
        return BSpline(t, c, k, extrapolate, axis, _fast=True)

    def _wrap(self, t, c2, k: int, ext: int):
        b = BSpline(t, c2.reshape((c2.shape[0],) + self.tsh), k, True, 0, _fast=True)
        b._ext = ext
        b.axis = self.axis
        return b

    @staticmethod
    def basis_element(t, extrapolate = True):
        ta = np.asarray(t, dtype=float)
        k = ta.size - 2
        if k < 0:
            raise ValueError("BSpline.basis_element requires at least 2 knots")
        tt = np.empty(ta.size + 2 * k)
        for i in range(k):
            tt[i] = ta[0] - 1.0
            tt[ta.size + k + i] = ta[ta.size - 1] + 1.0
        for i in range(ta.size):
            tt[k + i] = ta[i]
        c = np.zeros(tt.size)
        c[k] = 1.0
        return BSpline(tt, c, k, extrapolate, 0, _fast=True)

    def __call__(self, x, nu: int = 0, extrapolate = None):
        ext = _ext_mode(extrapolate, self._ext)
        xa = np.asarray(x, dtype=float)
        xs = np.ascontiguousarray(xa.reshape(-1)).copy()
        if ext == 2:
            n = self.t.size - self.k - 1
            t0 = self.t[self.k]
            per = self.t[n] - t0
            for i in range(xs.size):
                xs[i] = t0 + _fmod_py(xs[i] - t0, per)
            ext = 0
        out = _bspl_eval(self.t, self._c2, self.k, xs, nu, ext == 1)
        return _finish(out, xa.shape, self.tsh, self.axis)

    def derivative(self, nu: int = 1):
        if nu < 0:
            return self.antiderivative(-nu)
        t, c2, k = _splder(self.t, _pad(self.t, self._c2), self.k, nu)
        return self._wrap(t, c2, k, self._ext)

    def antiderivative(self, nu: int = 1):
        if nu < 0:
            return self.derivative(-nu)
        t, c2, k = _splantider(self.t, _pad(self.t, self._c2), self.k, nu)
        return self._wrap(t, c2, k, 0 if self._ext == 2 else self._ext)

    def integrate(self, a: float, b: float, extrapolate = None):
        ext = _ext_mode(extrapolate, self._ext)
        sign = 1.0
        if b < a:
            a, b = b, a
            sign = -1.0
        n = self.t.size - self.k - 1
        if ext == 0:
            a = max(a, self.t[self.k])
            b = min(b, self.t[n])
        ta, ca, ka = _splantider(self.t, _pad(self.t, self._c2), self.k, 1)
        R = ca.shape[1]
        res = [0.0] * R
        if ext == 2:
            ts = self.t[self.k]
            te = self.t[n]
            period = te - ts
            interval = b - a
            n_periods = math.floor(interval / period)
            left = interval - n_periods * period
            if n_periods > 0:
                o = _bspl_eval(ta, ca, ka, np.array([ts, te]), 0, False)
                res = [(o[1, j] - o[0, j]) * n_periods for j in range(R)]
            a = ts + _fmod_py(a - ts, period)
            b = a + left
            if b <= te:
                o = _bspl_eval(ta, ca, ka, np.array([a, b]), 0, False)
                res = [res[j] + o[1, j] - o[0, j] for j in range(R)]
            else:
                o = _bspl_eval(ta, ca, ka, np.array([a, te]), 0, False)
                o2 = _bspl_eval(ta, ca, ka, np.array([ts, ts + b - te]), 0, False)
                res = [res[j] + o[1, j] - o[0, j] + o2[1, j] - o2[0, j] for j in range(R)]
        else:
            o = _bspl_eval(ta, ca, ka, np.array([a, b]), 0, ext == 1)
            res = [o[1, j] - o[0, j] for j in range(R)]
        out = np.array([v * sign for v in res]).reshape(self.tsh)
        if static.len(out.shape) == 0:
            return out.item()
        else:
            return out

    @staticmethod
    def design_matrix(x, t, k: int, extrapolate: bool = False):
        # dense version of scipy's sparse design matrix
        xa = np.asarray(x, dtype=float)
        ta = np.asarray(t, dtype=float)
        if k < 0:
            raise ValueError("Spline order cannot be negative.")
        if ta.size < 2 * k + 2:
            raise ValueError(f"Length t is not enough for k={k}.")
        nt = ta.size - k - 1
        out = np.zeros((xa.size, nt))
        ell = k
        for i in range(xa.size):
            xv = xa[i]
            if not extrapolate and (xv < ta[k] or xv > ta[nt]):
                raise ValueError(f"Out of bounds w/ x = {xa}.")
            ell = _bspl_find(ta, k, xv, ell, extrapolate)
            h = _deboor(ta, xv, k, ell, 0)
            for a in range(k + 1):
                out[i, ell - k + a] = h[a]
        return out
