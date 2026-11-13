# Ported from scipy 1.18.1 (BSD-3-Clause): scipy/spatial/transform/_rotation_spline.py

import numpy as np
from math import sqrt, sin, cos, tan
from scipy.linalg import solve_banded
from ._rotation import Rotation, _from_rotvec, compose_quat, _normalize_rows

def _skew(x, i: int):
    S = np.zeros((3, 3))
    S[0, 1] = -x[i, 2]
    S[0, 2] = x[i, 1]
    S[1, 0] = x[i, 2]
    S[1, 2] = -x[i, 0]
    S[2, 0] = -x[i, 1]
    S[2, 1] = x[i, 0]
    return S

def _nrm(x, i: int) -> float:
    return sqrt(x[i, 0] ** 2 + x[i, 1] ** 2 + x[i, 2] ** 2)

def _rate_to_rotvec_dot(rv):
    n = rv.shape[0]
    out = np.empty((n, 3, 3))
    for i in range(n):
        nm = _nrm(rv, i)
        k = 0.0
        if nm > 0.0001:
            k = (1 - 0.5 * nm / tan(0.5 * nm)) / nm ** 2
        else:
            k = 1 / 12 + 1 / 720 * nm ** 2
        S = _skew(rv, i)
        out[i] = np.identity(3) + 0.5 * S + k * (S @ S)
    return out

def _rotvec_dot_to_rate(rv):
    n = rv.shape[0]
    out = np.empty((n, 3, 3))
    for i in range(n):
        nm = _nrm(rv, i)
        k1 = 0.0
        k2 = 0.0
        if nm > 0.0001:
            k1 = (1 - cos(nm)) / nm ** 2
            k2 = (nm - sin(nm)) / nm ** 3
        else:
            k1 = 0.5 - nm ** 2 / 24
            k2 = 1 / 6 - nm ** 2 / 120
        S = _skew(rv, i)
        out[i] = np.identity(3) - k1 * S + k2 * (S @ S)
    return out

def _cr(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])

def _nonlinear(rv, rvd):
    n = rv.shape[0]
    out = np.empty((n, 3))
    for i in range(n):
        nm = _nrm(rv, i)
        dp = rv[i, 0] * rvd[i, 0] + rv[i, 1] * rvd[i, 1] + rv[i, 2] * rvd[i, 2]
        cp = _cr(rv[i], rvd[i])
        ccp = _cr(rv[i], cp)
        dccp = _cr(rvd[i], cp)
        k1 = 0.0
        k2 = 0.0
        k3 = 0.0
        if nm > 0.0001:
            k1 = (-nm * sin(nm) - 2 * (cos(nm) - 1)) / nm ** 4
            k2 = (-2 * nm + 3 * sin(nm) - nm * cos(nm)) / nm ** 5
            k3 = (nm - sin(nm)) / nm ** 3
        else:
            k1 = 1 / 12 - nm ** 2 / 180
            k2 = -1 / 60 + nm ** 2 / 12604
            k3 = 1 / 6 - nm ** 2 / 120
        for k in range(3):
            out[i, k] = dp * (k1 * cp[k] + k2 * ccp[k]) + k3 * dccp[k]
    return out

def _mv(A, b):
    n = b.shape[0]
    out = np.empty((n, 3))
    for i in range(n):
        for r in range(3):
            out[i, r] = A[i, r, 0] * b[i, 0] + A[i, r, 1] * b[i, 1] + A[i, r, 2] * b[i, 2]
    return out

def _compute_angular_rate(rv, rvd):
    return _mv(_rotvec_dot_to_rate(rv), rvd)

def _compute_angular_acceleration(rv, rvd, rvdd):
    return _compute_angular_rate(rv, rvdd) + _nonlinear(rv, rvd)

def _block3(A, B, d):
    nb = d.shape[0]
    u = 5
    res = np.zeros((11, 3 * nb))
    for b in range(A.shape[0]):
        for r in range(3):
            for c in range(3):
                i = r + 3 * (1 + b)
                j = c + 3 * b
                res[u + i - j, j] = A[b, r, c]
    for b in range(B.shape[0]):
        for r in range(3):
            for c in range(3):
                i = r + 3 * b
                j = c + 3 * (1 + b)
                res[u + i - j, j] = B[b, r, c]
    for b in range(nb):
        for r in range(3):
            res[u, 3 * b + r] = d[b]
    return res

class RotationSpline:
    def _solve(self, dt, rates, rv):
        first = rates[0].copy()
        A = _rate_to_rotvec_dot(rv)
        Ai = _rotvec_dot_to_rate(rv)
        n = rv.shape[0]
        Aa = np.empty((n - 2, 3, 3))
        Bb = np.empty((n - 2, 3, 3))
        for i in range(1, n - 1):
            Aa[i - 1] = 2 * Ai[i] / dt[i]
            Bb[i - 1] = 2 * A[i] / dt[i]
        d = np.empty(n - 1)
        for i in range(n - 1):
            d[i] = 4 * (1 / dt[i] + 1 / dt[i + 1])
        M = _block3(Aa, Bb, d)
        b0 = np.empty((n - 1, 3))
        for i in range(n - 1):
            for k in range(3):
                b0[i, k] = 6 * (rv[i, k] * dt[i] ** (-2) + rv[i + 1, k] * dt[i + 1] ** (-2))
        t0 = Ai[0] @ first
        tl = A[n - 1] @ rates[n - 1]
        for k in range(3):
            b0[0, k] -= 2 / dt[0] * t0[k]
            b0[n - 2, k] -= 2 / dt[n - 1] * tl[k]
        for it in range(10):
            rvd = _mv(A, rates)
            dbeta = _nonlinear(rv[:n - 1], rvd[:n - 1])
            b = b0 - dbeta
            new = solve_banded((5, 5), M, b.ravel()).reshape((-1, 3))
            ok = True
            for i in range(n - 1):
                for k in range(3):
                    if not (abs(new[i, k] - rates[i, k]) < 1e-9 * (1 + abs(new[i, k]))):
                        ok = False
                    rates[i, k] = new[i, k]
            if ok:
                break
        rvd = _mv(A, rates)
        out = np.empty((n, 3))
        out[0] = first
        for i in range(1, n):
            out[i] = rates[i - 1]
        return out, rvd

    def __init__(self, times, rotations: Rotation[False]):
        from scipy.interpolate import PPoly
        t = np.asarray(times, dtype=float)
        nr = len(rotations)
        if nr == 1:
            raise ValueError("`rotations` must contain at least 2 rotations.")
        if t.ndim != 1:
            raise ValueError("`times` must be 1-dimensional.")
        if t.shape[0] != nr:
            raise ValueError(f"Expected number of rotations to be equal to number of timestamps given, got {nr} rotations and {t.shape[0]} timestamps.")
        dt = np.diff(t)
        for i in range(dt.size):
            if dt[i] <= 0:
                raise ValueError("Values in `times` must be in a strictly increasing order.")
        q = rotations._quat
        r0 = Rotation[False](q[:nr - 1].copy())
        r1 = Rotation[False](q[1:].copy())
        rv = (r0.inv() * r1).as_rotvec()
        rates = rv / dt.reshape((-1, 1))
        rvd = rates
        if nr > 2:
            rates, rvd = self._solve(dt, rates.copy(), rv)
        dtc = dt.reshape((-1, 1))
        coeff = np.empty((4, nr - 1, 3))
        coeff[0] = (-2 * rv + dtc * rates + dtc * rvd) / dtc ** 3
        coeff[1] = (3 * rv - 2 * dtc * rates - dtc * rvd) / dtc ** 2
        coeff[2] = rates
        coeff[3] = 0.0
        self.times = t
        self.rotations = rotations
        self.interpolator = PPoly(coeff, t)

    def _rot(self, tt):
        rv = self.interpolator(tt)
        m = tt.shape[0]
        ns = self.times.shape[0] - 1
        q = np.empty((m, 4))
        dq = _from_rotvec(rv)
        for i in range(m):
            idx = int(np.searchsorted(self.times, tt[i], side='right')) - 1
            if idx < 0:
                idx = 0
            if idx > ns - 1:
                idx = ns - 1
            qi = compose_quat(self.rotations._quat[idx:idx + 1], dq[i:i + 1])
            _normalize_rows(qi)
            q[i] = qi[0]
        return q

    def __call__(self, times, order: Literal[int] = 0):
        ta = np.asarray(times, dtype=float)
        tt = np.atleast_1d(ta).ravel()
        single: Literal[bool] = isinstance(times, float) or isinstance(times, int)
        if order == 0:
            q = self._rot(tt)
            if single:
                return Rotation[True](q)
            else:
                return Rotation[False](q)
        else:
            rv = self.interpolator(tt)
            rvd = self.interpolator(tt, 1)
            res = np.zeros((1, 1))
            if order == 1:
                res = _compute_angular_rate(rv, rvd)
            elif order == 2:
                res = _compute_angular_acceleration(rv, rvd, self.interpolator(tt, 2))
            else:
                compile_error("`order` must be 0, 1 or 2.")
            if single:
                return res[0].copy()
            else:
                return res
