# Ported from scipy 1.18.1 (BSD-3-Clause): BPoly (scipy/interpolate/
# _interpolate.py, Bernstein evaluation from _ppoly.pyx).  Real coefficients.

import numpy as np
import math
import internal.static as static
from _common import NAN, _ext_mode, _fmod_py, _pp_find, _finish
from _ppoly import PPoly

def _comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return 0.0
    r = 1.0
    for i in range(min(k, n - k)):
        r = r * (n - i) / (i + 1)
    return float(round(r))

def _poch(a: float, m: int) -> float:
    r = 1.0
    for i in range(m):
        r *= a + i
    return r

def _bpoly1(s: float, c, ci: int, cj: int) -> float:
    k = c.shape[0] - 1
    s1 = 1.0 - s
    if k == 0:
        return c[0, ci, cj]
    elif k == 1:
        return c[0, ci, cj] * s1 + c[1, ci, cj] * s
    elif k == 2:
        return c[0, ci, cj] * s1 * s1 + c[1, ci, cj] * 2. * s1 * s + c[2, ci, cj] * s * s
    elif k == 3:
        return (c[0, ci, cj] * s1 * s1 * s1 + c[1, ci, cj] * 3. * s1 * s1 * s +
                c[2, ci, cj] * 3. * s1 * s * s + c[3, ci, cj] * s * s * s)
    res = 0.0
    cb = 1.0
    for j in range(k + 1):
        res += cb * s ** j * s1 ** (k - j) * c[j, ci, cj]
        cb *= 1. * (k - j) / (j + 1.)
    return res

def _bpoly1_deriv(s: float, c, ci: int, cj: int, nu: int, wrk) -> float:
    k = c.shape[0] - 1
    if nu == 0:
        return _bpoly1(s, c, ci, cj)
    poch = 1.0
    for a in range(nu):
        poch *= k - a
    for a in range(k - nu + 1):
        term = 0.0
        cb = 1.0
        for j in range(nu + 1):
            term += c[j + a, ci, cj] * (-1.0) ** (j + nu) * cb
            cb *= 1. * (nu - j) / (j + 1)
        wrk[a, 0, 0] = term * poch
    return _bpoly1(s, wrk, 0, 0)

def _eval_bernstein(c3, x, xs, nu: int, ext: bool):
    N = xs.size
    R = c3.shape[2]
    out = np.empty((N, R))
    wrk = np.empty((max(c3.shape[0] - nu, 1), 1, 1))
    for ip in range(N):
        xv = xs[ip]
        i = _pp_find(x, xv, ext)
        if i < 0:
            for jp in range(R):
                out[ip, jp] = NAN
            continue
        ds = x[i + 1] - x[i]
        dsn = ds ** nu
        for jp in range(R):
            s = (xv - x[i]) / ds
            if nu == 0:
                out[ip, jp] = _bpoly1(s, c3, i, jp)
            else:
                out[ip, jp] = _bpoly1_deriv(s, c3, i, jp, nu, wrk) / dsn
    return out

def _raise_degree(c, d: int):
    # c: (k+1, R)
    if d == 0:
        return c
    k = c.shape[0] - 1
    R = c.shape[1]
    out = np.zeros((c.shape[0] + d, R))
    for a in range(c.shape[0]):
        for r in range(R):
            f = c[a, r] * _comb(k, a)
            for j in range(d + 1):
                out[a + j, r] += f * _comb(d, j) / _comb(k + d, a + j)
    return out

def _construct_from_derivatives(xa: float, xb: float, ya, yb):
    # ya: (na, R), yb: (nb, R)
    na = ya.shape[0]
    nb = yb.shape[0]
    R = ya.shape[1]
    n = na + nb
    c = np.empty((n, R))
    for q in range(na):
        for r in range(R):
            c[q, r] = ya[q, r] / _poch(float(n - q), q) * (xb - xa) ** q
        for j in range(q):
            for r in range(R):
                c[q, r] -= (-1.0) ** (j + q) * _comb(q, j) * c[j, r]
    for q in range(nb):
        for r in range(R):
            c[n - q - 1, r] = yb[q, r] / _poch(float(n - q), q) * (-1.0) ** q * (xb - xa) ** q
        for j in range(q):
            for r in range(R):
                c[n - q - 1, r] -= (-1.0) ** (j + 1) * _comb(q, j + 1) * c[n - q + j, r]
    return c

class BPoly:
    def __init__(self, c, x, extrapolate = None, axis: int = 0, _ext: int = -1):
        ca = np.asarray(c, dtype=float)
        if static.len(ca.shape) < 2:
            compile_error("Coefficients array must be at least 2-dimensional.")
        if axis != 0:
            ca = np.moveaxis(ca, axis + 1, 0)
            ca = np.moveaxis(ca, axis + 1, 0)
        xa = np.ascontiguousarray(np.asarray(x, dtype=float))
        if xa.size < 2:
            raise ValueError("at least 2 breakpoints are needed")
        if ca.shape[1] != xa.size - 1:
            raise ValueError("number of coefficients != len(x)-1")
        self.c = np.ascontiguousarray(ca)
        self.x = xa
        self.tsh = ca.shape[2:]
        self._c3 = self.c.reshape((ca.shape[0], ca.shape[1], -1))
        self._ext = _ext if _ext >= 0 else _ext_mode(extrapolate, 1)
        self.axis = axis

    def _new(self, c3, ext: int):
        b = BPoly(c3.reshape((c3.shape[0], c3.shape[1]) + self.tsh), self.x, axis=0, _ext=ext)
        b.axis = self.axis
        return b

    @property
    def extrapolate(self) -> bool:
        return self._ext != 0

    def __call__(self, x, nu: int = 0, extrapolate = None):
        if nu < 0:
            raise NotImplementedError("Cannot do antiderivatives in the B-basis yet.")
        ext = _ext_mode(extrapolate, self._ext)
        xa = np.asarray(x, dtype=float)
        xs = np.ascontiguousarray(xa.reshape(-1)).copy()
        if ext == 2:
            x0 = self.x[0]
            per = self.x[self.x.size - 1] - x0
            for i in range(xs.size):
                xs[i] = x0 + _fmod_py(xs[i] - x0, per)
            ext = 0
        out = _eval_bernstein(self._c3, self.x, xs, nu, ext == 1)
        return _finish(out, xa.shape, self.tsh, self.axis)

    def derivative(self, nu: int = 1):
        if nu < 0:
            return self.antiderivative(-nu)
        bp = self
        for it in range(nu):
            c = bp._c3
            k = c.shape[0] - 1
            m = c.shape[1]
            R = c.shape[2]
            if k == 0:
                c2 = np.zeros((1, m, R))
            else:
                c2 = np.empty((k, m, R))
                for a in range(k):
                    for i in range(m):
                        dx = bp.x[i + 1] - bp.x[i]
                        for r in range(R):
                            c2[a, i, r] = k * (c[a + 1, i, r] - c[a, i, r]) / dx
            bp = bp._new(c2, bp._ext)
        return bp

    def antiderivative(self, nu: int = 1):
        if nu <= 0:
            return self.derivative(-nu)
        bp = self
        for it in range(nu):
            c = bp._c3
            k = c.shape[0]
            m = c.shape[1]
            R = c.shape[2]
            c2 = np.zeros((k + 1, m, R))
            for i in range(m):
                for r in range(R):
                    s = 0.0
                    for a in range(k):
                        s += c[a, i, r]
                        c2[a + 1, i, r] = s / k
            for i in range(m):
                d = bp.x[i + 1] - bp.x[i]
                for a in range(k + 1):
                    for r in range(R):
                        c2[a, i, r] *= d
            for r in range(R):
                acc = 0.0
                for i in range(1, m):
                    acc += c2[k, i - 1, r]
                    for a in range(k + 1):
                        c2[a, i, r] += acc
            bp = bp._new(c2, 0 if bp._ext == 2 else bp._ext)
        return bp

    def integrate(self, a: float, b: float, extrapolate = None):
        ib = self.antiderivative()
        ext = _ext_mode(extrapolate, self._ext)
        if ext == 2:
            sign = 1.0
            if b < a:
                a, b = b, a
                sign = -1.0
            xs = self.x[0]
            xe = self.x[self.x.size - 1]
            period = xe - xs
            interval = b - a
            n_periods = math.floor(interval / period)
            left = interval - n_periods * period
            res = n_periods * (ib(np.array([xe]))[0] - ib(np.array([xs]))[0])
            a = xs + _fmod_py(a - xs, period)
            b = a + left
            if b <= xe:
                res += ib(np.array([b]))[0] - ib(np.array([a]))[0]
            else:
                res += ib(np.array([xe]))[0] - ib(np.array([a]))[0] + ib(np.array([xs + left + a - xe]))[0] - ib(np.array([xs]))[0]
            return sign * res
        else:
            ib._ext = ext
            return ib(np.array([b]))[0] - ib(np.array([a]))[0]

    @staticmethod
    def from_power_basis(pp: PPoly, extrapolate = None):
        c3 = pp._c3
        k = c3.shape[0] - 1
        m = c3.shape[1]
        R = c3.shape[2]
        c = np.zeros((k + 1, m, R))
        for a in range(k + 1):
            for i in range(m):
                dx = pp.x[i + 1] - pp.x[i]
                for r in range(R):
                    factor = c3[a, i, r] / _comb(k, k - a) * dx ** (k - a)
                    for j in range(k - a, k + 1):
                        c[j, i, r] += factor * _comb(j, k - a)
        ext = pp._ext if extrapolate is None else _ext_mode(extrapolate, 1)
        b = BPoly(c.reshape((k + 1, m) + pp.tsh), pp.x, axis=0, _ext=ext)
        b.axis = pp.axis
        return b

    @staticmethod
    def from_derivatives(xi, yi, orders = None, extrapolate = None):
        xa = np.asarray(xi, dtype=float)
        ys = [np.atleast_1d(np.asarray(v, dtype=float)) for v in yi]
        if xa.size != len(ys):
            raise ValueError("xi and yi need to have the same length")
        for i in range(1, xa.size):
            if xa[i] - xa[0] <= 0:
                raise ValueError("x coordinates are not in increasing order")
        m = xa.size - 1
        k = 0
        for i in range(m):
            k = max(k, ys[i].shape[0] + ys[i + 1].shape[0])
        ords = [-1] * m
        if orders is not None:
            if isinstance(orders, int):
                ords = [orders] * m
            else:
                ords = [int(o) for o in orders]
            for o in ords:
                k = max(k, o)
                if o <= 0:
                    raise ValueError("Orders must be positive.")
        R = 1
        cs = List[np.ndarray[float, 2]]()
        for i in range(m):
            y1 = ys[i].reshape((ys[i].shape[0], -1))
            y2 = ys[i + 1].reshape((ys[i + 1].shape[0], -1))
            R = y1.shape[1]
            n1 = y1.shape[0]
            n2 = y2.shape[0]
            if ords[i] >= 0:
                n = ords[i] + 1
                n1 = min(n // 2, y1.shape[0])
                n2 = min(n - n1, y2.shape[0])
                n1 = min(n - n2, y2.shape[0])
                if n1 + n2 != n:
                    raise ValueError(f"Point {xa[i]} has {y1.shape[0]} derivatives, point {xa[i + 1]} has {y2.shape[0]} derivatives, but order {ords[i]} requested")
                if not (n1 <= y1.shape[0] and n2 <= y2.shape[0]):
                    raise ValueError("`order` input incompatible with length y1 or y2.")
            b = _construct_from_derivatives(xa[i], xa[i + 1], y1[:n1], y2[:n2])
            if b.shape[0] < k:
                b = _raise_degree(b, k - b.shape[0])
            cs.append(b)
        c = np.empty((k, m, R))
        for i in range(m):
            for a in range(k):
                for r in range(R):
                    c[a, i, r] = cs[i][a, r]
        tsh = ys[0].shape[1:]
        return BPoly(c.reshape((k, m) + tsh), xa, extrapolate)
