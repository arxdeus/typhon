# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# PPoly: piecewise polynomials in the power basis.  Coefficients are stored
# with shape (k, m) + tsh (tsh = trailing value shape) and, internally, as a
# (k, m, R) float array.

import numpy as np
import math
import internal.static as static
from _common import NAN, _ext_mode, _fmod_py, _pp_find, _finish, _poch

def _poly1(s: float, c, ci: int, cj: int, dx: int) -> float:
    k = c.shape[0]
    res = 0.0
    z = 1.0
    if dx < 0:
        for kp in range(-dx):
            z *= s
    for kp in range(k):
        prefactor = 1.0
        if dx == 0:
            prefactor = 1.0
        elif dx > 0:
            if kp < dx:
                continue
            for q in range(kp, kp - dx, -1):
                prefactor *= float(q)
        else:
            for q in range(kp + 1, kp - dx + 1):
                prefactor /= float(q)
        res += c[k - kp - 1, ci, cj] * z * prefactor
        if kp < k - 1 and kp >= dx:
            z *= s
    return res

def _pp_eval(c3, x, xs, nu: int, ext: bool):
    N = xs.size
    R = c3.shape[2]
    out = np.empty((N, R))
    for i in range(N):
        xv = xs[i]
        iv = _pp_find(x, xv, ext)
        for j in range(R):
            out[i, j] = NAN if iv < 0 else _poly1(xv - x[iv], c3, iv, j, nu)
    return out

def _pp_integrate(c3, x, a: float, b: float, ext: bool) -> List[float]:
    R = c3.shape[2]
    ia = _pp_find(x, a, ext)
    ib = _pp_find(x, b, ext)
    res = [0.0] * R
    if ia < 0 or ib < 0:
        return [NAN] * R
    for j in range(R):
        vtot = 0.0
        for iv in range(ia, ib + 1):
            if iv == ib:
                va = _poly1(b - x[iv], c3, iv, j, -1)
            else:
                va = _poly1(x[iv + 1] - x[iv], c3, iv, j, -1)
            if iv == ia:
                vb = _poly1(a - x[iv], c3, iv, j, -1)
            else:
                vb = _poly1(0.0, c3, iv, j, -1)
            vtot += va - vb
        res[j] = vtot
    return res

class PPoly:
    def __init__(self, c, x, extrapolate = None, axis: int = 0, _ext: int = -1):
        ca = np.asarray(c, dtype=float)
        if static.len(ca.shape) < 2:
            compile_error("Coefficients array must be at least 2-dimensional.")
        if not (0 <= axis and axis < ca.ndim - 1):
            raise ValueError(f"axis={axis} must be between 0 and {ca.ndim - 1}")
        if axis != 0:
            ca = np.moveaxis(ca, axis + 1, 0)
            ca = np.moveaxis(ca, axis + 1, 0)
        xa = np.ascontiguousarray(np.asarray(x, dtype=float))
        if static.len(xa.shape) != 1:
            compile_error("x must be 1-dimensional")
        if xa.size < 2:
            raise ValueError("at least 2 breakpoints are needed")
        if ca.shape[0] == 0:
            raise ValueError("polynomial must be at least of order 0")
        if ca.shape[1] != xa.size - 1:
            raise ValueError("number of coefficients != len(x)-1")
        for i in range(xa.size - 1):
            if xa[i + 1] < xa[i]:
                raise ValueError("`x` must be strictly increasing or decreasing.")
        self.c = np.ascontiguousarray(ca)
        self.x = xa
        self.tsh = ca.shape[2:]
        self._c3 = self.c.reshape((ca.shape[0], ca.shape[1], -1))
        self._ext = _ext if _ext >= 0 else _ext_mode(extrapolate, 1)
        self.axis = axis

    def _new(self, c3, ext: int):
        return PPoly(c3.reshape((c3.shape[0], c3.shape[1]) + self.tsh), self.x,
                     axis=0, _ext=ext)._setaxis(self.axis)

    def _setaxis(self, axis: int):
        self.axis = axis
        return self

    @property
    def extrapolate(self) -> bool:
        # True for extrapolate=True and 'periodic'
        return self._ext != 0

    def __call__(self, x, nu: int = 0, extrapolate = None):
        ext = _ext_mode(extrapolate, self._ext)
        xa = np.asarray(x, dtype=float)
        xs = np.ascontiguousarray(xa.reshape(-1))
        if ext == 2:
            x0 = self.x[0]
            per = self.x[self.x.size - 1] - x0
            xs = xs.copy()
            for i in range(xs.size):
                xs[i] = x0 + _fmod_py(xs[i] - x0, per)
            ext = 0
        out = _pp_eval(self._c3, self.x, xs, nu, ext == 1)
        return _finish(out, xa.shape, self.tsh, self.axis)

    def derivative(self, nu: int = 1):
        if nu < 0:
            return self.antiderivative(-nu)
        k = self._c3.shape[0]
        m = self._c3.shape[1]
        R = self._c3.shape[2]
        k2 = k - nu
        if k2 <= 0:
            return self._new(np.zeros((1, m, R)), self._ext)
        c2 = np.empty((k2, m, R))
        for r in range(k2):
            f = _poch(k2 - r, nu)
            for i in range(m):
                for j in range(R):
                    c2[r, i, j] = self._c3[r, i, j] * f
        return self._new(c2, self._ext)

    def antiderivative(self, nu: int = 1):
        if nu <= 0:
            return self.derivative(-nu)
        k = self._c3.shape[0]
        m = self._c3.shape[1]
        R = self._c3.shape[2]
        c = np.zeros((k + nu, m, R))
        for r in range(k):
            f = _poch(k - r, nu)
            for i in range(m):
                for j in range(R):
                    c[r, i, j] = self._c3[r, i, j] / f
        # fix continuity of derivatives 0..nu-1 at breakpoints
        kk = k + nu
        order = nu - 1
        for ip in range(1, self.x.size - 1):
            xv = self.x[ip]
            for j in range(R):
                for dx in range(order, -1, -1):
                    res = _poly1(xv - self.x[ip - 1], c, ip - 1, j, dx)
                    c[kk - dx - 1, ip, j] = res / math.gamma(float(dx + 1))
        return self._new(c, 0 if self._ext == 2 else self._ext)

    def integrate(self, a: float, b: float, extrapolate = None):
        ext = _ext_mode(extrapolate, self._ext)
        sign = 1.0
        if b < a:
            a, b = b, a
            sign = -1.0
        c3 = self._c3
        R = c3.shape[2]
        if ext == 2:
            xs = self.x[0]
            xe = self.x[self.x.size - 1]
            period = xe - xs
            interval = b - a
            n_periods = math.floor(interval / period)
            left = interval - n_periods * period
            if n_periods > 0:
                full = _pp_integrate(c3, self.x, xs, xe, False)
                res = [v * n_periods for v in full]
            else:
                res = [0.0] * R
            a = xs + _fmod_py(a - xs, period)
            b = a + left
            if b <= xe:
                r2 = _pp_integrate(c3, self.x, a, b, False)
                res = [res[j] + r2[j] for j in range(R)]
            else:
                r2 = _pp_integrate(c3, self.x, a, xe, False)
                r3 = _pp_integrate(c3, self.x, xs, xs + left + a - xe, False)
                res = [res[j] + r2[j] + r3[j] for j in range(R)]
        else:
            res = _pp_integrate(c3, self.x, a, b, ext == 1)
        out = np.array([v * sign for v in res])
        r = out.reshape(self.tsh)
        if static.len(r.shape) == 0:
            return r.item()
        else:
            return r

    def solve(self, y: float = 0.0, discontinuity: bool = True, extrapolate = None):
        # real roots of p(x) == y; one-dimensional values only
        if static.len(self.tsh) != 0:
            compile_error("PPoly.solve supports scalar-valued polynomials only")
        ext = _ext_mode(extrapolate, self._ext) == 1
        c3 = self._c3
        k = c3.shape[0]
        m = c3.shape[1]
        roots = List[float]()
        last = -float('inf')
        for i in range(m):
            lo = self.x[i]
            hi = self.x[i + 1]
            coefs = [c3[r, i, 0] for r in range(k)]
            coefs[k - 1] -= y
            st = 0
            while st < k and coefs[st] == 0.0:
                st += 1
            cand = List[float]()
            if st == k:
                # identically equal to y on this interval
                cand.append(lo)
                cand.append(NAN)
            elif st < k - 1:
                rr = np.roots(np.array(coefs[st:]))
                for z in rr:
                    if abs(z.imag) <= 1e-10 * max(1.0, abs(z.real)):
                        cand.append(z.real)
                cand.sort()
            for z in cand:
                if z != z:
                    roots.append(NAN)
                    continue
                xv = z + lo
                okl = xv >= lo or (ext and i == 0)
                okr = xv <= hi or (ext and i == m - 1)
                if okl and okr:
                    if len(roots) > 0 and abs(xv - last) <= 1e-12 * max(1.0, abs(xv)):
                        continue
                    roots.append(xv)
                    last = xv
            if discontinuity and i + 1 < m:
                lv = _poly1(hi - lo, c3, i, 0, 0) - y
                rv = c3[k - 1, i + 1, 0] - y
                if lv != 0.0 and rv != 0.0 and (lv < 0.0) != (rv < 0.0):
                    if not (len(roots) > 0 and roots[len(roots) - 1] == hi):
                        roots.append(hi)
                        last = hi
        out = np.empty(len(roots))
        for i in range(len(roots)):
            out[i] = roots[i]
        return out

    def roots(self, discontinuity: bool = True, extrapolate = None):
        return self.solve(0.0, discontinuity, extrapolate)

    @staticmethod
    def from_spline(tck, extrapolate = None):
        from _bspl import _bspl_eval
        t = np.asarray(tck[0], dtype=float)
        c = np.asarray(tck[1], dtype=float)
        k = int(tck[2])
        nt = t.size - 1
        cvals = np.empty((k + 1, nt))
        c2 = c.reshape((c.shape[0], 1))
        for m in range(k, -1, -1):
            yv = _bspl_eval(t, c2, k, t[:nt].copy(), m, True)
            g = math.gamma(float(m + 1))
            for i in range(nt):
                cvals[k - m, i] = yv[i, 0] / g
        return PPoly(cvals, t, extrapolate)
