# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# Polynomial interpolators: KroghInterpolator, BarycentricInterpolator,
# krogh_interpolate, barycentric_interpolate, approximate_taylor_polynomial,
# plus lagrange and pade (returning numpy-style coefficient arrays wrapped
# in a minimal poly1d).  Real (float64) data only.

import numpy as np
import internal.static as static
from _common import _finish

def _fact(k: int) -> float:
    r = 1.0
    for i in range(2, k + 1):
        r *= float(i)
    return r

def _yprep(xi, yi, axis: int):
    ya = np.asarray(yi, dtype=float)
    ax = axis % ya.ndim
    if ya.shape[ax] != xi.size:
        raise ValueError("x and y arrays must be equal in length along interpolation axis.")
    ym = np.moveaxis(ya, ax, 0)
    tsh = ym.shape[1:]
    return np.ascontiguousarray(ym).reshape((xi.size, -1)).copy(), tsh, ax

class KroghInterpolator:
    def __init__(self, xi, yi, axis: int = 0):
        xa = np.ascontiguousarray(np.asarray(xi, dtype=float).ravel())
        y2, tsh, ax = _yprep(xa, yi, axis)
        self._tsh = tsh
        self._axis = ax
        self.xi = xa
        self.yi = y2
        n = y2.shape[0]
        r = y2.shape[1]
        self.n = n
        self.r = r
        c = np.zeros((n + 1, r))
        for j in range(r):
            c[0, j] = y2[0, j]
        Vk = np.zeros((n, r))
        for k in range(1, n):
            s = 0
            while s <= k and xa[k - s] == xa[k]:
                s += 1
            s -= 1
            f = _fact(s)
            for j in range(r):
                Vk[0, j] = y2[k, j] / f
            for i in range(k - s):
                if xa[i] == xa[k]:
                    raise ValueError("Elements of `xi` can't be equal.")
                d = xa[i] - xa[k]
                for j in range(r):
                    if s == 0:
                        Vk[i + 1, j] = (c[i, j] - Vk[i, j]) / d
                    else:
                        Vk[i + 1, j] = (Vk[i + 1, j] - Vk[i, j]) / d
            for j in range(r):
                c[k, j] = Vk[k - s, j]
        self.c = c

    def _evaluate(self, x):
        N = x.size
        p = np.zeros((N, self.r))
        for q in range(N):
            pi = 1.0
            for j in range(self.r):
                p[q, j] = self.c[0, j]
            for k in range(1, self.n):
                pi = (x[q] - self.xi[k - 1]) * pi
                for j in range(self.r):
                    p[q, j] += pi * self.c[k, j]
        return p

    def _evaluate_derivatives(self, x, der: int):
        n = self.n
        r = self.r
        N = x.size
        pi = np.zeros((n, N))
        w = np.zeros((n, N))
        for q in range(N):
            pi[0, q] = 1.0
        p = np.zeros((N, r))
        for q in range(N):
            for j in range(r):
                p[q, j] = self.c[0, j]
        for k in range(1, n):
            for q in range(N):
                w[k - 1, q] = x[q] - self.xi[k - 1]
                pi[k, q] = w[k - 1, q] * pi[k - 1, q]
                for j in range(r):
                    p[q, j] += pi[k, q] * self.c[k, j]
        cn = np.zeros((max(der, n + 1), N, r))
        for k in range(n + 1):
            for q in range(N):
                for j in range(r):
                    cn[k, q, j] += self.c[k, j]
        for q in range(N):
            for j in range(r):
                cn[0, q, j] = p[q, j]
        for k in range(1, n):
            for i in range(1, n - k + 1):
                for q in range(N):
                    pi[i, q] = w[k + i - 1, q] * pi[i - 1, q] + pi[i, q]
                    for j in range(r):
                        cn[k, q, j] = cn[k, q, j] + pi[i, q] * cn[k + i, q, j]
            f = _fact(k)
            for q in range(N):
                for j in range(r):
                    cn[k, q, j] *= f
        for q in range(N):
            for j in range(r):
                cn[n, q, j] = 0.0
        return cn[:der].copy()

    def __call__(self, x):
        xa = np.asarray(x, dtype=float)
        xs = np.ascontiguousarray(xa.ravel())
        return _finish(self._evaluate(xs), xa.shape, self._tsh, self._axis)

    def derivatives(self, x, der: Optional[int] = None):
        xa = np.asarray(x, dtype=float)
        xs = np.ascontiguousarray(xa.ravel())
        d = self.n if der is None else der
        y = self._evaluate_derivatives(xs, d)
        return _finish_ders(y, xa.shape, self._tsh, self._axis)

    def derivative(self, x, der: int = 1):
        xa = np.asarray(x, dtype=float)
        xs = np.ascontiguousarray(xa.ravel())
        y = self._evaluate_derivatives(xs, der + 1)
        return _finish(y[der].copy(), xa.shape, self._tsh, self._axis)

def _finish_ders(y, xshape, tsh, axis: int):
    D = y.shape[0]
    r = y.reshape((D,) + xshape + tsh)
    if axis != 0 and static.len(xshape) > 0:
        nx = static.len(xshape)
        l = [i for i in range(r.ndim)]
        l = [0] + l[nx + 1:nx + 1 + axis] + l[1:nx + 1] + l[nx + 1 + axis:]
        return r.transpose(l)
    return r

def krogh_interpolate(xi, yi, x, der = 0, axis: int = 0):
    P = KroghInterpolator(xi, yi, axis=axis)
    if isinstance(der, int):
        if der == 0:
            return P.derivative(x, 0)
        return P.derivative(x, der=der)
    else:
        d = [int(v) for v in der]
        return P.derivatives(x, der=max(d) + 1)[np.array(d)]

class BarycentricInterpolator:
    def __init__(self, xi, yi, axis: int = 0, wi = None, rng = None):
        xa = np.ascontiguousarray(np.asarray(xi, dtype=float).ravel())
        y2, tsh, ax = _yprep(xa, yi, axis)
        self._tsh = tsh
        self._axis = ax
        self.xi = xa
        self.yi = y2
        self.n = xa.size
        self.r = y2.shape[1]
        n = self.n
        self._inv_capacity = 4.0 / (np.max(xa) - np.min(xa))
        if wi is not None:
            self.wi = np.asarray(wi, dtype=float).copy()
        else:
            perm = np.arange(n)
            if rng is not None:
                perm = np.random.default_rng(rng).permutation(n)
            inv = np.zeros(n, dtype=int)
            for i in range(n):
                inv[perm[i]] = i
            self.wi = np.zeros(n)
            for i in range(n):
                prod = 1.0
                for j in range(n):
                    dist = 1.0 if j == inv[i] else self._inv_capacity * (xa[i] - xa[perm[j]])
                    prod *= dist
                if prod == 0.0:
                    raise ValueError('Interpolation points xi must be distinct.')
                self.wi[i] = 1.0 / prod

    def _evaluate(self, x):
        N = x.size
        p = np.zeros((N, self.r))
        for q in range(N):
            hit = -1
            den = 0.0
            for j in range(self.n):
                cj = x[q] - self.xi[j]
                if cj == 0.0:
                    if hit < 0:
                        hit = j
                    cj = 1.0
                cw = self.wi[j] / cj
                den += cw
                for k in range(self.r):
                    p[q, k] += cw * self.yi[j, k]
            if hit >= 0:
                for k in range(self.r):
                    p[q, k] = self.yi[hit, k]
            else:
                for k in range(self.r):
                    p[q, k] /= den
        return p

    def __call__(self, x):
        xa = np.asarray(x, dtype=float)
        xs = np.ascontiguousarray(xa.ravel())
        return _finish(self._evaluate(xs), xa.shape, self._tsh, self._axis)

    def _diff_matrix(self):
        n = self.n
        c = np.zeros((n, n))
        for i in range(n):
            s = 0.0
            for j in range(n):
                if i != j:
                    c[i, j] = self.wi[j] / ((self.xi[i] - self.xi[j]) * self.wi[i])
                    s += c[i, j]
            c[i, i] = -s
        return c

    def _eval_der(self, x, der: int, cij):
        # value of derivative (der - 1)
        if der == 1:
            return self._evaluate(x)
        if der > self.n:
            return np.zeros((x.size, self.r))
        sub = BarycentricInterpolator(self.xi, cij @ self.yi, wi=self.wi)
        return sub._eval_der(x, der - 1, cij)

    def derivative(self, x, der: int = 1):
        xa = np.asarray(x, dtype=float)
        xs = np.ascontiguousarray(xa.ravel())
        y = self._eval_der(xs, der + 1, self._diff_matrix())
        return _finish(y, xa.shape, self._tsh, self._axis)

    def derivatives(self, x, der: Optional[int] = None):
        xa = np.asarray(x, dtype=float)
        xs = np.ascontiguousarray(xa.ravel())
        d = self.n if der is None else der
        cij = self._diff_matrix()
        cn = np.zeros((d, xs.size, self.r))
        for k in range(d):
            cn[k] = self._eval_der(xs, k + 1, cij)
        return _finish_ders(cn, xa.shape, self._tsh, self._axis)

def barycentric_interpolate(xi, yi, x, axis: int = 0, der = 0, rng = None):
    P = BarycentricInterpolator(xi, yi, axis=axis, rng=rng)
    if isinstance(der, int):
        if der == 0:
            return P(x)
        return P.derivative(x, der=der)
    else:
        d = [int(v) for v in der]
        return P.derivatives(x, der=max(d) + 1)[np.array(d)]

class poly1d:
    # minimal numpy.poly1d: coeffs highest power first
    coeffs: np.ndarray[float, 1]

    def __init__(self, c):
        a = np.atleast_1d(np.asarray(c, dtype=float)).ravel()
        k = 0
        while k < a.size - 1 and a[k] == 0.0:
            k += 1
        self.coeffs = a[k:].copy()

    @property
    def order(self) -> int:
        return self.coeffs.size - 1

    @property
    def c(self):
        return self.coeffs

    def __call__(self, x):
        return np.polyval(self.coeffs, x)

    def __add__(self, o: poly1d):
        a = self.coeffs
        b = o.coeffs
        n = max(a.size, b.size)
        r = np.zeros(n)
        for i in range(a.size):
            r[n - a.size + i] += a[i]
        for i in range(b.size):
            r[n - b.size + i] += b[i]
        return poly1d(r)

    def __mul__(self, o: poly1d):
        return poly1d(np.convolve(self.coeffs, o.coeffs))

    def __truediv__(self, f: float):
        return poly1d(self.coeffs / f)

    def __getitem__(self, k: int) -> float:
        # coefficient of x**k
        if k > self.order:
            return 0.0
        return self.coeffs[self.order - k]

def lagrange(x, w):
    xa = np.asarray(x, dtype=float)
    wa = np.asarray(w, dtype=float)
    M = xa.size
    p = poly1d(0.0)
    for j in range(M):
        pt = poly1d(wa[j])
        for k in range(M):
            if k == j:
                continue
            fac = xa[j] - xa[k]
            pt = pt * (poly1d(np.array([1.0, -xa[k]])) / fac)
        p = p + pt
    return p

def pade(an, m: int, n: Optional[int] = None):
    a = np.asarray(an, dtype=float)
    nn = 0
    if n is None:
        nn = a.size - 1 - m
        if nn < 0:
            raise ValueError("Order of q <m> must be smaller than len(an)-1.")
    else:
        nn = n
    if nn < 0:
        raise ValueError("Order of p <n> must be greater than 0.")
    N = m + nn
    if N > a.size - 1:
        raise ValueError("Order of q+p <m+n> must be smaller than len(an).")
    C = np.zeros((N + 1, N + 1))
    for i in range(min(N + 1, nn + 1)):
        C[i, i] = 1.0
    for row in range(1, m + 1):
        for j in range(row):
            C[row, nn + 1 + j] = -a[row - 1 - j]
    for row in range(m + 1, N + 1):
        for j in range(m):
            C[row, nn + 1 + j] = -a[row - 1 - j]
    pq = np.linalg.solve(C, a[:N + 1].copy())
    p = pq[:nn + 1]
    q = np.empty(m + 1)
    q[0] = 1.0
    for i in range(m):
        q[i + 1] = pq[nn + 1 + i]
    return poly1d(p[::-1].copy()), poly1d(q[::-1].copy())

def approximate_taylor_polynomial(f, x: float, degree: int, scale: float, order: Optional[int] = None):
    o = degree if order is None else order
    n = o + 1
    xs = scale * np.cos(np.linspace(0.0, np.pi, n, endpoint=(n % 1) != 0)) + x
    P = KroghInterpolator(xs, f(xs))
    d = P.derivatives(np.array([x]), der=degree + 1).ravel()
    co = np.empty(degree + 1)
    for i in range(degree + 1):
        co[degree - i] = d[i] / _fact(i)
    return poly1d(co)
