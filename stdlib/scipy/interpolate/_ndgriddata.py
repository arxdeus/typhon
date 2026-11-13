# Ported from scipy 1.18.1 (BSD-3-Clause): scipy/interpolate/_interpnd.pyx,
# _ndgriddata.py.  LinearNDInterpolator, CloughTocher2DInterpolator,
# NearestNDInterpolator, griddata, estimate_gradients_2d_global.
# Real-valued data only.

import numpy as np
import internal.static as static
from math import sqrt, fabs
from scipy.spatial._qhull import Delaunay
from scipy.spatial._kdtree import KDTree

_EPS = 2.220446049250313e-16

def _coords(points):
    # (n, ndim) float array from an array or a tuple of coordinate arrays
    if isinstance(points, Tuple):
        cols = [np.asarray(c, dtype=float) for c in points]
        shp = cols[0].shape
        out = np.empty(shp + (len(cols),))
        for j in range(len(cols)):
            out[..., j] = cols[j]
        return out
    else:
        a = np.asarray(points, dtype=float)
        if a.ndim == 1:
            return a.reshape((-1, 1))
        else:
            return a

def _prep(points, rescale: bool):
    C = _coords(points)
    P = np.ascontiguousarray(C).reshape((-1, C.shape[C.ndim - 1])).copy()
    if P.shape[1] < 2:
        raise ValueError("input data must be at least 2-D")
    D = P.shape[1]
    offset = np.zeros(D)
    scale = np.ones(D)
    if rescale:
        offset = np.mean(P, axis=0)
        for j in range(D):
            mn = P[0, j]
            mx = P[0, j]
            for i in range(P.shape[0]):
                mn = min(mn, P[i, j])
                mx = max(mx, P[i, j])
            scale[j] = mx - mn if mx - mn > 0 else 1.0
        P = (P - offset) / scale
    return P, offset, scale

def _values2(values, n: int):
    v = np.asarray(values, dtype=float)
    if v.shape[0] != n:
        raise ValueError("different number of values and points")
    return np.ascontiguousarray(v).reshape((n, -1)).copy(), v.shape[1:]

def _xi(args, D: int, offset, scale):
    if static.len(args) == 1:
        X = _coords(args[0])
    else:
        X = _coords(args)
    if X.shape[X.ndim - 1] != D:
        raise ValueError("number of dimensions in xi does not match x")
    shp = X.shape
    X2 = np.ascontiguousarray(X).reshape((-1, D))
    X2 = (X2 - offset) / scale
    return X2, shp

class LinearNDInterpolator:
    def __init__(self, points, values, fill_value: float = np.nan, rescale: bool = False):
        P, off, sc = _prep(points, rescale)
        self.points = P
        self.offset = off
        self.scale = sc
        self.tri = Delaunay(P)
        self.values, self.values_shape = _values2(values, P.shape[0])
        self.fill_value = fill_value

    def __call__(self, *args, simplex_tolerance: float = 1.0):
        if simplex_tolerance <= 0:
            raise ValueError("simplex_tolerance must be positive")
        D = self.points.shape[1]
        X, shp = _xi(args, D, self.offset, self.scale)
        tri = self.tri
        T = tri.transform
        nv = self.values.shape[1]
        out = np.empty((X.shape[0], nv))
        eps = 100 * _EPS * simplex_tolerance
        eps_broad = sqrt(_EPS)
        c = np.zeros(D + 1)
        start = 0
        for i in range(X.shape[0]):
            s, start = tri._find(T, c, X[i], start, eps, eps_broad)
            if s == -1:
                for k in range(nv):
                    out[i, k] = self.fill_value
                continue
            for k in range(nv):
                out[i, k] = 0.0
            for j in range(D + 1):
                m = tri.simplices[s, j]
                for k in range(nv):
                    out[i, k] = out[i, k] + c[j] * self.values[m, k]
        return out.reshape(shp[:-1] + self.values_shape)

def _estimate_gradients_one(P, indptr, indices, data, maxiter: int, tol: float, y):
    n = P.shape[0]
    for i in range(2 * n):
        y[i] = 0.0
    for it in range(maxiter):
        err = 0.0
        for ip in range(n):
            Q0 = 0.0
            Q1 = 0.0
            Q3 = 0.0
            s0 = 0.0
            s1 = 0.0
            for jp in range(indptr[ip], indptr[ip + 1]):
                ip2 = indices[jp]
                ex = P[ip2, 0] - P[ip, 0]
                ey = P[ip2, 1] - P[ip, 1]
                L = sqrt(ex ** 2 + ey ** 2)
                L3 = L * L * L
                f1 = data[ip]
                f2 = data[ip2]
                df2 = -ex * y[2 * ip2] - ey * y[2 * ip2 + 1]
                Q0 += 4 * ex * ex / L3
                Q1 += 4 * ex * ey / L3
                Q3 += 4 * ey * ey / L3
                s0 += (6 * (f1 - f2) - 2 * df2) * ex / L3
                s1 += (6 * (f1 - f2) - 2 * df2) * ey / L3
            Q2 = Q1
            det = Q0 * Q3 - Q1 * Q2
            r0 = (Q3 * s0 - Q1 * s1) / det
            r1 = (-Q2 * s0 + Q0 * s1) / det
            change = max(fabs(y[2 * ip] + r0), fabs(y[2 * ip + 1] + r1))
            y[2 * ip] = -r0
            y[2 * ip + 1] = -r1
            change /= max(1.0, max(fabs(r0), fabs(r1)))
            err = max(err, change)
        if err < tol:
            return it + 1
    return 0

def estimate_gradients_2d_global(tri, y, maxiter: int = 400, tol: float = 1e-6):
    ya = np.asarray(y, dtype=float)
    n = tri.npoints
    if ya.shape[0] != n:
        raise ValueError("'y' has a wrong number of items")
    Y = np.ascontiguousarray(ya).reshape((n, -1))
    nv = Y.shape[1]
    indptr, indices = tri.vertex_neighbor_vertices
    G = np.empty((n, nv, 2))
    g = np.zeros(2 * n)
    for k in range(nv):
        col = Y[:, k].copy()
        _estimate_gradients_one(tri.points, indptr, indices, col, maxiter, tol, g)
        for i in range(n):
            G[i, k, 0] = g[2 * i]
            G[i, k, 1] = g[2 * i + 1]
    return G.reshape(ya.shape + (2,))

def _ct_single(P, S, N, T, s: int, b, f, df) -> float:
    s0 = S[s, 0]
    s1 = S[s, 1]
    s2 = S[s, 2]
    e12x = P[s1, 0] - P[s0, 0]
    e12y = P[s1, 1] - P[s0, 1]
    e23x = P[s2, 0] - P[s1, 0]
    e23y = P[s2, 1] - P[s1, 1]
    e31x = P[s0, 0] - P[s2, 0]
    e31y = P[s0, 1] - P[s2, 1]
    f1 = f[0]
    f2 = f[1]
    f3 = f[2]
    df12 = +(df[0] * e12x + df[1] * e12y)
    df21 = -(df[2] * e12x + df[3] * e12y)
    df23 = +(df[2] * e23x + df[3] * e23y)
    df32 = -(df[4] * e23x + df[5] * e23y)
    df31 = +(df[4] * e31x + df[5] * e31y)
    df13 = -(df[0] * e31x + df[1] * e31y)
    c3000 = f1
    c2100 = (df12 + 3 * c3000) / 3
    c2010 = (df13 + 3 * c3000) / 3
    c0300 = f2
    c1200 = (df21 + 3 * c0300) / 3
    c0210 = (df23 + 3 * c0300) / 3
    c0030 = f3
    c1020 = (df31 + 3 * c0030) / 3
    c0120 = (df32 + 3 * c0030) / 3
    c2001 = (c2100 + c2010 + c3000) / 3
    c0201 = (c1200 + c0300 + c0210) / 3
    c0021 = (c1020 + c0120 + c0030) / 3
    g = [0.0, 0.0, 0.0]
    c = [0.0, 0.0, 0.0]
    for k in range(3):
        itri = N[s, k]
        if itri == -1:
            g[k] = -1.0 / 2
            continue
        y0 = (P[S[itri, 0], 0] + P[S[itri, 1], 0] + P[S[itri, 2], 0]) / 3
        y1 = (P[S[itri, 0], 1] + P[S[itri, 1], 1] + P[S[itri, 2], 1]) / 3
        c[2] = 1.0
        for i in range(2):
            c[i] = T[s, i, 0] * (y0 - T[s, 2, 0]) + T[s, i, 1] * (y1 - T[s, 2, 1])
            c[2] -= c[i]
        if k == 0:
            g[k] = (2 * c[2] + c[1] - 1) / (2 - 3 * c[2] - 3 * c[1])
        elif k == 1:
            g[k] = (2 * c[0] + c[2] - 1) / (2 - 3 * c[0] - 3 * c[2])
        else:
            g[k] = (2 * c[1] + c[0] - 1) / (2 - 3 * c[1] - 3 * c[0])
    c0111 = (g[0] * (-c0300 + 3 * c0210 - 3 * c0120 + c0030) + (-c0300 + 2 * c0210 - c0120 + c0021 + c0201)) / 2
    c1011 = (g[1] * (-c0030 + 3 * c1020 - 3 * c2010 + c3000) + (-c0030 + 2 * c1020 - c2010 + c2001 + c0021)) / 2
    c1101 = (g[2] * (-c3000 + 3 * c2100 - 3 * c1200 + c0300) + (-c3000 + 2 * c2100 - c1200 + c2001 + c0201)) / 2
    c1002 = (c1101 + c1011 + c2001) / 3
    c0102 = (c1101 + c0111 + c0201) / 3
    c0012 = (c1011 + c0111 + c0021) / 3
    c0003 = (c1002 + c0102 + c0012) / 3
    minval = b[0]
    for k in range(3):
        if b[k] < minval:
            minval = b[k]
    b1 = b[0] - minval
    b2 = b[1] - minval
    b3 = b[2] - minval
    b4 = 3 * minval
    return (b1 ** 3 * c3000 + 3 * b1 ** 2 * b2 * c2100 + 3 * b1 ** 2 * b3 * c2010 +
            3 * b1 ** 2 * b4 * c2001 + 3 * b1 * b2 ** 2 * c1200 +
            6 * b1 * b2 * b4 * c1101 + 3 * b1 * b3 ** 2 * c1020 + 6 * b1 * b3 * b4 * c1011 +
            3 * b1 * b4 ** 2 * c1002 + b2 ** 3 * c0300 + 3 * b2 ** 2 * b3 * c0210 +
            3 * b2 ** 2 * b4 * c0201 + 3 * b2 * b3 ** 2 * c0120 + 6 * b2 * b3 * b4 * c0111 +
            3 * b2 * b4 ** 2 * c0102 + b3 ** 3 * c0030 + 3 * b3 ** 2 * b4 * c0021 +
            3 * b3 * b4 ** 2 * c0012 + b4 ** 3 * c0003)

class CloughTocher2DInterpolator:
    def __init__(self, points, values, fill_value: float = np.nan, tol: float = 1e-6,
                 maxiter: int = 400, rescale: bool = False):
        P, off, sc = _prep(points, rescale)
        if P.shape[1] != 2:
            raise ValueError("this mode of interpolation available only for 2-D data")
        self.points = P
        self.offset = off
        self.scale = sc
        self.tri = Delaunay(P)
        self.values, self.values_shape = _values2(values, P.shape[0])
        self.fill_value = fill_value
        self.grad = estimate_gradients_2d_global(self.tri, self.values, maxiter, tol)

    def __call__(self, *args, simplex_tolerance: float = 1.0):
        if simplex_tolerance <= 0:
            raise ValueError("simplex_tolerance must be positive")
        X, shp = _xi(args, 2, self.offset, self.scale)
        tri = self.tri
        T = tri.transform
        nv = self.values.shape[1]
        out = np.zeros((X.shape[0], nv))
        eps = 100 * _EPS * simplex_tolerance
        eps_broad = sqrt(eps)
        c = np.zeros(3)
        f = [0.0, 0.0, 0.0]
        df = [0.0] * 6
        start = 0
        for i in range(X.shape[0]):
            s, start = tri._find(T, c, X[i], start, eps, eps_broad)
            if s == -1:
                for k in range(nv):
                    out[i, k] = self.fill_value
                continue
            for k in range(nv):
                for j in range(3):
                    v = tri.simplices[s, j]
                    f[j] = self.values[v, k]
                    df[2 * j] = self.grad[v, k, 0]
                    df[2 * j + 1] = self.grad[v, k, 1]
                out[i, k] = _ct_single(tri.points, tri.simplices, tri.neighbors, T, s, c, f, df)
        return out.reshape(shp[:-1] + self.values_shape)

class NearestNDInterpolator:
    def __init__(self, x, y, rescale: bool = False, tree_options = None):
        P, off, sc = _prep(x, rescale)
        self.points = P
        self.offset = off
        self.scale = sc
        self.tree = KDTree(P)
        self.values = np.asarray(y)

    def __call__(self, *args):
        D = self.points.shape[1]
        X, shp = _xi(args, D, self.offset, self.scale)
        idx = np.empty(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            best = self.tree._knn(X[i], 1, 2.0, np.inf)
            idx[i] = best[0][1]
        r = self.values[idx]
        return r.reshape(shp[:-1] + self.values.shape[1:])

def griddata(points, values, xi, method: str = 'linear', fill_value: float = np.nan,
             rescale: bool = False, simplex_tolerance: float = 1.0):
    if simplex_tolerance <= 0:
        raise ValueError("simplex_tolerance must be positive")
    if method == 'nearest':
        ip = NearestNDInterpolator(points, values, rescale=rescale)
        return ip(xi)
    elif method == 'linear':
        ip = LinearNDInterpolator(points, values, fill_value=fill_value, rescale=rescale)
        return ip(xi, simplex_tolerance=simplex_tolerance)
    elif method == 'cubic':
        ip = CloughTocher2DInterpolator(points, values, fill_value=fill_value, rescale=rescale)
        return ip(xi, simplex_tolerance=simplex_tolerance)
    else:
        raise ValueError(f"Unknown interpolation method {method!r}")
