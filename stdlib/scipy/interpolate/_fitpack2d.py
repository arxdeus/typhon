# Ported from scipy 1.18.1 (BSD-3-Clause): RectBivariateSpline and the
# bivariate evaluation interface (scipy/interpolate/_fitpack2.py), bisplev.

import numpy as np
import _dierckx2 as D2

class RectBivariateSpline:
    def __init__(self, x, y, z, bbox = (None, None, None, None), kx: int = 3, ky: int = 3,
                 s: float = 0.0, maxit: int = 20):
        xa = np.ascontiguousarray(np.asarray(x, dtype=float).ravel())
        ya = np.ascontiguousarray(np.asarray(y, dtype=float).ravel())
        za = np.asarray(z, dtype=float)
        for i in range(xa.size - 1):
            if not (xa[i + 1] - xa[i] > 0.0):
                raise ValueError("x must be strictly increasing")
        for i in range(ya.size - 1):
            if not (ya[i + 1] - ya[i] > 0.0):
                raise ValueError("y must be strictly increasing")
        if xa.size != za.shape[0]:
            raise ValueError("x dimension of z must have same number of elements as x")
        if ya.size != za.shape[1]:
            raise ValueError("y dimension of z must have same number of elements as y")
        if not (s >= 0.0):
            raise ValueError("s should be s >= 0.0")
        xb = float(np.min(xa)) if bbox[0] is None else float(bbox[0])
        xe = float(np.max(xa)) if bbox[1] is None else float(bbox[1])
        yb = float(np.min(ya)) if bbox[2] is None else float(bbox[2])
        ye = float(np.max(ya)) if bbox[3] is None else float(bbox[3])
        zf = np.ascontiguousarray(za).ravel().copy()
        tx, ty, c, fp, ier = D2.regrid_smth(xa, ya, zf, xb, xe, yb, ye, kx, ky, s, maxit)
        if ier != 0 and ier != -1 and ier != -2:
            raise ValueError(f"ier={ier}: error in RectBivariateSpline fit")
        self.fp = fp
        self.tx = tx
        self.ty = ty
        self.c = c
        self.kx = kx
        self.ky = ky

    @property
    def tck(self):
        return (self.tx, self.ty, self.c)

    @property
    def degrees(self):
        return (self.kx, self.ky)

    def get_residual(self) -> float:
        return self.fp

    def get_knots(self):
        return (self.tx, self.ty)

    def get_coeffs(self):
        return self.c

    def _eval_grid(self, x, y, dx: int, dy: int):
        mx = x.size
        my = y.size
        z = np.empty(mx * my)
        if dx == 0 and dy == 0:
            D2.fpbisp(self.tx, self.tx.size, self.ty, self.ty.size, self.c, self.kx, self.ky, x, mx, y, my, z)
        else:
            if not (0 <= dx and dx < self.kx and 0 <= dy and dy < self.ky):
                raise ValueError("Error code returned by parder: 10")
            nc, nxx, nyy = D2.pardtc(self.tx, self.tx.size, self.ty, self.ty.size, self.c, self.kx, self.ky, dx, dy)
            txs = self.tx[dx:self.tx.size - dx].copy()
            tys = self.ty[dy:self.ty.size - dy].copy()
            D2.fpbisp(txs, txs.size, tys, tys.size, nc, self.kx - dx, self.ky - dy, x, mx, y, my, z)
        return z.reshape((mx, my))

    def __call__(self, x, y, dx: int = 0, dy: int = 0, grid: Literal[bool] = True):
        xa = np.atleast_1d(np.asarray(x, dtype=float))
        ya = np.atleast_1d(np.asarray(y, dtype=float))
        if grid:
            xr = np.ascontiguousarray(xa.ravel())
            yr = np.ascontiguousarray(ya.ravel())
            for i in range(xr.size - 1):
                if not (xr[i + 1] - xr[i] >= 0.0):
                    raise ValueError("x must be strictly increasing when `grid` is True")
            for i in range(yr.size - 1):
                if not (yr[i + 1] - yr[i] >= 0.0):
                    raise ValueError("y must be strictly increasing when `grid` is True")
            return self._eval_grid(xr, yr, dx, dy)
        else:
            xb, yb = np.broadcast_arrays(xa, ya)
            xr = np.ascontiguousarray(xb).ravel()
            yr = np.ascontiguousarray(yb).ravel()
            out = np.empty(xr.size)
            for i in range(xr.size):
                out[i] = self._eval_grid(xr[i:i + 1].copy(), yr[i:i + 1].copy(), dx, dy)[0, 0]
            return out.reshape(xb.shape)

    def ev(self, xi, yi, dx: int = 0, dy: int = 0):
        return self(xi, yi, dx=dx, dy=dy, grid=False)

    def integral(self, xa: float, xb: float, ya: float, yb: float) -> float:
        return D2.dblint(self.tx, self.tx.size, self.ty, self.ty.size, self.c, self.kx, self.ky, xa, xb, ya, yb)

    def partial_derivative(self, dx: int, dy: int):
        if not (dx >= 0 and dy >= 0):
            raise ValueError("order of derivative must be positive or zero")
        if not (dx < self.kx and dy < self.ky):
            raise ValueError("order of derivative must be less than degree of spline")
        nc, nxx, nyy = D2.pardtc(self.tx, self.tx.size, self.ty, self.ty.size, self.c, self.kx, self.ky, dx, dy)
        out = RectBivariateSpline.__new__()
        out.tx = self.tx[dx:self.tx.size - dx].copy()
        out.ty = self.ty[dy:self.ty.size - dy].copy()
        nclen = (self.tx.size - dx - self.kx - 1) * (self.ty.size - dy - self.ky - 1)
        out.c = nc[:nclen].copy()
        out.kx = self.kx - dx
        out.ky = self.ky - dy
        out.fp = 0.0
        return out

def bisplev(x, y, tck, dx: int = 0, dy: int = 0):
    tx = np.ascontiguousarray(np.asarray(tck[0], dtype=float))
    ty = np.ascontiguousarray(np.asarray(tck[1], dtype=float))
    c = np.ascontiguousarray(np.asarray(tck[2], dtype=float))
    kx = int(tck[3])
    ky = int(tck[4])
    xa = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    ya = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    s = RectBivariateSpline.__new__()
    s.tx = tx
    s.ty = ty
    s.c = c
    s.kx = kx
    s.ky = ky
    s.fp = 0.0
    z = s._eval_grid(np.ascontiguousarray(xa), np.ascontiguousarray(ya), dx, dy)
    if xa.size == 1 and ya.size == 1:
        return z.reshape((1, 1))
    else:
        return z

import _surfit as SF
from math import sqrt as _sqrt

def _from_raw(tx, ty, c, kx: int, ky: int, fp: float):
    s = RectBivariateSpline.__new__()
    s.tx = tx
    s.ty = ty
    s.c = c
    s.kx = kx
    s.ky = ky
    s.fp = fp
    return s

def _xyzw(x, y, z, w, kx: int, ky: int):
    xa = np.ascontiguousarray(np.asarray(x, dtype=float).ravel())
    ya = np.ascontiguousarray(np.asarray(y, dtype=float).ravel())
    za = np.ascontiguousarray(np.asarray(z, dtype=float).ravel())
    if not (xa.size == ya.size and ya.size == za.size):
        raise ValueError("x, y, and z should have a same length")
    wa = np.ones(xa.size) if w is None else np.ascontiguousarray(np.asarray(w, dtype=float).ravel())
    if wa.size != xa.size:
        raise ValueError("x, y, z, and w should have a same length")
    if not xa.size >= (kx + 1) * (ky + 1):
        raise ValueError("The length of x, y and z should be at least (kx+1) * (ky+1)")
    return xa, ya, za, wa

def SmoothBivariateSpline(x, y, z, w = None, bbox = (None, None, None, None), kx: int = 3, ky: int = 3,
                          s: Optional[float] = None, eps: float = 1e-16):
    xa, ya, za, wa = _xyzw(x, y, z, w, kx, ky)
    m = xa.size
    xb = float(np.min(xa)) if bbox[0] is None else float(bbox[0])
    xe = float(np.max(xa)) if bbox[1] is None else float(bbox[1])
    yb = float(np.min(ya)) if bbox[2] is None else float(bbox[2])
    ye = float(np.max(ya)) if bbox[3] is None else float(bbox[3])
    nxest = max(kx + 1 + int(_sqrt(m / 2)), 2 * (kx + 1))
    nyest = max(ky + 1 + int(_sqrt(m / 2)), 2 * (ky + 1))
    nmax = max(nxest, nyest)
    sv = float(m) if s is None else s
    tx = np.zeros(nmax)
    ty = np.zeros(nmax)
    nx, ny, c, fp, ier = SF.surfit(0, xa, ya, za, wa, xb, xe, yb, ye, kx, ky, sv, nxest, nyest, eps, tx, ty, nxest, nyest)
    if ier == 10:
        raise ValueError("Invalid input data")
    return _from_raw(tx[:nx].copy(), ty[:ny].copy(), c[:(nx - kx - 1) * (ny - ky - 1)].copy(), kx, ky, fp)

def _calc_b(data, knots, n: int) -> float:
    v1 = float(np.min(data))
    v2 = float(np.min(knots))
    if v2 > v1:
        return v1
    return v2 - (float(np.max(knots)) - v2) / n

def _calc_e(data, knots, n: int) -> float:
    v1 = float(np.max(data))
    v2 = float(np.max(knots))
    if v2 < v1:
        return v1
    return v2 + (v2 - float(np.min(knots))) / n

def LSQBivariateSpline(x, y, z, tx, ty, w = None, bbox = (None, None, None, None), kx: int = 3,
                       ky: int = 3, eps: Optional[float] = None):
    xa, ya, za, wa = _xyzw(x, y, z, w, kx, ky)
    txa = np.asarray(tx, dtype=float).ravel()
    tya = np.asarray(ty, dtype=float).ravel()
    nx = 2 * kx + 2 + txa.size
    ny = 2 * ky + 2 + tya.size
    nmax = max(nx, ny)
    tx1 = np.zeros(nmax)
    ty1 = np.zeros(nmax)
    for i in range(txa.size):
        tx1[kx + 1 + i] = txa[i]
    for i in range(tya.size):
        ty1[ky + 1 + i] = tya[i]
    xb = _calc_b(xa, tx1, nmax) if bbox[0] is None else float(bbox[0])
    xe = _calc_e(xa, tx1, nmax) if bbox[1] is None else float(bbox[1])
    yb = _calc_b(ya, ty1, nmax) if bbox[2] is None else float(bbox[2])
    ye = _calc_e(ya, ty1, nmax) if bbox[3] is None else float(bbox[3])
    e = 1e-16 if eps is None else eps
    nx2, ny2, c, fp, ier = SF.surfit(-1, xa, ya, za, wa, xb, xe, yb, ye, kx, ky, 0.0, nx, ny, e, tx1, ty1, nx, ny)
    if ier == 10:
        raise ValueError("Invalid input data")
    return _from_raw(tx1[:nx].copy(), ty1[:ny].copy(), c[:(nx - kx - 1) * (ny - ky - 1)].copy(), kx, ky, fp)

_SURF_CACHE_TX = [np.zeros(0)]
_SURF_CACHE_TY = [np.zeros(0)]

def bisplrep(x, y, z, w = None, xb = None, xe = None, yb = None, ye = None, kx: int = 3, ky: int = 3,
             task: int = 0, s: Optional[float] = None, eps: float = 1e-16, tx = None, ty = None,
             full_output: Literal[int] = 0, nxest: Optional[int] = None, nyest: Optional[int] = None,
             quiet: int = 1):
    xa, ya, za, wa = _xyzw(x, y, z, w, kx, ky)
    m = xa.size
    xbv = float(np.min(xa)) if xb is None else float(xb)
    xev = float(np.max(xa)) if xe is None else float(xe)
    ybv = float(np.min(ya)) if yb is None else float(yb)
    yev = float(np.max(ya)) if ye is None else float(ye)
    if task < -1 or task > 1:
        raise TypeError("task must be -1, 0 or 1")
    sv = float(m) - _sqrt(2.0 * m) if s is None else s
    nxe = int(kx + _sqrt(m / 2)) if nxest is None else nxest
    nye = int(ky + _sqrt(m / 2)) if nyest is None else nyest
    nxe = max(nxe, 2 * kx + 3)
    nye = max(nye, 2 * ky + 3)
    if task >= 0 and sv == 0:
        nxe = int(kx + _sqrt(3.0 * m))
        nye = int(ky + _sqrt(3.0 * m))
    nx0 = nxe
    ny0 = nye
    txa = np.zeros(0)
    tya = np.zeros(0)
    if tx is not None:
        txa = np.asarray(tx, dtype=float).ravel()
    if ty is not None:
        tya = np.asarray(ty, dtype=float).ravel()
    if task == 1:
        # continue from the knots of the previous call
        txa = _SURF_CACHE_TX[0]
        tya = _SURF_CACHE_TY[0]
        if txa.size == 0 or tya.size == 0:
            raise TypeError("task=1 requires a previous bisplrep call")
        nx0 = txa.size
        ny0 = tya.size
        nxe = max(nxe, nx0)
        nye = max(nye, ny0)
    if task == -1:
        if txa.size == 0 or tya.size == 0:
            raise TypeError("Knots must be given for task=-1")
        nx0 = txa.size
        ny0 = tya.size
        nxe = max(nxe, nx0)
        nye = max(nye, ny0)
    nmax = max(nxe, nye)
    T1 = np.zeros(nmax)
    T2 = np.zeros(nmax)
    for i in range(txa.size):
        T1[i] = txa[i]
    for i in range(tya.size):
        T2[i] = tya[i]
    nx, ny, c, fp, ier = SF.surfit(task, xa, ya, za, wa, xbv, xev, ybv, yev, kx, ky, sv, nxe, nye, eps, T1, T2, nx0, ny0)
    if ier == 10:
        raise ValueError("Invalid inputs.")
    ncoef = (nx - kx - 1) * (ny - ky - 1)
    _SURF_CACHE_TX[0] = T1[:nx].copy()
    _SURF_CACHE_TY[0] = T2[:ny].copy()
    tck = (T1[:nx].copy(), T2[:ny].copy(), c[:ncoef].copy(), kx, ky)
    if full_output:
        return tck, fp, ier
    else:
        return tck
