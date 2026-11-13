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
