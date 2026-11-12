# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# interp1d, RegularGridInterpolator and interpn.

import numpy as np
import internal.static as static
from _common import NAN, _searchsorted, _finish, _vals
from _fitpack import make_interp_spline

def _is_extrap(fill_value) -> bool:
    if isinstance(fill_value, str):
        if fill_value == 'extrapolate':
            return True
        raise ValueError(f"fill_value={fill_value} is not supported")
    return False

def _nextafter(v: float, d: float) -> float:
    return float(np.nextafter(v, d))

class interp1d:
    def __init__(self, x, y, kind = 'linear', axis: int = -1, copy: bool = True,
                 bounds_error = None, fill_value = NAN, assume_sorted: bool = False):
        order = -1
        kname = ''
        if isinstance(kind, int):
            order = kind
            kname = 'spline'
        elif kind == 'zero' or kind == 'slinear' or kind == 'quadratic' or kind == 'cubic':
            order = {'zero': 0, 'slinear': 1, 'quadratic': 2, 'cubic': 3}[kind]
            kname = 'spline'
        elif kind in ('linear', 'nearest', 'nearest-up', 'previous', 'next'):
            kname = kind
        else:
            raise NotImplementedError(f"{kind} is unsupported: Use fitpack routines for other types.")
        xa = np.array(np.asarray(x, dtype=float))
        ya = np.asarray(y, dtype=float)
        if static.len(xa.shape) != 1:
            compile_error("the x array must have exactly one dimension.")
        if static.len(ya.shape) == 0:
            compile_error("the y array must have at least one dimension.")
        ax = axis % ya.ndim
        ym = np.moveaxis(ya, ax, 0)
        n = xa.size
        if ym.shape[0] != n:
            raise ValueError(f"x and y arrays must be equal in length along interpolation axis.")
        y2 = np.ascontiguousarray(ym).reshape((n, -1)).copy()
        R = y2.shape[1]
        if not assume_sorted:
            ind = np.argsort(xa, kind='mergesort')
            xs = np.empty(n)
            ys = np.empty((n, R))
            for i in range(n):
                xs[i] = xa[ind[i]]
                for j in range(R):
                    ys[i, j] = y2[ind[i], j]
            xa = xs
            y2 = ys
        self.x = xa
        self.y = ya
        self.axis = ax
        self.tsh = ym.shape[1:]
        self._y2 = y2
        self._kind = kname
        self._spl_t = np.empty(0)
        self._spl_c = np.empty((0, R))
        self._spl_k = 0
        minval = 1
        if kname == 'spline':
            minval = order + 1
            if n < minval:
                raise ValueError(f"x and y arrays must have at least {minval} entries")
            spl = make_interp_spline(xa, y2, k=order, check_finite=False)
            self._spl_t = spl.t
            self._spl_c = spl._c2
            self._spl_k = order
        if n < minval:
            raise ValueError(f"x and y arrays must have at least {minval} entries")
        self.bounds_error = False
        be_set = False
        if bounds_error is not None:
            self.bounds_error = bool(bounds_error)
            be_set = True
        self._below = [NAN] * R
        self._above = [NAN] * R
        self._extrapolate = False
        if _is_extrap(fill_value):
            if self.bounds_error:
                raise ValueError("Cannot extrapolate and raise at the same time.")
            self.bounds_error = False
            if kname == 'previous':
                self._above = [y2[n - 1, j] for j in range(R)]
            elif kname == 'next':
                self._below = [y2[0, j] for j in range(R)]
            else:
                self._extrapolate = True
        else:
            if isinstance(fill_value, Tuple):
                if static.len(fill_value) != 2:
                    compile_error("fill_value tuple must have two elements (below, above)")
                self._below = _vals(fill_value[0], R)
                self._above = _vals(fill_value[1], R)
            else:
                self._below = _vals(fill_value, R)
                self._above = _vals(fill_value, R)
            if not be_set:
                self.bounds_error = True

    def _eval1(self, xv: float, out, i: int):
        x = self.x
        n = x.size
        y2 = self._y2
        R = y2.shape[1]
        kind = self._kind
        if kind == 'linear':
            hi = _searchsorted(x, xv)
            hi = min(max(hi, 1), n - 1)
            lo = hi - 1
            xl = x[lo]
            xh = x[hi]
            for j in range(R):
                out[i, j] = ((xv - xl) / (xh - xl)) * y2[hi, j] + ((xh - xv) / (xh - xl)) * y2[lo, j]
        elif kind == 'nearest' or kind == 'nearest-up':
            # boundaries between neighbouring points
            right = kind == 'nearest-up'
            lo = 0
            hi = n - 1
            if xv != xv:
                idx = n - 1
            else:
                while lo < hi:
                    mid = (lo + hi) // 2
                    bd = x[mid] / 2.0 + x[mid + 1] / 2.0
                    if (bd <= xv) if right else (bd < xv):
                        lo = mid + 1
                    else:
                        hi = mid
                idx = lo
            for j in range(R):
                out[i, j] = y2[idx, j]
        elif kind == 'previous':
            lo = 0
            hi = n
            if xv != xv:
                lo = n
            else:
                while lo < hi:
                    mid = (lo + hi) // 2
                    if _nextafter(x[mid], -np.inf) < xv:
                        lo = mid + 1
                    else:
                        hi = mid
            idx = min(max(lo, 1), n)
            for j in range(R):
                out[i, j] = y2[idx - 1, j]
        elif kind == 'next':
            lo = 0
            hi = n
            if xv != xv:
                lo = n
            else:
                while lo < hi:
                    mid = (lo + hi) // 2
                    if _nextafter(x[mid], np.inf) <= xv:
                        lo = mid + 1
                    else:
                        hi = mid
            idx = min(max(lo, 0), n - 1)
            for j in range(R):
                out[i, j] = y2[idx, j]

    def __call__(self, x):
        xa = np.asarray(x, dtype=float)
        xs = np.ascontiguousarray(xa.reshape(-1))
        N = xs.size
        R = self._y2.shape[1]
        if self._kind == 'spline':
            from _bspl import _bspl_eval
            out = _bspl_eval(self._spl_t, self._spl_c, self._spl_k, xs, 0, True)
        else:
            out = np.empty((N, R))
            for i in range(N):
                self._eval1(xs[i], out, i)
        if not self._extrapolate:
            x0 = self.x[0]
            x1 = self.x[self.x.size - 1]
            if self.bounds_error:
                for i in range(N):
                    if xs[i] < x0:
                        raise ValueError(f"A value ({xs[i]}) in x_new is below the interpolation range's minimum value ({x0}).")
                for i in range(N):
                    if xs[i] > x1:
                        raise ValueError(f"A value ({xs[i]}) in x_new is above the interpolation range's maximum value ({x1}).")
            for i in range(N):
                if xs[i] < x0:
                    for j in range(R):
                        out[i, j] = self._below[j]
                elif xs[i] > x1:
                    for j in range(R):
                        out[i, j] = self._above[j]
        return _finish(out, xa.shape, self.tsh, self.axis)

# ---------------------------------------------------------------- regular grid

def _grid_list(points) -> Tuple[List[np.ndarray[float, 1]], List[bool]]:
    grid = List[np.ndarray[float, 1]]()
    desc = List[bool]()
    i = 0
    for p in points:
        pa = np.array(np.asarray(p, dtype=float).reshape(-1))
        asc = True
        dsc = True
        for q in range(pa.size - 1):
            if not (pa[q + 1] > pa[q]):
                asc = False
            if not (pa[q + 1] < pa[q]):
                dsc = False
        if not asc:
            if dsc:
                pa = np.array(pa[::-1])
            else:
                raise ValueError(f"The points in dimension {i} must be strictly ascending or descending")
        grid.append(pa)
        desc.append(not asc)
        i += 1
    return grid, desc

def _coords(xi, D: int):
    # returns (flat (N, D) float array, leading shape)
    if isinstance(xi, Tuple):
        arrs = np.broadcast_arrays(*tuple(np.asarray(a, dtype=float) for a in xi))
        st = np.stack(arrs, axis=-1)
        return np.ascontiguousarray(st).reshape((-1, D)), st.shape[:-1], st.shape[st.ndim - 1]
    else:
        xa = np.asarray(xi, dtype=float)
        if static.len(xa.shape) == 0:
            compile_error("xi must be at least 1-dimensional")
        elif static.len(xa.shape) == 1:
            # scipy reshapes a 1-D xi to (-1, ndim)
            return np.ascontiguousarray(xa).reshape((-1, D)), (xa.size // D,), D
        else:
            return np.ascontiguousarray(xa).reshape((-1, xa.shape[xa.ndim - 1])), xa.shape[:-1], xa.shape[xa.ndim - 1]

class RegularGridInterpolator:
    def __init__(self, points, values, method: str = "linear", bounds_error: bool = True,
                 fill_value = NAN):
        if method != "linear" and method != "nearest":
            if method in ("slinear", "cubic", "quintic", "pchip"):
                raise NotImplementedError(f"Method '{method}' is not supported")
            raise ValueError(f"Method '{method}' is not defined")
        self.method = method
        self.bounds_error = bounds_error
        grid, desc = _grid_list(points)
        va = np.asarray(values, dtype=float)
        D = len(grid)
        if D > va.ndim:
            raise ValueError(f"There are {D} point arrays, but values has {va.ndim} dimensions")
        vals = np.array(va)
        for i in range(D):
            if grid[i].size != va.shape[i]:
                raise ValueError(f"There are {grid[i].size} points and {va.shape[i]} values in dimension {i}")
            if desc[i]:
                vals = np.flip(vals, axis=i)
        G = 1
        for i in range(D):
            G *= va.shape[i]
        self.grid = grid
        self._v2 = np.ascontiguousarray(vals).reshape((G, -1)).copy()
        self.values = va
        if isinstance(points, Tuple):
            self._trail = va.shape[static.len(points):]
        else:
            # list of points: the grid must span all value dimensions
            self._trail = va.shape[static.len(va.shape):]
            if D != va.ndim:
                raise ValueError("pass `points` as a tuple to use trailing value dimensions")
        self._fill_none = fill_value is None
        self.fill_value = NAN if fill_value is None else float(fill_value)

    def __call__(self, xi, method: Optional[str] = None):
        meth = self.method if method is None else method
        if meth != "linear" and meth != "nearest":
            raise ValueError(f"Method '{meth}' is not defined")
        grid = self.grid
        D = len(grid)
        x2, lead, dd = _coords(xi, D)
        if dd != D:
            raise ValueError(f"The requested sample points xi have dimension {dd} but this RegularGridInterpolator has dimension {D}")
        N = x2.shape[0]
        R = self._v2.shape[1]
        strides = [1] * D
        for d in range(D - 2, -1, -1):
            strides[d] = strides[d + 1] * grid[d + 1].size
        out = np.zeros((N, R))
        idx = [0] * D
        nd = [0.0] * D
        for p in range(N):
            nanp = False
            oob = False
            for d in range(D):
                xv = x2[p, d]
                g = grid[d]
                if xv != xv:
                    nanp = True
                if xv < g[0] or xv > g[g.size - 1]:
                    oob = True
                    if self.bounds_error:
                        raise ValueError(f"One of the requested xi is out of bounds in dimension {d}")
                i = _searchsorted(g, xv) - 1
                if i < 0:
                    i = 0
                if i > g.size - 2:
                    i = g.size - 2
                idx[d] = i
                nd[d] = (xv - g[i]) / (g[i + 1] - g[i]) if g.size > 1 else 0.0
            if meth == "nearest":
                off = 0
                for d in range(D):
                    off += (idx[d] if nd[d] <= 0.5 else idx[d] + 1) * strides[d]
                for j in range(R):
                    out[p, j] = self._v2[off, j]
            else:
                for corner in range(1 << D):
                    w = 1.0
                    off = 0
                    for d in range(D):
                        if (corner >> (D - 1 - d)) & 1:
                            w *= nd[d]
                            off += (idx[d] + 1) * strides[d]
                        else:
                            w *= 1.0 - nd[d]
                            off += idx[d] * strides[d]
                    for j in range(R):
                        out[p, j] += self._v2[off, j] * w
            if oob and not self.bounds_error and not self._fill_none:
                for j in range(R):
                    out[p, j] = self.fill_value
            if nanp:
                for j in range(R):
                    out[p, j] = NAN
        return self._shape_out(out, lead, D)

    def _shape_out(self, out, lead, D: int):
        r = out.reshape(lead + self._trail)
        if static.len(r.shape) == 0:
            return r.item()
        else:
            return r

def interpn(points, values, xi, method: str = "linear", bounds_error: bool = True,
            fill_value = NAN):
    if method not in ("linear", "nearest", "cubic", "quintic", "pchip", "splinef2d", "slinear"):
        raise ValueError(f"interpn only understands the methods 'linear', 'nearest', 'slinear', 'cubic', 'quintic', 'pchip', and 'splinef2d'. You provided {method}.")
    interp = RegularGridInterpolator(points, values, method=method,
                                     bounds_error=bounds_error, fill_value=fill_value)
    return interp(xi)
