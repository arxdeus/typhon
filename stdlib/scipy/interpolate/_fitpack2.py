# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# FITPACK object interface: UnivariateSpline, InterpolatedUnivariateSpline,
# LSQUnivariateSpline (scipy/interpolate/_fitpack2.py).  Instances do not
# change their __class__ after fitting (scipy reassigns it according to ier).

import numpy as np
import _dierckx as D
from _fitpack import splev, splint, splder, splantider, spalde, sproot

def _ext_code(ext) -> int:
    if isinstance(ext, str):
        if ext == 'extrapolate':
            return 0
        if ext == 'zeros':
            return 1
        if ext == 'raise':
            return 2
        if ext == 'const':
            return 3
        raise ValueError(f"Unknown extrapolation mode {ext}.")
    else:
        e = int(ext)
        if e < 0 or e > 3:
            raise ValueError(f"Unknown extrapolation mode {ext}.")
        return e

def _validate(x, y, w, k: int, s: Optional[float], positive_s: bool):
    if positive_s:
        for i in range(x.size - 1):
            if not (x[i + 1] - x[i] >= 0.0):
                raise ValueError("x must be increasing if s > 0")
    else:
        for i in range(x.size - 1):
            if not (x[i + 1] - x[i] > 0.0):
                raise ValueError("x must be strictly increasing if s = 0")
    if x.size != y.size:
        raise ValueError("x and y should have a same length")
    if w.size != x.size:
        raise ValueError("x, y, and w should have a same length")
    if not (1 <= k and k <= 5):
        raise ValueError("k should be 1 <= k <= 5")
    if s is not None:
        if not (s >= 0.0):
            raise ValueError("s should be s >= 0.0")

class UnivariateSpline:
    _x: np.ndarray[float, 1]
    _y: np.ndarray[float, 1]
    _w: np.ndarray[float, 1]
    _xb: float
    _xe: float
    _k: int
    _s: float
    _n: int
    _t: np.ndarray[float, 1]
    _c: np.ndarray[float, 1]
    _fp: float
    _ier: int
    _nest: int
    ext: int

    def __init__(self, x, y, w = None, bbox = (None, None), k: int = 3, s = None,
                 ext = 0, check_finite: bool = False):
        xa = np.ascontiguousarray(np.asarray(x, dtype=float).ravel())
        ya = np.ascontiguousarray(np.asarray(y, dtype=float).ravel())
        wa = np.ones(xa.size) if w is None else np.ascontiguousarray(np.asarray(w, dtype=float).ravel())
        sv: Optional[float] = None if s is None else float(s)
        pos = True
        if sv is not None:
            pos = sv > 0.0
        if check_finite:
            for i in range(xa.size):
                if not (np.isfinite(xa[i]) and np.isfinite(ya[i])):
                    raise ValueError("x and y array must not contain NaNs or infs.")
        _validate(xa, ya, wa, k, sv, pos)
        self.ext = _ext_code(ext)
        self._setup(xa, ya, wa, bbox, k)
        self._s = float(xa.size) if sv is None else sv
        self._nest = xa.size + k + 1 if self._s == 0.0 else max(xa.size // 2, 2 * (k + 1))
        self._fit_smooth()

    def _setup(self, xa, ya, wa, bbox, k: int):
        self._x = xa
        self._y = ya
        self._w = wa
        self._k = k
        self._xb = xa[0] if bbox[0] is None else float(bbox[0])
        self._xe = xa[xa.size - 1] if bbox[1] is None else float(bbox[1])

    def _store(self, wk: D.CurfitWork):
        n = wk.n
        k = self._k
        self._n = n
        self._t = wk.t[:n].copy()
        self._c = np.zeros(n)
        for i in range(n - k - 1):
            self._c[i] = wk.c[i]
        self._fp = wk.fp
        self._ier = wk.ier
        if wk.ier == 10:
            raise ValueError("Error on input data")

    def _fit_smooth(self):
        m = self._x.size
        k = self._k
        nest = self._nest
        wk = D.CurfitWork(nest)
        wk.n = nest
        D.curfit(0, self._x, self._y, self._w, self._xb, self._xe, k, self._s, wk)
        if wk.ier == 1:
            # _reset_nest: refit with nest = m + k + 1 (fresh FITPACK work arrays)
            wk2 = D.CurfitWork(m + k + 1)
            for i in range(wk.n):
                wk2.t[i] = wk.t[i]
            wk2.n = m + k + 1
            D.curfit(1, self._x, self._y, self._w, self._xb, self._xe, k, self._s, wk2)
            wk = wk2
            self._nest = m + k + 1
        self._store(wk)

    @staticmethod
    def _from_tck(t, c, k: int, ext: int = 0):
        self = UnivariateSpline.__new__()
        n = t.size
        self._t = np.asarray(t, dtype=float).copy()
        self._c = np.zeros(n)
        for i in range(min(n, c.size)):
            self._c[i] = c[i]
        self._k = k
        self._n = n
        self._s = -1.0
        self._fp = 0.0
        self._ier = 0
        self._nest = n
        self._x = np.zeros(0)
        self._y = np.zeros(0)
        self._w = np.zeros(0)
        self._xb = 0.0
        self._xe = 0.0
        self.ext = ext
        return self

    @property
    def _eval_args(self):
        return (self._t, self._c, self._k)

    def set_smoothing_factor(self, s: float):
        if self._x.size == 0 or self._s == -1.0:
            return
        self._s = float(s)
        self._fit_smooth()

    def __call__(self, x, nu: int = 0, ext = None):
        e = self.ext if ext is None else _ext_code(ext)
        return splev(x, self._eval_args, der=nu, ext=e)

    def get_knots(self):
        return self._t[self._k:self._n - self._k].copy()

    def get_coeffs(self):
        return self._c[:self._n - self._k - 1].copy()

    def get_residual(self) -> float:
        return self._fp

    def integral(self, a: float, b: float) -> float:
        return splint(a, b, self._eval_args)

    def derivatives(self, x: float):
        return spalde(float(x), self._eval_args)

    def roots(self):
        if self._k == 3:
            mest = 3 * (self._t.size - 7)
            return sproot(self._eval_args, mest=mest)
        raise NotImplementedError('finding roots unsupported for non-cubic splines')

    def derivative(self, n: int = 1):
        t, c, k = splder(self._eval_args, n)
        e = 1 if self.ext == 3 else self.ext
        return UnivariateSpline._from_tck(t, c, k, e)

    def antiderivative(self, n: int = 1):
        t, c, k = splantider(self._eval_args, n)
        return UnivariateSpline._from_tck(t, c, k, self.ext)

class InterpolatedUnivariateSpline(UnivariateSpline):
    def __init__(self, x, y, w = None, bbox = (None, None), k: int = 3, ext = 0,
                 check_finite: bool = False):
        xa = np.ascontiguousarray(np.asarray(x, dtype=float).ravel())
        ya = np.ascontiguousarray(np.asarray(y, dtype=float).ravel())
        wa = np.ones(xa.size) if w is None else np.ascontiguousarray(np.asarray(w, dtype=float).ravel())
        _validate(xa, ya, wa, k, None, False)
        self.ext = _ext_code(ext)
        self._setup(xa, ya, wa, bbox, k)
        self._s = 0.0
        self._nest = xa.size + k + 1
        self._fit_smooth()

class LSQUnivariateSpline(UnivariateSpline):
    def __init__(self, x, y, t, w = None, bbox = (None, None), k: int = 3, ext = 0,
                 check_finite: bool = False):
        xa = np.ascontiguousarray(np.asarray(x, dtype=float).ravel())
        ya = np.ascontiguousarray(np.asarray(y, dtype=float).ravel())
        wa = np.ones(xa.size) if w is None else np.ascontiguousarray(np.asarray(w, dtype=float).ravel())
        _validate(xa, ya, wa, k, None, True)
        self.ext = _ext_code(ext)
        self._setup(xa, ya, wa, bbox, k)
        ti = np.asarray(t, dtype=float).ravel()
        n = ti.size + 2 * (k + 1)
        wk = D.CurfitWork(n)
        for i in range(k + 1):
            wk.t[i] = self._xb
            wk.t[n - 1 - i] = self._xe
        for i in range(ti.size):
            wk.t[k + 1 + i] = ti[i]
        for i in range(k, n - k - 1):
            if not (wk.t[i + 1] - wk.t[i] > 0):
                raise ValueError('Interior knots t must satisfy Schoenberg-Whitney conditions')
        if D.fpchec(xa, xa.size, wk.t, n, k) != 0:
            raise ValueError("Interior knots t must satisfy Schoenberg-Whitney conditions")
        wk.n = n
        self._nest = n
        self._s = -1.0
        D.curfit(-1, xa, ya, wa, self._xb, self._xe, k, float(xa.size), wk)
        self._store(wk)
