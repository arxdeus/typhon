# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# make_lsq_spline (QR via Givens rotations as scipy's _dierckx.qr_reduce)
# and make_smoothing_spline (GCV criterion, banded solves).

import numpy as np
from math import sqrt, fabs
from _common import _bspl_find, _deboor
from _bspl import BSpline
from scipy.linalg import solve_banded, cholesky_banded
from scipy.optimize import minimize_scalar
from _fitpack import _not_a_knot, make_interp_spline
import _dierckx as D

def _dlartg(f: float, g: float):
    # LAPACK dlartg (3.10+ algorithm): returns (c, s, r)
    safmin = 2.2250738585072014e-308
    safmax = 1.0 / safmin
    rtmin = sqrt(safmin)
    rtmax = sqrt(safmax / 2)
    if g == 0.0:
        return (1.0, 0.0, f)
    if f == 0.0:
        return (0.0, 1.0 if g > 0 else -1.0, fabs(g))
    f1 = fabs(f)
    g1 = fabs(g)
    if f1 > rtmin and f1 < rtmax and g1 > rtmin and g1 < rtmax:
        d = sqrt(f * f + g * g)
        c = f1 / d
        r = d if f >= 0 else -d
        s = g / r
        return (c, s, r)
    u = min(safmax, max(safmin, max(f1, g1)))
    fs = f / u
    gs = g / u
    d = sqrt(fs * fs + gs * gs)
    c = fabs(fs) / d
    r = d if f >= 0 else -d
    s = gs / r
    return (c, s, r * u)

def _data_matrix(x, t, k: int, w):
    m = x.size
    nz = k + 1
    A = np.zeros((m, nz))
    off = np.zeros(m, dtype=int)
    ind = k
    for i in range(m):
        ind = _bspl_find(t, k, x[i], ind, False)
        if ind < 0:
            raise ValueError(f"find_interval: out of bounds with x = {x[i]}")
        off[i] = ind - k
        h = _deboor(t, x[i], k, ind, 0)
        for j in range(nz):
            A[i, j] = h[j] * w[i]
    return A, off

def _qr_reduce(A, off, nc: int, yw, startrow: int):
    m = A.shape[0]
    nz = A.shape[1]
    R = yw.shape[1]
    for i in range(startrow, m):
        oi = off[i]
        for j in range(oi, nc):
            if j >= min(i, nc):
                break
            c, s, r = _dlartg(A[j, 0], A[i, 0])
            A[j, 0] = r
            for l in range(1, nz):
                a = A[j, l]
                b = A[i, l]
                A[j, l] = c * a + s * b
                A[i, l - 1] = c * b - s * a
            A[i, nz - 1] = 0.0
            for l in range(R):
                a = yw[j, l]
                b = yw[i, l]
                yw[j, l] = c * a + s * b
                yw[i, l] = c * b - s * a
        if i < nc:
            off[i] = i

def _fpback_packed(A, nc: int, yw):
    nz = A.shape[1]
    R = yw.shape[1]
    cc = np.zeros((nc, R))
    for l in range(R):
        cc[nc - 1, l] = yw[nc - 1, l] / A[nc - 1, 0]
    for i in range(nc - 2, -1, -1):
        nel = min(nz, nc - i)
        for l in range(R):
            ssum = yw[i, l]
            for j in range(1, nel):
                ssum -= A[i, j] * cc[i + j, l]
            cc[i, l] = ssum / A[i, 0]
    return cc

def _residuals(x, y2, t, k: int, w, c):
    # scipy _dierckx._compute_residuals (FITPACK-style interval tracking)
    m = x.size
    R = y2.shape[1]
    res = np.zeros(m)
    fp = 0.0
    l = k + 2
    ind = k
    nt = t.size
    for it in range(1, m + 1):
        if not (x[it - 1] < t[l - 1] or l > nt - k - 1):
            l += 1
        xv = x[it - 1]
        ind = _bspl_find(t, k, xv, ind, False)
        h = _deboor(t, xv, k, ind, 0)
        for yi in range(R):
            term = 0.0
            l0 = l - k - 2
            for j in range(1, k + 2):
                l0 += 1
                term += c[l0 - 1, yi] * h[j - 1]
            delta = w[it - 1] * (term - y2[it - 1, yi])
            term = delta * delta
            res[it - 1] += term
            fp += term
    return res, fp

def _lsq_solve_qr(x, y2, t, k: int, w):
    # returns (R, Y, c, fp, residuals) like scipy _lsq_solve_qr
    A, off = _data_matrix(x, t, k, w)
    yw = y2 * w.reshape((-1, 1))
    nc = t.size - k - 1
    _qr_reduce(A, off, nc, yw, 0)
    c = _fpback_packed(A, nc, yw)
    res, fp = _residuals(x, y2, t, k, w, c)
    return A, yw, c, fp, res

def _lsq_qr(x, y2, t, k: int, w):
    A, yw, c, fp, res = _lsq_solve_qr(x, y2, t, k, w)
    return c

def make_lsq_spline(x, y, t, k: int = 3, w = None, axis: int = 0,
                    check_finite: bool = True, method: str = 'qr'):
    xa = np.ascontiguousarray(np.asarray(x, dtype=float))
    ya = np.asarray(y, dtype=float)
    ta = np.ascontiguousarray(np.asarray(t, dtype=float))
    wa = np.ones(xa.size) if w is None else np.asarray(w, dtype=float)
    ax = axis % ya.ndim
    ym = np.moveaxis(ya, ax, 0)
    if xa.shape[0] < k + 1:
        raise ValueError("Need more x points.")
    if k < 0:
        raise ValueError("Expect non-negative k.")
    for i in range(ta.size - 1):
        if ta[i + 1] - ta[i] < 0:
            raise ValueError("Expect t to be a 1D strictly increasing sequence.")
    if xa.size != ym.shape[0]:
        raise ValueError(f"Shapes of x {xa.shape} and y {ya.shape} are incompatible")
    if k > 0:
        for i in range(xa.size):
            if xa[i] < ta[k] or xa[i] > ta[ta.size - k - 1]:
                raise ValueError(f"Out of bounds w/ x = {xa}.")
    if xa.size != wa.size:
        raise ValueError(f"Shapes of x {xa.shape} and w {wa.shape} are incompatible")
    for i in range(xa.size - 1):
        if xa[i + 1] - xa[i] < 0:
            raise ValueError("Expect x to be a 1D non-decreasing sequence.")
    if method != 'qr' and method != 'norm-eq':
        raise ValueError(f"Unknown method ={method!r}.")
    y2 = np.ascontiguousarray(ym).reshape((xa.size, -1)).copy()
    c = _lsq_qr(xa, y2, ta, k, wa)
    n = ta.size - k - 1
    return BSpline(ta, c.reshape((n,) + ym.shape[1:]), k, True, ax, _fast=True)._setax(ax)

def _coeff_dd(x, lo: int, hi: int):
    n = hi - lo
    res = np.zeros(n)
    for i in range(n):
        pp = 1.0
        for kk in range(n):
            if kk != i:
                pp *= x[lo + i] - x[lo + kk]
        res[i] = 1.0 / pp
    return res

def _xtwy(X, w, Y):
    # compute_banded_symmetric_XT_W_Y
    n = X.shape[1]
    WY = Y.copy()
    for j in range(n):
        WY[2, j] *= w[j]
    for i in range(2):
        for j in range(2 - i, n):
            WY[i, j] *= w[j - (2 - i)]
        for j in range(0, n - 1 - i):
            WY[3 + i, j] *= w[j + 1 + i]
    res = np.zeros((4, n))
    for i in range(n):
        for j in range(min(n - i, 4)):
            s = 0.0
            for q in range(5 - j):
                s += X[j + q, i] * WY[q, i + j]
            res[3 - j, i + j] = s
    return res

def _b_inv(A):
    U = cholesky_banded(A)
    n = U.shape[1]
    for i in range(2, 5):
        for j in range(i - 1, n):
            U[4 - i, j] /= U[3, j - i + 1]
    D = np.empty(n)
    for j in range(n):
        D[j] = 1.0 / U[3, j] ** 2
        U[3, j] /= U[3, j]
    B = np.zeros((4, n))
    for i in range(n - 1, -1, -1):
        for j in range(min(3, n - i - 1), -1, -1):
            rng = min(3, n - i - 1)
            s = 0.0
            if j == 0:
                for kk in range(1, rng + 1):
                    s -= U[3 - kk, i + kk] * B[3 - kk, i + kk]
                s += D[i]
                B[3, i] = s
            else:
                for kk in range(1, rng + 1):
                    diag = abs(kk - j)
                    ind = i + min(kk, j)
                    s -= U[3 - kk, i + kk] * B[3 - diag, ind + diag]
                B[3 - j, i + j] = s
    for j in range(n):
        B[0, j] = 0.0
    return B

class _GCV:
    X: np.ndarray[float, 2]
    XtWX: np.ndarray[float, 2]
    wE: np.ndarray[float, 2]
    XtE: np.ndarray[float, 2]
    y: np.ndarray[float, 1]

    def __init__(self, X, XtWX, wE, XtE, y):
        self.X = X
        self.XtWX = XtWX
        self.wE = wE
        self.XtE = XtE
        self.y = y

    def __call__(self, lam: float) -> float:
        n = self.X.shape[1]
        c = solve_banded((2, 2), self.X + lam * self.wE, self.y)
        res = np.zeros(n)
        for i in range(n):
            for j in range(max(0, i - n + 3), min(5, i + 3)):
                res[i] += self.wE[j, i + 2 - j] * c[i + 2 - j]
        nr = 0.0
        for i in range(n):
            nr += (lam * res[i]) ** 2
        numer = nr / n
        lhs = self.XtWX + lam * self.XtE
        b = _b_inv(lhs)
        tot = 0.0
        for i in range(4):
            for j in range(n):
                v = b[i, j] * self.XtWX[i, j]
                if i < 3:
                    v *= 2
                tot += v
        denom = (1 - tot / n) ** 2
        return numer / denom

def make_smoothing_spline(x, y, w = None, lam: Optional[float] = None, axis: int = 0):
    xa = np.ascontiguousarray(np.asarray(x, dtype=float))
    ya = np.asarray(y, dtype=float)
    for i in range(xa.size - 1):
        if xa[i + 1] - xa[i] <= 0:
            raise ValueError("``x`` should be an ascending array")
    ax = axis % ya.ndim
    if xa.shape[0] != ya.shape[ax]:
        raise ValueError(f"``x`` should be 1D and x.shape = {xa.shape} == y.shape = {ya.shape}")
    n = xa.shape[0]
    wa = np.ones(n) if w is None else np.asarray(w, dtype=float)
    for i in range(wa.size):
        if wa[i] <= 0:
            raise ValueError("Invalid vector of weights")
    if n <= 4:
        raise ValueError("``x`` and ``y`` length must be at least 5")
    t = np.empty(n + 6)
    for i in range(3):
        t[i] = xa[0]
        t[n + 3 + i] = xa[n - 1]
    for i in range(n):
        t[3 + i] = xa[i]
    ym = np.moveaxis(ya, ax, 0)
    tsh = ym.shape[1:]
    y2 = np.ascontiguousarray(ym).reshape((n, -1)).copy()
    Xb = BSpline.design_matrix(xa, t, 3)
    X = np.zeros((5, n))
    for i in range(1, 4):
        for d in range(n - 4):
            X[i, 2 + d] = Xb[i + d, 3 + d]
    X[1, 1] = Xb[0, 0]
    X[2, 0] = (xa[2] + xa[1] - 2 * xa[0]) * Xb[0, 0]
    X[2, 1] = Xb[1, 1] + Xb[1, 2]
    X[3, 0] = (xa[2] - xa[0]) * Xb[1, 1]
    X[3, 1] = Xb[2, 2]
    X[1, n - 2] = Xb[n - 3, n - 1]
    X[1, n - 1] = (xa[n - 1] - xa[n - 3]) * Xb[n - 2, n + 2 - 2]
    X[2, n - 2] = Xb[n - 2, n + 2 - 3] + Xb[n - 2, n + 2 - 2]
    X[2, n - 1] = (2 * xa[n - 1] - xa[n - 2] - xa[n - 3]) * Xb[n - 1, n + 2 - 1]
    X[3, n - 2] = Xb[n - 1, n + 2 - 1]
    wE = np.zeros((5, n))
    cd = _coeff_dd(xa, 0, 3)
    for i in range(3):
        wE[2 + i, 0] = cd[i] / wa[i]
    cd = _coeff_dd(xa, 0, 4)
    for i in range(4):
        wE[1 + i, 1] = cd[i] / wa[i]
    for j in range(2, n - 2):
        cd = _coeff_dd(xa, j - 2, j + 3)
        for i in range(5):
            wE[i, j] = (xa[j + 2] - xa[j - 2]) * cd[i] / wa[j - 2 + i]
    cd = _coeff_dd(xa, n - 4, n)
    for i in range(4):
        wE[i, n - 2] = -cd[i] / wa[n - 4 + i]
    cd = _coeff_dd(xa, n - 3, n)
    for i in range(3):
        wE[i, n - 1] = cd[i] / wa[n - 3 + i]
    wE = wE * 6.0
    R = y2.shape[1]
    lams = np.empty(R)
    if lam is None:
        XtWX = _xtwy(X, wa, X)
        XtE = _xtwy(X, wa, wE)
        for r in range(R):
            g = _GCV(X, XtWX, wE, XtE, y2[:, r].copy())
            est = minimize_scalar(g, bounds=(0.0, float(n)), method='Bounded')
            if not est.success:
                raise ValueError(f"Unable to find minimum of the GCV function: {est.message}")
            lams[r] = est.x
    else:
        if lam < 0.0:
            raise ValueError("Regularization parameter should be non-negative")
        for r in range(R):
            lams[r] = lam
    c = np.empty((n, R))
    for r in range(R):
        cr = solve_banded((2, 2), X + lams[r] * wE, y2[:, r].copy())
        for i in range(n):
            c[i, r] = cr[i]
    c_ = np.empty((n + 2, R))
    for r in range(R):
        c0 = c[0, r]
        c1 = c[1, r]
        cm0 = c[n - 1, r]
        cm1 = c[n - 2, r]
        c_[0, r] = c0 * (t[5] + t[4] - 2 * t[3]) + c1
        c_[1, r] = c0 * (t[5] - t[3]) + c1
        for i in range(1, n - 1):
            c_[i + 1, r] = c[i, r]
        c_[n, r] = cm0 * (t[n + 2] - t[n]) + cm1
        c_[n + 1, r] = cm0 * (2 * t[n + 2] - t[n + 1] - t[n]) + cm1
    return BSpline(t, c_.reshape((n + 2,) + tsh), 3, True, 0, _fast=True)._setax(ax)

# ---------------------------------------------------------------- make_splrep
# Port of scipy/interpolate/_fitpack_repro.py (non-periodic).

_TOL = 0.001
_MAXIT = 20

def _get_residuals(x, y2, t, k: int, w):
    A, yw, c, fp, res = _lsq_solve_qr(x, y2, t, k, w)
    tot = 0.0
    for i in range(res.size):
        tot += res[i]
    if tot != tot:
        raise ValueError("error. the weighted sum of squared residuals is becoming NaN")
    return res, fp

def _fpknot_new(x, t, k: int, residuals) -> float:
    interval = k + 1
    nc = t.size - k - 1
    ix = [0]
    fparts = List[float]()
    fpart = 0.0
    for i in range(x.size):
        xv = x[i]
        rv = residuals[i]
        fpart += rv
        if xv >= t[interval] and interval < nc:
            carry = rv / 2.0
            fpart -= carry
            fparts.append(fpart)
            fpart = carry
            interval += 1
            ix.append(i)
    ix.append(x.size - 1)
    fparts.append(fpart)
    idx_max = -101
    fmax = -1e100
    for i in range(len(fparts)):
        if ix[i + 1] - ix[i] > 1 and fparts[i] > fmax:
            idx_max = i
            fmax = fparts[i]
    if idx_max == -101:
        raise ValueError("Internal error. Please report it to SciPy developers.")
    return x[(ix[idx_max] + ix[idx_max + 1] + 1) // 2]

def _add_knot(x, t, k: int, residuals):
    nk = _fpknot_new(x, t, k, residuals)
    idx = int(np.searchsorted(t, nk))
    out = np.empty(t.size + 1)
    for i in range(idx):
        out[i] = t[i]
    out[idx] = nk
    for i in range(idx, t.size):
        out[i + 1] = t[i]
    return out

def _generate_knots_impl(x, y2, w, xb: float, xe: float, k: int, s: float, nest: Optional[int]):
    acc = s * _TOL
    m = x.size
    ns = max(m + k + 1, 2 * k + 3)
    if nest is not None:
        ns = nest
        if ns < 2 * (k + 1):
            raise ValueError(f"`nest` too small: nest = {ns} < 2*(k+1) = {2 * (k + 1)}.")
    nmin = 2 * (k + 1)
    nmax = m + k + 1
    fp = 0.0
    fpold = 0.0
    t = np.empty(2 * (k + 1))
    for i in range(k + 1):
        t[i] = xb
        t[k + 1 + i] = xe
    n = t.size
    nplus = 0
    out = List[np.ndarray[float, 1]]()
    for it in range(m):
        out.append(t.copy())
        fpold = fp
        residuals, fp = _get_residuals(x, y2, t, k, w)
        fpms = fp - s
        if abs(fpms) < acc or fpms < 0:
            return out
        if n == nmin:
            nplus = 1
        else:
            delta = fpold - fp
            npl1 = int(nplus * fpms / delta) if delta > acc else nplus * 2
            nplus = min(nplus * 2, max(npl1, nplus // 2, 1))
        for j in range(nplus):
            t = _add_knot(x, t, k, residuals)
            n = t.size
            if n >= nmax:
                out.append(_not_a_knot(x, k))
                return out
            if n >= ns:
                out.append(t.copy())
                return out
            if j < nplus - 1:
                residuals, _ = _get_residuals(x, y2, t, k, w)
    return out

def _disc(t, k: int):
    n = t.size
    delta = t[n - k - 1] - t[k]
    nrint = n - 2 * k - 1
    matr = np.empty((nrint - 1, k + 2))
    for jj in range(nrint - 1):
        j = jj + k + 1
        for ii in range(k + 2):
            i = jj + ii
            res = 1.0
            for s in range(k + 2):
                if i + s != j:
                    res *= t[j] - t[i + s]
            matr[jj, ii] = (t[i + k + 1] - t[i]) / res
    fac = (delta / nrint) ** k
    return matr * fac

class _FRati:
    x: np.ndarray[float, 1]
    y: np.ndarray[float, 2]
    t: np.ndarray[float, 1]
    k: int
    w: np.ndarray[float, 1]
    s: float
    YY: np.ndarray[float, 2]
    AA: np.ndarray[float, 2]
    offset: np.ndarray[int, 1]
    nc: int
    b: np.ndarray[float, 2]
    c: np.ndarray[float, 2]

    def __init__(self, x, y, t, k: int, s: float, w, R, Y):
        self.x = x
        self.y = y
        self.t = t
        self.k = k
        self.w = w
        self.s = s
        b = _disc(t, k)
        nc = t.size - k - 1
        nz = k + 1
        nb = b.shape[0]
        R2 = Y.shape[1]
        self.YY = np.zeros((nc + nb, R2))
        for i in range(nc):
            for j in range(R2):
                self.YY[i, j] = Y[i, j]
        self.AA = np.zeros((nc + nb, k + 2))
        for i in range(nc):
            for j in range(nz):
                self.AA[i, j] = R[i, j]
        self.offset = np.empty(nc + nb, dtype=int)
        for i in range(nc):
            self.offset[i] = i
        for i in range(nb):
            self.offset[nc + i] = i
        self.nc = nc
        self.b = b
        self.c = np.zeros((nc, R2))

    def __call__(self, p: float) -> float:
        AB = self.AA.copy()
        off = self.offset.copy()
        nc = self.nc
        for i in range(self.b.shape[0]):
            for j in range(self.b.shape[1]):
                AB[nc + i, j] = self.b[i, j] / p
        QY = self.YY.copy()
        _qr_reduce(AB, off, nc, QY, nc)
        c = _fpback_packed(AB, nc, QY)
        res, fp = _residuals(self.x, self.y, self.t, self.k, self.w, c)
        self.c = c
        return fp - self.s

def _fprati_new(p1: float, f1: float, p2: float, f2: float, p3: float, f3: float) -> float:
    h1 = f1 * (f2 - f3)
    h2 = f2 * (f3 - f1)
    h3 = f3 * (f1 - f2)
    if p3 == np.inf:
        return -(p2 * h1 + p1 * h2) / h3
    return -(p1 * p2 * h3 + p2 * p3 * h1 + p1 * p3 * h2) / (p1 * h1 + p2 * h2 + p3 * h3)

def _root_rati(f: _FRati, p0: float, p1: float, f1: float, p3: float, f3: float, acc: float):
    con1 = 0.1
    con9 = 0.9
    con4 = 0.04
    ich1 = 0
    ich3 = 0
    p = p0
    for it in range(_MAXIT):
        p2 = p
        f2 = f(p)
        if abs(f2) < acc:
            return 0
        if ich3 == 0:
            if f2 - f3 <= acc:
                p3 = p2
                f3 = f2
                p = p * con4
                if p <= p1:
                    p = p1 * con9 + p2 * con1
                continue
            elif f2 < 0:
                ich3 = 1
        if ich1 == 0:
            if f1 - f2 <= acc:
                p1 = p2
                f1 = f2
                p = p / con4
                if p3 != np.inf and p <= p3:
                    p = p2 * con1 + p3 * con9
                continue
            elif f2 > 0:
                ich1 = 1
        if f1 <= f2 or f2 <= f3:
            return 2
        p = _fprati_new(p1, f1, p2, f2, p3, f3)
        if f2 < 0:
            p3 = p2
            f3 = f2
        else:
            p1 = p2
            f1 = f2
    return 3

def _make_splrep_impl(x, y2, w, xb: float, xe: float, k: int, s: float, t, nest: Optional[int]):
    # returns (t, c (nc, R))
    acc = s * _TOL
    m = x.size
    if nest is not None:
        if nest < 2 * (k + 1):
            raise ValueError(f"`nest` too small: nest = {nest} < 2*(k+1) = {2 * (k + 1)}.")
        if t is not None:
            raise ValueError("Either supply `t` or `nest`.")
    tt = np.zeros(0)
    if t is None:
        knots = _generate_knots_impl(x, y2, w, xb, xe, k, s, nest)
        tt = knots[len(knots) - 1]
    else:
        tt = np.ascontiguousarray(np.asarray(t, dtype=float))
        if D.fpchec(x, m, tt, tt.size, k) != 0:
            raise ValueError("Knots do not satisfy the Schoenberg-Whitney conditions")
    if tt.size == 2 * (k + 1):
        A, Y, c, fp, res = _lsq_solve_qr(x, y2, tt, k, w)
        return tt, c
    R, Y, c0, fp0_, res0 = _lsq_solve_qr(x, y2, tt, k, w)
    nc = tt.size - k - 1
    sr = 0.0
    for i in range(R.shape[0]):
        sr += R[i, 0]
    p = nc / sr
    _, fp = _get_residuals(x, y2, tt, k, w)
    fpinf = fp - s
    tpoly = np.empty(2 * (k + 1))
    for i in range(k + 1):
        tpoly[i] = xb
        tpoly[k + 1 + i] = xe
    _, fp0 = _get_residuals(x, y2, tpoly, k, w)
    fp0 = fp0 - s
    f = _FRati(x, y2, tt, k, s, w, R, Y)
    _root_rati(f, p, 0.0, fp0, np.inf, fpinf, acc)
    return tt, f.c

def _validate_xyw(x, y2, w, k: int, s: float):
    if s < 0:
        raise ValueError(f"`s` must be non-negative. Got s = {s}")
    for i in range(w.size):
        if w[i] < 0:
            raise ValueError("Weights must be non-negative")
    if w.shape[0] != x.shape[0]:
        raise ValueError(f"Weights is incompatible: w.shape ={w.shape} != {x.shape}.")
    if x.shape[0] != y2.shape[0]:
        raise ValueError(f"Data is incompatible: x.shape = {x.shape} and y.shape = {y2.shape}.")
    for i in range(x.size - 1):
        if x[i + 1] < x[i]:
            raise ValueError("Expect `x` to be an ordered 1D sequence.")

def make_splrep(x, y, w = None, xb = None, xe = None, k: int = 3, s: float = 0.0,
                t = None, nest: Optional[int] = None, bc_type = None):
    if bc_type is not None:
        if bc_type != 'not-a-knot':
            raise NotImplementedError("make_splrep: only bc_type='not-a-knot' is supported")
    xa = np.ascontiguousarray(np.asarray(x, dtype=float))
    ya = np.ascontiguousarray(np.asarray(y, dtype=float))
    if s == 0:
        if t is not None or w is not None or nest is not None:
            raise ValueError("s==0 is for interpolation only")
        return make_interp_spline(xa, ya, k=k)
    wa = np.ones(xa.size) if w is None else np.ascontiguousarray(np.asarray(w, dtype=float))
    y2 = ya.reshape((-1, 1)).copy()
    _validate_xyw(xa, y2, wa, k, s)
    xbv = float(np.min(xa)) if xb is None else float(xb)
    xev = float(np.max(xa)) if xe is None else float(xe)
    tt, c = _make_splrep_impl(xa, y2, wa, xbv, xev, k, s, t, nest)
    return BSpline(tt, c[:, 0].copy(), k, True, 0, _fast=True)

def make_splprep(x, w = None, u = None, ub = None, ue = None, k: int = 3, s: float = 0.0,
                 t = None, nest: Optional[int] = None, bc_type = None):
    if bc_type is not None:
        if bc_type != 'not-a-knot':
            raise NotImplementedError("make_splprep: only bc_type='not-a-knot' is supported")
    X = np.asarray(x, dtype=float)
    ndim = X.shape[0]
    m = X.shape[1]
    Xm = np.ascontiguousarray(X.T).copy()
    ua = np.zeros(m)
    if u is None:
        tot = 0.0
        for i in range(1, m):
            d = 0.0
            for j in range(ndim):
                d += (Xm[i, j] - Xm[i - 1, j]) ** 2
            tot += np.sqrt(d)
            ua[i] = tot
        for i in range(1, m):
            ua[i] = ua[i] / tot
    else:
        ua = np.ascontiguousarray(np.asarray(u, dtype=float))
    if s == 0:
        if t is not None or w is not None or nest is not None:
            raise ValueError("s==0 is for interpolation only")
        return make_interp_spline(ua, X, k=k, axis=1), ua
    wa = np.ones(m) if w is None else np.ascontiguousarray(np.asarray(w, dtype=float))
    _validate_xyw(ua, Xm, wa, k, s)
    ubv = float(np.min(ua)) if ub is None else float(ub)
    uev = float(np.max(ua)) if ue is None else float(ue)
    tt, c = _make_splrep_impl(ua, Xm, wa, ubv, uev, k, s, t, nest)
    return BSpline(tt, c.T.copy(), k, True, 1, _fast=True), ua

def generate_knots(x, y, w = None, xb = None, xe = None, k: int = 3, s: float = 0.0,
                   nest: Optional[int] = None, bc_type = None):
    xa = np.ascontiguousarray(np.asarray(x, dtype=float))
    out = List[np.ndarray[float, 1]]()
    if s == 0:
        if nest is not None or w is not None:
            raise ValueError("s == 0 is interpolation only")
        out.append(_not_a_knot(xa, k))
        return out
    ya = np.asarray(y, dtype=float)
    y2 = np.ascontiguousarray(ya).reshape((xa.size, -1)).copy()
    wa = np.ones(xa.size) if w is None else np.ascontiguousarray(np.asarray(w, dtype=float))
    _validate_xyw(xa, y2, wa, k, s)
    xbv = float(np.min(xa)) if xb is None else float(xb)
    xev = float(np.max(xa)) if xe is None else float(xe)
    return _generate_knots_impl(xa, y2, wa, xbv, xev, k, s, nest)
