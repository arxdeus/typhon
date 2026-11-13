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

def _lsq_qr(x, y2, t, k: int, w):
    # returns c (nc, R)
    m = x.size
    R = y2.shape[1]
    nz = k + 1
    nc = t.size - k - 1
    A = np.zeros((m, nz))
    off = np.zeros(m, dtype=int)
    yw = np.empty((m, R))
    ind = k
    for i in range(m):
        ind = _bspl_find(t, k, x[i], ind, False)
        if ind < 0:
            raise ValueError(f"Out of bounds w/ x = {x[i]}.")
        off[i] = ind - k
        h = _deboor(t, x[i], k, ind, 0)
        for j in range(nz):
            A[i, j] = h[j] * w[i]
        for l in range(R):
            yw[i, l] = y2[i, l] * w[i]
    for i in range(m):
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
