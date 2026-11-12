# Ported from scipy 1.18.1 (BSD-3-Clause)

# Wolfe line searches used by BFGS and CG: line_search_wolfe1 (MINPACK-2
# dcsrch, More-Thuente) with a fallback to line_search_wolfe2 (Nocedal &
# Wright Alg. 3.5 with zoom).

import math
import numpy as np
from ._common import _sqrt, _sign, INF

def _dcstep(stx: float, fx: float, dx: float, sty: float, fy: float, dy: float,
            stp: float, fp: float, dp: float, brackt: bool, stpmin: float,
            stpmax: float, cver: bool = False):
    # cver: follow the C (lbfgsb.c) variant, which differs from the Python
    # DCSRCH only in a strict comparison in the first case.
    sgnd = _sign(dp) * _sign(dx)
    stpf = 0.0
    if fp > fx:
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s * _sqrt((theta / s) ** 2 - dx / s * (dp / s))
        if stp < stx:
            gamma *= -1
        p = gamma - dx + theta
        q = gamma - dx + gamma + dp
        r = p / q
        stpc = stx + r * (stp - stx)
        stpq = stx + dx / ((fx - fp) / (stp - stx) + dx) / 2.0 * (stp - stx)
        if abs(stpc - stx) < abs(stpq - stx) or (not cver and abs(stpc - stx) == abs(stpq - stx)):
            stpf = stpc
        else:
            stpf = stpc + (stpq - stpc) / 2.0
        brackt = True
    elif sgnd < 0.0:
        theta = 3 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s * _sqrt((theta / s) ** 2 - dx / s * (dp / s))
        if stp > stx:
            gamma *= -1
        p = gamma - dp + theta
        q = gamma - dp + gamma + dx
        r = p / q
        stpc = stp + r * (stx - stp)
        stpq = stp + dp / (dp - dx) * (stx - stp)
        if abs(stpc - stp) > abs(stpq - stp):
            stpf = stpc
        else:
            stpf = stpq
        brackt = True
    elif abs(dp) < abs(dx):
        theta = 3 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s * _sqrt(max(0.0, (theta / s) ** 2 - dx / s * (dp / s)))
        if stp > stx:
            gamma = -gamma
        p = gamma - dp + theta
        q = gamma + (dx - dp) + gamma
        r = p / q
        stpc = 0.0
        if r < 0 and gamma != 0:
            stpc = stp + r * (stx - stp)
        elif stp > stx:
            stpc = stpmax
        else:
            stpc = stpmin
        stpq = stp + dp / (dp - dx) * (stx - stp)
        if brackt:
            if abs(stpc - stp) < abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            if stp > stx:
                stpf = min(stp + 0.66 * (sty - stp), stpf)
            else:
                stpf = max(stp + 0.66 * (sty - stp), stpf)
        else:
            if abs(stpc - stp) > abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            stpf = min(max(stpf, stpmin), stpmax)
    elif brackt:
        theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp
        s = max(abs(theta), abs(dy), abs(dp))
        gamma = s * _sqrt((theta / s) ** 2 - dy / s * (dp / s))
        if stp > sty:
            gamma = -gamma
        p = gamma - dp + theta
        q = gamma - dp + gamma + dy
        r = p / q
        stpc = stp + r * (sty - stp)
        stpf = stpc
    elif stp > stx:
        stpf = stpmax
    else:
        stpf = stpmin
    if fp > fx:
        sty = stp
        fy = fp
        dy = dp
    else:
        if sgnd < 0:
            sty = stx
            fy = fx
            dy = dx
        stx = stp
        fx = fp
        dx = dp
    stp = stpf
    return stx, fx, dx, sty, fy, dy, stp, brackt

class _DCSRCH:
    # task codes: 0 START, 1 FG, 2 CONVERGENCE, 3 WARNING, 4 ERROR
    brackt: bool
    stage: int
    ginit: float
    gtest: float
    gx: float
    gy: float
    finit: float
    fx: float
    fy: float
    stx: float
    sty: float
    stmin: float
    stmax: float
    width: float
    width1: float
    ftol: float
    gtol: float
    xtol: float
    stpmin: float
    stpmax: float
    cver: bool

    def __init__(self, ftol: float, gtol: float, xtol: float, stpmin: float,
                 stpmax: float, cver: bool = False):
        self.cver = cver
        self.ftol = ftol
        self.gtol = gtol
        self.xtol = xtol
        self.stpmin = stpmin
        self.stpmax = stpmax
        self.brackt = False
        self.stage = 1
        self.ginit = 0.0
        self.gtest = 0.0
        self.gx = 0.0
        self.gy = 0.0
        self.finit = 0.0
        self.fx = 0.0
        self.fy = 0.0
        self.stx = 0.0
        self.sty = 0.0
        self.stmin = 0.0
        self.stmax = 0.0
        self.width = 0.0
        self.width1 = 0.0

    def iterate(self, stp: float, f: float, g: float, task: int):
        p5 = 0.5
        p66 = 0.66
        xtrapl = 1.1
        xtrapu = 4.0
        if task == 0:
            if stp < self.stpmin:
                task = 4
            if stp > self.stpmax:
                task = 4
            if g >= 0:
                task = 4
            if self.ftol < 0 or self.gtol < 0 or self.xtol < 0:
                task = 4
            if self.stpmin < 0 or self.stpmax < self.stpmin:
                task = 4
            if task == 4:
                return stp, f, g, task
            self.brackt = False
            self.stage = 1
            self.finit = f
            self.ginit = g
            self.gtest = self.ftol * self.ginit
            self.width = self.stpmax - self.stpmin
            self.width1 = self.width / p5
            self.stx = 0.0
            self.fx = self.finit
            self.gx = self.ginit
            self.sty = 0.0
            self.fy = self.finit
            self.gy = self.ginit
            self.stmin = 0.0
            self.stmax = stp + xtrapu * stp
            return stp, f, g, 1
        ftest = self.finit + stp * self.gtest
        if self.stage == 1 and f <= ftest and g >= 0:
            self.stage = 2
        if self.brackt and (stp <= self.stmin or stp >= self.stmax):
            task = 3
        if self.brackt and self.stmax - self.stmin <= self.xtol * self.stmax:
            task = 3
        if stp == self.stpmax and f <= ftest and g <= self.gtest:
            task = 3
        if stp == self.stpmin and (f > ftest or g >= self.gtest):
            task = 3
        if f <= ftest and abs(g) <= self.gtol * -self.ginit:
            task = 2
        if task == 3 or task == 2:
            return stp, f, g, task
        if self.stage == 1 and f <= self.fx and f > ftest:
            fm = f - stp * self.gtest
            fxm = self.fx - self.stx * self.gtest
            fym = self.fy - self.sty * self.gtest
            gm = g - self.gtest
            gxm = self.gx - self.gtest
            gym = self.gy - self.gtest
            self.stx, fxm, gxm, self.sty, fym, gym, stp, self.brackt = _dcstep(
                self.stx, fxm, gxm, self.sty, fym, gym, stp, fm, gm,
                self.brackt, self.stmin, self.stmax, self.cver)
            self.fx = fxm + self.stx * self.gtest
            self.fy = fym + self.sty * self.gtest
            self.gx = gxm + self.gtest
            self.gy = gym + self.gtest
        else:
            (self.stx, self.fx, self.gx, self.sty, self.fy, self.gy, stp,
             self.brackt) = _dcstep(self.stx, self.fx, self.gx, self.sty,
                                    self.fy, self.gy, stp, f, g, self.brackt,
                                    self.stmin, self.stmax, self.cver)
        if self.brackt:
            if abs(self.sty - self.stx) >= p66 * self.width1:
                stp = self.stx + p5 * (self.sty - self.stx)
            self.width1 = self.width
            self.width = abs(self.sty - self.stx)
        if self.brackt:
            self.stmin = min(self.stx, self.sty)
            self.stmax = max(self.stx, self.sty)
        else:
            self.stmin = stp + xtrapl * (stp - self.stx)
            self.stmax = stp + xtrapu * (stp - self.stx)
        stp = min(max(stp, self.stpmin), self.stpmax)
        if ((self.brackt and (stp <= self.stmin or stp >= self.stmax)) or
                (self.brackt and self.stmax - self.stmin <= self.xtol * self.stmax)):
            stp = self.stx
        return stp, f, g, 1

class _LineFun[SF]:
    # phi(alpha) = f(xk + alpha pk), derphi(alpha) = g(xk + alpha pk) . pk
    sf: SF
    xk: np.ndarray[float, 1]
    pk: np.ndarray[float, 1]
    gval: np.ndarray[float, 1]
    gval_alpha: float
    has_gval: bool

    def __init__(self, sf: SF, xk: np.ndarray[float, 1], pk: np.ndarray[float, 1],
                 gfk: np.ndarray[float, 1]):
        self.sf = sf
        self.xk = xk
        self.pk = pk
        self.gval = gfk
        self.gval_alpha = math.nan
        self.has_gval = False

    def phi(self, s: float) -> float:
        return self.sf.fun(self.xk + s * self.pk)

    def derphi(self, s: float) -> float:
        self.gval = self.sf.grad(self.xk + s * self.pk)
        self.gval_alpha = s
        self.has_gval = True
        return float(np.dot(self.gval, self.pk))

def _wolfe1(lf, phi0: float, old_phi0: float, derphi0: float,
            c1: float, c2: float, amax: float, amin: float,
            xtol: float = 1e-14):
    # returns (ok, stp, phi1, phi0)
    alpha1 = 1.0
    # old_phi0 NaN stands for scipy's None
    if derphi0 != 0 and old_phi0 == old_phi0:
        alpha1 = min(1.0, 1.01 * 2 * (phi0 - old_phi0) / derphi0)
        if alpha1 < 0:
            alpha1 = 1.0
    dc = _DCSRCH(c1, c2, xtol, amin, amax)
    phi1 = phi0
    derphi1 = derphi0
    task = 0
    stp = alpha1
    ok = False
    converged = False
    for i in range(100):
        stp, phi1, derphi1, task = dc.iterate(alpha1, phi1, derphi1, task)
        if not (stp == stp and abs(stp) != INF):
            task = 3
            break
        if task == 1:
            alpha1 = stp
            phi1 = lf.phi(stp)
            derphi1 = lf.derphi(stp)
        else:
            converged = True
            break
    ok = converged and task != 4 and task != 3
    return ok, stp, phi1, phi0

def _cubicmin(a: float, fa: float, fpa: float, b: float, fb: float, c: float,
              fc: float):
    C = fpa
    db = b - a
    dc = c - a
    denom = (db * dc) ** 2 * (db - dc)
    if denom == 0.0:
        return False, 0.0
    d00 = dc ** 2
    d01 = -db ** 2
    d10 = -dc ** 3
    d11 = db ** 3
    v0 = fb - fa - C * db
    v1 = fc - fa - C * dc
    A = d00 * v0 + d01 * v1
    B = d10 * v0 + d11 * v1
    A /= denom
    B /= denom
    radical = B * B - 3 * A * C
    if radical < 0 or A == 0.0:
        return False, 0.0
    xmin = a + (-B + math.sqrt(radical)) / (3 * A)
    if not (xmin == xmin and abs(xmin) != INF):
        return False, 0.0
    return True, xmin

def _quadmin(a: float, fa: float, fpa: float, b: float, fb: float):
    D = fa
    C = fpa
    db = b - a * 1.0
    if db == 0.0:
        return False, 0.0
    B = (fb - D - C * db) / (db * db)
    if B == 0.0:
        return False, 0.0
    xmin = a - C / (2.0 * B)
    if not (xmin == xmin and abs(xmin) != INF):
        return False, 0.0
    return True, xmin

def _zoom(a_lo: float, a_hi: float, phi_lo: float, phi_hi: float,
          derphi_lo: float, lf, phi0: float, derphi0: float, c1: float,
          c2: float, extra_condition):
    maxiter = 10
    i = 0
    delta1 = 0.2
    delta2 = 0.1
    phi_rec = phi0
    a_rec = 0.0
    while True:
        dalpha = a_hi - a_lo
        a = a_lo
        b = a_hi
        if dalpha < 0:
            a, b = a_hi, a_lo
        cchk = 0.0
        ok = False
        a_j = 0.0
        if i > 0:
            cchk = delta1 * dalpha
            ok, a_j = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec,
                                phi_rec)
        if i == 0 or (not ok) or a_j > b - cchk or a_j < a + cchk:
            qchk = delta2 * dalpha
            ok, a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
            if (not ok) or a_j > b - qchk or a_j < a + qchk:
                a_j = a_lo + 0.5 * dalpha
        phi_aj = lf.phi(a_j)
        if phi_aj > phi0 + c1 * a_j * derphi0 or phi_aj >= phi_lo:
            phi_rec = phi_hi
            a_rec = a_hi
            a_hi = a_j
            phi_hi = phi_aj
        else:
            derphi_aj = lf.derphi(a_j)
            if abs(derphi_aj) <= -c2 * derphi0 and extra_condition(a_j, phi_aj):
                return True, a_j, phi_aj, derphi_aj
            if derphi_aj * (a_hi - a_lo) >= 0:
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_lo
                phi_hi = phi_lo
            else:
                phi_rec = phi_lo
                a_rec = a_lo
            a_lo = a_j
            phi_lo = phi_aj
            derphi_lo = derphi_aj
        i += 1
        if i > maxiter:
            return False, 0.0, 0.0, 0.0

def _wolfe2(lf, phi0: float, old_phi0: float, derphi0: float, c1: float,
            c2: float, amax: float, extra_condition, maxiter: int = 10):
    # returns (ok, alpha_star, phi_star, derphi_ok)
    alpha0 = 0.0
    alpha1 = 1.0
    if derphi0 != 0 and old_phi0 == old_phi0:
        alpha1 = min(1.0, 1.01 * 2 * (phi0 - old_phi0) / derphi0)
    if alpha1 < 0:
        alpha1 = 1.0
    alpha1 = min(alpha1, amax)
    phi_a1 = lf.phi(alpha1)
    phi_a0 = phi0
    derphi_a0 = derphi0
    for i in range(maxiter):
        if alpha1 == 0 or alpha0 > amax:
            return False, 0.0, phi0, False
        not_first_iteration = i > 0
        if phi_a1 > phi0 + c1 * alpha1 * derphi0 or (phi_a1 >= phi_a0 and not_first_iteration):
            ok, a, p, d = _zoom(alpha0, alpha1, phi_a0, phi_a1, derphi_a0, lf,
                                phi0, derphi0, c1, c2, extra_condition)
            return ok, a, p, ok
        derphi_a1 = lf.derphi(alpha1)
        if abs(derphi_a1) <= -c2 * derphi0:
            if extra_condition(alpha1, phi_a1):
                return True, alpha1, phi_a1, True
        if derphi_a1 >= 0:
            ok, a, p, d = _zoom(alpha1, alpha0, phi_a1, phi_a0, derphi_a1, lf,
                                phi0, derphi0, c1, c2, extra_condition)
            return ok, a, p, ok
        alpha2 = 2 * alpha1
        alpha2 = min(alpha2, amax)
        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = lf.phi(alpha1)
        derphi_a0 = derphi_a1
    # did not converge: scipy returns alpha1 with derphi_star None
    return True, alpha1, phi_a1, False

def _line_search_wolfe12(sf, xk: np.ndarray[float, 1], pk: np.ndarray[float, 1],
                         gfk: np.ndarray[float, 1], old_fval: float,
                         old_old_fval: float, c1: float, c2: float,
                         amin: float, amax: float, extra_condition):
    # returns (ok, alpha, new_fval, old_fval, gfkp1, has_gfkp1)
    derphi0 = float(np.dot(gfk, pk))
    lf = _LineFun(sf, xk, pk, gfk)
    ok, stp, fval, oldf = _wolfe1(lf, old_fval, old_old_fval, derphi0, c1, c2,
                                  amax, amin)
    if ok:
        if not extra_condition(stp, xk + stp * pk, fval, lf.gval):
            ok = False
    if ok:
        return True, stp, fval, oldf, lf.gval, True
    lf2 = _LineFun(sf, xk, pk, gfk)

    def extra2(alpha: float, phi: float) -> bool:
        if lf2.gval_alpha != alpha:
            lf2.derphi(alpha)
        return extra_condition(alpha, xk + alpha * pk, phi, lf2.gval)

    ok2, a2, f2, has_d = _wolfe2(lf2, old_fval, old_old_fval, derphi0, c1, c2,
                                 amax, extra2)
    if not ok2:
        return False, 0.0, 0.0, 0.0, gfk, False
    return True, a2, f2, old_fval, lf2.gval, has_d
