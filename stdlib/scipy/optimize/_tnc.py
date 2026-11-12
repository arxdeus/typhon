# Ported from scipy tnc (MIT): scipy/optimize/tnc/tnc.c (TNC 1.3,
# Copyright (c) 2002-2005 Jean-Sebastien Roy, based on Stephen G. Nash's
# Fortran TN) and scipy/optimize/_tnc.py (scipy 1.18.1, BSD-3-Clause).

import math
import numpy as np
from ._common import OptimizeResult, _opt_float, _opt_int, _has_opt, INF
from ._common import _ScalarFunction
from ._callback import _STOP_MSG

_DBL_EPSILON = 2.220446049250313e-16
_HUGE = INF

TNC_MINRC = -3
TNC_ENOMEM = -3
TNC_EINVAL = -2
TNC_INFEASIBLE = -1
TNC_LOCALMINIMUM = 0
TNC_FCONVERGED = 1
TNC_XCONVERGED = 2
TNC_MAXFUN = 3
TNC_LSFAIL = 4
TNC_CONSTANT = 5
TNC_NOPROGRESS = 6
TNC_USERABORT = 7

RCSTRINGS = {-1: 'Infeasible (lower bound > upper bound)',
             0: 'Local minimum reached (|pg| ~= 0)',
             1: 'Converged (|f_n-f_(n-1)| ~= 0)',
             2: 'Converged (|x_n-x_(n-1)| ~= 0)',
             3: 'Max. number of function evaluations reached',
             4: 'Linear search failed',
             5: 'All lower bounds are equal to the upper bounds',
             6: 'Unable to progress',
             7: 'User requested end of minimization'}

_GETPTC_OK = 0
_GETPTC_EVAL = 1
_GETPTC_EINVAL = 2
_GETPTC_FAIL = 3
_LS_OK = 0
_LS_MAXFUN = 1
_LS_FAIL = 2
_LS_USERABORT = 3

def _dnrm21(n: int, dx: np.ndarray[float, 1]) -> float:
    dssq = 1.0
    dscale = 0.0
    for i in range(n):
        if dx[i] != 0.0:
            dabsxi = abs(dx[i])
            if dscale < dabsxi:
                ratio = dscale / dabsxi
                dssq = 1.0 + dssq * ratio * ratio
                dscale = dabsxi
            else:
                ratio = dabsxi / dscale
                dssq += ratio * ratio
    return dscale * math.sqrt(dssq)

def _ddot1(n: int, dx: np.ndarray[float, 1], dy: np.ndarray[float, 1]) -> float:
    d = 0.0
    for i in range(n):
        d += dy[i] * dx[i]
    return d

def _project(n: int, x: np.ndarray[float, 1], pivot: np.ndarray[int, 1]):
    for i in range(n):
        if pivot[i] != 0:
            x[i] = 0.0

def _coercex(n: int, x: np.ndarray[float, 1], low: np.ndarray[float, 1],
             up: np.ndarray[float, 1]):
    for i in range(n):
        if x[i] < low[i]:
            x[i] = low[i]
        elif x[i] > up[i]:
            x[i] = up[i]

def _unscalex(n: int, x: np.ndarray[float, 1], xs: np.ndarray[float, 1],
              xo: np.ndarray[float, 1]):
    for i in range(n):
        x[i] = x[i] * xs[i] + xo[i]

def _scalex(n: int, x: np.ndarray[float, 1], xs: np.ndarray[float, 1],
            xo: np.ndarray[float, 1]):
    for i in range(n):
        if xs[i] > 0.0:
            x[i] = (x[i] - xo[i]) / xs[i]

def _scaleg(n: int, g: np.ndarray[float, 1], xs: np.ndarray[float, 1], fscale: float):
    for i in range(n):
        g[i] *= xs[i] * fscale

class _TNCFun[SF]:
    # function(x, &f, g, state) with evaluation counting by the caller
    sf: SF
    cb_stop: bool

    def __init__(self, sf: SF):
        self.sf = sf
        self.cb_stop = False

    def __call__(self, x: np.ndarray[float, 1], g: np.ndarray[float, 1]) -> float:
        f, gg = self.sf.fun_and_grad(x.copy())
        for i in range(x.size):
            g[i] = gg[i]
        return f

class _Getptc:
    reltol: float
    abstol: float
    u: float
    fu: float
    gu: float
    xmin: float
    fmin: float
    gmin: float
    xw: float
    fw: float
    gw: float
    a: float
    b: float
    oldf: float
    b1: float
    scxbnd: float
    e: float
    step: float
    factor: float
    braktd: bool
    gtest1: float
    gtest2: float
    tol: float

    def __init__(self):
        self.reltol = 0.0
        self.abstol = 0.0
        self.u = 0.0
        self.fu = 0.0
        self.gu = 0.0
        self.xmin = 0.0
        self.fmin = 0.0
        self.gmin = 0.0
        self.xw = 0.0
        self.fw = 0.0
        self.gw = 0.0
        self.a = 0.0
        self.b = 0.0
        self.oldf = 0.0
        self.b1 = 0.0
        self.scxbnd = 0.0
        self.e = 0.0
        self.step = 0.0
        self.factor = 0.0
        self.braktd = False
        self.gtest1 = 0.0
        self.gtest2 = 0.0
        self.tol = 0.0

    def init(self, tnytol: float, eta: float, rmu: float, xbnd: float) -> int:
        if self.u <= 0.0 or xbnd <= tnytol or self.gu > 0.0:
            return _GETPTC_EINVAL
        if xbnd < self.abstol:
            self.abstol = xbnd
        self.tol = self.abstol
        self.a = 0.0
        self.xw = 0.0
        self.xmin = 0.0
        self.oldf = self.fu
        self.fmin = self.fu
        self.fw = self.fu
        self.gw = self.gu
        self.gmin = self.gu
        self.step = self.u
        self.factor = 5.0
        self.braktd = False
        self.scxbnd = xbnd
        self.b = self.scxbnd + self.reltol * abs(self.scxbnd) + self.abstol
        self.e = self.b + self.b
        self.b1 = self.b
        self.gtest1 = -rmu * self.gu
        self.gtest2 = -eta * self.gu
        if self.step >= self.scxbnd:
            self.step = self.scxbnd
            self.scxbnd -= (self.reltol * abs(xbnd) + self.abstol) / (1.0 + self.reltol)
        self.u = self.step
        if abs(self.step) < self.tol and self.step < 0.0:
            self.u = -self.tol
        if abs(self.step) < self.tol and self.step >= 0.0:
            self.u = self.tol
        return _GETPTC_EVAL

    def iter(self, big: float, rtsmll: float, tnytol: float, fpresn: float,
             xbnd: float) -> int:
        conv_check = False
        if self.fu <= self.fmin:
            chordu = self.oldf - (self.xmin + self.u) * self.gtest1
            if self.fu > chordu:
                chordm = self.oldf - self.xmin * self.gtest1
                self.gu = -self.gmin
                denom = chordm - self.fmin
                if abs(denom) < 1e-15:
                    denom = 1e-15
                    if chordm - self.fmin < 0.0:
                        denom = -denom
                if self.xmin != 0.0:
                    self.gu = self.gmin * (chordu - self.fu) / denom
                self.fu = 0.5 * self.u * (self.gmin + self.gu) + self.fmin
                if self.fu < self.fmin:
                    self.fu = self.fmin
            else:
                self.fw = self.fmin
                self.fmin = self.fu
                self.gw = self.gmin
                self.gmin = self.gu
                self.xmin += self.u
                self.a -= self.u
                self.b -= self.u
                self.xw = -self.u
                self.scxbnd -= self.u
                if self.gu <= 0.0:
                    self.a = 0.0
                else:
                    self.b = 0.0
                    self.braktd = True
                self.tol = abs(self.xmin) * self.reltol + self.abstol
                conv_check = True
        if not conv_check:
            if self.u < 0.0:
                self.a = self.u
            else:
                self.b = self.u
                self.braktd = True
            self.xw = self.u
            self.fw = self.fu
            self.gw = self.gu
        twotol = self.tol + self.tol
        xmidpt = 0.5 * (self.a + self.b)
        convrg = (abs(xmidpt) <= twotol - 0.5 * (self.b - self.a)) or \
            (abs(self.gmin) <= self.gtest2 and self.fmin < self.oldf and
             ((abs(self.xmin - xbnd) > self.tol) or (not self.braktd)))
        if convrg:
            if self.xmin != 0.0:
                return _GETPTC_OK
            if abs(self.oldf - self.fw) <= fpresn:
                return _GETPTC_FAIL
            self.tol = 0.1 * self.tol
            if self.tol < tnytol:
                return _GETPTC_FAIL
            self.reltol = 0.1 * self.reltol
            self.abstol = 0.1 * self.abstol
            twotol = 0.1 * twotol
        r = 0.0
        q = 0.0
        s = 0.0
        min_found = False
        if abs(self.e) > self.tol:
            r = 3.0 * (self.fmin - self.fw) / self.xw + self.gmin + self.gw
            absr = abs(r)
            q = absr
            if self.gw != 0.0 and self.gmin != 0.0:
                abgw = abs(self.gw)
                abgmin = abs(self.gmin)
                s = math.sqrt(abgmin) * math.sqrt(abgw)
                if self.gw / abgw * self.gmin > 0.0:
                    if r >= s or r <= -s:
                        q = math.sqrt(abs(r + s)) * math.sqrt(abs(r - s))
                    else:
                        r = 0.0
                        q = 0.0
                        min_found = True
                else:
                    sumsq = 1.0
                    p = 0.0
                    scale = 0.0
                    if absr >= s:
                        if absr > rtsmll:
                            p = absr * rtsmll
                        if s >= p:
                            value = s / absr
                            sumsq = 1.0 + value * value
                        scale = absr
                    else:
                        if s > rtsmll:
                            p = s * rtsmll
                        if absr >= p:
                            value = absr / s
                            sumsq = 1.0 + value * value
                        scale = s
                    sumsq = math.sqrt(sumsq)
                    q = big
                    if scale < big / sumsq:
                        q = scale * sumsq
            if not min_found:
                if self.xw < 0.0:
                    q = -q
                s = self.xw * (self.gmin - r - q)
                q = self.gw - self.gmin + q + q
                if q > 0.0:
                    s = -s
                if q <= 0.0:
                    q = -q
                r = self.e
                if self.b1 != self.step or self.braktd:
                    self.e = self.step
        a1 = self.a
        self.b1 = self.b
        self.step = xmidpt
        if (not self.braktd) or ((self.a == 0.0 and self.xw < 0.0) or
                                 (self.b == 0.0 and self.xw > 0.0)):
            if self.braktd:
                d1 = self.xw
                d2 = self.a
                if self.a == 0.0:
                    d2 = self.b
                self.u = -d1 / d2
                self.step = 5.0 * d2 * (0.1 + 1.0 / self.u) / 11.0
                if self.u < 1.0:
                    self.step = 0.5 * d2 * math.sqrt(self.u)
            else:
                self.step = -self.factor * self.xw
                if self.step > self.scxbnd:
                    self.step = self.scxbnd
                if self.step != self.scxbnd:
                    self.factor = 5.0 * self.factor
            if self.step <= 0.0:
                a1 = self.step
            if self.step > 0.0:
                self.b1 = self.step
        if abs(s) <= abs(0.5 * q * r) or s <= q * a1 or s >= q * self.b1:
            self.e = self.b - self.a
        else:
            self.step = s / q
            if self.step - self.a < twotol or self.b - self.step < twotol:
                if xmidpt <= 0.0:
                    self.step = -self.tol
                else:
                    self.step = self.tol
        if self.step >= self.scxbnd:
            self.step = self.scxbnd
            self.scxbnd -= (self.reltol * abs(xbnd) + self.abstol) / (1.0 + self.reltol)
        self.u = self.step
        if abs(self.step) < self.tol and self.step < 0.0:
            self.u = -self.tol
        if abs(self.step) < self.tol and self.step >= 0.0:
            self.u = self.tol
        return _GETPTC_EVAL

# ------------------------------------------------------------- helpers

def _step_max(step: float, n: int, x: np.ndarray[float, 1], d: np.ndarray[float, 1],
              pivot: np.ndarray[int, 1], low: np.ndarray[float, 1],
              up: np.ndarray[float, 1], xs: np.ndarray[float, 1],
              xo: np.ndarray[float, 1]) -> float:
    for i in range(n):
        if pivot[i] == 0 and d[i] != 0.0:
            if d[i] < 0.0:
                t = (low[i] - xo[i]) / xs[i] - x[i]
                if t > step * d[i]:
                    step = t / d[i]
            else:
                t = (up[i] - xo[i]) / xs[i] - x[i]
                if t < step * d[i]:
                    step = t / d[i]
    return step

def _add_constraint(n: int, x: np.ndarray[float, 1], p: np.ndarray[float, 1],
                    pivot: np.ndarray[int, 1], low: np.ndarray[float, 1],
                    up: np.ndarray[float, 1], xs: np.ndarray[float, 1],
                    xo: np.ndarray[float, 1]) -> bool:
    newcon = False
    for i in range(n):
        if pivot[i] == 0 and p[i] != 0.0:
            if p[i] < 0.0 and low[i] != -_HUGE:
                tol = _DBL_EPSILON * 10.0 * (abs(low[i]) + 1.0)
                if x[i] * xs[i] + xo[i] - low[i] <= tol:
                    pivot[i] = -1
                    x[i] = (low[i] - xo[i]) / xs[i]
                    newcon = True
            elif up[i] != _HUGE:
                tol = _DBL_EPSILON * 10.0 * (abs(up[i]) + 1.0)
                if up[i] - (x[i] * xs[i] + xo[i]) <= tol:
                    pivot[i] = 1
                    x[i] = (up[i] - xo[i]) / xs[i]
                    newcon = True
    return newcon

def _remove_constraint(gtpnew: float, gnorm: float, pgtolfs: float, f: float,
                       fLastConstraint: float, g: np.ndarray[float, 1],
                       pivot: np.ndarray[int, 1], n: int) -> bool:
    if (fLastConstraint - f) <= (gtpnew * -0.5) and gnorm > pgtolfs:
        return False
    imax = -1
    cmax = 0.0
    for i in range(n):
        if pivot[i] == 2:
            continue
        t = -pivot[i] * g[i]
        if t < cmax:
            cmax = t
            imax = i
    if imax != -1:
        pivot[imax] = 0
        return True
    return False

def _ssbfgs(n: int, gamma: float, sj: np.ndarray[float, 1], hjv: np.ndarray[float, 1],
            hjyj: np.ndarray[float, 1], yjsj: float, yjhyj: float, vsj: float,
            vhyj: float, hjp1v: np.ndarray[float, 1]):
    delta = 0.0
    beta = 0.0
    if yjsj != 0.0:
        delta = (gamma * yjhyj / yjsj + 1.0) * vsj / yjsj - gamma * vhyj / yjsj
        beta = -gamma * vsj / yjsj
    for i in range(n):
        hjp1v[i] = gamma * hjv[i] + delta * sj[i] + beta * hjyj[i]

class _Precond:
    # quasi-Newton preconditioner state shared by msolve / initPreconditioner
    n: int
    sk: np.ndarray[float, 1]
    yk: np.ndarray[float, 1]
    sr: np.ndarray[float, 1]
    yr: np.ndarray[float, 1]
    diagb: np.ndarray[float, 1]
    upd1: bool
    yksk: float
    yrsr: float
    lreset: bool

    def __init__(self, n: int):
        self.n = n
        self.sk = np.zeros(n)
        self.yk = np.zeros(n)
        self.sr = np.zeros(n)
        self.yr = np.zeros(n)
        self.diagb = np.ones(n)
        self.upd1 = True
        self.yksk = 0.0
        self.yrsr = 0.0
        self.lreset = False

    def msolve(self, g: np.ndarray[float, 1], y: np.ndarray[float, 1]):
        n = self.n
        diagb = self.diagb
        if self.upd1:
            for i in range(n):
                y[i] = g[i] / diagb[i]
            return
        gsk = _ddot1(n, g, self.sk)
        hg = np.zeros(n)
        hyr = np.zeros(n)
        hyk = np.zeros(n)
        if self.lreset:
            for i in range(n):
                rdiagb = 1.0 / diagb[i]
                hg[i] = g[i] * rdiagb
                hyk[i] = self.yk[i] * rdiagb
            ykhyk = _ddot1(n, self.yk, hyk)
            ghyk = _ddot1(n, g, hyk)
            _ssbfgs(n, 1.0, self.sk, hg, hyk, self.yksk, ykhyk, gsk, ghyk, y)
        else:
            for i in range(n):
                rdiagb = 1.0 / diagb[i]
                hg[i] = g[i] * rdiagb
                hyk[i] = self.yk[i] * rdiagb
                hyr[i] = self.yr[i] * rdiagb
            gsr = _ddot1(n, g, self.sr)
            ghyr = _ddot1(n, g, hyr)
            yrhyr = _ddot1(n, self.yr, hyr)
            _ssbfgs(n, 1.0, self.sr, hg, hyr, self.yrsr, yrhyr, gsr, ghyr, hg)
            yksr = _ddot1(n, self.yk, self.sr)
            ykhyr = _ddot1(n, self.yk, hyr)
            _ssbfgs(n, 1.0, self.sr, hyk, hyr, self.yrsr, yrhyr, yksr, ykhyr, hyk)
            ykhyk = _ddot1(n, hyk, self.yk)
            ghyk = _ddot1(n, hyk, g)
            _ssbfgs(n, 1.0, self.sk, hg, hyk, self.yksk, ykhyk, gsk, ghyk, y)

    def init_emat(self, emat: np.ndarray[float, 1]):
        n = self.n
        diagb = self.diagb
        sk = self.sk
        yk = self.yk
        sr = self.sr
        yr = self.yr
        if self.upd1:
            for i in range(n):
                emat[i] = diagb[i]
            return
        bsk = np.zeros(n)
        yksk = self.yksk
        yrsr = self.yrsr
        if self.lreset:
            for i in range(n):
                bsk[i] = diagb[i] * sk[i]
            sds = _ddot1(n, sk, bsk)
            if yksk == 0.0:
                yksk = 1.0
            if sds == 0.0:
                sds = 1.0
            for i in range(n):
                td = diagb[i]
                emat[i] = td - td * td * sk[i] * sk[i] / sds + yk[i] * yk[i] / yksk
        else:
            for i in range(n):
                bsk[i] = diagb[i] * sr[i]
            sds = _ddot1(n, sr, bsk)
            srds = _ddot1(n, sk, bsk)
            yrsk = _ddot1(n, yr, sk)
            if yrsr == 0.0:
                yrsr = 1.0
            if sds == 0.0:
                sds = 1.0
            for i in range(n):
                td = diagb[i]
                bsk[i] = td * sk[i] - bsk[i] * srds / sds + yr[i] * yrsk / yrsr
                emat[i] = td - td * td * sr[i] * sr[i] / sds + yr[i] * yr[i] / yrsr
            sds = _ddot1(n, sk, bsk)
            if yksk == 0.0:
                yksk = 1.0
            if sds == 0.0:
                sds = 1.0
            for i in range(n):
                emat[i] -= bsk[i] * bsk[i] / sds + yk[i] * yk[i] / yksk

class _TNC[F]:
    fn: F
    n: int
    low: np.ndarray[float, 1]
    up: np.ndarray[float, 1]
    xs: np.ndarray[float, 1]
    xo: np.ndarray[float, 1]
    nfeval: int
    maxnfeval: int

    def __init__(self, fn: F, n: int, low: np.ndarray[float, 1],
                 up: np.ndarray[float, 1], xs: np.ndarray[float, 1],
                 xo: np.ndarray[float, 1], maxnfeval: int):
        self.fn = fn
        self.n = n
        self.low = low
        self.up = up
        self.xs = xs
        self.xo = xo
        self.nfeval = 0
        self.maxnfeval = maxnfeval

    def hessian_times_vector(self, v: np.ndarray[float, 1], gv: np.ndarray[float, 1],
                             x: np.ndarray[float, 1], g: np.ndarray[float, 1],
                             fscale: float, accuracy: float, xnorm: float):
        n = self.n
        delta = accuracy * (xnorm + 1.0)
        xv = np.empty(n)
        for i in range(n):
            xv[i] = x[i] + delta * v[i]
        _unscalex(n, xv, self.xs, self.xo)
        _coercex(n, xv, self.low, self.up)
        self.fn(xv, gv)
        _scaleg(n, gv, self.xs, fscale)
        dinv = 1.0 / delta
        for i in range(n):
            gv[i] = (gv[i] - g[i]) * dinv
        for i in range(n):
            if self.xs[i] == 0.0:
                gv[i] = 0.0

    def direction(self, zsol: np.ndarray[float, 1], pc: _Precond,
                  x: np.ndarray[float, 1], g: np.ndarray[float, 1],
                  maxCGit: int, fscale: float, pivot: np.ndarray[int, 1],
                  accuracy: float, gnorm: float, xnorm: float):
        n = self.n
        if maxCGit == 0:
            for i in range(n):
                zsol[i] = -g[i]
            _project(n, zsol, pivot)
            return
        rhsnrm = gnorm
        tol = 1e-12
        qold = 0.0
        rzold = 0.0
        r = np.empty(n)
        v = np.empty(n)
        zk = np.zeros(n)
        emat = np.zeros(n)
        gv = np.zeros(n)
        pc.init_emat(emat)
        for i in range(n):
            r[i] = -g[i]
            v[i] = 0.0
            zsol[i] = 0.0
        for k in range(maxCGit):
            _project(n, r, pivot)
            pc.msolve(r, zk)
            _project(n, zk, pivot)
            rz = _ddot1(n, r, zk)
            if (rz / rhsnrm < tol) or (self.nfeval >= (self.maxnfeval - 1)):
                if k == 0:
                    for i in range(n):
                        zsol[i] = -g[i]
                    _project(n, zsol, pivot)
                break
            beta = 0.0 if k == 0 else rz / rzold
            for i in range(n):
                v[i] = zk[i] + beta * v[i]
            _project(n, v, pivot)
            self.hessian_times_vector(v, gv, x, g, fscale, accuracy, xnorm)
            self.nfeval += 1
            _project(n, gv, pivot)
            vgv = _ddot1(n, v, gv)
            if vgv / rhsnrm < tol:
                if k == 0:
                    pc.msolve(g, zsol)
                    for i in range(n):
                        zsol[i] = -zsol[i]
                    _project(n, zsol, pivot)
                break
            # diagonalScaling
            vr = 1.0 / _ddot1(n, v, r)
            vgvi = 1.0 / _ddot1(n, v, gv)
            for i in range(n):
                emat[i] += -r[i] * r[i] * vr + gv[i] * gv[i] * vgvi
                if emat[i] <= 1e-6:
                    emat[i] = 1.0
            alpha = rz / vgv
            for i in range(n):
                zsol[i] += alpha * v[i]
            for i in range(n):
                r[i] += -alpha * gv[i]
            gtp = _ddot1(n, zsol, g)
            pr = _ddot1(n, r, zsol)
            qnew = (gtp + pr) * 0.5
            qtest = (k + 1) * (1.0 - qold / qnew)
            if qtest <= 0.5:
                break
            if gtp > 0.0:
                for i in range(n):
                    zsol[i] += -alpha * v[i]
                break
            qold = qnew
            rzold = rz
        for i in range(n):
            pc.diagb[i] = emat[i]

    def linear_search(self, fscale: float, pivot: np.ndarray[int, 1], eta: float,
                      ftol: float, xbnd: float, p: np.ndarray[float, 1],
                      x: np.ndarray[float, 1], f: float, alpha: float,
                      gfull: np.ndarray[float, 1]):
        # returns (rc, f, alpha)
        n = self.n
        maxlsit = 64
        temp = np.empty(n)
        tempgfull = np.zeros(n)
        newgfull = np.zeros(n)
        for i in range(n):
            temp[i] = gfull[i]
        _scaleg(n, temp, self.xs, fscale)
        gs = _Getptc()
        gs.gu = _ddot1(n, temp, p)
        for i in range(n):
            temp[i] = x[i]
        _project(n, temp, pivot)
        xnorm = _dnrm21(n, temp)
        rteps = math.sqrt(_DBL_EPSILON)
        pe = _dnrm21(n, p) + _DBL_EPSILON
        gs.reltol = rteps * (xnorm + 1.0) / pe
        gs.abstol = -_DBL_EPSILON * (1.0 + abs(f)) / (gs.gu - _DBL_EPSILON)
        tnytol = _DBL_EPSILON * (xnorm + 1.0) / pe
        rtsmll = _DBL_EPSILON
        big = 1.0 / (_DBL_EPSILON * _DBL_EPSILON)
        itcnt = 0
        fpresn = ftol
        gs.u = alpha
        gs.fu = f
        gs.fmin = f
        rmu = 1e-4
        itest = gs.init(tnytol, eta, rmu, xbnd)
        alpha = gs.xmin
        while itest == _GETPTC_EVAL:
            itcnt += 1
            if itcnt > maxlsit or self.nfeval >= self.maxnfeval:
                break
            ualpha = gs.xmin + gs.u
            for i in range(n):
                temp[i] = x[i] + ualpha * p[i]
            _unscalex(n, temp, self.xs, self.xo)
            _coercex(n, temp, self.low, self.up)
            fu = self.fn(temp, tempgfull)
            self.nfeval += 1
            gs.fu = fu * fscale
            for i in range(n):
                temp[i] = tempgfull[i]
            _scaleg(n, temp, self.xs, fscale)
            gs.gu = _ddot1(n, temp, p)
            itest = gs.iter(big, rtsmll, tnytol, fpresn, xbnd)
            if gs.xmin == ualpha:
                for i in range(n):
                    newgfull[i] = tempgfull[i]
        alpha = gs.xmin
        if itest == _GETPTC_OK:
            for i in range(n):
                x[i] += alpha * p[i]
            for i in range(n):
                gfull[i] = newgfull[i]
            return _LS_OK, gs.fmin, alpha
        elif itcnt > maxlsit:
            return _LS_FAIL, f, alpha
        elif itest != _GETPTC_EVAL:
            return _LS_FAIL, f, alpha
        return _LS_MAXFUN, f, alpha

    def minimize(self, x: np.ndarray[float, 1], f: float, gfull: np.ndarray[float, 1],
                 fscale: float, maxCGit: int, eta: float, stepmx: float,
                 accuracy: float, fmin: float, ftol: float, xtol: float,
                 pgtol: float, rescale: float, callback):
        # tnc_minimize; returns (rc, f, niter, stopped)
        n = self.n
        xs = self.xs
        xo = self.xo
        low = self.low
        up = self.up
        niter = 0
        oldg = np.zeros(n)
        g = np.zeros(n)
        temp = np.zeros(n)
        pk = np.zeros(n)
        pivot = np.zeros(n, int)
        pc = _Precond(n)
        difnew = 0.0
        epsred = 0.05
        icycle = n - 1
        newcon = True
        alpha = 0.0
        rc = TNC_ENOMEM
        stopped = False
        _scalex(n, x, xs, xo)
        f *= fscale
        for i in range(n):
            if xs[i] == 0.0:
                pivot[i] = 2
            elif low[i] != -_HUGE and (x[i] * xs[i] + xo[i] - low[i] <=
                                       _DBL_EPSILON * 10.0 * (abs(low[i]) + 1.0)):
                pivot[i] = -1
            elif up[i] != _HUGE and (x[i] * xs[i] + xo[i] - up[i] >=
                                     _DBL_EPSILON * 10.0 * (abs(up[i]) + 1.0)):
                pivot[i] = 1
            else:
                pivot[i] = 0
        for i in range(n):
            g[i] = gfull[i]
        _scaleg(n, g, xs, fscale)
        for i in range(n):
            if -pivot[i] * g[i] < 0.0:
                pivot[i] = 0
        _project(n, g, pivot)
        gnorm = _dnrm21(n, g)
        fLastConstraint = f
        fLastReset = f
        while True:
            if _dnrm21(n, g) <= pgtol * fscale:
                for i in range(n):
                    g[i] = gfull[i]
                _project(n, g, pivot)
                rc = TNC_LOCALMINIMUM
                break
            if self.nfeval >= self.maxnfeval:
                rc = TNC_MAXFUN
                break
            newscale = _dnrm21(n, g)
            if newscale > _DBL_EPSILON and abs(math.log10(newscale)) > rescale:
                newscale = 1.0 / newscale
                f *= newscale
                fscale *= newscale
                gnorm *= newscale
                fLastConstraint *= newscale
                fLastReset *= newscale
                difnew *= newscale
                for i in range(n):
                    g[i] *= newscale
                for i in range(n):
                    pc.diagb[i] = 1.0
                pc.upd1 = True
                icycle = n - 1
                newcon = True
            for i in range(n):
                temp[i] = x[i]
            _project(n, temp, pivot)
            xnorm = _dnrm21(n, temp)
            oldnfeval = self.nfeval
            self.direction(pk, pc, x, g, maxCGit, fscale, pivot, accuracy,
                           gnorm, xnorm)
            if not newcon:
                if not pc.lreset:
                    for i in range(n):
                        pc.sr[i] += pc.sk[i]
                        pc.yr[i] += pc.yk[i]
                    icycle += 1
                else:
                    for i in range(n):
                        pc.sr[i] = pc.sk[i]
                        pc.yr[i] = pc.yk[i]
                    fLastReset = f
                    icycle = 1
            for i in range(n):
                oldg[i] = g[i]
            oldf = f
            oldgtp = _ddot1(n, pk, g)
            ustpmax = stepmx / (_dnrm21(n, pk) + _DBL_EPSILON)
            spe = _step_max(ustpmax, n, x, pk, pivot, low, up, xs, xo)
            brk = False
            if spe > 0.0:
                # initialStep
                d = abs(f - fmin / fscale)
                alpha = 1.0
                if d * 2.0 <= -oldgtp and d >= _DBL_EPSILON:
                    alpha = d * -2.0 / oldgtp
                if alpha >= spe:
                    alpha = spe
                lsrc, f, alpha = self.linear_search(fscale, pivot, eta, ftol, spe,
                                                    pk, x, f, alpha, gfull)
                if lsrc == _LS_FAIL:
                    rc = TNC_LSFAIL
                    break
                if alpha >= 0.9 * ustpmax:
                    stepmx *= 1e2
                if alpha - spe >= -_DBL_EPSILON * 10.0:
                    newcon = True
                else:
                    if lsrc != _LS_OK:
                        if lsrc == _LS_MAXFUN:
                            rc = TNC_MAXFUN
                        else:
                            rc = TNC_LSFAIL
                        break
                    newcon = False
            else:
                newcon = True
            if newcon:
                if not _add_constraint(n, x, pk, pivot, low, up, xs, xo):
                    if self.nfeval == oldnfeval:
                        rc = TNC_NOPROGRESS
                        break
                fLastConstraint = f
            niter += 1
            if callback is not None:
                for i in range(n):
                    temp[i] = x[i]
                _unscalex(n, temp, xs, xo)
                try:
                    callback(temp.copy())
                except StopIteration:
                    stopped = True
                    rc = TNC_USERABORT
                    break
            difold = difnew
            difnew = oldf - f
            if icycle == 1:
                if difnew > difold * 2.0:
                    epsred += epsred
                if difnew < difold * 0.5:
                    epsred *= 0.5
            for i in range(n):
                g[i] = gfull[i]
            _scaleg(n, g, xs, fscale)
            for i in range(n):
                temp[i] = g[i]
            _project(n, temp, pivot)
            gnorm = _dnrm21(n, temp)
            remcon = _remove_constraint(oldgtp, gnorm, pgtol * fscale, f,
                                        fLastConstraint, g, pivot, n)
            if remcon:
                for i in range(n):
                    temp[i] = g[i]
                _project(n, temp, pivot)
                gnorm = _dnrm21(n, temp)
                fLastConstraint = f
            if (not remcon) and (not newcon):
                if abs(difnew) <= ftol * fscale:
                    rc = TNC_FCONVERGED
                    break
                if alpha * _dnrm21(n, pk) <= xtol:
                    rc = TNC_XCONVERGED
                    break
            _project(n, g, pivot)
            if not newcon:
                for i in range(n):
                    pc.yk[i] = g[i] - oldg[i]
                    pc.sk[i] = alpha * pk[i]
                pc.yksk = _ddot1(n, pc.yk, pc.sk)
                if icycle == (n - 1) or difnew < epsred * (fLastReset - f):
                    pc.lreset = True
                else:
                    pc.yrsr = _ddot1(n, pc.yr, pc.sr)
                    pc.lreset = pc.yrsr <= 0.0
                pc.upd1 = False
        _unscalex(n, x, xs, xo)
        _coercex(n, x, low, up)
        f /= fscale
        return rc, f, niter, stopped

def _minimize_tnc(fun, x0: np.ndarray[float, 1], args, jac, bnds, callback, options):
    eps = _opt_float(options, 'eps', 1e-8)
    maxCGit = _opt_int(options, 'maxCGit', -1)
    eta = _opt_float(options, 'eta', -1.0)
    stepmx = _opt_float(options, 'stepmx', 0.0)
    accuracy = _opt_float(options, 'accuracy', 0.0)
    fmin = _opt_float(options, 'minfev', 0.0)
    ftol = _opt_float(options, 'ftol', -1.0)
    xtol = _opt_float(options, 'xtol', -1.0)
    pgtol = _opt_float(options, 'gtol', -1.0)
    rescale = _opt_float(options, 'rescale', -1.0)
    n = x0.size
    maxfun = _opt_int(options, 'maxfun', max(100, 10 * n))
    frs: Optional[float] = None
    if _has_opt(options, 'finite_diff_rel_step'):
        frs = _opt_float(options, 'finite_diff_rel_step', 0.0)
    low = np.full(n, -INF)
    up = np.full(n, INF)
    if bnds is not None:
        low, up = bnds
        low = low.copy()
        up = up.copy()
    if isinstance(jac, str):
        sf = _ScalarFunction(fun, x0, args, jac, None, frs, low, up)
    else:
        sf = _ScalarFunction(fun, x0, args, jac, eps, frs, low, up)
    fn = _TNCFun(sf)
    x = x0.copy()
    g = np.zeros(n)
    f = INF
    rc = 0
    niter = 0
    stopped = False
    nfeval = 0
    for i in range(n):
        if low[i] > up[i]:
            rc = TNC_INFEASIBLE
    if rc == 0:
        _coercex(n, x, low, up)
        if maxfun < 1:
            rc = TNC_MAXFUN
        else:
            f = fn(x, g)
            nfeval = 1
            nc = 0
            for i in range(n):
                if low[i] == up[i]:
                    nc += 1
            if nc == n:
                rc = TNC_CONSTANT
            else:
                xs = np.empty(n)
                xo = np.empty(n)
                for i in range(n):
                    if low[i] != -_HUGE and up[i] != _HUGE:
                        xs[i] = up[i] - low[i]
                        xo[i] = (up[i] + low[i]) * 0.5
                    else:
                        xs[i] = 1.0 + abs(x[i])
                        xo[i] = x[i]
                rteps = math.sqrt(_DBL_EPSILON)
                if stepmx < rteps * 10.0:
                    stepmx = 1.0e1
                if eta < 0.0 or eta >= 1.0:
                    eta = 0.25
                if rescale < 0:
                    rescale = 1.3
                if maxCGit < 0:
                    maxCGit = n // 2
                    if maxCGit < 1:
                        maxCGit = 1
                    elif maxCGit > 50:
                        maxCGit = 50
                if maxCGit > n:
                    maxCGit = n
                if accuracy <= _DBL_EPSILON:
                    accuracy = rteps
                if ftol < 0.0:
                    ftol = accuracy
                if pgtol < 0.0:
                    pgtol = 1e-2 * math.sqrt(accuracy)
                if xtol < 0.0:
                    xtol = rteps
                t = _TNC(fn, n, low, up, xs, xo, maxfun)
                t.nfeval = nfeval
                rc, f, niter, stopped = t.minimize(x, f, g, 1.0, maxCGit, eta,
                                                   stepmx, accuracy, fmin, ftol,
                                                   xtol, pgtol, rescale, callback)
    funv, jacv = sf.fun_and_grad(x)
    r = OptimizeResult(x, funv, jacv.copy(), nit=niter, nfev=sf.nfev,
                       status=rc, message=RCSTRINGS[rc],
                       success=(-1 < rc < 3))
    if stopped:
        r.success = False
        r.status = 99
        r.message = _STOP_MSG
    return r
