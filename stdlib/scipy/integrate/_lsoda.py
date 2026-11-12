# Ported from ODEPACK lsoda (public domain)
#
# Pure Typhon port of LSODA (Petzold/Hindmarsh, automatic Adams/BDF method
# switching) following scipy 1.18.1's C translation `scipy/integrate/src/lsoda.c`
# (BSD-3-Clause). Used by solve_ivp(method='LSODA'), odeint and ode('lsoda').
# Arrays are 0-based. The Nordsieck history `yh` is stored column-major
# (yh[i + j*nyh]) exactly as in the reference. Banded Jacobians (jt = 4, 5) are
# assembled into dense storage; partial pivoting picks the same pivots as
# dgbtrf because entries outside the band are zero.
# Deviations: the iteration matrix is factored with an unblocked LU (dgetf2
# order) instead of LAPACK dgetrf; no diagnostic messages are printed (ixpr,
# mxhnil only counted); workspace-size errors (istate -7) cannot occur.

import math
import numpy as np

_SM1 = [0.5, 0.575, 0.55, 0.45, 0.35, 0.25, 0.20, 0.15, 0.10, 0.075, 0.050, 0.025]
_UROUND = 2.220446049250313e-16
_NCOL = 14

def _vmnorm(n: int, v: np.ndarray[float, 1], off: int, w: np.ndarray[float, 1]) -> float:
    vm = 0.0
    for i in range(n):
        x = abs(v[off + i]) * w[i]
        if x > vm:
            vm = x
    return vm

def _lu_getrf(a: np.ndarray[float, 2], n: int, piv: List[int]) -> int:
    # dgetf2: right-looking LU with partial pivoting (first max wins)
    info = 0
    for k in range(n):
        p = k
        amax = abs(a[k, k])
        for i in range(k + 1, n):
            v = abs(a[i, k])
            if v > amax:
                amax = v
                p = i
        piv[k] = p
        if a[p, k] == 0.0:
            if info == 0:
                info = k + 1
            continue
        if p != k:
            for j in range(n):
                tmp = a[k, j]
                a[k, j] = a[p, j]
                a[p, j] = tmp
        inv = 1.0 / a[k, k]
        for i in range(k + 1, n):
            a[i, k] *= inv
        for j in range(k + 1, n):
            akj = a[k, j]
            if akj != 0.0:
                for i in range(k + 1, n):
                    a[i, j] -= a[i, k] * akj
    return info

def _lu_getrs(a: np.ndarray[float, 2], n: int, piv: List[int], x: np.ndarray[float, 1]):
    for k in range(n):
        p = piv[k]
        if p != k:
            tmp = x[k]
            x[k] = x[p]
            x[p] = tmp
    for j in range(n):
        xj = x[j]
        if xj != 0.0:
            for i in range(j + 1, n):
                x[i] -= xj * a[i, j]
    for j in range(n - 1, -1, -1):
        if x[j] != 0.0:
            x[j] /= a[j, j]
            xj = x[j]
            for i in range(j):
                x[i] -= xj * a[i, j]

def _cfode(meth: int, elco: List[float], tesco: List[float]):
    pc = [0.0] * 12
    if meth == 1:
        elco[0] = 1.0
        elco[1] = 1.0
        tesco[0] = 0.0
        tesco[1] = 2.0
        tesco[3] = 1.0
        tesco[35] = 0.0
        pc[0] = 1.0
        rqfac = 1.0
        for nq in range(1, 12):
            rq1fac = rqfac
            rqfac = rqfac / (nq + 1)
            pc[nq] = 0.0
            for ib in range(nq):
                pc[nq - ib] = pc[nq - ib - 1] + nq * pc[nq - ib]
            pc[0] = nq * pc[0]
            pint = pc[0]
            xpin = pc[0] * 0.5
            tsign = 1.0
            for i in range(1, nq + 1):
                tsign = -tsign
                pint = pint + tsign * pc[i] / (i + 1)
                xpin = xpin + tsign * pc[i] / (i + 2)
            elco[0 + nq * 13] = pint * rq1fac
            elco[1 + nq * 13] = 1.0
            for i in range(1, nq + 1):
                elco[(i + 1) + nq * 13] = rq1fac * pc[i] / (i + 1)
            agamq = rqfac * xpin
            ragq = 1.0 / agamq
            tesco[1 + 3 * nq] = ragq
            if nq < 11:
                tesco[3 * (nq + 1)] = ragq * rqfac / (nq + 2)
            tesco[2 + 3 * (nq - 1)] = ragq
        return
    pc[0] = 1.0
    rq1fac = 1.0
    for nq in range(5):
        pc[nq + 1] = 0.0
        for ib in range(nq + 1):
            pc[nq - ib + 1] = pc[nq - ib] + (nq + 1) * pc[nq - ib + 1]
        pc[0] = (nq + 1) * pc[0]
        for i in range(nq + 2):
            elco[i + nq * 13] = pc[i] / pc[1]
        elco[1 + nq * 13] = 1.0
        tesco[0 + 3 * nq] = rq1fac
        tesco[1 + 3 * nq] = (nq + 2) / elco[nq * 13]
        tesco[2 + 3 * nq] = (nq + 3) / elco[nq * 13]
        rq1fac = rq1fac / (nq + 1)

class _Lsoda:
    # common block (doubles)
    conit: float
    crate: float
    el: List[float]
    elco: List[float]
    hold: float
    rmax: float
    tesco: List[float]
    ccmax: float
    el0: float
    h: float
    hmin: float
    hmxi: float
    hu: float
    rc: float
    tn: float
    uround: float
    tsw: float
    pdest: float
    pdlast: float
    ratio: float
    cm1: List[float]
    cm2: List[float]
    pdnorm: float
    # common block (ints)
    illin: int
    init: int
    mxstep: int
    mxhnil: int
    nhnil: int
    ntrep: int
    nslast: int
    nyh: int
    ialth: int
    ipup: int
    lmax: int
    nqnyh: int
    nslp: int
    icf: int
    ierpj: int
    iersl: int
    jcur: int
    jstart: int
    kflag: int
    l: int
    meth: int
    miter: int
    maxord: int
    maxcor: int
    msbp: int
    mxncf: int
    n: int
    nq: int
    nst: int
    nfe: int
    nje: int
    nqu: int
    icount: int
    irflag: int
    jtyp: int
    mused: int
    mxordn: int
    mxords: int
    # problem setup and optional inputs (rwork/iwork slots in the reference)
    jt: int
    ml: int
    mu: int
    rtol: np.ndarray[float, 1]
    atol: np.ndarray[float, 1]
    rtol_vec: bool
    iopt: int
    opt_h0: float
    opt_hmax: float
    opt_hmin: float
    opt_ixpr: int
    opt_mxstep: int
    opt_mxhnil: int
    opt_mxordn: int
    opt_mxords: int
    tcrit: float
    tolsf: float
    imxer: int
    # work arrays
    yh: np.ndarray[float, 1]
    ewt: np.ndarray[float, 1]
    savf: np.ndarray[float, 1]
    acor: np.ndarray[float, 1]
    wm: np.ndarray[float, 2]
    piv: List[int]
    ftem: np.ndarray[float, 1]
    # stoda corrector outputs
    _m: int
    _del: float

    def __init__(self):
        self.el = [0.0] * 13
        self.elco = [0.0] * 156
        self.tesco = [0.0] * 36
        self.cm1 = [0.0] * 12
        self.cm2 = [0.0] * 5
        self.conit = 0.0
        self.crate = 0.0
        self.hold = 0.0
        self.rmax = 0.0
        self.ccmax = 0.0
        self.el0 = 0.0
        self.h = 0.0
        self.hmin = 0.0
        self.hmxi = 0.0
        self.hu = 0.0
        self.rc = 0.0
        self.tn = 0.0
        self.uround = _UROUND
        self.tsw = 0.0
        self.pdest = 0.0
        self.pdlast = 0.0
        self.ratio = 0.0
        self.pdnorm = 0.0
        self.illin = 0
        self.init = 0
        self.mxstep = 0
        self.mxhnil = 0
        self.nhnil = 0
        self.ntrep = 0
        self.nslast = 0
        self.nyh = 0
        self.ialth = 0
        self.ipup = 0
        self.lmax = 0
        self.nqnyh = 0
        self.nslp = 0
        self.icf = 0
        self.ierpj = 0
        self.iersl = 0
        self.jcur = 0
        self.jstart = 0
        self.kflag = 0
        self.l = 0
        self.meth = 0
        self.miter = 0
        self.maxord = 0
        self.maxcor = 0
        self.msbp = 0
        self.mxncf = 0
        self.n = 0
        self.nq = 0
        self.nst = 0
        self.nfe = 0
        self.nje = 0
        self.nqu = 0
        self.icount = 0
        self.irflag = 0
        self.jtyp = 0
        self.mused = 0
        self.mxordn = 0
        self.mxords = 0
        self.jt = 2
        self.ml = 0
        self.mu = 0
        self.rtol = np.zeros(0)
        self.atol = np.zeros(0)
        self.rtol_vec = False
        self.iopt = 0
        self.opt_h0 = 0.0
        self.opt_hmax = 0.0
        self.opt_hmin = 0.0
        self.opt_ixpr = 0
        self.opt_mxstep = 0
        self.opt_mxhnil = 0
        self.opt_mxordn = 0
        self.opt_mxords = 0
        self.tcrit = 0.0
        self.tolsf = 0.0
        self.imxer = -1
        self.yh = np.zeros(0)
        self.ewt = np.zeros(0)
        self.savf = np.zeros(0)
        self.acor = np.zeros(0)
        self.wm = np.zeros((0, 0))
        self.piv = List[int]()
        self.ftem = np.zeros(0)
        self._m = 0
        self._del = 0.0

    def _alloc(self, n: int):
        self.yh = np.zeros(_NCOL * n)
        self.ewt = np.zeros(n)
        self.savf = np.zeros(n)
        self.acor = np.zeros(n)
        self.wm = np.zeros((n, n))
        self.piv = [0] * n
        self.ftem = np.zeros(n)

    def _call_f(self, f, t: float, y: np.ndarray[float, 1], out: np.ndarray[float, 1]):
        r = f(t, y)
        for i in range(self.n):
            out[i] = r[i]

    def _ewset(self, ycur: np.ndarray[float, 1]):
        for i in range(self.n):
            self.ewt[i] = self.rtol[i] * abs(ycur[i]) + self.atol[i]

    def _prja(self, y: np.ndarray[float, 1], f, jac):
        n = self.n
        self.nje += 1
        self.ierpj = 0
        self.jcur = 1
        hl0 = self.h * self.el0
        wm = self.wm
        ewt = self.ewt
        savf = self.savf
        ftem = self.ftem
        if self.miter == 1 or self.miter == 4:
            J = jac(self.tn, y)
            con = -hl0
            for i in range(n):
                for j in range(n):
                    wm[i, j] = J[i, j] * con
        elif self.miter == 2:
            fac = _vmnorm(n, savf, 0, ewt)
            r0 = 1000.0 * abs(self.h) * self.uround * n * fac
            if r0 == 0.0:
                r0 = 1.0
            srur = math.sqrt(self.uround)
            for j in range(n):
                yj = y[j]
                r = max(srur * abs(yj), r0 / ewt[j])
                y[j] = y[j] + r
                fac = -hl0 / r
                self._call_f(f, self.tn, y, ftem)
                for i in range(n):
                    wm[i, j] = (ftem[i] - savf[i]) * fac
                y[j] = yj
            self.nfe += n
        else:
            # miter 5: banded finite differences, mband column groups
            ml = self.ml
            mu = self.mu
            mband = ml + mu + 1
            mba = min(mband, n)
            for i in range(n):
                for j in range(n):
                    wm[i, j] = 0.0
            srur = math.sqrt(self.uround)
            fac = _vmnorm(n, savf, 0, ewt)
            r0 = 1000.0 * abs(self.h) * self.uround * n * fac
            if r0 == 0.0:
                r0 = 1.0
            for j in range(mba):
                i = j
                while i < n:
                    yi = y[i]
                    r = max(srur * abs(yi), r0 / ewt[i])
                    y[i] = y[i] + r
                    i += mband
                self._call_f(f, self.tn, y, ftem)
                jj = j
                while jj < n:
                    y[jj] = self.yh[jj]
                    yjj = y[jj]
                    r = max(srur * abs(yjj), r0 / ewt[jj])
                    fac = -hl0 / r
                    i1 = max(jj - mu, 0)
                    i2 = min(jj + ml, n - 1)
                    for i in range(i1, i2 + 1):
                        wm[i, jj] = (ftem[i] - savf[i]) * fac
                    jj += mband
            self.nfe += mba
        # norm of the Jacobian (fnorm == bnorm for banded storage)
        an = 0.0
        for i in range(n):
            s = 0.0
            for j in range(n):
                s = s + abs(wm[i, j]) / ewt[j]
            an = max(an, s * ewt[i])
        self.pdnorm = an / abs(hl0)
        for i in range(n):
            wm[i, i] = wm[i, i] + 1.0
        if _lu_getrf(wm, n, self.piv) != 0:
            self.ierpj = 1

    def _solsy(self, x: np.ndarray[float, 1]):
        self.iersl = 0
        _lu_getrs(self.wm, self.n, self.piv, x)

    def _first_call_init(self):
        self.lmax = self.maxord + 1
        self.nq = 1
        self.l = 2
        self.ialth = 2
        self.rmax = 10000.0
        self.rc = 0.0
        self.el0 = 1.0
        self.crate = 0.7
        self.hold = self.h
        self.nslp = 0
        self.ipup = self.miter
        self.icount = 20
        self.irflag = 0
        self.pdest = 0.0
        self.pdlast = 0.0
        self.ratio = 5.0
        _cfode(2, self.elco, self.tesco)
        for i in range(5):
            self.cm2[i] = self.tesco[1 + i * 3] * self.elco[i + 1 + i * 13]
        _cfode(1, self.elco, self.tesco)
        for i in range(12):
            self.cm1[i] = self.tesco[1 + i * 3] * self.elco[i + 1 + i * 13]

    def _reset(self):
        for i in range(self.l):
            self.el[i] = self.elco[i + (self.nq - 1) * 13]
        self.nqnyh = self.nq * self.nyh
        self.rc = self.rc * self.el[0] / self.el0
        self.el0 = self.el[0]
        self.conit = 0.5 / (self.nq + 2)

    def _adjust_step(self, rh: float) -> float:
        rh = min(rh, self.rmax)
        rh = rh / max(1.0, abs(self.h) * self.hmxi * rh)
        if self.meth != 2:
            self.irflag = 0
            pdh = max(abs(self.h) * self.pdlast, 0.000001)
            if rh * pdh * 1.00001 >= _SM1[self.nq - 1]:
                rh = _SM1[self.nq - 1] / pdh
                self.irflag = 1
        r = 1.0
        n = self.n
        nyh = self.nyh
        yh = self.yh
        for j in range(1, self.l):
            r = r * rh
            for i in range(n):
                yh[i + j * nyh] = yh[i + j * nyh] * r
        self.h = self.h * rh
        self.rc = self.rc * rh
        self.ialth = self.l
        return rh

    def _predict(self):
        if abs(self.rc - 1.0) > self.ccmax:
            self.ipup = self.miter
        if self.nst >= self.nslp + self.msbp:
            self.ipup = self.miter
        self.tn = self.tn + self.h
        yh = self.yh
        nyh = self.nyh
        i1 = self.nqnyh
        for jb in range(self.nq):
            i1 -= nyh
            for i in range(i1, self.nqnyh):
                yh[i] = yh[i] + yh[i + nyh]

    def _retract(self):
        yh = self.yh
        nyh = self.nyh
        i1 = self.nqnyh
        for jb in range(self.nq):
            i1 -= nyh
            for i in range(i1, self.nqnyh):
                yh[i] = yh[i] - yh[i + nyh]

    # Corrector loop (labels 220-450). Returns 0 converged, 1 no convergence,
    # 3 retry with a fresh Jacobian.
    def _corrector(self, y: np.ndarray[float, 1], f, jac, pnorm: float) -> int:
        n = self.n
        nyh = self.nyh
        yh = self.yh
        savf = self.savf
        acor = self.acor
        ewt = self.ewt
        m = 0
        rate = 0.0
        dl = 0.0
        delp = 0.0
        for i in range(n):
            y[i] = yh[i]
        self._call_f(f, self.tn, y, savf)
        self.nfe += 1
        if self.ipup > 0:
            self._prja(y, f, jac)
            self.ipup = 0
            self.rc = 1.0
            self.nslp = self.nst
            self.crate = 0.7
            if self.ierpj != 0:
                return 1
        for i in range(n):
            acor[i] = 0.0
        while True:
            if self.miter == 0:
                for i in range(n):
                    savf[i] = self.h * savf[i] - yh[i + nyh]
                    y[i] = savf[i] - acor[i]
                dl = _vmnorm(n, y, 0, ewt)
                for i in range(n):
                    y[i] = yh[i] + self.el[0] * savf[i]
                    acor[i] = savf[i]
            else:
                for i in range(n):
                    y[i] = self.h * savf[i] - (yh[i + nyh] + acor[i])
                self._solsy(y)
                dl = _vmnorm(n, y, 0, ewt)
                for i in range(n):
                    acor[i] = acor[i] + y[i]
                    y[i] = yh[i] + self.el[0] * acor[i]
            if dl <= 100.0 * pnorm * self.uround:
                break
            if m != 0 or self.meth != 1:
                if m != 0:
                    rm = 1024.0
                    if dl <= 1024.0 * delp:
                        rm = dl / delp
                    rate = max(rate, rm)
                    self.crate = max(0.2 * self.crate, rm)
                dcon = dl * min(1.0, 1.5 * self.crate) / (self.tesco[1 + (self.nq - 1) * 3] * self.conit)
                if dcon <= 1.0:
                    self.pdest = max(self.pdest, rate / abs(self.h * self.el[0]))
                    if self.pdest != 0.0:
                        self.pdlast = self.pdest
                    break
            m = m + 1
            if m == self.maxcor or (m >= 2 and dl > 2.0 * delp):
                if self.miter != 0 and self.jcur != 1:
                    self.icf = 1
                    self.ipup = self.miter
                    return 3
                return 1
            delp = dl
            self._call_f(f, self.tn, y, savf)
            self.nfe += 1
        self.jcur = 0
        self._m = m
        self._del = dl
        return 0

    def _corrector_failure(self, ncf: int, told: float) -> int:
        # returns -1 on fatal error, else 0
        self.icf = 2
        self.rmax = 2.0
        self.tn = told
        self._retract()
        if self.ierpj < 0 or self.iersl < 0:
            self.kflag = -3
            return -1
        if abs(self.h) <= self.hmin * 1.00001:
            self.kflag = -2
            return -1
        if ncf == self.mxncf:
            self.kflag = -2
            return -1
        rh = 0.25
        self.ipup = self.miter
        rh = max(rh, self.hmin / abs(self.h))
        self._adjust_step(rh)
        return 0

    def _finish_stoda(self, reset_rmax: bool):
        if reset_rmax:
            self.rmax = 10.0
        r = 1.0 / self.tesco[1 + (self.nqu - 1) * 3]
        for i in range(self.n):
            self.acor[i] = self.acor[i] * r
        self.hold = self.h
        self.jstart = 1

    def _stoda(self, y: np.ndarray[float, 1], f, jac):
        n = self.n
        nyh = self.nyh
        yh = self.yh
        ewt = self.ewt
        savf = self.savf
        acor = self.acor
        sm1 = _SM1
        rh = 0.0
        dsm = 0.0
        rhup = 0.0
        rhdn = 0.0
        rhsm = 0.0
        told = self.tn
        pdh = 0.0
        newq = 0
        ncf = 0
        reset_rmax = False
        self.kflag = 0
        self.ierpj = 0
        self.iersl = 0
        self.jcur = 0
        self.icf = 0
        if self.jstart == 0:
            self._first_call_init()
            self._reset()
        elif self.jstart == -1:
            self.ipup = self.miter
            self.lmax = self.maxord + 1
            if self.ialth == 1:
                self.ialth = 2
            if self.meth != self.mused:
                _cfode(self.meth, self.elco, self.tesco)
                self.ialth = self.l
                self._reset()
            if self.h != self.hold:
                rh = self.h / self.hold
                self.h = self.hold
                rh = max(rh, self.hmin / abs(self.h))
                self._adjust_step(rh)
        elif self.jstart == -2:
            if self.h != self.hold:
                rh = self.h / self.hold
                self.h = self.hold
                self._adjust_step(rh)
        self._predict()
        pnorm = _vmnorm(n, yh, 0, ewt)
        while True:
            cs = self._corrector(y, f, jac, pnorm)
            if cs == 3:
                continue
            if cs == 1:
                ncf += 1
                if self._corrector_failure(ncf, told) != 0:
                    self.hold = self.h
                    self.jstart = 1
                    return
                self._predict()
                pnorm = _vmnorm(n, yh, 0, ewt)
                continue
            self.jcur = 0
            m_c = self._m
            if m_c == 0:
                dsm = self._del / self.tesco[1 + (self.nq - 1) * 3]
            else:
                dsm = _vmnorm(n, acor, 0, ewt) / self.tesco[1 + (self.nq - 1) * 3]
            if dsm > 1.0:
                self.kflag -= 1
                self.tn = told
                self._retract()
                self.rmax = 2.0
                if abs(self.h) <= self.hmin * 1.00001:
                    self.kflag = -1
                    self.hold = self.h
                    self.jstart = 1
                    return
                if self.kflag <= -3:
                    if self.kflag == -10:
                        self.kflag = -1
                        self.hold = self.h
                        self.jstart = 1
                        return
                    rh = 0.1
                    rh = max(self.hmin / abs(self.h), rh)
                    self.h = self.h * rh
                    for i in range(n):
                        y[i] = yh[i]
                    self._call_f(f, self.tn, y, savf)
                    self.nfe += 1
                    for i in range(n):
                        yh[i + nyh] = self.h * savf[i]
                    self.ipup = self.miter
                    self.ialth = 5
                    if self.nq != 1:
                        self.nq = 1
                        self.l = 2
                        self._reset()
                    self._predict()
                    pnorm = _vmnorm(n, yh, 0, ewt)
                    continue
                rhup = 0.0
            else:
                self.kflag = 0
                reset_rmax = True
                self.nst += 1
                self.hu = self.h
                self.nqu = self.nq
                self.mused = self.meth
                for j in range(self.l):
                    ej = self.el[j]
                    for i in range(n):
                        yh[i + j * nyh] = yh[i + j * nyh] + ej * acor[i]
                self.icount -= 1
                switched = False
                if self.icount >= 0:
                    pass
                elif self.meth == 1:
                    cond_b = (dsm > 100.0 * pnorm * self.uround) and (self.pdest != 0.0)
                    if not (self.nq > 5 or ((not cond_b) and self.irflag == 0)):
                        rh2 = 0.0
                        rh1 = 0.0
                        nqm2 = 0
                        if not cond_b:
                            rh2 = 2.0
                            nqm2 = min(self.nq, self.mxords)
                        else:
                            exsm = 1.0 / float(self.l)
                            rh1 = 1.0 / (1.2 * math.pow(dsm, exsm) + 0.0000012)
                            rh1it = 2.0 * rh1
                            pdh = self.pdlast * abs(self.h)
                            if pdh * rh1 > 0.00001:
                                rh1it = sm1[self.nq - 1] / pdh
                            rh1 = min(rh1, rh1it)
                            if self.nq > self.mxords:
                                nqm2 = self.mxords
                                lm2 = self.mxords + 1
                                exm2 = 1.0 / float(lm2)
                                dm2 = _vmnorm(n, yh, lm2 * nyh, ewt) / self.cm2[self.mxords - 1]
                                rh2 = 1.0 / (1.2 * math.pow(dm2, exm2) + 0.0000012)
                            else:
                                dm2 = dsm * (self.cm1[self.nq - 1] / self.cm2[self.nq - 1])
                                rh2 = 1.0 / (1.2 * math.pow(dm2, exsm) + 0.0000012)
                                nqm2 = self.nq
                        if (cond_b and rh2 >= self.ratio * rh1) or (not cond_b):
                            rh = rh2
                            self.icount = 20
                            self.meth = 2
                            self.miter = self.jtyp
                            self.pdlast = 0.0
                            self.nq = nqm2
                            self.l = self.nq + 1
                            rh = max(rh, self.hmin / abs(self.h))
                            self._adjust_step(rh)
                            switched = True
                else:
                    exsm = 1.0 / float(self.l)
                    dm1 = 0.0
                    rh1 = 0.0
                    nqm1 = 0
                    exm1 = 0.0
                    if self.mxordn >= self.nq:
                        dm1 = dsm * (self.cm2[self.nq - 1] / self.cm1[self.nq - 1])
                        rh1 = 1.0 / (1.2 * math.pow(dm1, exsm) + 0.0000012)
                        nqm1 = self.nq
                        exm1 = exsm
                    else:
                        nqm1 = self.mxordn
                        lm1 = self.mxordn + 1
                        exm1 = 1.0 / float(lm1)
                        dm1 = _vmnorm(n, yh, lm1 * nyh, ewt) / self.cm1[self.mxordn - 1]
                        rh1 = 1.0 / (1.2 * math.pow(dm1, exm1) + 0.0000012)
                    rh1it = 2.0 * rh1
                    pdh = self.pdnorm * abs(self.h)
                    if pdh * rh1 > 0.00001:
                        rh1it = sm1[nqm1 - 1] / pdh
                    rh1 = min(rh1, rh1it)
                    rh2 = 1.0 / (1.2 * math.pow(dsm, exsm) + 0.0000012)
                    if rh1 * self.ratio >= 5.0 * rh2:
                        dm1 = math.pow(max(0.001, rh1), exm1) * dm1
                        if dm1 > 1000.0 * self.uround * pnorm:
                            rh = rh1
                            self.icount = 20
                            self.meth = 1
                            self.miter = 0
                            self.pdlast = 0.0
                            self.nq = nqm1
                            self.l = self.nq + 1
                            rh = max(rh, self.hmin / abs(self.h))
                            self._adjust_step(rh)
                            switched = True
                if switched:
                    break
                # 488
                self.ialth -= 1
                if self.ialth == 0:
                    rhup = 0.0
                    if self.l != self.lmax:
                        for i in range(n):
                            savf[i] = acor[i] - yh[i + (self.lmax - 1) * nyh]
                        dup = _vmnorm(n, savf, 0, ewt) / self.tesco[2 + (self.nq - 1) * 3]
                        exup = 1.0 / (float(self.l) + 1.0)
                        rhup = 1.0 / (1.4 * math.pow(dup, exup) + 0.0000014)
                else:
                    if not (self.ialth > 1 or self.l == self.lmax):
                        for i in range(n):
                            yh[i + (self.lmax - 1) * nyh] = acor[i]
                    self._finish_stoda(False)
                    return
            # 540
            exsm = 1.0 / float(self.l)
            rhsm = 1.0 / (1.2 * math.pow(dsm, exsm) + 0.0000012)
            rhdn = 0.0
            if self.nq != 1:
                ddn = _vmnorm(n, yh, (self.l - 1) * nyh, ewt) / self.tesco[(self.nq - 1) * 3]
                exdn = 1.0 / self.nq
                rhdn = 1.0 / (1.3 * math.pow(ddn, exdn) + 0.0000013)
            if self.meth == 1:
                pdh = max(abs(self.h) * self.pdlast, 0.000001)
                if self.l < self.lmax:
                    rhup = min(rhup, sm1[self.l - 1] / pdh)
                rhsm = min(rhsm, sm1[self.nq - 1] / pdh)
                if self.nq > 1:
                    rhdn = min(rhdn, sm1[self.nq - 2] / pdh)
                self.pdest = 0.0
            go_up = False
            if rhsm >= rhup:
                if rhsm >= rhdn:
                    newq = self.nq
                    rh = rhsm
                else:
                    newq = self.nq - 1
                    rh = rhdn
                    if self.kflag < 0 and rh > 1.0:
                        rh = 1.0
            else:
                if rhup > rhdn:
                    go_up = True
                else:
                    newq = self.nq - 1
                    rh = rhdn
                    if self.kflag < 0 and rh > 1.0:
                        rh = 1.0
            if go_up:
                newq = self.l
                rh = rhup
                if rh < 1.1:
                    self.ialth = 3
                    reset_rmax = False
                    break
                r = self.el[self.l - 1] / float(self.l)
                for i in range(n):
                    yh[i + newq * nyh] = acor[i] * r
                self.nq = newq
                self.l = self.nq + 1
                self._reset()
                rh = max(rh, self.hmin / abs(self.h))
                self._adjust_step(rh)
                reset_rmax = True
                break
            bypass = (self.meth == 1) and (rh * pdh * 1.00001 >= sm1[newq - 1])
            if (not bypass) and self.kflag == 0 and rh < 1.1:
                self.ialth = 3
                reset_rmax = False
                break
            if self.kflag <= -2:
                rh = min(rh, 0.2)
            if newq != self.nq:
                self.nq = newq
                self.l = self.nq + 1
                self._reset()
            rh = max(rh, self.hmin / abs(self.h))
            self._adjust_step(rh)
            if self.kflag == 0:
                reset_rmax = True
                break
            self._predict()
            pnorm = _vmnorm(n, yh, 0, ewt)
        self._finish_stoda(reset_rmax)

    def intdy(self, t: float, k: int, dky: np.ndarray[float, 1]) -> int:
        n = self.n
        nyh = self.nyh
        yh = self.yh
        hu = self.hu
        if k < 0 or k > self.nq:
            return -1
        tp = self.tn - hu - 100.0 * self.uround * (self.tn + hu)
        if (t - tp) * (t - self.tn) > 0.0:
            return -2
        s = (t - self.tn) / self.h
        ic = 1
        if k != 0:
            for jj in range(self.l - k, self.nq + 1):
                ic = ic * jj
        c = float(ic)
        for i in range(n):
            dky[i] = c * yh[i + (self.l - 1) * nyh]
        if k != self.nq:
            for jb in range(1, self.nq - k + 1):
                if k != 0:
                    for jj in range(self.nq - jb + 1 - k, self.nq - jb + 1):
                        ic = ic * jj
                c = float(ic)
                for i in range(n):
                    dky[i] = c * yh[i + (self.nq - jb) * nyh] + s * dky[i]
            if k == 0:
                return 0
        r = math.pow(self.h, float(-k))
        for i in range(n):
            dky[i] = r * dky[i]
        return 0

    def _mark_error(self, istate: int) -> int:
        if self.illin == 5:
            return -8
        self.illin += 1
        return -3

    def _setup_opts(self, first: bool) -> int:
        # optional inputs; returns 0 or an error istate
        if self.iopt == 1:
            self.mxstep = self.opt_mxstep
            if self.mxstep < 0:
                return -3
            if self.mxstep == 0:
                self.mxstep = 500
            self.mxhnil = self.opt_mxhnil
            if self.mxhnil < 0:
                return -3
            if self.mxhnil == 0:
                self.mxhnil = 10
            if first:
                self.mxordn = self.opt_mxordn
                if self.mxordn < 0:
                    return -3
                if self.mxordn == 0:
                    self.mxordn = 100
                self.mxordn = min(self.mxordn, 12)
                self.mxords = self.opt_mxords
                if self.mxords < 0:
                    return -3
                if self.mxords == 0:
                    self.mxords = 100
                self.mxords = min(self.mxords, 5)
            hmax = self.opt_hmax
            if hmax < 0.0:
                return -3
            self.hmxi = 0.0
            if hmax > 0.0:
                self.hmxi = 1.0 / hmax
            self.hmin = self.opt_hmin
            if self.hmin < 0.0:
                return -3
        else:
            self.mxstep = 500
            self.mxhnil = 10
            self.hmxi = 0.0
            self.hmin = 0.0
            if first:
                self.mxordn = 12
                self.mxords = 5
        return 0

    def _ret_yh(self, y: np.ndarray[float, 1]):
        for i in range(self.n):
            y[i] = self.yh[i]

    def _exit(self):
        self.illin = 0

    # Port of lsoda(): advances y in place. Returns (t, istate).
    def run(self, f, jac, y: np.ndarray[float, 1], t: float, tout: float,
            itask: int, istate: int):
        ihit = False
        tcrit = 0.0
        n = y.shape[0]
        if istate < 1 or istate > 3:
            return t, self._mark_error(istate)
        if itask < 1 or itask > 5:
            return t, self._mark_error(istate)
        if istate != 1 and self.init == 0:
            return t, self._mark_error(istate)
        initial_jump = True
        if istate == 1:
            self.init = 0
            if tout == t:
                self.ntrep += 1
                if self.ntrep < 5:
                    return t, istate
                return t, -8
            self.ntrep = 0
            if n <= 0:
                return t, self._mark_error(istate)
            self.n = n
            jt = self.jt
            if jt == 3 or jt < 1 or jt > 5:
                return t, self._mark_error(istate)
            self.jtyp = jt
            if jt > 2:
                if self.ml < 0 or self.ml >= n or self.mu < 0 or self.mu >= n:
                    return t, self._mark_error(istate)
            h0 = 0.0
            if self.iopt == 1:
                h0 = self.opt_h0
            if self._setup_opts(True) != 0:
                return t, self._mark_error(istate)
            if self.iopt == 1 and (tout - t) * h0 < 0.0:
                return t, self._mark_error(istate)
            self.meth = 1
            self.nyh = n
            self._alloc(n)
            for i in range(n):
                if self.rtol[i] < 0.0 or self.atol[i] < 0.0:
                    return t, self._mark_error(istate)
            self.uround = _UROUND
            self.tn = t
            self.tsw = t
            self.maxord = self.mxordn
            if itask == 4 or itask == 5:
                tcrit = self.tcrit
                if (tcrit - tout) * (tout - t) < 0.0:
                    return t, self._mark_error(istate)
                if h0 != 0.0 and (t + h0 - tcrit) * h0 > 0.0:
                    h0 = tcrit - t
            self.jstart = 0
            self.nhnil = 0
            self.nst = 0
            self.nje = 0
            self.nslast = 0
            self.hu = 0.0
            self.nqu = 0
            self.mused = 0
            self.miter = 0
            self.ccmax = 0.3
            self.maxcor = 3
            self.msbp = 20
            self.mxncf = 10
            f0 = np.zeros(n)
            self._call_f(f, t, y, f0)
            self.nfe = 1
            for i in range(n):
                self.yh[i] = y[i]
            self.nq = 1
            self.h = 1.0
            self._ewset(self.yh)
            for i in range(n):
                if self.ewt[i] <= 0.0:
                    return t, self._mark_error(istate)
                self.ewt[i] = 1.0 / self.ewt[i]
            if h0 == 0.0:
                tdist = abs(tout - t)
                w0 = max(abs(t), abs(tout))
                if tdist < 2.0 * self.uround * w0:
                    return t, self._mark_error(istate)
                tol = self.rtol[0]
                if self.rtol_vec:
                    for i in range(n):
                        tol = max(tol, self.rtol[i])
                if tol <= 0.0:
                    for i in range(n):
                        if abs(y[i]) != 0.0:
                            tol = max(tol, self.atol[i] / abs(y[i]))
                tol = max(tol, 100.0 * self.uround)
                tol = min(tol, 0.001)
                sm = _vmnorm(n, f0, 0, self.ewt)
                sm = 1.0 / (tol * w0 * w0) + (tol * sm * sm)
                h0 = 1.0 / math.sqrt(sm)
                h0 = min(h0, tdist)
                h0 = math.copysign(h0, tout - t)
            rh = abs(h0) * self.hmxi
            if rh > 1.0:
                h0 = h0 / rh
            self.h = h0
            for i in range(n):
                self.yh[i + n] = h0 * f0[i]
            initial_jump = False
        elif istate == 3:
            self.ntrep = 0
            if n <= 0 or n > self.n:
                return t, self._mark_error(istate)
            self.n = n
            if self._setup_opts(False) != 0:
                return t, self._mark_error(istate)
            for i in range(n):
                if self.rtol[i] < 0.0 or self.atol[i] < 0.0:
                    return t, self._mark_error(istate)
            self.jstart = -1
        if istate != 1:
            self.nslast = self.nst
            if itask == 1:
                if (self.tn - tout) * self.h >= 0.0:
                    if self.intdy(tout, 0, y) != 0:
                        return t, self._mark_error(istate)
                    self._exit()
                    return tout, 2
            elif itask == 3:
                tp = self.tn - self.hu * (1.0 + 100.0 * self.uround)
                if (tp - tout) * self.h > 0.0:
                    return t, self._mark_error(istate)
                if (self.tn - tout) * self.h >= 0.0:
                    self._ret_yh(y)
                    self._exit()
                    return self.tn, 2
            elif itask == 4:
                tcrit = self.tcrit
                if (self.tn - tcrit) * self.h > 0.0:
                    return t, self._mark_error(istate)
                if (tcrit - tout) * self.h < 0.0:
                    return t, self._mark_error(istate)
                if (self.tn - tout) * self.h < 0.0:
                    hmx = abs(self.tn) + abs(self.h)
                    ihit = abs(self.tn - tcrit) <= 100.0 * self.uround * hmx
                    if ihit:
                        self._ret_yh(y)
                        self._exit()
                        return tcrit, 2
                    tnext = self.tn + self.h * (1.0 + 4.0 * self.uround)
                    if (tnext - tcrit) * self.h > 0.0:
                        self.h = (tcrit - self.tn) * (1.0 - 4.0 * self.uround)
                        if istate == 2 and self.jstart >= 0:
                            self.jstart = -2
                else:
                    if self.intdy(tout, 0, y) != 0:
                        return t, self._mark_error(istate)
                    self._exit()
                    return tout, 2
            elif itask == 5:
                tcrit = self.tcrit
                if (self.tn - tcrit) * self.h > 0.0:
                    return t, self._mark_error(istate)
                hmx = abs(self.tn) + abs(self.h)
                ihit = abs(self.tn - tcrit) <= 100.0 * self.uround * hmx
                if ihit:
                    self._ret_yh(y)
                    self._exit()
                    return tcrit, 2
                tnext = self.tn + self.h * (1.0 + 4.0 * self.uround)
                if (tnext - tcrit) * self.h > 0.0:
                    self.h = (tcrit - self.tn) * (1.0 - 4.0 * self.uround)
                    if istate == 2 and self.jstart >= 0:
                        self.jstart = -2
        elif itask == 4 or itask == 5:
            tcrit = self.tcrit
        return self._main_loop(f, jac, y, t, tout, itask, initial_jump, tcrit)

    def _main_loop(self, f, jac, y: np.ndarray[float, 1], t: float, tout: float, itask: int,
                   initial_jump: bool, tcrit: float):
        n = self.n
        ij = initial_jump
        while True:
            if ij:
                if self.nst - self.nslast >= self.mxstep:
                    self._ret_yh(y)
                    self._exit()
                    return self.tn, -1
                self._ewset(self.yh)
                for i in range(n):
                    if self.ewt[i] <= 0.0:
                        self._ret_yh(y)
                        self._exit()
                        return self.tn, -6
                    self.ewt[i] = 1.0 / self.ewt[i]
            ij = True
            tolsf = self.uround * _vmnorm(n, self.yh, 0, self.ewt)
            if tolsf > 0.01:
                tolsf = tolsf * 200.0
                self.tolsf = tolsf
                if self.nst == 0:
                    return t, self._mark_error(2)
                self._ret_yh(y)
                self._exit()
                return self.tn, -2
            if abs(self.h) <= np.nextafter(abs(self.tn), 2 * abs(self.tn)) - abs(self.tn):
                self.nhnil += 1
            self._stoda(y, f, jac)
            if self.kflag == -1 or self.kflag == -2:
                ist = -4 if self.kflag == -1 else -5
                big = 0.0
                imxer = 0
                for i in range(n):
                    size = abs(self.acor[i] * self.ewt[i])
                    if size > big:
                        big = size
                        imxer = i + 1
                self.imxer = imxer
                self._ret_yh(y)
                self._exit()
                return self.tn, ist
            if self.kflag == -3:
                return self.tn, -3
            self.init = 1
            if self.meth != self.mused:
                self.tsw = self.tn
                self.maxord = self.mxords if self.meth == 2 else self.mxordn
                self.jstart = -1
            if itask == 1:
                if (self.tn - tout) * self.h < 0.0:
                    continue
                if self.intdy(tout, 0, y) != 0:
                    return t, self._mark_error(2)
                self._exit()
                return tout, 2
            elif itask == 2:
                self._ret_yh(y)
                self._exit()
                return self.tn, 2
            elif itask == 3:
                if (self.tn - tout) * self.h >= 0.0:
                    self._ret_yh(y)
                    self._exit()
                    return self.tn, 2
                continue
            elif itask == 4:
                if (self.tn - tout) * self.h >= 0.0:
                    self.intdy(tout, 0, y)
                    self._exit()
                    return tout, 2
                hmx = abs(self.tn) + abs(self.h)
                ihit = abs(self.tn - tcrit) <= 100.0 * self.uround * hmx
                if ihit:
                    self._ret_yh(y)
                    self._exit()
                    return tcrit, 2
                tnext = self.tn + self.h * (1.0 + 4.0 * self.uround)
                if (tnext - tcrit) * self.h <= 0.0:
                    continue
                self.h = (tcrit - self.tn) * (1.0 - 4.0 * self.uround)
                if self.jstart >= 0:
                    self.jstart = -2
                continue
            else:
                hmx = abs(self.tn) + abs(self.h)
                ihit = abs(self.tn - tcrit) <= 100.0 * self.uround * hmx
                self._ret_yh(y)
                self._exit()
                if ihit:
                    return tcrit, 2
                return self.tn, 2
        return t, 2

_ODE_MSGS = {2: "Integration successful.",
             1: "Nothing was done; the integration time was 0.",
             -1: "Excess work done on this call (perhaps wrong Dfun type).",
             -2: "Excess accuracy requested (tolerances too small).",
             -3: "Illegal input detected (internal error).",
             -4: "Repeated error test failures (internal error).",
             -5: "Repeated convergence failures (perhaps bad Jacobian or tolerances).",
             -6: "Error weight became zero during problem.",
             -7: "Internal workspace insufficient to finish (internal error).",
             -8: "Run terminated (internal error)."}

def _tol_vec(tol, n: int, default: float):
    # returns (vector, is_vector)
    out = np.empty(n)
    if tol is None:
        for i in range(n):
            out[i] = default
        return out, False
    ta = np.asarray(tol).astype(float)
    if ta.ndim == 0:
        v = ta.item()
        for i in range(n):
            out[i] = v
        return out, False
    if ta.size != n:
        raise ValueError("Tolerances must be an array of the same length as the\n     number of equations or a scalar.")
    tf = ta.reshape(n)
    for i in range(n):
        out[i] = tf[i]
    return out, True
