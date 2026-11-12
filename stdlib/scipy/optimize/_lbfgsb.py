# Ported from scipy 1.18.1 (BSD-3-Clause)

# L-BFGS-B (Byrd, Lu, Nocedal, Zhu; version 3.0 with the Morales-Nocedal
# subspace fix), translated from scipy's C implementation (lbfgsb.c).
# Matrices are kept as flat column-major arrays exactly as in the C code.

import math
import numpy as np
from ._common import *
from ._common import (_ScalarFunction, _approx_grad, _approx_jac, _bound_val, _call_scalar, _call_vec, _e1, _e2, _eps_for_method, _epsilon, _fd_steps, _has_opt, _norm2, _norm_inf, _old_bound_to_new, _opt_bool, _opt_float, _opt_int, _prepare_bounds_pair, _scalar_eps, _sign, _sqrt, _standardize_bounds, _status_message_maxfev, _status_message_maxiter, _status_message_nan, _status_message_oob, _status_message_pr_loss, _status_message_success, _vec)
from ._linesearch import _DCSRCH
from ._callback import _cb_halt, _STOP_MSG

_EPSMCH = 2.220446049250313e-016

def _cp(src: np.ndarray[float, 1], so: int, dst: np.ndarray[float, 1], do: int,
        n: int):
    for i in range(n):
        dst[do + i] = src[so + i]

def _dot(a: np.ndarray[float, 1], ao: int, b: np.ndarray[float, 1], bo: int,
         n: int) -> float:
    s = 0.0
    for i in range(n):
        s += a[ao + i] * b[bo + i]
    return s

def _dnrm2(x: np.ndarray[float, 1], n: int) -> float:
    # reference BLAS dnrm2 (scaled sum of squares)
    if n < 1:
        return 0.0
    if n == 1:
        return abs(x[0])
    scale = 0.0
    ssq = 1.0
    for i in range(n):
        if x[i] != 0.0:
            absxi = abs(x[i])
            if scale < absxi:
                ssq = 1.0 + ssq * (scale / absxi) ** 2
                scale = absxi
            else:
                ssq += (absxi / scale) ** 2
    return scale * math.sqrt(ssq)

def _potrf(a: np.ndarray[float, 1], off: int, lda: int, n: int) -> int:
    # upper Cholesky A = U^T U (column-major, in place); returns info
    for j in range(n):
        s = a[off + j + lda * j]
        for k in range(j):
            s -= a[off + k + lda * j] * a[off + k + lda * j]
        if s <= 0.0 or s != s:
            a[off + j + lda * j] = s
            return j + 1
        ajj = math.sqrt(s)
        a[off + j + lda * j] = ajj
        for i in range(j + 1, n):
            t = a[off + j + lda * i]
            for k in range(j):
                t -= a[off + k + lda * j] * a[off + k + lda * i]
            a[off + j + lda * i] = t / ajj
    return 0

def _trtrs(trans: bool, a: np.ndarray[float, 1], aoff: int, lda: int, n: int,
           nrhs: int, b: np.ndarray[float, 1], boff: int, ldb: int) -> int:
    # solve U^T x = b (trans) or U x = b with U upper triangular
    for i in range(n):
        if a[aoff + i + lda * i] == 0.0:
            return i + 1
    for r in range(nrhs):
        bo = boff + ldb * r
        if trans:
            for i in range(n):
                s = b[bo + i]
                for k in range(i):
                    s -= a[aoff + k + lda * i] * b[bo + k]
                b[bo + i] = s / a[aoff + i + lda * i]
        else:
            for i in range(n - 1, -1, -1):
                s = b[bo + i]
                for k in range(i + 1, n):
                    s -= a[aoff + i + lda * k] * b[bo + k]
                b[bo + i] = s / a[aoff + i + lda * i]
    return 0

def _bmv(m: int, sy: np.ndarray[float, 1], wt: np.ndarray[float, 1], col: int,
         v: np.ndarray[float, 1], vo: int, p: np.ndarray[float, 1], po: int) -> int:
    if col == 0:
        return 0
    p[po + col] = v[vo + col]
    for i in range(1, col):
        i2 = col + i
        ssum = 0.0
        for k in range(i):
            ssum = ssum + sy[i + m * k] * v[vo + k] / sy[k + m * k]
        p[po + i2] = v[vo + i2] + ssum
    info = _trtrs(True, wt, 0, m, col, 1, p, po + col, col)
    if info != 0:
        return info
    for i in range(col):
        p[po + i] = v[vo + i] / math.sqrt(sy[i + m * i])
    info = _trtrs(False, wt, 0, m, col, 1, p, po + col, col)
    if info != 0:
        return info
    for i in range(col):
        p[po + i] = -p[po + i] / math.sqrt(sy[i + m * i])
    for i in range(col):
        ssum = 0.0
        for k in range(i + 1, col):
            ssum = ssum + sy[k + m * i] * p[po + col + k] / sy[i + m * i]
        p[po + i] = p[po + i] + ssum
    return 0

def _projgr(n: int, l, u, nbd, x, g) -> float:
    sbgnrm = 0.0
    for i in range(n):
        gi = g[i]
        if gi != gi:
            return gi
        if nbd[i] != 0:
            if gi < 0.0:
                if nbd[i] >= 2:
                    gi = max(x[i] - u[i], gi)
            else:
                if nbd[i] <= 2:
                    gi = min(x[i] - l[i], gi)
        sbgnrm = max(sbgnrm, abs(gi))
    return sbgnrm

def _hpsolb(n: int, t: np.ndarray[float, 1], iorder: np.ndarray[int, 1],
            iheap: int):
    if iheap == 0:
        for k in range(2, n + 2):
            ddum = t[k - 1]
            indxin = iorder[k - 1]
            i = k
            while i > 1:
                j = i // 2
                if ddum < t[j - 1]:
                    t[i - 1] = t[j - 1]
                    iorder[i - 1] = iorder[j - 1]
                    i = j
                else:
                    break
            t[i - 1] = ddum
            iorder[i - 1] = indxin
    if n > 0:
        i = 1
        out = t[0]
        indxout = iorder[0]
        ddum = t[n]
        indxin = iorder[n]
        while True:
            j = i + i
            if j <= n:
                if t[j] < t[j - 1]:
                    j += 1
                if t[j - 1] < ddum:
                    t[i - 1] = t[j - 1]
                    iorder[i - 1] = iorder[j - 1]
                    i = j
                else:
                    break
            else:
                break
        t[i - 1] = ddum
        iorder[i - 1] = indxin
        t[n] = out
        iorder[n] = indxout

class _LB:
    # all L-BFGS-B work arrays and saved scalars
    n: int
    m: int
    x: np.ndarray[float, 1]
    l: np.ndarray[float, 1]
    u: np.ndarray[float, 1]
    nbd: np.ndarray[int, 1]
    g: np.ndarray[float, 1]
    f: float
    ws: np.ndarray[float, 1]
    wy: np.ndarray[float, 1]
    sy: np.ndarray[float, 1]
    ss: np.ndarray[float, 1]
    wt: np.ndarray[float, 1]
    wn: np.ndarray[float, 1]
    snd: np.ndarray[float, 1]
    z: np.ndarray[float, 1]
    r: np.ndarray[float, 1]
    d: np.ndarray[float, 1]
    t: np.ndarray[float, 1]
    xp: np.ndarray[float, 1]
    wa: np.ndarray[float, 1]
    index: np.ndarray[int, 1]
    iwhere: np.ndarray[int, 1]
    indx2: np.ndarray[int, 1]
    theta: float
    col: int
    head: int
    itail: int
    iupdat: int
    updatd: bool
    nseg: int
    nfree: int
    nenter: int
    ileave: int
    wrk: bool

    def __init__(self, x: np.ndarray[float, 1], l: np.ndarray[float, 1],
                 u: np.ndarray[float, 1], nbd: np.ndarray[int, 1], m: int):
        n = x.size
        self.n = n
        self.m = m
        self.x = x
        self.l = l
        self.u = u
        self.nbd = nbd
        self.g = np.zeros(n)
        self.f = 0.0
        self.ws = np.zeros(m * n)
        self.wy = np.zeros(m * n)
        self.sy = np.zeros(m * m)
        self.ss = np.zeros(m * m)
        self.wt = np.zeros(m * m)
        self.wn = np.zeros(4 * m * m)
        self.snd = np.zeros(4 * m * m)
        self.z = np.zeros(n)
        self.r = np.zeros(n)
        self.d = np.zeros(n)
        self.t = np.zeros(n)
        self.xp = np.zeros(n)
        self.wa = np.zeros(8 * m)
        self.index = np.zeros(n, int)
        self.iwhere = np.zeros(n, int)
        self.indx2 = np.zeros(n, int)
        self.theta = 1.0
        self.col = 0
        self.head = 0
        self.itail = 0
        self.iupdat = 0
        self.updatd = False
        self.nseg = 0
        self.nfree = n
        self.nenter = 0
        self.ileave = 0
        self.wrk = False

    def reset(self):
        self.col = 0
        self.head = 0
        self.theta = 1.0
        self.iupdat = 0
        self.updatd = False

def _active(s: _LB):
    n = s.n
    prjctd = False
    cnstnd = False
    boxed = True
    for i in range(n):
        if s.nbd[i] > 0:
            if s.nbd[i] <= 2 and s.x[i] <= s.l[i]:
                if s.x[i] < s.l[i]:
                    prjctd = True
                    s.x[i] = s.l[i]
            elif s.nbd[i] >= 2 and s.x[i] >= s.u[i]:
                if s.x[i] > s.u[i]:
                    prjctd = True
                    s.x[i] = s.u[i]
    for i in range(n):
        if s.nbd[i] != 2:
            boxed = False
        if s.nbd[i] == 0:
            s.iwhere[i] = -1
        else:
            cnstnd = True
            if s.nbd[i] == 2 and s.u[i] - s.l[i] <= 0.0:
                s.iwhere[i] = 3
            else:
                s.iwhere[i] = 0
    return prjctd, cnstnd, boxed

def _cauchy(s: _LB, sbgnrm: float) -> int:
    # generalized Cauchy point -> s.z (xcp); returns info
    n = s.n
    m = s.m
    x = s.x
    l = s.l
    u = s.u
    nbd = s.nbd
    g = s.g
    iorder = s.indx2
    iwhere = s.iwhere
    t = s.t
    d = s.d
    xcp = s.z
    wy = s.wy
    ws = s.ws
    theta = s.theta
    col = s.col
    head = s.head
    wa = s.wa
    P = 0
    C = 2 * m
    WBP = 4 * m
    V = 6 * m
    f1 = 0.0
    tl = 0.0
    tu = 0.0
    if sbgnrm <= 0.0:
        _cp(x, 0, xcp, 0, n)
        return 0
    bnded = True
    nfree = n
    nbreak = -1
    ibkmin = 0
    bkmin = 0.0
    col2 = 2 * col
    for i in range(col2):
        wa[P + i] = 0.0
    for i in range(n):
        neggi = -g[i]
        if iwhere[i] != 3 and iwhere[i] != -1:
            if nbd[i] <= 2:
                tl = x[i] - l[i]
            if nbd[i] >= 2:
                tu = u[i] - x[i]
            xlower = nbd[i] <= 2 and tl <= 0.0
            xupper = nbd[i] >= 2 and tu <= 0.0
            iwhere[i] = 0
            if xlower:
                if neggi <= 0.0:
                    iwhere[i] = 1
            elif xupper:
                if neggi >= 0.0:
                    iwhere[i] = 2
            else:
                if abs(neggi) <= 0.0:
                    iwhere[i] = -3
        pointr = head
        if iwhere[i] != 0 and iwhere[i] != -1:
            d[i] = 0.0
        else:
            d[i] = neggi
            f1 = f1 - neggi * neggi
            for j in range(col):
                wa[P + j] = wa[P + j] + wy[i + n * pointr] * neggi
                wa[P + col + j] = wa[P + col + j] + ws[i + n * pointr] * neggi
                pointr = (pointr + 1) % m
            if nbd[i] <= 2 and nbd[i] != 0 and neggi < 0.0:
                nbreak += 1
                iorder[nbreak] = i
                t[nbreak] = tl / (-neggi)
                if nbreak == 0 or t[nbreak] < bkmin:
                    bkmin = t[nbreak]
                    ibkmin = nbreak
            elif nbd[i] >= 2 and neggi > 0.0:
                nbreak += 1
                iorder[nbreak] = i
                t[nbreak] = tu / neggi
                if nbreak == 0 or t[nbreak] < bkmin:
                    bkmin = t[nbreak]
                    ibkmin = nbreak
            else:
                nfree -= 1
                iorder[nfree] = i
                if abs(neggi) > 0.0:
                    bnded = False
    if theta != 1.0:
        for j in range(col):
            wa[P + col + j] *= theta
    _cp(x, 0, xcp, 0, n)
    if nbreak == -1 and nfree == n:
        return 0
    for j in range(col2):
        wa[C + j] = 0.0
    f2 = -theta * f1
    f2_org = f2
    if col > 0:
        info = _bmv(m, s.sy, s.wt, col, wa, P, wa, V)
        if info != 0:
            return info
        f2 = f2 - _dot(wa, V, wa, P, col2)
    dtm = -f1 / f2
    tsum = 0.0
    s.nseg = 1
    nleft = nbreak
    iter = 0
    tj = 0.0
    while nbreak > -1:
        tj0 = tj
        ibp = 0
        if iter == 0:
            tj = bkmin
            ibp = iorder[ibkmin]
        else:
            if iter == 1:
                if ibkmin != nbreak:
                    t[ibkmin] = t[nbreak]
                    iorder[ibkmin] = iorder[nbreak]
            _hpsolb(nleft, t, iorder, iter - 1)
            tj = t[nleft]
            ibp = iorder[nleft]
        dt = tj - tj0
        if dtm < dt:
            if dtm <= 0.0:
                dtm = 0.0
            tsum = tsum + dtm
            for i in range(n):
                xcp[i] += tsum * d[i]
            if col > 0:
                for j in range(col2):
                    wa[C + j] += dtm * wa[P + j]
            return 0
        tsum = tsum + dt
        nleft -= 1
        iter += 1
        dibp = d[ibp]
        d[ibp] = 0.0
        zibp = 0.0
        if dibp > 0.0:
            zibp = u[ibp] - x[ibp]
            xcp[ibp] = u[ibp]
            iwhere[ibp] = 2
        else:
            zibp = l[ibp] - x[ibp]
            xcp[ibp] = l[ibp]
            iwhere[ibp] = 1
        if nleft == -1 and nbreak == n:
            dtm = dt
            if col > 0:
                for j in range(col2):
                    wa[C + j] += dtm * wa[P + j]
            return 0
        s.nseg += 1
        dibp2 = dibp * dibp
        f1 = f1 + dt * f2 + dibp2 - theta * dibp * zibp
        f2 = f2 - theta * dibp2
        if col > 0:
            for j in range(col2):
                wa[C + j] += dt * wa[P + j]
            pointr = head
            for j in range(col):
                wa[WBP + j] = wy[ibp + n * pointr]
                wa[WBP + col + j] = theta * ws[ibp + n * pointr]
                pointr = (pointr + 1) % m
            info = _bmv(m, s.sy, s.wt, col, wa, WBP, wa, V)
            if info != 0:
                return info
            wmc = _dot(wa, C, wa, V, col2)
            wmp = _dot(wa, P, wa, V, col2)
            wmw = _dot(wa, WBP, wa, V, col2)
            for j in range(col2):
                wa[P + j] += -dibp * wa[WBP + j]
            f1 = f1 + dibp * wmc
            f2 = f2 + dibp * 2.0 * wmp - dibp2 * wmw
        f2 = max(_EPSMCH * f2_org, f2)
        if nleft >= 0:
            dtm = -f1 / f2
        elif bnded:
            dtm = 0.0
            break
        else:
            dtm = -f1 / f2
            break
    if dtm <= 0.0:
        dtm = 0.0
    tsum = tsum + dtm
    for i in range(n):
        xcp[i] += tsum * d[i]
    if col > 0:
        for j in range(col2):
            wa[C + j] += dtm * wa[P + j]
    return 0

def _freev(s: _LB, cnstnd: bool, iter: int):
    n = s.n
    idx = s.index
    idx2 = s.indx2
    iwhere = s.iwhere
    s.nenter = 0
    s.ileave = n
    if iter > 0 and cnstnd:
        for i in range(s.nfree):
            k = idx[i]
            if iwhere[k] > 0:
                s.ileave -= 1
                idx2[s.ileave] = k
        for i in range(s.nfree, n):
            k = idx[i]
            if iwhere[k] <= 0:
                idx2[s.nenter] = k
                s.nenter += 1
    s.wrk = (s.ileave < n) or (s.nenter > 0) or s.updatd
    s.nfree = 0
    iact = n
    for i in range(n):
        if iwhere[i] <= 0:
            idx[s.nfree] = i
            s.nfree += 1
        else:
            iact -= 1
            idx[iact] = i

def _cmprlb(s: _LB, cnstnd: bool) -> int:
    n = s.n
    m = s.m
    col = s.col
    r = s.r
    if (not cnstnd) and col > 0:
        for i in range(n):
            r[i] = -s.g[i]
    else:
        for i in range(s.nfree):
            k = s.index[i]
            r[i] = -s.theta * (s.z[k] - s.x[k]) - s.g[k]
        info = _bmv(m, s.sy, s.wt, col, s.wa, 2 * m, s.wa, 0)
        if info != 0:
            return -8
        pointr = s.head
        for j in range(col):
            a1 = s.wa[j]
            a2 = s.theta * s.wa[col + j]
            for i in range(s.nfree):
                k = s.index[i]
                r[i] = r[i] + s.wy[k + n * pointr] * a1 + s.ws[k + n * pointr] * a2
            pointr = (pointr + 1) % m
    return 0

def _formk(s: _LB) -> int:
    n = s.n
    m = s.m
    nsub = s.nfree
    ind = s.index
    nenter = s.nenter
    ileave = s.ileave
    indx2 = s.indx2
    iupdat = s.iupdat
    wn = s.wn
    wn1 = s.snd
    ws = s.ws
    wy = s.wy
    sy = s.sy
    theta = s.theta
    col = s.col
    head = s.head
    m2 = 2 * m
    upcl = col
    if s.updatd:
        if iupdat > m:
            for jy in range(m - 1):
                js = m + jy
                cnt = m - (jy + 1)
                for q in range(cnt):
                    wn1[jy + m2 * jy + q] = wn1[(jy + 1) + m2 * (jy + 1) + q]
                    wn1[js + m2 * js + q] = wn1[(js + 1) + m2 * (js + 1) + q]
                for q in range(m - 1):
                    wn1[m + m2 * jy + q] = wn1[(m + 1) + m2 * (jy + 1) + q]
        pbegin = 0
        pend = nsub
        dbegin = nsub
        dend = n
        iy = col - 1
        is_ = m + col - 1
        ipntr = head + col - 1
        if ipntr >= m:
            ipntr -= m
        jpntr = head
        for jy in range(col):
            js = m + jy
            temp1 = 0.0
            temp2 = 0.0
            temp3 = 0.0
            for k in range(pbegin, pend):
                k1 = ind[k]
                temp1 = temp1 + wy[k1 + n * ipntr] * wy[k1 + n * jpntr]
            for k in range(dbegin, dend):
                k1 = ind[k]
                temp2 = temp2 + ws[k1 + n * ipntr] * ws[k1 + n * jpntr]
                temp3 = temp3 + ws[k1 + n * ipntr] * wy[k1 + n * jpntr]
            wn1[iy + m2 * jy] = temp1
            wn1[is_ + m2 * js] = temp2
            wn1[is_ + m2 * jy] = temp3
            jpntr = (jpntr + 1) % m
        jy = col - 1
        jpntr = head + col - 1
        if jpntr >= m:
            jpntr -= m
        ipntr = head
        for i in range(col):
            is_ = m + i
            temp3 = 0.0
            for k in range(pbegin, pend):
                k1 = ind[k]
                temp3 = temp3 + ws[k1 + n * ipntr] * wy[k1 + n * jpntr]
            ipntr = (ipntr + 1) % m
            wn1[is_ + m2 * jy] = temp3
        upcl = col - 1
    ipntr = head
    for iy in range(upcl):
        is_ = m + iy
        jpntr = head
        for jy in range(iy + 1):
            js = m + jy
            temp1 = 0.0
            temp2 = 0.0
            temp3 = 0.0
            temp4 = 0.0
            for k in range(nenter):
                k1 = indx2[k]
                temp1 = temp1 + wy[k1 + n * ipntr] * wy[k1 + n * jpntr]
                temp2 = temp2 + ws[k1 + n * ipntr] * ws[k1 + n * jpntr]
            for k in range(ileave, n):
                k1 = indx2[k]
                temp3 = temp3 + wy[k1 + n * ipntr] * wy[k1 + n * jpntr]
                temp4 = temp4 + ws[k1 + n * ipntr] * ws[k1 + n * jpntr]
            wn1[iy + m2 * jy] = wn1[iy + m2 * jy] + temp1 - temp3
            wn1[is_ + m2 * js] = wn1[is_ + m2 * js] - temp2 + temp4
            jpntr = (jpntr + 1) % m
        ipntr = (ipntr + 1) % m
    ipntr = head
    for is_ in range(m, m + upcl):
        jpntr = head
        for jy in range(upcl):
            temp1 = 0.0
            temp3 = 0.0
            for k in range(nenter):
                k1 = indx2[k]
                temp1 = temp1 + ws[k1 + n * ipntr] * wy[k1 + n * jpntr]
            for k in range(ileave, n):
                k1 = indx2[k]
                temp3 = temp3 + ws[k1 + n * ipntr] * wy[k1 + n * jpntr]
            if is_ <= jy + m:
                wn1[is_ + m2 * jy] = wn1[is_ + m2 * jy] + temp1 - temp3
            else:
                wn1[is_ + m2 * jy] = wn1[is_ + m2 * jy] - temp1 + temp3
            jpntr = (jpntr + 1) % m
        ipntr = (ipntr + 1) % m
    for iy in range(col):
        is_ = col + iy
        is1 = m + iy
        for jy in range(iy + 1):
            js = col + jy
            js1 = m + jy
            wn[jy + m2 * iy] = wn1[iy + m2 * jy] / theta
            wn[js + m2 * is_] = wn1[is1 + m2 * js1] * theta
        for jy in range(iy):
            wn[jy + m2 * is_] = -wn1[is1 + m2 * jy]
        for jy in range(iy, col):
            wn[jy + m2 * is_] = wn1[is1 + m2 * jy]
        wn[iy + m2 * iy] = wn[iy + m2 * iy] + sy[iy + m * iy]
    info = _potrf(wn, 0, m2, col)
    if info != 0:
        return -1
    col2 = 2 * col
    _trtrs(True, wn, 0, m2, col, col, wn, m2 * col, m2)
    for is_ in range(col, col2):
        for js in range(is_, col2):
            wn[is_ + m2 * js] = wn[is_ + m2 * js] + _dot(wn, m2 * is_, wn, m2 * js, col)
    info = _potrf(wn, col + m2 * col, m2, col)
    if info != 0:
        return -2
    return 0

def _formt(s: _LB) -> int:
    m = s.m
    wt = s.wt
    sy = s.sy
    ss = s.ss
    col = s.col
    theta = s.theta
    for j in range(col):
        wt[m * j] = theta * ss[m * j]
    for i in range(1, col):
        for j in range(i, col):
            k1 = min(i, j)
            ddum = 0.0
            for k in range(k1):
                ddum = ddum + (sy[i + m * k] * sy[j + m * k]) / sy[k + m * k]
            wt[i + m * j] = ddum + theta * ss[i + m * j]
    info = _potrf(wt, 0, m, col)
    if info != 0:
        return -3
    return 0

def _subsm(s: _LB) -> int:
    # uses d := s.r (reduced gradient), x := s.z, xx := s.x, gg := s.g
    n = s.n
    m = s.m
    nsub = s.nfree
    ind = s.index
    l = s.l
    u = s.u
    nbd = s.nbd
    x = s.z
    d = s.r
    xp = s.xp
    ws = s.ws
    wy = s.wy
    theta = s.theta
    xx = s.x
    gg = s.g
    col = s.col
    head = s.head
    wv = s.wa
    wn = s.wn
    if nsub <= 0:
        return 0
    pointr = head
    for i in range(col):
        temp1 = 0.0
        temp2 = 0.0
        for j in range(nsub):
            k = ind[j]
            temp1 += wy[k + n * pointr] * d[j]
            temp2 += ws[k + n * pointr] * d[j]
        wv[i] = temp1
        wv[col + i] = theta * temp2
        pointr = (pointr + 1) % m
    m2 = 2 * m
    col2 = 2 * col
    info = _trtrs(True, wn, 0, m2, col2, 1, wv, 0, m2)
    if info != 0:
        return info
    for i in range(col):
        wv[i] = -wv[i]
    info = _trtrs(False, wn, 0, m2, col2, 1, wv, 0, m2)
    if info != 0:
        return info
    pointr = head
    for jy in range(col):
        js = col + jy
        for i in range(nsub):
            k = ind[i]
            d[i] = d[i] + (wy[k + n * pointr] * wv[jy] / theta) + (ws[k + n * pointr] * wv[js])
        pointr = (pointr + 1) % m
    temp = 1.0 / theta
    for i in range(nsub):
        d[i] *= temp
    iword = 0
    _cp(x, 0, xp, 0, n)
    for i in range(nsub):
        k = ind[i]
        dk = d[i]
        xk = x[k]
        if nbd[k] != 0:
            if nbd[k] == 1:
                x[k] = max(l[k], xk + dk)
                if x[k] == l[k]:
                    iword = 1
            else:
                if nbd[k] == 2:
                    xk = max(l[k], xk + dk)
                    x[k] = min(u[k], xk)
                    if x[k] == l[k] or x[k] == u[k]:
                        iword = 1
                else:
                    if nbd[k] == 3:
                        x[k] = min(u[k], xk + dk)
                        if x[k] == u[k]:
                            iword = 1
        else:
            x[k] = xk + dk
    if iword == 0:
        return 0
    dd_p = 0.0
    for i in range(n):
        dd_p = dd_p + (x[i] - xx[i]) * gg[i]
    if dd_p > 0.0:
        _cp(xp, 0, x, 0, n)
    else:
        return 0
    alpha = 1.0
    temp1 = 1.0
    ibd = 0
    for i in range(nsub):
        k = ind[i]
        dk = d[i]
        if nbd[k] != 0:
            if dk < 0.0 and nbd[k] <= 2:
                temp2 = l[k] - x[k]
                if temp2 >= 0.0:
                    temp1 = 0.0
                elif dk * alpha < temp2:
                    temp1 = temp2 / dk
            elif dk > 0.0 and nbd[k] >= 2:
                temp2 = u[k] - x[k]
                if temp2 <= 0.0:
                    temp1 = 0.0
                elif dk * alpha > temp2:
                    temp1 = temp2 / dk
            if temp1 < alpha:
                alpha = temp1
                ibd = i
    if alpha < 1.0:
        dk = d[ibd]
        k = ind[ibd]
        if dk > 0.0:
            x[k] = u[k]
            d[ibd] = 0.0
        elif dk < 0.0:
            x[k] = l[k]
            d[ibd] = 0.0
    for i in range(nsub):
        k = ind[i]
        x[k] = x[k] + alpha * d[i]
    return 0

def _matupd(s: _LB, rr: float, dr: float, stp: float, dtd: float):
    n = s.n
    m = s.m
    ws = s.ws
    wy = s.wy
    sy = s.sy
    ss = s.ss
    d = s.d
    r = s.r
    iupdat = s.iupdat
    if iupdat <= m:
        s.col = iupdat
        s.itail = (s.head + iupdat - 1) % m
    else:
        s.itail = (s.itail + 1) % m
        s.head = (s.head + 1) % m
    col = s.col
    _cp(d, 0, ws, s.itail * n, n)
    _cp(r, 0, wy, s.itail * n, n)
    s.theta = rr / dr
    if iupdat > m:
        for j in range(1, col):
            for q in range(j):
                ss[m * (j - 1) + q] = ss[1 + m * j + q]
            for q in range(col - j):
                sy[(j - 1) + m * (j - 1) + q] = sy[j + m * j + q]
    pointr = s.head
    for j in range(col - 1):
        sy[col - 1 + m * j] = _dot(d, 0, wy, pointr * n, n)
        ss[j + m * (col - 1)] = _dot(ws, pointr * n, d, 0, n)
        pointr = (pointr + 1) % m
    ss[(col - 1) + m * (col - 1)] = dtd if stp == 1.0 else stp * stp * dtd
    sy[(col - 1) + m * (col - 1)] = dr

def _lbfgs_inv_hess(s: _LB, n_corrs: int) -> np.ndarray[float, 2]:
    # dense form of LbfgsInvHessProduct(sk, yk) built from the stored pairs
    n = s.n
    H = np.zeros((n, n))
    rho = np.empty(n_corrs)
    for i in range(n_corrs):
        rho[i] = 1.0 / _dot(s.ws, i * n, s.wy, i * n, n)
    alpha = np.empty(n_corrs)
    for c in range(n):
        q = np.zeros(n)
        q[c] = 1.0
        for i in range(n_corrs - 1, -1, -1):
            alpha[i] = rho[i] * _dot(s.ws, i * n, q, 0, n)
            for k in range(n):
                q[k] -= alpha[i] * s.wy[i * n + k]
        for i in range(n_corrs):
            beta = rho[i] * _dot(s.wy, i * n, q, 0, n)
            for k in range(n):
                q[k] += s.ws[i * n + k] * (alpha[i] - beta)
        for k in range(n):
            H[k, c] = q[k]
    return H

def _minimize_lbfgsb(fun, x0: np.ndarray[float, 1], args, jac,
                     bnds: Optional[Tuple[np.ndarray[float, 1], np.ndarray[float, 1]]],
                     options, callback = None):
    m = _opt_int(options, 'maxcor', 10)
    ftol = _opt_float(options, 'ftol', 2.2204460492503131e-09)
    gtol = _opt_float(options, 'gtol', 1e-5)
    eps = _opt_float(options, 'eps', 1e-8)
    maxfun = _opt_int(options, 'maxfun', 15000)
    maxiter = _opt_int(options, 'maxiter', 15000)
    maxls = _opt_int(options, 'maxls', 20)
    frs: Optional[float] = None
    if _has_opt(options, 'finite_diff_rel_step'):
        frs = _opt_float(options, 'finite_diff_rel_step', 0.0)
    pgtol = gtol
    factr = ftol / _EPSMCH
    n = x0.size
    lb = np.full(n, -INF)
    ub = np.full(n, INF)
    if bnds is not None:
        lb, ub = bnds
        for i in range(n):
            if lb[i] > ub[i]:
                raise ValueError('LBFGSB - one of the lower bounds is greater than an upper bound.')
        x0 = np.clip(x0, lb, ub)
    if not maxls > 0:
        raise ValueError('maxls must be positive.')
    if m <= 0:
        raise ValueError('maxcor must be positive.')
    sf = _ScalarFunction(fun, x0, args, jac, eps, frs, lb, ub)
    nbd = np.zeros(n, int)
    low = np.zeros(n)
    upp = np.zeros(n)
    for i in range(n):
        L = lb[i]
        U = ub[i]
        hl = not math.isinf(L)
        hu = not math.isinf(U)
        if hl:
            low[i] = L
        if hu:
            upp[i] = U
        if hl and hu:
            nbd[i] = 2
        elif hl:
            nbd[i] = 1
        elif hu:
            nbd[i] = 3
        else:
            nbd[i] = 0
    s = _LB(np.array(x0, float), low, upp, nbd, m)
    x = s.x
    tol = factr * _EPSMCH
    prjctd, cnstnd, boxed = _active(s)
    n_iterations = 0
    # task: 'conv_grad', 'conv_f', 'stop_iter', 'stop_fev', 'abnormal'
    task = ''

    def evaluate():
        f, g = sf.fun_and_grad(x)
        s.f = f
        for i in range(n):
            s.g[i] = g[i]

    evaluate()
    sbgnrm = _projgr(n, s.l, s.u, s.nbd, x, s.g)
    iter = 0
    fold = 0.0
    if sbgnrm <= pgtol:
        task = 'conv_grad'
    while task == '':
        info = 0
        # ---- compute search direction
        if cnstnd or s.col == 0:
            info = _cauchy(s, sbgnrm)
            if info != 0:
                s.reset()
                continue
            _freev(s, cnstnd, iter)
        else:
            _cp(x, 0, s.z, 0, n)
            s.wrk = s.updatd
            s.nseg = 0
        if s.nfree != 0 and s.col != 0:
            if s.wrk:
                info = _formk(s)
            if info != 0:
                s.reset()
                continue
            info = _cmprlb(s, cnstnd)
            if info == 0:
                info = _subsm(s)
            if info != 0:
                s.reset()
                continue
        for i in range(n):
            s.d[i] = s.z[i] - x[i]
        # ---- line search (lnsrlb)
        dnorm = _dnrm2(s.d, n)
        dtd = dnorm * dnorm
        stpmx = 1.0e10
        if cnstnd:
            if iter == 0:
                stpmx = 1.0
            else:
                for i in range(n):
                    a1 = s.d[i]
                    if nbd[i] != 0:
                        if a1 < 0.0 and nbd[i] <= 2:
                            a2 = s.l[i] - x[i]
                            if a2 >= 0.0:
                                stpmx = 0.0
                            elif a1 * stpmx < a2:
                                stpmx = a2 / a1
                        elif a1 > 0.0 and nbd[i] >= 2:
                            a2 = s.u[i] - x[i]
                            if a2 <= 0.0:
                                stpmx = 0.0
                            elif a1 * stpmx > a2:
                                stpmx = a2 / a1
        stp = 1.0
        if iter == 0 and not boxed:
            stp = min(1.0 / dnorm, stpmx)
        _cp(x, 0, s.t, 0, n)
        _cp(s.g, 0, s.r, 0, n)
        fold = s.f
        ifun = 0
        iback = 0
        ls_task = 0
        dc = _DCSRCH(1e-3, 0.9, 0.1, 0.0, stpmx, True)
        gd = 0.0
        gdold = 0.0
        lninfo = 0
        new_x = False
        while True:
            gd = _dot(s.g, 0, s.d, 0, n)
            if ifun == 0:
                gdold = gd
                if gd >= 0.0:
                    lninfo = -4
                    break
            stp, _f, _g, ls_task = dc.iterate(stp, s.f, gd, ls_task)
            if ls_task != 2 and ls_task != 3:
                ifun += 1
                iback = ifun - 1
                if iback >= maxls:
                    # the C code checks iback after returning from lnsrlb,
                    # before the requested evaluation is performed
                    pass
                if stp == 1.0:
                    _cp(s.z, 0, x, 0, n)
                else:
                    for i in range(n):
                        x[i] = stp * s.d[i] + s.t[i]
                        if nbd[i] == 1 or nbd[i] == 2:
                            x[i] = max(x[i], s.l[i])
                        if nbd[i] == 2 or nbd[i] == 3:
                            x[i] = min(x[i], s.u[i])
                if iback >= maxls:
                    break
                evaluate()
            else:
                new_x = True
                break
        if lninfo != 0 or iback >= maxls:
            _cp(s.t, 0, x, 0, n)
            _cp(s.r, 0, s.g, 0, n)
            s.f = fold
            if s.col == 0:
                task = 'abnormal'
                iter += 1
                break
            else:
                s.reset()
                continue
        # ---- new iterate
        iter += 1
        sbgnrm = _projgr(n, s.l, s.u, s.nbd, x, s.g)
        n_iterations += 1
        if _cb_halt(callback, x, s.f):
            task = 'callback'
            break
        if n_iterations >= maxiter:
            task = 'stop_iter'
            break
        elif sf.nfev > maxfun:
            task = 'stop_fev'
            break
        if sbgnrm <= pgtol:
            task = 'conv_grad'
            break
        ddum = max(abs(fold), abs(s.f), 1.0)
        if (fold - s.f) <= tol * ddum:
            task = 'conv_f'
            break
        for i in range(n):
            s.r[i] = s.g[i] - s.r[i]
        rn = _dnrm2(s.r, n)
        rr = rn * rn
        dr = 0.0
        if stp == 1.0:
            dr = gd - gdold
            ddum = -gdold
        else:
            dr = (gd - gdold) * stp
            for i in range(n):
                s.d[i] *= stp
            ddum = -gdold * stp
        if dr <= _EPSMCH * ddum:
            s.updatd = False
            continue
        s.updatd = True
        s.iupdat += 1
        _matupd(s, rr, dr, stp, dtd)
        if _formt(s) != 0:
            s.reset()
            continue
    warnflag = 2
    msg = ''
    if task == 'conv_grad':
        warnflag = 0
        msg = 'CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL'
    elif task == 'conv_f':
        warnflag = 0
        msg = 'CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH'
    elif task == 'stop_iter':
        msg = 'STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT'
    elif task == 'stop_fev':
        msg = 'STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT'
    elif task == 'callback':
        msg = 'STOP: CALLBACK REQUESTED STOP'
    else:
        msg = 'ABNORMAL: '
    if warnflag != 0 and (sf.nfev > maxfun or n_iterations >= maxiter):
        warnflag = 1
    n_corrs = min(s.iupdat, m)
    H = _lbfgs_inv_hess(s, n_corrs)
    r = OptimizeResult(x.copy(), s.f, s.g.copy(), nit=n_iterations,
                       nfev=sf.nfev, njev=sf.ngev, status=warnflag,
                       success=(warnflag == 0), message=msg, hess_inv=H)
    if task == 'callback':
        r.success = False
        r.status = 99
        r.message = _STOP_MSG
    return r
