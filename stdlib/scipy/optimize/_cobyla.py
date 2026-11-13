# Ported from PRIMA (BSD-3-Clause)
# Python reference implementation pyprima as vendored by scipy 1.18.1
# (scipy/_external/pyprima: cobyla/*.py and common/*.py), Copyright (c)
# Zaikun Zhang; scipy wrapper _cobyla_py.py (BSD-3-Clause).

import math
import numpy as np

REALMAX = 1.7976931348623157e308
REALMIN = 2.2250738585072014e-308
EPS = 2.220446049250313e-16
FUNCMAX = 1e30
CONSTRMAX = FUNCMAX
BOUNDMAX = REALMAX / 4
CTOL_DEFAULT = 1.4901161193847656e-08
CWEIGHT_DEFAULT = 1e8
MIN_MAXFILT = 200
MAXFILT_DEFAULT = 2000
MAXHISTMEM = 300000000

INFO_DEFAULT = 0
SMALL_TR_RADIUS = 0
FTARGET_ACHIEVED = 1
TRSUBP_FAILED = 2
MAXFUN_REACHED = 3
MAXTR_REACHED = 20
NAN_INF_X = -1
NAN_INF_F = -2
NAN_INF_MODEL = -3
NO_SPACE_BETWEEN_BOUNDS = 6
DAMAGING_ROUNDING = 7
ZERO_LINEAR_CONSTRAINT = 8
CALLBACK_TERMINATE = 30

def get_info_string(solver: str, info: int) -> str:
    reason = 'UNKNOWN EXIT FLAG'
    if info == FTARGET_ACHIEVED:
        reason = 'the target function value is achieved.'
    elif info == MAXFUN_REACHED:
        reason = 'the objective function has been evaluated MAXFUN times.'
    elif info == MAXTR_REACHED:
        reason = 'the maximal number of trust region iterations has been reached.'
    elif info == SMALL_TR_RADIUS:
        reason = 'the trust region radius reaches its lower bound.'
    elif info == TRSUBP_FAILED:
        reason = 'a trust region step has failed to reduce the quadratic model.'
    elif info == NAN_INF_X:
        reason = 'NaN or Inf occurs in x.'
    elif info == NAN_INF_F:
        reason = 'the objective function returns NaN/+Inf.'
    elif info == NAN_INF_MODEL:
        reason = 'NaN or Inf occurs in the models.'
    elif info == DAMAGING_ROUNDING:
        reason = 'rounding errors are becoming damaging.'
    elif info == NO_SPACE_BETWEEN_BOUNDS:
        reason = 'there is no space between the lower and upper bounds of variable.'
    elif info == ZERO_LINEAR_CONSTRAINT:
        reason = 'one of the linear constraints has a zero gradient'
    elif info == CALLBACK_TERMINATE:
        reason = 'the callback function requested termination'
    return f'Return from {solver} because {reason.strip()}'

# ----------------------------------------------------------------- linalg

def _isnan(v: float) -> bool:
    return v != v

def _cmax0(c: np.ndarray[float, 1]) -> float:
    # np.max(np.append(0, c)) with NaN propagation
    m = 0.0
    for v in c:
        if v != v:
            return v
        if v > m:
            m = v
    return m

def norm(x: np.ndarray[float, 1]) -> float:
    return float(np.linalg.norm(x))

def isminor(x: float, ref: float) -> bool:
    sensitivity = 0.1
    refa = abs(ref) + sensitivity * abs(x)
    refb = abs(ref) + 2 * sensitivity * abs(x)
    return abs(ref) >= refa or refa >= refb

def _sign(v: float) -> float:
    if v > 0:
        return 1.0
    elif v < 0:
        return -1.0
    return 0.0

def planerot(x0: float, x1: float) -> np.ndarray[float, 2]:
    c = 1.0
    s = 0.0
    if _isnan(x0) or _isnan(x1):
        c = 1.0
        s = 0.0
    elif math.isinf(x0) and math.isinf(x1):
        c = 1 / math.sqrt(2) * _sign(x0)
        s = 1 / math.sqrt(2) * _sign(x1)
    elif abs(x0) <= 0 and abs(x1) <= 0:
        c = 1.0
        s = 0.0
    elif abs(x1) <= EPS * abs(x0):
        c = _sign(x0)
        s = 0.0
    elif abs(x0) <= EPS * abs(x1):
        c = 0.0
        s = _sign(x1)
    elif math.sqrt(REALMIN) < abs(x0) < math.sqrt(REALMAX / 2.1) and \
            math.sqrt(REALMIN) < abs(x1) < math.sqrt(REALMAX / 2.1):
        r = norm(np.array([x0, x1]))
        c = x0 / r
        s = x1 / r
    elif abs(x0) > abs(x1):
        t = x1 / x0
        u = max(1.0, abs(t), math.sqrt(1 + t * t))
        u *= _sign(x0)
        c = 1 / u
        s = t / u
    else:
        t = x0 / x1
        u = max(1.0, abs(t), math.sqrt(1 + t * t))
        u *= _sign(x1)
        c = t / u
        s = 1 / u
    G = np.empty((2, 2))
    G[0, 0] = c
    G[0, 1] = s
    G[1, 0] = -s
    G[1, 1] = c
    return G

def _rot_cols(Q: np.ndarray[float, 2], a: int, b: int, M: np.ndarray[float, 2]):
    # Q[:, [a, b]] = Q[:, [a, b]] @ M
    for i in range(Q.shape[0]):
        qa = Q[i, a]
        qb = Q[i, b]
        # BLAS gemm accumulates with fused multiply-add: fma(qb, M10, qa*M00)
        Q[i, a] = math.fma(qb, M[1, 0], qa * M[0, 0])
        Q[i, b] = math.fma(qb, M[1, 1], qa * M[0, 1])

def qradd_Rdiag(c: np.ndarray[float, 1], Q: np.ndarray[float, 2],
                Rdiag: np.ndarray[float, 1], n: int):
    m = Q.shape[1]
    cq = c @ Q
    cqa = np.abs(c) @ np.abs(Q)
    for i in range(cq.size):
        if isminor(cq[i], cqa[i]):
            cq[i] = 0.0
    k = m - 2
    while k >= n:
        if abs(cq[k + 1]) > 0:
            G = planerot(cq[k], cq[k + 1])
            _rot_cols(Q, k, k + 1, G.T)
            cq[k] = float(np.hypot(cq[k], cq[k + 1]))
        k -= 1
    if n < m:
        if abs(cq[n]) > EPS ** 2 and not isminor(cq[n], cqa[n]):
            n += 1
    if n - 1 >= 0 and n - 1 < m:
        Rdiag[n - 1] = cq[n - 1]
    return Q, Rdiag, n

def qrexc_Rdiag(A: np.ndarray[float, 2], Q: np.ndarray[float, 2],
                Rdiag: np.ndarray[float, 1], i: int):
    n = A.shape[1]
    if i < 0 or i >= n:
        return Q, Rdiag
    for k in range(i, n - 1):
        G = planerot(Rdiag[k + 1], float(np.dot(Q[:, k], A[:, k + 1])))
        # Q[:, [k, k+1]] = Q[:, [k+1, k]] @ G.T
        GT = G.T
        for r in range(Q.shape[0]):
            a = Q[r, k + 1]
            b = Q[r, k]
            Q[r, k] = math.fma(b, GT[1, 0], a * GT[0, 0])
            Q[r, k + 1] = math.fma(b, GT[1, 1], a * GT[0, 1])
    for k in range(i, n - 1):
        Rdiag[k] = float(np.dot(Q[:, k], A[:, k + 1]))
    Rdiag[n - 1] = float(np.dot(Q[:, n - 1], A[:, i]))
    return Q, Rdiag

def lsqr(A: np.ndarray[float, 2], b: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
    if A.shape[1] == 0:
        return np.zeros(0)
    return np.asarray(np.linalg.lstsq(A, b, rcond=None)[0], float)

def _cols(A: np.ndarray[float, 2], idx: np.ndarray[int, 1], k: int) -> np.ndarray[float, 2]:
    R = np.empty((A.shape[0], k))
    for j in range(k):
        for i in range(A.shape[0]):
            R[i, j] = A[i, idx[j]]
    return R

# ------------------------------------------------------------- trstlp

class _LP:
    iact: np.ndarray[int, 1]
    nact: int
    d: np.ndarray[float, 1]
    vmultc: np.ndarray[float, 1]
    z: np.ndarray[float, 2]

    def __init__(self, mcon: int, n: int):
        self.iact = np.zeros(mcon, int)
        self.nact = 0
        self.d = np.zeros(n)
        self.vmultc = np.zeros(mcon)
        self.z = np.zeros((n, n))

def trstlp(A: np.ndarray[float, 2], b: np.ndarray[float, 1], delta: float,
           g: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
    num_constraints = A.shape[1]
    num_vars = A.shape[0]
    A_aug = np.empty((num_vars, num_constraints + 1))
    for i in range(num_vars):
        for j in range(num_constraints):
            A_aug[i, j] = A[i, j]
        A_aug[i, num_constraints] = g[i]
    b_aug = np.zeros(num_constraints + 1)
    for j in range(num_constraints):
        b_aug[j] = b[j]
    for i in range(num_constraints + 1):
        maxval = 0.0
        for r in range(num_vars):
            maxval = max(maxval, abs(A_aug[r, i]))
        if maxval > 1000000000000.0:
            modscal = max(2 * REALMIN, 1 / maxval)
            for r in range(num_vars):
                A_aug[r, i] *= modscal
            b_aug[i] *= modscal
    st = _LP(num_constraints + 1, num_vars)
    # stage 1 on the first num_constraints columns (views in pyprima)
    s1 = _LP(num_constraints, num_vars)
    _trstlp_sub(s1, 1, A_aug[:, :num_constraints].copy(), b_aug[:num_constraints].copy(), delta)
    for i in range(num_constraints):
        st.iact[i] = s1.iact[i]
        st.vmultc[i] = s1.vmultc[i]
    st.nact = s1.nact
    st.d = s1.d
    st.z = s1.z
    _trstlp_sub(st, 2, A_aug, b_aug, delta)
    return st.d

def _trstlp_sub(st: _LP, stage: int, A: np.ndarray[float, 2], b: np.ndarray[float, 1],
                delta: float):
    zdasav = np.zeros(st.z.shape[1])
    vmultd = np.zeros(st.vmultc.size)
    zdota = np.zeros(st.z.shape[1])
    mcon = A.shape[1]
    num_vars = A.shape[0]
    num_constraints = 0
    icon = 0
    cviol = 0.0
    sdirn = np.zeros(num_vars)
    if stage == 1:
        for i in range(mcon):
            st.iact[i] = i
        st.nact = 0
        st.d = np.zeros(num_vars)
        cviol = _cmax0(-b)
        st.vmultc = cviol + b
        st.z = np.eye(num_vars)
        if mcon == 0 or cviol <= 0:
            return
        allnan = True
        for v in b:
            if not _isnan(v):
                allnan = False
        if allnan:
            return
        best = -REALMAX
        icon = 0
        first = True
        for i in range(mcon):
            if not _isnan(b[i]):
                if first or -b[i] > best:
                    best = -b[i]
                    icon = i
                    first = False
        num_constraints = mcon
    else:
        if float(np.dot(st.d, st.d)) >= delta * delta:
            return
        st.iact[mcon - 1] = mcon - 1
        st.vmultc[mcon - 1] = 0.0
        num_constraints = mcon - 1
        icon = mcon - 1
        cviol = _cmax0(st.d @ A[:, :num_constraints] - b[:num_constraints])
    iact = st.iact
    vmultc = st.vmultc
    z = st.z
    d = st.d
    nact = st.nact
    for k in range(nact):
        zdota[k] = float(np.dot(z[:, k], A[:, iact[k]]))
    optold = REALMAX
    nactold = nact
    nfail = 0
    maxiter = min(10000, 100 * max(num_constraints, num_vars))
    for it in range(maxiter):
        optnew = cviol if stage == 1 else float(np.dot(d, A[:, mcon - 1]))
        if optnew < optold or nact > nactold:
            nactold = nact
            nfail = 0
        else:
            nfail += 1
        optold = min(optold, optnew)
        if nfail == 3:
            break
        if icon >= nact:
            for k in range(nact):
                zdasav[k] = zdota[k]
            nactsav = nact
            z, zdota, nact = qradd_Rdiag(A[:, iact[icon]].copy(), z, zdota, nact)
            if nact == nactsav + 1:
                if nact != icon + 1:
                    vmultc[icon] = vmultc[nact - 1]
                    vmultc[nact - 1] = 0.0
                    t = iact[icon]
                    iact[icon] = iact[nact - 1]
                    iact[nact - 1] = t
                else:
                    vmultc[nact - 1] = 0.0
            else:
                vd = lsqr(_cols(A, iact, nact), A[:, iact[icon]].copy())
                for k in range(nact):
                    vmultd[k] = vd[k]
                anyp = False
                for k in range(nact):
                    if vmultd[k] > 0 and iact[k] <= num_constraints:
                        anyp = True
                if not anyp:
                    break
                for k in range(nact, mcon):
                    vmultd[k] = -1.0
                frac = REALMAX
                for i in range(nact):
                    fm = REALMAX
                    if vmultd[i] > 0 and iact[i] <= num_constraints:
                        fm = vmultc[i] / vmultd[i]
                    frac = min(frac, fm)
                for k in range(nact):
                    vmultc[k] = max(0.0, vmultc[k] - frac * vmultd[k])
                if _isnan(zdota[nact - 1]) or abs(zdota[nact - 1]) <= EPS ** 2:
                    break
                vmultc[icon] = 0.0
                vmultc[nact - 1] = frac
                t = iact[icon]
                iact[icon] = iact[nact - 1]
                iact[nact - 1] = t
            if stage == 2 and iact[nact - 1] != mcon - 1:
                if nact <= 1:
                    break
                zs = zdota[:nact].copy()
                z, zs = qrexc_Rdiag(_cols(A, iact, nact), z, zs, nact - 2)
                for k in range(nact):
                    zdota[k] = zs[k]
                t = iact[nact - 2]
                iact[nact - 2] = iact[nact - 1]
                iact[nact - 1] = t
                tv = vmultc[nact - 2]
                vmultc[nact - 2] = vmultc[nact - 1]
                vmultc[nact - 1] = tv
            if _isnan(zdota[nact - 1]) or abs(zdota[nact - 1]) <= EPS ** 2:
                break
            if stage == 1:
                sdirn -= (float(np.dot(sdirn, A[:, iact[nact - 1]])) + 1) / zdota[nact - 1] * z[:, nact - 1]
            else:
                sdirn = -1 / zdota[nact - 1] * z[:, nact - 1]
        else:
            zs = zdota[:nact].copy()
            z, zs = qrexc_Rdiag(_cols(A, iact, nact), z, zs, icon)
            for k in range(nact):
                zdota[k] = zs[k]
            ti = iact[icon]
            tv = vmultc[icon]
            for k in range(icon, nact - 1):
                iact[k] = iact[k + 1]
                vmultc[k] = vmultc[k + 1]
            iact[nact - 1] = ti
            vmultc[nact - 1] = tv
            nact -= 1
            if stage == 2 and nact < 0:
                break
            if nact > 0:
                if _isnan(zdota[nact - 1]) or abs(zdota[nact - 1]) <= EPS ** 2:
                    break
            if stage == 1:
                sdirn -= float(np.dot(sdirn, z[:, nact])) * z[:, nact]
            else:
                sdirn = -1 / zdota[nact - 1] * z[:, nact - 1]
        dd = delta * delta - float(np.dot(d, d))
        ss = float(np.dot(sdirn, sdirn))
        sd = float(np.dot(sdirn, d))
        if dd <= 0 or ss <= EPS * delta * delta or _isnan(sd):
            break
        sqrtd = max(math.sqrt(ss * dd + sd * sd), abs(sd), math.sqrt(ss * dd))
        step = 0.0
        if sd > 0:
            step = dd / (sqrtd + sd)
        else:
            step = (sqrtd - sd) / ss
        if step <= 0 or not math.isfinite(step):
            break
        if stage == 1:
            if isminor(cviol, step):
                break
            step = min(step, cviol)
        dnew = d + step * sdirn
        Aact = _cols(A, iact, nact)
        if stage == 1:
            bb = np.empty(nact)
            for k in range(nact):
                bb[k] = b[iact[k]]
            cviol = _cmax0(dnew @ Aact - bb)
        vd = lsqr(Aact, dnew)
        for k in range(nact):
            vmultd[k] = -vd[k]
        if stage == 2:
            vmultd[nact - 1] = max(0.0, vmultd[nact - 1])
        Aall = _cols(A, iact, mcon)
        biact = np.empty(mcon)
        for k in range(mcon):
            biact[k] = b[iact[k]]
        cvshift = cviol - (dnew @ Aall - biact)
        cvsabs = np.abs(dnew) @ np.abs(Aall) + np.abs(biact) + cviol
        for k in range(mcon):
            if isminor(cvshift[k], cvsabs[k]):
                cvshift[k] = 0.0
        for k in range(nact, mcon):
            vmultd[k] = cvshift[k]
        frac = 1.0
        icon = -1
        for i in range(vmultd.size):
            fm = REALMAX
            if vmultd[i] < 0:
                fm = vmultc[i] / (vmultc[i] - vmultd[i])
            if fm < frac:
                frac = fm
                icon = i
        dold = d
        d = (1 - frac) * d + frac * dnew
        vmultc = np.maximum(0.0, (1 - frac) * vmultc + frac * vmultd)
        if not (math.isfinite(float(np.sum(np.abs(d)))) and math.isfinite(float(np.sum(np.abs(vmultc))))):
            d = dold
            break
        if stage == 1:
            cviol = _cmax0(d @ A - b)
        if icon < 0 or icon >= mcon:
            break
    st.iact = iact
    st.nact = nact
    st.d = d
    st.vmultc = vmultc
    st.z = z

def trrad(delta_in: float, dnorm: float, eta1: float, eta2: float, gamma1: float,
          gamma2: float, ratio: float) -> float:
    if ratio <= eta1:
        return gamma1 * dnorm
    elif ratio <= eta2:
        return max(gamma1 * delta_in, dnorm)
    return max(gamma1 * delta_in, gamma2 * dnorm)

# --------------------------------------------------- simplex updates

def _inv(A: np.ndarray[float, 2]) -> np.ndarray[float, 2]:
    return np.asarray(np.linalg.inv(A), float)

def _erri(simi: np.ndarray[float, 2], sim: np.ndarray[float, 2], n: int) -> float:
    E = simi @ sim[:, :n] - np.eye(n)
    m = 0.0
    for v in E.flat:
        if v != v:
            return v
        if abs(v) > m:
            m = abs(v)
    return m

class _Simplex:
    conmat: np.ndarray[float, 2]
    cval: np.ndarray[float, 1]
    fval: np.ndarray[float, 1]
    sim: np.ndarray[float, 2]
    simi: np.ndarray[float, 2]

    def __init__(self, conmat, cval, fval, sim, simi):
        self.conmat = conmat
        self.cval = cval
        self.fval = fval
        self.sim = sim
        self.simi = simi

    def copy(self):
        return _Simplex(self.conmat.copy(), self.cval.copy(), self.fval.copy(),
                        self.sim.copy(), self.simi.copy())

def findpole(cpen: float, cval: np.ndarray[float, 1], fval: np.ndarray[float, 1]) -> int:
    n1 = fval.size
    jopt = n1 - 1
    phi = fval + cpen * cval
    phimin = float(np.min(phi))
    anyb = False
    for i in range(n1):
        if cval[i] < cval[jopt] and phi[i] <= phi[jopt]:
            anyb = True
    if phimin < phi[jopt] or anyb:
        best = -1
        for i in range(n1):
            if not (phi[i] > phimin):
                if best < 0 or cval[i] < cval[best]:
                    best = i
        jopt = best if best >= 0 else 0
    return jopt

def _swap_cols(M: np.ndarray[float, 2], a: int, b: int):
    for i in range(M.shape[0]):
        t = M[i, a]
        M[i, a] = M[i, b]
        M[i, b] = t

def updatepole(cpen: float, S: _Simplex) -> int:
    n = S.sim.shape[0]
    info = INFO_DEFAULT
    jopt = findpole(cpen, S.cval, S.fval)
    sim_old = S.sim.copy()
    simi_old = S.simi.copy()
    sim = S.sim
    simi = S.simi
    if 0 <= jopt < n:
        for i in range(n):
            sim[i, n] += sim[i, jopt]
        sim_jopt = sim[:, jopt].copy()
        for i in range(n):
            sim[i, jopt] = 0.0
        for j in range(n):
            for i in range(n):
                sim[i, j] -= sim_jopt[i]
        cs = np.sum(simi, axis=0)
        for j in range(n):
            simi[jopt, j] = -cs[j]
    erri = _erri(simi, sim, n)
    itol = 1.0
    if erri > 0.1 * itol or _isnan(erri):
        simi_test = _inv(sim[:, :n].copy())
        erri_test = _erri(simi_test, sim, n)
        if erri_test < erri or (_isnan(erri) and not _isnan(erri_test)):
            simi = simi_test
            erri = erri_test
    if erri <= itol:
        S.sim = sim
        S.simi = simi
        if 0 <= jopt < n:
            t = S.fval[jopt]
            S.fval[jopt] = S.fval[n]
            S.fval[n] = t
            _swap_cols(S.conmat, jopt, n)
            t = S.cval[jopt]
            S.cval[jopt] = S.cval[n]
            S.cval[n] = t
    else:
        info = DAMAGING_ROUNDING
        S.sim = sim_old
        S.simi = simi_old
    return info

def updatexfc(jdrop: int, constr: np.ndarray[float, 1], cpen: float, cstrv: float,
              d: np.ndarray[float, 1], f: float, S: _Simplex) -> int:
    # jdrop < 0 stands for None
    n = S.sim.shape[0]
    if jdrop < 0:
        return INFO_DEFAULT
    # pyprima keeps references (sim_old is sim): a rejected update is not
    # rolled back for the in-place modified arrays
    sim = S.sim
    simi = S.simi
    if jdrop < n:
        for i in range(n):
            sim[i, jdrop] = d[i]
        simi_jdrop = simi[jdrop, :] / float(np.dot(simi[jdrop, :], d))
        simi = simi - np.outer(simi @ d, simi_jdrop)
        for j in range(n):
            simi[jdrop, j] = simi_jdrop[j]
    else:
        for i in range(n):
            sim[i, n] += d[i]
        for j in range(n):
            for i in range(n):
                sim[i, j] -= d[i]
        simid = simi @ d
        sum_simi = np.sum(simi, axis=0)
        simi = simi + np.outer(simid, sum_simi / (1 - float(np.sum(simid))))
    itol = 1.0
    erri = _erri(simi, sim, n)
    if erri > 0.1 * itol or _isnan(erri):
        simi_test = _inv(sim[:, :n].copy())
        erri_test = _erri(simi_test, sim, n)
        if erri_test < erri or (_isnan(erri) and not _isnan(erri_test)):
            simi = simi_test
            erri = erri_test
    S.sim = sim
    if erri <= itol:
        S.simi = simi
        S.fval[jdrop] = f
        for i in range(constr.size):
            S.conmat[i, jdrop] = constr[i]
        S.cval[jdrop] = cstrv
        return updatepole(cpen, S)
    return DAMAGING_ROUNDING

def setdrop_tr(ximproved: bool, d: np.ndarray[float, 1], delta: float, rho: float,
               sim: np.ndarray[float, 2], simi: np.ndarray[float, 2]) -> int:
    n = sim.shape[0]
    distsq = np.zeros(n + 1)
    if ximproved:
        for j in range(n):
            s = 0.0
            for i in range(n):
                t = sim[i, j] - d[i]
                s += t * t
            distsq[j] = s
        distsq[n] = float(np.sum(d * d))
    else:
        for j in range(n):
            s = 0.0
            for i in range(n):
                s += sim[i, j] * sim[i, j]
            distsq[j] = s
        distsq[n] = 0.0
    r = max(rho, delta / 10)
    weight = np.maximum(1.0, distsq / (r * r))
    simid = simi @ d
    score = np.empty(n + 1)
    for j in range(n):
        score[j] = weight[j] * abs(simid[j])
    score[n] = weight[n] * abs(1 - float(np.sum(simid)))
    if not ximproved:
        score[n] = -1.0
    for j in range(n + 1):
        if _isnan(score[j]):
            score[j] = -1.0
    jdrop = -1
    best = 0.0
    for j in range(n + 1):
        if score[j] > 0 and (jdrop < 0 or score[j] > best):
            best = score[j]
            jdrop = j
    if ximproved and jdrop < 0:
        jdrop = int(np.argmax(distsq))
    return jdrop

def _Amat(amat: np.ndarray[float, 2], m_lcon: int, conmat: np.ndarray[float, 2],
          simi: np.ndarray[float, 2], n: int) -> np.ndarray[float, 2]:
    m = conmat.shape[0]
    A = np.zeros((n, m))
    for j in range(m_lcon):
        for i in range(n):
            A[i, j] = amat[j, i]
    if m > m_lcon:
        D = np.empty((m - m_lcon, n))
        for r in range(m - m_lcon):
            for k in range(n):
                D[r, k] = conmat[m_lcon + r, k] - conmat[m_lcon + r, n]
        P = np.zeros((m - m_lcon, n))
        for r in range(m - m_lcon):
            for i in range(n):
                acc = 0.0
                for k in range(n):
                    acc = math.fma(D[r, k], simi[k, i], acc)
                P[r, i] = acc
        for r in range(m - m_lcon):
            for i in range(n):
                A[i, m_lcon + r] = P[r, i]
    return A

def geostep(jdrop: int, amat: np.ndarray[float, 2], m_lcon: int, S: _Simplex,
            cpen: float, delbar: float) -> np.ndarray[float, 1]:
    n = S.simi.shape[0]
    d = S.simi[jdrop, :].copy()
    d = delbar * (d / norm(d))
    g = (S.fval[:n] - S.fval[n]) @ S.simi
    A = _Amat(amat, m_lcon, S.conmat, S.simi, n)
    cvpd = _cmax0(S.conmat[:, n] + d @ A)
    cvnd = _cmax0(S.conmat[:, n] - d @ A)
    if -float(np.dot(d, g)) + cpen * cvnd < float(np.dot(d, g)) + cpen * cvpd:
        d *= -1
    return d

# ------------------------------------------------------------- filter

def _anyinf(f: float, c: float) -> bool:
    return _isnan(f) or _isnan(c) or (math.isinf(f) and f > 0) or (math.isinf(c) and c > 0)

def isbetter(f1: float, c1: float, f2: float, c2: float, ctol: float) -> bool:
    is_better = _anyinf(f1, c1) and not _anyinf(f2, c2)
    is_better = is_better or (f1 < f2 and c1 <= c2)
    is_better = is_better or (f1 <= f2 and c1 < c2)
    cref = 10 * max(EPS, min(ctol, 0.01 * CONSTRMAX))
    is_better = is_better or (f1 < REALMAX and c1 <= ctol and (c2 > max(ctol, cref) or _isnan(c2)))
    return is_better

class _Filt:
    nfilt: int
    cfilt: np.ndarray[float, 1]
    ffilt: np.ndarray[float, 1]
    xfilt: np.ndarray[float, 2]
    confilt: np.ndarray[float, 2]

    def __init__(self, maxfilt: int, n: int, m: int):
        self.nfilt = 0
        self.cfilt = np.zeros(maxfilt)
        self.ffilt = np.zeros(maxfilt)
        self.xfilt = np.zeros((n, maxfilt))
        self.confilt = np.zeros((m, maxfilt))

    def save(self, cstrv: float, ctol: float, cweight: float, f: float,
             x: np.ndarray[float, 1], constr: np.ndarray[float, 1]):
        nfilt = self.nfilt
        maxfilt = self.ffilt.size
        for i in range(nfilt):
            if isbetter(self.ffilt[i], self.cfilt[i], f, cstrv, ctol):
                return
            if self.ffilt[i] <= f and self.cfilt[i] <= cstrv:
                return
        keep = np.empty(nfilt, bool)
        for i in range(nfilt):
            keep[i] = not isbetter(f, cstrv, self.ffilt[i], self.cfilt[i], ctol)
        nk = 0
        for i in range(nfilt):
            if keep[i]:
                nk += 1
        if nk == maxfilt:
            cfs = np.maximum(self.cfilt - ctol, 0.0)
            phi = np.zeros(maxfilt)
            if cweight <= 0:
                phi = self.ffilt.copy()
            elif math.isinf(cweight):
                phi = cfs
            else:
                phi = np.maximum(self.ffilt, -REALMAX)
                for i in range(maxfilt):
                    if _isnan(phi[i]):
                        phi[i] = -REALMAX
                phi = phi + cweight * cfs
            phimax = float(np.max(phi))
            cref = -REALMAX
            for i in range(maxfilt):
                if phi[i] >= phimax:
                    cref = max(cref, cfs[i])
            fref = -REALMAX
            for i in range(maxfilt):
                if cfs[i] >= cref:
                    fref = max(fref, self.ffilt[i])
            kworst = -1
            for i in range(maxfilt):
                if not (self.ffilt[i] > fref):
                    if kworst < 0 or self.cfilt[i] > self.cfilt[kworst]:
                        kworst = i
            if kworst < 0 or kworst >= keep.size:
                kworst = 0
            keep[kworst] = False
        k = 0
        for i in range(nfilt):
            if keep[i]:
                self.xfilt[:, k] = self.xfilt[:, i]
                self.ffilt[k] = self.ffilt[i]
                self.cfilt[k] = self.cfilt[i]
                self.confilt[:, k] = self.confilt[:, i]
                k += 1
        nfilt = k
        self.xfilt[:, nfilt] = x
        self.ffilt[nfilt] = f
        self.cfilt[nfilt] = cstrv
        if constr.size > 0:
            self.confilt[:, nfilt] = constr
        self.nfilt = nfilt + 1

def selectx(fhist: np.ndarray[float, 1], chist: np.ndarray[float, 1], cweight: float,
            ctol: float) -> int:
    nhist = fhist.size
    def anyp(fr: float, cr: float) -> bool:
        for i in range(nhist):
            if fhist[i] < fr and chist[i] < cr:
                return True
        return False
    fref = REALMAX
    cref = REALMAX
    if anyp(FUNCMAX, CONSTRMAX):
        fref = FUNCMAX
        cref = CONSTRMAX
    elif anyp(REALMAX, CONSTRMAX):
        fref = REALMAX
        cref = CONSTRMAX
    elif anyp(FUNCMAX, REALMAX):
        fref = FUNCMAX
        cref = REALMAX
    if not anyp(fref, cref):
        return nhist - 1
    chs = np.maximum(chist - ctol, 0.0)
    cmin = REALMAX
    for i in range(nhist):
        if fhist[i] < fref:
            cmin = min(cmin, chs[i])
    cref = max(EPS, 2 * cmin)
    phi = np.zeros(nhist)
    if cweight <= 0:
        phi = fhist.copy()
    elif math.isinf(cweight):
        phi = chs
    else:
        phi = np.maximum(fhist, -REALMAX) + cweight * chs
    phimin = REALMAX
    first = True
    for i in range(nhist):
        if fhist[i] < fref and chs[i] <= cref:
            if first or phi[i] < phimin:
                phimin = phi[i]
                first = False
    cref2 = REALMAX
    first = True
    for i in range(nhist):
        if fhist[i] < fref and phi[i] <= phimin:
            if first or chs[i] < cref2:
                cref2 = chs[i]
                first = False
    fref2 = REALMAX
    first = True
    for i in range(nhist):
        if chs[i] <= cref2:
            if first or fhist[i] < fref2:
                fref2 = fhist[i]
                first = False
    kopt = -1
    for i in range(nhist):
        if not (fhist[i] > fref2):
            if kopt < 0 or chist[i] < chist[kopt]:
                kopt = i
    return kopt if kopt >= 0 else 0

def checkbreak_con(maxfun: int, nf: int, cstrv: float, ctol: float, f: float,
                   ftarget: float, x: np.ndarray[float, 1]) -> int:
    info = INFO_DEFAULT
    for v in x:
        if _isnan(v) or math.isinf(v):
            info = NAN_INF_X
    if _isnan(f) or (math.isinf(f) and f > 0) or _isnan(cstrv) or (math.isinf(cstrv) and cstrv > 0):
        info = NAN_INF_F
    if cstrv <= ctol and f <= ftarget:
        info = FTARGET_ACHIEVED
    if nf >= maxfun:
        info = MAXFUN_REACHED
    return info

def redrho(rho_in: float, rhoend: float) -> float:
    rho_ratio = rho_in / rhoend
    if rho_ratio > 250:
        return 0.1 * rho_in
    elif rho_ratio <= 16:
        return rhoend
    return math.sqrt(rho_ratio) * rhoend

def redrat(ared: float, pred: float, rshrink: float) -> float:
    if _isnan(ared):
        return -REALMAX
    elif _isnan(pred) or pred <= 0:
        if ared > 0:
            return rshrink / 2
        return -REALMAX
    elif math.isinf(pred) and pred > 0 and math.isinf(ared) and ared > 0:
        return 1.0
    elif math.isinf(pred) and pred > 0 and math.isinf(ared) and ared < 0:
        return -REALMAX
    return ared / pred

def moderatef(f: float) -> float:
    if _isnan(f):
        f = FUNCMAX
    return min(max(f, -REALMAX), FUNCMAX)

def moderatec(c: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
    r = c.copy()
    for i in range(r.size):
        v = r[i]
        if _isnan(v):
            v = CONSTRMAX
        elif math.isinf(v):
            v = REALMAX if v > 0 else -REALMAX
        r[i] = min(max(v, -CONSTRMAX), CONSTRMAX)
    return r

# ------------------------------------------------------------- cobylb

def fcratio(conmat: np.ndarray[float, 2], fval: np.ndarray[float, 1]) -> float:
    m = conmat.shape[0]
    n1 = conmat.shape[1]
    fmin = float(np.min(fval))
    fmax = float(np.max(fval))
    anyc = False
    denom = INF_
    for i in range(m):
        cmin = REALMAX
        cmax = -REALMAX
        for j in range(n1):
            v = -conmat[i, j]
            cmin = min(cmin, v)
            cmax = max(cmax, v)
        if cmin < 0.5 * cmax:
            anyc = True
            denom = min(denom, max(cmax, 0.0) - cmin)
    if anyc and fmin < fmax:
        return (fmax - fmin) / denom
    return 0.0

INF_ = math.inf

class _Cob[CF]:
    calcfc: CF
    amat: np.ndarray[float, 2]
    bvec: np.ndarray[float, 1]
    m_lcon: int
    m_nlcon: int
    n: int

    def __init__(self, calcfc: CF, amat: np.ndarray[float, 2], bvec: np.ndarray[float, 1],
                 m_nlcon: int, n: int):
        self.calcfc = calcfc
        self.amat = amat
        self.bvec = bvec
        self.m_lcon = bvec.size
        self.m_nlcon = m_nlcon
        self.n = n

    def evaluate(self, x: np.ndarray[float, 1]):
        constr = np.zeros(self.m_lcon + self.m_nlcon)
        if self.m_lcon > 0:
            lc = x @ self.amat.T - self.bvec
            for i in range(self.m_lcon):
                constr[i] = lc[i]
        anynan = False
        for v in x:
            if _isnan(v):
                anynan = True
        if anynan:
            f = float(np.sum(x))
            constr = np.ones(self.m_nlcon) * f
            return f, constr
        f, nl = self.calcfc(x.copy())
        f = moderatef(f)
        nl = moderatec(nl)
        for i in range(self.m_nlcon):
            constr[self.m_lcon + i] = nl[i]
        return f, constr

    def _A(self, S: _Simplex) -> np.ndarray[float, 2]:
        return _Amat(self.amat, self.m_lcon, S.conmat, S.simi, self.n)

    def getcpen(self, S0: _Simplex, cpen: float, delta: float) -> float:
        S = S0.copy()
        n = self.n
        for it in range(n + 1):
            info = updatepole(cpen, S)
            if info == DAMAGING_ROUNDING:
                break
            g = (S.fval[:n] - S.fval[n]) @ S.simi
            A = self._A(S)
            d = trstlp(A, -S.conmat[:, n], delta, g)
            preref = -float(np.dot(d, g))
            prerec = S.cval[n] - _cmax0(S.conmat[:, n] + d @ A)
            if not (prerec > 0 and preref < 0):
                break
            cpen = max(cpen, min(-2 * preref / prerec, REALMAX))
            if findpole(cpen, S.cval, S.fval) == n:
                break
        return cpen

    def cobylb(self, maxfilt: int, maxfun: int, ctol: float, cweight: float,
               eta1: float, eta2: float, ftarget: float, gamma1: float,
               gamma2: float, rhobeg: float, rhoend: float,
               constr0: np.ndarray[float, 1], f0: float, x0: np.ndarray[float, 1],
               callback):
        n = self.n
        mcon = constr0.size
        cpenmin = EPS
        # initxfc
        info = INFO_DEFAULT
        sim = np.zeros((n, n + 1))
        for i in range(n):
            sim[i, i] = rhobeg
        for i in range(n):
            sim[i, n] = x0[i]
        simi = np.eye(n) / rhobeg
        evaluated = np.zeros(n + 1, bool)
        fval = np.zeros(n + 1) + REALMAX
        cval = np.zeros(n + 1) + REALMAX
        conmat = np.zeros((mcon, n + 1)) + REALMAX
        for k in range(n + 1):
            x = sim[:, n].copy()
            f = 0.0
            constr = np.zeros(mcon)
            j = 0
            if k == 0:
                j = n
                f = f0
                constr = constr0
            else:
                j = k - 1
                x[j] += rhobeg
                f, constr = self.evaluate(x)
            cstrv = _cmax0(constr)
            evaluated[j] = True
            fval[j] = f
            for i in range(mcon):
                conmat[i, j] = constr[i]
            cval[j] = cstrv
            subinfo = checkbreak_con(maxfun, k, cstrv, ctol, f, ftarget, x)
            if subinfo != INFO_DEFAULT:
                info = subinfo
                break
            if j < n and fval[j] < fval[n]:
                t = fval[j]
                fval[j] = fval[n]
                fval[n] = t
                t = cval[j]
                cval[j] = cval[n]
                cval[n] = t
                _swap_cols(conmat, j, n)
                for i in range(n):
                    sim[i, n] = x[i]
                for c in range(j + 1):
                    sim[j, c] = -rhobeg
        nf = 0
        alle = True
        for e in evaluated:
            if e:
                nf += 1
            else:
                alle = False
        if alle:
            simi = _inv(sim[:, :n].copy())
        S = _Simplex(conmat, cval, fval, sim, simi)
        mf = min(max(maxfilt, 1), maxfun)
        filt = _Filt(mf, n, mcon)
        for i in range(n + 1):
            if evaluated[i]:
                xi = S.sim[:, i] + S.sim[:, n] if i < n else S.sim[:, i].copy()
                filt.save(S.cval[i], ctol, cweight, S.fval[i], xi, S.conmat[:, i].copy())
        if info != INFO_DEFAULT:
            kopt = selectx(filt.ffilt[:filt.nfilt].copy(), filt.cfilt[:filt.nfilt].copy(), cweight, ctol)
            return filt.xfilt[:, kopt].copy(), filt.ffilt[kopt], filt.confilt[:, kopt].copy(), filt.cfilt[kopt], nf, info, 0
        rho = rhobeg
        delta = rhobeg
        cpen = max(cpenmin, min(1000.0, fcratio(S.conmat, S.fval)))
        shortd = False
        ratio = -1.0
        jdrop_tr = 0
        gamma3 = max(1.0, min(0.75 * gamma2, 1.5))
        maxtr = 10 * maxfun
        info = MAXTR_REACHED
        d = np.zeros(n)
        distsq = np.zeros(n + 1)
        ntr = 0
        for tr in range(maxtr):
            ntr = tr
            cpen = self.getcpen(S, cpen, delta)
            subinfo = updatepole(cpen, S)
            if subinfo == DAMAGING_ROUNDING:
                info = subinfo
                break
            adequate_geo = True
            for j in range(n):
                s = 0.0
                for i in range(n):
                    s += S.sim[i, j] * S.sim[i, j]
                if not (s <= 4 * (delta * delta)):
                    adequate_geo = False
            g = (S.fval[:n] - S.fval[n]) @ S.simi
            A = self._A(S)
            d = trstlp(A, -S.conmat[:, n], delta, g)
            dnorm = min(delta, norm(d))
            shortd = dnorm <= 0.1 * rho
            preref = -float(np.dot(d, g))
            prerec = S.cval[n] - _cmax0(S.conmat[:, n] + d @ A)
            prerem = preref + cpen * prerec
            trfail = not (prerem > 1e-06 * min(cpen, 1.0) * rho)
            if shortd or trfail:
                delta *= 0.1
                if delta <= gamma3 * rho:
                    delta = rho
            else:
                x = S.sim[:, n] + d
                fx, cx, cs, nf2 = self._eval_near(x, S, distsq, rhoend, nf, filt, ctol, cweight)
                nf = nf2
                actrem = S.fval[n] + cpen * S.cval[n] - (fx + cpen * cs)
                ratio = redrat(actrem, prerem, eta1)
                delta = trrad(delta, dnorm, eta1, eta2, gamma1, gamma2, ratio)
                if delta <= gamma3 * rho:
                    delta = rho
                ximproved = actrem > 0
                jdrop_tr = setdrop_tr(ximproved, d, delta, rho, S.sim, S.simi)
                subinfo = updatexfc(jdrop_tr, cx, cpen, cs, d, fx, S)
                if subinfo == DAMAGING_ROUNDING:
                    info = subinfo
                    break
                subinfo = checkbreak_con(maxfun, nf, cs, ctol, fx, ftarget, x)
                if subinfo != INFO_DEFAULT:
                    info = subinfo
                    break
            bad_trstep = shortd or trfail or ratio <= 0 or jdrop_tr < 0
            improve_geo = bad_trstep and not adequate_geo
            reduce_rho = bad_trstep and adequate_geo and max(delta, dnorm) <= rho
            if improve_geo:
                geo_ok = True
                jmax = 0
                vmax = -1.0
                for j in range(n):
                    s = 0.0
                    for i in range(n):
                        s += S.sim[i, j] * S.sim[i, j]
                    if not (s <= 4 * (delta * delta)):
                        geo_ok = False
                    if s > vmax:
                        vmax = s
                        jmax = j
                if not geo_ok:
                    jdrop_geo = jmax
                    delbar = delta / 2
                    d = geostep(jdrop_geo, self.amat, self.m_lcon, S, cpen, delbar)
                    x = S.sim[:, n] + d
                    fx, cx, cs, nf2 = self._eval_near(x, S, distsq, rhoend, nf, filt, ctol, cweight)
                    nf = nf2
                    subinfo = updatexfc(jdrop_geo, cx, cpen, cs, d, fx, S)
                    if subinfo == DAMAGING_ROUNDING:
                        info = subinfo
                        break
                    subinfo = checkbreak_con(maxfun, nf, cs, ctol, fx, ftarget, x)
                    if subinfo != INFO_DEFAULT:
                        info = subinfo
                        break
            if reduce_rho:
                if rho <= rhoend:
                    info = SMALL_TR_RADIUS
                    break
                delta = max(0.5 * rho, redrho(rho, rhoend))
                rho = redrho(rho, rhoend)
                cpen = max(cpenmin, min(cpen, fcratio(S.conmat, S.fval)))
                subinfo = updatepole(cpen, S)
                if subinfo == DAMAGING_ROUNDING:
                    info = subinfo
                    break
            if callback is not None:
                if _cob_cb(callback, S.sim[:, n].copy(), S.fval[n], nf, tr, S.cval[n]):
                    info = CALLBACK_TERMINATE
                    break
        x = S.sim[:, n] + d
        if info == SMALL_TR_RADIUS and shortd and norm(x - S.sim[:, n]) > 0.001 * rhoend and nf < maxfun:
            f, constr = self.evaluate(x)
            cstrv = _cmax0(constr)
            nf += 1
            filt.save(cstrv, ctol, cweight, f, x, constr)
        kopt = selectx(filt.ffilt[:filt.nfilt].copy(), filt.cfilt[:filt.nfilt].copy(),
                       max(cpen, cweight), ctol)
        return filt.xfilt[:, kopt].copy(), filt.ffilt[kopt], filt.confilt[:, kopt].copy(), filt.cfilt[kopt], nf, info, ntr

    def _eval_near(self, x: np.ndarray[float, 1], S: _Simplex, distsq: np.ndarray[float, 1],
                   rhoend: float, nf: int, filt: _Filt, ctol: float, cweight: float):
        n = self.n
        dx = x - S.sim[:, n]
        distsq[n] = float(np.sum(dx * dx))
        for j in range(n):
            s = 0.0
            for i in range(n):
                t = x[i] - (S.sim[i, n] + S.sim[i, j])
                s += t * t
            distsq[j] = s
        j = int(np.argmin(distsq))
        if distsq[j] <= (0.0001 * rhoend) ** 2:
            return S.fval[j], S.conmat[:, j].copy(), S.cval[j], nf
        f, constr = self.evaluate(x)
        cstrv = _cmax0(constr)
        filt.save(cstrv, ctol, cweight, f, x.copy(), constr)
        return f, constr, cstrv, nf + 1

def _cob_cb(callback, x: np.ndarray[float, 1], f: float, nf: int, tr: int, cstrv: float) -> bool:
    try:
        from ._common import OptimizeResult
        r = OptimizeResult(x.copy(), f, np.empty(0, float), nit=tr, nfev=nf)
        r.maxcv = cstrv
        if static.function.can_call(callback, intermediate_result=r):
            callback(intermediate_result=r)
        else:
            callback(x.copy())
    except StopIteration:
        return True
    return False

# ------------------------------------------------------------- driver

from ._common import OptimizeResult, _call_scalar, _opt_float, _opt_int, _has_opt
from ._constraints import LinearConstraint, NonlinearConstraint
from ._cons import _u1, _ustr

def _nl_values(c, x: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
    # transform_constraint_function: [lb - v (finite lb)], [v - ub (finite ub)]
    if isinstance(c, LinearConstraint):
        return np.zeros(0)
    elif isinstance(c, NonlinearConstraint):
        v = _u1(c.fun(x))
        lb = c.lb if c.lb.size == v.size else np.full(v.size, c.lb[0])
        ub = c.ub if c.ub.size == v.size else np.full(v.size, c.ub[0])
        out = List[float]()
        for i in range(v.size):
            if lb[i] > -INF_:
                out.append(lb[i] - v[i])
        for i in range(v.size):
            if ub[i] < INF_:
                out.append(v[i] - ub[i])
        return np.array(out, float) if len(out) > 0 else np.zeros(0)
    else:
        t = _ustr(c['type']).lower()
        if t != 'eq' and t != 'ineq':
            raise ValueError(f"Unknown constraint type '{t}'.")
        v = _u1(c['fun'](x))
        if t == 'eq':
            return np.concatenate((-v, v))
        return -v

class _CalcFC[F, A, CS]:
    fun: F
    args: A
    cs: CS
    nfev: int

    def __init__(self, fun: F, args: A, cs: CS):
        self.fun = fun
        self.args = args
        self.cs = cs
        self.nfev = 0

    def __call__(self, x: np.ndarray[float, 1]):
        self.nfev += 1
        f = _call_scalar(self.fun, x.copy(), self.args)
        vals = np.zeros(0)
        if isinstance(self.cs, Tuple):
            if static.len(self.cs) > 0:
                for c in self.cs:
                    vals = np.concatenate((vals, _nl_values(c, x)))
        elif isinstance(self.cs, List):
            for c in self.cs:
                vals = np.concatenate((vals, _nl_values(c, x)))
        elif self.cs is None:
            pass
        else:
            vals = _nl_values(self.cs, x)
        return f, vals

def _lin_rows(cs, n: int):
    # combine LinearConstraint objects -> (A, lb, ub)
    A = np.zeros((0, n))
    lb = np.zeros(0)
    ub = np.zeros(0)
    if isinstance(cs, Tuple):
        if static.len(cs) > 0:
            for c in cs:
                if isinstance(c, LinearConstraint):
                    A = np.concatenate((A, c.A), axis=0)
                    lb = np.concatenate((lb, c.lb))
                    ub = np.concatenate((ub, c.ub))
    elif isinstance(cs, LinearConstraint):
        A = cs.A
        lb = cs.lb
        ub = cs.ub
    elif isinstance(cs, List):
        for c in cs:
            if isinstance(c, LinearConstraint):
                A = np.concatenate((A, c.A), axis=0)
                lb = np.concatenate((lb, c.lb))
                ub = np.concatenate((ub, c.ub))
    return A, lb, ub

def _has_nl(cs) -> bool:
    if isinstance(cs, Tuple):
        r = False
        if static.len(cs) > 0:
            for c in cs:
                if not isinstance(c, LinearConstraint):
                    r = True
        return r
    elif isinstance(cs, List):
        r = False
        for c in cs:
            if not isinstance(c, LinearConstraint):
                r = True
        return r
    elif cs is None:
        return False
    elif isinstance(cs, LinearConstraint):
        return False
    return True

def _minimize_cobyla(fun, x0: np.ndarray[float, 1], args, constraints, bnds,
                     callback, options):
    rhobeg = _opt_float(options, 'rhobeg', 1.0)
    rhoend = _opt_float(options, 'tol', 1e-4)
    maxfun = _opt_int(options, 'maxiter', 1000)
    ctol = _opt_float(options, 'catol', CTOL_DEFAULT)
    ftarget = _opt_float(options, 'f_target', -INF_)
    n = x0.size
    lb = np.full(n, -INF_)
    ub = np.full(n, INF_)
    if bnds is not None:
        lb, ub = bnds
    calc = _CalcFC(fun, args, constraints)
    Al, ll, ul = _lin_rows(constraints, n)
    x = x0.copy()
    if not _has_nl(constraints):
        x = np.minimum(np.maximum(x, lb), ub)
    # separate_LC_into_eq_and_ineq
    rows = List[np.ndarray[float, 1]]()
    rhs = List[float]()
    xl = lb.copy()
    xu = ub.copy()
    for i in range(n):
        if xl[i] != xl[i] or xl[i] < -BOUNDMAX:
            xl[i] = -BOUNDMAX
        if xu[i] != xu[i] or xu[i] > BOUNDMAX:
            xu[i] = BOUNDMAX
    for i in range(n):
        if xl[i] > -BOUNDMAX:
            r = np.zeros(n)
            r[i] = -1.0
            rows.append(r)
            rhs.append(-xl[i])
    for i in range(n):
        if xu[i] < BOUNDMAX:
            r = np.zeros(n)
            r[i] = 1.0
            rows.append(r)
            rhs.append(xu[i])
    k = ll.size
    eq = np.zeros(k, bool)
    for i in range(k):
        eq[i] = ul[i] <= ll[i] + 2 * EPS
    for i in range(k):
        if eq[i]:
            rows.append(-Al[i])
            rhs.append(-(ll[i] + ul[i]) / 2.0)
    for i in range(k):
        if eq[i]:
            rows.append(Al[i].copy())
            rhs.append((ll[i] + ul[i]) / 2.0)
    for i in range(k):
        if (not eq[i]) and ll[i] > -INF_:
            rows.append(-Al[i])
            rhs.append(-ll[i])
    for i in range(k):
        if (not eq[i]) and ul[i] < INF_:
            rows.append(Al[i].copy())
            rhs.append(ul[i])
    m_l = len(rows)
    amat = np.zeros((m_l, n))
    bvec = np.zeros(m_l)
    for i in range(m_l):
        amat[i] = rows[i]
        bvec[i] = rhs[i]
    f0, nl0 = calc(x)
    m_nl = nl0.size
    cob = _Cob(calc, amat, bvec, m_nl, n)
    mmm = m_l + m_nl
    constr = np.zeros(mmm)
    f = moderatef(f0)
    if m_l > 0:
        lc = moderatec(amat @ x - bvec)
        for i in range(m_l):
            constr[i] = lc[i]
    nl0m = moderatec(nl0)
    for i in range(m_nl):
        constr[m_l + i] = nl0m[i]
    # preproc (cobyla defaults)
    if maxfun < n + 2:
        maxfun = n + 2
    if abs(rhobeg - rhoend) < 100.0 * EPS * max(abs(rhobeg), 1.0):
        rhoend = rhobeg
    if rhobeg <= 0 or rhobeg != rhobeg or math.isinf(rhobeg):
        rhobeg = max(10 * rhoend, 1.0) if (math.isfinite(rhoend) and rhoend > 0) else 1.0
    if rhoend <= 0 or rhobeg < rhoend or rhoend != rhoend or math.isinf(rhoend):
        rhoend = max(EPS, min(0.1 * rhobeg, 1e-6))
    if ctol != ctol or ctol < 0:
        ctol = CTOL_DEFAULT
    maxfilt = MAXFILT_DEFAULT
    unit_memo = (mmm + n + 2) * 8
    if maxfilt > MAXHISTMEM / unit_memo:
        maxfilt = int(MAXHISTMEM / unit_memo)
    maxfilt = min(maxfun, max(MIN_MAXFILT, maxfilt))
    xr, fr, cr, cstrv, nf, info, ntr = cob.cobylb(maxfilt, maxfun, ctol, CWEIGHT_DEFAULT,
                                                  0.1, 0.7, ftarget, 0.5, 2.0, rhobeg,
                                                  rhoend, constr, f, x, callback)
    success = False
    message = ''
    if cstrv > ctol:
        message = ('Did not converge to a solution satisfying the constraints. See '
                   '`maxcv` for the magnitude of the violation.')
    else:
        success = info == SMALL_TR_RADIUS or info == FTARGET_ACHIEVED
        message = get_info_string('COBYLA', info)
    r = OptimizeResult(xr, fr, np.empty(0, float), nfev=nf, status=info,
                       success=success, message=message)
    r.maxcv = cstrv
    return r
