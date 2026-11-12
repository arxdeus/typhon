# Ported from scipy 1.18.1 _trlib (trlib, MIT License,
# Copyright (c) 2016--2017 Felix Lenders): trlib_krylov.c,
# trlib_tri_factor.c, trlib_leftmost.c, trlib_eigen_inverse.c,
# trlib_quadratic_zero.c and the _trlib.pyx reverse-communication driver.
# The LAPACK tridiagonal kernels (dpttrf, dpttrs, dptrfs, dlagtm) are
# ported from reference LAPACK. trlib's eigen_inverse seeds rand() with
# time(NULL); this port uses a fixed deterministic start vector instead.

import math
import numpy as np

_EPS = 2.2204460492503131e-16
_EPS_POW_4 = 5.4774205922939014e-07
_EPS_POW_5 = 1.4901161193847656e-08
_EPS_POW_75 = 1.8189894035458565e-12

# krylov return codes
_CLR_CONV_BOUND = 0
_CLR_CONV_INTERIOR = 1
_CLR_APPROX_HARD = 2
_CLR_HARD_INIT_LAM = 4
_CLR_FAIL_HARD = 5
_CLR_UNBDBEL = 6
_CLR_UNLIKE_CONV = 7
_CLR_CONTINUE = 10
_CLR_ITMAX = -1
_CLR_FAIL_NUMERIC = -5
_CLR_FAIL_TTR = -7
_CLR_PCINDEF = -8
_CLR_UNEXPECT_INT = -9
_CLT_CG = 1
_CLT_L = 2
_CLA_TRIVIAL = 0
_CLA_INIT = 1
_CLA_RETRANSF = 2
_CLA_UPDATE_STATIO = 3
_CLA_UPDATE_GRAD = 4
_CLA_UPDATE_DIR = 5
_CLA_NEW_KRYLOV = 6
_CLA_CONV_HARD = 7
_CLA_OBJVAL = 8
_CLS_INIT = 1
_CLS_HOTSTART = 2
_CLS_HOTSTART_G = 3
_CLS_HOTSTART_P = 4
_CLS_VEC_INIT = 7
_CLS_CG_NEW_ITER = 8
_CLS_CG_UPDATE_S = 9
_CLS_CG_UPDATE_GV = 10
_CLS_CG_UPDATE_P = 11
_CLS_LANCZOS_SWT = 12
_CLS_L_UPDATE_P = 13
_CLS_L_CMP_CONV = 14
_CLS_L_CMP_CONV_RT = 15
_CLS_L_CHK_CONV = 16
_CLS_L_NEW_ITER = 17
_CLS_CG_IF_IRBLK_P = 18
_CLS_CG_IF_IRBLK_C = 19
_CLS_CG_IF_IRBLK_N = 20
_CLC_NO_EXP_INV = 0
_CLC_EXP_INV_LOC = 1
_CLC_EXP_INV_GLO = 2
# tri_factor return codes
_TTR_CONV_BOUND = 0
_TTR_CONV_INTERIOR = 1
_TTR_HARD = 2
_TTR_NEWTON_BREAK = 3
_TTR_HARD_INIT_LAM = 4
_TTR_ITMAX = -1
_TTR_FAIL_FACTOR = -2
_TTR_FAIL_LINSOLVE = -3
_TTR_FAIL_EIG = -4
_LMR_CONV = 0
_LMR_ITMAX = -1
_LMR_NEWTON_BREAK = 3
_EIR_CONV = 0
_EIR_ITMAX = -1
_EIR_N_STARTVEC = 5

_DLAMCH_E = 1.1102230246251565e-16
_DLAMCH_S = 2.2250738585072014e-308

# ------------------------------------------------- reference LAPACK kernels

def _dpttrf(n: int, d: np.ndarray[float, 1], od: int, e: np.ndarray[float, 1],
            oe: int) -> int:
    for i in range(n - 1):
        if d[od + i] <= 0.0:
            return i + 1
        ei = e[oe + i]
        e[oe + i] = ei / d[od + i]
        d[od + i + 1] = d[od + i + 1] - e[oe + i] * ei
    if n > 0 and d[od + n - 1] <= 0.0:
        return n
    return 0

def _dpttrs(n: int, d: np.ndarray[float, 1], od: int, e: np.ndarray[float, 1],
            oe: int, b: np.ndarray[float, 1], ob: int):
    if n == 0:
        return
    if n == 1:
        b[ob] = b[ob] * (1.0 / d[od])
        return
    for i in range(1, n):
        b[ob + i] = b[ob + i] - b[ob + i - 1] * e[oe + i - 1]
    b[ob + n - 1] = b[ob + n - 1] / d[od + n - 1]
    i = n - 2
    while i >= 0:
        b[ob + i] = b[ob + i] / d[od + i] - b[ob + i + 1] * e[oe + i]
        i -= 1

def _dptrfs(n: int, d: np.ndarray[float, 1], e: np.ndarray[float, 1],
            df: np.ndarray[float, 1], ef: np.ndarray[float, 1],
            b: np.ndarray[float, 1], x: np.ndarray[float, 1]):
    # iterative refinement of x (forward error bound not needed)
    if n == 0:
        return
    eps = _DLAMCH_E
    safe1 = 4 * _DLAMCH_S
    safe2 = safe1 / eps
    work = np.zeros(2 * n)
    count = 1
    lstres = 3.0
    while True:
        if n == 1:
            bi = b[0]
            dx = d[0] * x[0]
            work[n] = bi - dx
            work[0] = abs(bi) + abs(dx)
        else:
            bi = b[0]
            dx = d[0] * x[0]
            ex = e[0] * x[1]
            work[n] = bi - dx - ex
            work[0] = abs(bi) + abs(dx) + abs(ex)
            for i in range(1, n - 1):
                bi = b[i]
                cx = e[i - 1] * x[i - 1]
                dx = d[i] * x[i]
                ex = e[i] * x[i + 1]
                work[n + i] = bi - cx - dx - ex
                work[i] = abs(bi) + abs(cx) + abs(dx) + abs(ex)
            bi = b[n - 1]
            cx = e[n - 2] * x[n - 2]
            dx = d[n - 1] * x[n - 1]
            work[n + n - 1] = bi - cx - dx
            work[n - 1] = abs(bi) + abs(cx) + abs(dx)
        s = 0.0
        for i in range(n):
            if work[i] > safe2:
                s = max(s, abs(work[n + i]) / work[i])
            else:
                s = max(s, (abs(work[n + i]) + safe1) / (work[i] + safe1))
        berr = s
        if berr > eps and 2.0 * berr <= lstres and count <= 5:
            _dpttrs(n, df, 0, ef, 0, work, n)
            for i in range(n):
                x[i] = x[i] + work[n + i]
            lstres = berr
            count += 1
        else:
            break

def _dlagtm_add(n: int, dl: np.ndarray[float, 1], d: np.ndarray[float, 1],
                du: np.ndarray[float, 1], x: np.ndarray[float, 1],
                b: np.ndarray[float, 1]):
    # b := A*x + b (alpha = beta = 1)
    if n == 1:
        b[0] = b[0] + d[0] * x[0]
    elif n > 1:
        b[0] = b[0] + d[0] * x[0] + du[0] * x[1]
        b[n - 1] = b[n - 1] + dl[n - 2] * x[n - 2] + d[n - 1] * x[n - 1]
        for i in range(1, n - 1):
            b[i] = b[i] + dl[i - 1] * x[i - 1] + d[i] * x[i] + du[i] * x[i + 1]

def _nrm2(n: int, x: np.ndarray[float, 1], o: int = 0) -> float:
    # reference dnrm2 (scaled sum of squares)
    if n < 1:
        return 0.0
    if n == 1:
        return abs(x[o])
    scale = 0.0
    ssq = 1.0
    for i in range(n):
        v = x[o + i]
        if v != 0.0:
            a = abs(v)
            if scale < a:
                ssq = 1.0 + ssq * (scale / a) ** 2
                scale = a
            else:
                ssq += (a / scale) ** 2
    return scale * math.sqrt(ssq)

def _dot(n: int, x: np.ndarray[float, 1], y: np.ndarray[float, 1]) -> float:
    s = 0.0
    for i in range(n):
        s += x[i] * y[i]
    return s

def _quadratic_zero(c_abs: float, c_lin: float, tol: float):
    lin_sq = c_lin * c_lin
    t1 = 0.0
    t2 = 0.0
    if abs(c_abs) > tol * lin_sq:
        q = lin_sq - 4.0 * c_abs
        if abs(q) <= (_EPS * c_lin) * (_EPS * c_lin):
            t1 = -.5 * c_lin
            t2 = t1
        elif q < 0.0:
            return 0.0, 0.0
        else:
            t1 = -.5 * (c_lin + math.copysign(math.sqrt(q), c_lin))
            t2 = c_abs / t1
            if t2 < t1:
                q = t2
                t2 = t1
                t1 = q
    else:
        if c_lin < 0.0:
            t1 = 0.0
            t2 = -c_lin
        else:
            t1 = -c_lin
            t2 = 0.0
    q = (t1 + c_lin) * t1 + c_abs
    dq = 2.0 * t1 + c_lin
    if dq != 0.0:
        t1 = t1 - q / dq
    q = (t2 + c_lin) * t2 + c_abs
    dq = 2.0 * t2 + c_lin
    if dq != 0.0:
        t2 = t2 - q / dq
    return t1, t2

# ------------------------------------------------------------- leftmost

def _leftmost_irreducible(n: int, diag: np.ndarray[float, 1], od: int,
                          offdiag: np.ndarray[float, 1], oo: int, warm: int,
                          leftmost_minor: float, itmax: int, tol_abs: float):
    # returns (ret, leftmost)
    if n == 1:
        return _LMR_CONV, diag[od]
    oabs0 = abs(offdiag[oo])
    oabs1 = abs(offdiag[oo + n - 2])
    low = min(diag[od] - oabs0, diag[od + n - 1] - oabs1)
    up = max(diag[od] + oabs0, diag[od + n - 1] - oabs1)
    for ii in range(1, n - 1):
        oabs1 = abs(offdiag[oo + ii])
        low = min(low, diag[od + ii] - oabs0 - oabs1)
        up = max(up, diag[od + ii] + oabs0 + oabs1)
        oabs0 = oabs1
    leftmost = 0.0
    if warm:
        up = min(up, leftmost_minor)
        leftmost = leftmost_minor - .1 * (up - low)
    else:
        leftmost_minor = 0.0
        leftmost = low + .1 * (up - low)
    itmax = itmax * n
    iter_pr = 0
    prlp = 0.0
    dprlp = 0.0
    ddprlp = 0.0
    leftmost_attempt = 0.0
    dleftmost = 0.0
    while True:
        iter_pr += 1
        if iter_pr > itmax:
            return _LMR_ITMAX, leftmost
        n_neg_piv = 0
        continue_outer_loop = False
        for jj in range(n):
            if jj == 0:
                prlp = diag[od] - leftmost
                dprlp = -1.0
                ddprlp = 0.0
            else:
                o = offdiag[oo + jj - 1]
                obyprlp = o / prlp
                ddprlp = obyprlp * obyprlp * (ddprlp - 2.0 * dprlp * dprlp / prlp)
                dprlp = -1.0 + o * o * dprlp / (prlp * prlp)
                prlp = diag[od + jj] - leftmost - o * obyprlp
            if prlp == 0.0:
                if n_neg_piv == 0 and jj + 1 == n:
                    return _LMR_CONV, leftmost
                else:
                    up = leftmost
                    leftmost = 0.5 * (low + up)
                    continue_outer_loop = True
                    break
            elif prlp < 0.0:
                n_neg_piv += 1
                up = leftmost
                if n_neg_piv > 1:
                    leftmost = 0.5 * (low + up)
                    continue_outer_loop = True
                    break
        if continue_outer_loop:
            continue
        if n_neg_piv == 0:
            low = leftmost
        else:
            up = leftmost
        if up - low <= tol_abs * max(1.0, max(abs(low), abs(up))) or abs(prlp) <= tol_abs:
            return _LMR_CONV, leftmost
        if warm:
            if warm == 2 or (warm == 1 and up - low >= .1 * max(1.0, abs(leftmost))):
                quad_lin = -(2.0 * leftmost + prlp + (leftmost - leftmost_minor) * dprlp)
                quad_abs = -((leftmost - leftmost_minor) * prlp + leftmost * (quad_lin + leftmost))
                leftmost_attempt, _ = _quadratic_zero(quad_abs, quad_lin, _EPS_POW_75)
                dleftmost = leftmost_attempt - leftmost
            if warm > 2 or (warm == 1 and up - low < .1 * max(1.0, abs(leftmost))):
                quad_qua = -(dprlp + .5 * (leftmost - leftmost_minor) * ddprlp)
                quad_lin = -(prlp + (leftmost - leftmost_minor) * dprlp)
                quad_abs = (leftmost_minor - leftmost) * prlp
                if warm == 4 or abs(quad_qua) <= _EPS * _EPS or \
                        quad_lin * quad_lin - 4.0 * quad_qua * quad_abs <= 0.0:
                    dleftmost = -quad_abs / quad_lin
                    leftmost_attempt = leftmost + dleftmost
                else:
                    quad_lin = quad_lin / quad_qua
                    quad_abs = quad_abs / quad_qua
                    dleftmost, _ = _quadratic_zero(quad_abs, quad_lin, _EPS_POW_75)
                    leftmost_attempt = leftmost + dleftmost
        else:
            dleftmost = -prlp / dprlp
            leftmost_attempt = leftmost + dleftmost
        if low <= leftmost_attempt and leftmost_attempt <= up:
            if abs(dleftmost) <= tol_abs * max(1.0, max(abs(low), abs(up))):
                return _LMR_NEWTON_BREAK, leftmost
            leftmost = leftmost_attempt
        else:
            if warm:
                quad_lin = -(prlp + (leftmost - leftmost_minor) * dprlp)
                quad_abs = (leftmost_minor - leftmost) * prlp
                dleftmost = -quad_abs / quad_lin
                leftmost_attempt = leftmost + dleftmost
            if low > leftmost_attempt or up < leftmost_attempt:
                dleftmost = -prlp / dprlp
                leftmost_attempt = leftmost + dleftmost
            if low > leftmost_attempt or up < leftmost_attempt:
                leftmost = .5 * (low + up)
    return _LMR_CONV, leftmost

def _eigen_inverse(n: int, diag: np.ndarray[float, 1], od: int,
                   offdiag: np.ndarray[float, 1], oo: int, lam_init: float,
                   itmax: int, tol_abs: float, diag_fac: np.ndarray[float, 1],
                   offdiag_fac: np.ndarray[float, 1], eig: np.ndarray[float, 1],
                   oe: int) -> int:
    # inverse iteration for the eigenvector to the leftmost eigenvalue;
    # diag_fac/offdiag_fac are written at offsets od/oo, eig at oe
    pert = 0.0
    minuslam = -lam_init
    while pert <= 1.0 / _EPS:
        for i in range(n):
            diag_fac[od + i] = diag[od + i] + minuslam
        for i in range(n - 1):
            offdiag_fac[oo + i] = offdiag[oo + i]
        info_fac = _dpttrf(n, diag_fac, od, offdiag_fac, oo)
        if info_fac == 0:
            break
        if pert == 0.0:
            pert = _EPS_POW_4 * max(1.0, -lam_init)
        else:
            pert = 10.0 * pert
        minuslam = pert - lam_init
    for jj in range(_EIR_N_STARTVEC):
        # deterministic start vectors (trlib: rand() seeded by time)
        for kk in range(n):
            eig[oe + kk] = 1.0 / (1.0 + float((kk * 7 + jj * 13) % 17))
        invnorm = 1.0 / _nrm2(n, eig, oe)
        for kk in range(n):
            eig[oe + kk] *= invnorm
        it = 0
        while True:
            it += 1
            if it > itmax:
                break
            _dpttrs(n, diag_fac, od, offdiag_fac, oo, eig, oe)
            invnorm = 1.0 / _nrm2(n, eig, oe)
            for kk in range(n):
                eig[oe + kk] *= invnorm
            if abs(invnorm - pert) <= tol_abs:
                return _EIR_CONV
    return _EIR_ITMAX

# ------------------------------------------------------- tri_factor_min

class _TrlibState:
    # trlib iwork/fwork contents (single irreducible block: scipy calls
    # trlib with ctl_invariant = 0, so nirblk is always 1)
    itmax: int
    n: int
    status: int
    ii: int
    pos_def: int
    interior: int
    warm_leftmost: int
    ileftmost: int
    warm_lam0: int
    warm_lam: int
    lanczos_switch: int
    exit_tri: int
    sub_fail_tri: int
    iter_tri: int
    outerstatus: int
    nirblk: int
    stop_i: float
    stop_b: float
    v_g: float
    p_Hp: float
    cgl: float
    cglm: float
    lam0: float
    lam: float
    obj: float
    s_Mp: float
    p_Mp: float
    s_Ms: float
    sigma: float
    raymax: float
    raymin: float
    alpha: np.ndarray[float, 1]
    beta: np.ndarray[float, 1]
    neglin: np.ndarray[float, 1]
    h0: np.ndarray[float, 1]
    h: np.ndarray[float, 1]
    delta: np.ndarray[float, 1]
    delta_fac0: np.ndarray[float, 1]
    delta_fac: np.ndarray[float, 1]
    gamma: np.ndarray[float, 1]
    gamma_fac0: np.ndarray[float, 1]
    gamma_fac: np.ndarray[float, 1]
    leftmost: np.ndarray[float, 1]
    regdiag: np.ndarray[float, 1]
    convhist: np.ndarray[float, 1]
    # vectors managed by the _trlib.pyx driver
    s: np.ndarray[float, 1]
    g: np.ndarray[float, 1]
    v: np.ndarray[float, 1]
    gm: np.ndarray[float, 1]
    p: np.ndarray[float, 1]
    Hp: np.ndarray[float, 1]
    Q: np.ndarray[float, 2]
    init: int

    def __init__(self, n: int):
        self.n = n
        itmax = int(min(1e9 / max(n, 1), 2.0 * n))
        self.itmax = itmax
        m = itmax + 1
        self.status = 0
        self.ii = 0
        self.pos_def = 0
        self.interior = 0
        self.warm_leftmost = 0
        self.ileftmost = 0
        self.warm_lam0 = 0
        self.warm_lam = 0
        self.lanczos_switch = -1
        self.exit_tri = 0
        self.sub_fail_tri = 0
        self.iter_tri = 0
        self.outerstatus = 0
        self.nirblk = 1
        self.stop_i = 0.0
        self.stop_b = 0.0
        self.v_g = 0.0
        self.p_Hp = 0.0
        self.cgl = 0.0
        self.cglm = 0.0
        self.lam0 = 0.0
        self.lam = 0.0
        self.obj = 0.0
        self.s_Mp = 0.0
        self.p_Mp = 0.0
        self.s_Ms = 0.0
        self.sigma = 0.0
        self.raymax = 0.0
        self.raymin = 0.0
        self.alpha = np.zeros(m)
        self.beta = np.zeros(m)
        self.neglin = np.zeros(m)
        self.h0 = np.zeros(m)
        self.h = np.zeros(m)
        self.delta = np.zeros(m)
        self.delta_fac0 = np.zeros(m)
        self.delta_fac = np.zeros(m)
        self.gamma = np.zeros(m)
        self.gamma_fac0 = np.zeros(m)
        self.gamma_fac = np.zeros(m)
        self.leftmost = np.zeros(m)
        self.regdiag = np.zeros(m)
        self.convhist = np.zeros(m)
        self.s = np.zeros(n)
        self.g = np.zeros(n)
        self.v = np.zeros(n)
        self.gm = np.zeros(n)
        self.p = np.zeros(n)
        self.Hp = np.zeros(n)
        self.Q = np.zeros((m, n))
        self.init = _CLS_INIT

def _fac_shift(n0: int, diag: np.ndarray[float, 1], offdiag: np.ndarray[float, 1],
               lam: float, diag_lam: np.ndarray[float, 1],
               diag_fac: np.ndarray[float, 1], offdiag_fac: np.ndarray[float, 1]) -> int:
    for i in range(n0):
        diag_lam[i] = diag[i] + lam
        diag_fac[i] = diag_lam[i]
    for i in range(n0 - 1):
        offdiag_fac[i] = offdiag[i]
    return _dpttrf(n0, diag_fac, 0, offdiag_fac, 0)

def _tri_obj(n0: int, st: _TrlibState, diag: np.ndarray[float, 1],
             offdiag: np.ndarray[float, 1], sol: np.ndarray[float, 1]) -> float:
    w = np.zeros(n0)
    for i in range(n0):
        w[i] = -2.0 * st.neglin[i]
    _dlagtm_add(n0, offdiag, diag, offdiag, sol, w)
    return 0.5 * _dot(n0, sol, w)

def _tri_factor_min(st: _TrlibState, n0: int, diag: np.ndarray[float, 1],
                    offdiag: np.ndarray[float, 1], radius: float, itmax: int,
                    tol_rel: float, tol_newton_tiny: float,
                    warm_fac0: int) -> int:
    # trlib_tri_factor_min for one irreducible block, equality = 0,
    # refine = 1; solution in st.h0 / st.h, multiplier in st.lam0 / st.lam
    neglin = st.neglin
    sol0 = st.h0
    sol = st.h
    diag_fac0 = st.delta_fac0
    offdiag_fac0 = st.gamma_fac0
    diag_lam = np.zeros(n0)
    info_fac = 0
    ret = 0
    norm_sol0 = 0.0
    st.iter_tri = 0
    st.sub_fail_tri = 0
    for i in range(n0):
        sol[i] = 0.0
    if st.warm_lam0:
        if not warm_fac0:
            info_fac = _fac_shift(n0, diag, offdiag, st.lam0, diag_lam, diag_fac0, offdiag_fac0)
            if info_fac != 0:
                st.warm_lam0 = 0
            else:
                warm_fac0 = 1
        if warm_fac0:
            for i in range(n0):
                sol0[i] = neglin[i]
            _dpttrs(n0, diag_fac0, 0, offdiag_fac0, 0, sol0, 0)
            norm_sol0 = _nrm2(n0, sol0)
            st.warm_lam0 = 1 if norm_sol0 >= radius else 0
    if not st.warm_lam0:
        st.lam0 = 0.0
        for i in range(n0):
            diag_fac0[i] = diag[i]
            sol0[i] = neglin[i]
        for i in range(n0 - 1):
            offdiag_fac0[i] = offdiag[i]
        info_fac = _dpttrf(n0, diag_fac0, 0, offdiag_fac0, 0)
        if info_fac == 0:
            for i in range(n0):
                sol0[i] = neglin[i]
            _dpttrs(n0, diag_fac0, 0, offdiag_fac0, 0, sol0, 0)
            norm_sol0 = _nrm2(n0, sol0)
        if info_fac != 0:
            sf, lm = _leftmost_irreducible(n0, diag, 0, offdiag, 0,
                                           st.warm_leftmost, st.leftmost[0],
                                           1000, _EPS_POW_75)
            st.sub_fail_tri = sf
            st.leftmost[0] = lm
            pert_low = -_EPS_POW_5 * abs(lm)
            pert_up = 1.0 / _EPS
            jj = 0
            lam_pert = 0.0
            while jj < 50:
                st.lam0 = -lm + lam_pert
                info_fac = _fac_shift(n0, diag, offdiag, st.lam0, diag_lam, diag_fac0, offdiag_fac0)
                if info_fac == 0:
                    pert_up = lam_pert
                    for i in range(n0):
                        sol0[i] = neglin[i]
                    _dpttrs(n0, diag_fac0, 0, offdiag_fac0, 0, sol0, 0)
                    norm_sol0 = _nrm2(n0, sol0)
                    if norm_sol0 >= radius:
                        break
                    else:
                        lam_pert = .5 * (pert_low + pert_up)
                else:
                    pert_low = lam_pert
                    if pert_up == 1.0 / _EPS:
                        if lam_pert == 0.0:
                            lam_pert = (1.0 + abs(lm)) * _EPS_POW_75
                        else:
                            lam_pert = 2.0 * lam_pert
                    else:
                        lam_pert = .5 * (pert_low + pert_up)
                jj += 1
            if info_fac != 0:
                lam_pert = pert_up
                st.lam0 = -lm + lam_pert
                info_fac = _fac_shift(n0, diag, offdiag, st.lam0, diag_lam, diag_fac0, offdiag_fac0)
                if info_fac != 0:
                    return _TTR_FAIL_FACTOR
                _dpttrs(n0, diag_fac0, 0, offdiag_fac0, 0, sol0, 0)
                norm_sol0 = _nrm2(n0, sol0)
    if norm_sol0 >= radius:
        w = np.zeros(n0)
        while True:
            ferr = 1.0 / norm_sol0
            for i in range(n0):
                w[i] = sol0[i] * ferr
            invD_norm_w_sq = w[0] * w[0] / diag_fac0[0]
            for jj in range(1, n0):
                w[jj] = w[jj] - offdiag_fac0[jj - 1] * w[jj - 1]
                invD_norm_w_sq += w[jj] * w[jj] / diag_fac0[jj]
            dlam = (norm_sol0 - radius) / (radius * invD_norm_w_sq)
            st.iter_tri += 1
            if st.lam0 + dlam == st.lam0 or \
                    abs(dlam) <= tol_newton_tiny * max(1.0, abs(st.lam0)) or \
                    st.iter_tri > itmax:
                if st.iter_tri > itmax:
                    ret = _TTR_ITMAX
                    break
                ret = _TTR_NEWTON_BREAK
                break
            st.lam0 += dlam
            info_fac = _fac_shift(n0, diag, offdiag, st.lam0, diag_lam, diag_fac0, offdiag_fac0)
            if info_fac != 0:
                return _TTR_FAIL_FACTOR
            for i in range(n0):
                sol0[i] = neglin[i]
            _dpttrs(n0, diag_fac0, 0, offdiag_fac0, 0, sol0, 0)
            _dptrfs(n0, diag_lam, offdiag, diag_fac0, offdiag_fac0, neglin, sol0)
            norm_sol0 = _nrm2(n0, sol0)
            if norm_sol0 - radius <= tol_rel * radius:
                ret = _TTR_CONV_BOUND
                break
    st.warm_lam0 = 1
    if abs(radius - norm_sol0) >= _EPS_POW_5 * radius:
        if st.lam0 == 0.0:
            ret = _TTR_CONV_INTERIOR
        else:
            st.sub_fail_tri = _eigen_inverse(n0, diag, 0, offdiag, 0,
                                             st.leftmost[0], 10, _EPS_POW_5,
                                             st.delta_fac, st.gamma_fac, sol, 0)
            dot = _dot(n0, sol0, sol)
            ferr, berr = _quadratic_zero(norm_sol0 * norm_sol0 - radius * radius,
                                         2.0 * dot, _EPS_POW_75)
            dot2 = _dot(n0, sol, neglin)
            lm = st.leftmost[0]
            if .5 * lm * ferr * ferr + (lm * dot - dot2) * ferr <= \
                    .5 * lm * berr * berr + (lm * dot - dot2) * berr:
                for i in range(n0):
                    sol0[i] += ferr * sol[i]
            else:
                for i in range(n0):
                    sol0[i] += berr * sol[i]
            ret = _TTR_HARD_INIT_LAM
    st.lam = st.lam0
    st.warm_lam = 1
    for i in range(n0):
        sol[i] = sol0[i]
    st.obj = _tri_obj(n0, st, diag, offdiag, sol)
    return ret

def _regularize_posdef(n: int, diag: np.ndarray[float, 1],
                       offdiag: np.ndarray[float, 1], tol_away: float,
                       security_step: float, regdiag: np.ndarray[float, 1]):
    regdiag[0] = 0.0
    if diag[0] <= tol_away:
        regdiag[0] = security_step * tol_away
    diag_fac = diag[0] + regdiag[0]
    for pivot in range(n - 1):
        regdiag[pivot + 1] = 0.0
        o2 = offdiag[pivot] * offdiag[pivot]
        if diag[pivot + 1] - o2 / diag_fac <= tol_away * diag_fac:
            regdiag[pivot + 1] = security_step * abs(o2 / diag_fac - diag[pivot + 1])
        diag_fac = diag[pivot + 1] + regdiag[pivot + 1] - o2 / diag_fac

# ---------------------------------------------------------- krylov_min

class _Rc:
    # reverse-communication outputs of trlib_krylov_min
    action: int
    it: int
    ityp: int
    flt1: float
    flt2: float
    flt3: float

    def __init__(self):
        self.action = 0
        self.it = 0
        self.ityp = 0
        self.flt1 = 0.0
        self.flt2 = 0.0
        self.flt3 = 0.0

def _tri_min(st: _TrlibState, radius: float, warm_fac0: int) -> int:
    return _tri_factor_min(st, st.ii + 1, st.delta, st.gamma, radius,
                           100 + 3 * st.ii, _EPS, 1e-11, warm_fac0)

def _krylov_internal(st: _TrlibState, rc: _Rc, init: int, radius: float,
                     itmax: int, itmax_lanczos: int, tol_rel_i: float,
                     tol_rel_b: float, zero: float, obj_lo: float,
                     g_dot_g: float, v_dot_g: float, p_dot_Hp: float) -> int:
    # trlib_krylov_min_internal with equality = 0, ctl_invariant = 0,
    # earlyterm = 1, tol_abs_i = tol_abs_b = 0
    returnvalue = _CLR_CONTINUE
    rc.it = st.ii
    if init == _CLS_INIT or init == _CLS_HOTSTART or init == _CLS_HOTSTART_P:
        st.status = init
    while True:
        status = st.status
        if status == _CLS_INIT:
            st.ii = 0
            rc.it = 0
            st.pos_def = 1
            st.interior = 1
            st.warm_leftmost = 0
            st.nirblk = 1
            st.warm_lam0 = 0
            st.warm_lam = 0
            st.lanczos_switch = -1
            st.exit_tri = 0
            st.sub_fail_tri = 0
            st.iter_tri = 0
            for i in range(itmax):
                st.gamma[i] = 0.0
            rc.ityp = _CLT_CG
            st.status = _CLS_VEC_INIT
            rc.action = _CLA_INIT
        elif status == _CLS_VEC_INIT:
            if v_dot_g <= 0.0 and g_dot_g > 0.0:
                rc.ityp = _CLT_CG
                rc.action = _CLA_TRIVIAL
                returnvalue = _CLR_PCINDEF
            else:
                eta_i = 0.0
                eta_b = 0.0
                if tol_rel_i >= 0.0:
                    eta_i = tol_rel_i
                if abs(tol_rel_i + 1.0) < 1e-8:
                    eta_i = min(5e-1, math.sqrt(math.sqrt(v_dot_g)))
                if abs(tol_rel_i + 2.0) < 1e-8:
                    eta_i = min(5e-1, math.sqrt(v_dot_g))
                st.stop_i = max(0.0, eta_i * math.sqrt(v_dot_g))
                st.stop_i = st.stop_i * st.stop_i
                if tol_rel_b >= 0.0:
                    eta_b = tol_rel_b
                if abs(tol_rel_b + 1.0) < 1e-8 or abs(tol_rel_b + 3.0) < 1e-8:
                    eta_b = min(5e-1, math.sqrt(math.sqrt(v_dot_g)))
                if abs(tol_rel_b + 2.0) < 1e-8 or abs(tol_rel_b + 4.0) < 1e-8:
                    eta_b = min(5e-1, math.sqrt(v_dot_g))
                if tol_rel_b < -2.5:
                    eta_b = max(1e-6, eta_b)
                st.stop_b = max(0.0, eta_b * math.sqrt(v_dot_g))
                st.v_g = v_dot_g
                st.p_Hp = p_dot_Hp
                st.neglin[0] = -math.sqrt(v_dot_g)
                st.cgl = 1.0
                st.cglm = 1.0
                st.sigma = 1.0
                st.leftmost[0] = 0.0
                st.lam = 0.0
                st.obj = 0.0
                st.s_Mp = 0.0
                st.p_Mp = 0.0
                st.s_Ms = 0.0
                st.p_Mp = st.v_g
                st.raymax = st.p_Hp / st.p_Mp
                st.raymin = st.raymax
                st.delta[0] = 0.0
                rc.ityp = _CLT_CG
                st.status = _CLS_CG_NEW_ITER
                rc.action = _CLA_TRIVIAL
        elif status == _CLS_CG_NEW_ITER:
            ii = st.ii
            if abs(st.p_Hp) <= zero:
                rc.action = _CLA_TRIVIAL
                st.status = _CLS_LANCZOS_SWT
            else:
                st.alpha[ii] = st.v_g / st.p_Hp
                st.delta[ii] += st.p_Hp / st.v_g
                st.pos_def = 1 if (st.pos_def and st.alpha[ii] > 0.0) else 0
                if ii > 0:
                    st.sigma = -math.copysign(1.0, st.alpha[ii - 1]) * st.sigma
                st.cglm = st.cgl
                st.cgl = st.sigma / math.sqrt(st.v_g)
                if ii > 0:
                    st.s_Mp = st.beta[ii - 1] * (st.s_Mp + st.alpha[ii - 1] * st.p_Mp)
                    st.p_Mp = st.v_g + st.beta[ii - 1] * st.beta[ii - 1] * st.p_Mp
                sp_Msp = st.s_Ms + st.alpha[ii] * (2.0 * st.s_Mp + st.alpha[ii] * st.p_Mp)
                st.raymax = max(st.raymax, st.p_Hp / st.p_Mp)
                st.raymin = min(st.raymin, st.p_Hp / st.p_Mp)
                st.interior = 1 if (st.interior and st.pos_def and sp_Msp < radius * radius) else 0
                if st.interior:
                    st.s_Ms = sp_Msp
                    st.obj = st.obj - .5 * st.alpha[ii] * st.alpha[ii] * st.p_Hp
                    rc.ityp = _CLT_CG
                    st.status = _CLS_CG_UPDATE_S
                    rc.flt1 = st.alpha[ii]
                    rc.action = _CLA_UPDATE_STATIO
                else:
                    warm_fac0 = 0
                    if st.warm_lam0:
                        warm_fac0 = 1 if st.delta_fac0[ii - 1] != 0.0 else 0
                        if warm_fac0:
                            st.gamma_fac0[ii - 1] = st.gamma[ii - 1] / st.delta_fac0[ii - 1]
                            st.delta_fac0[ii] = st.delta[ii] + st.lam0 - \
                                st.gamma[ii - 1] * st.gamma[ii - 1] / st.delta_fac0[ii - 1]
                            warm_fac0 = 1 if st.delta_fac0[ii] > 0.0 else 0
                    st.exit_tri = _tri_min(st, radius, warm_fac0)
                    if st.exit_tri < 0 and st.exit_tri != _TTR_NEWTON_BREAK:
                        rc.ityp = _CLT_CG
                        rc.action = _CLA_RETRANSF
                        returnvalue = _CLR_FAIL_TTR
                    elif st.exit_tri == _TTR_CONV_INTERIOR and st.pos_def:
                        rc.ityp = _CLT_CG
                        rc.action = _CLA_RETRANSF
                        returnvalue = _CLR_UNEXPECT_INT
                    else:
                        rc.ityp = _CLT_CG
                        st.status = _CLS_CG_UPDATE_S
                        rc.action = _CLA_TRIVIAL
        elif status == _CLS_CG_UPDATE_S:
            rc.ityp = _CLT_CG
            st.status = _CLS_CG_UPDATE_GV
            rc.flt1 = st.alpha[st.ii]
            rc.flt2 = st.cgl
            rc.action = _CLA_UPDATE_GRAD
        elif status == _CLS_CG_UPDATE_GV:
            returnvalue = _krylov_cg_update_gv(st, rc, radius, zero, obj_lo,
                                               g_dot_g, v_dot_g)
        elif status == _CLS_CG_UPDATE_P:
            st.p_Hp = p_dot_Hp
            st.ii += 1
            if st.ii >= itmax:
                rc.action = _CLA_TRIVIAL if st.interior else _CLA_RETRANSF
                rc.ityp = _CLT_CG
                returnvalue = _CLR_ITMAX
            else:
                rc.ityp = _CLT_CG
                st.status = _CLS_CG_NEW_ITER
                rc.action = _CLA_TRIVIAL
        elif status == _CLS_HOTSTART:
            returnvalue = _krylov_hotstart(st, rc, radius, obj_lo)
        elif status == _CLS_HOTSTART_P:
            n1 = st.ii + 1
            _regularize_posdef(n1, st.delta, st.gamma, 1e-12, 10.0, st.regdiag)
            for i in range(n1):
                st.delta[i] += st.regdiag[i]
            for i in range(n1):
                st.h[i] = 0.0
            st.warm_lam0 = 0
            st.warm_lam = 0
            st.warm_leftmost = 0
            st.exit_tri = _tri_min(st, radius, 0)
            for i in range(n1):
                st.delta[i] -= st.regdiag[i]
            rc.action = _CLA_RETRANSF
            returnvalue = st.exit_tri
        else:
            returnvalue = _krylov_lanczos(st, rc, status, radius, itmax,
                                          itmax_lanczos, zero, obj_lo,
                                          g_dot_g, v_dot_g, p_dot_Hp)
        # trlib compares the *pointer* `action` with TRLIB_CLA_TRIVIAL, so
        # the internal loop always performs exactly one state transition
        break
    return returnvalue

def _krylov_cg_update_gv(st: _TrlibState, rc: _Rc, radius: float, zero: float,
                         obj_lo: float, g_dot_g: float, v_dot_g: float) -> int:
    ii = st.ii
    if v_dot_g != v_dot_g:
        if st.interior:
            rc.action = _CLA_TRIVIAL
        else:
            rc.ityp = _CLT_CG
            rc.action = _CLA_RETRANSF
        return _CLR_FAIL_NUMERIC
    if g_dot_g > 0.0 and v_dot_g <= 0.0:
        if st.interior:
            rc.action = _CLA_TRIVIAL
        else:
            rc.ityp = _CLT_CG
            rc.action = _CLA_RETRANSF
        return _CLR_PCINDEF
    if g_dot_g <= zero:
        rc.action = _CLA_TRIVIAL if st.interior else _CLA_RETRANSF
        rc.ityp = _CLT_CG
        st.gamma[ii] = 0.0
        return _CLR_FAIL_HARD
    st.beta[ii] = v_dot_g / st.v_g
    st.delta[ii + 1] = (v_dot_g * st.p_Hp) / (st.v_g * st.v_g)
    st.gamma[ii] = abs(math.sqrt(v_dot_g) * st.p_Hp / (math.sqrt(st.v_g) * st.v_g))
    st.v_g = v_dot_g
    if st.interior:
        st.convhist[ii] = math.sqrt(st.v_g) / st.stop_i
    else:
        st.convhist[ii] = (st.gamma[ii] * abs(st.h[ii])) / st.stop_b
    if st.interior and st.v_g <= st.stop_i:
        rc.ityp = _CLT_CG
        rc.action = _CLA_TRIVIAL
        return _CLR_CONV_INTERIOR
    elif (not st.interior) and st.gamma[ii] * abs(st.h[ii]) <= st.stop_b:
        rc.ityp = _CLT_CG
        rc.action = _CLA_RETRANSF
        return _CLR_CONV_BOUND
    if (not st.interior) and ii > 10 and st.convhist[ii - 10] > 1e-1 * st.convhist[ii]:
        doit = True
        for cit in range(ii - 10, ii):
            if st.convhist[cit + 1] > st.convhist[cit]:
                doit = False
        if doit:
            rc.ityp = _CLT_CG
            rc.action = _CLA_TRIVIAL if st.interior else _CLA_RETRANSF
            return _CLR_UNLIKE_CONV
        rc.ityp = _CLT_CG
        st.status = _CLS_CG_UPDATE_P
        rc.flt1 = -1.0
        rc.flt2 = st.beta[ii]
        rc.action = _CLA_UPDATE_DIR
        return _CLR_CONTINUE
    elif st.obj != st.obj or st.obj < obj_lo:
        rc.ityp = _CLT_CG
        rc.action = _CLA_TRIVIAL if st.interior else _CLA_RETRANSF
        return _CLR_UNBDBEL
    rc.ityp = _CLT_CG
    st.status = _CLS_CG_UPDATE_P
    rc.flt1 = -1.0
    rc.flt2 = st.beta[ii]
    rc.action = _CLA_UPDATE_DIR
    return _CLR_CONTINUE

def _krylov_hotstart(st: _TrlibState, rc: _Rc, radius: float, obj_lo: float) -> int:
    ii = st.ii
    st.exit_tri = _tri_min(st, radius, 0)
    if st.exit_tri < 0 and st.exit_tri != _TTR_NEWTON_BREAK:
        rc.ityp = _CLT_CG
        rc.action = _CLA_RETRANSF
        return _CLR_FAIL_TTR
    if st.exit_tri == _TTR_HARD_INIT_LAM:
        rc.ityp = _CLT_CG
        rc.action = _CLA_RETRANSF
        return _CLR_HARD_INIT_LAM
    # earlyterm
    rc.ityp = _CLT_CG
    rc.action = _CLA_RETRANSF
    return _CLR_UNLIKE_CONV

def _krylov_lanczos(st: _TrlibState, rc: _Rc, status: int, radius: float,
                    itmax: int, itmax_lanczos: int, zero: float, obj_lo: float,
                    g_dot_g: float, v_dot_g: float, p_dot_Hp: float) -> int:
    ii = st.ii
    if status == _CLS_LANCZOS_SWT:
        st.lanczos_switch = ii
        if ii > 0:
            st.sigma = -math.copysign(1.0, st.alpha[ii - 1]) * st.sigma
        rc.ityp = _CLT_L
        st.status = _CLS_L_UPDATE_P
        rc.flt1 = st.sigma / math.sqrt(st.v_g)
        rc.flt2 = 0.0
        rc.action = _CLA_UPDATE_DIR
        return _CLR_CONTINUE
    elif status == _CLS_L_UPDATE_P:
        if abs(p_dot_Hp) <= zero:
            # ctl_invariant = 0 <= EXP_INV_GLO
            rc.ityp = _CLT_CG
            rc.action = _CLA_RETRANSF
            return _CLR_FAIL_HARD
        st.delta[ii] = p_dot_Hp
        st.raymax = max(st.raymax, p_dot_Hp)
        st.raymin = min(st.raymin, p_dot_Hp)
        warm_fac0 = 0
        if st.warm_lam0:
            warm_fac0 = 1 if st.delta_fac0[ii - 1] != 0.0 else 0
            if warm_fac0:
                st.gamma_fac0[ii - 1] = st.gamma[ii - 1] / st.delta_fac0[ii - 1]
                st.delta_fac0[ii] = st.delta[ii] + st.lam0 - \
                    st.gamma[ii - 1] * st.gamma[ii - 1] / st.delta_fac0[ii - 1]
                warm_fac0 = 1 if st.delta_fac0[ii] > 0.0 else 0
        st.exit_tri = _tri_min(st, radius, warm_fac0)
        if st.exit_tri < 0 and st.exit_tri != _TTR_NEWTON_BREAK:
            rc.ityp = _CLT_L
            rc.action = _CLA_RETRANSF
            return _CLR_FAIL_TTR
        if st.exit_tri == _TTR_HARD_INIT_LAM:
            rc.ityp = _CLT_L
            rc.action = _CLA_RETRANSF
            return _CLR_HARD_INIT_LAM
        if ii < 2:
            rc.flt1 = -st.delta[ii] / st.gamma[ii - 1]
            rc.flt2 = -st.gamma[ii - 1]
            rc.flt3 = 1.0
        else:
            rc.flt1 = -st.delta[ii] / st.gamma[ii - 1]
            rc.flt2 = -st.gamma[ii - 1] / st.gamma[ii - 2]
            rc.flt3 = 1.0
        if ii == st.lanczos_switch:
            rc.flt1 = -st.delta[ii] / math.sqrt(st.v_g) * st.sigma
            rc.flt2 = -st.gamma[ii - 1] * st.cgl
            rc.flt3 = st.gamma[ii - 1] / math.sqrt(st.v_g)
            st.cgl = 1.0
            st.cglm = 1.0
        if ii == 0:
            # irblk[nirblk-1] == 0 (new irreducible block at the start)
            rc.flt1 = -st.delta[ii] / st.gamma[ii - 1]
            rc.flt2 = 0.0
            rc.flt3 = 1.0
        rc.ityp = _CLT_L
        rc.action = _CLA_UPDATE_GRAD
        st.status = _CLS_L_CMP_CONV
        return _CLR_CONTINUE
    elif status == _CLS_L_CMP_CONV:
        if v_dot_g != v_dot_g:
            if st.interior:
                rc.action = _CLA_TRIVIAL
            else:
                rc.ityp = _CLT_L
                rc.action = _CLA_RETRANSF
            return _CLR_FAIL_NUMERIC
        if v_dot_g <= 0.0 and g_dot_g > 0.0:
            if st.interior:
                rc.action = _CLA_TRIVIAL
            else:
                rc.ityp = _CLT_L
                rc.action = _CLA_RETRANSF
            return _CLR_PCINDEF
        ret = _CLR_CONTINUE
        if g_dot_g <= zero:
            if st.interior:
                rc.action = _CLA_TRIVIAL
            else:
                rc.ityp = _CLT_L
                rc.action = _CLA_RETRANSF
            ret = _CLR_APPROX_HARD
        st.gamma[ii] = math.sqrt(v_dot_g)
        st.v_g = v_dot_g * st.h[ii] * st.h[ii]
        rc.ityp = _CLT_L
        rc.action = _CLA_TRIVIAL
        st.status = _CLS_L_CHK_CONV
        return ret
    elif status == _CLS_L_CHK_CONV:
        if st.interior:
            st.convhist[ii] = math.sqrt(st.v_g) / math.sqrt(st.stop_i)
        else:
            st.convhist[ii] = st.v_g / math.sqrt(st.stop_b)
        if st.exit_tri != _TTR_CONV_INTERIOR and st.v_g <= math.sqrt(st.stop_b):
            rc.ityp = _CLT_L
            rc.action = _CLA_RETRANSF
            return st.exit_tri
        elif st.exit_tri == _TTR_CONV_INTERIOR and st.v_g <= math.sqrt(st.stop_i):
            rc.ityp = _CLT_L
            rc.action = _CLA_RETRANSF
            return st.exit_tri
        if (not st.interior) and ii > 10 and st.convhist[ii - 10] > 1e-1 * st.convhist[ii]:
            doit = True
            for cit in range(ii - 10, ii):
                if st.convhist[cit + 1] > st.convhist[cit]:
                    doit = False
            if doit:
                rc.ityp = _CLT_L
                rc.action = _CLA_RETRANSF
                return _CLR_UNLIKE_CONV
            rc.ityp = _CLT_L
            rc.action = _CLA_TRIVIAL
            st.status = _CLS_L_NEW_ITER
            return _CLR_CONTINUE
        elif st.obj != st.obj or st.obj < obj_lo:
            rc.ityp = _CLT_L
            rc.action = _CLA_RETRANSF
            return _CLR_UNBDBEL
        rc.ityp = _CLT_L
        rc.action = _CLA_TRIVIAL
        st.status = _CLS_L_NEW_ITER
        return _CLR_CONTINUE
    elif status == _CLS_L_NEW_ITER:
        st.ii += 1
        if st.ii >= itmax or (st.ii - st.lanczos_switch >= itmax_lanczos):
            rc.action = _CLA_RETRANSF
            rc.ityp = _CLT_L
            return _CLR_ITMAX
        rc.it = st.ii
        rc.ityp = _CLT_L
        rc.flt1 = 1.0 / st.gamma[st.ii - 1]
        rc.flt2 = 0.0
        rc.action = _CLA_UPDATE_DIR
        st.status = _CLS_L_UPDATE_P
        return _CLR_CONTINUE
    rc.action = _CLA_TRIVIAL
    return _CLR_CONTINUE

def _krylov_min(st: _TrlibState, rc: _Rc, init: int, radius: float,
                tol_rel_i: float, tol_rel_b: float, g_dot_g: float,
                v_dot_g: float, p_dot_Hp: float) -> int:
    itmax = st.itmax
    itmax_lanczos = 100
    zero = 2e-16
    obj_lo = -1e20
    ret = -1000
    rc.it = st.ii
    if init == _CLS_INIT or init == _CLS_HOTSTART:
        st.outerstatus = 0
    if st.outerstatus < 100 or st.outerstatus == 300:
        while True:
            ret = _krylov_internal(st, rc, init, radius, itmax, itmax_lanczos,
                                   tol_rel_i, tol_rel_b, zero, obj_lo,
                                   g_dot_g, v_dot_g, p_dot_Hp)
            if init > 0 or ret < 10 or rc.action != _CLA_TRIVIAL:
                break
    if ret >= 0 or ret == -1000:
        if st.outerstatus < 100 and ret < 10 and rc.action != _CLA_TRIVIAL:
            st.outerstatus = 100 + ret
            return 10
        if st.outerstatus >= 100 and st.outerstatus < 200:
            ret = st.outerstatus - 100
            st.outerstatus = 0
            rc.action = _CLA_TRIVIAL
        if ret < 10 and st.outerstatus < 100:
            lam = st.lam
            if lam > 1e-2 * max(1.0, st.raymax) and st.raymin < 0.0 and \
                    abs(st.raymin) < 1e-8 * st.raymax:
                st.outerstatus = 200 + ret
                rc.action = _CLA_OBJVAL
                return 10
        if st.outerstatus > 200 and st.outerstatus < 300:
            obj = st.obj
            realobj = g_dot_g
            if abs(obj - realobj) > max(1e-6, 1e-1 * abs(realobj)) or realobj > 0.0:
                ret = _krylov_internal(st, rc, _CLS_HOTSTART_P, radius, itmax,
                                       itmax_lanczos, tol_rel_i, tol_rel_b,
                                       zero, obj_lo, g_dot_g, v_dot_g, p_dot_Hp)
                st.outerstatus = 300
                return ret
            else:
                ret = st.outerstatus - 200
                st.outerstatus = 0
                return ret
        if st.outerstatus == 300 and ret < 10:
            st.outerstatus = 0
            return ret
    return ret

def _trlib_solve(m, trust_radius: float):
    # TRLIBQuadraticSubproblem.solve (_trlib.pyx)
    st = m.kstate
    rc = _Rc()
    init = st.init
    g_dot_g = 0.0
    v_dot_g = 0.0
    p_dot_Hp = 0.0
    jac = m.jac()
    while True:
        ret = _krylov_min(st, rc, init, trust_radius, m.tol_rel_i, m.tol_rel_b,
                          g_dot_g, v_dot_g, p_dot_Hp)
        init = 0
        action = rc.action
        it = rc.it
        if action == _CLA_INIT:
            st.s[:] = 0.0
            st.gm[:] = 0.0
            st.g[:] = jac
            st.v[:] = st.g
            g_dot_g = float(np.dot(st.g, st.g))
            v_dot_g = float(np.dot(st.v, st.g))
            st.p[:] = -st.v
            st.Hp[:] = m.hessp(st.p)
            p_dot_Hp = float(np.dot(st.p, st.Hp))
            st.Q[0, :] = st.v / math.sqrt(v_dot_g)
        if action == _CLA_RETRANSF:
            st.s[:] = np.dot(st.h[:it + 1], st.Q[:it + 1, :])
        if action == _CLA_UPDATE_STATIO:
            if rc.ityp == _CLT_CG:
                st.s += rc.flt1 * st.p
        if action == _CLA_UPDATE_GRAD:
            if rc.ityp == _CLT_CG:
                st.Q[it, :] = rc.flt2 * st.v
                st.gm[:] = st.g
                st.g += rc.flt1 * st.Hp
            if rc.ityp == _CLT_L:
                st.s[:] = st.Hp + rc.flt1 * st.g + rc.flt2 * st.gm
                st.gm[:] = rc.flt3 * st.g
                st.g[:] = st.s
            st.v[:] = st.g
            g_dot_g = float(np.dot(st.g, st.g))
            v_dot_g = float(np.dot(st.v, st.g))
        if action == _CLA_UPDATE_DIR:
            st.p[:] = rc.flt1 * st.v + rc.flt2 * st.p
            st.Hp[:] = m.hessp(st.p)
            p_dot_Hp = float(np.dot(st.p, st.Hp))
            if rc.ityp == _CLT_L:
                st.Q[it, :] = st.p
        if action == _CLA_OBJVAL:
            g_dot_g = .5 * float(np.dot(st.s, m.hessp(st.s)))
            g_dot_g += float(np.dot(st.s, jac))
        if ret < 10:
            break
        st.init = _CLS_HOTSTART
    return st.s.copy(), st.lam > 0.0
