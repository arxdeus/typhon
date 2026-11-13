# Ported from scikit-learn 1.9.1 (BSD-3-Clause)
# Port of sklearn/svm/src/liblinear (linear.cpp, tron.cpp; LIBLINEAR,
# BSD-3-Clause, Chih-Jen Lin et al.) as vendored and patched by sklearn
# (sorted labels, sample weights, max_iter, mt19937 bounded_rand_int).

# Dense rows: liblinear's feature_node lists skip zeros, and every update is
# a fused `acc += a*b` (clang -ffp-contract=on), so iterating over all
# columns with _fma gives identical results (fma(a, 0, c) == c).
# Solver types: 0 L2R_LR, 1 L2R_L2LOSS_SVC_DUAL, 2 L2R_L2LOSS_SVC,
# 3 L2R_L1LOSS_SVC_DUAL, 5 L1R_L2LOSS_SVC, 6 L1R_LR, 7 L2R_LR_DUAL,
# 11 L2R_L2LOSS_SVR, 12 L2R_L2LOSS_SVR_DUAL, 13 L2R_L1LOSS_SVR_DUAL.
# (4 = Crammer-Singer is not ported.)

import math
import numpy as np
from numpy.linalg.blas import fint, _suffix_blas
from ._newrand import _StdMT, _fma

_INF = float("inf")

@C
@_C.linkname("cblas_dnrm2" + _suffix_blas)
def _cblas_dnrm2(n: fint, x: cobj, incx: fint) -> float:
    pass

@C
@_C.linkname("cblas_ddot" + _suffix_blas)
def _cblas_ddot(n: fint, x: cobj, incx: fint, y: cobj, incy: fint) -> float:
    pass

@C
@_C.linkname("cblas_daxpy" + _suffix_blas)
def _cblas_daxpy(n: fint, a: float, x: cobj, incx: fint, y: cobj, incy: fint) -> None:
    pass

@C
@_C.linkname("cblas_dscal" + _suffix_blas)
def _cblas_dscal(n: fint, a: float, x: cobj, incx: fint) -> None:
    pass

def _nrm2(n: int, x: Ptr[float]) -> float:
    return _cblas_dnrm2(fint(n), x.as_byte(), fint(1))

def _dot(n: int, x: Ptr[float], y: Ptr[float]) -> float:
    return _cblas_ddot(fint(n), x.as_byte(), fint(1), y.as_byte(), fint(1))

def _axpy(n: int, a: float, x: Ptr[float], y: Ptr[float]):
    _cblas_daxpy(fint(n), a, x.as_byte(), fint(1), y.as_byte(), fint(1))

def _scal(n: int, a: float, x: Ptr[float]):
    _cblas_dscal(fint(n), a, x.as_byte(), fint(1))

class _Prob:
    """Dense problem: X row-major (l, n) incl. the bias column, y, W."""
    X: Ptr[float]
    l: int
    n: int
    y: List[float]
    W: List[float]

    def __init__(self, X: Ptr[float], l: int, n: int, y: List[float], W: List[float]):
        self.X = X
        self.l = l
        self.n = n
        self.y = y
        self.W = W

    def row(self, i: int) -> Ptr[float]:
        return self.X + i * self.n

    def xw(self, i: int, w: Ptr[float]) -> float:
        r = self.row(i)
        s = 0.0
        for k in range(self.n):
            s = _fma(w[k], r[k], s)
        return s

    def add_row(self, i: int, d: float, w: Ptr[float]):
        r = self.row(i)
        for k in range(self.n):
            w[k] = _fma(d, r[k], w[k])

# ---------------------------------------------------------------- TRON

class _L2RFun:
    """l2r_lr_fun (kind 0), l2r_l2_svc_fun (kind 1), l2r_l2_svr_fun (kind 2)."""
    prob: _Prob
    C: List[float]
    z: List[float]
    D: List[float]
    I: List[int]
    sizeI: int
    kind: int
    p: float

    def __init__(self, prob: _Prob, C: List[float], kind: int, p: float = 0.0):
        self.prob = prob
        self.C = C
        self.z = [0.0] * prob.l
        self.D = [0.0] * prob.l
        self.I = [0] * prob.l
        self.sizeI = 0
        self.kind = kind
        self.p = p

    def nvar(self) -> int:
        return self.prob.n

    def _Xv(self, v: Ptr[float]):
        for i in range(self.prob.l):
            self.z[i] = self.prob.xw(i, v)

    def fun(self, w: Ptr[float]) -> float:
        pr = self.prob
        self._Xv(w)
        f = 0.0
        for i in range(pr.n):
            f = _fma(w[i], w[i], f)
        f /= 2.0
        y = pr.y
        C = self.C
        if self.kind == 0:
            for i in range(pr.l):
                yz = y[i] * self.z[i]
                if yz >= 0:
                    f = _fma(C[i], math.log(1 + math.exp(-yz)), f)
                else:
                    f = _fma(C[i], -yz + math.log(1 + math.exp(yz)), f)
        elif self.kind == 1:
            for i in range(pr.l):
                self.z[i] = y[i] * self.z[i]
                d = 1 - self.z[i]
                if d > 0:
                    f = _fma(C[i] * d, d, f)
        else:
            p = self.p
            for i in range(pr.l):
                d = self.z[i] - y[i]
                if d < -p:
                    f = _fma(C[i] * (d + p), d + p, f)
                elif d > p:
                    f = _fma(C[i] * (d - p), d - p, f)
        return f

    def _subXTv(self, v: List[float], out: Ptr[float], allrows: bool):
        pr = self.prob
        for k in range(pr.n):
            out[k] = 0.0
        cnt = pr.l if allrows else self.sizeI
        for t in range(cnt):
            i = t if allrows else self.I[t]
            r = pr.row(i)
            vi = v[t]
            for k in range(pr.n):
                out[k] = _fma(vi, r[k], out[k])

    def grad(self, w: Ptr[float], g: Ptr[float]):
        pr = self.prob
        y = pr.y
        C = self.C
        n = pr.n
        if self.kind == 0:
            for i in range(pr.l):
                zi = 1 / (1 + math.exp(-y[i] * self.z[i]))
                self.D[i] = zi * (1 - zi)
                self.z[i] = C[i] * (zi - 1) * y[i]
            self._subXTv(self.z, g, True)
            for i in range(n):
                g[i] = w[i] + g[i]
            return
        self.sizeI = 0
        if self.kind == 1:
            for i in range(pr.l):
                if self.z[i] < 1:
                    self.z[self.sizeI] = C[i] * y[i] * (self.z[i] - 1)
                    self.I[self.sizeI] = i
                    self.sizeI += 1
        else:
            p = self.p
            for i in range(pr.l):
                d = self.z[i] - y[i]
                if d < -p:
                    self.z[self.sizeI] = C[i] * (d + p)
                    self.I[self.sizeI] = i
                    self.sizeI += 1
                elif d > p:
                    self.z[self.sizeI] = C[i] * (d - p)
                    self.I[self.sizeI] = i
                    self.sizeI += 1
        self._subXTv(self.z, g, False)
        for i in range(n):
            g[i] = _fma(2.0, g[i], w[i])

    def Hv(self, s: Ptr[float], Hs: Ptr[float]):
        pr = self.prob
        n = pr.n
        if self.kind == 0:
            wa = [0.0] * pr.l
            for i in range(pr.l):
                wa[i] = self.C[i] * self.D[i] * pr.xw(i, s)
            self._subXTv(wa, Hs, True)
            for i in range(n):
                Hs[i] = s[i] + Hs[i]
        else:
            wa = [0.0] * self.sizeI
            for t in range(self.sizeI):
                wa[t] = self.C[self.I[t]] * pr.xw(self.I[t], s)
            self._subXTv(wa, Hs, False)
            for i in range(n):
                Hs[i] = _fma(2.0, Hs[i], s[i])

def _trcg(f: _L2RFun, delta: float, g: Ptr[float], s: Ptr[float], r: Ptr[float]) -> int:
    n = f.nvar()
    d = Ptr[float](n)
    Hd = Ptr[float](n)
    for i in range(n):
        s[i] = 0.0
        r[i] = -g[i]
        d[i] = r[i]
    cgtol = 0.1 * _nrm2(n, g)
    cg_iter = 0
    rTr = _dot(n, r, r)
    while True:
        if _nrm2(n, r) <= cgtol:
            break
        cg_iter += 1
        f.Hv(d, Hd)
        alpha = rTr / _dot(n, d, Hd)
        _axpy(n, alpha, d, s)
        if _nrm2(n, s) > delta:
            alpha = -alpha
            _axpy(n, alpha, d, s)
            std = _dot(n, s, d)
            sts = _dot(n, s, s)
            dtd = _dot(n, d, d)
            dsq = delta * delta
            rad = math.sqrt(_fma(std, std, dtd * (dsq - sts)))
            if std >= 0:
                alpha = (dsq - sts) / (std + rad)
            else:
                alpha = (rad - std) / dtd
            _axpy(n, alpha, d, s)
            alpha = -alpha
            _axpy(n, alpha, Hd, r)
            break
        alpha = -alpha
        _axpy(n, alpha, Hd, r)
        rnew = _dot(n, r, r)
        beta = rnew / rTr
        _scal(n, beta, d)
        _axpy(n, 1.0, r, d)
        rTr = rnew
    return cg_iter

def _tron(f: _L2RFun, w: Ptr[float], eps: float, max_iter: int) -> int:
    eta0 = 1e-4; eta1 = 0.25; eta2 = 0.75
    sigma1 = 0.25; sigma2 = 0.5; sigma3 = 4.0
    n = f.nvar()
    s = Ptr[float](n)
    r = Ptr[float](n)
    w_new = Ptr[float](n)
    g = Ptr[float](n)
    for i in range(n):
        w[i] = 0.0
    fv = f.fun(w)
    f.grad(w, g)
    delta = _nrm2(n, g)
    gnorm1 = delta
    gnorm = gnorm1
    search = gnorm > eps * gnorm1
    it = 1
    while it <= max_iter and search:
        _trcg(f, delta, g, s, r)
        for i in range(n):
            w_new[i] = w[i]
        _axpy(n, 1.0, s, w_new)
        gs = _dot(n, g, s)
        prered = -0.5 * (gs - _dot(n, s, r))
        fnew = f.fun(w_new)
        actred = fv - fnew
        snorm = _nrm2(n, s)
        if it == 1:
            delta = min(delta, snorm)
        alpha = 0.0
        if fnew - fv - gs <= 0:
            alpha = sigma3
        else:
            alpha = max(sigma1, -0.5 * (gs / (fnew - fv - gs)))
        if actred < eta0 * prered:
            delta = min(max(alpha, sigma1) * snorm, sigma2 * delta)
        elif actred < eta1 * prered:
            delta = max(sigma1 * delta, min(alpha * snorm, sigma2 * delta))
        elif actred < eta2 * prered:
            delta = max(sigma1 * delta, min(alpha * snorm, sigma3 * delta))
        else:
            delta = max(delta, min(alpha * snorm, sigma3 * delta))
        if actred > eta0 * prered:
            it += 1
            for i in range(n):
                w[i] = w_new[i]
            fv = fnew
            f.grad(w, g)
            gnorm = _nrm2(n, g)
            if gnorm <= eps * gnorm1:
                break
        if fv < -1.0e+32:
            break
        if abs(actred) <= 0 and prered <= 0:
            break
        if abs(actred) <= 1.0e-12 * abs(fv) and abs(prered) <= 1.0e-12 * abs(fv):
            break
    return it - 1

# ---------------------------------------------------------- dual CD solvers

def _solve_l2r_l1l2_svc(prob: _Prob, w: Ptr[float], eps: float, Cp: float, Cn: float,
                        solver_type: int, max_iter: int, rng: _StdMT) -> int:
    l = prob.l
    n = prob.n
    it = 0
    QD = [0.0] * l
    index = list(range(l))
    alpha = [0.0] * l
    y = [0] * l
    active_size = l
    PGmax_old = _INF
    PGmin_old = -_INF
    diag = [0.0] * l
    ub = [0.0] * l
    for i in range(l):
        Ci = prob.W[i] * Cp if prob.y[i] > 0 else prob.W[i] * Cn
        if solver_type == 3:
            diag[i] = 0.0
            ub[i] = Ci
        else:
            diag[i] = 0.5 / Ci
            ub[i] = _INF
        y[i] = 1 if prob.y[i] > 0 else -1
    for k in range(n):
        w[k] = 0.0
    for i in range(l):
        q = diag[i]
        r = prob.row(i)
        for k in range(n):
            q = _fma(r[k], r[k], q)
        QD[i] = q
    while it < max_iter:
        PGmax_new = -_INF
        PGmin_new = _INF
        for i in range(active_size):
            j = i + rng.rint(active_size - i)
            index[i], index[j] = index[j], index[i]
        s = 0
        while s < active_size:
            i = index[s]
            yi = float(y[i])
            G = prob.xw(i, w)
            G = _fma(G, yi, -1.0)
            C = ub[i]
            G = _fma(alpha[i], diag[i], G)
            PG = 0.0
            if alpha[i] == 0:
                if G > PGmax_old:
                    active_size -= 1
                    index[s], index[active_size] = index[active_size], index[s]
                    continue
                elif G < 0:
                    PG = G
            elif alpha[i] == C:
                if G < PGmin_old:
                    active_size -= 1
                    index[s], index[active_size] = index[active_size], index[s]
                    continue
                elif G > 0:
                    PG = G
            else:
                PG = G
            PGmax_new = max(PGmax_new, PG)
            PGmin_new = min(PGmin_new, PG)
            if abs(PG) > 1.0e-12:
                a_old = alpha[i]
                alpha[i] = min(max(alpha[i] - G / QD[i], 0.0), C)
                d = (alpha[i] - a_old) * yi
                prob.add_row(i, d, w)
            s += 1
        it += 1
        if PGmax_new - PGmin_new <= eps:
            if active_size == l:
                break
            else:
                active_size = l
                PGmax_old = _INF
                PGmin_old = -_INF
                continue
        PGmax_old = PGmax_new
        PGmin_old = PGmin_new
        if PGmax_old <= 0:
            PGmax_old = _INF
        if PGmin_old >= 0:
            PGmin_old = -_INF
    return it

def _solve_l2r_l1l2_svr(prob: _Prob, w: Ptr[float], C0: float, p: float, eps: float,
                        solver_type: int, max_iter: int, rng: _StdMT) -> int:
    l = prob.l
    n = prob.n
    it = 0
    active_size = l
    index = list(range(l))
    Gmax_old = _INF
    Gnorm1_init = -1.0
    beta = [0.0] * l
    QD = [0.0] * l
    y = prob.y
    lam = [0.0] * l
    ub = [0.0] * l
    for i in range(l):
        Ci = prob.W[i] * C0
        if solver_type == 13:
            lam[i] = 0.0
            ub[i] = Ci
        else:
            lam[i] = 0.5 / Ci
            ub[i] = _INF
    for k in range(n):
        w[k] = 0.0
    for i in range(l):
        r = prob.row(i)
        q = 0.0
        for k in range(n):
            q = _fma(r[k], r[k], q)
        QD[i] = q
    while it < max_iter:
        Gmax_new = 0.0
        Gnorm1_new = 0.0
        for i in range(active_size):
            j = i + rng.rint(active_size - i)
            index[i], index[j] = index[j], index[i]
        s = 0
        while s < active_size:
            i = index[s]
            G = _fma(lam[i], beta[i], -y[i])
            H = QD[i] + lam[i]
            r = prob.row(i)
            for k in range(n):
                G = _fma(r[k], w[k], G)
            Gp = G + p
            Gn = G - p
            violation = 0.0
            if beta[i] == 0:
                if Gp < 0:
                    violation = -Gp
                elif Gn > 0:
                    violation = Gn
                elif Gp > Gmax_old and Gn < -Gmax_old:
                    active_size -= 1
                    index[s], index[active_size] = index[active_size], index[s]
                    continue
            elif beta[i] >= ub[i]:
                if Gp > 0:
                    violation = Gp
                elif Gp < -Gmax_old:
                    active_size -= 1
                    index[s], index[active_size] = index[active_size], index[s]
                    continue
            elif beta[i] <= -ub[i]:
                if Gn < 0:
                    violation = -Gn
                elif Gn > Gmax_old:
                    active_size -= 1
                    index[s], index[active_size] = index[active_size], index[s]
                    continue
            elif beta[i] > 0:
                violation = abs(Gp)
            else:
                violation = abs(Gn)
            Gmax_new = max(Gmax_new, violation)
            Gnorm1_new += violation
            d = 0.0
            if Gp < H * beta[i]:
                d = -Gp / H
            elif Gn > H * beta[i]:
                d = -Gn / H
            else:
                d = -beta[i]
            if abs(d) < 1.0e-12:
                s += 1
                continue
            b_old = beta[i]
            beta[i] = min(max(beta[i] + d, -ub[i]), ub[i])
            d = beta[i] - b_old
            if d != 0:
                prob.add_row(i, d, w)
            s += 1
        if it == 0:
            Gnorm1_init = Gnorm1_new
        it += 1
        if Gnorm1_new <= eps * Gnorm1_init:
            if active_size == l:
                break
            else:
                active_size = l
                Gmax_old = _INF
                continue
        Gmax_old = Gmax_new
    return it

def _solve_l2r_lr_dual(prob: _Prob, w: Ptr[float], eps: float, Cp: float, Cn: float,
                       max_iter: int, rng: _StdMT) -> int:
    l = prob.l
    n = prob.n
    it = 0
    xTx = [0.0] * l
    index = list(range(l))
    alpha = [0.0] * (2 * l)
    y = [0] * l
    max_inner_iter = 100
    innereps = 1e-2
    innereps_min = min(1e-8, eps)
    ub = [0.0] * l
    for i in range(l):
        if prob.y[i] > 0:
            ub[i] = prob.W[i] * Cp
            y[i] = 1
        else:
            ub[i] = prob.W[i] * Cn
            y[i] = -1
    for i in range(l):
        alpha[2 * i] = min(0.001 * ub[i], 1e-8)
        alpha[2 * i + 1] = ub[i] - alpha[2 * i]
    for k in range(n):
        w[k] = 0.0
    for i in range(l):
        r = prob.row(i)
        q = 0.0
        coef = float(y[i]) * alpha[2 * i]
        for k in range(n):
            if r[k] != 0.0:
                q = _fma(r[k], r[k], q)
                w[k] = _fma(coef, r[k], w[k])
        xTx[i] = q
    while it < max_iter:
        for i in range(l):
            j = i + rng.rint(l - i)
            index[i], index[j] = index[j], index[i]
        newton_iter = 0
        Gmax = 0.0
        for s in range(l):
            i = index[s]
            yi = float(y[i])
            C = ub[i]
            ywTx = prob.xw(i, w)
            ywTx *= yi
            a = xTx[i]
            b = ywTx
            ind1 = 2 * i; ind2 = 2 * i + 1; sign = 1.0
            if _fma(0.5 * a, alpha[ind2] - alpha[ind1], b) < 0:
                ind1 = 2 * i + 1
                ind2 = 2 * i
                sign = -1.0
            alpha_old = alpha[ind1]
            z = alpha_old
            if C - z < 0.5 * C:
                z = 0.1 * z
            gp = _fma(a, z - alpha_old, sign * b) + math.log(z / (C - z))
            Gmax = max(Gmax, abs(gp))
            eta = 0.1
            inner = 0
            while inner <= max_inner_iter:
                if abs(gp) < innereps:
                    break
                gpp = a + C / (C - z) / z
                tmpz = z - gp / gpp
                if tmpz <= 0:
                    z *= eta
                else:
                    z = tmpz
                gp = _fma(a, z - alpha_old, sign * b) + math.log(z / (C - z))
                newton_iter += 1
                inner += 1
            if inner > 0:
                alpha[ind1] = z
                alpha[ind2] = C - z
                prob.add_row(i, sign * (z - alpha_old) * yi, w)
        it += 1
        if Gmax < eps:
            break
        if newton_iter <= l // 10:
            innereps = max(innereps_min, 0.1 * innereps)
    return it

# ------------------------------------------------- L1-regularized (columns)

class _Cols:
    """transpose(): per feature j the (row, value) pairs with value != 0."""
    idx: List[List[int]]
    val: List[List[float]]

    def __init__(self, prob: _Prob):
        self.idx = [List[int]() for _ in range(prob.n)]
        self.val = [List[float]() for _ in range(prob.n)]
        for i in range(prob.l):
            r = prob.row(i)
            for j in range(prob.n):
                if r[j] != 0.0:
                    self.idx[j].append(i)
                    self.val[j].append(r[j])

def _solve_l1r_l2_svc(prob: _Prob, w: Ptr[float], eps: float, Cp: float, Cn: float,
                      max_iter: int, rng: _StdMT) -> int:
    l = prob.l
    w_size = prob.n
    it = 0
    active_size = w_size
    max_num_linesearch = 20
    sigma = 0.01
    Gmax_old = _INF
    Gnorm1_init = -1.0
    index = list(range(w_size))
    y = [0] * l
    b = [1.0] * l
    xj_sq = [0.0] * w_size
    C = [0.0] * l
    cols = _Cols(prob)
    for j in range(w_size):
        w[j] = 0.0
    for j in range(l):
        if prob.y[j] > 0:
            y[j] = 1
            C[j] = prob.W[j] * Cp
        else:
            y[j] = -1
            C[j] = prob.W[j] * Cn
    for j in range(w_size):
        ci = cols.idx[j]
        cv = cols.val[j]
        for t in range(len(ci)):
            ind = ci[t]
            cv[t] *= float(y[ind])
            v = cv[t]
            b[ind] = _fma(-w[j], v, b[ind])
            xj_sq[j] = _fma(C[ind] * v, v, xj_sq[j])
    while it < max_iter:
        Gmax_new = 0.0
        Gnorm1_new = 0.0
        for j in range(active_size):
            i = j + rng.rint(active_size - j)
            index[i], index[j] = index[j], index[i]
        s = 0
        while s < active_size:
            j = index[s]
            ci = cols.idx[j]
            cv = cols.val[j]
            nn = len(ci)
            G_loss = 0.0
            H = 0.0
            for t in range(nn):
                ind = ci[t]
                if b[ind] > 0:
                    v = cv[t]
                    tmp = C[ind] * v
                    G_loss = _fma(-tmp, b[ind], G_loss)
                    H = _fma(tmp, v, H)
            G_loss *= 2
            G = G_loss
            H *= 2
            H = max(H, 1e-12)
            Gp = G + 1
            Gn = G - 1
            violation = 0.0
            if w[j] == 0:
                if Gp < 0:
                    violation = -Gp
                elif Gn > 0:
                    violation = Gn
                elif Gp > Gmax_old / l and Gn < -Gmax_old / l:
                    active_size -= 1
                    index[s], index[active_size] = index[active_size], index[s]
                    continue
            elif w[j] > 0:
                violation = abs(Gp)
            else:
                violation = abs(Gn)
            Gmax_new = max(Gmax_new, violation)
            Gnorm1_new += violation
            d = 0.0
            if Gp < H * w[j]:
                d = -Gp / H
            elif Gn > H * w[j]:
                d = -Gn / H
            else:
                d = -w[j]
            if abs(d) < 1.0e-12:
                s += 1
                continue
            delta = _fma(G, d, abs(w[j] + d) - abs(w[j]))
            d_old = 0.0
            loss_old = 0.0
            loss_new = 0.0
            nls = 0
            while nls < max_num_linesearch:
                d_diff = d_old - d
                cond = _fma(-sigma, delta, abs(w[j] + d) - abs(w[j]))
                appxcond = _fma(xj_sq[j] * d, d, _fma(G_loss, d, cond))
                if appxcond <= 0:
                    for t in range(nn):
                        b[ci[t]] = _fma(d_diff, cv[t], b[ci[t]])
                    break
                if nls == 0:
                    loss_old = 0.0
                    loss_new = 0.0
                    for t in range(nn):
                        ind = ci[t]
                        if b[ind] > 0:
                            loss_old = _fma(C[ind] * b[ind], b[ind], loss_old)
                        b_new = _fma(d_diff, cv[t], b[ind])
                        b[ind] = b_new
                        if b_new > 0:
                            loss_new = _fma(C[ind] * b_new, b_new, loss_new)
                else:
                    loss_new = 0.0
                    for t in range(nn):
                        ind = ci[t]
                        b_new = _fma(d_diff, cv[t], b[ind])
                        b[ind] = b_new
                        if b_new > 0:
                            loss_new = _fma(C[ind] * b_new, b_new, loss_new)
                cond = cond + loss_new - loss_old
                if cond <= 0:
                    break
                else:
                    d_old = d
                    d *= 0.5
                    delta *= 0.5
                nls += 1
            w[j] += d
            if nls >= max_num_linesearch:
                for i in range(l):
                    b[i] = 1.0
                for i in range(w_size):
                    if w[i] == 0:
                        continue
                    for t in range(len(cols.idx[i])):
                        b[cols.idx[i][t]] = _fma(-w[i], cols.val[i][t], b[cols.idx[i][t]])
            s += 1
        if it == 0:
            Gnorm1_init = Gnorm1_new
        it += 1
        if Gnorm1_new <= eps * Gnorm1_init:
            if active_size == w_size:
                break
            else:
                active_size = w_size
                Gmax_old = _INF
                continue
        Gmax_old = Gmax_new
    return it

def _solve_l1r_lr(prob: _Prob, w: Ptr[float], eps: float, Cp: float, Cn: float,
                  max_newton_iter: int, rng: _StdMT) -> int:
    l = prob.l
    w_size = prob.n
    newton_iter = 0
    it = 0
    max_iter = 1000
    max_num_linesearch = 20
    QP_no_change = 0
    nu = 1e-12
    inner_eps = 1.0
    sigma = 0.01
    Gnorm1_init = -1.0
    Gmax_old = _INF
    index = list(range(w_size))
    y = [0] * l
    Hdiag = [0.0] * w_size
    Grad = [0.0] * w_size
    wpd = [0.0] * w_size
    xjneg_sum = [0.0] * w_size
    xTd = [0.0] * l
    exp_wTx = [0.0] * l
    exp_wTx_new = [0.0] * l
    tau = [0.0] * l
    D = [0.0] * l
    C = [0.0] * l
    cols = _Cols(prob)
    for j in range(w_size):
        w[j] = 0.0
    for j in range(l):
        if prob.y[j] > 0:
            y[j] = 1
            C[j] = prob.W[j] * Cp
        else:
            y[j] = -1
            C[j] = prob.W[j] * Cn
    w_norm = 0.0
    for j in range(w_size):
        w_norm += abs(w[j])
        wpd[j] = w[j]
        ci = cols.idx[j]; cv = cols.val[j]
        for t in range(len(ci)):
            ind = ci[t]
            exp_wTx[ind] = _fma(w[j], cv[t], exp_wTx[ind])
            if y[ind] == -1:
                xjneg_sum[j] = _fma(C[ind], cv[t], xjneg_sum[j])
    for j in range(l):
        exp_wTx[j] = math.exp(exp_wTx[j])
        tt = 1 / (1 + exp_wTx[j])
        tau[j] = C[j] * tt
        D[j] = C[j] * exp_wTx[j] * tt * tt
    while newton_iter < max_newton_iter:
        Gmax_new = 0.0
        Gnorm1_new = 0.0
        active_size = w_size
        s = 0
        while s < active_size:
            j = index[s]
            Hdiag[j] = nu
            Grad[j] = 0.0
            tmp = 0.0
            ci = cols.idx[j]; cv = cols.val[j]
            for t in range(len(ci)):
                ind = ci[t]
                Hdiag[j] = _fma(cv[t] * cv[t], D[ind], Hdiag[j])
                tmp = _fma(cv[t], tau[ind], tmp)
            Grad[j] = -tmp + xjneg_sum[j]
            Gp = Grad[j] + 1
            Gn = Grad[j] - 1
            violation = 0.0
            if w[j] == 0:
                if Gp < 0:
                    violation = -Gp
                elif Gn > 0:
                    violation = Gn
                elif Gp > Gmax_old / l and Gn < -Gmax_old / l:
                    active_size -= 1
                    index[s], index[active_size] = index[active_size], index[s]
                    continue
            elif w[j] > 0:
                violation = abs(Gp)
            else:
                violation = abs(Gn)
            Gmax_new = max(Gmax_new, violation)
            Gnorm1_new += violation
            s += 1
        if newton_iter == 0:
            Gnorm1_init = Gnorm1_new
        if Gnorm1_new <= eps * Gnorm1_init or QP_no_change >= 10:
            break
        QP_no_change += 1
        it = 0
        QP_Gmax_old = _INF
        QP_active_size = active_size
        for i in range(l):
            xTd[i] = 0.0
        while it < max_iter:
            QP_Gmax_new = 0.0
            QP_Gnorm1_new = 0.0
            for j in range(QP_active_size):
                i = j + rng.rint(QP_active_size - j)
                index[i], index[j] = index[j], index[i]
            s = 0
            while s < QP_active_size:
                j = index[s]
                H = Hdiag[j]
                ci = cols.idx[j]; cv = cols.val[j]
                G = _fma(wpd[j] - w[j], nu, Grad[j])
                for t in range(len(ci)):
                    ind = ci[t]
                    G = _fma(cv[t] * D[ind], xTd[ind], G)
                Gp = G + 1
                Gn = G - 1
                violation = 0.0
                if wpd[j] == 0:
                    if Gp < 0:
                        violation = -Gp
                    elif Gn > 0:
                        violation = Gn
                    elif Gp > QP_Gmax_old / l and Gn < -QP_Gmax_old / l:
                        QP_active_size -= 1
                        index[s], index[QP_active_size] = index[QP_active_size], index[s]
                        continue
                elif wpd[j] > 0:
                    violation = abs(Gp)
                else:
                    violation = abs(Gn)
                z = 0.0
                if Gp < H * wpd[j]:
                    z = -Gp / H
                elif Gn > H * wpd[j]:
                    z = -Gn / H
                else:
                    z = -wpd[j]
                if abs(z) < 1.0e-12:
                    s += 1
                    continue
                z = min(max(z, -10.0), 10.0)
                QP_no_change = 0
                QP_Gmax_new = max(QP_Gmax_new, violation)
                QP_Gnorm1_new += violation
                wpd[j] += z
                for t in range(len(ci)):
                    xTd[ci[t]] = _fma(cv[t], z, xTd[ci[t]])
                s += 1
            it += 1
            if QP_Gnorm1_new <= inner_eps * Gnorm1_init:
                if QP_active_size == active_size:
                    break
                else:
                    QP_active_size = active_size
                    QP_Gmax_old = _INF
                    continue
            QP_Gmax_old = QP_Gmax_new
        delta = 0.0
        w_norm_new = 0.0
        for j in range(w_size):
            delta = _fma(Grad[j], wpd[j] - w[j], delta)
            if wpd[j] != 0:
                w_norm_new += abs(wpd[j])
        delta += (w_norm_new - w_norm)
        negsum_xTd = 0.0
        for i in range(l):
            if y[i] == -1:
                negsum_xTd = _fma(C[i], xTd[i], negsum_xTd)
        nls = 0
        while nls < max_num_linesearch:
            cond = _fma(-sigma, delta, w_norm_new - w_norm + negsum_xTd)
            for i in range(l):
                e = math.exp(xTd[i])
                exp_wTx_new[i] = exp_wTx[i] * e
                cond = _fma(C[i], math.log((1 + exp_wTx_new[i]) / (e + exp_wTx_new[i])), cond)
            if cond <= 0:
                w_norm = w_norm_new
                for j in range(w_size):
                    w[j] = wpd[j]
                for i in range(l):
                    exp_wTx[i] = exp_wTx_new[i]
                    tt = 1 / (1 + exp_wTx[i])
                    tau[i] = C[i] * tt
                    D[i] = C[i] * exp_wTx[i] * tt * tt
                break
            else:
                w_norm_new = 0.0
                for j in range(w_size):
                    wpd[j] = (w[j] + wpd[j]) * 0.5
                    if wpd[j] != 0:
                        w_norm_new += abs(wpd[j])
                delta *= 0.5
                negsum_xTd *= 0.5
                for i in range(l):
                    xTd[i] *= 0.5
            nls += 1
        if nls >= max_num_linesearch:
            for i in range(l):
                exp_wTx[i] = 0.0
            for i in range(w_size):
                if w[i] == 0:
                    continue
                for t in range(len(cols.idx[i])):
                    ind = cols.idx[i][t]
                    exp_wTx[ind] = _fma(w[i], cols.val[i][t], exp_wTx[ind])
            for i in range(l):
                exp_wTx[i] = math.exp(exp_wTx[i])
        if it == 1:
            inner_eps *= 0.25
        newton_iter += 1
        Gmax_old = Gmax_new
    return newton_iter

class _McsvmCS:
    """liblinear Solver_MCSVM_CS (Crammer-Singer multi-class SVM)."""
    prob: _Prob
    nr_class: int
    C: List[float]
    eps: float
    max_iter: int

    def __init__(self, prob: _Prob, nr_class: int, weighted_C: List[float], eps: float, max_iter: int):
        self.prob = prob
        self.nr_class = nr_class
        self.eps = eps
        self.max_iter = max_iter
        self.C = [prob.W[i] * weighted_C[int(prob.y[i])] for i in range(prob.l)]

    def sub_problem(self, B: List[float], A_i: float, yi: int, C_yi: float, active_i: int,
                    alpha_new: List[float]):
        D = [B[k] for k in range(active_i)]
        if yi < active_i:
            D[yi] += A_i * C_yi
        D.sort(reverse=True)
        beta = D[0] - A_i * C_yi
        r = 1
        while r < active_i and beta < r * D[r]:
            beta += D[r]
            r += 1
        beta /= r
        for r in range(active_i):
            if r == yi:
                alpha_new[r] = min(C_yi, (beta - B[r]) / A_i)
            else:
                alpha_new[r] = min(0.0, (beta - B[r]) / A_i)

    def shrunk(self, G: List[float], i: int, m: int, yi: int, alpha_i: float, minG: float) -> bool:
        bound = 0.0
        if m == yi:
            bound = self.C[i]
        return alpha_i == bound and G[m] < minG

    def solve(self, w: Ptr[float], rng: _StdMT) -> int:
        prob = self.prob
        l = prob.l
        nc = self.nr_class
        n = prob.n
        eps = self.eps
        B = [0.0] * nc
        G = [0.0] * nc
        alpha = [0.0] * (l * nc)
        alpha_new = [0.0] * nc
        index = list(range(l))
        QD = [0.0] * l
        d_ind = [0] * nc
        d_val = [0.0] * nc
        alpha_index = [0] * (nc * l)
        y_index = [0] * l
        active_size = l
        active_size_i = [nc] * l
        eps_shrink = max(10.0 * eps, 1.0)
        start_from_all = True
        for k in range(n * nc):
            w[k] = 0.0
        for i in range(l):
            for m in range(nc):
                alpha_index[i * nc + m] = m
            r = prob.row(i)
            q = 0.0
            for k in range(n):
                q += r[k] * r[k]
            QD[i] = q
            y_index[i] = int(prob.y[i])
        it = 0
        while it < self.max_iter:
            stopping = -_INF
            for i in range(active_size):
                j = i + rng.rint(active_size - i)
                index[i], index[j] = index[j], index[i]
            s = 0
            while s < active_size:
                i = index[s]
                Ai = QD[i]
                ab = i * nc
                if Ai > 0:
                    asz = active_size_i[i]
                    for m in range(asz):
                        G[m] = 1.0
                    if y_index[i] < asz:
                        G[y_index[i]] = 0.0
                    xr = prob.row(i)
                    for f in range(n):
                        v = xr[f]
                        for m in range(asz):
                            G[m] += w[f * nc + alpha_index[ab + m]] * v
                    minG = _INF
                    maxG = -_INF
                    for m in range(asz):
                        if alpha[ab + alpha_index[ab + m]] < 0 and G[m] < minG:
                            minG = G[m]
                        if G[m] > maxG:
                            maxG = G[m]
                    if y_index[i] < asz:
                        if alpha[ab + int(prob.y[i])] < self.C[i] and G[y_index[i]] < minG:
                            minG = G[y_index[i]]
                    m = 0
                    while m < active_size_i[i]:
                        if self.shrunk(G, i, m, y_index[i], alpha[ab + alpha_index[ab + m]], minG):
                            active_size_i[i] -= 1
                            while active_size_i[i] > m:
                                a = active_size_i[i]
                                if not self.shrunk(G, i, a, y_index[i], alpha[ab + alpha_index[ab + a]], minG):
                                    alpha_index[ab + m], alpha_index[ab + a] = alpha_index[ab + a], alpha_index[ab + m]
                                    G[m], G[a] = G[a], G[m]
                                    if y_index[i] == a:
                                        y_index[i] = m
                                    elif y_index[i] == m:
                                        y_index[i] = a
                                    break
                                active_size_i[i] -= 1
                        m += 1
                    if active_size_i[i] <= 1:
                        active_size -= 1
                        index[s], index[active_size] = index[active_size], index[s]
                        continue
                    if maxG - minG <= 1e-12:
                        s += 1
                        continue
                    else:
                        stopping = max(maxG - minG, stopping)
                    asz = active_size_i[i]
                    for m in range(asz):
                        B[m] = G[m] - Ai * alpha[ab + alpha_index[ab + m]]
                    self.sub_problem(B, Ai, y_index[i], self.C[i], asz, alpha_new)
                    nz = 0
                    for m in range(asz):
                        d = alpha_new[m] - alpha[ab + alpha_index[ab + m]]
                        alpha[ab + alpha_index[ab + m]] = alpha_new[m]
                        if abs(d) >= 1e-12:
                            d_ind[nz] = alpha_index[ab + m]
                            d_val[nz] = d
                            nz += 1
                    for f in range(n):
                        v = xr[f]
                        for m in range(nz):
                            w[f * nc + d_ind[m]] += d_val[m] * v
                s += 1
            it += 1
            if stopping < eps_shrink:
                if stopping < eps and start_from_all:
                    break
                else:
                    active_size = l
                    for i in range(l):
                        active_size_i[i] = nc
                    eps_shrink = max(eps_shrink / 2, eps)
                    start_from_all = True
            else:
                start_from_all = False
        return it

# --------------------------------------------------------------- driver

def _train_one(prob: _Prob, solver_type: int, w: Ptr[float], eps: float, Cp: float, Cn: float,
               C0: float, p: float, max_iter: int, rng: _StdMT) -> int:
    pos = 0
    for i in range(prob.l):
        if prob.y[i] > 0:
            pos += 1
    neg = prob.l - pos
    primal_tol = eps * float(max(min(pos, neg), 1)) / float(prob.l)
    if solver_type == 0 or solver_type == 2:
        C = [prob.W[i] * Cp if prob.y[i] > 0 else prob.W[i] * Cn for i in range(prob.l)]
        f = _L2RFun(prob, C, 0 if solver_type == 0 else 1)
        return _tron(f, w, primal_tol, max_iter)
    elif solver_type == 1 or solver_type == 3:
        return _solve_l2r_l1l2_svc(prob, w, eps, Cp, Cn, solver_type, max_iter, rng)
    elif solver_type == 5:
        return _solve_l1r_l2_svc(prob, w, primal_tol, Cp, Cn, max_iter, rng)
    elif solver_type == 6:
        return _solve_l1r_lr(prob, w, primal_tol, Cp, Cn, max_iter, rng)
    elif solver_type == 7:
        return _solve_l2r_lr_dual(prob, w, eps, Cp, Cn, max_iter, rng)
    elif solver_type == 11:
        C = [prob.W[i] * C0 for i in range(prob.l)]
        f = _L2RFun(prob, C, 2, p)
        return _tron(f, w, eps, max_iter)
    elif solver_type == 12 or solver_type == 13:
        return _solve_l2r_l1l2_svr(prob, w, C0, p, eps, solver_type, max_iter, rng)
    else:
        raise ValueError(f"liblinear solver_type {solver_type} is not supported")

def train_wrap(X, y, solver_type: int, eps: float, bias: float, C: float, class_weight,
               max_iter: int, seed: int, epsilon: float, sample_weight):
    """liblinear.train_wrap (dense X): returns (raw_coef (nr, nf [+1]), n_iter).
    y holds class indices 0..k-1 as floats (or targets for SVR);
    class_weight is per class index (empty array for regression)."""
    Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
    ya = np.asarray(y, dtype=float)
    swa = np.asarray(sample_weight, dtype=float)
    cw = np.asarray(class_weight, dtype=float)
    l0, nf = Xa.shape
    nb = nf + 1 if bias > 0 else nf
    # remove_zero_weight + bias column
    keep = List[int]()
    for i in range(l0):
        if swa[i] > 0:
            keep.append(i)
    l = len(keep)
    Xb = np.empty((l, nb))
    yv = [0.0] * l
    Wv = [0.0] * l
    for t in range(l):
        i = keep[t]
        for k in range(nf):
            Xb[t, k] = Xa[i, k]
        if bias > 0:
            Xb[t, nf] = bias
        yv[t] = ya[i]
        Wv[t] = swa[i]
    rng = _StdMT(u32(seed))
    regression = solver_type == 11 or solver_type == 12 or solver_type == 13
    if regression:
        prob = _Prob(Xb.data, l, nb, yv, Wv)
        w = np.zeros((1, nb))
        it = _train_one(prob, solver_type, w.data, eps, 0.0, 0.0, C, epsilon, max_iter, rng)
        return w, np.array([it])
    # group_classes (labels sorted)
    labels = List[int]()
    for t in range(l):
        lab = int(yv[t])
        if lab not in labels:
            labels.append(lab)
    labels.sort()
    nr_class = len(labels)
    count = [0] * nr_class
    for t in range(l):
        count[labels.index(int(yv[t]))] += 1
    start = [0] * nr_class
    for c in range(1, nr_class):
        start[c] = start[c - 1] + count[c - 1]
    perm = [0] * l
    pos = start.copy()
    for t in range(l):
        c = labels.index(int(yv[t]))
        perm[pos[c]] = t
        pos[c] += 1
    wC = [C] * nr_class
    for i in range(cw.shape[0]):
        for j in range(nr_class):
            if labels[j] == i:
                wC[j] *= cw[i]
    Xs = np.empty((l, nb))
    Ws = [0.0] * l
    for k in range(l):
        for c in range(nb):
            Xs[k, c] = Xb[perm[k], c]
        Ws[k] = Wv[perm[k]]
    sy = [0.0] * l
    sub = _Prob(Xs.data, l, nb, sy, Ws)
    if solver_type == 4:
        for c in range(nr_class):
            for k in range(start[c], start[c] + count[c]):
                sy[k] = float(c)
        W4 = np.zeros((nr_class, nb))
        flat = Ptr[float](nr_class * nb)
        solver = _McsvmCS(sub, nr_class, wC, eps, 100000)
        it4 = solver.solve(flat, rng)
        for f in range(nb):
            for m in range(nr_class):
                W4[m, f] = flat[f * nr_class + m]
        return W4, np.array([it4])
    if nr_class == 2:
        e0 = start[0] + count[0]
        for k in range(l):
            sy[k] = -1.0 if k < e0 else 1.0
        w = np.zeros((1, nb))
        it = _train_one(sub, solver_type, w.data, eps, wC[1], wC[0], C, epsilon, max_iter, rng)
        return w, np.array([it])
    W = np.zeros((nr_class, nb))
    iters = np.zeros(nr_class, dtype=int)
    wtmp = Ptr[float](nb)
    for c in range(nr_class):
        si = start[c]
        ei = si + count[c]
        for k in range(l):
            sy[k] = 1.0 if (k >= si and k < ei) else -1.0
        iters[c] = _train_one(sub, solver_type, wtmp, eps, wC[c], C, C, epsilon, max_iter, rng)
        for k in range(nb):
            W[c, k] = wtmp[k]
    return W, iters

def solver_type(multi_class: str, penalty: str, loss: str, dual: bool) -> int:
    """_get_liblinear_solver_type."""
    if multi_class == "crammer_singer":
        return 4
    if multi_class != "ovr":
        raise ValueError(f"`multi_class` must be one of `ovr`, `crammer_singer`, got {multi_class!r}")
    r = -1
    if loss == "logistic_regression":
        if penalty == "l1" and not dual:
            r = 6
        elif penalty == "l2":
            r = 7 if dual else 0
    elif loss == "hinge":
        if penalty == "l2" and dual:
            r = 3
    elif loss == "squared_hinge":
        if penalty == "l1" and not dual:
            r = 5
        elif penalty == "l2":
            r = 1 if dual else 2
    elif loss == "epsilon_insensitive":
        if penalty == "l2" and dual:
            r = 13
    elif loss == "squared_epsilon_insensitive":
        if penalty == "l2":
            r = 12 if dual else 11
    else:
        raise ValueError(f"Unsupported set of arguments: loss='{loss}' is not supported, Parameters: penalty={penalty!r}, loss={loss!r}, dual={dual!r}")
    if r < 0:
        raise ValueError(f"Unsupported set of arguments: The combination of penalty='{penalty}' and loss='{loss}' are not supported when dual={dual}, Parameters: penalty={penalty!r}, loss={loss!r}, dual={dual!r}")
    return r
