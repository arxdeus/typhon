# Ported from scikit-learn 1.9.1 (BSD-3-Clause)
#
# sklearn.utils.optimize._newton_cg (+ _cg, _line_search_wolfe12) and
# sklearn.linear_model._glm._newton_solver.NewtonCholeskySolver on top of
# LinearModelLoss (_linear_loss.py).

import math
import numpy as np
from ._linear_loss import LinearModelLoss, _LossFG, lbfgs_fit, MULTINOMIAL

_EPS = 2.220446049250313e-16

def _cg(lm: LinearModelLoss, l2: float, fgrad: np.ndarray[float, 1], maxiter: int,
        tol: float) -> np.ndarray[float, 1]:
    eps = 16.0 * _EPS
    xsupi = np.zeros(fgrad.shape[0])
    ri = fgrad.copy()
    psupi = -ri
    i = 0
    dri0 = float(ri @ ri)
    psupi_norm2 = dri0
    while i <= maxiter:
        if float(np.sum(np.abs(ri))) <= tol:
            break
        Ap = lm.hessp(psupi, l2)
        curv = float(psupi @ Ap)
        if 0.0 <= curv and curv <= eps * psupi_norm2:
            break
        elif curv < 0.0:
            if i > 0:
                break
            else:
                xsupi += dri0 / curv * psupi
                break
        alphai = dri0 / curv
        xsupi += alphai * psupi
        ri += alphai * Ap
        dri1 = float(ri @ ri)
        betai = dri1 / dri0
        psupi = -ri + betai * psupi
        psupi_norm2 = dri1 + betai * betai * psupi_norm2
        i = i + 1
        dri0 = dri1
    return xsupi

class _NoExtra:
    def __call__(self, a: float, x, f: float, g) -> bool:
        return True

class _Wolfe2Extra:
    def __call__(self, a: float, phi: float) -> bool:
        return True

def _ls_wolfe12(fg: _LossFG, xk: np.ndarray[float, 1], pk: np.ndarray[float, 1],
                gfk: np.ndarray[float, 1], old_fval: float, old_old_fval: float):
    """sklearn's _line_search_wolfe12. Returns (ok, alpha, new_fval, old_fval)."""
    from scipy.optimize._linesearch import _LineFun, _wolfe1, _wolfe2
    eps = 16.0 * _EPS
    derphi0 = float(gfk @ pk)
    lf = _LineFun(fg, xk, pk, gfk)
    ok, stp, fval, oldf = _wolfe1(lf, old_fval, old_old_fval, derphi0, 1e-4, 0.9, 50.0, 1e-8)
    if ok:
        return True, stp, fval, old_fval
    fv = fg.fun(xk + pk)
    tiny_loss = abs(old_fval * eps)
    if abs(fv - old_fval) <= tiny_loss:
        s_old = float(np.sum(np.abs(gfk)))
        gr = fg.grad(xk + pk)
        if float(np.sum(np.abs(gr))) < s_old:
            return True, 1.0, fv, old_fval
    lf2 = _LineFun(fg, xk, pk, gfk)
    ok2, a2, f2, hd = _wolfe2(lf2, old_fval, old_old_fval, derphi0, 1e-4, 0.9, math.inf,
                             _Wolfe2Extra())
    if not ok2:
        return False, 0.0, 0.0, old_fval
    return True, a2, f2, old_fval

def newton_cg(lm: LinearModelLoss, x0: np.ndarray[float, 1], l2: float, tol: float,
              maxiter: int, maxinner: int = 200):
    fg = _LossFG(lm, l2, 1)
    xk = x0.copy()
    k = 0
    old_fval = fg.fun(x0)
    old_old_fval = math.nan
    while k < maxiter:
        fgrad = lm.grad_hessp_setup(xk, l2)
        absgrad = np.abs(fgrad)
        if float(np.max(absgrad)) <= tol:
            break
        maggrad = float(np.sum(absgrad))
        eta = min(0.5, math.sqrt(maggrad))
        termcond = eta * maggrad
        xsupi = _cg(lm, l2, fgrad, maxinner, termcond)
        ok, alphak, nf, of = _ls_wolfe12(fg, xk, xsupi, fgrad, old_fval, old_old_fval)
        if not ok:
            break
        old_old_fval = of
        old_fval = nf
        xk += alphak * xsupi
        k += 1
    return xk, k

class NewtonCholeskySolver:
    lm: LinearModelLoss
    coef: np.ndarray[float, 1]
    l2: float
    tol: float
    max_iter: int
    iteration: int
    converged: bool
    use_fallback: bool
    loss_value: float
    gradient: np.ndarray[float, 1]
    hessian: np.ndarray[float, 2]
    hessian_warning: bool
    coef_newton: np.ndarray[float, 1]
    gradient_times_newton: float
    raw1: np.ndarray[float, 1]
    rawK: np.ndarray[float, 2]
    multi_int: bool
    multi_nopen: bool

    def __init__(self, lm: LinearModelLoss, coef: np.ndarray[float, 1], l2: float, tol: float,
                 max_iter: int):
        self.lm = lm
        self.coef = coef.copy()
        self.l2 = l2
        self.tol = tol
        self.max_iter = max_iter
        self.iteration = 0
        self.converged = False
        self.use_fallback = False
        self.loss_value = 0.0
        self.gradient = np.zeros(0)
        self.hessian = np.zeros((0, 0))
        self.hessian_warning = False
        self.coef_newton = np.zeros(0)
        self.gradient_times_newton = 0.0
        self.raw1 = np.zeros(0)
        self.rawK = np.zeros((0, 0))
        self.multi_int = False
        self.multi_nopen = False

    def _is_multi(self) -> bool:
        return self.lm.kind == MULTINOMIAL

    def setup(self):
        if self._is_multi():
            self.rawK = self.lm.rawK(self.coef)
            lo, G, P = self.lm._multi_lg(self.rawK, True)
            f = float(np.sum(lo) / self.lm.sw_sum)
            if self.l2 > 0.0:
                f += self.lm.l2_penalty(self.coef, self.l2)
            self.loss_value = f
        else:
            self.raw1 = self.lm.raw1(self.coef)
            self.loss_value = self.lm.loss_raw1(self.coef, self.raw1, self.l2)
        self.multi_int = self._is_multi() and self.lm.fit_intercept
        self.multi_nopen = self._is_multi() and self.l2 == 0.0
        K = self.lm.nc
        if self.multi_nopen:
            nd = self.lm.n_dof()
            for j in range(nd):
                last = self.coef[j * K + K - 1]
                for k in range(K):
                    self.coef[j * K + k] -= last
        elif self.multi_int:
            b = self.lm.p * K
            last = self.coef[self.coef.shape[0] - 1]
            for k in range(K):
                self.coef[b + k] -= last

    def inner_solve(self):
        import scipy.linalg as sl
        if self.hessian_warning:
            self.use_fallback = True
            return
        N = self.coef.shape[0]
        K = self.lm.nc
        if self.multi_nopen:
            nd = N // K
            m = N - nd
            for j in range(nd):
                self.gradient[j * K + K - 1] = 0.0
                for c in range(N):
                    self.hessian[j * K + K - 1, c] = 0.0
                    self.hessian[c, j * K + K - 1] = 0.0
            g = np.empty(m)
            H = np.empty((m, m))
            idx = List[int]()
            for j in range(nd):
                for k in range(K - 1):
                    idx.append(j * K + k)
            for a in range(m):
                g[a] = self.gradient[idx[a]]
                for b in range(m):
                    H[a, b] = self.hessian[idx[a], idx[b]]
        elif self.multi_int:
            self.gradient[N - 1] = 0.0
            for c in range(N):
                self.hessian[N - 1, c] = 0.0
                self.hessian[c, N - 1] = 0.0
            g = self.gradient[:N - 1].copy()
            H = self.hessian[:N - 1, :N - 1].copy()
        else:
            g = self.gradient
            H = self.hessian
        try:
            sol = sl.solve(H, -g, check_finite=False, assume_a="sym")
        except:
            self.use_fallback = True
            return
        if not np.all(np.isfinite(sol)):
            self.use_fallback = True
            return
        if self.multi_nopen:
            cn = np.zeros(N)
            a = 0
            for j in range(N // K):
                for k in range(K - 1):
                    cn[j * K + k] = sol[a]
                    a += 1
            self.coef_newton = cn
        elif self.multi_int:
            cn = np.zeros(N)
            for a in range(N - 1):
                cn[a] = sol[a]
            self.coef_newton = cn
        else:
            self.coef_newton = sol
        self.gradient_times_newton = float(self.gradient @ self.coef_newton)
        if self.gradient_times_newton > 0.0:
            self.use_fallback = True

    def line_search(self):
        beta = 0.5
        sigma = 0.00048828125
        eps = 16.0 * _EPS
        t = 1.0
        armijo = sigma * self.gradient_times_newton
        coef_old = self.coef
        loss_old = self.loss_value
        grad_old = self.gradient
        sum_abs_grad_old = -1.0
        multi = self._is_multi()
        rn1 = np.zeros(0)
        rnK = np.zeros((0, 0))
        if multi:
            # raw of coef_newton includes its intercept part
            rnK = self.lm.rawK(self.coef_newton)
        else:
            rn1 = self.lm.raw1(self.coef_newton)
        done = False
        r1 = self.raw1
        rK = self.rawK
        for i in range(21):
            self.coef = coef_old + t * self.coef_newton
            if multi:
                rK = self.rawK + t * rnK
                self.loss_value, self.gradient = self.lm.lg_rawK(self.coef, rK, self.l2)
            else:
                r1 = self.raw1 + t * rn1
                self.loss_value, self.gradient = self.lm.lg_raw1(self.coef, r1, self.l2)
            imp = self.loss_value - loss_old
            if imp <= t * armijo:
                done = True
                break
            if abs(imp) <= abs(loss_old * eps):
                if sum_abs_grad_old < 0.0:
                    sum_abs_grad_old = float(np.sum(np.abs(grad_old)))
                if float(np.sum(np.abs(self.gradient))) < sum_abs_grad_old:
                    done = True
                    break
            t *= beta
        if not done:
            self.use_fallback = True
            return
        self.raw1 = r1
        self.rawK = rK

    def check_convergence(self):
        if float(np.max(np.abs(self.gradient))) > self.tol:
            return
        d2 = float(self.coef_newton @ (self.hessian @ self.coef_newton))
        if not (0.5 * d2 <= self.tol):
            return
        self.converged = True

    def fallback(self):
        mi = self.max_iter - self.iteration
        x, nit, st = lbfgs_fit(self.lm, self.coef, self.l2, mi, self.tol)
        self.iteration += nit
        self.coef = x
        self.converged = st == 0

    def finalize(self):
        K = self.lm.nc
        if self.multi_nopen:
            nd = self.lm.n_dof()
            for j in range(nd):
                m = 0.0
                for k in range(K):
                    m += self.coef[j * K + k]
                m /= float(K)
                for k in range(K):
                    self.coef[j * K + k] -= m
        elif self.multi_int:
            b = self.lm.p * K
            m = float(np.mean(self.coef[b:]))
            for k in range(K):
                self.coef[b + k] -= m

    def solve(self) -> np.ndarray[float, 1]:
        self.setup()
        self.iteration = 1
        self.converged = False
        self.use_fallback = False
        while self.iteration <= self.max_iter and not self.converged:
            self.use_fallback = False
            g, H, w = self.lm.gradient_hessian(self.coef, self.l2)
            self.gradient = g
            self.hessian = H
            self.hessian_warning = w
            self.inner_solve()
            if self.use_fallback:
                break
            self.line_search()
            if self.use_fallback:
                break
            self.check_convergence()
            self.iteration += 1
        if not self.converged:
            if self.use_fallback:
                self.fallback()
        self.iteration -= 1
        self.finalize()
        return self.coef
