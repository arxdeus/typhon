# Ported from scikit-learn 1.9.1 (BSD-3-Clause)
#
# LinearModelLoss (sklearn/linear_model/_linear_loss.py) with the pointwise
# losses of sklearn/_loss/_loss.pyx.tp, dense float64 only.
# kind: BINOMIAL (HalfBinomialLoss), MULTINOMIAL (HalfMultinomialLoss),
# TWEEDIE_LOG (HalfPoisson p=1, HalfGamma p=2, HalfTweedieLoss),
# TWEEDIE_ID (HalfTweedieLossIdentity, p=0 is HalfSquaredError).
# Multinomial coefficients are flat in Fortran order of (n_classes, n_dof):
# coef[j * n_classes + k]. Sample weights are always materialized (ones when
# None): multiplying by 1.0 is exact, so values equal the unweighted path.

import math
import numpy as np

BINOMIAL = 0
MULTINOMIAL = 1
TWEEDIE_LOG = 2
TWEEDIE_ID = 3

def _log1pexp(x: float) -> float:
    if x <= -37.0:
        return math.exp(x)
    elif x <= -2.0:
        return math.log1p(math.exp(x))
    elif x <= 18.0:
        return math.log(1.0 + math.exp(x))
    elif x <= 33.3:
        return x + math.exp(-x)
    else:
        return x

def _closs(kind: int, p: float, y: float, r: float) -> float:
    if kind == BINOMIAL:
        return _log1pexp(r) - y * r
    elif kind == TWEEDIE_LOG:
        if p == 0.0:
            e = math.exp(r)
            return 0.5 * (e - y) * (e - y)
        elif p == 1.0:
            return math.exp(r) - y * r
        elif p == 2.0:
            return r + y * math.exp(-r)
        return (math.exp((2.0 - p) * r) / (2.0 - p) - y * math.exp((1.0 - p) * r) / (1.0 - p))
    else:
        if p == 0.0:
            return 0.5 * (r - y) * (r - y)
        elif p == 1.0:
            if y == 0.0:
                return r
            return y * math.log(y / r) + r - y
        elif p == 2.0:
            return math.log(r / y) + y / r - 1.0
        t = math.pow(r, 1.0 - p)
        t = r * t / (2.0 - p) - y * t / (1.0 - p)
        if y > 0.0:
            t += math.pow(y, 2.0 - p) / ((1.0 - p) * (2.0 - p))
        return t

def _closs_grad(kind: int, p: float, y: float, r: float) -> Tuple[float, float]:
    if kind == BINOMIAL:
        if r <= -37.0:
            t = math.exp(r)
            return t - y * r, t - y
        elif r <= -2.0:
            t = math.exp(r)
            return math.log1p(t) - y * r, ((1.0 - y) * t - y) / (1.0 + t)
        elif r <= 18.0:
            t = math.exp(-r)
            return math.log1p(t) + (1.0 - y) * r, ((1.0 - y) - y * t) / (1.0 + t)
        t = math.exp(-r)
        return t + (1.0 - y) * r, ((1.0 - y) - y * t) / (1.0 + t)
    elif kind == TWEEDIE_LOG:
        if p == 0.0:
            e = math.exp(r)
            return 0.5 * (e - y) * (e - y), e * (e - y)
        elif p == 1.0:
            t = math.exp(r)
            return t - y * r, t - y
        elif p == 2.0:
            t = math.exp(-r)
            return r + y * t, 1.0 - y * t
        e1 = math.exp((1.0 - p) * r)
        e2 = math.exp((2.0 - p) * r)
        return e2 / (2.0 - p) - y * e1 / (1.0 - p), e2 - y * e1
    else:
        if p == 0.0:
            g = r - y
            return 0.5 * g * g, g
        elif p == 1.0:
            l = r if y == 0.0 else y * math.log(y / r) + r - y
            return l, 1.0 - y / r
        elif p == 2.0:
            return math.log(r / y) + y / r - 1.0, (r - y) / (r * r)
        t = math.pow(r, 1.0 - p)
        l = r * t / (2.0 - p) - y * t / (1.0 - p)
        if y > 0.0:
            l += math.pow(y, 2.0 - p) / ((1.0 - p) * (2.0 - p))
        return l, t * (1.0 - y / r)

def _cgrad(kind: int, p: float, y: float, r: float) -> float:
    if kind == BINOMIAL:
        if r > -37.0:
            t = math.exp(-r)
            return ((1.0 - y) - y * t) / (1.0 + t)
        return math.exp(r) - y
    elif kind == TWEEDIE_LOG:
        if p == 0.0:
            e = math.exp(r)
            return e * (e - y)
        elif p == 1.0:
            return math.exp(r) - y
        elif p == 2.0:
            return 1.0 - y * math.exp(-r)
        return math.exp((2.0 - p) * r) - y * math.exp((1.0 - p) * r)
    else:
        if p == 0.0:
            return r - y
        elif p == 1.0:
            return 1.0 - y / r
        elif p == 2.0:
            return (r - y) / (r * r)
        return math.pow(r, -p) * (r - y)

def _cgrad_hess(kind: int, p: float, y: float, r: float) -> Tuple[float, float]:
    if kind == BINOMIAL:
        if r > -37.0:
            t = math.exp(-r)
            return ((1.0 - y) - y * t) / (1.0 + t), t / ((1.0 + t) * (1.0 + t))
        t = math.exp(r)
        return t - y, t
    elif kind == TWEEDIE_LOG:
        if p == 0.0:
            e = math.exp(r)
            return e * (e - y), e * (2.0 * e - y)
        elif p == 1.0:
            h = math.exp(r)
            return h - y, h
        elif p == 2.0:
            t = math.exp(-r)
            return 1.0 - y * t, t * y
        e1 = math.exp((1.0 - p) * r)
        e2 = math.exp((2.0 - p) * r)
        return e2 - y * e1, (2.0 - p) * e2 - (1.0 - p) * y * e1
    else:
        if p == 0.0:
            return r - y, 1.0
        elif p == 1.0:
            return 1.0 - y / r, y / (r * r)
        elif p == 2.0:
            t = r * r
            return (r - y) / t, (-1.0 + 2.0 * y / r) / t
        t = math.pow(r, -p)
        return t * (r - y), t * ((1.0 - p) + p * y / r)

class LinearModelLoss:
    X: np.ndarray[float, 2]
    y: np.ndarray[float, 1]
    sw: np.ndarray[float, 1]
    sw_sum: float
    kind: int
    power: float
    nc: int
    fit_intercept: bool
    n: int
    p: int
    # cached by gradient_hessian_product (newton-cg)
    _hp: np.ndarray[float, 1]
    _hX: np.ndarray[float, 2]
    _hX_sum: np.ndarray[float, 1]
    _hsum: float
    _proba: np.ndarray[float, 2]

    def __init__(self, X: np.ndarray[float, 2], y: np.ndarray[float, 1], sw: np.ndarray[float, 1],
                 kind: int, power: float, nc: int, fit_intercept: bool):
        self.X = X
        self.y = y
        self.sw = sw
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.sw_sum = float(np.sum(sw))
        self.kind = kind
        self.power = power
        self.nc = nc
        self.fit_intercept = fit_intercept
        self._hp = np.zeros(0)
        self._hX = np.zeros((0, 0))
        self._hX_sum = np.zeros(0)
        self._hsum = 0.0
        self._proba = np.zeros((0, 0))

    def n_dof(self) -> int:
        return self.p + (1 if self.fit_intercept else 0)

    def multiclass(self) -> bool:
        return self.kind == MULTINOMIAL

    # --- binary / single-output ---
    def _w1(self, coef: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        return coef[:self.p]

    def raw1(self, coef: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        r = self.X @ coef[:self.p]
        if self.fit_intercept:
            r = r + coef[self.p]
        return r

    # --- multinomial: W (K, p), intercept (K,) ---
    def _W(self, coef: np.ndarray[float, 1]) -> np.ndarray[float, 2]:
        K = self.nc
        W = np.empty((K, self.p))
        for j in range(self.p):
            for k in range(K):
                W[k, j] = coef[j * K + k]
        return W

    def rawK(self, coef: np.ndarray[float, 1]) -> np.ndarray[float, 2]:
        K = self.nc
        R = self.X @ self._W(coef).T
        if self.fit_intercept:
            b = self.p * K
            for i in range(self.n):
                for k in range(K):
                    R[i, k] += coef[b + k]
        return R

    def l2_penalty(self, coef: np.ndarray[float, 1], l2: float) -> float:
        if self.kind == MULTINOMIAL:
            w = coef[:self.p * self.nc]
            return float(0.5 * l2 * float(np.dot(w, w)))
        w = coef[:self.p]
        return float(0.5 * l2 * float(w @ w))

    def _multi_lg(self, R: np.ndarray[float, 2], want_loss: bool):
        # HalfMultinomialLoss.loss_gradient / gradient_proba (per-sample)
        n = self.n
        K = self.nc
        lo = np.empty(n)
        G = np.empty((n, K))
        P = np.empty((n, K))
        pk = np.empty(K)
        for i in range(n):
            mx = R[i, 0]
            for k in range(1, K):
                if mx < R[i, k]:
                    mx = R[i, k]
            se = 0.0
            for k in range(K):
                pk[k] = math.exp(R[i, k] - mx)
                se += pk[k]
            li = math.log(se) + mx
            yi = self.y[i]
            for k in range(K):
                if yi == float(k):
                    li -= R[i, k]
                pk[k] /= se
                P[i, k] = pk[k]
                G[i, k] = (pk[k] - (1.0 if yi == float(k) else 0.0)) * self.sw[i]
            lo[i] = li * self.sw[i]
        return lo, G, P

    def loss(self, coef: np.ndarray[float, 1], l2: float) -> float:
        if self.kind == MULTINOMIAL:
            lo, G, P = self._multi_lg(self.rawK(coef), True)
            f = float(np.sum(lo) / self.sw_sum)
        else:
            r = self.raw1(coef)
            lo = np.empty(self.n)
            for i in range(self.n):
                lo[i] = self.sw[i] * _closs(self.kind, self.power, self.y[i], r[i])
            f = float(np.sum(lo) / self.sw_sum)
        if l2 > 0.0:
            f += self.l2_penalty(coef, l2)
        return f

    def _assemble_grad1(self, coef: np.ndarray[float, 1], gp: np.ndarray[float, 1], l2: float) -> np.ndarray[float, 1]:
        g = np.empty(coef.shape[0])
        gx = self.X.T @ gp + l2 * coef[:self.p]
        for j in range(self.p):
            g[j] = gx[j]
        if self.fit_intercept:
            g[self.p] = float(np.sum(gp))
        return g

    def _assemble_gradK(self, coef: np.ndarray[float, 1], G: np.ndarray[float, 2], l2: float) -> np.ndarray[float, 1]:
        K = self.nc
        g = np.empty(coef.shape[0])
        gx = G.T @ self.X + l2 * self._W(coef)
        for j in range(self.p):
            for k in range(K):
                g[j * K + k] = gx[k, j]
        if self.fit_intercept:
            s = np.sum(G, axis=0)
            for k in range(K):
                g[self.p * K + k] = s[k]
        return g

    def loss_gradient(self, coef: np.ndarray[float, 1], l2: float) -> Tuple[float, np.ndarray[float, 1]]:
        if self.kind == MULTINOMIAL:
            lo, G, P = self._multi_lg(self.rawK(coef), True)
            f = float(np.sum(lo) / self.sw_sum)
            f += self.l2_penalty(coef, l2)
            G /= self.sw_sum
            return f, self._assemble_gradK(coef, G, l2)
        r = self.raw1(coef)
        lo = np.empty(self.n)
        gp = np.empty(self.n)
        for i in range(self.n):
            a, b = _closs_grad(self.kind, self.power, self.y[i], r[i])
            lo[i] = self.sw[i] * a
            gp[i] = self.sw[i] * b
        f = float(np.sum(lo) / self.sw_sum)
        f += self.l2_penalty(coef, l2)
        gp /= self.sw_sum
        return f, self._assemble_grad1(coef, gp, l2)

    def gradient(self, coef: np.ndarray[float, 1], l2: float) -> np.ndarray[float, 1]:
        if self.kind == MULTINOMIAL:
            lo, G, P = self._multi_lg(self.rawK(coef), False)
            G /= self.sw_sum
            return self._assemble_gradK(coef, G, l2)
        r = self.raw1(coef)
        gp = np.empty(self.n)
        for i in range(self.n):
            gp[i] = self.sw[i] * _cgrad(self.kind, self.power, self.y[i], r[i])
        gp /= self.sw_sum
        return self._assemble_grad1(coef, gp, l2)

    def gradient_hessian(self, coef: np.ndarray[float, 1], l2: float):
        """Returns (grad, hess, hessian_warning)."""
        nd = self.n_dof()
        p = self.p
        if self.kind != MULTINOMIAL:
            r = self.raw1(coef)
            gp = np.empty(self.n)
            hp = np.empty(self.n)
            for i in range(self.n):
                a, b = _cgrad_hess(self.kind, self.power, self.y[i], r[i])
                gp[i] = self.sw[i] * a
                hp[i] = self.sw[i] * b
            gp /= self.sw_sum
            hp /= self.sw_sum
            nonpos = 0.0
            for i in range(self.n):
                if hp[i] <= 0.0:
                    nonpos += self.sw[i]
            warn = nonpos / self.sw_sum > 0.25
            hp = np.abs(hp)
            g = self._assemble_grad1(coef, gp, l2)
            H = np.empty((nd, nd))
            if warn:
                return g, H, True
            WX = hp[:, None] * self.X
            S = self.X.T @ WX
            for a in range(p):
                for b in range(p):
                    H[a, b] = S[a, b]
            if l2 > 0.0:
                for a in range(p):
                    H[a, a] += l2
            if self.fit_intercept:
                Xh = self.X.T @ hp
                for a in range(p):
                    H[a, p] = Xh[a]
                    H[p, a] = Xh[a]
                H[p, p] = float(np.sum(hp))
            return g, H, False
        K = self.nc
        lo, G, P = self._multi_lg(self.rawK(coef), False)
        G /= self.sw_sum
        g = self._assemble_gradK(coef, G, l2)
        N = coef.shape[0]
        H = np.zeros((N, N))
        swn = self.sw / self.sw_sum
        for k in range(K):
            for l in range(k, K):
                h = np.empty(self.n)
                for i in range(self.n):
                    if k == l:
                        h[i] = P[i, k] * (1.0 - P[i, k]) * swn[i]
                    else:
                        h[i] = -P[i, k] * P[i, l] * swn[i]
                S = self.X.T @ (h[:, None] * self.X)
                for a in range(p):
                    for b in range(p):
                        H[a * K + k, b * K + l] = S[a, b]
                if self.fit_intercept:
                    Xh = self.X.T @ h
                    for a in range(p):
                        H[a * K + k, p * K + l] = Xh[a]
                        H[p * K + k, a * K + l] = Xh[a]
                    H[p * K + k, p * K + l] = float(np.sum(h))
                if l != k:
                    for a in range(nd):
                        for b in range(nd):
                            H[a * K + l, b * K + k] = H[b * K + k, a * K + l]
        if l2 > 0.0:
            for a in range(p * K):
                H[a, a] += l2
        return g, H, False

    def grad_hessp_setup(self, coef: np.ndarray[float, 1], l2: float) -> np.ndarray[float, 1]:
        """gradient_hessian_product: returns grad, caches the state used by hessp."""
        if self.kind != MULTINOMIAL:
            r = self.raw1(coef)
            gp = np.empty(self.n)
            hp = np.empty(self.n)
            for i in range(self.n):
                a, b = _cgrad_hess(self.kind, self.power, self.y[i], r[i])
                gp[i] = self.sw[i] * a
                hp[i] = self.sw[i] * b
            gp /= self.sw_sum
            hp /= self.sw_sum
            g = self._assemble_grad1(coef, gp, l2)
            self._hsum = float(np.sum(hp))
            self._hX = hp[:, None] * self.X
            if self.fit_intercept:
                self._hX_sum = np.sum(self._hX, axis=0)
            return g
        lo, G, P = self._multi_lg(self.rawK(coef), False)
        G /= self.sw_sum
        self._proba = P
        return self._assemble_gradK(coef, G, l2)

    def hessp(self, s: np.ndarray[float, 1], l2: float) -> np.ndarray[float, 1]:
        p = self.p
        ret = np.empty(s.shape[0])
        if self.kind != MULTINOMIAL:
            sp = s[:p]
            v = self.X.T @ (self._hX @ sp)
            for j in range(p):
                ret[j] = v[j] + l2 * sp[j]
            if self.fit_intercept:
                for j in range(p):
                    ret[j] += s[p] * self._hX_sum[j]
                ret[p] = float(self._hX_sum @ sp) + self._hsum * s[p]
            return ret
        K = self.nc
        Ws = self._W(s)
        tmp = self.X @ Ws.T
        if self.fit_intercept:
            for i in range(self.n):
                for k in range(K):
                    tmp[i, k] += s[p * K + k]
        P = self._proba
        ps = np.sum(P * tmp, axis=1)
        for i in range(self.n):
            for k in range(K):
                tmp[i, k] = (tmp[i, k] - ps[i]) * P[i, k] * self.sw[i]
        hx = tmp.T @ self.X / self.sw_sum + l2 * Ws
        for j in range(p):
            for k in range(K):
                ret[j * K + k] = hx[k, j]
        if self.fit_intercept:
            ts = np.sum(tmp, axis=0) / self.sw_sum
            for k in range(K):
                ret[p * K + k] = ts[k]
        return ret

    # --- variants with a given raw prediction (NewtonSolver line search) ---
    def lg_raw1(self, coef: np.ndarray[float, 1], r: np.ndarray[float, 1], l2: float) -> Tuple[float, np.ndarray[float, 1]]:
        lo = np.empty(self.n)
        gp = np.empty(self.n)
        for i in range(self.n):
            a, b = _closs_grad(self.kind, self.power, self.y[i], r[i])
            lo[i] = self.sw[i] * a
            gp[i] = self.sw[i] * b
        f = float(np.sum(lo) / self.sw_sum)
        f += self.l2_penalty(coef, l2)
        gp /= self.sw_sum
        return f, self._assemble_grad1(coef, gp, l2)

    def lg_rawK(self, coef: np.ndarray[float, 1], R: np.ndarray[float, 2], l2: float) -> Tuple[float, np.ndarray[float, 1]]:
        lo, G, P = self._multi_lg(R, True)
        f = float(np.sum(lo) / self.sw_sum)
        f += self.l2_penalty(coef, l2)
        G /= self.sw_sum
        return f, self._assemble_gradK(coef, G, l2)

    def loss_raw1(self, coef: np.ndarray[float, 1], r: np.ndarray[float, 1], l2: float) -> float:
        lo = np.empty(self.n)
        for i in range(self.n):
            lo[i] = self.sw[i] * _closs(self.kind, self.power, self.y[i], r[i])
        f = float(np.sum(lo) / self.sw_sum)
        if l2 > 0.0:
            f += self.l2_penalty(coef, l2)
        return f

class _LossFG:
    """fun/grad pair for scipy-style optimizers. mode 0: loss_gradient (one
    pass, as jac=True); mode 1: separate loss / gradient (newton-cg)."""
    lm: LinearModelLoss
    l2: float
    mode: int
    _x: np.ndarray[float, 1]
    _g: np.ndarray[float, 1]
    _has: bool

    def __init__(self, lm: LinearModelLoss, l2: float, mode: int):
        self.lm = lm
        self.l2 = l2
        self.mode = mode
        self._x = np.zeros(0)
        self._g = np.zeros(0)
        self._has = False

    def _same(self, x: np.ndarray[float, 1]) -> bool:
        if not self._has or self._x.shape[0] != x.shape[0]:
            return False
        for i in range(x.shape[0]):
            if self._x[i] != x[i]:
                return False
        return True

    def fun(self, x: np.ndarray[float, 1]) -> float:
        if self.mode == 1:
            return self.lm.loss(x, self.l2)
        f, g = self.lm.loss_gradient(x, self.l2)
        self._x = x.copy()
        self._g = g
        self._has = True
        return f

    def grad(self, x: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        if self.mode == 1:
            return self.lm.gradient(x, self.l2)
        if not self._same(x):
            self.fun(x)
        return self._g.copy()

def lbfgs_fit(lm: LinearModelLoss, w0: np.ndarray[float, 1], l2: float, max_iter: int, tol: float):
    """scipy L-BFGS-B as sklearn calls it (maxls=50, gtol=tol, ftol=64 eps).
    Returns (x, n_iter, status)."""
    from scipy.optimize import minimize
    fg = _LossFG(lm, l2, 0)
    opts = {"maxiter": float(max_iter), "maxls": 50.0, "gtol": tol,
            "ftol": 64.0 * 2.220446049250313e-16}
    res = minimize(fg.fun, w0, method="L-BFGS-B", jac=fg.grad, options=opts)
    return res.x.copy(), min(res.nit, max_iter), res.status
