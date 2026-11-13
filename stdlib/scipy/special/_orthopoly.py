# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)
# orthopoly1d: orthogonal polynomial objects (scipy.special._orthogonal).
# The scipy implementation stores Python callables for the weight function and
# the evaluation function; here they are encoded as (kind, a, b) triples.

import math
import numpy as np
from numpy import poly1d, polyval, poly
from . import _orth as O

# evaluation kinds (ek)
#  0 none (polyval), -1 ones, 1 jacobi(n,a,b), 2 sh_jacobi(n,a,b), 3 genlaguerre(n,a),
#  4 laguerre, 5 hermite, 6 hermitenorm, 7 gegenbauer(n,a), 8 chebyt, 10 chebyc,
#  11 chebys, 14 legendre, 15 sh_legendre
# weight kinds (wk) share the numbering: 1 (1-x)^a (1+x)^b, 2 (1-x)^(a-b) x^(b-1),
#  3 exp(-x) x^a, 5 exp(-x^2), 6 exp(-x^2/2), 8 1/sqrt(1-x^2), 10 1/sqrt(1-x^2/4),
#  11 sqrt(1-x^2/4), 14 1

def _wfunc(wk: int, wa: float, wb: float, x: float) -> float:
    if wk == 1:
        return (1 - x) ** wa * (1 + x) ** wb
    elif wk == 2:
        return (1.0 - x) ** (wa - wb) * x ** (wb - 1.0)
    elif wk == 3:
        return math.exp(-x) * x ** wa
    elif wk == 5:
        return math.exp(-x * x)
    elif wk == 6:
        return math.exp(-x * x / 2.0)
    elif wk == 8:
        return 1.0 / math.sqrt(1 - x * x)
    elif wk == 10:
        return 1.0 / math.sqrt(1 - x * x / 4.0)
    elif wk == 11:
        return math.sqrt(1 - x * x / 4.0)
    return 1.0

def _efunc(ek: int, n: int, a: float, b: float, x: float) -> float:
    if ek == -1:
        return 1.0
    elif ek == 1:
        return O.eval_jacobi_l(n, a, b, x)
    elif ek == 2:
        return O.eval_sh_jacobi_l(n, a, b, x)
    elif ek == 3:
        return O.eval_genlaguerre_l(n, a, x)
    elif ek == 4:
        return O.eval_laguerre_l(n, x)
    elif ek == 5:
        return O.eval_hermite(n, x)
    elif ek == 6:
        return O.eval_hermitenorm(n, x)
    elif ek == 7:
        return O.eval_gegenbauer(float(n), a, x)
    elif ek == 8:
        return O.eval_chebyt_l(n, x)
    elif ek == 10:
        return O.eval_chebyc_l(n, x)
    elif ek == 11:
        return O.eval_chebys_l(n, x)
    elif ek == 14:
        return O.eval_legendre_l(n, x)
    elif ek == 15:
        return O.eval_sh_legendre_l(n, x)
    return 1.0

class orthopoly1d(poly1d):
    weights: np.ndarray[float, 2]
    normcoef: float
    limits: Tuple[float, float]
    ek: int
    en: int
    ea: float
    eb: float
    escale: float
    wk: int
    wa: float
    wb: float

    def __init__(self, roots: np.ndarray[float, 1], weights: np.ndarray[float, 1],
                 hn: float, kn: float, wk: int, wa: float, wb: float,
                 limits: Tuple[float, float], monic: bool,
                 ek: int, en: int, ea: float, eb: float):
        m = len(roots)
        mu = math.sqrt(hn)
        escale = 1.0
        if monic:
            if ek != 0:
                escale = 1.0 / kn
            mu = mu / abs(kn)
            kn = 1.0
        if m == 0:
            c = np.array([kn])
        else:
            c = poly(roots).astype(float) * kn
        poly1d.__init__(self, c)
        self.weights = np.empty((m, 3), dtype=float)
        for k in range(m):
            self.weights[k, 0] = roots[k]
            self.weights[k, 1] = weights[k]
            self.weights[k, 2] = weights[k] / _wfunc(wk, wa, wb, roots[k])
        self.wk = wk
        self.wa = wa
        self.wb = wb
        self.limits = limits
        self.normcoef = mu
        self.ek = ek
        self.en = en
        self.ea = ea
        self.eb = eb
        self.escale = escale

    def weight_func(self, x: float) -> float:
        return _wfunc(self.wk, self.wa, self.wb, x)

    def _ev(self, x: float) -> float:
        return _efunc(self.ek, self.en, self.ea, self.eb, x) * self.escale

    def __call__(self, v):
        if self.ek == 0:
            return polyval(self.coeffs, v)
        elif isinstance(v, float) or isinstance(v, int):
            return self._ev(float(v))
        else:
            a = np.asarray(v).astype(float)
            out = np.empty(a.shape, dtype=float)
            for idx in np.ndindex(a.shape):
                out[idx] = self._ev(float(a[idx]))
            return out

    def _scale(self, p: float):
        if p == 1.0:
            return
        self.coeffs = self.coeffs * p
        if self.ek != 0:
            self.escale = self.escale * p
        self.normcoef = self.normcoef * p
