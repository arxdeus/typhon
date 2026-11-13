# Ported from scikit-learn 1.9.1 (BSD-3-Clause)
# Port of sklearn/svm/src/libsvm/svm.cpp (libsvm 3.x, BSD-3-Clause,
# Chih-Chung Chang and Chih-Jen Lin) as vendored and patched by sklearn
# (sample weights, sorted labels, max_iter, mt19937 bounded_rand_int).

# Dense input, kernels linear / poly / rbf / sigmoid / precomputed. The SMO
# follows libsvm step by step (WSS3, shrinking, float32 Q entries, gradient
# reconstruction, Solver_NU), so iterate paths and eps-stopped solutions are
# libsvm's. Kernel values come from a precomputed matrix over the training
# rows instead of the LRU cache (identical values).

import math
import numpy as np
from ._newrand import _StdMT, _fma, _f32

_INF = float("inf")
_TAU = 1e-12
_LOWER: Literal[int] = 0
_UPPER: Literal[int] = 1
_FREE: Literal[int] = 2

C_SVC: Literal[int] = 0
NU_SVC: Literal[int] = 1
ONE_CLASS: Literal[int] = 2
EPSILON_SVR: Literal[int] = 3
NU_SVR: Literal[int] = 4

def _powi(base: float, times: int) -> float:
    tmp = base
    ret = 1.0
    t = times
    while t > 0:
        if t % 2 == 1:
            ret *= tmp
        tmp = tmp * tmp
        t //= 2
    return ret

def _dot(a: Ptr[float], b: Ptr[float], n: int) -> float:
    # libsvm calls BLAS ddot (scipy's BLAS)
    from numpy.linalg.blas import cblas_ddot, fint
    return cblas_ddot(fint(n), a.as_byte(), fint(1), b.as_byte(), fint(1))

class _Param:
    svm_type: int
    kernel: str
    degree: int
    gamma: float
    coef0: float
    C: float
    nu: float
    p: float
    eps: float
    shrinking: bool
    probability: bool
    max_iter: int
    class_weight: List[float]

    def __init__(self, svm_type: int, kernel: str, degree: int, gamma: float, coef0: float,
                 C: float, nu: float, p: float, eps: float, shrinking: bool,
                 probability: bool, max_iter: int, class_weight: List[float]):
        self.svm_type = svm_type
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.C = C
        self.nu = nu
        self.p = p
        self.eps = eps
        self.shrinking = shrinking
        self.probability = probability
        self.max_iter = max_iter
        self.class_weight = class_weight

    def copy(self) -> _Param:
        return _Param(self.svm_type, self.kernel, self.degree, self.gamma, self.coef0, self.C,
                      self.nu, self.p, self.eps, self.shrinking, self.probability,
                      self.max_iter, self.class_weight.copy())

class _Data:
    """Training matrix (row-major) + squared norms; kernel evaluation."""
    X: Ptr[float]
    p: int
    xsq: List[float]
    par: _Param
    keep: np.ndarray[float, 2]

    def __init__(self, Xa: np.ndarray[float, 2], par: _Param):
        self.keep = Xa
        self.X = Xa.data
        self.p = Xa.shape[1]
        self.par = par
        n = Xa.shape[0]
        self.xsq = [0.0] * n
        if par.kernel == "rbf":
            for i in range(n):
                self.xsq[i] = _dot(self.X + i * self.p, self.X + i * self.p, self.p)

    def k_train(self, i: int, j: int) -> float:
        # Kernel::kernel_* (training path: rbf uses x_square)
        par = self.par
        p = self.p
        a = self.X + i * p
        b = self.X + j * p
        if par.kernel == "rbf":
            return math.exp(-par.gamma * (self.xsq[i] + self.xsq[j] - 2 * _dot(a, b, p)))
        elif par.kernel == "linear":
            return _dot(a, b, p)
        elif par.kernel == "poly":
            return _powi(_fma(par.gamma, _dot(a, b, p), par.coef0), par.degree)
        elif par.kernel == "sigmoid":
            return math.tanh(_fma(par.gamma, _dot(a, b, p), par.coef0))
        else:
            # precomputed (dense rep): x[i].values[x[j].ind]
            return a[j]

def k_function(par: _Param, a: Ptr[float], b: Ptr[float], p: int) -> float:
    # Kernel::k_function (prediction path: rbf via the difference vector)
    if par.kernel == "rbf":
        m = Ptr[float](p)
        for k in range(p):
            m[k] = a[k] - b[k]
        return math.exp(-par.gamma * _dot(m, m, p))
    elif par.kernel == "linear":
        return _dot(a, b, p)
    elif par.kernel == "poly":
        return _powi(_fma(par.gamma, _dot(a, b, p), par.coef0), par.degree)
    elif par.kernel == "sigmoid":
        return math.tanh(_fma(par.gamma, _dot(a, b, p), par.coef0))
    else:
        raise ValueError("precomputed kernel needs the column index")

class _Solver:
    l: int
    active_size: int
    y: List[int]
    G: List[float]
    alpha_status: List[int]
    alpha: List[float]
    QD: List[float]
    eps: float
    C: List[float]
    p: List[float]
    active_set: List[int]
    G_bar: List[float]
    unshrink: bool
    nu: bool
    r: float
    # Q(i, j) = f32(s[a] * s[b] * K[kx[a], kx[b]]) for original variables a, b
    K: np.ndarray[float, 2]
    s: List[int]
    kx: List[int]

    def __init__(self, nu: bool):
        self.nu = nu
        self.r = 0.0
        self.l = 0
        self.active_size = 0
        self.unshrink = False
        self.eps = 0.0
        self.K = np.empty((0, 0))
        self.y = []; self.G = []; self.alpha_status = []; self.alpha = []; self.QD = []
        self.C = []; self.p = []; self.active_set = []; self.G_bar = []; self.s = []; self.kx = []

    def q(self, i: int, j: int) -> float:
        a = self.active_set[i]
        b = self.active_set[j]
        return _f32(float(self.s[a] * self.s[b]) * self.K[self.kx[a], self.kx[b]])

    def update_alpha_status(self, i: int):
        if self.alpha[i] >= self.C[i]:
            self.alpha_status[i] = _UPPER
        elif self.alpha[i] <= 0:
            self.alpha_status[i] = _LOWER
        else:
            self.alpha_status[i] = _FREE

    def is_upper(self, i: int) -> bool:
        return self.alpha_status[i] == _UPPER

    def is_lower(self, i: int) -> bool:
        return self.alpha_status[i] == _LOWER

    def is_free(self, i: int) -> bool:
        return self.alpha_status[i] == _FREE

    def swap_index(self, i: int, j: int):
        self.QD[i], self.QD[j] = self.QD[j], self.QD[i]
        self.y[i], self.y[j] = self.y[j], self.y[i]
        self.G[i], self.G[j] = self.G[j], self.G[i]
        self.alpha_status[i], self.alpha_status[j] = self.alpha_status[j], self.alpha_status[i]
        self.alpha[i], self.alpha[j] = self.alpha[j], self.alpha[i]
        self.p[i], self.p[j] = self.p[j], self.p[i]
        self.active_set[i], self.active_set[j] = self.active_set[j], self.active_set[i]
        self.G_bar[i], self.G_bar[j] = self.G_bar[j], self.G_bar[i]
        self.C[i], self.C[j] = self.C[j], self.C[i]

    def reconstruct_gradient(self):
        l = self.l
        if self.active_size == l:
            return
        for j in range(self.active_size, l):
            self.G[j] = self.G_bar[j] + self.p[j]
        nr_free = 0
        for j in range(self.active_size):
            if self.is_free(j):
                nr_free += 1
        if nr_free * l > 2 * self.active_size * (l - self.active_size):
            for i in range(self.active_size, l):
                for j in range(self.active_size):
                    if self.is_free(j):
                        self.G[i] = _fma(self.alpha[j], self.q(i, j), self.G[i])
        else:
            for i in range(self.active_size):
                if self.is_free(i):
                    ai = self.alpha[i]
                    for j in range(self.active_size, l):
                        self.G[j] = _fma(ai, self.q(i, j), self.G[j])

    def select_working_set(self) -> Tuple[int, int, int]:
        if self.nu:
            return self._select_nu()
        Gmax = -_INF
        Gmax2 = -_INF
        Gmax_idx = -1
        Gmin_idx = -1
        obj_diff_min = _INF
        for t in range(self.active_size):
            if self.y[t] == 1:
                if not self.is_upper(t):
                    if -self.G[t] >= Gmax:
                        Gmax = -self.G[t]
                        Gmax_idx = t
            else:
                if not self.is_lower(t):
                    if self.G[t] >= Gmax:
                        Gmax = self.G[t]
                        Gmax_idx = t
        i = Gmax_idx
        for j in range(self.active_size):
            if self.y[j] == 1:
                if not self.is_lower(j):
                    grad_diff = Gmax + self.G[j]
                    if self.G[j] >= Gmax2:
                        Gmax2 = self.G[j]
                    if grad_diff > 0:
                        quad_coef = self.QD[i] + self.QD[j] - 2.0 * float(self.y[i]) * self.q(i, j)
                        obj_diff = 0.0
                        if quad_coef > 0:
                            obj_diff = -(grad_diff * grad_diff) / quad_coef
                        else:
                            obj_diff = -(grad_diff * grad_diff) / _TAU
                        if obj_diff <= obj_diff_min:
                            Gmin_idx = j
                            obj_diff_min = obj_diff
            else:
                if not self.is_upper(j):
                    grad_diff = Gmax - self.G[j]
                    if -self.G[j] >= Gmax2:
                        Gmax2 = -self.G[j]
                    if grad_diff > 0:
                        quad_coef = self.QD[i] + self.QD[j] + 2.0 * float(self.y[i]) * self.q(i, j)
                        obj_diff = 0.0
                        if quad_coef > 0:
                            obj_diff = -(grad_diff * grad_diff) / quad_coef
                        else:
                            obj_diff = -(grad_diff * grad_diff) / _TAU
                        if obj_diff <= obj_diff_min:
                            Gmin_idx = j
                            obj_diff_min = obj_diff
        if Gmax + Gmax2 < self.eps or Gmin_idx == -1:
            return (1, 0, 0)
        return (0, Gmax_idx, Gmin_idx)

    def _select_nu(self) -> Tuple[int, int, int]:
        Gmaxp = -_INF; Gmaxp2 = -_INF; Gmaxp_idx = -1
        Gmaxn = -_INF; Gmaxn2 = -_INF; Gmaxn_idx = -1
        Gmin_idx = -1
        obj_diff_min = _INF
        for t in range(self.active_size):
            if self.y[t] == 1:
                if not self.is_upper(t):
                    if -self.G[t] >= Gmaxp:
                        Gmaxp = -self.G[t]
                        Gmaxp_idx = t
            else:
                if not self.is_lower(t):
                    if self.G[t] >= Gmaxn:
                        Gmaxn = self.G[t]
                        Gmaxn_idx = t
        ip = Gmaxp_idx
        iN = Gmaxn_idx
        for j in range(self.active_size):
            if self.y[j] == 1:
                if not self.is_lower(j):
                    grad_diff = Gmaxp + self.G[j]
                    if self.G[j] >= Gmaxp2:
                        Gmaxp2 = self.G[j]
                    if grad_diff > 0:
                        quad_coef = self.QD[ip] + self.QD[j] - 2 * self.q(ip, j)
                        obj_diff = 0.0
                        if quad_coef > 0:
                            obj_diff = -(grad_diff * grad_diff) / quad_coef
                        else:
                            obj_diff = -(grad_diff * grad_diff) / _TAU
                        if obj_diff <= obj_diff_min:
                            Gmin_idx = j
                            obj_diff_min = obj_diff
            else:
                if not self.is_upper(j):
                    grad_diff = Gmaxn - self.G[j]
                    if -self.G[j] >= Gmaxn2:
                        Gmaxn2 = -self.G[j]
                    if grad_diff > 0:
                        quad_coef = self.QD[iN] + self.QD[j] - 2 * self.q(iN, j)
                        obj_diff = 0.0
                        if quad_coef > 0:
                            obj_diff = -(grad_diff * grad_diff) / quad_coef
                        else:
                            obj_diff = -(grad_diff * grad_diff) / _TAU
                        if obj_diff <= obj_diff_min:
                            Gmin_idx = j
                            obj_diff_min = obj_diff
        if max(Gmaxp + Gmaxp2, Gmaxn + Gmaxn2) < self.eps or Gmin_idx == -1:
            return (1, 0, 0)
        oi = Gmaxp_idx if self.y[Gmin_idx] == 1 else Gmaxn_idx
        return (0, oi, Gmin_idx)

    def be_shrunk(self, i: int, Gmax1: float, Gmax2: float) -> bool:
        if self.is_upper(i):
            if self.y[i] == 1:
                return -self.G[i] > Gmax1
            else:
                return -self.G[i] > Gmax2
        elif self.is_lower(i):
            if self.y[i] == 1:
                return self.G[i] > Gmax2
            else:
                return self.G[i] > Gmax1
        else:
            return False

    def be_shrunk_nu(self, i: int, Gmax1: float, Gmax2: float, Gmax3: float, Gmax4: float) -> bool:
        if self.is_upper(i):
            if self.y[i] == 1:
                return -self.G[i] > Gmax1
            else:
                return -self.G[i] > Gmax4
        elif self.is_lower(i):
            if self.y[i] == 1:
                return self.G[i] > Gmax2
            else:
                return self.G[i] > Gmax3
        else:
            return False

    def do_shrinking(self):
        if self.nu:
            self._shrink_nu()
            return
        Gmax1 = -_INF
        Gmax2 = -_INF
        for i in range(self.active_size):
            if self.y[i] == 1:
                if not self.is_upper(i):
                    if -self.G[i] >= Gmax1:
                        Gmax1 = -self.G[i]
                if not self.is_lower(i):
                    if self.G[i] >= Gmax2:
                        Gmax2 = self.G[i]
            else:
                if not self.is_upper(i):
                    if -self.G[i] >= Gmax2:
                        Gmax2 = -self.G[i]
                if not self.is_lower(i):
                    if self.G[i] >= Gmax1:
                        Gmax1 = self.G[i]
        if not self.unshrink and Gmax1 + Gmax2 <= self.eps * 10:
            self.unshrink = True
            self.reconstruct_gradient()
            self.active_size = self.l
        i = 0
        while i < self.active_size:
            if self.be_shrunk(i, Gmax1, Gmax2):
                self.active_size -= 1
                while self.active_size > i:
                    if not self.be_shrunk(self.active_size, Gmax1, Gmax2):
                        self.swap_index(i, self.active_size)
                        break
                    self.active_size -= 1
            i += 1

    def _shrink_nu(self):
        Gmax1 = -_INF; Gmax2 = -_INF; Gmax3 = -_INF; Gmax4 = -_INF
        for i in range(self.active_size):
            if not self.is_upper(i):
                if self.y[i] == 1:
                    if -self.G[i] > Gmax1:
                        Gmax1 = -self.G[i]
                elif -self.G[i] > Gmax4:
                    Gmax4 = -self.G[i]
            if not self.is_lower(i):
                if self.y[i] == 1:
                    if self.G[i] > Gmax2:
                        Gmax2 = self.G[i]
                elif self.G[i] > Gmax3:
                    Gmax3 = self.G[i]
        if not self.unshrink and max(Gmax1 + Gmax2, Gmax3 + Gmax4) <= self.eps * 10:
            self.unshrink = True
            self.reconstruct_gradient()
            self.active_size = self.l
        i = 0
        while i < self.active_size:
            if self.be_shrunk_nu(i, Gmax1, Gmax2, Gmax3, Gmax4):
                self.active_size -= 1
                while self.active_size > i:
                    if not self.be_shrunk_nu(self.active_size, Gmax1, Gmax2, Gmax3, Gmax4):
                        self.swap_index(i, self.active_size)
                        break
                    self.active_size -= 1
            i += 1

    def calculate_rho(self) -> float:
        if self.nu:
            return self._rho_nu()
        nr_free = 0
        ub = _INF
        lb = -_INF
        sum_free = 0.0
        for i in range(self.active_size):
            yG = float(self.y[i]) * self.G[i]
            if self.is_upper(i):
                if self.y[i] == -1:
                    ub = min(ub, yG)
                else:
                    lb = max(lb, yG)
            elif self.is_lower(i):
                if self.y[i] == 1:
                    ub = min(ub, yG)
                else:
                    lb = max(lb, yG)
            else:
                nr_free += 1
                sum_free += yG
        if nr_free > 0:
            return sum_free / float(nr_free)
        return (ub + lb) / 2

    def _rho_nu(self) -> float:
        nr_free1 = 0; nr_free2 = 0
        ub1 = _INF; ub2 = _INF
        lb1 = -_INF; lb2 = -_INF
        sum_free1 = 0.0; sum_free2 = 0.0
        for i in range(self.active_size):
            if self.y[i] == 1:
                if self.is_upper(i):
                    lb1 = max(lb1, self.G[i])
                elif self.is_lower(i):
                    ub1 = min(ub1, self.G[i])
                else:
                    nr_free1 += 1
                    sum_free1 += self.G[i]
            else:
                if self.is_upper(i):
                    lb2 = max(lb2, self.G[i])
                elif self.is_lower(i):
                    ub2 = min(ub2, self.G[i])
                else:
                    nr_free2 += 1
                    sum_free2 += self.G[i]
        r1 = sum_free1 / float(nr_free1) if nr_free1 > 0 else (ub1 + lb1) / 2
        r2 = sum_free2 / float(nr_free2) if nr_free2 > 0 else (ub2 + lb2) / 2
        self.r = (r1 + r2) / 2
        return (r1 - r2) / 2

    def solve(self, K: np.ndarray[float, 2], s: List[int], kx: List[int], p_: List[float],
              y_: List[int], alpha_: List[float], C_: List[float], eps: float,
              shrinking: bool, max_iter: int) -> Tuple[List[float], float, int]:
        l = len(y_)
        self.l = l
        self.K = K
        self.s = s
        self.kx = kx
        self.p = p_.copy()
        self.y = y_.copy()
        self.alpha = alpha_.copy()
        self.C = C_.copy()
        self.eps = eps
        self.unshrink = False
        self.active_set = list(range(l))
        self.active_size = l
        self.QD = [K[kx[i], kx[i]] for i in range(l)]
        self.alpha_status = [0] * l
        for i in range(l):
            self.update_alpha_status(i)
        self.G = [self.p[i] for i in range(l)]
        self.G_bar = [0.0] * l
        for i in range(l):
            if not self.is_lower(i):
                ai = self.alpha[i]
                for j in range(l):
                    self.G[j] = _fma(ai, self.q(i, j), self.G[j])
                if self.is_upper(i):
                    for j in range(l):
                        self.G_bar[j] = _fma(self.C[i], self.q(i, j), self.G_bar[j])
        it = 0
        counter = min(l, 1000) + 1
        while True:
            if max_iter != -1 and it >= max_iter:
                break
            counter -= 1
            if counter == 0:
                counter = min(l, 1000)
                if shrinking:
                    self.do_shrinking()
            st, i, j = self.select_working_set()
            if st != 0:
                self.reconstruct_gradient()
                self.active_size = l
                st, i, j = self.select_working_set()
                if st != 0:
                    break
                else:
                    counter = 1
            it += 1
            C_i = self.C[i]
            C_j = self.C[j]
            old_ai = self.alpha[i]
            old_aj = self.alpha[j]
            Qij = self.q(i, j)
            if self.y[i] != self.y[j]:
                quad_coef = self.QD[i] + self.QD[j] + 2 * Qij
                if quad_coef <= 0:
                    quad_coef = _TAU
                delta = (-self.G[i] - self.G[j]) / quad_coef
                diff = self.alpha[i] - self.alpha[j]
                self.alpha[i] += delta
                self.alpha[j] += delta
                if diff > 0:
                    if self.alpha[j] < 0:
                        self.alpha[j] = 0.0
                        self.alpha[i] = diff
                else:
                    if self.alpha[i] < 0:
                        self.alpha[i] = 0.0
                        self.alpha[j] = -diff
                if diff > C_i - C_j:
                    if self.alpha[i] > C_i:
                        self.alpha[i] = C_i
                        self.alpha[j] = C_i - diff
                else:
                    if self.alpha[j] > C_j:
                        self.alpha[j] = C_j
                        self.alpha[i] = C_j + diff
            else:
                quad_coef = self.QD[i] + self.QD[j] - 2 * Qij
                if quad_coef <= 0:
                    quad_coef = _TAU
                delta = (self.G[i] - self.G[j]) / quad_coef
                sm = self.alpha[i] + self.alpha[j]
                self.alpha[i] -= delta
                self.alpha[j] += delta
                if sm > C_i:
                    if self.alpha[i] > C_i:
                        self.alpha[i] = C_i
                        self.alpha[j] = sm - C_i
                else:
                    if self.alpha[j] < 0:
                        self.alpha[j] = 0.0
                        self.alpha[i] = sm
                if sm > C_j:
                    if self.alpha[j] > C_j:
                        self.alpha[j] = C_j
                        self.alpha[i] = sm - C_j
                else:
                    if self.alpha[i] < 0:
                        self.alpha[i] = 0.0
                        self.alpha[j] = sm
            dai = self.alpha[i] - old_ai
            daj = self.alpha[j] - old_aj
            for k in range(self.active_size):
                self.G[k] += _fma(self.q(i, k), dai, self.q(j, k) * daj)
            ui = self.is_upper(i)
            uj = self.is_upper(j)
            self.update_alpha_status(i)
            self.update_alpha_status(j)
            if ui != self.is_upper(i):
                if ui:
                    for k in range(l):
                        self.G_bar[k] = _fma(-C_i, self.q(i, k), self.G_bar[k])
                else:
                    for k in range(l):
                        self.G_bar[k] = _fma(C_i, self.q(i, k), self.G_bar[k])
            if uj != self.is_upper(j):
                if uj:
                    for k in range(l):
                        self.G_bar[k] = _fma(-C_j, self.q(j, k), self.G_bar[k])
                else:
                    for k in range(l):
                        self.G_bar[k] = _fma(C_j, self.q(j, k), self.G_bar[k])
        rho = self.calculate_rho()
        out = [0.0] * l
        for i in range(l):
            out[self.active_set[i]] = self.alpha[i]
        return out, rho, it

# ---------------------------------------------------------------- models

class Model:
    nr_class: int
    label: List[int]
    sv_rows: List[int]      # rows of the training data
    sv_ind: List[int]       # index within the zero-weight-filtered problem
    nSV: List[int]
    sv_coef: List[List[float]]
    rho: List[float]
    n_iter: List[int]
    probA: List[float]
    probB: List[float]

    def __init__(self):
        self.nr_class = 2
        self.label = []
        self.sv_rows = []
        self.sv_ind = []
        self.nSV = []
        self.sv_coef = []
        self.rho = []
        self.n_iter = []
        self.probA = []
        self.probB = []

class _Prob:
    rows: List[int]
    y: List[float]
    W: List[float]

    def __init__(self, rows: List[int], y: List[float], W: List[float]):
        self.rows = rows
        self.y = y
        self.W = W

def _kmat(d: _Data, rows: List[int]) -> np.ndarray[float, 2]:
    l = len(rows)
    K = np.empty((l, l))
    for a in range(l):
        for b in range(a, l):
            v = d.k_train(rows[a], rows[b])
            K[a, b] = v
            K[b, a] = v
    return K

def _train_one(d: _Data, prob: _Prob, par: _Param, Cp: float, Cn: float) -> Tuple[List[float], float, int]:
    l = len(prob.rows)
    K = _kmat(d, prob.rows)
    st = par.svm_type
    if st == C_SVC:
        y = [1 if prob.y[i] > 0 else -1 for i in range(l)]
        C = [prob.W[i] * (Cp if prob.y[i] > 0 else Cn) for i in range(l)]
        s = _Solver(False)
        alpha, rho, it = s.solve(K, y, list(range(l)), [-1.0] * l, y, [0.0] * l, C,
                                 par.eps, par.shrinking, par.max_iter)
        for i in range(l):
            alpha[i] *= float(y[i])
        return alpha, rho, it
    elif st == NU_SVC:
        y = [1 if prob.y[i] > 0 else -1 for i in range(l)]
        C = prob.W.copy()
        nu_l = 0.0
        for i in range(l):
            nu_l = _fma(par.nu, C[i], nu_l)
        sum_pos = nu_l / 2
        sum_neg = nu_l / 2
        a0 = [0.0] * l
        for i in range(l):
            if y[i] == 1:
                a0[i] = min(C[i], sum_pos)
                sum_pos -= a0[i]
            else:
                a0[i] = min(C[i], sum_neg)
                sum_neg -= a0[i]
        s = _Solver(True)
        alpha, rho, it = s.solve(K, y, list(range(l)), [0.0] * l, y, a0, C,
                                 par.eps, par.shrinking, par.max_iter)
        r = s.r
        for i in range(l):
            alpha[i] *= float(y[i]) / r
        return alpha, rho / r, it
    elif st == ONE_CLASS:
        C = prob.W.copy()
        nu_l = 0.0
        for i in range(l):
            nu_l = _fma(C[i], par.nu, nu_l)
        a0 = [0.0] * l
        i = 0
        while nu_l > 0:
            a0[i] = min(C[i], nu_l)
            nu_l -= a0[i]
            i += 1
        ones = [1] * l
        s = _Solver(False)
        return s.solve(K, ones, list(range(l)), [0.0] * l, ones, a0, C,
                       par.eps, par.shrinking, par.max_iter)
    else:
        sg = [1] * l + [-1] * l
        kx = list(range(l)) + list(range(l))
        lin = [0.0] * (2 * l)
        C = [0.0] * (2 * l)
        a0 = [0.0] * (2 * l)
        nu = st == NU_SVR
        if not nu:
            for i in range(l):
                lin[i] = par.p - prob.y[i]
                lin[i + l] = par.p + prob.y[i]
                C[i] = prob.W[i] * par.C
                C[i + l] = prob.W[i] * par.C
        else:
            sm = 0.0
            for i in range(l):
                C[i] = prob.W[i] * par.C
                C[i + l] = C[i]
                sm = _fma(C[i], par.nu, sm)
            sm /= 2
            for i in range(l):
                a0[i] = min(sm, C[i])
                a0[i + l] = a0[i]
                sm -= a0[i]
                lin[i] = -prob.y[i]
                lin[i + l] = prob.y[i]
        s = _Solver(nu)
        a2, rho, it = s.solve(K, sg, kx, lin, sg, a0, C, par.eps, par.shrinking, par.max_iter)
        alpha = [a2[i] - a2[i + l] for i in range(l)]
        return alpha, rho, it

def _sigmoid_train(dec: List[float], yv: List[float]):
    l = len(dec)
    prior1 = 0.0
    prior0 = 0.0
    for i in range(l):
        if yv[i] > 0:
            prior1 += 1
        else:
            prior0 += 1
    max_iter = 100
    min_step = 1e-10
    sigma = 1e-12
    eps = 1e-5
    hiTarget = (prior1 + 1.0) / (prior1 + 2.0)
    loTarget = 1 / (prior0 + 2.0)
    t = [0.0] * l
    A = 0.0
    B = math.log((prior0 + 1.0) / (prior1 + 1.0))
    fval = 0.0
    for i in range(l):
        t[i] = hiTarget if yv[i] > 0 else loTarget
        fApB = _fma(dec[i], A, B)
        if fApB >= 0:
            fval += _fma(t[i], fApB, math.log(1 + math.exp(-fApB)))
        else:
            fval += _fma(t[i] - 1, fApB, math.log(1 + math.exp(fApB)))
    for it in range(max_iter):
        h11 = sigma; h22 = sigma; h21 = 0.0; g1 = 0.0; g2 = 0.0
        for i in range(l):
            fApB = _fma(dec[i], A, B)
            p = 0.0; q = 0.0
            if fApB >= 0:
                p = math.exp(-fApB) / (1.0 + math.exp(-fApB))
                q = 1.0 / (1.0 + math.exp(-fApB))
            else:
                p = 1.0 / (1.0 + math.exp(fApB))
                q = math.exp(fApB) / (1.0 + math.exp(fApB))
            d2 = p * q
            h11 = _fma(dec[i] * dec[i], d2, h11)
            h22 += d2
            h21 = _fma(dec[i], d2, h21)
            d1 = t[i] - p
            g1 = _fma(dec[i], d1, g1)
            g2 += d1
        if abs(g1) < eps and abs(g2) < eps:
            break
        det = _fma(h11, h22, -(h21 * h21))
        dA = -_fma(h22, g1, -(h21 * g2)) / det
        dB = -_fma(-h21, g1, h11 * g2) / det
        gd = _fma(g1, dA, g2 * dB)
        stepsize = 1.0
        while stepsize >= min_step:
            newA = _fma(stepsize, dA, A)
            newB = _fma(stepsize, dB, B)
            newf = 0.0
            for i in range(l):
                fApB = _fma(dec[i], newA, newB)
                if fApB >= 0:
                    newf += _fma(t[i], fApB, math.log(1 + math.exp(-fApB)))
                else:
                    newf += _fma(t[i] - 1, fApB, math.log(1 + math.exp(fApB)))
            if newf < _fma(0.0001 * stepsize, gd, fval):
                A = newA; B = newB; fval = newf
                break
            else:
                stepsize = stepsize / 2.0
        if stepsize < min_step:
            break
    return A, B

def sigmoid_predict(d: float, A: float, B: float) -> float:
    fApB = _fma(d, A, B)
    if fApB >= 0:
        return math.exp(-fApB) / (1.0 + math.exp(-fApB))
    else:
        return 1.0 / (1 + math.exp(fApB))

def multiclass_probability(k: int, r: List[List[float]]) -> List[float]:
    p = [1.0 / k] * k
    Q = [[0.0] * k for _ in range(k)]
    Qp = [0.0] * k
    max_iter = max(100, k)
    eps = 0.005 / k
    for t in range(k):
        Q[t][t] = 0.0
        for j in range(t):
            Q[t][t] = _fma(r[j][t], r[j][t], Q[t][t])
            Q[t][j] = Q[j][t]
        for j in range(t + 1, k):
            Q[t][t] = _fma(r[j][t], r[j][t], Q[t][t])
            Q[t][j] = -r[j][t] * r[t][j]
    for it in range(max_iter):
        pQp = 0.0
        for t in range(k):
            Qp[t] = 0.0
            for j in range(k):
                Qp[t] = _fma(Q[t][j], p[j], Qp[t])
            pQp = _fma(p[t], Qp[t], pQp)
        max_error = 0.0
        for t in range(k):
            e = abs(Qp[t] - pQp)
            if e > max_error:
                max_error = e
        if max_error < eps:
            break
        for t in range(k):
            diff = (-Qp[t] + pQp) / Q[t][t]
            p[t] += diff
            pQp = (pQp + diff * _fma(diff, Q[t][t], 2 * Qp[t])) / (1 + diff) / (1 + diff)
            for j in range(k):
                Qp[j] = _fma(diff, Q[t][j], Qp[j]) / (1 + diff)
                p[j] /= (1 + diff)
    return p

def predict_values(d: _Data, m: Model, par: _Param, x: Ptr[float], xrow: int) -> Tuple[float, List[float]]:
    """libsvm predict_values on a model whose SVs are rows of d (xrow: row
    index of x when x is itself a training row, for precomputed kernels)."""
    total = len(m.sv_rows)
    p = d.p
    kv = [0.0] * total
    for k in range(total):
        if par.kernel == "precomputed":
            kv[k] = x[m.sv_rows[k]]
        else:
            kv[k] = k_function(par, x, d.X + m.sv_rows[k] * p, p)
    st = par.svm_type
    if st == ONE_CLASS or st == EPSILON_SVR or st == NU_SVR:
        s = 0.0
        for k in range(total):
            s = _fma(m.sv_coef[0][k], kv[k], s)
        s -= m.rho[0]
        if st == ONE_CLASS:
            return (1.0 if s > 0 else -1.0), [s]
        return s, [s]
    nc = m.nr_class
    start = [0] * nc
    for c in range(1, nc):
        start[c] = start[c - 1] + m.nSV[c - 1]
    vote = [0] * nc
    out = List[float]()
    pp = 0
    for i in range(nc):
        for j in range(i + 1, nc):
            s = 0.0
            si = start[i]; sj = start[j]
            coef1 = m.sv_coef[j - 1]
            coef2 = m.sv_coef[i]
            for k in range(m.nSV[i]):
                s = _fma(coef1[si + k], kv[si + k], s)
            for k in range(m.nSV[j]):
                s = _fma(coef2[sj + k], kv[sj + k], s)
            s -= m.rho[pp]
            out.append(s)
            if s > 0:
                vote[i] += 1
            else:
                vote[j] += 1
            pp += 1
    best = 0
    for i in range(1, nc):
        if vote[i] > vote[best]:
            best = i
    return float(m.label[best]), out

def _group_classes(prob: _Prob):
    labels = List[int]()
    count = List[int]()
    for v in prob.y:
        lab = int(v)
        found = False
        for j in range(len(labels)):
            if labels[j] == lab:
                count[j] += 1
                found = True
                break
        if not found:
            labels.append(lab)
            count.append(1)
    nc = len(labels)
    for j in range(1, nc):
        i = j - 1
        tl = labels[j]; tc = count[j]
        while i >= 0 and labels[i] > tl:
            labels[i + 1] = labels[i]
            count[i + 1] = count[i]
            i -= 1
        labels[i + 1] = tl
        count[i + 1] = tc
    l = len(prob.y)
    start = [0] * nc
    for c in range(1, nc):
        start[c] = start[c - 1] + count[c - 1]
    perm = [0] * l
    pos = start.copy()
    for i in range(l):
        c = labels.index(int(prob.y[i]))
        perm[pos[c]] = i
        pos[c] += 1
    return labels, count, start, perm

class Trainer:
    """svm_train with sklearn's newrand stream."""
    d: _Data
    rng: _StdMT

    def __init__(self, d: _Data):
        self.d = d
        self.rng = _StdMT(u32(0))

    def train(self, prob0: _Prob, par: _Param, wl: List[int], wv: List[float], seed: int) -> Model:
        keep = [i for i in range(len(prob0.rows)) if prob0.W[i] > 0]
        prob = _Prob([prob0.rows[i] for i in keep], [prob0.y[i] for i in keep], [prob0.W[i] for i in keep])
        if seed >= 0:
            self.rng = _StdMT(u32(seed))
        m = Model()
        st = par.svm_type
        l = len(prob.rows)
        if st == ONE_CLASS or st == EPSILON_SVR or st == NU_SVR:
            m.nr_class = 2
            if par.probability and st != ONE_CLASS:
                m.probA = [self._svr_probability(prob, par)]
            alpha, rho, it = _train_one(self.d, prob, par, 0.0, 0.0)
            m.rho = [rho]
            m.n_iter = [it]
            coef = List[float]()
            for i in range(l):
                if abs(alpha[i]) > 0:
                    m.sv_rows.append(prob.rows[i])
                    m.sv_ind.append(i)
                    coef.append(alpha[i])
            m.sv_coef = [coef]
            m.nSV = [len(coef), len(coef)]
            return m
        labels, count, start, perm = _group_classes(prob)
        nc = len(labels)
        wC = [par.C] * nc
        for t in range(len(wl)):
            for j in range(nc):
                if wl[t] == labels[j]:
                    wC[j] *= wv[t]
                    break
        xr = [prob.rows[perm[i]] for i in range(l)]
        W = [prob.W[perm[i]] for i in range(l)]
        nonzero = [False] * l
        alphas = List[List[float]]()
        for i in range(nc):
            for j in range(i + 1, nc):
                si = start[i]; sj = start[j]
                ci = count[i]; cj = count[j]
                sub = _Prob(xr[si:si + ci] + xr[sj:sj + cj], [1.0] * ci + [-1.0] * cj,
                            W[si:si + ci] + W[sj:sj + cj])
                if par.probability:
                    A, B = self._binary_probability(sub, par, wC[i], wC[j], seed)
                    m.probA.append(A); m.probB.append(B)
                alpha, rho, it = _train_one(self.d, sub, par, wC[i], wC[j])
                for k in range(ci):
                    if abs(alpha[k]) > 0:
                        nonzero[si + k] = True
                for k in range(cj):
                    if abs(alpha[ci + k]) > 0:
                        nonzero[sj + k] = True
                alphas.append(alpha)
                m.rho.append(rho)
                m.n_iter.append(it)
        m.nr_class = nc
        m.label = labels
        m.nSV = [0] * nc
        for c in range(nc):
            for k in range(count[c]):
                if nonzero[start[c] + k]:
                    m.nSV[c] += 1
        total = 0
        for v in m.nSV:
            total += v
        for i in range(l):
            if nonzero[i]:
                m.sv_rows.append(xr[i])
                m.sv_ind.append(perm[i])
        nz_start = [0] * nc
        for c in range(1, nc):
            nz_start[c] = nz_start[c - 1] + m.nSV[c - 1]
        m.sv_coef = [[0.0] * total for _ in range(nc - 1)]
        pp = 0
        for i in range(nc):
            for j in range(i + 1, nc):
                si = start[i]; sj = start[j]
                ci = count[i]; cj = count[j]
                q = nz_start[i]
                for k in range(ci):
                    if nonzero[si + k]:
                        m.sv_coef[j - 1][q] = alphas[pp][k]
                        q += 1
                q = nz_start[j]
                for k in range(cj):
                    if nonzero[sj + k]:
                        m.sv_coef[i][q] = alphas[pp][ci + k]
                        q += 1
                pp += 1
        return m

    def _xptr(self, row: int) -> Ptr[float]:
        return self.d.X + row * self.d.p

    def _binary_probability(self, prob: _Prob, par: _Param, Cp: float, Cn: float, seed: int):
        l = len(prob.rows)
        nr_fold = 5
        perm = list(range(l))
        for i in range(l):
            j = i + self.rng.rint(l - i)
            perm[i], perm[j] = perm[j], perm[i]
        dec = [0.0] * l
        for f in range(nr_fold):
            begin = f * l // nr_fold
            end = (f + 1) * l // nr_fold
            idx = [perm[j] for j in range(begin)] + [perm[j] for j in range(end, l)]
            pc = 0; ncnt = 0
            for j in idx:
                if prob.y[j] > 0:
                    pc += 1
                else:
                    ncnt += 1
            if pc == 0 and ncnt == 0:
                for j in range(begin, end):
                    dec[perm[j]] = 0.0
            elif pc > 0 and ncnt == 0:
                for j in range(begin, end):
                    dec[perm[j]] = 1.0
            elif pc == 0 and ncnt > 0:
                for j in range(begin, end):
                    dec[perm[j]] = -1.0
            else:
                sub = _Prob([prob.rows[j] for j in idx], [prob.y[j] for j in idx], [prob.W[j] for j in idx])
                sp = par.copy()
                sp.probability = False
                sp.C = 1.0
                m = self.train(sub, sp, [1, -1], [Cp, Cn], seed)
                for j in range(begin, end):
                    r = prob.rows[perm[j]]
                    _, dv = predict_values(self.d, m, sp, self._xptr(r), r)
                    dec[perm[j]] = dv[0] * float(m.label[0])
        return _sigmoid_train(dec, prob.y)

    def _svr_probability(self, prob: _Prob, par: _Param) -> float:
        nr_fold = 5
        l = len(prob.rows)
        sp = par.copy()
        sp.probability = False
        # cross_validation (random_seed = -1: no reseed); regression branch
        perm = list(range(l))
        for i in range(l):
            j = i + self.rng.rint(l - i)
            perm[i], perm[j] = perm[j], perm[i]
        ymv = [0.0] * l
        for f in range(nr_fold):
            begin = f * l // nr_fold
            end = (f + 1) * l // nr_fold
            idx = [perm[j] for j in range(begin)] + [perm[j] for j in range(end, l)]
            sub = _Prob([prob.rows[j] for j in idx], [prob.y[j] for j in idx], [prob.W[j] for j in idx])
            m = self.train(sub, sp, List[int](), List[float](), -1)
            for j in range(begin, end):
                r = prob.rows[perm[j]]
                v, _ = predict_values(self.d, m, sp, self._xptr(r), r)
                ymv[perm[j]] = v
        mae = 0.0
        for i in range(l):
            ymv[i] = prob.y[i] - ymv[i]
            mae += abs(ymv[i])
        mae /= float(l)
        std = math.sqrt(2 * mae * mae)
        count = 0
        mae = 0.0
        for i in range(l):
            if abs(ymv[i]) > 5 * std:
                count += 1
            else:
                mae += abs(ymv[i])
        mae /= float(l - count)
        return mae
