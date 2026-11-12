# Ported from scikit-learn 1.9.1 (BSD-3-Clause)
# The SMO solver is a port of sklearn/svm/src/libsvm/svm.cpp (libsvm 3.x,
# BSD-3-Clause, Chih-Chung Chang and Chih-Jen Lin) as vendored by sklearn.

# SVC (C-SVC, one-vs-one), dense input, kernels: rbf / linear / poly / sigmoid.
# The solver follows libsvm step by step (WSS3 working set selection,
# shrinking, float32 Q entries, gradient reconstruction), so the iterate path
# and the eps-stopped solution are those of libsvm. Kernel rows are taken from
# a precomputed l x l matrix instead of libsvm's LRU cache (identical values;
# memory O(l^2)).

import math
import numpy as np
from ..base import _accuracy

_INF = float("inf")
_TAU = 1e-12
_LOWER: Literal[int] = 0
_UPPER: Literal[int] = 1
_FREE: Literal[int] = 2

# sklearn's libsvm is compiled by clang with -ffp-contract=on: `a*b + c` inside
# one expression becomes a fused multiply-add. _fma mirrors those sites so the
# SMO iterate path (and hence the eps-stopped solution) is the same.
@llvm
def _fma(a: float, b: float, c: float) -> float:
    declare double @llvm.fma.f64(double, double, double)
    %0 = call double @llvm.fma.f64(double %a, double %b, double %c)
    ret double %0

def _f32(x: float) -> float:
    return float(float32(x))

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
    # Q(i, j) for current (swapped) positions via the original indices
    K: np.ndarray[float, 2]
    yo: List[int]

    def __init__(self):
        self.l = 0
        self.active_size = 0
        self.y = []
        self.G = []
        self.alpha_status = []
        self.alpha = []
        self.QD = []
        self.eps = 0.0
        self.C = []
        self.p = []
        self.active_set = []
        self.G_bar = []
        self.unshrink = False
        self.K = np.empty((0, 0))
        self.yo = []

    def q(self, i: int, j: int) -> float:
        a = self.active_set[i]
        b = self.active_set[j]
        return _f32(float(self.yo[a] * self.yo[b]) * self.K[a, b])

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
        # Q->swap_index(i,j) swaps QD and the kernel row identity; we keep the
        # identity in active_set.
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

    def do_shrinking(self):
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

    def calculate_rho(self) -> float:
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

    def solve(self, K: np.ndarray[float, 2], yv: List[int], Cv: List[float],
              eps: float, shrinking: bool, max_iter: int):
        l = len(yv)
        self.l = l
        self.K = K
        self.yo = yv
        self.y = yv.copy()
        self.p = [-1.0] * l
        self.alpha = [0.0] * l
        self.C = Cv.copy()
        self.eps = eps
        self.unshrink = False
        self.active_set = list(range(l))
        self.active_size = l
        self.QD = [K[i, i] for i in range(l)]
        self.alpha_status = [0] * l
        for i in range(l):
            self.update_alpha_status(i)
        self.G = [self.p[i] for i in range(l)]
        self.G_bar = [0.0] * l
        # alpha starts at 0 => no initial gradient terms
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

def _farr(v: List[float]):
    a = np.empty(len(v))
    for i in range(len(v)):
        a[i] = v[i]
    return a

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
    # libsvm calls BLAS ddot (scipy's BLAS); use the same routine
    from numpy.linalg.blas import cblas_ddot, fint
    return cblas_ddot(fint(n), a.as_byte(), fint(1), b.as_byte(), fint(1))

class _Model:
    nc: int
    sv_rows: List[int]
    sv_local: List[int]
    nSV: List[int]
    sv_coef: List[List[float]]
    rho: List[float]
    iters: List[int]
    probA: List[float]
    probB: List[float]

    def __init__(self, nc: int, sv_rows: List[int], sv_local: List[int], nSV: List[int],
                 sv_coef: List[List[float]], rho: List[float], iters: List[int],
                 probA: List[float], probB: List[float]):
        self.nc = nc
        self.sv_rows = sv_rows
        self.sv_local = sv_local
        self.nSV = nSV
        self.sv_coef = sv_coef
        self.rho = rho
        self.iters = iters
        self.probA = probA
        self.probB = probB

class _StdMT:
    """std::mt19937 (seed via the standard init) + sklearn's newrand
    bounded_rand_int (Lemire)."""
    mt: List[u32]
    pos: int

    def __init__(self, seed: u32):
        self.mt = [u32(0)] * 624
        self.mt[0] = seed
        for i in range(1, 624):
            prev = self.mt[i - 1]
            self.mt[i] = u32(1812433253) * (prev ^ (prev >> u32(30))) + u32(i)
        self.pos = 624

    def next(self) -> u32:
        if self.pos >= 624:
            mt = self.mt
            for k in range(624):
                y = (mt[k] & u32(0x80000000)) | (mt[(k + 1) % 624] & u32(0x7fffffff))
                v = mt[(k + 397) % 624] ^ (y >> u32(1))
                if y & u32(1):
                    v ^= u32(0x9908b0df)
                mt[k] = v
            self.pos = 0
        y = self.mt[self.pos]
        self.pos += 1
        y ^= (y >> u32(11))
        y ^= (y << u32(7)) & u32(0x9d2c5680)
        y ^= (y << u32(15)) & u32(0xefc60000)
        y ^= (y >> u32(18))
        return y

    def bounded(self, rng: u32) -> u32:
        x = self.next()
        m = u64(x) * u64(rng)
        l = u32(m)
        if l < rng:
            t = u32(0) - rng
            if t >= rng:
                t -= rng
                if t >= rng:
                    t %= rng
            while l < t:
                x = self.next()
                m = u64(x) * u64(rng)
                l = u32(m)
        return u32(m >> u64(32))

def _legacy_randint_imax(seed: int) -> int:
    # np.random.RandomState(seed).randint(np.iinfo("i").max): masked rejection
    from ..utils import _LegacyMT
    g = _LegacyMT(seed)
    rng = u32(0x7fffffff - 1)
    mask = rng
    mask |= mask >> u32(1); mask |= mask >> u32(2); mask |= mask >> u32(4)
    mask |= mask >> u32(8); mask |= mask >> u32(16)
    v = g.genrand_int32() & mask
    while v > rng:
        v = g.genrand_int32() & mask
    return int(v)

def _sigmoid_train(dec: List[float], yv: List[int]):
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

def _sigmoid_predict(d: float, A: float, B: float) -> float:
    fApB = _fma(d, A, B)
    if fApB >= 0:
        return math.exp(-fApB) / (1.0 + math.exp(-fApB))
    else:
        return 1.0 / (1 + math.exp(fApB))

def _multiclass_probability(k: int, r: List[List[float]]) -> List[float]:
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

class SVC:
    C: float
    kernel: str
    degree: int
    gamma: str
    gamma_value: float
    coef0: float
    shrinking: bool
    probability: bool
    tol: float
    cache_size: float
    max_iter: int
    decision_function_shape: str
    break_ties: bool
    random_state: Optional[int]
    classes_: np.ndarray[int, 1]
    support_: np.ndarray[int, 1]
    support_vectors_: np.ndarray[float, 2]
    n_support_: np.ndarray[int, 1]
    dual_coef_: np.ndarray[float, 2]
    intercept_: np.ndarray[float, 1]
    _gamma: float
    n_iter_: np.ndarray[int, 1]
    _rho: List[float]
    _sv_coef: List[List[float]]
    probA_: np.ndarray[float, 1]
    probB_: np.ndarray[float, 1]
    _rng: _StdMT
    _seed: int
    _train_data: Ptr[float]
    _keep: np.ndarray[float, 2]
    _model: _Model

    def __init__(self, *, C: float = 1.0, kernel: str = "rbf", degree: int = 3,
                 gamma = "scale", coef0: float = 0.0, shrinking: bool = True,
                 probability: bool = False, tol: float = 1e-3, cache_size: float = 200.0,
                 class_weight = None, verbose: bool = False, max_iter: int = -1,
                 decision_function_shape: str = "ovr", break_ties: bool = False,
                 random_state: Optional[int] = None):
        if kernel != "rbf" and kernel != "linear" and kernel != "poly" and kernel != "sigmoid":
            raise ValueError(f"SVC kernel '{kernel}' is not supported")
        self.C = C
        self.kernel = kernel
        self.degree = degree
        if isinstance(gamma, str):
            self.gamma = gamma
            self.gamma_value = 0.0
        else:
            self.gamma = ""
            self.gamma_value = float(gamma)
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.random_state = random_state
        self.classes_ = np.empty(0, dtype=int)
        self.support_ = np.empty(0, dtype=int)
        self.support_vectors_ = np.empty((0, 0))
        self.n_support_ = np.empty(0, dtype=int)
        self.dual_coef_ = np.empty((0, 0))
        self.intercept_ = np.empty(0)
        self._gamma = 0.0
        self.n_iter_ = np.empty(0, dtype=int)
        self._rho = []
        self._sv_coef = []
        self.probA_ = np.empty(0)
        self.probB_ = np.empty(0)
        self._rng = _StdMT(u32(5489))
        self._seed = 0
        self._keep = np.empty((0, 0))
        self._train_data = self._keep.data
        self._model = _Model(0, [], [], [], [], [], [], [], [])

    def _k_train(self, a: Ptr[float], b: Ptr[float], sa: float, sb: float, p: int) -> float:
        # Kernel::kernel_* (training path: rbf uses x_square)
        if self.kernel == "rbf":
            return math.exp(-self._gamma * (sa + sb - 2 * _dot(a, b, p)))
        elif self.kernel == "linear":
            return _dot(a, b, p)
        elif self.kernel == "poly":
            return _powi(_fma(self._gamma, _dot(a, b, p), self.coef0), self.degree)
        else:
            return math.tanh(_fma(self._gamma, _dot(a, b, p), self.coef0))

    def _k_func(self, a: Ptr[float], b: Ptr[float], p: int) -> float:
        # Kernel::k_function (prediction path: rbf via the difference vector)
        if self.kernel == "rbf":
            m = Ptr[float](p)
            for k in range(p):
                m[k] = a[k] - b[k]
            return math.exp(-self._gamma * _dot(m, m, p))
        elif self.kernel == "linear":
            return _dot(a, b, p)
        elif self.kernel == "poly":
            return _powi(_fma(self._gamma, _dot(a, b, p), self.coef0), self.degree)
        else:
            return math.tanh(_fma(self._gamma, _dot(a, b, p), self.coef0))

    def _gamma_for(self, Xa) -> float:
        p = Xa.shape[1]
        if self.gamma == "scale":
            v = float(np.var(Xa))
            return 1.0 / (float(p) * v) if v != 0 else 1.0
        elif self.gamma == "auto":
            return 1.0 / float(p)
        else:
            return self.gamma_value

    def _train(self, data: Ptr[float], p: int, xsq: List[float], rows: List[int],
               lab: List[int], nc: int, wC: List[float]) -> _Model:
        """svm_train (C_SVC) on the rows `rows` with class indices `lab`
        (0..nc-1, sorted label order) and per-class weighted C."""
        # svm_train: set_seed(param->random_seed). The nested CV trains of the
        # probability estimate copy the same seed, so they reseed it too.
        self._rng = _StdMT(u32(self._seed))
        n = len(rows)
        count = [0] * nc
        for i in range(n):
            count[lab[i]] += 1
        start = [0] * nc
        for c in range(1, nc):
            start[c] = start[c - 1] + count[c - 1]
        perm = [0] * n
        pos = start.copy()
        for i in range(n):
            perm[pos[lab[i]]] = i
            pos[lab[i]] += 1
        nonzero = [False] * n
        alphas = List[List[float]]()
        rhos = List[float]()
        iters = List[int]()
        probA = List[float]()
        probB = List[float]()
        for i in range(nc):
            for j in range(i + 1, nc):
                si = start[i]; sj = start[j]
                ci = count[i]; cj = count[j]
                l = ci + cj
                idx = [rows[perm[si + k]] for k in range(ci)] + [rows[perm[sj + k]] for k in range(cj)]
                yv = [1] * ci + [-1] * cj
                if self.probability:
                    A, B = self._binary_probability(data, p, xsq, idx, yv, wC[i], wC[j])
                    probA.append(A); probB.append(B)
                alpha, rho, it = self._solve_pair(data, p, xsq, idx, yv, wC[i], wC[j])
                for k in range(ci):
                    if abs(alpha[k]) > 0:
                        nonzero[si + k] = True
                for k in range(cj):
                    if abs(alpha[ci + k]) > 0:
                        nonzero[sj + k] = True
                alphas.append(alpha)
                rhos.append(rho)
                iters.append(it)
        nSV = [0] * nc
        for c in range(nc):
            for k in range(count[c]):
                if nonzero[start[c] + k]:
                    nSV[c] += 1
        total = sum(nSV)
        sv_rows = List[int]()
        sv_local = List[int]()
        for i in range(n):
            if nonzero[i]:
                sv_rows.append(rows[perm[i]])
                sv_local.append(perm[i])
        nz_start = [0] * nc
        for c in range(1, nc):
            nz_start[c] = nz_start[c - 1] + nSV[c - 1]
        sv_coef = [[0.0] * total for _ in range(nc - 1)]
        pp = 0
        for i in range(nc):
            for j in range(i + 1, nc):
                si = start[i]; sj = start[j]
                ci = count[i]; cj = count[j]
                q = nz_start[i]
                for k in range(ci):
                    if nonzero[si + k]:
                        sv_coef[j - 1][q] = alphas[pp][k]
                        q += 1
                q = nz_start[j]
                for k in range(cj):
                    if nonzero[sj + k]:
                        sv_coef[i][q] = alphas[pp][ci + k]
                        q += 1
                pp += 1
        return _Model(nc, sv_rows, sv_local, nSV, sv_coef, rhos, iters, probA, probB)

    def _solve_pair(self, data: Ptr[float], p: int, xsq: List[float], idx: List[int],
                    yv: List[int], Cp: float, Cn: float):
        l = len(idx)
        Cv = [Cp if yv[k] > 0 else Cn for k in range(l)]
        K = np.empty((l, l))
        for a in range(l):
            for b in range(l):
                K[a, b] = self._k_train(data + idx[a] * p, data + idx[b] * p,
                                        xsq[idx[a]], xsq[idx[b]], p)
        s = _Solver()
        alpha, rho, it = s.solve(K, yv, Cv, self.tol, self.shrinking, self.max_iter)
        for k in range(l):
            alpha[k] *= float(yv[k])
        return alpha, rho, it

    def _binary_probability(self, data: Ptr[float], p: int, xsq: List[float],
                            idx: List[int], yv: List[int], Cp: float, Cn: float):
        # svm_binary_svc_probability: 5-fold CV decision values + Platt sigmoid
        l = len(idx)
        nr_fold = 5
        perm = list(range(l))
        for i in range(l):
            j = i + int(self._rng.bounded(u32(l - i)))
            perm[i], perm[j] = perm[j], perm[i]
        dec = [0.0] * l
        for f in range(nr_fold):
            begin = f * l // nr_fold
            end = (f + 1) * l // nr_fold
            sub = [perm[j] for j in range(begin)] + [perm[j] for j in range(end, l)]
            pc = 0; ncnt = 0
            for j in sub:
                if yv[j] > 0:
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
                # subparam: C=1, weights +1 -> Cp, -1 -> Cn, probability off.
                # svm_train groups labels in sorted order: -1 first, then +1.
                rows = [idx[j] for j in sub]
                lab = [0 if yv[j] < 0 else 1 for j in sub]
                saved = self.probability
                self.probability = False
                m = self._train(data, p, xsq, rows, lab, 2, [Cn, Cp])
                self.probability = saved
                # predict_values gives dec for (label[0]=-1 vs label[1]=+1);
                # libsvm multiplies by label[0] = -1
                for j in range(begin, end):
                    r = idx[perm[j]]
                    dec[perm[j]] = -self._pair_dec(m, data + r * p, p)[0]
        return _sigmoid_train(dec, yv)

    def fit(self, X, y, sample_weight = None):
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        ya = np.asarray(y, dtype=int)
        n, p = Xa.shape
        classes, yi = np.unique(ya, return_inverse=True)
        nc = classes.shape[0]
        if nc < 2:
            raise ValueError("The number of classes has to be greater than one")
        self.classes_ = classes
        self._gamma = self._gamma_for(Xa)
        # BaseLibSVM.fit: seed = RandomState(random_state).randint(INT_MAX),
        # svm_train calls set_seed(seed) on libsvm's std::mt19937
        if self.random_state is None:
            from time import time
            seed = int(time() * 1e6) & 0x7fffffff
        else:
            seed = _legacy_randint_imax(self.random_state)
        self._seed = seed
        data = Xa.data
        self._train_data = data
        self._keep = Xa
        xsq = [_dot(data + i * p, data + i * p, p) for i in range(n)]
        m = self._train(data, p, xsq, list(range(n)), [int(v) for v in yi], nc, [self.C] * nc)
        self._model = m
        total = len(m.sv_rows)
        self._sv_coef = m.sv_coef
        self._rho = m.rho
        self.support_ = np.array(m.sv_rows)
        sv = np.empty((total, p))
        for a in range(total):
            for b in range(p):
                sv[a, b] = Xa[m.sv_rows[a], b]
        self.support_vectors_ = sv
        self.n_support_ = np.array(m.nSV)
        dc = np.empty((nc - 1, total))
        for a in range(nc - 1):
            for b in range(total):
                dc[a, b] = m.sv_coef[a][b]
        self.intercept_ = -np.array(m.rho)
        self.probA_ = _farr(m.probA)
        self.probB_ = _farr(m.probB)
        if nc == 2:
            # sklearn flips the binary sign (classes_[1] is the positive class)
            dc = -dc
            self.intercept_ = -self.intercept_
        self.dual_coef_ = dc
        self.n_iter_ = np.array(m.iters)
        return self

    def _pair_dec(self, m: _Model, x: Ptr[float], p: int) -> List[float]:
        # predict_values on a model whose SVs are rows of the training data
        nc = m.nc
        total = len(m.sv_rows)
        kv = [0.0] * total
        for k in range(total):
            kv[k] = self._k_func(x, self._train_data + m.sv_rows[k] * p, p)
        start = [0] * nc
        for c in range(1, nc):
            start[c] = start[c - 1] + m.nSV[c - 1]
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
                pp += 1
        return out

    def _dec_values(self, X):
        # libsvm predict_values: one column per class pair (i<j)
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        n, p = Xa.shape
        nc = self.classes_.shape[0]
        total = self.support_vectors_.shape[0]
        svd = self.support_vectors_.data
        xd = Xa.data
        start = [0] * nc
        for c in range(1, nc):
            start[c] = start[c - 1] + int(self.n_support_[c - 1])
        npair = nc * (nc - 1) // 2
        out = np.empty((n, npair))
        kv = [0.0] * total
        for r in range(n):
            for k in range(total):
                kv[k] = self._k_func(xd + r * p, svd + k * p, p)
            pp = 0
            for i in range(nc):
                for j in range(i + 1, nc):
                    s = 0.0
                    si = start[i]; sj = start[j]
                    ci = int(self.n_support_[i]); cj = int(self.n_support_[j])
                    coef1 = self._sv_coef[j - 1]
                    coef2 = self._sv_coef[i]
                    for k in range(ci):
                        s = _fma(coef1[si + k], kv[si + k], s)
                    for k in range(cj):
                        s = _fma(coef2[sj + k], kv[sj + k], s)
                    s -= self._rho[pp]
                    out[r, pp] = s
                    pp += 1
        return out

    def decision_function(self, X):
        d = self._dec_values(X)
        nc = self.classes_.shape[0]
        # DEVIATION: always 2-D (static return type); binary is (n, 1)
        if nc == 2:
            return -d
        if self.decision_function_shape == "ovo":
            return d
        return _ovr_decision_function(d < 0, -d, nc)

    def predict(self, X):
        d = self._dec_values(X)
        nc = self.classes_.shape[0]
        n = d.shape[0]
        out = np.empty(n, dtype=int)
        for r in range(n):
            vote = [0] * nc
            pp = 0
            for i in range(nc):
                for j in range(i + 1, nc):
                    if d[r, pp] > 0:
                        vote[i] += 1
                    else:
                        vote[j] += 1
                    pp += 1
            best = 0
            for i in range(1, nc):
                if vote[i] > vote[best]:
                    best = i
            out[r] = self.classes_[best]
        return out

    def predict_proba(self, X):
        if not self.probability:
            raise AttributeError("predict_proba is not available when probability=False")
        d = self._dec_values(X)
        nc = self.classes_.shape[0]
        n = d.shape[0]
        out = np.empty((n, nc))
        min_prob = 1e-7
        for r in range(n):
            pw = [[0.0] * nc for _ in range(nc)]
            k = 0
            for i in range(nc):
                for j in range(i + 1, nc):
                    v = min(max(_sigmoid_predict(d[r, k], self.probA_[k], self.probB_[k]), min_prob), 1 - min_prob)
                    pw[i][j] = v
                    pw[j][i] = 1 - v
                    k += 1
            pe = _multiclass_probability(nc, pw)
            for i in range(nc):
                out[r, i] = pe[i]
        return out

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def score(self, X, y, sample_weight = None):
        return _accuracy(y, self.predict(X))

def _ovr_decision_function(predictions, confidences, n_classes: int):
    n = predictions.shape[0]
    votes = np.zeros((n, n_classes))
    sum_conf = np.zeros((n, n_classes))
    k = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            for r in range(n):
                sum_conf[r, i] -= confidences[r, k]
                sum_conf[r, j] += confidences[r, k]
                if predictions[r, k]:
                    votes[r, j] += 1
                else:
                    votes[r, i] += 1
            k += 1
    m = np.max(np.abs(sum_conf))
    if m != 0:
        tc = sum_conf / (3 * (np.abs(sum_conf) + 1))
    else:
        tc = sum_conf
    return votes + tc
