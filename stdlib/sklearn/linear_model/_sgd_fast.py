# Ported from scikit-learn 1.9.1 (BSD-3-Clause)
#
# _plain_sgd of sklearn/linear_model/_sgd_fast.pyx.tp with the WeightVector
# (sklearn/utils/_weight_vector.pyx.tp) and the SequentialDataset shuffle
# (our_rand_r Fisher-Yates) for dense float64 data. Multiply-adds are fused
# (float._fma) where clang contracts them in the compiled Cython (macOS arm64).

import math
import numpy as np
from ._sag import _our_rand_r

NO_PENALTY = 0
L1 = 1
L2 = 2
ELASTICNET = 3
CONSTANT = 1
OPTIMAL = 2
INVSCALING = 3
ADAPTIVE = 4
PA1 = 5
PA2 = 6

# loss codes
HINGE = 0
SQUARED_HINGE = 1
LOG = 2
MODIFIED_HUBER = 3
SQUARED = 4
HUBER = 5
EPS_INSENSITIVE = 6
SQ_EPS_INSENSITIVE = 7

def _fma(a: float, b: float, c: float) -> float:
    return float._fma(a, b, c)

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

class SGDLoss:
    kind: int
    param: float

    def __init__(self, kind: int, param: float = 0.0):
        self.kind = kind
        self.param = param

    def loss(self, y: float, p: float) -> float:
        k = self.kind
        if k == HINGE:
            z = p * y
            if z <= self.param:
                return self.param - z
            return 0.0
        elif k == SQUARED_HINGE:
            z = _fma(-p, y, self.param)
            if z > 0.0:
                return z * z
            return 0.0
        elif k == LOG:
            return _fma(-y, p, _log1pexp(p))
        elif k == MODIFIED_HUBER:
            z = p * y
            if z >= 1.0:
                return 0.0
            elif z >= -1.0:
                return (1.0 - z) * (1.0 - z)
            else:
                return -4.0 * z
        elif k == SQUARED:
            return 0.5 * (p - y) * (p - y)
        elif k == HUBER:
            ae = abs(y - p)
            if ae <= self.param:
                return 0.5 * ae * ae
            return self.param * _fma(-0.5, self.param, ae)
        elif k == EPS_INSENSITIVE:
            r = abs(y - p) - self.param
            return r if r > 0.0 else 0.0
        else:
            r = abs(y - p) - self.param
            return r * r if r > 0.0 else 0.0

    def gradient(self, y: float, p: float) -> float:
        k = self.kind
        if k == HINGE:
            z = p * y
            if z <= self.param:
                return -y
            return 0.0
        elif k == SQUARED_HINGE:
            z = _fma(-p, y, self.param)
            if z > 0.0:
                return -2.0 * y * z
            return 0.0
        elif k == LOG:
            if p > -37.0:
                e = math.exp(-p)
                return _fma(-y, e, 1.0 - y) / (1.0 + e)
            return math.exp(p) - y
        elif k == MODIFIED_HUBER:
            z = p * y
            if z >= 1.0:
                return 0.0
            elif z >= -1.0:
                return 2.0 * (1.0 - z) * -y
            else:
                return -4.0 * y
        elif k == SQUARED:
            return p - y
        elif k == HUBER:
            res = p - y
            if abs(res) <= self.param:
                return res
            return self.param if res >= 0.0 else -self.param
        elif k == EPS_INSENSITIVE:
            if y - p > self.param:
                return -1.0
            elif p - y > self.param:
                return 1.0
            return 0.0
        else:
            z = y - p
            if z > self.param:
                return -2.0 * (z - self.param)
            elif z < -self.param:
                return 2.0 * (-z - self.param)
            return 0.0

class _WeightVector:
    w: Ptr[float]
    aw: Ptr[float]
    has_aw: bool
    n: int
    wscale: float
    sq_norm: float
    l1_norm: float
    average_a: float
    average_b: float

    def __init__(self, w: Ptr[float], aw: Ptr[float], has_aw: bool, n: int):
        self.w = w
        self.aw = aw
        self.has_aw = has_aw
        self.n = n
        self.wscale = 1.0
        s = 0.0
        a = 0.0
        for i in range(n):
            s = _fma(w[i], w[i], s)
            a += abs(w[i])
        self.sq_norm = s
        self.l1_norm = a
        self.average_a = 0.0
        self.average_b = 1.0

    def add(self, x: Ptr[float], c: float):
        l2 = 0.0
        l1 = 0.0
        ws = self.wscale
        w = self.w
        cw = c / ws
        for j in range(self.n):
            w[j] = _fma(x[j], cw, w[j])
            l2 = _fma(w[j], w[j], l2)
            l1 += abs(w[j])
        self.sq_norm = l2 * (ws * ws)
        self.l1_norm = l1 * ws

    def add_average(self, x: Ptr[float], c: float, num_iter: float):
        mu = 1.0 / num_iter
        ws = self.wscale
        aw = self.aw
        f = -c / ws
        for j in range(self.n):
            aw[j] = _fma(self.average_a * x[j], f, aw[j])
        if num_iter > 1.0:
            self.average_b /= (1.0 - mu)
        self.average_a = _fma(mu * self.average_b, ws, self.average_a)

    def dot(self, x: Ptr[float]) -> float:
        # clang unrolls by 4 (mul then add, unfused) with a fused tail
        s = 0.0
        w = self.w
        n4 = self.n & ~3
        j = 0
        while j < n4:
            a0 = w[j] * x[j]
            a1 = w[j + 1] * x[j + 1]
            a2 = w[j + 2] * x[j + 2]
            a3 = w[j + 3] * x[j + 3]
            s = (((s + a0) + a1) + a2) + a3
            j += 4
        for k in range(n4, self.n):
            s = _fma(w[k], x[k], s)
        s *= self.wscale
        return s

    def scale(self, c: float):
        self.wscale *= c
        self.sq_norm *= (c * c)
        self.l1_norm *= abs(c)
        if self.wscale < 1e-9:
            self.reset_wscale()

    def reset_wscale(self):
        if self.has_aw:
            for i in range(self.n):
                self.aw[i] = _fma(self.average_a, self.w[i], self.aw[i])
            r = 1.0 / self.average_b
            for i in range(self.n):
                self.aw[i] *= r
            self.average_a = 0.0
            self.average_b = 1.0
        for i in range(self.n):
            self.w[i] *= self.wscale
        self.wscale = 1.0

    def norm(self) -> float:
        return math.sqrt(self.sq_norm)

def _l1penalty(w: _WeightVector, q: Ptr[float], n: int, u: float):
    ws = w.wscale
    wd = w.w
    for idx in range(n):
        z = wd[idx]
        if ws * z > 0.0:
            wd[idx] = max(0.0, wd[idx] - ((u + q[idx]) / ws))
        elif ws * z < 0.0:
            wd[idx] = min(0.0, wd[idx] + ((u - q[idx]) / ws))
        q[idx] = _fma(ws, wd[idx] - z, q[idx])

class SGDResult:
    intercept: float
    average_intercept: float
    n_iter: int
    def __init__(self, intercept: float, average_intercept: float, n_iter: int):
        self.intercept = intercept
        self.average_intercept = average_intercept
        self.n_iter = n_iter

def plain_sgd(weights: np.ndarray[float, 1], intercept: float,
              average_weights: np.ndarray[float, 1], has_average: bool,
              average_intercept: float, loss: SGDLoss, penalty_type: int, alpha: float,
              l1_ratio: float, X: np.ndarray[float, 2], Y: np.ndarray[float, 1],
              SW: np.ndarray[float, 1], validation_mask: np.ndarray[bool, 1],
              early_stopping: bool, score_cb, n_iter_no_change: int, max_iter: int,
              tol: float, fit_intercept: bool, shuffle: bool, seed: int, weight_pos: float,
              weight_neg: float, learning_rate: int, eta0: float, power_t: float,
              one_class: bool, t: float, intercept_decay: float, average: int) -> SGDResult:
    """weights / average_weights are updated in place (contiguous)."""
    n_samples = X.shape[0]
    n_features = weights.shape[0]
    Xc = np.ascontiguousarray(X)
    xp = Xc.data
    w = _WeightVector(weights.data, average_weights.data, has_average, n_features)
    index = Ptr[int](max(n_samples, 1))
    for i in range(n_samples):
        index[i] = i
    current_index = -1
    no_improvement_count = 0
    infinity = False
    eta = 0.0
    sumloss = 0.0
    objective_sum = 0.0
    best_objective = math.inf
    best_score = -math.inf
    score = 0.0
    nval = 0
    for i in range(n_samples):
        if validation_mask[i]:
            nval += 1
    train_count = float(n_samples - nval)
    is_hinge = loss.kind == HINGE
    optimal_init = 0.0
    MAX_DLOSS = 1e12
    q = Ptr[float](max(n_features, 1))
    use_q = penalty_type == L1 or penalty_type == ELASTICNET
    for j in range(n_features):
        q[j] = 0.0
    u = 0.0
    if penalty_type == L2:
        l1_ratio = 0.0
    elif penalty_type == L1:
        l1_ratio = 1.0
    eta = eta0
    if learning_rate == OPTIMAL:
        typw = math.sqrt(1.0 / math.sqrt(alpha))
        initial_eta0 = typw / max(1.0, loss.gradient(1.0, -typw))
        optimal_init = 1.0 / (initial_eta0 * alpha)
    epoch = 0
    seed32 = u32(seed)
    for ep in range(max_iter):
        epoch = ep
        sumloss = 0.0
        objective_sum = 0.0
        if shuffle:
            sd = seed32
            for i in range(n_samples - 1):
                j = i + int(u32(_our_rand_r(__ptr__(sd))) % u32(n_samples - i))
                ti = index[i]
                index[i] = index[j]
                index[j] = ti
        for i in range(n_samples):
            if current_index >= n_samples - 1:
                current_index = -1
            current_index += 1
            sample_index = index[current_index]
            x = xp + sample_index * n_features
            y = Y[sample_index]
            sample_weight = SW[sample_index]
            if validation_mask[sample_index]:
                continue
            p = w.dot(x) + intercept
            if learning_rate == OPTIMAL:
                eta = 1.0 / (alpha * (optimal_init + t - 1.0))
            elif learning_rate == INVSCALING:
                eta = eta0 / math.pow(t, power_t)
            if not early_stopping:
                cur = loss.loss(y, p)
                sumloss += cur
                objective_sum += cur
                if learning_rate != PA1 and learning_rate != PA2:
                    if penalty_type > 0:
                        nn = w.norm()
                        objective_sum = _fma(alpha, _fma((1.0 - l1_ratio) * 0.5, nn * nn,
                                                         l1_ratio * w.l1_norm), objective_sum)
                    if one_class:
                        objective_sum = _fma(intercept, alpha, objective_sum)
            class_weight = weight_pos if y > 0.0 else weight_neg
            update = 0.0
            if learning_rate == PA1:
                sq = 0.0
                for j in range(n_features):
                    sq = _fma(x[j], x[j], sq)
                if sq == 0.0:
                    continue
                update = min(eta0, loss.loss(y, p) / sq)
            elif learning_rate == PA2:
                sq = 0.0
                for j in range(n_features):
                    sq = _fma(x[j], x[j], sq)
                update = loss.loss(y, p) / (sq + 0.5 / eta0)
            else:
                dloss = loss.gradient(y, p)
                if dloss < -MAX_DLOSS:
                    dloss = -MAX_DLOSS
                elif dloss > MAX_DLOSS:
                    dloss = MAX_DLOSS
                update = -eta * dloss
            if learning_rate >= PA1:
                if is_hinge:
                    update *= y
                elif y - p < 0.0:
                    update *= -1.0
            update *= class_weight * sample_weight
            if penalty_type >= L2:
                w.scale(max(0.0, _fma(-((1.0 - l1_ratio) * eta), alpha, 1.0)))
            if update != 0.0:
                w.add(x, update)
            if fit_intercept:
                intercept_update = update
                if one_class:
                    intercept_update -= eta * alpha
                if intercept_update != 0.0:
                    intercept = _fma(intercept_update, intercept_decay, intercept)
            if 0 < average and float(average) <= t:
                w.add_average(x, update, t - float(average) + 1.0)
                average_intercept += ((intercept - average_intercept) / (t - float(average) + 1.0))
            if use_q:
                u = _fma(l1_ratio * eta, alpha, u)
                _l1penalty(w, q, n_features, u)
            t += 1.0
        nonfinite = not math.isfinite(intercept)
        for j in range(n_features):
            if not math.isfinite(weights[j]):
                nonfinite = True
        if nonfinite:
            infinity = True
            break
        if early_stopping:
            score = score_cb(weights, intercept)
            if tol > -math.inf and score < best_score + tol:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            if score > best_score:
                best_score = score
        else:
            if tol > -math.inf and objective_sum / train_count > best_objective - tol:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            if objective_sum / train_count < best_objective:
                best_objective = objective_sum / train_count
        if no_improvement_count >= n_iter_no_change:
            if learning_rate == ADAPTIVE and eta > 1e-6:
                eta = eta / 5.0
                no_improvement_count = 0
            else:
                break
    if infinity:
        raise ValueError(f"Floating-point under-/overflow occurred at epoch #{epoch + 1}. Scaling input data with StandardScaler or MinMaxScaler might help.")
    w.reset_wscale()
    return SGDResult(intercept, average_intercept, epoch + 1)
