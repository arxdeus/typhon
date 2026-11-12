# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Split criteria (sklearn/tree/_criterion.pyx). One class dispatches on
# `kind` instead of the Cython class hierarchy; the arithmetic (order of
# operations included) follows the Cython code so impurities, improvements
# and node values are bit-identical.

import math
from ._utils import WeightedFenwickTree, simultaneous_sort

GINI: Literal[int] = 0
ENTROPY: Literal[int] = 1
MSE: Literal[int] = 2
MAE: Literal[int] = 3
POISSON: Literal[int] = 4

_EPS_POISSON: float = 10.0 * 2.220446049250313e-16
_INF: float = float("inf")

def _log2(x: float) -> float:
    return math.log(x) / math.log(2.0)

@llvm
def _fma(a: float, b: float, c: float) -> float:
    declare double @llvm.fma.f64(double, double, double)
    %0 = call double @llvm.fma.f64(double %a, double %b, double %c)
    ret double %0

def _xlogy(x: float, y: float) -> float:
    if x == 0.0 and not math.isnan(y):
        return 0.0
    return x * math.log(y)

class Criterion:
    kind: int
    n_outputs: int
    n_classes: List[int]
    max_n_classes: int
    n_samples: int
    # data
    y: List[float]              # n x n_outputs, row major
    sample_weight: List[float]
    sample_indices: List[int]
    start: int
    pos: int
    end: int
    n_node_samples: int
    weighted_n_samples: float
    weighted_n_node_samples: float
    weighted_n_left: float
    weighted_n_right: float
    # classification: [k * max_n_classes + c]; regression: [k]
    sum_total: List[float]
    sum_left: List[float]
    sum_right: List[float]
    sq_sum_total: float
    # MAE
    node_medians: List[float]
    left_abs_errors: List[float]
    right_abs_errors: List[float]
    left_medians: List[float]
    right_medians: List[float]
    sorted_y: List[float]
    sorted_indices: List[int]
    ranks: List[int]
    prefix_sum_tree: WeightedFenwickTree

    def __init__(self, kind: int, n_outputs: int, n_classes: List[int], n_samples: int):
        self.kind = kind
        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.start = 0; self.pos = 0; self.end = 0
        self.n_node_samples = 0
        self.weighted_n_samples = 0.0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0
        self.sq_sum_total = 0.0
        self.y = List[float]()
        self.sample_weight = List[float]()
        self.sample_indices = List[int]()
        if kind <= ENTROPY:
            self.n_classes = list(n_classes)
            m = 0
            for c in n_classes:
                if c > m:
                    m = c
            self.max_n_classes = m
            sz = n_outputs * m
        else:
            self.n_classes = [1] * n_outputs
            self.max_n_classes = 1
            sz = n_outputs
        self.sum_total = [0.0] * sz
        self.sum_left = [0.0] * sz
        self.sum_right = [0.0] * sz
        self.node_medians = [0.0] * n_outputs
        if kind == MAE:
            self.left_abs_errors = [0.0] * n_samples
            self.right_abs_errors = [0.0] * n_samples
            self.left_medians = [0.0] * n_samples
            self.right_medians = [0.0] * n_samples
            self.sorted_y = [0.0] * n_samples
            self.sorted_indices = [0] * n_samples
            self.ranks = [0] * n_samples
            self.prefix_sum_tree = WeightedFenwickTree(n_samples)
        else:
            self.left_abs_errors = List[float]()
            self.right_abs_errors = List[float]()
            self.left_medians = List[float]()
            self.right_medians = List[float]()
            self.sorted_y = List[float]()
            self.sorted_indices = List[int]()
            self.ranks = List[int]()
            self.prefix_sum_tree = WeightedFenwickTree(0)

    def init(self, y: List[float], sample_weight: List[float], weighted_n_samples: float,
             sample_indices: List[int], start: int, end: int):
        self.y = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0
        no = self.n_outputs
        if self.kind <= ENTROPY:
            mc = self.max_n_classes
            for k in range(no):
                for c in range(self.n_classes[k]):
                    self.sum_total[k * mc + c] = 0.0
            for p in range(start, end):
                i = sample_indices[p]
                w = sample_weight[i]
                for k in range(no):
                    c = int(y[i * no + k])
                    self.sum_total[k * mc + c] += w
                self.weighted_n_node_samples += w
        elif self.kind == MAE:
            for p in range(start, end):
                self.weighted_n_node_samples += sample_weight[sample_indices[p]]
        else:
            self.sq_sum_total = 0.0
            for k in range(no):
                self.sum_total[k] = 0.0
            for p in range(start, end):
                i = sample_indices[p]
                w = sample_weight[i]
                for k in range(no):
                    y_ik = y[i * no + k]
                    w_y_ik = w * y_ik
                    self.sum_total[k] += w_y_ik
                    self.sq_sum_total = _fma(w_y_ik, y_ik, self.sq_sum_total)
                self.weighted_n_node_samples += w
        self.reset()

    def _move_sums(self, to_left_zero: bool):
        # _move_sums_{classification,regression}
        n = len(self.sum_total)
        if to_left_zero:
            for j in range(n):
                self.sum_left[j] = 0.0
                self.sum_right[j] = self.sum_total[j]
            self.weighted_n_left = 0.0
            self.weighted_n_right = self.weighted_n_node_samples
        else:
            for j in range(n):
                self.sum_right[j] = 0.0
                self.sum_left[j] = self.sum_total[j]
            self.weighted_n_right = 0.0
            self.weighted_n_left = self.weighted_n_node_samples

    def reset(self):
        if self.kind == MAE:
            self._mae_reset()
            return
        self.pos = self.start
        self._move_sums(True)

    def reverse_reset(self):
        self.pos = self.end
        self._move_sums(False)

    def update(self, new_pos: int):
        pos = self.pos
        no = self.n_outputs
        y = self.y
        sw = self.sample_weight
        si = self.sample_indices
        if self.kind == MAE:
            for p in range(pos, new_pos):
                self.weighted_n_left += sw[si[p]]
            self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
            self.pos = new_pos
            return
        cls = self.kind <= ENTROPY
        mc = self.max_n_classes
        if (new_pos - pos) <= (self.end - new_pos):
            for p in range(pos, new_pos):
                i = si[p]
                w = sw[i]
                for k in range(no):
                    if cls:
                        self.sum_left[k * mc + int(y[i * no + k])] += w
                    else:
                        self.sum_left[k] = _fma(w, y[i * no + k], self.sum_left[k])
                self.weighted_n_left += w
        else:
            self.reverse_reset()
            p = self.end - 1
            while p > new_pos - 1:
                i = si[p]
                w = sw[i]
                for k in range(no):
                    if cls:
                        self.sum_left[k * mc + int(y[i * no + k])] -= w
                    else:
                        self.sum_left[k] = _fma(-w, y[i * no + k], self.sum_left[k])
                self.weighted_n_left -= w
                p -= 1
        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        if cls:
            for k in range(no):
                for c in range(self.n_classes[k]):
                    j = k * mc + c
                    self.sum_right[j] = self.sum_total[j] - self.sum_left[j]
        else:
            for k in range(no):
                self.sum_right[k] = self.sum_total[k] - self.sum_left[k]
        self.pos = new_pos

    # ---------------------------------------------------------- impurity

    def node_impurity(self) -> float:
        kind = self.kind
        no = self.n_outputs
        wn = self.weighted_n_node_samples
        if kind == GINI:
            mc = self.max_n_classes
            gini = 0.0
            for k in range(no):
                sq = 0.0
                for c in range(self.n_classes[k]):
                    ck = self.sum_total[k * mc + c]
                    sq += ck * ck
                gini += 1.0 - sq / (wn * wn)
            return gini / float(no)
        elif kind == ENTROPY:
            mc = self.max_n_classes
            ent = 0.0
            for k in range(no):
                for c in range(self.n_classes[k]):
                    ck = self.sum_total[k * mc + c]
                    if ck > 0.0:
                        ck /= wn
                        ent -= ck * _log2(ck)
            return ent / float(no)
        elif kind == MSE:
            imp = self.sq_sum_total / wn
            for k in range(no):
                imp -= (self.sum_total[k] / wn) ** 2.0
            return imp / float(no)
        elif kind == MAE:
            return self.right_abs_errors[0] / (wn * float(no))
        else:
            return self._poisson_loss(self.start, self.end, self.sum_total, wn)

    def children_impurity(self) -> Tuple[float, float]:
        kind = self.kind
        no = self.n_outputs
        wl = self.weighted_n_left
        wr = self.weighted_n_right
        if kind == GINI:
            mc = self.max_n_classes
            gl = 0.0; gr = 0.0
            for k in range(no):
                sl = 0.0; sr = 0.0
                for c in range(self.n_classes[k]):
                    ck = self.sum_left[k * mc + c]
                    sl += ck * ck
                    ck = self.sum_right[k * mc + c]
                    sr += ck * ck
                gl += 1.0 - sl / (wl * wl)
                gr += 1.0 - sr / (wr * wr)
            return (gl / float(no), gr / float(no))
        elif kind == ENTROPY:
            mc = self.max_n_classes
            el = 0.0; er = 0.0
            for k in range(no):
                for c in range(self.n_classes[k]):
                    ck = self.sum_left[k * mc + c]
                    if ck > 0.0:
                        ck /= wl
                        el -= ck * _log2(ck)
                    ck = self.sum_right[k * mc + c]
                    if ck > 0.0:
                        ck /= wr
                        er -= ck * _log2(ck)
            return (el / float(no), er / float(no))
        elif kind == MSE:
            sq_left = 0.0
            for p in range(self.start, self.pos):
                i = self.sample_indices[p]
                w = self.sample_weight[i]
                for k in range(no):
                    y_ik = self.y[i * no + k]
                    sq_left = _fma(w * y_ik, y_ik, sq_left)
            sq_right = self.sq_sum_total - sq_left
            il = sq_left / wl
            ir = sq_right / wr
            for k in range(no):
                il -= (self.sum_left[k] / wl) ** 2.0
                ir -= (self.sum_right[k] / wr) ** 2.0
            return (il / float(no), ir / float(no))
        elif kind == MAE:
            il = 0.0; ir = 0.0
            if self.pos > self.start:
                il += self.left_abs_errors[self.pos - 1]
            if self.pos < self.end:
                ir += self.right_abs_errors[self.pos]
            return (il / (wl * float(no)), ir / (wr * float(no)))
        else:
            return (self._poisson_loss(self.start, self.pos, self.sum_left, wl),
                    self._poisson_loss(self.pos, self.end, self.sum_right, wr))

    def proxy_impurity_improvement(self) -> float:
        no = self.n_outputs
        if self.kind == MSE:
            pl = 0.0; pr = 0.0
            for k in range(no):
                pl += self.sum_left[k] * self.sum_left[k]
                pr += self.sum_right[k] * self.sum_right[k]
            return pl / self.weighted_n_left + pr / self.weighted_n_right
        elif self.kind == POISSON:
            pl = 0.0; pr = 0.0
            for k in range(no):
                if self.sum_left[k] <= _EPS_POISSON or self.sum_right[k] <= _EPS_POISSON:
                    return -_INF
                yl = self.sum_left[k] / self.weighted_n_left
                yr = self.sum_right[k] / self.weighted_n_right
                pl -= self.sum_left[k] * _log2(yl)
                pr -= self.sum_right[k] * _log2(yr)
            return -pl - pr
        il, ir = self.children_impurity()
        # clang contracts (-wr)*ir - wl*il into fma(-wr, ir, -(wl*il))
        return _fma(-self.weighted_n_right, ir, -(self.weighted_n_left * il))

    def impurity_improvement(self, parent: float, il: float, ir: float) -> float:
        wn = self.weighted_n_node_samples
        return ((wn / self.weighted_n_samples) *
                (parent - (self.weighted_n_right / wn * ir)
                        - (self.weighted_n_left / wn * il)))

    def node_value(self, dest: List[float], off: int):
        no = self.n_outputs
        if self.kind <= ENTROPY:
            mc = self.max_n_classes
            for k in range(no):
                for c in range(self.n_classes[k]):
                    dest[off + k * mc + c] = self.sum_total[k * mc + c] / self.weighted_n_node_samples
        elif self.kind == MAE:
            for k in range(no):
                dest[off + k] = self.node_medians[k]
        else:
            for k in range(no):
                dest[off + k] = self.sum_total[k] / self.weighted_n_node_samples

    def middle_value(self) -> float:
        if self.kind == MAE:
            return (self.left_medians[self.pos - 1] + self.right_medians[self.pos]) / 2.0
        return (self.sum_left[0] / (2.0 * self.weighted_n_left) +
                self.sum_right[0] / (2.0 * self.weighted_n_right))

    def check_monotonicity(self, cst: int, lower: float, upper: float) -> bool:
        if self.kind == MAE:
            vl = self.left_medians[self.pos - 1]
            vr = self.right_medians[self.pos]
        else:
            vl = self.sum_left[0] / self.weighted_n_left
            vr = self.sum_right[0] / self.weighted_n_right
        return (vl >= lower and vr >= lower and vl <= upper and vr <= upper and
                (vl - vr) * float(cst) <= 0.0)

    def clip_node_value(self, dest: List[float], off: int, lower: float, upper: float):
        if dest[off] < lower:
            dest[off] = lower
        elif dest[off] > upper:
            dest[off] = upper
        if self.kind <= ENTROPY:
            dest[off + 1] = 1.0 - dest[off]

    # ---------------------------------------------------------- poisson

    def _poisson_loss(self, start: int, end: int, y_sum: List[float], wsum: float) -> float:
        no = self.n_outputs
        loss = 0.0
        for k in range(no):
            if y_sum[k] <= _EPS_POISSON:
                return _INF
            y_mean = y_sum[k] / wsum
            for p in range(start, end):
                i = self.sample_indices[p]
                w = self.sample_weight[i]
                yi = self.y[i * no + k]
                loss += w * _xlogy(yi, yi / y_mean)
        return loss / (wsum * float(no))

    # -------------------------------------------------------------- MAE

    def _precompute_abs_errors(self, start: int, end: int, abs_errors: List[float],
                               medians: List[float]):
        tree = self.prefix_sum_tree
        sorted_y = self.sorted_y
        step = 1
        n = 0
        if start < end:
            n = end - start
        else:
            n = start - end
            step = -1
        tree.reset(n)
        p = start
        for _ in range(n):
            i = self.sample_indices[p]
            w = self.sample_weight[i]
            rank = self.ranks[p]
            tree.add(rank, sorted_y[rank], w)
            half_weight = 0.5 * tree.total_w
            median_rank, w_left, wy_left, median_prev_rank = tree.search(half_weight)
            median = 0.0
            if median_rank != median_prev_rank:
                median = (sorted_y[median_prev_rank] + sorted_y[median_rank]) / 2.0
            else:
                median = sorted_y[median_rank]
            w_right = tree.total_w - w_left
            wy_right = tree.total_wy - wy_left
            medians[p] = median
            abs_errors[p] += _fma(-median, w_right, wy_right) + _fma(median, w_left, -wy_left)
            p += step

    def _mae_reset(self):
        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start
        start = self.start; end = self.end
        no = self.n_outputs
        for p in range(start, end):
            self.left_abs_errors[p] = 0.0
            self.right_abs_errors[p] = 0.0
        n = self.n_node_samples
        for k in range(no):
            for p in range(start, end):
                self.sorted_y[p - start] = self.y[self.sample_indices[p] * no + k]
            for i in range(n):
                self.sorted_indices[i] = i
            simultaneous_sort(self.sorted_y, self.sorted_indices, 0, n, True)
            for i in range(n):
                self.ranks[start + self.sorted_indices[i]] = i
            self._precompute_abs_errors(start, end, self.left_abs_errors, self.left_medians)
            self._precompute_abs_errors(end - 1, start - 1, self.right_abs_errors, self.right_medians)
            self.node_medians[k] = self.right_medians[start]
