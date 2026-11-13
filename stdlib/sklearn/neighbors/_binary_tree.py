# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# sklearn.neighbors BallTree / KDTree: the BinaryTree of _binary_tree.pxi.tp
# (node layout, splitting with libc++ nth_element ordering, depth/breadth
# first single-tree and dual-tree k-neighbour queries, radius queries,
# kernel density, two-point correlation) with the DistanceMetric kernels of
# metrics/_dist_metrics.pyx.tp for the metrics BallTree/KDTree accept.

import numpy as np
import math
from ._tree_util import (simultaneous_sort_range, heap_push, NodeHeap, nth_element)

def _metric_info(metric: str, p: float):
    # (code, p) ; codes: 0 euclid 1 manhattan 2 chebyshev 3 minkowski
    # 4 hamming 5 canberra 6 braycurtis 7 jaccard 8 dice 9 rogerstanimoto
    # 10 russellrao 11 sokalmichener 12 sokalsneath 13 haversine
    # 14 seuclidean 15 mahalanobis
    if metric == "euclidean" or metric == "l2":
        return 0, 2.0
    if metric == "manhattan" or metric == "cityblock" or metric == "l1":
        return 1, 1.0
    if metric == "chebyshev" or metric == "infinity":
        return 2, float("inf")
    if metric == "minkowski" or metric == "p":
        if p < 1.0:
            raise ValueError("p must be greater than or equal to 1 for BallTree/KDTree")
        return 3, p
    if metric == "hamming":
        return 4, 2.0
    if metric == "canberra":
        return 5, 2.0
    if metric == "braycurtis":
        return 6, 2.0
    if metric == "jaccard":
        return 7, 2.0
    if metric == "dice":
        return 8, 2.0
    if metric == "rogerstanimoto":
        return 9, 2.0
    if metric == "russellrao":
        return 10, 2.0
    if metric == "sokalmichener":
        return 11, 2.0
    if metric == "sokalsneath":
        return 12, 2.0
    if metric == "haversine":
        return 13, 2.0
    if metric == "seuclidean":
        return 14, 2.0
    if metric == "mahalanobis":
        return 15, 2.0
    raise ValueError(f"Unrecognized metric '{metric}'")

def _metric_class_name(code: int) -> str:
    names = ["EuclideanDistance64", "ManhattanDistance64", "ChebyshevDistance64",
             "MinkowskiDistance64", "HammingDistance64", "CanberraDistance64",
             "BrayCurtisDistance64", "JaccardDistance64", "DiceDistance64",
             "RogersTanimotoDistance64", "RussellRaoDistance64",
             "SokalMichenerDistance64", "SokalSneathDistance64",
             "HaversineDistance64", "SEuclideanDistance64", "MahalanobisDistance64"]
    return names[code]

def _cdiv(a: float, b: float) -> float:
    # IEEE division (0/0 = nan, x/0 = inf) as in C
    if b == 0.0:
        if a == 0.0 or a != a:
            return float("nan")
        return float("inf") if a > 0.0 else -float("inf")
    return a / b

def _log_add_exp(x1: float, x2: float) -> float:
    a = max(x1, x2)
    if a == -float("inf"):
        return -float("inf")
    return a + math.log(math.exp(x1 - a) + math.exp(x2 - a))

def _log_sub_exp(x1: float, x2: float) -> float:
    if x1 <= x2:
        return -float("inf")
    return x1 + math.log(1.0 - math.exp(x2 - x1))

def _kernel_id(kernel: str) -> int:
    if kernel == "gaussian":
        return 1
    if kernel == "tophat":
        return 2
    if kernel == "epanechnikov":
        return 3
    if kernel == "exponential":
        return 4
    if kernel == "linear":
        return 5
    if kernel == "cosine":
        return 6
    raise ValueError(f"kernel = '{kernel}' not recognized")

def _log_vn(n: int) -> float:
    return 0.5 * float(n) * math.log(math.pi) - math.lgamma(0.5 * float(n) + 1.0)

def _log_sn(n: int) -> float:
    return math.log(2.0 * math.pi) + _log_vn(n - 1)

def _log_kernel_norm_k(h: float, d: int, kc: int) -> float:
    factor = 0.0
    if kc == 1:
        factor = 0.5 * float(d) * math.log(2.0 * math.pi)
    elif kc == 2:
        factor = _log_vn(d)
    elif kc == 3:
        factor = _log_vn(d) + math.log(2.0 / (float(d) + 2.0))
    elif kc == 4:
        factor = _log_sn(d - 1) + math.lgamma(float(d))
    elif kc == 5:
        factor = _log_vn(d) - math.log(float(d) + 1.0)
    else:
        tmp = 2.0 / math.pi
        k = 1
        while k < d + 1:
            factor += tmp
            tmp *= -float((d - k) * (d - k - 1)) * (2.0 / math.pi) ** 2
            k += 2
        lf = math.log(factor) if factor > 0.0 else float("nan")
        factor = lf + _log_sn(d - 1)
    return -factor - float(d) * math.log(h)

def _clk(dist: float, h: float, kc: int) -> float:
    NEG = -float("inf")
    if kc == 1:
        return -0.5 * (dist * dist) / (h * h)
    if kc == 4:
        return -dist / h
    if kc == 2:
        return 0.0 if dist < h else NEG
    if kc == 3:
        return math.log(1.0 - (dist * dist) / (h * h)) if dist < h else NEG
    if kc == 5:
        return math.log(1.0 - dist / h) if dist < h else NEG
    return math.log(math.cos(0.5 * math.pi * dist / h)) if dist < h else NEG

def _opt_vec(x):
    if isinstance(x, NoneType):
        return np.empty(0), False
    else:
        return np.ascontiguousarray(np.asarray(x, dtype=float).ravel()), True

def _opt_mat(x):
    if isinstance(x, NoneType):
        return np.empty((0, 0)), False
    else:
        A = np.ascontiguousarray(np.asarray(x, dtype=float))
        side = int(math.sqrt(float(A.size)) + 0.5)
        if side * side != A.size:
            return np.empty((0, 0)), False
        return np.ascontiguousarray(A.reshape((side, side))), True

class _BinaryTree:
    kind: int                 # 0 KDTree, 1 BallTree
    data: np.ndarray[float, 2]
    sample_weight: np.ndarray[float, 1]
    has_sw: bool
    sum_weight: float
    idx_array: np.ndarray[int, 1]
    nd_start: np.ndarray[int, 1]
    nd_end: np.ndarray[int, 1]
    nd_leaf: np.ndarray[int, 1]
    nd_radius: np.ndarray[float, 1]
    b0: np.ndarray[float, 2]   # KD: lower bounds ; Ball: centroids
    b1: np.ndarray[float, 2]   # KD: upper bounds
    leaf_size: int
    n_levels: int
    n_nodes: int
    code: int
    pp: float
    metric: str
    vec: np.ndarray[float, 1]
    has_vec: bool
    mat: np.ndarray[float, 2]
    n_trims: int
    n_leaves: int
    n_splits: int
    n_calls: int

    def __init__(self, kind: int, data, leaf_size: int, metric: str, p: float,
                 w_a: np.ndarray[float, 1], has_w: bool, V_a: np.ndarray[float, 1], has_V: bool,
                 VM: np.ndarray[float, 2], has_VM: bool, VI_a: np.ndarray[float, 2], has_VI: bool,
                 sw_a: np.ndarray[float, 1], has_swa: bool):
        X = np.asarray(data, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")
        self.kind = kind
        self.data = np.ascontiguousarray(X)
        if self.data.size == 0:
            raise ValueError("X is an empty array")
        if not np.all(np.isfinite(self.data)):
            raise ValueError("Input contains NaN or infinity.")
        n_samples = self.data.shape[0]
        n_features = self.data.shape[1]
        if leaf_size < 1:
            raise ValueError("leaf_size must be greater than or equal to 1")
        self.leaf_size = leaf_size
        code, pp = _metric_info(metric, float(p))
        self.code = code
        self.pp = pp
        self.metric = metric
        self.vec = np.empty(0)
        self.has_vec = False
        self.mat = np.empty((0, 0))
        if code == 3 and has_w:
            wa = w_a
            if np.any(wa < 0.0):
                raise ValueError("w cannot contain negative weights")
            if wa.shape[0] != n_features:
                raise ValueError(f"MinkowskiDistance: the size of w must match the number of features ({n_features}). Currently len(w)={wa.shape[0]}.")
            self.vec = wa
            self.has_vec = True
        if code == 14:
            if not has_V:
                raise ValueError("SEuclidean dist: size of V does not match")
            va = V_a
            if va.shape[0] != n_features:
                raise ValueError("SEuclidean dist: size of V does not match")
            self.vec = va
            self.has_vec = True
        if code == 15:
            if not has_VI:
                if not has_VM:
                    raise ValueError("Must provide either V or VI for Mahalanobis distance")
                self.mat = np.ascontiguousarray(np.linalg.inv(VM))
            else:
                self.mat = VI_a
            if self.mat.ndim != 2 or self.mat.shape[0] != self.mat.shape[1]:
                raise ValueError("V/VI must be square")
            if self.mat.shape[0] != n_features:
                raise ValueError("Mahalanobis dist: size of V does not match")
        if code == 13 and n_features != 2:
            raise ValueError("Haversine distance only valid in 2 dimensions")
        if kind == 0 and code > 3:
            raise ValueError(f"metric {_metric_class_name(code)} is not valid for KDTree")
        self.n_levels = int(math.log2(max(1.0, float(n_samples - 1) / float(leaf_size))) + 1.0)
        self.n_nodes = int(2.0 ** float(self.n_levels)) - 1
        self.idx_array = np.arange(n_samples)
        self.nd_start = np.zeros(self.n_nodes, dtype=int)
        self.nd_end = np.zeros(self.n_nodes, dtype=int)
        self.nd_leaf = np.zeros(self.n_nodes, dtype=int)
        self.nd_radius = np.zeros(self.n_nodes)
        self.b0 = np.zeros((self.n_nodes, n_features))
        self.b1 = np.zeros((self.n_nodes if kind == 0 else 0, n_features))
        self.has_sw = False
        self.sample_weight = np.empty(0)
        self.sum_weight = float(n_samples)
        if has_swa:
            sw = sw_a
            if sw.shape[0] != n_samples:
                raise ValueError("sample_weight must have the same number of samples as data")
            self.sample_weight = sw
            self.has_sw = True
            self.sum_weight = float(np.sum(sw))
        self.n_trims = 0
        self.n_leaves = 0
        self.n_splits = 0
        self.n_calls = 0
        self._recursive_build(0, 0, n_samples)

    # ---- metric kernels ---------------------------------------------------
    def _rd(self, A: np.ndarray[float, 2], ia: int, B: np.ndarray[float, 2], ib: int) -> float:
        c = self.code
        n = A.shape[1]
        d = 0.0
        if c == 0:
            for j in range(n):
                t = A[ia, j] - B[ib, j]
                d += t * t
            return d
        if c == 1:
            for j in range(n):
                d += abs(A[ia, j] - B[ib, j])
            return d
        if c == 2:
            for j in range(n):
                d = max(d, abs(A[ia, j] - B[ib, j]))
            return d
        if c == 3:
            if self.has_vec:
                for j in range(n):
                    d += self.vec[j] * (abs(A[ia, j] - B[ib, j]) ** self.pp)
            else:
                for j in range(n):
                    d += abs(A[ia, j] - B[ib, j]) ** self.pp
            return d
        if c == 4:
            ne = 0
            for j in range(n):
                if A[ia, j] != B[ib, j]:
                    ne += 1
            return float(ne) / float(n)
        if c == 5:
            for j in range(n):
                den = abs(A[ia, j]) + abs(B[ib, j])
                if den > 0.0:
                    d += abs(A[ia, j] - B[ib, j]) / den
            return d
        if c == 6:
            num = 0.0
            den = 0.0
            for j in range(n):
                num += abs(A[ia, j] - B[ib, j])
                den += abs(A[ia, j]) + abs(B[ib, j])
            if den > 0.0:
                return num / den
            return 0.0
        if c >= 7 and c <= 12:
            ntt = 0
            nneq = 0
            for j in range(n):
                t1 = A[ia, j] != 0.0
                t2 = B[ib, j] != 0.0
                if t1 and t2:
                    ntt += 1
                if t1 != t2:
                    nneq += 1
            if c == 7:
                nnz = ntt + nneq
                if nnz == 0:
                    return 0.0
                return float(nnz - ntt) * 1.0 / float(nnz)
            if c == 8:
                return _cdiv(float(nneq), 2.0 * float(ntt) + float(nneq))
            if c == 9 or c == 11:
                return (2.0 * float(nneq)) / (float(n) + float(nneq))
            if c == 10:
                return float(n - ntt) * 1.0 / float(n)
            return _cdiv(float(nneq), 0.5 * float(ntt) + float(nneq))
        if c == 13:
            s0 = math.sin(0.5 * (A[ia, 0] - B[ib, 0]))
            s1 = math.sin(0.5 * (A[ia, 1] - B[ib, 1]))
            return s0 * s0 + math.cos(A[ia, 0]) * math.cos(B[ib, 0]) * s1 * s1
        if c == 14:
            for j in range(n):
                t = A[ia, j] - B[ib, j]
                d += t * t / self.vec[j]
            return d
        buf = np.empty(n)
        for j in range(n):
            buf[j] = A[ia, j] - B[ib, j]
        for i in range(n):
            tmp = 0.0
            for j in range(n):
                tmp += self.mat[i, j] * buf[j]
            d += tmp * buf[i]
        return d

    def _r2d(self, r: float) -> float:
        c = self.code
        if c == 0 or c == 14 or c == 15:
            return math.sqrt(r)
        if c == 3:
            return r ** (1.0 / self.pp)
        if c == 13:
            return 2.0 * math.asin(math.sqrt(r))
        return r

    def _d2r(self, d: float) -> float:
        c = self.code
        if c == 0 or c == 14 or c == 15:
            return d * d
        if c == 3:
            return d ** self.pp
        if c == 13:
            t = math.sin(0.5 * d)
            return t * t
        return d

    def _rdist(self, A: np.ndarray[float, 2], ia: int, B: np.ndarray[float, 2], ib: int) -> float:
        self.n_calls += 1
        return self._rd(A, ia, B, ib)

    def _dist(self, A: np.ndarray[float, 2], ia: int, B: np.ndarray[float, 2], ib: int) -> float:
        self.n_calls += 1
        return self._r2d(self._rd(A, ia, B, ib))

    def _clone_for(self, Xa: np.ndarray[float, 2]) -> _BinaryTree:
        nov = np.empty(0)
        nom = np.empty((0, 0))
        return _BinaryTree(self.kind, Xa, self.leaf_size, self.metric, self.pp,
                           self.vec, self.code == 3 and self.has_vec,
                           self.vec, self.code == 14,
                           self.mat, self.code == 15,
                           nom, False, nov, False)

    # ---- build ------------------------------------------------------------
    def _init_node(self, i_node: int, idx_start: int, idx_end: int):
        nf = self.data.shape[1]
        if self.kind == 0:
            for j in range(nf):
                self.b0[i_node, j] = float("inf")
                self.b1[i_node, j] = -float("inf")
            for i in range(idx_start, idx_end):
                r = self.idx_array[i]
                for j in range(nf):
                    self.b0[i_node, j] = min(self.b0[i_node, j], self.data[r, j])
                    self.b1[i_node, j] = max(self.b1[i_node, j], self.data[r, j])
            rad = 0.0
            for j in range(nf):
                if self.pp == float("inf"):
                    rad = max(rad, 0.5 * (self.b1[i_node, j] - self.b0[i_node, j]))
                else:
                    rad += (0.5 * abs(self.b1[i_node, j] - self.b0[i_node, j])) ** self.pp
            self.nd_start[i_node] = idx_start
            self.nd_end[i_node] = idx_end
            self.nd_radius[i_node] = rad ** (1.0 / self.pp)
        else:
            npts = idx_end - idx_start
            for j in range(nf):
                self.b0[i_node, j] = 0.0
            if self.has_sw:
                sws = 0.0
                for i in range(idx_start, idx_end):
                    r = self.idx_array[i]
                    sws += self.sample_weight[r]
                    for j in range(nf):
                        self.b0[i_node, j] += self.data[r, j] * self.sample_weight[r]
                for j in range(nf):
                    self.b0[i_node, j] /= sws
            else:
                for i in range(idx_start, idx_end):
                    r = self.idx_array[i]
                    for j in range(nf):
                        self.b0[i_node, j] += self.data[r, j]
                for j in range(nf):
                    self.b0[i_node, j] /= float(npts)
            radius = 0.0
            for i in range(idx_start, idx_end):
                radius = max(radius, self._rdist(self.b0, i_node, self.data, self.idx_array[i]))
            self.nd_radius[i_node] = self._r2d(radius)
            self.nd_start[i_node] = idx_start
            self.nd_end[i_node] = idx_end

    def _recursive_build(self, i_node: int, idx_start: int, idx_end: int):
        n_points = idx_end - idx_start
        n_mid = n_points // 2
        nf = self.data.shape[1]
        self._init_node(i_node, idx_start, idx_end)
        if 2 * i_node + 1 >= self.n_nodes:
            self.nd_leaf[i_node] = 1
        elif idx_end - idx_start < 2:
            self.nd_leaf[i_node] = 1
        else:
            self.nd_leaf[i_node] = 0
            # find_node_split_dim
            j_max = 0
            max_spread = 0.0
            for j in range(nf):
                mx = self.data[self.idx_array[idx_start], j]
                mn = mx
                for i in range(1, n_points):
                    v = self.data[self.idx_array[idx_start + i], j]
                    mx = max(mx, v)
                    mn = min(mn, v)
                spread = mx - mn
                if spread > max_spread:
                    max_spread = spread
                    j_max = j
            nth_element(self.idx_array, self.data, j_max, idx_start, idx_start + n_mid, idx_end)
            self._recursive_build(2 * i_node + 1, idx_start, idx_start + n_mid)
            self._recursive_build(2 * i_node + 2, idx_start + n_mid, idx_end)

    # ---- node bounds ------------------------------------------------------
    def _min_rdist(self, i_node: int, P: np.ndarray[float, 2], ip: int) -> float:
        if self.kind == 0:
            nf = self.data.shape[1]
            rd = 0.0
            if self.pp == float("inf"):
                for j in range(nf):
                    d_lo = self.b0[i_node, j] - P[ip, j]
                    d_hi = P[ip, j] - self.b1[i_node, j]
                    d = (d_lo + abs(d_lo)) + (d_hi + abs(d_hi))
                    rd = max(rd, 0.5 * d)
            else:
                for j in range(nf):
                    d_lo = self.b0[i_node, j] - P[ip, j]
                    d_hi = P[ip, j] - self.b1[i_node, j]
                    d = (d_lo + abs(d_lo)) + (d_hi + abs(d_hi))
                    rd += (0.5 * d) ** self.pp
            return rd
        return self._d2r(self._min_dist(i_node, P, ip))

    def _max_rdist(self, i_node: int, P: np.ndarray[float, 2], ip: int) -> float:
        if self.kind == 0:
            nf = self.data.shape[1]
            rd = 0.0
            if self.pp == float("inf"):
                for j in range(nf):
                    rd = max(rd, abs(P[ip, j] - self.b0[i_node, j]))
                    rd = max(rd, abs(P[ip, j] - self.b1[i_node, j]))
            else:
                for j in range(nf):
                    d_lo = abs(P[ip, j] - self.b0[i_node, j])
                    d_hi = abs(P[ip, j] - self.b1[i_node, j])
                    rd += max(d_lo, d_hi) ** self.pp
            return rd
        return self._d2r(self._max_dist(i_node, P, ip))

    def _min_dist(self, i_node: int, P: np.ndarray[float, 2], ip: int) -> float:
        if self.kind == 0:
            if self.pp == float("inf"):
                return self._min_rdist(i_node, P, ip)
            return self._min_rdist(i_node, P, ip) ** (1.0 / self.pp)
        dp = self._dist(P, ip, self.b0, i_node)
        return max(0.0, dp - self.nd_radius[i_node])

    def _max_dist(self, i_node: int, P: np.ndarray[float, 2], ip: int) -> float:
        if self.kind == 0:
            if self.pp == float("inf"):
                return self._max_rdist(i_node, P, ip)
            return self._max_rdist(i_node, P, ip) ** (1.0 / self.pp)
        dp = self._dist(P, ip, self.b0, i_node)
        return dp + self.nd_radius[i_node]

    def _min_max_dist(self, i_node: int, P: np.ndarray[float, 2], ip: int):
        if self.kind == 0:
            nf = self.data.shape[1]
            mn = 0.0
            mx = 0.0
            if self.pp == float("inf"):
                for j in range(nf):
                    d_lo = self.b0[i_node, j] - P[ip, j]
                    d_hi = P[ip, j] - self.b1[i_node, j]
                    d = (d_lo + abs(d_lo)) + (d_hi + abs(d_hi))
                    mn = max(mn, 0.5 * d)
                    mx = max(mx, abs(d_lo))
                    mx = max(mx, abs(d_hi))
            else:
                for j in range(nf):
                    d_lo = self.b0[i_node, j] - P[ip, j]
                    d_hi = P[ip, j] - self.b1[i_node, j]
                    d = (d_lo + abs(d_lo)) + (d_hi + abs(d_hi))
                    mn += (0.5 * d) ** self.pp
                    mx += max(abs(d_lo), abs(d_hi)) ** self.pp
                mn = mn ** (1.0 / self.pp)
                mx = mx ** (1.0 / self.pp)
            return mn, mx
        dp = self._dist(P, ip, self.b0, i_node)
        rad = self.nd_radius[i_node]
        return max(0.0, dp - rad), dp + rad

    def _min_rdist_dual(self, i1: int, other: _BinaryTree, i2: int) -> float:
        if self.kind == 0:
            nf = self.data.shape[1]
            rd = 0.0
            if self.pp == float("inf"):
                for j in range(nf):
                    d1 = self.b0[i1, j] - other.b1[i2, j]
                    d2 = other.b0[i2, j] - self.b1[i1, j]
                    d = (d1 + abs(d1)) + (d2 + abs(d2))
                    rd = max(rd, 0.5 * d)
            else:
                for j in range(nf):
                    d1 = self.b0[i1, j] - other.b1[i2, j]
                    d2 = other.b0[i2, j] - self.b1[i1, j]
                    d = (d1 + abs(d1)) + (d2 + abs(d2))
                    rd += (0.5 * d) ** self.pp
            return rd
        return self._d2r(self._min_dist_dual(i1, other, i2))

    def _min_dist_dual(self, i1: int, other: _BinaryTree, i2: int) -> float:
        if self.kind == 0:
            return self._r2d(self._min_rdist_dual(i1, other, i2))
        dp = self._dist(other.b0, i2, self.b0, i1)
        return max(0.0, dp - self.nd_radius[i1] - other.nd_radius[i2])

    def _max_dist_dual(self, i1: int, other: _BinaryTree, i2: int) -> float:
        if self.kind == 0:
            nf = self.data.shape[1]
            rd = 0.0
            if self.pp == float("inf"):
                for j in range(nf):
                    rd = max(rd, abs(self.b0[i1, j] - other.b1[i2, j]))
                    rd = max(rd, abs(self.b1[i1, j] - other.b0[i2, j]))
            else:
                for j in range(nf):
                    d1 = abs(self.b0[i1, j] - other.b1[i2, j])
                    d2 = abs(self.b1[i1, j] - other.b0[i2, j])
                    rd += max(d1, d2) ** self.pp
            return self._r2d(rd)
        dp = self._dist(other.b0, i2, self.b0, i1)
        return dp + self.nd_radius[i1] + other.nd_radius[i2]

    # ---- k-neighbour traversals ------------------------------------------
    def _qs_depth(self, i_node: int, P: np.ndarray[float, 2], ip: int,
                  HD: np.ndarray[float, 2], HI: np.ndarray[int, 2], k: int, lb: float):
        if lb > HD[ip, 0]:
            self.n_trims += 1
        elif self.nd_leaf[i_node] == 1:
            self.n_leaves += 1
            for i in range(self.nd_start[i_node], self.nd_end[i_node]):
                dp = self._rdist(P, ip, self.data, self.idx_array[i])
                heap_push(HD, HI, ip, k, dp, self.idx_array[i])
        else:
            self.n_splits += 1
            i1 = 2 * i_node + 1
            i2 = i1 + 1
            lb1 = self._min_rdist(i1, P, ip)
            lb2 = self._min_rdist(i2, P, ip)
            if lb1 <= lb2:
                self._qs_depth(i1, P, ip, HD, HI, k, lb1)
                self._qs_depth(i2, P, ip, HD, HI, k, lb2)
            else:
                self._qs_depth(i2, P, ip, HD, HI, k, lb2)
                self._qs_depth(i1, P, ip, HD, HI, k, lb1)

    def _qs_breadth(self, P: np.ndarray[float, 2], ip: int, HD: np.ndarray[float, 2],
                    HI: np.ndarray[int, 2], k: int, nh: NodeHeap):
        nh.push(self._min_rdist(0, P, ip), 0)
        while nh.n > 0:
            lb, i_node = nh.pop()
            if lb > HD[ip, 0]:
                self.n_trims += 1
            elif self.nd_leaf[i_node] == 1:
                self.n_leaves += 1
                for i in range(self.nd_start[i_node], self.nd_end[i_node]):
                    dp = self._rdist(P, ip, self.data, self.idx_array[i])
                    heap_push(HD, HI, ip, k, dp, self.idx_array[i])
            else:
                self.n_splits += 1
                for i in range(2 * i_node + 1, 2 * i_node + 3):
                    nh.push(self._min_rdist(i, P, ip), i)

    def _push_leaf_pair(self, other: _BinaryTree, i_node1: int, i_node2: int, lb: float,
                        HD: np.ndarray[float, 2], HI: np.ndarray[int, 2], k: int,
                        bounds: np.ndarray[float, 1], init_val: float):
        bounds[i_node2] = init_val
        for i2 in range(other.nd_start[i_node2], other.nd_end[i_node2]):
            i_pt = other.idx_array[i2]
            if HD[i_pt, 0] <= lb:
                continue
            for i1 in range(self.nd_start[i_node1], self.nd_end[i_node1]):
                dp = self._rdist(self.data, self.idx_array[i1], other.data, i_pt)
                heap_push(HD, HI, i_pt, k, dp, self.idx_array[i1])
            bounds[i_node2] = max(bounds[i_node2], HD[i_pt, 0])

    def _qd_depth(self, i_node1: int, other: _BinaryTree, i_node2: int,
                  bounds: np.ndarray[float, 1], HD: np.ndarray[float, 2],
                  HI: np.ndarray[int, 2], k: int, lb: float):
        leaf1 = self.nd_leaf[i_node1] == 1
        leaf2 = other.nd_leaf[i_node2] == 1
        if lb > bounds[i_node2]:
            pass
        elif leaf1 and leaf2:
            self._push_leaf_pair(other, i_node1, i_node2, lb, HD, HI, k, bounds, 0.0)
            while i_node2 > 0:
                ipar = (i_node2 - 1) // 2
                bmax = max(bounds[2 * ipar + 1], bounds[2 * ipar + 2])
                if bmax < bounds[ipar]:
                    bounds[ipar] = bmax
                    i_node2 = ipar
                else:
                    break
        elif leaf1 or ((not leaf2) and other.nd_radius[i_node2] > self.nd_radius[i_node1]):
            lb1 = self._min_rdist_dual(i_node1, other, 2 * i_node2 + 1)
            lb2 = self._min_rdist_dual(i_node1, other, 2 * i_node2 + 2)
            if lb1 < lb2:
                self._qd_depth(i_node1, other, 2 * i_node2 + 1, bounds, HD, HI, k, lb1)
                self._qd_depth(i_node1, other, 2 * i_node2 + 2, bounds, HD, HI, k, lb2)
            else:
                self._qd_depth(i_node1, other, 2 * i_node2 + 2, bounds, HD, HI, k, lb2)
                self._qd_depth(i_node1, other, 2 * i_node2 + 1, bounds, HD, HI, k, lb1)
        else:
            lb1 = self._min_rdist_dual(2 * i_node1 + 1, other, i_node2)
            lb2 = self._min_rdist_dual(2 * i_node1 + 2, other, i_node2)
            if lb1 < lb2:
                self._qd_depth(2 * i_node1 + 1, other, i_node2, bounds, HD, HI, k, lb1)
                self._qd_depth(2 * i_node1 + 2, other, i_node2, bounds, HD, HI, k, lb2)
            else:
                self._qd_depth(2 * i_node1 + 2, other, i_node2, bounds, HD, HI, k, lb2)
                self._qd_depth(2 * i_node1 + 1, other, i_node2, bounds, HD, HI, k, lb1)

    def _qd_breadth(self, other: _BinaryTree, HD: np.ndarray[float, 2],
                    HI: np.ndarray[int, 2], k: int, nh: NodeHeap):
        bounds = np.full(other.n_nodes, float("inf"))
        nn2 = other.n_nodes
        nh.push(self._min_rdist_dual(0, other, 0), 0)
        while nh.n > 0:
            lb, key = nh.pop()
            i_node1 = key // nn2
            i_node2 = key % nn2
            leaf1 = self.nd_leaf[i_node1] == 1
            leaf2 = other.nd_leaf[i_node2] == 1
            if lb > bounds[i_node2]:
                pass
            elif leaf1 and leaf2:
                self._push_leaf_pair(other, i_node1, i_node2, lb, HD, HI, k, bounds, -1.0)
            elif leaf1 or ((not leaf2) and other.nd_radius[i_node2] > self.nd_radius[i_node1]):
                for i2 in range(2 * i_node2 + 1, 2 * i_node2 + 3):
                    nh.push(self._min_rdist_dual(i_node1, other, i2), i_node1 * nn2 + i2)
            else:
                for i1 in range(2 * i_node1 + 1, 2 * i_node1 + 3):
                    nh.push(self._min_rdist_dual(i1, other, i_node2), i1 * nn2 + i_node2)

    # ---- public queries ---------------------------------------------------
    def _check_X(self, X) -> np.ndarray[float, 2]:
        A = np.asarray(X, dtype=float)
        if A.ndim != 2:
            raise ValueError(f"Expected 2D array, got {A.ndim}D array instead")
        if A.shape[1] != self.data.shape[1]:
            raise ValueError("query data dimension must match training data dimension")
        if not np.all(np.isfinite(A)):
            raise ValueError("Input contains NaN or infinity.")
        return np.ascontiguousarray(A)

    def query(self, X, k: int = 1, return_distance: Literal[bool] = True,
              dualtree: bool = False, breadth_first: bool = False,
              sort_results: bool = True):
        Xa = self._check_X(X)
        if self.data.shape[0] < k:
            raise ValueError("k must be less than or equal to the number of training points")
        nq = Xa.shape[0]
        HD = np.full((nq, k), float("inf"))
        HI = np.zeros((nq, k), dtype=int)
        self.n_trims = 0
        self.n_leaves = 0
        self.n_splits = 0
        if dualtree:
            other = self._clone_for(Xa)
            if breadth_first:
                nh = NodeHeap(self.data.shape[0] // self.leaf_size)
                self._qd_breadth(other, HD, HI, k, nh)
            else:
                lb = self._min_rdist_dual(0, other, 0)
                bounds = np.full(other.n_nodes, float("inf"))
                self._qd_depth(0, other, 0, bounds, HD, HI, k, lb)
        else:
            if breadth_first:
                nh = NodeHeap(self.data.shape[0] // self.leaf_size)
                for i in range(nq):
                    self._qs_breadth(Xa, i, HD, HI, k, nh)
            else:
                for i in range(nq):
                    lb = self._min_rdist(0, Xa, i)
                    self._qs_depth(0, Xa, i, HD, HI, k, lb)
        if sort_results:
            for r in range(nq):
                rv = np.ascontiguousarray(HD[r])
                ri = np.ascontiguousarray(HI[r])
                simultaneous_sort_range(rv, ri, 0, k)
                for c in range(k):
                    HD[r, c] = rv[c]
                    HI[r, c] = ri[c]
        if return_distance:
            for r in range(nq):
                for c in range(k):
                    HD[r, c] = self._r2d(HD[r, c])
            return HD, HI
        else:
            return HI

    def _qr_single(self, i_node: int, P: np.ndarray[float, 2], ip: int, r: float,
                   ind: np.ndarray[int, 1], dst: np.ndarray[float, 1], count0: int,
                   count_only: bool, return_distance: bool) -> int:
        count = count0
        mn, mx = self._min_max_dist(i_node, P, ip)
        if mn > r:
            pass
        elif mx <= r:
            if count_only:
                count += self.nd_end[i_node] - self.nd_start[i_node]
            else:
                for i in range(self.nd_start[i_node], self.nd_end[i_node]):
                    if count < 0 or count >= self.data.shape[0]:
                        return -1
                    ind[count] = self.idx_array[i]
                    if return_distance:
                        dst[count] = self._dist(P, ip, self.data, self.idx_array[i])
                    count += 1
        elif self.nd_leaf[i_node] == 1:
            reduced_r = self._d2r(r)
            for i in range(self.nd_start[i_node], self.nd_end[i_node]):
                dp = self._rdist(P, ip, self.data, self.idx_array[i])
                if dp <= reduced_r:
                    if count < 0 or count >= self.data.shape[0]:
                        return -1
                    if not count_only:
                        ind[count] = self.idx_array[i]
                        if return_distance:
                            dst[count] = self._r2d(dp)
                    count += 1
        else:
            count = self._qr_single(2 * i_node + 1, P, ip, r, ind, dst, count, count_only, return_distance)
            count = self._qr_single(2 * i_node + 2, P, ip, r, ind, dst, count, count_only, return_distance)
        return count

    def _radius_array(self, r, nq: int) -> np.ndarray[float, 1]:
        ra = np.ascontiguousarray(np.atleast_1d(np.asarray(r, dtype=float)))
        if ra.ndim != 1:
            raise ValueError("r must be broadcastable to X.shape")
        if ra.shape[0] == 1:
            return np.full(nq, ra[0])
        if ra.shape[0] != nq:
            raise ValueError("r must be broadcastable to X.shape")
        return ra

    def _radius_core(self, X, r, return_distance: bool, count_only: bool, sort_results: bool):
        if count_only and return_distance:
            raise ValueError("count_only and return_distance cannot both be true")
        if sort_results and not return_distance:
            raise ValueError("return_distance must be True if sort_results is True")
        Xa = self._check_X(X)
        nq = Xa.shape[0]
        ra = self._radius_array(r, nq)
        n = self.data.shape[0]
        ind_buf = np.zeros(n, dtype=int)
        dst_buf = np.zeros(n)
        counts = np.zeros(nq, dtype=int)
        inds = List[np.ndarray[int, 1]]()
        dsts = List[np.ndarray[float, 1]]()
        for i in range(nq):
            c = self._qr_single(0, Xa, i, ra[i], ind_buf, dst_buf, 0, count_only, return_distance)
            counts[i] = c
            if count_only:
                continue
            if sort_results:
                simultaneous_sort_range(dst_buf, ind_buf, 0, c)
            inds.append(ind_buf[:c].copy())
            if return_distance:
                dsts.append(dst_buf[:c].copy())
        return counts, inds, dsts

    # ---- kernel density ---------------------------------------------------
    def _total_node_weight(self, i_node: int) -> float:
        N = 0.0
        for i in range(self.nd_start[i_node], self.nd_end[i_node]):
            N += self.sample_weight[self.idx_array[i]]
        return N

    def _kde_depth(self, i_node: int, P: np.ndarray[float, 2], ip: int, kc: int, h: float,
                   log_knorm: float, log_atol: float, log_rtol: float,
                   local_min: float, local_spread: float, g: np.ndarray[float, 1]):
        # g[0] = global_log_min_bound, g[1] = global_log_bound_spread
        n1 = 0.0
        n2 = 0.0
        if self.has_sw:
            n1 = self._total_node_weight(i_node)
            n2 = self.sum_weight
        else:
            n1 = float(self.nd_end[i_node] - self.nd_start[i_node])
            n2 = float(self.data.shape[0])
        if (log_knorm + local_spread - math.log(n1) + math.log(n2)
                <= _log_add_exp(log_atol, log_rtol + log_knorm + local_min)):
            pass
        elif (log_knorm + g[1]
                <= _log_add_exp(log_atol, log_rtol + log_knorm + g[0])):
            pass
        elif self.nd_leaf[i_node] == 1:
            g[0] = _log_sub_exp(g[0], local_min)
            g[1] = _log_sub_exp(g[1], local_spread)
            for i in range(self.nd_start[i_node], self.nd_end[i_node]):
                dp = self._dist(P, ip, self.data, self.idx_array[i])
                ld = _clk(dp, h, kc)
                lw = 0.0
                if self.has_sw:
                    lw = math.log(self.sample_weight[self.idx_array[i]])
                g[0] = _log_add_exp(g[0], ld + lw)
        else:
            i1 = 2 * i_node + 1
            i2 = 2 * i_node + 2
            if self.has_sw:
                n1 = self._total_node_weight(i1)
                n2 = self._total_node_weight(i2)
            else:
                n1 = float(self.nd_end[i1] - self.nd_start[i1])
                n2 = float(self.nd_end[i2] - self.nd_start[i2])
            lb1, ub1 = self._min_max_dist(i1, P, ip)
            c1min = math.log(n1) + _clk(ub1, h, kc)
            c1spr = _log_sub_exp(math.log(n1) + _clk(lb1, h, kc), c1min)
            lb2, ub2 = self._min_max_dist(i2, P, ip)
            c2min = math.log(n2) + _clk(ub2, h, kc)
            c2spr = _log_sub_exp(math.log(n2) + _clk(lb2, h, kc), c2min)
            g[0] = _log_sub_exp(g[0], local_min)
            g[0] = _log_add_exp(g[0], c1min)
            g[0] = _log_add_exp(g[0], c2min)
            g[1] = _log_sub_exp(g[1], local_spread)
            g[1] = _log_add_exp(g[1], c1spr)
            g[1] = _log_add_exp(g[1], c2spr)
            self._kde_depth(i1, P, ip, kc, h, log_knorm, log_atol, log_rtol, c1min, c1spr, g)
            self._kde_depth(i2, P, ip, kc, h, log_knorm, log_atol, log_rtol, c2min, c2spr, g)

    def _kde_breadth(self, P: np.ndarray[float, 2], ip: int, kc: int, h: float,
                     log_knorm: float, log_atol: float, log_rtol: float, nh: NodeHeap,
                     nmin: np.ndarray[float, 1], nspr: np.ndarray[float, 1]) -> float:
        N = self.sum_weight if self.has_sw else float(self.data.shape[0])
        v0 = self._min_dist(0, P, ip)
        nh.push(v0, 0)
        gmin = math.log(N) + _clk(self._max_dist(0, P, ip), h, kc)
        gmax = math.log(N) + _clk(v0, h, kc)
        gspr = _log_sub_exp(gmax, gmin)
        nmin[0] = gmin
        nspr[0] = gspr
        while nh.n > 0:
            v, i_node = nh.pop()
            n1 = 0.0
            if self.has_sw:
                n1 = self._total_node_weight(i_node)
            else:
                n1 = float(self.nd_end[i_node] - self.nd_start[i_node])
            if (log_knorm + nspr[i_node] - math.log(n1) + math.log(N)
                    <= _log_add_exp(log_atol, log_rtol + log_knorm + nmin[i_node])):
                pass
            elif log_knorm + gspr <= _log_add_exp(log_atol, log_rtol + log_knorm + gmin):
                break
            elif self.nd_leaf[i_node] == 1:
                gmin = _log_sub_exp(gmin, nmin[i_node])
                gspr = _log_sub_exp(gspr, nspr[i_node])
                for i in range(self.nd_start[i_node], self.nd_end[i_node]):
                    dp = self._dist(P, ip, self.data, self.idx_array[i])
                    ld = _clk(dp, h, kc)
                    lw = 0.0
                    if self.has_sw:
                        lw = math.log(self.sample_weight[self.idx_array[i]])
                    gmin = _log_add_exp(gmin, ld + lw)
            else:
                i1 = 2 * i_node + 1
                i2 = 2 * i_node + 2
                m1 = 0.0
                m2 = 0.0
                if self.has_sw:
                    m1 = self._total_node_weight(i1)
                    m2 = self._total_node_weight(i2)
                else:
                    m1 = float(self.nd_end[i1] - self.nd_start[i1])
                    m2 = float(self.nd_end[i2] - self.nd_start[i2])
                lb1, ub1 = self._min_max_dist(i1, P, ip)
                lb2, ub2 = self._min_max_dist(i2, P, ip)
                nmin[i1] = math.log(m1) + _clk(ub1, h, kc)
                nspr[i1] = math.log(m1) + _clk(lb1, h, kc)
                nmin[i2] = math.log(m2) + _clk(ub2, h, kc)
                nspr[i2] = math.log(m2) + _clk(lb2, h, kc)
                gmin = _log_sub_exp(gmin, nmin[i_node])
                gmin = _log_add_exp(gmin, nmin[i1])
                gmin = _log_add_exp(gmin, nmin[i2])
                gspr = _log_sub_exp(gspr, nspr[i_node])
                gspr = _log_add_exp(gspr, nspr[i1])
                gspr = _log_add_exp(gspr, nspr[i2])
                nh.push(lb1, i1)
                nh.push(lb2, i2)
        nh.clear()
        return _log_add_exp(gmin, gspr - math.log(2.0))

    def kernel_density(self, X, h: float, kernel: str = "gaussian", atol: float = 0.0,
                       rtol: float = 1e-8, breadth_first: bool = True,
                       return_log: bool = False):
        kc = _kernel_id(kernel)
        nf = self.data.shape[1]
        log_atol = math.log(atol) if atol > 0.0 else -float("inf")
        log_rtol = math.log(rtol) if rtol > 0.0 else -float("inf")
        log_knorm = _log_kernel_norm_k(h, nf, kc)
        Xa = self._check_X(X)
        nq = Xa.shape[0]
        out = np.zeros(nq)
        if breadth_first:
            nh = NodeHeap(self.data.shape[0] // self.leaf_size)
            nmin = np.full(self.n_nodes, -float("inf"))
            nspr = np.zeros(self.n_nodes)
            for i in range(nq):
                out[i] = self._kde_breadth(Xa, i, kc, h, log_knorm, log_atol, log_rtol,
                                           nh, nmin, nspr)
        else:
            g = np.zeros(2)
            for i in range(nq):
                lb, ub = self._min_max_dist(0, Xa, i)
                sw = math.log(self.sum_weight)
                lmin = sw + _clk(ub, h, kc)
                lmax = sw + _clk(lb, h, kc)
                lspr = _log_sub_exp(lmax, lmin)
                g[0] = lmin
                g[1] = lspr
                self._kde_depth(0, Xa, i, kc, h, log_knorm, log_atol, log_rtol, lmin, lspr, g)
                out[i] = _log_add_exp(g[0], g[1] - math.log(2.0))
        for i in range(nq):
            out[i] += log_knorm
        if return_log:
            return out
        return np.exp(out)

    # ---- two-point correlation -------------------------------------------
    def _tp_single(self, i_node: int, P: np.ndarray[float, 2], ip: int, r: np.ndarray[float, 1],
                   count: np.ndarray[int, 1], i_min0: int, i_max0: int):
        i_min = i_min0
        i_max = i_max0
        mn, mx = self._min_max_dist(i_node, P, ip)
        while i_min < i_max:
            if mn > r[i_min]:
                i_min += 1
            else:
                break
        while i_max > i_min:
            npts = self.nd_end[i_node] - self.nd_start[i_node]
            if mx <= r[i_max - 1]:
                count[i_max - 1] += npts
                i_max -= 1
            else:
                break
        if i_min < i_max:
            if self.nd_leaf[i_node] == 1:
                for i in range(self.nd_start[i_node], self.nd_end[i_node]):
                    dp = self._dist(P, ip, self.data, self.idx_array[i])
                    j = i_max - 1
                    while j >= i_min and dp <= r[j]:
                        count[j] += 1
                        j -= 1
            else:
                self._tp_single(2 * i_node + 1, P, ip, r, count, i_min, i_max)
                self._tp_single(2 * i_node + 2, P, ip, r, count, i_min, i_max)

    def _tp_dual(self, i_node1: int, other: _BinaryTree, i_node2: int, r: np.ndarray[float, 1],
                 count: np.ndarray[int, 1], i_min0: int, i_max0: int):
        i_min = i_min0
        i_max = i_max0
        lb = self._min_dist_dual(i_node1, other, i_node2)
        ub = self._max_dist_dual(i_node1, other, i_node2)
        while i_min < i_max:
            if lb > r[i_min]:
                i_min += 1
            else:
                break
        while i_max > i_min:
            npts = ((self.nd_end[i_node1] - self.nd_start[i_node1])
                    * (other.nd_end[i_node2] - other.nd_start[i_node2]))
            if ub <= r[i_max - 1]:
                count[i_max - 1] += npts
                i_max -= 1
            else:
                break
        if i_min < i_max:
            leaf1 = self.nd_leaf[i_node1] == 1
            leaf2 = other.nd_leaf[i_node2] == 1
            if leaf1 and leaf2:
                for i1 in range(self.nd_start[i_node1], self.nd_end[i_node1]):
                    for i2 in range(other.nd_start[i_node2], other.nd_end[i_node2]):
                        dp = self._dist(self.data, self.idx_array[i1], other.data, other.idx_array[i2])
                        j = i_max - 1
                        while j >= i_min and dp <= r[j]:
                            count[j] += 1
                            j -= 1
            elif leaf1:
                for i2 in range(2 * i_node2 + 1, 2 * i_node2 + 3):
                    self._tp_dual(i_node1, other, i2, r, count, i_min, i_max)
            elif leaf2:
                for i1 in range(2 * i_node1 + 1, 2 * i_node1 + 3):
                    self._tp_dual(i1, other, i_node2, r, count, i_min, i_max)
            else:
                for i1 in range(2 * i_node1 + 1, 2 * i_node1 + 3):
                    for i2 in range(2 * i_node2 + 1, 2 * i_node2 + 3):
                        self._tp_dual(i1, other, i2, r, count, i_min, i_max)

    def two_point_correlation(self, X, r, dualtree: bool = False):
        Xa = self._check_X(X)
        ra = np.ascontiguousarray(np.atleast_1d(np.asarray(r, dtype=float)))
        if ra.ndim != 1:
            raise ValueError("r must be a 1-dimensional array")
        rs = np.ascontiguousarray(ra[np.argsort(ra)])
        count = np.zeros(rs.shape[0], dtype=int)
        if dualtree:
            other = self._clone_for(Xa)
            self._tp_dual(0, other, 0, rs, count, 0, rs.shape[0])
        else:
            for i in range(Xa.shape[0]):
                self._tp_single(0, Xa, i, rs, count, 0, rs.shape[0])
        return count

    # ---- introspection ----------------------------------------------------
    def get_tree_stats(self):
        return (self.n_trims, self.n_leaves, self.n_splits)

    def reset_n_calls(self):
        self.n_calls = 0

    def get_n_calls(self) -> int:
        return self.n_calls

    def get_arrays(self):
        """(data, idx_array, node idx_start, node idx_end, node is_leaf,
        node radius, node_bounds as (2 or 1, n_nodes, n_features) list)."""
        return (self.data, self.idx_array, self.nd_start, self.nd_end, self.nd_leaf,
                self.nd_radius, self.b0, self.b1)

class _TreeAPI:
    _t: _BinaryTree

    def __init__(self, t: _BinaryTree):
        self._t = t

    @property
    def data(self):
        return self._t.data

    @property
    def sample_weight(self):
        return self._t.sample_weight

    @property
    def idx_array(self):
        return self._t.idx_array

    def query(self, X, k: int = 1, return_distance: Literal[bool] = True,
              dualtree: bool = False, breadth_first: bool = False, sort_results: bool = True):
        return self._t.query(X, k, return_distance, dualtree, breadth_first, sort_results)

    def query_radius(self, X, r, return_distance: Literal[bool] = False,
                     count_only: Literal[bool] = False, sort_results: bool = False):
        counts, inds, dsts = self._t._radius_core(X, r, return_distance, count_only, sort_results)
        if count_only:
            return counts
        elif return_distance:
            return inds, dsts
        else:
            return inds

    def kernel_density(self, X, h: float, kernel: str = "gaussian", atol: float = 0.0,
                       rtol: float = 1e-8, breadth_first: bool = True, return_log: bool = False):
        return self._t.kernel_density(X, h, kernel, atol, rtol, breadth_first, return_log)

    def two_point_correlation(self, X, r, dualtree: bool = False):
        return self._t.two_point_correlation(X, r, dualtree)

    def get_tree_stats(self):
        return self._t.get_tree_stats()

    def reset_n_calls(self):
        self._t.reset_n_calls()

    def get_n_calls(self) -> int:
        return self._t.get_n_calls()

    def get_arrays(self):
        return self._t.get_arrays()

class KDTree(_TreeAPI):
    """sklearn.neighbors.KDTree (euclidean, manhattan, chebyshev, minkowski)."""

    def __init__(self, data, leaf_size: int = 40, metric: str = "minkowski", *,
                 sample_weight = None, p: float = 2.0, w = None):
        wa, hw = _opt_vec(w)
        sa, hs = _opt_vec(sample_weight)
        _TreeAPI.__init__(self, _BinaryTree(0, data, leaf_size, metric, float(p), wa, hw,
                                            np.empty(0), False, np.empty((0, 0)), False,
                                            np.empty((0, 0)), False, sa, hs))

class BallTree(_TreeAPI):
    """sklearn.neighbors.BallTree (all DistanceMetric metrics except callables)."""

    def __init__(self, data, leaf_size: int = 40, metric: str = "minkowski", *,
                 sample_weight = None, p: float = 2.0, w = None, V = None, VI = None):
        wa, hw = _opt_vec(w)
        sa, hs = _opt_vec(sample_weight)
        va, hv = _opt_vec(V)
        vm, hvm = _opt_mat(V)
        vi, hvi = _opt_mat(VI)
        _TreeAPI.__init__(self, _BinaryTree(1, data, leaf_size, metric, float(p), wa, hw,
                                            va, hv, vm, hvm, vi, hvi, sa, hs))
