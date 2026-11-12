# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Histograms, node splitting and the best-first tree grower
# (sklearn/ensemble/_hist_gradient_boosting/histogram.pyx, splitting.pyx,
# grower.py, predictor.py, _predictor.pyx). Single-threaded: the OpenMP
# loops of sklearn are order-independent per bin / per feature, so results
# are identical.

import math
import heapq
import numpy as np
from numpy.random import Generator, PCG64
from C import qsort(cobj, int, int, cobj)

INF: float = float("inf")

class _SplitInfo:
    gain: float
    feature_idx: int
    bin_idx: int
    missing_go_to_left: bool
    sum_gradient_left: float
    sum_gradient_right: float
    sum_hessian_left: float
    sum_hessian_right: float
    n_samples_left: int
    n_samples_right: int
    value_left: float
    value_right: float
    is_categorical: bool
    left_cat_bitset: List[bool]

    def __init__(self):
        self.gain = -1.0
        self.feature_idx = 0
        self.bin_idx = 0
        self.missing_go_to_left = False
        self.sum_gradient_left = 0.0
        self.sum_gradient_right = 0.0
        self.sum_hessian_left = 0.0
        self.sum_hessian_right = 0.0
        self.n_samples_left = 0
        self.n_samples_right = 0
        self.value_left = 0.0
        self.value_right = 0.0
        self.is_categorical = False
        self.left_cat_bitset = List[bool]()

class _Hist:
    g: List[float]
    h: List[float]
    c: List[int]

    def __init__(self, size: int):
        self.g = [0.0] * size
        self.h = [0.0] * size
        self.c = [0] * size

class TreeNode:
    depth: int
    start: int          # partition_start
    stop: int           # partition_stop
    n_samples: int
    sum_gradients: float
    sum_hessians: float
    value: float
    is_leaf: bool
    allowed_features: List[int]
    has_allowed: bool
    interaction_cst_indices: List[int]
    children_lower_bound: float
    children_upper_bound: float
    split_info: _SplitInfo
    has_split: bool
    left_child: Optional[TreeNode]
    right_child: Optional[TreeNode]
    histograms: Optional[_Hist]

    def __init__(self, depth: int, start: int, stop: int, sum_gradients: float,
                 sum_hessians: float, value: float):
        self.depth = depth
        self.start = start
        self.stop = stop
        self.n_samples = stop - start
        self.sum_gradients = sum_gradients
        self.sum_hessians = sum_hessians
        self.value = value
        self.is_leaf = False
        self.allowed_features = List[int]()
        self.has_allowed = False
        self.interaction_cst_indices = List[int]()
        self.children_lower_bound = -INF
        self.children_upper_bound = INF
        self.split_info = _SplitInfo()
        self.has_split = False
        self.left_child = None
        self.right_child = None
        self.histograms = None

    def __lt__(self, other: TreeNode) -> bool:
        return self.split_info.gain > other.split_info.gain

def compute_node_value(sum_gradient: float, sum_hessian: float, lower_bound: float,
                       upper_bound: float, l2: float) -> float:
    value = -sum_gradient / (sum_hessian + l2 + 1e-15)
    if value < lower_bound:
        value = lower_bound
    elif value > upper_bound:
        value = upper_bound
    return value

def _split_gain(sgl: float, shl: float, sgr: float, shr: float, loss_current: float,
                mcst: int, lower: float, upper: float, l2: float) -> float:
    vl = compute_node_value(sgl, shl, lower, upper, l2)
    vr = compute_node_value(sgr, shr, lower, upper, l2)
    if (mcst == 1 and vl > vr) or (mcst == -1 and vl < vr):
        return -1.0
    gain = loss_current
    gain -= sgl * vl
    gain -= sgr * vr
    return gain

def _cmp_cat(a: cobj, b: cobj) -> i32:
    # compare_cat_infos: -1 if a.value < b.value else 1 (records are
    # (bin_idx, value) pairs of doubles); libc qsort like sklearn
    va = Ptr[float](a)[1]
    vb = Ptr[float](b)[1]
    return i32(-1) if va < vb else i32(1)

def _qsort_cat(bins: List[int], vals: List[float]) -> List[int]:
    n = len(bins)
    buf = Ptr[float](2 * n)
    for i in range(n):
        buf[2 * i] = float(bins[i])
        buf[2 * i + 1] = vals[i]
    f = _cmp_cat
    qsort(buf.as_byte(), n, 16, f.__raw__())
    return [int(buf[2 * i]) for i in range(n)]

class HistogramBuilder:
    X_binned: List[int]       # column-major n * p
    n: int
    n_features: int
    n_bins: int
    gradients: List[float32]
    hessians: List[float32]
    hessians_are_constant: bool

    def __init__(self, X_binned: List[int], n: int, n_features: int, n_bins: int,
                 gradients: List[float32], hessians: List[float32], hessians_are_constant: bool):
        self.X_binned = X_binned
        self.n = n
        self.n_features = n_features
        self.n_bins = n_bins
        self.gradients = gradients
        self.hessians = hessians
        self.hessians_are_constant = hessians_are_constant

    def compute_histograms_brute(self, part: List[int], start: int, stop: int,
                                 features: List[int]) -> _Hist:
        nb = self.n_bins
        hist = _Hist(self.n_features * nb)
        hg = hist.g
        hh = hist.h
        hc = hist.c
        g = self.gradients
        h = self.hessians
        const = self.hessians_are_constant
        n = self.n
        Xb = self.X_binned
        for f in features:
            base = f * n
            off = f * nb
            for pos in range(start, stop):
                i = part[pos]
                b = off + Xb[base + i]
                hg[b] += float(g[i])
                if not const:
                    hh[b] += float(h[i])
                hc[b] += 1
        return hist

    def compute_histograms_subtraction(self, parent: _Hist, sibling: _Hist,
                                       features: List[int]) -> _Hist:
        nb = self.n_bins
        for f in features:
            for b in range(f * nb, f * nb + nb):
                parent.g[b] -= sibling.g[b]
                parent.h[b] -= sibling.h[b]
                parent.c[b] -= sibling.c[b]
        return parent

class Splitter:
    X_binned: List[int]
    n: int
    n_features: int
    n_bins: int
    n_bins_non_missing: List[int]
    missing_values_bin_idx: int
    has_missing_values: List[bool]
    is_categorical: List[bool]
    monotonic_cst: List[int]
    hessians_are_constant: bool
    l2_regularization: float
    min_hessian_to_split: float
    min_samples_leaf: int
    min_gain_to_split: float
    feature_fraction_per_split: float
    rng: Generator[PCG64]
    partition: List[int]

    def __init__(self, X_binned: List[int], n: int, n_features: int, n_bins: int,
                 n_bins_non_missing: List[int], missing_values_bin_idx: int,
                 has_missing_values: List[bool], is_categorical: List[bool],
                 monotonic_cst: List[int], l2_regularization: float,
                 min_hessian_to_split: float, min_samples_leaf: int, min_gain_to_split: float,
                 hessians_are_constant: bool, feature_fraction_per_split: float,
                 rng: Generator[PCG64]):
        self.X_binned = X_binned
        self.n = n
        self.n_features = n_features
        self.n_bins = n_bins
        self.n_bins_non_missing = n_bins_non_missing
        self.missing_values_bin_idx = missing_values_bin_idx
        self.has_missing_values = has_missing_values
        self.is_categorical = is_categorical
        self.monotonic_cst = monotonic_cst
        self.l2_regularization = l2_regularization
        self.min_hessian_to_split = min_hessian_to_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain_to_split = min_gain_to_split
        self.hessians_are_constant = hessians_are_constant
        self.feature_fraction_per_split = feature_fraction_per_split
        self.rng = rng
        self.partition = list(range(n))

    def split_indices(self, si: _SplitInfo, start: int, stop: int) -> int:
        """Stable in-place partition of partition[start:stop]; returns the
        number of samples going left (right_child_position)."""
        f = si.feature_idx
        base = f * self.n
        Xb = self.X_binned
        part = self.partition
        left = List[int](stop - start)
        right = List[int](stop - start)
        mb = self.missing_values_bin_idx
        for pos in range(start, stop):
            i = part[pos]
            b = Xb[base + i]
            go_left = False
            if si.is_categorical:
                go_left = si.left_cat_bitset[b]
            else:
                go_left = (si.missing_go_to_left and b == mb) or b <= si.bin_idx
            if go_left:
                left.append(i)
            else:
                right.append(i)
        p = start
        for i in left:
            part[p] = i
            p += 1
        for i in right:
            part[p] = i
            p += 1
        return len(left)

    def find_node_split(self, n_samples: int, hist: _Hist, sum_gradients: float,
                        sum_hessians: float, value: float, lower_bound: float,
                        upper_bound: float, allowed: List[int], has_allowed: bool) -> _SplitInfo:
        n_allowed = len(allowed) if has_allowed else self.n_features
        mask = List[bool]()
        if self.feature_fraction_per_split < 1.0:
            n_sub = max(1, int(math.ceil(self.feature_fraction_per_split * float(n_allowed))))
            m = np.full(n_allowed, False)
            for t in range(n_sub):
                m[t] = True
            self.rng.shuffle(m)
            mask = [bool(m[t]) for t in range(n_allowed)]
        infos = List[_SplitInfo](n_allowed)
        for idx in range(n_allowed):
            f = allowed[idx] if has_allowed else idx
            si = _SplitInfo()
            si.feature_idx = f
            si.gain = -1.0
            si.is_categorical = self.is_categorical[f]
            infos.append(si)
            if len(mask) > 0 and not mask[idx]:
                continue
            if self.is_categorical[f]:
                self._split_category(f, self.has_missing_values[f], hist, n_samples,
                                     sum_gradients, sum_hessians, value,
                                     self.monotonic_cst[f], lower_bound, upper_bound, si)
            else:
                self._split_l2r(f, self.has_missing_values[f], hist, n_samples,
                                sum_gradients, sum_hessians, value, self.monotonic_cst[f],
                                lower_bound, upper_bound, si)
                if self.has_missing_values[f]:
                    self._split_r2l(f, hist, n_samples, sum_gradients, sum_hessians, value,
                                    self.monotonic_cst[f], lower_bound, upper_bound, si)
        best = 0
        for idx in range(1, n_allowed):
            if infos[idx].gain > infos[best].gain:
                best = idx
        return infos[best]

    def _finish(self, si: _SplitInfo, best_gain: float, best_bin: int, mgl: bool,
                bsgl: float, bshl: float, bnl: int, n_samples: int, sum_gradients: float,
                sum_hessians: float, lower: float, upper: float):
        si.gain = best_gain
        si.bin_idx = best_bin
        si.missing_go_to_left = mgl
        si.sum_gradient_left = bsgl
        si.sum_gradient_right = sum_gradients - bsgl
        si.sum_hessian_left = bshl
        si.sum_hessian_right = sum_hessians - bshl
        si.n_samples_left = bnl
        si.n_samples_right = n_samples - bnl
        l2 = self.l2_regularization
        si.value_left = compute_node_value(si.sum_gradient_left, si.sum_hessian_left, lower, upper, l2)
        si.value_right = compute_node_value(si.sum_gradient_right, si.sum_hessian_right, lower, upper, l2)

    def _split_l2r(self, f: int, has_missing: bool, hist: _Hist, n_samples: int,
                   sum_gradients: float, sum_hessians: float, value: float, mcst: int,
                   lower: float, upper: float, si: _SplitInfo):
        end = self.n_bins_non_missing[f] - 1 + (1 if has_missing else 0)
        sgl = 0.0
        shl = 0.0
        nl = 0
        loss_current = sum_gradients * value
        found = False
        best_gain = -1.0
        best_bin = 0
        bsgl = 0.0
        bshl = 0.0
        bnl = 0
        off = f * self.n_bins
        const = self.hessians_are_constant
        msl = self.min_samples_leaf
        mh = self.min_hessian_to_split
        for b in range(end):
            c = hist.c[off + b]
            nl += c
            nr = n_samples - nl
            if const:
                shl += float(c)
            else:
                shl += hist.h[off + b]
            shr = sum_hessians - shl
            sgl += hist.g[off + b]
            sgr = sum_gradients - sgl
            if nl < msl:
                continue
            if nr < msl:
                break
            if shl < mh:
                continue
            if shr < mh:
                break
            gain = _split_gain(sgl, shl, sgr, shr, loss_current, mcst, lower, upper,
                               self.l2_regularization)
            if gain > best_gain and gain > self.min_gain_to_split:
                found = True
                best_gain = gain
                best_bin = b
                bsgl = sgl
                bshl = shl
                bnl = nl
        if found:
            self._finish(si, best_gain, best_bin, False, bsgl, bshl, bnl, n_samples,
                         sum_gradients, sum_hessians, lower, upper)

    def _split_r2l(self, f: int, hist: _Hist, n_samples: int, sum_gradients: float,
                   sum_hessians: float, value: float, mcst: int, lower: float, upper: float,
                   si: _SplitInfo):
        sgr = 0.0
        shr = 0.0
        nr = 0
        loss_current = sum_gradients * value
        start = self.n_bins_non_missing[f] - 2
        found = False
        best_gain = si.gain
        best_bin = 0
        bsgl = 0.0
        bshl = 0.0
        bnl = 0
        off = f * self.n_bins
        const = self.hessians_are_constant
        msl = self.min_samples_leaf
        mh = self.min_hessian_to_split
        b = start
        while b >= 0:
            c = hist.c[off + b + 1]
            nr += c
            nl = n_samples - nr
            if const:
                shr += float(c)
            else:
                shr += hist.h[off + b + 1]
            shl = sum_hessians - shr
            sgr += hist.g[off + b + 1]
            sgl = sum_gradients - sgr
            b -= 1
            if nr < msl:
                continue
            if nl < msl:
                break
            if shr < mh:
                continue
            if shl < mh:
                break
            gain = _split_gain(sgl, shl, sgr, shr, loss_current, mcst, lower, upper,
                               self.l2_regularization)
            if gain > best_gain and gain > self.min_gain_to_split:
                found = True
                best_gain = gain
                best_bin = b + 1
                bsgl = sgl
                bshl = shl
                bnl = nl
        if found:
            self._finish(si, best_gain, best_bin, True, bsgl, bshl, bnl, n_samples,
                         sum_gradients, sum_hessians, lower, upper)

    def _split_category(self, f: int, has_missing: bool, hist: _Hist, n_samples: int,
                        sum_gradients: float, sum_hessians: float, value: float, mcst: int,
                        lower: float, upper: float, si: _SplitInfo):
        nbnm = self.n_bins_non_missing[f]
        mb = self.missing_values_bin_idx
        off = f * self.n_bins
        const = self.hessians_are_constant
        MIN_CAT_SUPPORT = 10.0
        support_factor = float(n_samples) / sum_hessians
        cbins = List[int]()
        cvals = List[float]()
        for b in range(nbnm):
            shb = float(hist.c[off + b]) if const else hist.h[off + b]
            if shb * support_factor >= MIN_CAT_SUPPORT:
                cbins.append(b)
                cvals.append(hist.g[off + b] / (shb + MIN_CAT_SUPPORT))
        if has_missing:
            shb = float(hist.c[off + mb]) if const else hist.h[off + mb]
            if shb * support_factor >= MIN_CAT_SUPPORT:
                cbins.append(mb)
                cvals.append(hist.g[off + mb] / (shb + MIN_CAT_SUPPORT))
        n_used = len(cbins)
        if n_used <= 1:
            return
        sorted_bins = _qsort_cat(cbins, cvals)
        loss_current = sum_gradients * value
        found = False
        best_gain = -1.0
        bsgl = 0.0
        bshl = 0.0
        bnl = 0
        best_thresh = 0
        best_dir = 0
        msl = self.min_samples_leaf
        mh = self.min_hessian_to_split
        for direction in (1, -1):
            middle = (n_used + 1) // 2 if direction == 1 else (n_used + 1) // 2 - 1
            sgl = 0.0
            shl = 0.0
            nl = 0
            for i in range(middle):
                sidx = i if direction == 1 else n_used - 1 - i
                b = sorted_bins[sidx]
                c = hist.c[off + b]
                nl += c
                nr = n_samples - nl
                if const:
                    shl += float(c)
                else:
                    shl += hist.h[off + b]
                shr = sum_hessians - shl
                sgl += hist.g[off + b]
                sgr = sum_gradients - sgl
                if nl < msl or shl < mh:
                    continue
                if nr < msl or shr < mh:
                    break
                gain = _split_gain(sgl, shl, sgr, shr, loss_current, mcst, lower, upper,
                                   self.l2_regularization)
                if gain > best_gain and gain > self.min_gain_to_split:
                    found = True
                    best_gain = gain
                    best_thresh = sidx
                    bsgl = sgl
                    bshl = shl
                    bnl = nl
                    best_dir = direction
        if found:
            self._finish(si, best_gain, 0, si.missing_go_to_left, bsgl, bshl, bnl, n_samples,
                         sum_gradients, sum_hessians, lower, upper)
            bs = [False] * 256
            if best_dir == 1:
                for s in range(best_thresh + 1):
                    bs[sorted_bins[s]] = True
            else:
                s = n_used - 1
                while s >= best_thresh:
                    bs[sorted_bins[s]] = True
                    s -= 1
            si.left_cat_bitset = bs
            if has_missing:
                si.missing_go_to_left = bs[mb]

class TreeGrower:
    X_binned: List[int]
    n: int
    n_features: int
    max_leaf_nodes: Optional[int]
    max_depth: Optional[int]
    min_samples_leaf: int
    n_bins_non_missing: List[int]
    has_missing_values: List[bool]
    is_categorical: List[bool]
    monotonic_cst: List[int]
    with_monotonic_cst: bool
    interaction_cst: List[List[int]]
    has_interaction_cst: bool
    shrinkage: float
    histogram_builder: HistogramBuilder
    splitter: Splitter
    splittable_nodes: List[TreeNode]
    finalized_leaves: List[TreeNode]
    n_categorical_splits: int
    n_nodes: int
    root: TreeNode
    all_features: List[int]

    def __init__(self, X_binned: List[int], n: int, n_features: int, gradients: List[float32],
                 hessians: List[float32], hessians_are_constant: bool,
                 max_leaf_nodes: Optional[int], max_depth: Optional[int], min_samples_leaf: int,
                 n_bins: int, n_bins_non_missing: List[int], has_missing_values: List[bool],
                 is_categorical: List[bool], monotonic_cst: List[int],
                 interaction_cst: List[List[int]], has_interaction_cst: bool,
                 l2_regularization: float, feature_fraction_per_split: float,
                 rng: Generator[PCG64], shrinkage: float):
        self.X_binned = X_binned
        self.n = n
        self.n_features = n_features
        self.with_monotonic_cst = False
        for v in monotonic_cst:
            if v != 0:
                self.with_monotonic_cst = True
        for f in range(n_features):
            if is_categorical[f] and monotonic_cst[f] != 0:
                raise ValueError("Categorical features cannot have monotonic constraints.")
        self.histogram_builder = HistogramBuilder(X_binned, n, n_features, n_bins, gradients,
                                                  hessians, hessians_are_constant)
        self.splitter = Splitter(X_binned, n, n_features, n_bins, n_bins_non_missing, n_bins - 1,
                                 has_missing_values, is_categorical, monotonic_cst,
                                 l2_regularization, 1e-3, min_samples_leaf, 0.0,
                                 hessians_are_constant, feature_fraction_per_split, rng)
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_bins_non_missing = n_bins_non_missing
        self.has_missing_values = has_missing_values
        self.is_categorical = is_categorical
        self.monotonic_cst = monotonic_cst
        self.interaction_cst = interaction_cst
        self.has_interaction_cst = has_interaction_cst
        self.shrinkage = shrinkage
        self.splittable_nodes = List[TreeNode]()
        self.finalized_leaves = List[TreeNode]()
        self.n_categorical_splits = 0
        self.all_features = list(range(n_features))
        self._initialize_root(hessians, hessians_are_constant)
        self.n_nodes = 1

    def grow(self):
        while len(self.splittable_nodes) > 0:
            self.split_next()
        for leaf in self.finalized_leaves:
            leaf.value *= self.shrinkage

    def _initialize_root(self, hessians: List[float32], const: bool):
        allowed = List[int]()
        arbitrary = 0
        if self.has_interaction_cst:
            seen = [False] * self.n_features
            for grp in self.interaction_cst:
                for f in grp:
                    seen[f] = True
            for f in range(self.n_features):
                if seen[f]:
                    allowed.append(f)
            arbitrary = allowed[0]
        feats = allowed if self.has_interaction_cst else self.all_features
        n = self.n
        hist = self.histogram_builder.compute_histograms_brute(self.splitter.partition, 0, n, feats)
        nb = self.histogram_builder.n_bins
        ga = np.empty(nb)
        ha = np.empty(nb)
        for b in range(nb):
            ga[b] = hist.g[arbitrary * nb + b]
            ha[b] = hist.h[arbitrary * nb + b]
        sum_gradients = float(np.sum(ga))
        sum_hessians = 0.0
        if const:
            sum_hessians = float(hessians[0]) * float(n)
        else:
            sum_hessians = float(np.sum(ha))
        self.root = TreeNode(0, 0, n, sum_gradients, sum_hessians, 0.0)
        if self.root.n_samples < 2 * self.min_samples_leaf:
            self._finalize_leaf(self.root)
            return
        if sum_hessians < self.splitter.min_hessian_to_split:
            self._finalize_leaf(self.root)
            return
        if self.has_interaction_cst:
            self.root.interaction_cst_indices = list(range(len(self.interaction_cst)))
            self.root.allowed_features = allowed
            self.root.has_allowed = True
        self.root.histograms = hist
        self._compute_best_split_and_push(self.root)

    def _compute_best_split_and_push(self, node: TreeNode):
        node.split_info = self.splitter.find_node_split(
            node.n_samples, node.histograms, node.sum_gradients, node.sum_hessians,
            node.value, node.children_lower_bound, node.children_upper_bound,
            node.allowed_features, node.has_allowed)
        node.has_split = True
        if node.split_info.gain <= 0.0:
            self._finalize_leaf(node)
        else:
            heapq.heappush(self.splittable_nodes, node)

    def _finalize_leaf(self, node: TreeNode):
        node.is_leaf = True
        self.finalized_leaves.append(node)

    def _compute_interactions(self, node: TreeNode) -> Tuple[List[int], List[int]]:
        seen = [False] * self.n_features
        idxs = List[int]()
        f0 = node.split_info.feature_idx
        for i in node.interaction_cst_indices:
            if f0 in self.interaction_cst[i]:
                idxs.append(i)
                for f in self.interaction_cst[i]:
                    seen[f] = True
        allowed = [f for f in range(self.n_features) if seen[f]]
        return allowed, idxs

    def split_next(self):
        node = heapq.heappop(self.splittable_nodes)
        si = node.split_info
        right_pos = self.splitter.split_indices(si, node.start, node.stop)
        depth = node.depth + 1
        n_leaf_nodes = len(self.finalized_leaves) + len(self.splittable_nodes) + 2
        left = TreeNode(depth, node.start, node.start + right_pos, si.sum_gradient_left,
                        si.sum_hessian_left, si.value_left)
        right = TreeNode(depth, left.stop, node.stop, si.sum_gradient_right,
                         si.sum_hessian_right, si.value_right)
        node.right_child = right
        node.left_child = left
        if self.has_interaction_cst:
            af, idxs = self._compute_interactions(node)
            left.allowed_features = af
            left.interaction_cst_indices = idxs
            left.has_allowed = True
            right.allowed_features = af
            right.interaction_cst_indices = idxs
            right.has_allowed = True
        if not self.has_missing_values[si.feature_idx]:
            si.missing_go_to_left = left.n_samples > right.n_samples
        self.n_nodes += 2
        if si.is_categorical:
            self.n_categorical_splits += 1
        if self.max_leaf_nodes is not None and n_leaf_nodes == self.max_leaf_nodes:
            self._finalize_leaf(left)
            self._finalize_leaf(right)
            while len(self.splittable_nodes) > 0:
                self._finalize_leaf(self.splittable_nodes.pop())
            node.histograms = None
            return
        if self.max_depth is not None and depth == self.max_depth:
            self._finalize_leaf(left)
            self._finalize_leaf(right)
            node.histograms = None
            return
        if left.n_samples < self.min_samples_leaf * 2:
            self._finalize_leaf(left)
        if right.n_samples < self.min_samples_leaf * 2:
            self._finalize_leaf(right)
        if self.with_monotonic_cst:
            mc = self.monotonic_cst[si.feature_idx]
            lo_l = node.children_lower_bound
            up_l = node.children_upper_bound
            lo_r = node.children_lower_bound
            up_r = node.children_upper_bound
            if mc != 0:
                mid = (left.value + right.value) / 2.0
                if mc == 1:
                    up_l = mid
                    lo_r = mid
                else:
                    lo_l = mid
                    up_r = mid
            left.children_lower_bound = lo_l
            left.children_upper_bound = up_l
            right.children_lower_bound = lo_r
            right.children_upper_bound = up_r
        split_left = not left.is_leaf
        split_right = not right.is_leaf
        if split_left or split_right:
            smallest = left
            largest = right
            if not (left.n_samples < right.n_samples):
                smallest = right
                largest = left
            feats = smallest.allowed_features if smallest.has_allowed else self.all_features
            smallest.histograms = self.histogram_builder.compute_histograms_brute(
                self.splitter.partition, smallest.start, smallest.stop, feats)
            largest.histograms = self.histogram_builder.compute_histograms_subtraction(
                node.histograms, smallest.histograms, feats)
            node.histograms = None
            if split_left:
                self._compute_best_split_and_push(left)
            if split_right:
                self._compute_best_split_and_push(right)
            left.histograms = None if left.is_leaf else left.histograms
            right.histograms = None if right.is_leaf else right.histograms
        node.histograms = None

class TreePredictor:
    # flat node arrays (PREDICTOR_RECORD_DTYPE)
    value: List[float]
    count: List[int]
    feature_idx: List[int]
    num_threshold: List[float]
    missing_go_to_left: List[bool]
    left: List[int]
    right: List[int]
    gain: List[float]
    depth: List[int]
    is_leaf: List[bool]
    bin_threshold: List[int]
    is_categorical: List[bool]
    bitset_idx: List[int]
    binned_left_cat_bitsets: List[List[bool]]
    raw_left_cat_bitsets: List[List[bool]]

    def __init__(self, n_nodes: int):
        self.value = [0.0] * n_nodes
        self.count = [0] * n_nodes
        self.feature_idx = [0] * n_nodes
        self.num_threshold = [0.0] * n_nodes
        self.missing_go_to_left = [False] * n_nodes
        self.left = [0] * n_nodes
        self.right = [0] * n_nodes
        self.gain = [0.0] * n_nodes
        self.depth = [0] * n_nodes
        self.is_leaf = [False] * n_nodes
        self.bin_threshold = [0] * n_nodes
        self.is_categorical = [False] * n_nodes
        self.bitset_idx = [0] * n_nodes
        self.binned_left_cat_bitsets = List[List[bool]]()
        self.raw_left_cat_bitsets = List[List[bool]]()

    def get_n_leaf_nodes(self) -> int:
        c = 0
        for v in self.is_leaf:
            if v:
                c += 1
        return c

    def get_max_depth(self) -> int:
        m = 0
        for d in self.depth:
            if d > m:
                m = d
        return m

    def predict_row(self, X, r: int, known_cat: List[List[bool]]) -> float:
        node = 0
        while True:
            if self.is_leaf[node]:
                return self.value[node]
            v = X[r, self.feature_idx[node]]
            if math.isnan(v):
                node = self.left[node] if self.missing_go_to_left[node] else self.right[node]
            elif self.is_categorical[node]:
                if v < 0.0:
                    node = self.left[node] if self.missing_go_to_left[node] else self.right[node]
                else:
                    c = int(v) & 255   # <X_BINNED_DTYPE_C>data_val
                    if self.raw_left_cat_bitsets[self.bitset_idx[node]][c]:
                        node = self.left[node]
                    elif known_cat[self.feature_idx[node]][c]:
                        node = self.right[node]
                    else:
                        node = self.left[node] if self.missing_go_to_left[node] else self.right[node]
            else:
                if v <= self.num_threshold[node]:
                    node = self.left[node]
                else:
                    node = self.right[node]
        return 0.0

    def predict_binned_row(self, Xb: List[int], n: int, r: int, mb: int) -> float:
        node = 0
        while True:
            if self.is_leaf[node]:
                return self.value[node]
            b = Xb[self.feature_idx[node] * n + r]
            if b == mb:
                node = self.left[node] if self.missing_go_to_left[node] else self.right[node]
            elif self.is_categorical[node]:
                if self.binned_left_cat_bitsets[self.bitset_idx[node]][b]:
                    node = self.left[node]
                else:
                    node = self.right[node]
            else:
                if b <= self.bin_threshold[node]:
                    node = self.left[node]
                else:
                    node = self.right[node]
        return 0.0

    def compute_partial_dependence(self, grid, target_features: List[int], out: List[float]):
        nn = len(self.value)
        stack = [0] * nn
        wstack = [0.0] * nn
        for s in range(grid.shape[0]):
            size = 1
            stack[0] = 0
            wstack[0] = 1.0
            while size > 0:
                size -= 1
                cur = stack[size]
                if self.is_leaf[cur]:
                    out[s] += wstack[size] * self.value[cur]
                else:
                    fi = -1
                    for t in range(len(target_features)):
                        if target_features[t] == self.feature_idx[cur]:
                            fi = t
                            break
                    if fi >= 0:
                        if grid[s, fi] <= self.num_threshold[cur]:
                            stack[size] = self.left[cur]
                        else:
                            stack[size] = self.right[cur]
                        size += 1
                    else:
                        stack[size] = self.left[cur]
                        frac = float(self.count[self.left[cur]]) / float(self.count[cur])
                        cw = wstack[size]
                        wstack[size] = cw * frac
                        size += 1
                        stack[size] = self.right[cur]
                        wstack[size] = cw * (1.0 - frac)
                        size += 1

def make_predictor(grower: TreeGrower, thresholds: List[List[float]]) -> TreePredictor:
    pred = TreePredictor(grower.n_nodes)
    _fill(pred, grower.root, thresholds, grower.n_bins_non_missing, 0)
    return pred

def _fill(pred: TreePredictor, gn: TreeNode, thresholds: List[List[float]],
          nbnm: List[int], idx: int) -> int:
    pred.count[idx] = gn.n_samples
    pred.depth[idx] = gn.depth
    pred.gain[idx] = gn.split_info.gain if gn.has_split else -1.0
    pred.value[idx] = gn.value
    if gn.is_leaf:
        pred.is_leaf[idx] = True
        return idx + 1
    si = gn.split_info
    f = si.feature_idx
    pred.feature_idx[idx] = f
    pred.bin_threshold[idx] = si.bin_idx
    pred.missing_go_to_left[idx] = si.missing_go_to_left
    pred.is_categorical[idx] = si.is_categorical
    if si.bin_idx == nbnm[f] - 1:
        pred.num_threshold[idx] = INF
    elif si.is_categorical:
        cats = thresholds[f]
        pred.bitset_idx[idx] = len(pred.binned_left_cat_bitsets)
        pred.binned_left_cat_bitsets.append(si.left_cat_bitset)
        raw = [False] * 256
        for b in range(len(cats)):
            if si.left_cat_bitset[b]:
                raw[int(cats[b]) & 255] = True
        pred.raw_left_cat_bitsets.append(raw)
    else:
        pred.num_threshold[idx] = thresholds[f][si.bin_idx]
    nxt = idx + 1
    pred.left[idx] = nxt
    nxt = _fill(pred, gn.left_child, thresholds, nbnm, nxt)
    pred.right[idx] = nxt
    return _fill(pred, gn.right_child, thresholds, nbnm, nxt)
