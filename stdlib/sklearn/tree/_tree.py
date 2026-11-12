# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Tree structure, depth-first and best-first builders and minimal
# cost-complexity pruning (sklearn/tree/_tree.pyx). Node arrays are stored
# as Python lists while building and exposed as numpy arrays.

import math
import numpy as np
from ._splitter import Splitter, SplitRecord, ParentInfo

TREE_LEAF: Literal[int] = -1
TREE_UNDEFINED: Literal[int] = -2
_EPSILON: float = 2.220446049250313e-16
_INF: float = float("inf")

class Tree:
    n_features: int
    n_outputs: int
    n_classes_list: List[int]
    max_n_classes: int
    value_stride: int
    max_depth: int
    node_count: int
    left: List[int]
    right: List[int]
    feat: List[int]
    thr: List[float]
    imp: List[float]
    nns: List[int]
    wnns: List[float]
    mgl: List[bool]
    val: List[float]

    def __init__(self, n_features: int, n_classes: List[int], n_outputs: int):
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_classes_list = list(n_classes)
        m = 0
        for c in n_classes:
            if c > m:
                m = c
        self.max_n_classes = m
        self.value_stride = n_outputs * m
        self.max_depth = 0
        self.node_count = 0
        self.left = List[int](); self.right = List[int](); self.feat = List[int]()
        self.thr = List[float](); self.imp = List[float](); self.nns = List[int]()
        self.wnns = List[float](); self.mgl = List[bool](); self.val = List[float]()

    def _add_node(self, parent: int, is_left: bool, is_leaf: bool, feature: int,
                  threshold: float, impurity: float, n_node_samples: int,
                  wn: float, missing_go_to_left: bool) -> int:
        node_id = self.node_count
        self.imp.append(impurity)
        self.nns.append(n_node_samples)
        self.wnns.append(wn)
        if parent != TREE_UNDEFINED:
            if is_left:
                self.left[parent] = node_id
            else:
                self.right[parent] = node_id
        if is_leaf:
            self.left.append(TREE_LEAF); self.right.append(TREE_LEAF)
            self.feat.append(TREE_UNDEFINED); self.thr.append(float(TREE_UNDEFINED))
            self.mgl.append(False)
        else:
            # children are filled in later
            self.left.append(0); self.right.append(0)
            self.feat.append(feature); self.thr.append(threshold)
            self.mgl.append(missing_go_to_left)
        for _ in range(self.value_stride):
            self.val.append(0.0)
        self.node_count += 1
        return node_id

    # ------------------------------------------------------ properties

    @property
    def children_left(self):
        return np.array(self.left)

    @property
    def children_right(self):
        return np.array(self.right)

    @property
    def feature(self):
        return np.array(self.feat)

    @property
    def threshold(self):
        return np.array(self.thr)

    @property
    def impurity(self):
        return np.array(self.imp)

    @property
    def n_node_samples(self):
        return np.array(self.nns)

    @property
    def weighted_n_node_samples(self):
        return np.array(self.wnns)

    @property
    def missing_go_to_left(self):
        return np.array([1 if b else 0 for b in self.mgl])

    @property
    def n_leaves(self) -> int:
        c = 0
        for i in range(self.node_count):
            if self.left[i] == TREE_LEAF and self.right[i] == TREE_LEAF:
                c += 1
        return c

    @property
    def value(self):
        a = np.array(self.val)
        return a.reshape((self.node_count, self.n_outputs, self.max_n_classes))

    @property
    def n_classes(self):
        return np.array(self.n_classes_list)

    # ------------------------------------------------------ inference

    def _leaf_of(self, X, i: int) -> int:
        node = 0
        while self.left[node] != TREE_LEAF:
            v = float32(X[i, self.feat[node]])
            if math.isnan(float(v)):
                node = self.left[node] if self.mgl[node] else self.right[node]
            elif float(v) <= self.thr[node]:
                node = self.left[node]
            else:
                node = self.right[node]
        return node

    def apply(self, X):
        n = X.shape[0]
        out = np.zeros(n, dtype=int)
        for i in range(n):
            out[i] = self._leaf_of(X, i)
        return out

    def predict(self, X):
        """-> (n_samples, n_outputs * max_n_classes) leaf values"""
        n = X.shape[0]
        s = self.value_stride
        out = np.zeros((n, s))
        for i in range(n):
            leaf = self._leaf_of(X, i)
            for j in range(s):
                out[i, j] = self.val[leaf * s + j]
        return out

    def decision_path(self, X):
        """Dense (n_samples, node_count) 0/1 indicator matrix."""
        n = X.shape[0]
        out = np.zeros((n, self.node_count), dtype=int)
        for i in range(n):
            node = 0
            while self.left[node] != TREE_LEAF:
                out[i, node] = 1
                v = float32(X[i, self.feat[node]])
                if math.isnan(float(v)):
                    node = self.left[node] if self.mgl[node] else self.right[node]
                elif float(v) <= self.thr[node]:
                    node = self.left[node]
                else:
                    node = self.right[node]
            out[i, node] = 1
        return out

    def compute_node_depths(self):
        d = np.empty(self.node_count, dtype=int)
        d[0] = 1
        for nid in range(self.node_count):
            if self.left[nid] != TREE_LEAF:
                d[self.left[nid]] = d[nid] + 1
                d[self.right[nid]] = d[nid] + 1
        return d

    def compute_feature_importances(self, normalize: bool = True):
        imps = np.zeros(self.n_features)
        for nid in range(self.node_count):
            l = self.left[nid]
            if l != TREE_LEAF:
                r = self.right[nid]
                imps[self.feat[nid]] += (self.wnns[nid] * self.imp[nid] -
                                         self.wnns[l] * self.imp[l] -
                                         self.wnns[r] * self.imp[r])
        for i in range(self.n_features):
            imps[i] /= self.wnns[0]
        if normalize:
            normalizer = float(np.sum(imps))
            if normalizer > 0.0:
                for i in range(self.n_features):
                    imps[i] /= normalizer
        return imps

# ----------------------------------------------------------- builders

def _child_bounds(splitter: Splitter, feature: int, lower: float, upper: float,
                  middle: float) -> Tuple[float, float, float, float]:
    """-> (left_min, left_max, right_min, right_max)"""
    if not splitter.with_monotonic_cst or splitter.monotonic_cst[feature] == 0:
        return (lower, upper, lower, upper)
    elif splitter.monotonic_cst[feature] == 1:
        return (lower, middle, middle, upper)
    else:
        return (middle, upper, lower, middle)

def build_depth_first(tree: Tree, splitter: Splitter, min_samples_split: int,
                      min_samples_leaf: int, min_weight_leaf: float, max_depth: int,
                      min_impurity_decrease: float):
    # stack entries: (start, end, depth, parent, is_left, impurity, n_const, lower, upper)
    stack = [(0, splitter.n_samples, 0, TREE_UNDEFINED, False, _INF, 0, -_INF, _INF)]
    pr = ParentInfo()
    first = True
    max_depth_seen = -1
    while len(stack) > 0:
        start, end, depth, parent, is_left, impurity, n_const, lower, upper = stack.pop()
        pr.impurity = impurity
        pr.n_constant_features = n_const
        pr.lower_bound = lower
        pr.upper_bound = upper
        n_node_samples = end - start
        wn = splitter.node_reset(start, end)
        is_leaf = (depth >= max_depth or n_node_samples < min_samples_split or
                   n_node_samples < 2 * min_samples_leaf or wn < 2.0 * min_weight_leaf)
        if first:
            pr.impurity = splitter.node_impurity()
            first = False
        is_leaf = is_leaf or pr.impurity <= _EPSILON
        split = SplitRecord(end)
        if not is_leaf:
            split = splitter.node_split(pr)
            is_leaf = (split.pos >= end or
                       split.improvement + _EPSILON < min_impurity_decrease)
        node_id = tree._add_node(parent, is_left, is_leaf, split.feature, split.threshold,
                                 pr.impurity, n_node_samples, wn, split.missing_go_to_left)
        off = node_id * tree.value_stride
        splitter.node_value(tree.val, off)
        if splitter.with_monotonic_cst:
            splitter.criterion.clip_node_value(tree.val, off, pr.lower_bound, pr.upper_bound)
        if not is_leaf:
            middle = 0.0
            if splitter.with_monotonic_cst and splitter.monotonic_cst[split.feature] != 0:
                middle = splitter.criterion.middle_value()
            lmin, lmax, rmin, rmax = _child_bounds(splitter, split.feature,
                                                   pr.lower_bound, pr.upper_bound, middle)
            stack.append((split.pos, end, depth + 1, node_id, False, split.impurity_right,
                          pr.n_constant_features, rmin, rmax))
            stack.append((start, split.pos, depth + 1, node_id, True, split.impurity_left,
                          pr.n_constant_features, lmin, lmax))
        if depth > max_depth_seen:
            max_depth_seen = depth
    tree.max_depth = max_depth_seen

class _Frontier:
    node_id: int
    start: int
    end: int
    pos: int
    depth: int
    is_leaf: bool
    impurity: float
    impurity_left: float
    impurity_right: float
    improvement: float
    lower_bound: float
    upper_bound: float
    middle_value: float

    def __init__(self):
        self.node_id = 0; self.start = 0; self.end = 0; self.pos = 0; self.depth = 0
        self.is_leaf = True; self.impurity = 0.0; self.impurity_left = 0.0
        self.impurity_right = 0.0; self.improvement = 0.0; self.lower_bound = 0.0
        self.upper_bound = 0.0; self.middle_value = 0.0

def _push_heap(h: List[_Frontier], rec: _Frontier):
    # std::push_heap with comparator a.improvement < b.improvement (max-heap)
    h.append(rec)
    i = len(h) - 1
    v = h[i]
    while i > 0:
        parent = (i - 1) // 2
        if h[parent].improvement < v.improvement:
            h[i] = h[parent]
            i = parent
        else:
            break
    h[i] = v

def _pop_heap(h: List[_Frontier]) -> _Frontier:
    # libc++ std::pop_heap: Floyd's sift-down-to-leaf then sift-up
    n = len(h)
    top = h[0]
    if n > 1:
        # __floyd_sift_down
        hole = 0
        length = n
        while True:
            child = 2 * hole + 1
            if child + 1 < length and h[child].improvement < h[child + 1].improvement:
                child += 1
            h[hole] = h[child]
            hole = child
            if hole > (length - 2) // 2:
                break
        last = n - 1
        if hole == last:
            h[hole] = top
        else:
            h[hole] = h[last]
            h[last] = top
            # __sift_up on [0, hole]
            i = hole
            v = h[i]
            while i > 0:
                parent = (i - 1) // 2
                if h[parent].improvement < v.improvement:
                    h[i] = h[parent]
                    i = parent
                else:
                    break
            h[i] = v
    return h.pop()

def _add_split_node(tree: Tree, splitter: Splitter, start: int, end: int, is_first: bool,
                    is_left: bool, parent: int, depth: int, pr: ParentInfo,
                    min_samples_split: int, min_samples_leaf: int, min_weight_leaf: float,
                    max_depth: int, min_impurity_decrease: float) -> _Frontier:
    wn = splitter.node_reset(start, end)
    pr.n_constant_features = 0
    if is_first:
        pr.impurity = splitter.node_impurity()
    n_node_samples = end - start
    is_leaf = (depth >= max_depth or n_node_samples < min_samples_split or
               n_node_samples < 2 * min_samples_leaf or wn < 2.0 * min_weight_leaf or
               pr.impurity <= _EPSILON)
    split = SplitRecord(end)
    if not is_leaf:
        split = splitter.node_split(pr)
        is_leaf = split.pos >= end or split.improvement + _EPSILON < min_impurity_decrease
    node_id = tree._add_node(parent, is_left, is_leaf, split.feature, split.threshold,
                             pr.impurity, n_node_samples, wn, split.missing_go_to_left)
    off = node_id * tree.value_stride
    splitter.node_value(tree.val, off)
    if splitter.with_monotonic_cst:
        splitter.criterion.clip_node_value(tree.val, off, pr.lower_bound, pr.upper_bound)
    res = _Frontier()
    res.node_id = node_id
    res.start = start
    res.end = end
    res.depth = depth
    res.impurity = pr.impurity
    res.lower_bound = pr.lower_bound
    res.upper_bound = pr.upper_bound
    res.middle_value = 0.0
    if splitter.with_monotonic_cst and not is_leaf:
        res.middle_value = splitter.criterion.middle_value()
    if not is_leaf:
        res.pos = split.pos
        res.is_leaf = False
        res.improvement = split.improvement
        res.impurity_left = split.impurity_left
        res.impurity_right = split.impurity_right
    else:
        res.pos = end
        res.is_leaf = True
        res.improvement = 0.0
        res.impurity_left = pr.impurity
        res.impurity_right = pr.impurity
    return res

def build_best_first(tree: Tree, splitter: Splitter, min_samples_split: int,
                     min_samples_leaf: int, min_weight_leaf: float, max_depth: int,
                     max_leaf_nodes: int, min_impurity_decrease: float):
    max_split_nodes = max_leaf_nodes - 1
    max_depth_seen = -1
    pr = ParentInfo()
    frontier = List[_Frontier]()
    rec0 = _add_split_node(tree, splitter, 0, splitter.n_samples, True, True,
                           TREE_UNDEFINED, 0, pr, min_samples_split, min_samples_leaf,
                           min_weight_leaf, max_depth, min_impurity_decrease)
    _push_heap(frontier, rec0)
    while len(frontier) > 0:
        record = _pop_heap(frontier)
        nid = record.node_id
        is_leaf = record.is_leaf or max_split_nodes <= 0
        if is_leaf:
            tree.left[nid] = TREE_LEAF
            tree.right[nid] = TREE_LEAF
            tree.feat[nid] = TREE_UNDEFINED
            tree.thr[nid] = float(TREE_UNDEFINED)
        else:
            lmin, lmax, rmin, rmax = _child_bounds(splitter, tree.feat[nid], record.lower_bound,
                                                   record.upper_bound, record.middle_value)
            max_split_nodes -= 1
            pr.lower_bound = lmin
            pr.upper_bound = lmax
            pr.impurity = record.impurity_left
            left = _add_split_node(tree, splitter, record.start, record.pos, False, True,
                                   nid, record.depth + 1, pr, min_samples_split,
                                   min_samples_leaf, min_weight_leaf, max_depth,
                                   min_impurity_decrease)
            pr.lower_bound = rmin
            pr.upper_bound = rmax
            pr.impurity = record.impurity_right
            right = _add_split_node(tree, splitter, record.pos, record.end, False, False,
                                    nid, record.depth + 1, pr, min_samples_split,
                                    min_samples_leaf, min_weight_leaf, max_depth,
                                    min_impurity_decrease)
            _push_heap(frontier, left)
            _push_heap(frontier, right)
        if record.depth > max_depth_seen:
            max_depth_seen = record.depth
    tree.max_depth = max_depth_seen

# ------------------------------------------------------------ pruning

def _cost_complexity_prune(orig: Tree, ccp_alpha: float, path: bool,
                           alphas: List[float], imps: List[float]) -> List[bool]:
    """Returns leaves_in_subtree. When `path`, records the full pruning path
    (ccp_pruning_path) instead of stopping at ccp_alpha."""
    n_nodes = orig.node_count
    total = orig.wnns[0]
    r_node = [orig.wnns[i] * orig.imp[i] / total for i in range(n_nodes)]
    child_l = orig.left
    child_r = orig.right
    parent = [0] * n_nodes
    leaves = [False] * n_nodes
    st = [(0, TREE_UNDEFINED)]
    while len(st) > 0:
        node_idx, par = st.pop()
        parent[node_idx] = par
        if child_l[node_idx] == TREE_LEAF:
            leaves[node_idx] = True
        else:
            st.append((child_l[node_idx], node_idx))
            st.append((child_r[node_idx], node_idx))
    n_leaves = [0] * n_nodes
    r_branch = [0.0] * n_nodes
    for leaf_idx0 in range(n_nodes):
        if not leaves[leaf_idx0]:
            continue
        leaf_idx = leaf_idx0
        r_branch[leaf_idx] = r_node[leaf_idx]
        current_r = r_node[leaf_idx]
        while leaf_idx != 0:
            p = parent[leaf_idx]
            r_branch[p] += current_r
            n_leaves[p] += 1
            leaf_idx = p
    candidate = [not leaves[i] for i in range(n_nodes)]
    in_subtree = [True] * n_nodes
    if path:
        alphas.append(0.0); imps.append(r_branch[0])
    max_f = 1.7976931348623157e308
    while candidate[0]:
        effective_alpha = max_f
        pruned = 0
        for i in range(n_nodes):
            if not candidate[i]:
                continue
            subtree_alpha = (r_node[i] - r_branch[i]) / float(n_leaves[i] - 1)
            if subtree_alpha < effective_alpha:
                effective_alpha = subtree_alpha
                pruned = i
        if not path and ccp_alpha < effective_alpha:
            break
        nst = [pruned]
        while len(nst) > 0:
            node_idx = nst.pop()
            if not in_subtree[node_idx]:
                continue
            candidate[node_idx] = False
            leaves[node_idx] = False
            in_subtree[node_idx] = False
            if child_l[node_idx] != TREE_LEAF:
                nst.append(child_l[node_idx])
                nst.append(child_r[node_idx])
        leaves[pruned] = True
        in_subtree[pruned] = True
        n_pruned_leaves = n_leaves[pruned] - 1
        n_leaves[pruned] = 0
        r_diff = r_node[pruned] - r_branch[pruned]
        r_branch[pruned] = r_node[pruned]
        node_idx = parent[pruned]
        while node_idx != TREE_UNDEFINED:
            n_leaves[node_idx] -= n_pruned_leaves
            r_branch[node_idx] += r_diff
            node_idx = parent[node_idx]
        if path:
            alphas.append(effective_alpha); imps.append(r_branch[0])
    return leaves

def build_pruned_tree_ccp(orig: Tree, ccp_alpha: float) -> Tree:
    leaves = _cost_complexity_prune(orig, ccp_alpha, False, List[float](), List[float]())
    tree = Tree(orig.n_features, orig.n_classes_list, orig.n_outputs)
    vs = orig.value_stride
    st = [(0, 0, TREE_UNDEFINED, False)]
    max_depth_seen = -1
    while len(st) > 0:
        oid, depth, parent, is_left = st.pop()
        is_leaf = leaves[oid]
        nid = tree._add_node(parent, is_left, is_leaf, orig.feat[oid], orig.thr[oid],
                             orig.imp[oid], orig.nns[oid], orig.wnns[oid], orig.mgl[oid])
        for j in range(vs):
            tree.val[nid * vs + j] = orig.val[oid * vs + j]
        if not is_leaf:
            st.append((orig.right[oid], depth + 1, nid, False))
            st.append((orig.left[oid], depth + 1, nid, True))
        if depth > max_depth_seen:
            max_depth_seen = depth
    tree.max_depth = max_depth_seen
    return tree

def ccp_pruning_path(orig: Tree):
    alphas = List[float]()
    imps = List[float]()
    _cost_complexity_prune(orig, 0.0, True, alphas, imps)
    return (np.array(alphas), np.array(imps))
