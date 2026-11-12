# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Dense best / random splitters (sklearn/tree/_splitter.pyx with the
# DensePartitioner of _partitioner.pyx). X is float32 stored column-major,
# feature values are float32 like sklearn, thresholds are float64.

import math
from ._utils import _RandR, simultaneous_sort, FEATURE_THRESHOLD
from ._criterion import Criterion

_INF: float = float("inf")

class SplitRecord:
    feature: int
    pos: int
    threshold: float
    improvement: float
    impurity_left: float
    impurity_right: float
    missing_go_to_left: bool

    def __init__(self, start_pos: int):
        self.impurity_left = _INF
        self.impurity_right = _INF
        self.pos = start_pos
        self.feature = 0
        self.threshold = 0.0
        self.improvement = -_INF
        self.missing_go_to_left = False

    def copy(self) -> SplitRecord:
        r = SplitRecord(self.pos)
        r.feature = self.feature
        r.threshold = self.threshold
        r.improvement = self.improvement
        r.impurity_left = self.impurity_left
        r.impurity_right = self.impurity_right
        r.missing_go_to_left = self.missing_go_to_left
        return r

class ParentInfo:
    n_constant_features: int
    impurity: float
    lower_bound: float
    upper_bound: float

    def __init__(self):
        self.n_constant_features = 0
        self.impurity = _INF
        self.lower_bound = -_INF
        self.upper_bound = _INF

class Splitter:
    criterion: Criterion
    random: bool
    max_features: int
    min_samples_leaf: int
    min_weight_leaf: float
    rng: _RandR
    monotonic_cst: List[int]
    with_monotonic_cst: bool
    # data
    X: List[float32]        # column-major: X[f * n_total + i]
    n_total: int
    y: List[float]
    sample_weight: List[float]
    samples: List[int]
    n_samples: int
    weighted_n_samples: float
    features: List[int]
    n_features: int
    feature_values: List[float32]
    constant_features: List[int]
    start: int
    end: int
    # partitioner
    missing_mask: List[bool]
    has_missing_mask: bool
    n_missing: int

    def __init__(self, criterion: Criterion, random: bool, max_features: int,
                 min_samples_leaf: int, min_weight_leaf: float, seed: int,
                 monotonic_cst: List[int]):
        self.criterion = criterion
        self.random = random
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.rng = _RandR(seed)
        self.monotonic_cst = monotonic_cst
        self.with_monotonic_cst = len(monotonic_cst) > 0
        self.n_missing = 0
        self.start = 0
        self.end = 0
        self.X = List[float32]()
        self.n_total = 0
        self.y = List[float]()
        self.sample_weight = List[float]()
        self.samples = List[int]()
        self.n_samples = 0
        self.weighted_n_samples = 0.0
        self.features = List[int]()
        self.n_features = 0
        self.feature_values = List[float32]()
        self.constant_features = List[int]()
        self.missing_mask = List[bool]()
        self.has_missing_mask = False

    def init(self, X: List[float32], n_total: int, n_features: int, y: List[float],
             sample_weight: List[float], has_weight: bool, missing_mask: List[bool]):
        self.X = X
        self.n_total = n_total
        self.samples = [0] * n_total
        j = 0
        wn = 0.0
        for i in range(n_total):
            if not has_weight or sample_weight[i] != 0.0:
                self.samples[j] = i
                j += 1
            wn += sample_weight[i]
        self.n_samples = j
        self.weighted_n_samples = wn
        self.features = list(range(n_features))
        self.n_features = n_features
        self.feature_values = [float32(0.0)] * n_total
        self.constant_features = [0] * n_features
        self.y = y
        self.sample_weight = sample_weight
        self.missing_mask = missing_mask
        self.has_missing_mask = len(missing_mask) > 0

    def node_reset(self, start: int, end: int) -> float:
        self.start = start
        self.end = end
        self.criterion.init(self.y, self.sample_weight, self.weighted_n_samples,
                            self.samples, start, end)
        return self.criterion.weighted_n_node_samples

    def node_value(self, dest: List[float], off: int):
        self.criterion.node_value(dest, off)

    def node_impurity(self) -> float:
        return self.criterion.node_impurity()

    # ------------------------------------------------------- partitioner

    def _x(self, i: int, f: int) -> float32:
        return self.X[f * self.n_total + i]

    def sort_samples_and_feature_values(self, f: int):
        fv = self.feature_values
        samples = self.samples
        n_missing = 0
        if self.has_missing_mask and self.missing_mask[f]:
            i = self.start
            current_end = self.end - 1
            while i <= current_end:
                if math.isnan(float(self._x(samples[current_end], f))):
                    n_missing += 1
                    current_end -= 1
                    continue
                if math.isnan(float(self._x(samples[i], f))):
                    t = samples[i]; samples[i] = samples[current_end]; samples[current_end] = t
                    n_missing += 1
                    current_end -= 1
                fv[i] = self._x(samples[i], f)
                i += 1
        else:
            base = f * self.n_total
            X = self.X
            for i in range(self.start, self.end):
                fv[i] = X[base + samples[i]]
        simultaneous_sort(fv, samples, self.start, self.end - self.start - n_missing, True)
        self.n_missing = n_missing

    def shift_missing_to_the_left(self):
        n_non_missing = self.end - self.start - self.n_missing
        s = self.start
        a = self.samples[s:self.end]
        b = self.feature_values[s:self.end]
        n = self.end - s
        for t in range(n):
            src = (t + n_non_missing) % n
            self.samples[s + t] = a[src]
            self.feature_values[s + t] = b[src]

    def find_min_max(self, f: int) -> Tuple[float32, float32]:
        fv = self.feature_values
        mn = float32(_INF)
        mx = float32(-_INF)
        n_missing = 0
        seen = False
        for p in range(self.start, self.end):
            v = self._x(self.samples[p], f)
            fv[p] = v
            if math.isnan(float(v)):
                n_missing += 1
            elif not seen:
                mn = v; mx = v; seen = True
            elif v < mn:
                mn = v
            elif v > mx:
                mx = v
        self.n_missing = n_missing
        return (mn, mx)

    def next_p(self, p: int, missing_go_to_left: bool) -> Tuple[int, int]:
        end_non_missing = self.end if missing_go_to_left else self.end - self.n_missing
        if p == end_non_missing and not missing_go_to_left:
            return (self.end, self.end)
        if missing_go_to_left and p == self.start:
            p = self.start + self.n_missing
        p += 1
        fv = self.feature_values
        while p < end_non_missing and fv[p] <= fv[p - 1] + FEATURE_THRESHOLD:
            p += 1
        return (p - 1, p)

    def partition_samples(self, threshold: float, missing_go_to_left: bool) -> int:
        ps = self.start
        pe = self.end
        fv = self.feature_values
        samples = self.samples
        while ps < pe:
            v = fv[ps]
            go_left = missing_go_to_left if math.isnan(float(v)) else float(v) <= threshold
            if go_left:
                ps += 1
            else:
                pe -= 1
                t = fv[ps]; fv[ps] = fv[pe]; fv[pe] = t
                u = samples[ps]; samples[ps] = samples[pe]; samples[pe] = u
        return pe

    def partition_samples_final(self, best: SplitRecord):
        ps = self.start
        pe = self.end
        samples = self.samples
        f = best.feature
        while ps < pe:
            v = self._x(samples[ps], f)
            go_left = best.missing_go_to_left if math.isnan(float(v)) else float(v) <= best.threshold
            if go_left:
                ps += 1
            else:
                pe -= 1
                u = samples[ps]; samples[ps] = samples[pe]; samples[pe] = u

    # ----------------------------------------------------------- split

    def node_split(self, parent: ParentInfo) -> SplitRecord:
        if self.random:
            return self._node_split_random(parent)
        return self._node_split_best(parent)

    def _finish(self, best: SplitRecord, parent: ParentInfo, n_known: int,
                n_found: int, n_total_c: int, do_partition: bool):
        criterion = self.criterion
        if best.pos < self.end:
            if do_partition:
                self.partition_samples_final(best)
            criterion.reset()
            criterion.update(best.pos)
            il, ir = criterion.children_impurity()
            best.impurity_left = il
            best.impurity_right = ir
            best.improvement = criterion.impurity_improvement(parent.impurity, il, ir)
        features = self.features
        cf = self.constant_features
        for t in range(n_known):
            features[t] = cf[t]
        for t in range(n_found):
            cf[n_known + t] = features[n_known + t]
        parent.n_constant_features = n_total_c

    def _node_split_best(self, parent: ParentInfo) -> SplitRecord:
        criterion = self.criterion
        start = self.start
        end = self.end
        features = self.features
        fv = self.feature_values
        max_features = self.max_features
        msl = self.min_samples_leaf
        mwl = self.min_weight_leaf
        best = SplitRecord(end)
        cur = SplitRecord(end)
        best_proxy = -_INF
        f_i = self.n_features
        n_visited = 0
        n_found = 0
        n_drawn = 0
        n_known = parent.n_constant_features
        n_total_c = n_known
        self.n_missing = 0
        while f_i > n_total_c and (n_visited < max_features or n_visited <= n_found + n_drawn):
            n_visited += 1
            f_j = self.rng.rand_int(n_drawn, f_i - n_found)
            if f_j < n_known:
                t = features[n_drawn]; features[n_drawn] = features[f_j]; features[f_j] = t
                n_drawn += 1
                continue
            f_j += n_found
            cur.feature = features[f_j]
            self.sort_samples_and_feature_values(cur.feature)
            n_missing = self.n_missing
            end_nm = end - n_missing
            if end_nm == start or (fv[end_nm - 1] <= fv[start] + FEATURE_THRESHOLD and n_missing == 0):
                t = features[f_j]; features[f_j] = features[n_total_c]; features[n_total_c] = t
                n_found += 1
                n_total_c += 1
                continue
            f_i -= 1
            t = features[f_i]; features[f_i] = features[f_j]; features[f_j] = t
            n_searches = 2 if n_missing != 0 else 1
            for si in range(n_searches):
                mgl = si == 1
                if mgl:
                    self.shift_missing_to_the_left()
                criterion.reset()
                p = start
                p_prev = start
                while p < end:
                    p_prev, p = self.next_p(p, mgl)
                    if p == end:
                        continue
                    n_left = p - start
                    n_right = end - p
                    if n_left < msl or n_right < msl:
                        continue
                    cur.pos = p
                    criterion.update(p)
                    if (self.with_monotonic_cst and self.monotonic_cst[cur.feature] != 0 and
                            not criterion.check_monotonicity(self.monotonic_cst[cur.feature],
                                                             parent.lower_bound, parent.upper_bound)):
                        continue
                    if criterion.weighted_n_left < mwl or criterion.weighted_n_right < mwl:
                        continue
                    cpi = criterion.proxy_impurity_improvement()
                    if cpi > best_proxy:
                        best_proxy = cpi
                        if p == end_nm and not mgl:
                            cur.threshold = _INF
                        else:
                            cur.threshold = float(fv[p_prev]) / 2.0 + float(fv[p]) / 2.0
                        if n_missing == 0:
                            cur.missing_go_to_left = n_left > n_right
                        else:
                            cur.missing_go_to_left = mgl
                        best = cur.copy()
        self._finish(best, parent, n_known, n_found, n_total_c, True)
        return best

    def _node_split_random(self, parent: ParentInfo) -> SplitRecord:
        criterion = self.criterion
        start = self.start
        end = self.end
        features = self.features
        max_features = self.max_features
        msl = self.min_samples_leaf
        mwl = self.min_weight_leaf
        best = SplitRecord(end)
        cur = SplitRecord(end)
        best_proxy = -_INF
        f_i = self.n_features
        n_visited = 0
        n_found = 0
        n_drawn = 0
        n_known = parent.n_constant_features
        n_total_c = n_known
        self.n_missing = 0
        while f_i > n_total_c and (n_visited < max_features or n_visited <= n_found + n_drawn):
            n_visited += 1
            f_j = self.rng.rand_int(n_drawn, f_i - n_found)
            if f_j < n_known:
                t = features[n_drawn]; features[n_drawn] = features[f_j]; features[f_j] = t
                n_drawn += 1
                continue
            f_j += n_found
            cur.feature = features[f_j]
            mn, mx = self.find_min_max(cur.feature)
            n_missing = self.n_missing
            if end - start == n_missing or (mx <= mn + FEATURE_THRESHOLD and n_missing == 0):
                features[f_j] = features[n_total_c]
                features[n_total_c] = cur.feature
                n_found += 1
                n_total_c += 1
                continue
            f_i -= 1
            t = features[f_i]; features[f_i] = features[f_j]; features[f_j] = t
            has_missing = n_missing != 0
            cur.threshold = self.rng.rand_uniform(float(mn), float(mx))
            mgl = False
            if has_missing:
                mgl = self.rng.rand_int(0, 2) != 0
            if cur.threshold == float(mx):
                cur.threshold = float(mn)
            cur.pos = self.partition_samples(cur.threshold, mgl)
            n_left = cur.pos - start
            n_right = end - cur.pos
            if n_left < msl or n_right < msl:
                continue
            criterion.reset()
            criterion.update(cur.pos)
            if criterion.weighted_n_left < mwl or criterion.weighted_n_right < mwl:
                continue
            if (self.with_monotonic_cst and self.monotonic_cst[cur.feature] != 0 and
                    not criterion.check_monotonicity(self.monotonic_cst[cur.feature],
                                                     parent.lower_bound, parent.upper_bound)):
                continue
            cpi = criterion.proxy_impurity_improvement()
            if cpi > best_proxy:
                if has_missing:
                    cur.missing_go_to_left = mgl
                else:
                    cur.missing_go_to_left = n_left > n_right
                best_proxy = cpi
                best = cur.copy()
        self._finish(best, parent, n_known, n_found, n_total_c,
                     cur.feature != best.feature)
        return best
