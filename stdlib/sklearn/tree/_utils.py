# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Helpers shared by the tree builders: the xorshift rand_r used by the
# splitters (sklearn/utils/_random.pxd), the introsort used to sort feature
# values together with sample indices (sklearn/utils/_sorting.pyx) and the
# weighted Fenwick tree of the absolute-error criterion (tree/_utils.pyx).
# All ports are exact so that split orders and float sums match sklearn.

import math

RAND_R_MAX: Literal[int] = 2147483647
FEATURE_THRESHOLD: float32 = float32(1e-7)

class _RandR:
    seed: u32

    def __init__(self, seed: int):
        self.seed = u32(seed)

    def next(self) -> int:
        if self.seed == u32(0):
            self.seed = u32(1)
        s = self.seed
        s ^= s << u32(13)
        s ^= s >> u32(17)
        s ^= s << u32(5)
        self.seed = s
        return int(s % u32(2147483648))

    def rand_int(self, low: int, high: int) -> int:
        return low + self.next() % (high - low)

    def rand_uniform(self, low: float, high: float) -> float:
        return ((high - low) * float(self.next()) / float(RAND_R_MAX)) + low

# ------------------------------------------------------------- introsort

def _swap(v: List[T], ix: List[int], off: int, i: int, j: int, T: type):
    a = v[off + i]; v[off + i] = v[off + j]; v[off + j] = a
    b = ix[off + i]; ix[off + i] = ix[off + j]; ix[off + j] = b

def _median3(v: List[T], off: int, n: int, T: type) -> T:
    a = v[off]; b = v[off + n // 2]; c = v[off + n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b

def _inplace_median3(v: List[T], ix: List[int], off: int, n: int, T: type) -> T:
    pivot_idx = n // 2
    if v[off] > v[off + n - 1]:
        _swap(v, ix, off, 0, n - 1)
    if v[off + n - 1] > v[off + pivot_idx]:
        _swap(v, ix, off, n - 1, pivot_idx)
        if v[off] > v[off + n - 1]:
            _swap(v, ix, off, 0, n - 1)
    return v[off + n - 1]

def _insertion_sort(v: List[T], ix: List[int], off: int, n: int, T: type):
    for i in range(1, n):
        tv = v[off + i]
        ti = ix[off + i]
        j = i
        while j > 0 and v[off + j - 1] > tv:
            v[off + j] = v[off + j - 1]
            ix[off + j] = ix[off + j - 1]
            j -= 1
        v[off + j] = tv
        ix[off + j] = ti

def _sift_down(v: List[T], ix: List[int], off: int, start: int, end: int, T: type):
    root = start
    while True:
        child = root * 2 + 1
        maxind = root
        if child < end and v[off + maxind] < v[off + child]:
            maxind = child
        if child + 1 < end and v[off + maxind] < v[off + child + 1]:
            maxind = child + 1
        if maxind == root:
            break
        _swap(v, ix, off, root, maxind)
        root = maxind

def _heapsort(v: List[T], ix: List[int], off: int, n: int, T: type):
    start = (n - 2) // 2
    end = n
    while True:
        _sift_down(v, ix, off, start, end)
        if start == 0:
            break
        start -= 1
    end = n - 1
    while end > 0:
        _swap(v, ix, off, 0, end)
        _sift_down(v, ix, off, 0, end)
        end -= 1

def _introsort_3way(v: List[T], ix: List[int], off: int, n: int, maxd: int, T: type):
    while n > 15:
        if maxd <= 0:
            _heapsort(v, ix, off, n)
            return
        maxd -= 1
        pivot = _median3(v, off, n)
        i = 0; l = 0; r = n
        while i < r:
            if v[off + i] < pivot:
                _swap(v, ix, off, i, l)
                i += 1
                l += 1
            elif v[off + i] > pivot:
                r -= 1
                _swap(v, ix, off, i, r)
            else:
                i += 1
        _introsort_3way(v, ix, off, l, maxd)
        off += r
        n -= r
    _insertion_sort(v, ix, off, n)

def _introsort_2way(v: List[T], ix: List[int], off: int, n: int, maxd: int, T: type):
    while n > 15:
        if maxd <= 0:
            _heapsort(v, ix, off, n)
            return
        maxd -= 1
        pivot = _inplace_median3(v, ix, off, n)
        i = 1
        j = n - 2
        while True:
            while i <= j and v[off + i] < pivot:
                i += 1
            while i <= j and v[off + j] > pivot:
                j -= 1
            if i >= j:
                break
            _swap(v, ix, off, i, j)
            i += 1
            j -= 1
        pivot_idx = i
        _swap(v, ix, off, pivot_idx, n - 1)
        _introsort_2way(v, ix, off, pivot_idx, maxd)
        off += pivot_idx + 1
        n -= pivot_idx + 1
    _insertion_sort(v, ix, off, n)

def simultaneous_sort(v: List[T], ix: List[int], off: int, n: int,
                      three_way: bool = False, T: type = float):
    """Sort v[off:off+n] and ix[off:off+n] by v (sklearn utils/_sorting)."""
    if n == 0:
        return
    maxd = 2 * int(math.log2(float(n)))
    if three_way:
        _introsort_3way(v, ix, off, n, maxd)
    else:
        _introsort_2way(v, ix, off, n, maxd)

# --------------------------------------------------------- Fenwick tree

class WeightedFenwickTree:
    size: int
    tree_w: List[float]
    tree_wy: List[float]
    total_w: float
    total_wy: float
    max_pow2: int

    def __init__(self, capacity: int):
        self.size = 0
        self.tree_w = [0.0] * (capacity + 1)
        self.tree_wy = [0.0] * (capacity + 1)
        self.total_w = 0.0
        self.total_wy = 0.0
        self.max_pow2 = 0

    def reset(self, size: int):
        self.size = size
        for p in range(size + 1):
            self.tree_w[p] = 0.0
            self.tree_wy[p] = 0.0
        self.total_w = 0.0
        self.total_wy = 0.0
        p = 1
        while p <= size:
            p <<= 1
        self.max_pow2 = p >> 1

    def add(self, idx: int, y_value: float, weight: float):
        wy = weight * y_value
        k = idx + 1
        while k <= self.size:
            self.tree_w[k] += weight
            self.tree_wy[k] += wy
            k += k & -k
        self.total_w += weight
        self.total_wy += wy

    def search(self, target_weight: float) -> Tuple[int, float, float, int]:
        """-> (idx, cumul_weight, cumul_weighted_y, prev_idx)"""
        current_idx = 0
        cumul_weight = 0.0
        cumul_wy = 0.0
        search_bit = self.max_pow2
        equal_target = 0.0
        equal_bit = 0
        while search_bit != 0:
            next_idx = current_idx + search_bit
            if next_idx <= self.size:
                node_weight = self.tree_w[next_idx]
                if target_weight == node_weight:
                    equal_target = target_weight
                    equal_bit = search_bit
                    break
                elif target_weight > node_weight:
                    target_weight -= node_weight
                    current_idx = next_idx
                    cumul_weight += node_weight
                    cumul_wy += self.tree_wy[next_idx]
            search_bit >>= 1
        if search_bit == 0:
            return (current_idx, cumul_weight, cumul_wy, current_idx)
        prev_idx = current_idx
        while search_bit != 0:
            next_idx = prev_idx + search_bit
            if next_idx <= self.size:
                node_weight = self.tree_w[next_idx]
                if target_weight > node_weight:
                    target_weight -= node_weight
                    prev_idx = next_idx
            search_bit >>= 1
        search_bit = equal_bit
        target_weight = equal_target
        while search_bit != 0:
            next_idx = current_idx + search_bit
            if next_idx <= self.size:
                node_weight = self.tree_w[next_idx]
                if target_weight >= node_weight:
                    target_weight -= node_weight
                    current_idx = next_idx
                    cumul_weight += node_weight
                    cumul_wy += self.tree_wy[next_idx]
            search_bit >>= 1
        return (current_idx, cumul_weight, cumul_wy, prev_idx)
