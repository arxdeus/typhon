# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Helpers of sklearn.neighbors BinaryTree: utils/_sorting.pyx introsort
# (simultaneous_sort), utils/_heap.pyx heap_push, the NodeHeap min-heap and
# libc++ std::nth_element used by _partition_nodes.pyx.  Ported operation
# for operation so ties resolve exactly like scikit-learn.

import numpy as np
import math

def _swap2(v: np.ndarray[float, 1], ix: np.ndarray[int, 1], i: int, j: int):
    t = v[i]
    v[i] = v[j]
    v[j] = t
    u = ix[i]
    ix[i] = ix[j]
    ix[j] = u

def _insertion_sort(v: np.ndarray[float, 1], ix: np.ndarray[int, 1], o: int, n: int):
    for i in range(1, n):
        tv = v[o + i]
        ti = ix[o + i]
        j = i
        while j > 0 and v[o + j - 1] > tv:
            v[o + j] = v[o + j - 1]
            ix[o + j] = ix[o + j - 1]
            j -= 1
        v[o + j] = tv
        ix[o + j] = ti

def _sift_down(v: np.ndarray[float, 1], ix: np.ndarray[int, 1], o: int, start: int, end: int):
    root = start
    while True:
        child = root * 2 + 1
        maxind = root
        if child < end and v[o + maxind] < v[o + child]:
            maxind = child
        if child + 1 < end and v[o + maxind] < v[o + child + 1]:
            maxind = child + 1
        if maxind == root:
            break
        _swap2(v, ix, o + root, o + maxind)
        root = maxind

def _heapsort2(v: np.ndarray[float, 1], ix: np.ndarray[int, 1], o: int, n: int):
    start = (n - 2) // 2
    end = n
    while True:
        _sift_down(v, ix, o, start, end)
        if start == 0:
            break
        start -= 1
    end = n - 1
    while end > 0:
        _swap2(v, ix, o, o + end)
        _sift_down(v, ix, o, 0, end)
        end -= 1

def _introsort_2way(v: np.ndarray[float, 1], ix: np.ndarray[int, 1], o0: int, n0: int, maxd0: int):
    o = o0
    n = n0
    maxd = maxd0
    while n > 15:
        if maxd <= 0:
            _heapsort2(v, ix, o, n)
            return
        maxd -= 1
        # inplace_median3
        pi = n // 2
        if v[o] > v[o + n - 1]:
            _swap2(v, ix, o, o + n - 1)
        if v[o + n - 1] > v[o + pi]:
            _swap2(v, ix, o + n - 1, o + pi)
            if v[o] > v[o + n - 1]:
                _swap2(v, ix, o, o + n - 1)
        pivot = v[o + n - 1]
        i = 1
        j = n - 2
        while True:
            while i <= j and v[o + i] < pivot:
                i += 1
            while i <= j and v[o + j] > pivot:
                j -= 1
            if i >= j:
                break
            _swap2(v, ix, o + i, o + j)
            i += 1
            j -= 1
        pivot_idx = i
        _swap2(v, ix, o + pivot_idx, o + n - 1)
        _introsort_2way(v, ix, o, pivot_idx, maxd)
        o += pivot_idx + 1
        n -= pivot_idx + 1
    _insertion_sort(v, ix, o, n)

def simultaneous_sort_range(v: np.ndarray[float, 1], ix: np.ndarray[int, 1], o: int, n: int):
    """sklearn.utils._sorting.simultaneous_sort on v[o:o+n], ix[o:o+n]."""
    if n == 0:
        return
    maxd = 2 * int(math.log2(float(n)))
    _introsort_2way(v, ix, o, n, maxd)

def simultaneous_sort(distances: np.ndarray[float, 2], indices: np.ndarray[int, 2]):
    """sklearn.neighbors._binary_tree.simultaneous_sort: sort every row."""
    for r in range(distances.shape[0]):
        n = distances.shape[1]
        rv = np.ascontiguousarray(distances[r])
        ri = np.ascontiguousarray(indices[r])
        simultaneous_sort_range(rv, ri, 0, n)
        for c in range(n):
            distances[r, c] = rv[c]
            indices[r, c] = ri[c]

def heap_push(HD: np.ndarray[float, 2], HI: np.ndarray[int, 2], row: int, size: int,
              val: float, val_idx: int):
    """Push (val, val_idx) onto the fixed-size max-heap stored in row `row`."""
    if val >= HD[row, 0]:
        return
    HD[row, 0] = val
    HI[row, 0] = val_idx
    cur = 0
    while True:
        left = 2 * cur + 1
        right = left + 1
        swap_idx = 0
        if left >= size:
            break
        elif right >= size:
            if HD[row, left] > val:
                swap_idx = left
            else:
                break
        elif HD[row, left] >= HD[row, right]:
            if val < HD[row, left]:
                swap_idx = left
            else:
                break
        else:
            if val < HD[row, right]:
                swap_idx = right
            else:
                break
        HD[row, cur] = HD[row, swap_idx]
        HI[row, cur] = HI[row, swap_idx]
        cur = swap_idx
    HD[row, cur] = val
    HI[row, cur] = val_idx

class NodeHeap:
    """Growable min-heap of (val, node) used by breadth-first queries."""
    v: np.ndarray[float, 1]
    k: np.ndarray[int, 1]
    n: int

    def __init__(self, size_guess: int = 100):
        sz = max(size_guess, 1)
        self.v = np.zeros(sz)
        self.k = np.zeros(sz, dtype=int)
        self.n = 0

    def clear(self):
        self.n = 0

    def _swap(self, i: int, j: int):
        t = self.v[i]
        self.v[i] = self.v[j]
        self.v[j] = t
        u = self.k[i]
        self.k[i] = self.k[j]
        self.k[j] = u

    def push(self, val: float, node: int):
        self.n += 1
        if self.n > self.v.shape[0]:
            nv = np.zeros(2 * self.n)
            nk = np.zeros(2 * self.n, dtype=int)
            for t in range(self.v.shape[0]):
                nv[t] = self.v[t]
                nk[t] = self.k[t]
            self.v = nv
            self.k = nk
        i = self.n - 1
        self.v[i] = val
        self.k[i] = node
        while i > 0:
            ip = (i - 1) // 2
            if self.v[ip] <= self.v[i]:
                break
            self._swap(i, ip)
            i = ip

    def pop(self):
        if self.n == 0:
            raise ValueError("cannot pop on empty heap")
        rv = self.v[0]
        rk = self.k[0]
        self.v[0] = self.v[self.n - 1]
        self.k[0] = self.k[self.n - 1]
        self.n -= 1
        i = 0
        while i < self.n:
            c1 = 2 * i + 1
            c2 = 2 * i + 2
            isw = 0
            if c2 < self.n:
                if self.v[c1] <= self.v[c2]:
                    isw = c1
                else:
                    isw = c2
            elif c1 < self.n:
                isw = c1
            else:
                break
            if isw > 0 and self.v[isw] <= self.v[i]:
                self._swap(i, isw)
                i = isw
            else:
                break
        return rv, rk

def nodeheap_sort(vals: np.ndarray[float, 1]):
    """Ascending sort of vals through a NodeHeap: (sorted vals, indices)."""
    n = vals.shape[0]
    h = NodeHeap(0)
    for i in range(n):
        h.push(vals[i], i)
    out = np.zeros(n)
    idx = np.zeros(n, dtype=int)
    for i in range(n):
        v, k = h.pop()
        out[i] = v
        idx[i] = k
    return out, idx

# --- libc++ std::nth_element with the (value, index) comparator -------------

def _lt(data: np.ndarray[float, 2], dim: int, x: int, y: int) -> bool:
    vx = data[x, dim]
    vy = data[y, dim]
    if vx == vy:
        return x < y
    return vx < vy

def _sw(a: np.ndarray[int, 1], i: int, j: int):
    t = a[i]
    a[i] = a[j]
    a[j] = t

def _sort3(a: np.ndarray[int, 1], data: np.ndarray[float, 2], dim: int, x: int, y: int, z: int) -> int:
    r = 0
    if not _lt(data, dim, a[y], a[x]):
        if not _lt(data, dim, a[z], a[y]):
            return r
        _sw(a, y, z)
        r = 1
        if _lt(data, dim, a[y], a[x]):
            _sw(a, x, y)
            r = 2
        return r
    if _lt(data, dim, a[z], a[y]):
        _sw(a, x, z)
        r = 1
        return r
    _sw(a, x, y)
    r = 1
    if _lt(data, dim, a[z], a[y]):
        _sw(a, y, z)
        r = 2
    return r

def _selection_sort(a: np.ndarray[int, 1], data: np.ndarray[float, 2], dim: int, first: int, last: int):
    lm1 = last - 1
    f = first
    while f != lm1:
        best = f
        t = f + 1
        while t < last:
            if _lt(data, dim, a[t], a[best]):
                best = t
            t += 1
        if best != f:
            _sw(a, f, best)
        f += 1

def nth_element(a: np.ndarray[int, 1], data: np.ndarray[float, 2], dim: int,
                first0: int, nth: int, last0: int):
    """std::nth_element(a+first0, a+nth, a+last0) ordered by (data[., dim], index)."""
    first = first0
    last = last0
    limit = 7
    while True:
        if nth == last:
            return
        ln = last - first
        if ln <= 1:
            return
        if ln == 2:
            last -= 1
            if _lt(data, dim, a[last], a[first]):
                _sw(a, first, last)
            return
        if ln == 3:
            _sort3(a, data, dim, first, first + 1, last - 1)
            return
        if ln <= limit:
            _selection_sort(a, data, dim, first, last)
            return
        m = first + ln // 2
        lm1 = last - 1
        n_swaps = _sort3(a, data, dim, first, m, lm1)
        i = first
        j = lm1
        if not _lt(data, dim, a[i], a[m]):
            found = False
            while True:
                j -= 1
                if i == j:
                    break
                if _lt(data, dim, a[j], a[m]):
                    found = True
                    break
            if found:
                _sw(a, i, j)
                n_swaps += 1
            else:
                i += 1
                j = last
                j -= 1
                if not _lt(data, dim, a[first], a[j]):
                    while True:
                        if i == j:
                            return
                        elif _lt(data, dim, a[first], a[i]):
                            _sw(a, i, j)
                            n_swaps += 1
                            i += 1
                            break
                        i += 1
                if i == j:
                    return
                while True:
                    while not _lt(data, dim, a[first], a[i]):
                        i += 1
                    j -= 1
                    while _lt(data, dim, a[first], a[j]):
                        j -= 1
                    if i >= j:
                        break
                    _sw(a, i, j)
                    n_swaps += 1
                    i += 1
                if nth < i:
                    return
                first = i
                continue
        i += 1
        if i < j:
            while True:
                while _lt(data, dim, a[i], a[m]):
                    i += 1
                j -= 1
                while not _lt(data, dim, a[j], a[m]):
                    j -= 1
                if i >= j:
                    break
                _sw(a, i, j)
                n_swaps += 1
                if m == i:
                    m = j
                i += 1
        if i != m and _lt(data, dim, a[m], a[i]):
            _sw(a, i, m)
            n_swaps += 1
        if nth == i:
            return
        if n_swaps == 0:
            if nth < i:
                j = first
                m = first
                while True:
                    j += 1
                    if j == i:
                        return
                    if _lt(data, dim, a[j], a[m]):
                        break
                    m = j
            else:
                j = i
                m = i
                while True:
                    j += 1
                    if j == last:
                        return
                    if _lt(data, dim, a[j], a[m]):
                        break
                    m = j
        if nth < i:
            last = i
        else:
            i += 1
            first = i
