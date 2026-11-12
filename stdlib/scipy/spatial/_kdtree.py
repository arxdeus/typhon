# Ported from scipy 1.18.1 (BSD-3-Clause)

# kd-tree for nearest-neighbour lookup (scipy.spatial.KDTree / cKDTree).
#
# The tree uses scipy's sliding-midpoint/median split rule with tight
# node bounding boxes. Neighbours at exactly equal distance are reported
# in increasing index order (scipy's order for exact ties depends on its
# internal heap and is not reproduced). ``eps`` is accepted and ignored
# (exact search is always performed, which satisfies any eps >= 0).

import numpy as np
import internal.static as static
from math import sqrt, fabs, inf

def _pdist1(x, y, p: float) -> float:
    # Minkowski distance raised to the p-th power (max for p=inf).
    n = len(x)
    s = 0.0
    if p == inf:
        for k in range(n):
            d = fabs(x[k] - y[k])
            if d > s:
                s = d
    elif p == 2.0:
        for k in range(n):
            d = x[k] - y[k]
            s += d * d
    elif p == 1.0:
        for k in range(n):
            s += fabs(x[k] - y[k])
    else:
        for k in range(n):
            s += fabs(x[k] - y[k]) ** p
    return s

def _root(s: float, p: float) -> float:
    if p == inf or p == 1.0:
        return s
    if p == 2.0:
        return sqrt(s)
    return s ** (1.0 / p)

def _pow(r: float, p: float) -> float:
    if p == inf or p == 1.0 or r == inf:
        return r
    if p == 2.0:
        return r * r
    return r ** p

def _accum(s: float, d: float, p: float) -> float:
    if p == inf:
        return s if s > d else d
    if p == 2.0:
        return s + d * d
    if p == 1.0:
        return s + d
    return s + d ** p

def minkowski_distance_p(x, y, p: float = 2.0):
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if p == inf:
        return np.max(np.abs(b - a), axis=-1)
    elif p == 1.0:
        return np.sum(np.abs(b - a), axis=-1)
    else:
        return np.sum(np.abs(b - a) ** p, axis=-1)

def minkowski_distance(x, y, p: float = 2.0):
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if p == inf or p == 1.0:
        return minkowski_distance_p(a, b, p)
    else:
        return minkowski_distance_p(a, b, p) ** (1.0 / p)

def distance_matrix(x, y, p: float = 2.0, threshold: int = 1000000):
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if static.len(a.shape) != 2 or static.len(b.shape) != 2:
        compile_error("distance_matrix expects 2-D inputs")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"x contains {a.shape[1]}-dimensional vectors but y contains {b.shape[1]}-dimensional vectors")
    out = np.empty((a.shape[0], b.shape[0]), dtype=float)
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            out[i, j] = _root(_pdist1(a[i], b[j], p), p)
    return out

class KDTree:
    data: np.ndarray[float, 2]
    n: int
    m: int
    leafsize: int
    maxes: np.ndarray[float, 1]
    mins: np.ndarray[float, 1]
    indices: np.ndarray[int, 1]
    # node arrays
    _start: List[int]
    _end: List[int]
    _left: List[int]
    _right: List[int]
    _dim: List[int]
    _split: List[float]
    _lo: List[np.ndarray[float, 1]]
    _hi: List[np.ndarray[float, 1]]

    def __init__(self, data, leafsize: int = 10, compact_nodes: bool = True,
                 copy_data: bool = False, balanced_tree: bool = True,
                 boxsize = None):
        a = np.asarray(data, dtype=float)
        if static.len(a.shape) != 2:
            compile_error("data must be 2 dimensions")
        if leafsize < 1:
            raise ValueError("leafsize must be at least 1")
        if boxsize is not None:
            compile_error("periodic boxsize is not supported")
        self.data = a.copy()
        self.n = a.shape[0]
        self.m = a.shape[1]
        self.leafsize = leafsize
        self.indices = np.arange(self.n)
        self._start = []
        self._end = []
        self._left = []
        self._right = []
        self._dim = []
        self._split = []
        self._lo = []
        self._hi = []
        if self.n > 0:
            self.maxes = np.max(self.data, axis=0)
            self.mins = np.min(self.data, axis=0)
        else:
            self.maxes = np.zeros(self.m, dtype=float)
            self.mins = np.zeros(self.m, dtype=float)
        self._build(0, self.n, balanced_tree)

    @property
    def size(self) -> int:
        return len(self._start)

    def _bbox(self, start: int, end: int):
        lo = np.full(self.m, inf, dtype=float)
        hi = np.full(self.m, -inf, dtype=float)
        for t in range(start, end):
            i = self.indices[t]
            for k in range(self.m):
                v = self.data[i, k]
                if v < lo[k]:
                    lo[k] = v
                if v > hi[k]:
                    hi[k] = v
        return lo, hi

    def _build(self, start: int, end: int, balanced: bool) -> int:
        node = len(self._start)
        lo, hi = self._bbox(start, end)
        self._start.append(start)
        self._end.append(end)
        self._left.append(-1)
        self._right.append(-1)
        self._dim.append(-1)
        self._split.append(0.0)
        self._lo.append(lo)
        self._hi.append(hi)
        if end - start <= self.leafsize:
            return node
        d = 0
        best = -1.0
        for k in range(self.m):
            w = hi[k] - lo[k]
            if w > best:
                best = w
                d = k
        if best == 0.0:
            return node
        idx = self.indices
        data = self.data
        split = 0.0
        if balanced:
            # median by (value, index) order
            seg = [(data[idx[t], d], idx[t]) for t in range(start, end)]
            seg.sort()
            for t in range(start, end):
                idx[t] = seg[t - start][1]
            split = data[idx[(start + end) // 2], d]
        else:
            split = (hi[d] + lo[d]) / 2.0
        p = start
        q = end - 1
        while p <= q:
            if data[idx[p], d] < split:
                p += 1
            elif data[idx[q], d] >= split:
                q -= 1
            else:
                tmp = idx[p]
                idx[p] = idx[q]
                idx[q] = tmp
                p += 1
                q -= 1
        if p == start:
            j = start
            split = data[idx[j], d]
            for t in range(start + 1, end):
                if data[idx[t], d] < split:
                    j = t
                    split = data[idx[t], d]
            tmp = idx[start]
            idx[start] = idx[j]
            idx[j] = tmp
            p = start + 1
        elif p == end:
            j = end - 1
            split = data[idx[j], d]
            for t in range(start, end - 1):
                if data[idx[t], d] > split:
                    j = t
                    split = data[idx[t], d]
            tmp = idx[end - 1]
            idx[end - 1] = idx[j]
            idx[j] = tmp
            p = end - 1
        self._dim[node] = d
        self._split[node] = split
        l = self._build(start, p, balanced)
        r = self._build(p, end, balanced)
        self._left[node] = l
        self._right[node] = r
        return node

    def _rect_min(self, node: int, x, p: float) -> float:
        # p-th power of the minimum distance from x to the node's box
        lo = self._lo[node]
        hi = self._hi[node]
        s = 0.0
        for k in range(self.m):
            d = 0.0
            if x[k] < lo[k]:
                d = lo[k] - x[k]
            elif x[k] > hi[k]:
                d = x[k] - hi[k]
            s = _accum(s, d, p)
        return s

    def _rect_max(self, node: int, x, p: float) -> float:
        lo = self._lo[node]
        hi = self._hi[node]
        s = 0.0
        for k in range(self.m):
            d = max(fabs(x[k] - lo[k]), fabs(x[k] - hi[k]))
            s = _accum(s, d, p)
        return s

    # ---- k nearest neighbours ------------------------------------------

    def _knn(self, x, k: int, p: float, ub: float):
        # returns sorted list of (dist_p, index); ub is the p-power bound
        best: List[Tuple[float, int]] = []
        bound = ub
        if self.n == 0:
            return best
        stack = [(self._rect_min(0, x, p), 0)]
        while len(stack) > 0:
            md, node = stack.pop()
            if md > bound or (md == bound and len(best) >= k):
                continue
            if self._left[node] < 0:
                for t in range(self._start[node], self._end[node]):
                    i = self.indices[t]
                    dd = _pdist1(x, self.data[i], p)
                    if dd < ub:
                        # insert keeping (dist, index) order
                        item = (dd, i)
                        if len(best) == k and not (item < best[k - 1]):
                            continue
                        pos = len(best)
                        best.append(item)
                        while pos > 0 and item < best[pos - 1]:
                            best[pos] = best[pos - 1]
                            pos -= 1
                        best[pos] = item
                        if len(best) > k:
                            best.pop()
                        if len(best) == k:
                            bound = best[k - 1][0]
            else:
                l = self._left[node]
                r = self._right[node]
                dl = self._rect_min(l, x, p)
                dr = self._rect_min(r, x, p)
                if dl <= dr:
                    stack.append((dr, r))
                    stack.append((dl, l))
                else:
                    stack.append((dl, l))
                    stack.append((dr, r))
        return best

    def _check_p(self, p: float):
        if p < 1.0:
            raise ValueError("Only p-norms with 1<=p<=infinity permitted")

    def query(self, x, k: Literal[int] = 1, eps: float = 0.0, p: float = 2.0,
              distance_upper_bound: float = inf, workers: int = 1):
        """Query the kd-tree for nearest neighbors."""
        if k < 1:
            compile_error("k must be at least 1")
        self._check_p(p)
        if eps < 0:
            raise ValueError("eps must be non-negative")
        a = np.asarray(x, dtype=float)
        ub = _pow(distance_upper_bound, p)
        if static.len(a.shape) == 1:
            if a.shape[0] != self.m:
                raise ValueError(f"x must consist of vectors of length {self.m} but has shape {a.shape}")
            best = self._knn(a, k, p, ub)
            if k == 1:
                if len(best) == 0:
                    return inf, self.n
                return _root(best[0][0], p), best[0][1]
            else:
                dd = np.full(k, inf, dtype=float)
                ii = np.full(k, self.n, dtype=int)
                for t in range(len(best)):
                    dd[t] = _root(best[t][0], p)
                    ii[t] = best[t][1]
                return dd, ii
        elif static.len(a.shape) == 2:
            if a.shape[1] != self.m:
                raise ValueError(f"x must consist of vectors of length {self.m} but has shape {a.shape}")
            nq = a.shape[0]
            if k == 1:
                dd = np.full(nq, inf, dtype=float)
                ii = np.full(nq, self.n, dtype=int)
                for q in range(nq):
                    best = self._knn(a[q], 1, p, ub)
                    if len(best) > 0:
                        dd[q] = _root(best[0][0], p)
                        ii[q] = best[0][1]
                return dd, ii
            else:
                dd = np.full((nq, k), inf, dtype=float)
                ii = np.full((nq, k), self.n, dtype=int)
                for q in range(nq):
                    best = self._knn(a[q], k, p, ub)
                    for t in range(len(best)):
                        dd[q, t] = _root(best[t][0], p)
                        ii[q, t] = best[t][1]
                return dd, ii
        else:
            compile_error("x must be 1-D or 2-D")

    # ---- ball queries --------------------------------------------------

    def _ball(self, x, rp: float, p: float, out: List[int]):
        if self.n == 0:
            return
        stack = [0]
        while len(stack) > 0:
            node = stack.pop()
            if self._rect_min(node, x, p) > rp:
                continue
            if self._rect_max(node, x, p) <= rp:
                for t in range(self._start[node], self._end[node]):
                    out.append(self.indices[t])
            elif self._left[node] < 0:
                for t in range(self._start[node], self._end[node]):
                    i = self.indices[t]
                    if _pdist1(x, self.data[i], p) <= rp:
                        out.append(i)
            else:
                stack.append(self._right[node])
                stack.append(self._left[node])

    def query_ball_point(self, x, r: float, p: float = 2.0, eps: float = 0.0,
                         workers: int = 1, return_sorted = None,
                         return_length: Literal[bool] = False):
        """Find all points within distance r of point(s) x."""
        self._check_p(p)
        a = np.asarray(x, dtype=float)
        rp = _pow(r, p)
        if static.len(a.shape) == 1:
            if a.shape[0] != self.m:
                raise ValueError(f"Searching for a {a.shape[0]}-dimensional point in a {self.m}-dimensional KDTree.")
            out: List[int] = []
            self._ball(a, rp, p, out)
            if return_length:
                return len(out)
            else:
                out.sort()
                return out
        elif static.len(a.shape) == 2:
            if a.shape[1] != self.m:
                raise ValueError(f"Searching for a {a.shape[1]}-dimensional point in a {self.m}-dimensional KDTree.")
            if return_length:
                lens: List[int] = []
                for q in range(a.shape[0]):
                    o: List[int] = []
                    self._ball(a[q], rp, p, o)
                    lens.append(len(o))
                return np.array(lens)
            else:
                res: List[List[int]] = []
                for q in range(a.shape[0]):
                    o: List[int] = []
                    self._ball(a[q], rp, p, o)
                    o.sort()
                    res.append(o)
                return res
        else:
            compile_error("x must be 1-D or 2-D")

    def query_ball_tree(self, other: KDTree, r: float, p: float = 2.0, eps: float = 0.0):
        self._check_p(p)
        if self.m != other.m:
            raise ValueError("KDTrees must have the same dimension")
        rp = _pow(r, p)
        res: List[List[int]] = []
        for i in range(self.n):
            o: List[int] = []
            other._ball(self.data[i], rp, p, o)
            o.sort()
            res.append(o)
        return res

    def query_pairs(self, r: float, p: float = 2.0, eps: float = 0.0,
                    output_type: str = 'set'):
        """Find all pairs (i, j), i < j, of points within distance r."""
        self._check_p(p)
        if output_type != 'set':
            raise ValueError("Invalid output type")
        rp = _pow(r, p)
        s = Set[Tuple[int, int]]()
        for i in range(self.n):
            o: List[int] = []
            self._ball(self.data[i], rp, p, o)
            for j in o:
                if j > i:
                    s.add((i, j))
        return s

    def count_neighbors(self, other: KDTree, r, p: float = 2.0,
                        weights = None, cumulative: bool = True):
        """Count pairs (x1, x2) with distance(x1, x2) <= r."""
        self._check_p(p)
        if weights is not None:
            compile_error("weights are not supported")
        if not cumulative:
            raise ValueError("only cumulative=True is supported")
        if self.m != other.m:
            raise ValueError("KDTrees must have the same dimension")
        if isinstance(r, float) or isinstance(r, int):
            rp = _pow(float(r), p)
            c = 0
            for i in range(self.n):
                o: List[int] = []
                other._ball(self.data[i], rp, p, o)
                c += len(o)
            return c
        else:
            ra = np.asarray(r, dtype=float)
            flat = ra.ravel()
            counts = np.zeros(flat.shape[0], dtype=int)
            for t in range(flat.shape[0]):
                rp = _pow(flat[t], p)
                c = 0
                for i in range(self.n):
                    o: List[int] = []
                    other._ball(self.data[i], rp, p, o)
                    c += len(o)
                counts[t] = c
            return counts.reshape(ra.shape)

cKDTree = KDTree
