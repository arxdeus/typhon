# Ported from scipy 1.18.1 (BSD-3-Clause)

# Hierarchical clustering (scipy.cluster.hierarchy subset): linkage via
# scipy's MST / nearest-neighbour-chain / generic algorithms (identical
# tie-breaking), flat clusters, cophenetic distances, inconsistency
# statistics, dendrogram layout (no plotting), trees and cuts.

import numpy as np
import internal.static as static
from math import sqrt, inf
import scipy.spatial.distance as distance

_INF = inf

def _cidx(n: int, i: int, j: int) -> int:
    if i < j:
        return n * i - (i * (i + 1)) // 2 + (j - i - 1)
    return n * j - (j * (j + 1)) // 2 + (i - j - 1)

def _new_dist(method: int, d_xi: float, d_yi: float, d_xy: float,
              size_x: int, size_y: int, size_i: int) -> float:
    if method == 0:
        return min(d_xi, d_yi)
    elif method == 1:
        return max(d_xi, d_yi)
    elif method == 2:
        return (size_x * d_xi + size_y * d_yi) / (size_x + size_y)
    elif method == 3:
        return sqrt((((size_x * d_xi * d_xi) + (size_y * d_yi * d_yi)) -
                     (size_x * size_y * d_xy * d_xy) / (size_x + size_y)) /
                    (size_x + size_y))
    elif method == 4:
        return sqrt(0.5 * (d_xi * d_xi + d_yi * d_yi) - 0.25 * d_xy * d_xy)
    elif method == 5:
        t = 1.0 / (size_x + size_y + size_i)
        return sqrt((size_i + size_x) * t * d_xi * d_xi +
                    (size_i + size_y) * t * d_yi * d_yi -
                    size_i * t * d_xy * d_xy)
    else:
        return 0.5 * (d_xi + d_yi)

def _method_code(method: str) -> int:
    if method == 'single':
        return 0
    if method == 'complete':
        return 1
    if method == 'average':
        return 2
    if method == 'centroid':
        return 3
    if method == 'median':
        return 4
    if method == 'ward':
        return 5
    if method == 'weighted':
        return 6
    raise ValueError(f"Invalid method: {method}")

# ---- Z stored as flat list of 4-tuples rows -------------------------------

def _zget(Z) -> List[float]:
    a = np.ascontiguousarray(np.asarray(Z, dtype=float))
    if static.len(a.shape) != 2:
        compile_error("linkage matrix must be 2-D")
    if a.shape[1] != 4:
        raise ValueError("Linkage matrix 'Z' must have 4 columns.")
    out = List[float](a.size)
    for i in range(a.size):
        out.append(a.data[i])
    return out

def _zarr(Zl: List[float], m: int):
    out = np.empty((m, 4), dtype=float)
    for i in range(4 * m):
        out.data[i] = Zl[i]
    return out

def _uf_find(parent: List[int], x: int) -> int:
    p = x
    while parent[x] != x:
        x = parent[x]
    while parent[p] != x:
        q = parent[p]
        parent[p] = x
        p = q
    return x

def _sort_and_label(Zl: List[float], n: int) -> List[float]:
    m = n - 1
    # stable sort by distance (mergesort)
    order = list(range(m))
    order.sort(key=lambda r: Zl[4 * r + 2])
    S = List[float](4 * m)
    for r in order:
        for c in range(4):
            S.append(Zl[4 * r + c])
    parent = list(range(2 * n - 1))
    size = [1] * (2 * n - 1)
    nxt = n
    for i in range(m):
        x = int(S[4 * i])
        y = int(S[4 * i + 1])
        xr = _uf_find(parent, x)
        yr = _uf_find(parent, y)
        if xr < yr:
            S[4 * i] = float(xr)
            S[4 * i + 1] = float(yr)
        else:
            S[4 * i] = float(yr)
            S[4 * i + 1] = float(xr)
        parent[xr] = nxt
        parent[yr] = nxt
        sz = size[xr] + size[yr]
        size[nxt] = sz
        nxt += 1
        S[4 * i + 3] = float(sz)
    return S

def _mst_single(dists: List[float], n: int) -> List[float]:
    Z = [0.0] * (4 * (n - 1))
    merged = [False] * n
    D = [_INF] * n
    x = 0
    y = 0
    for k in range(n - 1):
        cur = _INF
        merged[x] = True
        for i in range(n):
            if merged[i]:
                continue
            dist = dists[_cidx(n, x, i)]
            if D[i] > dist:
                D[i] = dist
            if D[i] < cur:
                y = i
                cur = D[i]
        Z[4 * k] = float(x)
        Z[4 * k + 1] = float(y)
        Z[4 * k + 2] = cur
        x = y
    return _sort_and_label(Z, n)

def _nn_chain(dists: List[float], n: int, method: int) -> List[float]:
    Z = [0.0] * (4 * (n - 1))
    D = list(dists)
    size = [1] * n
    chain = [0] * n
    clen = 0
    y = 0
    x = 0
    cur = 0.0
    for k in range(n - 1):
        if clen == 0:
            clen = 1
            for i in range(n):
                if size[i] > 0:
                    chain[0] = i
                    break
        while True:
            x = chain[clen - 1]
            if clen > 1:
                y = chain[clen - 2]
                cur = D[_cidx(n, x, y)]
            else:
                cur = _INF
            for i in range(n):
                if size[i] == 0 or x == i:
                    continue
                dist = D[_cidx(n, x, i)]
                if dist < cur:
                    cur = dist
                    y = i
            if clen > 1 and y == chain[clen - 2]:
                break
            chain[clen] = y
            clen += 1
        clen -= 2
        if x > y:
            x, y = y, x
        nx = size[x]
        ny = size[y]
        Z[4 * k] = float(x)
        Z[4 * k + 1] = float(y)
        Z[4 * k + 2] = cur
        Z[4 * k + 3] = float(nx + ny)
        size[x] = 0
        size[y] = nx + ny
        for i in range(n):
            ni = size[i]
            if ni == 0 or i == y:
                continue
            D[_cidx(n, i, y)] = _new_dist(method, D[_cidx(n, i, x)],
                                          D[_cidx(n, i, y)], cur, nx, ny, ni)
    return _sort_and_label(Z, n)

class _Heap:
    index_by_key: List[int]
    key_by_index: List[int]
    values: List[float]
    size: int

    def __init__(self, values: List[float]):
        self.size = len(values)
        self.index_by_key = list(range(self.size))
        self.key_by_index = list(range(self.size))
        self.values = list(values)
        for i in reversed(range(self.size // 2)):
            self.sift_down(i)

    def remove_min(self):
        self.swap(0, self.size - 1)
        self.size -= 1
        self.sift_down(0)

    def change_value(self, key: int, value: float):
        index = self.index_by_key[key]
        old = self.values[index]
        self.values[index] = value
        if value < old:
            self.sift_up(index)
        else:
            self.sift_down(index)

    def sift_up(self, index: int):
        parent = (index - 1) >> 1
        while index > 0 and self.values[parent] > self.values[index]:
            self.swap(index, parent)
            index = parent
            parent = (index - 1) >> 1

    def sift_down(self, index: int):
        child = (index << 1) + 1
        while child < self.size:
            if child + 1 < self.size and self.values[child + 1] < self.values[child]:
                child += 1
            if self.values[index] > self.values[child]:
                self.swap(index, child)
                index = child
                child = (index << 1) + 1
            else:
                break

    def swap(self, i: int, j: int):
        self.values[i], self.values[j] = self.values[j], self.values[i]
        ki = self.key_by_index[i]
        kj = self.key_by_index[j]
        self.key_by_index[i] = kj
        self.key_by_index[j] = ki
        self.index_by_key[ki] = j
        self.index_by_key[kj] = i

def _find_min_dist(n: int, D: List[float], size: List[int], x: int) -> Tuple[int, float]:
    cur = _INF
    y = -1
    for i in range(x + 1, n):
        if size[i] == 0:
            continue
        dist = D[_cidx(n, x, i)]
        if dist < cur:
            cur = dist
            y = i
    if y == -1:
        raise ValueError("find_min_dist cannot find any neighbors closer than inf away. "
                         "Check that distances contain no negative/infinite/NaN entries. ")
    return (y, cur)

def _fast_linkage(dists: List[float], n: int, method: int) -> List[float]:
    Z = [0.0] * (4 * (n - 1))
    D = list(dists)
    size = [1] * n
    cluster_id = list(range(n))
    neighbor = [0] * (n - 1)
    min_dist = [0.0] * (n - 1)
    for x in range(n - 1):
        p = _find_min_dist(n, D, size, x)
        neighbor[x] = p[0]
        min_dist[x] = p[1]
    heap = _Heap(min_dist)
    x = 0
    y = 0
    dist = 0.0
    for k in range(n - 1):
        for i in range(n - k):
            x = heap.key_by_index[0]
            dist = heap.values[0]
            y = neighbor[x]
            if dist == D[_cidx(n, x, y)]:
                break
            p = _find_min_dist(n, D, size, x)
            y = p[0]
            dist = p[1]
            neighbor[x] = y
            min_dist[x] = dist
            heap.change_value(x, dist)
        heap.remove_min()
        id_x = cluster_id[x]
        id_y = cluster_id[y]
        nx = size[x]
        ny = size[y]
        if id_x > id_y:
            id_x, id_y = id_y, id_x
        Z[4 * k] = float(id_x)
        Z[4 * k + 1] = float(id_y)
        Z[4 * k + 2] = dist
        Z[4 * k + 3] = float(nx + ny)
        size[x] = 0
        size[y] = nx + ny
        cluster_id[y] = n + k
        for z in range(n):
            nz = size[z]
            if nz == 0 or z == y:
                continue
            D[_cidx(n, z, y)] = _new_dist(method, D[_cidx(n, z, x)],
                                          D[_cidx(n, z, y)], dist, nx, ny, nz)
        for z in range(x):
            if size[z] > 0 and neighbor[z] == x:
                neighbor[z] = y
        for z in range(y):
            if size[z] == 0:
                continue
            dd = D[_cidx(n, z, y)]
            if dd < min_dist[z]:
                neighbor[z] = y
                min_dist[z] = dd
                heap.change_value(z, dd)
        if y < n - 1:
            p = _find_min_dist(n, D, size, y)
            neighbor[y] = p[0]
            min_dist[y] = p[1]
            heap.change_value(y, p[1])
    return Z

def _num_obs_y(m: int) -> int:
    k = int(sqrt(m * 2.0)) + 1
    if k * (k - 1) != 2 * m:
        raise ValueError("Incompatible vector size. It must be a binomial coefficient n choose 2 for some integer n >= 2.")
    return k

def _condensed(y, metric) -> List[float]:
    a = np.asarray(y, dtype=float)
    if static.len(a.shape) == 1:
        out = List[float](a.shape[0])
        for i in range(a.shape[0]):
            out.append(a[i])
        return out
    elif static.len(a.shape) == 2:
        d = distance.pdist(a, metric)
        out = List[float](d.shape[0])
        for i in range(d.shape[0]):
            out.append(d[i])
        return out
    else:
        compile_error("`y` must be 1 or 2 dimensional.")

def linkage(y, method: str = 'single', metric = 'euclidean', optimal_ordering: bool = False):
    code = _method_code(method)
    a = np.asarray(y, dtype=float)
    if static.len(a.shape) == 2:
        if isinstance(metric, str):
            if (method == 'centroid' or method == 'median' or method == 'ward') and metric != 'euclidean':
                raise ValueError(f"`method={method}` requires the distance metric to be Euclidean")
    D = _condensed(a, metric)
    for v in D:
        if not (v - v == 0.0):
            raise ValueError("The condensed distance matrix must contain only finite values.")
    n = _num_obs_y(len(D))
    Zl = List[float]()
    if code == 0:
        Zl = _mst_single(D, n)
    elif code == 1 or code == 2 or code == 6 or code == 5:
        Zl = _nn_chain(D, n, code)
    else:
        Zl = _fast_linkage(D, n, code)
    Z = _zarr(Zl, n - 1)
    if optimal_ordering:
        return optimal_leaf_ordering(Z, D)
    return Z

def single(y):
    return linkage(y, method='single')
def complete(y):
    return linkage(y, method='complete')
def average(y):
    return linkage(y, method='average')
def weighted(y):
    return linkage(y, method='weighted')
def centroid(y):
    return linkage(y, method='centroid')
def median(y):
    return linkage(y, method='median')
def ward(y):
    return linkage(y, method='ward')

# ---- validity ------------------------------------------------------------

def num_obs_linkage(Z) -> int:
    a = np.asarray(Z)
    return a.shape[0] + 1

def _valid_linkage_msg(Zl: List[float], m: int) -> str:
    n = m + 1
    if m < 1:
        return "Linkage must be computed on at least two observations."
    for i in range(m):
        a = Zl[4 * i]
        b = Zl[4 * i + 1]
        if a < 0 or b < 0:
            return "Linkage contains negative indices."
        if Zl[4 * i + 2] < 0:
            return "Linkage contains negative distances."
        if Zl[4 * i + 3] < 0:
            return "Linkage contains negative counts."
        if a >= n + i or b >= n + i:
            return "Linkage uses non-singleton cluster before it is formed."
    seen = set[int]()
    for i in range(m):
        for c in range(2):
            v = int(Zl[4 * i + c])
            if v in seen:
                return "Linkage uses the same cluster more than once."
            seen.add(v)
    return ""

def is_valid_linkage(Z, warning: bool = False, throw: bool = False, name = None) -> bool:
    a = np.asarray(Z, dtype=float)
    if static.len(a.shape) != 2:
        if throw:
            raise TypeError("Linkage matrix must be 2-dimensional.")
        return False
    if a.shape[1] != 4:
        if throw:
            raise ValueError("Linkage matrix must have 4 columns.")
        return False
    msg = _valid_linkage_msg(_zget(a), a.shape[0])
    if msg != "":
        if throw:
            raise ValueError(msg)
        return False
    return True

def is_monotonic(Z) -> bool:
    Zl = _zget(Z)
    m = len(Zl) // 4
    for i in range(1, m):
        if Zl[4 * i + 2] < Zl[4 * (i - 1) + 2]:
            return False
    return True

def is_valid_im(R, warning: bool = False, throw: bool = False, name = None) -> bool:
    a = np.asarray(R, dtype=float)
    if a.shape[1] != 4:
        if throw:
            raise ValueError("Inconsistency matrix must have 4 columns.")
        return False
    for i in range(a.shape[0]):
        if a[i, 0] < 0 or a[i, 1] < 0 or a[i, 2] < 0:
            if throw:
                raise ValueError("Inconsistency matrix contains negative values.")
            return False
    return True

# ---- tree traversals (ports of _hierarchy.pyx) ---------------------------

def _max_dist_each(Zl: List[float], n: int) -> List[float]:
    MD = [0.0] * (n - 1)
    curr = [0] * n
    visited = [False] * (2 * n - 1)
    k = 0
    curr[0] = 2 * n - 2
    while k >= 0:
        root = curr[k] - n
        lc = int(Zl[4 * root])
        rc = int(Zl[4 * root + 1])
        if lc >= n and not visited[lc]:
            visited[lc] = True
            k += 1
            curr[k] = lc
            continue
        if rc >= n and not visited[rc]:
            visited[rc] = True
            k += 1
            curr[k] = rc
            continue
        md = Zl[4 * root + 2]
        if lc >= n and MD[lc - n] > md:
            md = MD[lc - n]
        if rc >= n and MD[rc - n] > md:
            md = MD[rc - n]
        MD[root] = md
        k -= 1
    return MD

def _max_rfield_each(Zl: List[float], Rl: List[float], n: int, rf: int) -> List[float]:
    MD = [0.0] * (n - 1)
    curr = [0] * n
    visited = [False] * (2 * n - 1)
    k = 0
    curr[0] = 2 * n - 2
    while k >= 0:
        root = curr[k] - n
        lc = int(Zl[4 * root])
        rc = int(Zl[4 * root + 1])
        if lc >= n and not visited[lc]:
            visited[lc] = True
            k += 1
            curr[k] = lc
            continue
        if rc >= n and not visited[rc]:
            visited[rc] = True
            k += 1
            curr[k] = rc
            continue
        md = Rl[4 * root + rf]
        if lc >= n and MD[lc - n] > md:
            md = MD[lc - n]
        if rc >= n and MD[rc - n] > md:
            md = MD[rc - n]
        MD[root] = md
        k -= 1
    return MD

def _cluster_monocrit(Zl: List[float], MC: List[float], T: List[int], cutoff: float, n: int):
    n_cluster = 0
    leader = -1
    curr = [0] * n
    visited = [False] * (2 * n - 1)
    k = 0
    curr[0] = 2 * n - 2
    while k >= 0:
        root = curr[k] - n
        lc = int(Zl[4 * root])
        rc = int(Zl[4 * root + 1])
        if leader == -1 and MC[root] <= cutoff:
            leader = root
            n_cluster += 1
        if lc >= n and not visited[lc]:
            visited[lc] = True
            k += 1
            curr[k] = lc
            continue
        if rc >= n and not visited[rc]:
            visited[rc] = True
            k += 1
            curr[k] = rc
            continue
        if lc < n:
            if leader == -1:
                n_cluster += 1
            T[lc] = n_cluster
        if rc < n:
            if leader == -1:
                n_cluster += 1
            T[rc] = n_cluster
        if leader == root:
            leader = -1
        k -= 1

def _cluster_maxclust_monocrit(Zl: List[float], MC: List[float], T: List[int], n: int, max_nc: int):
    if max_nc >= n:
        for i in range(n):
            T[i] = i + 1
        return
    curr = [0] * n
    lower = -1
    upper = n - 1
    while upper - lower > 1:
        i = (lower + upper) >> 1
        thresh = MC[i]
        visited = [False] * (2 * n - 1)
        nc = 0
        k = 0
        curr[0] = 2 * n - 2
        while k >= 0:
            root = curr[k] - n
            lc = int(Zl[4 * root])
            rc = int(Zl[4 * root + 1])
            if MC[root] <= thresh:
                nc += 1
                if nc > max_nc:
                    break
                k -= 1
                visited[lc] = True
                visited[rc] = True
                continue
            if not visited[lc]:
                visited[lc] = True
                if lc >= n:
                    k += 1
                    curr[k] = lc
                    continue
                else:
                    nc += 1
                    if nc > max_nc:
                        break
            if not visited[rc]:
                visited[rc] = True
                if rc >= n:
                    k += 1
                    curr[k] = rc
                    continue
                else:
                    nc += 1
                    if nc > max_nc:
                        break
            k -= 1
        if nc > max_nc:
            lower = i
        else:
            upper = i
    _cluster_monocrit(Zl, MC, T, MC[upper], n)

def _split(a: float) -> Tuple[float, float]:
    c = 134217729.0 * a
    hi = c - (c - a)
    return (hi, a - hi)

def _prod_err(a: float, b: float) -> float:
    # exact rounding error of a*b (Dekker two-product)
    p = a * b
    x = _split(a)
    y = _split(b)
    return ((x[0] * y[0] - p) + x[0] * y[1] + x[1] * y[0]) + x[1] * y[1]

def _inconsistent_l(Zl: List[float], n: int, d: int) -> List[float]:
    R = [0.0] * (4 * (n - 1))
    curr = [0] * n
    for i in range(n - 1):
        k = 0
        cnt = 0
        s = 0.0
        ss = 0.0
        visited = [False] * (2 * n - 1)
        curr[0] = i
        while k >= 0:
            root = curr[k]
            if k < d - 1:
                lc = int(Zl[4 * root])
                if lc >= n and not visited[lc]:
                    visited[lc] = True
                    k += 1
                    curr[k] = lc - n
                    continue
                rc = int(Zl[4 * root + 1])
                if rc >= n and not visited[rc]:
                    visited[rc] = True
                    k += 1
                    curr[k] = rc - n
                    continue
            dist = Zl[4 * root + 2]
            cnt += 1
            s += dist
            ss += dist * dist
            k -= 1
        R[4 * i] = s / cnt
        R[4 * i + 2] = float(cnt)
        std = 0.0
        if cnt < 2:
            # scipy's compiled kernel contracts this to fma(-s, s, ss)
            std = (ss - s * s - _prod_err(s, s)) / cnt
        else:
            std = (ss - ((s * s) / cnt)) / (cnt - 1)
        if std > 0:
            std = sqrt(std)
            R[4 * i + 1] = std
            R[4 * i + 3] = (Zl[4 * i + 2] - R[4 * i]) / std
        else:
            R[4 * i + 1] = 0.0
    return R

def _prelist(Zl: List[float], n: int) -> List[int]:
    members = List[int](n)
    curr = [0] * n
    visited = [False] * (2 * n - 1)
    k = 0
    curr[0] = 2 * n - 2
    while k >= 0:
        root = curr[k] - n
        lc = int(Zl[4 * root])
        if not visited[lc]:
            visited[lc] = True
            if lc >= n:
                k += 1
                curr[k] = lc
                continue
            else:
                members.append(lc)
        rc = int(Zl[4 * root + 1])
        if not visited[rc]:
            visited[rc] = True
            if rc >= n:
                k += 1
                curr[k] = rc
                continue
            else:
                members.append(rc)
        k -= 1
    return members

def _cophenetic_l(Zl: List[float], n: int) -> List[float]:
    d = [0.0] * (n * (n - 1) // 2)
    curr = [0] * n
    members = [0] * n
    left_start = [0] * n
    visited = [False] * (2 * n - 1)
    k = 0
    curr[0] = 2 * n - 2
    while k >= 0:
        root = curr[k] - n
        lc = int(Zl[4 * root])
        rc = int(Zl[4 * root + 1])
        n_lc = 1
        n_rc = 1
        if lc >= n:
            n_lc = int(Zl[4 * (lc - n) + 3])
            if not visited[lc]:
                visited[lc] = True
                k += 1
                curr[k] = lc
                left_start[k] = left_start[k - 1]
                continue
        else:
            members[left_start[k]] = lc
        if rc >= n:
            n_rc = int(Zl[4 * (rc - n) + 3])
            if not visited[rc]:
                visited[rc] = True
                k += 1
                curr[k] = rc
                left_start[k] = left_start[k - 1] + n_lc
                continue
        else:
            members[left_start[k] + n_lc] = rc
        dist = Zl[4 * root + 2]
        rs = left_start[k] + n_lc
        for i in range(left_start[k], rs):
            for j in range(rs, rs + n_rc):
                d[_cidx(n, members[i], members[j])] = dist
        k -= 1
    return d

def _farr(L: List[float]):
    out = np.empty(len(L), dtype=float)
    for i in range(len(L)):
        out[i] = L[i]
    return out

def _iarr32(L: List[int]):
    out = np.empty(len(L), dtype=np.int32)
    for i in range(len(L)):
        out[i] = np.int32(L[i])
    return out

def _checked(Z) -> List[float]:
    Zl = _zget(Z)
    msg = _valid_linkage_msg(Zl, len(Zl) // 4)
    if msg != "":
        raise ValueError(msg)
    return Zl

# ---- public API -----------------------------------------------------------

def inconsistent(Z, d: int = 2):
    if d < 0:
        raise ValueError(f"The second argument d must be a nonnegative integer value.")
    Zl = _checked(Z)
    m = len(Zl) // 4
    return _zarr(_inconsistent_l(Zl, m + 1, d), m)

def maxdists(Z):
    Zl = _checked(Z)
    return _farr(_max_dist_each(Zl, len(Zl) // 4 + 1))

def maxinconsts(Z, R):
    Zl = _checked(Z)
    Rl = _zget(R)
    if len(Rl) != len(Zl):
        raise ValueError("The inconsistency matrix and linkage matrix each have a different number of rows.")
    return _farr(_max_rfield_each(Zl, Rl, len(Zl) // 4 + 1, 3))

def maxRstat(Z, R, i: int):
    Zl = _checked(Z)
    Rl = _zget(R)
    if i < 0 or i > 3:
        raise ValueError('The third argument must be an integer between 0 and 3 inclusive.')
    return _farr(_max_rfield_each(Zl, Rl, len(Zl) // 4 + 1, i))

def fcluster(Z, t, criterion: str = 'inconsistent', depth: int = 2, R = None, monocrit = None):
    Zl = _checked(Z)
    n = len(Zl) // 4 + 1
    T = [0] * n
    if criterion == 'inconsistent':
        Rl = List[float]()
        if R is None:
            Rl = _inconsistent_l(Zl, n, depth)
        else:
            Rl = _zget(R)
        _cluster_monocrit(Zl, _max_rfield_each(Zl, Rl, n, 3), T, float(t), n)
    elif criterion == 'distance':
        _cluster_monocrit(Zl, _max_dist_each(Zl, n), T, float(t), n)
    elif criterion == 'maxclust':
        _cluster_maxclust_monocrit(Zl, _max_dist_each(Zl, n), T, n, int(t))
    elif criterion == 'monocrit' or criterion == 'maxclust_monocrit':
        if monocrit is None:
            raise ValueError("monocrit must be given")
        else:
            mc = np.asarray(monocrit, dtype=float)
            MC = [float(mc[i]) for i in range(mc.shape[0])]
            if criterion == 'monocrit':
                _cluster_monocrit(Zl, MC, T, float(t), n)
            else:
                _cluster_maxclust_monocrit(Zl, MC, T, n, int(t))
    else:
        raise ValueError(f'Invalid cluster formation criterion: {criterion}')
    return _iarr32(T)

def fclusterdata(X, t, criterion: str = 'inconsistent', metric = 'euclidean',
                 depth: int = 2, method: str = 'single', R = None):
    Y = distance.pdist(np.asarray(X, dtype=float), metric)
    Z = linkage(Y, method=method)
    if R is None:
        return fcluster(Z, t, criterion=criterion, depth=depth, R=inconsistent(Z, depth))
    else:
        return fcluster(Z, t, criterion=criterion, depth=depth, R=R)

def leaves_list(Z):
    Zl = _checked(Z)
    return _iarr32(_prelist(Zl, len(Zl) // 4 + 1))

def cophenet(Z, Y = None):
    Zl = _checked(Z)
    n = len(Zl) // 4 + 1
    zz = _farr(_cophenetic_l(Zl, n))
    if Y is None:
        return zz
    else:
        Ya = np.asarray(Y, dtype=float)
        z = np.mean(zz)
        y = np.mean(Ya)
        Yy = Ya - y
        Zz = zz - z
        c = np.sum(Yy * Zz) / np.sqrt(np.sum(Yy ** 2) * np.sum(Zz ** 2))
        return (float(c), zz)

def leaders(Z, T):
    Zl = _checked(Z)
    n = len(Zl) // 4 + 1
    Ta = np.asarray(T)
    if Ta.shape[0] != n:
        raise ValueError('Mismatch: len(T)!=Z.shape[0] + 1.')
    ids = [-1] * (2 * n - 1)
    uniq = set[int]()
    for i in range(n):
        ids[i] = int(Ta[i])
        uniq.add(ids[i])
    nc = len(uniq)
    L = List[int]()
    M = List[int]()
    curr = [0] * n
    visited = [False] * (2 * n - 1)
    k = 0
    curr[0] = 2 * n - 2
    result = -1
    while k >= 0:
        root = curr[k] - n
        lc = int(Zl[4 * root])
        rc = int(Zl[4 * root + 1])
        if lc >= n and not visited[lc]:
            visited[lc] = True
            k += 1
            curr[k] = lc
            continue
        if rc >= n and not visited[rc]:
            visited[rc] = True
            k += 1
            curr[k] = rc
            continue
        cl = ids[lc]
        cr = ids[rc]
        if cl == cr:
            ids[root + n] = cl
        else:
            if cl != -1:
                if len(L) >= nc:
                    result = root + n
                    break
                L.append(lc)
                M.append(cl)
            if cr != -1:
                if len(L) >= nc:
                    result = root + n
                    break
                L.append(rc)
                M.append(cr)
            ids[root + n] = -1
        k -= 1
    if result == -1:
        lc = int(Zl[4 * (n - 2)])
        rc = int(Zl[4 * (n - 2) + 1])
        cl = ids[lc]
        cr = ids[rc]
        if cl == cr and cl != -1:
            if len(L) >= nc:
                result = 2 * n - 2
            else:
                L.append(2 * n - 2)
                M.append(cl)
    if result >= 0:
        raise ValueError(f'T is not a valid assignment vector. Error found when examining linkage node {result} (< 2n-1).')
    return (_iarr32(L), _iarr32(M))

# ---- trees ----------------------------------------------------------------

class ClusterNode:
    id: int
    left: Optional[ClusterNode]
    right: Optional[ClusterNode]
    dist: float
    count: int

    def __init__(self, id: int, left: Optional[ClusterNode] = None,
                 right: Optional[ClusterNode] = None, dist: float = 0.0, count: int = 1):
        if id < 0:
            raise ValueError('The id must be non-negative.')
        if dist < 0:
            raise ValueError('The distance must be non-negative.')
        if (left is None and right is not None) or (left is not None and right is None):
            raise ValueError('Only full or proper binary trees are permitted.  This node has one child.')
        if count < 1:
            raise ValueError('A cluster must contain at least one original observation.')
        self.id = id
        self.left = left
        self.right = right
        self.dist = dist
        if left is None:
            self.count = count
        else:
            self.count = left.count + right.count

    def __lt__(self, node: ClusterNode) -> bool:
        return self.dist < node.dist
    def __gt__(self, node: ClusterNode) -> bool:
        return self.dist > node.dist
    def __eq__(self, node: ClusterNode) -> bool:
        return self.dist == node.dist

    def get_id(self) -> int:
        return self.id
    def get_count(self) -> int:
        return self.count
    def get_left(self) -> Optional[ClusterNode]:
        return self.left
    def get_right(self) -> Optional[ClusterNode]:
        return self.right
    def is_leaf(self) -> bool:
        return self.left is None

    def pre_order(self, func = None):
        stack = [self]
        if func is None:
            out = List[int]()
            while len(stack) > 0:
                nd = stack.pop()
                if nd.left is None:
                    out.append(nd.id)
                else:
                    stack.append(nd.right)
                    stack.append(nd.left)
            return out
        else:
            out2 = []
            while len(stack) > 0:
                nd = stack.pop()
                if nd.left is None:
                    out2.append(func(nd))
                else:
                    stack.append(nd.right)
                    stack.append(nd.left)
            return out2

    def __repr__(self) -> str:
        return f"<ClusterNode id={self.id} count={self.count} dist={self.dist}>"

def _to_tree_all(Zl: List[float]) -> List[ClusterNode]:
    m = len(Zl) // 4
    n = m + 1
    d = List[ClusterNode](2 * n - 1)
    for i in range(n):
        d.append(ClusterNode(i))
    for i in range(m):
        fi = int(Zl[4 * i])
        fj = int(Zl[4 * i + 1])
        if fi > i + n:
            raise ValueError(f'Corrupt matrix Z. Index to derivative cluster is used before it is formed. See row {fi}, column 0')
        if fj > i + n:
            raise ValueError(f'Corrupt matrix Z. Index to derivative cluster is used before it is formed. See row {fj}, column 1')
        nd = ClusterNode(i + n, d[fi], d[fj], Zl[4 * i + 2])
        if int(Zl[4 * i + 3]) != nd.count:
            raise ValueError(f'Corrupt matrix Z. The count Z[{i},3] is incorrect.')
        d.append(nd)
    return d

def to_tree(Z, rd: Literal[bool] = False):
    d = _to_tree_all(_checked(Z))
    if rd:
        return (d[len(d) - 1], d)
    else:
        return d[len(d) - 1]

def _order_cluster_tree(Zl: List[float]) -> List[ClusterNode]:
    d = _to_tree_all(Zl)
    q = [d[len(d) - 1]]
    head = 0
    nodes = List[ClusterNode]()
    while head < len(q):
        node = q[head]
        head += 1
        if not node.is_leaf():
            # bisect.insort_left by dist
            lo = 0
            hi = len(nodes)
            while lo < hi:
                mid = (lo + hi) // 2
                if nodes[mid].dist < node.dist:
                    lo = mid + 1
                else:
                    hi = mid
            nodes.insert(lo, node)
            q.append(node.right)
            q.append(node.left)
    return nodes

def cut_tree(Z, n_clusters = None, height = None):
    Zl = _checked(Z)
    nobs = len(Zl) // 4 + 1
    nodes = _order_cluster_tree(Zl)
    cols = List[int]()
    if height is not None and n_clusters is not None:
        raise ValueError("At least one of either height or n_clusters must be None")
    if height is None and n_clusters is None:
        cols = list(range(nobs))
    elif height is not None:
        hs = np.atleast_1d(np.asarray(height, dtype=float))
        for h in hs:
            lo = 0
            hi = len(nodes)
            while lo < hi:
                mid = (lo + hi) // 2
                if nodes[mid].dist < h:
                    lo = mid + 1
                else:
                    hi = mid
            cols.append(lo)
    elif n_clusters is not None:
        cs = np.atleast_1d(np.asarray(n_clusters))
        for c in cs:
            # nobs - searchsorted(arange(nobs), c)
            v = int(c)
            p = 0 if v < 0 else (nobs if v > nobs else v)
            cols.append(nobs - p)
    ncols = len(cols)
    groups = np.zeros((ncols, nobs), dtype=int)
    last = list(range(nobs))
    for c in range(ncols):
        if cols[c] == 0:
            for j in range(nobs):
                groups[c, j] = last[j]
    for i in range(len(nodes)):
        idx = nodes[i].pre_order()
        this = list(last)
        mn = last[idx[0]]
        mx = last[idx[0]]
        for t in idx:
            if last[t] < mn:
                mn = last[t]
            if last[t] > mx:
                mx = last[t]
        for t in idx:
            this[t] = mn
        for j in range(nobs):
            if this[j] > mx:
                this[j] -= 1
        for c in range(ncols):
            if cols[c] == i + 1:
                for j in range(nobs):
                    groups[c, j] = this[j]
        last = this
    return groups.T.copy()

def optimal_leaf_ordering(Z, y, metric = 'euclidean'):
    raise NotImplementedError("optimal_leaf_ordering is not supported")
    return np.asarray(Z, dtype=float)

def correspond(Z, Y) -> bool:
    return num_obs_linkage(Z) == distance.num_obs_y(Y)

def is_isomorphic(T1, T2) -> bool:
    a = np.asarray(T1)
    b = np.asarray(T2)
    if a.shape[0] != b.shape[0]:
        raise ValueError('T1 and T2 must have the same number of elements.')
    d1 = Dict[int, int]()
    d2 = Dict[int, int]()
    for i in range(a.shape[0]):
        x = int(a[i])
        y = int(b[i])
        if x in d1:
            if y not in d2:
                return False
            if d1[x] != y or d2[y] != x:
                return False
        elif y in d2:
            return False
        else:
            d1[x] = y
            d2[y] = x
    return True

# ---- dendrogram -----------------------------------------------------------

_link_line_colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

def set_link_color_palette(palette = None):
    _link_line_colors.clear()
    if palette is None:
        for c in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']:
            _link_line_colors.append(c)
    else:
        for c in palette:
            _link_line_colors.append(c)

class Dendrogram:
    icoord: List[List[float]]
    dcoord: List[List[float]]
    ivl: List[str]
    leaves: List[int]
    color_list: List[str]
    leaves_color_list: List[str]

    def __init__(self):
        self.icoord = List[List[float]]()
        self.dcoord = List[List[float]]()
        self.ivl = List[str]()
        self.leaves = List[int]()
        self.color_list = List[str]()
        self.leaves_color_list = List[str]()

    def __getitem__(self, key: Literal[str]):
        if key == 'icoord':
            return self.icoord
        elif key == 'dcoord':
            return self.dcoord
        elif key == 'ivl':
            return self.ivl
        elif key == 'leaves':
            return self.leaves
        elif key == 'color_list':
            return self.color_list
        elif key == 'leaves_color_list':
            return self.leaves_color_list
        else:
            compile_error("unknown dendrogram key")

    def __contains__(self, key: str) -> bool:
        return key in ('icoord', 'dcoord', 'ivl', 'leaves', 'color_list', 'leaves_color_list')

    def keys(self) -> List[str]:
        return ['icoord', 'dcoord', 'ivl', 'leaves', 'color_list', 'leaves_color_list']

class _DState:
    Zl: List[float]
    p: int
    trunc: str
    color_threshold: float
    labels: Optional[List[str]]
    count_sort: str
    distance_sort: str
    show_leaf_counts: bool
    n: int
    R: Dendrogram
    current_color: int
    below: bool
    above_color: str

    def __init__(self, Zl: List[float], p: int, trunc: str, ct: float,
                 labels: Optional[List[str]], count_sort: str, distance_sort: str,
                 show_leaf_counts: bool, n: int, above_color: str):
        self.Zl = Zl
        self.p = p
        self.trunc = trunc
        self.color_threshold = ct
        self.labels = labels
        self.count_sort = count_sort
        self.distance_sort = distance_sort
        self.show_leaf_counts = show_leaf_counts
        self.n = n
        self.R = Dendrogram()
        self.current_color = 0
        self.below = False
        self.above_color = above_color

def _d_single(S: _DState, i: int):
    S.R.leaves.append(i)
    lab = S.labels
    if lab is not None:
        S.R.ivl.append(lab[i])
    else:
        S.R.ivl.append(str(i))

def _d_nonsingle(S: _DState, i: int):
    S.R.leaves.append(i)
    if S.show_leaf_counts:
        S.R.ivl.append("(" + str(int(S.Zl[4 * (i - S.n) + 3])) + ")")
    else:
        S.R.ivl.append("")

def _d_info(S: _DState, i: int, iv: float, level: int) -> Tuple[float, float, float, float]:
    n = S.n
    Zl = S.Zl
    if S.trunc == 'lastp':
        if 2 * n - S.p > i and i >= n:
            d = Zl[4 * (i - n) + 2]
            _d_nonsingle(S, i)
            return (iv + 5.0, 10.0, 0.0, d)
        elif i < n:
            _d_single(S, i)
            return (iv + 5.0, 10.0, 0.0, 0.0)
    elif S.trunc == 'level':
        if i > n and level > S.p:
            d = Zl[4 * (i - n) + 2]
            _d_nonsingle(S, i)
            return (iv + 5.0, 10.0, 0.0, d)
        elif i < n:
            _d_single(S, i)
            return (iv + 5.0, 10.0, 0.0, 0.0)
    if i < n:
        _d_single(S, i)
        return (iv + 5.0, 10.0, 0.0, 0.0)
    aa = int(Zl[4 * (i - n)])
    ab = int(Zl[4 * (i - n) + 1])
    na = 1.0
    da = 0.0
    nb = 1.0
    db = 0.0
    if aa >= n:
        na = Zl[4 * (aa - n) + 3]
        da = Zl[4 * (aa - n) + 2]
    if ab >= n:
        nb = Zl[4 * (ab - n) + 3]
        db = Zl[4 * (ab - n) + 2]
    ua = aa
    ub = ab
    if S.count_sort == 'ascending':
        if na > nb:
            ua = ab
            ub = aa
    elif S.count_sort == 'descending':
        if not (na > nb):
            ua = ab
            ub = aa
    elif S.distance_sort == 'ascending':
        if da > db:
            ua = ab
            ub = aa
    elif S.distance_sort == 'descending':
        if not (da > db):
            ua = ab
            ub = aa
    ra = _d_info(S, ua, iv, level + 1)
    h = Zl[4 * (i - n) + 2]
    c = ''
    if h >= S.color_threshold or S.color_threshold <= 0:
        c = S.above_color
        if S.below:
            S.current_color = (S.current_color + 1) % len(_link_line_colors)
        S.below = False
    else:
        S.below = True
        c = _link_line_colors[S.current_color]
    rb = _d_info(S, ub, iv + ra[1], level + 1)
    max_dist = max(ra[3], rb[3], h)
    S.R.icoord.append([ra[0], ra[0], rb[0], rb[0]])
    S.R.dcoord.append([ra[2], h, h, rb[2]])
    S.R.color_list.append(c)
    return ((ra[0] + rb[0]) / 2, ra[1] + rb[1], h, max_dist)

def _sortflag(v) -> str:
    if isinstance(v, bool):
        return 'ascending' if v else ''
    elif isinstance(v, str):
        return v
    else:
        return ''

def dendrogram(Z, p: int = 30, truncate_mode: Optional[str] = None, color_threshold = None,
               get_leaves: bool = True, orientation: str = 'top', labels = None,
               count_sort = False, distance_sort = False, show_leaf_counts: bool = True,
               no_plot: bool = False, no_labels: bool = False, leaf_font_size = None,
               leaf_rotation = None, leaf_label_func = None, show_contracted: bool = False,
               link_color_func = None, ax = None, above_threshold_color: str = 'C0'):
    if orientation not in ("top", "left", "bottom", "right"):
        raise ValueError("orientation must be one of 'top', 'left', 'bottom', or 'right'")
    if not no_plot:
        raise NotImplementedError("dendrogram plotting is not supported; pass no_plot=True")
    Zl = _checked(Z)
    n = len(Zl) // 4 + 1
    lab: Optional[List[str]] = None
    if labels is not None:
        ll = List[str]()
        for x in labels:
            ll.append(str(x))
        if len(ll) != n:
            raise ValueError("Dimensions of Z and labels must be consistent.")
        lab = ll
    tm = ''
    if truncate_mode is not None:
        tm = truncate_mode
    if tm not in ('lastp', 'mtica', 'level', 'none', ''):
        raise ValueError('Invalid truncation mode.')
    pp = p
    if tm == 'lastp':
        if pp > n or pp == 0:
            pp = n
    if tm == 'mtica':
        tm = 'level'
    if tm == 'level':
        if pp <= 0:
            pp = 1 << 62
    ct = 0.0
    if color_threshold is None:
        mx = Zl[2]
        for k in range(n - 1):
            if Zl[4 * k + 2] > mx:
                mx = Zl[4 * k + 2]
        ct = mx * 0.7
    elif isinstance(color_threshold, str):
        mx = Zl[2]
        for k in range(n - 1):
            if Zl[4 * k + 2] > mx:
                mx = Zl[4 * k + 2]
        ct = mx * 0.7
    else:
        ct = float(color_threshold)
    S = _DState(Zl, pp, tm, ct, lab, _sortflag(count_sort), _sortflag(distance_sort),
                show_leaf_counts, n, above_threshold_color)
    _d_info(S, 2 * n - 2, 0.0, 0)
    R = S.R
    lcl = [''] * len(R.leaves)
    for t in range(len(R.icoord)):
        for q in range(4):
            xi = R.icoord[t][q]
            yi = R.dcoord[t][q]
            if yi == 0.0 and (xi % 5 == 0 and xi % 2 == 1):
                lcl[(int(xi) - 5) // 10] = R.color_list[t]
    R.leaves_color_list = lcl
    return R
