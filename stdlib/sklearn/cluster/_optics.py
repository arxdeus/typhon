# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# OPTICS (exact brute-force neighborhoods), compute_optics_graph,
# cluster_optics_dbscan and cluster_optics_xi (steep-area cluster
# extraction with predecessor correction).

from ..base import estimator_repr as _estimator_repr
import numpy as np
from ._kmeans import _as2d

def _pdist_row(X: np.ndarray[float, 2], i: int, metric: str, p: float) -> np.ndarray[float, 1]:
    n, d = X.shape
    out = np.empty(n)
    for j in range(n):
        if metric == "precomputed":
            out[j] = X[i, j]
            continue
        r = 0.0
        if metric == "euclidean" or metric == "l2" or (metric == "minkowski" and p == 2.0):
            for k in range(d):
                t = X[i, k] - X[j, k]
                r += t * t
            r = np.sqrt(r)
        elif metric == "manhattan" or metric == "cityblock" or metric == "l1" or (metric == "minkowski" and p == 1.0):
            for k in range(d):
                r += abs(X[i, k] - X[j, k])
        elif metric == "chebyshev":
            for k in range(d):
                r = max(r, abs(X[i, k] - X[j, k]))
        elif metric == "minkowski":
            for k in range(d):
                r += abs(X[i, k] - X[j, k]) ** p
            r = r ** (1.0 / p)
        else:
            raise ValueError(f"OPTICS: unsupported metric {metric}")
        out[j] = r
    return out

def _round15(x: float) -> float:
    if x == np.inf or x != x:
        return x
    return float(np.around(np.array([x]), 15)[0])

def compute_optics_graph(X, *, min_samples = 5, max_eps: float = np.inf, metric: str = "minkowski",
                         p: float = 2.0, metric_params = None, algorithm: str = "auto",
                         leaf_size: int = 30, n_jobs = None):
    Xa = _as2d(X)
    n = Xa.shape[0]
    ms = 0
    if isinstance(min_samples, float):
        if min_samples > 1.0:
            raise ValueError("min_samples as float must be in (0, 1]")
        ms = max(2, int(min_samples * n))
    else:
        ms = int(min_samples)
        if ms > n:
            raise ValueError(f"min_samples must be no greater than the number of samples ({n}). Got {ms}")
    reach = np.full(n, np.inf)
    pred = np.full(n, -1, dtype=int)
    D = np.empty((n, n))
    for i in range(n):
        D[i] = _pdist_row(Xa, i, metric, p)
    core = np.empty(n)
    for i in range(n):
        row = np.sort(D[i])
        c = row[ms - 1]
        core[i] = np.inf if c > max_eps else _round15(c)
    processed = np.zeros(n, dtype=bool)
    ordering = np.zeros(n, dtype=int)
    for oi in range(n):
        point = -1
        best = np.inf
        for q in range(n):
            if not processed[q]:
                if point < 0 or reach[q] < best:
                    point = q
                    best = reach[q]
        processed[point] = True
        ordering[oi] = point
        if core[point] != np.inf:
            for j in range(n):
                if processed[j] or D[point, j] > max_eps:
                    continue
                rd = _round15(max(D[point, j], core[point]))
                if rd < reach[j]:
                    reach[j] = rd
                    pred[j] = point
    return ordering, core, reach, pred

def cluster_optics_dbscan(*, reachability, core_distances, ordering, eps: float):
    n = core_distances.shape[0]
    labels = np.zeros(n, dtype=int)
    c = -1
    for t in range(n):
        i = int(ordering[t])
        if reachability[i] > eps and core_distances[i] <= eps:
            c += 1
        labels[i] = c
    for i in range(n):
        if reachability[i] > eps and not (core_distances[i] <= eps):
            labels[i] = -1
    return labels

def _extend_region(steep: np.ndarray[bool, 1], xward: np.ndarray[bool, 1], start: int,
                   min_samples: int) -> int:
    n = steep.shape[0]
    non_x = 0
    index = start
    end = start
    while index < n:
        if steep[index]:
            non_x = 0
            end = index
        elif not xward[index]:
            non_x += 1
            if non_x > min_samples:
                break
        else:
            return end
        index += 1
    return end

def _correct_predecessor(rp: np.ndarray[float, 1], pp: np.ndarray[int, 1], ordering, s: int, e: int):
    while s < e:
        if rp[s] > rp[e]:
            return s, e
        p_e = pp[e]
        for i in range(s, e):
            if p_e == ordering[i]:
                return s, e
        e -= 1
    return -1, -1

def _xi_cluster(rp0: np.ndarray[float, 1], pp: np.ndarray[int, 1], ordering, xi: float,
                min_samples: int, min_cluster_size: int, predecessor_correction: bool):
    n0 = rp0.shape[0]
    rp = np.empty(n0 + 1)
    rp[:n0] = rp0
    rp[n0] = np.inf
    xc = 1.0 - xi
    sd_start = List[int]()
    sd_end = List[int]()
    sd_mib = List[float]()
    clusters = List[Tuple[int, int]]()
    index = 0
    mib = 0.0
    su = np.zeros(n0, dtype=bool)
    sdn = np.zeros(n0, dtype=bool)
    dw = np.zeros(n0, dtype=bool)
    uw = np.zeros(n0, dtype=bool)
    for i in range(n0):
        a = rp[i]
        b = rp[i + 1]
        r = np.nan
        if b != 0.0 and not (a == np.inf and b == np.inf):
            r = a / b
        elif b == 0.0 and a != 0.0:
            r = np.inf
        su[i] = r <= xc
        sdn[i] = r >= 1.0 / xc
        dw[i] = r > 1.0
        uw[i] = r < 1.0

    def filt(mib: float, ss: List[int], se: List[int], sm: List[float]):
        ns = List[int]()
        ne = List[int]()
        nm = List[float]()
        if mib == np.inf:
            return ns, ne, nm
        for q in range(len(ss)):
            if mib <= rp[ss[q]] * xc:
                ns.append(ss[q])
                ne.append(se[q])
                nm.append(max(sm[q], mib))
        return ns, ne, nm

    for si in range(n0):
        if not (su[si] or sdn[si]):
            continue
        if si < index:
            continue
        for q in range(index, si + 1):
            mib = max(mib, rp[q])
        if sdn[si]:
            sd_start, sd_end, sd_mib = filt(mib, sd_start, sd_end, sd_mib)
            d_end = _extend_region(sdn, uw, si, min_samples)
            sd_start.append(si)
            sd_end.append(d_end)
            sd_mib.append(0.0)
            index = d_end + 1
            mib = rp[index]
        else:
            sd_start, sd_end, sd_mib = filt(mib, sd_start, sd_end, sd_mib)
            u_start = si
            u_end = _extend_region(su, dw, u_start, min_samples)
            index = u_end + 1
            mib = rp[index]
            uc = List[Tuple[int, int]]()
            for q in range(len(sd_start)):
                c_start = sd_start[q]
                c_end = u_end
                if rp[c_end + 1] * xc < sd_mib[q]:
                    continue
                d_max = rp[sd_start[q]]
                if d_max * xc >= rp[c_end + 1]:
                    while rp[c_start + 1] > rp[c_end + 1] and c_start < sd_end[q]:
                        c_start += 1
                elif rp[c_end + 1] * xc >= d_max:
                    while rp[c_end - 1] > d_max and c_end > u_start:
                        c_end -= 1
                if predecessor_correction:
                    c_start, c_end = _correct_predecessor(rp, pp, ordering, c_start, c_end)
                if c_start < 0:
                    continue
                if c_end - c_start + 1 < min_cluster_size:
                    continue
                if c_start > sd_end[q]:
                    continue
                if c_end < u_start:
                    continue
                uc.append((c_start, c_end))
            uc.reverse()
            for c in uc:
                clusters.append(c)
    out = np.zeros((len(clusters), 2), dtype=int)
    for q in range(len(clusters)):
        out[q, 0] = clusters[q][0]
        out[q, 1] = clusters[q][1]
    return out

def _extract_xi_labels(ordering: np.ndarray[int, 1], clusters: np.ndarray[int, 2]):
    n = ordering.shape[0]
    labels = np.full(n, -1, dtype=int)
    label = 0
    for q in range(clusters.shape[0]):
        a = clusters[q, 0]
        b = clusters[q, 1]
        free = True
        for i in range(a, b + 1):
            if labels[i] != -1:
                free = False
        if free:
            for i in range(a, b + 1):
                labels[i] = label
            label += 1
    out = np.empty(n, dtype=int)
    for i in range(n):
        out[ordering[i]] = labels[i]
    return out

def cluster_optics_xi(*, reachability, predecessor, ordering, min_samples = 5,
                      min_cluster_size = None, xi: float = 0.05,
                      predecessor_correction: bool = True):
    n = reachability.shape[0]
    ms = 0
    if isinstance(min_samples, float):
        ms = max(2, int(min_samples * n))
    else:
        ms = int(min_samples)
    mcs = ms
    if min_cluster_size is not None:
        if isinstance(min_cluster_size, float):
            mcs = max(2, int(min_cluster_size * n))
        else:
            mcs = int(min_cluster_size)
    clusters = _xi_cluster(reachability[ordering], predecessor[ordering], ordering, xi, ms, mcs,
                           predecessor_correction)
    return _extract_xi_labels(ordering, clusters), clusters

class OPTICS:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    min_samples_i: int
    min_samples_f: float
    max_eps: float
    metric: str
    p: float
    cluster_method: str
    eps: Optional[float]
    xi: float
    predecessor_correction: bool
    min_cluster_size_i: int
    min_cluster_size_f: float
    ordering_: np.ndarray[int, 1]
    core_distances_: np.ndarray[float, 1]
    reachability_: np.ndarray[float, 1]
    predecessor_: np.ndarray[int, 1]
    labels_: np.ndarray[int, 1]
    cluster_hierarchy_: np.ndarray[int, 2]
    n_features_in_: int

    def __init__(self, *, min_samples = 5, max_eps: float = np.inf, metric: str = "minkowski",
                 p: float = 2.0, metric_params = None, cluster_method: str = "xi",
                 eps: Optional[float] = None, xi: float = 0.05, predecessor_correction: bool = True,
                 min_cluster_size = None, algorithm: str = "auto", leaf_size: int = 30,
                 memory = None, n_jobs = None):
        self.min_samples_i = -1
        self.min_samples_f = 0.0
        if isinstance(min_samples, float):
            self.min_samples_f = min_samples
        else:
            self.min_samples_i = int(min_samples)
        self.max_eps = max_eps
        self.metric = metric
        self.p = p
        if cluster_method != "xi" and cluster_method != "dbscan":
            raise ValueError(f"cluster_method should be 'xi' or 'dbscan', got {cluster_method}")
        self.cluster_method = cluster_method
        self.eps = eps
        self.xi = xi
        self.predecessor_correction = predecessor_correction
        self.min_cluster_size_i = -1
        self.min_cluster_size_f = 0.0
        if min_cluster_size is not None:
            if isinstance(min_cluster_size, float):
                self.min_cluster_size_f = min_cluster_size
            else:
                self.min_cluster_size_i = int(min_cluster_size)
        self.ordering_ = np.empty(0, dtype=int)
        self.core_distances_ = np.empty(0)
        self.reachability_ = np.empty(0)
        self.predecessor_ = np.empty(0, dtype=int)
        self.labels_ = np.empty(0, dtype=int)
        self.cluster_hierarchy_ = np.empty((0, 2), dtype=int)
        self.n_features_in_ = 0

    def fit(self, X, y = None):
        Xa = _as2d(X)
        n = Xa.shape[0]
        ms = self.min_samples_i if self.min_samples_i >= 0 else max(2, int(self.min_samples_f * n))
        o, c, r, pr = compute_optics_graph(Xa, min_samples=ms, max_eps=self.max_eps,
                                           metric=self.metric, p=self.p)
        self.ordering_ = o
        self.core_distances_ = c
        self.reachability_ = r
        self.predecessor_ = pr
        if self.cluster_method == "xi":
            mcs = ms
            if self.min_cluster_size_i >= 0:
                mcs = self.min_cluster_size_i
            elif self.min_cluster_size_f > 0.0:
                mcs = max(2, int(self.min_cluster_size_f * n))
            lab, cl = cluster_optics_xi(reachability=r, predecessor=pr, ordering=o, min_samples=ms,
                                        min_cluster_size=mcs, xi=self.xi,
                                        predecessor_correction=self.predecessor_correction)
            self.labels_ = lab
            self.cluster_hierarchy_ = cl
        else:
            eps = self.max_eps if self.eps is None else self.eps.__val__()
            if eps > self.max_eps:
                raise ValueError(f"Specify an epsilon smaller than {self.max_eps}. Got {eps}.")
            self.labels_ = cluster_optics_dbscan(reachability=r, core_distances=c, ordering=o, eps=eps)
        self.n_features_in_ = Xa.shape[1]
        return self

    def fit_predict(self, X, y = None):
        return self.fit(X).labels_
