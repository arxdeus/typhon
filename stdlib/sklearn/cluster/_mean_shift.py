# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# MeanShift (flat kernel, optional bin seeding), estimate_bandwidth,
# get_bin_seeds and AffinityPropagation. Neighbor searches are exact
# brute-force euclidean scans.

from ..base import estimator_repr as _estimator_repr
import numpy as np
from numpy.random.legacy import RandomState, check_random_state
from ._kmeans import _as2d

def _dist(a: np.ndarray[float, 1], B: np.ndarray[float, 2], j: int) -> float:
    s = 0.0
    for k in range(a.shape[0]):
        t = a[k] - B[j, k]
        s += t * t
    return np.sqrt(s)

def estimate_bandwidth(X, *, quantile: float = 0.3, n_samples: Optional[int] = None,
                       random_state = 0, n_jobs = None) -> float:
    Xa = _as2d(X)
    if n_samples is not None:
        rs = check_random_state(random_state)
        idx = rs.permutation(Xa.shape[0])[:n_samples.__val__()]
        Xa = np.ascontiguousarray(Xa[idx])
    n = Xa.shape[0]
    k = int(n * quantile)
    if k < 1:
        k = 1
    bw = 0.0
    for i in range(n):
        d = np.empty(n)
        for j in range(n):
            d[j] = _dist(Xa[i], Xa, j)
        d.sort()
        bw += d[k - 1]
    return bw / n

def get_bin_seeds(X, bin_size: float, min_bin_freq: int = 1):
    Xa = _as2d(X)
    if bin_size == 0.0:
        return Xa
    p = Xa.shape[1]
    keys = List[List[float]]()
    counts = List[int]()
    index = Dict[str, int]()
    for i in range(Xa.shape[0]):
        b = [float(np.round(Xa[i, k] / bin_size)) for k in range(p)]
        key = str(b)
        if key in index:
            counts[index[key]] += 1
        else:
            index[key] = len(keys)
            keys.append(b)
            counts.append(1)
    sel = [q for q in range(len(keys)) if counts[q] >= min_bin_freq]
    if len(sel) == Xa.shape[0]:
        return Xa
    out = np.empty((len(sel), p))
    for r in range(len(sel)):
        for k in range(p):
            # sklearn stores the bins as float32 before scaling
            out[r, k] = float(float32(keys[sel[r]][k])) * bin_size
    return out

class MeanShift:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    bandwidth: Optional[float]
    seeds: Optional[np.ndarray[float, 2]]
    bin_seeding: bool
    min_bin_freq: int
    cluster_all: bool
    max_iter: int
    cluster_centers_: np.ndarray[float, 2]
    labels_: np.ndarray[int, 1]
    n_iter_: int
    n_features_in_: int

    def __init__(self, *, bandwidth: Optional[float] = None, seeds = None, bin_seeding: bool = False,
                 min_bin_freq: int = 1, cluster_all: bool = True, n_jobs = None, max_iter: int = 300):
        self.bandwidth = bandwidth
        self.seeds = None if seeds is None else _as2d(seeds)
        self.bin_seeding = bin_seeding
        self.min_bin_freq = min_bin_freq
        self.cluster_all = cluster_all
        self.max_iter = max_iter
        self.cluster_centers_ = np.empty((0, 0))
        self.labels_ = np.empty(0, dtype=int)
        self.n_iter_ = 0
        self.n_features_in_ = 0

    def fit(self, X, y = None):
        Xa = _as2d(X)
        n, p = Xa.shape
        bw = estimate_bandwidth(Xa) if self.bandwidth is None else self.bandwidth.__val__()
        seeds = Xa
        if self.seeds is not None:
            seeds = self.seeds.__val__()
        elif self.bin_seeding:
            seeds = get_bin_seeds(Xa, bw, self.min_bin_freq)
        stop = 1e-3 * bw
        cents = List[List[float]]()
        inten = List[int]()
        idx_of = Dict[str, int]()
        mx_it = 0
        for s in range(seeds.shape[0]):
            mean = seeds[s].copy()
            it = 0
            cnt = 0
            while True:
                acc = np.zeros(p)
                cnt = 0
                for j in range(n):
                    if _dist(mean, Xa, j) <= bw:
                        acc += Xa[j]
                        cnt += 1
                if cnt == 0:
                    break
                old = mean
                mean = acc / float(cnt)
                if np.sqrt(float(((mean - old) ** 2).sum())) <= stop or it == self.max_iter:
                    break
                it += 1
            if it > mx_it:
                mx_it = it
            if cnt > 0:
                key = [float(mean[k]) for k in range(p)]
                ks = str(key)
                if ks in idx_of:
                    inten[idx_of[ks]] = cnt
                else:
                    idx_of[ks] = len(cents)
                    cents.append(key)
                    inten.append(cnt)
        self.n_iter_ = mx_it
        if len(cents) == 0:
            raise ValueError(f"No point was within bandwidth={bw} of any seed. Try a different seeding strategy or increase the bandwidth.")
        order = list(range(len(cents)))
        order.sort(key=lambda q: (inten[q], cents[q]), reverse=True)
        m = len(order)
        sc = np.empty((m, p))
        for r in range(m):
            for k in range(p):
                sc[r, k] = cents[order[r]][k]
        unique = np.ones(m, dtype=bool)
        for i in range(m):
            if unique[i]:
                for j in range(m):
                    if _dist(sc[i], sc, j) <= bw:
                        unique[j] = False
                unique[i] = True
        cc = np.ascontiguousarray(sc[unique])
        labels = np.zeros(n, dtype=int)
        for i in range(n):
            best = 0
            bd = _dist(Xa[i], cc, 0)
            for c in range(1, cc.shape[0]):
                d = _dist(Xa[i], cc, c)
                if d < bd:
                    bd = d
                    best = c
            if self.cluster_all or bd <= bw:
                labels[i] = best
            else:
                labels[i] = -1
        self.cluster_centers_ = cc
        self.labels_ = labels
        self.n_features_in_ = p
        return self

    def predict(self, X):
        Xa = _as2d(X)
        out = np.empty(Xa.shape[0], dtype=int)
        for i in range(Xa.shape[0]):
            best = 0
            bd = _dist(Xa[i], self.cluster_centers_, 0)
            for c in range(1, self.cluster_centers_.shape[0]):
                d = _dist(Xa[i], self.cluster_centers_, c)
                if d < bd:
                    bd = d
                    best = c
            out[i] = best
        return out

    def fit_predict(self, X, y = None):
        return self.fit(X).labels_

def _sq_euclid(X: np.ndarray[float, 2]) -> np.ndarray[float, 2]:
    # sklearn euclidean_distances(squared=True): x.x - 2 x.y + y.y, diag 0
    n = X.shape[0]
    G = X @ X.T
    D = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            v = G[i, i] - 2.0 * G[i, j] + G[j, j]
            D[i, j] = v if v > 0.0 else 0.0
        D[i, i] = 0.0
    return D

def _affinity_propagation(S0: np.ndarray[float, 2], preference: np.ndarray[float, 1],
                          convergence_iter: int, max_iter: int, damping: float, rs: RandomState):
    n = S0.shape[0]
    S = S0.copy()
    allsame = True
    for i in range(n):
        for j in range(n):
            if i != j and S[i, j] != S[0, 1 if n > 1 else 0]:
                allsame = False
    for i in range(n):
        if preference[i] != preference[0]:
            allsame = False
    if n == 1 or allsame:
        if preference[0] > S.ravel()[n - 1]:
            return np.arange(n), np.arange(n), 0
        else:
            return np.array([0]), np.zeros(n, dtype=int), 0
    for i in range(n):
        S[i, i] = preference[i]
    A = np.zeros((n, n))
    R = np.zeros((n, n))
    eps = np.finfo(float).eps
    tiny = np.finfo(float).tiny
    noise = rs.standard_normal(size=(n, n))
    S += (eps * S + tiny * 100) * noise
    e = np.zeros((n, convergence_iter))
    E = np.zeros(n, dtype=bool)
    it = 0
    never = True
    while it < max_iter:
        tmp = A + S
        I = np.argmax(tmp, axis=1)
        Y = np.empty(n)
        for i in range(n):
            Y[i] = tmp[i, I[i]]
            tmp[i, I[i]] = -np.inf
        Y2 = np.max(tmp, axis=1)
        tmp = S - Y[:, None]
        for i in range(n):
            tmp[i, I[i]] = S[i, I[i]] - Y2[i]
        tmp *= 1 - damping
        R *= damping
        R += tmp
        tmp = np.maximum(R, 0.0)
        for i in range(n):
            tmp[i, i] = R[i, i]
        tmp -= np.sum(tmp, axis=0)
        dA = np.diag(tmp).copy()
        tmp = np.clip(tmp, 0.0, np.inf)
        for i in range(n):
            tmp[i, i] = dA[i]
        tmp *= 1 - damping
        A *= damping
        A -= tmp
        for i in range(n):
            E[i] = (A[i, i] + R[i, i]) > 0.0
            e[i, it % convergence_iter] = 1.0 if E[i] else 0.0
        K = int(E.sum())
        if it >= convergence_iter:
            se = np.sum(e, axis=1)
            conv = 0
            for i in range(n):
                if se[i] == float(convergence_iter) or se[i] == 0.0:
                    conv += 1
            unconverged = conv != n
            if (not unconverged and K > 0) or it == max_iter:
                never = False
                break
        it += 1
    if never:
        it = max_iter - 1
    Il = [i for i in range(n) if E[i]]
    K = len(Il)
    if K > 0:
        I = np.array(Il, dtype=int)
        c = np.argmax(S[:, I], axis=1)
        for k in range(K):
            c[I[k]] = k
        for k in range(K):
            ii = [q for q in range(n) if c[q] == k]
            best = 0
            bs = -np.inf
            for jj in range(len(ii)):
                s = 0.0
                for q in ii:
                    s += S[q, ii[jj]]
                if s > bs:
                    bs = s
                    best = jj
            I[k] = ii[best]
        c = np.argmax(S[:, I], axis=1)
        for k in range(K):
            c[I[k]] = k
        labels = I[c]
        centers = np.unique(labels)
        return centers, np.searchsorted(centers, labels), it + 1
    else:
        return np.empty(0, dtype=int), np.full(n, -1, dtype=int), it + 1

class AffinityPropagation:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    damping: float
    max_iter: int
    convergence_iter: int
    affinity: str
    preference: Optional[np.ndarray[float, 1]]
    random_state: Optional[int]
    cluster_centers_indices_: np.ndarray[int, 1]
    cluster_centers_: np.ndarray[float, 2]
    labels_: np.ndarray[int, 1]
    affinity_matrix_: np.ndarray[float, 2]
    n_iter_: int
    n_features_in_: int

    def __init__(self, *, damping: float = 0.5, max_iter: int = 200, convergence_iter: int = 15,
                 copy: bool = True, preference = None, affinity: str = "euclidean",
                 verbose: bool = False, random_state: Optional[int] = None):
        if not (0.5 <= damping < 1.0):
            raise ValueError(f"The 'damping' parameter of AffinityPropagation must be a float in the range [0.5, 1.0). Got {damping} instead.")
        if affinity != "euclidean" and affinity != "precomputed":
            raise ValueError(f"affinity must be 'euclidean' or 'precomputed', got {affinity}")
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.affinity = affinity
        if preference is None:
            self.preference = None
        elif isinstance(preference, float) or isinstance(preference, int):
            self.preference = np.array([float(preference)])
        else:
            self.preference = np.asarray(preference, dtype=float).ravel()
        self.random_state = random_state
        self.cluster_centers_indices_ = np.empty(0, dtype=int)
        self.cluster_centers_ = np.empty((0, 0))
        self.labels_ = np.empty(0, dtype=int)
        self.affinity_matrix_ = np.empty((0, 0))
        self.n_iter_ = 0
        self.n_features_in_ = 0

    def fit(self, X, y = None):
        Xa = _as2d(X)
        if self.affinity == "precomputed":
            self.affinity_matrix_ = Xa.copy()
        else:
            self.affinity_matrix_ = -_sq_euclid(Xa)
        n = self.affinity_matrix_.shape[0]
        if n != self.affinity_matrix_.shape[1]:
            raise ValueError(f"The matrix of similarities must be a square array. Got {self.affinity_matrix_.shape} instead.")
        pref = np.empty(n)
        if self.preference is None:
            pref[:] = float(np.median(self.affinity_matrix_))
        else:
            pv = self.preference.__val__()
            if pv.shape[0] == 1:
                pref[:] = pv[0]
            else:
                pref = pv.copy()
        rs = check_random_state(self.random_state)
        ci, lab, it = _affinity_propagation(self.affinity_matrix_, pref, self.convergence_iter,
                                            self.max_iter, self.damping, rs)
        self.cluster_centers_indices_ = ci
        self.labels_ = lab
        self.n_iter_ = it
        self.n_features_in_ = Xa.shape[1]
        if self.affinity != "precomputed":
            self.cluster_centers_ = np.ascontiguousarray(Xa[ci])
        return self

    def predict(self, X):
        Xa = _as2d(X)
        if self.affinity == "precomputed":
            raise ValueError("Predict method is not supported when affinity='precomputed'.")
        out = np.empty(Xa.shape[0], dtype=int)
        if self.cluster_centers_.shape[0] == 0:
            out[:] = -1
            return out
        for i in range(Xa.shape[0]):
            best = 0
            bd = _dist(Xa[i], self.cluster_centers_, 0)
            for c in range(1, self.cluster_centers_.shape[0]):
                d = _dist(Xa[i], self.cluster_centers_, c)
                if d < bd:
                    bd = d
                    best = c
            out[i] = best
        return out

    def fit_predict(self, X, y = None):
        return self.fit(X).labels_
