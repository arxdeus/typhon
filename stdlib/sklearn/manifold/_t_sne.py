# Ported from scikit-learn 1.9.1 (BSD-3-Clause)
# t-SNE: exact gradient and Barnes-Hut (quad/oct tree) gradient, float32
# parameters as in sklearn so a fixed random_state reproduces its output.

import numpy as np
from ..decomposition._extmath import euclidean_distances_sq

_MACHINE_EPSILON = 2.220446049250313e-16
_FLOAT32_TINY = 1.1754943508222875e-38
_EPS_DBL = 1e-8
_PERPLEXITY_TOL = float(float32(1e-5))
_QT_EPS = float32(1e-6)

# sklearn's compiled Barnes-Hut / quad-tree kernels contract these
# multiply-adds into fused FMA instructions; mirror them for identical bits.
def _fma32(a: float32, b: float32, c: float32) -> float32:
    return float32._fma(a, b, c)

def _fma64(a: float, b: float, c: float) -> float:
    return float._fma(a, b, c)

def _binary_search_perplexity(sqd, desired_perplexity: float):
    """sqd: float32 (n, k). Returns float64 conditional P (n, k)."""
    n = sqd.shape[0]
    k = sqd.shape[1]
    using_nb = k < n
    P = np.zeros((n, k))
    desired_entropy = np.log(desired_perplexity)
    for i in range(n):
        beta_min = -np.inf
        beta_max = np.inf
        beta = 1.0
        for _ in range(100):
            sum_Pi = 0.0
            for j in range(k):
                if j != i or using_nb:
                    P[i, j] = np.exp(-float(sqd[i, j]) * beta)
                    sum_Pi += P[i, j]
            if sum_Pi == 0.0:
                sum_Pi = _EPS_DBL
            sdp = 0.0
            for j in range(k):
                P[i, j] /= sum_Pi
                sdp += float(sqd[i, j]) * P[i, j]
            entropy = np.log(sum_Pi) + beta * sdp
            diff = entropy - desired_entropy
            if abs(diff) <= _PERPLEXITY_TOL:
                break
            if diff > 0.0:
                beta_min = beta
                if beta_max == np.inf:
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                beta_max = beta
                if beta_min == -np.inf:
                    beta /= 2.0
                else:
                    beta = (beta + beta_min) / 2.0
    return P

def _joint_probabilities(D, perplexity: float):
    """exact: returns the condensed (squareform) P vector."""
    n = D.shape[0]
    cP = _binary_search_perplexity(D.astype(np.float32), perplexity)
    Pm = cP + cP.T
    sum_P = max(float(np.sum(Pm)), _MACHINE_EPSILON)
    out = np.empty(n * (n - 1) // 2)
    t = 0
    for i in range(n):
        for j in range(i + 1, n):
            out[t] = max(Pm[i, j] / sum_P, _MACHINE_EPSILON)
            t += 1
    return out

def _kl_divergence_exact(params, P, dof: float, n: int, nc: int, compute_error: bool):
    X = params.reshape((n, nc))
    m = n * (n - 1) // 2
    dist = np.empty(m)
    t = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = 0.0
            for a in range(nc):
                d = float(X[i, a]) - float(X[j, a])
                s += d * d
            dist[t] = s
            t += 1
    dist /= dof
    dist += 1.0
    dist **= (dof + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * float(np.sum(dist))), _MACHINE_EPSILON)
    kl = np.nan
    if compute_error:
        kl = 2.0 * float(np.dot(P, np.log(np.maximum(P, _MACHINE_EPSILON) / Q)))
    PQd = np.zeros((n, n))
    W = (P - Q) * dist
    t = 0
    for i in range(n):
        for j in range(i + 1, n):
            PQd[i, j] = W[t]
            PQd[j, i] = W[t]
            t += 1
    grad = np.empty((n, nc), dtype=np.float32)
    for i in range(n):
        # X[i] - X is a float32 difference, upcast by the float64 dot
        g = PQd[i] @ (X[i] - X).astype(float)
        for a in range(nc):
            grad[i, a] = float32(g[a])
    c = 2.0 * (dof + 1.0) / dof
    g2 = grad.ravel()
    g2 *= float32(c)
    return kl, g2

# ---------------------------------------------------------------- quad tree

class _QuadTree:
    nd: int
    ncpc: int
    # per cell arrays (capacity-growing lists)
    parent: List[int]
    is_leaf: List[bool]
    depth: List[int]
    sq_max_width: List[float32]
    cum_size: List[int]
    point_index: List[int]
    children: List[int]       # ncpc per cell, -1 = none
    min_b: List[float32]      # 3 per cell
    max_b: List[float32]
    center: List[float32]
    bary: List[float32]
    n_points: int

    def __init__(self, nd: int):
        self.nd = nd
        self.ncpc = 1 << nd
        self.parent = []
        self.is_leaf = []
        self.depth = []
        self.sq_max_width = []
        self.cum_size = []
        self.point_index = []
        self.children = []
        self.min_b = []
        self.max_b = []
        self.center = []
        self.bary = []
        self.n_points = 0

    def _new_cell(self, parent: int, depth: int) -> int:
        cid = len(self.parent)
        self.parent.append(parent)
        self.is_leaf.append(True)
        self.depth.append(depth)
        self.sq_max_width.append(float32(0.0))
        self.cum_size.append(0)
        self.point_index.append(-1)
        for _ in range(8):
            self.children.append(-1)
        for _ in range(3):
            self.min_b.append(float32(0.0))
            self.max_b.append(float32(0.0))
            self.center.append(float32(0.0))
            self.bary.append(float32(0.0))
        return cid

    def build(self, X):
        n = X.shape[0]
        nd = self.nd
        m = np.min(X, axis=0)
        M = np.max(X, axis=0)
        root = self._new_cell(-1, 0)
        for i in range(nd):
            Mi = float(M[i])
            # np.maximum(M * (1. + 1e-3 * np.sign(M)), M + 1e-3) in float32
            sg = float32(1.0) if Mi > 0 else (float32(-1.0) if Mi < 0 else float32(0.0))
            a = M[i] * (float32(1.0) + float32(1e-3) * sg)
            b = M[i] + float32(1e-3)
            mx = a if a > b else b
            self.min_b[root * 3 + i] = m[i]
            self.max_b[root * 3 + i] = mx
            self.center[root * 3 + i] = float32(float(mx + m[i]) / 2.0)
            w = mx - m[i]
            if w * w > self.sq_max_width[root]:
                self.sq_max_width[root] = w * w
        pt = [float32(0.0)] * 3
        for i in range(n):
            for j in range(nd):
                pt[j] = X[i, j]
            self._insert(pt, i, 0)

    def _is_dup(self, p: List[float32], cid: int) -> bool:
        for i in range(self.nd):
            if not (abs(p[i] - self.bary[cid * 3 + i]) <= _QT_EPS):
                return False
        return True

    def _select_child(self, p: List[float32], cid: int) -> int:
        sel = 0
        for i in range(self.nd):
            sel *= 2
            if p[i] >= self.center[cid * 3 + i]:
                sel += 1
        return self.children[cid * 8 + sel]

    def _insert(self, p: List[float32], pidx: int, cid: int):
        n_point = self.cum_size[cid]
        if n_point == 0:
            self.cum_size[cid] = 1
            self.n_points += 1
            for i in range(self.nd):
                self.bary[cid * 3 + i] = p[i]
            self.point_index[cid] = pidx
            return
        if not self.is_leaf[cid]:
            for ax in range(self.nd):
                b = self.bary[cid * 3 + ax]
                # (n_point * barycenter + point) / (n_point + 1), float32 arithmetic
                self.bary[cid * 3 + ax] = _fma32(float32(n_point), b, p[ax]) / float32(n_point + 1)
            self.cum_size[cid] += 1
            sc = self._select_child(p, cid)
            if sc == -1:
                self.n_points += 1
                self._new_child(p, cid, pidx, 1)
                return
            self._insert(p, pidx, sc)
            return
        if self._is_dup(p, cid):
            self.cum_size[cid] += 1
            self.n_points += 1
            return
        bp = [self.bary[cid * 3], self.bary[cid * 3 + 1], self.bary[cid * 3 + 2]]
        self._new_child(bp, cid, self.point_index[cid], self.cum_size[cid])
        self._insert(p, pidx, cid)

    def _new_child(self, p: List[float32], cid: int, pidx: int, size: int):
        ch = self._new_cell(cid, self.depth[cid] + 1)
        self.is_leaf[cid] = False
        self.point_index[cid] = -1
        ccid = 0
        for i in range(self.nd):
            ccid *= 2
            c = self.center[cid * 3 + i]
            if p[i] >= c:
                ccid += 1
                self.min_b[ch * 3 + i] = c
                self.max_b[ch * 3 + i] = self.max_b[cid * 3 + i]
            else:
                self.min_b[ch * 3 + i] = self.min_b[cid * 3 + i]
                self.max_b[ch * 3 + i] = c
            lo = self.min_b[ch * 3 + i]
            hi = self.max_b[ch * 3 + i]
            self.center[ch * 3 + i] = float32(float(lo + hi) / 2.0)
            w = hi - lo
            self.bary[ch * 3 + i] = p[i]
            if w * w > self.sq_max_width[ch]:
                self.sq_max_width[ch] = w * w
        self.point_index[ch] = pidx
        self.cum_size[ch] = size
        self.children[cid * 8 + ccid] = ch

    def summarize(self, p: List[float32], res: List[float32], sq_theta: float32, cid: int, idx: int) -> int:
        nd = self.nd
        idx_d = idx + nd
        res[idx_d] = float32(0.0)
        dup = True
        for i in range(nd):
            r = p[i] - self.bary[cid * 3 + i]
            res[idx + i] = r
            res[idx_d] = _fma32(r, r, res[idx_d])
            dup = dup and abs(r) <= _QT_EPS
        if dup and self.is_leaf[cid]:
            return idx
        if self.is_leaf[cid] or (self.sq_max_width[cid] / res[idx_d]) < sq_theta:
            res[idx_d + 1] = float32(self.cum_size[cid])
            return idx + nd + 2
        for c in range(self.ncpc):
            ch = self.children[cid * 8 + c]
            if ch != -1:
                idx = self.summarize(p, res, sq_theta, ch, idx)
        return idx

def _kl_divergence_bh(params, Pdata, Pind, Pptr, dof: float, n: int, nc: int,
                      angle: float, compute_error: bool):
    X = params.reshape((n, nc))
    qt = _QuadTree(nc)
    qt.build(X)
    dofi = int(dof)
    fdof = float32(dofi)
    expo = float32((dofi + 1.0) / 2.0)
    offset = nc + 2
    # negative forces
    res = [float32(0.0)] * (n * offset)
    neg = np.zeros((n, nc), dtype=np.float32)
    sum_Q = 0.0
    pos = [float32(0.0)] * 3
    sq_theta = float32(angle) * float32(angle)
    for i in range(n):
        for a in range(nc):
            pos[a] = X[i, a]
        nf = [float32(0.0)] * 3
        idx = qt.summarize(pos, res, sq_theta, 0, 0)
        for j in range(idx // offset):
            d2 = res[j * offset + nc]
            size = res[j * offset + nc + 1]
            qijZ = float(fdof / (fdof + d2))
            if dofi != 1:
                qijZ = qijZ ** float(expo)
            sum_Q = _fma64(float(size), qijZ, sum_Q)
            mult = float32(float(size) * qijZ * qijZ)
            for a in range(nc):
                nf[a] = _fma32(mult, res[j * offset + a], nf[a])
        for a in range(nc):
            neg[i, a] = nf[a]
    sum_Q = max(sum_Q, _MACHINE_EPSILON)
    # positive forces
    C = float32(0.0)
    posf = np.zeros((n, nc), dtype=np.float32)
    buff = [float32(0.0)] * 3
    for i in range(n):
        for k in range(Pptr[i], Pptr[i + 1]):
            j = Pind[k]
            dij = float32(0.0)
            pij = float32(Pdata[k])
            for a in range(nc):
                buff[a] = X[i, a] - X[j, a]
                dij = _fma32(buff[a], buff[a], dij)
            qij = fdof / (fdof + dij)
            if dofi != 1:
                qij = qij ** expo  # float32 powf, as in the compiled kernel
            dd = pij * qij
            if compute_error:
                qq = float32(float(qij) / sum_Q)
                ratio = max(pij, float32(_FLOAT32_TINY)) / max(qq, float32(_FLOAT32_TINY))
                C = float32(_fma64(float(pij), np.log(float(ratio)), float(C)))
            for a in range(nc):
                posf[i, a] = _fma32(dd, buff[a], posf[i, a])
    grad = np.empty((n, nc), dtype=np.float32)
    for i in range(n):
        for a in range(nc):
            grad[i, a] = float32(float(posf[i, a]) - float(neg[i, a]) / sum_Q)
    c = 2.0 * (dof + 1.0) / dof
    g = grad.ravel()
    g *= float32(c)
    err = float(C) if compute_error else np.nan
    return err, g

def _gradient_descent(method: str, p, P, Pdata, Pind, Pptr, dof: float, n: int, nc: int,
                      angle: float, it0: int, max_iter: int, n_iter_check: int,
                      n_iter_without_progress: int, momentum: float, learning_rate: float,
                      min_gain: float, min_grad_norm: float, lr_is_f64: bool):
    """In-place on p (float32), returns (error, last iteration). As in
    sklearn, update/gains restart at zeros/ones on every call. With
    learning_rate='auto' the rate is a numpy float64 scalar, which promotes
    `update` to float64; a user float keeps it float32."""
    update = np.zeros(p.shape[0])
    gains = np.ones(p.shape[0], dtype=np.float32)
    error = 1.7976931348623157e308
    best_error = 1.7976931348623157e308
    best_iter = it0
    i = it0
    mom = float32(momentum)
    for ii in range(it0, max_iter):
        i = ii
        check = (ii + 1) % n_iter_check == 0
        ce = check or ii == max_iter - 1
        grad = np.empty(0, dtype=np.float32)
        if method == "exact":
            error, grad = _kl_divergence_exact(p, P, dof, n, nc, ce)
        else:
            error, grad = _kl_divergence_bh(p, Pdata, Pind, Pptr, dof, n, nc, angle, ce)
        for t in range(p.shape[0]):
            if update[t] * float(grad[t]) < 0.0:
                gains[t] += float32(0.2)
            else:
                gains[t] *= float32(0.8)
            if gains[t] < float32(min_gain):
                gains[t] = float32(min_gain)
            grad[t] *= gains[t]
            # momentum * update - learning_rate * grad: learning_rate_ is a
            # numpy float64 scalar, so the product is computed in float64
            if lr_is_f64:
                u = float(momentum) * update[t] - learning_rate * float(grad[t])
                update[t] = u
                p[t] = float32(float(p[t]) + u)
            else:
                u32 = mom * float32(update[t]) - float32(learning_rate) * grad[t]
                update[t] = float(u32)
                p[t] += u32
        if check:
            gn = 0.0
            for t in range(grad.shape[0]):
                gn += float(grad[t]) * float(grad[t])
            grad_norm = float(float32(np.sqrt(gn)))
            if error < best_error:
                best_error = error
                best_iter = ii
            elif ii - best_iter > n_iter_without_progress:
                break
            if grad_norm <= min_grad_norm:
                break
    return error, i

class TSNE:
    """method 'exact' / 'barnes_hut' (n_components <= 3), init 'pca' /
    'random' / array, learning_rate 'auto' or float, metric 'euclidean' or
    'precomputed' (distance matrix)."""
    n_components: int
    perplexity: float
    early_exaggeration: float
    learning_rate: float
    lr_auto: bool
    max_iter: int
    n_iter_without_progress: int
    min_grad_norm: float
    metric: str
    init: str
    init_array: Optional[np.ndarray[float, 2]]
    random_state: Optional[int]
    method: str
    angle: float
    embedding_: np.ndarray[float32, 2]
    kl_divergence_: float
    n_iter_: int
    learning_rate_: float

    def __init__(self, n_components: int = 2, *, perplexity: float = 30.0,
                 early_exaggeration: float = 12.0, learning_rate = "auto", max_iter: int = 1000,
                 n_iter_without_progress: int = 300, min_grad_norm: float = 1e-7,
                 metric: str = "euclidean", metric_params = None, init = "pca", verbose: int = 0,
                 random_state: Optional[int] = None, method: str = "barnes_hut", angle: float = 0.5,
                 n_jobs = None):
        if method != "exact" and method != "barnes_hut":
            raise ValueError("'method' must be 'barnes_hut' or 'exact'")
        if metric != "euclidean" and metric != "precomputed":
            raise NotImplementedError("TSNE: only metric 'euclidean' and 'precomputed' are ported")
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        if isinstance(learning_rate, str):
            self.lr_auto = True
            self.learning_rate = 0.0
        else:
            self.lr_auto = False
            self.learning_rate = float(learning_rate)
        self.max_iter = max_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.init_array = None
        if isinstance(init, str):
            self.init = init
        else:
            self.init = "array"
            self.init_array = np.array(init, dtype=float)
        self.random_state = random_state
        self.method = method
        self.angle = angle
        self.embedding_ = np.empty((0, 0), dtype=np.float32)
        self.kl_divergence_ = 0.0
        self.n_iter_ = 0
        self.learning_rate_ = 0.0

    def fit_transform(self, X, y = None):
        Xa = np.asarray(X, dtype=float)
        n = Xa.shape[0]
        nc = self.n_components
        if self.perplexity >= n:
            raise ValueError("perplexity must be less than n_samples")
        if self.lr_auto:
            self.learning_rate_ = max(n / self.early_exaggeration / 4, 50.0)
        else:
            self.learning_rate_ = self.learning_rate
        if self.method == "barnes_hut" and nc > 3:
            raise ValueError("'n_components' should be inferior to 4 for the barnes_hut algorithm as it relies on quad-tree or oct-tree.")
        if self.metric == "precomputed" and self.init == "pca":
            raise ValueError('The parameter init="pca" cannot be used with metric="precomputed".')
        rs = np.random.RandomState(self.random_state)
        D = np.empty((0, 0))
        if self.metric == "precomputed":
            D = Xa ** 2
        else:
            D = euclidean_distances_sq(Xa, Xa, True)
        P = np.empty(0)
        Pdata = np.empty(0)
        Pind = np.empty(0, dtype=int)
        Pptr = np.empty(0, dtype=int)
        if self.method == "exact":
            P = _joint_probabilities(D, self.perplexity)
        else:
            Pdata, Pind, Pptr = self._joint_probabilities_nn(Xa)
        # init
        X0 = np.empty((n, nc), dtype=np.float32)
        if self.init == "array":
            X0 = self.init_array.__val__().astype(np.float32)
        elif self.init == "pca":
            from ..decomposition import PCA
            Z = PCA(nc, svd_solver="auto", random_state=self.random_state).fit_transform(Xa).astype(np.float32)
            sd = float32(np.std(Z[:, 0]))
            X0 = Z / sd * float32(1e-4)
        elif self.init == "random":
            X0 = float32(1e-4) * rs.standard_normal(size=(n, nc)).astype(np.float32)
        else:
            raise ValueError(f"invalid init {self.init}")
        dof = float(max(nc - 1, 1))
        p = np.ascontiguousarray(X0).ravel().copy()
        ee = self.early_exaggeration
        P *= ee
        Pdata *= ee
        EXPL = 250
        kl, it = _gradient_descent(self.method, p, P, Pdata, Pind, Pptr, dof, n, nc, self.angle,
                                   0, EXPL, 50, EXPL, 0.5, self.learning_rate_, 0.01,
                                   self.min_grad_norm, self.lr_auto)
        P /= ee
        Pdata /= ee
        if it < EXPL or self.max_iter - EXPL > 0:
            kl, it = _gradient_descent(self.method, p, P, Pdata, Pind, Pptr, dof, n, nc, self.angle,
                                       it + 1, self.max_iter, 50, self.n_iter_without_progress, 0.8,
                                       self.learning_rate_, 0.01, self.min_grad_norm, self.lr_auto)
        self.n_iter_ = it
        self.kl_divergence_ = kl
        self.embedding_ = p.reshape((n, nc))
        return self.embedding_

    def fit(self, X, y = None):
        self.fit_transform(X)
        return self

    def _joint_probabilities_nn(self, Xa):
        """kNN (k = min(n - 1, 3 * perplexity + 1)) sparse symmetric P in
        CSR form (float64 data; the gradient casts it to float32)."""
        n = Xa.shape[0]
        k = min(n - 1, int(3.0 * self.perplexity + 1))
        # neighbour distances as a tree query computes them: sqrt(sum diff^2)
        D = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                if self.metric == "precomputed":
                    D[i, j] = Xa[i, j]
                else:
                    s = 0.0
                    for a in range(Xa.shape[1]):
                        d = Xa[i, a] - Xa[j, a]
                        s += d * d
                    D[i, j] = np.sqrt(s)
        idx = np.empty((n, k), dtype=int)
        dk = np.empty((n, k), dtype=np.float32)
        for i in range(n):
            row = D[i].copy()
            row[i] = np.inf
            order = np.argsort(row, kind="stable")
            for t in range(k):
                idx[i, t] = order[t]
                # kneighbors_graph distances are sqrt'd, then squared again
                dk[i, t] = float32(D[i, order[t]] ** 2)
        # csr sort_indices: columns ascending within each row
        for i in range(n):
            o = np.argsort(idx[i], kind="stable")
            idx[i] = idx[i][o]
            dk[i] = dk[i][o]
        cP = _binary_search_perplexity(dk, self.perplexity)
        # P + P.T (float64), then /= sum
        dense = np.zeros((n, n))
        for i in range(n):
            for t in range(k):
                dense[i, idx[i, t]] += cP[i, t]
        S = dense + dense.T
        nnz = 0
        for i in range(n):
            for j in range(n):
                if S[i, j] != 0.0:
                    nnz += 1
        data = np.empty(nnz)
        ind = np.empty(nnz, dtype=int)
        ptr = np.zeros(n + 1, dtype=int)
        t = 0
        for i in range(n):
            for j in range(n):
                if S[i, j] != 0.0:
                    data[t] = S[i, j]
                    ind[t] = j
                    t += 1
            ptr[i + 1] = t
        # sparse P.sum() is a pairwise sum over the CSR data array
        tot = max(float(np.sum(data)), _MACHINE_EPSILON)
        data /= tot
        return data, ind, ptr
