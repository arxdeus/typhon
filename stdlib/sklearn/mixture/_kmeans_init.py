# Ported from scikit-learn 1.9.1 (BSD-3-Clause)
# Minimal dense KMeans(n_init=1, init='k-means++', algorithm='lloyd') and
# kmeans_plusplus, reproducing sklearn's random stream, used to initialise
# mixture models (init_params='kmeans' / 'k-means++').

import numpy as np
from ..decomposition._extmath import _row_sq_einsum

def _sq_dists(A, B, B_sq):
    # sklearn _euclidean_distances(A, B, Y_norm_squared=B_sq, squared=True)
    AA = _row_sq_einsum(A)
    D = -2.0 * (A @ B.T)
    D += AA[:, None]
    D += B_sq[None, :]
    np.maximum(D, 0.0, out=D)
    return D

def kmeans_plusplus(X, n_clusters: int, x_squared_norms, rs):
    """returns (centers, indices); unit sample weights."""
    n_samples, n_features = X.shape
    centers = np.empty((n_clusters, n_features))
    n_local_trials = 2 + int(np.log(float(n_clusters)))
    sw = np.ones(n_samples)
    center_id = int(rs.choice(n_samples, p=sw / sw.sum()))
    indices = np.full(n_clusters, -1, dtype=int)
    centers[0] = X[center_id]
    indices[0] = center_id
    closest = _sq_dists(X[center_id:center_id + 1], X, x_squared_norms)[0]
    current_pot = float(closest @ sw)
    for c in range(1, n_clusters):
        rand_vals = rs.uniform(size=n_local_trials) * current_pot
        cand = np.searchsorted(np.cumsum(sw * closest), rand_vals)
        for t in range(n_local_trials):
            if cand[t] > closest.size - 1:
                cand[t] = closest.size - 1
        dc = _sq_dists(X[cand], X, x_squared_norms)
        np.minimum(closest[None, :], dc, out=dc)
        pots = dc @ sw
        best = int(np.argmin(pots))
        current_pot = float(pots[best])
        closest = dc[best].copy()
        bc = int(cand[best])
        centers[c] = X[bc]
        indices[c] = bc
    return centers, indices

def _lloyd_iter(X, centers_old, centers_new, weight, labels, center_shift, update: bool):
    n_samples, n_features = X.shape
    k = centers_old.shape[0]
    csq = np.sum(centers_old * centers_old, axis=1)
    pd = X @ centers_old.T
    if update:
        centers_new.fill(0.0)
        weight.fill(0.0)
    for i in range(n_samples):
        best = 0
        mn = csq[0] - 2.0 * pd[i, 0]
        for j in range(1, k):
            d = csq[j] - 2.0 * pd[i, j]
            if d < mn:
                mn = d
                best = j
        labels[i] = best
        if update:
            weight[best] += 1.0
            for f in range(n_features):
                centers_new[best, f] += X[i, f]
    if not update:
        return
    # _relocate_empty_clusters_dense
    n_empty = 0
    for j in range(k):
        if weight[j] == 0:
            n_empty += 1
    if n_empty > 0:
        dist = np.empty(n_samples)
        for i in range(n_samples):
            s = 0.0
            for f in range(n_features):
                t = X[i, f] - centers_old[labels[i], f]
                s += t * t
            dist[i] = s
        if float(np.max(dist)) != 0.0:
            far = np.argsort(-dist, kind="stable")[:n_empty]
            e = 0
            for j in range(k):
                if weight[j] == 0:
                    fi = int(far[e])
                    old = labels[fi]
                    for f in range(n_features):
                        centers_new[old, f] -= X[fi, f]
                        centers_new[j, f] = X[fi, f]
                    weight[j] = 1.0
                    weight[old] -= 1.0
                    e += 1
    amax = int(np.argmax(weight))
    for j in range(k):
        if weight[j] > 0:
            a = 1.0 / weight[j]
            for f in range(n_features):
                centers_new[j, f] *= a
        else:
            for f in range(n_features):
                centers_new[j, f] = centers_new[amax, f]
    for j in range(k):
        s = 0.0
        for f in range(n_features):
            t = centers_new[j, f] - centers_old[j, f]
            s += t * t
        center_shift[j] = np.sqrt(s)

def kmeans_labels(X0, n_clusters: int, rs, max_iter: int = 300, tol: float = 1e-4):
    """KMeans(n_clusters, n_init=1, random_state=rs).fit(X).labels_"""
    X = np.array(X0, dtype=float)
    X -= X.mean(axis=0)
    xsq = _row_sq_einsum(X)
    centers, _ = kmeans_plusplus(X, n_clusters, xsq, rs)
    tol_ = float(np.mean(np.var(X, axis=0))) * tol
    n = X.shape[0]
    centers_new = np.zeros_like(centers)
    labels = np.full(n, -1, dtype=int)
    labels_old = labels.copy()
    weight = np.zeros(n_clusters)
    shift = np.zeros(n_clusters)
    strict = False
    for _ in range(max_iter):
        _lloyd_iter(X, centers, centers_new, weight, labels, shift, True)
        tmp = centers
        centers = centers_new
        centers_new = tmp
        same = True
        for i in range(n):
            if labels[i] != labels_old[i]:
                same = False
                break
        if same:
            strict = True
            break
        if float((shift ** 2).sum()) <= tol_:
            break
        labels_old[:] = labels
    if not strict:
        _lloyd_iter(X, centers, centers, weight, labels, shift, False)
    return labels
