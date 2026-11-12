# Ported from scikit-learn 1.9.1 (BSD-3-Clause)
# Shared helpers for the manifold estimators: k-nearest neighbours with
# sklearn's distance formulas, Dijkstra shortest paths, isotonic regression
# (scipy's PAVA) and linear interpolation.

import numpy as np
from ..decomposition._extmath import _row_sq_einsum

def _fma64(a: float, b: float, c: float) -> float:
    return float._fma(a, b, c)

def _sq_norms(X):
    return _row_sq_einsum(X)

def _use_brute(Xfit, n_neighbors_param: int) -> bool:
    """NearestNeighbors(algorithm='auto') choice made at fit time from the
    constructor's n_neighbors."""
    return Xfit.shape[1] > 15 or n_neighbors_param >= Xfit.shape[0] // 2

def _knn(Xfit, Xq, k: int, exclude_self: bool, brute: bool):
    """(distances, indices) of the k nearest rows of Xfit for each row of Xq,
    ascending; when exclude_self, Xq is Xfit and each point skips itself
    (sklearn kneighbors(X=None)). 'brute' distances use the ArgKmin GEMM
    expansion (||x||^2 + (-2 x.y)) + ||y||^2; tree distances sum squared
    differences."""
    n = Xfit.shape[0]
    q = Xq.shape[0]
    d = Xfit.shape[1]
    D = np.empty((q, n))
    if brute:
        M = -2.0 * (Xq @ Xfit.T)
        xs = _sq_norms(Xq)
        ys = _sq_norms(Xfit)
        for i in range(q):
            for j in range(n):
                v = (xs[i] + M[i, j]) + ys[j]
                D[i, j] = np.sqrt(v if v > 0.0 else 0.0)
    else:
        for i in range(q):
            for j in range(n):
                s = 0.0
                for a in range(d):
                    t = Xq[i, a] - Xfit[j, a]
                    s += t * t
                D[i, j] = np.sqrt(s)
    dist = np.empty((q, k))
    ind = np.empty((q, k), dtype=int)
    for i in range(q):
        row = D[i].copy()
        if exclude_self:
            row[i] = np.inf
        order = np.argsort(row, kind="stable")
        for t in range(k):
            ind[i, t] = order[t]
            dist[i, t] = row[order[t]]
    return dist, ind

def _euclidean_distances(X, Y, same: bool):
    """sklearn euclidean_distances (gemm expansion, diagonal zeroed when X is Y)."""
    XX = _sq_norms(X)
    YY = _sq_norms(Y)
    D = -2.0 * (X @ Y.T)
    D += XX[:, None]
    D += YY[None, :]
    np.maximum(D, 0.0, out=D)
    if same:
        for i in range(D.shape[0]):
            D[i, i] = 0.0
    return np.sqrt(D)

def _components(adj):
    """connected components of an undirected dense adjacency (inf = no edge)."""
    n = adj.shape[0]
    lab = np.full(n, -1, dtype=int)
    c = 0
    for s in range(n):
        if lab[s] >= 0:
            continue
        stack = [s]
        lab[s] = c
        while len(stack) > 0:
            u = stack.pop()
            for v in range(n):
                if lab[v] < 0 and adj[u, v] < np.inf:
                    lab[v] = c
                    stack.append(v)
        c += 1
    return c, lab

def _dijkstra_all(adj):
    """all-pairs shortest paths on a dense undirected adjacency (inf = no edge)."""
    n = adj.shape[0]
    out = np.full((n, n), np.inf)
    for s in range(n):
        dist = np.full(n, np.inf)
        done = np.zeros(n, dtype=bool)
        dist[s] = 0.0
        for _ in range(n):
            u = -1
            best = np.inf
            for v in range(n):
                if not done[v] and dist[v] < best:
                    best = dist[v]
                    u = v
            if u < 0:
                break
            done[u] = True
            for v in range(n):
                w = adj[u, v]
                if w < np.inf and not done[v]:
                    nd = dist[u] + w
                    if nd < dist[v]:
                        dist[v] = nd
        out[s] = dist
    return out

def _pava(y, w):
    """scipy.optimize.isotonic_regression (increasing), Busing (2022) PAVA
    with the fused multiply-adds of scipy's compiled kernel."""
    x = y.copy()
    ww = w.copy()
    n = x.shape[0]
    r = np.zeros(n + 1, dtype=int)
    if n == 0:
        return x
    r[1] = 1
    b = 0
    xp = x[0]
    wp = ww[0]
    i = 1
    while i < n:
        b += 1
        xb = x[i]
        wb = ww[i]
        sb = 0.0
        if xp >= xb:
            b -= 1
            sb = _fma64(wp, xp, wb * xb)
            wb += wp
            xb = sb / wb
            while i < n - 1 and xb >= x[i + 1]:
                i += 1
                sb = _fma64(ww[i], x[i], sb)
                wb += ww[i]
                xb = sb / wb
            while b > 0 and x[b - 1] >= xb:
                b -= 1
                sb = _fma64(ww[b], x[b], sb)
                wb += ww[b]
                xb = sb / wb
        x[b] = xb
        xp = xb
        ww[b] = wb
        wp = wb
        r[b + 1] = i + 1
        i += 1
    f = n - 1
    k = b
    while k >= 0:
        t = r[k]
        xk = x[k]
        j = f
        while j >= t:
            x[j] = xk
            j -= 1
        f = t - 1
        k -= 1
    return x

def _isotonic_fit_transform(X, y):
    """IsotonicRegression(out_of_bounds='clip').fit_transform(X, y)."""
    n = X.shape[0]
    # np.lexsort((y, X)): by X, then y
    order = np.argsort(y, kind="stable")
    order = order[np.argsort(X[order], kind="stable")]
    Xs = X[order]
    ys = y[order]
    # _make_unique, eps = finfo(float64).resolution
    eps = 1e-15
    ux: List[float] = []
    uy: List[float] = []
    uw: List[float] = []
    cx = Xs[0]
    cy = 0.0
    cw = 0.0
    for j in range(n):
        xv = Xs[j]
        if xv - cx >= eps:
            ux.append(cx)
            uw.append(cw)
            uy.append(cy / cw)
            cx = xv
            cw = 1.0
            cy = ys[j]
        else:
            cw += 1.0
            cy += ys[j]
    ux.append(cx)
    uw.append(cw)
    uy.append(cy / cw)
    xu = np.array(ux)
    yu = _pava(np.array(uy), np.array(uw))
    x_min = float(np.min(xu))
    x_max = float(np.max(xu))
    m = yu.shape[0]
    keep = np.ones(m, dtype=bool)
    for t in range(1, m - 1):
        keep[t] = (yu[t] != yu[t - 1]) or (yu[t] != yu[t + 1])
    xt = xu[keep]
    yt = yu[keep]
    out = np.empty(n)
    if yt.shape[0] == 1:
        out.fill(yt[0])
        return out
    L = xt.shape[0]
    for i in range(n):
        v = min(max(X[i], x_min), x_max)
        idx = int(np.searchsorted(xt, v))
        if idx < 1:
            idx = 1
        if idx > L - 1:
            idx = L - 1
        lo = idx - 1
        # interp1d(kind='linear') dispatches to np.interp, whose compiled
        # kernel computes slope * (x - xlo) + ylo with a fused multiply-add
        if v == xt[idx]:
            out[i] = yt[idx]
        else:
            slope = (yt[idx] - yt[lo]) / (xt[idx] - xt[lo])
            out[i] = _fma64(slope, v - xt[lo], yt[lo])
    return out
