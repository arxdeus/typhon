# Ported from scipy 1.18.1 (BSD-3-Clause)

# K-means clustering and vector quantization (scipy.cluster.vq).
# Random initialisation uses ``numpy.random.default_rng(rng)`` (or the given
# Generator) and consumes the same streams as scipy.

import numpy as np
import internal.static as static
from math import inf, sqrt, fabs

class ClusterError(Exception):
    def __init__(self, msg: str = ''):
        super().__init__(msg)

def whiten(obs, check_finite: bool = True):
    a = np.asarray(obs, dtype=float)
    sd = np.std(a, axis=0)
    if static.len(a.shape) == 1:
        if sd == 0.0:
            return a / 1.0
        return a / sd
    else:
        for j in range(sd.shape[0]):
            if sd[j] == 0.0:
                sd[j] = 1.0
        return a / sd

def _as2(a):
    if static.len(a.shape) == 1:
        return a.reshape((a.shape[0], 1))
    elif static.len(a.shape) == 2:
        return a
    else:
        compile_error("Input of rank > 2 is not supported.")

def _vq2(obs, book) -> Tuple[List[int], List[float]]:
    n = obs.shape[0]
    k = book.shape[0]
    d = obs.shape[1]
    if book.shape[1] != d:
        raise ValueError("observation and code_book should have same number of features")
    codes = List[int](n)
    dists = List[float](n)
    for i in range(n):
        best = inf
        bi = 0
        for j in range(k):
            s = 0.0
            for t in range(d):
                diff = obs[i, t] - book[j, t]
                s += diff * diff
            if s < best:
                best = s
                bi = j
        codes.append(bi)
        dists.append(sqrt(best))
    return (codes, dists)

def vq(obs, code_book, check_finite: bool = True):
    o = np.asarray(obs, dtype=float)
    b = np.asarray(code_book, dtype=float)
    if static.len(o.shape) != static.len(b.shape):
        compile_error("Observation and code_book should have the same rank")
    r = _vq2(_as2(o), _as2(b))
    codes = np.empty(len(r[0]), dtype=np.int32)
    dist = np.empty(len(r[1]), dtype=float)
    for i in range(len(r[0])):
        codes[i] = np.int32(r[0][i])
        dist[i] = r[1][i]
    return codes, dist

def _update_means(obs, codes: List[int], nc: int):
    d = obs.shape[1]
    cb = np.zeros((nc, d), dtype=float)
    cnt = [0] * nc
    for i in range(obs.shape[0]):
        c = codes[i]
        cnt[c] += 1
        for t in range(d):
            cb[c, t] += obs[i, t]
    has = [False] * nc
    for c in range(nc):
        if cnt[c] > 0:
            has[c] = True
            for t in range(d):
                cb[c, t] /= cnt[c]
    return cb, has

def _mean(L: List[float]) -> float:
    s = 0.0
    for v in L:
        s += v
    return s / len(L)

def _kmeans(obs2, guess2, thresh: float):
    book = guess2
    diff = inf
    prev0 = inf
    prev1 = inf
    while diff > thresh:
        r = _vq2(obs2, book)
        prev0 = prev1
        prev1 = _mean(r[1])
        cbh = _update_means(obs2, r[0], book.shape[0])
        cb = cbh[0]
        has = cbh[1]
        nk = 0
        for h in has:
            if h:
                nk += 1
        nb = np.empty((nk, obs2.shape[1]), dtype=float)
        q = 0
        for c in range(len(has)):
            if has[c]:
                for t in range(obs2.shape[1]):
                    nb[q, t] = cb[c, t]
                q += 1
        book = nb
        diff = fabs(prev0 - prev1)
    r2 = _vq2(obs2, book)
    return book, _mean(r2[1])

def _get_rng(rng):
    if rng is None:
        return np.random.default_rng()
    elif isinstance(rng, int):
        return np.random.default_rng(rng)
    else:
        return rng

def _take_rows(data2, idx):
    out = np.empty((len(idx), data2.shape[1]), dtype=float)
    for i in range(len(idx)):
        for t in range(data2.shape[1]):
            out[i, t] = data2[int(idx[i]), t]
    return out

def _out_book(book, nd: Literal[int]):
    if nd == 1:
        return book[:, 0].copy()
    else:
        return book

def kmeans(obs, k_or_guess, iter: int = 20, thresh: float = 1e-5,
           check_finite: bool = True, rng = None):
    o = np.asarray(obs, dtype=float)
    o2 = _as2(o)
    if iter < 1:
        raise ValueError(f"iter must be at least 1, got {iter}")
    if isinstance(k_or_guess, int):
        k = k_or_guess
        if k < 1:
            raise ValueError(f"Asked for {k} clusters.")
        g = _get_rng(rng)
        best_book = np.empty((0, o2.shape[1]), dtype=float)
        best = inf
        for i in range(iter):
            idx = g.choice(o2.shape[0], size=k, replace=False)
            r = _kmeans(o2, _take_rows(o2, idx), thresh)
            if r[1] < best:
                best_book = r[0]
                best = r[1]
        return _out_book(best_book, static.len(o.shape)), best
    else:
        gs = np.asarray(k_or_guess, dtype=float)
        if gs.size < 1:
            raise ValueError("Asked for 0 clusters.")
        r = _kmeans(o2, _as2(gs).copy(), thresh)
        return _out_book(r[0], static.len(o.shape)), r[1]

def _krandinit(data2, k: int, g, nd: Literal[int]):
    n = data2.shape[0]
    d = data2.shape[1]
    mu = np.mean(data2, axis=0)
    if nd == 1:
        var = 0.0
        for i in range(n):
            var += (data2[i, 0] - mu[0]) ** 2
        var /= (n - 1)
        x = g.standard_normal(size=k)
        out = np.empty((k, 1), dtype=float)
        for i in range(k):
            out[i, 0] = x[i] * sqrt(var) + mu[0]
        return out
    elif d > n:
        U, s, vh = np.linalg.svd(data2 - mu, full_matrices=False)
        x = g.standard_normal(size=(k, s.shape[0]))
        sVh = s[:, None] * vh / sqrt(n - 1.0)
        return x @ sVh + mu
    else:
        c = data2 - mu
        cov = (c.T @ c) / (n - 1.0)
        x = g.standard_normal(size=(k, d))
        return x @ np.linalg.cholesky(cov).T + mu

def _kpp(data2, k: int, g):
    n = data2.shape[0]
    d = data2.shape[1]
    init = np.empty((k, d), dtype=float)
    for i in range(k):
        idx = 0
        if i == 0:
            idx = int(g.integers(n))
        else:
            D2 = np.empty(n, dtype=float)
            for j in range(n):
                best = inf
                for q in range(i):
                    s = 0.0
                    for t in range(d):
                        df = init[q, t] - data2[j, t]
                        s += df * df
                    if s < best:
                        best = s
                D2[j] = best
            probs = D2 / D2.sum()
            cum = probs.cumsum()
            r = g.uniform()
            idx = int(np.searchsorted(cum, r))
        for t in range(d):
            init[i, t] = data2[idx, t]
    return init

def kmeans2(data, k, iter: int = 10, thresh: float = 1e-5, minit: str = 'random',
            missing: str = 'warn', check_finite: bool = True, rng = None):
    if iter < 1:
        raise ValueError(f"Invalid iter ({iter}), must be a positive integer.")
    if missing != 'warn' and missing != 'raise':
        raise ValueError(f"Unknown missing method {missing!r}")
    a = np.asarray(data, dtype=float)
    nd: Literal[int] = static.len(a.shape)
    d2 = _as2(a)
    if a.size < 1:
        raise ValueError("Empty input is not supported.")
    book = np.empty((0, d2.shape[1]), dtype=float)
    nc = 0
    if isinstance(k, int):
        nc = k
        if nc < 1:
            raise ValueError(f"Cannot ask kmeans2 for {nc} clusters (k was {k})")
        g = _get_rng(rng)
        if minit == 'random':
            book = _krandinit(d2, nc, g, nd)
        elif minit == 'points':
            idx = g.choice(d2.shape[0], size=nc, replace=False)
            book = _take_rows(d2, idx)
        elif minit == '++':
            book = _kpp(d2, nc, g)
        else:
            raise ValueError(f"Unknown init method {minit!r}")
    else:
        kb = np.asarray(k, dtype=float)
        if static.len(kb.shape) != nd:
            compile_error("k array doesn't match data rank")
        book = _as2(kb).copy()
        nc = book.shape[0]
        if book.shape[1] != d2.shape[1]:
            raise ValueError("k array doesn't match data dimension")
    labels = List[int]()
    for _ in range(iter):
        labels = _vq2(d2, book)[0]
        cbh = _update_means(d2, labels, nc)
        nb = cbh[0]
        has = cbh[1]
        for c in range(nc):
            if not has[c]:
                if missing == 'raise':
                    raise ClusterError("One of the clusters is empty. Re-run kmeans with a different initialization.")
                for t in range(d2.shape[1]):
                    nb[c, t] = book[c, t]
        book = nb
    lab = np.empty(len(labels), dtype=np.int32)
    for i in range(len(labels)):
        lab[i] = np.int32(labels[i])
    return _out_book(book, nd), lab
