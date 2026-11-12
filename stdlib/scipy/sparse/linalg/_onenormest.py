# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# onenormest: block 1-norm estimator of Higham & Tisseur (scipy
# _onenormest.py), including the resampling of parallel columns with the
# legacy global numpy RNG (np.random.randint), so results match scipy for the
# same np.random.seed. Real operators only (as scipy's float work arrays).

import numpy as np
from ._interface import aslinearoperator

def _vectors_are_parallel(v, w, n: int) -> bool:
    s = 0.0
    for i in range(n):
        s += v[i] * w[i]
    return s == float(n)

def _col(X, j: int):
    return np.ascontiguousarray(X[:, j])

def _column_needs_resampling(i: int, X, Y, haveY: bool) -> bool:
    n = X.shape[0]
    v = _col(X, i)
    for j in range(i):
        if _vectors_are_parallel(v, _col(X, j), n):
            return True
    if haveY:
        for j in range(Y.shape[1]):
            if _vectors_are_parallel(v, _col(Y, j), n):
                return True
    return False

def _resample_column(i: int, X):
    r = np.random.randint(0, 2, size=X.shape[0])
    for q in range(X.shape[0]):
        X[q, i] = float(r[q] * 2 - 1)

def _every_col_parallel(X, Y) -> bool:
    n = X.shape[0]
    for a in range(X.shape[1]):
        v = _col(X, a)
        found = False
        for b in range(Y.shape[1]):
            if _vectors_are_parallel(v, _col(Y, b), n):
                found = True
                break
        if not found:
            return False
    return True

def _in(v: int, lst: List[int]) -> bool:
    for u in lst:
        if u == v:
            return True
    return False

def _onenormest_core(Aop, AHop, t: int, itmax: int):
    if itmax < 2:
        raise ValueError("at least two iterations are required")
    if t < 1:
        raise ValueError("at least one column is required")
    n = Aop.shape[0]
    if t >= n:
        raise ValueError("t should be smaller than the order of A")
    X = np.ones((n, t), dtype=float)
    if t > 1:
        for i in range(1, t):
            _resample_column(i, X)
        for i in range(t):
            while _column_needs_resampling(i, X, X, False):
                _resample_column(i, X)
    X = X / float(n)
    ind_hist = List[int]()
    est_old = 0.0
    S = np.zeros((n, t), dtype=float)
    k = 1
    ind = List[int]()
    ind_best = 0
    w = np.zeros(n, dtype=float)
    est = 0.0
    while True:
        Y = np.asarray(Aop.matmat(X)).astype(float)
        mags = np.zeros(t, dtype=float)
        for j in range(t):
            s = 0.0
            for q in range(n):
                s += abs(Y[q, j])
            mags[j] = s
        best_j = 0
        for j in range(t):
            if mags[j] > mags[best_j]:
                best_j = j
        est = mags[best_j]
        if est > est_old or k == 2:
            if k >= 2:
                ind_best = ind[best_j]
            w = _col(Y, best_j)
        if k >= 2 and est <= est_old:
            est = est_old
            break
        est_old = est
        S_old = S
        if k > itmax:
            break
        S = Y.copy()
        for a in range(n):
            for b in range(t):
                if S[a, b] == 0.0:
                    S[a, b] = 1.0
                S[a, b] = S[a, b] / abs(S[a, b])
        if _every_col_parallel(S, S_old):
            break
        if t > 1:
            for i in range(t):
                while _column_needs_resampling(i, S, S_old, True):
                    _resample_column(i, S)
        Z = np.asarray(AHop.matmat(S)).astype(float)
        h = np.zeros(n, dtype=float)
        for a in range(n):
            mx = 0.0
            for b in range(t):
                mx = max(mx, abs(Z[a, b]))
            h[a] = mx
        if k >= 2:
            hm = h[0]
            for a in range(n):
                hm = max(hm, h[a])
            if hm == h[ind_best]:
                break
        srt = np.argsort(h)
        full = List[int]()
        for q in range(n):
            full.append(int(srt[n - 1 - q]))
        ind = full[:t + len(ind_hist)]
        if t > 1:
            allin = True
            for q in range(t):
                if not _in(ind[q], ind_hist):
                    allin = False
            if allin:
                break
            a1 = List[int]()
            a2 = List[int]()
            for q in ind:
                if _in(q, ind_hist):
                    a2.append(q)
                else:
                    a1.append(q)
            ind = a1 + a2
        for j in range(t):
            for q in range(n):
                X[q, j] = 0.0
            X[ind[j], j] = 1.0
        for q in range(t):
            if not _in(ind[q], ind_hist):
                ind_hist.append(ind[q])
        k += 1
    v = np.zeros(n, dtype=float)
    v[ind_best] = 1.0
    return est, v, w

def onenormest(A, t: int = 2, itmax: int = 5, compute_v: Literal[bool] = False,
               compute_w: Literal[bool] = False):
    Aop = aslinearoperator(A)
    if Aop.shape[0] != Aop.shape[1]:
        raise ValueError("expected the operator to act like a square matrix")
    n = Aop.shape[1]
    if t >= n:
        E = np.asarray(Aop.matmat(np.identity(n)))
        best = 0
        sums = np.zeros(n, dtype=float)
        for j in range(n):
            s = 0.0
            for q in range(n):
                s += abs(E[q, j])
            sums[j] = s
            if s > sums[best]:
                best = j
        v = np.zeros(n, dtype=float)
        v[best] = 1.0
        w = np.ascontiguousarray(E[:, best]).astype(float)
        est = sums[best]
    else:
        est, v, w = _onenormest_core(Aop, Aop.H, t, itmax)
    if compute_v:
        if compute_w:
            return est, v, w
        else:
            return est, v
    else:
        if compute_w:
            return est, w
        else:
            return est
