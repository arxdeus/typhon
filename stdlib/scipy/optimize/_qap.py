# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)
# Quadratic assignment problem: 'faq' (Fast Approximate QAP, Vogelstein et al.)
# and '2opt' (scipy/optimize/_qap.py). Options are explicit keyword arguments.

import numpy as np
from ._lsap import linear_sum_assignment

class QAPResult:
    col_ind: np.ndarray[int, 1]
    fun: float
    nit: int
    def __init__(self, col_ind: np.ndarray[int, 1], fun: float, nit: int):
        self.col_ind = col_ind
        self.fun = fun
        self.nit = nit

def _calc_score(A: np.ndarray[float, 2], B: np.ndarray[float, 2], perm: np.ndarray[int, 1]) -> float:
    n = A.shape[0]
    s = 0.0
    for i in range(n):
        for j in range(n):
            s += A[i, j] * B[perm[i], perm[j]]
    return s

def _as_matrix(X) -> np.ndarray[float, 2]:
    a = np.atleast_2d(np.asarray(X)).astype(float)
    return a

def _pm_array(pm) -> np.ndarray[int, 2]:
    if pm is None:
        return np.zeros((0, 2), dtype=int)
    a = np.atleast_2d(np.asarray(pm)).astype(int)
    if a.size == 0:
        return np.zeros((0, 2), dtype=int)
    return a

def _validate(A: np.ndarray[float, 2], B: np.ndarray[float, 2], pm: np.ndarray[int, 2], what: str):
    n = A.shape[0]
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("`A` and `B` must have exactly two dimensions")
    if A.shape[0] != A.shape[1]:
        raise ValueError("`A` must be square")
    if B.shape[0] != B.shape[1]:
        raise ValueError("`B` must be square")
    if A.shape[0] != B.shape[0]:
        raise ValueError("`A` and `B` matrices must be of equal size")
    if pm.shape[0] > n:
        raise ValueError(f"`{what}` can have only as many seeds as there are nodes")
    if pm.shape[1] != 2:
        raise ValueError(f"`{what}` must have two columns")
    seen0 = [False] * n
    seen1 = [False] * n
    for i in range(pm.shape[0]):
        for c in range(2):
            if pm[i, c] < 0:
                raise ValueError(f"`{what}` must contain only positive indices")
            if pm[i, c] >= n:
                raise ValueError(f"`{what}` entries must be less than number of nodes")
        if seen0[pm[i, 0]] or seen1[pm[i, 1]]:
            raise ValueError(f"`{what}` column entries must be unique")
        seen0[pm[i, 0]] = True
        seen1[pm[i, 1]] = True

def _sub(X: np.ndarray[float, 2], rows: np.ndarray[int, 1], cols: np.ndarray[int, 1], r0: int, r1: int, c0: int, c1: int) -> np.ndarray[float, 2]:
    out = np.empty((r1 - r0, c1 - c0))
    for i in range(r0, r1):
        for j in range(c0, c1):
            out[i - r0, j - c0] = X[rows[i], cols[j]]
    return out

def _doubly_stochastic(P: np.ndarray[float, 2], tol: float = 1e-3) -> np.ndarray[float, 2]:
    max_iter = 1000
    c = 1.0 / P.sum(axis=0)
    r = 1.0 / np.dot(P, c)
    P_eps = P
    for it in range(max_iter):
        ok = True
        rs = P_eps.sum(axis=1)
        cs = P_eps.sum(axis=0)
        for i in range(rs.shape[0]):
            if abs(rs[i] - 1.0) >= tol or abs(cs[i] - 1.0) >= tol:
                ok = False
        if ok:
            break
        c = 1.0 / np.dot(r, P)
        r = 1.0 / np.dot(P, c)
        P_eps = r.reshape(-1, 1) * P * c
    return P_eps

def _setdiff(n: int, used: np.ndarray[int, 1]) -> np.ndarray[int, 1]:
    mask = [False] * n
    for i in range(used.shape[0]):
        mask[used[i]] = True
    out = List[int]()
    for i in range(n):
        if not mask[i]:
            out.append(i)
    return np.array(out, dtype=int)

def _faq(A: np.ndarray[float, 2], B: np.ndarray[float, 2], maximize: bool, pm: np.ndarray[int, 2], rng,
         P0, shuffle_input: bool, maxiter: int, tol: float) -> QAPResult:
    _validate(A, B, pm, 'partial_match')
    n = A.shape[0]
    n_seeds = pm.shape[0]
    n_unseed = n - n_seeds
    if maxiter <= 0:
        raise ValueError("'maxiter' must be a positive integer")
    if tol <= 0:
        raise ValueError("'tol' must be a positive float")
    g = np.random.default_rng(rng)
    P = np.ones((n_unseed, n_unseed)) / max(n_unseed, 1)
    if isinstance(P0, str):
        if P0 == 'randomized':
            J = np.ones((n_unseed, n_unseed)) / n_unseed
            K = _doubly_stochastic(g.uniform(size=(n_unseed, n_unseed)))
            P = (J + K) / 2
        elif P0 != 'barycenter':
            raise ValueError("Invalid 'P0' parameter string")
    else:
        P = _as_matrix(P0)
        if P.shape[0] != n_unseed or P.shape[1] != n_unseed:
            raise ValueError("`P0` matrix must have shape m' x m', where m'=n-m")
        rs = P.sum(axis=1)
        cs = P.sum(axis=0)
        for i in range(n_unseed):
            if rs[i] < 1 - 1e-8 or rs[i] > 1 + 1e-8 or cs[i] < 1 - 1e-8 or cs[i] > 1 + 1e-8:
                raise ValueError("`P0` matrix must be doubly stochastic")
            for j in range(n_unseed):
                if P[i, j] < 0:
                    raise ValueError("`P0` matrix must be doubly stochastic")
    if n == 0 or n_seeds == n:
        pc = np.empty(n_seeds, dtype=int)
        for i in range(n_seeds):
            pc[i] = pm[i, 1]
        return QAPResult(pc, _calc_score(A, B, pc), 0)
    scal = -1.0 if maximize else 1.0
    seedB = np.empty(n_seeds, dtype=int)
    seedA = np.empty(n_seeds, dtype=int)
    for i in range(n_seeds):
        seedB[i] = pm[i, 1]
        seedA[i] = pm[i, 0]
    nonseed_B = _setdiff(n, seedB)
    if shuffle_input:
        nonseed_B = g.permutation(nonseed_B)
    nonseed_A = _setdiff(n, seedA)
    perm_A = np.concatenate((seedA, nonseed_A))
    perm_B = np.concatenate((seedB, nonseed_B))
    A12 = _sub(A, perm_A, perm_A, 0, n_seeds, n_seeds, n)
    A21 = _sub(A, perm_A, perm_A, n_seeds, n, 0, n_seeds)
    A22 = _sub(A, perm_A, perm_A, n_seeds, n, n_seeds, n)
    B12 = _sub(B, perm_B, perm_B, 0, n_seeds, n_seeds, n)
    B21 = _sub(B, perm_B, perm_B, n_seeds, n, 0, n_seeds)
    B22 = _sub(B, perm_B, perm_B, n_seeds, n, n_seeds, n)
    const_sum = np.dot(A21, B21.T) + np.dot(A12.T, B12)
    A22T = A22.T.copy()
    B22T = B22.T.copy()
    n_iter = 0
    for it in range(1, maxiter + 1):
        n_iter = it
        grad_fp = const_sum + np.dot(np.dot(A22, P), B22T) + np.dot(np.dot(A22T, P), B22)
        lsa = linear_sum_assignment(grad_fp, maximize)
        cols = lsa[1]
        Q = np.zeros((n_unseed, n_unseed))
        for i in range(n_unseed):
            Q[i, cols[i]] = 1.0
        R = P - Q
        RT = R.T.copy()
        b21 = (np.dot(RT, A21) * B21).sum()
        b12 = (np.dot(RT, A12.T) * B12.T).sum()
        AR22 = np.dot(A22T, R)
        BR22 = np.dot(B22, RT)
        b22a = 0.0
        b22b = 0.0
        for i in range(n_unseed):
            for j in range(n_unseed):
                b22a += AR22[i, j] * B22T[cols[i], j]
                b22b += A22[i, j] * BR22[cols[i], j]
        a = (AR22.T * BR22).sum()
        b = b21 + b12 + b22a + b22b
        alpha = 0.0
        if a * scal > 0 and 0 <= -b / (2 * a) and -b / (2 * a) <= 1:
            alpha = -b / (2 * a)
        else:
            v1 = (b + a) * scal
            alpha = 1.0 if v1 < 0.0 else 0.0
        P_i1 = alpha * P + (1 - alpha) * Q
        d = P - P_i1
        nrm = 0.0
        for i in range(n_unseed):
            for j in range(n_unseed):
                nrm += d[i, j] * d[i, j]
        if np.sqrt(nrm) / np.sqrt(float(n_unseed)) < tol:
            P = P_i1
            break
        P = P_i1
    lsa2 = linear_sum_assignment(P, True)
    col = lsa2[1]
    unshuf = np.zeros(n, dtype=int)
    for i in range(n_seeds):
        unshuf[perm_A[i]] = perm_B[i]
    for i in range(n_unseed):
        unshuf[perm_A[n_seeds + i]] = perm_B[n_seeds + col[i]]
    return QAPResult(unshuf, _calc_score(A, B, unshuf), n_iter)

def _two_opt(A: np.ndarray[float, 2], B: np.ndarray[float, 2], maximize: bool, rng, pm: np.ndarray[int, 2],
             pg: np.ndarray[int, 2]) -> QAPResult:
    g = np.random.default_rng(rng)
    _validate(A, B, pm, 'partial_match')
    N = A.shape[0]
    if N == 0 or pm.shape[0] == N:
        pc = np.empty(pm.shape[0], dtype=int)
        for i in range(pm.shape[0]):
            pc[i] = pm[i, 1]
        return QAPResult(pc, _calc_score(A, B, pc), 0)
    _validate(A, B, pg, 'partial_guess')
    perm = np.zeros(N, dtype=int)
    fixed_rows = [False] * N
    has_part = pm.shape[0] > 0 or pg.shape[0] > 0
    if has_part:
        guess_rows = [False] * N
        guess_cols = [False] * N
        fixed_cols = [False] * N
        for i in range(pg.shape[0]):
            guess_rows[pg[i, 0]] = True
            guess_cols[pg[i, 1]] = True
            perm[pg[i, 0]] = pg[i, 1]
        for i in range(pm.shape[0]):
            fixed_rows[pm[i, 0]] = True
            fixed_cols[pm[i, 1]] = True
            perm[pm[i, 0]] = pm[i, 1]
        rcols = List[int]()
        for i in range(N):
            if not fixed_cols[i] and not guess_cols[i]:
                rcols.append(i)
        shuffled = g.permutation(np.array(rcols, dtype=int))
        k = 0
        for i in range(N):
            if not fixed_rows[i] and not guess_rows[i]:
                perm[i] = shuffled[k]
                k += 1
    else:
        perm = g.permutation(np.arange(N))
    best = _calc_score(A, B, perm)
    free = List[int]()
    for i in range(N):
        if not (has_part and fixed_rows[i]):
            free.append(i)
    nf = len(free)
    n_iter = 0
    done = False
    while not done:
        swapped = False
        for ii in range(nf):
            for jj in range(ii, nf):
                i = free[ii]
                j = free[jj]
                n_iter += 1
                t = perm[i]
                perm[i] = perm[j]
                perm[j] = t
                score = _calc_score(A, B, perm)
                if (maximize and score > best) or ((not maximize) and score < best):
                    best = score
                    swapped = True
                    break
                t = perm[i]
                perm[i] = perm[j]
                perm[j] = t
            if swapped:
                break
        if not swapped:
            done = True
    return QAPResult(perm, best, n_iter)

def quadratic_assignment(A, B, method: str = 'faq', maximize: bool = False, partial_match=None, rng=None,
                         P0='barycenter', shuffle_input: bool = False, maxiter: int = 30, tol: float = 0.03,
                         partial_guess=None):
    a = _as_matrix(A)
    b = _as_matrix(B)
    pm = _pm_array(partial_match)
    if method == 'faq':
        return _faq(a, b, maximize, pm, rng, P0, shuffle_input, maxiter, tol)
    elif method == '2opt':
        return _two_opt(a, b, maximize, rng, pm, _pm_array(partial_guess))
    raise ValueError(f"method must be one of {{'faq', '2opt'}}, got {method}")
