# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# multiscale_graphcorr (MGC) with the default euclidean distance.

import numpy as np
import scipy.special._kernels as K
from scipy.spatial.distance import cdist
from numpy.random.legacy import RandomState
from ._distn_infrastructure import check_random_state
from ._continuous_distns import beta as _beta_dist

class MGCResult:
    statistic: float
    pvalue: float
    mgc_map: np.ndarray[float, 2]
    opt_scale: List[int]
    null_dist: np.ndarray[float, 1]
    def __init__(self, statistic: float, pvalue: float, mgc_map: np.ndarray[float, 2],
                 opt_scale: List[int], null_dist: np.ndarray[float, 1]):
        self.statistic = statistic
        self.pvalue = pvalue
        self.mgc_map = mgc_map
        self.opt_scale = opt_scale
        self.null_dist = null_dist
    @property
    def stat(self) -> float:
        return self.statistic

def _dense_rank_col(x: np.ndarray[float, 1]) -> np.ndarray[int, 1]:
    n = x.shape[0]
    order = np.argsort(x)
    out = np.empty(n, dtype=int)
    r = 0
    prev = 0.0
    for i in range(n):
        v = x[int(order[i])]
        if i == 0 or v != prev:
            r += 1
            prev = v
        out[int(order[i])] = r
    return out

def _rank_matrix(d: np.ndarray[float, 2]) -> np.ndarray[int, 2]:
    n = d.shape[0]
    out = np.empty((n, n), dtype=int)
    for j in range(n):
        col = _dense_rank_col(np.ascontiguousarray(d[:, j]))
        for i in range(n):
            out[i, j] = col[i]
    return out

def _center(d: np.ndarray[float, 2]) -> np.ndarray[float, 2]:
    n = d.shape[0]
    out = np.empty((n, n), dtype=float)
    for j in range(n):
        s = 0.0
        for i in range(n):
            s += d[i, j]
        m = s / n * n / (n - 1)
        for i in range(n):
            out[i, j] = d[i, j] - m
    for i in range(n):
        out[i, i] = 0.0
    return out

def _local_cov(distx: np.ndarray[float, 2], disty: np.ndarray[float, 2],
               rx: np.ndarray[int, 2], ry: np.ndarray[int, 2]) -> np.ndarray[float, 2]:
    # distx, disty already arranged as in scipy (disty is transposed by caller)
    n = distx.shape[0]
    nx = int(np.max(rx))
    ny = int(np.max(ry))
    cov = np.zeros((nx, ny))
    ex = np.zeros(nx)
    ey = np.zeros(ny)
    for i in range(n):
        for j in range(n):
            a = distx[i, j]
            b = disty[i, j]
            k = int(rx[i, j]) - 1
            l = int(ry[i, j]) - 1
            cov[k, l] += a * b
            ex[k] += a
            ey[l] += b
    for k in range(1, nx):
        cov[k, 0] += cov[k - 1, 0]
        ex[k] += ex[k - 1]
    for l in range(1, ny):
        cov[0, l] += cov[0, l - 1]
        ey[l] += ey[l - 1]
    for k in range(nx - 1):
        for l in range(ny - 1):
            cov[k + 1, l + 1] += cov[k + 1, l] + cov[k, l + 1] - cov[k, l]
    for k in range(nx):
        for l in range(ny):
            cov[k, l] = cov[k, l] - ex[k] * ey[l] / (n * n)
    return cov

def _local_correlations(distx: np.ndarray[float, 2], disty: np.ndarray[float, 2]) -> np.ndarray[float, 2]:
    rx = _rank_matrix(distx)
    ry = _rank_matrix(disty)
    cx = _center(distx)
    cy = _center(disty)
    cov = _local_cov(cx, cy.T, rx, ry.T)
    vx = _local_cov(cx, cx.T, rx, rx.T)
    vy = _local_cov(cy, cy.T, ry, ry.T)
    nx = cov.shape[0]
    ny = cov.shape[1]
    corr = np.empty((nx, ny), dtype=float)
    for k in range(nx):
        for l in range(ny):
            a = vx[k, k]
            b = vy[l, l]
            c = cov[k, l] / K.sqrt(a * b) if a * b > 0 else (K.NAN if a * b == 0 else K.NAN)
            if c > 1:
                c = 1.0
            corr[k, l] = c
    for k in range(nx):
        if vx[k, k] <= 0:
            for l in range(ny):
                corr[k, l] = 0.0
    for l in range(ny):
        if vy[l, l] <= 0:
            for k in range(nx):
                corr[k, l] = 0.0
    return corr

def _label_largest(mask: np.ndarray[bool, 2]) -> np.ndarray[bool, 2]:
    # 4-connected components in raster-scan order; keep the (first) largest
    m = mask.shape[0]
    n = mask.shape[1]
    lab = np.zeros((m, n), dtype=int)
    counts = List[int]()
    nl = 0
    for i in range(m):
        for j in range(n):
            if mask[i, j] and lab[i, j] == 0:
                nl += 1
                stack = [(i, j)]
                lab[i, j] = nl
                cnt = 0
                while len(stack) > 0:
                    p = stack.pop()
                    cnt += 1
                    for d in range(4):
                        ni = p[0] + (-1 if d == 0 else (1 if d == 1 else 0))
                        nj = p[1] + (-1 if d == 2 else (1 if d == 3 else 0))
                        if ni >= 0 and ni < m and nj >= 0 and nj < n and mask[ni, nj] and lab[ni, nj] == 0:
                            lab[ni, nj] = nl
                            stack.append((ni, nj))
                counts.append(cnt)
    best = 0
    for q in range(1, len(counts)):
        if counts[q] > counts[best]:
            best = q
    out = np.zeros((m, n), dtype=bool)
    if nl > 0:
        for i in range(m):
            for j in range(n):
                out[i, j] = lab[i, j] == best + 1
    return out

def _mgc_stat(distx: np.ndarray[float, 2], disty: np.ndarray[float, 2]):
    smap = _local_correlations(distx, disty)
    m = smap.shape[0]
    n = smap.shape[1]
    stat = smap[m - 1, n - 1]
    opt = [m, n]
    if m == 1 or n == 1:
        return (stat, smap, [m * n])
    samp = distx.shape[0] - 1
    per_sig = 1 - (0.02 / samp)
    thr = samp * (samp - 3) / 4 - 1 / 2
    thr = _beta_dist.ppf(per_sig, thr, thr) * 2 - 1
    thr = max(thr, smap[m - 1, n - 1])
    sig = np.zeros((m, n), dtype=bool)
    anyc = False
    for i in range(m):
        for j in range(n):
            if smap[i, j] > thr:
                sig[i, j] = True
                anyc = True
    if anyc:
        sig = _label_largest(sig)
    else:
        sig = np.zeros((m, n), dtype=bool)
    total = 0
    mx = -K.INF
    for i in range(m):
        for j in range(n):
            if sig[i, j]:
                total += 1
                if smap[i, j] > mx:
                    mx = smap[i, j]
    if total > 0:
        if total >= int(K.ceil(0.02 * max(m, n))) * min(m, n):
            if mx >= stat:
                stat = mx
                best = -1
                for i in range(m):
                    for j in range(n):
                        if smap[i, j] >= mx and sig[i, j]:
                            best = max(best, i * n + j)
                opt = [best // n + 1, best % n + 1]
    return (stat, smap, opt)

def _two_sample(u: np.ndarray[float, 2], v: np.ndarray[float, 2]):
    nx = u.shape[0]
    ny = v.shape[0]
    x = np.concatenate((u, v), axis=0)
    y = np.concatenate((np.zeros(nx), np.ones(ny))).reshape((-1, 1))
    return (x, y)

def multiscale_graphcorr(x, y, reps: int = 1000, workers: int = 1, is_twosamp: bool = False,
                         random_state = None):
    xa = np.asarray(x).astype(float)
    ya = np.asarray(y).astype(float)
    if xa.ndim == 1:
        xa = xa.reshape((-1, 1))
    elif xa.ndim != 2:
        raise ValueError(f"Expected a 2-D array `x`, found shape {xa.shape}")
    if ya.ndim == 1:
        ya = ya.reshape((-1, 1))
    elif ya.ndim != 2:
        raise ValueError(f"Expected a 2-D array `y`, found shape {ya.shape}")
    nx = xa.shape[0]
    ny = ya.shape[0]
    if bool(np.any(np.isnan(xa))) or bool(np.any(np.isnan(ya))):
        raise ValueError("The input contains nan values")
    if bool(np.any(np.isinf(xa))) or bool(np.any(np.isinf(ya))):
        raise ValueError("Inputs contain infinities")
    two = is_twosamp
    if nx != ny:
        if xa.shape[1] == ya.shape[1]:
            two = True
        else:
            raise ValueError("Shape mismatch, x and y must have shape [n, p] and [m, p].")
    if nx < 5 or ny < 5:
        raise ValueError("MGC requires at least 5 samples to give reasonable results.")
    if reps < 0:
        raise ValueError("Number of reps must be an integer greater than 0.")
    if two:
        t = _two_sample(xa, ya)
        xa = t[0]
        ya = t[1]
    dx = cdist(xa, xa)
    dy = cdist(ya, ya)
    r = _mgc_stat(dx, dy)
    stat = r[0]
    rs = check_random_state(random_state)
    n = dy.shape[0]
    null = np.empty(reps, dtype=float)
    seeds = rs.randint(0, 1 << 32, size=(reps, 4))
    for i in range(reps):
        prs = RandomState(np.asarray(seeds[i]).astype(np.uint32))
        order = prs.permutation(n)
        permy = np.empty((n, n), dtype=float)
        for a in range(n):
            for b in range(n):
                permy[a, b] = dy[int(order[a]), int(order[b])]
        null[i] = _mgc_stat(dx, permy)[0]
    cnt = 0
    for i in range(reps):
        if null[i] >= stat:
            cnt += 1
    pvalue = (1 + cnt) / (1 + reps)
    return MGCResult(stat, pvalue, r[1], r[2], null)
