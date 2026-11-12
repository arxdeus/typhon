# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Bipartite matching (scipy/sparse/csgraph/_matching.pyx):
# maximum_bipartite_matching (Hopcroft-Karp) and
# min_weight_full_bipartite_matching (LAPJVsp), line-by-line ports so the
# returned matchings are identical to scipy's.

import numpy as np
from .._base import _issparse

_INF_I = 2147483647
_INF = float("inf")

def _hopcroft_karp(indices, indptr, i: int, j: int):
    INF = _INF_I
    x = np.full(i, -1, dtype=int)
    y = np.full(j, i, dtype=int)
    dist = np.empty(i + 1, dtype=int)
    q = np.empty(i + 1, dtype=int)
    stack = np.empty(i + 1, dtype=int)
    parents = np.empty(max(i, 1), dtype=int)
    while True:
        head = 0
        tail = 0
        for v in range(i):
            if x[v] < 0:
                dist[v] = 0
                q[tail] = v
                tail += 1
            else:
                dist[v] = INF
        dist[i] = INF
        while head < tail:
            v = q[head]
            head += 1
            if dist[v] < dist[i]:
                for up in range(indptr[v], indptr[v + 1]):
                    u = indices[up]
                    if dist[y[u]] == INF:
                        dist[y[u]] = dist[v] + 1
                        q[tail] = y[u]
                        tail += 1
        if dist[i] == INF:
            break
        for w in range(i):
            if x[w] < 0:
                done = False
                stack[0] = w
                stack_head = 1
                while stack_head != 0:
                    stack_head -= 1
                    v = stack[stack_head]
                    for up in range(indptr[v], indptr[v + 1]):
                        u = indices[up]
                        yu = y[u]
                        if dist[yu] == dist[v] + 1:
                            if yu == i:
                                done = True
                                while True:
                                    dist[v] = INF
                                    u_old = x[v]
                                    y[u] = v
                                    x[v] = u
                                    u = u_old
                                    if v == w:
                                        break
                                    v = parents[v]
                                break
                            stack[stack_head] = yu
                            stack_head += 1
                            parents[yu] = v
                    if done:
                        break
    for k in range(j):
        if y[k] == i:
            y[k] = -1
    return x, y

def maximum_bipartite_matching(graph, perm_type: str = "row"):
    if not _issparse(graph):
        compile_error("graph must be sparse")
    G = graph.tocsr()
    i, j = G.shape
    x, y = _hopcroft_karp(G.indices, G.indptr, i, j)
    return x if perm_type == "column" else y

def _update_dual(nc: int, d, v, todo, last: int, min_diff: float):
    for k in range(last, nc):
        j0 = todo[k]
        v[j0] += d[j0] - min_diff

def _update_assignments(lab, y, x, j: int, i0: int):
    while True:
        i = lab[j]
        y[j] = i
        k = j
        j = x[i]
        x[i] = k
        if i == i0:
            return

def _lapjvsp_single_l(l: int, nc: int, d, ok, free, first, kk, cc, v, lab,
                      todo, y, x, td1: int) -> int:
    for jp in range(nc):
        d[jp] = _INF
        ok[jp] = 0
    min_diff = _INF
    i0 = free[l]
    for t in range(first[i0], first[i0 + 1]):
        j = kk[t]
        dj = cc[t] - v[j]
        d[j] = dj
        lab[j] = i0
        if dj <= min_diff:
            if dj < min_diff:
                td1 = -1
                min_diff = dj
            td1 += 1
            todo[td1] = j
    for hp in range(td1 + 1):
        j = todo[hp]
        if y[j] == -1:
            _update_assignments(lab, y, x, j, i0)
            return td1
        ok[j] = 1
    td2 = nc - 1
    last = nc
    while True:
        if td1 < 0:
            raise ValueError("no full matching exists")
        j0 = todo[td1]
        td1 -= 1
        i = y[j0]
        todo[td2] = j0
        td2 -= 1
        tp = first[i]
        while kk[tp] != j0:
            tp += 1
        h = cc[tp] - v[j0] - min_diff
        for t in range(first[i], first[i + 1]):
            j = kk[t]
            if ok[j] == 0:
                vj = cc[t] - v[j] - h
                if vj < d[j]:
                    d[j] = vj
                    lab[j] = i
                    if vj == min_diff:
                        if y[j] == -1:
                            _update_dual(nc, d, v, todo, last, min_diff)
                            _update_assignments(lab, y, x, j, i0)
                            return td1
                        td1 += 1
                        todo[td1] = j
                        ok[j] = 1
        if td1 == -1:
            min_diff = _INF
            last = td2 + 1
            for jp in range(nc):
                if d[jp] != _INF and d[jp] <= min_diff and ok[jp] == 0:
                    if d[jp] < min_diff:
                        td1 = -1
                        min_diff = d[jp]
                    td1 += 1
                    todo[td1] = jp
            for hp in range(td1 + 1):
                j = todo[hp]
                if y[j] == -1:
                    _update_dual(nc, d, v, todo, last, min_diff)
                    _update_assignments(lab, y, x, j, i0)
                    return td1
                ok[j] = 1
    return td1

def _lapjvsp(first, kk, cc, nr: int, nc: int):
    v = np.zeros(nc, dtype=float)
    x = np.full(nr, -1, dtype=int)
    y = np.full(nc, -1, dtype=int)
    u = np.zeros(nr, dtype=float)
    d = np.zeros(nc, dtype=float)
    ok = np.zeros(nc, dtype=int)
    xinv = np.zeros(nr, dtype=int)
    free = np.full(nr, -1, dtype=int)
    todo = np.full(nc, -1, dtype=int)
    lab = np.zeros(nc, dtype=int)
    if nr == nc:
        for jp in range(nc):
            v[jp] = _INF
        for i in range(nr):
            for t in range(first[i], first[i + 1]):
                jp = kk[t]
                if cc[t] < v[jp]:
                    v[jp] = cc[t]
                    y[jp] = i
        for jq in range(nc):
            jp = nc - 1 - jq
            i = y[jp]
            if i == -1:
                raise ValueError("no full matching exists")
            if x[i] == -1:
                x[i] = jp
            else:
                y[jp] = -1
                xinv[i] = 1
        lp = 0
        for i in range(nr):
            if xinv[i] == 1:
                continue
            if x[i] != -1:
                min_diff = _INF
                j1 = x[i]
                for t in range(first[i], first[i + 1]):
                    jp = kk[t]
                    if jp != j1:
                        if cc[t] - v[jp] < min_diff:
                            min_diff = cc[t] - v[jp]
                u[i] = min_diff
                tp = first[i]
                while kk[tp] != j1:
                    tp += 1
                v[j1] = cc[tp] - min_diff
            else:
                free[lp] = i
                lp += 1
        for _ in range(2):
            h = 0
            l0p = lp
            lp = 0
            while h < l0p:
                i = free[h]
                h += 1
                j0p = -1
                j1p = -1
                v0 = _INF
                vj = _INF
                for t in range(first[i], first[i + 1]):
                    jp = kk[t]
                    dj = cc[t] - v[jp]
                    if dj < vj:
                        if dj >= v0:
                            vj = dj
                            j1p = jp
                        else:
                            vj = v0
                            v0 = dj
                            j1p = j0p
                            j0p = jp
                if j0p < 0:
                    raise ValueError("no full matching exists")
                i0 = y[j0p]
                u[i] = vj
                if v0 < vj:
                    v[j0p] += v0 - vj
                elif i0 != -1:
                    j0p = j1p
                    i0 = y[j0p]
                x[i] = j0p
                y[j0p] = i
                if i0 != -1:
                    if v0 < vj:
                        h -= 1
                        free[h] = i0
                    else:
                        free[lp] = i0
                        lp += 1
        l0 = lp
    else:
        l0 = nr
        for i in range(nr):
            free[i] = i
    td1 = -1
    for l in range(l0):
        td1 = _lapjvsp_single_l(l, nc, d, ok, free, first, kk, cc, v, lab,
                                todo, y, x, td1)
    return x

def min_weight_full_bipartite_matching(biadjacency, maximize: bool = False):
    if not _issparse(biadjacency):
        compile_error("graph must be sparse")
    B = biadjacency.tocsr().astype(float)
    B.sum_duplicates()
    if maximize:
        B = -B
    for k in range(len(B.data)):
        if B.data[k] == _INF:
            B.data[k] = 0.0
    B.eliminate_zeros()
    i, j = B.shape
    a = np.arange(min(i, j))
    if j < i:
        Bt = B.T.tocsr()
        m, _ = _hopcroft_karp(Bt.indices, Bt.indptr, j, i)
        cnt = 0
        for q in range(len(m)):
            if m[q] != -1:
                cnt += 1
        if cnt != min(i, j):
            raise ValueError("no full matching exists")
        bb = _lapjvsp(Bt.indptr, Bt.indices, Bt.data, j, i)
        idx = np.argsort(bb, kind="stable")
        return bb[idx], a[idx]
    else:
        m, _ = _hopcroft_karp(B.indices, B.indptr, i, j)
        cnt = 0
        for q in range(len(m)):
            if m[q] != -1:
                cnt += 1
        if cnt != min(i, j):
            raise ValueError("no full matching exists")
        bb = _lapjvsp(B.indptr, B.indices, B.data, i, j)
        return a, bb
