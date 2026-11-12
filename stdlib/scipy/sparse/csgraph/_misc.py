# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# minimum_spanning_tree (Kruskal, _min_spanning_tree.pyx),
# reverse_cuthill_mckee and structural_rank (_reordering.pyx), laplacian
# (_laplacian.py, dense/sparse array forms), maximum_flow (Edmonds-Karp and
# Dinic, _flow.pyx).

import math
import numpy as np
from .._base import _compressed, _coo
from .._construct import csr_array, csr_matrix, coo_array
from ._validation import _csr_graph, _transpose_csr, _issparse
from ._traversal import _is_spmatrix

def minimum_spanning_tree(csgraph, overwrite: bool = False):
    G = _csr_graph(csgraph)
    N = G.shape[0]
    data = G.data.copy()
    indices = G.indices
    indptr = G.indptr
    nd = len(data)
    rank = np.zeros(N, dtype=int)
    pred = np.arange(N)
    i_sort = np.argsort(data, kind="stable")
    row_indices = np.zeros(nd, dtype=int)
    for i in range(N):
        for j in range(indptr[i], indptr[i + 1]):
            row_indices[j] = i
    n_edges = 0
    i = 0
    while i < nd and n_edges < N - 1:
        j = i_sort[i]
        V1 = row_indices[j]
        V2 = indices[j]
        R1 = V1
        while pred[R1] != R1:
            R1 = pred[R1]
        R2 = V2
        while pred[R2] != R2:
            R2 = pred[R2]
        while pred[V1] != R1:
            pred[V1] = R1
        while pred[V2] != R2:
            pred[V2] = R2
        if R1 != R2:
            n_edges += 1
            if rank[R1] > rank[R2]:
                pred[R2] = R1
            elif rank[R1] < rank[R2]:
                pred[R1] = R2
            else:
                pred[R2] = R1
                rank[R1] += 1
        else:
            data[j] = 0.0
        i += 1
    while i < nd:
        data[i_sort[i]] = 0.0
        i += 1
    if _is_spmatrix(csgraph):
        T = csr_matrix((data, indices.copy(), indptr.copy()), shape=(N, N))
        T.eliminate_zeros()
        return T
    else:
        T2 = csr_array((data, indices.copy(), indptr.copy()), shape=(N, N))
        T2.eliminate_zeros()
        return T2

def _node_degrees(ind, ptr, n: int):
    degree = np.zeros(n, dtype=int)
    for ii in range(n):
        degree[ii] = ptr[ii + 1] - ptr[ii]
        for jj in range(ptr[ii], ptr[ii + 1]):
            if ind[jj] == ii:
                degree[ii] += 1
                break
    return degree

def reverse_cuthill_mckee(graph, symmetric_mode: bool = False):
    if not _issparse(graph):
        compile_error("Input graph must be sparse")
    if graph.format != "csr" and graph.format != "csc":
        raise TypeError("Input must be in CSC or CSR sparse format.")
    # CSC stores the CSR arrays of the transpose; scipy passes them as-is
    G0 = graph if isinstance(graph, _compressed) else graph.tocsr()
    nrows = G0.shape[0]
    if not symmetric_mode:
        G = G0 + G0.transpose()
        ind = G.indices
        ptr = G.indptr
    else:
        ind = G0.indices
        ptr = G0.indptr
    order = np.zeros(nrows, dtype=int)
    degree = _node_degrees(ind, ptr, nrows)
    inds = np.argsort(degree)
    rev_inds = np.argsort(inds)
    temp_degrees = np.zeros(max(int(degree.max()) if nrows > 0 else 0, 1), dtype=int)
    N = 0
    for zz in range(nrows):
        if inds[zz] != -1:
            seed = inds[zz]
            order[N] = seed
            N += 1
            inds[rev_inds[seed]] = -1
            level_start = N - 1
            level_end = N
            while level_start < level_end:
                for ii in range(level_start, level_end):
                    i = order[ii]
                    N_old = N
                    for jj in range(ptr[i], ptr[i + 1]):
                        j = ind[jj]
                        if inds[rev_inds[j]] != -1:
                            inds[rev_inds[j]] = -1
                            order[N] = j
                            N += 1
                    level_len = 0
                    for kk in range(N_old, N):
                        temp_degrees[level_len] = degree[order[kk]]
                        level_len += 1
                    for kk in range(1, level_len):
                        temp = temp_degrees[kk]
                        temp2 = order[N_old + kk]
                        ll = kk
                        while ll > 0 and temp < temp_degrees[ll - 1]:
                            temp_degrees[ll] = temp_degrees[ll - 1]
                            order[N_old + ll] = order[N_old + ll - 1]
                            ll -= 1
                        temp_degrees[ll] = temp
                        order[N_old + ll] = temp2
                level_start = level_end
                level_end = N
        if N == nrows:
            break
    return np.ascontiguousarray(order[::-1])

def structural_rank(graph):
    # maximum bipartite matching size (Hopcroft-Karp equivalent result)
    G = graph.tocsr()
    m, n = G.shape
    match_col = np.full(n, -1, dtype=int)
    rank = 0
    for r in range(m):
        seen = np.zeros(n, dtype=bool)
        # iterative augmenting path search (DFS)
        stack_r = [r]
        stack_p = [G.indptr[r]]
        parent_col = List[int]()
        found = -1
        path_cols = List[int]()
        while len(stack_r) > 0:
            u = stack_r[len(stack_r) - 1]
            p = stack_p[len(stack_p) - 1]
            if p >= G.indptr[u + 1]:
                stack_r.pop()
                stack_p.pop()
                if len(path_cols) > 0:
                    path_cols.pop()
                continue
            stack_p[len(stack_p) - 1] = p + 1
            c = G.indices[p]
            if seen[c]:
                continue
            seen[c] = True
            path_cols.append(c)
            if match_col[c] < 0:
                found = c
                break
            stack_r.append(match_col[c])
            stack_p.append(G.indptr[match_col[c]])
        if found >= 0:
            # augment along stack_r / path_cols
            for q in range(len(stack_r)):
                match_col[path_cols[q]] = stack_r[q]
            rank += 1
    return rank

def laplacian(csgraph, normed: bool = False, return_diag: Literal[bool] = False,
              use_out_degree: bool = False, copy: bool = True,
              form: str = "array", dtype: type = NoneType,
              symmetrized: bool = False):
    if form != "array":
        raise ValueError(f"{form!r} must be 'array' (function/lo forms are not supported)")
    axis = 1 if use_out_degree else 0
    if _issparse(csgraph):
        A = csgraph.tocsr(copy=True)
        if A.shape[0] != A.shape[1]:
            raise ValueError("csgraph must be a square matrix or array")
        T = type(A.data.dtype() if dtype is NoneType else dtype())
        R = A.astype(T)
        if symmetrized:
            R = R + R.conj().transpose().tocsr()
        R.sum_duplicates()
        n = R.shape[0]
        diag = R.diagonal()
        w = R.sum(axis=axis) - diag
        c = R.tocoo()
        rows = List[int]()
        cols = List[int]()
        vals = List[T]()
        if normed:
            isolated = np.array([w[q] == T() for q in range(n)], dtype=bool)
            ww = np.empty(n, dtype=T)
            for q in range(n):
                ww[q] = T(1) if isolated[q] else T(math.sqrt(float(w[q])))
            for k in range(len(c.data)):
                if c.row[k] != c.col[k]:
                    rows.append(c.row[k])
                    cols.append(c.col[k])
                    vals.append(-c.data[k] / ww[c.row[k]] / ww[c.col[k]])
            for q in range(n):
                rows.append(q)
                cols.append(q)
                vals.append(T(0) if isolated[q] else T(1))
            wout = ww
        else:
            for k in range(len(c.data)):
                if c.row[k] != c.col[k]:
                    rows.append(c.row[k])
                    cols.append(c.col[k])
                    vals.append(-c.data[k])
            for q in range(n):
                rows.append(q)
                cols.append(q)
                vals.append(w[q])
            wout = w
        L = coo_array((np.array(vals, dtype=T), (np.array(rows, dtype=int),
                       np.array(cols, dtype=int))), shape=(n, n))
        if return_diag:
            return L, wout
        else:
            return L
    else:
        G = np.asarray(csgraph)
        T = type(G.dtype() if dtype is NoneType else dtype())
        m = G.astype(T).copy()
        n = m.shape[0]
        if symmetrized:
            m = m + np.conj(m.T) if T is complex else m + m.T
        for q in range(n):
            m[q, q] = T()
        w = m.sum(axis=axis)
        if normed:
            isolated = np.array([w[q] == T() for q in range(n)], dtype=bool)
            ww = np.empty(n, dtype=T)
            for q in range(n):
                ww[q] = T(1) if isolated[q] else T(math.sqrt(float(w[q])))
            for i in range(n):
                for j in range(n):
                    m[i, j] = -(m[i, j] / ww[j] / ww[i])
            for q in range(n):
                m[q, q] = T(0) if isolated[q] else T(1)
            if return_diag:
                return m, ww
            else:
                return m
        else:
            for i in range(n):
                for j in range(n):
                    m[i, j] = -m[i, j]
            for q in range(n):
                m[q, q] = w[q]
            if return_diag:
                return m, w
            else:
                return m

class MaximumFlowResult[F]:
    flow_value: int
    flow: F

    def __init__(self, flow_value: int, flow: F):
        self.flow_value = flow_value
        self.flow = flow

    def __repr__(self):
        return f"MaximumFlowResult with value {self.flow_value}"

def _add_reverse_edges(a):
    n = a.shape[0]
    at = a.transpose().tocsr()
    res_data = np.zeros(2 * a.nnz, dtype=int)
    res_indices = np.zeros(2 * a.nnz, dtype=int)
    res_indptr = np.zeros(n + 1, dtype=int)
    res_ptr = 0
    for i in range(n):
        a_ptr = a.indptr[i]
        a_end = a.indptr[i + 1]
        at_ptr = at.indptr[i]
        at_end = at.indptr[i + 1]
        while a_ptr != a_end or at_ptr != at_end:
            move_a = a_ptr != a_end and (at_ptr == at_end or
                                         a.indices[a_ptr] <= at.indices[at_ptr])
            move_at = at_ptr != at_end and (a_ptr == a_end or
                                            at.indices[at_ptr] <= a.indices[a_ptr])
            if move_a:
                res_indices[res_ptr] = a.indices[a_ptr]
                res_data[res_ptr] = a.data[a_ptr]
                a_ptr += 1
            if move_at:
                res_indices[res_ptr] = at.indices[at_ptr]
                at_ptr += 1
            res_ptr += 1
        res_indptr[i + 1] = res_ptr
    return csr_array((res_data[:res_ptr].copy(), res_indices[:res_ptr].copy(), res_indptr),
                     shape=(n, n))

def _make_edge_pointers(a):
    n = a.shape[0]
    b = csr_array((np.arange(len(a.data)), a.indices.copy(), a.indptr.copy()),
                  shape=(n, n))
    return b.transpose().tocsr().data

def _edmonds_karp(edge_ptr, tails, heads, capacities, rev, source: int, sink: int):
    n_verts = len(edge_ptr) - 1
    flow = np.zeros(len(capacities), dtype=int)
    q = np.empty(n_verts, dtype=int)
    pred_edge = np.empty(n_verts, dtype=int)
    while True:
        for k in range(n_verts):
            pred_edge[k] = -1
        q[0] = source
        start = 0
        end = 1
        path_found = False
        while start != end and not path_found:
            cur = q[start]
            start += 1
            for e in range(edge_ptr[cur], edge_ptr[cur + 1]):
                t = heads[e]
                if pred_edge[t] == -1 and t != source and capacities[e] > flow[e]:
                    pred_edge[t] = e
                    if t == sink:
                        path_found = True
                        break
                    q[end] = t
                    end += 1
        if path_found:
            df = 9223372036854775807
            t = sink
            while t != source:
                e = pred_edge[t]
                df = min(df, capacities[e] - flow[e])
                t = tails[e]
            t = sink
            while t != source:
                e = pred_edge[t]
                flow[e] += df
                flow[rev[e]] -= df
                t = tails[e]
        else:
            break
    return flow

def _dinic(edge_ptr, heads, capacities, rev, source: int, sink: int):
    n_verts = len(edge_ptr) - 1
    levels = np.empty(n_verts, dtype=int)
    progress = np.empty(n_verts, dtype=int)
    q = np.empty(n_verts, dtype=int)
    stack = np.empty((n_verts, 2), dtype=int)
    flows = np.zeros(len(capacities), dtype=int)
    while True:
        for i in range(n_verts):
            levels[i] = -1
        # build level graph
        q[0] = source
        start = 0
        end = 1
        levels[source] = 0
        reached = False
        while start != end:
            cur = q[start]
            start += 1
            if cur == sink:
                reached = True
                break
            for e in range(edge_ptr[cur], edge_ptr[cur + 1]):
                dv = heads[e]
                if capacities[e] > 0 and levels[dv] == -1:
                    levels[dv] = levels[cur] + 1
                    q[end] = dv
                    end += 1
        if not reached:
            break
        for i in range(n_verts):
            progress[i] = edge_ptr[i]
        while True:
            # augment one path
            top = 0
            stack[0, 0] = source
            stack[0, 1] = 2147483647
            found = False
            while True:
                current = stack[top, 0]
                fl = stack[top, 1]
                e = progress[current]
                dv = heads[e]
                if capacities[e] > 0 and levels[dv] == levels[current] + 1:
                    cf = min(fl, capacities[e])
                    if dv == sink:
                        while top > -1:
                            e2 = progress[stack[top, 0]]
                            capacities[e2] -= cf
                            capacities[rev[e2]] += cf
                            flows[e2] += cf
                            flows[rev[e2]] -= cf
                            top -= 1
                        found = True
                        break
                    top += 1
                    stack[top, 0] = dv
                    stack[top, 1] = cf
                else:
                    dead = False
                    while progress[current] == edge_ptr[current + 1] - 1:
                        top -= 1
                        if top < 0:
                            dead = True
                            break
                        current = stack[top, 0]
                    if dead:
                        break
                    progress[current] += 1
            if not found:
                break
    return flows

def maximum_flow(csgraph, source: int, sink: int, method: str = "dinic"):
    if not _issparse(csgraph):
        compile_error("graph must be in CSR format")
    if csgraph.format != "csr":
        raise TypeError("graph must be in CSR format")
    if source == sink:
        raise ValueError("source and sink vertices must differ")
    if csgraph.shape[0] != csgraph.shape[1]:
        raise ValueError("graph must be specified as a square matrix.")
    N = csgraph.shape[0]
    if source < 0 or source >= N:
        raise ValueError(f"source value ({source}) must be between 0 and {N - 1}")
    if sink < 0 or sink >= N:
        raise ValueError(f"sink value ({sink}) must be between 0 and {N - 1}")
    g = csgraph.astype(int)
    if not g.has_sorted_indices:
        g = g.sorted_indices()
    m = _add_reverse_edges(g)
    rev = _make_edge_pointers(m)
    if method == "edmonds_karp":
        tails = np.empty(len(m.data), dtype=int)
        for i in range(N):
            for j in range(m.indptr[i], m.indptr[i + 1]):
                tails[j] = i
        flow = _edmonds_karp(m.indptr, tails, m.indices, m.data, rev, source, sink)
    elif method == "dinic":
        flow = _dinic(m.indptr, m.indices, m.data.copy(), rev, source, sink)
    else:
        raise ValueError(f"{method} method is not supported yet.")
        flow = np.zeros(0, dtype=int)
    fm = csr_array((flow, m.indices.copy(), m.indptr.copy()), shape=(N, N))
    fv = 0
    for j in range(m.indptr[source], m.indptr[source + 1]):
        fv += flow[j]
    if _is_spmatrix(csgraph):
        return MaximumFlowResult(fv, fm._as_matrix())
    else:
        return MaximumFlowResult(fv, fm)
