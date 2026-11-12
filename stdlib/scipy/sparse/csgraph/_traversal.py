# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Graph traversal (scipy/sparse/csgraph/_traversal.pyx, _tools.pyx):
# connected_components (union by DFS for weak, Tarjan-Zwick iterative SCC for
# strong), breadth/depth-first orders and trees, reconstruct_path. Loops and
# tie-breaking are those of the Cython sources.

import numpy as np
from .._base import _compressed, _coo, _issparse
from .._construct import csr_array, csr_matrix
from ._validation import _csr_graph, _transpose_csr, NULL_IDX, _issparse

def _cc_directed(indices, indptr, N: int, labels) -> int:
    VOID = -1
    if N == 0:
        return 0
    NONLEAD = 1 << 40           # flag bit (scipy uses the int32 sign bit)
    index = N
    n_comp = 0
    root_hl = 0
    done = False
    succ_pos = [VOID for _ in range(N)]
    dfs_stack = List[int]()
    comp_stack = List[int]()
    for q in range(N):
        labels[q] = 0
    for v0 in range(N):
        v = v0
        if succ_pos[v] != VOID:
            continue
        root_hl = index
        dfs_stack.append(v)
        labels[v] = index
        index -= 1
        succ_pos[v] = indptr[v]
        while len(dfs_stack) > 0:
            v = dfs_stack[len(dfs_stack) - 1] & (NONLEAD - 1)
            if succ_pos[v] < indptr[v + 1]:
                w = indices[succ_pos[v]]
                succ_pos[v] += 1
                if succ_pos[w] == VOID:
                    dfs_stack.append(w)
                    labels[w] = index
                    index -= 1
                    succ_pos[w] = indptr[w]
                else:
                    if labels[v] < labels[w]:
                        dfs_stack[len(dfs_stack) - 1] = v | NONLEAD
                        labels[v] = labels[w]
                        if labels[v] == root_hl and index == 0:
                            labels[v] = n_comp
                            for j in range(len(comp_stack)):
                                labels[comp_stack[j]] = n_comp
                            for j in range(len(dfs_stack) - 1):
                                labels[dfs_stack[j] & (NONLEAD - 1)] = n_comp
                            n_comp += 1
                            dfs_stack.clear()
                            comp_stack.clear()
                            done = True
                            break
            else:
                if dfs_stack[len(dfs_stack) - 1] & NONLEAD == 0:
                    dfs_stack.pop()
                    while len(comp_stack) > 0:
                        top = comp_stack[len(comp_stack) - 1]
                        if labels[v] < labels[top]:
                            break
                        labels[top] = n_comp
                        comp_stack.pop()
                        index += 1
                    labels[v] = n_comp
                    index += 1
                    n_comp += 1
                else:
                    dfs_stack.pop()
                    comp_stack.append(v)
                    if len(dfs_stack) > 0:
                        parent = dfs_stack[len(dfs_stack) - 1] & (NONLEAD - 1)
                        if labels[parent] < labels[v]:
                            dfs_stack[len(dfs_stack) - 1] = parent | NONLEAD
                            labels[parent] = labels[v]
        if done:
            break
    return n_comp

def _cc_undirected(ind1, ptr1, ind2, ptr2, N: int, labels) -> int:
    VOID = -1
    END = -2
    for q in range(N):
        labels[q] = VOID
    label = 0
    SS = labels
    SS_head = END
    for v0 in range(N):
        if labels[v0] == VOID:
            SS_head = v0
            SS[v0] = END
            while SS_head != END:
                v = SS_head
                SS_head = SS[v]
                labels[v] = label
                for j in range(ptr1[v], ptr1[v + 1]):
                    w = ind1[j]
                    if SS[w] == VOID:
                        SS[w] = SS_head
                        SS_head = w
                for j in range(ptr2[v], ptr2[v + 1]):
                    w = ind2[j]
                    if SS[w] == VOID:
                        SS[w] = SS_head
                        SS_head = w
            label += 1
    return label

def connected_components(csgraph, directed: bool = True, connection: str = "weak",
                         return_labels: Literal[bool] = True):
    conn = connection.lower()
    if conn != "weak" and conn != "strong":
        raise ValueError("connection must be 'weak' or 'strong'")
    d = directed and conn == "strong"
    G = _csr_graph(csgraph)
    N = G.shape[0]
    labels = np.empty(N, dtype=int)
    if d:
        nc = _cc_directed(G.indices, G.indptr, N, labels)
    else:
        GT = _transpose_csr(G)
        nc = _cc_undirected(G.indices, G.indptr, GT.indices, GT.indptr, N, labels)
    if return_labels:
        return nc, labels
    else:
        return nc

def _bfs(head: int, ind1, ptr1, ind2, ptr2, undirected: bool, node_list, pred) -> int:
    node_list[0] = head
    i_nl = 0
    i_nl_end = 1
    while i_nl < i_nl_end:
        pnode = node_list[i_nl]
        for i in range(ptr1[pnode], ptr1[pnode + 1]):
            cnode = ind1[i]
            if cnode == head:
                continue
            elif pred[cnode] == NULL_IDX:
                node_list[i_nl_end] = cnode
                pred[cnode] = pnode
                i_nl_end += 1
        if undirected:
            for i in range(ptr2[pnode], ptr2[pnode + 1]):
                cnode = ind2[i]
                if cnode == head:
                    continue
                elif pred[cnode] == NULL_IDX:
                    node_list[i_nl_end] = cnode
                    pred[cnode] = pnode
                    i_nl_end += 1
        i_nl += 1
    return i_nl

def breadth_first_order(csgraph, i_start: int, directed: bool = True,
                        return_predecessors: Literal[bool] = True):
    G = _csr_graph(csgraph)
    N = G.shape[0]
    node_list = np.full(N, NULL_IDX, dtype=int)
    pred = np.full(N, NULL_IDX, dtype=int)
    if directed:
        length = _bfs(i_start, G.indices, G.indptr, G.indices, G.indptr, False,
                      node_list, pred)
    else:
        GT = _transpose_csr(G)
        length = _bfs(i_start, G.indices, G.indptr, GT.indices, GT.indptr, True,
                      node_list, pred)
    if return_predecessors:
        return node_list[:length].copy(), pred
    else:
        return node_list[:length].copy()

def _dfs(head: int, ind1, ptr1, ind2, ptr2, undirected: bool, N: int,
         node_list, pred) -> int:
    root_list = np.full(N, NULL_IDX, dtype=int)
    flag = np.zeros(N, dtype=int)
    node_list[0] = head
    root_list[0] = head
    i_root = 0
    i_nl_end = 1
    flag[head] = 1
    while i_root >= 0:
        pnode = root_list[i_root]
        no_children = True
        for i in range(ptr1[pnode], ptr1[pnode + 1]):
            cnode = ind1[i]
            if flag[cnode]:
                continue
            else:
                i_root += 1
                root_list[i_root] = cnode
                node_list[i_nl_end] = cnode
                pred[cnode] = pnode
                flag[cnode] = 1
                i_nl_end += 1
                no_children = False
                break
        if undirected and no_children:
            for i in range(ptr2[pnode], ptr2[pnode + 1]):
                cnode = ind2[i]
                if flag[cnode]:
                    continue
                else:
                    i_root += 1
                    root_list[i_root] = cnode
                    node_list[i_nl_end] = cnode
                    pred[cnode] = pnode
                    flag[cnode] = 1
                    i_nl_end += 1
                    no_children = False
                    break
        if i_nl_end == N:
            break
        if no_children:
            i_root -= 1
    return i_nl_end

def depth_first_order(csgraph, i_start: int, directed: bool = True,
                      return_predecessors: Literal[bool] = True):
    G = _csr_graph(csgraph)
    N = G.shape[0]
    node_list = np.full(N, NULL_IDX, dtype=int)
    pred = np.full(N, NULL_IDX, dtype=int)
    if directed:
        length = _dfs(i_start, G.indices, G.indptr, G.indices, G.indptr, False, N,
                      node_list, pred)
    else:
        GT = _transpose_csr(G)
        length = _dfs(i_start, G.indices, G.indptr, GT.indices, GT.indptr, True, N,
                      node_list, pred)
    if return_predecessors:
        return node_list[:length].copy(), pred
    else:
        return node_list[:length].copy()

def _is_spmatrix(g) -> Literal[bool]:
    if _issparse(g):
        return g._mstatic()
    else:
        return False

def reconstruct_path(csgraph, predecessors, directed: bool = True):
    G = _csr_graph(csgraph)
    P = np.asarray(predecessors)
    N = G.shape[0]
    # indices = argsort(predecessors)[nnull:] (stable for the integer keys
    # numpy sorts with radix/timsort-equivalent ordering)
    order = sorted(range(N), key=lambda q: P[q])
    nnull = 0
    for q in range(N):
        if P[q] < 0:
            nnull += 1
    idx = np.array(order[nnull:], dtype=int)
    pind = np.array([P[q] for q in order[nnull:]], dtype=int)
    indptr = np.zeros(N + 1, dtype=int)
    for r in range(N + 1):
        c = 0
        for q in range(len(pind)):
            if pind[q] < r:
                c += 1
        indptr[r] = c
    data = np.empty(len(idx))
    for q in range(len(idx)):
        data[q] = G._get(pind[q], idx[q])
    if not directed:
        for q in range(len(idx)):
            d2 = G._get(idx[q], pind[q])
            a = data[q] if data[q] != 0 else np.inf
            b = d2 if d2 != 0 else np.inf
            data[q] = min(a, b)
    if _is_spmatrix(csgraph):
        return csr_matrix((data, idx, indptr), shape=(N, N))
    else:
        return csr_array((data, idx, indptr), shape=(N, N))

def breadth_first_tree(csgraph, i_start: int, directed: bool = True):
    _nl, pred = breadth_first_order(csgraph, i_start, directed, True)
    return reconstruct_path(csgraph, pred, directed)

def depth_first_tree(csgraph, i_start: int, directed: bool = True):
    _nl, pred = depth_first_order(csgraph, i_start, directed, True)
    return reconstruct_path(csgraph, pred, directed)
