# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Input validation / conversion shared by the csgraph routines
# (scipy/sparse/csgraph/_validation.py and _tools.pyx).

import math
import numpy as np
from .._base import _compressed, _coo, _issparse
from .._construct import csr_array

NULL_IDX: Literal[int] = -9999

class NegativeCycleError(Exception):
    def __init__(self, msg: str = ""):
        super().__init__(msg)

def _issparse(g) -> Literal[bool]:
    return _issparse(g)

def _is_null(v: float, null_value: float, infinity_null: bool, nan_null: bool) -> bool:
    if math.isnan(v):
        return nan_null or math.isnan(null_value)
    if math.isinf(v) and infinity_null:
        return True
    return v == null_value

def csgraph_from_dense(graph, null_value: float = 0.0, nan_null: bool = True,
                       infinity_null: bool = True):
    G = np.asarray(graph).astype(float)
    if G.ndim != 2:
        raise ValueError("graph should have two dimensions")
    m, n = G.shape
    if m != n:
        raise ValueError("graph should be a square array")
    Bp = np.zeros(m + 1, dtype=int)
    Bj = List[int]()
    Bx = List[float]()
    for i in range(m):
        for j in range(n):
            v = G[i, j]
            if not _is_null(v, null_value, infinity_null, nan_null):
                Bj.append(j)
                Bx.append(v)
        Bp[i + 1] = len(Bj)
    return csr_array((np.array(Bx, dtype=float), np.array(Bj, dtype=int), Bp),
                     shape=(m, n))

def csgraph_to_dense(csgraph, null_value: float = 0.0):
    A = csgraph.tocsr().astype(float)
    N = A.shape[0]
    G = np.full((N, A.shape[1]), math.inf)
    null = np.ones((N, A.shape[1]), dtype=bool)
    for row in range(N):
        for i in range(A.indptr[row], A.indptr[row + 1]):
            col = A.indices[i]
            null[row, col] = False
            if A.data[i] < G[row, col]:
                G[row, col] = A.data[i]
    for row in range(N):
        for col in range(A.shape[1]):
            if null[row, col]:
                G[row, col] = null_value
    return G

def csgraph_masked_from_dense(graph, null_value: float = 0.0, nan_null: bool = True,
                              infinity_null: bool = True, copy: bool = True):
    G = np.asarray(graph).astype(float)
    mask = np.zeros(G.shape, dtype=bool)
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            mask[i, j] = _is_null(G[i, j], null_value, infinity_null, nan_null)
    return G, mask

def _csr_graph(csgraph):
    # validate_graph(..., dense_output=False): float64 CSR
    if _issparse(csgraph):
        A = csgraph.tocsr(copy=True).astype(float)
        if A.shape[0] != A.shape[1]:
            raise ValueError("compressed-sparse graph must be shape (N, N)")
        return A._as_array()
    else:
        return csgraph_from_dense(csgraph, null_value=0.0)

def _dense_graph(csgraph):
    # validate_graph(..., csr_output=False): float64 dense, null -> inf
    if _issparse(csgraph):
        if csgraph.shape[0] != csgraph.shape[1]:
            raise ValueError("compressed-sparse graph must be shape (N, N)")
        return csgraph_to_dense(csgraph, null_value=math.inf)
    else:
        G = np.asarray(csgraph).astype(float).copy()
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise ValueError("compressed-sparse graph must be shape (N, N)")
        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                if _is_null(G[i, j], 0.0, True, True):
                    G[i, j] = math.inf
        return G

def _transpose_csr(A):
    return A.transpose().tocsr()

def _index_list(indices, N: int) -> List[int]:
    if indices is None:
        return list(range(N))
    elif isinstance(indices, int):
        return [indices + N if indices < 0 else indices]
    else:
        out = List[int]()
        for v in indices:
            iv = int(v)
            out.append(iv + N if iv < 0 else iv)
        for v in out:
            if v < 0 or v >= N:
                raise ValueError("indices out of range 0...N")
        return out

def _squeeze(indices, X):
    # scipy returns 1-D results when indices is a scalar
    if isinstance(indices, int):
        return X[0].copy()
    else:
        return X
