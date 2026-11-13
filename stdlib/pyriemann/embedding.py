# Copyright (C) 2026 the Typhon authors
# Ported from pyriemann 0.12 (BSD-3-Clause)

# Embedding SPD matrices via manifold learning: SpectralEmbedding and
# LocallyLinearEmbedding (TSNE is not provided).

import numpy as np
from scipy.linalg import solve
from sklearn.manifold import spectral_embedding
from .geometry.distance import pairwise_distance
from .geometry.kernel import kernel as kernel_fct

def _check_dimensions(X, Y = None, n_components: int = -1, n_neighbors: int = -1):
    n_matrices = X.shape[0]
    n_ch = X.shape[1]
    if Y is not None:
        if Y.shape[1] != n_ch or Y.shape[2] != n_ch:
            raise ValueError(
                "Dimension of matrices in data to be transformed must match "
                "dimension of data used for fitting.")
    nc = n_components
    nn = n_neighbors
    if n_components < 0:
        nc = n_matrices - 1
    elif n_components >= n_matrices:
        raise ValueError(
            f"n_components (is {n_components}) must be smaller than n_matrices (is {n_matrices}).")
    if n_neighbors < 0:
        nn = n_matrices - 1
    elif n_matrices <= n_neighbors:
        nn = n_matrices - 1
    return nc, nn

class SpectralEmbedding:
    n_components: int
    metric: str
    eps: Optional[float]
    embedding_: np.ndarray[float, 2]

    def __init__(self, n_components: int = 2, metric: str = "riemann",
                 eps: Optional[float] = None):
        self.n_components = n_components
        self.metric = metric
        self.eps = eps
        self.embedding_ = np.empty((0, 0))

    def _get_affinity_matrix(self, X):
        distmatrix = pairwise_distance(X, metric=self.metric)
        eps = 0.0
        if self.eps is None:
            eps = np.median(distmatrix) ** 2 / 2
        else:
            eps = self.eps
        kern = np.exp(-(distmatrix ** 2) / (4 * eps))
        q = kern @ np.ones(kern.shape[0])
        return kern / np.outer(q, q)

    def fit(self, X, y = None):
        X = np.asarray(X, dtype=float)
        _check_dimensions(X, None, self.n_components)
        affinity_matrix = self._get_affinity_matrix(X)
        embd = spectral_embedding(affinity_matrix, n_components=self.n_components,
                                  norm_laplacian=True)
        mn = np.min(embd, axis=0)
        self.embedding_ = 2 * (embd - mn) / np.ptp(embd, axis=0) - 1
        return self

    def fit_transform(self, X, y = None):
        self.fit(X)
        return self.embedding_

def barycenter_weights(X, Y, indices, metric: str = "riemann", kernel = None,
                       reg: float = 1e-3):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n_matrices = indices.shape[0]
    n_neighbors = indices.shape[1]
    if X.shape[0] != n_matrices:
        raise ValueError(
            f"Number of index-sets in indices (is {n_matrices}) must match number of matrices in X (is {X.shape[0]}).")
    B = np.empty((n_matrices, n_neighbors))
    v = np.ones(n_neighbors)
    for i in range(n_matrices):
        X_neighbors = Y[indices[i]]
        G = kernel_fct(X_neighbors, Cref=X[i], metric=metric)
        tr = np.trace(G)
        R = reg * tr if tr > 0 else reg
        for d in range(n_neighbors):
            G[d, d] += R
        w = solve(G, v, assume_a="pos")
        B[i] = w / np.sum(w)
    return B

def _neighbors(dists, n_neighbors: int):
    n = dists.shape[0]
    out = np.empty((n, n_neighbors), dtype=int)
    for i in range(n):
        order = np.argsort(dists[i])
        for k in range(n_neighbors):
            out[i, k] = order[k + 1]
    return out

def locally_linear_embedding(X, n_components: int = 2, n_neighbors: int = 5,
                             metric: str = "riemann", kernel = None,
                             reg: float = 1e-3):
    X = np.asarray(X, dtype=float)
    n_matrices = X.shape[0]
    pd = pairwise_distance(X, metric=metric)
    neighbors = _neighbors(pd, n_neighbors)
    B = barycenter_weights(X, X, neighbors, metric=metric, kernel=kernel, reg=reg)
    W = np.zeros((n_matrices, n_matrices))
    for i in range(n_matrices):
        for k in range(n_neighbors):
            W[i, neighbors[i, k]] += B[i, k]
    M = W.T @ W - W.T - W
    for i in range(n_matrices):
        M[i, i] += 1.0
    ev, evec = np.linalg.eigh(M)
    vals = ev[1:n_components + 1]
    vecs = evec[:, 1:n_components + 1]
    index = np.argsort(np.abs(vals))
    return vecs[:, index], float(np.sum(vals))

class LocallyLinearEmbedding:
    n_components: int
    n_neighbors: int
    metric: str
    reg: float
    data_: np.ndarray[float, 3]
    embedding_: np.ndarray[float, 2]
    error_: float

    def __init__(self, n_components: int = 2, n_neighbors: int = 5,
                 metric: str = "riemann", kernel = None, reg: float = 1e-3):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.reg = reg
        self.data_ = np.empty((0, 0, 0))
        self.embedding_ = np.empty((0, 0))
        self.error_ = 0.0

    def fit(self, X, y = None):
        X = np.asarray(X, dtype=float)
        self.data_ = X
        nc, nn = _check_dimensions(X, None, self.n_components, self.n_neighbors)
        self.n_components = nc
        self.n_neighbors = nn
        emb, err = locally_linear_embedding(X, nc, nn, self.metric, None, self.reg)
        self.embedding_ = emb
        self.error_ = err
        return self

    def transform(self, X, y = None):
        X = np.asarray(X, dtype=float)
        _check_dimensions(self.data_, X)
        pd = pairwise_distance(X, self.data_, metric=self.metric)
        ind = _neighbors(pd, self.n_neighbors)
        weights = barycenter_weights(X, self.data_, ind, metric=self.metric,
                                     kernel=None, reg=self.reg)
        X_new = np.empty((X.shape[0], self.n_components))
        for i in range(X.shape[0]):
            X_new[i] = self.embedding_[ind[i]].T @ weights[i]
        return X_new

    def fit_transform(self, X, y = None):
        self.fit(X)
        return self.embedding_
