# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Function forms: k_means, affinity_propagation, mean_shift, spectral_clustering.

import numpy as np
from ._kmeans import KMeans
from ._mean_shift import MeanShift, AffinityPropagation
from ._spectral import SpectralClustering

def k_means(X, n_clusters: int, *, sample_weight = None, init = "k-means++", n_init = "auto",
            max_iter: int = 300, verbose: bool = False, tol: float = 1e-4, random_state = None,
            copy_x: bool = True, algorithm: str = "lloyd", return_n_iter: Literal[bool] = False):
    est = KMeans(n_clusters, init=init, n_init=n_init, max_iter=max_iter, verbose=0, tol=tol,
                 random_state=random_state, copy_x=copy_x, algorithm=algorithm)
    est.fit(X, sample_weight=sample_weight)
    if return_n_iter:
        return est.cluster_centers_, est.labels_, est.inertia_, est.n_iter_
    return est.cluster_centers_, est.labels_, est.inertia_

def affinity_propagation(S, *, preference = None, convergence_iter: int = 15, max_iter: int = 200,
                         damping: float = 0.5, copy: bool = True, verbose: bool = False,
                         return_n_iter: Literal[bool] = False, random_state = None):
    est = AffinityPropagation(damping=damping, max_iter=max_iter, convergence_iter=convergence_iter,
                              copy=copy, preference=preference, affinity="precomputed",
                              random_state=random_state)
    est.fit(S)
    if return_n_iter:
        return est.cluster_centers_indices_, est.labels_, est.n_iter_
    return est.cluster_centers_indices_, est.labels_

def mean_shift(X, *, bandwidth = None, seeds = None, bin_seeding: bool = False, min_bin_freq: int = 1,
               cluster_all: bool = True, max_iter: int = 300, n_jobs = None):
    est = MeanShift(bandwidth=bandwidth, seeds=seeds, bin_seeding=bin_seeding,
                    min_bin_freq=min_bin_freq, cluster_all=cluster_all, max_iter=max_iter)
    est.fit(X)
    return est.cluster_centers_, est.labels_

def spectral_clustering(affinity, *, n_clusters: int = 8, n_components = None, eigen_solver = None,
                        random_state = None, n_init: int = 10, eigen_tol = "auto",
                        assign_labels: str = "kmeans", verbose: bool = False):
    est = SpectralClustering(n_clusters, eigen_solver=eigen_solver, n_components=n_components,
                             random_state=random_state, n_init=n_init, affinity="precomputed",
                             assign_labels=assign_labels)
    est.fit(affinity)
    return est.labels_
