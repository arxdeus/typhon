from ._unsupervised import NearestNeighbors
from ._supervised import (KNeighborsClassifier, KNeighborsRegressor,
                          RadiusNeighborsClassifier, RadiusNeighborsRegressor)
from ._nearest_centroid import NearestCentroid
from ._kde import KernelDensity, kernel_norm
from ._binary_tree import BallTree, KDTree
from ._lof import LocalOutlierFactor
from ._nca import NeighborhoodComponentsAnalysis
from ._graph import (kneighbors_graph, radius_neighbors_graph, sort_graph_by_row_values,
                     KNeighborsTransformer, RadiusNeighborsTransformer)

VALID_METRICS = {
    "ball_tree": ["braycurtis", "canberra", "chebyshev", "cityblock", "dice", "euclidean", "hamming",
                  "haversine", "infinity", "jaccard", "l1", "l2", "mahalanobis", "manhattan",
                  "minkowski", "p", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener",
                  "sokalsneath"],
    "kd_tree": ["chebyshev", "cityblock", "euclidean", "infinity", "l1", "l2", "manhattan", "minkowski", "p"],
    "brute": ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan", "braycurtis", "canberra",
              "chebyshev", "correlation", "dice", "hamming", "jaccard", "mahalanobis", "minkowski",
              "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath",
              "sqeuclidean", "yule", "precomputed"],
}
VALID_METRICS_SPARSE = {
    "ball_tree": [],
    "kd_tree": [],
    "brute": ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan", "braycurtis", "canberra",
              "chebyshev", "correlation", "hamming", "jaccard", "minkowski", "rogerstanimoto",
              "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule",
              "precomputed"],
}
