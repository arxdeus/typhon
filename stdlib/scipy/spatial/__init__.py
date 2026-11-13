# Ported from scipy 1.18.1 (BSD-3-Clause)

# scipy.spatial subset: KDTree/cKDTree, distance_matrix, minkowski_distance
# and the ``distance`` submodule. The qhull-based ConvexHull, Delaunay and
# Voronoi are not provided.

from ._kdtree import KDTree, cKDTree, distance_matrix, minkowski_distance, \
                     minkowski_distance_p
import scipy.spatial.distance as distance
from ._procrustes import procrustes
from ._geometric_slerp import geometric_slerp
