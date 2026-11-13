# Ported from scipy 1.18.1 (BSD-3-Clause)

# scipy.spatial: KDTree/cKDTree, distance_matrix, minkowski_distance, the
# ``distance`` submodule and the Qhull-based ConvexHull, Delaunay, Voronoi,
# HalfspaceIntersection (beneath-beyond reimplementation, see _qhull.py).

from ._kdtree import KDTree, cKDTree, Rectangle, distance_matrix, minkowski_distance, \
                     minkowski_distance_p
import scipy.spatial.distance as distance
from ._procrustes import procrustes
from ._geometric_slerp import geometric_slerp
from ._qhull import Delaunay, ConvexHull, Voronoi, HalfspaceIntersection, tsearch
from ._hull import QhullError
import scipy.spatial.transform as transform
from ._spherical_voronoi import SphericalVoronoi
