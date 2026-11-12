# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Native Typhon implementation of sklearn.neighbors. Every algorithm
# ('auto', 'brute', 'kd_tree', 'ball_tree') uses one exact brute-force
# backend; results match scikit-learn (ties broken by lower sample index).

from ._unsupervised import NearestNeighbors
from ._supervised import (KNeighborsClassifier, KNeighborsRegressor,
                          RadiusNeighborsClassifier, RadiusNeighborsRegressor)
from ._nearest_centroid import NearestCentroid
from ._kde import KernelDensity, kernel_norm
