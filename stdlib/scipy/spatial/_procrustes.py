# Ported from scipy 1.18.1 (BSD-3-Clause)

import numpy as np
from scipy.linalg import orthogonal_procrustes

def procrustes(data1, data2):
    mtx1 = np.array(data1, dtype=float)
    mtx2 = np.array(data2, dtype=float)
    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")
    mtx1 = mtx1 - np.mean(mtx1, 0)
    mtx2 = mtx2 - np.mean(mtx2, 0)
    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)
    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")
    mtx1 = mtx1 / norm1
    mtx2 = mtx2 / norm2
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s
    disparity = float(np.sum(np.square(mtx1 - mtx2)))
    return mtx1, mtx2, disparity
