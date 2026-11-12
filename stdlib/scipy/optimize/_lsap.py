# Ported from scipy 1.18.1 (BSD-3-Clause)

# linear_sum_assignment: shortest augmenting path (Crouse 2016, a
# Jonker-Volgenant variant), translated from rectangular_lsap.cpp.

import math
import numpy as np

def _augmenting_path(nc: int, cost: np.ndarray[float, 1], u: np.ndarray[float, 1],
                     v: np.ndarray[float, 1], path: np.ndarray[int, 1],
                     row4col: np.ndarray[int, 1],
                     shortestPathCosts: np.ndarray[float, 1], i: int,
                     SR: np.ndarray[bool, 1], SC: np.ndarray[bool, 1],
                     remaining: np.ndarray[int, 1]):
    minVal = 0.0
    num_remaining = nc
    for it in range(nc):
        remaining[it] = nc - it - 1
    SR.fill(False)
    SC.fill(False)
    shortestPathCosts.fill(math.inf)
    sink = -1
    while sink == -1:
        index = -1
        lowest = math.inf
        SR[i] = True
        for it in range(num_remaining):
            j = remaining[it]
            r = minVal + cost[i * nc + j] - u[i] - v[j]
            if r < shortestPathCosts[j]:
                path[j] = i
                shortestPathCosts[j] = r
            if (shortestPathCosts[j] < lowest or
                    (shortestPathCosts[j] == lowest and row4col[j] == -1)):
                lowest = shortestPathCosts[j]
                index = it
        minVal = lowest
        if minVal == math.inf:
            return -1, minVal
        j = remaining[index]
        if row4col[j] == -1:
            sink = j
        else:
            i = row4col[j]
        SC[j] = True
        num_remaining -= 1
        remaining[index] = remaining[num_remaining]
    return sink, minVal

def linear_sum_assignment(cost_matrix, maximize: bool = False):
    c = np.asarray(cost_matrix)
    if c.ndim != 2:
        compile_error("expected a matrix (2-D array)")
    cm = c.astype(float)
    nr0 = cm.shape[0]
    nc0 = cm.shape[1]
    if nr0 == 0 or nc0 == 0:
        return np.empty(0, int), np.empty(0, int)
    transpose = nc0 < nr0
    nr = nr0
    nc = nc0
    cost = np.empty(nr * nc, float)
    if transpose:
        for i in range(nr0):
            for j in range(nc0):
                cost[j * nr0 + i] = cm[i, j]
        nr, nc = nc0, nr0
    else:
        for i in range(nr0):
            for j in range(nc0):
                cost[i * nc0 + j] = cm[i, j]
    if maximize:
        for i in range(nr * nc):
            cost[i] = -cost[i]
    for i in range(nr * nc):
        if cost[i] != cost[i] or cost[i] == -math.inf:
            raise ValueError("matrix contains invalid numeric entries")
    u = np.zeros(nr)
    v = np.zeros(nc)
    shortestPathCosts = np.empty(nc)
    path = np.full(nc, -1, int)
    col4row = np.full(nr, -1, int)
    row4col = np.full(nc, -1, int)
    SR = np.zeros(nr, bool)
    SC = np.zeros(nc, bool)
    remaining = np.zeros(nc, int)
    for curRow in range(nr):
        sink, minVal = _augmenting_path(nc, cost, u, v, path, row4col,
                                        shortestPathCosts, curRow, SR, SC,
                                        remaining)
        if sink < 0:
            raise ValueError("cost matrix is infeasible")
        u[curRow] += minVal
        for i in range(nr):
            if SR[i] and i != curRow:
                u[i] += minVal - shortestPathCosts[col4row[i]]
        for j in range(nc):
            if SC[j]:
                v[j] -= minVal - shortestPathCosts[j]
        j = sink
        while True:
            i = path[j]
            row4col[j] = i
            tmp = col4row[i]
            col4row[i] = j
            j = tmp
            if i == curRow:
                break
    a = np.empty(nr, int)
    b = np.empty(nr, int)
    if transpose:
        order = np.argsort(col4row, kind='stable')
        for k in range(nr):
            a[k] = col4row[order[k]]
            b[k] = order[k]
    else:
        for i in range(nr):
            a[i] = i
            b[i] = col4row[i]
    return a, b
