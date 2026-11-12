# Ported from numpy 2.5.4 (BSD-3-Clause)
#
# Masked reduction kernels: every reduction is computed lane by lane over
# the unmasked entries of a (lanes, n) view of the data.

import numpy as np
from numpy.util import tuple_delete, normalize_axis_index

def _sum_type(X: type):
    if X is bool:
        return int()
    else:
        return X()

def lanes(d, m, axis: int):
    N: Literal[int] = static.len(d.shape)
    ax = normalize_axis_index(axis, N)
    n = d.shape[ax]
    L = d.size // n if n > 0 else 0
    d2 = np.ascontiguousarray(np.moveaxis(d, ax, -1)).reshape((L, n))
    m2 = np.ascontiguousarray(np.moveaxis(m, ax, -1)).reshape((L, n))
    return d2, m2, tuple_delete(d.shape, ax)

def flat2(d, m):
    n = d.size
    return d.ravel().reshape((1, n)), m.ravel().reshape((1, n))

def lane(d2, m2, r: int, kind: Literal[str], p: float, R: type) -> Tuple[R, bool]:
    n = d2.shape[1]
    if kind == "sum":
        s = R()
        c = 0
        for j in range(n):
            if not m2[r, j]:
                s += R(d2[r, j])
                c += 1
        return (s, c > 0)
    elif kind == "prod":
        s = R(1)
        c = 0
        for j in range(n):
            if not m2[r, j]:
                s *= R(d2[r, j])
                c += 1
        return (s, c > 0)
    elif kind == "mean":
        s = 0.0
        c = 0
        for j in range(n):
            if not m2[r, j]:
                s += float(d2[r, j])
                c += 1
        return (R(s / c) if c > 0 else R(), c > 0)
    elif kind == "var" or kind == "std":
        s = 0.0
        c = 0
        for j in range(n):
            if not m2[r, j]:
                s += float(d2[r, j])
                c += 1
        if c == 0:
            return (R(), False)
        mu = s / c
        q = 0.0
        for j in range(n):
            if not m2[r, j]:
                t = float(d2[r, j]) - mu
                q += t * t
        den = float(c) - p
        if den <= 0.0:
            return (R(), False)
        v = q / den
        if kind == "std":
            v = np.sqrt(v)
        return (R(v), True)
    elif kind == "min" or kind == "max" or kind == "argmin" or kind == "argmax":
        best = 0
        found = False
        for j in range(n):
            if not m2[r, j]:
                if not found:
                    best = j
                    found = True
                elif kind == "min" or kind == "argmin":
                    if d2[r, j] < d2[r, best]:
                        best = j
                else:
                    if d2[r, j] > d2[r, best]:
                        best = j
        if kind == "argmin" or kind == "argmax":
            return (R(best), True)
        else:
            if found:
                return (R(d2[r, best]), True)
            return (R(), False)
    elif kind == "any" or kind == "all":
        c = 0
        acc = kind == "all"
        for j in range(n):
            if not m2[r, j]:
                c += 1
                if kind == "any":
                    acc = acc or bool(d2[r, j])
                else:
                    acc = acc and bool(d2[r, j])
        return (R(acc), c > 0)
    elif kind == "count":
        c = 0
        for j in range(n):
            if not m2[r, j]:
                c += 1
        return (R(c), True)
    elif kind == "median":
        v = List[float]()
        for j in range(n):
            if not m2[r, j]:
                v.append(float(d2[r, j]))
        c = len(v)
        if c == 0:
            return (R(), False)
        v.sort()
        if c % 2 == 1:
            return (R(v[c // 2]), True)
        return (R(0.5 * (v[c // 2 - 1] + v[c // 2])), True)
    else:
        compile_error("unknown masked reduction")
