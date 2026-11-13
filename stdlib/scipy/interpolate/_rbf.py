# Ported from scipy 1.18.1 (BSD-3-Clause): legacy Rbf
# (scipy/interpolate/_rbf.py).  Named basis functions only (no callables),
# mode '1-D'; keyword options: function, epsilon, smooth, norm.

import numpy as np
import internal.static as static
from math import sqrt, exp, log, pow
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.linalg import solve

def _fcode(name: str) -> int:
    f = name.lower()
    if f == 'multiquadric':
        return 0
    if f == 'inverse' or f == 'inverse multiquadric' or f == 'inverse_multiquadric':
        return 1
    if f == 'gaussian':
        return 2
    if f == 'linear':
        return 3
    if f == 'cubic':
        return 4
    if f == 'quintic':
        return 5
    if f == 'thin_plate' or f == 'thin-plate':
        return 6
    raise ValueError("function must be a callable or one of multiquadric, inverse_multiquadric, gaussian, linear, cubic, quintic, thin_plate")

def _h(code: int, eps: float, r: float) -> float:
    if code == 0:
        return sqrt((1.0 / eps * r) ** 2 + 1)
    if code == 1:
        return 1.0 / sqrt((1.0 / eps * r) ** 2 + 1)
    if code == 2:
        return exp(-(1.0 / eps * r) ** 2)
    if code == 3:
        return r
    if code == 4:
        return r ** 3
    if code == 5:
        return r ** 5
    return 0.0 if r == 0.0 else r ** 2 * log(r)

def _amat(xi, norm: str, code: int, eps: float, smooth: float):
    r = squareform(pdist(xi.T, norm))
    n = xi.shape[1]
    out = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            out[i, j] = _h(code, eps, r[i, j]) - (smooth if i == j else 0.0)
    return out

class Rbf:
    def __init__(self, *args, function: str = 'multiquadric', epsilon: Optional[float] = None,
                 smooth: float = 0.0, norm: str = 'euclidean'):
        arrs = List[np.ndarray[float, 1]]()
        for a in args:
            arrs.append(np.ascontiguousarray(np.asarray(a, dtype=float).ravel()))
        nd = len(arrs) - 1
        n = arrs[0].size
        xi = np.empty((nd, n))
        for k in range(nd):
            if arrs[k].size != n:
                raise ValueError("All arrays must be equal length.")
            for i in range(n):
                xi[k, i] = arrs[k][i]
        di = arrs[nd]
        if di.size != n:
            raise ValueError("All arrays must be equal length.")
        self.xi = xi
        self.N = n
        self.di = di
        self.norm = norm
        if epsilon is None:
            prod = 1.0
            cnt = 0
            for k in range(nd):
                e = float(np.max(xi[k])) - float(np.min(xi[k]))
                if e != 0:
                    prod *= e
                    cnt += 1
            self.epsilon = pow(prod / n, 1.0 / cnt)
        else:
            self.epsilon = epsilon
        self.smooth = smooth
        self.function = function.lower()
        self._code = _fcode(function)
        self.nodes = solve(_amat(xi, norm, self._code, self.epsilon, smooth), di)

    @property
    def A(self):
        return _amat(self.xi, self.norm, self._code, self.epsilon, self.smooth)

    def __call__(self, *args):
        a0 = np.asarray(args[0], dtype=float)
        shp = a0.shape
        arrs = List[np.ndarray[float, 1]]()
        for a in args:
            arrs.append(np.ascontiguousarray(np.asarray(a, dtype=float).ravel()))
        nd = len(arrs)
        m = a0.size
        xa = np.empty((nd, m))
        for k in range(nd):
            if arrs[k].size != m:
                raise ValueError("Array lengths must be equal")
            for i in range(m):
                xa[k, i] = arrs[k][i]
        r = cdist(xa.T, self.xi.T, self.norm)
        out = np.empty(m)
        for i in range(m):
            s = 0.0
            for j in range(self.N):
                s += _h(self._code, self.epsilon, r[i, j]) * self.nodes[j]
            out[i] = s
        return out.reshape(shp)
