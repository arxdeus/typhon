# Ported from scipy 1.18.1 (BSD-3-Clause)

# Pairwise distance computations (scipy.spatial.distance subset).
# Metrics may be given as strings or as callables ``f(u, v) -> float``
# taking two 1-D float ndarrays.

import numpy as np
import internal.static as static
from math import sqrt, fabs, inf

def _as2d(X):
    a = np.asarray(X, dtype=float)
    if static.len(a.shape) != 2:
        compile_error("A 2-dimensional array must be passed.")
    return a

def _as1d(u):
    a = np.asarray(u, dtype=float)
    if static.len(a.shape) != 1:
        compile_error("Input vector should be 1-D.")
    return a

def _row(a, i: int):
    n = a.shape[1]
    r = np.empty(n, dtype=float)
    for k in range(n):
        r[k] = a[i, k]
    return r

# ---- metric kernels on rows (a[i], b[j]) ------------------------------

_EUCLID = 0
_SQEUCLID = 1
_CITY = 2
_CHEB = 3
_MINK = 4
_COS = 5
_CORR = 6
_CANB = 7
_BRAY = 8
_SEUC = 9
_MAHA = 10
_HAMM = 11
_JACC = 12
_DICE = 13

def _metric_code(metric: str) -> int:
    m = metric.lower()
    if m in ('euclidean', 'euclid', 'eu', 'e'):
        return _EUCLID
    if m in ('sqeuclidean', 'sqe', 'sqeuclid'):
        return _SQEUCLID
    if m in ('cityblock', 'cblock', 'cb', 'c', 'manhattan'):
        return _CITY
    if m in ('chebyshev', 'chebychev', 'cheby', 'cheb', 'ch'):
        return _CHEB
    if m in ('minkowski', 'mi', 'm', 'pnorm'):
        return _MINK
    if m in ('cosine', 'cos'):
        return _COS
    if m in ('correlation', 'co'):
        return _CORR
    if m in ('canberra',):
        return _CANB
    if m in ('braycurtis',):
        return _BRAY
    if m in ('seuclidean', 'se', 's'):
        return _SEUC
    if m in ('mahalanobis', 'mahal', 'mah'):
        return _MAHA
    if m in ('hamming', 'hamm', 'ha', 'h'):
        return _HAMM
    if m in ('jaccard', 'jacc', 'ja', 'j'):
        return _JACC
    if m in ('dice',):
        return _DICE
    raise ValueError(f"Unknown Distance Metric: {metric}")

def _dist(code: int, a, i: int, b, j: int, p: float, V, VI) -> float:
    n = a.shape[1]
    if code == _EUCLID or code == _SQEUCLID:
        s = 0.0
        for k in range(n):
            d = a[i, k] - b[j, k]
            s += d * d
        return sqrt(s) if code == _EUCLID else s
    elif code == _CITY:
        s = 0.0
        for k in range(n):
            s += fabs(a[i, k] - b[j, k])
        return s
    elif code == _CHEB:
        s = 0.0
        for k in range(n):
            d = fabs(a[i, k] - b[j, k])
            if d > s:
                s = d
        return s
    elif code == _MINK:
        if p == inf:
            return _dist(_CHEB, a, i, b, j, p, V, VI)
        if p == 1.0:
            return _dist(_CITY, a, i, b, j, p, V, VI)
        if p == 2.0:
            return _dist(_EUCLID, a, i, b, j, p, V, VI)
        s = 0.0
        for k in range(n):
            s += fabs(a[i, k] - b[j, k]) ** p
        return s ** (1.0 / p)
    elif code == _COS or code == _CORR:
        mu = 0.0
        mv = 0.0
        if code == _CORR:
            for k in range(n):
                mu += a[i, k]
                mv += b[j, k]
            mu /= n
            mv /= n
        uv = 0.0
        uu = 0.0
        vv = 0.0
        for k in range(n):
            x = a[i, k] - mu
            y = b[j, k] - mv
            uv += x * y
            uu += x * x
            vv += y * y
        d = 1.0 - uv / sqrt(uu * vv)
        if d < 0.0:
            d = 0.0
        if d > 2.0:
            d = 2.0
        return d
    elif code == _CANB:
        s = 0.0
        for k in range(n):
            num = fabs(a[i, k] - b[j, k])
            den = fabs(a[i, k]) + fabs(b[j, k])
            if den != 0.0:
                s += num / den
        return s
    elif code == _BRAY:
        s1 = 0.0
        s2 = 0.0
        for k in range(n):
            s1 += fabs(a[i, k] - b[j, k])
            s2 += fabs(a[i, k] + b[j, k])
        return s1 / s2
    elif code == _SEUC:
        s = 0.0
        for k in range(n):
            d = a[i, k] - b[j, k]
            s += d * d / V[k]
        return sqrt(s)
    elif code == _MAHA:
        s = 0.0
        for k in range(n):
            dk = a[i, k] - b[j, k]
            t = 0.0
            for l in range(n):
                t += VI[k, l] * (a[i, l] - b[j, l])
            s += dk * t
        return sqrt(s)
    elif code == _HAMM:
        c = 0
        for k in range(n):
            if a[i, k] != b[j, k]:
                c += 1
        return c / n
    elif code == _JACC:
        nz = 0
        ne = 0
        for k in range(n):
            x = a[i, k]
            y = b[j, k]
            if x != 0.0 or y != 0.0:
                nz += 1
                if x != y:
                    ne += 1
        return ne / nz if nz != 0 else 0.0
    else:  # dice (boolean)
        ntt = 0
        ndiff = 0
        for k in range(n):
            x = a[i, k] != 0.0
            y = b[j, k] != 0.0
            if x and y:
                ntt += 1
            elif x != y:
                ndiff += 1
        return ndiff / (2 * ntt + ndiff)

def _var_cols(X):
    m = X.shape[0]
    n = X.shape[1]
    V = np.empty(n, dtype=float)
    for k in range(n):
        mu = 0.0
        for i in range(m):
            mu += X[i, k]
        mu /= m
        s = 0.0
        for i in range(m):
            d = X[i, k] - mu
            s += d * d
        V[k] = s / (m - 1)
    return V

def _inv_cov(X):
    return np.linalg.inv(np.cov(X.T))

def _prep_V(V, X):
    if V is None:
        return _var_cols(X)
    else:
        v = _as1d(V)
        if v.shape[0] != X.shape[1]:
            raise ValueError("Variance vector V must be of the same dimension as the vectors on which the distances are computed.")
        return v

def _prep_VI(VI, X):
    if VI is None:
        if X.shape[0] <= X.shape[1]:
            raise ValueError(f"The number of observations ({X.shape[0]}) is too small; the covariance matrix is singular. For observations with {X.shape[1]} dimensions, at least {X.shape[1] + 1} observations are required.")
        return _inv_cov(X)
    else:
        return _as2d(VI)

def pdist(X, metric = 'euclidean', p: float = 2.0, V = None, VI = None):
    """Pairwise distances between observations in n-dimensional space."""
    a = _as2d(X)
    m = a.shape[0]
    out = np.empty(m * (m - 1) // 2, dtype=float)
    if isinstance(metric, str):
        code = _metric_code(metric)
        Vv = _prep_V(V, a) if code == _SEUC else np.ones(a.shape[1], dtype=float)
        VIv = _prep_VI(VI, a) if code == _MAHA else np.zeros((1, 1), dtype=float)
        idx = 0
        for i in range(m):
            for j in range(i + 1, m):
                out[idx] = _dist(code, a, i, a, j, p, Vv, VIv)
                idx += 1
    else:
        idx = 0
        for i in range(m):
            for j in range(i + 1, m):
                out[idx] = float(metric(_row(a, i), _row(a, j)))
                idx += 1
    return out

def cdist(XA, XB, metric = 'euclidean', p: float = 2.0, V = None, VI = None):
    """Distance between each pair of the two collections of inputs."""
    a = _as2d(XA)
    b = _as2d(XB)
    if a.shape[1] != b.shape[1]:
        raise ValueError("XA and XB must have the same number of columns (i.e. feature dimension.)")
    ma = a.shape[0]
    mb = b.shape[0]
    out = np.empty((ma, mb), dtype=float)
    if isinstance(metric, str):
        code = _metric_code(metric)
        Vv = np.ones(a.shape[1], dtype=float)
        VIv = np.zeros((1, 1), dtype=float)
        if code == _SEUC:
            Vv = _prep_V(V, np.vstack((a, b)))
        elif code == _MAHA:
            VIv = _prep_VI(VI, np.vstack((a, b)))
        for i in range(ma):
            for j in range(mb):
                out[i, j] = _dist(code, a, i, b, j, p, Vv, VIv)
    else:
        for i in range(ma):
            ri = _row(a, i)
            for j in range(mb):
                out[i, j] = float(metric(ri, _row(b, j)))
    return out

def _single(code: int, u, v, p: float, V, VI) -> float:
    a = _as1d(u).reshape((1, -1))
    b = _as1d(v).reshape((1, -1))
    return _dist(code, a, 0, b, 0, p, V, VI)

def _nov():
    return np.zeros(1, dtype=float)

def _novi():
    return np.zeros((1, 1), dtype=float)

def euclidean(u, v) -> float:
    return _single(_EUCLID, u, v, 2.0, _nov(), _novi())

def sqeuclidean(u, v) -> float:
    return _single(_SQEUCLID, u, v, 2.0, _nov(), _novi())

def cityblock(u, v) -> float:
    return _single(_CITY, u, v, 2.0, _nov(), _novi())

def chebyshev(u, v) -> float:
    return _single(_CHEB, u, v, 2.0, _nov(), _novi())

def minkowski(u, v, p: float = 2.0) -> float:
    if p <= 0:
        raise ValueError("p must be greater than 0")
    return _single(_MINK, u, v, p, _nov(), _novi())

def cosine(u, v) -> float:
    return _single(_COS, u, v, 2.0, _nov(), _novi())

def correlation(u, v) -> float:
    return _single(_CORR, u, v, 2.0, _nov(), _novi())

def canberra(u, v) -> float:
    return _single(_CANB, u, v, 2.0, _nov(), _novi())

def braycurtis(u, v) -> float:
    return _single(_BRAY, u, v, 2.0, _nov(), _novi())

def seuclidean(u, v, V) -> float:
    return _single(_SEUC, u, v, 2.0, _as1d(V), _novi())

def mahalanobis(u, v, VI) -> float:
    return _single(_MAHA, u, v, 2.0, _nov(), _as2d(VI))

def hamming(u, v) -> float:
    return _single(_HAMM, u, v, 2.0, _nov(), _novi())

def jaccard(u, v) -> float:
    return _single(_JACC, u, v, 2.0, _nov(), _novi())

def dice(u, v) -> float:
    return _single(_DICE, u, v, 2.0, _nov(), _novi())

def is_valid_dm(D, tol: float = 0.0, throw: bool = False, name: str = 'D') -> bool:
    a = np.asarray(D, dtype=float)
    if static.len(a.shape) != 2:
        if throw:
            raise ValueError(f"Distance matrix '{name}' must have shape=2 (i.e. be two-dimensional).")
        return False
    else:
        n = a.shape[0]
        if a.shape[1] != n:
            if throw:
                raise ValueError(f"Distance matrix '{name}' must be symmetric.")
            return False
        for i in range(n):
            if a[i, i] != 0.0:
                if throw:
                    raise ValueError(f"Distance matrix '{name}' diagonal must be zero.")
                return False
            for j in range(n):
                if tol == 0.0:
                    bad = a[i, j] != a[j, i]
                else:
                    bad = fabs(a[i, j] - a[j, i]) > tol
                if bad:
                    if throw:
                        raise ValueError(f"Distance matrix '{name}' must be symmetric.")
                    return False
        return True

def is_valid_y(y, throw: bool = False, name: str = 'y') -> bool:
    a = np.asarray(y, dtype=float)
    if static.len(a.shape) != 1:
        if throw:
            raise ValueError(f"Condensed distance matrix '{name}' must have shape=1 (i.e. be one-dimensional).")
        return False
    else:
        n = a.shape[0]
        d = int(np.ceil(np.sqrt(n * 2)))
        if (d * (d - 1) // 2) != n:
            if throw:
                raise ValueError("Length n of condensed distance matrix 'y' must be a binomial coefficient, i.e.there must be a k such that (k \\choose 2)=n)!")
            return False
        return True

def num_obs_y(Y) -> int:
    a = np.asarray(Y, dtype=float)
    is_valid_y(a, throw=True, name='Y')
    k = a.shape[0]
    if k == 0:
        raise ValueError("The number of observations cannot be determined on an empty distance matrix.")
    return int(np.ceil(np.sqrt(k * 2)))

def num_obs_dm(d) -> int:
    a = np.asarray(d, dtype=float)
    is_valid_dm(a, tol=inf, throw=True, name='d')
    return a.shape[0]

def squareform(X, force: str = 'no', checks: bool = True):
    """Convert between condensed and square distance matrices."""
    a = np.asarray(X, dtype=float)
    f = force.lower()
    if static.len(a.shape) != 1:
        if f == 'tomatrix':
            raise ValueError("Forcing 'tomatrix' but input X is not a distance vector.")
    if static.len(a.shape) != 2:
        if f == 'tovector':
            raise ValueError("Forcing 'tovector' but input X is not a distance matrix.")
    if static.len(a.shape) == 1:
        s = a.shape[0]
        if s == 0:
            return np.zeros((1, 1), dtype=float)
        d = int(np.ceil(np.sqrt(s * 2)))
        if d * (d - 1) != s * 2:
            raise ValueError("Incompatible vector size. It must be a binomial coefficient n choose 2 for some integer n >= 2.")
        M = np.zeros((d, d), dtype=float)
        k = 0
        for i in range(d):
            for j in range(i + 1, d):
                M[i, j] = a[k]
                M[j, i] = a[k]
                k += 1
        return M
    elif static.len(a.shape) == 2:
        if a.shape[0] != a.shape[1]:
            raise ValueError("The matrix argument must be square.")
        if checks:
            is_valid_dm(a, throw=True, name='X')
        d = a.shape[0]
        out = np.empty(d * (d - 1) // 2, dtype=float)
        k = 0
        for i in range(d):
            for j in range(i + 1, d):
                out[k] = a[i, j]
                k += 1
        return out
    else:
        compile_error("The first argument must be one or two dimensional array.")
