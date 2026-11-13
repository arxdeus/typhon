# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# RBFInterpolator (scipy/interpolate/_rbfinterp.py, _rbfinterp_pythran.py).
# The augmented system is solved with LAPACK dsysv like scipy.  Real data
# values only (complex `d` is not supported).

import numpy as np
from math import sqrt, log, exp
from scipy.linalg._basic import _sysv
from scipy.spatial._kdtree import KDTree

_K_LINEAR = 0
_K_TPS = 1
_K_CUBIC = 2
_K_QUINTIC = 3
_K_MQ = 4
_K_IMQ = 5
_K_IQ = 6
_K_GAUSS = 7

def _kernel_code(kernel: str) -> int:
    k = kernel.lower()
    if k == 'linear':
        return _K_LINEAR
    if k == 'thin_plate_spline':
        return _K_TPS
    if k == 'cubic':
        return _K_CUBIC
    if k == 'quintic':
        return _K_QUINTIC
    if k == 'multiquadric':
        return _K_MQ
    if k == 'inverse_multiquadric':
        return _K_IMQ
    if k == 'inverse_quadratic':
        return _K_IQ
    if k == 'gaussian':
        return _K_GAUSS
    raise ValueError("`kernel` must be one of {'linear', 'thin_plate_spline', 'cubic', 'quintic', 'multiquadric', 'inverse_multiquadric', 'inverse_quadratic', 'gaussian'}.")

def _min_degree(code: int) -> int:
    if code == _K_MQ or code == _K_LINEAR:
        return 0
    if code == _K_TPS or code == _K_CUBIC:
        return 1
    if code == _K_QUINTIC:
        return 2
    return -1

def _kfun(code: int, r: float) -> float:
    if code == _K_LINEAR:
        return -r
    if code == _K_TPS:
        return 0.0 if r == 0 else r ** 2 * log(r)
    if code == _K_CUBIC:
        return r ** 3
    if code == _K_QUINTIC:
        return -r ** 5
    if code == _K_MQ:
        return -sqrt(r ** 2 + 1)
    if code == _K_IMQ:
        return 1 / sqrt(r ** 2 + 1)
    if code == _K_IQ:
        return 1 / (r ** 2 + 1)
    return exp(-r ** 2)

def _comb(n: int, k: int) -> int:
    r = 1
    for i in range(k):
        r = r * (n - i) // (i + 1)
    return r

def _cwr(ndim: int, deg: int, start: int, cur: List[int], out: List[List[int]]):
    # combinations_with_replacement(range(ndim), deg) in lexicographic order
    if len(cur) == deg:
        row = [0] * ndim
        for v in cur:
            row[v] += 1
        out.append(row)
        return
    for v in range(start, ndim):
        cur.append(v)
        _cwr(ndim, deg, v, cur, out)
        cur.pop()

def _monomial_powers(ndim: int, degree: int):
    out = List[List[int]]()
    for deg in range(degree + 1):
        _cwr(ndim, deg, 0, List[int](), out)
    P = np.zeros((len(out), ndim), dtype=int)
    for i in range(len(out)):
        for j in range(ndim):
            P[i, j] = out[i][j]
    return P

def _norm_diff(a, i: int, b, j: int, eps: float) -> float:
    s = 0.0
    for k in range(a.shape[1]):
        d = a[i, k] * eps - b[j, k] * eps
        s += d * d
    return sqrt(s)

def _mono(xh, i: int, powers, j: int) -> float:
    p = 1.0
    for k in range(powers.shape[1]):
        p *= xh[i, k] ** float(powers[j, k])
    return p

def _shift_scale(y):
    n = y.shape[1]
    shift = np.empty(n)
    scale = np.empty(n)
    for k in range(n):
        mn = y[0, k]
        mx = y[0, k]
        for i in range(y.shape[0]):
            mn = min(mn, y[i, k])
            mx = max(mx, y[i, k])
        shift[k] = (mx + mn) / 2
        scale[k] = (mx - mn) / 2
        if scale[k] == 0.0:
            scale[k] = 1.0
    return shift, scale

def _build_and_solve(y, d, smoothing, code: int, epsilon: float, powers):
    p = d.shape[0]
    s = d.shape[1]
    r = powers.shape[0]
    shift, scale = _shift_scale(y)
    yhat = (y - shift) / scale
    N = p + r
    # Fortran (column-major) storage: lhs[i, j] at i + j * N
    lhs = np.zeros(N * N)
    for i in range(p):
        for j in range(i + 1):
            v = _kfun(code, _norm_diff(y, i, y, j, epsilon))
            lhs[i + j * N] = v
            lhs[j + i * N] = v
    for i in range(p):
        for j in range(r):
            v = _mono(yhat, i, powers, j)
            lhs[i + (p + j) * N] = v
            lhs[(p + j) + i * N] = v
    for i in range(p):
        lhs[i + i * N] += smoothing[i]
    rhs = np.zeros(N * s)
    for j in range(s):
        for i in range(p):
            rhs[i + j * N] = d[i, j]
    info = _sysv(lhs.data, N, False, False, rhs.data, s)
    if info < 0:
        raise ValueError(f"The {-info}-th argument had an illegal value.")
    elif info > 0:
        msg = "Singular matrix."
        if r > 0:
            pm = np.empty((p, r))
            for i in range(p):
                for j in range(r):
                    pm[i, j] = _mono(yhat, i, powers, j)
            rank = int(np.linalg.matrix_rank(pm))
            if rank < r:
                msg = f"Singular matrix. The matrix of monomials evaluated at the data point coordinates does not have full column rank ({rank}/{r})."
        raise np.linalg.LinAlgError(msg)
    coeffs = np.empty((N, s))
    for j in range(s):
        for i in range(N):
            coeffs[i, j] = rhs[i + j * N]
    return shift, scale, coeffs

def _evaluate(x, y, code: int, epsilon: float, powers, shift, scale, coeffs):
    q = x.shape[0]
    p = y.shape[0]
    r = powers.shape[0]
    xhat = (x - shift) / scale
    vec = np.empty((q, p + r))
    for i in range(q):
        for j in range(p):
            vec[i, j] = _kfun(code, _norm_diff(x, i, y, j, epsilon))
        for j in range(r):
            vec[i, p + j] = _mono(xhat, i, powers, j)
    return vec @ coeffs

class RBFInterpolator:
    def __init__(self, y, d, neighbors: Optional[int] = None, smoothing = 0.0,
                 kernel: str = 'thin_plate_spline', epsilon: Optional[float] = None,
                 degree: Optional[int] = None):
        ya = np.ascontiguousarray(np.asarray(y, dtype=float))
        if ya.ndim != 2:
            raise ValueError("`y` must be a 2-dimensional array.")
        ny = ya.shape[0]
        ndim = ya.shape[1]
        da = np.asarray(d, dtype=float)
        if da.shape[0] != ny:
            raise ValueError(f"Expected the first axis of `d` to have length {ny}.")
        d_shape = da.shape[1:]
        d2 = np.ascontiguousarray(da).reshape((ny, -1)).copy()
        sm = np.empty(ny)
        if isinstance(smoothing, float) or isinstance(smoothing, int):
            for i in range(ny):
                sm[i] = float(smoothing)
        else:
            sa = np.asarray(smoothing, dtype=float).ravel()
            if sa.size != ny:
                raise ValueError(f"Expected `smoothing` to be a scalar or have shape ({ny},).")
            for i in range(ny):
                sm[i] = sa[i]
        code = _kernel_code(kernel)
        eps = 1.0
        if epsilon is None:
            if code > _K_QUINTIC:
                raise ValueError("`epsilon` must be specified if `kernel` is not one of {'linear', 'thin_plate_spline', 'cubic', 'quintic'}.")
        else:
            eps = float(epsilon)
        mind = _min_degree(code)
        deg = max(mind, 0)
        if degree is not None:
            deg = degree
            if deg < -1:
                raise ValueError("`degree` must be at least -1.")
        nobs = ny
        nb = 0
        if neighbors is not None:
            nb = min(neighbors, ny)
            nobs = nb
        powers = _monomial_powers(ndim, deg)
        if powers.shape[0] > nobs:
            raise ValueError(f"At least {powers.shape[0]} data points are required when `degree` is {deg} and the number of dimensions is {ndim}.")
        self.y = ya
        self.d = d2
        self.d_shape = d_shape
        self.neighbors = nb
        self.smoothing = sm
        self.kernel = kernel.lower()
        self._code = code
        self.epsilon = eps
        self.powers = powers
        self._tree = KDTree(ya)
        if nb == 0:
            self._shift, self._scale, self._coeffs = _build_and_solve(ya, d2, sm, code, eps, powers)
        else:
            self._shift = np.zeros(ndim)
            self._scale = np.ones(ndim)
            self._coeffs = np.zeros((1, 1))

    def __call__(self, x):
        xa = np.ascontiguousarray(np.asarray(x, dtype=float))
        if xa.ndim != 2:
            raise ValueError("`x` must be a 2-dimensional array.")
        nx = xa.shape[0]
        if xa.shape[1] != self.y.shape[1]:
            raise ValueError(f"Expected the second axis of `x` to have length {self.y.shape[1]}.")
        S = self.d.shape[1]
        out = np.empty((nx, S))
        if self.neighbors == 0:
            out = _evaluate(xa, self.y, self._code, self.epsilon, self.powers, self._shift, self._scale, self._coeffs)
        else:
            k = self.neighbors
            # group query points by their (sorted) neighbour index sets
            groups = Dict[str, List[int]]()
            keys = Dict[str, List[int]]()
            order = List[str]()
            for i in range(nx):
                best = self._tree._knn(xa[i], k, 2.0, np.inf)
                idx = sorted([best[j][1] for j in range(len(best))])
                key = ','.join([str(v) for v in idx])
                if key not in groups:
                    groups[key] = List[int]()
                    keys[key] = idx
                    order.append(key)
                groups[key].append(i)
            for key in order:
                yidx = keys[key]
                xidx = groups[key]
                yn = np.empty((len(yidx), self.y.shape[1]))
                dn = np.empty((len(yidx), S))
                sn = np.empty(len(yidx))
                for a in range(len(yidx)):
                    yn[a] = self.y[yidx[a]]
                    dn[a] = self.d[yidx[a]]
                    sn[a] = self.smoothing[yidx[a]]
                xn = np.empty((len(xidx), xa.shape[1]))
                for a in range(len(xidx)):
                    xn[a] = xa[xidx[a]]
                sh, sc, co = _build_and_solve(yn, dn, sn, self._code, self.epsilon, self.powers)
                v = _evaluate(xn, yn, self._code, self.epsilon, self.powers, sh, sc, co)
                for a in range(len(xidx)):
                    out[xidx[a]] = v[a]
        return out.reshape((nx,) + self.d_shape)
