# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# DCT/DST types 1-4 (scipy.fft) computed through complex FFTs of length
# 2N (or 2(N+-1) for type 1) along the last axis of a (rows, N) view.
# Results agree with pocketfft's dedicated r2r kernels to rounding.

import numpy as np
import numpy.fft as _f
from numpy.util import normalize_axis_index, tuple_set

def _r1(X, t: int, dst: bool):
    # X: float64 2-D (rows, N). Unnormalized type-t transform along axis 1.
    rows = X.shape[0]
    N = X.shape[1]
    if not dst:
        if t == 1:
            if N == 1:
                return X.copy()
            V = np.zeros((rows, 2 * N - 2), dtype=complex)
            V[:, :N] = X
            for j in range(1, N - 1):
                V[:, 2 * N - 2 - j] = X[:, j]
            return _f.fft(V, axis=1)[:, :N].real.copy()
        if t == 2:
            V = np.zeros((rows, 2 * N), dtype=complex)
            V[:, :N] = X
            for j in range(N):
                V[:, 2 * N - 1 - j] = X[:, j]
            F = _f.fft(V, axis=1)[:, :N]
            w = np.exp(-1j * np.pi * np.arange(N) / (2.0 * N))
            return (F * w).real.copy()
        if t == 3:
            c = np.full(N, 2.0)
            c[0] = 1.0
            w = c * np.exp(1j * np.pi * np.arange(N) / (2.0 * N))
            Z = np.zeros((rows, 2 * N), dtype=complex)
            Z[:, :N] = X * w
            return (_f.ifft(Z, axis=1, norm='forward')[:, :N]).real.copy()
        w = np.exp(-1j * np.pi * np.arange(N) / (2.0 * N))
        Z = np.zeros((rows, 2 * N), dtype=complex)
        Z[:, :N] = X * w
        k = np.arange(N)
        w2 = np.exp(-1j * np.pi * (2.0 * k + 1.0) / (4.0 * N))
        return 2.0 * (_f.fft(Z, axis=1)[:, :N] * w2).real
    if t == 1:
        V = np.zeros((rows, 2 * N + 2), dtype=complex)
        for j in range(N):
            V[:, 1 + j] = X[:, j]
            V[:, 2 * N + 1 - j] = -X[:, j]
        return -(_f.fft(V, axis=1)[:, 1:N + 1].imag)
    s = np.ones(N)
    for j in range(1, N, 2):
        s[j] = -1.0
    if t == 2:
        Y = _r1(X * s, 2, False)
        return Y[:, ::-1].copy()
    Y = _r1(X[:, ::-1].copy(), t, False)
    return Y * s

def _tr2d(X, t: int, norm: str, dst: bool, orth: bool):
    N = X.shape[1]
    r2 = np.sqrt(2.0)
    if orth and (t == 1 or t == 3):
        X = X.copy()
        if t == 1 and not dst:
            X[:, 0] *= r2
            X[:, N - 1] *= r2
        if t == 3:
            if dst:
                X[:, N - 1] *= r2
            else:
                X[:, 0] *= r2
    Y = _r1(X, t, dst)
    m = 2 * N
    if t == 1:
        m = 2 * (N + 1) if dst else 2 * (N - 1)
    if norm == 'ortho':
        Y = Y / np.sqrt(float(m))
    elif norm == 'forward':
        Y = Y / float(m)
    if orth and (t == 1 or t == 2):
        if t == 1 and not dst:
            Y[:, 0] /= r2
            Y[:, N - 1] /= r2
        if t == 2:
            if dst:
                Y[:, N - 1] /= r2
            else:
                Y[:, 0] /= r2
    return Y

def _norm_str(norm: Optional[str]) -> str:
    if norm is None:
        return 'backward'
    n = norm
    if n != 'backward' and n != 'ortho' and n != 'forward':
        raise ValueError(f'Invalid norm value {n}; should be "backward", "ortho" or "forward"')
    return n

def _inv_norm(norm: str) -> str:
    if norm == 'backward':
        return 'forward'
    if norm == 'forward':
        return 'backward'
    return norm

def _real_1d(x, t: int, n: Optional[int], axis: int, norm: str, dst: bool,
             orthogonalize: Optional[bool]):
    if t < 1 or t > 4:
        raise ValueError('invalid type, should be 1, 2, 3 or 4')
    a = np.asarray(x)
    if a.dtype is complex or a.dtype is complex64:
        re = _real_1d(a.real.astype(float), t, n, axis, norm, dst, orthogonalize)
        im = _real_1d(a.imag.astype(float), t, n, axis, norm, dst, orthogonalize)
        return re + 1j * im
    else:
        orth = norm == 'ortho'
        if orthogonalize is not None:
            orth = orthogonalize
        b = a.astype(float)
        ax = normalize_axis_index(axis, b.ndim)
        b = np.moveaxis(b, ax, -1)
        N0 = b.shape[b.ndim - 1]
        N = N0
        if n is not None:
            N = n
        if N < 1:
            raise ValueError(f'invalid number of data points ({N}) specified')
        if t == 1 and not dst and N < 2:
            raise ValueError('DCT-I is not defined for size < 2')
        shp = b.shape
        rows = b.size // N0 if N0 > 0 else 0
        X0 = np.ascontiguousarray(b).reshape(rows, N0)
        X = np.zeros((rows, N), dtype=float)
        k = min(N, N0)
        X[:, :k] = X0[:, :k]
        Y = _tr2d(X, t, norm, dst, orth)
        oshp = tuple_set(shp, b.ndim - 1, N)
        out = Y.reshape(oshp)
        return np.moveaxis(out, -1, ax).copy()

_INV = (0, 1, 3, 2, 4)

def dct(x, type: int = 2, n: Optional[int] = None, axis: int = -1,
        norm: Optional[str] = None, overwrite_x: bool = False,
        workers: Optional[int] = None, orthogonalize: Optional[bool] = None):
    return _real_1d(x, type, n, axis, _norm_str(norm), False, orthogonalize)

def idct(x, type: int = 2, n: Optional[int] = None, axis: int = -1,
         norm: Optional[str] = None, overwrite_x: bool = False,
         workers: Optional[int] = None, orthogonalize: Optional[bool] = None):
    if type < 1 or type > 4:
        raise ValueError('invalid type, should be 1, 2, 3 or 4')
    return _real_1d(x, _INV[type], n, axis, _inv_norm(_norm_str(norm)), False, orthogonalize)

def dst(x, type: int = 2, n: Optional[int] = None, axis: int = -1,
        norm: Optional[str] = None, overwrite_x: bool = False,
        workers: Optional[int] = None, orthogonalize: Optional[bool] = None):
    return _real_1d(x, type, n, axis, _norm_str(norm), True, orthogonalize)

def idst(x, type: int = 2, n: Optional[int] = None, axis: int = -1,
         norm: Optional[str] = None, overwrite_x: bool = False,
         workers: Optional[int] = None, orthogonalize: Optional[bool] = None):
    if type < 1 or type > 4:
        raise ValueError('invalid type, should be 1, 2, 3 or 4')
    return _real_1d(x, _INV[type], n, axis, _inv_norm(_norm_str(norm)), True, orthogonalize)

def _nd_axes(ndim: int, s, axes) -> List[int]:
    res = List[int]()
    if axes is None:
        if s is None:
            for i in range(ndim):
                res.append(i)
        else:
            m = len(s)
            for i in range(ndim - m, ndim):
                res.append(i)
    elif isinstance(axes, int):
        res.append(normalize_axis_index(axes, ndim))
    else:
        for a in axes:
            res.append(normalize_axis_index(a, ndim))
    return res

def _nd_sizes(s, k: int) -> List[int]:
    res = List[int]()
    if s is None:
        for i in range(k):
            res.append(-1)
    elif isinstance(s, int):
        res.append(s)
    else:
        for v in s:
            res.append(int(v))
    if len(res) != k:
        raise ValueError('when given, axes and shape arguments have to be of the same length')
    return res

def _real_nd(x, t: int, s, axes, norm: str, dst: bool, orthogonalize: Optional[bool]):
    a = np.asarray(x).astype(float) if not (np.asarray(x).dtype is complex) else np.asarray(x)
    ax = _nd_axes(a.ndim, s, axes)
    sz = _nd_sizes(s, len(ax))
    y = _real_1d(a, t, None if sz[0] < 0 else sz[0], ax[0], norm, dst, orthogonalize) if len(ax) > 0 else a.copy()
    for i in range(1, len(ax)):
        y = _real_1d(y, t, None if sz[i] < 0 else sz[i], ax[i], norm, dst, orthogonalize)
    return y

def dctn(x, type: int = 2, s = None, axes = None, norm: Optional[str] = None,
         overwrite_x: bool = False, workers: Optional[int] = None,
         orthogonalize: Optional[bool] = None):
    return _real_nd(x, type, s, axes, _norm_str(norm), False, orthogonalize)

def idctn(x, type: int = 2, s = None, axes = None, norm: Optional[str] = None,
          overwrite_x: bool = False, workers: Optional[int] = None,
          orthogonalize: Optional[bool] = None):
    if type < 1 or type > 4:
        raise ValueError('invalid type, should be 1, 2, 3 or 4')
    return _real_nd(x, _INV[type], s, axes, _inv_norm(_norm_str(norm)), False, orthogonalize)

def dstn(x, type: int = 2, s = None, axes = None, norm: Optional[str] = None,
         overwrite_x: bool = False, workers: Optional[int] = None,
         orthogonalize: Optional[bool] = None):
    return _real_nd(x, type, s, axes, _norm_str(norm), True, orthogonalize)

def idstn(x, type: int = 2, s = None, axes = None, norm: Optional[str] = None,
          overwrite_x: bool = False, workers: Optional[int] = None,
          orthogonalize: Optional[bool] = None):
    if type < 1 or type > 4:
        raise ValueError('invalid type, should be 1, 2, 3 or 4')
    return _real_nd(x, _INV[type], s, axes, _inv_norm(_norm_str(norm)), True, orthogonalize)
