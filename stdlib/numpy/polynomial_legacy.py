# Legacy polynomial helpers (np.polyval, np.polyfit, np.roots, ...) and
# assorted small routines (unwrap, angle, real_if_close, trapezoid,
# cumulative_sum).

import util
import internal.static as static

from .ndarray import ndarray
from .routines import asarray, array, empty, zeros, ones, atleast_1d, \
                      trim_zeros, vander, concatenate, iscomplexobj, \
                      swapaxes, moveaxis, expand_dims
from .ndmath import trapz, cumsum, diff, convolve
from .linalg import eigvals, lstsq, inv
from .linalg_sym import dot

def _float_type(T: type):
    if T is complex or T is complex64:
        return T()
    elif T is float32:
        return float32()
    else:
        return float()

def polyval(p, x):
    p = asarray(p)
    if isinstance(x, ndarray) or isinstance(x, List) or isinstance(x, Tuple):
        xa = asarray(x)
        y = zeros(xa.shape, dtype=type(util.coerce(xa.dtype, p.dtype)))
        for i in range(p.size):
            y = y * xa + p[i]
        return y
    else:
        R = type(util.coerce(type(x), p.dtype))
        y = util.zero(R)
        for i in range(p.size):
            y = y * util.cast(x, R) + util.cast(p[i], R)
        return y

def poly(seq_of_zeros):
    a = asarray(seq_of_zeros)
    if a.ndim == 2:
        if a.shape[0] == a.shape[1] and a.shape[0] != 0:
            zs = eigvals(a)
            return _poly_from_roots(zs, real_out=not iscomplexobj(a))
        else:
            raise ValueError("input must be 1d or non-empty square 2d array.")
    elif a.ndim == 1:
        return _poly_from_roots(a, real_out=True)
    else:
        compile_error("input must be 1d or non-empty square 2d array.")

def _poly_from_roots(zs, real_out: bool):
    T = zs.dtype
    n = zs.size
    c = zeros(n + 1, dtype=T)
    c[0] = T(1)
    for k in range(n):
        z = zs[k]
        for i in range(k + 1, 0, -1):
            c[i] = c[i] - z * c[i - 1]
    if T is complex or T is complex64:
        # numpy returns real coefficients when roots come in conjugate pairs
        return c
    elif T is float or T is float32:
        return c
    else:
        return c.astype(float)

def roots(p):
    """
    Return the roots of a polynomial with coefficients given in p.
    Note: always returns a complex array (numpy returns a real array if all
    roots are real; use np.real_if_close for that).
    """
    p = atleast_1d(asarray(p))
    if p.ndim != 1:
        raise ValueError("Input must be a rank-1 array.")
    nz = [i for i in range(p.size) if p[i] != type(p[0])()]
    if len(nz) == 0:
        return empty(0, dtype=complex)
    trailing = p.size - nz[-1] - 1
    q = p[nz[0]:nz[-1] + 1]
    N = q.size
    if N > 1:
        F = type(util.coerce(q.dtype, float))
        A = zeros((N - 1, N - 1), dtype=F)
        for i in range(1, N - 1):
            A[i, i - 1] = F(1)
        for j in range(N - 1):
            A[0, j] = util.cast(-q[j + 1], F) / util.cast(q[0], F)
        r = eigvals(A).astype(complex)
    else:
        r = empty(0, dtype=complex)
    if trailing > 0:
        r = concatenate((r, zeros(trailing, dtype=complex)))
    return r

def polyder(p, m: int = 1):
    if m < 0:
        raise ValueError("Order of derivative must be positive (see polyint)")
    p = asarray(p)
    T = type(util.coerce(p.dtype, int))
    val = p.astype(T)
    for _ in range(m):
        n = val.size - 1
        if n <= 0:
            val = zeros(1, dtype=T) if n <= 0 else val
            continue
        nv = empty(n, dtype=T)
        for i in range(n):
            nv[i] = val[i] * T(n - i)
        val = nv
    return val

def polyint(p, m: int = 1, k = None):
    if m < 0:
        raise ValueError("Order of integral must be positive (see polyder)")
    p = asarray(p)
    T = type(util.coerce(p.dtype, float))
    kk = zeros(m, dtype=T)
    if k is not None:
        if isinstance(k, ndarray) or isinstance(k, List) or isinstance(k, Tuple):
            ka = asarray(k)
            if ka.size == 1:
                for i in range(m):
                    kk[i] = util.cast(ka[0], T)
            elif ka.size < m:
                raise ValueError("k must be a scalar or a rank-1 array of length 1 or >m.")
            else:
                for i in range(m):
                    kk[i] = util.cast(ka[i], T)
        else:
            for i in range(m):
                kk[i] = util.cast(k, T)
    val = p.astype(T)
    for j in range(m):
        n = val.size
        nv = empty(n + 1, dtype=T)
        for i in range(n):
            nv[i] = val[i] / T(n - i)
        nv[n] = kk[j]
        val = nv
    return val

def polyadd(a1, a2):
    a1 = atleast_1d(asarray(a1))
    a2 = atleast_1d(asarray(a2))
    T = type(util.coerce(a1.dtype, a2.dtype))
    n = max(a1.size, a2.size)
    r = zeros(n, dtype=T)
    for i in range(a1.size):
        r[n - a1.size + i] += util.cast(a1[i], T)
    for i in range(a2.size):
        r[n - a2.size + i] += util.cast(a2[i], T)
    return r

def polysub(a1, a2):
    return polyadd(a1, -asarray(a2))

def polymul(a1, a2):
    return convolve(atleast_1d(asarray(a1)), atleast_1d(asarray(a2)))

def polyfit(x, y, deg: int, rcond = None, full: Literal[bool] = False, w = None,
            cov: Literal[bool] = False):
    x = asarray(x).astype(float)
    y = asarray(y)
    if deg < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        compile_error("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        compile_error("expected 1D or 2D array for y")
    if x.shape[0] != y.shape[0]:
        raise TypeError("expected x and y to have same length")
    order = deg + 1
    yv = y.astype(type(util.coerce(y.dtype, float)))
    lhs = vander(x, order)
    if w is not None:
        wa = asarray(w).astype(float)
        if wa.ndim != 1:
            raise TypeError("expected a 1-d array for weights")
        if wa.shape[0] != y.shape[0]:
            raise TypeError("expected w and y to have the same length")
        lhs = lhs * wa.reshape(wa.size, 1)
        if y.ndim == 2:
            yv = yv * wa.reshape(wa.size, 1)
        else:
            yv = yv * wa
    rc = float(x.size) * util.eps(float) if rcond is None else float(rcond)
    # scale lhs to improve condition number
    scale = empty(order, dtype=float)
    for j in range(order):
        s = 0.0
        for i in range(x.size):
            s += lhs[i, j] * lhs[i, j]
        scale[j] = util.sqrt(s) if s != 0.0 else 1.0
    lhs = lhs / scale
    c, resids, rank, s = lstsq(lhs, yv, rcond=rc)
    if y.ndim == 2:
        c = (c.T / scale).T
    else:
        c = c / scale
    if full:
        return c, resids, rank, s, rc
    elif cov:
        Vbase = inv(dot(lhs.T, lhs))
        Vbase = Vbase / (scale.reshape(order, 1) * scale.reshape(1, order))
        n = x.size
        if n <= order:
            raise ValueError("the number of data points must exceed order to scale the covariance matrix")
        fac = 1.0
        if y.ndim == 1:
            r = resids.sum() if resids.size > 0 else 0.0
            fac = r / float(n - order)
            return c, Vbase * fac
        else:
            K = y.shape[1]
            outc = empty((order, order, K), dtype=float)
            for kcol in range(K):
                f = (resids[kcol] if resids.size > 0 else 0.0) / float(n - order)
                for i in range(order):
                    for j in range(order):
                        outc[i, j, kcol] = Vbase[i, j] * f
            return c, outc
    else:
        return c

def angle(z, deg: bool = False):
    if isinstance(z, ndarray) or isinstance(z, List) or isinstance(z, Tuple):
        za = asarray(z)
        out = empty(za.shape, dtype=float)
        i = 0
        for v in za.flat:
            if isinstance(v, complex) or isinstance(v, complex64):
                a = util.atan2(float(v.imag), float(v.real))
            else:
                a = util.atan2(0.0, float(v))
            out.data[i] = a * (180.0 / util.PI) if deg else a
            i += 1
        return out
    else:
        if isinstance(z, complex) or isinstance(z, complex64):
            a = util.atan2(float(z.imag), float(z.real))
        else:
            a = util.atan2(0.0, float(z))
        return a * (180.0 / util.PI) if deg else a

def real_if_close(a, tol: float = 100):
    """
    Returns the real part if all imaginary parts are close to zero.
    Deviation: since the return type must be static, a complex input always
    returns a float array; ValueError is raised if any imaginary part is not
    within tol (numpy would return the complex input unchanged). Use
    np.iscomplexobj / np.isreal checks beforehand if that matters.
    """
    a = asarray(a)
    if a.dtype is complex or a.dtype is complex64:
        F = float if a.dtype is complex else float32
        t = tol
        if t > 1.0:
            t = t * util.eps(F)
        out = empty(a.shape, dtype=F)
        i = 0
        for v in a.flat:
            if abs(float(v.imag)) >= t:
                raise ValueError("real_if_close(): imaginary part not close to zero")
            out.data[i] = F(v.real)
            i += 1
        return out
    else:
        return a

def unwrap(p, discont = None, axis: int = -1, period: float = 6.283185307179586):
    p = asarray(p)
    F = type(_float_type(p.dtype))
    a = p.astype(F)
    nd: Literal[int] = static.len(a.shape)
    ax = util.normalize_axis_index(axis, nd)
    d = discont if discont is not None else period / 2.0
    dd = float(d)
    if dd < period / 2.0:
        dd = period / 2.0
    high = period / 2.0
    low = -high
    # work on axis moved to the end, contiguous copy
    m = moveaxis(a, ax, -1).copy()
    n = m.shape[nd - 1]
    rows = m.size // n if n > 0 else 0
    data = m.data
    for r in range(rows):
        base = r * n
        cum = 0.0
        prev = float(data[base]) if n > 0 else 0.0
        for i in range(1, n):
            cur = float(data[base + i])
            dlt = cur - prev
            prev = cur
            ddmod = ((dlt - low) % period) + low
            if ddmod == low and dlt > 0:
                ddmod = high
            phc = ddmod - dlt
            if abs(dlt) < dd:
                phc = 0.0
            cum += phc
            data[base + i] = F(cur + cum)
    return moveaxis(m, -1, ax).copy()

def trapezoid(y, x = None, dx = 1.0, axis: int = -1):
    return trapz(y, x=x, dx=dx, axis=axis)

def cumulative_sum(x, axis = None, dtype: type = NoneType, out = None,
                   include_initial: bool = False):
    xa = asarray(x)
    if axis is None:
        if xa.ndim != 1:
            raise ValueError("axis must be specified for multi-dimensional arrays")
        ax = 0
    else:
        ax = axis
    c = cumsum(xa, axis=ax, dtype=dtype)
    if include_initial:
        shp = c.shape
        ax2 = util.normalize_axis_index(ax, xa.ndim)
        zshape = util.tuple_set(shp, ax2, 1)
        c = concatenate((zeros(zshape, dtype=c.dtype), c), axis=ax2)
    if out is not None:
        out[...] = c
        return out
    return c
