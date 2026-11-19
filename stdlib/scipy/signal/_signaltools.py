# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# Static typing note: lfilter/sosfilt return `y` when zi is None (the default)
# and the tuple `(y, zf)` when zi is given, decided at compile time.

import numpy as np
import numpy.linalg as _alg
import math
import cmath
from numpy.util import tuple_set, normalize_axis_index
from _arraytools import _to_rows, _from_rows, _ext_rows, axis_slice, axis_reverse, \
    odd_ext, even_ext, const_ext, zero_ext
from _filter_design import _flist

def _dt_for(x):
    # floating result dtype: complex stays complex, everything else float
    if x.dtype is complex:
        return complex()
    else:
        return float()

def _rows_as(x, axis: int, T: type):
    x2, msh, ax = _to_rows(np.asarray(x), axis)
    if x2.dtype is T:
        return x2, msh, ax
    else:
        return x2.astype(T), msh, ax

# ----------------------------------------------------------------- lfilter

def _lfilter_row(b: List[float], a: List[float], x, r: int, y, z: List):
    # direct form II transposed; b, a normalized and same length n; z length n-1
    n = len(b)
    N = x.shape[1]
    for i in range(N):
        xi = x[r, i]
        yi = (b[0] * xi + z[0]) if n > 1 else b[0] * xi
        for k in range(n - 2):
            z[k] = b[k + 1] * xi + z[k + 1] - a[k + 1] * yi
        if n > 1:
            z[n - 2] = b[n - 1] * xi - a[n - 1] * yi
        y[r, i] = yi

def _norm_ba(b, a):
    bl = _flist(b)
    al = _flist(a)
    if al[0] == 0.0:
        raise ValueError('BUG: filter coefficient a[0] == 0 not supported yet')
    a0 = al[0]
    n = max(len(bl), len(al))
    bl = [v / a0 for v in bl] + [0.0] * (n - len(bl))
    al = [v / a0 for v in al] + [0.0] * (n - len(al))
    return bl, al

def lfilter(b, a, x, axis: int = -1, zi = None):
    bl, al = _norm_ba(b, a)
    xa = np.asarray(x)
    T = type(_dt_for(xa))
    x2, msh, ax = _rows_as(xa, axis, T)
    M = x2.shape[0]
    n = len(bl)
    y = np.empty(x2.shape, dtype=T)
    if zi is None:
        for r in range(M):
            z = [T()] * max(n - 1, 1)
            _lfilter_row(bl, al, x2, r, y, z)
        return _from_rows(y, msh, ax)
    else:
        zia = np.asarray(zi)
        if zia.shape[ax] != n - 1:
            raise ValueError('Unexpected shape for zi: expected (..., n-1, ...)')
        zi2, zsh, _ = _rows_as(np.broadcast_to(zia, tuple_set(xa.shape, ax, n - 1)), axis, T)
        zf = np.empty(zi2.shape, dtype=T)
        for r in range(M):
            z = [zi2[r, k] for k in range(n - 1)]
            if n == 1:
                z = [T()]
            _lfilter_row(bl, al, x2, r, y, z)
            for k in range(n - 1):
                zf[r, k] = z[k]
        return _from_rows(y, msh, ax), _from_rows(zf, zsh, ax)

def lfilter_zi(b, a):
    bl = _flist(b)
    al = _flist(a)
    if al[0] == 0:
        raise ValueError('First coefficient of parameter `a` must be non-zero!')
    if al[0] != 1:
        a0 = al[0]
        bl = [v / a0 for v in bl]
        al = [v / a0 for v in al]
    sa = 0.0
    for v in al:
        sa += v
    if sa == 0:
        raise ValueError('Filter not stable due to sum(a) == 0, i.e., having a pole at z = 1!')
    sb = 0.0
    for v in bl:
        sb += v
    y_inf = sb / sa
    n = max(len(al), len(bl))
    bl += [0.0] * (n - len(bl))
    al += [0.0] * (n - len(al))
    d = [bl[i] - y_inf * al[i] for i in range(n)]
    # flip(cumsum(flip(d)))[1:]
    cs = [0.0] * n
    acc = 0.0
    for i in range(n - 1, -1, -1):
        acc += d[i]
        cs[i] = acc
    out = np.empty(n - 1, dtype=float)
    for i in range(1, n):
        out[i - 1] = cs[i]
    return out

# ----------------------------------------------------------------- sosfilt

def _validate_sos(sos):
    s = np.asarray(sos)
    if s.ndim != 2:
        compile_error('sos array must be 2D')
    if s.shape[1] != 6:
        raise ValueError('sos array must be shape (n_sections, 6)')
    for i in range(s.shape[0]):
        if s[i, 3] != 1:
            raise ValueError('sos[:, 3] should be all ones')
    return s.astype(float), s.shape[0]

def _sosfilt_rows_dir(s, x2, zz, rev: bool):
    # zz: (M, n_sections, 2) states, x2 overwritten in place, rows walked left
    # to right (or right to left when rev). Per-sample arithmetic is exactly
    # yc = b0*xc + z0; z0 = (b1*xc - a1*yc) + z1; z1 = b2*xc - a2*yc.
    ns = s.shape[0]
    M = x2.shape[0]
    N = x2.shape[1]
    # hoisted coefficients: c[5k..5k+4] = b0, b1, b2, a1, a2
    c = Ptr[float](5 * ns if ns > 0 else 1)
    for k in range(ns):
        c[5 * k] = float(s[k, 0])
        c[5 * k + 1] = float(s[k, 1])
        c[5 * k + 2] = float(s[k, 2])
        c[5 * k + 3] = float(s[k, 4])
        c[5 * k + 4] = float(s[k, 5])
    i0 = N - 1 if rev else 0
    st = -1 if rev else 1
    if x2.dtype is float and zz.dtype is float and x2._contig[0] and zz._contig[0]:
        xp = x2.data
        zp = zz.data
        for r in range(M):
            row = xp + r * N
            z = zp + r * (2 * ns)
            i = i0
            for _ in range(N):
                xc = row[i]
                for k in range(ns):
                    cc = c + 5 * k
                    zk = z + 2 * k
                    yc = cc[0] * xc + zk[0]
                    zk[0] = cc[1] * xc - cc[3] * yc + zk[1]
                    zk[1] = cc[2] * xc - cc[4] * yc
                    xc = yc
                row[i] = xc
                i += st
        return
    for r in range(M):
        i = i0
        for _ in range(N):
            xc = x2[r, i]
            for k in range(ns):
                b0 = s[k, 0]
                b1 = s[k, 1]
                b2 = s[k, 2]
                a1 = s[k, 4]
                a2 = s[k, 5]
                yc = b0 * xc + zz[r, k, 0]
                zz[r, k, 0] = b1 * xc - a1 * yc + zz[r, k, 1]
                zz[r, k, 1] = b2 * xc - a2 * yc
                xc = yc
            x2[r, i] = xc
            i += st

def _sosfilt_rows(s, x2, zz):
    _sosfilt_rows_dir(s, x2, zz, False)

def _sosfilt_rows_rev(s, x2, zz):
    _sosfilt_rows_dir(s, x2, zz, True)

def _sosfiltfilt_inplace(s, zi, ext):
    # Forward-backward SOS filtering of every row of 2-D ext in place, initial
    # states zi * (first sample of each pass); s is a float (ns, 6) sos array.
    ns = s.shape[0]
    M = ext.shape[0]
    L = ext.shape[1]
    zz = np.empty((M, ns, 2), dtype=ext.dtype)
    for r in range(M):
        x0 = ext[r, 0]
        for k in range(ns):
            zz[r, k, 0] = zi[k, 0] * x0
            zz[r, k, 1] = zi[k, 1] * x0
    _sosfilt_rows(s, ext, zz)
    for r in range(M):
        y0 = ext[r, L - 1]
        for k in range(ns):
            zz[r, k, 0] = zi[k, 0] * y0
            zz[r, k, 1] = zi[k, 1] * y0
    _sosfilt_rows_rev(s, ext, zz)

def sosfilt(sos, x, axis: int = -1, zi = None):
    s, ns = _validate_sos(sos)
    xa = np.asarray(x)
    T = type(_dt_for(xa))
    x2, msh, ax = _rows_as(xa, axis, T)
    x2 = x2.copy()
    M = x2.shape[0]
    zz = np.zeros((M, ns, 2), dtype=T)
    if zi is not None:
        zia = np.asarray(zi)
        # zi shape: (n_sections, ..., 2 at axis+1, ...)
        if zia.ndim != xa.ndim + 1 or zia.shape[0] != ns or zia.shape[ax + 1] != 2:
            raise ValueError(f'Invalid zi shape. With axis={axis}, an input with shape {xa.shape}, and an sos array with {ns} sections, zi must have shape (n_sections, ..., 2, ...)')
        for k in range(ns):
            zk2, _, _ = _rows_as(zia[k], axis, T)
            for r in range(M):
                zz[r, k, 0] = zk2[r, 0]
                zz[r, k, 1] = zk2[r, 1]
    _sosfilt_rows(s, x2, zz)
    y = _from_rows(x2, msh, ax)
    if zi is None:
        return y
    else:
        zf = np.empty(np.asarray(zi).shape, dtype=T)
        zmsh = tuple_set(msh, len(msh) - 1, 2)
        for k in range(ns):
            zk = np.empty((M, 2), dtype=T)
            for r in range(M):
                zk[r, 0] = zz[r, k, 0]
                zk[r, 1] = zz[r, k, 1]
            zf[k] = _from_rows(zk, zmsh, ax)
        return y, zf

def sosfilt_zi(sos):
    s, ns = _validate_sos(sos)
    zi = np.empty((ns, 2), dtype=float)
    scale = 1.0
    for k in range(ns):
        b = [s[k, 0], s[k, 1], s[k, 2]]
        a = [s[k, 3], s[k, 4], s[k, 5]]
        z = lfilter_zi(b, a)
        zi[k, 0] = scale * z[0]
        zi[k, 1] = scale * z[1]
        scale *= (b[0] + b[1] + b[2]) / (a[0] + a[1] + a[2])
    return zi

# --------------------------------------------------------- forward-backward

def _pad_kind(padtype: Optional[str]) -> str:
    if padtype is None:
        return 'none'
    p: str = padtype
    if p not in ('even', 'odd', 'constant'):
        raise ValueError(f"Unknown value '{p}' given to padtype. padtype must be 'even', 'odd', 'constant', or None.")
    return p

def _edge(kind: str, padlen: Optional[int], N: int, ntaps: int) -> int:
    edge = 0
    if kind == 'none':
        edge = 0
    elif padlen is None:
        edge = ntaps * 3
    else:
        edge = padlen
    if N <= edge:
        raise ValueError(f'The length of the input vector x must be greater than padlen, which is {edge}.')
    return edge

def _pad_rows(x2, edge: int, kind: str):
    if kind == 'none' or edge == 0:
        return x2.copy()
    return _ext_rows(x2, edge, kind)

def filtfilt(b, a, x, axis: int = -1, padtype: Optional[str] = 'odd',
             padlen: Optional[int] = None, method: str = 'pad', irlen: Optional[int] = None):
    if method == 'gust':
        return _filtfilt_gust(b, a, x, axis, irlen)
    if method != 'pad':
        raise ValueError("method must be 'pad' or 'gust'.")
    bl, al = _norm_ba(b, a)
    nb = len(_flist(b))
    na = len(_flist(a))
    xa = np.asarray(x)
    T = type(_dt_for(xa))
    x2, msh, ax = _rows_as(xa, axis, T)
    kind = _pad_kind(padtype)
    edge = _edge(kind, padlen, x2.shape[1], max(na, nb))
    ext = _pad_rows(x2, edge, kind)
    zi = lfilter_zi(_flist(b), _flist(a))
    n = len(bl)
    M = ext.shape[0]
    L = ext.shape[1]
    y = np.empty(ext.shape, dtype=T)
    for r in range(M):
        x0 = ext[r, 0]
        z = [T(zi[k] * x0) for k in range(n - 1)] if n > 1 else [T()]
        _lfilter_row(bl, al, ext, r, y, z)
    yr = np.empty(ext.shape, dtype=T)
    for r in range(M):
        for i in range(L):
            yr[r, i] = y[r, L - 1 - i]
    y2 = np.empty(ext.shape, dtype=T)
    for r in range(M):
        y0 = yr[r, 0]
        z = [T(zi[k] * y0) for k in range(n - 1)] if n > 1 else [T()]
        _lfilter_row(bl, al, yr, r, y2, z)
    N = x2.shape[1]
    out = np.empty((M, N), dtype=T)
    for r in range(M):
        for i in range(N):
            out[r, i] = y2[r, L - 1 - (edge + i)]
    return _from_rows(out, msh, ax)

def _lf_list(bl: List[float], al: List[float], xs: List[float], zi: List[float]) -> List[float]:
    n = len(bl)
    z = list(zi) if n > 1 else [0.0]
    out = [0.0] * len(xs)
    for i in range(len(xs)):
        xi = xs[i]
        yi = (bl[0] * xi + z[0]) if n > 1 else bl[0] * xi
        for k in range(n - 2):
            z[k] = bl[k + 1] * xi + z[k + 1] - al[k + 1] * yi
        if n > 1:
            z[n - 2] = bl[n - 1] * xi - al[n - 1] * yi
        out[i] = yi
    return out

def _filtfilt_gust(b, a, x, axis: int, irlen: Optional[int]):
    # Gustafsson's method (real input). Returns y only.
    bl, al = _norm_ba(b, a)
    order = len(bl) - 1
    x2, msh, ax = _rows_as(np.asarray(x), axis, float)
    R = x2.shape[0]
    n = x2.shape[1]
    if order == 0:
        sc = bl[0] * bl[0]
        y0 = np.empty(x2.shape, dtype=float)
        for r in range(R):
            for i in range(n):
                y0[r, i] = sc * x2[r, i]
        return _from_rows(y0, msh, ax)
    m: int = n
    if irlen is not None:
        if n > 2 * irlen:
            m = irlen
    zi0 = [0.0] * order
    zi0[0] = 1.0
    zeros_z = [0.0] * order
    o0 = _lf_list(bl, al, [0.0] * m, zi0)
    Obs = [[0.0] * order for _ in range(m)]
    for i in range(m):
        Obs[i][0] = o0[i]
    for k in range(1, order):
        for i in range(k, m):
            Obs[i][k] = o0[i - k]
    # S = lfilter(b, a, Obs[::-1], axis=0)
    S = [[0.0] * order for _ in range(m)]
    for k in range(order):
        col = [Obs[m - 1 - i][k] for i in range(m)]
        f = _lf_list(bl, al, col, zeros_z)
        for i in range(m):
            S[i][k] = f[i]
    full = (m == n)
    rowsM = m if full else 2 * m
    Mm = np.zeros((rowsM, 2 * order), dtype=float)
    W = np.zeros((rowsM, 2 * order), dtype=float)
    for i in range(m):
        for k in range(order):
            sr = S[m - 1 - i][k]
            obr = Obs[m - 1 - i][k]
            if full:
                Mm[i, k] = sr - Obs[i][k]
                Mm[i, order + k] = obr - S[i][k]
                W[i, k] = sr
                W[i, order + k] = obr
            else:
                Mm[i, k] = sr - Obs[i][k]
                Mm[m + i, order + k] = obr - S[i][k]
                W[i, k] = sr
                W[m + i, order + k] = obr
    D = np.empty((rowsM, R), dtype=float)
    yfb_all = np.empty((R, n), dtype=float)
    for r in range(R):
        xs = [x2[r, i] for i in range(n)]
        yf = _lf_list(bl, al, xs, zeros_z)
        yfb = _lf_list(bl, al, yf[::-1], zeros_z)[::-1]
        yb = _lf_list(bl, al, xs[::-1], zeros_z)[::-1]
        ybf = _lf_list(bl, al, yb, zeros_z)
        for i in range(n):
            yfb_all[r, i] = yfb[i]
        if full:
            for i in range(n):
                D[i, r] = ybf[i] - yfb[i]
        else:
            for i in range(m):
                D[i, r] = ybf[i] - yfb[i]
                D[m + i, r] = ybf[n - m + i] - yfb[n - m + i]
    ic = _alg.lstsq(Mm, D)[0]  # (2*order, R)
    for r in range(R):
        for i in range(rowsM):
            wic = 0.0
            for k in range(2 * order):
                wic += ic[k, r] * W[i, k]
            if full:
                yfb_all[r, i] += wic
            elif i < m:
                yfb_all[r, i] += wic
            else:
                yfb_all[r, n - m + (i - m)] += wic
    return _from_rows(yfb_all, msh, ax)

def sosfiltfilt(sos, x, axis: int = -1, padtype: Optional[str] = 'odd',
                padlen: Optional[int] = None):
    s, ns = _validate_sos(sos)
    xa = np.asarray(x)
    T = type(_dt_for(xa))
    x2, msh, ax = _rows_as(xa, axis, T)
    ntaps = 2 * ns + 1
    c2 = 0
    c5 = 0
    for k in range(ns):
        if s[k, 2] == 0:
            c2 += 1
        if s[k, 5] == 0:
            c5 += 1
    ntaps -= min(c2, c5)
    kind = _pad_kind(padtype)
    edge = _edge(kind, padlen, x2.shape[1], ntaps)
    ext = _pad_rows(x2, edge, kind)
    zi = sosfilt_zi(s)
    _sosfiltfilt_inplace(s, zi, ext)
    if edge == 0:
        return _from_rows(ext, msh, ax)
    M = ext.shape[0]
    N = x2.shape[1]
    out = np.empty((M, N), dtype=T)
    for r in range(M):
        for i in range(N):
            out[r, i] = ext[r, edge + i]
    return _from_rows(out, msh, ax)

# ------------------------------------------------------------------ detrend

def _solve2(A00: float, A01: float, A11: float, b0, b1):
    det = A00 * A11 - A01 * A01
    return (A11 * b0 - A01 * b1) / det, (A00 * b1 - A01 * b0) / det

def detrend(data, axis: int = -1, type: str = 'linear', bp = 0, overwrite_data: bool = False):
    if type not in ('linear', 'l', 'constant', 'c'):
        raise ValueError("Trend type must be 'linear' or 'constant'.")
    xa = np.asarray(data)
    T = type_of_float(xa)
    x2, msh, ax = _rows_as(xa, axis, T)
    x2 = x2.copy()
    M = x2.shape[0]
    N = x2.shape[1]
    if type in ('constant', 'c'):
        for r in range(M):
            s = T()
            for i in range(N):
                s += x2[r, i]
            m = s / N
            for i in range(N):
                x2[r, i] -= m
        return _from_rows(x2, msh, ax)
    bps = sorted(set([0, N] + [int(v) for v in _flist(bp)]))
    for v in bps:
        if v > N:
            raise ValueError('Breakpoints must be less than length of data along given axis.')
    for m in range(len(bps) - 1):
        lo = bps[m]
        Npts = bps[m + 1] - lo
        if Npts == 0:
            continue
        # least squares fit of y = c0 * t + c1 with t = (1..Npts)/Npts
        A = np.ones((Npts, 2), dtype=float)
        for i in range(Npts):
            A[i, 0] = (i + 1) / Npts
        Y = np.empty((Npts, M), dtype=T)
        for r in range(M):
            for i in range(Npts):
                Y[i, r] = x2[r, lo + i]
        coef = _alg.lstsq(A, Y)[0]
        for r in range(M):
            for i in range(Npts):
                x2[r, lo + i] -= A[i, 0] * coef[0, r] + coef[1, r]
    return _from_rows(x2, msh, ax)

def type_of_float(x):
    if x.dtype is complex:
        return complex
    elif x.dtype is float32:
        return float32
    else:
        return float

# --------------------------------------------------------------- convolution

def _conv_full(a, v):
    # direct convolution (1-D), result dtype from inputs
    n = a.size
    m = v.size
    out = np.zeros(n + m - 1, dtype=type(a[0] * v[0]) if n and m else float)
    for i in range(n):
        ai = a[i]
        for j in range(m):
            out[i + j] += ai * v[j]
    return out

def _mode_slice(full, n: int, m: int, mode: str):
    if mode == 'full':
        return full
    elif mode == 'same':
        start = (full.size - n) // 2
        return full[start:start + n].copy()
    elif mode == 'valid':
        L = max(n, m) - min(n, m) + 1
        start = min(n, m) - 1
        return full[start:start + L].copy()
    raise ValueError("acceptable mode flags are 'valid', 'same', or 'full'")

def _shape_from(lst: List[int], like):
    return Ptr[type(like)](lst._ptr.as_byte())[0]

def _strides_of(sh: List[int]) -> List[int]:
    st = [1] * len(sh)
    for d in range(len(sh) - 2, -1, -1):
        st[d] = st[d + 1] * sh[d + 1]
    return st

def _conv_nd(a, v, caxes: List[bool], mode: str):
    # direct N-D convolution; non-convolved axes broadcast elementwise
    if a.ndim != v.ndim:
        compile_error('in1 and in2 should have the same dimensionality')
    nd = a.ndim
    A = np.ascontiguousarray(a)
    V = np.ascontiguousarray(v)
    ash = [A.shape[d] for d in range(nd)]
    vsh = [V.shape[d] for d in range(nd)]
    fsh = [0] * nd
    for d in range(nd):
        if caxes[d]:
            fsh[d] = ash[d] + vsh[d] - 1
        else:
            if ash[d] != vsh[d] and ash[d] != 1 and vsh[d] != 1:
                raise ValueError('incompatible shapes for in1 and in2')
            fsh[d] = max(ash[d], vsh[d])
    T = type(A.data[0] * V.data[0])
    fst = _strides_of(fsh)
    ast = _strides_of(ash)
    vst = _strides_of(vsh)
    ftot = 1
    for d in range(nd):
        ftot *= fsh[d]
    full = np.zeros(ftot, dtype=T)
    atot = A.size
    vtot = V.size
    ia = [0] * nd
    iv = [0] * nd
    for pa in range(atot):
        r = pa
        for d in range(nd):
            ia[d] = r // ast[d]
            r %= ast[d]
        av = A.data[pa]
        for pv in range(vtot):
            r2 = pv
            ok = True
            off = 0
            for d in range(nd):
                iv[d] = r2 // vst[d]
                r2 %= vst[d]
                if caxes[d]:
                    off += (ia[d] + iv[d]) * fst[d]
                else:
                    if ash[d] == vsh[d]:
                        if ia[d] != iv[d]:
                            ok = False
                            break
                        off += ia[d] * fst[d]
                    elif ash[d] == 1:
                        off += iv[d] * fst[d]
                    else:
                        if iv[d] != 0:
                            pass
                        off += ia[d] * fst[d]
            if ok:
                full.data[off] += av * V.data[pv]
    # mode slicing per convolved axis
    osh = list(fsh)
    ost = [0] * nd
    for d in range(nd):
        if caxes[d]:
            n = ash[d]
            m = vsh[d]
            if mode == 'full':
                pass
            elif mode == 'same':
                osh[d] = n
                ost[d] = (fsh[d] - n) // 2
            elif mode == 'valid':
                osh[d] = max(n, m) - min(n, m) + 1
                ost[d] = min(n, m) - 1
            else:
                raise ValueError("acceptable mode flags are 'valid', 'same', or 'full'")
    out = np.empty(_shape_from(osh, A.shape), dtype=T)
    otot = out.size
    oo = _strides_of(osh)
    for po in range(otot):
        r = po
        off = 0
        for d in range(nd):
            i = r // oo[d]
            r %= oo[d]
            off += (i + ost[d]) * fst[d]
        out.data[po] = full.data[off]
    return out

def _axes_mask(nd: int, axes) -> List[bool]:
    if axes is None:
        return [True] * nd
    elif isinstance(axes, int):
        mk = [False] * nd
        mk[normalize_axis_index(axes, nd)] = True
        return mk
    else:
        mk = [False] * nd
        for ax in axes:
            mk[normalize_axis_index(ax, nd)] = True
        return mk

def _valid_swap(a, v, mode: str) -> bool:
    # scipy swaps inputs in 'valid' mode when in2 is larger in every dimension
    if mode != 'valid':
        return False
    ge = True
    le = True
    for d in range(a.ndim):
        if a.shape[d] < v.shape[d]:
            ge = False
        if a.shape[d] > v.shape[d]:
            le = False
    if not ge and not le:
        raise ValueError("For 'valid' mode, one must be at least as large as the other in every dimension")
    return not ge

def _flip_all(a, conj: bool):
    A = np.ascontiguousarray(a)
    out = np.empty(A.shape, dtype=A.dtype)
    n = A.size
    for i in range(n):
        out.data[i] = A.data[n - 1 - i]
    if conj and A.dtype is complex:
        return np.conj(out)
    return out

def convolve(in1, in2, mode: str = 'full', method: str = 'auto'):
    a = np.asarray(in1)
    v = np.asarray(in2)
    if a.ndim == 1 and v.ndim == 1:
        if _valid_swap(a, v, mode):
            return _mode_slice(_conv_full(v, a), v.size, a.size, mode)
        return _mode_slice(_conv_full(a, v), a.size, v.size, mode)
    if _valid_swap(a, v, mode):
        return _conv_nd(v, a, [True] * a.ndim, mode)
    return _conv_nd(a, v, [True] * a.ndim, mode)

def correlate(in1, in2, mode: str = 'full', method: str = 'auto'):
    a = np.asarray(in1)
    v = np.asarray(in2)
    if _valid_swap(a, v, mode):
        # result is reverse(conj(correlate(in2, in1)))
        return _flip_all(convolve(v, _flip_all(a, True), mode), True)
    return convolve(a, _flip_all(v, True), mode)

def fftconvolve(in1, in2, mode: str = 'full', axes = None):
    # Computed by direct summation for N-D input or explicit axes (exact up to
    # rounding, scipy uses FFTs); 1-D input uses an FFT like scipy.
    a = np.asarray(in1)
    v = np.asarray(in2)
    if a.ndim == 1 and v.ndim == 1:
        if _valid_swap(a, v, mode):
            return fftconvolve(v, a, mode)
        n = a.size
        m = v.size
        L = n + m - 1
        nf = _next_fast_len(L)
        if a.dtype is complex or v.dtype is complex:
            fa = np.fft.fft(a.astype(complex), nf)
            fv = np.fft.fft(v.astype(complex), nf)
            return _mode_slice(np.fft.ifft(fa * fv, nf)[:L].copy(), n, m, mode)
        else:
            fa = np.fft.rfft(a.astype(float), nf)
            fv = np.fft.rfft(v.astype(float), nf)
            return _mode_slice(np.fft.irfft(fa * fv, nf)[:L].copy(), n, m, mode)
    mk = _axes_mask(a.ndim, axes)
    if _valid_swap(a, v, mode):
        return _conv_nd(v.astype(float) if v.dtype is int else v, a, mk, mode)
    return _conv_nd(a.astype(float) if a.dtype is int else a, v, mk, mode)

def oaconvolve(in1, in2, mode: str = 'full', axes = None):
    return fftconvolve(in1, in2, mode, axes)

def _next_fast_len(target: int) -> int:
    # smallest 5-smooth number >= target (scipy.fft.next_fast_len, real=False)
    if target <= 6:
        return max(target, 1)
    best = 1 << 62
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            q = p35
            while q < target:
                q *= 2
            if q < best:
                best = q
            if p35 >= target:
                break
            p35 *= 3
        if p5 >= target:
            break
        p5 *= 5
    return best

# ------------------------------------------------------------------- hilbert

def hilbert(x, N: Optional[int] = None, axis: int = -1):
    xa = np.asarray(x)
    if xa.dtype is complex:
        compile_error('x must be real.')
    x2, msh, ax = _rows_as(xa, axis, float)
    n: int = x2.shape[1]
    if N is not None:
        n = N
    if n <= 0:
        raise ValueError('N must be positive.')
    Xf = np.fft.fft(x2, n, axis=-1)
    h = np.zeros(n, dtype=float)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        for i in range(1, n // 2):
            h[i] = 2.0
    else:
        h[0] = 1.0
        for i in range(1, (n + 1) // 2):
            h[i] = 2.0
    y = np.fft.ifft(Xf * h, axis=-1)
    return _from_rows(y, msh, ax)

# ------------------------------------------------------------------- medfilt

def medfilt(volume, kernel_size: int = 3):
    a = np.asarray(volume)
    if a.ndim != 1:
        compile_error('medfilt: only 1-D input is supported natively')
    if kernel_size % 2 != 1:
        raise ValueError('Each element of kernel_size should be odd.')
    n = a.size
    h = kernel_size // 2
    out = np.empty(n, dtype=float)
    for i in range(n):
        w = [0.0] * kernel_size
        for k in range(kernel_size):
            j = i - h + k
            w[k] = float(a[j]) if 0 <= j < n else 0.0
        w.sort()
        out[i] = w[h]
    return out

# ---------------------------------------------------------------- savgol

def savgol_coeffs(window_length: int, polyorder: int, deriv: int = 0, delta: float = 1.0,
                  pos: Optional[int] = None, use: str = 'conv'):
    if polyorder >= window_length:
        raise ValueError('polyorder must be less than window_length.')
    halflen, rem = divmod(window_length, 2)
    p = 0.0
    if pos is None:
        p = float(halflen) if rem != 0 else halflen - 0.5
    else:
        if not (0 <= pos < window_length):
            raise ValueError('pos must be nonnegative and less than window_length.')
        p = float(pos)
    if use not in ('conv', 'dot'):
        raise ValueError("`use` must be 'conv' or 'dot'")
    if deriv > polyorder:
        return np.zeros(window_length, dtype=float)
    # x = arange(-pos, window_length - pos); if use == 'conv' x reversed
    A = np.empty((polyorder + 1, window_length), dtype=float)
    for j in range(window_length):
        xv = -p + j
        if use == 'conv':
            xv = (window_length - 1 - p) - j
            xv = -xv
            xv = -p + (window_length - 1 - j)
        for i in range(polyorder + 1):
            A[i, j] = xv ** i
    y = np.zeros(polyorder + 1, dtype=float)
    y[deriv] = math.factorial(deriv) / (delta ** deriv)
    coeffs = _alg.lstsq(A, y)[0]
    return coeffs

def _polyfit_eval(xs: List[float], ys: List[float], order: int, deriv: int, x0: float, delta: float) -> float:
    n = len(xs)
    A = np.empty((n, order + 1), dtype=float)
    for i in range(n):
        for j in range(order + 1):
            A[i, j] = xs[i] ** (order - j)
    Y = np.empty(n, dtype=float)
    for i in range(n):
        Y[i] = ys[i]
    c = _alg.lstsq(A, Y)[0]
    # derivative of polynomial (descending coefficients)
    cl = [c[i] for i in range(order + 1)]
    for _ in range(deriv):
        m = len(cl) - 1
        cl = [cl[i] * (m - i) for i in range(m)] if m > 0 else [0.0]
    r = 0.0
    for v in cl:
        r = r * x0 + v
    return r / (delta ** deriv)

def savgol_filter(x, window_length: int, polyorder: int, deriv: int = 0, delta: float = 1.0,
                  axis: int = -1, mode: str = 'interp', cval: float = 0.0):
    if mode not in ('mirror', 'constant', 'nearest', 'wrap', 'interp'):
        raise ValueError("mode must be 'mirror', 'constant', 'nearest' 'wrap' or 'interp'.")
    x2, msh, ax = _rows_as(np.asarray(x), axis, float)
    coeffs = savgol_coeffs(window_length, polyorder, deriv, delta)
    M = x2.shape[0]
    N = x2.shape[1]
    W = window_length
    h = W // 2
    out = np.empty((M, N), dtype=float)
    if mode == 'interp':
        if W > N:
            raise ValueError("If mode is 'interp', window_length must be less than or equal to the size of x.")
    for r in range(M):
        for i in range(N):
            s = 0.0
            for k in range(W):
                j = i + h - k  # convolution: coeffs[k] * x[i + h - k]
                v = 0.0
                if 0 <= j < N:
                    v = x2[r, j]
                elif mode == 'constant':
                    v = cval
                elif mode == 'nearest':
                    v = x2[r, 0] if j < 0 else x2[r, N - 1]
                elif mode == 'wrap':
                    v = x2[r, j % N]
                elif mode == 'mirror':
                    jj = j
                    while jj < 0 or jj >= N:
                        if jj < 0:
                            jj = -jj
                        if jj >= N:
                            jj = 2 * (N - 1) - jj
                    v = x2[r, jj]
                else:
                    v = 0.0
                s += coeffs[k] * v
            out[r, i] = s
        if mode == 'interp' and W <= N:
            xs = [float(t) for t in range(W)]
            ys0 = [x2[r, t] for t in range(W)]
            ys1 = [x2[r, N - W + t] for t in range(W)]
            for t in range(h):
                out[r, t] = _polyfit_eval(xs, ys0, polyorder, deriv, float(t), delta)
            for t in range(W - h, W):
                out[r, N - W + t] = _polyfit_eval(xs, ys1, polyorder, deriv, float(t), delta)
    return _from_rows(out, msh, ax)

# ---------------------------------------------------------- resampling

def upfirdn(h, x, up: int = 1, down: int = 1, axis: int = -1, mode: str = 'constant', cval: float = 0.0):
    if mode != 'constant' or cval != 0.0:
        raise ValueError("only mode='constant' with cval=0 is supported")
    hl = _flist(h)
    x2, msh, ax = _rows_as(np.asarray(x), axis, float)
    M = x2.shape[0]
    N = x2.shape[1]
    L = len(hl)
    nup = (N - 1) * up + 1
    full_len = nup + L - 1
    out_len = (full_len + down - 1) // down
    out = np.zeros((M, out_len), dtype=float)
    for r in range(M):
        for o in range(out_len):
            n = o * down
            s = 0.0
            # y[n] = sum_k h[k] * xu[n - k], xu[j] = x[j/up] if j % up == 0
            kmin = max(0, n - (nup - 1))
            kmax = min(L - 1, n)
            k = kmin
            while k <= kmax:
                j = n - k
                if j % up == 0:
                    s += hl[k] * x2[r, j // up]
                k += 1
            out[r, o] = s
    return _from_rows(out, msh, ax)

def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a

def resample_poly(x, up: int, down: int, axis: int = -1, window = ('kaiser', 5.0),
                  padtype: str = 'constant', cval: Optional[float] = None):
    from _fir_filter_design import firwin
    if padtype != 'constant' or cval is not None:
        raise ValueError("only padtype='constant' is supported")
    if up < 1 or down < 1:
        raise ValueError('up and down must be >= 1')
    g = _gcd(up, down)
    up //= g
    down //= g
    x2, msh, ax = _rows_as(np.asarray(x), axis, float)
    n_in = x2.shape[1]
    n_out = n_in * up
    n_out = n_out // down + (1 if n_out % down else 0)
    if up == 1 and down == 1:
        return _from_rows(x2.copy(), msh, ax)
    hl: List[float] = []
    half_len = 0
    if isinstance(window, str) or isinstance(window, Tuple) or isinstance(window, float):
        max_rate = max(up, down)
        f_c = 1.0 / max_rate
        half_len = 10 * max_rate
        hh = firwin(2 * half_len + 1, f_c, window=window)
        hl = [hh[i] * up for i in range(hh.size)]
    else:
        # an explicit FIR filter given as `window`
        hl = [v * up for v in _flist(window)]
        half_len = (len(hl) - 1) // 2
    n_pre_pad = down - half_len % down
    n_post_pad = 0
    n_pre_remove = (half_len + n_pre_pad) // down
    while _output_len(len(hl) + n_pre_pad + n_post_pad, n_in, up, down) < n_out + n_pre_remove:
        n_post_pad += 1
    hp = [0.0] * n_pre_pad + hl + [0.0] * n_post_pad
    y = upfirdn(hp, x2, up, down, axis=-1)
    out = np.empty((x2.shape[0], n_out), dtype=float)
    for r in range(x2.shape[0]):
        for i in range(n_out):
            out[r, i] = y[r, n_pre_remove + i]
    return _from_rows(out, msh, ax)

def _output_len(len_h: int, in_len: int, up: int, down: int) -> int:
    return (((in_len - 1) * up + len_h) - 1) // down + 1

def resample(x, num: int, axis: int = -1):
    # Fourier method, real input, no window, t=None
    x2, msh, ax = _rows_as(np.asarray(x), axis, float)
    Nx = x2.shape[1]
    X = np.fft.rfft(x2, axis=-1)
    nY = num // 2 + 1
    Y = np.zeros((x2.shape[0], nY), dtype=complex)
    nm = min(num, Nx)
    nk = nm // 2 + 1
    for r in range(x2.shape[0]):
        for k in range(nk):
            Y[r, k] = X[r, k]
        if nm % 2 == 0:
            if num < Nx:
                Y[r, nm // 2] = complex(Y[r, nm // 2].real * 2, 0.0) if False else Y[r, nm // 2] * 2.0 if False else Y[r, nm // 2]
                # scipy: when downsampling to even num, Nyquist bin doubled real part
                Y[r, nm // 2] = complex(2 * X[r, nm // 2].real, 0.0) if num < Nx else Y[r, nm // 2]
            elif num > Nx:
                Y[r, nm // 2] = Y[r, nm // 2] * 0.5
    y = np.fft.irfft(Y, num, axis=-1) * (float(num) / float(Nx))
    return _from_rows(y, msh, ax)

def decimate(x, q: int, n: Optional[int] = None, ftype: str = 'iir', axis: int = -1,
             zero_phase: bool = True):
    from _filter_design import cheby1
    from _fir_filter_design import firwin
    xa = np.asarray(x)
    if ftype == 'fir':
        nn = 20 * q if n is None else n
        b = firwin(nn + 1, 1.0 / q, window='hamming')
        if zero_phase:
            return resample_poly(xa, 1, q, axis=axis, window=b)
        else:
            yfull = lfilter(b, [1.0], xa, axis=axis)
            return _downsample(yfull, q, axis)
    elif ftype == 'iir':
        nn = 8 if n is None else n
        sos = cheby1(nn, 0.05, 0.8 / q, output='sos')
        if zero_phase:
            y = sosfiltfilt(sos, xa, axis=axis)
        else:
            y = sosfilt(sos, xa, axis=axis)
        return _downsample(y, q, axis)
    raise ValueError('invalid ftype')

def _downsample(y, q: int, axis: int):
    y2, msh, ax = _rows_as(y, axis, float)
    N = y2.shape[1]
    n = (N + q - 1) // q
    out = np.empty((y2.shape[0], n), dtype=float)
    for r in range(y2.shape[0]):
        for i in range(n):
            out[r, i] = y2[r, i * q]
    return _from_rows(out, msh, ax)
