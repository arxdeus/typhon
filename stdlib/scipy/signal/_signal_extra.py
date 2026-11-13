# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# FIR design (remez, firls, minimum_phase), filtering helpers (lfiltic,
# deconvolve, wiener, medfilt2d, order_filter, hilbert2, correlation_lags,
# choose_conv_method, vectorstrength), waveforms (sawtooth, square,
# gausspulse, chirp, sweep_poly, unit_impulse), chirp-z (czt_points, czt,
# zoom_fft, CZT, ZoomFFT) and lombscargle.
#
# Deviations: wiener, medfilt2d and order_filter are 2-D (or 1-D for
# wiener) only. gausspulse returns yI only for retquad/retenv False (static
# flags). chirp(complex=True) is selected through the compile-time flag.

import numpy as np
import math
import cmath
from ._remez import _pre_remez
from ._signaltools import lfilter, convolve, correlate

# ---------------------------------------------------------------- FIR design

def remez(numtaps: int, bands, desired, *, weight = None, type: str = 'bandpass',
          maxiter: int = 25, grid_density: int = 16, fs: Optional[float] = None):
    fsv = 1.0 if fs is None else fs
    if type == 'bandpass':
        tnum = 1
    elif type == 'differentiator':
        tnum = 2
    elif type == 'hilbert':
        tnum = 3
    else:
        raise ValueError("Type must be 'bandpass', 'differentiator', or 'hilbert'")
    if numtaps < 2:
        raise ValueError('The number of taps must be greater than 1.')
    b0 = np.asarray(bands).ravel()
    d0 = np.asarray(desired).ravel()
    bl = [float(b0[i]) for i in range(b0.size)]
    dl = [float(d0[i]) for i in range(d0.size)]
    if weight is None:
        wl = [1.0 for _ in range(len(dl))]
    else:
        w0 = np.asarray(weight).ravel()
        wl = [float(w0[i]) for i in range(w0.size)]
    nb = len(dl)
    if len(bl) != 2 * nb or len(wl) != nb:
        raise ValueError('The inputs desired and weight must have same length.\n  The input bands must have twice this length.')
    old = 0.0
    for k in range(2 * nb):
        if bl[k] < old:
            raise ValueError('Bands must be monotonic starting at zero.')
        if bl[k] * 2 > fsv:
            raise ValueError('Band edges should be less than 1/2 the sampling frequency')
        old = bl[k]
        bl[k] = old / fsv
    h = _pre_remez(numtaps, nb, bl, dl, wl, tnum, maxiter, grid_density)
    out = np.empty(numtaps)
    for i in range(numtaps):
        out[i] = h[i]
    return out

def _sinc(x: float) -> float:
    if x == 0.0:
        return 1.0
    y = math.pi * x
    return math.sin(y) / y

def firls(numtaps: int, bands, desired, *, weight = None, fs: Optional[float] = None):
    from scipy.linalg import toeplitz, hankel, solve, lstsq
    fsv = 2.0 if fs is None else fs
    nyq = 0.5 * fsv
    if numtaps % 2 == 0 or numtaps < 1:
        raise ValueError('numtaps must be odd and >= 1')
    M = (numtaps - 1) // 2
    if nyq <= 0:
        raise ValueError(f'nyq must be positive, got {nyq} <= 0.')
    b0 = np.asarray(bands).ravel()
    bl = [float(b0[i]) / nyq for i in range(b0.size)]
    if len(bl) % 2 != 0:
        raise ValueError('bands must contain frequency pairs.')
    for v in bl:
        if v < 0 or v > 1:
            raise ValueError('bands must be between 0 and 1 relative to Nyquist')
    nb = len(bl) // 2
    d0 = np.asarray(desired).ravel()
    if d0.size != len(bl):
        raise ValueError(f'desired must have one entry per frequency, got {d0.size} gains for {len(bl)} frequencies.')
    dl = [float(d0[i]) for i in range(d0.size)]
    for i in range(nb):
        if bl[2 * i + 1] - bl[2 * i] <= 0 or (i > 0 and bl[2 * i] < bl[2 * i - 2]):
            raise ValueError('bands must be monotonically nondecreasing and have width > 0.')
        if i > 0 and bl[2 * i - 1] > bl[2 * i]:
            raise ValueError('bands must not overlap.')
    for v in dl:
        if v < 0:
            raise ValueError('desired must be non-negative.')
    if weight is None:
        wl = [1.0 for _ in range(nb)]
    else:
        w0 = np.asarray(weight).ravel()
        wl = [float(w0[i]) for i in range(w0.size)]
    if len(wl) != nb:
        raise ValueError(f'weight must be the same size as the number of band pairs ({nb}).')
    for v in wl:
        if v < 0:
            raise ValueError('weight must be non-negative.')
    q = np.empty(numtaps)
    for n in range(numtaps):
        s = 0.0
        for i in range(nb):
            b1 = bl[2 * i]; b2 = bl[2 * i + 1]
            s += (_sinc(b2 * n) * b2 - _sinc(b1 * n) * b1) * wl[i]
        q[n] = s
    Q = toeplitz(q[:M + 1].copy()) + hankel(q[:M + 1].copy(), q[M:].copy())
    bvec = np.zeros(M + 1)
    for i in range(nb):
        b1 = bl[2 * i]; b2 = bl[2 * i + 1]
        m = (dl[2 * i + 1] - dl[2 * i]) / (b2 - b1)
        c = dl[2 * i] - b1 * m
        for n in range(M + 1):
            v2 = b2 * (m * b2 + c) * _sinc(b2 * n)
            v1 = b1 * (m * b1 + c) * _sinc(b1 * n)
            if n == 0:
                v2 -= m * b2 * b2 / 2.0
                v1 -= m * b1 * b1 / 2.0
            else:
                v2 += m * math.cos(n * math.pi * b2) / (math.pi * n) ** 2
                v1 += m * math.cos(n * math.pi * b1) / (math.pi * n) ** 2
            bvec[n] += (v2 - v1) * wl[i]
    a = solve(Q, bvec, assume_a='pos', check_finite=False)
    coeffs = np.empty(2 * M + 1)
    for i in range(M):
        coeffs[i] = a[M - i]
    coeffs[M] = 2 * a[0]
    for i in range(1, M + 1):
        coeffs[M + i] = a[i]
    return coeffs

def _dhtm(mag):
    n = mag.shape[0]
    sig = np.zeros(n)
    midpt = n // 2
    for i in range(1, midpt):
        sig[i] = 1.0
    for i in range(midpt + 1, n):
        sig[i] = -1.0
    lm = np.fft.ifft(np.log(mag).astype(complex))
    t = np.fft.fft(sig * lm)
    recon = np.fft.ifft(mag * np.exp(t))
    return recon.real.copy()

def minimum_phase(h, method: str = 'homomorphic', n_fft: Optional[int] = None, *, half: bool = True):
    ha = np.asarray(h).astype(float)
    if ha.ndim != 1 or ha.shape[0] <= 2:
        raise ValueError('h must be 1-D and at least 2 samples long')
    N = ha.shape[0]
    n_half = N // 2
    if method != 'homomorphic' and method != 'hilbert':
        raise ValueError(f'method must be "homomorphic" or "hilbert", got {method!r}')
    if method == 'hilbert' and not half:
        raise ValueError("`half=False` is only supported when `method='homomorphic'`")
    nf = 2 ** int(math.ceil(math.log2(2 * (N - 1) / 0.01))) if n_fft is None else n_fft
    if nf < N:
        raise ValueError(f'n_fft must be at least len(h)=={N}')
    F = np.fft.fft(ha.astype(complex), nf)
    if method == 'hilbert':
        H = np.empty(nf)
        for i in range(nf):
            w = i * (2 * math.pi / nf * n_half)
            H[i] = (F[i] * cmath.exp(complex(0.0, w))).real
        mx = -math.inf
        mn = math.inf
        for v in H:
            mx = max(mx, v)
            mn = min(mn, v)
        dp = mx - 1
        ds = 0 - mn
        S = 4.0 / (math.sqrt(1 + dp + ds) + math.sqrt(1 - dp + ds)) ** 2
        for i in range(nf):
            H[i] = math.sqrt((H[i] + ds) * S) + 1e-10
        h_min = _dhtm(H)
    else:
        ht = np.empty(nf)
        mnpos = math.inf
        for i in range(nf):
            ht[i] = abs(F[i])
            if ht[i] > 0:
                mnpos = min(mnpos, ht[i])
        for i in range(nf):
            ht[i] = math.log(ht[i] + 1e-07 * mnpos)
            if half:
                ht[i] *= 0.5
        hr = np.fft.ifft(ht.astype(complex)).real.copy()
        stop = nf // 2
        for i in range(nf):
            if i == 0:
                wv = 1.0
            elif i < stop:
                wv = 2.0
            elif i == stop:
                wv = 1.0 + nf % 2
            else:
                wv = 0.0
            hr[i] *= wv
        hm = np.fft.ifft(np.exp(np.fft.fft(hr.astype(complex))))
        h_min = hm.real.copy()
    n_out = n_half + N % 2 if half else N
    return h_min[:n_out].copy()

# ---------------------------------------------------------------- filtering helpers

def lfiltic(b, a, y, x = None):
    bl = np.atleast_1d(np.asarray(b)).astype(float)
    al = np.atleast_1d(np.asarray(a)).astype(float)
    N = al.size - 1
    M = bl.size - 1
    K = max(M, N)
    if N < 0:
        raise ValueError('There must be at least one `a` coefficient.')
    y0 = np.atleast_1d(np.asarray(y)).astype(float)
    xv = np.zeros(M)
    if x is not None:
        x0 = np.atleast_1d(np.asarray(x)).astype(float)
        for i in range(min(M, x0.size)):
            xv[i] = x0[i]
    yv = np.zeros(max(N, y0.size))
    for i in range(y0.size):
        yv[i] = y0[i]
    zi = np.zeros(K)
    for m in range(M):
        s = 0.0
        for j in range(M - m):
            s += bl[m + 1 + j] * xv[j]
        zi[m] = s
    for m in range(N):
        s = 0.0
        for j in range(N - m):
            s += al[m + 1 + j] * yv[j]
        zi[m] -= s
    if al[0] != 1.0:
        if al[0] == 0.0:
            raise ValueError('First `a` filter coefficient must be non-zero.')
        for i in range(K):
            zi[i] /= al[0]
    return zi

def deconvolve(signal, divisor):
    num = np.atleast_1d(np.asarray(signal)).astype(float)
    den = np.atleast_1d(np.asarray(divisor)).astype(float)
    if num.ndim != 1 or num.size == 0:
        raise ValueError(f'Parameter signal must be non-empty 1d array, but its shape is {num.shape}!')
    if den.ndim != 1 or den.size == 0:
        raise ValueError(f'Parameter divisor must be non-empty 1d array, but its shape is {den.shape}!')
    N = num.size
    D = den.size
    if D > N:
        return np.empty(0), num
    inp = np.zeros(N - D + 1)
    inp[0] = 1.0
    quot = lfilter(num, den, inp)
    rem = num - convolve(den, quot, mode='full')
    return quot, rem

def _box_mean(im, my: int, mx: int):
    # correlate(im, ones((my, mx)), 'same') with zero padding
    R, C = im.shape
    out = np.empty((R, C))
    oy = (my - 1) // 2
    ox = (mx - 1) // 2
    for i in range(R):
        for j in range(C):
            s = 0.0
            for di in range(my):
                ii = i + di - oy
                if 0 <= ii < R:
                    for dj in range(mx):
                        jj = j + dj - ox
                        if 0 <= jj < C:
                            s += im[ii, jj]
            out[i, j] = s
    return out

def wiener(im, mysize = None, noise: Optional[float] = None):
    a = np.asarray(im).astype(float)
    if a.ndim == 1:
        return wiener(a.reshape((1, a.size)), (1, 3 if mysize is None else int(mysize)), noise).ravel()
    elif a.ndim != 2:
        compile_error('wiener: only 1-D and 2-D input is supported')
    else:
        if mysize is None:
            my = 3; mx = 3
        elif isinstance(mysize, int):
            my = mysize; mx = mysize
        else:
            my = int(mysize[0]); mx = int(mysize[1])
        size = float(my * mx)
        lMean = _box_mean(a, my, mx) / size
        lVar = _box_mean(a * a, my, mx) / size - lMean ** 2
        nz = 0.0
        if noise is None:
            for v in lVar.flat:
                nz += v
            nz /= lVar.size
        else:
            nz = noise
        out = np.empty(a.shape)
        R, C = a.shape
        for i in range(R):
            for j in range(C):
                if lVar[i, j] < nz:
                    out[i, j] = lMean[i, j]
                else:
                    out[i, j] = (a[i, j] - lMean[i, j]) * (1 - nz / lVar[i, j]) + lMean[i, j]
        return out

def _rank2d(a, ky: int, kx: int, mask, rank_of_count: int):
    # generic zero-padded rank filter over an odd window; rank_of_count < 0 -> median
    R, C = a.shape
    out = np.empty((R, C), dtype=a.dtype)
    oy = ky // 2
    ox = kx // 2
    buf = List[float](ky * kx)
    for i in range(R):
        for j in range(C):
            buf.clear()
            for di in range(ky):
                for dj in range(kx):
                    if mask[di, dj]:
                        ii = i + di - oy
                        jj = j + dj - ox
                        if 0 <= ii < R and 0 <= jj < C:
                            buf.append(float(a[ii, jj]))
                        else:
                            buf.append(0.0)
            buf.sort()
            k = len(buf) // 2 if rank_of_count < 0 else rank_of_count
            out[i, j] = a.dtype(buf[k])
    return out

def medfilt2d(input, kernel_size = 3):
    a = np.asarray(input).astype(float)
    if a.ndim != 2:
        compile_error('medfilt2d: input must be 2-D')
    if isinstance(kernel_size, int):
        ky = kernel_size; kx = kernel_size
    else:
        ky = int(kernel_size[0]); kx = int(kernel_size[1])
    if ky % 2 != 1 or kx % 2 != 1:
        raise ValueError('Each element of kernel_size should be odd.')
    mask = np.ones((ky, kx), dtype=bool)
    return _rank2d(a, ky, kx, mask, -1)

def order_filter(a, domain, rank: int):
    dom = np.asarray(domain)
    for d in dom.shape:
        if d % 2 != 1:
            raise ValueError('Each dimension of domain argument should have an odd number of elements.')
    aa = np.asarray(a)
    if aa.ndim == 1:
        r = order_filter(aa.reshape((1, aa.size)), dom.reshape((1, dom.size)), rank)
        return r.ravel()
    elif aa.ndim != 2:
        compile_error('order_filter: only 1-D and 2-D input is supported')
    else:
        mask = dom != 0
        return _rank2d(aa, dom.shape[0], dom.shape[1], mask, rank)

def hilbert2(x, N = None):
    xa = np.atleast_2d(np.asarray(x)).astype(float)
    if N is None:
        n0 = xa.shape[0]; n1 = xa.shape[1]
    elif isinstance(N, int):
        if N <= 0:
            raise ValueError('N must be positive.')
        n0 = N; n1 = N
    else:
        n0 = int(N[0]); n1 = int(N[1])
        if n0 <= 0 or n1 <= 0:
            raise ValueError('When given as a tuple, N must hold exactly two positive integers')
    from scipy.fft import fft2, ifft2
    Xf = fft2(xa, (n0, n1))
    k0 = (n0 + 1) // 2
    k1 = (n1 + 1) // 2
    for i in range(n0):
        for j in range(n1):
            f = 1.0
            if i >= k0 or j >= k1:
                f = 0.0
            else:
                if 1 <= i < k0 and k0 > 1:
                    f *= 2.0
                if 1 <= j < k1 and k1 > 1:
                    f *= 2.0
            Xf[i, j] = Xf[i, j] * f
    return ifft2(Xf)

def correlation_lags(in1_len: int, in2_len: int, mode: str = 'full'):
    if mode == 'full' or mode == 'same':
        lags = np.arange(-in2_len + 1, in1_len)
        if mode == 'same':
            mid = lags.size // 2
            lb = in1_len // 2
            if in1_len % 2 == 0:
                return lags[mid - lb:mid + lb].copy()
            else:
                return lags[mid - lb:mid + lb + 1].copy()
        return lags
    elif mode == 'valid':
        lb = in1_len - in2_len
        if lb >= 0:
            return np.arange(lb + 1)
        else:
            return np.arange(lb, 1)
    else:
        raise ValueError(f'Mode {mode} is invalid')

def _prod(t) -> int:
    r = 1
    for v in t:
        r *= v
    return r

def choose_conv_method(in1, in2, mode: str = 'full', measure: bool = False) -> str:
    v = np.asarray(in1)
    k = np.asarray(in2)
    if v.dtype is bool or k.dtype is bool:
        return 'direct'
    if isinstance(v.dtype(), int) or isinstance(k.dtype(), int):
        mv = 0
        for e in v.flat:
            mv = max(mv, abs(int(e)))
        mk = 0
        for e in k.flat:
            mk = max(mk, abs(int(e)))
        if mv * mk * min(v.size, k.size) > 2 ** 52 - 1:
            return 'direct'
    # _conv_ops
    xs = v.shape
    hs = k.shape
    nd = len(xs)
    if mode == 'full':
        out_shape = [xs[i] + hs[i] - 1 for i in range(nd)]
    elif mode == 'valid':
        out_shape = [abs(xs[i] - hs[i]) + 1 for i in range(nd)]
    elif mode == 'same':
        out_shape = [xs[i] for i in range(nd)]
    else:
        raise ValueError("Acceptable mode flags are 'valid', 'same', or 'full'.")
    s1 = _prod(xs)
    s2 = _prod(hs)
    if nd == 1:
        if mode == 'full':
            direct_ops = s1 * s2
        elif mode == 'valid':
            direct_ops = (s2 - s1 + 1) * s1 if s2 >= s1 else (s1 - s2 + 1) * s2
        else:
            direct_ops = s1 * s2 if s1 < s2 else s1 * s2 - (s2 // 2) * ((s2 + 1) // 2)
    else:
        direct_ops = min(s1, s2) * _prod(out_shape)
    full_out = 1
    for i in range(nd):
        full_out *= xs[i] + hs[i] - 1
    N = float(full_out)
    fft_ops = 3 * N * math.log(N)
    offset = -0.001 if nd == 1 else -0.0001
    if nd == 1:
        if mode == 'valid':
            c = (1.89095737e-09, 2.1364985e-10, offset)
        elif mode == 'full':
            c = (1.764907e-09, 2.1414831e-10, offset)
        else:
            c = (3.2646654e-09, 2.8478277e-10, offset) if k.size <= v.size else (3.21635404e-09, 1.1773253e-08, -1e-05)
    else:
        if mode == 'valid':
            c = (1.85927e-09, 2.11242e-08, offset)
        elif mode == 'full':
            c = (1.99817e-09, 1.66174e-08, offset)
        else:
            c = (2.04735e-09, 1.55367e-08, offset)
    return 'fft' if c[0] * fft_ops < c[1] * direct_ops + c[2] else 'direct'

def vectorstrength(events, period):
    ev = np.atleast_1d(np.asarray(events)).astype(float)
    if isinstance(period, float) or isinstance(period, int):
        p = float(period)
        if p <= 0:
            raise ValueError('periods must be positive')
        s = complex()
        for e in ev:
            s += cmath.exp(complex(0.0, 2.0 * math.pi / p * e))
        s = s / ev.size
        return abs(s), math.atan2(s.imag, s.real)
    else:
        pa = np.atleast_1d(np.asarray(period)).astype(float)
        st = np.empty(pa.size)
        ph = np.empty(pa.size)
        for i in range(pa.size):
            if pa[i] <= 0:
                raise ValueError('periods must be positive')
            s = complex()
            for e in ev:
                s += cmath.exp(complex(0.0, 2.0 * math.pi / pa[i] * e))
            s = s / ev.size
            st[i] = abs(s)
            ph[i] = math.atan2(s.imag, s.real)
        return st, ph

# ---------------------------------------------------------------- waveforms

def _fmod_pos(t: float, m: float) -> float:
    r = math.fmod(t, m)
    if r != 0.0 and (r < 0) != (m < 0):
        r += m
    return r

def sawtooth(t, width = 1.0):
    ta = np.asarray(t).astype(float)
    out = np.empty(ta.shape)
    w = float(width)
    for i in range(ta.size):
        tv = ta.flat[i]
        if w > 1 or w < 0:
            out.flat[i] = math.nan
            continue
        tmod = _fmod_pos(tv, 2 * math.pi)
        if tmod < w * 2 * math.pi:
            out.flat[i] = tmod / (math.pi * w) - 1
        else:
            out.flat[i] = (math.pi * (w + 1) - tmod) / (math.pi * (1 - w))
    return out

def square(t, duty = 0.5):
    ta = np.asarray(t).astype(float)
    out = np.empty(ta.shape)
    w = float(duty)
    for i in range(ta.size):
        tv = ta.flat[i]
        if w > 1 or w < 0:
            out.flat[i] = math.nan
            continue
        tmod = _fmod_pos(tv, 2 * math.pi)
        out.flat[i] = 1.0 if tmod < w * 2 * math.pi else -1.0
    return out

def _gp_a(fc: float, bw: float, bwr: float) -> float:
    if fc < 0:
        raise ValueError(f'Center frequency (fc={fc:.2f}) must be >=0.')
    if bw <= 0:
        raise ValueError(f'Fractional bandwidth (bw={bw:.2f}) must be > 0.')
    if bwr >= 0:
        raise ValueError(f'Reference level for bandwidth (bwr={bwr:.2f}) must be < 0 dB')
    ref = 10.0 ** (bwr / 20.0)
    return -(math.pi * fc * bw) ** 2 / (4.0 * math.log(ref))

def gausspulse(t, fc: float = 1000, bw: float = 0.5, bwr: float = -6, tpr: float = -60,
               retquad: Literal[bool] = False, retenv: Literal[bool] = False):
    a = _gp_a(fc, bw, bwr)
    if isinstance(t, str):
        if t != 'cutoff':
            raise ValueError("If `t` is a string, it must be 'cutoff'")
        if tpr >= 0:
            raise ValueError('Reference level for time cutoff must be < 0 dB')
        tref = 10.0 ** (tpr / 20.0)
        return math.sqrt(-math.log(tref) / a)
    else:
        ta = np.asarray(t).astype(float)
        yenv = np.exp(-a * ta * ta)
        yI = yenv * np.cos(2 * math.pi * fc * ta)
        if retquad:
            yQ = yenv * np.sin(2 * math.pi * fc * ta)
            if retenv:
                return (yI, yQ, yenv)
            return (yI, yQ)
        elif retenv:
            return (yI, yenv)
        else:
            return yI

def _chirp_phase(t, f0: float, t1: float, f1: float, method: str, vertex_zero: bool):
    ta = np.asarray(t).astype(float)
    ph = np.empty(ta.shape)
    tp = 2 * math.pi
    if method in ('linear', 'lin', 'li'):
        beta = (f1 - f0) / t1
        for i in range(ta.size):
            tv = ta.flat[i]
            ph.flat[i] = tp * (f0 * tv + 0.5 * beta * tv * tv)
    elif method in ('quadratic', 'quad', 'q'):
        beta = (f1 - f0) / t1 ** 2
        for i in range(ta.size):
            tv = ta.flat[i]
            if vertex_zero:
                ph.flat[i] = tp * (f0 * tv + beta * tv ** 3 / 3)
            else:
                ph.flat[i] = tp * (f1 * tv + beta * ((t1 - tv) ** 3 - t1 ** 3) / 3)
    elif method in ('logarithmic', 'log', 'lo'):
        if f0 * f1 <= 0.0:
            raise ValueError('For a logarithmic chirp, f0 and f1 must be nonzero and have the same sign.')
        for i in range(ta.size):
            tv = ta.flat[i]
            if f0 == f1:
                ph.flat[i] = tp * f0 * tv
            else:
                beta = t1 / math.log(f1 / f0)
                ph.flat[i] = tp * beta * f0 * ((f1 / f0) ** (tv / t1) - 1.0)
    elif method in ('hyperbolic', 'hyp'):
        if f0 == 0 or f1 == 0:
            raise ValueError('For a hyperbolic chirp, f0 and f1 must be nonzero.')
        for i in range(ta.size):
            tv = ta.flat[i]
            if f0 == f1:
                ph.flat[i] = tp * f0 * tv
            else:
                sing = -f1 * t1 / (f0 - f1)
                ph.flat[i] = tp * (-sing * f0) * math.log(abs(1 - tv / sing))
    else:
        raise ValueError(f"method must be 'linear', 'quadratic', 'logarithmic', or 'hyperbolic', but a value of '{method}' was given.")
    return ph

def chirp(t, f0: float, t1: float, f1: float, method: str = 'linear', phi: float = 0,
          vertex_zero: bool = True, *, complex: Literal[bool] = False):
    phase = _chirp_phase(t, f0, t1, f1, method, vertex_zero) + phi * math.pi / 180.0
    if complex:
        return np.exp(phase.astype(type(1j)) * 1j)
    else:
        return np.cos(phase)

def sweep_poly(t, poly, phi: float = 0):
    ip = np.polyint(np.asarray(poly).astype(float))
    phase = 2 * math.pi * np.polyval(ip, np.asarray(t).astype(float))
    return np.cos(phase + phi * math.pi / 180)

def unit_impulse(shape, idx = None, dtype: type = float):
    if isinstance(shape, int):
        out = np.zeros(shape, dtype=dtype)
        if idx is None:
            out[0] = dtype(1)
        elif isinstance(idx, str):
            if idx != 'mid':
                raise ValueError("idx must be None, 'mid', an int or a tuple")
            out[shape // 2] = dtype(1)
        else:
            out[idx] = dtype(1)
        return out
    else:
        out = np.zeros(shape, dtype=dtype)
        if idx is None:
            out.flat[0] = dtype(1)
        elif isinstance(idx, str):
            mid = tuple(s // 2 for s in shape)
            out[mid] = dtype(1)
        elif isinstance(idx, int):
            ii = tuple(idx for _ in shape)
            out[ii] = dtype(1)
        else:
            out[idx] = dtype(1)
        return out

# ---------------------------------------------------------------- chirp-z

def czt_points(m: int, w = None, a = complex(1.0, 0.0)):
    if m < 1:
        raise ValueError(f'Invalid number of CZT output points ({m}) specified. m must be positive and integer type.')
    av = complex(a)
    out = np.empty(m, dtype=complex)
    for k in range(m):
        if w is None:
            out[k] = av * cmath.exp(complex(0.0, 2.0 * math.pi * k / m))
        else:
            out[k] = av * complex(w) ** (-k)
    return out

class CZT:
    m: int
    n: int
    w: complex
    a: complex
    _Awk2: np.ndarray[complex, 1]
    _Fwk2: np.ndarray[complex, 1]
    _wk2: np.ndarray[complex, 1]
    _nfft: int

    def __init__(self, n: int, m: Optional[int] = None, w = None, a = complex(1.0, 0.0)):
        if n < 1:
            raise ValueError(f'Invalid number of CZT data points ({n}) specified. n must be positive and integer type.')
        mm = n if m is None else m
        if mm < 1:
            raise ValueError(f'Invalid number of CZT output points ({mm}) specified. m must be positive and integer type.')
        L = max(mm, n)
        wk2 = np.empty(L, dtype=complex)
        if w is None:
            ww = cmath.exp(complex(0.0, -2.0 * math.pi / mm))
            for k in range(L):
                wk2[k] = cmath.exp(-(complex(0.0, 1.0) * math.pi * float((k * k) % (2 * mm))) / mm)
        else:
            ww = complex(w)
            for k in range(L):
                wk2[k] = ww ** (k * k / 2.0)
        av = complex(a)
        self.w = ww
        self.a = av
        self.m = mm
        self.n = n
        from scipy.fft import next_fast_len
        nfft = next_fast_len(n + mm - 1)
        self._nfft = nfft
        A = np.empty(n, dtype=complex)
        for k in range(n):
            A[k] = av ** (-k) * wk2[k]
        self._Awk2 = A
        st = np.empty(n - 1 + mm, dtype=complex)
        for i in range(n - 1):
            st[i] = 1.0 / wk2[n - 1 - i]
        for i in range(mm):
            st[n - 1 + i] = 1.0 / wk2[i]
        self._Fwk2 = np.fft.fft(st, nfft)
        self._wk2 = wk2[:mm].copy()

    def __call__(self, x, *, axis: int = -1):
        xa = np.asarray(x).astype(complex)
        if xa.ndim != 1:
            compile_error('CZT: only 1-D input is supported')
        if xa.shape[0] != self.n:
            raise ValueError(f'CZT defined for length {self.n}, not {xa.shape[0]}')
        y = np.fft.ifft(self._Fwk2 * np.fft.fft(xa * self._Awk2, self._nfft))
        out = np.empty(self.m, dtype=complex)
        for i in range(self.m):
            out[i] = y[self.n - 1 + i] * self._wk2[i]
        return out

    def points(self):
        return czt_points(self.m, self.w, self.a)

def ZoomFFT(n: int, fn, m: Optional[int] = None, *, fs: float = 2, endpoint: bool = False):
    mm = n if m is None else m
    if isinstance(fn, float) or isinstance(fn, int):
        f1 = 0.0; f2 = float(fn)
    else:
        f1 = float(fn[0]); f2 = float(fn[1])
    scale = (f2 - f1) * mm / (fs * (mm - 1)) if endpoint else (f2 - f1) / fs
    t = CZT(n, mm)
    L = max(mm, n)
    wk2 = np.empty(L, dtype=complex)
    for k in range(L):
        wk2[k] = cmath.exp(-(complex(0.0, 1.0) * math.pi * scale * float(k * k)) / mm)
    t.w = cmath.exp(complex(0.0, -2.0 * math.pi / mm * scale))
    t.a = cmath.exp(complex(0.0, 2.0 * math.pi * f1 / fs))
    A = np.empty(n, dtype=complex)
    for k in range(n):
        A[k] = cmath.exp(complex(0.0, -2.0 * math.pi * f1 / fs * k)) * wk2[k]
    t._Awk2 = A
    st = np.empty(n - 1 + mm, dtype=complex)
    for i in range(n - 1):
        st[i] = 1.0 / wk2[n - 1 - i]
    for i in range(mm):
        st[n - 1 + i] = 1.0 / wk2[i]
    t._Fwk2 = np.fft.fft(st, t._nfft)
    t._wk2 = wk2[:mm].copy()
    return t

def czt(x, m: Optional[int] = None, w = None, a = complex(1.0, 0.0), *, axis: int = -1):
    xa = np.asarray(x)
    return CZT(xa.shape[xa.ndim - 1], m, w, a)(xa)

def zoom_fft(x, fn, m: Optional[int] = None, *, fs: float = 2, endpoint: bool = False, axis: int = -1):
    xa = np.asarray(x)
    return ZoomFFT(xa.shape[xa.ndim - 1], fn, m, fs=fs, endpoint=endpoint)(xa)

# ---------------------------------------------------------------- Lomb-Scargle

def lombscargle(x, y, freqs, *, normalize = False, weights = None, floating_mean: bool = False):
    xa = np.asarray(x).astype(float)
    ya = np.asarray(y).astype(float)
    fa = np.asarray(freqs).astype(float)
    n = xa.size
    if not (xa.ndim == 1 and n > 0 and ya.size == n):
        raise ValueError('Parameters x, y, weights must be 1-D arrays of equal non-zero length!')
    if not (fa.ndim == 1 and fa.size > 0):
        raise ValueError('Parameter freqs must be a 1-D array of non-zero length!')
    wa = np.ones(n) if weights is None else np.asarray(weights).astype(float)
    if wa.size != n:
        raise ValueError('Parameters x, y, weights must be 1-D arrays of equal non-zero length!')
    if isinstance(normalize, bool):
        mode = 'normalize' if normalize else 'power'
    else:
        mode = normalize
    if mode != 'power' and mode != 'normalize' and mode != 'amplitude':
        raise ValueError("Normalize must be: False (or 'power'), True (or 'normalize'), or 'amplitude'.")
    ws = 0.0
    for v in wa:
        if v < 0:
            raise ValueError('Parameter weights must have only non-negative entries which sum to a positive value!')
        ws += v
    if ws <= 0:
        raise ValueError('Parameter weights must have only non-negative entries which sum to a positive value!')
    w = wa * (1.0 / ws)
    nf = fa.size
    pg = np.empty(nf)
    pgc = np.empty(nf, dtype=complex)
    epsneg = 1.1102230246251565e-16
    Y = 0.0
    YY = 0.0
    for i in range(n):
        Y += w[i] * ya[i]
        YY += w[i] * ya[i] * ya[i]
    for k in range(nf):
        f = fa[k]
        CC = 0.0; CS = 0.0; C = 0.0; S = 0.0
        for i in range(n):
            c = math.cos(f * xa[i]); s = math.sin(f * xa[i])
            CC += w[i] * c * c
            CS += w[i] * c * s
            C += w[i] * c
            S += w[i] * s
        SS = 1.0 - CC
        if floating_mean:
            CC -= C * C
            SS -= S * S
            CS -= C * S
        tau = 0.5 * math.atan2(2.0 * CS, CC - SS)
        YC = 0.0; YS = 0.0; CC2 = 0.0; C2 = 0.0; S2 = 0.0
        for i in range(n):
            ft = f * xa[i] - tau
            c = math.cos(ft); s = math.sin(ft)
            YC += w[i] * ya[i] * c
            YS += w[i] * ya[i] * s
            CC2 += w[i] * c * c
            C2 += w[i] * c
            S2 += w[i] * s
        SS2 = 1.0 - CC2
        if floating_mean:
            YC -= Y * C2
            YS -= Y * S2
            CC2 -= C2 * C2
            SS2 -= S2 * S2
        if CC2 < epsneg:
            CC2 = epsneg
        if SS2 < epsneg:
            SS2 = epsneg
        a = YC / CC2
        b = YS / SS2
        p = 2.0 * (a * YC + b * YS)
        if mode == 'power':
            pg[k] = p * float(n) / 4.0
        elif mode == 'normalize':
            yy = YY - Y * Y if floating_mean else YY
            pg[k] = p * 0.5 / yy
        else:
            pgc[k] = complex(a, b) * cmath.exp(complex(0.0, tau))
    if mode == 'amplitude':
        # scipy returns complex amplitudes; the static return type here is
        # float, so the magnitude is returned (deviation).
        for k in range(nf):
            pg[k] = abs(pgc[k])
    return pg
