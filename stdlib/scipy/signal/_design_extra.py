# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Transfer-function transforms and analog responses: lp2lp, lp2hp, lp2bp,
# lp2bs, bilinear (b, a form), findfreqs, freqs, freqs_zpk, cheb2ord,
# iirdesign, iircomb.

import numpy as np
import math
import cmath
from ._filter_design import normalize, _flist, _farr, _carr, _clist, _roots, \
    _ord_setup, _post_wn, _validate_gpass_gstop, buttord, cheb1ord, ellipord, iirfilter

def _comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return 0.0
    r = 1
    kk = min(k, n - k)
    for i in range(1, kk + 1):
        r = r * (n - kk + i) // i
    return float(r)

def lp2lp(b, a, wo: float = 1.0):
    bl = _flist(b)
    al = _flist(a)
    d = len(al)
    n = len(bl)
    M = max(d, n)
    pwo = [wo ** float(M - 1 - i) for i in range(M)]
    start1 = max(n - d, 0)
    start2 = max(d - n, 0)
    bo = [bl[i] * pwo[start1] / pwo[start2 + i] for i in range(n)]
    ao = [al[i] * pwo[start1] / pwo[start1 + i] for i in range(d)]
    return normalize(_farr(bo), _farr(ao))

def lp2hp(b, a, wo: float = 1.0):
    bl = _flist(b)
    al = _flist(a)
    d = len(al)
    n = len(bl)
    M = max(d, n)
    pwo = [wo ** float(i) if wo != 1 else 1.0 for i in range(M)]
    if d >= n:
        outa = [al[d - 1 - i] * pwo[i] for i in range(d)]
        outb = [0.0] * d
        for i in range(n):
            outb[i] = bl[n - 1 - i] * pwo[i]
    else:
        outb = [bl[n - 1 - i] * pwo[i] for i in range(n)]
        outa = [0.0] * n
        for i in range(d):
            outa[i] = al[d - 1 - i] * pwo[i]
    return normalize(_farr(outb), _farr(outa))

def lp2bp(b, a, wo: float = 1.0, bw: float = 1.0):
    bl = _flist(b)
    al = _flist(a)
    D = len(al) - 1
    N = len(bl) - 1
    ma = max(N, D)
    Np = N + ma
    Dp = D + ma
    bp = [0.0] * (Np + 1)
    ap = [0.0] * (Dp + 1)
    wosq = wo * wo
    for j in range(Np + 1):
        val = 0.0
        for i in range(N + 1):
            for k in range(i + 1):
                if ma - i + 2 * k == j:
                    val += _comb(i, k) * bl[N - i] * wosq ** (i - k) / bw ** i
        bp[Np - j] = val
    for j in range(Dp + 1):
        val = 0.0
        for i in range(D + 1):
            for k in range(i + 1):
                if ma - i + 2 * k == j:
                    val += _comb(i, k) * al[D - i] * wosq ** (i - k) / bw ** i
        ap[Dp - j] = val
    return normalize(_farr(bp), _farr(ap))

def lp2bs(b, a, wo: float = 1.0, bw: float = 1.0):
    bl = _flist(b)
    al = _flist(a)
    D = len(al) - 1
    N = len(bl) - 1
    M = max(N, D)
    Np = 2 * M
    Dp = 2 * M
    bp = [0.0] * (Np + 1)
    ap = [0.0] * (Dp + 1)
    wosq = wo * wo
    for j in range(Np + 1):
        val = 0.0
        for i in range(N + 1):
            for k in range(M - i + 1):
                if i + 2 * k == j:
                    val += _comb(M - i, k) * bl[N - i] * wosq ** (M - i - k) * bw ** i
        bp[Np - j] = val
    for j in range(Dp + 1):
        val = 0.0
        for i in range(D + 1):
            for k in range(M - i + 1):
                if i + 2 * k == j:
                    val += _comb(M - i, k) * al[D - i] * wosq ** (M - i - k) * bw ** i
        ap[Dp - j] = val
    return normalize(_farr(bp), _farr(ap))

def _ppow(p: List[float], n: int) -> List[float]:
    r = [1.0]
    for _ in range(n):
        out = [0.0] * (len(r) + len(p) - 1)
        for i in range(len(r)):
            for j in range(len(p)):
                out[i + j] += r[i] * p[j]
        r = out
    return r

def _pmul(a: List[float], b: List[float]) -> List[float]:
    out = [0.0] * (len(a) + len(b) - 1)
    for i in range(len(a)):
        for j in range(len(b)):
            out[i + j] += a[i] * b[j]
    return out

def _trim_f(x: List[float]) -> List[float]:
    i = 0
    while i < len(x) and x[i] == 0.0:
        i += 1
    return x[i:]

def bilinear(b, a, fs: float = 1.0):
    bl = _trim_f(_flist(b))
    al = _trim_f(_flist(a))
    fac = math.sqrt(fs * 2)
    # np.polynomial (ascending) coefficients
    zp1 = [1.0 / fac, 1.0 / fac]
    zm1 = [-1.0 * fac, 1.0 * fac]
    N = max(len(al), len(bl)) - 1
    def acc(c: List[float]) -> List[float]:
        tot = [0.0]
        cr = c[::-1]
        for q in range(len(cr)):
            term = _pmul(_ppow(zp1, N - q), _ppow(zm1, q))
            term = [cr[q] * v for v in term]
            if len(term) > len(tot):
                tot = tot + [0.0] * (len(term) - len(tot))
            for i in range(len(term)):
                tot[i] += term[i]
        # Polynomial arithmetic trims trailing zeros of the highest degree
        while len(tot) > 1 and tot[-1] == 0.0:
            tot = tot[:-1]
        return tot[::-1]
    return normalize(_farr(acc(bl)), _farr(acc(al)))

# ---------------------------------------------------------------- analog response

def _logspace(start: float, stop: float, num: int):
    w = np.empty(num, dtype=float)
    for i in range(num):
        e = start + (stop - start) * i / (num - 1) if num > 1 else start
        w[i] = 10.0 ** e
    return w

def _findfreqs_c(ep0: List[complex], tz: List[complex], N: int):
    ep = ep0 if len(ep0) > 0 else [complex(-1000.0, 0.0)]
    ez: List[complex] = [v for v in ep if v.imag >= 0]
    for v in tz:
        if abs(v) < 100000.0 and v.imag >= 0:
            ez.append(v)
    mx = -math.inf
    mn = math.inf
    for v in ez:
        integ = 1.0 if abs(v) < 1e-10 else 0.0
        mx = max(mx, 3 * abs(v.real + integ) + 1.5 * v.imag)
        mn = min(mn, abs((v + integ).real) + 2 * v.imag)
    hfreq = float(round(math.log10(mx) + 0.5))
    lfreq = float(round(math.log10(0.1 * mn) - 0.5))
    return _logspace(lfreq, hfreq, N)

def findfreqs(num, den, N: int, kind: str = 'ba'):
    if kind == 'ba':
        return _findfreqs_c(_roots(_flist(den)), _roots(_flist(num)), N)
    elif kind == 'zp':
        return _findfreqs_c(_clist(den), _clist(num), N)
    else:
        raise ValueError("input must be one of {'ba', 'zp'}")

def _polyval_c(c: List[float], x: complex) -> complex:
    r = complex(0.0)
    for v in c:
        r = r * x + v
    return r

def freqs(b, a, worN = 200, plot = None):
    bl = _flist(b)
    al = _flist(a)
    if isinstance(worN, int):
        w = findfreqs(b, a, worN)
    elif worN is None:
        w = findfreqs(b, a, 200)
    else:
        w = _farr(_flist(worN))
    h = np.empty(w.size, dtype=complex)
    for i in range(w.size):
        s = complex(0.0, w[i])
        h[i] = _polyval_c(bl, s) / _polyval_c(al, s)
    return w, h

def freqs_zpk(z, p, k: float, worN = 200):
    zl = _clist(z)
    pl = _clist(p)
    if isinstance(worN, int):
        w = _findfreqs_c(pl, zl, worN)
    elif worN is None:
        w = _findfreqs_c(pl, zl, 200)
    else:
        w = _farr(_flist(worN))
    h = np.empty(w.size, dtype=complex)
    for i in range(w.size):
        s = complex(0.0, w[i])
        num = complex(1.0)
        for v in zl:
            num *= s - v
        den = complex(1.0)
        for v in pl:
            den *= s - v
        h[i] = k * num / den
    return w, h

# ---------------------------------------------------------------- order selection

def cheb2ord(wp, ws, gpass: float, gstop: float, analog: bool = False, fs: Optional[float] = None):
    ftype, passb, nat = _ord_setup(wp, ws, gpass, gstop, analog, fs, 'cheby')
    GSTOP = 10 ** (0.1 * abs(gstop))
    GPASS = 10 ** (0.1 * abs(gpass))
    v_pass_stop = math.acosh(math.sqrt((GSTOP - 1.0) / (GPASS - 1.0)))
    ord = int(math.ceil(v_pass_stop / math.acosh(nat)))
    new_freq = 1.0 / math.cosh(1.0 / ord * v_pass_stop)
    if ftype == 1:
        WN = [v / new_freq for v in passb]
    elif ftype == 2:
        WN = [v * new_freq for v in passb]
    elif ftype == 3:
        nat0 = new_freq / 2.0 * (passb[0] - passb[1]) + math.sqrt(new_freq ** 2 * (passb[1] - passb[0]) ** 2 / 4.0 + passb[1] * passb[0])
        WN = [nat0, passb[1] * passb[0] / nat0]
    else:
        nat0 = 1.0 / (2.0 * new_freq) * (passb[0] - passb[1]) + math.sqrt((passb[1] - passb[0]) ** 2 / (4.0 * new_freq ** 2) + passb[1] * passb[0])
        WN = [nat0, passb[0] * passb[1] / nat0]
    return ord, _post_wn(WN, analog, fs, wp)

def _ord_any(ftype: str, wp, ws, gpass: float, gstop: float, analog: bool, fs: Optional[float]):
    if ftype in ('butter', 'butterworth'):
        return buttord(wp, ws, gpass, gstop, analog, fs)
    elif ftype in ('cauer', 'elliptic', 'ellip'):
        return ellipord(wp, ws, gpass, gstop, analog, fs)
    elif ftype in ('cheby1', 'chebyshev1', 'chebyshevi'):
        return cheb1ord(wp, ws, gpass, gstop, analog, fs)
    elif ftype in ('cheby2', 'chebyshev2', 'chebyshevii'):
        return cheb2ord(wp, ws, gpass, gstop, analog, fs)
    elif ftype in ('bessel', 'bessel_phase', 'bessel_delay', 'bessel_mag'):
        raise ValueError(f'{ftype} does not have order selection. Use iirfilter function.')
    else:
        raise ValueError(f'Invalid IIR filter type: {ftype}')

def iirdesign(wp, ws, gpass: float, gstop: float, analog: bool = False, ftype: str = 'ellip',
              output: Literal[str] = 'ba', fs: Optional[float] = None):
    _validate_gpass_gstop(gpass, gstop)
    wpl = _flist(wp)
    wsl = _flist(ws)
    if len(wpl) != len(wsl) or (len(wpl) != 1 and len(wpl) != 2):
        raise ValueError('wp and ws must have one or two elements each, and the same shape')
    for v in wpl + wsl:
        if v <= 0:
            raise ValueError('Values for wp, ws must be greater than 0')
    if not analog:
        if fs is None:
            for v in wpl + wsl:
                if v >= 1:
                    raise ValueError('Values for wp, ws must be less than 1')
        else:
            for v in wpl + wsl:
                if v >= fs / 2:
                    raise ValueError(f'Values for wp, ws must be less than fs/2 (fs={fs} -> fs/2={fs / 2})')
    if len(wpl) == 2:
        if not ((wsl[0] < wpl[0] and wpl[1] < wsl[1]) or (wpl[0] < wsl[0] and wsl[1] < wpl[1])):
            raise ValueError('Passband must lie strictly inside stopband or vice versa')
    band_type = 2 * (len(wpl) - 1) + 1
    if wpl[0] >= wsl[0]:
        band_type += 1
    btype = ('lowpass', 'highpass', 'bandstop', 'bandpass')[band_type - 1]
    N, Wn = _ord_any(ftype, _farr(wpl), _farr(wsl), gpass, gstop, analog, fs)
    return iirfilter(N, Wn, rp=gpass, rs=gstop, analog=analog, btype=btype, ftype=ftype,
                     output=output, fs=fs)

def iircomb(w0: float, Q: float, ftype: str = 'notch', fs: float = 2.0, *, pass_zero: bool = False):
    ft = ftype.lower()
    if not (0 < w0 < fs / 2):
        raise ValueError(f'w0 must be between 0 and {fs / 2} (Nyquist), but given {w0}.')
    if ft != 'notch' and ft != 'peak':
        raise ValueError('ftype must be either notch or peak.')
    N = int(round(fs / w0))
    if abs(w0 - fs / N) / fs > 1e-14:
        raise ValueError('fs must be divisible by w0.')
    w0r = 2 * math.pi * w0 / fs
    w_delta = w0r / Q
    G0, G = (1.0, 0.0) if ft == 'notch' else (0.0, 1.0)
    beta = math.tan(N * w_delta / 4)
    ax = (1 - beta) / (1 + beta)
    bx = (G0 + G * beta) / (1 + beta)
    cx = (G0 - G * beta) / (1 + beta)
    negative = (ft == 'peak' and pass_zero) or (ft == 'notch' and not pass_zero)
    sgn = -1.0 if negative else 1.0
    b = np.zeros(N + 1)
    a = np.zeros(N + 1)
    b[0] = bx
    b[N] = sgn * cx
    a[0] = 1.0
    a[N] = sgn * ax
    return b, a
