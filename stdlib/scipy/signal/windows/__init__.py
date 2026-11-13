# Ported from scipy 1.18.1 (BSD-3-Clause)

import numpy as np
import internal.static as static
import math
import cmath
from math import pi as _PI, cos as _cos, sin as _sin, exp as _exp, sqrt as _sqrt, floor as _floor

def _i0(x: float) -> float:
    return np.i0(np.array([x]))[0]

def _len_guards(M: int):
    if M < 0:
        raise ValueError('Window length M must be a non-negative integer')
    return M <= 1

def _ones(M: int):
    return np.ones(M, dtype=float)

def _general_cosine_impl(M: int, a: List[float], sym: bool):
    if _len_guards(M):
        return _ones(M)
    Me = M if sym else M + 1
    w = np.zeros(Me, dtype=float)
    fac = np.linspace(-_PI, _PI, Me)
    for i in range(Me):
        s = 0.0
        for k in range(len(a)):
            s += a[k] * _cos(k * fac[i])
        w[i] = s
    return w if sym else w[:M].copy()

def general_cosine(M: int, a, sym: bool = True):
    return _general_cosine_impl(M, [float(v) for v in a], sym)

def general_hamming(M: int, alpha: float, sym: bool = True):
    return _general_cosine_impl(M, [alpha, 1.0 - alpha], sym)

def hann(M: int, sym: bool = True):
    return general_hamming(M, 0.5, sym)

def hamming(M: int, sym: bool = True):
    return general_hamming(M, 0.54, sym)

def blackman(M: int, sym: bool = True):
    return _general_cosine_impl(M, [0.42, 0.5, 0.08], sym)

def blackmanharris(M: int, sym: bool = True):
    return _general_cosine_impl(M, [0.35875, 0.48829, 0.14128, 0.01168], sym)

def nuttall(M: int, sym: bool = True):
    return _general_cosine_impl(M, [0.3635819, 0.4891775, 0.1365995, 0.0106411], sym)

def flattop(M: int, sym: bool = True):
    return _general_cosine_impl(M, [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368], sym)

def _finish(w, M: int, sym: bool):
    return w if sym else w[:M].copy()

def boxcar(M: int, sym: bool = True):
    return _ones(M)

def triang(M: int, sym: bool = True):
    if _len_guards(M):
        return _ones(M)
    Me = M if sym else M + 1
    w = np.empty(Me, dtype=float)
    half = (Me + 1) // 2
    for i in range(half):
        n = float(i + 1)
        v = (2 * n - 1.0) / Me if Me % 2 == 0 else 2 * n / (Me + 1.0)
        w[i] = v
        w[Me - 1 - i] = v
    return _finish(w, M, sym)

def bartlett(M: int, sym: bool = True):
    if _len_guards(M):
        return _ones(M)
    Me = M if sym else M + 1
    w = np.empty(Me, dtype=float)
    for i in range(Me):
        n = float(i)
        w[i] = 2.0 * n / (Me - 1) if n <= (Me - 1) / 2.0 else 2.0 - 2.0 * n / (Me - 1)
    return _finish(w, M, sym)

def cosine(M: int, sym: bool = True):
    if _len_guards(M):
        return _ones(M)
    Me = M if sym else M + 1
    w = np.empty(Me, dtype=float)
    for i in range(Me):
        w[i] = _sin(_PI / Me * (i + 0.5))
    return _finish(w, M, sym)

def tukey(M: int, alpha: float = 0.5, sym: bool = True):
    if _len_guards(M):
        return _ones(M)
    if alpha <= 0:
        return _ones(M)
    elif alpha >= 1.0:
        return hann(M, sym)
    Me = M if sym else M + 1
    width = int(_floor(alpha * (Me - 1) / 2.0))
    w = np.ones(Me, dtype=float)
    for i in range(Me):
        n = float(i)
        if i <= width:
            w[i] = 0.5 * (1 + _cos(_PI * (-1 + 2.0 * n / alpha / (Me - 1))))
        elif i >= Me - width - 1:
            w[i] = 0.5 * (1 + _cos(_PI * (-2.0 / alpha + 1 + 2.0 * n / alpha / (Me - 1))))
    return _finish(w, M, sym)

def kaiser(M: int, beta: float, sym: bool = True):
    if _len_guards(M):
        return _ones(M)
    Me = M if sym else M + 1
    alpha = (Me - 1) / 2.0
    arg = np.empty(Me, dtype=float)
    for i in range(Me):
        r = (i - alpha) / alpha
        arg[i] = beta * _sqrt(1 - r ** 2.0)
    w = np.i0(arg) / _i0(beta)
    return _finish(w, M, sym)

def gaussian(M: int, std: float, sym: bool = True):
    if _len_guards(M):
        return _ones(M)
    Me = M if sym else M + 1
    sig2 = 2 * std * std
    w = np.empty(Me, dtype=float)
    for i in range(Me):
        n = i - (Me - 1.0) / 2.0
        w[i] = _exp(-n ** 2 / sig2)
    return _finish(w, M, sym)

def exponential(M: int, center: Optional[float] = None, tau: float = 1.0, sym: bool = True):
    if sym and center is not None:
        raise ValueError("If sym==True, center must be None.")
    if _len_guards(M):
        return _ones(M)
    Me = M if sym else M + 1
    c = (Me - 1) / 2 if center is None else center
    w = np.empty(Me, dtype=float)
    for i in range(Me):
        w[i] = _exp(-abs(i - c) / tau)
    return _finish(w, M, sym)

def parzen(M: int, sym: bool = True):
    if _len_guards(M):
        return _ones(M)
    Me = M if sym else M + 1
    w = np.empty(Me, dtype=float)
    start = -(Me - 1) / 2.0
    for i in range(Me):
        n = start + i
        an = abs(n)
        if an <= (Me - 1) / 4.0:
            w[i] = 1 - 6 * (an / (Me / 2.0)) ** 2.0 + 6 * (an / (Me / 2.0)) ** 3.0
        else:
            w[i] = 2 * (1 - an / (Me / 2.0)) ** 3.0
    return _finish(w, M, sym)

def bohman(M: int, sym: bool = True):
    if _len_guards(M):
        return _ones(M)
    Me = M if sym else M + 1
    lin = np.linspace(-1.0, 1.0, Me)
    w = np.zeros(Me, dtype=float)
    for i in range(1, Me - 1):
        fac = abs(lin[i])
        w[i] = (1 - fac) * _cos(_PI * fac) + 1.0 / _PI * _sin(_PI * fac)
    return _finish(w, M, sym)

def barthann(M: int, sym: bool = True):
    if _len_guards(M):
        return _ones(M)
    Me = M if sym else M + 1
    w = np.empty(Me, dtype=float)
    for i in range(Me):
        fac = abs(i / (Me - 1.0) - 0.5)
        w[i] = 0.62 - 0.48 * fac + 0.38 * _cos(2 * _PI * fac)
    return _finish(w, M, sym)

def general_gaussian(M: int, p: float, sig: float, sym: bool = True):
    if _len_guards(M):
        return _ones(M)
    Me = M if sym else M + 1
    w = np.empty(Me, dtype=float)
    for i in range(Me):
        n = i - (Me - 1.0) / 2.0
        w[i] = _exp(-0.5 * abs(n / sig) ** (2 * p))
    return _finish(w, M, sym)

def kaiser_bessel_derived(M: int, beta: float, *, sym: bool = True):
    if not sym:
        raise ValueError('Kaiser-Bessel Derived windows are only defined for symmetric shapes')
    elif M < 1:
        return np.empty(0, dtype=float)
    elif M % 2:
        raise ValueError('Kaiser-Bessel Derived windows are only defined for even number of points')
    kw = kaiser(M // 2 + 1, beta)
    h = M // 2
    csum = np.cumsum(kw)
    w = np.empty(M, dtype=float)
    tot = csum[h]
    for i in range(h):
        v = _sqrt(csum[i] / tot)
        w[i] = v
        w[M - 1 - i] = v
    return w

def chebwin(M: int, at: float, sym: bool = True):
    if abs(at) < 45:
        print('UserWarning: This window is not suitable for spectral analysis for attenuation values lower than about 45dB because the equivalent noise bandwidth of a Chebyshev window does not grow monotonically with increasing sidelobe attenuation when the attenuation is smaller than about 45 dB.')
    if _len_guards(M):
        return _ones(M)
    Me = M if sym else M + 1
    order = Me - 1.0
    beta = math.cosh(1.0 / order * math.acosh(10 ** (abs(at) / 20.0)))
    p = np.empty(Me, dtype=complex)
    for k in range(Me):
        x = beta * _cos(_PI * k / Me)
        if x > 1:
            v = math.cosh(order * math.acosh(x))
        elif x < -1:
            v = (2 * (Me % 2) - 1) * math.cosh(order * math.acosh(-x))
        else:
            v = _cos(order * math.acos(x))
        if Me % 2:
            p[k] = complex(v, 0.0)
        else:
            p[k] = v * cmath.exp(complex(0.0, _PI / Me * k))
    F = np.fft.fft(p)
    if Me % 2:
        n = (Me + 1) // 2
        w = np.empty(2 * n - 1, dtype=float)
        for i in range(n):
            w[n - 1 + i] = F[i].real
        for i in range(1, n):
            w[n - 1 - i] = F[i].real
    else:
        n = Me // 2 + 1
        w = np.empty(2 * (n - 1), dtype=float)
        for i in range(1, n):
            w[n - 2 + i] = F[i].real
            w[n - 1 - i] = F[i].real
    mx = -math.inf
    for v in w:
        mx = max(mx, v)
    for i in range(w.size):
        w[i] = w[i] / mx
    return _finish(w, M, sym)

def taylor(M: int, nbar: int = 4, sll: float = 30, norm: bool = True, sym: bool = True):
    if _len_guards(M):
        return _ones(M)
    Me = M if sym else M + 1
    B = 10 ** (sll / 20)
    A = math.acosh(B) / _PI
    s2 = nbar ** 2 / (A ** 2 + (nbar - 0.5) ** 2)
    ma = [float(i) for i in range(1, nbar)]
    m2 = [v * v for v in ma]
    Fm = [0.0 for _ in range(nbar - 1)]
    for mi in range(nbar - 1):
        sign = 1.0 if mi % 2 == 0 else -1.0
        numer = 1.0
        for v in ma:
            numer *= 1 - m2[mi] / s2 / (A ** 2 + (v - 0.5) ** 2)
        numer *= sign
        denom = 1.0
        for j in range(mi):
            denom *= 1 - m2[mi] / m2[j]
        for j in range(mi + 1, nbar - 1):
            denom *= 1 - m2[mi] / m2[j]
        Fm[mi] = numer / (2 * denom)
    def W(n: float) -> float:
        s = 0.0
        for k in range(nbar - 1):
            s += Fm[k] * _cos(2 * _PI * ma[k] * (n - Me / 2.0 + 0.5) / Me)
        return 1 + 2 * s
    w = np.empty(Me, dtype=float)
    for i in range(Me):
        w[i] = W(float(i))
    if norm:
        scale = 1.0 / W((Me - 1) / 2)
        for i in range(Me):
            w[i] *= scale
    return _finish(w, M, sym)

def _sinc(x: float) -> float:
    if x == 0.0:
        return 1.0
    y = _PI * x
    return _sin(y) / y

def lanczos(M: int, *, sym: bool = True):
    if _len_guards(M):
        return _ones(M)
    Me = M if sym else M + 1
    w = np.empty(Me, dtype=float)
    if Me % 2 == 0:
        h = Me // 2
        for k in range(h):
            v = _sinc(2.0 * (h + k) / (Me - 1) - 1.0)
            w[h + k] = v
            w[h - 1 - k] = v
    else:
        h = (Me + 1) // 2
        # right side: n in arange((M+1)/2, M)
        for k in range(Me - h):
            v = _sinc(2.0 * (h + k) / (Me - 1) - 1.0)
            w[h + k] = v
            w[Me - h - 1 - k] = v
        w[Me - h] = 1.0
    return _finish(w, M, sym)

def _by_name(name: str, Nx: int, sym: bool, args: List[float]):
    def need(n: int):
        if len(args) != n:
            raise ValueError(f"'{name}' requires {n} parameter(s), got {len(args)}")
    if name in ('hann', 'han'):
        need(0); return hann(Nx, sym)
    if name in ('hamming', 'hamm', 'ham'):
        need(0); return hamming(Nx, sym)
    if name in ('blackman', 'black', 'blk'):
        need(0); return blackman(Nx, sym)
    if name in ('blackmanharris', 'blackharr', 'bkh'):
        need(0); return blackmanharris(Nx, sym)
    if name in ('nuttall', 'nutl', 'nut'):
        need(0); return nuttall(Nx, sym)
    if name in ('boxcar', 'box', 'ones', 'rect', 'rectangular'):
        need(0); return boxcar(Nx, sym)
    if name in ('bartlett', 'bart', 'brt'):
        need(0); return bartlett(Nx, sym)
    if name in ('triang', 'triangle', 'tri'):
        need(0); return triang(Nx, sym)
    if name in ('flattop', 'flat', 'flt'):
        need(0); return flattop(Nx, sym)
    if name in ('cosine', 'halfcosine'):
        need(0); return cosine(Nx, sym)
    if name in ('tukey', 'tuk'):
        if len(args) == 0:
            raise ValueError(f"'{name}' must have parameters")
        return tukey(Nx, args[0], sym)
    if name in ('kaiser', 'ksr'):
        need(1); return kaiser(Nx, args[0], sym)
    if name in ('gaussian', 'gauss', 'gss'):
        need(1); return gaussian(Nx, args[0], sym)
    if name in ('general hamming', 'general_hamming'):
        need(1); return general_hamming(Nx, args[0], sym)
    if name in ('barthann', 'brthan', 'bth'):
        need(0); return barthann(Nx, sym)
    if name in ('bohman', 'bman', 'bmn'):
        need(0); return bohman(Nx, sym)
    if name in ('parzen', 'parz', 'par'):
        need(0); return parzen(Nx, sym)
    if name in ('lanczos', 'sinc'):
        need(0); return lanczos(Nx, sym=sym)
    if name in ('chebwin', 'cheb'):
        need(1); return chebwin(Nx, args[0], sym)
    if name in ('general gaussian', 'general_gaussian', 'general gauss', 'general_gauss', 'ggs'):
        need(2); return general_gaussian(Nx, args[0], args[1], sym)
    if name in ('kaiser bessel derived', 'kaiser_bessel_derived', 'kbd'):
        need(1); return kaiser_bessel_derived(Nx, args[0], sym=sym)
    if name in ('general cosine', 'general_cosine'):
        if len(args) == 0:
            raise ValueError(f"'{name}' must have parameters")
        return general_cosine(Nx, args, sym)
    if name in ('taylor', 'taylorwin'):
        if len(args) == 0:
            return taylor(Nx, sym=sym)
        elif len(args) == 1:
            return taylor(Nx, int(args[0]), sym=sym)
        elif len(args) == 2:
            return taylor(Nx, int(args[0]), args[1], sym=sym)
        return taylor(Nx, int(args[0]), args[1], args[2] != 0.0, sym)
    if name == 'dpss':
        need(1); return dpss(Nx, args[0], None, sym)
    if name in ('exponential', 'poisson'):
        if len(args) == 0:
            raise ValueError(f"'{name}' must have parameters")
        # scipy passes (center, tau) positionally
        if len(args) == 1:
            return exponential(Nx, args[0], 1.0, sym)
        return exponential(Nx, args[0], args[1], sym)
    raise ValueError(f"Invalid window name '{name}'")

def get_window(window, Nx: int, fftbins: bool = True):
    if Nx <= 0:
        raise ValueError(f'Parameter Nx={Nx} is not a positive integer')
    sym = not fftbins
    if isinstance(window, str):
        name = window
        args: List[float] = []
    elif isinstance(window, Tuple):
        name = window[0]
        args = []
        for i in static.range(1, static.len(window)):
            args.append(float(window[i]))
    else:
        return kaiser(Nx, float(window), sym)
    if name.endswith('_symmetric'):
        sym = True
        name = name[:-10]
    elif name.endswith('_periodic'):
        sym = False
        name = name[:-9]
    return _by_name(name, Nx, sym, args)

# ------------------------------------------------------------------- dpss

def _dpss_norm_kind(norm) -> int:
    # 0: 2-norm, 1: approximate, 2: subsample, -1: default
    if norm is None:
        return -1
    elif isinstance(norm, int):
        if norm != 2:
            raise ValueError(f'norm must be one of (2, "approximate", "subsample"), got {norm}')
        return 0
    elif isinstance(norm, str):
        if norm == 'approximate':
            return 1
        elif norm == 'subsample':
            return 2
        raise ValueError(f'norm must be one of (2, "approximate", "subsample"), got {norm}')
    else:
        compile_error('norm must be None, 2, "approximate" or "subsample"')

def _dpss_impl(M: int, NW: float, K: int, sym: bool, nk: int, want_ratios: bool):
    if not (0 < K <= M):
        raise ValueError('Kmax must be greater than 0 and less than M')
    if NW >= M / 2.0:
        raise ValueError('NW must be less than M/2.')
    if NW <= 0:
        raise ValueError('NW must be positive')
    Me = M if sym else M + 1
    W = float(NW) / Me
    d = [((Me - 1 - 2.0 * i) / 2.0) ** 2 * math.cos(2 * math.pi * W) for i in range(Me)]
    e = [i * (Me - i) / 2.0 for i in range(1, Me)]
    # scipy: linalg.eigh_tridiagonal(d, e, select='i', select_range=(M-Kmax, M-1))
    # (LAPACK ?stebz + ?stein), reversed so the largest concentration comes first
    from scipy.linalg import eigh_tridiagonal
    _, vecs = eigh_tridiagonal(np.array(d), np.array(e), select='i',
                               select_range=(Me - K, Me - 1))
    wins = np.empty((K, Me), dtype=float)
    for kk in range(K):
        for i in range(Me):
            wins[kk, i] = vecs[i, K - 1 - kk]
    for kk in range(0, K, 2):
        s = 0.0
        for i in range(Me):
            s += wins[kk, i]
        if s < 0:
            for i in range(Me):
                wins[kk, i] = -wins[kk, i]
    thresh = max(1e-7, 1.0 / Me)
    for kk in range(1, K, 2):
        for i in range(Me):
            v = wins[kk, i]
            if v * v > thresh:
                if v < 0:
                    for j in range(Me):
                        wins[kk, j] = -wins[kk, j]
                break
    ratios = np.empty(K, dtype=float)
    if want_ratios:
        nfft = Me
        target = 2 * Me - 1
        nfft = target
        while True:
            t = nfft
            for p in (2, 3, 5):
                while t % p == 0:
                    t //= p
            if t == 1:
                break
            nfft += 1
        r = [0.0] * Me
        r[0] = 2 * W
        for i in range(1, Me):
            xx = 2 * W * i
            r[i] = 4 * W * math.sin(math.pi * xx) / (math.pi * xx)
        for kk in range(K):
            row = np.empty(Me, dtype=float)
            for i in range(Me):
                row[i] = wins[kk, i]
            F = np.fft.rfft(row, nfft)
            cxy = np.fft.irfft(F * np.conj(F), nfft)
            s = 0.0
            for i in range(Me):
                s += cxy[i] * r[i]
            ratios[kk] = s
    if nk != 0:
        mx = -math.inf
        for kk in range(K):
            for i in range(Me):
                mx = max(mx, wins[kk, i])
        for kk in range(K):
            for i in range(Me):
                wins[kk, i] /= mx
        if Me % 2 == 0:
            corr = 1.0
            if nk == 1:
                corr = Me ** 2 / float(Me ** 2 + NW)
            else:
                row = np.empty(Me, dtype=float)
                for i in range(Me):
                    row[i] = wins[0, i]
                sF = np.fft.rfft(row)
                tot = sF[0].real
                for j in range(1, sF.size):
                    shift = -(1 - 1.0 / Me) * j
                    tot += (sF[j] * 2 * cmath.exp(-1j * math.pi * shift)).real
                corr = Me / tot
            for kk in range(K):
                for i in range(Me):
                    wins[kk, i] *= corr
    out = np.empty((K, M), dtype=float)
    for kk in range(K):
        for i in range(M):
            out[kk, i] = wins[kk, i]
    return out, ratios

def dpss(M: int, NW: float, Kmax = None, sym: bool = True, norm = None,
         return_ratios: Literal[bool] = False):
    # Kmax=None returns a single 1-D window (ratio a float), as in scipy.
    nk = _dpss_norm_kind(norm)
    if isinstance(Kmax, NoneType):
        if nk == -1:
            nk = 1
        if M <= 1:
            if return_ratios:
                return np.ones(M, dtype=float), 1.0
            else:
                return np.ones(M, dtype=float)
        w, r = _dpss_impl(M, NW, 1, sym, nk, return_ratios)
        w1 = np.empty(M, dtype=float)
        for i in range(M):
            w1[i] = w[0, i]
        if return_ratios:
            return w1, r[0]
        else:
            return w1
    else:
        if nk == -1:
            nk = 0
        if M <= 1:
            # scipy returns a 1-D ones array here; we keep the 2-D type (1, M)
            if return_ratios:
                return np.ones((1, M), dtype=float), np.ones(1, dtype=float)
            else:
                return np.ones((1, M), dtype=float)
        w, r = _dpss_impl(M, NW, int(Kmax), sym, nk, return_ratios)
        if return_ratios:
            return w, r
        else:
            return w
