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
# Eigenvectors of the symmetric tridiagonal DPSS matrix are computed by
# Sturm-sequence bisection (eigenvalues to full precision) followed by
# inverse iteration with a partially pivoted tridiagonal LU solve, which is
# equivalent to LAPACK stebz/stein used by scipy. Sign conventions follow
# scipy.

def _sturm_count(d: List[float], e2: List[float], x: float) -> int:
    # number of eigenvalues < x
    cnt = 0
    q = d[0] - x
    if q < 0:
        cnt += 1
    for i in range(1, len(d)):
        if q == 0.0:
            q = 1e-300
        q = d[i] - x - e2[i - 1] / q
        if q < 0:
            cnt += 1
    return cnt

def _tri_eig_bisect(d: List[float], e: List[float], k: int) -> float:
    # k-th smallest eigenvalue (0-based)
    n = len(d)
    e2 = [v * v for v in e]
    lo = math.inf
    hi = -math.inf
    for i in range(n):
        r = (abs(e[i - 1]) if i > 0 else 0.0) + (abs(e[i]) if i < n - 1 else 0.0)
        lo = min(lo, d[i] - r)
        hi = max(hi, d[i] + r)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            break
        if _sturm_count(d, e2, mid) > k:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

def _tri_solve(d: List[float], e: List[float], lam: float, b: List[float]) -> List[float]:
    # solve (T - lam I) x = b with partial pivoting (Gaussian elimination, banded)
    n = len(d)
    # rows stored as [a0, a1, a2] at columns i, i+1, i+2
    A0 = [d[i] - lam for i in range(n)]
    A1 = [e[i] if i < n - 1 else 0.0 for i in range(n)]
    A2 = [0.0] * n
    L = [0.0] * n  # sub-diagonal element below row i (original)
    for i in range(n - 1):
        L[i] = e[i]
    rhs = list(b)
    for i in range(n - 1):
        # candidates: row i (A0[i], A1[i], A2[i]) and row i+1 with sub elem L[i]
        sub = L[i]
        if abs(sub) > abs(A0[i]):
            # swap row i and i+1. Row i+1 entries at cols i, i+1, i+2: sub, A0[i+1], A1[i+1]
            r0, r1, r2 = sub, A0[i + 1], A1[i + 1]
            o0, o1, o2 = A0[i], A1[i], A2[i]
            A0[i], A1[i], A2[i] = r0, r1, r2
            rhs[i], rhs[i + 1] = rhs[i + 1], rhs[i]
            f = o0 / r0
            A0[i + 1] = o1 - f * r1
            A1[i + 1] = o2 - f * r2
            rhs[i + 1] -= f * rhs[i]
        else:
            piv = A0[i] if A0[i] != 0.0 else 1e-300
            A0[i] = piv
            f = sub / piv
            A0[i + 1] = A0[i + 1] - f * A1[i]
            A1[i + 1] = A1[i + 1] - f * A2[i]
            rhs[i + 1] -= f * rhs[i]
    if A0[n - 1] == 0.0:
        A0[n - 1] = 1e-300
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = rhs[i]
        if i + 1 < n:
            s -= A1[i] * x[i + 1]
        if i + 2 < n:
            s -= A2[i] * x[i + 2]
        x[i] = s / A0[i]
    return x

def _tri_eigvec(d: List[float], e: List[float], lam: float, seed: int) -> List[float]:
    n = len(d)
    # perturb shift slightly to avoid an exactly singular system
    scale = max([abs(v) for v in d] + [abs(v) for v in e] + [1.0])
    shift = lam + 2.220446049250313e-16 * scale * 10
    v = [1.0 + 0.1 * math.sin(1.0 + i * 0.7 + seed) for i in range(n)]
    for _ in range(4):
        v = _tri_solve(d, e, shift, v)
        nrm = math.sqrt(sum([t * t for t in v]))
        v = [t / nrm for t in v]
    return v

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
    wins = np.empty((K, Me), dtype=float)
    for kk in range(K):
        lam = _tri_eig_bisect(d, e, Me - 1 - kk)
        v = _tri_eigvec(d, e, lam, kk)
        for i in range(Me):
            wins[kk, i] = v[i]
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
