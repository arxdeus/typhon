# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# Spectral estimation (periodogram, welch, csd, coherence, spectrogram) for
# real or complex N-D input along `axis`. Deviations from scipy:
#  - `window` may be a str, a tuple like ('tukey', 0.25), a float (kaiser beta)
#    or a 1-D ndarray.
#  - `detrend` may be 'constant', 'linear' or False (callables unsupported).
#  - spectrogram `mode` is a compile-time Literal[str] since 'complex' changes
#    the dtype of the result.

import numpy as np
import math
import cmath
from numpy.util import tuple_set, normalize_axis_index
from _arraytools import _to_rows
from windows import get_window

def _is_complex_arr(x) -> Literal[bool]:
    return x.dtype is complex or x.dtype is complex64

def _win_array(window, nperseg: Optional[int], n: int, default_n: int = 256):
    # _triage_segments
    if isinstance(window, str) or isinstance(window, Tuple) or isinstance(window, float):
        nps = default_n if nperseg is None else nperseg
        if nps < 1:
            raise ValueError('nperseg must be a positive integer')
        if nps > n:
            nps = n
        return get_window(window, nps), nps
    else:
        w = np.asarray(window).astype(float)
        if w.ndim != 1:
            compile_error('window must be 1-D')
        if n < w.size:
            raise ValueError('window is longer than input signal')
        if nperseg is not None and nperseg != w.size:
            raise ValueError('value specified for nperseg is different from length of window')
        return w, w.size

def _detrend_seg(seg: List, kind: int):
    # kind 0: none, 1: constant, 2: linear (least squares on 1..N / N)
    n = len(seg)
    if kind == 1:
        s = seg[0] * 0
        for v in seg:
            s += v
        m = s / n
        for i in range(n):
            seg[i] -= m
    elif kind == 2:
        # fit y = c0 * t + c1, t_i = (i+1)/n, via normal equations
        st = 0.0
        stt = 0.0
        for i in range(n):
            t = (i + 1) / n
            st += t
            stt += t * t
        sy = seg[0] * 0
        sty = seg[0] * 0
        for i in range(n):
            t = (i + 1) / n
            sy += seg[i]
            sty += t * seg[i]
        det = stt * n - st * st
        c0 = (n * sty - st * sy) / det
        c1 = (stt * sy - st * sty) / det
        for i in range(n):
            seg[i] -= c0 * ((i + 1) / n) + c1

def _detrend_kind(detrend) -> int:
    if isinstance(detrend, bool):
        if detrend:
            raise ValueError('detrend=True is not valid; use a trend type')
        return 0
    elif isinstance(detrend, str):
        if detrend in ('constant', 'c'):
            return 1
        elif detrend in ('linear', 'l'):
            return 2
        raise ValueError("Trend type must be 'linear' or 'constant'.")
    elif detrend is None:
        return 0
    else:
        compile_error("detrend must be 'constant', 'linear' or False")

def _stft_rows(x2, y2, same: bool, win, nperseg: int, noverlap: int, nfft: int,
               dkind: int, onesided: bool, cross: bool):
    # returns complex array (M, nseg, nf) of per-segment spectra:
    # cross=True:  conj(X) * Y ; cross=False: X (stft)
    M = x2.shape[0]
    N = x2.shape[1]
    step = nperseg - noverlap
    nseg = (N - nperseg) // step + 1 if N >= nperseg else 0
    nf = nfft // 2 + 1 if onesided else nfft
    out = np.empty((M, nseg, nf), dtype=complex)
    buf = np.zeros((nseg, nfft), dtype=x2.dtype)
    bufy = np.zeros((nseg, nfft), dtype=x2.dtype)
    for r in range(M):
        for s in range(nseg):
            seg = [x2[r, s * step + i] for i in range(nperseg)]
            _detrend_seg(seg, dkind)
            for i in range(nperseg):
                buf[s, i] = win[i] * seg[i]
            if not same:
                segy = [y2[r, s * step + i] for i in range(nperseg)]
                _detrend_seg(segy, dkind)
                for i in range(nperseg):
                    bufy[s, i] = win[i] * segy[i]
        X = _fft_rows(buf, nfft, onesided)
        if same:
            for s in range(nseg):
                for k in range(nf):
                    v = X[s, k]
                    out[r, s, k] = v.conjugate() * v if cross else v
        else:
            Y = _fft_rows(bufy, nfft, onesided)
            for s in range(nseg):
                for k in range(nf):
                    out[r, s, k] = X[s, k].conjugate() * Y[s, k]
    return out

def _fft_rows(buf, nfft: int, onesided: bool):
    if onesided:
        if buf.dtype is complex:
            compile_error('onesided spectrum requires real input')
        else:
            return np.fft.rfft(buf, nfft, axis=-1)
    else:
        return np.fft.fft(buf.astype(complex), nfft, axis=-1)

def _freqs(nfft: int, fs: float, onesided: bool):
    if onesided:
        return np.fft.rfftfreq(nfft, 1.0 / fs)
    else:
        return np.fft.fftfreq(nfft, 1.0 / fs)

def _prep(x, axis: int):
    xa = np.asarray(x)
    if _is_complex_arr(xa):
        x2, msh, ax = _to_rows(xa.astype(complex), axis)
        return x2, msh, ax
    else:
        x2, msh, ax = _to_rows(xa.astype(float), axis)
        return x2, msh, ax

def _scale(win, fs: float, scaling: str) -> float:
    s = 0.0
    if scaling == 'density':
        for i in range(win.size):
            s += win[i] * win[i]
        return 1.0 / (fs * s)
    elif scaling == 'spectrum':
        for i in range(win.size):
            s += win[i]
        return 1.0 / (s * s)
    raise ValueError(f'Unknown scaling: {scaling}')

def _median_bias(n: int) -> float:
    s = 0.0
    for i in range(1, (n - 1) // 2 + 1):
        ii2 = 2.0 * i
        s += 1.0 / (ii2 + 1) - 1.0 / ii2
    return 1 + s

def _median(v: List[float]) -> float:
    v = sorted(v)
    n = len(v)
    if n % 2:
        return v[n // 2]
    return 0.5 * (v[n // 2 - 1] + v[n // 2])

def _csd_impl(x, y, same: bool, fs: float, window, nperseg: Optional[int], noverlap: Optional[int],
              nfft: Optional[int], detrend, return_onesided: bool, scaling: str, axis: int,
              average: str):
    x2, msh, ax = _prep(x, axis)
    y2, _, _ = _prep(y, axis)
    n = x2.shape[1]
    win, nps = _win_array(window, nperseg, n)
    nf = nps if nfft is None else nfft
    if nf < nps:
        raise ValueError(f'nfft={nf} must be greater than or equal to nperseg={nps}!')
    nov = nps // 2 if noverlap is None else noverlap
    if nov >= nps:
        raise ValueError(f'noverlap={nov} must be less than nperseg={nps}!')
    onesided = return_onesided
    if _is_complex_arr(x2) or _is_complex_arr(y2):
        onesided = False
    sc = _scale(win, fs, scaling)
    P = _stft_rows(x2, y2, same, win, nps, nov, nf, _detrend_kind(detrend), onesided, True)
    M = P.shape[0]
    nseg = P.shape[1]
    nfr = P.shape[2]
    out = np.empty((M, nfr), dtype=complex)
    for r in range(M):
        for k in range(nfr):
            fac = sc
            if onesided and k >= 1 and (k < nfr - 1 or nf % 2 == 1):
                fac *= 2.0
            if nseg == 1:
                out[r, k] = P[r, 0, k] * fac
            elif average == 'mean':
                s = complex(0.0)
                for t in range(nseg):
                    s += P[r, t, k] * fac
                out[r, k] = s / nseg
            elif average == 'median':
                re = [P[r, t, k].real * fac for t in range(nseg)]
                im = [P[r, t, k].imag * fac for t in range(nseg)]
                out[r, k] = complex(_median(re), _median(im)) / _median_bias(nseg)
            else:
                raise ValueError(f"Parameter average={average} must be 'median' or 'mean'!")
    return _freqs(nf, fs, onesided), out, msh, ax

def _rows_back(out2, msh, ax: int):
    sh = tuple_set(msh, len(msh) - 1, out2.shape[1])
    return np.ascontiguousarray(np.moveaxis(out2.reshape(sh), -1, ax))

def csd(x, y, fs: float = 1.0, window = 'hann_periodic', nperseg: Optional[int] = None,
        noverlap: Optional[int] = None, nfft: Optional[int] = None, detrend = 'constant',
        return_onesided: bool = True, scaling: str = 'density', axis: int = -1,
        average: str = 'mean'):
    xa = np.asarray(x)
    ya = np.asarray(y)
    if xa.shape != ya.shape:
        raise ValueError('x and y must have the same shape (broadcasting is not supported natively)')
    f, out, msh, ax = _csd_impl(xa, ya, False, fs, window, nperseg, noverlap, nfft, detrend,
                                return_onesided, scaling, axis, average)
    return f, _rows_back(out, msh, ax)

def welch(x, fs: float = 1.0, window = 'hann_periodic', nperseg: Optional[int] = None,
          noverlap: Optional[int] = None, nfft: Optional[int] = None, detrend = 'constant',
          return_onesided: bool = True, scaling: str = 'density', axis: int = -1,
          average: str = 'mean'):
    xa = np.asarray(x)
    f, out, msh, ax = _csd_impl(xa, xa, True, fs, window, nperseg, noverlap, nfft, detrend,
                                return_onesided, scaling, axis, average)
    return f, _rows_back(out.real, msh, ax)

def periodogram(x, fs: float = 1.0, window = 'boxcar', nfft: Optional[int] = None,
                detrend = 'constant', return_onesided: bool = True, scaling: str = 'density',
                axis: int = -1):
    xa = np.asarray(x)
    ax = normalize_axis_index(axis, xa.ndim)
    n = xa.shape[ax]
    nperseg = n
    nf: Optional[int] = nfft
    if nfft is not None:
        if nfft < n:
            nperseg = nfft
            nf = None
            idx = np.arange(nfft)
            xa = np.take(xa, idx, axis=ax)
    return welch(xa, fs, window, nperseg, 0, nf, detrend, return_onesided, scaling, axis)

def coherence(x, y, fs: float = 1.0, window = 'hann_periodic', nperseg: Optional[int] = None,
              noverlap: Optional[int] = None, nfft: Optional[int] = None, detrend = 'constant',
              axis: int = -1):
    f, Pxx = welch(x, fs, window, nperseg, noverlap, nfft, detrend, axis=axis)
    _, Pyy = welch(y, fs, window, nperseg, noverlap, nfft, detrend, axis=axis)
    _, Pxy = csd(x, y, fs, window, nperseg, noverlap, nfft, detrend, axis=axis)
    return f, np.abs(Pxy) ** 2 / Pxx / Pyy

def spectrogram(x, fs: float = 1.0, window = ('tukey_periodic', 0.25), nperseg: Optional[int] = None,
                noverlap: Optional[int] = None, nfft: Optional[int] = None, detrend = 'constant',
                return_onesided: bool = True, scaling: str = 'density', axis: int = -1,
                mode: Literal[str] = 'psd'):
    if mode != 'psd' and mode != 'complex' and mode != 'magnitude' and mode != 'angle' and mode != 'phase':
        compile_error("unknown value for mode, must be one of ['psd', 'complex', 'magnitude', 'angle', 'phase']")
    x2, msh, ax = _prep(x, axis)
    n = x2.shape[1]
    win, nps = _win_array(window, nperseg, n)
    nov = nps // 8 if noverlap is None else noverlap
    nf = nps if nfft is None else nfft
    if nf < nps:
        raise ValueError('nfft must be greater than or equal to nperseg.')
    if nov >= nps:
        raise ValueError('noverlap must be less than nperseg.')
    onesided = return_onesided
    if _is_complex_arr(x2):
        onesided = False
    sc = _scale(win, fs, scaling)
    psd: Literal[bool] = (mode == 'psd')
    if not psd:
        sc = math.sqrt(sc)
    P = _stft_rows(x2, x2, True, win, nps, nov, nf, _detrend_kind(detrend), onesided, psd)
    M = P.shape[0]
    nseg = P.shape[1]
    nfr = P.shape[2]
    step = nps - nov
    t = np.empty(nseg, dtype=float)
    for i in range(nseg):
        t[i] = (nps / 2 + i * step) / fs
    freqs = _freqs(nf, fs, onesided)
    sh = (*tuple_set(msh, len(msh) - 1, nfr), nseg)
    if mode == 'psd':
        R = np.empty((M, nfr, nseg), dtype=float)
        for r in range(M):
            for k in range(nfr):
                fac = sc
                if onesided and k >= 1 and (k < nfr - 1 or nf % 2 == 1):
                    fac *= 2.0
                for s in range(nseg):
                    R[r, k, s] = P[r, s, k].real * fac
        return freqs, t, np.ascontiguousarray(np.moveaxis(R.reshape(sh), -2, ax))
    elif mode == 'complex':
        C = np.empty((M, nfr, nseg), dtype=complex)
        for r in range(M):
            for k in range(nfr):
                for s in range(nseg):
                    C[r, k, s] = P[r, s, k] * sc
        return freqs, t, np.ascontiguousarray(np.moveaxis(C.reshape(sh), -2, ax))
    else:
        R = np.empty((M, nfr, nseg), dtype=float)
        for r in range(M):
            for k in range(nfr):
                prev = 0.0
                for s in range(nseg):
                    v = P[r, s, k] * sc
                    if mode == 'magnitude':
                        R[r, k, s] = abs(v)
                    else:
                        a = math.atan2(v.imag, v.real)
                        if mode == 'phase' and s > 0:
                            # numpy.unwrap along the time axis
                            d = a - prev
                            dd = ((d + math.pi) % (2 * math.pi)) - math.pi
                            if dd == -math.pi and d > 0:
                                dd = math.pi
                            if abs(d) >= math.pi:
                                a = prev + dd
                        R[r, k, s] = a
                        prev = a
        return freqs, t, np.ascontiguousarray(np.moveaxis(R.reshape(sh), -2, ax))

# ------------------------------------------------------------- stft / istft

def _boundary_kind(boundary) -> str:
    if boundary is None:
        return 'none'
    elif isinstance(boundary, str):
        if boundary not in ('even', 'odd', 'constant', 'zeros'):
            raise ValueError(f"Unknown boundary option '{boundary}', must be one of: ['even', 'odd', 'constant', 'zeros', None]")
        return boundary
    else:
        compile_error('boundary must be a str or None')

def stft(x, fs: float = 1.0, window = 'hann_periodic', nperseg: int = 256,
         noverlap: Optional[int] = None, nfft: Optional[int] = None, detrend = False,
         return_onesided: bool = True, boundary = 'zeros', padded: bool = True,
         axis: int = -1, scaling: str = 'spectrum'):
    from _arraytools import _ext_rows
    if scaling != 'spectrum' and scaling != 'psd':
        raise ValueError(f"Parameter scaling={scaling} not in ['spectrum', 'psd']!")
    scl = 'density' if scaling == 'psd' else 'spectrum'
    x2, msh, ax = _prep(x, axis)
    win, nps = _win_array(window, nperseg, x2.shape[1])
    nf = nps if nfft is None else nfft
    if nf < nps:
        raise ValueError('nfft must be greater than or equal to nperseg.')
    nov = nps // 2 if noverlap is None else noverlap
    if nov >= nps:
        raise ValueError('noverlap must be less than nperseg.')
    nstep = nps - nov
    bk = _boundary_kind(boundary)
    if bk != 'none':
        x2 = _ext_rows(x2, nps // 2, bk)
    if padded:
        L = x2.shape[1]
        nadd = (-(L - nps) % nstep) % nps
        if nadd > 0:
            xp = np.zeros((x2.shape[0], L + nadd), dtype=x2.dtype)
            for r in range(x2.shape[0]):
                for i in range(L):
                    xp[r, i] = x2[r, i]
            x2 = xp
    onesided = return_onesided
    if _is_complex_arr(x2):
        onesided = False
    sc = math.sqrt(_scale(win, fs, scl))
    P = _stft_rows(x2, x2, True, win, nps, nov, nf, _detrend_kind(detrend), onesided, False)
    M = P.shape[0]
    nseg = P.shape[1]
    nfr = P.shape[2]
    t = np.empty(nseg, dtype=float)
    for i in range(nseg):
        t[i] = (nps / 2 + i * nstep) / fs
        if bk != 'none':
            t[i] -= nps / 2 / fs
    C = np.empty((M, nfr, nseg), dtype=complex)
    for r in range(M):
        for k in range(nfr):
            for s in range(nseg):
                C[r, k, s] = P[r, s, k] * sc
    sh = (*tuple_set(msh, len(msh) - 1, nfr), nseg)
    return _freqs(nf, fs, onesided), t, np.ascontiguousarray(np.moveaxis(C.reshape(sh), -2, ax))

def istft(Zxx, fs: float = 1.0, window = 'hann_periodic', nperseg: Optional[int] = None,
          noverlap: Optional[int] = None, nfft: Optional[int] = None,
          input_onesided: Literal[bool] = True, boundary: bool = True,
          time_axis: int = -1, freq_axis: int = -2, scaling: str = 'spectrum'):
    # Returns (t, x); x is float for input_onesided=True and complex otherwise.
    Z = np.asarray(Zxx).astype(complex)
    if Z.ndim < 2:
        compile_error('Input stft must be at least 2d!')
    fa = normalize_axis_index(freq_axis, Z.ndim)
    ta = normalize_axis_index(time_axis, Z.ndim)
    if fa == ta:
        raise ValueError('Must specify differing time and frequency axes!')
    # move (freq, time) to the last two axes
    Zm = np.ascontiguousarray(np.moveaxis(Z, (fa, ta), (-2, -1)))
    nfr = Zm.shape[-2]
    nseg = Zm.shape[-1]
    Z3 = Zm.reshape(-1, nfr, nseg)
    n_default = 2 * (nfr - 1) if input_onesided else nfr
    nps = n_default if nperseg is None else nperseg
    if nps < 1:
        raise ValueError('nperseg must be a positive integer')
    nf = 0
    if nfft is None:
        nf = nps if (input_onesided and nps == n_default + 1) else n_default
    else:
        nf = nfft
        if nf < nps:
            raise ValueError('nfft must be greater than or equal to nperseg.')
    nov = nps // 2 if noverlap is None else noverlap
    if nov >= nps:
        raise ValueError('noverlap must be less than nperseg.')
    nstep = nps - nov
    win, _ = _win_array(window, nps, nps)
    ws = 0.0
    w2 = 0.0
    for i in range(nps):
        ws += win[i]
        w2 += win[i] ** 2
    fac = 0.0
    if scaling == 'spectrum':
        fac = ws
    elif scaling == 'psd':
        fac = math.sqrt(fs * w2)
    else:
        raise ValueError(f"Parameter scaling={scaling} not in ['spectrum', 'psd']!")
    M = Z3.shape[0]
    outlen = nps + (nseg - 1) * nstep
    norm = [0.0] * outlen
    for s in range(nseg):
        for i in range(nps):
            norm[s * nstep + i] += win[i] ** 2
    lo = nps // 2 if boundary else 0
    hi = outlen - nps // 2 if boundary else outlen
    T = float if input_onesided else complex
    X = np.zeros((M, hi - lo), dtype=T)
    for r in range(M):
        acc = [complex(0.0)] * outlen
        for s in range(nseg):
            col = np.empty(nfr, dtype=complex)
            for k in range(nfr):
                col[k] = Z3[r, k, s]
            if input_onesided:
                seg = np.fft.irfft(col, nf)
                for i in range(nps):
                    acc[s * nstep + i] += complex(seg[i] * fac * win[i])
            else:
                seg = np.fft.ifft(col, nf)
                for i in range(nps):
                    acc[s * nstep + i] += seg[i] * fac * win[i]
        for i in range(lo, hi):
            nv = norm[i] if norm[i] > 1e-10 else 1.0
            v = acc[i] / nv
            if input_onesided:
                X[r, i - lo] = v.real
            else:
                X[r, i - lo] = v
    # output shape: outer dims of Z (without freq/time) + time inserted at time_axis
    outer = Zm.shape[:-2]
    Xr = X.reshape((*outer, hi - lo))
    tpos = ta if fa > ta else ta - 1
    Xo = np.ascontiguousarray(np.moveaxis(Xr, -1, tpos))
    t = np.empty(Xo.shape[0], dtype=float)
    for i in range(t.size):
        t[i] = i / fs
    return t, Xo

def check_COLA(window, nperseg: int, noverlap: int, tol: float = 1e-10) -> bool:
    return _check_ola(window, nperseg, noverlap, tol, False)

def check_NOLA(window, nperseg: int, noverlap: int, tol: float = 1e-10) -> bool:
    return _check_ola(window, nperseg, noverlap, tol, True)

def _check_ola(window, nperseg: int, noverlap: int, tol: float, sq: bool) -> bool:
    if nperseg < 1:
        raise ValueError('nperseg must be a positive integer')
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    win, _ = _win_array(window, nperseg, nperseg)
    step = nperseg - noverlap
    bins = [0.0] * step
    for ii in range(nperseg // step):
        for j in range(step):
            v = win[ii * step + j]
            bins[j] += v * v if sq else v
    rem = nperseg % step
    for j in range(rem):
        v = win[nperseg - rem + j]
        bins[j] += v * v if sq else v
    if sq:
        return min(bins) > tol
    med = sorted(bins)
    m = med[len(med) // 2] if len(med) % 2 else 0.5 * (med[len(med) // 2 - 1] + med[len(med) // 2])
    return max([abs(b - m) for b in bins]) < tol
