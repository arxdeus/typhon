# Ported from scipy 1.18.1 (BSD-3-Clause)

import numpy as np
import math
import cmath
from windows import get_window
from _filter_design import _flist

def kaiser_beta(a: float) -> float:
    if a > 50:
        return 0.1102 * (a - 8.7)
    elif a > 21:
        return 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21)
    else:
        return 0.0

def kaiser_atten(numtaps: int, width: float) -> float:
    return 2.285 * (numtaps - 1) * math.pi * width + 7.95

def kaiserord(ripple: float, width: float):
    A = abs(ripple)
    if A < 8:
        raise ValueError(f'Requested maximum ripple attenuation {A} is too small for the Kaiser formula.')
    beta = kaiser_beta(A)
    numtaps = (A - 7.95) / 2.285 / (math.pi * width) + 1
    return int(math.ceil(numtaps)), beta

def _sinc(x: float) -> float:
    if x == 0.0:
        return 1.0
    y = math.pi * x
    return math.sin(y) / y

def _pass_zero(pass_zero, ncut: int) -> bool:
    if isinstance(pass_zero, bool):
        return pass_zero
    elif isinstance(pass_zero, str):
        if pass_zero in ('bandstop', 'lowpass'):
            if pass_zero == 'lowpass' and ncut != 1:
                raise ValueError(f'cutoff must have one element if pass_zero=="lowpass", got {ncut}')
            if pass_zero == 'bandstop' and ncut <= 1:
                raise ValueError(f'cutoff must have at least two elements if pass_zero=="bandstop", got {ncut}')
            return True
        elif pass_zero in ('bandpass', 'highpass'):
            if pass_zero == 'highpass' and ncut != 1:
                raise ValueError(f'cutoff must have one element if pass_zero=="highpass", got {ncut}')
            if pass_zero == 'bandpass' and ncut <= 1:
                raise ValueError(f'cutoff must have at least two elements if pass_zero=="bandpass", got {ncut}')
            return False
        raise ValueError(f"Parameter pass_zero={pass_zero} not in (True, False, 'bandpass', 'lowpass', 'highpass', 'bandstop')")
    else:
        compile_error('pass_zero must be bool or str')

def _firwin_win(window, numtaps: int, width: Optional[float], nyq: float):
    if width is not None:
        atten = kaiser_atten(numtaps, width / nyq)
        return get_window(('kaiser', kaiser_beta(atten)), numtaps, fftbins=False)
    return get_window(window, numtaps, fftbins=False)

def firwin(numtaps: int, cutoff, width: Optional[float] = None, window = 'hamming',
           pass_zero = True, scale: bool = True, fs: Optional[float] = None):
    fsv = 2.0 if fs is None else fs
    nyq = 0.5 * fsv
    cut = [c / nyq for c in _flist(cutoff)]
    if len(cut) == 0:
        raise ValueError('At least one cutoff frequency must be given.')
    if min(cut) <= 0 or max(cut) >= 1:
        raise ValueError('Invalid cutoff frequency: frequencies must be greater than 0 and less than fs/2.')
    for i in range(1, len(cut)):
        if cut[i] - cut[i - 1] <= 0:
            raise ValueError('Invalid cutoff frequencies: the frequencies must be strictly increasing.')
    pz = _pass_zero(pass_zero, len(cut))
    pass_nyquist = (len(cut) % 2 == 0) == pz
    if pass_nyquist and numtaps % 2 == 0:
        raise ValueError('A filter with an even number of coefficients must have zero response at the Nyquist frequency.')
    c = ([0.0] if pz else List[float]()) + cut + ([1.0] if pass_nyquist else List[float]())
    nb = len(c) // 2
    alpha = 0.5 * (numtaps - 1)
    h = np.zeros(numtaps, dtype=float)
    for j in range(nb):
        left = c[2 * j]
        right = c[2 * j + 1]
        for i in range(numtaps):
            m = i - alpha
            h[i] += right * _sinc(right * m)
        for i in range(numtaps):
            m = i - alpha
            h[i] -= left * _sinc(left * m)
    win = _firwin_win(window, numtaps, width, nyq)
    for i in range(numtaps):
        h[i] *= win[i]
    if scale:
        left = c[0]
        right = c[1]
        sf = 0.0
        if left == 0:
            sf = 0.0
        elif right == 1:
            sf = 1.0
        else:
            sf = 0.5 * (left + right)
        s = 0.0
        for i in range(numtaps):
            s += h[i] * math.cos(math.pi * (i - alpha) * sf)
        for i in range(numtaps):
            h[i] /= s
    return h

def firwin2(numtaps: int, freq, gain, nfreqs: Optional[int] = None, window = 'hamming',
            antisymmetric: bool = False, fs: Optional[float] = None):
    fsv = 2.0 if fs is None else fs
    nyq = 0.5 * fsv
    f = _flist(freq)
    g = _flist(gain)
    if len(f) != len(g):
        raise ValueError('freq and gain must be of same length.')
    if nfreqs is not None and numtaps >= nfreqs:
        raise ValueError('ntaps must be less than nfreqs')
    if f[0] != 0 or f[-1] != nyq:
        raise ValueError('freq must start with 0 and end with fs/2.')
    for i in range(len(f) - 1):
        if f[i + 1] - f[i] < 0:
            raise ValueError('The values in freq must be nondecreasing.')
    ftype = 0
    if antisymmetric:
        ftype = 4 if numtaps % 2 == 0 else 3
    else:
        ftype = 2 if numtaps % 2 == 0 else 1
    if ftype == 2 and g[-1] != 0.0:
        raise ValueError('A Type II filter must have zero gain at the Nyquist frequency.')
    elif ftype == 3 and (g[0] != 0.0 or g[-1] != 0.0):
        raise ValueError('A Type III filter must have zero gain at zero and Nyquist frequencies.')
    elif ftype == 4 and g[0] != 0.0:
        raise ValueError('A Type IV filter must have zero gain at zero frequency.')
    nf = 0
    if nfreqs is None:
        nf = 1 + (1 << int(math.ceil(math.log(numtaps, 2))))
    else:
        nf = nfreqs
    eps = 2.220446049250313e-16 * nyq
    for k in range(len(f) - 1):
        if f[k] == f[k + 1]:
            f[k] = f[k] - eps
            f[k + 1] = f[k + 1] + eps
    x = np.linspace(0.0, nyq, nf)
    fx = np.interp(x, _farr2(f), _farr2(g))
    fx2 = np.empty(nf, dtype=complex)
    for i in range(nf):
        sh = cmath.exp(-(numtaps - 1) / 2.0 * 1j * math.pi * x[i] / nyq)
        if ftype > 2:
            sh *= 1j
        fx2[i] = fx[i] * sh
    out_full = np.fft.irfft(fx2)
    out = np.empty(numtaps, dtype=float)
    if window is None:
        for i in range(numtaps):
            out[i] = out_full[i]
    else:
        wind = get_window(window, numtaps, fftbins=False)
        for i in range(numtaps):
            out[i] = out_full[i] * wind[i]
    if ftype == 3:
        out[numtaps // 2] = 0.0
    return out

def _farr2(x: List[float]):
    a = np.empty(len(x), dtype=float)
    for i in range(len(x)):
        a[i] = x[i]
    return a
