# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# scipy.fft on top of the native numpy.fft (pocketfft) implementation.
# Real inputs are transformed in double precision like numpy.fft; scipy keeps
# float32 inputs in single precision (deviation). `workers`, `overwrite_x`
# and `plan` are accepted for signature compatibility and ignored.

import numpy as np
import numpy.fft as _f

def fft(x, n: Optional[int] = None, axis: int = -1, norm: Optional[str] = None,
        overwrite_x: bool = False, workers: Optional[int] = None, plan = None):
    return _f.fft(x, n, axis, norm)

def ifft(x, n: Optional[int] = None, axis: int = -1, norm: Optional[str] = None,
         overwrite_x: bool = False, workers: Optional[int] = None, plan = None):
    return _f.ifft(x, n, axis, norm)

def rfft(x, n: Optional[int] = None, axis: int = -1, norm: Optional[str] = None,
         overwrite_x: bool = False, workers: Optional[int] = None, plan = None):
    return _f.rfft(x, n, axis, norm)

def irfft(x, n: Optional[int] = None, axis: int = -1, norm: Optional[str] = None,
          overwrite_x: bool = False, workers: Optional[int] = None, plan = None):
    return _f.irfft(x, n, axis, norm)

def hfft(x, n: Optional[int] = None, axis: int = -1, norm: Optional[str] = None,
         overwrite_x: bool = False, workers: Optional[int] = None, plan = None):
    return _f.hfft(x, n, axis, norm)

def ihfft(x, n: Optional[int] = None, axis: int = -1, norm: Optional[str] = None,
          overwrite_x: bool = False, workers: Optional[int] = None, plan = None):
    return _f.ihfft(x, n, axis, norm)

def fftn(x, s = None, axes = None, norm: Optional[str] = None,
         overwrite_x: bool = False, workers: Optional[int] = None, plan = None):
    return _f.fftn(x, s, axes, norm)

def ifftn(x, s = None, axes = None, norm: Optional[str] = None,
          overwrite_x: bool = False, workers: Optional[int] = None, plan = None):
    return _f.ifftn(x, s, axes, norm)

def rfftn(x, s = None, axes = None, norm: Optional[str] = None,
          overwrite_x: bool = False, workers: Optional[int] = None, plan = None):
    return _f.rfftn(x, s, axes, norm)

def irfftn(x, s = None, axes = None, norm: Optional[str] = None,
           overwrite_x: bool = False, workers: Optional[int] = None, plan = None):
    return _f.irfftn(x, s, axes, norm)

def fft2(x, s = None, axes = (-2, -1), norm: Optional[str] = None,
         overwrite_x: bool = False, workers: Optional[int] = None, plan = None):
    return _f.fft2(x, s, axes, norm)

def ifft2(x, s = None, axes = (-2, -1), norm: Optional[str] = None,
          overwrite_x: bool = False, workers: Optional[int] = None, plan = None):
    return _f.ifft2(x, s, axes, norm)

def rfft2(x, s = None, axes = (-2, -1), norm: Optional[str] = None,
          overwrite_x: bool = False, workers: Optional[int] = None, plan = None):
    return _f.rfft2(x, s, axes, norm)

def irfft2(x, s = None, axes = (-2, -1), norm: Optional[str] = None,
           overwrite_x: bool = False, workers: Optional[int] = None, plan = None):
    return _f.irfft2(x, s, axes, norm)

def fftfreq(n: int, d: float = 1.0):
    return _f.fftfreq(n, d)

def rfftfreq(n: int, d: float = 1.0):
    return _f.rfftfreq(n, d)

def fftshift(x, axes = None):
    return _f.fftshift(x, axes)

def ifftshift(x, axes = None):
    return _f.ifftshift(x, axes)

def _is_smooth(n: int, real: bool) -> bool:
    primes = [2, 3, 5] if real else [2, 3, 5, 7, 11]
    for p in primes:
        while n % p == 0:
            n //= p
    return n == 1

def next_fast_len(target: int, real: bool = False) -> int:
    # pocketfft (scipy.fft backend): 5-smooth for real, 11-smooth for complex
    if target < 0:
        raise ValueError('Target cannot be negative')
    if target == 0:
        return 0
    n = target
    while not _is_smooth(n, real):
        n += 1
    return n

def prev_fast_len(target: int, real: bool = False) -> int:
    if target < 0:
        raise ValueError('Target cannot be negative')
    if target == 0:
        return 0
    n = target
    while not _is_smooth(n, real):
        n -= 1
    return n
