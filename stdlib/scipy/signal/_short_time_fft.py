# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)
# ShortTimeFFT (scipy/signal/_short_time_fft.py): STFT / inverse STFT /
# spectrogram with arbitrary windows, hop sizes, FFT modes and scaling.
# Deviations: real-valued windows and 1-D signals `x` only; `stft*`/
# `spectrogram` accept 1-D arrays and return (f_pts, p_num) complex arrays;
# `istft` returns a float array (real part) and `istft_complex` the complex
# array (needed for the 'twosided' / 'centered' modes); `detr` may be
# 'linear', 'constant', None or a callable on 1-D arrays; `axis` options and
# complex windows are not supported; dual_win/win are plain float arrays.

import math
import numpy as np
from windows import get_window
from _signaltools import detrend

def _calc_dual_canonical_window(win: np.ndarray[float, 1], hop: int) -> np.ndarray[float, 1]:
    n = win.shape[0]
    if hop > n:
        raise ValueError(f"hop={hop} is larger than window length of {n} => STFT not invertible!")
    w2 = win * win
    DD = w2.copy()
    k = hop
    while k < n:
        for i in range(k, n):
            DD[i] += w2[i - k]
        for i in range(0, n - k):
            DD[i] += w2[i + k]
        k += hop
    mx = DD.max()
    rr = 1e-15 * mx
    for i in range(n):
        if not DD[i] >= rr:
            raise ValueError("Short-time Fourier Transform not invertible!")
    return win / DD

def closest_STFT_dual_window(win, hop: int, desired_dual = None, *, scaled: bool = True):
    w = np.asarray(win).astype(float)
    if w.ndim != 1:
        raise ValueError("Parameters `win` and `desired_dual` are not 1d arrays of equal length!")
    n = w.shape[0]
    dd = np.ones(n)
    if desired_dual is not None:
        dd = np.asarray(desired_dual).astype(float)
        if dd.ndim != 1 or dd.shape[0] != n:
            raise ValueError("Parameters `win` and `desired_dual` are not 1d arrays of equal length!")
    for i in range(n):
        if not math.isfinite(w[i]):
            raise ValueError("Parameter win must have finite entries!")
        if not math.isfinite(dd[i]):
            raise ValueError("Parameter desired_dual must have finite entries!")
    if not (1 <= hop and hop <= n):
        raise ValueError(f"Parameter hop={hop} is not an integer between 1 and len(win)={n}!")
    w_d = _calc_dual_canonical_window(w, hop)
    wdd = w * dd
    q_d = wdd.copy()
    k = hop
    while k < n:
        for i in range(k, n):
            q_d[i] += wdd[i - k]
        for i in range(0, n - k):
            q_d[i] += wdd[i + k]
        k += hop
    q_d = w_d * q_d
    if not scaled:
        return w_d + dd - q_d, 1.0
    numerator = float((q_d * w_d).sum())
    denominator = float((q_d * q_d).sum())
    if not (abs(numerator) > 0 and denominator > 1e-15):
        raise ValueError("Unable to calculate scaled closest dual window due to numerically unstable scaling factor! Try setting parameter `scaled` to False.")
    alpha = numerator / denominator
    return w_d + alpha * (dd - q_d), alpha

def _pad_index(i: int, n: int, mode: int) -> int:
    # mode 0 zeros(-1), 1 edge, 2 even(reflect); returns source index or -1
    if i >= 0 and i < n:
        return i
    if mode == 0:
        return -1
    if mode == 1:
        return 0 if i < 0 else n - 1
    if n == 1:
        return 0
    period = 2 * (n - 1)
    j = i % period
    if j >= n:
        j = period - j
    return j

class ShortTimeFFT:
    _win: np.ndarray[float, 1]
    _dual_win: Optional[np.ndarray[float, 1]]
    _hop: int
    _fs: float
    _fft_mode: str
    _mfft: int
    _scaling: str          # '' for None
    _phase_shift: Optional[int]
    _fac_mag: Optional[float]
    _fac_psd: Optional[float]
    _lower_border_end: Optional[Tuple[int, int]]

    def __init__(self, win, hop: int, fs: float, *, fft_mode: str = 'onesided', mfft: Optional[int] = None,
                 dual_win = None, scale_to: Optional[str] = None, phase_shift: Optional[int] = 0):
        w = np.asarray(win).astype(float)
        if w.ndim != 1 or w.size == 0:
            raise ValueError(f"Parameter win must be 1d, but win.shape={w.shape}!")
        for i in range(w.shape[0]):
            if not math.isfinite(w[i]):
                raise ValueError("Parameter win must have finite entries!")
        if hop < 1:
            raise ValueError(f"Parameter hop={hop} is not an integer >= 1!")
        self._win = w.copy()
        self._hop = hop
        self._fs = 1.0
        self._fac_mag = None
        self._fac_psd = None
        self._lower_border_end = None
        self._scaling = ''
        self._fft_mode = 'onesided'
        self._mfft = w.shape[0]
        self._phase_shift = 0
        self._dual_win = None
        self.fs = fs
        if mfft is not None:
            self.mfft = mfft
        if dual_win is not None:
            dw = np.asarray(dual_win).astype(float)
            if dw.ndim != 1 or dw.shape[0] != w.shape[0]:
                raise ValueError(f"dual_win.shape={dw.shape} must equal win.shape={w.shape}!")
            for i in range(dw.shape[0]):
                if not math.isfinite(dw[i]):
                    raise ValueError("Parameter dual_win must be a finite array!")
            self._dual_win = dw.copy()
        if scale_to is not None:
            self.scale_to(scale_to)
        self.fft_mode = fft_mode
        self.phase_shift = phase_shift

    @staticmethod
    def from_dual(dual_win, hop: int, fs: float, *, fft_mode: str = 'onesided', mfft: Optional[int] = None,
                  scale_to: Optional[str] = None, phase_shift: Optional[int] = 0):
        dw = np.asarray(dual_win).astype(float)
        win = _calc_dual_canonical_window(dw, hop)
        return ShortTimeFFT(win, hop, fs, fft_mode=fft_mode, mfft=mfft, dual_win=dw, scale_to=scale_to,
                            phase_shift=phase_shift)

    @staticmethod
    def from_window(win_param, fs: float, nperseg: int, noverlap: int, *, symmetric_win: bool = False,
                    fft_mode: str = 'onesided', mfft: Optional[int] = None, scale_to: Optional[str] = None,
                    phase_shift: Optional[int] = 0):
        win = get_window(win_param, nperseg, not symmetric_win)
        return ShortTimeFFT(win, nperseg - noverlap, fs, fft_mode=fft_mode, mfft=mfft, scale_to=scale_to,
                            phase_shift=phase_shift)

    @staticmethod
    def from_win_equals_dual(desired_win, hop: int, fs: float, *, fft_mode: str = 'onesided',
                             mfft: Optional[int] = None, scale_to: Optional[str] = None,
                             phase_shift: Optional[int] = 0):
        dwin = np.asarray(desired_win).astype(float)
        if dwin.ndim != 1 or dwin.size == 0:
            raise ValueError(f"Parameter desired_win is not 1d, but desired_win.shape={dwin.shape}!")
        n = dwin.shape[0]
        for i in range(n):
            if not math.isfinite(dwin[i]):
                raise ValueError("Parameter desired_win must have finite entries!")
        if not (1 <= hop and hop <= n):
            raise ValueError(f"Parameter hop={hop} is not an integer between 1 and len(desired_win)={n}!")
        if scale_to is not None and scale_to != 'magnitude' and scale_to != 'psd' and scale_to != 'unitary':
            raise ValueError(f"Parameter scale_to={scale_to} not in ['magnitude', 'psd', 'unitary', None]!")
        mf = n if mfft is None else mfft
        s_fac = math.sqrt(float(mf)) if scale_to == 'unitary' else 1.0
        win = dwin.copy()
        rr = 1e-15 * dwin.max()
        for m in range(hop):
            a = 0.0
            i = m
            while i < n:
                a += dwin[i] * dwin[i]
                i += hop
            a = math.sqrt(a)
            if not (a > rr):
                raise ValueError(f"Parameter desired_win does not have valid STFT dual window for hop={hop}!")
            i = m
            while i < n:
                win[i] /= a
                i += hop
        sc: Optional[str] = scale_to
        if scale_to == 'unitary':
            sc = None
        sft = ShortTimeFFT(win / s_fac, hop, fs, fft_mode=fft_mode, mfft=mf, dual_win=win * s_fac,
                           scale_to=sc, phase_shift=phase_shift)
        if scale_to == 'unitary':
            sft._scaling = 'unitary'
        return sft

    # ---- properties
    @property
    def win(self) -> np.ndarray[float, 1]:
        return self._win

    @property
    def hop(self) -> int:
        return self._hop

    @property
    def T(self) -> float:
        return 1.0 / self._fs

    @T.setter
    def T(self, v: float):
        if not (v > 0):
            raise ValueError(f"Sampling interval T={v} must be positive!")
        self._fs = 1.0 / v

    @property
    def fs(self) -> float:
        return self._fs

    @fs.setter
    def fs(self, v: float):
        if not (v > 0):
            raise ValueError(f"Sampling frequency fs={v} must be positive!")
        self._fs = v

    @property
    def fft_mode(self) -> str:
        return self._fft_mode

    @fft_mode.setter
    def fft_mode(self, t: str):
        if t != 'twosided' and t != 'centered' and t != 'onesided' and t != 'onesided2X':
            raise ValueError(f"fft_mode='{t}' not in ('twosided', 'centered', 'onesided', 'onesided2X')!")
        if t == 'onesided2X' and self._scaling == '':
            raise ValueError(f"For scaling is None, fft_mode='{t}' is invalid!Do scale_to('psd') or scale_to('magnitude')!")
        self._fft_mode = t

    @property
    def mfft(self) -> int:
        return self._mfft

    @mfft.setter
    def mfft(self, n_: int):
        if not (n_ >= self.m_num):
            raise ValueError(f"Attribute mfft={n_} needs to be at least the window length m_num={self.m_num}!")
        self._mfft = n_

    @property
    def scaling(self) -> Optional[str]:
        if self._scaling == '':
            return None
        return self._scaling

    @property
    def phase_shift(self) -> Optional[int]:
        return self._phase_shift

    @phase_shift.setter
    def phase_shift(self, v: Optional[int]):
        if v is None:
            self._phase_shift = None
            return
        vv: int = v
        if not (-self.mfft < vv and vv < self.mfft):
            raise ValueError(f"-mfft < phase_shift < mfft does not hold for mfft={self.mfft}, phase_shift={vv}!")
        self._phase_shift = vv

    def scale_to(self, scaling: str):
        if scaling != 'magnitude' and scaling != 'psd':
            raise ValueError(f"scaling={scaling} not in {{'magnitude', 'psd'}}!")
        if self._scaling == scaling:
            return
        s_fac = self.fac_psd if scaling == 'psd' else self.fac_magnitude
        self._win = self._win * s_fac
        if self._dual_win is not None:
            self._dual_win = self._dual_win / s_fac
        self._fac_mag = None
        self._fac_psd = None
        self._scaling = scaling

    @property
    def fac_magnitude(self) -> float:
        if self._scaling == 'magnitude':
            return 1.0
        if self._fac_mag is None:
            self._fac_mag = 1.0 / abs(float(self._win.sum()))
        return self._fac_mag

    @property
    def fac_psd(self) -> float:
        if self._scaling == 'psd':
            return 1.0
        if self._fac_psd is None:
            self._fac_psd = 1.0 / math.sqrt(float((self._win * self._win).sum()) / self.T)
        return self._fac_psd

    @property
    def m_num(self) -> int:
        return self._win.shape[0]

    @property
    def m_num_mid(self) -> int:
        return self.m_num // 2

    @property
    def delta_t(self) -> float:
        return self.T * self._hop

    @property
    def delta_f(self) -> float:
        return 1.0 / (self._mfft * self.T)

    @property
    def onesided_fft(self) -> bool:
        return self._fft_mode == 'onesided' or self._fft_mode == 'onesided2X'

    @property
    def f_pts(self) -> int:
        return self._mfft // 2 + 1 if self.onesided_fft else self._mfft

    @property
    def f(self) -> np.ndarray[float, 1]:
        if self.onesided_fft:
            return np.fft.rfftfreq(self._mfft, self.T)
        if self._fft_mode == 'twosided':
            return np.fft.fftfreq(self._mfft, self.T)
        return np.fft.fftshift(np.fft.fftfreq(self._mfft, self.T))

    @property
    def dual_win(self) -> np.ndarray[float, 1]:
        if self._dual_win is None:
            self._dual_win = _calc_dual_canonical_window(self._win, self._hop)
        return self._dual_win

    @property
    def invertible(self) -> bool:
        if self._dual_win is not None:
            return True
        try:
            self._dual_win = _calc_dual_canonical_window(self._win, self._hop)
            return True
        except ValueError:
            return False

    # ---- slice bookkeeping
    def _pre_padding(self) -> Tuple[int, int]:
        n = self.m_num
        n0 = -self.m_num_mid
        p_ = 0
        k = n0
        while k >= n0 - n - 1:
            n_ = k
            n_next = n_ - self._hop
            ok = n_next + n <= 0
            if not ok:
                zero = True
                for i in range(n + n_next, n):
                    if self._win[i] != 0.0:
                        zero = False
                ok = zero
            if ok:
                return n_, -p_
            p_ += 1
            k -= self._hop
        raise RuntimeError("This code line should never run! Please file a bug.")

    @property
    def k_min(self) -> int:
        return self._pre_padding()[0]

    @property
    def p_min(self) -> int:
        return self._pre_padding()[1]

    def _post_padding(self, n: int) -> Tuple[int, int]:
        m2p = self.m_num - self.m_num_mid
        if n < m2p:
            raise ValueError(f"Parameter n must be >= ceil(m_num/2) = {m2p}!")
        q1 = n // self._hop
        k1 = q1 * self._hop - self.m_num_mid
        q_ = q1
        k_ = k1
        while k_ < n + self.m_num:
            n_next = k_ + self._hop
            ok = n_next >= n
            if not ok:
                zero = True
                for i in range(min(n - n_next, self.m_num)):
                    if self._win[i] != 0.0:
                        zero = False
                ok = zero
            if ok:
                return k_ + self.m_num, q_ + 1
            q_ += 1
            k_ += self._hop
        raise RuntimeError("This code line should never run! Please file a bug.")

    def k_max(self, n: int) -> int:
        return self._post_padding(n)[0]

    def p_max(self, n: int) -> int:
        return self._post_padding(n)[1]

    def p_num(self, n: int) -> int:
        return self.p_max(n) - self.p_min

    @property
    def lower_border_end(self) -> Tuple[int, int]:
        if self._lower_border_end is not None:
            return self._lower_border_end
        m0 = 0
        for i in range(self.m_num):
            if self._win[i] != 0.0:
                m0 = i
                break
        k0 = -self.m_num_mid + m0
        q_ = 0
        k_ = k0
        while k_ < self._hop + 1:
            if k_ + self._hop >= 0:
                self._lower_border_end = (k_ + self.m_num, q_ + 1)
                return (k_ + self.m_num, q_ + 1)
            q_ += 1
            k_ += self._hop
        r = (0, max(self.p_min, 0))
        self._lower_border_end = r
        return r

    def upper_border_begin(self, n: int) -> Tuple[int, int]:
        m2p = self.m_num - self.m_num_mid
        if n < m2p:
            raise ValueError(f"Parameter n must be >= ceil(m_num/2) = {m2p}!")
        q2 = n // self._hop + 1
        q1 = max((n - self.m_num) // self._hop - 1, -1)
        q_ = q2
        while q_ > q1:
            k_ = q_ * self._hop + (self.m_num - self.m_num_mid)
            ok = k_ <= n
            if not ok:
                zero = True
                st = n - k_
                if st < 0:
                    st = max(self.m_num + st, 0)
                for i in range(st, self.m_num):
                    if self._win[i] != 0.0:
                        zero = False
                ok = zero
            if ok:
                return (q_ + 1) * self._hop - self.m_num_mid, q_ + 1
            q_ -= 1
        raise RuntimeError("This code line should never run! Please file a bug.")

    def p_range(self, n: int, p0: Optional[int] = None, p1: Optional[int] = None) -> Tuple[int, int]:
        p_max = self.p_max(n)
        p0_ = self.p_min if p0 is None else p0
        p1_ = p_max if p1 is None else p1
        if not (self.p_min <= p0_ and p0_ < p1_ and p1_ <= p_max):
            raise ValueError(f"Invalid Parameter p0={p0}, p1={p1}, i.e., self.p_min={self.p_min} <= p0 < p1 <= p_max={p_max} does not hold for signal length n={n}!")
        return p0_, p1_

    def t(self, n: int, p0: Optional[int] = None, p1: Optional[int] = None, k_offset: int = 0) -> np.ndarray[float, 1]:
        if n <= 0:
            raise ValueError(f"Parameter n={n} is not a positive integer!")
        r = self.p_range(n, p0, p1)
        out = np.empty(r[1] - r[0])
        for i in range(r[1] - r[0]):
            out[i] = (r[0] + i) * self.delta_t + k_offset * self.T
        return out

    def nearest_k_p(self, k: int, left: bool = True) -> int:
        p_q = k // self._hop
        rem = k - p_q * self._hop
        if rem == 0:
            return k
        return p_q * self._hop if left else (p_q + 1) * self._hop

    def extent(self, n: int, axes_seq: str = 'tf', center_bins: bool = False) -> Tuple[float, float, float, float]:
        if axes_seq != 'tf' and axes_seq != 'ft':
            raise ValueError(f"Parameter axes_seq={axes_seq} not in ['tf', 'ft']!")
        q0 = 0
        q1 = 0
        if self.onesided_fft:
            q0 = 0
            q1 = self.f_pts
        elif self._fft_mode == 'centered':
            q0 = -(self._mfft // 2)
            q1 = self._mfft // 2 if self._mfft % 2 == 0 else self._mfft // 2 + 1
        else:
            raise ValueError(f"Attribute fft_mode={self._fft_mode} must be in ['centered', 'onesided', 'onesided2X']")
        p0 = self.p_min
        p1 = self.p_max(n)
        if center_bins:
            t0 = self.delta_t * (p0 - 0.5)
            t1 = self.delta_t * (p1 - 0.5)
            f0 = self.delta_f * (q0 - 0.5)
            f1 = self.delta_f * (q1 - 0.5)
        else:
            t0 = self.delta_t * p0
            t1 = self.delta_t * p1
            f0 = self.delta_f * q0
            f1 = self.delta_f * q1
        if axes_seq == 'tf':
            return (t0, t1, f0, f1)
        return (f0, f1, t0, t1)

    # ---- transforms
    def _fft_func(self, x0: np.ndarray[float, 1]) -> np.ndarray[complex, 1]:
        mfft = self._mfft
        x = x0
        if self._phase_shift is not None:
            xp = np.zeros(max(mfft, x0.shape[0]))
            for i in range(x0.shape[0]):
                xp[i] = x0[i]
            ps = (self._phase_shift + self.m_num_mid) % self.m_num
            x = np.roll(xp, -ps)
        mode = self._fft_mode
        if mode == 'twosided':
            return np.fft.fft(x, mfft)
        if mode == 'centered':
            return np.fft.fftshift(np.fft.fft(x, mfft))
        X = np.fft.rfft(x, mfft)
        if mode == 'onesided2X':
            fac = math.sqrt(2.0) if self._scaling == 'psd' else 2.0
            hi = X.shape[0] - 1 if mfft % 2 == 0 else X.shape[0]
            for i in range(1, hi):
                X[i] = X[i] * fac
        return X

    def _ifft_func(self, X: np.ndarray[complex, 1]) -> np.ndarray[complex, 1]:
        mfft = self._mfft
        mode = self._fft_mode
        xc = np.zeros(mfft, dtype=complex)
        if mode == 'twosided':
            xc = np.fft.ifft(X, mfft)
        elif mode == 'centered':
            xc = np.fft.ifft(np.fft.ifftshift(X), mfft)
        else:
            Xc = X.copy()
            if mode == 'onesided2X':
                fac = math.sqrt(2.0) if self._scaling == 'psd' else 2.0
                hi = Xc.shape[0] - 1 if mfft % 2 == 0 else Xc.shape[0]
                for i in range(1, hi):
                    Xc[i] = Xc[i] / fac
            xr = np.fft.irfft(Xc, mfft)
            for i in range(mfft):
                xc[i] = complex(xr[i], 0.0)
        out = np.zeros(self.m_num, dtype=complex)
        if self._phase_shift is None:
            for i in range(self.m_num):
                out[i] = xc[i]
            return out
        ps = (self._phase_shift + self.m_num_mid) % self.m_num
        xr2 = np.roll(xc, ps)
        for i in range(self.m_num):
            out[i] = xr2[i]
        return out

    def _slice(self, x: np.ndarray[float, 1], k_off: int, p: int, p0: int, mode: int) -> np.ndarray[float, 1]:
        # window-length slice number (p - p0) of x (padded), starting sample
        # k = p * hop - m_num_mid + k_off
        n = x.shape[0]
        k0 = p * self._hop - self.m_num_mid + k_off
        out = np.empty(self.m_num)
        for i in range(self.m_num):
            j = _pad_index(k0 + i, n, mode)
            out[i] = 0.0 if j < 0 else x[j]
        return out

    def stft_detrend(self, x, detr = None, p0: Optional[int] = None, p1: Optional[int] = None, *,
                     k_offset: int = 0, padding: str = 'zeros'):
        xa = np.asarray(x).astype(float)
        if xa.ndim != 1:
            raise ValueError("only 1-D signals are supported")
        mode = 0
        if padding == 'zeros':
            mode = 0
        elif padding == 'edge':
            mode = 1
        elif padding == 'even':
            mode = 2
        elif padding == 'odd':
            raise ValueError("padding='odd' is not supported")
        else:
            raise ValueError(f"Parameter padding={padding} not in ('zeros', 'edge', 'even', 'odd')!")
        n = xa.shape[0]
        m2p = self.m_num - self.m_num_mid
        if n < m2p:
            raise ValueError(f"len(x)={n} must be >= ceil(m_num/2) = {m2p}!")
        r = self.p_range(n, p0, p1)
        S = np.zeros((self.f_pts, r[1] - r[0]), dtype=complex)
        for p_ in range(r[0], r[1]):
            xs = self._slice(xa, k_offset, p_, r[0], mode)
            if isinstance(detr, str):
                xs = detrend(xs, -1, detr)
            elif not isinstance(detr, NoneType):
                xs = np.asarray(detr(xs)).astype(float)
            X = self._fft_func(xs * self._win)
            for f in range(self.f_pts):
                S[f, p_ - r[0]] = X[f]
        return S

    def stft(self, x, p0: Optional[int] = None, p1: Optional[int] = None, *, k_offset: int = 0,
             padding: str = 'zeros'):
        return self.stft_detrend(x, None, p0, p1, k_offset=k_offset, padding=padding)

    def spectrogram(self, x, y = None, detr = None, *, p0: Optional[int] = None, p1: Optional[int] = None,
                    k_offset: int = 0, padding: str = 'zeros'):
        Sx = self.stft_detrend(x, detr, p0, p1, k_offset=k_offset, padding=padding)
        if isinstance(y, NoneType):
            out = np.empty(Sx.shape)
            for i in range(Sx.shape[0]):
                for j in range(Sx.shape[1]):
                    out[i, j] = Sx[i, j].real ** 2 + Sx[i, j].imag ** 2
            return out
        Sy = self.stft_detrend(y, detr, p0, p1, k_offset=k_offset, padding=padding)
        return Sx * Sy.conj()

    def istft_complex(self, S, k0: int = 0, k1: Optional[int] = None):
        Sa = np.asarray(S).astype(complex)
        if Sa.ndim != 2:
            raise ValueError("only 2-D (f, t) spectrograms are supported")
        if Sa.shape[0] != self.f_pts:
            raise ValueError(f"S.shape[-2]={Sa.shape[0]} must be equal to self.f_pts={self.f_pts} (S.shape={Sa.shape})!")
        n_min = self.m_num - self.m_num_mid
        q_num = self.p_num(n_min)
        if Sa.shape[1] < q_num:
            raise ValueError(f"S.shape[-1]={Sa.shape[1]} needs to have at least {q_num} slices (S.shape={Sa.shape})!")
        p_min = self.p_min
        q_max = Sa.shape[1] + p_min
        k_max = (q_max - 1) * self._hop + self.m_num - self.m_num_mid
        k1v = k_max if k1 is None else k1
        if not (self.k_min <= k0 and k0 < k1v and k1v <= k_max):
            raise ValueError(f"(self.k_min={self.k_min}) <= (k0={k0}) < (k1={k1v}) <= (k_max={k_max}) is false!")
        num_pts = k1v - k0
        if num_pts < n_min:
            raise ValueError(f"(k1={k1v}) - (k0={k0}) = {num_pts} has to be at least the half the window length {n_min}!")
        q0 = k0 // self._hop + p_min if k0 >= 0 else k0 // self._hop
        q1 = min(self.p_max(k1v), q_max)
        k_q0 = self.nearest_k_p(k0)
        k_q1 = self.nearest_k_p(k1v, False)
        n_pts = k_q1 - k_q0 + self.m_num - self.m_num_mid
        x = np.zeros(n_pts, dtype=complex)
        dw = self.dual_win
        for q_ in range(q0, q1):
            col = np.empty(self.f_pts, dtype=complex)
            for f in range(self.f_pts):
                col[f] = Sa[f, q_ - p_min]
            xs = self._ifft_func(col)
            i0 = q_ * self._hop - self.m_num_mid
            i1 = min(i0 + self.m_num, n_pts + k0)
            j0 = 0
            j1 = i1 - i0
            if i0 < k0:
                j0 += k0 - i0
                i0 = k0
            for j in range(j0, j1):
                x[i0 - k0 + (j - j0)] += xs[j] * dw[j]
        out = np.empty(k1v - k0, dtype=complex)
        for i in range(k1v - k0):
            out[i] = x[i]
        return out

    def istft(self, S, k0: int = 0, k1: Optional[int] = None):
        xc = self.istft_complex(S, k0, k1)
        out = np.empty(xc.shape[0])
        for i in range(xc.shape[0]):
            out[i] = xc[i].real
        return out
