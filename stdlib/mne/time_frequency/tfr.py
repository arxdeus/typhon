# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne/time_frequency/tfr.py array API: morlet, fwhm, _make_dpss, cwt,
# tfr_array_morlet, tfr_array_multitaper (via _compute_tfr) and
# mne/time_frequency/_stockwell.py tfr_array_stockwell.

import numpy as np
from scipy.fft import fft as _fft, ifft as _ifft, fftfreq as _fftfreq
from .multitaper import dpss_windows


def _next_fast_len(target: int) -> int:
    """mne.filter.next_fast_len: smallest 5-smooth number >= target."""
    if target <= 6:
        return target
    n = target
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def _cnorm(w: np.ndarray[complex, 1]) -> float:
    # 2-norm of a complex vector (np.linalg.norm on complex input is not reliable here)
    s = 0.0
    for v in w:
        s += v.real * v.real + v.imag * v.imag
    return np.sqrt(s)


class _Decim:
    start: int
    stop: Optional[int]
    step: int

    def __init__(self, start: int, stop: Optional[int], step: int):
        self.start = start
        self.stop = stop
        self.step = step

    def indices(self, n: int) -> List[int]:
        stop = n if self.stop is None else min(self.stop.__val__(), n)
        return list(range(self.start, stop, self.step))


def _ensure_decim(decim) -> _Decim:
    if isinstance(decim, int):
        if decim < 1:
            raise ValueError(f"decim must be >= 1, got {decim}")
        return _Decim(0, None, decim)
    elif isinstance(decim, slice):
        st = 0 if decim.start is None else decim.start
        sp = 1 if decim.step is None else decim.step
        return _Decim(st, decim.stop, sp)
    else:
        return decim


def _n_cycles_list(n_cycles, n_freqs: int) -> List[float]:
    if isinstance(n_cycles, int) or isinstance(n_cycles, float):
        return [float(n_cycles)]
    out = [float(v) for v in np.asarray(n_cycles, dtype=float).ravel()]
    if len(out) != 1 and len(out) != n_freqs:
        raise ValueError("n_cycles should be fixed or defined for each frequency.")
    return out


def _morlet_list(sfreq: float, freqs: List[float], n_cycles: List[float],
                 sigma: Optional[float], zero_mean: bool) -> List[np.ndarray[complex, 1]]:
    Ws = List[np.ndarray[complex, 1]]()
    for k in range(len(freqs)):
        f = freqs[k]
        if f <= 0:
            raise ValueError("all frequencies in 'freqs' must be greater than 0.")
        this_n_cycles = n_cycles[k] if len(n_cycles) != 1 else n_cycles[0]
        if sigma is None:
            sigma_t = this_n_cycles / (2.0 * np.pi * f)
        else:
            sigma_t = this_n_cycles / (2.0 * np.pi * sigma.__val__())
        t = np.arange(0.0, 5.0 * sigma_t, 1.0 / sfreq)
        nt = t.shape[0]
        tt = np.empty(2 * nt - 1)
        for i in range(nt):
            tt[i] = -t[nt - 1 - i]
        for i in range(1, nt):
            tt[nt - 1 + i] = t[i]
        osc = np.exp(2.0 * 1j * np.pi * f * tt)
        if zero_mean:
            real_offset = np.exp(-2 * (np.pi * f * sigma_t) ** 2)
            osc -= real_offset
        env = np.exp(-tt ** 2 / (2.0 * sigma_t ** 2))
        W = osc * env
        W /= np.sqrt(0.5) * _cnorm(W)
        Ws.append(W)
    return Ws


def morlet(sfreq: float, freqs, n_cycles = 7.0, sigma = None, zero_mean: bool = False):
    """Compute Morlet wavelets for the given frequency range (mne).

    Returns a list of complex arrays (a single array for a scalar freq)."""
    sg: Optional[float] = None if isinstance(sigma, NoneType) else float(sigma)
    if isinstance(freqs, int) or isinstance(freqs, float):
        return _morlet_list(float(sfreq), [float(freqs)], _n_cycles_list(n_cycles, 1), sg, zero_mean)[0]
    else:
        fl = [float(v) for v in np.asarray(freqs, dtype=float).ravel()]
        return _morlet_list(float(sfreq), fl, _n_cycles_list(n_cycles, len(fl)), sg, zero_mean)


def fwhm(freq, n_cycles):
    """Compute the full-width half maximum of a Morlet wavelet (mne)."""
    return n_cycles * np.sqrt(2 * np.log(2)) / (np.pi * freq)


def _make_dpss(sfreq: float, freqs: List[float], n_cycles: List[float], time_bandwidth: float,
               zero_mean: bool) -> Tuple[List[List[np.ndarray[complex, 1]]], np.ndarray[float, 2]]:
    for f in freqs:
        if f <= 0:
            raise ValueError("all frequencies in 'freqs' must be greater than 0.")
    if time_bandwidth < 2.0:
        raise ValueError("time_bandwidth should be >= 2.0 for good tapers")
    n_taps = int(np.floor(time_bandwidth - 1))
    Ws = List[List[np.ndarray[complex, 1]]]()
    Cs = np.zeros((n_taps, len(freqs)))
    for m in range(n_taps):
        Wm = List[np.ndarray[complex, 1]]()
        for k in range(len(freqs)):
            f = freqs[k]
            this_n_cycles = n_cycles[k] if len(n_cycles) != 1 else n_cycles[0]
            t_win = this_n_cycles / float(f)
            t = np.arange(0.0, t_win, 1.0 / sfreq)
            osc = np.exp(2.0 * 1j * np.pi * f * (t - t_win / 2.0))
            tapers, conc = dpss_windows(t.shape[0], time_bandwidth / 2.0, n_taps, sym=False)
            Wk = osc * tapers[m]
            if zero_mean:
                Wk -= Wk.mean()
            Wk /= np.sqrt(0.5) * _cnorm(Wk)
            Wm.append(Wk)
            Cs[m, k] = np.sqrt(conc[m])
        Ws.append(Wm)
    return Ws, Cs


def _cwt_one(x: np.ndarray[float, 1], W: np.ndarray[complex, 1], fft_x: np.ndarray[complex, 1],
             fft_W: np.ndarray[complex, 1], use_fft: bool, n_times: int) -> np.ndarray[complex, 1]:
    """'same'-mode convolution of x with W (centered n_times samples)."""
    if use_fft:
        full = _ifft(fft_x * fft_W)
        nfull = n_times + W.shape[0] - 1
        start = (nfull - n_times) // 2
        return full[start:start + n_times].copy()
    else:
        re = np.convolve(x, W.real.copy(), mode="same")
        im = np.convolve(x, W.imag.copy(), mode="same")
        return re + 1j * im


def _tf_loop(X: np.ndarray[float, 2], Ws: List[List[np.ndarray[complex, 1]]], output: str,
             use_fft: bool, decim: _Decim, weights: Optional[np.ndarray[float, 2]]):
    """_time_frequency_loop for one channel. X: (n_epochs, n_times).

    Returns complex array shaped by output:
      avg/itc: (n_freqs, n_out); per-taper complex/phase: (n_epochs, n_tapers, n_freqs, n_out);
      else (n_epochs, n_freqs, n_out)."""
    n_epochs = X.shape[0]
    n_times = X.shape[1]
    tidx = np.array(decim.indices(n_times), dtype=int)
    n_out = tidx.shape[0]
    n_tapers = len(Ws)
    n_freqs = len(Ws[0])
    is_avg = output.startswith("avg_") or output == "itc"
    per_taper = (output == "complex" or output == "phase") and weights is not None
    if is_avg:
        tfrs = np.zeros((1, 1, n_freqs, n_out), dtype=complex)
    elif per_taper:
        tfrs = np.zeros((n_epochs, n_tapers, n_freqs, n_out), dtype=complex)
    else:
        tfrs = np.zeros((n_epochs, 1, n_freqs, n_out), dtype=complex)
    for taper_idx in range(n_tapers):
        W = Ws[taper_idx]
        max_size = 0
        for w in W:
            max_size = max(max_size, w.shape[0])
        nfft = _next_fast_len(n_times + max_size - 1)
        fft_Ws = List[np.ndarray[complex, 1]]()
        if use_fft:
            for w in W:
                fft_Ws.append(_fft(w, nfft))
        plf = np.zeros((n_freqs, n_out), dtype=complex)
        for e in range(n_epochs):
            x = X[e].copy()
            fft_x = _fft(x, nfft) if use_fft else np.zeros(0, dtype=complex)
            for ii in range(n_freqs):
                fw = fft_Ws[ii] if use_fft else np.zeros(0, dtype=complex)
                ret = _cwt_one(x, W[ii], fft_x, fw, use_fft, n_times)
                tfr = ret[tidx]
                if output != "complex" and output != "phase" and weights is not None:
                    tfr = tfr * weights.__val__()[taper_idx, ii]
                if output == "power" or output == "avg_power":
                    val = (tfr * tfr.conj()).real.astype(complex)
                elif output == "phase":
                    val = np.angle(tfr).astype(complex)
                elif output == "avg_power_itc":
                    tfr_abs = np.abs(tfr)
                    plf[ii] += tfr / tfr_abs
                    val = (tfr_abs ** 2).astype(complex)
                elif output == "itc":
                    plf[ii] += tfr / np.abs(tfr)
                    continue
                else:
                    val = tfr
                if is_avg:
                    tfrs[0, 0, ii] += val
                elif per_taper:
                    tfrs[e, taper_idx, ii] += val
                else:
                    tfrs[e, 0, ii] += val
        if output == "avg_power_itc":
            tfrs[0, 0] += 1j * np.abs(plf)
        elif output == "itc":
            tfrs[0, 0] += np.abs(plf).astype(complex)
    if is_avg:
        tfrs /= n_epochs
    if n_tapers > 1 and output != "complex" and output != "phase" and output != "itc":
        w = weights.__val__()
        for ii in range(n_freqs):
            s = 0.0
            for m in range(n_tapers):
                s += w[m, ii] * w[m, ii]
            sc = 2.0 / s
            for a in range(tfrs.shape[0]):
                for b in range(tfrs.shape[1]):
                    for t in range(n_out):
                        v = tfrs[a, b, ii, t]
                        im = v.imag / n_tapers if output == "avg_power_itc" else v.imag
                        tfrs[a, b, ii, t] = complex(v.real * sc, im)
    return tfrs


def _check_output(output: str):
    if output not in ("complex", "power", "phase", "avg_power_itc", "avg_power", "itc"):
        raise ValueError(f"Invalid value for the 'output' parameter. Allowed values are 'complex', 'power', 'phase', 'avg_power_itc', 'avg_power', and 'itc', but got '{output}' instead.")


def _compute_tfr_c(data, freqs, sfreq: float, method: str, n_cycles, zero_mean: bool,
                   time_bandwidth: float, use_fft: bool, decim, output: str):
    """Complex-valued core of mne _compute_tfr.

    Returns (out, weights) with out of shape
      (n_chans, n_freqs, n_out) for avg_*/itc,
      (n_epochs, n_chans, n_tapers, n_freqs, n_out) for multitaper complex/phase,
      (n_epochs, n_chans, n_freqs, n_out) otherwise (as 5-D with taper axis 1)."""
    _check_output(output)
    if method != "morlet" and method != "multitaper":
        raise ValueError(f"Invalid value for the 'method' parameter. Allowed values are 'multitaper' and 'morlet', but got '{method}' instead.")
    ed = np.asarray(data, dtype=float)
    if ed.ndim != 3:
        raise ValueError(f"epoch_data must be of shape (n_epochs, n_chans, n_times), got {ed.shape}")
    fl = [float(v) for v in np.asarray(freqs, dtype=float).ravel()]
    for f in fl:
        if f > sfreq / 2.0:
            raise ValueError(f"Cannot compute freq above Nyquist freq of the data ({sfreq / 2.0:0.1f} Hz), got {max(fl):0.1f} Hz")
    if not (isinstance(n_cycles, int) or isinstance(n_cycles, float)):
        if len(n_cycles) != len(fl):
            raise ValueError(f"n_cycles must be a float or an array of length {len(fl)} frequencies, got {len(n_cycles)} cycles instead.")
    nc = _n_cycles_list(n_cycles, len(fl))
    dec = _ensure_decim(decim)
    weights: Optional[np.ndarray[float, 2]] = None
    if method == "morlet":
        Ws = [_morlet_list(sfreq, fl, nc, None, zero_mean)]
    else:
        Ws, w = _make_dpss(sfreq, fl, nc, time_bandwidth, zero_mean)
        weights = w
    if Ws[0][0].shape[0] > ed.shape[2]:
        raise ValueError(f"At least one of the wavelets is longer than the signal ({Ws[0][0].shape[0]} > {ed.shape[2]} samples). Use a longer signal or shorter wavelets.")
    n_epochs = ed.shape[0]
    n_chans = ed.shape[1]
    n_out = len(dec.indices(ed.shape[2]))
    n_freqs = len(fl)
    n_tapers = len(Ws)
    is_avg = output.startswith("avg_") or output == "itc"
    per_taper = (output == "complex" or output == "phase") and method == "multitaper"
    if is_avg:
        out = np.empty((1, n_chans, 1, n_freqs, n_out), dtype=complex)
    elif per_taper:
        out = np.empty((n_epochs, n_chans, n_tapers, n_freqs, n_out), dtype=complex)
    else:
        out = np.empty((n_epochs, n_chans, 1, n_freqs, n_out), dtype=complex)
    for c in range(n_chans):
        X = np.ascontiguousarray(ed[:, c, :])
        r = _tf_loop(X, Ws, output, use_fft, dec, weights)
        if is_avg:
            out[0, c] = r[0]
        else:
            for e in range(n_epochs):
                out[e, c] = r[e]
    wout = np.zeros((0, 0)) if weights is None else weights.__val__()
    return out, wout


def _real_or_complex(out5, output: Literal[str], method: Literal[str]):
    if output == "complex":
        if method == "multitaper":
            return out5
        else:
            return out5[:, :, 0].copy()
    elif output == "phase":
        if method == "multitaper":
            return out5.real.copy()
        else:
            return out5[:, :, 0].real.copy()
    elif output == "power":
        return out5[:, :, 0].real.copy()
    elif output == "avg_power_itc":
        return out5[0, :, 0].copy()
    else:
        return out5[0, :, 0].real.copy()


def tfr_array_morlet(data, sfreq: float, freqs, n_cycles = 7.0, zero_mean: bool = True,
                     use_fft: bool = True, decim = 1, output: Literal[str] = "complex",
                     n_jobs = None, *, verbose = None):
    """Compute Time-Frequency Representation (TFR) using Morlet wavelets (mne)."""
    out5, _ = _compute_tfr_c(data, freqs, float(sfreq), "morlet", n_cycles, zero_mean, 0.0,
                             use_fft, decim, output)
    return _real_or_complex(out5, output, "morlet")


def tfr_array_multitaper(data, sfreq: float, freqs, n_cycles = 7.0, zero_mean: bool = True,
                         time_bandwidth: float = 4.0, use_fft: bool = True, decim = 1,
                         output: Literal[str] = "complex", n_jobs = None, *,
                         return_weights: Literal[bool] = False, verbose = None):
    """Compute Time-Frequency Representation (TFR) using DPSS tapers (mne)."""
    out5, w = _compute_tfr_c(data, freqs, float(sfreq), "multitaper", n_cycles, zero_mean,
                             float(time_bandwidth), use_fft, decim, output)
    res = _real_or_complex(out5, output, "multitaper")
    if return_weights and (output == "complex" or output == "phase"):
        return res, w
    else:
        return res


def cwt(X, Ws, use_fft: bool = True, mode: str = "same", decim = 1):
    """Compute time-frequency decomposition with continuous wavelet transform (mne).

    Only mode='same' is supported."""
    if mode != "same":
        raise ValueError("only mode='same' is supported")
    Xa = np.asarray(X, dtype=float)
    Wl = List[np.ndarray[complex, 1]]()
    for w in Ws:
        Wl.append(np.asarray(w, dtype=complex))
    n_times = Xa.shape[1]
    max_size = 0
    for w in Wl:
        max_size = max(max_size, w.shape[0])
    nfft = _next_fast_len(n_times + max_size - 1)
    dec = _ensure_decim(decim)
    tidx = np.array(dec.indices(n_times), dtype=int)
    out = np.empty((Xa.shape[0], len(Wl), tidx.shape[0]), dtype=complex)
    fWs = [_fft(w, nfft) for w in Wl] if use_fft else List[np.ndarray[complex, 1]]()
    for s in range(Xa.shape[0]):
        x = Xa[s].copy()
        fx = _fft(x, nfft) if use_fft else np.zeros(0, dtype=complex)
        for k in range(len(Wl)):
            fw = fWs[k] if use_fft else np.zeros(0, dtype=complex)
            out[s, k] = _cwt_one(x, Wl[k], fx, fw, use_fft, n_times)[tidx]
    return out


# ---------------------------------------------------------------------------
# Stockwell transform (mne/time_frequency/_stockwell.py)

def _precompute_st_windows(n_samp: int, start_f: int, stop_f: int, sfreq: float,
                           width: float) -> np.ndarray[complex, 2]:
    tw0 = _fftfreq(n_samp, 1.0 / sfreq) / n_samp
    tw = np.empty(n_samp)
    tw[0] = tw0[0]
    for i in range(1, n_samp):
        tw[i] = tw0[n_samp - i]
    k = width
    nf = max(stop_f - start_f, 0)
    windows = np.empty((nf, n_samp), dtype=complex)
    for i_f in range(nf):
        f = float(start_f + i_f)
        if f == 0.0:
            window = np.ones(n_samp)
        else:
            window = f / (np.sqrt(2.0 * np.pi) * k) * np.exp(-0.5 * (1.0 / k ** 2.0) * f ** 2.0 * tw ** 2.0)
        window /= window.sum()
        windows[i_f] = _fft(window)
    return windows


def tfr_array_stockwell(data, sfreq: float, fmin = None, fmax = None, n_fft = None,
                        width: float = 1.0, decim = 1, return_itc: bool = False,
                        n_jobs = None, *, verbose = None):
    """Compute power and intertrial coherence using Stockwell (S) transform (mne).

    Returns (psd, itc, freqs); itc is an empty array when return_itc=False."""
    x0 = np.asarray(data, dtype=float)
    if x0.ndim != 3:
        raise ValueError(f"data must be 3D with shape (n_epochs, n_channels, n_times), got {x0.shape}")
    dec = _ensure_decim(decim)
    n_epochs = x0.shape[0]
    n_channels = x0.shape[1]
    n_times = x0.shape[2]
    out_idx = dec.indices(n_times)
    n_out = len(out_idx)
    nf_ = 0
    if isinstance(n_fft, NoneType):
        nf_ = 1 << int(np.ceil(np.log2(n_times)))
    else:
        nfv = int(n_fft)
        is_pow2 = nfv > 0 and (nfv & (nfv - 1)) == 0
        if (not is_pow2) and n_times > nfv:
            nf_ = 1 << int(np.ceil(np.log2(n_times)))
        elif nfv < n_times:
            raise ValueError(f"n_fft cannot be smaller than signal size. Got {nfv} < {n_times}.")
        else:
            nf_ = nfv
    zero_pad = nf_ - n_times if n_times < nf_ else 0
    x = np.zeros((n_epochs, n_channels, n_times + zero_pad))
    x[:, :, :n_times] = x0
    freqs_all = _fftfreq(nf_, 1.0 / sfreq)
    if isinstance(fmin, NoneType):
        fmn = 0.0
        for v in freqs_all:
            if v > 0:
                fmn = float(v)
                break
    else:
        fmn = float(fmin)
    fmx = float(freqs_all.max()) if isinstance(fmax, NoneType) else float(fmax)
    start_f = int(np.abs(freqs_all - fmn).argmin())
    stop_f = int(np.abs(freqs_all - fmx).argmin())
    freqs = freqs_all[start_f:stop_f].copy()
    n_samp = x.shape[2]
    W = _precompute_st_windows(n_samp, start_f, stop_f, sfreq, width)
    n_freq = stop_f - start_f
    psd = np.empty((n_channels, n_freq, n_out))
    itc = np.empty((n_channels, n_freq, n_out) if return_itc else (0, 0, 0))
    oi = np.array(out_idx, dtype=int)
    for c in range(n_channels):
        xc = np.ascontiguousarray(x[:, c, :])
        X = _fft(xc)
        XX = np.concatenate((X, X), axis=-1)
        for i_f in range(n_freq):
            f = start_f + i_f
            ST = _ifft(XX[:, f:f + n_samp] * W[i_f])
            TFR = ST[:, oi]
            TFR_abs = np.abs(TFR)
            for a in range(TFR_abs.shape[0]):
                for b in range(TFR_abs.shape[1]):
                    if TFR_abs[a, b] == 0:
                        TFR_abs[a, b] = 1.0
            if return_itc:
                Tn = TFR / TFR_abs
                itc[c, i_f] = np.abs(np.mean(Tn, axis=0))
            psd[c, i_f] = np.mean(TFR_abs * TFR_abs, axis=0)
    return psd, itc, freqs


# ---------------------------------------------------------------------------
# STFT (mne/time_frequency/_stft.py)

def stft(x, wsize: int, tstep = None, verbose = None):
    """STFT Short-Term Fourier Transform using a sine window (mne).

    Returns (n_signals, wsize // 2 + 1, n_step) complex array."""
    from scipy.fft import rfft
    xa = np.asarray(x, dtype=float)
    x2 = xa.reshape((-1, xa.shape[xa.ndim - 1]))
    n_signals = x2.shape[0]
    T = x2.shape[1]
    if wsize % 4:
        raise ValueError("The window length must be a multiple of 4.")
    ts = wsize // 2 if isinstance(tstep, NoneType) else int(tstep)
    if wsize % ts or ts % 2:
        raise ValueError("The step size must be a multiple of 2 and a divider of the window length.")
    if ts > wsize / 2:
        raise ValueError("The step size must be smaller than half the window length.")
    n_step = int(np.ceil(T / float(ts)))
    n_freq = wsize // 2 + 1
    X = np.zeros((n_signals, n_freq, n_step), dtype=complex)
    if n_signals == 0:
        return X
    win = np.sin(np.arange(0.5, wsize + 0.5) / wsize * np.pi)
    win2 = win ** 2
    swin = np.zeros((n_step - 1) * ts + wsize)
    for t in range(n_step):
        swin[t * ts:t * ts + wsize] += win2
    swin = np.sqrt(wsize * swin)
    xp = np.zeros((n_signals, wsize + (n_step - 1) * ts))
    off = (wsize - ts) // 2
    xp[:, off:off + T] = x2
    for t in range(n_step):
        wwin = win / swin[t * ts:t * ts + wsize]
        frame = xp[:, t * ts:t * ts + wsize] * wwin[None, :]
        X[:, :, t] = rfft(frame)
    return X


def istft(X, tstep = None, Tx = None):
    """ISTFT Inverse Short-Term Fourier Transform using a sine window (mne).

    X: (n_signals, n_win, n_step). Returns (n_signals, Tx)."""
    from scipy.fft import irfft
    Xa = np.asarray(X, dtype=complex)
    n_win = Xa.shape[Xa.ndim - 2]
    n_step = Xa.shape[Xa.ndim - 1]
    X3 = Xa.reshape((-1, n_win, n_step))
    n_sig = X3.shape[0]
    if n_win % 2 == 0:
        raise ValueError("The number of rows of the STFT matrix must be odd.")
    wsize = 2 * (n_win - 1)
    ts = wsize // 2 if isinstance(tstep, NoneType) else int(tstep)
    if wsize % ts:
        raise ValueError("The step size must be a divider of two times the number of rows of the STFT matrix minus two.")
    if ts > wsize / 2:
        raise ValueError("The step size must be smaller than the number of rows of the STFT matrix minus one.")
    T = n_step * ts
    tx = T if isinstance(Tx, NoneType) else int(Tx)
    x = np.zeros((n_sig, T + wsize - ts))
    win = np.sin(np.arange(0.5, wsize + 0.5) / wsize * np.pi)
    swin = np.zeros(T + wsize - ts)
    for t in range(n_step):
        swin[t * ts:t * ts + wsize] += win ** 2
    swin = np.sqrt(swin / wsize)
    for t in range(n_step):
        frame = irfft(np.ascontiguousarray(X3[:, :, t]), wsize)
        frame *= (win / swin[t * ts:t * ts + wsize])[None, :]
        x[:, t * ts:t * ts + wsize] += frame
    off = (wsize - ts) // 2
    end = min(off + T + 1, x.shape[1])
    xx = x[:, off:end]
    return xx[:, :min(tx, xx.shape[1])].copy()


def stftfreq(wsize: int, sfreq = None):
    """Compute frequencies of stft transformation (mne)."""
    from scipy.fft import rfftfreq
    freqs = rfftfreq(wsize)
    if not isinstance(sfreq, NoneType):
        freqs *= float(sfreq)
    return freqs
