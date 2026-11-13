# Ported from mne 1.13.2 (BSD-3-Clause)
#
# FIR filtering (mne.filter). Only the FIR code paths are native:
# method='fir' (and its alias 'fft') with firwin or firwin2 design. IIR
# filtering raises NotImplementedError.

import math
import internal.static as static
import numpy as np
from scipy.signal import firwin as _sp_firwin, firwin2 as _sp_firwin2
from scipy.signal import minimum_phase as _sp_minimum_phase
from scipy.fft import rfft as _rfft, irfft as _irfft

_known_phases_fir = ("zero", "zero-double", "minimum", "minimum-half")
_known_fir_windows = ("hann", "hamming", "blackman")
_known_fir_designs = ("firwin", "firwin2")


def _length_factor(window: str) -> float:
    if window == "hann":
        return 3.1
    if window == "hamming":
        return 3.3
    if window == "blackman":
        return 5.0
    raise ValueError(f"Invalid value for the 'fir_window' parameter. Allowed values are 'hann', 'hamming', and 'blackman', but got '{window}' instead.")


def next_fast_len(target: int) -> int:
    """Find the next 5-smooth number >= target (mne.filter.next_fast_len)."""
    if target <= 6:
        return target
    if (target & (target - 1)) == 0:
        return target
    match = 1 << 62
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            quotient = -(-target // p35)
            p2 = 2 ** int(quotient - 1).bit_length()
            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match


def _check_option(name: str, value: str, allowed):
    if value not in allowed:
        raise ValueError(f"Invalid value for the '{name}' parameter. Allowed values are {list(allowed)}, but got '{value}' instead.")


def _to_samples(filter_length, sfreq: float, phase: str, fir_design: str) -> int:
    if isinstance(filter_length, str):
        fl = filter_length.lower()
        err_msg = f'filter_length, if a string, must be a human-readable time, e.g. "10s", or "auto", not "{fl}"'
        mult_fact = 1.0
        if fl.endswith("ms"):
            mult_fact = 1e-3
            fl = fl[:-2]
        elif len(fl) > 0 and fl[-1] == "s":
            fl = fl[:-1]
        else:
            raise ValueError(err_msg)
        v = 0.0
        try:
            v = float(fl)
        except:
            raise ValueError(err_msg)
        n = max(int(math.ceil(v * mult_fact * sfreq)), 1)
        if fir_design == "firwin":
            n += (n - 1) % 2
        return n
    else:
        return int(filter_length)


def _trans_auto_l(l_trans_bandwidth, l_freq: float) -> float:
    if isinstance(l_trans_bandwidth, NoneType):
        return 0.0
    if isinstance(l_trans_bandwidth, str):
        if l_trans_bandwidth != "auto":
            raise ValueError(f'l_trans_bandwidth must be "auto" if string, got "{l_trans_bandwidth}"')
        return min(max(0.25 * l_freq, 2.0), l_freq)
    return float(l_trans_bandwidth)


def _trans_auto_h(h_trans_bandwidth, h_freq: float, sfreq: float) -> float:
    if isinstance(h_trans_bandwidth, NoneType):
        return 0.0
    if isinstance(h_trans_bandwidth, str):
        if h_trans_bandwidth != "auto":
            raise ValueError(f'h_trans_bandwidth must be "auto" if string, got "{h_trans_bandwidth}"')
        return min(max(0.25 * h_freq, 2.0), sfreq / 2.0 - h_freq)
    return float(h_trans_bandwidth)


def _triage_filter_params(len_x: int, sfreq: float, l_freq: Optional[float], h_freq: Optional[float],
                          l_trans_bandwidth, h_trans_bandwidth, filter_length,
                          method: str, phase: str, fir_window: str, fir_design: str):
    """Returns (l_freq, h_freq, l_stop, h_stop, filter_length) like mne."""
    _check_option("phase", phase, _known_phases_fir)
    _check_option("fir_window", fir_window, _known_fir_windows)
    _check_option("fir_design", fir_design, _known_fir_designs)
    if l_freq is not None:
        if l_freq <= 0:
            raise ValueError(f"highpass frequency {l_freq} must be greater than zero")
    if h_freq is not None:
        if h_freq >= sfreq / 2.0:
            raise ValueError(f"lowpass frequency {h_freq} must be less than Nyquist ({sfreq / 2.0})")
    l_stop = 0.0
    h_stop = 0.0
    l_tb = 0.0
    h_tb = 0.0
    if l_freq is not None:
        l_tb = _trans_auto_l(l_trans_bandwidth, l_freq)
        if l_tb <= 0:
            raise ValueError(f"l_trans_bandwidth must be positive, got {l_tb}")
        l_stop = l_freq - l_tb
        if l_stop < 0:
            raise ValueError(f"Filter specification invalid: Lower stop frequency negative ({l_stop:0.2f} Hz). Increase pass frequency or reduce the transition bandwidth (l_trans_bandwidth)")
    if h_freq is not None:
        h_tb = _trans_auto_h(h_trans_bandwidth, h_freq, sfreq)
        if h_tb <= 0:
            raise ValueError(f"h_trans_bandwidth must be positive, got {h_tb}")
        h_stop = h_freq + h_tb
        if h_stop > sfreq / 2:
            raise ValueError(f"Effective band-stop frequency ({h_stop}) is too high (maximum based on Nyquist is {sfreq / 2.0})")
    n = 0
    next_pow_2 = False
    if isinstance(filter_length, str) and filter_length.lower() == "auto":
        h_check = np.inf
        l_check = np.inf
        if h_freq is not None:
            h_check = h_tb
        if l_freq is not None:
            l_check = l_tb
        mult_fact = 2.0 if fir_design == "firwin2" else 1.0
        secs = _length_factor(fir_window) * mult_fact / float(min(h_check, l_check))
        # mne goes through f"{secs}s" and float() which is exact for repr
        n = _to_samples(f"{secs}s", sfreq, phase, fir_design)
    else:
        next_pow_2 = isinstance(filter_length, str) and phase == "zero-double"
        n = _to_samples(filter_length, sfreq, phase, fir_design)
    if fir_design == "firwin" or phase == "zero":
        n += (n - 1) % 2
    if n <= 0:
        raise ValueError(f"filter_length must be positive, got {n}")
    if next_pow_2:
        n = 2 ** int(math.ceil(math.log2(n)))
        if fir_design == "firwin":
            n += (n - 1) % 2
    return l_freq, h_freq, l_stop, h_stop, n


def _check_zero_phase_length(N: int, phase: str, gain_nyq: float = 0.0) -> int:
    if N % 2 == 0:
        if phase == "zero":
            raise RuntimeError(f'filter_length must be odd if phase="zero", got {N}')
        elif phase == "zero-double" and gain_nyq == 1:
            N += 1
    return N


def _firwin_design(N: int, freq: List[float], gain: List[float], window: str, sfreq: float):
    """Construct a FIR filter using firwin (port of mne.filter._firwin_design)."""
    assert freq[0] == 0
    assert len(freq) > 1
    assert len(freq) == len(gain)
    assert N % 2 == 1
    h = np.zeros(N)
    prev_freq = freq[-1]
    prev_gain = gain[-1]
    if gain[-1] == 1:
        h[N // 2] = 1.0
    assert prev_gain == 0 or prev_gain == 1
    i = len(freq) - 2
    while i >= 0:
        this_freq = freq[i]
        this_gain = gain[i]
        assert this_gain == 0 or this_gain == 1
        if this_gain != prev_gain:
            transition = (prev_freq - this_freq) / 2.0
            this_N = int(round(_length_factor(window) / transition))
            this_N += 1 - this_N % 2
            if this_N > N:
                raise ValueError(f"The requested filter length {N} is too short for the requested {transition * sfreq:0.2f} Hz transition band, which requires {this_N} samples")
            this_h = _sp_firwin(this_N, (prev_freq + this_freq) / 2.0, window=window,
                                pass_zero=True, fs=freq[-1] * 2)
            assert this_h.shape[0] == this_N
            offset = (N - this_N) // 2
            if this_gain == 0:
                for k in range(this_N):
                    h[offset + k] -= this_h[k]
            else:
                for k in range(this_N):
                    h[offset + k] += this_h[k]
        prev_gain = this_gain
        prev_freq = this_freq
        i -= 1
    return h


def _construct_fir_filter(sfreq: float, freq: List[float], gain: List[float], filter_length: int,
                          phase: str, fir_window: str, fir_design: str):
    assert freq[0] == 0
    nfreq = [f / (sfreq / 2.0) for f in freq]
    if nfreq[0] != 0 or nfreq[-1] != 1:
        raise ValueError(f"freq must start at 0 and end an Nyquist ({sfreq / 2.0}), got {freq}")
    N = _check_zero_phase_length(filter_length, phase, gain[-1])
    h = np.zeros(1)
    if phase == "minimum-half":
        h0 = np.zeros(1)
        if fir_design == "firwin2":
            h0 = _sp_firwin2(N * 2 - 1, nfreq, gain, window=fir_window)
        else:
            h0 = _firwin_design(N * 2 - 1, nfreq, gain, fir_window, sfreq)
        h = _sp_minimum_phase(h0)
    elif phase == "minimum":
        h0 = np.zeros(1)
        if fir_design == "firwin2":
            h0 = _sp_firwin2(N, nfreq, gain, window=fir_window)
        else:
            h0 = _firwin_design(N, nfreq, gain, fir_window, sfreq)
        h = _sp_minimum_phase(h0, half=False)
    elif fir_design == "firwin2":
        h = _sp_firwin2(N, nfreq, gain, window=fir_window)
    else:
        h = _firwin_design(N, nfreq, gain, fir_window, sfreq)
    assert h.size == N
    return h


def create_filter(data, sfreq: float, l_freq, h_freq, filter_length = "auto",
                  l_trans_bandwidth = "auto", h_trans_bandwidth = "auto",
                  method: str = "fir", iir_params = None, phase: str = "zero",
                  fir_window: str = "hamming", fir_design: str = "firwin", verbose = None):
    """Create a FIR filter (method='fir' only). Returns the impulse response."""
    sfreq = float(sfreq)
    if sfreq < 0:
        raise ValueError("sfreq must be positive")
    _check_option("method", method, ("iir", "fir", "fft"))
    if method == "iir":
        raise NotImplementedError("create_filter(method='iir') returns a dict in mne; use mne.filter.construct_iir_filter in Typhon")
    if not isinstance(iir_params, NoneType):
        raise ValueError('iir_params must be None if method != "iir"')
    hf: Optional[float] = None if h_freq is None else float(h_freq)
    lf: Optional[float] = None if l_freq is None else float(l_freq)
    if hf is not None:
        if hf > sfreq / 2.0:
            raise ValueError(f"h_freq ({hf}) must be less than the Nyquist frequency {sfreq / 2.0}")
    if lf is not None:
        if lf == 0:
            lf = None
    len_x = -1
    if not isinstance(data, NoneType):
        len_x = data.shape[-1]
    freq = List[float]()
    gain = List[float]()
    fl = 0
    if lf is None and hf is None:
        _, _, _, _, fl = _triage_filter_params(len_x, sfreq, None, None, None, None, filter_length,
                                               "fir", phase, fir_window, fir_design)
        freq = [0.0, sfreq / 2.0]
        gain = [1.0, 1.0]
    elif lf is None:
        _, f_p, _, f_s, fl = _triage_filter_params(len_x, sfreq, None, hf, None, h_trans_bandwidth,
                                                   filter_length, "fir", phase, fir_window, fir_design)
        freq = [0.0, f_p.__val__(), f_s]
        gain = [1.0, 1.0, 0.0]
        if f_s != sfreq / 2.0:
            freq.append(sfreq / 2.0)
            gain.append(0.0)
    elif hf is None:
        pass_, _, stop, _, fl = _triage_filter_params(len_x, sfreq, lf, None, l_trans_bandwidth, None,
                                                      filter_length, "fir", phase, fir_window, fir_design)
        freq = [stop, pass_.__val__(), sfreq / 2.0]
        gain = [0.0, 1.0, 1.0]
        if stop != 0:
            freq = [0.0] + freq
            gain = [0.0] + gain
    else:
        if lf.__val__() < hf.__val__():
            f_p1, f_p2, f_s1, f_s2, fl = _triage_filter_params(
                len_x, sfreq, lf, hf, l_trans_bandwidth, h_trans_bandwidth, filter_length,
                "fir", phase, fir_window, fir_design)
            freq = [f_s1, f_p1.__val__(), f_p2.__val__(), f_s2]
            gain = [0.0, 1.0, 1.0, 0.0]
            if f_s2 != sfreq / 2.0:
                freq.append(sfreq / 2.0)
                gain.append(0.0)
            if f_s1 != 0:
                freq = [0.0] + freq
                gain = [0.0] + gain
        else:
            # band-stop (single band): mne calls _triage_filter_params with
            # (h_freq, l_freq) swapped, swapped bandwidths and reverse=True
            _check_option("phase", phase, _known_phases_fir)
            _check_option("fir_window", fir_window, _known_fir_windows)
            _check_option("fir_design", fir_design, _known_fir_designs)
            lo = hf.__val__()
            hi = lf.__val__()
            if lo <= 0:
                raise ValueError(f"highpass frequency {lo} must be greater than zero")
            if hi >= sfreq / 2.0:
                raise ValueError(f"lowpass frequency {hi} must be less than Nyquist ({sfreq / 2.0})")
            h_tb = _trans_auto_l(h_trans_bandwidth, lo)
            l_tb = _trans_auto_h(l_trans_bandwidth, hi, sfreq)
            if h_tb <= 0 or l_tb <= 0:
                raise ValueError("transition bandwidths must be positive")
            f_p1 = lo
            f_s1 = lo + h_tb
            f_p2 = hi
            f_s2 = hi - l_tb
            if isinstance(filter_length, str) and filter_length.lower() == "auto":
                mult_fact = 2.0 if fir_design == "firwin2" else 1.0
                secs = _length_factor(fir_window) * mult_fact / float(min(h_tb, l_tb))
                fl = _to_samples(f"{secs}s", sfreq, phase, fir_design)
            else:
                fl = _to_samples(filter_length, sfreq, phase, fir_design)
            if fir_design == "firwin" or phase == "zero":
                fl += (fl - 1) % 2
            pairs = sorted([(f_p1, 1.0), (f_s1, 0.0), (f_s2, 0.0), (f_p2, 1.0)])
            freq = [p[0] for p in pairs]
            gain = [p[1] for p in pairs]
            if freq[0] != 0:
                freq = [0.0] + freq
                gain = [1.0] + gain
            if freq[-1] != sfreq / 2.0:
                freq.append(sfreq / 2.0)
                gain.append(1.0)
    return _construct_fir_filter(sfreq, freq, gain, fl, phase, fir_window, fir_design)


def _smart_pad(x, n_pad: Tuple[int, int], pad: str = "reflect_limited"):
    """Pad a 1-D array (port of mne.cuda._smart_pad for the last axis)."""
    if n_pad[0] == 0 and n_pad[1] == 0:
        return x
    if n_pad[0] < 0 or n_pad[1] < 0:
        raise RuntimeError("n_pad must be non-negative")
    n_times = x.shape[-1]
    if pad == "reflect_limited":
        lz = max(n_pad[0] - n_times + 1, 0)
        rz = max(n_pad[1] - n_times + 1, 0)
        nl = min(n_pad[0], n_times - 1)
        nr = min(n_pad[1], n_times - 1)
        out = np.zeros(lz + nl + n_times + nr + rz, dtype=x.dtype)
        o = lz
        x0 = x[0]
        for k in range(nl):
            out[o + k] = 2 * x0 - x[nl - k]
        o += nl
        for k in range(n_times):
            out[o + k] = x[k]
        o += n_times
        xl = x[n_times - 1]
        for k in range(nr):
            out[o + k] = 2 * xl - x[n_times - 2 - k]
        return out
    elif pad == "edge":
        return np.pad(x, (n_pad[0], n_pad[1]), "edge")
    elif pad == "reflect":
        return np.pad(x, (n_pad[0], n_pad[1]), "reflect", reflect_type="odd")
    elif pad == "constant":
        return np.pad(x, (n_pad[0], n_pad[1]), "constant")
    elif pad == "mean":
        return np.pad(x, (n_pad[0], n_pad[1]), "mean")
    elif pad == "median":
        return np.pad(x, (n_pad[0], n_pad[1]), "median")
    elif pad == "symmetric":
        return np.pad(x, (n_pad[0], n_pad[1]), "symmetric")
    elif pad == "wrap":
        return np.pad(x, (n_pad[0], n_pad[1]), "wrap")
    elif pad == "minimum":
        return np.pad(x, (n_pad[0], n_pad[1]), "minimum")
    elif pad == "maximum":
        return np.pad(x, (n_pad[0], n_pad[1]), "maximum")
    elif pad == "linear_ramp":
        return np.pad(x, (n_pad[0], n_pad[1]), "linear_ramp")
    else:
        raise ValueError(f"Unsupported pad mode {pad}")


def _1d_overlap_filter(x, n_h: int, n_edge: int, phase: str, h_fft, pad: str, n_fft: int):
    x_ext = _smart_pad(x, (n_edge, n_edge), pad)
    n_x = len(x_ext)
    x_filtered = np.zeros(n_x)
    n_seg = n_fft - n_h + 1
    n_segments = int(math.ceil(n_x / float(n_seg)))
    shift = ((n_h - 1) // 2 if phase.startswith("zero") else 0) + n_edge
    seg = np.zeros(n_fft)
    for seg_idx in range(n_segments):
        start = seg_idx * n_seg
        stop = min((seg_idx + 1) * n_seg, n_x)
        for k in range(n_fft):
            seg[k] = 0.0
        for k in range(start, stop):
            seg[k - start] = x_ext[k]
        xf = _rfft(seg, n_fft)
        xf *= h_fft
        prod = _irfft(xf, n_fft)
        start_filt = max(0, start - shift)
        stop_filt = min(start - shift + n_fft, n_x)
        start_prod = max(0, shift - start)
        for k in range(stop_filt - start_filt):
            x_filtered[start_filt + k] += prod[start_prod + k]
    return x_filtered[:n_x - 2 * n_edge]


def _overlap_add_filter_2d(x, h, n_fft: Optional[int], phase: str, picks: List[int], pad: str):
    """In-place overlap-add filtering of selected rows of 2-D float x."""
    nh = h.shape[0]
    _check_zero_phase_length(nh, phase)
    if nh == 1:
        hv = h[0] * h[0] if phase == "zero-double" else h[0]
        for p in picks:
            x[p, :] = x[p, :] * hv
        return
    n_edge = max(min(nh, x.shape[1]) - 1, 0)
    n_x = x.shape[1] + 2 * n_edge
    hh = h
    if phase == "zero-double":
        hh = np.convolve(h, h[::-1])
        nh = hh.shape[0]
    min_fft = 2 * nh - 1
    nf = 0
    if n_fft is None:
        max_fft = n_x
        if max_fft >= min_fft:
            lo = int(math.ceil(math.log2(min_fft)))
            hi = int(math.ceil(math.log2(max_fft)))
            best_cost = np.inf
            for e in range(lo, hi + 1):
                N = 1 << e
                cost = math.ceil(n_x / float(N - nh + 1)) * N * (math.log2(N) + 1)
                cost += 4e-5 * N * n_x
                if cost < best_cost:
                    best_cost = cost
                    nf = N
        else:
            nf = next_fast_len(min_fft)
    else:
        nf = n_fft
    if nf < min_fft:
        raise ValueError(f"n_fft is too short, has to be at least 2 * len(h) - 1 ({min_fft}), got {nf}")
    h_fft = _rfft(hh, nf)
    for p in picks:
        x[p, :] = _1d_overlap_filter(x[p, :], nh, n_edge, phase, h_fft, pad, nf)


def _filter_inplace_3d(data, sfreq: float, l_freq, h_freq, picks: List[int], filter_length,
                       l_trans_bandwidth, h_trans_bandwidth, method: str, iir_params,
                       phase: str, fir_window: str, fir_design: str, pad: str):
    """Filter (n_epochs, n_ch, n_times) or (n_ch, n_times) float data in place."""
    if method == "iir":
        p = _create_iir(sfreq, l_freq, h_freq, l_trans_bandwidth, h_trans_bandwidth,
                        iir_params, phase)
        if data.ndim == 3:
            n_ep = data.shape[0]
            n_ch = data.shape[1]
            x2 = data.reshape((n_ep * n_ch, data.shape[2]))
            allp = List[int]()
            for e in range(n_ep):
                for pp in picks:
                    allp.append(e * n_ch + pp)
            _iir_filter_2d(x2, p, allp, phase)
        elif data.ndim == 2:
            _iir_filter_2d(data, p, picks, phase)
        else:
            compile_error("data must be 2D or 3D")
        return
    filt = create_filter(data, sfreq, l_freq, h_freq, filter_length, l_trans_bandwidth,
                         h_trans_bandwidth, method, None, phase, fir_window, fir_design)
    if data.ndim == 3:
        n_ep = data.shape[0]
        n_ch = data.shape[1]
        x2 = data.reshape((n_ep * n_ch, data.shape[2]))
        allp = List[int]()
        for e in range(n_ep):
            for p in picks:
                allp.append(e * n_ch + p)
        _overlap_add_filter_2d(x2, filt, None, phase, allp, pad)
    elif data.ndim == 2:
        _overlap_add_filter_2d(data, filt, None, phase, picks, pad)
    else:
        compile_error("data must be 2D or 3D")


def filter_data(data, sfreq: float, l_freq, h_freq, picks = None, filter_length = "auto",
                l_trans_bandwidth = "auto", h_trans_bandwidth = "auto", n_jobs = None,
                method: str = "fir", iir_params = None, copy: bool = True, phase: str = "zero",
                fir_window: str = "hamming", fir_design: str = "firwin",
                pad: str = "reflect_limited", verbose = None):
    """Filter a subset of channels (mne.filter.filter_data, FIR only)."""
    x = np.asarray(data)
    if x.dtype is not float:
        compile_error("data must be float64 (mne requires real floating-point data)")
    if copy:
        x = x.copy()
    if x.ndim == 1:
        x2 = x.reshape((1, x.shape[0]))
        _filter_inplace_3d(x2, sfreq, l_freq, h_freq, [0], filter_length, l_trans_bandwidth,
                           h_trans_bandwidth, method, iir_params, phase, fir_window, fir_design, pad)
        return x
    else:
        n_rows = x.shape[x.ndim - 2]
        pk = List[int]()
        if isinstance(picks, NoneType):
            pk = list(range(n_rows))
        else:
            for p in picks:
                pk.append(int(p))
        if x.ndim == 2 or x.ndim == 3:
            _filter_inplace_3d(x, sfreq, l_freq, h_freq, pk, filter_length, l_trans_bandwidth,
                               h_trans_bandwidth, method, iir_params, phase, fir_window, fir_design, pad)
        else:
            if not isinstance(picks, NoneType):
                raise ValueError("picks argument is not supported for data with more than three dimensions")
            x2 = x.reshape((-1, x.shape[x.ndim - 1]))
            _filter_inplace_3d(x2, sfreq, l_freq, h_freq, list(range(x2.shape[0])), filter_length,
                               l_trans_bandwidth, h_trans_bandwidth, method, iir_params, phase,
                               fir_window, fir_design, pad)
        return x


def _bandstop_multi_fir(sfreq: float, lows: List[float], highs: List[float], tb: float,
                        filter_length, phase: str, fir_window: str, fir_design: str):
    """mne.filter.create_filter for l_freq=highs > h_freq=lows arrays (band-stop
    with several stop bands, FIR), as used by notch_filter."""
    _check_option("phase", phase, _known_phases_fir)
    _check_option("fir_window", fir_window, _known_fir_windows)
    _check_option("fir_design", fir_design, _known_fir_designs)
    for lo in lows:
        if lo <= 0:
            raise ValueError(f"highpass frequency {lo} must be greater than zero")
    for hi in highs:
        if hi >= sfreq / 2.0:
            raise ValueError(f"lowpass frequency {hi} must be less than Nyquist ({sfreq / 2.0})")
    if tb <= 0:
        raise ValueError(f"l_trans_bandwidth must be positive, got {tb}")
    fl = 0
    if isinstance(filter_length, str) and filter_length.lower() == "auto":
        mult_fact = 2.0 if fir_design == "firwin2" else 1.0
        secs = _length_factor(fir_window) * mult_fact / float(tb)
        fl = _to_samples(f"{secs}s", sfreq, phase, fir_design)
    else:
        fl = _to_samples(filter_length, sfreq, phase, fir_design)
    if fir_design == "firwin" or phase == "zero":
        fl += (fl - 1) % 2
    if fl <= 0:
        raise ValueError(f"filter_length must be positive, got {fl}")
    pairs = List[Tuple[float, float]]()
    for lo in lows:
        pairs.append((lo, 1.0))
    for lo in lows:
        pairs.append((lo + tb, 0.0))
    for hi in highs:
        pairs.append((hi - tb, 0.0))
    for hi in highs:
        pairs.append((hi, 1.0))
    pairs = sorted(pairs, key=lambda p: p[0])
    freq = [p[0] for p in pairs]
    gain = [p[1] for p in pairs]
    if freq[0] != 0:
        freq = [0.0] + freq
        gain = [1.0] + gain
    if freq[-1] != sfreq / 2.0:
        freq.append(sfreq / 2.0)
        gain.append(1.0)
    for k in range(len(gain) - 2):
        if abs(gain[k + 2] - 2.0 * gain[k + 1] + gain[k]) > 1:
            raise ValueError("Stop bands are not sufficiently separated.")
    return _construct_fir_filter(sfreq, freq, gain, fl, phase, fir_window, fir_design)


def _notch_bands(freqs, notch_widths, trans_bandwidth: float) -> Tuple[List[float], List[float]]:
    fl = List[float]()
    if isinstance(freqs, float) or isinstance(freqs, int):
        fl.append(float(freqs))
    else:
        for f in freqs:
            fl.append(float(f))
    nw = List[float]()
    if isinstance(notch_widths, NoneType):
        nw = [f / 200.0 for f in fl]
    elif isinstance(notch_widths, float) or isinstance(notch_widths, int):
        nw = [float(notch_widths) for _ in fl]
    else:
        for w in notch_widths:
            nw.append(float(w))
        if len(nw) != len(fl):
            raise ValueError("notch_widths must be None, scalar, or the same length as freqs")
    tb_2 = trans_bandwidth / 2.0
    lows = [fl[i] - nw[i] / 2.0 - tb_2 for i in range(len(fl))]
    highs = [fl[i] + nw[i] / 2.0 + tb_2 for i in range(len(fl))]
    return lows, highs


def _notch_inplace(data, sfreq: float, freqs, notch_widths, trans_bandwidth: float,
                   picks: List[int], filter_length, phase: str, fir_window: str,
                   fir_design: str, pad: str, method: str = "fir", iir_params = None):
    """FIR notch of a 2-D (or 3-D) array in place (one combined band-stop filter)."""
    lows, highs = _notch_bands(freqs, notch_widths, trans_bandwidth)
    if method == "iir":
        if len(lows) != 1:
            raise ValueError("Multiple stop-bands can only be used with method='fir' and method='spectrum_fit'")
        _filter_inplace_3d(data, sfreq, highs[0], lows[0], picks, filter_length, trans_bandwidth / 2.0,
                           trans_bandwidth / 2.0, "iir", iir_params, phase, fir_window, fir_design, pad)
        return
    h = _bandstop_multi_fir(sfreq, lows, highs, trans_bandwidth / 2.0, filter_length, phase,
                            fir_window, fir_design)
    if data.ndim == 3:
        n_ep = data.shape[0]
        n_ch = data.shape[1]
        x2 = data.reshape((n_ep * n_ch, data.shape[2]))
        allp = List[int]()
        for e in range(n_ep):
            for p in picks:
                allp.append(e * n_ch + p)
        _overlap_add_filter_2d(x2, h, None, phase, allp, pad)
    elif data.ndim == 2:
        _overlap_add_filter_2d(data, h, None, phase, picks, pad)
    else:
        compile_error("data must be 2D or 3D")


def notch_filter(x, Fs: float, freqs, filter_length = "auto", notch_widths = None,
                 trans_bandwidth: float = 1.0, method: str = "fir", iir_params = None,
                 mt_bandwidth = None, p_value: float = 0.05, picks = None, n_jobs = None,
                 copy: bool = True, phase: str = "zero", fir_window: str = "hamming",
                 fir_design: str = "firwin", pad: str = "reflect_limited", verbose = None):
    """FIR notch filter (mne.filter.notch_filter, method='fir'/'fft')."""
    if method != "fir" and method != "fft" and method != "iir":
        raise NotImplementedError("only method='fir'/'iir' notch filtering is supported in Typhon mne")
    a = np.asarray(x)
    if a.dtype is not float:
        compile_error("data must be float64 (mne requires real floating-point data)")
    out = a.copy() if copy else a
    if out.ndim == 1:
        x2 = out.reshape((1, out.shape[0]))
        _notch_inplace(x2, float(Fs), freqs, notch_widths, trans_bandwidth, [0], filter_length,
                       phase, fir_window, fir_design, pad, method, iir_params)
    else:
        n_rows = out.shape[out.ndim - 2]
        pk = List[int]()
        if isinstance(picks, NoneType):
            pk = list(range(n_rows))
        else:
            for p in picks:
                pk.append(int(p))
        _notch_inplace(out, float(Fs), freqs, notch_widths, trans_bandwidth, pk, filter_length,
                       phase, fir_window, fir_design, pad, method, iir_params)
    return out


# ---------------------------------------------------------------------------
# IIR filtering (mne.filter.construct_iir_filter / _iir_filter)
# ---------------------------------------------------------------------------

from scipy.signal import iirfilter as _sp_iirfilter, sosfilt as _sp_sosfilt, \
    lfilter as _sp_lfilter, sosfiltfilt as _sp_sosfiltfilt, filtfilt as _sp_filtfilt, \
    sos2zpk as _sp_sos2zpk, tf2zpk as _sp_tf2zpk
from scipy.signal import get_window as _sp_get_window

_known_iir_ftypes = ("bessel", "butter", "butterworth", "cauer", "cheby1", "cheby2",
                     "chebyshev1", "chebyshev2", "chebyshevi", "chebyshevii",
                     "ellip", "elliptic")


class IIRParams:
    """Resolved IIR filter (the dict mne.filter.construct_iir_filter returns).

    Exactly one of sos or (b, a) is set, selected by `output`.
    """
    output: str
    order: int
    ftype: str
    rp: Optional[float]
    rs: Optional[float]
    padlen: int
    sos: np.ndarray[float, 2]
    b: np.ndarray[float, 1]
    a: np.ndarray[float, 1]

    def __init__(self, order: int = 4, ftype: str = "butter", output: str = "sos",
                 rp: Optional[float] = None, rs: Optional[float] = None, padlen: int = -1):
        self.output = output
        self.order = order
        self.ftype = ftype
        self.rp = rp
        self.rs = rs
        self.padlen = padlen
        self.sos = np.zeros((0, 6))
        self.b = np.ones(1)
        self.a = np.ones(1)

    def __getitem__(self, key: Literal[str]):
        if key == "sos":
            return self.sos
        elif key == "b":
            return self.b
        elif key == "a":
            return self.a
        elif key == "padlen":
            return self.padlen
        elif key == "order":
            return self.order
        elif key == "ftype":
            return self.ftype
        elif key == "output":
            return self.output
        else:
            compile_error("unknown iir_params key")

    def __contains__(self, key: str) -> bool:
        if key in ("sos",):
            return self.output == "sos"
        if key in ("b", "a"):
            return self.output == "ba"
        return key in ("padlen", "order", "ftype", "output")


def _iir_set_value(p: IIRParams, k: str, v):
    if isinstance(v, Union):
        for tag, T in static.vars_types(v, with_index=True):
            if Union._get_tag(v) == tag:
                _iir_set_value(p, k, Union._get_data(v, T))
    elif isinstance(v, str):
        if k == "ftype":
            p.ftype = v
        elif k == "output":
            p.output = v
        else:
            raise ValueError(f"iir_params[{k!r}] must be numeric, got a str")
    elif isinstance(v, NoneType):
        if k == "rp":
            p.rp = None
        elif k == "rs":
            p.rs = None
        elif k not in ("padlen", "gpass", "gstop"):
            raise ValueError(f"iir_params[{k!r}] cannot be None")
    elif isinstance(v, Optional):
        if v is None:
            _iir_set_value(p, k, None)
        else:
            _iir_set_value(p, k, v.__val__())
    elif isinstance(v, int) or isinstance(v, float) or isinstance(v, bool):
        if k == "order":
            p.order = int(v)
        elif k == "padlen":
            p.padlen = int(v)
        elif k == "rp":
            p.rp = float(v)
        elif k == "rs":
            p.rs = float(v)
        else:
            raise ValueError(f"unsupported numeric iir_params key {k}")
    else:
        compile_error("unsupported iir_params value type")


def _iir_params_from(iir_params) -> IIRParams:
    """Accept None, IIRParams, or a str-keyed dict.

    mne's mixed dict(order=4, ftype='butter', output='sos') is a dict with
    Union[int|str] values; each entry is dispatched on its runtime member.
    """
    p = IIRParams()
    if isinstance(iir_params, NoneType):
        return p
    elif isinstance(iir_params, IIRParams):
        return iir_params
    elif hasattr(iir_params, "items"):
        for k, v in iir_params.items():
            _iir_set_value(p, k, v)
        return p
    else:
        compile_error("iir_params must be None, mne.filter.IIRParams or a dict")


def _iir_default_output_check(p: IIRParams):
    if p.output != "ba" and p.output != "sos":
        raise ValueError(f"Invalid value for the 'output' parameter. Allowed values are 'ba' and 'sos', but got '{p.output}' instead.")
    if p.ftype not in _known_iir_ftypes:
        raise RuntimeError(f"ftype must be in filter_dict from scipy.signal (e.g., butter, cheby1, etc.) not {p.ftype}")


def _check_coefficients_p(p: IIRParams):
    if p.output == "sos":
        _, poles, _ = _sp_sos2zpk(p.sos)
        for q in poles:
            if abs(q) > 1.0:
                raise RuntimeError("Filter poles outside unit circle, filter will be unstable. Consider using different filter coefficients.")
    else:
        _, poles, _ = _sp_tf2zpk(p.b, p.a)
        for q in poles:
            if abs(q) > 1.0:
                raise RuntimeError("Filter poles outside unit circle, filter will be unstable. Consider using different filter coefficients.")


def estimate_ringing_samples(p: IIRParams, max_try: int = 100000) -> int:
    n_per_chunk = 1000
    n_chunks_max = int(math.ceil(max_try / float(n_per_chunk)))
    x = np.zeros(n_per_chunk)
    x[0] = 1.0
    last_good = n_per_chunk
    thresh_val = 0.0
    zs = np.zeros((p.sos.shape[0], 2))
    zb = np.zeros(max(p.a.shape[0], p.b.shape[0]) - 1)
    for ii in range(n_chunks_max):
        h = np.zeros(0)
        if p.output == "ba":
            h, zb = _sp_lfilter(p.b, p.a, x, zi=zb)
        else:
            h, zs = _sp_sosfilt(p.sos, x, zi=zs)
        x[0] = 0.0
        h = np.abs(h)
        thresh_val = max(0.001 * float(np.max(h)), thresh_val)
        idx = -1
        for k in range(h.shape[0]):
            if h[k] > thresh_val:
                idx = k
        if idx >= 0:
            last_good = idx
        else:
            return (ii - 1) * n_per_chunk + last_good
    return n_per_chunk * n_chunks_max


def construct_iir_filter(iir_params = None, f_pass = None, f_stop = None, sfreq: float = 0.0,
                         btype: str = "", return_copy: bool = True, *, phase: str = "zero",
                         verbose = None) -> IIRParams:
    """Design an IIR filter by order (gpass/gstop iirdesign is not supported)."""
    p0 = _iir_params_from(iir_params)
    p = IIRParams()
    p.output = p0.output
    p.order = p0.order
    p.ftype = p0.ftype
    p.rp = p0.rp
    p.rs = p0.rs
    p.padlen = p0.padlen
    _iir_default_output_check(p)
    _check_option("phase", phase, ("zero", "zero-double", "forward"))
    fp = List[float]()
    if isinstance(f_pass, NoneType):
        raise ValueError("f_pass is required")
    elif isinstance(f_pass, float) or isinstance(f_pass, int):
        fp.append(float(f_pass))
    else:
        for f in f_pass:
            fp.append(float(f))
    Wp = [f / (sfreq / 2.0) for f in fp]
    singleton = btype in ("low", "lowpass", "high", "highpass")
    if p.output == "sos":
        if singleton:
            p.sos = _sp_iirfilter(p.order, Wp[0], rp=p.rp, rs=p.rs, btype=btype, ftype=p.ftype, output="sos")
        else:
            p.sos = _sp_iirfilter(p.order, Wp, rp=p.rp, rs=p.rs, btype=btype, ftype=p.ftype, output="sos")
    else:
        if singleton:
            bb, aa = _sp_iirfilter(p.order, Wp[0], rp=p.rp, rs=p.rs, btype=btype, ftype=p.ftype, output="ba")
            p.b = np.asarray(bb).astype(float)
            p.a = np.asarray(aa).astype(float)
        else:
            bb, aa = _sp_iirfilter(p.order, Wp, rp=p.rp, rs=p.rs, btype=btype, ftype=p.ftype, output="ba")
            p.b = np.asarray(bb).astype(float)
            p.a = np.asarray(aa).astype(float)
    _check_coefficients_p(p)
    if p.padlen < 0:
        p.padlen = estimate_ringing_samples(p)
    return p


def _iir_filter_2d(x, p: IIRParams, picks: List[int], phase: str):
    if len(picks) == 0:
        return
    n_t = x.shape[1]
    if phase == "zero" or phase == "zero-double":
        padlen = min(p.padlen, n_t - 1)
        sub = np.zeros((len(picks), n_t))
        for i in range(len(picks)):
            sub[i] = x[picks[i]]
        ext = sub
        if padlen > 0:
            ext = np.zeros((len(picks), n_t + 2 * padlen))
            for i in range(len(picks)):
                ext[i] = _smart_pad(sub[i], (padlen, padlen), "reflect_limited")
        y = np.zeros((1, 1))
        if p.output == "sos":
            y = _sp_sosfiltfilt(p.sos, ext, axis=-1, padlen=0)
        else:
            y = _sp_filtfilt(p.b, p.a, ext, axis=-1, padlen=0)
        for i in range(len(picks)):
            x[picks[i]] = y[i, padlen:padlen + n_t]
    else:
        for i in range(len(picks)):
            row = x[picks[i]].copy()
            if p.output == "sos":
                x[picks[i]] = _sp_sosfilt(p.sos, row)
            else:
                x[picks[i]] = _sp_lfilter(p.b, p.a, row)


def _create_iir(sfreq: float, l_freq, h_freq, l_trans_bandwidth, h_trans_bandwidth,
                iir_params, phase: str) -> IIRParams:
    _check_option("phase", phase, ("zero", "zero-double", "forward"))
    hf: Optional[float] = None if h_freq is None else float(h_freq)
    lf: Optional[float] = None if l_freq is None else float(l_freq)
    if hf is not None and hf.__val__() > sfreq / 2.0:
        raise ValueError(f"h_freq ({hf}) must be less than the Nyquist frequency {sfreq / 2.0}")
    if lf is not None and lf.__val__() == 0:
        lf = None
    if lf is not None and lf.__val__() <= 0:
        raise ValueError(f"highpass frequency {lf} must be greater than zero")
    if hf is not None and hf.__val__() >= sfreq / 2.0:
        raise ValueError(f"lowpass frequency {hf} must be less than Nyquist ({sfreq / 2.0})")
    if lf is None and hf is None:
        p = _iir_params_from(iir_params)
        q = IIRParams()
        q.output = "ba"
        q.padlen = 0 if p.padlen < 0 else p.padlen
        return q
    elif lf is None:
        return construct_iir_filter(iir_params, hf.__val__(), None, sfreq, "lowpass", phase=phase)
    elif hf is None:
        return construct_iir_filter(iir_params, lf.__val__(), None, sfreq, "highpass", phase=phase)
    elif lf.__val__() < hf.__val__():
        return construct_iir_filter(iir_params, [lf.__val__(), hf.__val__()], None, sfreq, "bandpass", phase=phase)
    else:
        return construct_iir_filter(iir_params, [hf.__val__(), lf.__val__()], None, sfreq, "bandstop", phase=phase)


# ---------------------------------------------------------------------------
# FFT resampling (mne.filter.resample, method='fft')
# ---------------------------------------------------------------------------

def _resample_fft_rows(x2, ratio: float, final_len: int, pad: str, window, npad):
    x_len = x2.shape[1]
    pd = "reflect_limited" if pad == "auto" else pad
    win = "boxcar"
    if isinstance(window, str):
        if window != "auto":
            win = window
    elif isinstance(window, NoneType):
        pass
    else:
        compile_error("window must be a str or None in Typhon mne.resample")
    np0 = 0
    np1 = 0
    if isinstance(npad, str):
        if npad != "auto":
            raise ValueError(f"Invalid value for the 'npad' parameter when a string. Allowed values are 'auto', but got '{npad}' instead.")
        min_add = min(x_len // 8, 100) * 2
        n = 2 ** int(math.ceil(math.log2(x_len + min_add))) - x_len
        np0 = n // 2
        np1 = n // 2 + n % 2
    else:
        np0 = int(npad)
        np1 = int(npad)
    orig_len = x_len + np0 + np1
    new_len = max(int(round(ratio * orig_len)), 1)
    tr0 = int(round(ratio * np0))
    tr1 = new_len - final_len - tr0
    W = np.fft.ifftshift(_sp_get_window(win, orig_len))
    W *= float(new_len) / float(orig_len)
    rl = orig_len // 2 + 1
    Wr = W.copy()
    Wrev = W[::-1]
    for k in range(1, rl):
        Wr[k] = (W[k] + Wrev[k - 1]) / 2.0
    Wh = Wr[:rl].copy()
    y = np.zeros((x2.shape[0], new_len - tr0 - tr1))
    shorter = new_len < orig_len
    use_len = new_len if shorter else orig_len
    for r in range(x2.shape[0]):
        xe = _smart_pad(x2[r].copy(), (np0, np1), pd)
        xf = _rfft(xe)
        if use_len % 2 == 0:
            nyq = use_len // 2
            if nyq < xf.shape[0]:
                xf[nyq] *= 2.0 if shorter else 0.5
        xf *= Wh
        yy = _irfft(xf, new_len)
        if tr0 > 0 or tr1 > 0:
            y[r] = yy[tr0:yy.shape[0] - tr1]
        else:
            y[r] = yy
    return y


def resample(x, up: float = 1.0, down: float = 1.0, *, axis: int = -1, window = "auto",
             n_jobs = None, pad: str = "auto", npad = 100, method: str = "fft", verbose = None):
    """Resample an array along the last axis (method='fft' only)."""
    if method != "fft":
        if method == "polyphase":
            raise NotImplementedError("method='polyphase' is not supported in Typhon mne")
        raise ValueError(f"Invalid value for the 'method' parameter. Allowed values are 'fft' and 'polyphase', but got '{method}' instead.")
    xa = np.asarray(x).astype(float)
    if axis != -1 and axis != xa.ndim - 1:
        raise NotImplementedError("Typhon mne.filter.resample only supports axis=-1")
    ratio = float(up) / down
    n = xa.shape[xa.ndim - 1]
    final_len = max(int(round(ratio * n)), 1)
    x2 = xa.reshape((-1, n))
    y = _resample_fft_rows(x2, ratio, final_len, pad, window, npad)
    return y.reshape(xa.shape[:-1] + (final_len,))


def _check_resamp_noop(sfreq: float, o_sfreq: float, rtol: float = 1e-6) -> bool:
    return abs(sfreq - o_sfreq) <= rtol * abs(o_sfreq)


def design_mne_c_filter(sfreq: float, l_freq: Optional[float] = None, h_freq: Optional[float] = 40.0,
                        l_trans_bandwidth: Optional[float] = None,
                        h_trans_bandwidth: float = 5.0, verbose = None):
    """Create a FIR filter like that used by MNE-C (mne 1.13.2)."""
    n_freqs = (4096 + 2 * 2048) // 2 + 1
    freq_resp = np.ones(n_freqs)
    lf = 0.0
    if l_freq is not None:
        lf = float(l_freq)
    l_width = 3
    if l_trans_bandwidth is not None:
        l_width = (int(float(n_freqs - 1) * l_trans_bandwidth / (0.5 * sfreq)) + 1) // 2
    hf = sfreq / 2.0
    if h_freq is not None:
        hf = float(h_freq)
    h_width = (int(float(n_freqs - 1) * h_trans_bandwidth / (0.5 * sfreq)) + 1) // 2
    h_start = int(float(n_freqs - 1) * hf / (0.5 * sfreq))
    l_start = int(float(n_freqs - 1) * lf / (0.5 * sfreq))
    if lf > 0:
        start = l_start - l_width + 1
        stop = start + 2 * l_width - 1
        if start < 0 or stop >= n_freqs:
            raise RuntimeError("l_freq too low or l_trans_bandwidth too large")
        for i in range(start):
            freq_resp[i] = 0.0
        for i in range(start, stop):
            k = float(i - start - l_width + 1) / float(l_width) + 3.0
            c = math.cos(math.pi / 4.0 * k)
            freq_resp[i] = c * c
    if hf < sfreq / 2.0:
        start = h_start - h_width + 1
        stop = start + 2 * h_width - 1
        if start < 0 or stop >= n_freqs:
            raise RuntimeError("h_freq too high or h_trans_bandwidth too large")
        for i in range(start, stop):
            k = float(i - start - h_width + 1) / float(h_width) + 1.0
            c = math.cos(math.pi / 4.0 * k)
            freq_resp[i] *= c * c
        for i in range(stop, n_freqs):
            freq_resp[i] = 0.0
    h = _irfft(freq_resp, n=2 * n_freqs - 1)
    return np.roll(h, n_freqs - 1)
