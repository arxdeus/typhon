# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne/time_frequency/csd.py: CrossSpectralDensity, csd_array_fourier,
# csd_array_multitaper, csd_array_morlet (+ Epochs wrappers csd_fourier,
# csd_multitaper, csd_morlet).

import numpy as np
from scipy.fft import rfft as _rfft, rfftfreq as _rfftfreq
from .multitaper import _compute_mt_params, _psd_from_mt_row, _psd_from_mt_w2, _trapezoid_dx
from .tfr import _morlet_list, _n_cycles_list, _next_fast_len, _cwt_one, _ensure_decim
from scipy.fft import fft as _fft


def _n_dims_from_triu(n: int) -> int:
    return int(np.ceil(np.sqrt(n * 2))) - 1


def _vector_to_sym_mat(vec: np.ndarray[complex, 1]) -> np.ndarray[complex, 2]:
    dim = _n_dims_from_triu(len(vec))
    mat = np.zeros((dim, dim), dtype=complex)
    k = 0
    for i in range(dim):
        for j in range(i, dim):
            mat[i, j] = vec[k]
            k += 1
    out = mat + mat.T.conj()
    for i in range(dim):
        out[i, i] /= 2
    return out


class CrossSpectralDensity:
    """Cross-spectral density (mne).

    ``_data``: (n_channels * (n_channels + 1) // 2, n_freqs) upper-triangle vectors.
    ``frequencies`` is the list of frequencies; after sum()/mean() the binned
    frequencies are available from ``frequency_bins``."""
    _data: np.ndarray[complex, 2]
    ch_names: List[str]
    tmin: Optional[float]
    tmax: Optional[float]
    _bins: List[List[float]]
    _is_sum: bool
    n_fft: int

    def __init__(self, data, ch_names: List[str], frequencies, n_fft: int,
                 tmin: Optional[float] = None, tmax: Optional[float] = None, projs = None):
        d = np.asarray(data, dtype=complex)
        if static.len(d.shape) == 1:
            self._data = d.reshape((d.shape[0], 1)).copy()
        else:
            self._data = d
        if len(ch_names) != _n_dims_from_triu(self._data.shape[0]):
            raise ValueError("Number of ch_names does not match the number of time series in the CSD matrix.")
        self.ch_names = list(ch_names)
        self.tmin = tmin
        self.tmax = tmax
        self._bins = List[List[float]]()
        self._is_sum = False
        if isinstance(frequencies, int) or isinstance(frequencies, float):
            self._bins.append([float(frequencies)])
        elif isinstance(frequencies, List[List[float]]):
            self._bins = frequencies
            self._is_sum = True
        else:
            for f in frequencies:
                self._bins.append([float(f)])
        if len(self._bins) != self._data.shape[1]:
            raise ValueError(f"Number of frequencies does not match the number of CSD matrices in the data array ({len(self._bins)} != {self._data.shape[1]}).")
        self.n_fft = n_fft

    @property
    def frequencies(self) -> List[float]:
        if self._is_sum:
            return [float(np.mean(np.array(b))) for b in self._bins]
        return [b[0] for b in self._bins]

    @property
    def frequency_bins(self) -> List[List[float]]:
        return self._bins

    @property
    def n_channels(self) -> int:
        return len(self.ch_names)

    def __len__(self) -> int:
        return len(self._bins)

    def __repr__(self) -> str:
        fs = List[str]()
        for b in self._bins:
            if len(b) == 1:
                fs.append(str(b[0]))
            else:
                fs.append(f"{min(b)}-{max(b)}")
        ts = f"{self.tmin} to {self.tmax} s" if (self.tmin is not None and self.tmax is not None) else "unknown"
        return f"<CrossSpectralDensity | n_channels={self.n_channels}, time={ts}, frequencies={', '.join(fs)} Hz.>"

    def copy(self):
        o = CrossSpectralDensity(self._data.copy(), list(self.ch_names), [0.0] * len(self._bins),
                                 self.n_fft, self.tmin, self.tmax)
        o._bins = [list(b) for b in self._bins]
        o._is_sum = self._is_sum
        return o

    def _get_frequency_index(self, freq: float) -> int:
        if self._is_sum:
            raise ValueError("This CSD object represents a mean across frequencies. Cannot select a specific frequency.")
        best = 0
        bd = np.inf
        for i in range(len(self._bins)):
            d = abs(self._bins[i][0] - freq)
            if d < bd:
                bd = d
                best = i
        if bd > 1:
            raise IndexError(f"Frequency {freq:f} is not available.")
        return best

    def sum(self, fmin = None, fmax = None):
        if self._is_sum:
            raise RuntimeError("This CSD matrix already represents a mean or sum across frequencies.")
        fr = self.frequencies
        if isinstance(fmin, NoneType) and isinstance(fmax, NoneType):
            lo = [fr[0]]
            hi = [fr[-1]]
        else:
            lo = _flist(fmin)
            hi = _flist(fmax)
            if len(lo) == 0:
                lo = [fr[0]] * len(hi)
            if len(hi) == 0:
                hi = [fr[-1]] * len(lo)
        for a, b in zip(lo, hi):
            if a > b:
                raise ValueError("Some lower bounds are higher than the corresponding upper bounds.")
        if len(lo) != len(hi):
            raise ValueError("The length of fmin does not match the length of fmax.")
        nb = len(lo)
        new_data = np.zeros((self._data.shape[0], nb), dtype=complex)
        bins = List[List[float]]()
        for i in range(nb):
            a = self._get_frequency_index(lo[i])
            b = self._get_frequency_index(hi[i]) + 1
            new_data[:, i] = self._data[:, a:b].sum(axis=1)
            bins.append(fr[a:b])
        o = CrossSpectralDensity(new_data, list(self.ch_names), [0.0] * nb, self.n_fft, self.tmin, self.tmax)
        o._bins = bins
        o._is_sum = True
        return o

    def mean(self, fmin = None, fmax = None):
        c = self.sum(fmin, fmax)
        for i in range(len(c._bins)):
            c._data[:, i] /= len(c._bins[i])
        return c

    def get_data(self, frequency = None, index = None, as_cov: bool = False):
        if isinstance(frequency, NoneType) and isinstance(index, NoneType):
            if self._data.shape[1] > 1:
                raise ValueError("Specify either the frequency or index of the frequency bin for which to obtain the CSD matrix.")
            idx = 0
        elif not isinstance(frequency, NoneType):
            if not isinstance(index, NoneType):
                raise ValueError("Cannot specify both a frequency and index.")
            idx = self._get_frequency_index(float(frequency))
        else:
            idx = int(index)
        return _vector_to_sym_mat(self._data[:, idx].copy())

    def __getitem__(self, sel: int):
        o = CrossSpectralDensity(self._data[:, sel:sel + 1].copy(), list(self.ch_names), [0.0],
                                 self.n_fft, self.tmin, self.tmax)
        o._bins = [list(self._bins[sel])]
        o._is_sum = self._is_sum
        return o

    def pick_frequency(self, freq = None, index = None):
        if isinstance(freq, NoneType) and isinstance(index, NoneType):
            raise ValueError('Use either the "freq" or "index" parameter to select the desired frequency.')
        if not isinstance(freq, NoneType):
            if not isinstance(index, NoneType):
                raise ValueError("Cannot specify both a frequency and index.")
            return self[self._get_frequency_index(float(freq))]
        return self[int(index)]


def _flist(x) -> List[float]:
    if isinstance(x, NoneType):
        return List[float]()
    elif isinstance(x, int) or isinstance(x, float):
        return [float(x)]
    else:
        return [float(v) for v in x]


def _prepare_csd_array(X, sfreq: float, t0: float, tmin, tmax, fmin: float, fmax: float):
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim != 3:
        raise ValueError("X must be n_epochs x n_channels x n_times.")
    n_times = Xa.shape[2]
    tstep = 1.0 / sfreq
    times = np.arange(n_times) * tstep + t0
    tmx = float(times.max()) if isinstance(tmax, NoneType) else float(tmax)
    tmn = float(times.min()) if isinstance(tmin, NoneType) else float(tmin)
    if tmx <= tmn:
        raise ValueError("tmax must be larger than tmin")
    if tmn < times[0] - tstep:
        raise ValueError("tmin should be larger than the smallest data time point")
    if tmx > times[-1] + tstep:
        raise ValueError("tmax should be smaller than the largest data time point")
    if fmax <= fmin:
        raise ValueError("fmax must be larger than fmin")
    return Xa, times


def _crop_times(Xa, times, tmin, tmax):
    a = 0 if isinstance(tmin, NoneType) else int(np.searchsorted(times, float(tmin) - 1e-10))
    b = times.shape[0] if isinstance(tmax, NoneType) else int(np.searchsorted(times, float(tmax) + 1e-10))
    return np.ascontiguousarray(Xa[:, :, a:b]), times[a:b].copy()


def _ch_names_or_default(ch_names, n: int) -> List[str]:
    if isinstance(ch_names, NoneType):
        return [f"SERIES{i + 1:03d}" for i in range(n)]
    return [str(c) for c in ch_names]


def _mt_spectra_2d(x: np.ndarray[float, 2], dpss: np.ndarray[float, 2], n_fft: int) -> np.ndarray[complex, 3]:
    n_ch = x.shape[0]
    nt = dpss.shape[0]
    nf = n_fft // 2 + 1
    out = np.zeros((n_ch, nt, nf), dtype=complex)
    r2 = np.sqrt(2.0)
    for c in range(n_ch):
        s = x[c] - x[c].mean()
        for k in range(nt):
            out[c, k] = _rfft(s * dpss[k], n_fft)
            out[c, k, 0] /= r2
            if n_fft % 2 == 0:
                out[c, k, nf - 1] /= r2
    return out


def _adaptive_weights(xfull: np.ndarray[complex, 2], eigvals, lo_idx: np.ndarray[int, 1],
                      max_iter: int) -> np.ndarray[float, 2]:
    """Weights d_k (n_tapers, n_sel_freqs) of _psd_from_mt_adaptive."""
    nt = xfull.shape[0]
    nfull = xfull.shape[1]
    rt = np.sqrt(eigvals)
    psd_est = _psd_from_mt_row(xfull, rt)
    var = _trapezoid_dx(psd_est, np.pi / nfull) / (2 * np.pi)
    xk = xfull[:, lo_idx].copy()
    nf = lo_idx.shape[0]
    psd_iter = _psd_from_mt_row(xk[:2, :], rt[:2])
    err = np.zeros((nt, nf))
    d_k = np.zeros((nt, nf))
    for n in range(max_iter):
        d_k = np.zeros((nt, nf))
        for k in range(nt):
            for f in range(nf):
                d = psd_iter[f] / (eigvals[k] * psd_iter[f] + (1 - eigvals[k]) * var)
                d_k[k, f] = d * rt[k]
        err -= d_k
        mx = 0.0
        for f in range(nf):
            m = 0.0
            for k in range(nt):
                m += err[k, f] ** 2
            m /= nt
            if m > mx:
                mx = m
        if mx < 1e-10:
            break
        psd_iter = _psd_from_mt_w2(xk, d_k)
        err = d_k.copy()
    return d_k


def _csd_from_mt_w(x_mt: np.ndarray[complex, 3], wts: np.ndarray[float, 3]) -> np.ndarray[complex, 2]:
    """x_mt (n_ch, n_tapers, n_freqs), weights (n_ch|1, n_tapers, n_freqs|1) -> triu CSD."""
    n_ch = x_mt.shape[0]
    nt = x_mt.shape[1]
    nf = x_mt.shape[2]
    per_ch = wts.shape[0] > 1
    per_f = wts.shape[2] > 1
    out = np.zeros((n_ch * (n_ch + 1) // 2, nf), dtype=complex)
    k = 0
    for i in range(n_ch):
        wi = i if per_ch else 0
        for j in range(i, n_ch):
            wj = j if per_ch else 0
            for f in range(nf):
                fw = f if per_f else 0
                acc = complex(0.0, 0.0)
                dx = 0.0
                dy = 0.0
                for t in range(nt):
                    a = wts[wi, t, fw]
                    b = wts[wj, t, fw]
                    acc += a * x_mt[i, t, f] * (b * x_mt[j, t, f]).conjugate()
                    dx += a * a
                    dy += b * b
                out[k, f] = acc * (2.0 / (np.sqrt(dx) * np.sqrt(dy)))
            k += 1
    return out


def csd_array_fourier(X, sfreq: float, t0: float = 0, fmin: float = 0, fmax: float = np.inf,
                      tmin = None, tmax = None, ch_names = None, n_fft = None, projs = None,
                      n_jobs = None, *, verbose = None):
    """Estimate cross-spectral density from an array using short-time fourier (mne)."""
    Xa, times = _prepare_csd_array(X, float(sfreq), float(t0), tmin, tmax, float(fmin), float(fmax))
    Xc, tc = _crop_times(Xa, times, tmin, tmax)
    n_times = tc.shape[0]
    nfft = n_times if isinstance(n_fft, NoneType) else int(n_fft)
    of = _rfftfreq(nfft, 1.0 / sfreq)
    sel = np.flatnonzero((of > 0) & (of >= fmin) & (of <= fmax))
    if sel.shape[0] == 0:
        raise ValueError("No discrete fourier transform results within the given frequency window. Please widen either the frequency window or the time window")
    win = np.hanning(n_times).reshape((1, n_times))
    n_ch = Xc.shape[1]
    acc = np.zeros((n_ch * (n_ch + 1) // 2, sel.shape[0]), dtype=complex)
    ones = np.ones((1, 1, 1))
    for e in range(Xc.shape[0]):
        x_mt = _mt_spectra_2d(np.ascontiguousarray(Xc[e]), win, nfft)
        c = _csd_from_mt_w(np.ascontiguousarray(x_mt[:, :, sel]), ones)
        c /= n_times
        c *= 8 / 3.0
        c /= sfreq
        acc += c
    acc /= Xc.shape[0]
    return CrossSpectralDensity(acc, _ch_names_or_default(ch_names, n_ch), of[sel], nfft,
                                float(tc[0]), float(tc[-1]))


def csd_array_multitaper(X, sfreq: float, t0: float = 0, fmin: float = 0, fmax: float = np.inf,
                         tmin = None, tmax = None, ch_names = None, n_fft = None, bandwidth = None,
                         adaptive: bool = False, low_bias: bool = True, projs = None,
                         n_jobs = None, max_iter: int = 250, *, verbose = None):
    """Estimate cross-spectral density from an array using a multitaper method (mne)."""
    Xa, times = _prepare_csd_array(X, float(sfreq), float(t0), tmin, tmax, float(fmin), float(fmax))
    Xc, tc = _crop_times(Xa, times, tmin, tmax)
    n_times = tc.shape[0]
    nfft = n_times if isinstance(n_fft, NoneType) else int(n_fft)
    window_fun, eigvals, ad = _compute_mt_params(n_times, float(sfreq), bandwidth, low_bias, adaptive)
    of = _rfftfreq(nfft, 1.0 / sfreq)
    sel = np.flatnonzero((of > 0) & (of >= fmin) & (of <= fmax))
    if sel.shape[0] == 0:
        raise ValueError("No discrete fourier transform results within the given frequency window. Please widen either the frequency window or the time window")
    n_ch = Xc.shape[1]
    nt = window_fun.shape[0]
    acc = np.zeros((n_ch * (n_ch + 1) // 2, sel.shape[0]), dtype=complex)
    for e in range(Xc.shape[0]):
        x_mt = _mt_spectra_2d(np.ascontiguousarray(Xc[e]), window_fun, nfft)
        if ad:
            w = np.empty((n_ch, nt, sel.shape[0]))
            for c in range(n_ch):
                w[c] = _adaptive_weights(x_mt[c], eigvals, sel, max_iter)
        else:
            w = np.sqrt(eigvals).reshape((1, nt, 1))
        c = _csd_from_mt_w(np.ascontiguousarray(x_mt[:, :, sel]), w)
        c /= sfreq
        acc += c
    acc /= Xc.shape[0]
    return CrossSpectralDensity(acc, _ch_names_or_default(ch_names, n_ch), of[sel], nfft,
                                float(tc[0]), float(tc[-1]))


def csd_array_morlet(X, sfreq: float, frequencies, t0: float = 0, tmin = None, tmax = None,
                     ch_names = None, n_cycles = 7, use_fft: bool = True, decim: int = 1,
                     projs = None, n_jobs = None, *, verbose = None):
    """Estimate cross-spectral density from an array using Morlet wavelets (mne)."""
    Xa, times = _prepare_csd_array(X, float(sfreq), float(t0), tmin, tmax, 0.0, np.inf)
    n_times = times.shape[0]
    fl = [float(v) for v in frequencies]
    Ws = _morlet_list(float(sfreq), fl, _n_cycles_list(n_cycles, len(fl)), None, False)
    imin = 0
    for i in range(len(fl)):
        if fl[i] < fl[imin]:
            imin = i
    wave_length = Ws[imin].shape[0] // 2
    a = 0
    b = n_times
    if not isinstance(tmin, NoneType):
        a = max(0, int(np.searchsorted(times, float(tmin))) - wave_length)
    if not isinstance(tmax, NoneType):
        b = min(n_times, int(np.searchsorted(times, float(tmax))) + wave_length)
    Xc = np.ascontiguousarray(Xa[:, :, a:b])
    tt = times[a:b].copy()
    ca = 0 if isinstance(tmin, NoneType) else int(np.searchsorted(tt, float(tmin) - 1e-10))
    cb = tt.shape[0] if isinstance(tmax, NoneType) else int(np.searchsorted(tt, float(tmax) + 1e-10))
    tcsd = tt[ca:cb].copy()
    nt = Xc.shape[2]
    max_size = 0
    for w in Ws:
        max_size = max(max_size, w.shape[0])
    nfft = _next_fast_len(nt + max_size - 1)
    fWs = [_fft(w, nfft) for w in Ws]
    tidx = list(range(0, nt, decim))
    sa = ca // decim
    sb = cb // decim
    n_ch = Xc.shape[1]
    nf = len(fl)
    acc = np.zeros((n_ch * (n_ch + 1) // 2, nf), dtype=complex)
    for e in range(Xc.shape[0]):
        ps = np.empty((n_ch, nf, len(tidx)), dtype=complex)
        for c in range(n_ch):
            x = Xc[e, c].copy()
            fx = _fft(x, nfft)
            for k in range(nf):
                ps[c, k] = _cwt_one(x, Ws[k], fx, fWs[k], use_fft, nt)[np.array(tidx)]
        ps = ps[:, :, sa:sb].copy()
        k = 0
        for i in range(n_ch):
            for j in range(i, n_ch):
                for f in range(nf):
                    acc[k, f] += np.mean(ps[i, f] * np.conj(ps[j, f])) / sfreq
                k += 1
    acc /= Xc.shape[0]
    return CrossSpectralDensity(acc, _ch_names_or_default(ch_names, n_ch), fl, 1,
                                float(tcsd[0]), float(tcsd[-1]))


def _epochs_csd_input(epochs, picks):
    from .._info import _picks_to_idx
    pk = _picks_to_idx(epochs.info, picks, "data", "bads")
    data = epochs.get_data()[:, np.array(pk), :].copy()
    return data, [epochs.info.ch_names[i] for i in pk]


def csd_fourier(epochs, fmin: float = 0, fmax: float = np.inf, tmin = None, tmax = None,
                picks = None, n_fft = None, projs = None, n_jobs = None, *, verbose = None):
    """Estimate cross-spectral density from epochs using short-time fourier (mne)."""
    d, names = _epochs_csd_input(epochs, picks)
    return csd_array_fourier(d, epochs.info.sfreq, epochs.tmin, fmin, fmax, tmin, tmax, names, n_fft)


def csd_multitaper(epochs, fmin: float = 0, fmax: float = np.inf, tmin = None, tmax = None,
                   picks = None, n_fft = None, bandwidth = None, adaptive: bool = False,
                   low_bias: bool = True, projs = None, n_jobs = None, *, verbose = None):
    """Estimate cross-spectral density from epochs using a multitaper method (mne)."""
    d, names = _epochs_csd_input(epochs, picks)
    return csd_array_multitaper(d, epochs.info.sfreq, epochs.tmin, fmin, fmax, tmin, tmax, names,
                                n_fft, bandwidth, adaptive, low_bias)


def csd_morlet(epochs, frequencies, tmin = None, tmax = None, picks = None, n_cycles = 7,
               use_fft: bool = True, decim: int = 1, projs = None, n_jobs = None, *, verbose = None):
    """Estimate cross-spectral density from epochs using Morlet wavelets (mne)."""
    d, names = _epochs_csd_input(epochs, picks)
    return csd_array_morlet(d, epochs.info.sfreq, frequencies, epochs.tmin, tmin, tmax, names,
                            n_cycles, use_fft, decim)
