# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# mne.time_frequency additions: BaseTFR/RawTFR/RawTFRArray, EpochsTFRComplex
# (complex-valued epochs TFR for csd_tfr), SpectrumArray, EpochsSpectrumArray,
# combine_spectrum, combine_tfr, csd_tfr, pick_channels_csd, fit_iir_model_raw.
# HDF5 based read_tfrs/write_tfrs/read_spectrum/read_csd are not available.

import numpy as np
from .._info import Info, _picks_to_idx, _scaling_cov_rank
from .._containers import Spectrum, EpochsSpectrum
from ._tfr_containers import AverageTFR, _of
from .csd import CrossSpectralDensity, _vector_to_sym_mat, _n_dims_from_triu


class BaseTFR:
    """Common fields of the TFR containers (info, freqs, times)."""
    info: Info
    freqs: np.ndarray[float, 1]
    times: np.ndarray[float, 1]
    method: str

    def __init__(self, info: Info, freqs, times, method: str = ""):
        self.info = info
        self.freqs = np.asarray(freqs, dtype=float)
        self.times = np.asarray(times, dtype=float)
        self.method = method

    @property
    def ch_names(self):
        return self.info.ch_names


class RawTFR(BaseTFR):
    """TFR of continuous data: data (n_channels, n_freqs, n_times)."""
    data: np.ndarray[float, 3]

    def __init__(self, info: Info, data, times, freqs, method: str = ""):
        super().__init__(info, freqs, times, method)
        self.data = np.asarray(data, dtype=float)
        if self.data.shape[0] != info.nchan or self.data.shape[1] != self.freqs.shape[0] or self.data.shape[2] != self.times.shape[0]:
            raise ValueError(f"Shape of data {self.data.shape} does not match channels/freqs/times")

    @property
    def shape(self):
        return self.data.shape

    def copy(self):
        return RawTFR(self.info.copy(), self.data.copy(), self.times.copy(), self.freqs.copy(), self.method)

    def get_data(self, picks = None, exclude = "bads", fmin = None, fmax = None, tmin = None, tmax = None):
        pk = _picks_to_idx(self.info, picks, "data_or_ica", exclude)
        fm = (self.freqs >= (-np.inf if isinstance(fmin, NoneType) else float(fmin))) & (self.freqs <= (np.inf if isinstance(fmax, NoneType) else float(fmax)))
        tm = (self.times >= (float(self.times[0]) if isinstance(tmin, NoneType) else float(tmin))) & (self.times <= (float(self.times[-1]) if isinstance(tmax, NoneType) else float(tmax)))
        return self.data[np.array(pk)][:, np.flatnonzero(fm)][:, :, np.flatnonzero(tm)].copy()

    def crop(self, tmin = None, tmax = None, fmin = None, fmax = None):
        tm = (self.times >= (float(self.times[0]) if isinstance(tmin, NoneType) else float(tmin))) & (self.times <= (float(self.times[-1]) if isinstance(tmax, NoneType) else float(tmax)))
        ti = np.flatnonzero(tm)
        self.times = self.times[ti].copy()
        self.data = self.data[:, :, ti].copy()
        fm = (self.freqs >= (-np.inf if isinstance(fmin, NoneType) else float(fmin))) & (self.freqs <= (np.inf if isinstance(fmax, NoneType) else float(fmax)))
        fi = np.flatnonzero(fm)
        self.freqs = self.freqs[fi].copy()
        self.data = self.data[:, fi, :].copy()
        return self

    def __repr__(self) -> str:
        return f"<RawTFR | {self.data.shape[0]} channels, {self.freqs.shape[0]} freqs, {self.times.shape[0]} times>"


def RawTFRArray(info: Info, data, times, freqs, *, method: str = ""):
    """Data object for precomputed spectrotemporal representations of continuous data."""
    return RawTFR(info, data, times, freqs, method)


class EpochsTFRComplex(BaseTFR):
    """Epochs TFR with complex data (n_epochs, n_channels, n_freqs, n_times)."""
    data: np.ndarray[complex, 4]
    events: np.ndarray[int, 2]

    def __init__(self, info: Info, data, times, freqs, events = None, method: str = ""):
        super().__init__(info, freqs, times, method)
        self.data = np.asarray(data, dtype=complex)
        n_ep = self.data.shape[0]
        if isinstance(events, NoneType):
            ev = np.zeros((n_ep, 3), dtype=int)
            for i in range(n_ep):
                ev[i, 0] = i
                ev[i, 2] = 1
            self.events = ev
        else:
            self.events = np.asarray(events).astype(int)
        if self.data.shape[1] != info.nchan or self.data.shape[2] != self.freqs.shape[0] or self.data.shape[3] != self.times.shape[0]:
            raise ValueError(f"Shape of data {self.data.shape} does not match channels/freqs/times")

    def __len__(self) -> int:
        return self.data.shape[0]

    def copy(self):
        return EpochsTFRComplex(self.info.copy(), self.data.copy(), self.times.copy(), self.freqs.copy(),
                                self.events.copy(), self.method)


def SpectrumArray(data, info: Info, freqs, dim_names = ("channel", "freq"), weights = None, *,
                  verbose = None) -> Spectrum:
    """mne.time_frequency.SpectrumArray: data (n_channels, n_freqs[, n_segments])."""
    d = np.asarray(data, dtype=float)
    fr = np.asarray(freqs, dtype=float)
    dn = [str(x) for x in dim_names]
    if "channel" not in dn or "freq" not in dn:
        raise ValueError("Data must contain 'channel' and 'freq' dimensions.")
    if d.shape[0] != info.nchan:
        raise ValueError(f"The number of channels in `data` ({d.shape[0]}) must match the number of channels in `info` ({info.nchan}).")
    if d.shape[dn.index("freq")] != fr.shape[0]:
        raise ValueError(f"The number of frequencies in `data` ({d.shape[dn.index('freq')]}) must match the number of elements in `freqs` ({fr.shape[0]}).")
    if static.len(d.shape) == 2:
        return Spectrum(d.reshape((d.shape[0], d.shape[1], 1)).copy(), fr, info.copy(), "unknown", False)
    if "taper" in dn:
        raise NotImplementedError("Typhon mne: SpectrumArray with a 'taper' dimension is not supported")
    return Spectrum(d.copy(), fr, info.copy(), "unknown", True)


def EpochsSpectrumArray(data, info: Info, freqs, events = None, event_id = None,
                        tmin: float = 0.0, dim_names = ("epoch", "channel", "freq"), weights = None,
                        *, verbose = None) -> EpochsSpectrum:
    """mne.time_frequency.EpochsSpectrumArray: data (n_epochs, n_channels, n_freqs[, n_segments])."""
    d = np.asarray(data, dtype=float)
    fr = np.asarray(freqs, dtype=float)
    if d.shape[1] != info.nchan:
        raise ValueError(f"The number of channels in `data` ({d.shape[1]}) must match the number of channels in `info` ({info.nchan}).")
    if d.shape[2] != fr.shape[0]:
        raise ValueError(f"The number of frequencies in `data` ({d.shape[2]}) must match the number of elements in `freqs` ({fr.shape[0]}).")
    n_ep = d.shape[0]
    ev = np.zeros((n_ep, 3), dtype=int)
    if isinstance(events, NoneType):
        for i in range(n_ep):
            ev[i, 0] = i
            ev[i, 2] = 1
    else:
        ev = np.asarray(events).astype(int)
    if static.len(d.shape) == 3:
        return EpochsSpectrum(d.reshape((n_ep, d.shape[1], d.shape[2], 1)).copy(), fr, info.copy(), "unknown", ev, False)
    return EpochsSpectrum(d.copy(), fr, info.copy(), "unknown", ev, True)


def _weights_for(weights, n: int, nave) -> np.ndarray[float, 1]:
    w = np.zeros(n)
    if isinstance(weights, str):
        if weights == "nave":
            tot = 0.0
            for i in range(n):
                tot += float(nave[i])
            for i in range(n):
                w[i] = float(nave[i]) / tot
        elif weights == "equal":
            for i in range(n):
                w[i] = 1.0 / n
        else:
            raise ValueError('Weights must be a list of float, or "nave" or "equal"')
    else:
        w = np.array([float(x) for x in weights])
        if w.shape[0] != n:
            raise ValueError("Weights must be the same size as all_tfr")
    return w


def combine_tfr(all_tfr, weights = "nave"):
    """mne.time_frequency.combine_tfr: weighted sum of AverageTFR objects."""
    n = len(all_tfr)
    nave = [t.nave for t in all_tfr]
    w = _weights_for(weights, n, nave)
    tfr = all_tfr[0].copy()
    bads = List[str]()
    for t in all_tfr:
        if t.ch_names != tfr.ch_names:
            raise AssertionError(f"{tfr} and {t} do not contain the same channels")
        if np.max(np.abs(t.times - tfr.times)) >= 1e-7:
            raise AssertionError(f"{tfr} and {t} do not contain the same time instants")
        for b in t.info.bads:
            if b not in bads:
                bads.append(b)
    tfr.info.bads = bads
    acc = np.zeros(tfr.data.shape)
    s = 0.0
    for i in range(n):
        acc += w[i] * all_tfr[i].data
        s += w[i] * w[i] / float(all_tfr[i].nave)
    tfr.data = acc
    tfr.nave = max(int(1.0 / s), 1)
    return tfr


def combine_spectrum(all_spectrum, weights = "equal"):
    """mne.time_frequency.combine_spectrum: weighted sum of Spectrum objects."""
    n = len(all_spectrum)
    if isinstance(weights, str):
        if weights == "nave":
            raise ValueError(f"The 'nave' attribute is not specified for {all_spectrum[0]}")
    w = _weights_for(weights, n, List[int]())
    sp = all_spectrum[0].copy()
    bads = List[str]()
    for s in all_spectrum:
        if s.ch_names != sp.ch_names:
            raise AssertionError(f"{sp} and {s} do not contain the same channels")
        if np.max(np.abs(s.freqs - sp.freqs)) >= 1e-7:
            raise AssertionError(f"{sp} and {s} do not contain the same frequencies")
        for b in s.info.bads:
            if b not in bads:
                bads.append(b)
    sp.info.bads = bads
    acc = np.zeros(sp._data.shape)
    for i in range(n):
        acc += w[i] * all_spectrum[i]._data
    sp._data = acc
    return sp


def pick_channels_csd(csd: CrossSpectralDensity, include = (), exclude = (), ordered: bool = True,
                      copy: bool = True, *, verbose = None) -> CrossSpectralDensity:
    """mne.time_frequency.pick_channels_csd."""
    from .. import pick_channels
    c = csd.copy() if copy else csd
    sel = pick_channels(c.ch_names, include, exclude, ordered)
    k = len(sel)
    out = np.zeros((k * (k + 1) // 2, c._data.shape[1]), dtype=complex)
    for f in range(c._data.shape[1]):
        mat = _vector_to_sym_mat(c._data[:, f].copy())
        r = 0
        for i in range(k):
            for j in range(i, k):
                out[r, f] = mat[sel[i], sel[j]]
                r += 1
    c._data = out
    c.ch_names = [c.ch_names[int(i)] for i in sel]
    return c


def csd_tfr(epochs_tfr, tmin = None, tmax = None, picks = None, projs = None, verbose = None):
    """mne.time_frequency.csd_tfr for complex epochs TFR (EpochsTFRComplex)."""
    times = epochs_tfr.times
    sf = epochs_tfr.info.sfreq
    tmn = _of(tmin)
    tmx = _of(tmax)
    if tmn is not None and tmn.__val__() < times[0] - (times[1] - times[0]):
        raise ValueError("tmin should be larger than the smallest data time point")
    if tmx is not None and tmx.__val__() > times[-1] + (times[1] - times[0]):
        raise ValueError("tmax should be smaller than the largest data time point")
    if tmn is not None and tmx is not None:
        if tmx.__val__() < tmn.__val__():
            raise ValueError("tmax must be larger than tmin")
    pk = _picks_to_idx(epochs_tfr.info, picks, "data", "bads")
    names = [epochs_tfr.info.ch_names[i] for i in pk]
    nch = len(pk)
    nf = epochs_tfr.freqs.shape[0]
    t0 = 0
    t1 = times.shape[0]
    if tmn is not None:
        t0 = int(np.searchsorted(times, tmn.__val__() - 1e-10))
    if tmx is not None:
        t1 = int(np.searchsorted(times, tmx.__val__() + 1e-10))
    nt = t1 - t0
    n_ep = epochs_tfr.data.shape[0]
    data = np.zeros((nch * (nch + 1) // 2, nf), dtype=complex)
    for e in range(n_ep):
        r = 0
        for i in range(nch):
            for j in range(i, nch):
                for f in range(nf):
                    acc = 0j
                    for t in range(t0, t1):
                        acc += epochs_tfr.data[e, pk[i], f, t] * np.conj(epochs_tfr.data[e, pk[j], f, t])
                    data[r, f] += acc / nt / sf
                r += 1
    data /= n_ep
    return CrossSpectralDensity(data, names, epochs_tfr.freqs, 0, tmn, tmx)


def fit_iir_model_raw(raw, order: int = 2, picks = None, tmin = None, tmax = None, verbose = None):
    """mne.time_frequency.fit_iir_model_raw: Yule-Walker AR fit, returns (b, a)."""
    start = 0
    stop = raw._data.shape[1]
    if not isinstance(tmin, NoneType):
        start = int(raw.time_as_index(float(tmin))[0])
    if not isinstance(tmax, NoneType):
        stop = int(raw.time_as_index(float(tmax))[0]) + 1
    pk = _picks_to_idx(raw.info, picks, "data", "bads")
    n = stop - start
    X = np.zeros((len(pk), n))
    for k in range(len(pk)):
        sc = _scaling_cov_rank(raw.info.ch_types[pk[k]])
        X[k] = raw._data[pk[k], start:stop] * sc
    r = np.zeros(order + 1)
    for k in range(len(pk)):
        d = X[k] - X[k].mean()
        r[0] += np.dot(d, d)
        for q in range(1, order + 1):
            r[q] += np.dot(d[0:n - q], d[q:])
    for q in range(order + 1):
        r[q] = r[q] / ((n - q) * len(pk))
    T = np.zeros((order, order))
    for i in range(order):
        for j in range(order):
            T[i, j] = r[abs(i - j)]
    rho = np.linalg.solve(T, r[1:])
    a = np.zeros(order + 1)
    a[0] = 1.0
    for i in range(order):
        a[i + 1] = -rho[i]
    return np.array([1.0]), a
