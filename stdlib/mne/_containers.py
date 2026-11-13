# Ported from mne 1.13.2 (BSD-3-Clause)
#
# EpochsArray, Evoked, RawArray and their Spectrum objects (data held in
# memory as float64 ndarrays).

import numpy as np
from ._info import Info, _picks_to_idx, _time_mask, _get_ch_factors
from .filter import _filter_inplace_3d, _check_option, _resample_fft_rows, _check_resamp_noop
from .time_frequency.psd import _psd_welch_2d
from .time_frequency.multitaper import _psd_mt_2d
from .annotations import Annotations


def _filter_picks(info: Info, picks) -> List[int]:
    return _picks_to_idx(info, picks, "data_or_ica", ())


def _update_filter_info(info: Info, picks: List[int], l_freq, h_freq):
    data_picks = _picks_to_idx(info, None, "data_or_ica", (), allow_empty=True)
    if len(data_picks) == 0:
        return
    for p in data_picks:
        if p not in picks:
            return
    lf: Optional[float] = None if l_freq is None else float(l_freq)
    hf: Optional[float] = None if h_freq is None else float(h_freq)
    if hf is not None and (lf is None or lf.__val__() < hf.__val__()) and hf.__val__() < info.lowpass:
        info.lowpass = hf.__val__()
    if lf is not None and (hf is None or lf.__val__() < hf.__val__()) and lf.__val__() > info.highpass:
        info.highpass = lf.__val__()


def _welch_kwargs_nfft(n_fft: Optional[int], n_sel: int) -> int:
    # _set_legacy_nfft_default: n_fft = min(n_times_in_window, 2048)
    if n_fft is None:
        return min(n_sel, 2048)
    return n_fft


class Spectrum:
    """Spectrum of continuous data (mne.time_frequency.Spectrum).

    Data is stored as (n_channels, n_freqs, n_segments); n_segments is 1
    unless the spectrum was computed with average=None, in which case
    get_data() returns the 3-D array like mne. Use get_data() for 2-D
    aggregated output.
    """
    _data: np.ndarray[float, 3]
    freqs: np.ndarray[float, 1]
    info: Info
    method: str
    _unagg: bool

    def __init__(self, data: np.ndarray[float, 3], freqs: np.ndarray[float, 1], info: Info,
                 method: str, unagg: bool = False):
        self._data = data
        self.freqs = freqs
        self.info = info
        self.method = method
        self._unagg = unagg

    @property
    def ch_names(self):
        return self.info.ch_names

    @property
    def shape(self):
        # (..., n_freqs); the segment axis of average=None spectra is in shape_full
        return self._data.shape[:-1]

    @property
    def shape_full(self):
        return self._data.shape

    def get_data(self, picks = None, exclude = "bads", fmin: float = 0, fmax: float = np.inf,
                 return_freqs: Literal[bool] = False):
        if self._unagg:
            raise ValueError("this Spectrum is unaggregated (average=None): use get_data_segments()")
        pk = _picks_to_idx(self.info, picks, "data_or_ica", exclude)
        lo = int(np.searchsorted(self.freqs, fmin))
        hi = int(np.searchsorted(self.freqs, fmax, side="right"))
        out = self._data[np.array(pk)][:, lo:hi, 0].copy()
        if return_freqs:
            return out, self.freqs[lo:hi].copy()
        else:
            return out

    def get_data_segments(self, picks = None, exclude = "bads", fmin: float = 0, fmax: float = np.inf):
        """(n_channels, n_freqs, n_segments) data of an average=None spectrum."""
        pk = _picks_to_idx(self.info, picks, "data_or_ica", exclude)
        lo = int(np.searchsorted(self.freqs, fmin))
        hi = int(np.searchsorted(self.freqs, fmax, side="right"))
        return self._data[np.array(pk)][:, lo:hi, :].copy()

    def copy(self):
        return Spectrum(self._data.copy(), self.freqs.copy(), self.info.copy(), self.method, self._unagg)

    def __repr__(self) -> str:
        return f"<Power Spectrum (from Raw, {self.method} method) | {self._data.shape[0]} channels × {self.freqs.shape[0]} freqs, {self.freqs[0]}-{self.freqs[-1]} Hz>"


class EpochsSpectrum:
    """Spectrum of epoched data (mne.time_frequency.EpochsSpectrum).

    Stored as (n_epochs, n_channels, n_freqs, n_segments); see Spectrum.
    """
    _data: np.ndarray[float, 4]
    freqs: np.ndarray[float, 1]
    info: Info
    method: str
    events: np.ndarray[int, 2]
    _unagg: bool

    def __init__(self, data: np.ndarray[float, 4], freqs: np.ndarray[float, 1], info: Info,
                 method: str, events: np.ndarray[int, 2], unagg: bool = False):
        self._data = data
        self.freqs = freqs
        self.info = info
        self.method = method
        self.events = events
        self._unagg = unagg

    @property
    def ch_names(self):
        return self.info.ch_names

    @property
    def shape(self):
        # (..., n_freqs); the segment axis of average=None spectra is in shape_full
        return self._data.shape[:-1]

    @property
    def shape_full(self):
        return self._data.shape

    def __len__(self) -> int:
        return self._data.shape[0]

    def get_data(self, picks = None, exclude = "bads", fmin: float = 0, fmax: float = np.inf,
                 return_freqs: Literal[bool] = False):
        if self._unagg:
            raise ValueError("this EpochsSpectrum is unaggregated (average=None): use get_data_segments()")
        pk = _picks_to_idx(self.info, picks, "data_or_ica", exclude)
        lo = int(np.searchsorted(self.freqs, fmin))
        hi = int(np.searchsorted(self.freqs, fmax, side="right"))
        out = self._data[:, np.array(pk), lo:hi, 0].copy()
        if return_freqs:
            return out, self.freqs[lo:hi].copy()
        else:
            return out

    def get_data_segments(self, picks = None, exclude = "bads", fmin: float = 0, fmax: float = np.inf):
        """(n_epochs, n_channels, n_freqs, n_segments) data of an average=None spectrum."""
        pk = _picks_to_idx(self.info, picks, "data_or_ica", exclude)
        lo = int(np.searchsorted(self.freqs, fmin))
        hi = int(np.searchsorted(self.freqs, fmax, side="right"))
        return self._data[:, np.array(pk), lo:hi, :].copy()

    def average(self, method: str = "mean"):
        d = np.zeros((1, 1, 1))
        if method == "mean":
            d = self._data.mean(axis=0)
        elif method == "median":
            d = np.median(self._data, axis=0)
        else:
            raise ValueError("method must be 'mean' or 'median'")
        return Spectrum(d, self.freqs.copy(), self.info.copy(), self.method, self._unagg)

    def copy(self):
        return EpochsSpectrum(self._data.copy(), self.freqs.copy(), self.info.copy(), self.method, self.events.copy(), self._unagg)

    def __repr__(self) -> str:
        return f"<Power Spectrum (from Epochs, {self.method} method) | {self._data.shape[0]} epochs × {self._data.shape[1]} channels × {self.freqs.shape[0]} freqs>"


def _psd_method(method: str, kind: str) -> str:
    m = method
    if m == "auto":
        m = "welch" if kind == "Raw" else "multitaper"
    _check_option("method", m, ("welch", "multitaper"))
    return m


def _spectra_rows(x2, sf: float, m: str, fmin: float, fmax: float, nf: int, n_overlap: int,
                  n_per_seg: Optional[int], avg: Optional[str], window, remove_dc: bool,
                  bandwidth, adaptive: bool, low_bias: bool, normalization: str, max_iter: int):
    if m == "welch":
        return _psd_welch_2d(x2, sf, fmin, fmax, nf, n_overlap, n_per_seg, avg, window, remove_dc)
    ps, freqs = _psd_mt_2d(x2, sf, fmin, fmax, bandwidth, adaptive, low_bias, normalization,
                           remove_dc, max_iter)
    return ps.reshape((ps.shape[0], ps.shape[1], 1)), freqs


def _welch_bad_spans(x2, good: List[bool], sf: float, fmin: float, fmax: float, nf: int,
                     n_overlap: int, n_per_seg: Optional[int], avg: Optional[str], window,
                     remove_dc: bool):
    """mne _psd_welch with NaNs aligned across channels (bad annotations): each
    good span is analysed separately (nperseg shrinks for short spans, as in
    scipy.signal.spectrogram) and per-span averages are combined with
    np.average weighted by the number of used samples."""
    from scipy.signal import spectrogram as _spg
    n = len(good)
    nps = nf if n_per_seg is None else min(n_per_seg.__val__(), nf)
    nps = min(nps, n)
    step = max(nps - n_overlap, 1)
    freqs_all = np.arange(nf // 2 + 1, dtype=float) * (sf / nf)
    lo = -1
    hi = -1
    for i in range(freqs_all.shape[0]):
        if freqs_all[i] >= fmin and freqs_all[i] <= fmax:
            if lo < 0:
                lo = i
            hi = i
    if lo < 0:
        raise ValueError(f"No frequencies found between fmin={fmin} and fmax={fmax}")
    if avg is None:
        raise NotImplementedError("average=None with bad annotations is not supported in Typhon mne")
    spans = List[Tuple[int, int]]()
    j = 0
    while j < n:
        if good[j]:
            k = j
            while k < n and good[k]:
                k += 1
            spans.append((j, k))
            j = k
        else:
            j += 1
    nfr = hi - lo + 1
    acc = np.zeros((x2.shape[0], nfr))
    wsum = 0.0
    for sp in spans:
        seg = x2[:, sp[0]:sp[1]].copy()
        w = sp[1] - sp[0]
        wt = float(w if w < nps else w - ((w - n_overlap) % step))
        p = min(nps, w)
        nov = n_overlap
        S = np.zeros((1, 1, 1))
        if remove_dc:
            _, _, S = _spg(seg, detrend="constant", noverlap=nov, nperseg=p, nfft=nf, fs=sf,
                           window=window, mode="psd")
        else:
            _, _, S = _spg(seg, detrend=False, noverlap=nov, nperseg=p, nfft=nf, fs=sf,
                           window=window, mode="psd")
        S = S[:, lo:hi + 1, :]
        if avg.__val__() == "mean":
            m = np.nanmean(S, axis=-1)
        else:
            from .time_frequency.psd import _median_biases
            b = _median_biases(S.shape[2])
            m = np.nanmedian(S, axis=-1) / b[S.shape[2]]
        acc += wt * m
        wsum += wt
    out = acc / wsum
    return out.reshape((out.shape[0], out.shape[1], 1)), freqs_all[lo:hi + 1].copy()


def _handle_tmin_tmax(times, sfreq: float, tmin: Optional[float], tmax: Optional[float]) -> Tuple[int, int]:
    """TimeMixin._handle_tmin_tmax (truncating time_as_index, stop exclusive)."""
    n = times.shape[0]
    start = 0 if tmin is None else int((tmin.__val__() - float(times[0])) * sfreq)
    stop = n if tmax is None else int((tmax.__val__() - float(times[0])) * sfreq)
    return min(max(0, start), n), min(max(0, stop), n)


def _opt_float(x) -> Optional[float]:
    if isinstance(x, NoneType):
        return None
    else:
        return float(x)


def _check_baseline(baseline, times, sfreq: float) -> Tuple[float, float]:
    """mne.baseline._check_baseline (on_baseline_outside_data='raise')."""
    bmin = _opt_float(baseline[0])
    bmax = _opt_float(baseline[1])
    t0 = float(times[0])
    t1 = float(times[-1])
    tstep = 1.0 / sfreq
    if bmin is None and bmax is not None and bmax.__val__() == 0.0 and t0 == 0.0:
        raise ValueError("Baseline interval is only one sample. Use `baseline=(0, 0)` if this is desired.")
    b0 = t0 if bmin is None else bmin.__val__()
    b1 = t1 if bmax is None else bmax.__val__()
    if b0 > b1:
        raise ValueError(f"Baseline min ({b0}) must be less than baseline max ({b1})")
    if b0 < t0 - tstep or b1 > t1 + tstep:
        raise ValueError(f"Baseline interval [{b0}, {b1}] s is outside of epochs data [{t0}, {t1}] s. Epochs were probably cropped.")
    return (b0, b1)


def _baseline_idx(times, bl: Tuple[float, float]) -> Tuple[int, int]:
    """Index range used by mne.baseline.rescale."""
    n = times.shape[0]
    imin = -1
    imax = -1
    for i in range(n):
        if imin < 0 and times[i] >= bl[0]:
            imin = i
        if times[i] <= bl[1]:
            imax = i + 1
    if imin < 0:
        raise ValueError(f"bmin is too large ({bl[0]}), it exceeds the largest time value")
    if imax < 0:
        raise ValueError(f"bmax is too small ({bl[1]}), it is smaller than the smallest time value")
    if imin >= imax:
        raise ValueError(f"Bad rescaling slice ({imin}:{imax}) from time values {bl[0]}, {bl[1]}")
    return imin, imax


def _rescale_3d(data, times, bl: Tuple[float, float], picks: List[int]):
    imin, imax = _baseline_idx(times, bl)
    for e in range(data.shape[0]):
        for p in picks:
            m = data[e, p, imin:imax].mean()
            data[e, p, :] -= m


def _detrend_picks(info: Info) -> List[int]:
    """_pick_data_channels(info, with_ref_meg=True, with_aux=True, exclude=())."""
    out = List[int]()
    for i in range(info.nchan):
        t = info.ch_types[i]
        if t in ("mag", "grad", "eeg", "csd", "seeg", "ecog", "dbs", "hbo", "hbr",
                 "fnirs_cw_amplitude", "fnirs_fd_ac_amplitude", "fnirs_fd_phase",
                 "fnirs_od", "ref_meg", "eog", "ecg", "emg", "bio", "eyegaze", "pupil"):
            out.append(i)
    return out


def _resample_info_times(info: Info, sfreq: float):
    info.lowpass = min(info.lowpass, sfreq / 2.0)
    info.sfreq = float(sfreq)


class Evoked:
    """Averaged epochs (mne.Evoked)."""
    data: np.ndarray[float, 2]
    info: Info
    times: np.ndarray[float, 1]
    nave: int
    comment: str
    baseline: Optional[Tuple[float, float]]
    kind: str

    def __init__(self, data: np.ndarray[float, 2], info: Info, times: np.ndarray[float, 1],
                 nave: int, comment: str = ""):
        self.data = data
        self.info = info
        self.times = times
        self.nave = nave
        self.comment = comment
        self.baseline = None
        self.kind = "average"

    @property
    def first(self) -> int:
        return int(round(float(self.times[0]) * self.info.sfreq))

    @property
    def last(self) -> int:
        return self.first + self.times.shape[0] - 1

    def apply_baseline(self, baseline = (None, 0), *, verbose = None):
        if isinstance(baseline, NoneType):
            return self
        bl = _check_baseline(baseline, self.times, self.info.sfreq)
        d3 = self.data.reshape((1, self.data.shape[0], self.data.shape[1]))
        _rescale_3d(d3, self.times, bl, list(range(self.info.nchan)))
        self.data = d3.reshape((self.data.shape[0], self.data.shape[1]))
        self.baseline = bl
        return self

    def get_peak(self, ch_type: Optional[str] = None, tmin: Optional[float] = None,
                 tmax: Optional[float] = None, mode: str = "abs", time_as_index: bool = False,
                 merge_grads: bool = False, return_amplitude: Literal[bool] = False, *,
                 strict: bool = True):
        """Evoked.get_peak: (ch_name, latency[, amplitude])."""
        used = self.info.get_channel_types(None, True, True)
        if ch_type is None and len(used) > 1:
            raise RuntimeError('Multiple data channel types found. Please pass the "ch_type" parameter.')
        pk = List[int]()
        if ch_type is None:
            pk = _picks_to_idx(self.info, None, "data", "bads")
        else:
            if ch_type.__val__() not in used:
                raise ValueError(f'Channel type "{ch_type}" not found in this evoked object.')
            pk = _picks_to_idx(self.info, ch_type.__val__(), "all", "bads")
        t = self.times
        lo = float(t[0]) if tmin is None else tmin.__val__()
        hi = float(t[-1]) if tmax is None else tmax.__val__()
        if lo < float(t[0]) or hi > float(t[-1]):
            nm = "tmin" if lo < float(t[0]) else "tmax"
            raise ValueError(f"{nm} ({lo if nm == 'tmin' else hi}) is out of bounds. It must be between {float(t[0])} and {float(t[-1])}")
        if lo > hi:
            raise ValueError(f"tmin ({lo}) must be <= tmax ({hi})")
        if mode != "abs" and mode != "pos" and mode != "neg":
            raise ValueError(f"Invalid value for the 'mode' parameter. Allowed values are 'abs', 'neg', and 'pos', but got '{mode}' instead.")
        best = -np.inf
        bi = -1
        bj = -1
        anyok = False
        for i in range(len(pk)):
            for j in range(t.shape[0]):
                if t[j] < lo or t[j] > hi:
                    continue
                v = self.data[pk[i], j]
                if (mode == "pos" and v > 0) or (mode == "neg" and v < 0):
                    anyok = True
                s = abs(v) if mode == "abs" else (v if mode == "pos" else -v)
                if s > best:
                    best = s
                    bi = i
                    bj = j
        if strict and mode == "pos" and not anyok:
            raise ValueError("No positive values encountered. Cannot operate in pos mode.")
        if strict and mode == "neg" and not anyok:
            raise ValueError("No negative values encountered. Cannot operate in neg mode.")
        name = self.info.ch_names[pk[bi]]
        amp = float(self.data[pk[bi], bj])
        lat = float(bj) if time_as_index else float(t[bj])
        if return_amplitude:
            return name, lat, amp
        else:
            return name, lat

    @property
    def ch_names(self):
        return self.info.ch_names

    @property
    def tmin(self) -> float:
        return self.times[0]

    @property
    def tmax(self) -> float:
        return self.times[-1]

    def get_data(self, picks = None, units = None, tmin: Optional[float] = None,
                 tmax: Optional[float] = None):
        pk = _picks_to_idx(self.info, picks, "all", ())
        a, b = _handle_tmin_tmax(self.times, self.info.sfreq, tmin, tmax)
        out = self.data[np.array(pk)][:, a:b].copy()
        if not isinstance(units, NoneType):
            out *= _get_ch_factors(self.info, units, pk)[:, None]
        return out

    def copy(self):
        e = Evoked(self.data.copy(), self.info.copy(), self.times.copy(), self.nave, self.comment)
        e.baseline = self.baseline
        e.kind = self.kind
        return e

    def crop(self, tmin: Optional[float] = None, tmax: Optional[float] = None,
             include_tmax: bool = True, verbose = None):
        m = _time_mask(self.times, tmin, tmax, sfreq=self.info.sfreq, include_tmax=include_tmax)
        self.data = self.data[:, m].copy()
        self.times = self.times[m].copy()
        return self

    def filter(self, l_freq, h_freq, picks = None, filter_length = "auto",
               l_trans_bandwidth = "auto", h_trans_bandwidth = "auto", n_jobs = None,
               method: str = "fir", iir_params = None, phase: str = "zero",
               fir_window: str = "hamming", fir_design: str = "firwin",
               skip_by_annotation = ("edge", "bad_acq_skip"), pad: str = "edge", verbose = None):
        pk = _filter_picks(self.info, picks)
        _filter_inplace_3d(self.data, self.info.sfreq, l_freq, h_freq, pk, filter_length,
                           l_trans_bandwidth, h_trans_bandwidth, method, iir_params, phase,
                           fir_window, fir_design, pad)
        _update_filter_info(self.info, pk, l_freq, h_freq)
        return self

    def resample(self, sfreq: float, *, npad = "auto", window = "auto", n_jobs = None,
                 pad: str = "edge", method: str = "fft", verbose = None):
        if method != "fft":
            raise NotImplementedError("only method='fft' resampling is supported in Typhon mne")
        o = self.info.sfreq
        if _check_resamp_noop(float(sfreq), o):
            return self
        ratio = float(sfreq) / o
        n = self.data.shape[1]
        final_len = max(int(round(ratio * n)), 1)
        t0 = float(self.times[0])
        self.data = _resample_fft_rows(self.data, ratio, final_len, pad, window, npad)
        _resample_info_times(self.info, float(sfreq))
        self.times = np.arange(final_len).astype(float) / float(sfreq) + t0
        return self

    def set_eeg_reference(self, ref_channels = "average", projection: bool = False, ch_type = "eeg",
                          joint: bool = False, verbose = None):
        """Evoked re-referencing (average / channel list / projection)."""
        from ._proj import make_eeg_average_ref_proj, _has_eeg_average_ref_proj, _add_projs
        if projection:
            if not (isinstance(ref_channels, str) and ref_channels == "average"):
                raise ValueError("Setting projection=True is only supported for ref_channels=\"average\"")
            if not _has_eeg_average_ref_proj(self.info):
                _add_projs(self.info, [make_eeg_average_ref_proj(self.info, ch_type, False)], False)
            return self
        eeg = [i for i in range(self.info.nchan)
               if self.info.ch_types[i] == "eeg" and self.info.ch_names[i] not in self.info.bads]
        refs = List[int]()
        if isinstance(ref_channels, str):
            if ref_channels == "average":
                refs = eeg
            else:
                refs = [self.info.ch_names.index(ref_channels)]
        else:
            for c in ref_channels:
                refs.append(self.info.ch_names.index(str(c)))
        if len(refs) == 0:
            return self
        ref = self.data[np.array(refs, dtype=int), :].mean(axis=0)
        for i in eeg:
            self.data[i, :] -= ref
        return self

    def __repr__(self) -> str:
        return f"<Evoked | '{self.comment}' (average, N={self.nave}), {self.times[0]:.5f} – {self.times[-1]:.5f} s, {self.data.shape[0]} ch>"


class EpochsArray:
    """Epochs object from numpy array (mne.EpochsArray)."""
    _data: np.ndarray[float, 3]
    info: Info
    events: np.ndarray[int, 2]
    event_id: Dict[str, int]
    times: np.ndarray[float, 1]
    baseline: Optional[Tuple[float, float]]
    preload: bool
    drop_log: List[List[str]]
    selection: List[int]
    reject: Optional[Dict[str, float]]
    flat: Optional[Dict[str, float]]
    _raw_sfreq: float
    _kind: str
    reject_tmin: Optional[float]
    reject_tmax: Optional[float]

    def __init__(self, data, info: Info, events = None, tmin: float = 0.0, event_id = None,
                 reject = None, flat = None, reject_tmin = None, reject_tmax = None,
                 baseline = None, proj: bool = True, on_missing: str = "raise",
                 metadata = None, selection = None, *, drop_log = None,
                 raw_sfreq = None, verbose = None):
        d = np.asarray(data)
        if d.ndim != 3:
            compile_error("Data must be a 3D array of shape (n_epochs, n_channels, n_samples)")
        self._data = d.astype(float)  # astype copies, like mne (copy=True)
        if self._data.shape[1] != info.nchan:
            raise ValueError(f"Info ({info.nchan}) and data ({self._data.shape[1]}) must have same number of channels.")
        n_ep = self._data.shape[0]
        self.info = info.copy()
        sfreq = info.sfreq
        if isinstance(events, NoneType):
            ev = np.zeros((n_ep, 3), dtype=int)
            for i in range(n_ep):
                ev[i, 0] = i
                ev[i, 2] = 1
            self.events = ev
        else:
            ev = np.asarray(events).astype(int)
            if ev.ndim != 2 or ev.shape[1] != 3:
                raise ValueError("events must be of shape (N, 3)")
            if ev.shape[0] != n_ep:
                raise ValueError("The number of epochs and the number of events must match")
            self.events = ev.copy()
        ids = Dict[str, int]()
        if isinstance(event_id, NoneType):
            seen = List[int]()
            for i in range(n_ep):
                v = int(self.events[i, 2])
                if v not in seen:
                    seen.append(v)
            seen.sort()
            for v in seen:
                ids[str(v)] = v
        elif isinstance(event_id, int):
            ids[str(event_id)] = event_id
        else:
            for k, v in event_id.items():
                ids[k] = v
        self.event_id = ids
        start_idx = int(round(tmin * sfreq))
        n_t = self._data.shape[2]
        self.times = np.arange(start_idx, start_idx + n_t).astype(float) / sfreq
        self.preload = True
        self.baseline = None
        self.drop_log = [List[str]() for _ in range(n_ep)]
        self.selection = list(range(n_ep))
        self.reject = None
        self.flat = None
        self._raw_sfreq = sfreq
        self._kind = "EpochsArray"
        self.reject_tmin = reject_tmin
        self.reject_tmax = reject_tmax
        if not isinstance(baseline, NoneType):
            bmin: Optional[float] = baseline[0]
            bmax: Optional[float] = baseline[1]
            b0 = float(self.times[0]) if bmin is None else bmin.__val__()
            b1 = float(self.times[-1]) if bmax is None else bmax.__val__()
            self.baseline = (b0, b1)
            self.apply_baseline((b0, b1))

    @property
    def tmin(self) -> float:
        return self.times[0]

    @property
    def tmax(self) -> float:
        return self.times[-1]

    @property
    def ch_names(self):
        return self.info.ch_names

    def __len__(self) -> int:
        return self._data.shape[0]

    def __repr__(self) -> str:
        return f"<{self._kind} | {len(self)} events (all good), {self.times[0]:.3g} – {self.times[-1]:.3g} s (baseline off), ~{self._data.nbytes / 1024.0:.1f} KiB, data loaded>"

    def apply_baseline(self, baseline = (None, 0), verbose = None):
        """mne BaseEpochs.apply_baseline (rescale mode='mean' on data+aux channels)."""
        if isinstance(baseline, NoneType):
            if self.baseline is not None:
                raise RuntimeError("You cannot remove baseline correction from preloaded data once it has been applied.")
            return self
        bl = _check_baseline(baseline, self.times, self.info.sfreq)
        picks = _detrend_picks(self.info)
        _rescale_3d(self._data, self.times, bl, picks)
        self.baseline = bl
        return self

    def get_data(self, picks = None, item = None, units = None, tmin: Optional[float] = None,
                 tmax: Optional[float] = None, *, copy: bool = True, verbose = None):
        if not isinstance(item, NoneType):
            compile_error("get_data(item=...) is not supported in Typhon mne")
        pk = _picks_to_idx(self.info, picks, "all", ())
        if tmin is None and tmax is None and isinstance(picks, NoneType) and isinstance(units, NoneType):
            return self._data.copy() if copy else self._data
        a, b = _handle_tmin_tmax(self.times, self.info.sfreq, tmin, tmax)
        out = self._data[:, np.array(pk), a:b].copy()
        if not isinstance(units, NoneType):
            out *= _get_ch_factors(self.info, units, pk)[None, :, None]
        return out

    def copy(self):
        n = EpochsArray(self._data, self.info, self.events, 0.0, self.event_id)
        n.times = self.times.copy()
        n.baseline = self.baseline
        n.drop_log = [list(d) for d in self.drop_log]
        n.selection = list(self.selection)
        n.reject = None if self.reject is None else self.reject.__val__().copy()
        n.flat = None if self.flat is None else self.flat.__val__().copy()
        n._raw_sfreq = self._raw_sfreq
        n._kind = self._kind
        return n

    def _get_name(self) -> str:
        """BaseEpochs._get_name(count='frac', ms='×', sep='+')."""
        keys = list(self.event_id.keys())
        if len(keys) == 1:
            return keys[0]
        n = self.events.shape[0]
        parts = List[str]()
        for k in keys:
            c = 0
            for i in range(n):
                if int(self.events[i, 2]) == self.event_id[k]:
                    c += 1
            parts.append(f"{float(c) / n:.2f} × {k}")
        return " + ".join(parts)

    def crop(self, tmin: Optional[float] = None, tmax: Optional[float] = None,
             include_tmax: bool = True, verbose = None):
        m = _time_mask(self.times, tmin, tmax, sfreq=self.info.sfreq, include_tmax=include_tmax)
        idx = np.nonzero(m)[0]
        self._data = self._data[:, :, idx[0]:idx[-1] + 1].copy()
        self.times = self.times[m].copy()
        return self

    def filter(self, l_freq, h_freq, picks = None, filter_length = "auto",
               l_trans_bandwidth = "auto", h_trans_bandwidth = "auto", n_jobs = None,
               method: str = "fir", iir_params = None, phase: str = "zero",
               fir_window: str = "hamming", fir_design: str = "firwin",
               skip_by_annotation = ("edge", "bad_acq_skip"), pad: str = "edge", verbose = None):
        """Filter in place (FIR only) and return self."""
        pk = _filter_picks(self.info, picks)
        _filter_inplace_3d(self._data, self.info.sfreq, l_freq, h_freq, pk, filter_length,
                           l_trans_bandwidth, h_trans_bandwidth, method, iir_params, phase,
                           fir_window, fir_design, pad)
        _update_filter_info(self.info, pk, l_freq, h_freq)
        return self

    def average(self, picks = None, method: str = "mean", by_event_type: Literal[bool] = False):
        if by_event_type:
            evs = List[Evoked]()
            for k in self.event_id.keys():
                evs.append(self[k]._average_one(picks, method))
            return evs
        else:
            return self._average_one(picks, method)

    def _average_one(self, picks, method: str):
        pk = _picks_to_idx(self.info, picks, "data_or_ica", "bads")
        sub = self._data[:, np.array(pk), :]
        d = np.zeros((1, 1))
        if method == "mean":
            d = sub.mean(axis=0)
        elif method == "median":
            d = np.median(sub, axis=0)
        else:
            raise ValueError("method must be 'mean' or 'median'")
        ev = Evoked(d, self.info._pick(pk), self.times.copy(), len(self), self._get_name())
        ev.baseline = self.baseline
        return ev

    def compute_psd(self, method: str = "multitaper", fmin: float = 0, fmax: float = np.inf,
                    tmin: Optional[float] = None, tmax: Optional[float] = None, picks = None,
                    proj: bool = False, remove_dc: bool = True, exclude = (), *, n_jobs = 1,
                    verbose = None, n_fft: Optional[int] = None, n_overlap: int = 0,
                    n_per_seg: Optional[int] = None, average = "mean", window = "hamming",
                    bandwidth = None, adaptive: bool = False, low_bias: bool = True,
                    normalization: str = "length", max_iter: int = 150):
        """Compute PSD (welch or multitaper). Returns an EpochsSpectrum.

        For method='welch' with average=None the result has an extra trailing
        segment axis (n_epochs, n_channels, n_freqs, n_segments), like mne.
        """
        m = _psd_method(method, "Epochs")
        sf = self.info.sfreq
        if np.isfinite(fmax) and fmax > sf / 2:
            raise ValueError(f"Requested fmax ({fmax} Hz) must not exceed ½ the sampling frequency of the data ({0.5 * sf} Hz).")
        tm = _time_mask(self.times, tmin, tmax, sfreq=sf)
        idx = np.nonzero(tm)[0]
        pk = _picks_to_idx(self.info, picks, "data", exclude)
        nf = _welch_kwargs_nfft(n_fft, idx.shape[0])
        sub = self._data[:, np.array(pk), :][:, :, idx[0]:idx[-1] + 1]
        n_ep = sub.shape[0]
        n_ch = sub.shape[1]
        x2 = sub.reshape((n_ep * n_ch, sub.shape[2]))
        avg: Optional[str] = None
        if isinstance(average, str):
            avg = average
        elif isinstance(average, bool):
            if average:
                raise ValueError("average must be 'mean', 'median', None or False")
        if m != "welch":
            avg = "mean"
        ps, freqs = _spectra_rows(x2, sf, m, float(fmin), float(fmax), nf, n_overlap, n_per_seg,
                                  avg, window, remove_dc, bandwidth, adaptive, low_bias,
                                  normalization, max_iter)
        data = ps.reshape((n_ep, n_ch, freqs.shape[0], ps.shape[2]))
        return EpochsSpectrum(data, freqs, self.info._pick(pk), m, self.events.copy(), avg is None)

    def resample(self, sfreq: float, *, npad = "auto", window = "auto", n_jobs = None,
                 pad: str = "edge", method: str = "fft", verbose = None):
        """FFT resampling in place (mne.Epochs.resample)."""
        if method != "fft":
            raise NotImplementedError("only method='fft' resampling is supported in Typhon mne")
        o = self.info.sfreq
        if _check_resamp_noop(float(sfreq), o):
            return self
        ratio = float(sfreq) / o
        n = self._data.shape[2]
        final_len = max(int(round(ratio * n)), 1)
        x2 = self._data.reshape((-1, n))
        y = _resample_fft_rows(x2, ratio, final_len, pad, window, npad)
        t0 = float(self.times[0])
        self._data = y.reshape((self._data.shape[0], self._data.shape[1], final_len))
        _resample_info_times(self.info, float(sfreq))
        self.times = np.arange(final_len).astype(float) / float(sfreq) + t0
        return self


class RawArray:
    """Raw object from numpy array (mne.io.RawArray)."""
    _data: np.ndarray[float, 2]
    info: Info
    first_samp: int
    times: np.ndarray[float, 1]
    annotations: Annotations
    _raw_kind: str

    def __init__(self, data, info: Info, first_samp: int = 0, copy: str = "auto", verbose = None):
        d = np.asarray(data)
        if d.ndim != 2:
            compile_error("Data must be a 2D array of shape (n_channels, n_samples)")
        self._data = d.astype(float)
        if self._data.shape[0] != info.nchan:
            raise ValueError(f"len(data) ({self._data.shape[0]}) does not match len(info['ch_names']) ({info.nchan})")
        self.info = info.copy()
        self.first_samp = first_samp
        self.times = np.arange(self._data.shape[1]).astype(float) / info.sfreq
        self.annotations = Annotations(List[float](), List[float](), List[str](), info.meas_date)
        self._raw_kind = "RawArray"

    @property
    def ch_names(self):
        return self.info.ch_names

    @property
    def n_times(self) -> int:
        return self._data.shape[1]

    def __len__(self) -> int:
        return self._data.shape[1]

    def __repr__(self) -> str:
        return f"<{self._raw_kind} | {self._data.shape[0]} x {self._data.shape[1]} (data loaded)>"

    def get_data(self, picks = None, start: int = 0, stop: Optional[int] = None,
                 reject_by_annotation = None, return_times: Literal[bool] = False,
                 units = None, tmin: Optional[float] = None, tmax: Optional[float] = None,
                 verbose = None):
        pk = _picks_to_idx(self.info, picks, "all", ())
        n = self._data.shape[1]
        ts, te = _handle_tmin_tmax(self.times, self.info.sfreq, tmin, tmax)
        s = ts if start == 0 else start
        e = te if stop is None else stop.__val__()
        s = min(max(0, s), n)
        e = min(max(0, e), n)
        out = self._data[np.array(pk), s:e].copy()
        tt = self.times[s:e].copy()
        if not isinstance(reject_by_annotation, NoneType) and len(self.annotations) > 0:
            mode = reject_by_annotation.lower()
            if mode != "omit" and mode != "nan":
                raise ValueError(f"Invalid value for the 'reject_by_annotation' parameter. Allowed values are 'omit' and 'nan', but got '{reject_by_annotation}' instead.")
            used = self._bad_annot_mask(s, e)
            if mode == "nan":
                for j in range(e - s):
                    if not used[j]:
                        out[:, j] = np.nan
            else:
                keep = np.array([j for j in range(e - s) if used[j]], dtype=int)
                out = out[:, keep].copy()
                tt = tt[keep].copy()
        if not isinstance(units, NoneType):
            out *= _get_ch_factors(self.info, units, pk)[:, None]
        if return_times:
            return out, tt
        else:
            return out

    def _bad_annot_mask(self, start: int, stop: int) -> List[bool]:
        """True for samples in [start, stop) not covered by BAD annotations."""
        used = [True for _ in range(stop - start)]
        sf = self.info.sfreq
        ft = self.first_time
        t0 = float(self.times[0])
        ann = self.annotations
        for i in range(len(ann)):
            if not ann.description[i].upper().startswith("BAD"):
                continue
            on = ann.onset[i] - ft
            en = on + ann.duration[i]
            a = int(np.round((on - t0) * sf))
            b = int(np.round((en - t0) * sf))
            if not (a < stop and b > start):
                continue
            a = max(a, start)
            b = min(b, stop)
            for j in range(a, b):
                used[j - start] = False
        return used

    def copy(self):
        n = RawArray(self._data, self.info, self.first_samp)
        n.annotations = self.annotations.copy()
        n._raw_kind = self._raw_kind
        return n

    @property
    def first_time(self) -> float:
        return self.first_samp / self.info.sfreq

    @property
    def last_samp(self) -> int:
        return self.first_samp + self._data.shape[1] - 1

    def get_channel_types(self, picks = None, unique: bool = False, only_data_chs: bool = False) -> List[str]:
        return self.info.get_channel_types(picks, unique, only_data_chs)

    def set_annotations(self, annotations: Optional[Annotations], emit_warning: bool = True,
                        on_missing: str = "raise", *, verbose = None):
        """Set annotations. orig_time=None onsets are relative to the first sample;
        otherwise they are relative to orig_time (seconds since epoch) and are
        re-referenced to info.meas_date."""
        md = self.info.meas_date
        if annotations is None:
            self.annotations = Annotations(List[float](), List[float](), List[str](), md)
            return self
        if md is None and annotations.orig_time is not None:
            raise RuntimeError("Ambiguous operation. Setting an Annotation object with known ``orig_time`` to a raw object which has ``meas_date`` set to None is ambiguous. Please, either set a meaningful ``meas_date`` to the raw object; or set ``orig_time`` to None in which case the annotation onsets would be taken in reference to the first sample of the raw object.")
        sf = self.info.sfreq
        end = (self._data.shape[1] - 1) / sf + 1.0 / sf
        ft = self.first_time
        a = annotations.copy()
        if annotations.orig_time is None:
            a.crop(0.0, end)
            a.onset = [o + ft for o in a.onset]
        else:
            shift = md.__val__() - annotations.orig_time.__val__()
            a.crop(shift + ft, shift + ft + end)
            a.onset = [o - shift for o in a.onset]
        a._orig_time = md
        self.annotations = a
        return self

    def time_as_index(self, times, use_rounding: bool = False, origin = None):
        t = np.atleast_1d(np.asarray(times).astype(float))
        idx = (t - self.times[0]) * self.info.sfreq
        if use_rounding:
            idx = np.round(idx)
        return idx.astype(int)

    def _pick_keep(self, keep: List[int]):
        self._data = self._data[np.array(keep, dtype=int), :].copy()
        self.info = self.info._pick(keep)
        return self

    def pick(self, picks, exclude = (), *, verbose = None):
        return self._pick_keep(_picks_to_idx(self.info, picks, "all", exclude))

    def pick_channels(self, ch_names, ordered: bool = True, *, verbose = None):
        keep = List[int]()
        if ordered:
            for c in ch_names:
                if str(c) not in self.info.ch_names:
                    raise ValueError(f"Missing channels from ch_names required by include: {c}")
                keep.append(self.info.ch_names.index(str(c)))
        else:
            names = [str(c) for c in ch_names]
            for i in range(self.info.nchan):
                if self.info.ch_names[i] in names:
                    keep.append(i)
        return self._pick_keep(keep)

    def pick_types(self, meg = False, eeg: bool = False, stim: bool = False, eog: bool = False,
                   ecg: bool = False, emg: bool = False, misc: bool = False, exclude = "bads"):
        want = List[str]()
        if eeg: want.append("eeg")
        if stim: want.append("stim")
        if eog: want.append("eog")
        if ecg: want.append("ecg")
        if emg: want.append("emg")
        if misc: want.append("misc")
        excl = list(self.info.bads) if isinstance(exclude, str) else [str(e) for e in exclude]
        keep = List[int]()
        for i in range(self.info.nchan):
            t = self.info.ch_types[i]
            ok = t in want
            if isinstance(meg, bool):
                if meg and (t == "mag" or t == "grad"):
                    ok = True
            else:
                if t == meg:
                    ok = True
            if ok and self.info.ch_names[i] not in excl:
                keep.append(i)
        return self._pick_keep(keep)

    def drop_channels(self, ch_names, on_missing: str = "raise"):
        names = [ch_names] if isinstance(ch_names, str) else [str(c) for c in ch_names]
        for c in names:
            if c not in self.info.ch_names and on_missing == "raise":
                raise ValueError(f"Channel(s) {c} not found, nothing dropped.")
        keep = [i for i in range(self.info.nchan) if self.info.ch_names[i] not in names]
        return self._pick_keep(keep)

    def set_eeg_reference(self, ref_channels = "average", copy: bool = False, projection: bool = False,
                          ch_type = "auto", forward = None, *, joint: bool = False, verbose = None):
        """Average or channel re-referencing of EEG channels (in place, returns self)."""
        if projection:
            from ._proj import make_eeg_average_ref_proj, _has_eeg_average_ref_proj, _add_projs
            if not (isinstance(ref_channels, str) and ref_channels == "average"):
                raise ValueError("Setting projection=True is only supported for ref_channels=\"average\"")
            if not _has_eeg_average_ref_proj(self.info):
                _add_projs(self.info, [make_eeg_average_ref_proj(self.info, ch_type if not isinstance(ch_type, str) or ch_type != "auto" else "eeg", False)], False)
            return self
        eeg = [i for i in range(self.info.nchan)
               if self.info.ch_types[i] == "eeg" and self.info.ch_names[i] not in self.info.bads]
        refs = List[int]()
        if isinstance(ref_channels, str):
            if ref_channels == "average":
                refs = eeg
            elif ref_channels in self.info.ch_names:
                refs = [self.info.ch_names.index(ref_channels)]
            else:
                raise ValueError(f"Invalid ref_channels {ref_channels}")
        else:
            for c in ref_channels:
                refs.append(self.info.ch_names.index(str(c)))
        if len(refs) == 0:
            return self
        ref = self._data[np.array(refs, dtype=int), :].mean(axis=0)
        for i in eeg:
            self._data[i, :] -= ref
        return self

    def apply_function(self, fun, picks = None, dtype = None, n_jobs = None,
                       channel_wise: bool = True, verbose = None, **kwargs):
        """Apply fun to the data in place. channel_wise=True calls fun on 1-D
        rows; channel_wise=False calls it on the 2-D picked array."""
        pk = _picks_to_idx(self.info, picks, "all", ())
        if channel_wise:
            for ci in pk:
                row = self._data[ci, :].copy()
                self._data[ci, :] = np.asarray(fun(row, **kwargs))
        else:
            sel = np.array(pk, dtype=int)
            out = np.asarray(fun(self._data[sel, :].copy(), **kwargs))
            for k in range(len(pk)):
                self._data[pk[k], :] = out[k]
        return self

    def load_data(self, verbose = None):
        return self

    @property
    def preload(self) -> bool:
        return True

    def crop(self, tmin: float = 0.0, tmax: Optional[float] = None, include_tmax: bool = True,
             verbose = None):
        sf = self.info.sfreq
        m = _time_mask(self.times, tmin, tmax, sfreq=sf, include_tmax=include_tmax)
        idx = np.nonzero(m)[0]
        s = int(idx[0])
        e = int(idx[-1]) + 1
        ann = self.annotations.copy()
        self._data = self._data[:, s:e].copy()
        self.first_samp += s
        ft = self.first_time
        ann.onset = [o - ft for o in ann.onset]
        self.times = np.arange(self._data.shape[1]).astype(float) / sf
        self.set_annotations(ann, False)
        return self

    def filter(self, l_freq, h_freq, picks = None, filter_length = "auto",
               l_trans_bandwidth = "auto", h_trans_bandwidth = "auto", n_jobs = None,
               method: str = "fir", iir_params = None, phase: str = "zero",
               fir_window: str = "hamming", fir_design: str = "firwin",
               skip_by_annotation = ("edge", "bad_acq_skip"), pad: str = "reflect_limited",
               verbose = None):
        """Filter in place (FIR only) and return self."""
        pk = _filter_picks(self.info, picks)
        _filter_inplace_3d(self._data, self.info.sfreq, l_freq, h_freq, pk, filter_length,
                           l_trans_bandwidth, h_trans_bandwidth, method, iir_params, phase,
                           fir_window, fir_design, pad)
        _update_filter_info(self.info, pk, l_freq, h_freq)
        return self

    def compute_psd(self, method: str = "welch", fmin: float = 0, fmax: float = np.inf,
                    tmin: Optional[float] = None, tmax: Optional[float] = None, picks = None,
                    exclude = (), proj: bool = False, remove_dc: bool = True,
                    reject_by_annotation: bool = True, *, n_jobs = 1, verbose = None,
                    n_fft: Optional[int] = None, n_overlap: int = 0,
                    n_per_seg: Optional[int] = None, average = "mean", window = "hamming",
                    bandwidth = None, adaptive: bool = False, low_bias: bool = True,
                    normalization: str = "length", max_iter: int = 150):
        """Compute PSD (welch or multitaper). Returns a Spectrum."""
        m = _psd_method(method, "Raw")
        sf = self.info.sfreq
        if np.isfinite(fmax) and fmax > sf / 2:
            raise ValueError(f"Requested fmax ({fmax} Hz) must not exceed ½ the sampling frequency of the data ({0.5 * sf} Hz).")
        tm = _time_mask(self.times, tmin, tmax, sfreq=sf)
        idx = np.nonzero(tm)[0]
        pk = _picks_to_idx(self.info, picks, "data", exclude)
        nf = _welch_kwargs_nfft(n_fft, idx.shape[0])
        sub = self._data[np.array(pk), idx[0]:idx[-1] + 1].copy()
        avg: Optional[str] = None
        if isinstance(average, str):
            avg = average
        elif isinstance(average, bool):
            if average:
                raise ValueError("average must be 'mean', 'median', None or False")
        if m != "welch":
            avg = "mean"
        good = List[bool]()
        has_bad = False
        if reject_by_annotation and len(self.annotations) > 0:
            good = self._bad_annot_mask(int(idx[0]), int(idx[-1]) + 1)
            for g in good:
                if not g:
                    has_bad = True
        if has_bad:
            if m == "multitaper":
                raise NotImplementedError('Cannot use method="multitaper" when reject_by_annotation=True. Please use method="welch" instead.')
            ps, freqs = _welch_bad_spans(sub, good, sf, float(fmin), float(fmax), nf, n_overlap,
                                         n_per_seg, avg, window, remove_dc)
            return Spectrum(ps, freqs, self.info._pick(pk), m, avg is None)
        ps, freqs = _spectra_rows(sub, sf, m, float(fmin), float(fmax), nf, n_overlap, n_per_seg,
                                  avg, window, remove_dc, bandwidth, adaptive, low_bias,
                                  normalization, max_iter)
        return Spectrum(ps, freqs, self.info._pick(pk), m, avg is None)

    def resample(self, sfreq: float, *, npad = "auto", window = "auto", stim_picks = None,
                 n_jobs = None, events = None, pad: str = "auto", method: str = "fft",
                 verbose = None):
        o = self.info.sfreq
        self._resample_core(sfreq, npad, window, stim_picks, pad, method)
        if isinstance(events, NoneType):
            return self
        else:
            ratio = float(sfreq) / o
            if abs(float(sfreq) - o) <= 1e-6 * o:
                return self, np.asarray(events).astype(int).copy()
            ev = np.asarray(events).astype(int).copy()
            for i in range(ev.shape[0]):
                ev[i, 0] = min(int(np.round(ev[i, 0] * ratio)), self._data.shape[1] + self.first_samp - 1)
            return self, ev

    def _resample_core(self, sfreq: float, npad, window, stim_picks, pad: str, method: str):
        """FFT resampling in place (mne.io.Raw.resample). Stim channels use
        mne's sample-picking rule. Passing events is not supported."""
        if method != "fft":
            raise NotImplementedError("only method='fft' resampling is supported in Typhon mne")
        o = self.info.sfreq
        if _check_resamp_noop(float(sfreq), o):
            return self
        ratio = float(sfreq) / o
        n = self._data.shape[1]
        final_len = max(int(round(ratio * n)), 1)
        y = _resample_fft_rows(self._data, ratio, final_len, pad, window, npad)
        stims = List[int]()
        if isinstance(stim_picks, NoneType):
            for i in range(self.info.nchan):
                if self.info.ch_types[i] == "stim":
                    stims.append(i)
        else:
            stims = _picks_to_idx(self.info, stim_picks, "all", ())
        for ci in stims:
            st = self._data[ci]
            picks_s = np.zeros(final_len, dtype=int)
            for k in range(final_len):
                picks_s[k] = min(int(k / ratio), n - 1)
            for k in range(final_len):
                w0 = int(picks_s[k])
                w1 = int(picks_s[k + 1]) if k + 1 < final_len else n
                val = st[w0]
                for j in range(w0, w1):
                    if st[j] != 0:
                        val = st[j]
                        break
                y[ci, k] = val
        self._data = y
        self.first_samp = int(round(self.first_samp * ratio))
        _resample_info_times(self.info, float(sfreq))
        self.times = np.arange(final_len).astype(float) / float(sfreq)
        return self
