# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne/time_frequency/tfr.py containers: AverageTFR, EpochsTFR (data, times,
# freqs, apply_baseline, crop, average, copy, arithmetic) and the
# tfr_morlet / tfr_multitaper / compute_tfr entry points for Epochs/Evoked.

import numpy as np
from .._info import Info, _picks_to_idx, _time_mask
from .tfr import _compute_tfr_c, _ensure_decim, tfr_array_stockwell


def _freq_mask(freqs, sfreq: float, fmin: Optional[float], fmax: Optional[float]):
    fmn = float(freqs[0]) if fmin is None else fmin.__val__()
    fmx = float(freqs[-1]) if fmax is None else fmax.__val__()
    if not np.isfinite(fmn):
        fmn = float(freqs[0])
    if not np.isfinite(fmx):
        fmx = float(freqs[-1])
    fmn = int(round(fmn * sfreq)) / sfreq - 0.5 / sfreq
    fmx = int(round(fmx * sfreq)) / sfreq + 0.5 / sfreq
    if fmn > fmx:
        raise ValueError(f"fmin ({fmin}) must be less than or equal to fmax ({fmax})")
    mask = (freqs >= fmn) & (freqs <= fmx)
    if not mask.any():
        raise ValueError(f"No frequencies remain when using fmin={fmin} and fmax={fmax}")
    return mask


def _of(x) -> Optional[float]:
    if isinstance(x, NoneType):
        return None
    elif isinstance(x, Optional):
        return x
    else:
        return float(x)


def _bl_bounds(baseline, times) -> Tuple[float, float]:
    from .._containers import _check_baseline
    return _check_baseline(baseline, times, 1.0e12)


def _rescale_last(data, times, bmin: float, bmax: float, mode: str):
    """mne.baseline.rescale on the last axis, in place (data: 2-D view rows x times)."""
    n = times.shape[0]
    imin = -1
    imax = -1
    for i in range(n):
        if imin < 0 and times[i] >= bmin:
            imin = i
        if times[i] <= bmax:
            imax = i + 1
    if imin < 0:
        raise ValueError(f"bmin is too large ({bmin}), it exceeds the largest time value")
    if imax < 0:
        raise ValueError(f"bmax is too small ({bmax}), it is smaller than the smallest time value")
    if imin >= imax:
        raise ValueError(f"Bad rescaling slice ({imin}:{imax}) from time values {bmin}, {bmax}")
    if mode not in ("logratio", "ratio", "zscore", "mean", "percent", "zlogratio", "meanlogratio"):
        raise ValueError(f"Invalid value for the 'mode' parameter. Allowed values are 'logratio', 'ratio', 'zscore', 'mean', 'percent', 'zlogratio', and 'meanlogratio', but got '{mode}' instead.")
    for r in range(data.shape[0]):
        d = data[r]
        m = d[imin:imax].mean()
        if mode == "mean":
            d -= m
        elif mode == "ratio":
            d /= m
        elif mode == "logratio":
            d /= m
            d[:] = np.log10(d)
        elif mode == "meanlogratio":
            d /= m
            d[:] = np.log10(d)
            d -= d[imin:imax].mean()
        elif mode == "percent":
            d -= m
            d /= m
        elif mode == "zscore":
            d -= m
            d /= np.std(d[imin:imax])
        else:
            d /= m
            d[:] = np.log10(d)
            d /= np.std(d[imin:imax])


class AverageTFR:
    """Data object for spectrotemporal representations of averaged data (mne).

    data: (n_channels, n_freqs, n_times)."""
    info: Info
    data: np.ndarray[float, 3]
    times: np.ndarray[float, 1]
    freqs: np.ndarray[float, 1]
    nave: int
    comment: str
    method: str
    baseline: Optional[Tuple[float, float]]

    def __init__(self, info: Info, data, times, freqs, nave: int = 1, comment: str = "",
                 method: str = ""):
        self.info = info
        self.data = np.asarray(data, dtype=float)
        self.times = np.asarray(times, dtype=float)
        self.freqs = np.asarray(freqs, dtype=float)
        self.nave = nave
        self.comment = comment
        self.method = method
        self.baseline = None
        if self.data.shape[0] != info.nchan or self.data.shape[1] != self.freqs.shape[0] or self.data.shape[2] != self.times.shape[0]:
            raise ValueError(f"Shape of data {self.data.shape} does not match channels/freqs/times")

    @property
    def ch_names(self):
        return self.info.ch_names

    @property
    def sfreq(self) -> float:
        return self.info.sfreq

    @property
    def shape(self):
        return self.data.shape

    @property
    def tmin(self) -> float:
        return float(self.times[0])

    @property
    def tmax(self) -> float:
        return float(self.times[-1])

    def copy(self):
        o = AverageTFR(self.info.copy(), self.data.copy(), self.times.copy(), self.freqs.copy(),
                       self.nave, self.comment, self.method)
        o.baseline = self.baseline
        return o

    def get_data(self, picks = None, exclude = "bads", fmin = None, fmax = None, tmin = None,
                 tmax = None):
        pk = _picks_to_idx(self.info, picks, "data_or_ica", exclude)
        fm = (self.freqs >= (-np.inf if isinstance(fmin, NoneType) else float(fmin))) & (self.freqs <= (np.inf if isinstance(fmax, NoneType) else float(fmax)))
        tm = (self.times >= (float(self.times[0]) if isinstance(tmin, NoneType) else float(tmin))) & (self.times <= (float(self.times[-1]) if isinstance(tmax, NoneType) else float(tmax)))
        return self.data[np.array(pk)][:, np.flatnonzero(fm)][:, :, np.flatnonzero(tm)].copy()

    def apply_baseline(self, baseline, mode: str = "mean", verbose = None):
        """Baseline correct the data (modes: mean, ratio, logratio, percent, zscore, zlogratio, meanlogratio)."""
        bl = _bl_bounds(baseline, self.times)
        self.baseline = bl
        d2 = self.data.reshape((-1, self.data.shape[2]))
        _rescale_last(d2, self.times, bl[0], bl[1], mode)
        return self

    def crop(self, tmin = None, tmax = None, fmin = None, fmax = None, include_tmax: bool = True):
        tmn = self.tmin if isinstance(tmin, NoneType) else max(float(tmin), self.tmin)
        inc = include_tmax
        tmx = self.tmax
        if not isinstance(tmax, NoneType):
            if float(tmax) > self.tmax:
                inc = True
            else:
                tmx = float(tmax)
        tm = _time_mask(self.times, tmn, tmx, sfreq=self.info.sfreq, include_tmax=inc)
        ti = np.flatnonzero(tm)
        self.times = self.times[ti].copy()
        self.data = self.data[:, :, ti].copy()
        if not isinstance(fmin, NoneType) or not isinstance(fmax, NoneType):
            fm = _freq_mask(self.freqs, self.info.sfreq, _of(fmin), _of(fmax))
            fi = np.flatnonzero(fm)
            self.freqs = self.freqs[fi].copy()
            self.data = self.data[:, fi, :].copy()
        return self

    def __add__(self, other: AverageTFR):
        o = self.copy()
        o.data += other.data
        return o

    def __sub__(self, other: AverageTFR):
        o = self.copy()
        o.data -= other.data
        return o

    def __mul__(self, num: float):
        o = self.copy()
        o.data *= num
        return o

    def __truediv__(self, num: float):
        o = self.copy()
        o.data /= num
        return o

    def __abs__(self):
        o = self.copy()
        o.data = np.abs(o.data)
        return o

    def __repr__(self) -> str:
        return f"<AverageTFR from Epochs, {self.method} method | {self.data.shape[0]} channels × {self.freqs.shape[0]} freqs × {self.times.shape[0]} times, {self.freqs[0]:.1f} - {self.freqs[-1]:.1f} Hz, {self.times[0]:.2f} - {self.times[-1]:.2f} s, nave={self.nave}>"


def AverageTFRArray(info: Info, data, times, freqs, *, nave: int = 1, comment: str = "",
                    method: str = ""):
    """Data object for *precomputed* spectrotemporal representations of averaged data (mne)."""
    return AverageTFR(info, data, times, freqs, nave, comment, method)


class EpochsTFR:
    """Data object for spectrotemporal representations of epoched data (mne).

    data: (n_epochs, n_channels, n_freqs, n_times) power/phase (real)."""
    info: Info
    data: np.ndarray[float, 4]
    times: np.ndarray[float, 1]
    freqs: np.ndarray[float, 1]
    events: np.ndarray[int, 2]
    event_id: Dict[str, int]
    method: str
    baseline: Optional[Tuple[float, float]]

    def __init__(self, info: Info, data, times, freqs, events = None, event_id = None,
                 method: str = ""):
        self.info = info
        self.data = np.asarray(data, dtype=float)
        self.times = np.asarray(times, dtype=float)
        self.freqs = np.asarray(freqs, dtype=float)
        n_ep = self.data.shape[0]
        if isinstance(events, NoneType):
            ev = np.zeros((n_ep, 3), dtype=int)
            for i in range(n_ep):
                ev[i, 0] = i
                ev[i, 2] = 1
            self.events = ev
        else:
            self.events = np.asarray(events).astype(int)
        if isinstance(event_id, NoneType):
            self.event_id = {"1": 1}
        else:
            self.event_id = event_id
        self.method = method
        self.baseline = None

    @property
    def ch_names(self):
        return self.info.ch_names

    @property
    def sfreq(self) -> float:
        return self.info.sfreq

    @property
    def shape(self):
        return self.data.shape

    @property
    def tmin(self) -> float:
        return float(self.times[0])

    @property
    def tmax(self) -> float:
        return float(self.times[-1])

    def __len__(self) -> int:
        return self.data.shape[0]

    def copy(self):
        o = EpochsTFR(self.info.copy(), self.data.copy(), self.times.copy(), self.freqs.copy(),
                      self.events.copy(), dict(self.event_id), self.method)
        o.baseline = self.baseline
        return o

    def get_data(self, picks = None, exclude = "bads"):
        pk = _picks_to_idx(self.info, picks, "data_or_ica", exclude)
        return self.data[:, np.array(pk)].copy()

    def apply_baseline(self, baseline, mode: str = "mean", verbose = None):
        bl = _bl_bounds(baseline, self.times)
        self.baseline = bl
        d2 = self.data.reshape((-1, self.data.shape[3]))
        _rescale_last(d2, self.times, bl[0], bl[1], mode)
        return self

    def crop(self, tmin = None, tmax = None, fmin = None, fmax = None, include_tmax: bool = True):
        tmn = self.tmin if isinstance(tmin, NoneType) else max(float(tmin), self.tmin)
        inc = include_tmax
        tmx = self.tmax
        if not isinstance(tmax, NoneType):
            if float(tmax) > self.tmax:
                inc = True
            else:
                tmx = float(tmax)
        tm = _time_mask(self.times, tmn, tmx, sfreq=self.info.sfreq, include_tmax=inc)
        ti = np.flatnonzero(tm)
        self.times = self.times[ti].copy()
        self.data = self.data[:, :, :, ti].copy()
        if not isinstance(fmin, NoneType) or not isinstance(fmax, NoneType):
            fm = _freq_mask(self.freqs, self.info.sfreq, _of(fmin), _of(fmax))
            fi = np.flatnonzero(fm)
            self.freqs = self.freqs[fi].copy()
            self.data = self.data[:, :, fi, :].copy()
        return self

    def average(self, method: str = "mean", *, dim: str = "epochs", copy: bool = False):
        """Average over epochs (returns AverageTFR). method: 'mean' | 'median'."""
        if dim != "epochs":
            raise ValueError("only dim='epochs' is supported")
        if method == "mean":
            d = self.data.mean(axis=0)
        elif method == "median":
            d = np.median(self.data, axis=0)
        else:
            raise ValueError(f"Combine option must be mean, median, got {method}")
        n = self.data.shape[0]
        return AverageTFR(self.info.copy(), d, self.times.copy(), self.freqs.copy(), n,
                          f"{method} of {n} EpochsTFR{'s' if n != 1 else ''}", self.method)

    def __getitem__(self, item):
        o = self.copy()
        if isinstance(item, int):
            o.data = self.data[item:item + 1].copy()
            o.events = self.events[item:item + 1].copy()
        else:
            idx = np.arange(self.data.shape[0])[item]
            o.data = self.data[idx].copy()
            o.events = self.events[idx].copy()
        return o

    def __repr__(self) -> str:
        return f"<EpochsTFR from Epochs, {self.method} method | {self.data.shape[0]} epochs × {self.data.shape[1]} channels × {self.freqs.shape[0]} freqs × {self.times.shape[0]} times>"


def EpochsTFRArray(info: Info, data, times, freqs, *, events = None, event_id = None,
                   method: str = ""):
    """Data object for *precomputed* spectrotemporal representations of epoched data (mne)."""
    return EpochsTFR(info, data, times, freqs, events, event_id, method)


def _inst_data(inst):
    from .._containers import Evoked
    if isinstance(inst, Evoked):
        return inst.data.reshape((1, inst.data.shape[0], inst.data.shape[1])), True
    else:
        return inst.get_data(), False


def _compute_tfr_inst(inst, method: str, freqs, n_cycles, zero_mean: bool, time_bandwidth: float,
                      use_fft: bool, decim, picks, output: str, tmin = None, tmax = None):
    data0, is_evoked = _inst_data(inst)
    tm = _time_mask(inst.times, _of(tmin), _of(tmax), sfreq=inst.info.sfreq)
    ti = np.flatnonzero(tm)
    data = np.ascontiguousarray(data0[:, :, ti])
    all_times = inst.times[ti].copy()
    pk = _picks_to_idx(inst.info, picks, "data", "bads")
    sub = np.ascontiguousarray(data[:, np.array(pk), :])
    out5, w = _compute_tfr_c(sub, freqs, inst.info.sfreq, method, n_cycles, zero_mean,
                             time_bandwidth, use_fft, decim, output)
    dec = _ensure_decim(decim)
    tidx = np.array(dec.indices(all_times.shape[0]), dtype=int)
    times = all_times[tidx].copy()
    info = inst.info._pick(pk)
    info.sfreq = info.sfreq / dec.step
    return out5, info, times, np.asarray(freqs, dtype=float).copy(), data.shape[0], is_evoked


def tfr_morlet(inst, freqs, n_cycles, use_fft: bool = False, return_itc: Literal[bool] = True,
               decim = 1, n_jobs = None, picks = None, zero_mean: bool = True,
               average: Literal[bool] = True, output: Literal[str] = "power", verbose = None):
    """Compute Time-Frequency Representation (TFR) using Morlet wavelets (mne, legacy).

    Returns AverageTFR, (AverageTFR power, AverageTFR itc) or EpochsTFR."""
    return _tfr_aux("morlet", inst, freqs, n_cycles, use_fft, return_itc, decim, picks,
                    zero_mean, 4.0, average, output)


def tfr_multitaper(inst, freqs, n_cycles, time_bandwidth: float = 4.0, use_fft: bool = True,
                   return_itc: Literal[bool] = True, decim = 1, n_jobs = None, picks = None,
                   average: Literal[bool] = True, *, verbose = None):
    """Compute Time-Frequency Representation (TFR) using DPSS tapers (mne, legacy)."""
    return _tfr_aux("multitaper", inst, freqs, n_cycles, use_fft, return_itc, decim, picks,
                    True, float(time_bandwidth), average, "power")


def _tfr_aux(method: str, inst, freqs, n_cycles, use_fft: bool, return_itc: Literal[bool],
             decim, picks, zero_mean: bool, time_bandwidth: float, average: Literal[bool],
             output: Literal[str], tmin = None, tmax = None):
    if average:
        if output == "complex":
            raise ValueError('output must be "power" if average=True')
        o = "avg_power_itc" if return_itc else "avg_power"
        out5, info, times, fr, nave, is_ev = _compute_tfr_inst(inst, method, freqs, n_cycles, zero_mean,
                                                         time_bandwidth, use_fft, decim, picks, o, tmin, tmax)
        if is_ev and return_itc:
            raise ValueError("return_itc must be False for evoked data")
        d = out5[0, :, 0]
        power = AverageTFR(info, d.real.copy(), times, fr, nave, "", method)
        if return_itc:
            itc = AverageTFR(info.copy(), d.imag.copy(), times.copy(), fr.copy(), nave, "", method)
            return power, itc
        else:
            return power
    else:
        if return_itc:
            raise ValueError("Inter-trial coherence is not supported with average=False")
        if output != "power" and output != "phase":
            compile_error("EpochsTFR supports output='power' or 'phase' (use tfr_array_* for complex)")
        out5, info, times, fr, nave, is_ev = _compute_tfr_inst(inst, method, freqs, n_cycles, zero_mean,
                                                         time_bandwidth, use_fft, decim, picks, output, tmin, tmax)
        if method == "multitaper" and output == "phase":
            raise ValueError("multitaper phase output has a taper dimension; use tfr_array_multitaper")
        d = out5[:, :, 0].real.copy()
        if is_ev:
            return EpochsTFR(info, d, times, fr, None, None, method)
        else:
            return EpochsTFR(info, d, times, fr, inst.events.copy(), dict(inst.event_id), method)


def compute_tfr(inst, method: str, freqs, *, tmin = None, tmax = None, picks = None,
                proj: bool = False, output: Literal[str] = "power", average: Literal[bool] = False,
                return_itc: Literal[bool] = False, decim = 1, n_jobs = None, verbose = None,
                n_cycles = 7.0, zero_mean = None, time_bandwidth: float = 4.0, use_fft: bool = True):
    """Epochs/Evoked.compute_tfr (method 'morlet' | 'multitaper')."""
    # tfr_array_morlet / tfr_array_multitaper both default to zero_mean=True
    zm = True if isinstance(zero_mean, NoneType) else bool(zero_mean)
    res = _tfr_aux(method, inst, freqs, n_cycles, use_fft, return_itc, decim, picks, zm,
                   float(time_bandwidth), average, output, tmin, tmax)
    return res


def tfr_stockwell(inst, fmin = None, fmax = None, n_fft = None, width: float = 1.0, decim = 1,
                  return_itc: Literal[bool] = False, n_jobs = None, verbose = None):
    """Compute Time-Frequency Representation (TFR) using Stockwell Transform (mne, legacy)."""
    data, is_ev = _inst_data(inst)
    if is_ev and return_itc:
        raise ValueError("return_itc must be False for evoked data")
    pk = _picks_to_idx(inst.info, None, "data", ())
    sub = np.ascontiguousarray(data[:, np.array(pk), :])
    power, itc, freqs = tfr_array_stockwell(sub, inst.info.sfreq, fmin, fmax, n_fft, width, decim, return_itc)
    dec = _ensure_decim(decim)
    times = inst.times[np.array(dec.indices(inst.times.shape[0]), dtype=int)].copy()
    info = inst.info._pick(pk)
    nave = data.shape[0]
    out = AverageTFR(info, power, times, freqs, nave, "", "stockwell-power")
    if return_itc:
        return out, AverageTFR(info.copy(), itc, times.copy(), freqs.copy(), nave, "", "stockwell-itc")
    else:
        return out
