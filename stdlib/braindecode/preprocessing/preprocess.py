# Ported from braindecode 2.0.0 (BSD-3-Clause)
#
# Preprocessor / preprocess and the array-level preprocessing functions.
# A Preprocessor holds a closure over its keyword arguments: `fn` given as a
# string literal resolves the mne RawArray method at compile time; a callable
# is either applied to the raw object (apply_on_array=False) or through
# RawArray.apply_function (apply_on_array=True, the braindecode default).

import numpy as np
from scipy.signal import lfilter
from mne.io import RawArray
from mne._info import _picks_to_idx
import mne
from ..datasets.base import BaseConcatDataset, RawDataset, EEGWindowsDataset

RawFn = Callable[[RawArray], None]


def _run_str(r: RawArray, fn: Literal[str], kw):
    if not hasattr(r, fn):
        compile_error("MNE object does not have this method")
    getattr(r, fn)(**kw)

def _mk_str(fn: Literal[str], kw):
    return lambda r: _run_str(r, fn, kw)

def _run_call(r: RawArray, fn, kw):
    fn(r, **kw)

def _mk_call(fn, kw):
    return lambda r: _run_call(r, fn, kw)

def _apply_kw(x, fn, kw):
    return fn(x, **kw)

def _run_array(r: RawArray, fn, channel_wise: Literal[bool], picks, kw):
    pk = _picks_to_idx(r.info, picks, "all", ())
    if channel_wise:
        for ci in pk:
            r._data[ci, :] = np.asarray(fn(r._data[ci, :].copy(), **kw))
    else:
        out = np.asarray(fn(r._data[np.array(pk, dtype=int), :].copy(), **kw))
        for k in range(len(pk)):
            r._data[pk[k], :] = out[k]

def _mk_array(fn, channel_wise: Literal[bool], picks, kw):
    return lambda r: _run_array(r, fn, channel_wise, picks, kw)


def _kw_repr(kw) -> str:
    s = str(kw)
    # NamedTuple repr is "(a=1, b=2)"
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    return s


class Preprocessor:
    """Preprocessor for an mne RawArray (braindecode Preprocessor)."""
    fn_name: str
    apply_on_array: bool
    kwargs_repr: str
    _function: RawFn

    def __init__(self, fn, *, apply_on_array: Literal[bool] = True, channel_wise: Literal[bool] = False,
                 picks = None, n_jobs: int = 1, **kwargs):
        self.fn_name = "<callable>"
        self.apply_on_array = apply_on_array
        self.kwargs_repr = _kw_repr(kwargs)
        if apply_on_array:
            self._function = _mk_array(fn, channel_wise, picks, kwargs)
        else:
            self._function = _mk_call(fn, kwargs)

    def __init__(self, fn: Literal[str], *, apply_on_array: bool = True, **kwargs):
        self.fn_name = fn
        self.apply_on_array = False
        self.kwargs_repr = _kw_repr(kwargs)
        self._function = _mk_str(fn, kwargs)

    def apply(self, raw: RawArray) -> RawArray:
        self._function(raw)
        return raw

    def __repr__(self) -> str:
        return f"Preprocessor(fn={repr(self.fn_name)}, apply_on_array={self.apply_on_array}, kwargs=({self.kwargs_repr}))"


def preprocess(concat_ds, preprocessors: List[Preprocessor], save_dir: Optional[str] = None,
               overwrite: bool = False, n_jobs = None, offset: int = 0, copy_data = None,
               parallel_kwargs = None, max_nbytes = "1M"):
    """Apply preprocessors in place to every recording (n_jobs is ignored)."""
    if save_dir is not None:
        raise NotImplementedError("save_dir is not supported in Codon braindecode")
    for ds in concat_ds.datasets:
        for p in preprocessors:
            p.apply(ds.raw)
        for p in preprocessors:
            ds.raw_preproc_kwargs.append(p.__repr__())
    concat_ds._recompute()
    return concat_ds


# -- named preprocessor classes mapping to mne RawArray methods

def Pick(picks, exclude = ()):
    return Preprocessor("pick", picks=picks, exclude=exclude)

def PickChannels(ch_names, ordered: bool = True):
    return Preprocessor("pick_channels", ch_names=ch_names, ordered=ordered)

def PickTypes(**kwargs):
    return Preprocessor("pick_types", **kwargs)

def DropChannels(ch_names, on_missing: str = "raise"):
    return Preprocessor("drop_channels", ch_names=ch_names, on_missing=on_missing)

def Filter(l_freq, h_freq, **kwargs):
    return Preprocessor("filter", l_freq=l_freq, h_freq=h_freq, **kwargs)

def Resample(sfreq: float, **kwargs):
    return Preprocessor("resample", sfreq=sfreq, **kwargs)

def Crop(tmin: float = 0.0, tmax: Optional[float] = None, include_tmax: bool = True):
    return Preprocessor("crop", tmin=tmin, tmax=tmax, include_tmax=include_tmax)

def SetEEGReference(ref_channels = "average", **kwargs):
    return Preprocessor("set_eeg_reference", ref_channels=ref_channels, **kwargs)


# -- array functions

def _exponential_moving_mean(data, factor_new: float):
    inv = 1.0 - factor_new
    n_times = data.shape[1]
    nan_mask = np.isnan(data)
    if not nan_mask.any():
        d = lfilter([1.0], [1.0, -inv], np.ones(n_times))
        n = lfilter([1.0], [1.0, -inv], data, axis=1)
        return n / d
    d = lfilter([1.0], [1.0, -inv], (~nan_mask).astype(float), axis=1)
    n = lfilter([1.0], [1.0, -inv], np.where(nan_mask, 0.0, data), axis=1)
    out = np.full(n.shape, np.nan)
    for i in range(n.shape[0]):
        for j in range(n.shape[1]):
            if d[i, j] > 0:
                out[i, j] = n[i, j] / d[i, j]
    return out


def exponential_moving_standardize(data, factor_new: float = 0.001,
                                   init_block_size: Optional[int] = None, eps: float = 1e-4):
    """Exponential moving standardization along the last axis of (n_ch, n_times)."""
    if not (0 < factor_new <= 1):
        raise ValueError(f"factor_new must be between 0 and 1, got {factor_new}")
    x = np.asarray(data).astype(float)
    meaned = _exponential_moving_mean(x, factor_new)
    demeaned = x - meaned
    squared = demeaned * demeaned
    square_ewmed = _exponential_moving_mean(squared, factor_new)
    standardized = demeaned / np.maximum(eps, np.sqrt(square_ewmed))
    if init_block_size is not None:
        b = init_block_size.__val__()
        blk = x[:, :b]
        m = np.mean(blk, axis=1, keepdims=True)
        s = np.std(blk, axis=1, keepdims=True)
        standardized[:, :b] = (blk - m) / np.maximum(eps, s)
    return standardized


def exponential_moving_demean(data, factor_new: float = 0.001, init_block_size: Optional[int] = None):
    """Exponential moving demeaning along the last axis of (n_ch, n_times)."""
    if not (0 < factor_new <= 1):
        raise ValueError(f"factor_new must be between 0 and 1, got {factor_new}")
    x = np.asarray(data).astype(float)
    demeaned = x - _exponential_moving_mean(x, factor_new)
    if init_block_size is not None:
        b = init_block_size.__val__()
        blk = x[:, :b]
        demeaned[:, :b] = blk - np.mean(blk, axis=1, keepdims=True)
    return demeaned


def filterbank(raw: RawArray, frequency_bands, drop_original_signals: bool = True,
               order_by_frequency_band: bool = False, **mne_filter_kwargs):
    """Add band-pass filtered copies of all channels (in place)."""
    if len(frequency_bands) == 0:
        raise ValueError(f"Expected at least one frequency band, got {frequency_bands}")
    orig = list(raw.info.ch_names)
    n_ch = len(orig)
    datas = List[np.ndarray[float, 2]]()
    names = List[List[str]]()
    types = list(raw.info.ch_types)
    for band in frequency_bands:
        l_freq, h_freq = band
        f = raw.copy()
        f.filter(l_freq=l_freq, h_freq=h_freq, **mne_filter_kwargs)
        datas.append(f._data)
        nm = List[str]()
        for o in orig:
            full = o + f"_{l_freq}-{h_freq}"
            nm.append(full[max(0, len(full) - 15):])
        names.append(nm)
    all_names = list(orig)
    all_types = list(types)
    rows = [raw._data]
    for k in range(len(datas)):
        all_names.extend(names[k])
        all_types.extend(types)
        rows.append(datas[k])
    stacked = np.concatenate(rows, axis=0)
    nb = len(datas) + 1
    order = List[int]()
    if order_by_frequency_band:
        order = list(range(n_ch * nb))
    else:
        for i in range(n_ch):
            for b in range(nb):
                order.append(b * n_ch + i)
    if drop_original_signals:
        order = [o for o in order if o >= n_ch]
    info = mne.create_info([all_names[o] for o in order], raw.info.sfreq, [all_types[o] for o in order])
    info.highpass = raw.info.highpass
    info.lowpass = raw.info.lowpass
    raw._data = stacked[np.array(order, dtype=int), :].copy()
    raw.info = info
    return raw
