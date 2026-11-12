# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# Additional BaseRaw API on RawArray: save, rename_channels,
# set_channel_types, reorder_channels, add_channels, append /
# concatenate_raws, notch_filter, __getitem__, set_montage hooks.

import numpy as np
from ._info import Info, _picks_to_idx
from ._containers import RawArray, EpochsArray, Evoked, _filter_picks, _update_filter_info
from .filter import _filter_inplace_3d
from .annotations import Annotations
from .io.fiff import _write_raw_fif


def _rename_info(info: Info, mapping):
    m = Dict[str, str]()
    if isinstance(mapping, Dict):
        for k, v in mapping.items():
            m[str(k)] = str(v)
    else:
        for c in info.ch_names:
            m[c] = str(mapping(c))
    info.rename_channels(m)


def _set_types_info(info: Info, mapping):
    m = Dict[str, str]()
    for k, v in mapping.items():
        m[str(k)] = str(v)
    info.set_channel_types(m)


def _reorder_idx(info: Info, ch_names) -> List[int]:
    idx = List[int]()
    for c in ch_names:
        ii = info.ch_names.index(str(c))
        if ii in idx:
            raise ValueError(f"Channel name repeated: {c}")
        idx.append(ii)
    return idx


@extend
class RawArray:
    def save(self, fname, picks = None, tmin: float = 0, tmax: Optional[float] = None,
             buffer_size_sec: Optional[float] = None, drop_small_buffer: bool = False,
             proj: bool = False, fmt: str = "single", overwrite: bool = False,
             split_size: str = "2GB", split_naming: str = "neuromag", verbose = None):
        if fmt != "single" and fmt != "double":
            raise ValueError(f"Typhon mne: Raw.save fmt must be 'single' or 'double', got {fmt}")
        _write_raw_fif(self, str(fname), picks, float(tmin), tmax, buffer_size_sec, fmt)

    def rename_channels(self, mapping, allow_duplicates: bool = False, *, verbose = None):
        _rename_info(self.info, mapping)
        return self

    def set_channel_types(self, mapping, *, on_unit_change: str = "warn", verbose = None):
        _set_types_info(self.info, mapping)
        return self

    def reorder_channels(self, ch_names):
        return self._pick_keep(_reorder_idx(self.info, ch_names))

    def add_channels(self, add_list, force_update_info: bool = False):
        n = self._data.shape[1]
        rows = [self._data]
        info = self.info.copy()
        for r in add_list:
            if r._data.shape[1] != n:
                raise ValueError(f"All data dimensions except channels must match, got {n} != {r._data.shape[1]}")
            for i in range(r.info.nchan):
                if r.info.ch_names[i] in info.ch_names:
                    raise ValueError(f"Channel names are not unique, found duplicates for: {{'{r.info.ch_names[i]}'}}")
                info._append_ch(r.info, i)
            rows.append(r._data)
        self._data = np.concatenate(rows, axis=0)
        self.info = info
        return self

    def append(self, raws, preload = None):
        """Concatenate raws in time; adds 'BAD boundary'/'EDGE boundary' annotations."""
        lst = List[RawArray]()
        if isinstance(raws, RawArray):
            lst.append(raws)
        else:
            for r in raws:
                lst.append(r)
        sf = self.info.sfreq
        ann = self.annotations.copy()
        edges = List[int]()
        datas = [self._data]
        # onsets of self.annotations are absolute (first_time-based)
        for r in lst:
            if r.info.nchan != self.info.nchan:
                raise ValueError("raw[%d]['info'][nchan] must match")
            n_one = sum([d.shape[1] for d in datas])
            edges.append(n_one)
            shift = n_one / sf + self.first_samp / sf - r.first_samp / sf
            for i in range(len(r.annotations)):
                ann.append(r.annotations.onset[i] + shift, r.annotations.duration[i],
                           r.annotations.description[i])
            datas.append(r._data)
        self._data = np.concatenate(datas, axis=1)
        self.times = np.arange(self._data.shape[1]).astype(float) / sf
        ft = self.first_time
        ann.onset = [o - ft for o in ann.onset]
        self.set_annotations(ann, False)
        for e in edges:
            on = e / sf + ft
            self.annotations.append(on, 0.0, "BAD boundary")
            self.annotations.append(on, 0.0, "EDGE boundary")
        return self

    def notch_filter(self, freqs, picks = None, filter_length = "auto", notch_widths = None,
                     trans_bandwidth: float = 1.0, n_jobs = None, method: str = "fir",
                     iir_params = None, mt_bandwidth = None, p_value: float = 0.05,
                     phase: str = "zero", fir_window: str = "hamming",
                     fir_design: str = "firwin", pad: str = "reflect_limited",
                     skip_by_annotation = ("edge", "bad_acq_skip"), verbose = None):
        """FIR notch filter (single or multiple frequencies) in place."""
        if method != "fir":
            raise NotImplementedError("only method='fir' notch filtering is supported in Typhon mne")
        fl = List[float]()
        if isinstance(freqs, float) or isinstance(freqs, int):
            fl.append(float(freqs))
        else:
            for f in freqs:
                fl.append(float(f))
        if len(fl) != 1:
            raise NotImplementedError("Typhon mne: notch_filter supports a single frequency per call (mne designs one combined multi-stop filter)")
        pk = _filter_picks(self.info, picks)
        tb = trans_bandwidth / 2.0
        for f in fl:
            nw = f / 200.0 if isinstance(notch_widths, NoneType) else float(notch_widths)
            lo = f - nw / 2.0 - tb
            hi = f + nw / 2.0 + tb
            _filter_inplace_3d(self._data, self.info.sfreq, hi, lo, pk, filter_length, tb, tb,
                               method, iir_params, phase, fir_window, fir_design, pad)
        return self

    def __getitem__(self, item):
        """raw[picks, start:stop] -> (data, times)."""
        pk = _picks_to_idx(self.info, item[0], "all", ())
        sl = item[1]
        idx = list(range(self._data.shape[1]))[sl]
        sa = np.array(idx, dtype=int)
        return self._data[np.array(pk, dtype=int)][:, sa].copy(), self.times[sa].copy()


def concatenate_raws(raws, preload = None, events_list = None, *, on_mismatch: str = "raise",
                     verbose = None):
    first = raws[0]
    rest = List[RawArray]()
    for k in range(1, len(raws)):
        rest.append(raws[k])
    if not isinstance(events_list, NoneType):
        from .event import concatenate_events
        fs = [r.first_samp for r in raws]
        ls = [r.last_samp for r in raws]
        ev = concatenate_events(events_list, fs, ls)
        first.append(rest)
        return first, ev
    else:
        return first.append(rest)


@extend
class EpochsArray:
    def rename_channels(self, mapping, allow_duplicates: bool = False, *, verbose = None):
        _rename_info(self.info, mapping)
        return self

    def set_channel_types(self, mapping, *, on_unit_change: str = "warn", verbose = None):
        _set_types_info(self.info, mapping)
        return self

    def _pick_keep(self, keep: List[int]):
        self._data = self._data[:, np.array(keep, dtype=int), :].copy()
        self.info = self.info._pick(keep)
        return self

    def pick(self, picks, exclude = (), *, verbose = None):
        return self._pick_keep(_picks_to_idx(self.info, picks, "all", exclude))

    def drop_channels(self, ch_names, on_missing: str = "raise"):
        names = [ch_names] if isinstance(ch_names, str) else [str(c) for c in ch_names]
        for c in names:
            if c not in self.info.ch_names and on_missing == "raise":
                raise ValueError(f"Channel(s) {c} not found, nothing dropped.")
        return self._pick_keep([i for i in range(self.info.nchan) if self.info.ch_names[i] not in names])

    def reorder_channels(self, ch_names):
        return self._pick_keep(_reorder_idx(self.info, ch_names))


@extend
class Evoked:
    def rename_channels(self, mapping, allow_duplicates: bool = False, *, verbose = None):
        _rename_info(self.info, mapping)
        return self

    def _pick_keep(self, keep: List[int]):
        self.data = self.data[np.array(keep, dtype=int), :].copy()
        self.info = self.info._pick(keep)
        return self

    def pick(self, picks, exclude = (), *, verbose = None):
        return self._pick_keep(_picks_to_idx(self.info, picks, "all", exclude))

    def save(self, fname, *, overwrite: bool = False, verbose = None):
        from .epochs import write_evokeds
        write_evokeds(fname, self)
