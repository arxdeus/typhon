# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# mne/epochs.py subset: Epochs(raw, events, ...) (built eagerly; values equal
# to mne's preload=True and preload=False paths), drop logs, reject/flat,
# decim, detrend, __getitem__, drop, equalize_event_counts, save /
# read_epochs, and Evoked write/read (write_evokeds / read_evokeds),
# combine_evoked.

import numpy as np
from ._info import Info, _picks_to_idx, _time_mask
from ._containers import (EpochsArray, Evoked, RawArray, _check_baseline, _rescale_3d,
                          _detrend_picks, _opt_float)
from .io._fiff import (FifFile, FifWriter, read_meas_info, FIFFB_MEAS, FIFFB_PROCESSED_DATA,
                       FIFFB_MNE_EPOCHS, FIFFB_MNE_EVENTS, FIFF_MNE_EVENT_LIST, FIFF_COMMENT,
                       FIFF_FIRST_SAMPLE, FIFF_LAST_SAMPLE, FIFF_MNE_EPOCHS_RAW_SFREQ,
                       FIFF_MNE_BASELINE_MIN, FIFF_MNE_BASELINE_MAX, FIFF_EPOCH,
                       FIFF_MNE_EPOCHS_DROP_LOG, FIFF_MNE_EPOCHS_SELECTION, FIFF_BLOCK_ID,
                       FIFFB_EVOKED, FIFFB_ASPECT, FIFF_FIRST_TIME, FIFF_NO_SAMPLES,
                       FIFF_ASPECT_KIND, FIFF_NAVE, _parse_drop_log, _dump_drop_log)

_REJECT_TYPES = ("mag", "grad", "eeg", "csd", "seeg", "ecog", "dbs", "hbo", "hbr",
                 "fnirs_cw_amplitude", "fnirs_fd_ac_amplitude", "fnirs_fd_phase",
                 "fnirs_od", "eog", "ecg", "emg", "ref_meg", "misc", "resp", "chpi",
                 "exci", "ias", "syst", "bio", "temperature", "gsr", "dipole", "gof",
                 "stim", "eyegaze", "pupil")


def _event_id_dict(event_id, events: np.ndarray[int, 2]) -> Dict[str, int]:
    """_check_event_id."""
    ids = Dict[str, int]()
    if isinstance(event_id, NoneType):
        vals = sorted(set([int(events[i, 2]) for i in range(events.shape[0])]))
        for v in vals:
            ids[str(v)] = v
    elif isinstance(event_id, int):
        ids[str(event_id)] = event_id
    elif isinstance(event_id, Dict):
        for k, v in event_id.items():
            ids[k] = v
    else:
        for v in event_id:
            ids[str(v)] = int(v)
    return ids


def _reject_dict(d) -> Optional[Dict[str, float]]:
    if isinstance(d, NoneType):
        return None
    else:
        out = Dict[str, float]()
        for k, v in d.items():
            if k not in _REJECT_TYPES:
                raise KeyError(f"Unknown channel types found in reject/flat: {{'{k}'}}")
            if v < 0:
                raise ValueError(f"If using numerical criteria, the value must be >= 0, not {v}")
            out[k] = float(v)
        return out if len(out) > 0 else None


def _is_good(e: np.ndarray[float, 2], info: Info, reject: Optional[Dict[str, float]],
             flat: Optional[Dict[str, float]], t0: int, t1: int) -> List[str]:
    """mne.epochs._is_good with full_report=True; returns bad names (empty if good)."""
    bad = List[str]()
    for pass_ in range(2):
        refl = reject if pass_ == 0 else flat
        if refl is None:
            continue
        for key, crit in refl.__val__().items():
            for c in range(info.nchan):
                if info.ch_types[c] != key or info.ch_names[c] in info.bads:
                    continue
                mx = -np.inf
                mn = np.inf
                for j in range(t0, t1):
                    v = e[c, j]
                    if v > mx:
                        mx = v
                    if v < mn:
                        mn = v
                d = mx - mn
                if (pass_ == 0 and d > crit) or (pass_ == 1 and d < crit):
                    bad.append(info.ch_names[c])
    return bad


def Epochs(raw, events = None, event_id = None, tmin: float = -0.2, tmax: float = 0.5,
           baseline = (None, 0), picks = None, preload: bool = False, reject = None,
           flat = None, proj: bool = True, decim: int = 1,
           reject_tmin: Optional[float] = None, reject_tmax: Optional[float] = None,
           detrend: Optional[int] = None, on_missing: str = "raise",
           reject_by_annotation: bool = True, metadata = None,
           event_repeated: str = "error", verbose = None) -> EpochsArray:
    """Extract epochs from a Raw instance (mne.Epochs). Always returns loaded
    data (identical to mne after load_data())."""
    if not isinstance(metadata, NoneType):
        compile_error("Typhon mne: Epochs(metadata=...) is not supported")
    ev = np.zeros((0, 3), dtype=int)
    if isinstance(events, NoneType):
        from .annotations import events_from_annotations
        ev2, ids2 = events_from_annotations(raw)
        ev = ev2
        if isinstance(event_id, NoneType):
            event_id = ids2
    else:
        ev = np.asarray(events).astype(int)
    ids = _event_id_dict(event_id, ev)
    for k, v in ids.items():
        found = False
        for i in range(ev.shape[0]):
            if int(ev[i, 2]) == v:
                found = True
        if not found:
            msg = f"No matching events found for {k} (event id {v})"
            if on_missing == "raise":
                raise ValueError(msg)
            elif on_missing == "warn":
                print("RuntimeWarning:", msg)
    vals = list(ids.values())
    n_all = ev.shape[0]
    selected = [i for i in range(n_all) if int(ev[i, 2]) in vals]
    drop_log = [List[str]() if k in selected else ["IGNORED"] for k in range(n_all)]
    if len(selected) == 0:
        raise ValueError("No desired events found.")
    times_seen = Set[int]()
    for i in selected:
        s = int(ev[i, 0])
        if s in times_seen and event_repeated == "error":
            raise RuntimeError("Event time samples were not unique. Consider setting the `event_repeated` parameter.")
        times_seen.add(s)
    if not (detrend is None or detrend.__val__() == 0 or detrend.__val__() == 1):
        raise ValueError("detrend must be None, 0, or 1")
    if tmin > tmax:
        raise ValueError("tmin has to be less than or equal to tmax")
    pk = _picks_to_idx(raw.info, picks, "all", ())
    info = raw.info._pick(pk)
    sfreq = info.sfreq
    start_idx = int(round(tmin * sfreq))
    raw_times = np.arange(start_idx, int(round(tmax * sfreq)) + 1).astype(float) / sfreq
    n_rt = raw_times.shape[0]
    # decimation (BaseEpochs.decimate on not-preloaded data)
    if decim < 1:
        raise ValueError("decim must be an integer > 0")
    i_start = int(round(-raw_times[0] * sfreq)) % decim
    dsel = list(range(i_start, n_rt, decim))
    times = raw_times[np.array(dsel, dtype=int)].copy()
    rtmin: Optional[float] = reject_tmin
    rtmax: Optional[float] = reject_tmax
    if rtmin is not None:
        if np.isclose(rtmin.__val__(), tmin):
            rtmin = float(times[0])
        elif rtmin.__val__() < tmin:
            raise ValueError(f"reject_tmin needs to be None or >= tmin (got {rtmin})")
    if rtmax is not None:
        if np.isclose(rtmax.__val__(), tmax):
            rtmax = float(times[-1])
        elif rtmax.__val__() > tmax:
            raise ValueError(f"reject_tmax needs to be None or <= tmax (got {rtmax})")
    bl: Optional[Tuple[float, float]] = None
    if not isinstance(baseline, NoneType):
        bl = _check_baseline(baseline, times, sfreq / decim)
    rej = _reject_dict(reject)
    fl = _reject_dict(flat)
    for dd in (rej, fl):
        if dd is not None:
            for k in dd.__val__().keys():
                if k not in info.ch_types:
                    raise ValueError(f"No {k.upper()} channel found. Cannot reject based on {k.upper()}.")
    nt = times.shape[0]
    rt0 = 0
    rt1 = nt
    if rtmin is not None or rtmax is not None:
        if rtmin is not None:
            for j in range(nt):
                if times[j] >= rtmin.__val__():
                    rt0 = j
                    break
        if rtmax is not None:
            for j in range(nt):
                if times[j] <= rtmax.__val__():
                    rt1 = j  # mne: slice(imin, idxs[-1]) excludes the last index
    dpk = _detrend_picks(info)
    n_raw = raw.n_times
    first_samp = raw.first_samp
    ann = raw.annotations
    ft = raw.first_time
    good = List[int]()
    datas = List[np.ndarray[float, 2]]()
    sel_arr = np.array(pk, dtype=int)
    for idx in selected:
        es = int(ev[idx, 0])
        start = int(round(es + raw_times[0] * sfreq)) - first_samp
        stop = start + n_rt
        if start < 0:
            drop_log[idx].append("NO_DATA")
            continue
        rs_tmin = float(raw_times[0]) if reject_tmin is None else reject_tmin.__val__()
        reject_start = int(round(es + rs_tmin * sfreq)) - first_samp
        rs_tmax = float(raw_times[-1]) if reject_tmax is None else reject_tmax.__val__()
        reject_stop = stop - int(round((float(raw_times[-1]) - rs_tmax) * sfreq))
        bad_desc = ""
        if reject_by_annotation and len(ann) > 0:
            for a in range(len(ann)):
                on = ann.onset[a] - ft
                if on < reject_stop / sfreq and on + ann.duration[a] > reject_start / sfreq:
                    if ann.description[a].lower().startswith("bad"):
                        bad_desc = ann.description[a]
                        break
        if bad_desc != "":
            drop_log[idx].append(bad_desc)
            continue
        if stop > n_raw:
            drop_log[idx].append("TOO_SHORT")
            continue
        ep = raw._data[sel_arr, start:stop].copy()
        if detrend is not None:
            from scipy.signal import detrend as _sdt
            if len(dpk) > 0:
                dsub = ep[np.array(dpk, dtype=int), :]
                dd = _sdt(dsub, axis=1, type="constant" if detrend.__val__() == 0 else "linear")
                for k in range(len(dpk)):
                    ep[dpk[k], :] = dd[k]
        if bl is not None:
            e3 = ep.reshape((1, ep.shape[0], ep.shape[1]))
            _rescale_3d(e3, raw_times, bl.__val__(), dpk)
            ep = e3.reshape((ep.shape[0], ep.shape[1]))
        ep = ep[:, np.array(dsel, dtype=int)].copy()
        bads = _is_good(ep, info, rej, fl, rt0, rt1)
        if len(bads) > 0:
            for b in bads:
                drop_log[idx].append(b)
            continue
        good.append(idx)
        datas.append(ep)
    data = np.zeros((len(good), info.nchan, nt))
    for k in range(len(good)):
        data[k] = datas[k]
    gev = np.zeros((len(good), 3), dtype=int)
    for k in range(len(good)):
        for j in range(3):
            gev[k, j] = ev[good[k], j]
    info.sfreq = sfreq / decim
    out = EpochsArray(data, info, gev, 0.0, ids)
    out.times = times
    out.baseline = bl
    out.drop_log = drop_log
    out.selection = good
    out.reject = rej
    out.flat = fl
    out._raw_sfreq = sfreq
    out._kind = "Epochs"
    if len(good) == 0:
        print("RuntimeWarning: All epochs were dropped!")
    return out


def _match_event_names(event_id: Dict[str, int], keys: List[str]) -> List[str]:
    out = Set[str]()
    for key in keys:
        parts = key.split("/")
        for name in event_id.keys():
            nparts = name.split("/")
            if all([p in nparts for p in parts]):
                out.add(name)
    if len(out) == 0:
        raise KeyError(f'Event name "{keys[-1]}" could not be found. The following events are present in the data: {", ".join(list(event_id.keys()))}')
    return sorted(out)


@extend
class EpochsArray:
    def _item_to_select(self, item) -> List[int]:
        n = self.events.shape[0]
        if isinstance(item, str):
            return self._keys_to_idx([item])
        elif isinstance(item, slice):
            return list(range(n))[item]
        elif isinstance(item, int):
            i = item if item >= 0 else item + n
            return [i]
        elif isinstance(item, List[str]):
            return self._keys_to_idx([str(x) for x in item])
        else:
            a = np.atleast_1d(np.asarray(item))
            if a.dtype is bool:
                return [i for i in range(a.shape[0]) if a[i]]
            return [int(a[i]) if int(a[i]) >= 0 else int(a[i]) + n for i in range(a.shape[0])]

    def _keys_to_idx(self, keys: List[str]) -> List[int]:
        names = _match_event_names(self.event_id, keys)
        codes = [self.event_id[k] for k in names]
        return [i for i in range(self.events.shape[0]) if int(self.events[i, 2]) in codes]

    def _getitem(self, sel: List[int], reason: str, drop_event_id: bool):
        """Restrict in place to sel (BaseEpochs._getitem, copy=False)."""
        keep_sel = [self.selection[i] for i in sel]
        for s in self.selection:
            if s not in keep_sel:
                self.drop_log[s] = [reason]
        self.selection = keep_sel
        sa = np.array(sel, dtype=int)
        self.events = np.atleast_2d(self.events[sa, :]).copy() if len(sel) > 0 else np.zeros((0, 3), dtype=int)
        self._data = self._data[sa].copy() if len(sel) > 0 else np.zeros((0, self._data.shape[1], self._data.shape[2]))
        if drop_event_id:
            present = Set[int]([int(self.events[i, 2]) for i in range(self.events.shape[0])])
            nid = Dict[str, int]()
            for k, v in self.event_id.items():
                if v in present:
                    nid[k] = v
            self.event_id = nid
        return self

    def __getitem__(self, item):
        sel = self._item_to_select(item)
        return self.copy()._getitem(sel, "IGNORED", True)

    def drop(self, indices, reason: str = "USER", verbose = None):
        n = self.events.shape[0]
        idx = self._item_to_select(indices) if not isinstance(indices, str) else List[int]()
        for i in idx:
            if i < 0 or i >= n:
                raise IndexError(f"Epoch index {i} is out of bounds")
        keep = [i for i in range(n) if i not in idx]
        return self._getitem(keep, reason, False)

    def drop_bad(self, reject = "existing", flat = "existing", verbose = None):
        """Apply (stricter) reject/flat criteria to loaded data."""
        rej: Optional[Dict[str, float]] = self.reject
        fl: Optional[Dict[str, float]] = self.flat
        if not isinstance(reject, str):
            r = _reject_dict(reject)
            if r is not None:
                base = Dict[str, float]() if rej is None else rej.__val__().copy()
                for k, v in r.__val__().items():
                    if k in base and v > base[k]:
                        raise ValueError(f'reject["{k}"] == {v} > {base[k]} (old value), new reject values must be at least as stringent as previous ones')
                    base[k] = v
                rej = base
        if not isinstance(flat, str):
            f = _reject_dict(flat)
            if f is not None:
                base = Dict[str, float]() if fl is None else fl.__val__().copy()
                for k, v in f.__val__().items():
                    if k in base and v < base[k]:
                        raise ValueError(f'flat["{k}"] == {v} < {base[k]} (old value), new flat values must be at least as stringent as previous ones')
                    base[k] = v
                fl = base
        self.reject = rej
        self.flat = fl
        if rej is None and fl is None:
            return self
        keep = List[int]()
        for i in range(self._data.shape[0]):
            bads = _is_good(self._data[i], self.info, rej, fl, 0, self.times.shape[0])
            if len(bads) > 0:
                for b in bads:
                    self.drop_log[self.selection[i]].append(b)
            else:
                keep.append(i)
        sel_keep = [self.selection[i] for i in keep]
        self.selection = sel_keep
        ka = np.array(keep, dtype=int)
        self.events = self.events[ka, :].copy()
        self._data = self._data[ka].copy()
        return self

    def decimate(self, decim: int, offset: int = 0, *, verbose = None):
        if decim < 1:
            raise ValueError("decim must be an integer > 0")
        if not (0 <= offset < decim):
            raise ValueError(f"decim must be at least 0 and less than {decim}, got {offset}")
        sf = self.info.sfreq
        start_idx = int(round(-float(self.times[0]) * sf))
        i_start = start_idx % decim + offset
        sel = np.array(list(range(i_start, self.times.shape[0], decim)), dtype=int)
        self._data = self._data[:, :, sel].copy()
        self.times = self.times[sel].copy()
        self.info.sfreq = sf / decim
        return self

    def load_data(self):
        return self

    def equalize_event_counts(self, event_ids = None, method: str = "mintime", *,
                              random_state = None):
        keys = List[List[str]]()
        if isinstance(event_ids, NoneType):
            for k in self.event_id.keys():
                keys.append([k])
        else:
            for e in event_ids:
                if isinstance(e, str):
                    keys.append([e])
                else:
                    keys.append([str(x) for x in e])
        if len(keys) == 0:
            raise ValueError("event_ids must have at least one element")
        eq = [self._keys_to_idx(k) for k in keys]
        nums = [[int(self.events[i, 0]) for i in e] for e in eq]
        small = 0
        for k in range(len(nums)):
            if len(nums[k]) < len(nums[small]):
                small = k
        drop = List[int]()
        for k in range(len(nums)):
            keep = List[bool]()
            if method == "mintime":
                keep = _minimize_time_diff(nums[small], nums[k])
            elif method == "truncate":
                keep = [j < len(nums[small]) for j in range(len(nums[k]))]
            else:
                raise ValueError(f"Invalid value for the 'method' parameter. Allowed values are 'mintime', 'truncate', and 'random', but got '{method}' instead.")
            for j in range(len(keep)):
                if not keep[j]:
                    drop.append(eq[k][j])
        self.drop(drop, reason="EQUALIZED_COUNT")
        return self, np.array(drop, dtype=int)

    def save(self, fname, split_size: str = "2GB", fmt: str = "single", overwrite: bool = False,
             split_naming: str = "neuromag", verbose = None):
        _write_epochs_fif(self, str(fname), fmt)


def _interp_lin(xp_n: int, fp: List[float], x: int, fill: float) -> float:
    # interp1d(arange(xp_n), fp, bounds_error=False, fill_value=fill) at integer x
    if x < 0 or x > xp_n - 1:
        return fill
    return fp[x]


def _minimize_time_diff(t_shorter: List[int], t_longer: List[int]) -> List[bool]:
    nl = len(t_longer)
    ns = len(t_shorter)
    keep = [True for _ in range(nl)]
    if ns < 2:
        keep = [False for _ in range(nl)]
        if ns == 1:
            bi = 0
            for i in range(nl):
                if abs(t_longer[i] - t_shorter[0]) < abs(t_longer[bi] - t_shorter[0]):
                    bi = i
            keep[bi] = True
        return keep
    ts = [float(v) for v in t_shorter]
    for ii in range(nl - ns):
        scores = [np.inf for _ in range(nl)]
        kept = [i for i in range(nl) if keep[i]]
        n2 = nl - ii - 1
        for r in kept:
            tk = [float(t_longer[i]) for i in kept if i != r]
            d = 0.0
            for x in range(ns):
                d += abs(_interp_lin(n2, tk, x, tk[-1]) - ts[x])
            for x in range(n2):
                d += abs(_interp_lin(ns, ts, x, ts[-1]) - tk[x])
            scores[r] = d
        bi = 0
        for i in range(nl):
            if scores[i] < scores[bi]:
                bi = i
        keep[bi] = False
    return keep


def _write_epochs_fif(ep: EpochsArray, fname: str, fmt: str):
    info = ep.info
    w = FifWriter()
    w.start_block(FIFFB_MEAS)
    w.write_id(FIFF_BLOCK_ID)
    w.write_meas_info(info)
    w.start_block(FIFFB_PROCESSED_DATA)
    w.start_block(FIFFB_MNE_EPOCHS)
    w.start_block(FIFFB_MNE_EVENTS)
    v = List[int]()
    for i in range(ep.events.shape[0]):
        for j in range(3):
            v.append(int(ep.events[i, j]))
    w.write_ints(FIFF_MNE_EVENT_LIST, v)
    w.write_string(FIFF_COMMENT, ";".join([k + ":" + str(val) for k, val in ep.event_id.items()]))
    w.end_block(FIFFB_MNE_EVENTS)
    first = int(round(float(ep.times[0]) * info.sfreq))
    w.write_int(FIFF_FIRST_SAMPLE, first)
    w.write_int(FIFF_LAST_SAMPLE, first + ep.times.shape[0] - 1)
    w.write_float(FIFF_MNE_EPOCHS_RAW_SFREQ, ep._raw_sfreq)
    if ep.baseline is not None:
        b = ep.baseline.__val__()
        w.write_float(FIFF_MNE_BASELINE_MIN, b[0])
        w.write_float(FIFF_MNE_BASELINE_MAX, b[1])
    d = ep._data.copy()
    for c in range(info.nchan):
        d[:, c, :] /= info.cals[c]
    w.write_float_matrix(FIFF_EPOCH, d.ravel(), [d.shape[0], d.shape[1], d.shape[2]])
    w.write_string(FIFF_MNE_EPOCHS_DROP_LOG, _dump_drop_log(ep.drop_log))
    w.write_ints(FIFF_MNE_EPOCHS_SELECTION, ep.selection)
    w.end_block(FIFFB_MNE_EPOCHS)
    w.end_block(FIFFB_PROCESSED_DATA)
    w.end_block(FIFFB_MEAS)
    w.end_file(fname)


def read_epochs(fname, proj: bool = True, preload: bool = True, verbose = None) -> EpochsArray:
    f = FifFile(str(fname))
    info = read_meas_info(f)
    nodes = f.tree.find(FIFFB_MNE_EPOCHS)
    if len(nodes) == 0:
        raise ValueError("Could not find epochs data")
    nd = nodes[0]
    first = 0
    last = 0
    raw_sfreq = info.sfreq
    bmin: Optional[float] = None
    bmax: Optional[float] = None
    flat = np.zeros(0)
    dims = List[int]()
    sel = List[int]()
    dlog = List[List[str]]()
    has_log = False
    for t in nd.entries:
        if t.kind == FIFF_FIRST_SAMPLE:
            first = f.int(t)
        elif t.kind == FIFF_LAST_SAMPLE:
            last = f.int(t)
        elif t.kind == FIFF_MNE_EPOCHS_RAW_SFREQ:
            raw_sfreq = f.float(t)
        elif t.kind == FIFF_MNE_BASELINE_MIN:
            bmin = f.float(t)
        elif t.kind == FIFF_MNE_BASELINE_MAX:
            bmax = f.float(t)
        elif t.kind == FIFF_EPOCH:
            flat, dims = f.matrix(t)
        elif t.kind == FIFF_MNE_EPOCHS_SELECTION:
            sel = f.ints(t)
        elif t.kind == FIFF_MNE_EPOCHS_DROP_LOG:
            dlog = _parse_drop_log(f.string(t))
            has_log = True
    en = nd.find(FIFFB_MNE_EVENTS)[0]
    et = en.tag(FIFF_MNE_EVENT_LIST)
    evl = f.ints(et.__val__())
    events = np.array(evl, dtype=int).reshape((len(evl) // 3, 3))
    ids = Dict[str, int]()
    ct = en.tag(FIFF_COMMENT)
    if ct is not None:
        for m in f.string(ct.__val__()).split(";"):
            j = m.rfind(":")
            ids[m[:j]] = int(m[j + 1:])
    n_ep = events.shape[0]
    data = np.zeros((n_ep, info.nchan, last - first + 1))
    if len(dims) == 3:
        data = flat.reshape((dims[0], dims[1], dims[2]))
    for c in range(info.nchan):
        data[:, c, :] *= info.cals[c]
    ep = EpochsArray(data, info, events, 0.0, ids if len(ids) > 0 else None)
    ep.times = np.arange(first, last + 1).astype(float) / info.sfreq
    if bmin is not None and bmax is not None:
        ep.baseline = (bmin.__val__(), bmax.__val__())
    if len(sel) == n_ep:
        ep.selection = sel
    if has_log:
        ep.drop_log = dlog
    ep._raw_sfreq = raw_sfreq
    ep._kind = "EpochsFIF"
    return ep


read_epochs_fif = read_epochs


def write_evokeds(fname, evoked, *, on_mismatch: str = "raise", overwrite: bool = False,
                  verbose = None):
    evs = List[Evoked]()
    if isinstance(evoked, Evoked):
        evs.append(evoked)
    else:
        for e in evoked:
            evs.append(e)
    if len(evs) == 0:
        raise ValueError("No evoked data to write")
    w = FifWriter()
    w.start_block(FIFFB_MEAS)
    w.write_id(FIFF_BLOCK_ID)
    w.write_meas_info(evs[0].info)
    w.start_block(FIFFB_PROCESSED_DATA)
    for e in evs:
        w.start_block(FIFFB_EVOKED)
        if len(e.comment) > 0:
            w.write_string(FIFF_COMMENT, e.comment)
        w.write_float(FIFF_FIRST_TIME, float(e.times[0]))
        w.write_int(FIFF_NO_SAMPLES, e.times.shape[0])
        w.write_int(FIFF_FIRST_SAMPLE, e.first)
        w.write_int(FIFF_LAST_SAMPLE, e.last)
        if e.baseline is not None:
            b = e.baseline.__val__()
            w.write_float(FIFF_MNE_BASELINE_MIN, b[0])
            w.write_float(FIFF_MNE_BASELINE_MAX, b[1])
        w.start_block(FIFFB_ASPECT)
        w.write_int(FIFF_ASPECT_KIND, 100 if e.kind == "average" else 101)
        w.write_int(FIFF_NAVE, int(round(float(e.nave))))
        d = e.data.copy()
        for c in range(e.info.nchan):
            d[c, :] /= e.info.cals[c]
        w.write_float_matrix(FIFF_EPOCH, d.ravel(), [d.shape[0], d.shape[1]])
        w.end_block(FIFFB_ASPECT)
        w.end_block(FIFFB_EVOKED)
    w.end_block(FIFFB_PROCESSED_DATA)
    w.end_block(FIFFB_MEAS)
    w.end_file(str(fname))


def read_evokeds(fname, condition = None, baseline = None, kind: str = "average",
                 proj: bool = True, allow_maxshield: bool = False, verbose = None):
    f = FifFile(str(fname))
    info = read_meas_info(f)
    out = List[Evoked]()
    for nd in f.tree.find(FIFFB_EVOKED):
        comment = ""
        first = 0
        last = 0
        bmin: Optional[float] = None
        bmax: Optional[float] = None
        for t in nd.entries:
            if t.kind == FIFF_COMMENT:
                comment = f.string(t)
            elif t.kind == FIFF_FIRST_SAMPLE:
                first = f.int(t)
            elif t.kind == FIFF_LAST_SAMPLE:
                last = f.int(t)
            elif t.kind == FIFF_MNE_BASELINE_MIN:
                bmin = f.float(t)
            elif t.kind == FIFF_MNE_BASELINE_MAX:
                bmax = f.float(t)
        asp = nd.find(FIFFB_ASPECT)[0]
        nave = 1
        akind = 100
        data = np.zeros((info.nchan, last - first + 1))
        for t in asp.entries:
            if t.kind == FIFF_NAVE:
                nave = f.int(t)
            elif t.kind == FIFF_ASPECT_KIND:
                akind = f.int(t)
            elif t.kind == FIFF_EPOCH:
                flat, dims = f.matrix(t)
                data = flat.reshape((dims[0], dims[1]))
        for c in range(info.nchan):
            data[c, :] *= info.cals[c]
        times = np.arange(first, last + 1).astype(float) / info.sfreq
        e = Evoked(data, info.copy(), times, nave, comment)
        e.kind = "average" if akind == 100 else "standard_error"
        if bmin is not None and bmax is not None:
            e.baseline = (bmin.__val__(), bmax.__val__())
        out.append(e)
    if not isinstance(baseline, NoneType):
        for e in out:
            e.apply_baseline(baseline)
    if isinstance(condition, int):
        if condition >= len(out) or condition < -len(out):
            raise ValueError(f"Dataset {condition} was not found")
        return out[condition]
    elif isinstance(condition, str):
        for e in out:
            if e.comment == condition:
                return e
        raise ValueError(f'Could not find evoked data with comment "{condition}"')
    else:
        return out


def combine_evoked(all_evoked, weights) -> Evoked:
    evs = [e for e in all_evoked]
    naves = np.array([float(e.nave) for e in evs])
    w = np.zeros(len(evs))
    if isinstance(weights, str):
        if weights == "nave":
            w = naves / naves.sum()
        elif weights == "equal":
            w = np.ones(len(evs)) / len(evs)
        else:
            raise ValueError(f"Invalid value for the 'weights' parameter. Allowed values are 'nave' and 'equal', but got '{weights}' instead.")
    else:
        w = np.array([float(x) for x in weights])
    if w.shape[0] != len(evs):
        raise ValueError("weights must be the same size as all_evoked")
    new_nave = 1.0 / np.sum(w ** 2 / naves)
    out = evs[0].copy()
    bads = Set[str]()
    for e in evs:
        for b in e.info.bads:
            bads.add(b)
    out.info.bads = list(bads)
    d = np.zeros(out.data.shape)
    for k in range(len(evs)):
        d = d + w[k] * evs[k].data
    out.data = d
    out.nave = int(round(new_nave))
    comment = ""
    for k in range(len(evs)):
        wk = float(w[k])
        sign = "" if wk >= 0 else "-"
        weight = "" if np.isclose(abs(wk), 1.0) else f"{abs(wk):0.3f}"
        mult = " × " if weight != "" else ""
        c = evs[k].comment
        this = f"({c})" if " + " in c else (c if c != "" else "unknown")
        if k == 0:
            comment += f"{sign}{weight}{mult}{this}"
        else:
            comment += f" {sign if sign != '' else '+'} {weight}{mult}{this}"
    out.comment = comment.replace(" - - ", " + ")
    return out


def EvokedArray(data, info: Info, tmin: float = 0.0, comment: str = "", nave: int = 1,
                kind: str = "average", baseline = None, *, verbose = None) -> Evoked:
    """Evoked object from numpy array (mne.EvokedArray)."""
    d = np.asarray(data)
    if d.ndim != 2:
        compile_error("Data must be a 2D array of shape (n_channels, n_samples)")
    x = d.astype(float)
    if x.shape[0] != info.nchan:
        raise ValueError(f"Info ({info.nchan}) and data ({x.shape[0]}) must have same number of channels.")
    if kind != "average" and kind != "standard_error":
        raise ValueError(f"Invalid value for the 'kind' parameter. Allowed values are 'average' and 'standard_error', but got '{kind}' instead.")
    first = int(round(tmin * info.sfreq))
    times = np.arange(first, first + x.shape[1]).astype(float) / info.sfreq
    e = Evoked(x, info.copy(), times, nave, comment)
    e.kind = kind
    if not isinstance(baseline, NoneType):
        e.apply_baseline(baseline)
    return e
