# Ported from braindecode 2.0.0 (BSD-3-Clause)
#
# create_windows_from_events and create_fixed_length_windows (EEGWindowsDataset
# path, i.e. use_mne_epochs=False). Per-event-type dict offsets are supported
# for trial_start_offset_samples / trial_stop_offset_samples /
# window_stride_samples.

import numpy as np
import pandas as pd
import mne
from ..datasets.base import BaseConcatDataset, RawDataset, EEGWindowsDataset


def _resolve_on_last_window(drop_last_window: Optional[bool], on_last_window: Optional[str],
                            default: Optional[str]) -> Optional[str]:
    if drop_last_window is not None and on_last_window is not None:
        raise ValueError("Cannot specify both `drop_last_window` and `on_last_window`. "
                         "Use `on_last_window` only, as `drop_last_window` is deprecated.")
    if on_last_window is not None and on_last_window.__val__() not in ("overlap", "drop"):
        raise ValueError(f"on_last_window must be one of 'overlap' or 'drop', got {repr(on_last_window)}.")
    if drop_last_window is not None:
        return "drop" if drop_last_window else "overlap"
    return default if on_last_window is None else on_last_window


def _compute_window_inds(starts: List[int], stops: List[int], start_offset: int, stop_offset: int,
                         size: int, stride: int, drop_last_window: bool, accepted_bads_ratio: float):
    st = [s + start_offset for s in starts]
    sp = [s + stop_offset for s in stops]
    bads = [size > (sp[i] - st[i]) for i in range(len(st))]
    nb = sum([1 for b in bads if b])
    if nb > 0:
        if nb <= accepted_bads_ratio * len(st):
            st = [st[i] for i in range(len(st)) if not bads[i]]
            sp = [sp[i] for i in range(len(sp)) if not bads[i]]
        else:
            md = min([sp[i] - st[i] for i in range(len(st))])
            r = nb / len(bads)
            raise ValueError(f"Window size {size} exceeds trial duration ({md}) for too many trials "
                             f"({r * 100}%). Set accepted_bads_ratio to at least {r}"
                             "and restart training to be able to continue.")
    i_win = List[int]()
    i_tr = List[int]()
    w_st = List[int]()
    for ti in range(len(st)):
        start = st[ti]
        stop = sp[ti]
        k = 0
        s = start
        while s < stop:
            if s + size <= stop:
                w_st.append(s)
                i_win.append(k)
                i_tr.append(ti)
            s += stride
            k += 1
        if not drop_last_window:
            if w_st[-1] + size != stop:
                w_st.append(stop - size)
                i_win.append(i_win[-1] + 1)
                i_tr.append(ti)
    w_sp = [s + size for s in w_st]
    return i_tr, i_win, w_st, w_sp


def _per_type(v, name: str, default: int) -> int:
    if isinstance(v, Dict):
        return v[name]
    elif isinstance(v, NoneType):
        return default
    else:
        return int(v)


def _windows_df(i_win: List[int], i_trial: Optional[List[int]], starts: List[int],
                stops: List[int], targets: List[int]) -> pd.DataFrame:
    d = Dict[str, List[int]]()
    d["i_window_in_trial"] = i_win
    if i_trial is not None:
        d["i_trial_in_dataset"] = i_trial.__val__()
    d["i_start_in_trial"] = starts
    d["i_stop_in_trial"] = stops
    d["target"] = targets
    return pd.DataFrame(d)


def _create_windows_from_events(ds: RawDataset, infer_window_size_stride: bool,
                                trial_start_offset_samples, trial_stop_offset_samples,
                                window_size_samples: Optional[int], window_stride_samples,
                                drop_last_window: bool, mapping: Dict[str, int],
                                accepted_bads_ratio: float, on_overlapping_events: str) -> EEGWindowsDataset:
    raw = ds.raw
    event_mapping = mapping
    vals = Set[int]()
    for v in mapping.values():
        vals.add(v)
    if len(vals) < len(mapping):
        event_mapping = Dict[str, int]()
        for name in mapping.keys():
            event_mapping[name] = len(event_mapping)
    events, events_id = mne.events_from_annotations(raw, event_mapping)
    targets = Dict[int, int]()
    code_to_name = Dict[int, str]()
    for name, code in events_id.items():
        targets[code] = mapping[name]
        code_to_name[code] = name
    ann = raw.annotations
    rows = [i for i in range(len(ann)) if ann.description[i] in events_id]
    sf = raw.info.sfreq
    n_ev = events.shape[0]
    onsets = [int(events[i, 0]) for i in range(n_ev)]
    codes = [int(events[i, 2]) for i in range(n_ev)]
    stops = [onsets[i] + int(ann.duration[rows[i]] * sf) for i in range(n_ev)]
    last_samp = raw.first_samp + raw.n_times - 1
    for i in range(n_ev):
        so = _per_type(trial_stop_offset_samples, code_to_name[codes[i]], 0)
        if stops[i] + so > last_samp + 1:
            trial = "last trial" if i == n_ev - 1 else f"trial {i}"
            raise ValueError(f'"trial_stop_offset_samples" too large. Stop of {trial} ({stops[i]}) + '
                             f'"trial_stop_offset_samples" ({so}) must be smaller than length of '
                             f"recording ({len(ds)}).")
    i_trials = List[int]()
    i_wins = List[int]()
    starts = List[int]()
    wstops = List[int]()
    size = 0
    is_dict = (isinstance(trial_start_offset_samples, Dict) or isinstance(trial_stop_offset_samples, Dict)
               or isinstance(window_stride_samples, Dict))
    onsets = [o - raw.first_samp for o in onsets]
    stops = [s - raw.first_samp for s in stops]
    if is_dict:
        size = window_size_samples.__val__()
        all_tr = List[int]()
        all_w = List[int]()
        all_s = List[int]()
        all_e = List[int]()
        for name, code in events_id.items():
            idx = [i for i in range(n_ev) if codes[i] == code]
            if len(idx) == 0:
                continue
            so = _per_type(trial_start_offset_samples, name, 0)
            eo = _per_type(trial_stop_offset_samples, name, 0)
            stride = _per_type(window_stride_samples, name, 0)
            t_on = [onsets[i] for i in idx]
            t_st = [stops[i] for i in idx]
            good = [idx[k] for k in range(len(idx)) if not (size > (t_st[k] + eo) - (t_on[k] + so))]
            tr, w, s, e = _compute_window_inds(t_on, t_st, so, eo, size, stride, drop_last_window,
                                               accepted_bads_ratio)
            all_tr.extend([good[i] for i in tr])
            all_w.extend(w)
            all_s.extend(s)
            all_e.extend(e)
        order = np.argsort(np.array(all_s), kind="stable")
        for k in range(len(all_s)):
            j = int(order[k])
            i_trials.append(all_tr[j])
            i_wins.append(all_w[j])
            starts.append(all_s[j])
            wstops.append(all_e[j])
    else:
        so = _per_type(trial_start_offset_samples, "", 0)
        eo = _per_type(trial_stop_offset_samples, "", 0)
        stride = 0
        if infer_window_size_stride:
            size = stops[0] + eo - (onsets[0] + so)
            stride = size
            keep = [i for i in range(len(onsets)) if (stops[i] + eo) - (onsets[i] + so) == size]
            if len(keep) != len(onsets):
                onsets = [onsets[i] for i in keep]
                stops = [stops[i] for i in keep]
                codes = [codes[i] for i in keep]
                rows = [rows[i] for i in keep]
        else:
            size = window_size_samples.__val__()
            stride = _per_type(window_stride_samples, "", size)
        good = [i for i in range(len(onsets)) if size <= (stops[i] + eo) - (onsets[i] + so)]
        tr, w, s, e = _compute_window_inds(onsets, stops, so, eo, size, stride, drop_last_window,
                                           accepted_bads_ratio)
        i_trials = [good[i] for i in tr]
        i_wins = w
        starts = s
        wstops = e
    if len(starts) > 0 and min(starts) < 0:
        raise ValueError('"trial_start_offset_samples" too small. A window would start '
                         f"{-min(starts)} samples before the start of the recording.")
    if on_overlapping_events != "ignore":
        for k in range(1, len(starts)):
            if starts[k] - starts[k - 1] <= 0:
                if on_overlapping_events == "raise":
                    raise ValueError("Overlapping trials detected. You can ignore, warn, or raise an error, "
                                     "using the on_overlapping_events argument.")
                break
    tg = [targets[codes[i_trials[k]]] for k in range(len(starts))]
    src = [rows[i_trials[k]] for k in range(len(starts))]
    md = _windows_df(i_wins, src, starts, wstops, tg)
    w = EEGWindowsDataset(raw, md, ds._description)
    return w


def create_windows_from_events(concat_ds: BaseConcatDataset[RawDataset],
                               trial_start_offset_samples = 0, trial_stop_offset_samples = 0,
                               window_size_samples: Optional[int] = None,
                               window_stride_samples = None,
                               drop_last_window: Optional[bool] = None,
                               mapping: Optional[Dict[str, int]] = None, preload: bool = False,
                               drop_bad_windows: Optional[bool] = None, picks = None, reject = None,
                               flat = None, on_missing: str = "error",
                               accepted_bads_ratio: float = 0.0, use_mne_epochs: Optional[bool] = None,
                               on_overlapping_events: str = "raise", n_jobs: int = 1,
                               verbose = "error", *, on_last_window: Optional[str] = None):
    """Create windows based on events (annotations) of each recording."""
    if not isinstance(picks, NoneType) or not isinstance(reject, NoneType) or not isinstance(flat, NoneType):
        compile_error("Typhon braindecode: picks/reject/flat need mne.Epochs (use_mne_epochs) and are not supported")
    if use_mne_epochs is not None and use_mne_epochs:
        raise NotImplementedError("use_mne_epochs=True is not supported in Typhon braindecode")
    olw = _resolve_on_last_window(drop_last_window, on_last_window, "overlap")
    drop = olw.__val__() == "drop"
    if on_overlapping_events not in ("raise", "warn", "ignore"):
        raise ValueError(f"Invalid value {on_overlapping_events} for on_overlapping_events.")
    if window_size_samples is not None:
        if window_size_samples.__val__() <= 0:
            raise AssertionError("window size has to be larger than 0")
    has_dict = (isinstance(trial_start_offset_samples, Dict) or isinstance(trial_stop_offset_samples, Dict)
                or isinstance(window_stride_samples, Dict))
    if has_dict and mapping is None:
        raise ValueError("mapping must be provided when any of trial_start_offset_samples, "
                         "trial_stop_offset_samples, or window_stride_samples is a dict.")
    if has_dict and window_size_samples is None:
        raise ValueError("window_size_samples must be provided (not None) when any of "
                         "trial_start_offset_samples, trial_stop_offset_samples, "
                         "or window_stride_samples is a dict.")
    mp = Dict[str, int]()
    if mapping is None:
        for ds in concat_ds.datasets:
            for name in sorted(list(Set[str](ds.raw.annotations.description))):
                if name not in mp:
                    mp[name] = len(mp)
    else:
        for k, v in mapping.items():
            mp[k] = v
    infer = window_size_samples is None
    out = List[EEGWindowsDataset]()
    for ds in concat_ds.datasets:
        out.append(_create_windows_from_events(ds, infer, trial_start_offset_samples,
                                               trial_stop_offset_samples, window_size_samples,
                                               window_stride_samples, drop, mp,
                                               accepted_bads_ratio, on_overlapping_events))
    return BaseConcatDataset(out)


def _create_fixed_length_windows(ds: RawDataset, start_offset_samples: int,
                                 stop_offset_samples: Optional[int], window_size_samples: Optional[int],
                                 window_stride_samples: Optional[int], drop_last_window: Optional[bool],
                                 mapping: Optional[Dict[str, int]]) -> EEGWindowsDataset:
    stop = ds.raw.n_times if stop_offset_samples is None else stop_offset_samples.__val__()
    size = stop - start_offset_samples if window_size_samples is None else window_size_samples.__val__()
    stride = size if window_stride_samples is None else window_stride_samples.__val__()
    last = stop - size
    target = -1
    if ds.target_name is not None:
        t = ds._description[ds.target_name.__val__()]
        if mapping is not None and str(t) in mapping.__val__():
            target = mapping.__val__()[str(t)]
        else:
            target = int(t)
    starts = List[int]()
    s = start_offset_samples
    while s < last + 1:
        starts.append(s)
        s += stride
    dlw = drop_last_window if drop_last_window is not None else False
    if not dlw and starts[-1] < last:
        starts.append(last)
    md = _windows_df(list(range(len(starts))), None, starts, [s + size for s in starts],
                     [target for _ in range(len(starts))])
    return EEGWindowsDataset(ds.raw, md, ds._description)


def create_fixed_length_windows(concat_ds: BaseConcatDataset[RawDataset], start_offset_samples: int = 0,
                                stop_offset_samples: Optional[int] = None,
                                window_size_samples: Optional[int] = None,
                                window_stride_samples: Optional[int] = None,
                                drop_last_window: Optional[bool] = None,
                                mapping: Optional[Dict[str, int]] = None, preload: bool = False,
                                drop_bad_windows: Optional[bool] = None, picks = None, reject = None,
                                flat = None, targets_from: str = "metadata", last_target_only: bool = True,
                                lazy_metadata: bool = False, on_missing: str = "error",
                                use_mne_epochs: Optional[bool] = None, n_jobs: int = 1,
                                verbose = "error", *, on_last_window: Optional[str] = None):
    """Create sliding windows over each continuous recording."""
    if not isinstance(picks, NoneType) or not isinstance(reject, NoneType) or not isinstance(flat, NoneType):
        compile_error("Typhon braindecode: picks/reject/flat need mne.Epochs (use_mne_epochs) and are not supported")
    olw = _resolve_on_last_window(drop_last_window, on_last_window, None)
    stride = window_stride_samples
    if window_size_samples is not None and stride is None:
        stride = window_size_samples
        if olw is None:
            olw = "drop"
    if (window_size_samples is None) != (stride is None):
        raise AssertionError("window_size_samples and window_stride_samples must both be None or both be set")
    if window_size_samples is not None:
        if window_size_samples.__val__() <= 0:
            raise AssertionError("window size has to be larger than 0")
        if stride.__val__() <= 0:
            raise AssertionError("window stride has to be larger than 0")
    sos = stop_offset_samples
    if sos is not None and sos.__val__() == 0:
        sos = None
    if window_size_samples is not None and olw is None:
        raise ValueError("on_last_window must be set if both window_size_samples & "
                         "window_stride_samples have also been set. Use 'drop' or 'overlap'.")
    if window_size_samples is None:
        olw = None
    if olw is not None and olw.__val__() != "drop" and lazy_metadata:
        raise ValueError("Cannot use on_last_window other than 'drop' with lazy_metadata=True.")
    dlw: Optional[bool] = None
    if olw is not None:
        dlw = olw.__val__() == "drop"
    if window_size_samples is not None:
        for ds in concat_ds.datasets:
            if window_size_samples.__val__() > ds.raw.n_times:
                mn = min([d.raw.n_times for d in concat_ds.datasets])
                raise ValueError(f"Window size {window_size_samples.__val__()} exceeds trial duration {mn}.")
    out = List[EEGWindowsDataset]()
    for ds in concat_ds.datasets:
        out.append(_create_fixed_length_windows(ds, start_offset_samples, sos, window_size_samples,
                                                stride, dlw, mapping))
    return BaseConcatDataset(out)
