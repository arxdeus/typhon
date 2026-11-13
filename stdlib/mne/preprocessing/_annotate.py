# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# annotate_amplitude and annotate_break (orig_time=None recordings).

import numpy as np
from .._info import Info, _picks_to_idx, _picks_by_type
from ..annotations import Annotations


def _mask_to_onsets_offsets(mask: List[bool]) -> Tuple[List[int], List[int]]:
    on = List[int]()
    off = List[int]()
    prev = False
    for i in range(len(mask)):
        if mask[i] and not prev:
            on.append(i)
        elif (not mask[i]) and prev:
            off.append(i)
        prev = mask[i]
    if prev:
        off.append(len(mask))
    return on, off


def _thresholds(v, name: str, types: List[str]) -> Optional[Dict[str, float]]:
    if isinstance(v, NoneType):
        return None
    out = Dict[str, float]()
    if isinstance(v, dict):
        for k, x in v.items():
            if float(x) < 0:
                raise ValueError(f"Argument '{name}' should define positive thresholds. Provided for channel type '{k}': '{x}'.")
            out[k] = float(x)
    else:
        if float(v) < 0:
            raise ValueError(f"Argument '{name}' should define a positive threshold. Provided: '{v}'.")
        for t in types:
            out[t] = float(v)
    return out


def _starts_stops_skip(raw) -> Tuple[List[int], List[int]]:
    n = raw._data.shape[1]
    ft = raw.first_time
    sf = raw.info.sfreq
    mask = [True for _ in range(n)]
    ann = raw.annotations
    extra_on = List[int]()
    for a in range(len(ann)):
        if ann.description[a].upper().startswith("BAD_ACQ_SKIP"):
            s = int(round((ann.onset[a] - ft) * sf))
            e = int(round((ann.onset[a] - ft + ann.duration[a]) * sf))
            for k in range(max(s, 0), min(e, n)):
                mask[k] = False
            if s == e:
                extra_on.append(s)
    on, off = _mask_to_onsets_offsets(mask)
    return on, off


def annotate_amplitude(raw, peak = None, flat = None, bad_percent = 5, min_duration = 0.005,
                       picks = None, *, verbose = None):
    """Annotate raw data based on peak-to-peak amplitude (mne 1.13.2)."""
    picks_ = _picks_to_idx(raw.info, picks, "data_or_ica", "bads")
    types = raw.info.get_channel_types(picks_)
    pk = _thresholds(peak, "peak", types)
    fl = _thresholds(flat, "flat", types)
    if pk is None and fl is None:
        raise ValueError("At least one of the arguments 'peak' or 'flat' must not be None.")
    bp = float(bad_percent)
    if bp < 0 or bp > 100:
        raise ValueError(f"Argument 'bad_percent' should define a percentage between 0% and 100%. Provided: {bp}%.")
    sf = raw.info.sfreq
    n_times = raw._data.shape[1]
    md = float(min_duration)
    if md < 0:
        raise ValueError(f"Argument 'min_duration' should define a positive duration in seconds. Provided: '{md}' seconds.")
    if md >= n_times / sf:
        raise ValueError(f"Argument 'min_duration' should define a positive duration in seconds shorter than the raw duration ({n_times / sf} seconds). Provided: '{md}' seconds.")
    md_s = int(round(md * sf))
    bads_idx = List[int]()
    onsets, ends = _starts_stops_skip(raw)
    index = List[int]()
    for k in range(len(onsets)):
        for j in range(onsets[k], ends[k]):
            index.append(j)
    n_idx = len(index)
    any_flat = [False for _ in range(n_times - 1)]
    any_peak = [False for _ in range(n_times - 1)]
    pset = Set[int](picks_)
    for item in _picks_by_type(raw.info, False, True):
        ch_type = item[0]
        sel = List[int]()
        for c in item[1]:
            if c in pset:
                sel.append(c)
        if len(sel) == 0:
            continue
        for kind in range(2):
            thr_d = fl if kind == 0 else pk
            if thr_d is None:
                continue
            thr = thr_d.__val__()[ch_type]
            for c in sel:
                m = [False for _ in range(max(n_idx - 1, 0))]
                for j in range(n_idx - 1):
                    d = abs(raw._data[c, index[j + 1]] - raw._data[c, index[j]])
                    m[j] = (d <= thr) if kind == 0 else (d >= thr)
                on, off = _mask_to_onsets_offsets(m)
                for q in range(len(on)):
                    if off[q] - on[q] < md_s:
                        for j in range(on[q], off[q]):
                            m[j] = False
                cnt = 0
                for v in m:
                    if v:
                        cnt += 1
                if cnt > 0:
                    cnt += 1
                mean = cnt / n_times * 100
                if mean >= bp:
                    bads_idx.append(c)
                elif 0 < mean and mean < bp:
                    for j in range(len(m)):
                        if m[j]:
                            if kind == 0:
                                any_flat[index[j]] = True
                            else:
                                any_peak[index[j]] = True
    ft = raw.first_time if raw.info.meas_date is not None else 0.0
    ons = List[float]()
    dur = List[float]()
    des = List[str]()
    for kind in range(2):
        arr = any_flat if kind == 0 else any_peak
        on, off = _mask_to_onsets_offsets(arr)
        for q in range(len(on)):
            ons.append(on[q] / sf + ft)
            dur.append((off[q] - on[q]) / sf)
            des.append("BAD_flat" if kind == 0 else "BAD_peak")
    bads = List[str]()
    for b in bads_idx:
        nm = raw.info.ch_names[b]
        if nm not in raw.info.bads:
            bads.append(nm)
    return Annotations(ons, dur, des, raw.info.meas_date), bads


def annotate_break(raw, events = None, min_break_duration: float = 15.0,
                   t_start_after_previous: float = 5.0, t_stop_before_next: float = 5.0,
                   ignore = ("bad", "edge"), *, verbose = None):
    """Annotations for breaks in a recording (mne 1.13.2)."""
    adur = min_break_duration - t_start_after_previous - t_stop_before_next
    if adur <= 0:
        raise ValueError(f"The result of min_break_duration - t_start_after_previous - t_stop_before_next must be greater than 0, but it is: {adur}")
    sf = raw.info.sfreq
    ign = List[str]()
    on_l = List[float]()
    du_l = List[float]()
    ds_l = List[str]()
    if isinstance(events, NoneType):
        for i in ignore:
            ign.append(str(i).lower())
        ann = raw.annotations
        for k in range(len(ann)):
            on_l.append(ann.onset[k])
            du_l.append(ann.duration[k])
            ds_l.append(ann.description[k])
    else:
        ev = np.asarray(events).astype(int)
        if ev.size == 0:
            raise ValueError("The events array must not be empty.")
        for k in range(ev.shape[0]):
            on_l.append(float(ev[k, 0]) / sf)
            du_l.append(0.0)
            ds_l.append(str(int(ev[k, 2])))
    if len(on_l) == 0:
        raise ValueError("Could not find (or generate) any annotations in your data.")
    iv = List[List[float]]()
    for k in range(len(on_l)):
        d = ds_l[k].lower()
        skip = False
        for g in ign:
            if d.startswith(g):
                skip = True
        if not skip:
            iv.append([on_l[k], on_l[k] + du_l[k]])
    if len(iv) == 0:
        raise ValueError("Could not find (or generate) any annotations in your data.")
    merged = [list(iv[0])]
    for it in iv:
        ms = merged[-1][1]
        if it[1] < ms:
            continue
        elif it[0] <= ms and it[1] >= ms:
            merged[-1][1] = it[1]
        else:
            merged.append(list(it))
    ft = raw.first_time if isinstance(events, NoneType) else 0.0
    if not isinstance(events, NoneType):
        ft = raw.first_time
    for m in merged:
        m[0] -= ft
        m[1] -= ft
    bo = List[float]()
    bd = List[float]()
    if 0 < merged[0][0] and merged[0][0] >= min_break_duration:
        bo.append(0.0)
        bd.append(merged[0][0] - t_stop_before_next)
    for k in range(1, len(merged)):
        ts = merged[k][0]
        ps = merged[k - 1][1]
        if ts - ps < min_break_duration:
            continue
        o = ps + t_start_after_previous
        bo.append(o)
        bd.append(ts - t_stop_before_next - o)
    t_last = float(raw.times[raw.times.shape[0] - 1])
    if t_last > merged[-1][1] and t_last - merged[-1][1] >= min_break_duration:
        o = merged[-1][1] + t_start_after_previous
        bo.append(o)
        bd.append(t_last - o)
    adj = raw.first_time if raw.info.meas_date is not None else 0.0
    out_on = [o + adj for o in bo]
    return Annotations(out_on, bd, ["BAD_break" for _ in bo], raw.info.meas_date)
