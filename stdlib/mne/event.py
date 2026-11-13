# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# mne/event.py: find_events, find_stim_steps, make_fixed_length_events,
# merge_events, pick_events, read_events/write_events, concatenate_events,
# shift_time_events.

import numpy as np
from .io.fiff import _read_events_fif, _write_events_fif


def _rows(ev: List[List[int]]) -> np.ndarray[int, 2]:
    out = np.zeros((len(ev), 3), dtype=int)
    for i in range(len(ev)):
        for j in range(3):
            out[i, j] = ev[i][j]
    return out


def _to_rows(events) -> List[List[int]]:
    e = np.asarray(events)
    out = List[List[int]]()
    for i in range(e.shape[0]):
        out.append([int(e[i, 0]), int(e[i, 1]), int(e[i, 2])])
    return out


def _find_stim_steps(d: List[int], first_samp: int, pad_start: Optional[int],
                     pad_stop: Optional[int], merge: int) -> List[List[int]]:
    steps = List[List[int]]()
    for i in range(len(d) - 1):
        if d[i + 1] != d[i]:
            steps.append([i + 1 + first_samp, d[i], d[i + 1]])
    if len(steps) == 0:
        return steps
    if pad_start is not None:
        v = steps[0][1]
        if v != pad_start.__val__():
            steps.insert(0, [0, pad_start.__val__(), v])
    if pad_stop is not None:
        v = steps[-1][2]
        if v != pad_stop.__val__():
            steps.append([len(d) + first_samp, v, pad_stop.__val__()])
    if merge != 0:
        n = len(steps)
        idx = [steps[i + 1][0] - steps[i][0] <= abs(merge) for i in range(n - 1)]
        if any(idx):
            keep = List[bool]()
            if merge > 0:
                for i in range(n - 1):
                    if idx[i]:
                        steps[i + 1][1] = steps[i][1]
                keep = [not x for x in idx] + [True]
            else:
                for i in range(n - 1):
                    if idx[i]:
                        steps[i][2] = steps[i + 1][2]
                keep = [True] + [not x for x in idx]
            steps = [steps[i] for i in range(n) if keep[i] and steps[i][1] != steps[i][2]]
    return steps


def _stim_channels(raw, stim_channel) -> List[str]:
    if isinstance(stim_channel, str):
        return [stim_channel]
    elif isinstance(stim_channel, NoneType):
        names = raw.info.ch_names
        if "STI101" in names:
            return ["STI101"]
        if "STI 014" in names:
            return ["STI 014"]
        out = [names[i] for i in range(len(names)) if raw.info.ch_types[i] == "stim"]
        if len(out) == 0:
            if len(raw.annotations) > 0:
                raise ValueError("No stim channels found, but the raw object has annotations. Consider using mne.events_from_annotations to convert these to events.")
            raise ValueError("No stim channels found. Consider specifying them manually using the 'stim_channel' parameter.")
        return out
    else:
        return [str(s) for s in stim_channel]


def _mask_trigs(ev: List[List[int]], mask: Optional[int], mask_type: str) -> List[List[int]]:
    if mask_type != "and" and mask_type != "not_and":
        raise ValueError(f"Invalid value for the 'mask_type' parameter. Allowed values are 'not_and' and 'and', but got '{mask_type}' instead.")
    if mask is not None:
        m = mask.__val__()
        if mask_type == "not_and":
            m = ~m
        for e in ev:
            e[1] = e[1] & m
            e[2] = e[2] & m
    return [e for e in ev if e[1] != e[2]]


def _find_events(d: List[int], first_samp: int, output: str, consecutive, min_samples: float,
                 mask: Optional[int], uint_cast: bool, mask_type: str, initial_event: bool,
                 ch_name: str) -> List[List[int]]:
    merge = 0
    if min_samples > 0:
        merge = int(min_samples // 1)
        if merge == min_samples:
            merge -= 1
    if uint_cast:
        d = [x & 0xFFFF for x in d]
    if min(d) < 0:
        d = [abs(x) for x in d]
    events = _find_stim_steps(d, first_samp, None, 0, merge)
    iv = d[0]
    if iv != 0:
        if initial_event:
            events.insert(0, [first_samp, 0, iv])
        else:
            print(f"Trigger channel {ch_name} has a non-zero initial value of {iv} (consider using initial_event=True to detect this event)")
    events = _mask_trigs(events, mask, mask_type)
    n = len(events)
    onsets = List[bool]()
    offsets = List[bool]()
    inc = isinstance(consecutive, str)
    cons = False
    if not inc:
        cons = bool(consecutive)
    for e in events:
        if inc:
            on = e[2] > e[1]
            onsets.append(on)
            offsets.append((on or e[2] == 0) and e[1] > 0)
        elif cons:
            onsets.append(e[2] > 0)
            offsets.append(e[1] > 0)
        else:
            onsets.append(e[1] == 0)
            offsets.append(e[2] == 0)
    on_idx = [i for i in range(n) if onsets[i]]
    off_idx = [i for i in range(n) if offsets[i]]
    if len(on_idx) == 0 or len(off_idx) == 0:
        return List[List[int]]()
    if on_idx[0] > off_idx[0]:
        off_idx = off_idx[1:]
    if on_idx[-1] > off_idx[-1]:
        on_idx = on_idx[:-1]
    if output == "onset":
        return [events[i] for i in on_idx]
    elif output == "step":
        u = sorted(set(on_idx + off_idx))
        return [events[i] for i in u]
    elif output == "offset":
        out = List[List[int]]()
        for k in range(len(off_idx)):
            e = list(events[off_idx[k]])
            eid = events[on_idx[k]][2]
            e[1] = e[2]
            e[2] = eid
            e[0] -= 1
            out.append(e)
        return out
    raise ValueError(f"Invalid value for the 'output' parameter. Allowed values are 'onset', 'step', and 'offset', but got '{output}' instead.")


def find_events(raw, stim_channel = None, output: str = "onset", consecutive = "increasing",
                min_duration: float = 0, shortest_event: int = 2, mask: Optional[int] = None,
                uint_cast: bool = False, mask_type: str = "and", initial_event: bool = False,
                verbose = None):
    """Find events from a stim channel (mne.find_events)."""
    min_samples = min_duration * raw.info.sfreq
    chans = _stim_channels(raw, stim_channel)
    allev = List[List[int]]()
    for ch in chans:
        if ch not in raw.info.ch_names:
            raise ValueError("No stim channel found to extract event triggers.")
        ci = raw.info.ch_names.index(ch)
        row = raw._data[ci]
        d = [int(row[k]) for k in range(row.shape[0])]
        ev = _find_events(d, raw.first_samp, output, consecutive, min_samples, mask,
                          uint_cast, mask_type, initial_event, ch)
        n_short = 0
        for k in range(len(ev) - 1):
            if ev[k + 1][0] - ev[k][0] < shortest_event:
                n_short += 1
        if n_short > 0:
            raise ValueError(f"You have {n_short} events shorter than the shortest_event. These are very unusual and you may want to set min_duration to a larger value e.g. x / raw.info['sfreq']. Where x = 1 sample shorter than the shortest event length.")
        allev.extend(ev)
    # _find_unique_events + stable time sort
    seen = Set[Tuple[int, int, int]]()
    uniq = List[List[int]]()
    for e in allev:
        k = (e[0], e[1], e[2])
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    uniq.sort(key=lambda e: (e[0], e[1], e[2]))
    return _rows(uniq)


def find_stim_steps(raw, pad_start: Optional[int] = None, pad_stop: Optional[int] = None,
                    merge: int = 0, stim_channel = None):
    chans = _stim_channels(raw, stim_channel)
    if len(chans) != 1:
        raise NotImplementedError("Typhon mne: find_stim_steps supports one stim channel")
    row = raw._data[raw.info.ch_names.index(chans[0])]
    d = [int(row[k]) for k in range(row.shape[0])]
    st = _find_stim_steps(d, raw.first_samp, pad_start, pad_stop, merge)
    if len(st) == 0:
        return np.empty((0, 3), dtype=int)
    return _rows(st)


def make_fixed_length_events(raw, id: int = 1, start: float = 0, stop: Optional[float] = None,
                             duration: float = 1.0, first_samp: bool = True,
                             overlap: float = 0.0):
    duration = float(duration)
    overlap = float(overlap)
    if not (0 <= overlap < duration):
        raise ValueError(f"overlap must be >=0 but < duration ({duration}), got {overlap}")
    sf = raw.info.sfreq
    s = int(raw.time_as_index(float(start), use_rounding=True)[0])
    e = 0
    if stop is not None:
        e = int(raw.time_as_index(stop.__val__(), use_rounding=True)[0])
    else:
        e = raw.last_samp + 1
    if first_samp:
        s = s + raw.first_samp
        e = min(e + raw.first_samp, raw.last_samp + 1)
    else:
        e = min(e, raw.n_times)
    e -= int(np.round(sf * duration))
    ts = np.arange(float(s), float(e + 1), sf * (duration - overlap)).astype(int)
    n = ts.shape[0]
    if n == 0:
        raise ValueError("No events produced, check the values of start, stop, and duration")
    out = np.zeros((n, 3), dtype=int)
    for i in range(n):
        out[i, 0] = ts[i]
        out[i, 2] = id
    return out


def merge_events(events, ids, new_id: int, replace_events: bool = True):
    ev = _to_rows(events)
    out = [list(e) for e in ev]
    idl = [int(i) for i in ids]
    touched = Set[int]()
    for col in (1, 2):
        for i in idl:
            for r in range(len(ev)):
                if ev[r][col] == i:
                    out[r][col] = new_id
                    touched.add(r)
    if not replace_events:
        for r in sorted(touched):
            out.append(list(ev[r]))
        out.sort(key=lambda e: (e[0], e[1], e[2]))
    return _rows(out)


def _int_list(x) -> List[int]:
    if isinstance(x, int):
        return [x]
    return [int(v) for v in x]


def pick_events(events, include = None, exclude = None, step: bool = False):
    ev = _to_rows(events)
    out = List[List[int]]()
    if not isinstance(include, NoneType):
        inc = _int_list(include)
        for e in ev:
            if e[2] in inc or (step and e[1] in inc):
                out.append(e)
    elif not isinstance(exclude, NoneType):
        exc = _int_list(exclude)
        for e in ev:
            if not (e[2] in exc or (step and e[1] in exc)):
                out.append(e)
    else:
        out = ev
    if len(out) == 0:
        raise RuntimeError("No events found")
    return _rows(out)


def shift_time_events(events, ids, tshift: float, sfreq: float):
    ev = np.asarray(events).copy()
    sh = int(tshift * sfreq)
    for i in range(ev.shape[0]):
        if isinstance(ids, NoneType):
            ev[i, 0] += sh
        elif int(ev[i, 2]) in _int_list(ids):
            ev[i, 0] += sh
    return ev


def concatenate_events(events, first_samps, last_samps):
    if not (len(events) == len(last_samps) and len(events) == len(first_samps)):
        raise ValueError("events, first_samps, and last_samps must all have the same lengths")
    fs = [int(x) for x in first_samps]
    ls = [int(x) for x in last_samps]
    out = _to_rows(events[0])
    n = 0
    for k in range(1, len(events)):
        n += ls[k - 1] - fs[k - 1] + 1
        for e in _to_rows(events[k]):
            e[0] = e[0] - fs[k] + n + fs[0]
            out.append(e)
    return _rows(out)


def read_events(filename, include = None, exclude = None, mask: Optional[int] = None,
                mask_type: str = "and", return_event_id: Literal[bool] = False,
                verbose = None):
    fn = str(filename)
    ev = np.zeros((0, 3), dtype=int)
    ids = Dict[str, int]()
    has_id = False
    if fn.endswith(".fif") or fn.endswith(".gz"):
        ev, ids, has_id = _read_events_fif(fn)
    else:
        rows = List[List[int]]()
        with open(fn) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 0:
                    continue
                vals = [int(float(p)) for p in parts]
                if len(vals) == 4:
                    rows.append([vals[0], vals[2], vals[3]])
                elif len(vals) == 3:
                    rows.append(vals)
                else:
                    raise ValueError("Unknown number of columns in event text file")
        if len(rows) == 0:
            raise ValueError("No text lines found")
        if mask is not None and rows[0][2] == 0:
            rows = rows[1:]
        ev = _rows(rows)
    ev = pick_events(ev, include, exclude)
    if mask is not None:
        ev = _rows(_mask_trigs(_to_rows(ev), mask, mask_type))
    if return_event_id:
        if not has_id:
            raise RuntimeError("No event_id found in the file")
        return ev, ids
    else:
        return ev


def write_events(filename, events, *, overwrite: bool = False, verbose = None):
    fn = str(filename)
    ev = np.asarray(events).astype(int)
    if fn.endswith(".fif") or fn.endswith(".gz"):
        _write_events_fif(fn, ev)
    else:
        with open(fn, "w") as f:
            for i in range(ev.shape[0]):
                f.write(f"{int(ev[i, 0]):6d} {int(ev[i, 1]):6d} {int(ev[i, 2]):3d}\n")


def count_events(events, ids = None) -> Dict[int, int]:
    """mne.count_events."""
    ev = np.asarray(events).astype(int)
    counts = Dict[int, int]()
    for i in range(ev.shape[0]):
        c = int(ev[i, 2])
        counts[c] = counts.get(c, 0) + 1
    out = Dict[int, int]()
    if isinstance(ids, NoneType):
        for k in sorted(list(counts.keys())):
            out[k] = counts[k]
    else:
        for i in ids:
            out[int(i)] = counts.get(int(i), 0)
    return out


def match_event_names(event_names, keys, *, on_missing: str = "raise") -> List[str]:
    """mne.match_event_names (event_names: list of str or event_id dict)."""
    names = List[str]()
    if isinstance(event_names, Dict):
        for k in event_names.keys():
            names.append(k)
    else:
        names = [str(x) for x in event_names]
    ks = List[str]()
    if isinstance(keys, str):
        ks.append(keys)
    else:
        ks = [str(x) for x in keys]
    matches = Set[str]()
    for key in ks:
        parts = key.split("/")
        for name in names:
            nparts = name.split("/")
            if all([p in nparts for p in parts]):
                matches.add(name)
    if len(matches) == 0:
        msg = f'Event name "{ks[-1]}" could not be found. The following events are present in the data: {", ".join(names)}'
        if on_missing == "raise":
            raise KeyError(msg)
        elif on_missing == "warn":
            print("RuntimeWarning:", msg)
    return sorted(list(matches))


def define_target_events(events, reference_id: int, target_id: int, sfreq: float, tmin: float,
                         tmax: float, new_id: Optional[int] = None, fill_na: Optional[int] = None):
    """mne.event.define_target_events. Empty results are returned as (0, 3) / (0,) arrays."""
    ev = np.asarray(events).astype(int)
    nid = reference_id if new_id is None else new_id.__val__()
    tsample = 1e3 / sfreq
    imin = int(tmin * sfreq)
    imax = int(tmax * sfreq)
    rows = List[List[int]]()
    lag = List[float]()
    for r in range(ev.shape[0]):
        if int(ev[r, 2]) != reference_id:
            continue
        lower = int(ev[r, 0]) + imin
        upper = int(ev[r, 0]) + imax
        first = -1
        for q in range(ev.shape[0]):
            if int(ev[q, 0]) > lower and int(ev[q, 0]) < upper and int(ev[q, 2]) == target_id:
                first = q
                break
        if first >= 0:
            lag.append(abs(float(int(ev[r, 0]) - int(ev[first, 0]))))
            rows.append([int(ev[r, 0]), int(ev[r, 1]), nid])
        elif fill_na is not None:
            rows.append([int(ev[r, 0]), int(ev[r, 1]), fill_na.__val__()])
            lag.append(float("nan"))
    out = np.zeros((len(rows), 3), dtype=int)
    for i in range(len(rows)):
        for j in range(3):
            out[i, j] = rows[i][j]
    anyl = False
    for v in lag:
        if np.isnan(v) or v != 0.0:
            anyl = True
    la = np.array(lag)
    if anyl:
        la = la * tsample
    else:
        la = np.zeros(0)
    return out, la
