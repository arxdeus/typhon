# Ported from mne 1.13.2 (BSD-3-Clause)
#
# make_metadata: per-trial event latencies. Typhon has no DataFrame in mne, so the
# metadata is returned as a small column store (Metadata) with float columns
# (latencies, NaN when absent) and string columns (event_name, first_*, last_*).

import numpy as np
from .event import match_event_names


class Metadata:
    columns: List[str]
    nums: Dict[str, List[float]]
    strs: Dict[str, List[Optional[str]]]
    n: int

    def __init__(self, columns: List[str], nums: Dict[str, List[float]],
                 strs: Dict[str, List[Optional[str]]], n: int):
        self.columns = columns
        self.nums = nums
        self.strs = strs
        self.n = n

    def __len__(self) -> int:
        return self.n

    def float_col(self, name: str) -> np.ndarray[float, 1]:
        v = self.nums[name]
        out = np.empty(len(v))
        for i in range(len(v)):
            out[i] = v[i]
        return out

    def str_col(self, name: str) -> List[Optional[str]]:
        return self.strs[name]

    def __getitem__(self, name: str):
        return self.float_col(name)


def _as_list(x) -> List[str]:
    out = List[str]()
    if isinstance(x, NoneType):
        return out
    elif isinstance(x, str):
        out.append(x)
    else:
        for v in x:
            out.append(str(v))
    return out


def make_metadata(events, event_id, tmin, tmax, sfreq: float, row_events = None,
                  keep_first = None, keep_last = None):
    """Generate Epochs metadata from events (mne 1.13.2). Returns (metadata, events, event_id)."""
    ev = np.asarray(events)
    ids = Dict[str, int]()
    names_order = List[str]()
    for k in event_id:
        ids[str(k)] = int(event_id[k])
        names_order.append(str(k))
    if len(names_order) == 0:
        raise ValueError("event_id dictionary must contain at least one entry")
    rows = _as_list(row_events)
    kfirst = _as_list(keep_first)
    klast = _as_list(keep_last)
    # tmin / tmax modes: 0 numeric, 1 list of names, 2 None
    tmin_mode = 2
    tmax_mode = 2
    tmin_names = List[str]()
    tmax_names = List[str]()
    start_sample = 0
    stop_sample = 0
    if isinstance(tmin, NoneType):
        tmin_mode = 2
    elif isinstance(tmin, str):
        tmin_mode = 1
        tmin_names = _as_list(tmin)
    elif isinstance(tmin, float) or isinstance(tmin, int):
        tmin_mode = 0
        start_sample = int(round(float(tmin) * sfreq))
    else:
        tmin_mode = 1
        tmin_names = _as_list(tmin)
    if isinstance(tmax, NoneType):
        tmax_mode = 2
    elif isinstance(tmax, str):
        tmax_mode = 1
        tmax_names = _as_list(tmax)
    elif isinstance(tmax, float) or isinstance(tmax, int):
        tmax_mode = 0
        stop_sample = int(round(float(tmax) * sfreq)) + 1
    else:
        tmax_mode = 1
        tmax_names = _as_list(tmax)
    both = List[str]()
    for k in kfirst:
        if k in klast:
            both.append(k)
    if len(both) > 0:
        raise ValueError("The event names in keep_first and keep_last must be mutually exclusive. Specified in both: " + ", ".join(sorted(both)))
    for pn in range(2):
        vals = kfirst if pn == 0 else klast
        for nm in vals:
            try:
                match_event_names(names_order, [nm])
            except KeyError:
                raise ValueError(f'Event "{nm}", specified in {"keep_first" if pn == 0 else "keep_last"}, cannot be found in event_id dictionary')
    for pair in ((rows, "row_events"), (tmin_names, "tmin"), (tmax_names, "tmax")):
        miss = List[str]()
        for nm in pair[0]:
            if nm not in ids:
                miss.append(nm)
        if len(miss) > 0:
            raise ValueError(f"Present in {pair[1]}, but missing from event_id: " + ", ".join(sorted(miss)))
    id_to_name = Dict[int, str]()
    for nm in names_order:
        id_to_name[ids[nm]] = nm
    # keep only events present in event_id
    sel = List[int]()
    for i in range(ev.shape[0]):
        if int(ev[i, 2]) in id_to_name:
            sel.append(i)
    ne = len(sel)
    sample = [int(ev[sel[i], 0]) for i in range(ne)]
    eid = [int(ev[sel[i], 2]) for i in range(ne)]
    keep_first_cols = [c for c in kfirst if c not in ids]
    keep_last_cols = [c for c in klast if c not in ids]
    first_cols = ["first_" + c for c in keep_first_cols]
    last_cols = ["last_" + c for c in keep_last_cols]
    columns = ["event_name"] + names_order + keep_first_cols + keep_last_cols + first_cols + last_cols
    nums = Dict[str, List[float]]()
    strs = Dict[str, List[Optional[str]]]()
    nan = float("nan")
    for c in names_order + keep_first_cols + keep_last_cols:
        nums[c] = [nan for _ in range(ne)]
    strs["event_name"] = [Optional[str]("") for _ in range(ne)]
    for c in first_cols + last_cols:
        strs[c] = [Optional[str](None) for _ in range(ne)]
    group_names = List[str]()
    for g in kfirst + klast:
        group_names.append(g)
    for r in range(ne):
        rs = sample[r]
        strs["event_name"][r] = id_to_name[eid[r]]
        w0 = 0
        if tmin_mode == 1:
            ids_t = [ids[nm] for nm in tmin_names]
            w0 = rs
            for q in range(ne):
                if sample[q] <= rs and eid[q] in ids_t:
                    w0 = sample[q]
        elif tmin_mode == 2:
            w0 = rs
        else:
            w0 = rs + start_sample
        w1 = 0
        if tmax_mode == 1:
            ids_t = [ids[nm] for nm in tmax_names]
            w1 = sample[ne - 1]
            for q in range(ne):
                if sample[q] >= rs and eid[q] in ids_t:
                    w1 = sample[q]
                    break
        elif tmax_mode == 2:
            has_next = False
            same_next = -1
            last_s = rs
            for q in range(ne):
                if sample[q] > rs:
                    has_next = True
                    last_s = sample[q]
                    if same_next < 0 and eid[q] == eid[r]:
                        same_next = sample[q]
            if not has_next:
                w1 = rs
            elif same_next >= 0:
                w1 = same_next - 1
            else:
                w1 = last_s
        else:
            w1 = rs + stop_sample
        for q in range(ne):
            if sample[q] < w0 or sample[q] > w1:
                continue
            esamp = sample[q] - rs
            etime = float(esamp) / sfreq
            if abs(etime) <= 1e-8:
                etime = 0.0
            ename = id_to_name[eid[q]]
            cur = nums[ename][r]
            if cur == cur:
                if ename not in klast:
                    continue
            nums[ename][r] = etime
            for grp in group_names:
                matches = match_event_names(names_order, [grp])
                if ename not in matches:
                    continue
                in_first = grp in kfirst
                col = ("first_" if in_first else "last_") + grp
                old = nums[grp][r]
                if old == old:
                    if (in_first and old <= etime) or ((not in_first) and old >= etime):
                        continue
                if grp not in ids:
                    nm2 = ename.replace(grp, "").replace("//", "/").strip("/")
                    strs[col][r] = nm2
                nums[grp][r] = etime
    # events kept (event_id membership) first
    ev2 = np.empty((ne, 3), dtype = int)
    for i in range(ne):
        for j in range(3):
            ev2[i, j] = ev[sel[i], j]
    out_ids = Dict[str, int]()
    for nm in names_order:
        out_ids[nm] = ids[nm]
    keep_rows = [i for i in range(ne)]
    if len(rows) > 0:
        out_ids = Dict[str, int]()
        for nm in names_order:
            if nm in rows:
                out_ids[nm] = ids[nm]
        kr = List[int]()
        for i in range(ne):
            if id_to_name[eid[i]] in rows:
                kr.append(i)
        keep_rows = kr
    nk = len(keep_rows)
    ev3 = np.empty((nk, 3), dtype = int)
    for a in range(nk):
        for j in range(3):
            ev3[a, j] = ev2[keep_rows[a], j]
    nums2 = Dict[str, List[float]]()
    for c in nums:
        nums2[c] = [nums[c][i] for i in keep_rows]
    strs2 = Dict[str, List[Optional[str]]]()
    for c in strs:
        strs2[c] = [strs[c][i] for i in keep_rows]
    return Metadata(columns, nums2, strs2, nk), ev3, out_ids
