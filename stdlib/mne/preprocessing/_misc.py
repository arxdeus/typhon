# Ported from mne 1.13.2 (BSD-3-Clause)
#
# fix_stim_artifact, annotate_nan, equalize_bads, compute_bridged_electrodes.

import numpy as np
from .._info import Info, _picks_to_idx
from .._containers import RawArray, EpochsArray, Evoked
from ..annotations import Annotations
from ..event import find_events
from ._regress import _annotations_from_mask
from scipy.stats import gaussian_kde
from scipy.optimize import minimize_scalar


def _get_window(start: int, end: int) -> np.ndarray[float, 1]:
    n = abs(end - start)
    w = np.ones(n + 0)
    for i in range(n):
        w[i] = 1.0
    w[0] = 0.0
    w[1] = 0.75
    w[n - 2] = 0.75
    w[n - 1] = 0.0
    return 1.0 - w


def _fix_artifact(data: np.ndarray[float, 2], window: np.ndarray[float, 1], picks: List[int],
                  first: int, last: int, b0: int, b1: int, mode: str):
    for c in picks:
        if mode == "linear":
            v0 = data[c, first]
            v1 = data[c, last]
            for x in range(first, last):
                data[c, x] = v0 + (v1 - v0) * (float(x - first) / float(last - first))
        elif mode == "window":
            for x in range(first, last):
                data[c, x] = data[c, x] * window[x - first]
        else:
            s = 0.0
            for x in range(b0, b1):
                s += data[c, x]
            m = s / float(b1 - b0)
            for x in range(first, last):
                data[c, x] = m


def _bl(baseline) -> List[float]:
    if isinstance(baseline, NoneType):
        raise TypeError("baseline must be an instance of tuple or list when mode='constant', got <class 'NoneType'> instead")
    else:
        bl = [float(v) for v in baseline]
        if len(bl) != 2:
            raise ValueError(f"Invalid value for the 'len(baseline)' parameter. Allowed values are 2, but got {len(bl)} instead.")
        return bl


def fix_stim_artifact(inst, events = None, event_id = None, tmin: float = 0.0, tmax: float = 0.01,
                      *, baseline = None, mode: str = "linear", stim_channel = None, picks = None):
    """Eliminate stimulation artifacts from an instance, in place (mne 1.13.2)."""
    if mode != "linear" and mode != "window" and mode != "constant":
        raise ValueError(f"Invalid value for the 'mode' parameter. Allowed values are 'linear', 'window', and 'constant', but got '{mode}' instead.")
    sf = inst.info.sfreq
    s_start = int(np.ceil(sf * tmin))
    s_end = int(np.ceil(sf * tmax))
    b_start = 0
    b_end = 0
    if mode == "constant":
        bl = _bl(baseline)
        b_start = int(np.ceil(sf * bl[0]))
        b_end = int(np.ceil(sf * bl[1]))
    if mode == "window" and (s_end - s_start) < 4:
        raise ValueError('Time range is too short. Use a larger interval or set mode to "linear".')
    window = np.ones(1)
    if mode == "window":
        window = _get_window(s_start, s_end)
    pk = _picks_to_idx(inst.info, picks, "data", ())
    if isinstance(inst, RawArray):
        ev = np.zeros((0, 3), dtype=int)
        if isinstance(events, NoneType):
            ev = find_events(inst, stim_channel = stim_channel)
        else:
            ev = np.asarray(events).astype(int)
        if ev.shape[0] == 0:
            raise ValueError("No events are found")
        for k in range(ev.shape[0]):
            if not isinstance(event_id, NoneType):
                if ev[k, 2] != int(event_id):
                    continue
            e0 = int(ev[k, 0]) - inst.first_samp
            _fix_artifact(inst._data, window, pk, e0 + s_start, e0 + s_end, e0 + b_start, e0 + b_end, mode)
    elif isinstance(inst, EpochsArray):
        if not isinstance(inst.reject, NoneType):
            raise RuntimeError("Reject is already applied. Use reject=None in the constructor.")
        e_start = int(np.ceil(sf * float(inst.times[0])))
        for i in range(inst._data.shape[0]):
            ep = inst._data[i].copy()
            _fix_artifact(ep, window, pk, s_start - e_start, s_end - e_start, b_start - e_start, b_end - e_start, mode)
            inst._data[i] = ep
    elif isinstance(inst, Evoked):
        f0 = inst.first
        _fix_artifact(inst.data, window, pk, s_start - f0, s_end - f0, b_start - f0, b_end - f0, mode)
    else:
        compile_error("Not a Raw or Epochs or Evoked")
    return inst


def annotate_nan(raw, *, verbose = None):
    """Detect segments with NaN (mne 1.13.2)."""
    on = List[float]()
    du = List[float]()
    for c in range(raw._data.shape[0]):
        row = raw._data[c]
        m = np.isnan(row)
        a = _annotations_from_mask(raw.times, m, "BAD_NAN")
        for k in range(len(a)):
            on.append(a.onset[k])
            du.append(a.duration[k])
    return Annotations(on, du, "BAD_NAN", raw.info.meas_date)


def equalize_bads(insts, interp_thresh: float = 1.0, copy: bool = True):
    """Interpolate or mark bads consistently across instances (mne 1.13.2)."""
    if interp_thresh < 0 or interp_thresh > 1:
        raise ValueError(f"interp_thresh must be between 0 and 1, got {interp_thresh}")
    out = [i.copy() if copy else i for i in insts]
    all_bads = List[str]()
    for inst in insts:
        for b in inst.info.bads:
            if b not in all_bads:
                all_bads.append(b)
    durs = List[float]()
    tot = 0.0
    for inst in insts:
        d = float(inst.times.shape[0])
        if isinstance(inst, EpochsArray):
            d = d * float(inst._data.shape[0])
        durs.append(d)
        tot += d
    keep = List[str]()
    for b in all_bads:
        g = 0.0
        for k in range(len(insts)):
            if b not in insts[k].info.bads:
                g += durs[k]
        if g / tot < interp_thresh:
            keep.append(b)
    for inst in out:
        need = False
        for b in inst.info.bads:
            if b not in keep:
                need = True
        if need:
            inst.interpolate_bads(exclude = keep)
        inst.info.bads = [b for b in keep]
    return out


class _KdeObj:
    kde: gaussian_kde
    cut: float

    def __init__(self, kde: gaussian_kde, cut: float):
        self.kde = kde
        self.cut = cut

    def __call__(self, x: float) -> float:
        if x < self.cut and x > 0.0:
            r = self.kde.evaluate(np.array([x]))
            return float(r[0])
        return float("inf")


def compute_bridged_electrodes(inst, lm_cutoff: float = 16, epoch_threshold: float = 0.5,
                               l_freq: float = 0.5, h_freq: float = 30, epoch_duration: float = 2,
                               bw_method = None, verbose = None):
    """Bridged EEG electrodes via the intrinsic Hjorth algorithm (mne 1.13.2)."""
    from ..epochs import make_fixed_length_epochs
    from .. import pick_types
    pk = pick_types(inst.info, eeg = True)
    if len(pk) == 0:
        raise RuntimeError("No EEG channels found, cannot compute electrode bridging")
    c = inst.copy()
    c.filter(l_freq = l_freq, h_freq = h_freq, picks = pk)
    data = np.zeros((1, 1, 1))
    if isinstance(c, RawArray):
        ep = make_fixed_length_epochs(c, duration = epoch_duration, preload = True)
        data = ep.get_data(picks = pk)
    elif isinstance(c, EpochsArray):
        data = c.get_data(picks = pk)
    else:
        d2 = c.data[np.array(pk)]
        data = d2.reshape((1, d2.shape[0], d2.shape[1]))
    ne = data.shape[0]
    npk = len(pk)
    nt = data.shape[2]
    ed = np.full((ne, npk, npk), float("nan"))
    for i in range(npk):
        for j in range(i + 1, npk):
            for e in range(ne):
                m = 0.0
                for t in range(nt):
                    m += data[e, i, t] - data[e, j, t]
                m /= float(nt)
                v = 0.0
                for t in range(nt):
                    dd = data[e, i, t] - data[e, j, t] - m
                    v += dd * dd
                ed[e, i, j] = v / float(nt) * 1e12
    bridged = List[Tuple[int, int]]()
    below = List[float]()
    for e in range(ne):
        for i in range(npk):
            for j in range(i + 1, npk):
                v = ed[e, i, j]
                if v < lm_cutoff:
                    below.append(v)
    if float(len(below)) / float(ne) < epoch_threshold:
        return bridged, ed
    kde = gaussian_kde(np.array(below), bw_method = bw_method)
    lm = float(minimize_scalar(_KdeObj(kde, float(lm_cutoff))).x)
    for i in range(npk):
        for j in range(i + 1, npk):
            cnt = 0
            for e in range(ne):
                if ed[e, i, j] < lm:
                    cnt += 1
            if float(cnt) / float(ne) > epoch_threshold:
                bridged.append((pk[i], pk[j]))
    return bridged, ed
