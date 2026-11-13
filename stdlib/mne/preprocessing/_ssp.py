# Ported from mne 1.13.2 (BSD-3-Clause)
#
# compute_proj_ecg / compute_proj_eog: SSP projectors from ECG/EOG artifacts.

import numpy as np
from .._info import Info, Projection
from .._containers import RawArray
from ._events import find_ecg_events, find_eog_events
from .._proj import compute_proj_evoked, compute_proj_epochs, make_eeg_average_ref_proj


def _count_type(info: Info, kinds: List[str]) -> int:
    n = 0
    for i in range(info.nchan):
        if info.ch_names[i] in info.bads:
            continue
        if info.ch_types[i] in kinds:
            n += 1
    return n


def _exg_dict(d, info: Info) -> Optional[Dict[str, float]]:
    if isinstance(d, NoneType):
        return None
    out = Dict[str, float]()
    for k in d:
        out[str(k)] = float(d[k])
    for t in ("grad", "mag", "eeg", "eog"):
        if t in out and _count_type(info, [t]) == 0:
            del out[t]
    return out


def _default_reject() -> Dict[str, float]:
    return {"grad": 2e-10, "mag": 3e-12, "eeg": 5e-05, "eog": 0.00025}


def _compute_exg_proj(mode: Literal[str], raw, raw_event, tmin: float, tmax: float, n_grad, n_mag, n_eeg,
                      l_freq: float, h_freq: float, average: bool, filter_length, ch_name,
                      reject, flat, bads, avg_ref: bool, no_proj: bool, event_id: int,
                      exg_l_freq: float, exg_h_freq: float, tstart: float, qrs_threshold,
                      filter_method: str, iir_params, copy: bool, meg: str):
    from ..epochs import Epochs
    r = raw.copy() if copy else raw
    projs = List[Projection]()
    if not no_proj:
        for p in r.info.projs:
            projs.append(p)
    if avg_ref:
        projs.append(make_eeg_average_ref_proj(r.info))
    ev_raw = r if isinstance(raw_event, NoneType) else raw_event
    events = np.zeros((0, 3), dtype=int)
    if mode == "ECG":
        events = find_ecg_events(ev_raw, event_id = event_id, ch_name = ch_name, tstart = tstart,
                                 l_freq = exg_l_freq, h_freq = exg_h_freq,
                                 qrs_threshold = qrs_threshold, filter_length = filter_length)[0]
    else:
        events = find_eog_events(ev_raw, event_id = event_id, l_freq = exg_l_freq,
                                 h_freq = exg_h_freq, filter_length = filter_length,
                                 ch_name = ch_name, tstart = tstart)
    if events.shape[0] < 1:
        return projs[:0], events
    my_info = r.info.copy()
    for b in bads:
        my_info.bads.append(str(b))
    rej = _exg_dict(reject, my_info)
    fl = _exg_dict(flat, my_info)
    picks = List[int]()
    for i in range(my_info.nchan):
        if my_info.ch_names[i] in my_info.bads:
            continue
        if my_info.ch_types[i] in ("mag", "grad", "eeg", "eog", "ecg"):
            picks.append(i)
    r.filter(l_freq, h_freq, picks = picks, filter_length = filter_length,
             l_trans_bandwidth = 0.5, h_trans_bandwidth = 0.5, method = filter_method,
             iir_params = iir_params, phase = "zero-double", fir_design = "firwin2")
    epochs = Epochs(r, events, None, tmin, tmax, None, picks, True, rej, fl, True)
    if epochs.events.shape[0] < 1:
        return projs[:0], events
    ev_projs = List[Projection]()
    if average:
        ev_projs = compute_proj_evoked(epochs.average(), n_grad, n_mag, n_eeg, meg)
    else:
        ev_projs = compute_proj_epochs(epochs, n_grad, n_mag, n_eeg, None, None, meg)
    for p in ev_projs:
        p.desc = mode + "-" + p.desc
        projs.append(p)
    return projs, events


def compute_proj_ecg(raw, raw_event = None, tmin: float = -0.2, tmax: float = 0.4, n_grad = 2,
                     n_mag = 2, n_eeg = 2, l_freq: float = 1.0, h_freq: float = 35.0,
                     average: bool = True, filter_length = "10s", n_jobs = None, ch_name = None,
                     reject = "default", flat = None, bads = (), avg_ref: bool = False,
                     no_proj: bool = False, event_id: int = 999, ecg_l_freq: float = 5,
                     ecg_h_freq: float = 35, tstart: float = 0.0, qrs_threshold = "auto",
                     filter_method: str = "fir", iir_params = None, copy: bool = True,
                     meg: str = "separate", verbose = None):
    """SSP projectors for ECG artifacts (mne 1.13.2)."""
    rj = _default_reject() if isinstance(reject, str) else reject
    return _compute_exg_proj("ECG", raw, raw_event, tmin, tmax, n_grad, n_mag, n_eeg, l_freq,
                             h_freq, average, filter_length, ch_name, rj, flat, bads, avg_ref,
                             no_proj, event_id, float(ecg_l_freq), float(ecg_h_freq), tstart,
                             qrs_threshold, filter_method, iir_params, copy, meg)


def compute_proj_eog(raw, raw_event = None, tmin: float = -0.2, tmax: float = 0.2, n_grad = 2,
                     n_mag = 2, n_eeg = 2, l_freq: float = 1.0, h_freq: float = 35.0,
                     average: bool = True, filter_length = "10s", n_jobs = None, reject = "default",
                     flat = None, bads = (), avg_ref: bool = False, no_proj: bool = False,
                     event_id: int = 998, eog_l_freq: float = 1, eog_h_freq: float = 10,
                     tstart: float = 0.0, filter_method: str = "fir", iir_params = None,
                     ch_name = None, copy: bool = True, meg: str = "separate", verbose = None):
    """SSP projectors for EOG artifacts (mne 1.13.2)."""
    rj = _default_reject() if isinstance(reject, str) else reject
    return _compute_exg_proj("EOG", raw, raw_event, tmin, tmax, n_grad, n_mag, n_eeg, l_freq,
                             h_freq, average, filter_length, ch_name, rj, flat, bads, avg_ref,
                             no_proj, event_id, float(eog_l_freq), float(eog_h_freq), tstart,
                             "auto", filter_method, iir_params, copy, meg)
