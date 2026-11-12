# Ported from mne 1.13.2 (BSD-3-Clause)
#
# Epoching raw data around detected events (the Epochs(raw, events, ...)
# subset used by create_ecg_epochs / create_eog_epochs: windows that do not
# fit inside the data are dropped, optional baseline correction) and
# cross-trial phase statistics (ctps, Kuiper test).

import numpy as np
import math
from .._containers import RawArray, EpochsArray
from ._events import find_ecg_events, find_eog_events

def _epochs_from_raw(raw, events: np.ndarray[int, 2], event_id: int, tmin: float, tmax: float,
                     picks, baseline, decim: int = 1):
    from .._info import _picks_to_idx
    sf = raw.info.sfreq
    pk = _picks_to_idx(raw.info, picks, "all", ())
    s0 = int(round(tmin * sf))
    s1 = int(round(tmax * sf))
    nt = s1 - s0 + 1
    keep = List[int]()
    for i in range(events.shape[0]):
        st = int(events[i, 0]) - raw.first_samp + s0
        if st >= 0 and st + nt <= raw._data.shape[1]:
            keep.append(i)
    if len(keep) == 0:
        raise ValueError("No epochs could be created (all windows fall outside the data).")
    data = np.empty((len(keep), len(pk), nt))
    ev = np.empty((len(keep), 3), int)
    for q in range(len(keep)):
        i = keep[q]
        st = int(events[i, 0]) - raw.first_samp + s0
        for c in range(len(pk)):
            data[q, c] = raw._data[pk[c], st:st + nt]
        ev[q, 0] = events[i, 0]
        ev[q, 1] = events[i, 1]
        ev[q, 2] = events[i, 2]
    if decim > 1:
        data = np.ascontiguousarray(data[:, :, ::decim])
    info = raw.info._pick(pk)
    return EpochsArray(data, info, ev, s0 / sf, {str(event_id): event_id}, baseline=baseline)

def create_ecg_epochs(raw, ch_name: Optional[str] = None, event_id: int = 999, picks = None,
                      tmin: float = -0.5, tmax: float = 0.5, l_freq: float = 8.0,
                      h_freq: float = 16.0, reject = None, flat = None, baseline = None,
                      preload: bool = True, keep_ecg: bool = False,
                      reject_by_annotation: bool = True, decim: int = 1, verbose = None):
    if keep_ecg:
        raise ValueError("keep_ecg can be True only if the ECG channel is created synthetically and preload=True.")
    events, _, _ = find_ecg_events(raw, event_id=event_id, ch_name=ch_name, l_freq=l_freq, h_freq=h_freq)
    if events.shape[0] == 0:
        raise ValueError("No ECG events could be found.")
    return _epochs_from_raw(raw, events, event_id, tmin, tmax, picks, baseline, decim)

def create_eog_epochs(raw, ch_name = None, event_id: int = 998, picks = None, tmin: float = -0.5,
                      tmax: float = 0.5, l_freq: float = 1.0, h_freq: float = 10.0, reject = None,
                      flat = None, baseline = None, preload: bool = True,
                      reject_by_annotation: bool = True, thresh: Optional[float] = None,
                      decim: int = 1, verbose = None):
    events = find_eog_events(raw, event_id=event_id, l_freq=l_freq, h_freq=h_freq,
                             ch_name=ch_name, thresh=thresh)
    return _epochs_from_raw(raw, events, event_id, tmin, tmax, picks, baseline, decim)

def _prob_kuiper(d: np.ndarray[float, 1], n_eff: int) -> np.ndarray[float, 1]:
    en = math.sqrt(float(n_eff))
    out = np.empty(d.shape[0])
    for t in range(d.shape[0]):
        kl = (en + 0.155 + 0.24 / en) * d[t]
        l2 = kl * kl
        # -logsumexp(a, b=b) with a_j = -2 j^2 l2, b_j = 2(4 j^2 l2 - 1)
        mx = -np.inf
        for j in range(1, 101):
            a = -2.0 * j * j * l2
            if a > mx:
                mx = a
        s = 0.0
        for j in range(1, 101):
            a = -2.0 * j * j * l2
            b = 2.0 * (4.0 * j * j * l2 - 1.0)
            s += b * math.exp(a - mx)
        v = -(mx + math.log(abs(s))) / (2.0 * n_eff)
        if s < 0.0:
            v = np.nan
        if kl < 0.4:
            v = 0.0
        if v > 1.0:
            v = 1.0
        out[t] = v
    return out

def kuiper(data: np.ndarray[float, 2]):
    """Kuiper statistic and normalized pK over trials (axis 0)."""
    x = np.sort(data, axis=0)
    n = x.shape[0]
    nt = x.shape[1]
    d = np.empty(nt)
    for t in range(nt):
        d1 = -np.inf
        d2 = -np.inf
        for i in range(n):
            a = (i + 1.0) / n - x[i, t]
            b = x[i, t] - float(i) / n
            if a > d1:
                d1 = a
            if b > d2:
                d2 = b
        d[t] = d1 + d2
    return d, _prob_kuiper(d, n)

def ctps(data, is_raw: bool = True):
    import scipy.signal
    x = np.asarray(data, dtype=float)
    if x.ndim != 3:
        compile_error("Data must have 3 dimensions")
    ne, ns, nt = x.shape
    ph = x.copy()
    if is_raw:
        for e in range(ne):
            for s in range(ns):
                h = scipy.signal.hilbert(x[e, s].copy())
                for t in range(nt):
                    ph[e, s, t] = (math.atan2(h[t].imag, h[t].real) + np.pi) / (2.0 * np.pi)
    ks = np.zeros((ns, nt))
    pk = np.zeros((ns, nt))
    for s in range(ns):
        src = np.ascontiguousarray(ph[:, s, :])
        k, p = kuiper(src)
        ks[s] = k
        pk[s] = p
    return ks, pk, ph
