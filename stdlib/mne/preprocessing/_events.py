# Ported from mne 1.13.2 (BSD-3-Clause)
#
# peak_finder (noise-tolerant extrema detection), EOG blink events and
# ECG QRS detection.

import numpy as np
from ..filter import filter_data
from .._info import Info

def peak_finder(x0, thresh: Optional[float] = None, extrema: int = 1, verbose = None):
    xa = np.asarray(x0, dtype=float)
    if xa.ndim != 1 or xa.size == 0:
        raise ValueError("The input data must be a non empty 1D vector")
    s = xa.size
    th = (float(np.max(xa)) - float(np.min(xa))) / 4.0 if thresh is None else thresh.__val__()
    if extrema != 1 and extrema != -1:
        raise ValueError("extrema must be -1 or 1")
    xx = xa * float(extrema)
    dx0 = np.diff(xx)
    eps = np.finfo(float).eps
    for i in range(dx0.size):
        if dx0[i] == 0.0:
            dx0[i] = -eps
    indl = List[int]()
    indl.append(0)
    for i in range(dx0.size - 1):
        if dx0[i] * dx0[i + 1] < 0.0:
            indl.append(i + 1)
    indl.append(s - 1)
    xl = [float(xx[i]) for i in indl]
    length = len(xl)
    min_mag = min(xl)
    out_i = List[int]()
    out_m = List[float]()
    if length > 2:
        temp_mag = min_mag
        found_peak = False
        left_min = min_mag
        sd0 = np.sign(xl[1] - xl[0])
        sd1 = np.sign(xl[2] - xl[1])
        ii = 0
        if sd0 <= 0:
            ii = -1
            if sd0 == sd1:
                xl = xl[:1] + xl[2:]
                indl = indl[:1] + indl[2:]
                length -= 1
        else:
            ii = 0
            if sd0 == sd1:
                xl = xl[1:]
                indl = indl[1:]
                length -= 1
        maxPeaks = int(np.ceil(length / 2.0))
        peak_loc = [0] * maxPeaks
        peak_mag = [0.0] * maxPeaks
        c_ind = 0
        temp_loc = 0
        while ii < length - 1:
            ii += 1
            if found_peak and ((xl[ii] > peak_mag[-1]) or (left_min < peak_mag[-1] - th)):
                temp_mag = min_mag
                found_peak = False
            if ii == length - 1:
                break
            if xl[ii] > temp_mag and xl[ii] > left_min + th:
                temp_loc = ii
                temp_mag = xl[ii]
            ii += 1
            if not found_peak and temp_mag > th + xl[ii]:
                found_peak = True
                left_min = xl[ii]
                peak_loc[c_ind] = temp_loc
                peak_mag[c_ind] = temp_mag
                c_ind += 1
            elif xl[ii] < left_min:
                left_min = xl[ii]
        if xl[-1] > temp_mag and xl[-1] > left_min + th:
            peak_loc[c_ind] = length - 1
            peak_mag[c_ind] = xl[-1]
            c_ind += 1
        elif not found_peak and temp_mag > min_mag:
            peak_loc[c_ind] = temp_loc
            peak_mag[c_ind] = temp_mag
            c_ind += 1
        for k in range(c_ind):
            out_i.append(indl[peak_loc[k]])
            out_m.append(peak_mag[k])
    else:
        xi = 0
        for k in range(length):
            if xl[k] > xl[xi]:
                xi = k
        if xl[xi] > min_mag + th:
            out_i.append(indl[xi])
            out_m.append(xl[xi])
    mags = _arr(out_m, float)
    if extrema < 0:
        mags = mags * -1.0
    return _arr(out_i, int), mags

def _arr(l, T: type):
    # np.array() of an empty list raises in Typhon numpy: build explicitly
    out = np.empty(len(l), T)
    for i in range(len(l)):
        out[i] = T(l[i])
    return out

def _hann2(x, sf: float, lo, hi, filter_length):
    return filter_data(x, sf, lo, hi, None, filter_length, 0.5, 0.5, phase="zero-double",
                       fir_window="hann", fir_design="firwin2")

def _find_eog_events(eog: np.ndarray[float, 2], event_id: int, l_freq: float, h_freq: float,
                     sampling_rate: float, first_samp: int, filter_length = "10s",
                     tstart: float = 0.0, thresh: Optional[float] = None):
    fmax = min(45.0, sampling_rate / 2.0 - 0.75)
    best = 0
    bestv = -1.0
    for r in range(eog.shape[0]):
        f = _hann2(eog[r].copy(), sampling_rate, 2.0, fmax, filter_length)
        v = float(np.sqrt((f * f).sum()))
        if v > bestv:
            bestv = v
            best = r
    filteog = _hann2(eog[best].copy(), sampling_rate, l_freq, h_freq, filter_length)
    temp = filteog - np.mean(filteog)
    n0 = int(sampling_rate * tstart)
    ev = np.empty(0, int)
    if abs(float(np.max(temp))) > abs(float(np.min(temp))):
        ev, _ = peak_finder(filteog[n0:], thresh, 1)
    else:
        ev, _ = peak_finder(filteog[n0:], thresh, -1)
    n = ev.size
    out = np.zeros((n, 3), int)
    for i in range(n):
        out[i, 0] = int(ev[i]) + n0 + first_samp
        out[i, 2] = event_id
    return out

def _type_idx(info: Info, kind: str, ch_name) -> List[int]:
    out = List[int]()
    if ch_name is None:
        for i in range(info.nchan):
            if info.ch_types[i] == kind and info.ch_names[i] not in info.bads:
                out.append(i)
    elif isinstance(ch_name, str) or isinstance(ch_name, Optional[str]):
        for c in str(ch_name).split(","):
            if c not in info.ch_names:
                raise ValueError(f"The specified channel cannot be found: {c}")
            out.append(info.ch_names.index(c))
    else:
        for c in ch_name:
            if str(c) not in info.ch_names:
                raise ValueError(f"The specified channel cannot be found: {c}")
            out.append(info.ch_names.index(str(c)))
    return out

def find_eog_events(raw, event_id: int = 998, l_freq: float = 1.0, h_freq: float = 10.0,
                    filter_length = "10s", ch_name = None, tstart: float = 0.0,
                    reject_by_annotation: bool = False, thresh: Optional[float] = None,
                    verbose = None):
    inds = _type_idx(raw.info, "eog", ch_name)
    if len(inds) == 0:
        raise RuntimeError("No EOG channel(s) found")
    eog = raw._data[np.array(inds), :].copy()
    return _find_eog_events(eog, event_id, l_freq, h_freq, raw.info.sfreq, raw.first_samp,
                            filter_length, tstart, thresh)

def qrs_detector(sfreq: float, ecg, thresh_value = 0.6, levels: float = 2.5, n_thresh: int = 3,
                 l_freq: Optional[float] = 5.0, h_freq: Optional[float] = 35.0,
                 tstart: float = 0.0, filter_length = "10s", verbose = None):
    win_size = int(round((60.0 * sfreq) / 120.0))
    e = np.asarray(ecg, dtype=float).copy()
    if l_freq is not None or h_freq is not None:
        e = _hann2(e, sfreq, l_freq, h_freq, filter_length)
    ecg_abs_all = np.abs(e)
    init = int(sfreq)
    n0 = int(sfreq * tstart)
    ecg_abs = ecg_abs_all[n0:]
    n_points = ecg_abs.size
    init_max = (float(np.max(ecg_abs[:init])) + float(np.max(ecg_abs[init:init * 2])) +
                float(np.max(ecg_abs[init * 2:init * 3]))) / 3.0
    runs = List[float]()
    if isinstance(thresh_value, str):
        if thresh_value != "auto":
            raise ValueError('threshold value must be "auto" or a float')
        for v in np.arange(0.3, 1.1, 0.05):
            runs.append(float(v))
    else:
        runs.append(float(thresh_value))
    clean = List[List[int]]()
    for tv in runs:
        thresh1 = init_max * tv
        numcross = List[int]()
        times = List[int]()
        rms = List[float]()
        ii = 0
        while ii < n_points - win_size:
            if ecg_abs[ii] > thresh1:
                w = ecg_abs[ii:ii + win_size]
                mt = int(np.argmax(w))
                times.append(ii + mt)
                above = w > thresh1
                nx = 0
                for k in range(win_size - 1):
                    nx += int(above[k + 1]) - int(above[k])
                numcross.append(nx)
                rms.append(float(np.sqrt((w * w).sum() / w.size)))
                ii += win_size
            else:
                ii += 1
        if len(rms) == 0:
            rms.append(0.0)
            times.append(0)
            numcross.append(0)
        ra = _arr(rms, float)
        rth = float(ra.mean()) + float(ra.std()) * levels
        ce = List[int]()
        for k in range(len(rms)):
            if rms[k] < rth and numcross[k] < n_thresh:
                ce.append(times[k] + n0)
        if len(ce) > 0:
            clean.append(ce)
    if len(clean) == 0:
        return np.empty(0, int)
    dur = e.size / float(sfreq)
    rates = np.array([60.0 * len(c) / dur for c in clean])
    good = List[float]()
    for r in rates:
        if 40.0 <= r <= 160.0:
            good.append(float(r))
    ideal = float(np.median(_arr(good, float))) if len(good) > 0 else 80.0
    bi = int(np.argmin(np.abs(rates - ideal)))
    return _arr(clean[bi], int)

def find_ecg_events(raw, event_id: int = 999, ch_name: Optional[str] = None, tstart: float = 0.0,
                    l_freq: float = 5.0, h_freq: float = 35.0, qrs_threshold = "auto",
                    filter_length = "10s", return_ecg: Literal[bool] = False,
                    reject_by_annotation: bool = True, verbose = None):
    inds = _type_idx(raw.info, "ecg", ch_name)
    if len(inds) == 0:
        raise RuntimeError("No ECG channel found (synthetic ECG is not supported in Typhon mne)")
    idx = inds[0]
    sf = raw.info.sfreq
    ecg = _hann2(raw._data[idx].copy(), sf, l_freq, h_freq, filter_length)
    ev = qrs_detector(sf, ecg, thresh_value=qrs_threshold, l_freq=None, h_freq=None, tstart=tstart)
    n = ev.size
    out = np.zeros((n, 3), int)
    for i in range(n):
        out[i, 0] = int(ev[i]) + raw.first_samp
        out[i, 2] = event_id
    dur_min = (ecg.size / sf - tstart) / 60.0
    pulse = n / dur_min
    if return_ecg:
        return out, idx, pulse, ecg.reshape((1, ecg.size))
    else:
        return out, idx, pulse
