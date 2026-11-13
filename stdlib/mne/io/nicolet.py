# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# Nicolet .data/.head reader (mne/io/nicolet/nicolet.py): int16 multiplexed.

import numpy as np
from ._bytes import _read_file, _read_text, _i16_le
from ._fmtutil import _utc_secs, _sel_idx, _prefix_idx, _set_kind
from .._info import Info
from .._containers import RawArray


def _kind_idx(sel, names: List[str], kind: str) -> List[int]:
    if isinstance(sel, str):
        return _prefix_idx(names, kind.upper())
    return _sel_idx(sel, names)


def read_raw_nicolet(input_fname, ch_type: str, eog = (), ecg = (), emg = (), misc = (),
                     preload = False, verbose = None) -> RawArray:
    fn = str(input_fname)
    if not fn.endswith(".data"):
        raise ValueError(f"File name should end with .data not \"{fn[fn.rfind('.'):]}\".")
    txt = _read_text(fn[:-5] + ".head")
    names = List[str]()
    sfreq = 0.0
    conv = 1.0
    nsamp = 0
    start = ""
    for line in txt.split("\n"):
        if "=" not in line:
            continue
        k = line.find("=")
        var = line[:k]
        val = line[k + 1:].strip()
        if var == "elec_names":
            inner = val[1:-1]
            for p in inner.split(","):
                names.append(p)
        elif var == "conversion_factor":
            conv = float(val)
        elif var == "num_samples":
            nsamp = int(val)
        elif var == "start_ts":
            start = val
        elif var == "sample_freq":
            sfreq = float(val)
    n = len(names)
    if ch_type != "eeg" and ch_type != "seeg":
        raise TypeError("Channel type not recognized. Available types are 'eeg' and 'seeg'.")
    types = [ch_type for _ in range(n)]
    info = Info(names, sfreq, types)
    for i in range(n):
        info.cals[i] = conv * 1e-6
    ds = start.split()
    d = ds[0].split("-")
    t = ds[1].split(":")
    sm = t[2].split(".")
    info.meas_date = _utc_secs(int(d[0]), int(d[1]), int(d[2]), int(t[0]), int(t[1]), int(sm[0]))
    info.highpass = 0.0
    info.lowpass = sfreq / 2.0
    # kind overrides (eog/ecg/emg/misc), auto = name prefix
    _set_kind(info, _kind_idx(eog, names, "eog"), "eog")
    _set_kind(info, _kind_idx(ecg, names, "ecg"), "ecg")
    _set_kind(info, _kind_idx(emg, names, "emg"), "emg")
    _set_kind(info, _kind_idx(misc, names, "misc"), "misc")
    b = _read_file(fn)
    data = np.zeros((n, nsamp))
    for s in range(nsamp):
        for c in range(n):
            data[c, s] = float(_i16_le(b, (s * n + c) * 2)) * info.cals[c]
    raw = RawArray(data, info)
    raw._raw_kind = "RawNicolet"
    return raw
