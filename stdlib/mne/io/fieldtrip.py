# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# FieldTrip .mat import (mne/io/fieldtrip). MAT v5 (uncompressed) only. Narrower:
# with info=None all channels are named-guessed (EOG/ECG/STI prefixes else misc),
# no elec/grad channel positions or dig.

import numpy as np
from scipy.io import loadmat
from .._info import Info
from .._containers import RawArray, EpochsArray
from ..epochs import EvokedArray


def _labels(v) -> List[str]:
    out = List[str]()
    if v.is_cell():
        for it in v.items:
            out.append(it.text)
    elif v.is_char():
        out.append(v.text)
    return out


def _sfreq(s) -> float:
    if s.has("fsample"):
        return s["fsample"].scalar()
    if not s.has("time"):
        raise ValueError("No Source for sfreq found")
    t = s["time"]
    a = t.items[0].data if t.is_cell() else t.data
    return 1.0 / (float(a[1]) - float(a[0]))


def _info(s, raw_info, names: List[str], sfreq: float) -> Tuple[Info, List[int]]:
    # returns (info, kept channel indices into the FieldTrip labels)
    keep = List[int]()
    if raw_info is not None:
        base = raw_info.copy()
        sel = List[int]()
        for i in range(len(names)):
            if names[i] in base.ch_names:
                keep.append(i)
                sel.append(base.ch_names.index(names[i]))
            else:
                print(f"RuntimeWarning: channel {names[i]} is present in the FieldTrip data but cannot be found in the provided info and will be removed")
        info = base._pick(sel)
        info.sfreq = sfreq
        return (info, keep)
    print("RuntimeWarning: Importing FieldTrip data without an info dict from the original file. Channel locations, orientations and types will be incorrect.")
    types = List[str]()
    for n in names:
        if n.startswith("EOG"):
            types.append("eog")
        elif n.startswith("ECG"):
            types.append("ecg")
        elif n.startswith("STI"):
            types.append("stim")
        else:
            types.append("misc")
        keep.append(len(keep))
    return (Info(names, sfreq, types), keep)


def _need_struct(m, data_name: str):
    if data_name not in m:
        raise KeyError(data_name)
    return m[data_name]


def read_raw_fieldtrip(fname, info, data_name: str = "data") -> RawArray:
    m = loadmat(str(fname))
    s = _need_struct(m, data_name)
    names = _labels(s["label"])
    sf = _sfreq(s)
    inf = _info(s, info, names, sf)
    tr = s["trial"]
    a = tr.items[0].array() if tr.is_cell() else tr.array()
    if tr.is_cell() and len(tr.items) > 1:
        raise RuntimeError("The data you are trying to load does not seem to be raw data")
    data = np.zeros((len(inf[1]), a.shape[1]))
    for k in range(len(inf[1])):
        data[k, :] = a[inf[1][k], :]
    return RawArray(data, inf[0])


def read_epochs_fieldtrip(fname, info, data_name: str = "data", trialinfo_column: int = 0):
    m = loadmat(str(fname))
    s = _need_struct(m, data_name)
    names = _labels(s["label"])
    sf = _sfreq(s)
    inf = _info(s, info, names, sf)
    tr = s["trial"]
    if not tr.is_cell():
        raise RuntimeError("Loading of epochs requires a cell array of trials")
    nt = len(tr.items)
    first = tr.items[0].array()
    nk = len(inf[1])
    data = np.zeros((nt, nk, first.shape[1]))
    for t in range(nt):
        a = tr.items[t].array()
        for k in range(nk):
            data[t, k, :] = a[inf[1][k], :]
    ev = None
    if s.has("trialinfo"):
        ti = s["trialinfo"].array()
        if trialinfo_column < 0:
            raise ValueError("trialinfo_column must be positive")
        if ti.shape[0] != nt and ti.shape[1] == nt:
            ti = ti.T
        if trialinfo_column > ti.shape[1] - 1:
            raise ValueError("trialinfo_column is higher than the amount of columns in trialinfo.")
        ev = np.zeros((nt, 3), dtype = int)
        for i in range(nt):
            ev[i, 0] = i
            ev[i, 2] = int(ti[i, trialinfo_column])
    tm = s["time"]
    t0 = tm.items[0].data[0] if tm.is_cell() else tm.data[0]
    if tm.is_cell():
        for it in tm.items:
            if it.data[0] != t0:
                raise RuntimeError("Loading data with non-uniform times per epoch is not supported")
    return EpochsArray(data, inf[0], ev, float(t0), proj = False)


def read_evoked_fieldtrip(fname, info, comment = None, data_name: str = "data"):
    m = loadmat(str(fname))
    s = _need_struct(m, data_name)
    names = _labels(s["label"])
    sf = _sfreq(s)
    inf = _info(s, info, names, sf)
    a = s["avg"].array()
    nk = len(inf[1])
    data = np.zeros((nk, a.shape[1]))
    for k in range(nk):
        data[k, :] = a[inf[1][k], :]
    return EvokedArray(data, inf[0], 0.0, "" if comment is None else str(comment))
