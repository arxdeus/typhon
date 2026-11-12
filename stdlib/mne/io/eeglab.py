# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# EEGLAB .set reader (mne/io/eeglab/eeglab.py): data embedded in the .set
# or in a separate .fdt (float32, multiplexed). Uses the minimal MAT v5
# reader in scipy.io (uncompressed files only).

import numpy as np
from scipy.io import loadmat
from ._bytes import _read_file, _arr_le
from .._info import Info, _VALID_CH_TYPES
from .._containers import RawArray
from ..annotations import Annotations

CAL = 1e-6


def _dirname(p: str) -> str:
    j = p.rfind("/")
    return p[:j + 1] if j >= 0 else ""


def read_raw_eeglab(input_fname, eog = (), preload = False, uint16_codec = None,
                    montage_units: str = "auto", verbose = None) -> RawArray:
    fn = str(input_fname)
    m = loadmat(fn)
    eeg = m["EEG"] if "EEG" in m else m["data"]
    use_top = "EEG" not in m
    trials = 1
    if use_top:
        trials = int(m["trials"].scalar())
    else:
        trials = int(eeg["trials"].scalar())
    if trials != 1:
        raise TypeError(f"The number of trials is {trials:d}. It must be 1 for raw files. Please use `mne.io.read_epochs_eeglab` if the .set file contains epochs.")
    srate = m["srate"].scalar() if use_top else eeg["srate"].scalar()
    nbchan = int(m["nbchan"].scalar() if use_top else eeg["nbchan"].scalar())
    pnts = int(m["pnts"].scalar() if use_top else eeg["pnts"].scalar())
    chanlocs = m["chanlocs"] if use_top else eeg["chanlocs"]
    names = List[str]()
    types = List[str]()
    if chanlocs.is_struct() and chanlocs.size > 0:
        for i in range(chanlocs.size):
            names.append(str(chanlocs.get(i, "labels")))
            t = "eeg"
            if chanlocs.has("type"):
                tv = chanlocs.get(i, "type")
                if tv.is_char():
                    tt = tv.text.strip().lower()
                    if tt in _VALID_CH_TYPES:
                        t = tt
            types.append(t)
    else:
        for i in range(nbchan):
            names.append(f"EEG {i:03d}")
            types.append("eeg")
    eogl = [str(x) for x in eog]
    for i in range(len(names)):
        if names[i] in eogl:
            types[i] = "eog"
    info = Info(names, srate, types)
    for i in range(len(names)):
        info.cals[i] = CAL
        if types[i] == "eog":
            info.coil_types[i] = 0
    dv = m["data"] if use_top else eeg["data"]
    data = np.zeros((nbchan, pnts))
    if dv.is_char():
        dname = dv.text
        dpath = _dirname(fn) + dname
        try:
            b = _read_file(dpath)
        except:
            dpath = fn[:fn.rfind(".")] + ".fdt"
            b = _read_file(dpath)
        flat = _arr_le(b, 0, nbchan * pnts, "f4")
        mm = flat.reshape((pnts, nbchan))
        for c in range(nbchan):
            data[c, :] = mm[:, c] * CAL
    else:
        a = dv.array()
        data = a * CAL
    raw = RawArray(data, info)
    raw._raw_kind = "RawEEGLAB"
    ev = m["event"] if use_top else eeg["event"]
    on = List[float]()
    du = List[float]()
    de = List[str]()
    if ev.is_struct():
        for i in range(ev.size):
            lat = ev.get(i, "latency").scalar() - 1
            if np.isnan(lat):
                continue
            tv = ev.get(i, "type")
            de.append(tv.text if tv.is_char() else str(int(tv.scalar())) if tv.scalar() == float(int(tv.scalar())) else str(tv.scalar()))
            on.append(lat / srate)
            d = 0.0
            if ev.has("duration"):
                dvv = ev.get(i, "duration")
                d = np.nan if dvv.size == 0 else dvv.scalar()
            du.append(d / srate)
    raw.set_annotations(Annotations(on, du, de), False)
    return raw
