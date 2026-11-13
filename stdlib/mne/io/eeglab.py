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


def read_epochs_eeglab(input_fname, events = None, event_id = None, eog = (), *,
                       uint16_codec = None, montage_units: str = "auto", verbose = None):
    """mne.io.read_epochs_eeglab: .set (embedded data or .fdt) with trials > 1."""
    from .._containers import EpochsArray
    fn = str(input_fname)
    m = loadmat(fn)
    eeg = m["EEG"] if "EEG" in m else m["data"]
    use_top = "EEG" not in m
    if (events is None) != (event_id is None):
        raise ValueError("Both `events` and `event_id` must be None or not None")
    trials = int(m["trials"].scalar() if use_top else eeg["trials"].scalar())
    if trials <= 1:
        raise ValueError("The file does not seem to contain epochs (trials less than 2). You should try using read_raw_eeglab function.")
    srate = m["srate"].scalar() if use_top else eeg["srate"].scalar()
    nbchan = int(m["nbchan"].scalar() if use_top else eeg["nbchan"].scalar())
    pnts = int(m["pnts"].scalar() if use_top else eeg["pnts"].scalar())
    xmin = m["xmin"].scalar() if use_top else eeg["xmin"].scalar()
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
    # data (nbchan, pnts, trials), column-major
    dv = m["data"] if use_top else eeg["data"]
    flat = np.zeros(0)
    if dv.is_char():
        dpath = _dirname(fn) + dv.text
        try:
            b = _read_file(dpath)
        except:
            dpath = fn[:fn.rfind(".")] + ".fdt"
            b = _read_file(dpath)
        flat = _arr_le(b, 0, nbchan * pnts * trials, "f4")
    else:
        flat = dv.data
    data = np.zeros((trials, nbchan, pnts))
    for t in range(trials):
        for p in range(pnts):
            for c in range(nbchan):
                data[t, c, p] = flat[c + nbchan * (p + pnts * t)] * CAL
    # events from the epoch struct
    ev = None
    ids = Dict[str, int]()
    if events is None:
        epv = m["epoch"] if use_top else eeg["epoch"]
        evv = m["event"] if use_top else eeg["event"]
        if epv.size == 0 or evv.size == 0:
            print("RuntimeWarning: The EEGLAB file contains no event information. All epochs will be assigned to a single 'unknown' event.")
            ids["unknown"] = 1
            ev = np.zeros((trials, 3), dtype = int)
            for i in range(trials):
                ev[i, 0] = i
                ev[i, 2] = 1
        else:
            ev_name = List[str]()
            ev_lat = List[int]()
            uniq = List[str]()
            ev_idx = 0
            for i in range(epv.size):
                et = epv.get(i, "eventtype")
                nm = ""
                cnt = 1
                if et.is_char():
                    nm = et.text
                elif et.is_cell():
                    parts = List[str]()
                    for it in et.items:
                        parts.append(it.text if it.is_char() else _num_str(it.scalar()))
                    nm = "/".join(parts)
                    cnt = len(parts)
                else:
                    nm = _num_str(et.scalar())
                ev_name.append(nm)
                ev_lat.append(int(evv.get(ev_idx, "latency").scalar()) - 1)
                ev_idx += cnt
                if nm not in uniq:
                    uniq.append(nm)
            for k in range(len(uniq)):
                ids[uniq[k]] = k + 1
            ev = np.zeros((trials, 3), dtype = int)
            prev = 0
            for i in range(trials):
                if i == 0:
                    prev = 0
                elif ev_lat[i] - ev_lat[i - 1] == 1:
                    prev = ids[ev_name[i - 1]]
                ev[i, 0] = ev_lat[i]
                ev[i, 1] = prev
                ev[i, 2] = ids[ev_name[i]]
    else:
        ev = np.asarray(events)
        for k, v in event_id.items():
            found = False
            for i in range(ev.shape[0]):
                if int(ev[i, 2]) == v:
                    found = True
            if not found:
                raise ValueError(f"No matching events found for {k} (event id {v})")
        ids = event_id
    e = EpochsArray(data, info, ev, xmin, ids, proj = False)
    return e


def _num_str(v: float) -> str:
    if v == float(int(v)):
        return str(int(v))
    return str(v)
