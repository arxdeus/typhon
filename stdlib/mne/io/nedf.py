# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# NeuroElectrics NEDF reader (mne/io/nedf/nedf.py). The XML header is
# parsed with a tiny tag scanner (no XML dependency).

import numpy as np
from ._bytes import _read_file, _ascii, _u8, _uint_be, _sint
from .._info import Info
from .._containers import RawArray

_HDRLEN = 10240


def _find_text(s: str, tag: str) -> Optional[str]:
    a = s.find("<" + tag + ">")
    if a < 0:
        return None
    a += len(tag) + 2
    e = s.find("</" + tag + ">", a)
    if e < 0:
        return None
    return s[a:e]


def _need(s: str, name: str) -> str:
    v = _find_text(s, name)
    if v is None or v == "":
        raise RuntimeError("NEDF header " + name + " not found")
    return v


def read_raw_nedf(filename, preload = False, verbose = None) -> RawArray:
    fn = str(filename)
    b = _read_file(fn)
    end = -1
    lim = min(len(b), _HDRLEN)
    for k in range(lim):
        if _u8(b, k) == 0:
            end = k
            break
    if end < 0:
        raise RuntimeError("End of header null not found")
    h = _ascii(b, 0, end)
    ver = _find_text(h, "NEDFversion")
    ver_s = ver if ver is not None else ""
    if ver_s != "1.3" and ver_s != "1.4":
        print("RuntimeWarning: NEDFversion unsupported, use with caution")
    add = _find_text(h, "AdditionalChannelStatus")
    if add is not None and add != "OFF":
        raise RuntimeError("Unknown additional channel, aborting.")
    nacc_s = _find_text(h, "NumberOfChannelsOfAccelerometer")
    n_acc = int(nacc_s) if nacc_s is not None and nacc_s != "" else 0
    eeg = _find_text(h, "EEGSettings")
    if eeg is None:
        raise RuntimeError("No EEG channels found")
    ntot = int(_need(eeg, "TotalNumberOfChannels"))
    sfreq = float(int(_need(eeg, "EEGSamplingRate")))
    mont = _find_text(eeg, "EEGMontage")
    names = List[str]()
    if mont is not None:
        pos = 0
        while True:
            a = mont.find(">", pos)
            if a < 0:
                break
            if mont[a - 1] == "/":
                pos = a + 1
                continue
            # find the matching close tag text
            e = mont.find("</", a)
            if e < 0:
                break
            names.append(mont[a + 1:e])
            pos = mont.find(">", e) + 1
    if ntot != len(names):
        raise RuntimeError(f"TotalNumberOfChannels ({ntot}) != channel count ({len(names)})")
    has_stim = _find_text(h, "STIMSettings") is not None
    trig_bytes = 4 if ver_s != "" else 1
    date_s = _find_text(h, "StartDate_firstEEGTimestamp")
    ms = int(date_s) if date_s is not None and date_s != "" else 0
    n_samples = int(_need(eeg, "NumberOfRecordsOfEEG"))
    sample_bytes = ntot * 3 + (2 * ntot * 3 if has_stim else 0) + trig_bytes
    chunk_bytes = n_acc * 2 + 5 * sample_bytes
    n_full = n_samples // 5
    n_last = n_samples % 5
    names.append("STI 014")
    types = List[str]()
    for k in range(ntot):
        types.append("eeg")
    types.append("stim")
    info = Info(names, sfreq, types)
    for k in range(ntot):
        info.cals[k] = 2.4 / (6.0 * 8388607.0)
    info.units[ntot] = -1
    info.coil_types[ntot] = 0
    info.meas_date = float(ms) / 1000.0
    data = np.zeros((ntot + 1, n_samples))
    for s in range(n_samples):
        ci = s // 5
        si = s % 5
        base = _HDRLEN + ci * chunk_bytes + n_acc * 2 + si * sample_bytes
        if ci >= n_full:
            base = _HDRLEN + n_full * chunk_bytes + n_acc * 2 + si * sample_bytes
        for c in range(ntot):
            v = _uint_be(b, base + c * 3, 3)
            if v > (1 << 23):
                v -= (1 << 24)
            data[c, s] = float(v) * info.cals[c]
        tb = base + ntot * 3 + (2 * ntot * 3 if has_stim else 0)
        if trig_bytes == 4:
            data[ntot, s] = float(_sint(_uint_be(b, tb, 4), 4))
        else:
            data[ntot, s] = float(_u8(b, tb))
    raw = RawArray(data, info)
    raw._raw_kind = "RawNedf"
    return raw
