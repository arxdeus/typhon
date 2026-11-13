# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# Persyst .lay/.dat reader (mne/io/persyst/persyst.py). subject_info is
# not stored.

import numpy as np
from ._bytes import _read_file, _read_text, _i16_le, _i32_le
from ._fmtutil import _utc_secs
from .._info import Info
from .._containers import RawArray
from ..annotations import Annotations


def _dir_of(p: str) -> str:
    j = p.rfind("/")
    return p[:j + 1] if j >= 0 else ""


def _base_of(p: str) -> str:
    j = max(p.rfind("/"), p.rfind("\\"))
    return p[j + 1:] if j >= 0 else p


def read_raw_persyst(fname, preload = False, verbose = None) -> RawArray:
    fn = str(fname)
    if not fn.endswith(".lay"):
        fn = fn + ".lay"
    txt = _read_text(fn)
    section = ""
    fileinfo = Dict[str, str]()
    patient = Dict[str, str]()
    ch_keys = List[str]()
    c_desc = List[str]()
    c_on = List[float]()
    c_du = List[float]()
    dat_path = ""
    for raw_line in txt.split("\n"):
        line = raw_line.strip()
        if line == "":
            continue
        if line[0] == "[" and line[-1] == "]" and len(line) >= 3:
            section = line[1:len(line) - 1].lower()
            continue
        if section == "comments":
            p = line.split(",", 4)
            c_desc.append(p[4])
            c_on.append(float(p[0]))
            c_du.append(float(p[1]))
            continue
        if "=" not in line:
            raise RuntimeError(f"The line {line} does not conform to the standards. Please check the .lay file.")
        pos = line.find("=")
        key = line[:pos]
        val = line[pos + 1:]
        if section != "channelmap":
            key = key.lower()
        if section == "fileinfo":
            if key == "file":
                dat_path = _dir_of(fn) + _base_of(val)
            fileinfo[key] = val
        elif section == "channelmap":
            ch_keys.append(key)
        elif section == "patient":
            patient[key] = val
    datatype = int(fileinfo["datatype"])
    cal = float(fileinfo["calibration"])
    n_chs = int(fileinfo["waveformcount"])
    meas = Optional[float](None)
    if "testdate" in patient:
        td = patient["testdate"]
        y = 0
        mo = 0
        d = 0
        ok = True
        if "/" in td:
            p = td.split("/")
            mo = int(p[0])
            d = int(p[1])
            y = int(p[2])
        elif "-" in td:
            p = td.split("-")
            d = int(p[0])
            mo = int(p[1])
            y = int(p[2])
        elif "." in td:
            p = td.split(".")
            y = int(p[0])
            mo = int(p[1])
            d = int(p[2])
        else:
            ok = False
        if ok:
            tt = patient["testtime"].split(":")
            meas = _utc_secs(y, mo, d, int(tt[0]), int(tt[1]), int(tt[2]))
        else:
            print("RuntimeWarning: Cannot read in the measurement date due to incompatible format.")
    if n_chs != len(ch_keys):
        raise RuntimeError("Channels in lay file do not match the number of channels in the .dat file.")
    names = List[str]()
    for k in ch_keys:
        up = k.upper()
        j = up.find("-REF")
        names.append(up[:j] if j >= 0 else up)
    sfreq = float(fileinfo["samplingrate"])
    types = ["eeg" for _ in range(n_chs)]
    info = Info(names, sfreq, types)
    for i in range(n_chs):
        info.cals[i] = cal * 1.0e-6
    info.meas_date = meas
    b = _read_file(dat_path)
    isz = 0
    if datatype == 7:
        isz = 4
    elif datatype == 0:
        isz = 2
    else:
        raise RuntimeError(f"Unknown format: {datatype}")
    ns = len(b) // (isz * n_chs)
    data = np.zeros((n_chs, ns))
    for s in range(ns):
        for c in range(n_chs):
            off = (s * n_chs + c) * isz
            v = _i32_le(b, off) if isz == 4 else _i16_le(b, off)
            data[c, s] = float(v) * info.cals[c]
    raw = RawArray(data, info)
    raw._raw_kind = "RawPersyst"
    # annotations grouped by first-seen description (dict order in mne)
    order = List[str]()
    for d in c_desc:
        if d not in order:
            order.append(d)
    on = List[float]()
    du = List[float]()
    ds = List[str]()
    for o in order:
        for i in range(len(c_desc)):
            if c_desc[i] == o:
                on.append(c_on[i])
                du.append(c_du[i])
                ds.append(o)
    raw.set_annotations(Annotations(on, du, ds), False)
    return raw
