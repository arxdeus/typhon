# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# NIRx (NIRStar 15.x, non-Aurora) reader (mne/io/nirx/nirx.py). Narrower:
# probeInfo.mat only (no json), English month names or ISO dates, no
# subject_info/dig, saturated="annotate"/"nan" need a single wl file pair
# (saturated flags are ignored when nosatflags files exist: "ignore").

import numpy as np
import os
from scipy.io import loadmat
from ._bytes import _read_text
from ._fmtutil import _utc_secs
from .._info import Info
from .._containers import RawArray
from ..annotations import Annotations

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_T = [[0.9999935422358484, 0.0035518702911386253, 0.00020202444732790286, -0.0017627306462569704],
      [-0.003557613783350072, 0.9983891728513405, 0.056626181675695435, 0.031094368998675088],
      [-5.861166682581691e-07, -0.056626298986206525, 0.9983954908642263, 0.0395972485325352]]


def _ints(s: str) -> List[int]:
    out = List[int]()
    cur = ""
    for ch in s:
        if ch.isdigit():
            cur += ch
        else:
            if cur != "":
                out.append(int(cur))
            cur = ""
    if cur != "":
        out.append(int(cur))
    return out


def _find_files(d: str, suffix: str) -> List[str]:
    out = List[str]()
    for f in sorted(os.listdir(d)):
        if f.endswith(suffix):
            out.append(d + "/" + f)
    return out


def _hdr_get(txt: str, section: str, key: str) -> str:
    sec = ""
    for line in txt.split("\n"):
        l = line.strip()
        if l.startswith("[") and l.endswith("]"):
            sec = l[1:len(l) - 1]
            continue
        if sec == section and "=" in l:
            k = l.find("=")
            if l[:k].strip() == key:
                return l[k + 1:].strip()
    raise KeyError(section + "/" + key)


def _strip_comments(txt: str) -> str:
    # re.sub("#.*?#", "", txt, flags=DOTALL)
    out = ""
    i = 0
    n = len(txt)
    while i < n:
        if txt[i] == "#":
            j = txt.find("#", i + 1)
            if j < 0:
                out += txt[i:]
                break
            i = j + 1
        else:
            out += txt[i]
            i += 1
    return out


def _meas_date(s: str) -> float:
    t = s.replace("\"", "")
    # ISO "YYYY-MM-DD HH:MM:SS.f"
    try:
        if len(t) >= 19 and t[4] == "-" and t[7] == "-":
            fr = 0.0
            if len(t) > 20 and t[19] == ".":
                fr = float("0" + t[19:])
            return _utc_secs(int(t[0:4]), int(t[5:7]), int(t[8:10]), int(t[11:13]), int(t[14:16]), int(t[17:19])) + fr
        p = t.find(":")
        hh = int(t[p - 2:p])
        year = int(t[p - 6:p - 2])
        mm = int(t[p + 1:p + 3])
        ss = int(t[p + 4:p + 6])
        fr = 0.0
        if len(t) > p + 6 and t[p + 6] == ".":
            fr = float("0" + t[p + 6:])
        head = t[:p - 6].replace(",", " ").split()
        mon = 0
        day = 0
        for tk in head:
            if tk in _MONTHS:
                mon = _MONTHS.index(tk) + 1
            elif tk.isdigit():
                day = int(tk)
        if mon == 0 or day == 0:
            raise ValueError("bad date")
        return _utc_secs(year, mon, day, hh, mm, ss) + fr
    except:
        print("RuntimeWarning: Extraction of measurement date from NIRX file failed. The date is being set to January 1st, 2000.")
        return 946684800.0


def read_raw_nirx(fname, saturated: str = "annotate", *, preload = False,
                  encoding: str = "latin-1", verbose = None) -> RawArray:
    if saturated not in ("annotate", "nan", "ignore"):
        raise ValueError(f"Invalid value for the 'saturated' parameter: {saturated}")
    d = str(fname)
    if d.endswith(".hdr"):
        j = d.rfind("/")
        d = d[:j] if j >= 0 else "."
    if len(_find_files(d, "config.json")) > 0:
        raise NotImplementedError("Aurora NIRx recordings are not supported in Typhon mne")
    hdr_f = _find_files(d, "hdr")
    if len(hdr_f) != 1:
        raise RuntimeError(f"Need one hdr file, got {len(hdr_f)}")
    for k in ("inf", "set", "tpl", "config.txt"):
        n = len(_find_files(d, k))
        if n != 1:
            raise RuntimeError(f"Need one {k} file, got {n}")
    wl = List[str]()
    for key in ("wl1", "wl2"):
        fs = _find_files(d, key)
        pick = ""
        if len(fs) == 1:
            pick = fs[0]
        else:
            nos = List[int]()
            for i in range(len(fs)):
                if "nosatflags_" in fs[i][fs[i].rfind("/") + 1:]:
                    nos.append(i)
            if len(nos) != 1 or len(fs) != 2:
                raise RuntimeError(f"Need one nosatflags and one standard {key} file, got {len(fs)}")
            if saturated == "ignore":
                pick = fs[nos[0]]
            else:
                raise NotImplementedError("saturated='annotate'/'nan' with nosatflags files is not supported in Typhon mne")
        wl.append(pick)
    pm = _find_files(d, "probeInfo.mat")
    if len(pm) != 1:
        raise RuntimeError(f"Need one probeInfo.mat or probeInfo.json file, got {len(pm)} .mat and 0 .json")
    hdr_all = _read_text(hdr_f[0])
    hdr = _strip_comments(hdr_all)
    ver = _hdr_get(hdr, "GeneralInfo", "NIRStar")
    if ver not in ("\"15.0\"", "\"15.2\"", "\"15.3\""):
        raise RuntimeError(f"MNE does not support this NIRStar version ({ver})")
    dev = _hdr_get(hdr, "GeneralInfo", "Device")
    if "NIRScout" not in dev and "NIRSport" not in dev:
        print(f"RuntimeWarning: Only import of data from NIRScout devices have been thoroughly tested. You are using a {dev} device.")
    meas = _meas_date(_hdr_get(hdr, "GeneralInfo", "Date") + _hdr_get(hdr, "GeneralInfo", "Time"))
    waves = _ints(_hdr_get(hdr, "ImagingParameters", "Wavelengths"))
    sdkey = _hdr_get(hdr, "DataStructure", "S-D-Key")
    nums = _ints(sdkey.replace(":", " ").replace("-", " ").replace(",", " "))
    sources = List[int]()
    detectors = List[int]()
    i = 0
    while i + 2 < len(nums):
        sources.append(nums[i])
        detectors.append(nums[i + 1])
        i += 3
    sfreq = float(_hdr_get(hdr, "ImagingParameters", "SamplingRate"))
    m = loadmat(pm[0])
    probes = m["probeInfo"]["probes"]
    idx_c = probes["index_c"].array()
    s3 = probes["coords_s3"].array()
    d3 = probes["coords_d3"].array()
    req = List[int]()
    for r in range(idx_c.shape[0]):
        for k in range(len(sources)):
            if sources[k] == int(idx_c[r, 0]) and detectors[k] == int(idx_c[r, 1]):
                req.append(k)
    names = List[str]()
    for k in req:
        for w in range(2):
            names.append(f"S{sources[k]}_D{detectors[k]} {waves[w]}")
    types = ["fnirs_cw_amplitude" for _ in names]
    info = Info(names, sfreq, types)
    info.meas_date = meas
    for ci in range(len(req)):
        k = req[ci]
        sp = _head(s3, sources[k] - 1)
        dp = _head(d3, detectors[k] - 1)
        for w in range(2):
            info.locs[ci * 2 + w] = [(sp[0] + dp[0]) / 2.0, (sp[1] + dp[1]) / 2.0, (sp[2] + dp[2]) / 2.0,
                                      sp[0], sp[1], sp[2], dp[0], dp[1], dp[2], float(waves[w]), 0.0, 0.0]
    # data
    cols = List[List[List[float]]]()
    n_samp = -1
    for w in range(2):
        txt = _read_text(wl[w])
        rows = List[List[float]]()
        for line in txt.split("\n"):
            s = line.strip()
            if s == "":
                continue
            p = s.split()
            r = List[float]()
            for k in req:
                r.append(float(p[k]))
            rows.append(r)
        cols.append(rows)
        if n_samp < 0 or len(rows) < n_samp:
            n_samp = len(rows)
    nch = len(names)
    data = np.zeros((nch, n_samp))
    for s in range(n_samp):
        for c in range(len(req)):
            data[c * 2, s] = cols[0][s][c]
            data[c * 2 + 1, s] = cols[1][s][c]
    raw = RawArray(data, info)
    raw._raw_kind = "RawNIRX"
    # triggers
    evt = hdr_f[0][:len(hdr_f[0]) - 3] + "evt"
    on = List[float]()
    du = List[float]()
    ds = List[str]()
    if os.path.exists(evt):
        for line in _read_text(evt).split("\n"):
            t = _ints(line)
            if len(t) == 0:
                continue
            bits = ""
            for q in range(len(t) - 1, 0, -1):
                bits += str(t[q])
            v = 0
            for ch in bits:
                v = v * 2 + (1 if ch == "1" else 0)
            on.append(float(t[0]) / sfreq)
            du.append(1.0)
            ds.append(str(float(v)))
    raw.set_annotations(Annotations(on, du, ds), False)
    return raw


def _head(a: np.ndarray[float, 2], i: int) -> List[float]:
    x = a[i, 0] / 100.0
    y = a[i, 1] / 100.0
    z = a[i, 2] / 100.0
    return [_T[0][0] * x + _T[0][1] * y + _T[0][2] * z + _T[0][3],
            _T[1][0] * x + _T[1][1] * y + _T[1][2] * z + _T[1][3],
            _T[2][0] * x + _T[2][1] * y + _T[2][2] * z + _T[2][3]]
