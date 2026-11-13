# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# Hitachi fNIRS CSV reader (mne/io/hitachi/hitachi.py). Single file only;
# subject_info is not stored.

import numpy as np
from ._bytes import _read_text
from ._fmtutil import _utc_secs
from .._info import Info
from .._containers import RawArray


def _compute_pairs(n_rows: int, n_cols: int, n: int) -> List[Tuple[int, int]]:
    pairs = List[Tuple[int, int]]()
    for ri in range(n_rows):
        for ci in range(n_cols - 1):
            a = (ri * n_cols + ci) // 2
            b = (ri * n_cols + ci + 1) // 2
            if (ri + ci) % 2 == 1:
                pairs.append((b, a))
            else:
                pairs.append((a, b))
        if ri >= n_rows - 1:
            continue
        for ci in range(n_cols):
            a = (ri * n_cols + ci) // 2
            b = ((ri + 1) * n_cols + ci) // 2
            if (ri + ci) % 2 == 1:
                pairs.append((b, a))
            else:
                pairs.append((a, b))
    if n > 1:
        m0 = 0
        m1 = 0
        for p in pairs:
            m0 = max(m0, p[0])
            m1 = max(m1, p[1])
        base = len(pairs)
        for i in range(base):
            pairs.append((pairs[i][0] + m0 + 1, pairs[i][1] + m1 + 1))
    return pairs


def _strip_end(s: str) -> str:
    while len(s) > 0 and (s[-1] == "," or s[-1] == "\r" or s[-1] == "\n"):
        s = s[:-1]
    return s


def read_raw_hitachi(fname, preload = False, verbose = None) -> RawArray:
    txt = _read_text(str(fname))
    txt = _strip_end(txt)
    sep = "\n" if "\n" in txt else "\r"
    lines = txt.split(sep)
    if _strip_end(lines[0]) != "Header":
        raise RuntimeError("Could not parse file: no header found")
    sfreq = 0.0
    desc = ""
    mode = ""
    hp = 0.0
    lp = 0.0
    has_hp = False
    has_lp = False
    meas = 946684800.0
    fn_waves = [0, 0]
    ch_wl = Dict[str, float]()
    li = 0
    for i in range(1, len(lines)):
        line = lines[i]
        li = i
        if len(line) == 0:
            continue
        parts = _strip_end(line).split(",")
        kind = parts[0]
        rest = parts[1:]
        if len(rest) == 0:
            rest = [""]
        if kind == "AnalyzeMode":
            if len(rest) != 1 or rest[0] != "Continuous":
                raise RuntimeError("Could not parse file: not continuous data")
        elif kind == "Sampling Period[s]":
            sfreq = 1.0 / float(rest[0])
        elif kind == "Exception":
            raise NotImplementedError(kind)
        elif kind == "Comment":
            desc = rest[0]
        elif kind == "Mode":
            mode = rest[0]
        elif kind == "HPF[Hz]":
            try:
                hp = float(rest[0])
                has_hp = True
            except:
                pass
        elif kind == "LPF[Hz]":
            try:
                lp = float(rest[0])
                has_lp = True
            except:
                pass
        elif kind == "Date":
            try:
                dp = rest[0].split(" ")
                hm = dp[1].split(":")
                dd = dp[0].split("/")
                a = int(dd[0])
                b = int(dd[1])
                c = int(dd[2])
                if len(dd[0]) == 4:
                    meas = _utc_secs(a, b, c, int(hm[0]), int(hm[1]), 0)
                else:
                    y = 1900 + c if c >= 69 else 2000 + c
                    meas = _utc_secs(y, a, b, int(hm[0]), int(hm[1]), 0)
            except:
                print("RuntimeWarning: Extraction of measurement date failed. The date is being set to January 1st, 2000")
        elif kind == "Wave[nm]":
            fn_waves = [int(rest[0]), int(rest[1])]
        elif kind == "Wave Length":
            for ent in rest:
                p = ent.rfind("(")
                ch_wl[ent] = float(ent[p + 1:len(ent) - 1])
        elif kind == "Data":
            break
    all_names = _strip_end(lines[li + 1]).split(",")
    keep = List[int]()
    names = List[str]()
    types = List[str]()
    for ci in range(len(all_names)):
        nm = all_names[ci]
        if nm == "Time":
            continue
        if nm.startswith("Probe") and len(nm) > 5 and nm[5].isdigit():
            continue
        keep.append(ci)
        names.append(nm)
        types.append("fnirs_cw_amplitude" if nm.startswith("CH") else "stim")
    nn = 0
    for t in types:
        if t == "fnirs_cw_amplitude":
            nn += 1
    if nn % 2 != 0:
        raise RuntimeError("Could not parse file: odd number of fNIRS channels")
    nr = 0
    nc = 0
    nst = 1
    if mode == "3x3":
        nr = 3
        nc = 3
        nst = 2
    elif mode == "3x5":
        nr = 3
        nc = 5
    elif mode == "4x4":
        nr = 4
        nc = 4
    elif mode == "3x11":
        nr = 3
        nc = 11
    else:
        raise ValueError(f"Invalid value for the 'Hitachi mode' parameter: {mode}")
    pairs = _compute_pairs(nr, nc, nst)
    if nn != len(pairs) * 2:
        raise RuntimeError("Could not parse file: channel count mismatch")
    wl_of = List[float]()
    k = 0
    for i in range(len(names)):
        if types[i] == "fnirs_cw_amplitude":
            acc = ch_wl[names[i]]
            wl_of.append(acc)
            sidx = pairs[k // 2][0]
            didx = pairs[k // 2][1]
            d0 = abs(acc - float(fn_waves[0]))
            d1 = abs(acc - float(fn_waves[1]))
            nom = fn_waves[0] if d0 <= d1 else fn_waves[1]
            names[i] = f"S{sidx + 1}_D{didx + 1} {nom}"
            k += 1
        else:
            wl_of.append(0.0)
    info = Info(names, sfreq, types)
    for i in range(len(names)):
        if types[i] == "fnirs_cw_amplitude":
            info.locs[i] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, wl_of[i], 0.0, 0.0]
    if has_hp:
        info.highpass = hp
    if has_lp:
        info.lowpass = lp
    if desc != "":
        info.description = desc
    info.meas_date = meas
    rows = List[List[float]]()
    for i in range(li + 2, len(lines)):
        ln = _strip_end(lines[i]).replace(":", "")
        if ln == "":
            continue
        p = ln.split(",")
        r = List[float]()
        for ci in keep:
            r.append(float(p[ci]))
        rows.append(r)
    ns = len(rows)
    data = np.zeros((len(names), ns))
    for s in range(ns):
        for c in range(len(names)):
            data[c, s] = rows[s][c]
    raw = RawArray(data, info)
    raw._raw_kind = "RawHitachi"
    return raw
