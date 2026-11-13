# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# BOXY (ISS Imagent) text reader (mne/io/boxy/boxy.py), parsed and
# non-parsed files. Numbers/columns are tokenized on whitespace.

import numpy as np
from ._bytes import _read_text
from .._info import Info
from .._containers import RawArray
from ..annotations import Annotations


def _isw(c: str) -> bool:
    return c.isalnum() or c == "_"


def _col_names(tok: str) -> List[str]:
    # re.findall(r"\w+\-\w+|\w+\-\d+|\w+")
    out = List[str]()
    i = 0
    n = len(tok)
    while i < n:
        if not _isw(tok[i]):
            i += 1
            continue
        j = i
        while j < n and _isw(tok[j]):
            j += 1
        if j + 1 < n and tok[j] == "-" and _isw(tok[j + 1]):
            k = j + 1
            while k < n and _isw(tok[k]):
                k += 1
            out.append(tok[i:k])
            i = k
        else:
            out.append(tok[i:j])
            i = j
    return out


def _first_tok(s: str) -> str:
    p = s.strip().split()
    return p[0] if len(p) > 0 else ""


def read_raw_boxy(fname, preload = False, verbose = None) -> RawArray:
    txt = _read_text(str(fname))
    lines = txt.split("\n")
    start_line = 1 << 60
    end_line = -1
    col_names = List[str]()
    filetype = ""
    has_mrk = False
    mrk_col = 0
    detect_num = 0
    source_num = 0
    sfreq = -1.0
    rows = List[List[float]]()
    mrk_data = List[float]()
    for ln in range(len(lines)):
        line = lines[ln]
        if ln >= start_line:
            if "#DATA ENDS" in line:
                end_line = ln
                break
            p = line.strip().split()
            if filetype == "non-parsed":
                if len(p) == len(col_names):
                    if has_mrk:
                        mrk_data.append(float(p[mrk_col]))
            else:
                if has_mrk:
                    mrk_data.append(float(p[mrk_col]))
            r = List[float]()
            for k in range(len(col_names)):
                r.append(float(p[k]) if k < len(p) else 0.0)
            rows.append(r)
        elif "BOXY.EXE:" in line:
            v = line.strip().split(" ")
            ver = v[len(v) - 1]
            if not (ver.endswith("0.40") or ver.endswith("0.84")):
                raise RuntimeError(f"MNE has not been tested with BOXY version ({ver})")
        elif "Detector Channels" in line:
            detect_num = int(_first_tok(line))
        elif "External MUX Channels" in line:
            source_num = int(_first_tok(line))
        elif "Update Rate (Hz)" in line or "Updata Rate (Hz)" in line:
            sfreq = float(_first_tok(line))
        elif "#DATA BEGINS" in line:
            start_line = ln + 3
        elif ln == start_line - 2:
            col_names = _col_names(line.split(" ")[0])
            filetype = "non-parsed" if "exmux" in col_names else "parsed"
            if "digaux" in col_names:
                mrk_col = col_names.index("digaux")
                has_mrk = True
    if sfreq < 0.0:
        raise RuntimeError("BOXY file has no update rate")
    names = List[str]()
    types = List[str]()
    cal = List[float]()
    for d in range(detect_num):
        for s in range(source_num):
            for ty in ("DC", "AC", "Ph"):
                names.append(f"S{s + 1}_D{d + 1} {ty}")
                if ty == "DC":
                    types.append("fnirs_cw_amplitude")
                    cal.append(1.0)
                elif ty == "AC":
                    types.append("fnirs_fd_ac_amplitude")
                    cal.append(1.0)
                else:
                    types.append("fnirs_fd_phase")
                    cal.append(3.141592653589793 / 180.0)
    info = Info(names, sfreq, types)
    for i in range(len(names)):
        info.cals[i] = cal[i]
    nrow = end_line - start_line
    ns = nrow
    if filetype == "non-parsed":
        ns = nrow // source_num
    data = np.zeros((len(names), ns))
    dets = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:detect_num]
    tys = ["DC", "AC", "Ph"]
    for t in range(ns):
        for di in range(detect_num):
            det = dets[di]
            for si in range(source_num):
                for ti in range(3):
                    ch = (di * source_num + si) * 3 + ti
                    if filetype == "parsed":
                        ci = col_names.index(f"{det}-{tys[ti]}{si + 1}")
                        val = rows[t][ci]
                    else:
                        ci = col_names.index(f"{det}-{tys[ti]}")
                        val = rows[t * source_num + si][ci]
                    data[ch, t] = val * cal[ch]
    raw = RawArray(data, info)
    raw._raw_kind = "RawBOXY"
    if has_mrk:
        prev = 0.0
        idx = List[int]()
        dur = List[float]()
        tmp = 0
        for i in range(len(mrk_data)):
            m = mrk_data[i]
            if m != 0.0 and m != prev:
                idx.append(i)
            if m != 0.0 and m == prev:
                tmp += 1
            if m == 0.0 and m != prev:
                dur.append(float(tmp + 1) / sfreq)
                tmp = 0
            prev = m
        on = List[float]()
        ds = List[str]()
        for i in idx:
            on.append(float(i) / sfreq)
            ds.append(str(mrk_data[i]))
        raw.set_annotations(Annotations(on, dur, ds), False)
    return raw
