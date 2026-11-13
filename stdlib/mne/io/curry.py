# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# Curry 7/8/9 reader (mne/io/curry/curry.py). The curryreader dependency is
# replaced by a minimal header/label parser. Narrower: no sensor positions /
# landmarks / HPI, no event (.cef/.ceo) annotations, raw recordings only.

import numpy as np
from ._bytes import _read_file, _read_text, _f32_le
from ._fmtutil import _utc_secs
from .._info import Info
from .._containers import RawArray

_DATA = [".cdt", ".dat"]
_HDR = [".cdt.dpa", ".cdt.dpo", ".dap"]
_LBL = [".cdt.dpa", ".cdt.dpo", ".rs3"]


def _exists(p: str) -> bool:
    try:
        f = open(p, "rb")
        f.close()
        return True
    except:
        return False


def _data_name(fn: str) -> str:
    for s in _DATA:
        if fn.endswith(s):
            return fn
    for s in _DATA:
        if fn.endswith(".cdt.dpa") or fn.endswith(".cdt.dpo"):
            base = fn[:fn.rfind(".cdt") ]
            if _exists(base + s):
                return base + s
    j = fn.rfind(".")
    base = fn[:j] if j >= 0 else fn
    for s in _DATA:
        if _exists(base + s):
            return base + s
    raise FileNotFoundError(f"no curry data file found (.dat or .cdt), checked {fn}")


def _with_suffix(dn: str, suffs: List[str], what: str) -> str:
    j = dn.rfind(".")
    base = dn[:j] if j >= 0 else dn
    for s in suffs:
        if _exists(base + s):
            return base + s
    raise FileNotFoundError(f"no corresponding {what} file found {suffs}")


def _val(txt: str, key: str) -> str:
    # regex `key\s*=\s*(.+)` -> last whitespace token of the first match
    p = 0
    while True:
        j = txt.find(key, p)
        if j < 0:
            raise ValueError(f"{key} not found in header")
        k = j + len(key)
        while k < len(txt) and txt[k].isspace():
            k += 1
        if k < len(txt) and txt[k] == "=":
            k += 1
            e = txt.find("\n", k)
            if e < 0:
                e = len(txt)
            toks = txt[k:e].split()
            return toks[len(toks) - 1] if len(toks) > 0 else ""
        p = j + len(key)


def _list(txt: str, name: str) -> List[str]:
    out = List[str]()
    parts = txt.split("START_LIST")
    for i in range(1, len(parts)):
        pw = parts[i - 1].split()
        if len(pw) > 0 and name in pw[len(pw) - 1]:
            lines = parts[i].split("\n")
            for k in range(1, len(lines)):
                if name in lines[k]:
                    break
                out.append(lines[k].strip())
    return out


def read_raw_curry(fname, preload = False, on_bad_hpi_match: str = "warn", verbose = None) -> RawArray:
    dn = _data_name(str(fname))
    hn = _with_suffix(dn, _HDR, "header")
    hdr = _read_text(hn)
    stime = float(_val(hdr, "SampleTimeUsec"))
    sfreq = float(_val(hdr, "SampleFreqHz"))
    if stime == 0.0:
        raise ValueError("Header file indicates a sampling interval of 0µs.")
    if abs(1e6 / stime - sfreq) > 1e-8 + 1e-5 * abs(sfreq):
        print("RuntimeWarning: Sample distance and sample frequency in header file do not match! sfreq will be derived from sample distance.")
    n_ch = int(_val(hdr, "NumChannels"))
    n_hdr_samp = int(_val(hdr, "NumSamples"))
    byteorder = _val(hdr, "DataByteOrder")
    is_ascii = byteorder == "ASCII"
    # types / units per device group
    ch_types = List[str]()
    units = List[str]()
    groups = hdr.split("DEVICE_PARAMETERS")
    gi = 1
    while gi < len(groups):
        g = " ".join(groups[gi].split())
        gid = g.split()[0]
        unit = g.split("DataUnit = ")[1].split()[0]
        n_g = int(g.split("NumChanThisGroup = ")[1].split()[0])
        t = "mag" if "MAG" in gid else ("misc" if "OTHER" in gid else "eeg")
        for _ in range(n_g):
            ch_types.append(t)
            units.append(unit)
        gi += 2
    ln = _with_suffix(dn, _LBL, "labels")
    lbl = _read_text(ln)
    names = _list(lbl, "LABELS")
    if not (len(ch_types) == len(units) == len(names) == n_ch):
        chfile = List[int]()
        for s in _list(hdr, "CHAN_IN_FILE"):
            chfile.append(int(s))
        order = List[int]()
        for i in range(len(chfile)):
            order.append(i)
        order.sort(key = lambda i: chfile[i])
        nn = List[str]()
        nt = List[str]()
        nu = List[str]()
        for i in order:
            if chfile[i] != 0:
                nn.append(names[i])
                nt.append(ch_types[i])
                nu.append(units[i])
        names = nn
        ch_types = nt
        units = nu
    if not (len(ch_types) == len(units) == len(names) == n_ch):
        raise RuntimeError("Curry channel count mismatch")
    info = Info(names, sfreq, ch_types)
    cals = List[float]()
    for i in range(n_ch):
        c = 1.0
        if units[i] == "fT":
            c = 1.0 / 1e15
        elif units[i] == "uV":
            c = 1.0 / 1e6
        cals.append(c)
        info.cals[i] = 1.0
    ys = int(_val(hdr, "StartYear"))
    mo = int(_val(hdr, "StartMonth"))
    da = int(_val(hdr, "StartDay"))
    hh = int(_val(hdr, "StartHour"))
    mi = int(_val(hdr, "StartMin"))
    se = int(_val(hdr, "StartSec"))
    ms = int(_val(hdr, "StartMillisec"))
    if ys >= 1 and mo >= 1 and mo <= 12 and da >= 1 and da <= 31:
        info.meas_date = _utc_secs(ys, mo, da, hh, mi, se) + float(ms) / 1000.0
    n_samp = n_hdr_samp
    rows = List[List[float]]()
    if is_ascii:
        for line in _read_text(dn).split("\n"):
            s = line.strip()
            if s == "":
                continue
            r = List[float]()
            for t in s.split():
                r.append(float(t))
            rows.append(r)
        n_samp = len(rows)
    else:
        b = _read_file(dn)
        n_samp = len(b) // (4 * n_ch)
    data = np.zeros((n_ch, n_samp))
    if is_ascii:
        for s in range(n_samp):
            for c in range(n_ch):
                data[c, s] = rows[s][c] * cals[c]
    else:
        for s in range(n_samp):
            for c in range(n_ch):
                data[c, s] = float(_f32_le(b, (s * n_ch + c) * 4)) * cals[c]
    raw = RawArray(data, info)
    raw._raw_kind = "RawCurry"
    return raw
