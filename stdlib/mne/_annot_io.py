# Ported from mne 1.13.2 (BSD-3-Clause)
#
# read_annotations for .txt / .csv (fif is handled by the Raw reader).

import numpy as np
from .annotations import Annotations


def _days_from_civil(y: int, m: int, d: int) -> int:
    if m <= 2:
        y -= 1
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    mp = (m + 9) % 12
    doy = (153 * mp + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def _parse_iso(s: str) -> Optional[float]:
    """POSIX seconds (UTC) of 'YYYY-MM-DD[ T]HH:MM:SS[.ffffff]' or None."""
    t = s.strip()
    if len(t) < 19 or t[4] != "-" or t[7] != "-" or (t[10] != " " and t[10] != "T"):
        return None
    if t[13] != ":" or t[16] != ":":
        return None
    for i in (0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18):
        if not t[i].isdigit():
            return None
    frac = 0.0
    if len(t) > 19:
        if t[19] != ".":
            return None
        digits = t[20:]
        if len(digits) == 0 or not digits.isdigit():
            return None
        frac = float(int(digits)) / (10.0 ** len(digits))
    days = _days_from_civil(int(t[0:4]), int(t[5:7]), int(t[8:10]))
    sec = int(t[11:13]) * 3600 + int(t[14:16]) * 60 + int(t[17:19])
    return float(days * 86400 + sec) + frac


def _split_csv_line(line: str) -> List[str]:
    return [c.strip() for c in line.rstrip("\r\n").split(",")]


def _read_txt(fname: str) -> Annotations:
    f = open(fname)
    lines = [l for l in f]
    f.close()
    orig: Optional[float] = None
    columns = List[str]()
    n_header = 0
    for l in lines:
        if not l.startswith("#"):
            break
        n_header += 1
        if l.startswith("# orig_time :"):
            v = _parse_iso(l[13:].strip())
            if v is not None and orig is None:
                orig = v
        elif l.startswith("# onset, duration, description") and len(columns) == 0:
            columns = [c.strip() for c in l[2:].split(",")]
    rows = List[List[str]]()
    for i in range(n_header, len(lines)):
        if len(lines[i].strip()) == 0:
            continue
        rows.append(_split_csv_line(lines[i]))
    if len(rows) == 0:
        return Annotations(List[float](), List[float](), List[str](), orig)
    ncol = len(rows[0])
    if len(columns) == 0:
        if ncol != 3 and ncol != 4:
            raise ValueError(f"Invalid value for the 'text header' parameter. Allowed values are 3 and 4, but got {ncol} instead.")
        columns = ["onset", "duration", "description"]
        if ncol == 4:
            columns.append("ch_names")
    io = -1
    idu = -1
    ide = -1
    for k in range(len(columns)):
        if columns[k] == "onset":
            io = k
        elif columns[k] == "duration":
            idu = k
        elif columns[k] == "description":
            ide = k
    if io < 0 or idu < 0 or ide < 0:
        raise ValueError("Column(s) not found in header. Please check the file format.")
    if ncol != len(columns):
        raise ValueError(f"Invalid value for the 'text header len' parameter. Allowed values are {len(columns)}, but got {ncol} instead.")
    onset = List[float]()
    dur = List[float]()
    desc = List[str]()
    for r in rows:
        onset.append(float(r[io]))
        dur.append(float(r[idu]))
        desc.append(r[ide])
    return Annotations(onset, dur, desc, orig)


def _read_csv(fname: str) -> Annotations:
    f = open(fname)
    lines = [l for l in f if len(l.strip()) > 0]
    f.close()
    head = _split_csv_line(lines[0])
    io = -1
    idu = -1
    ide = -1
    for k in range(len(head)):
        if head[k] == "onset":
            io = k
        elif head[k] == "duration":
            idu = k
        elif head[k] == "description":
            ide = k
    if io < 0 or idu < 0 or ide < 0:
        raise KeyError("onset")
    onset = List[float]()
    dur = List[float]()
    desc = List[str]()
    t0 = _parse_iso(_split_csv_line(lines[1])[io])
    if t0 is None:
        raise ValueError("It looks like you have provided annotation onsets as floats. These will be interpreted as MILLISECONDS. If that is not what you want, save your CSV as a TXT file; the TXT reader accepts onsets in seconds.")
    for i in range(1, len(lines)):
        r = _split_csv_line(lines[i])
        t = _parse_iso(r[io])
        if t is None:
            raise ValueError("invalid onset timestamp")
        onset.append(t - t0)
        dur.append(float(r[idu]))
        desc.append(r[ide])
    return Annotations(onset, dur, desc, t0)


def read_annotations(fname, sfreq = "ignored", uint16_codec = None, encoding = "utf8", verbose = None):
    """mne.read_annotations for .txt and .csv files."""
    p = str(fname)
    if p.endswith(".txt"):
        return _read_txt(p)
    if p.endswith(".csv"):
        return _read_csv(p)
    raise ValueError(f"Unknown annotation file format {p}. Supported .txt and .csv here")
