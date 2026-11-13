# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# BESA .avr / .mul evoked readers (mne/io/besa/besa.py).

import numpy as np
from ._bytes import _read_text
from .._info import Info
from ..epochs import EvokedArray


def _lines(fname: str) -> List[str]:
    out = List[str]()
    for ln in _read_text(fname).split("\n"):
        out.append(ln)
    return out


def _header_dict(header: str) -> Dict[str, str]:
    parts = header.split()
    d = Dict[str, str]()
    i = 0
    while i + 1 < len(parts):
        d[parts[i].replace("=", "")] = parts[i + 1]
        i += 2
    return d


def _load_rows(lines: List[str], skip: int) -> List[List[float]]:
    rows = List[List[float]]()
    for k in range(skip, len(lines)):
        t = lines[k].strip()
        if t == "" or t.startswith("#"):
            continue
        r = List[float]()
        for tok in t.replace(",", " ").split():
            r.append(float(tok))
        rows.append(r)
    return rows


def _elp_sidecar(fname: str) -> Tuple[bool, List[str], List[str]]:
    """(exists, names, types) of the .elp sidecar next to fname."""
    dot = fname.rfind(".")
    elp = fname[:dot] + ".elp"
    names = List[str]()
    types = List[str]()
    try:
        lines = _lines(elp)
    except:
        return (False, names, types)
    lines = [l for l in lines if l.strip() != ""]
    has_types = len(lines[0].split()) > 3
    for l in lines:
        p = l.split()
        if has_types:
            types.append(p[0].lower())
            names.append(p[1])
        else:
            names.append(p[0])
            types.append("eeg")
    return (True, names, types)


def _types_for(ch_names: List[str], present: bool, enames: List[str], etypes: List[str]) -> List[str]:
    out = List[str]()
    for c in ch_names:
        t = "eeg"
        if present:
            found = False
            for j in range(len(enames)):
                if enames[j] == c:
                    t = etypes[j]
                    found = True
                    break
            if not found:
                raise KeyError(c)
        out.append(t)
    return out


def _info_with_types(names: List[str], sfreq: float, types: List[str]) -> Info:
    info = Info(names, sfreq, types)
    for k in range(len(types)):
        if types[k] != "eeg":
            info.coil_types[k] = 0
    return info


def _read_evoked_besa_avr(fname: str):
    lines = _lines(fname)
    header = lines[0].strip()
    new_style = "Nchan=" in header
    ch_names = List[str]()
    if new_style:
        ch_names = lines[1].strip().split()
    fields = _header_dict(header)
    rows = _load_rows(lines, 2 if new_style else 1)
    n = len(rows)
    present, enames, etypes = _elp_sidecar(fname)
    if new_style:
        if len(ch_names) != n:
            raise RuntimeError(f"Mismatch between the number of channel names defined in the .avr file ({len(ch_names)}) and the number of rows in the data matrix ({n}).")
    else:
        if present:
            ch_names = enames
            if len(ch_names) != n:
                raise RuntimeError(f"Mismatch between the number of channels defined in the .avr file ({n}) and .elp file ({len(ch_names)}).")
        else:
            for i in range(n):
                ch_names.append(f"CH{i + 1:02d}")
    types = _types_for(ch_names, present, enames, etypes)
    if "DI" not in fields:
        raise RuntimeError('No "DI" field present. Could not determine sampling frequency.')
    di = float(fields["DI"])
    tsb = float(fields["TSB"]) if "TSB" in fields else 0.0
    sb = float(fields["SB"]) if "SB" in fields else 1.0
    seg = fields["SegmentName"] if "SegmentName" in fields else ""
    npts = len(rows[0])
    data = np.zeros((n, npts))
    for i in range(n):
        for j in range(npts):
            data[i, j] = rows[i][j] / sb / 1e6
    # mne builds the info with ch_types="eeg" regardless of the sidecar
    return EvokedArray(data, _info_with_types(ch_names, 1000.0 / di, ["eeg" for _ in ch_names]), tsb / 1000.0, seg)


def _read_evoked_besa_mul(fname: str):
    lines = _lines(fname)
    header = lines[0].strip()
    ch_names = lines[1].strip().split()
    fields = _header_dict(header)
    rows = _load_rows(lines, 2)
    nt = len(rows)
    nc = len(rows[0])
    if len(ch_names) != nc:
        raise RuntimeError(f"Mismatch between the number of channel names defined in the .mul file ({len(ch_names)}) and the number of columns in the data matrix ({nc}).")
    present, enames, etypes = _elp_sidecar(fname)
    types = _types_for(ch_names, present, enames, etypes)
    if "SamplingInterval[ms]" not in fields:
        raise RuntimeError('No "SamplingInterval[ms]" field present. Could not determine sampling frequency.')
    di = float(fields["SamplingInterval[ms]"])
    bs = float(fields["BeginSweep[ms]"]) if "BeginSweep[ms]" in fields else 0.0
    bins = float(fields["Bins/uV"]) if "Bins/uV" in fields else 1.0
    seg = fields["SegmentName"] if "SegmentName" in fields else ""
    data = np.zeros((nc, nt))
    for i in range(nt):
        for j in range(nc):
            data[j, i] = rows[i][j] / bins / 1e6
    return EvokedArray(data, _info_with_types(ch_names, 1000.0 / di, types), bs / 1000.0, seg)


def read_evoked_besa(fname, verbose = None):
    f = str(fname)
    if f.endswith(".avr"):
        return _read_evoked_besa_avr(f)
    if f.endswith(".mul"):
        return _read_evoked_besa_mul(f)
    raise ValueError("Filename must end in either .avr or .mul")
