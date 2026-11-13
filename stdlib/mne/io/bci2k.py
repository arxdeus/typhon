# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# BCI2000 .dat reader (mne/io/bci2k/bci2k.py). preload is always applied.

import numpy as np
from ._bytes import _read_file, _ascii, _arr_le, _u8
from .._info import Info
from .._containers import RawArray


def _parse_sampling_rate(val: str) -> float:
    t = val.strip()
    if t.lower().endswith("hz"):
        t = t[:len(t) - 2].strip()
    return float(t)


def _line_end(b: bytes, pos: int) -> int:
    n = len(b)
    while pos < n and _u8(b, pos) != 10:
        pos += 1
    return pos


def read_raw_bci2k(input_fname, preload = False, verbose = None) -> RawArray:
    fn = str(input_fname)
    b = _read_file(fn)
    e = _line_end(b, 0)
    first = _ascii(b, 0, e).strip()
    header = Dict[str, str]()
    toks = first.replace("=", " = ").split()
    i = 0
    while i + 2 < len(toks):
        if toks[i + 1] == "=":
            header[toks[i]] = toks[i + 2]
            i += 3
        else:
            i += 1
    for k in ("HeaderLen", "SourceCh", "StatevectorLen"):
        if k not in header:
            raise ValueError(f"BCI2000 header is missing required key(s): {k}")
    header_len = int(header["HeaderLen"])
    nch = int(header["SourceCh"])
    sv_len = int(header["StatevectorLen"])
    fmt = header["DataFormat"] if "DataFormat" in header else "int16"
    section = ""
    sfreq_s = ""
    st_names = List[str]()
    st_len = List[int]()
    st_byte = List[int]()
    st_bit = List[int]()
    pos = e + 1
    while pos < header_len:
        e2 = _line_end(b, pos)
        line = _ascii(b, pos, e2 - pos).strip()
        pos = e2 + 1
        if line == "":
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line
            continue
        if "Parameter Definition" in section:
            if "=" in line:
                j = line.find("=")
                left = line[:j].strip().split()
                right = line[j + 1:].strip().split()
                if len(left) > 0 and len(right) > 0 and left[-1] == "SamplingRate":
                    sfreq_s = right[0]
            continue
        if "State Vector Definition" in section:
            p = line.split()
            if len(p) >= 5:
                try:
                    ln = int(p[1])
                    bp = int(p[3])
                    bt = int(p[4])
                except:
                    continue
                st_names.append(p[0])
                st_len.append(ln)
                st_byte.append(bp)
                st_bit.append(bt)
    if sfreq_s == "":
        raise ValueError("Could not find 'SamplingRate' in the BCI2000 Parameter Definition section.")
    sfreq = _parse_sampling_rate(sfreq_s)
    bps = 2
    code = "i2"
    if fmt == "int16":
        bps = 2
    elif fmt == "int32":
        bps = 4
        code = "i4"
    elif fmt == "float32":
        bps = 4
        code = "f4"
    else:
        raise ValueError(f"Unsupported DataFormat: {fmt}")
    bpf = nch * bps + sv_len
    ns = (len(b) - header_len) // bpf
    names = List[str]()
    types = List[str]()
    for k in range(nch):
        names.append(f"EEG{k + 1}")
        types.append("eeg")
    stim_idx = -1
    for k in range(len(st_names)):
        if st_names[k] == "StimulusCode":
            stim_idx = k
    if stim_idx >= 0 and sv_len > 0:
        names.append("STI 014")
        types.append("stim")
    info = Info(names, sfreq, types)
    data = np.zeros((len(names), ns))
    for s in range(ns):
        base = header_len + s * bpf
        v = _arr_le(b, base, nch, code)
        for c in range(nch):
            data[c, s] = v[c]
    if stim_idx >= 0 and sv_len > 0:
        for s in range(ns):
            sb = header_len + s * bpf + nch * bps
            val = 0
            for bit in range(st_len[stim_idx]):
                off = st_bit[stim_idx] + bit
                tb = st_byte[stim_idx] + off // 8
                if tb < 0 or tb >= sv_len:
                    continue
                bv = (_u8(b, sb + tb) >> (off % 8)) & 1
                val |= bv << bit
            data[nch, s] = float(val)
        info.units[nch] = -1
        info.coil_types[nch] = 0
    raw = RawArray(data, info)
    raw._raw_kind = "RawBCI2k"
    return raw
