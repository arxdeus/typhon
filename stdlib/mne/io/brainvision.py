# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# BrainVision reader (mne/io/brainvision/brainvision.py): .vhdr header,
# .vmrk markers and binary .eeg data (INT_16 / INT_32 / IEEE_FLOAT_32,
# MULTIPLEXED or VECTORIZED). Amplifier filter settings in [Comment] and
# [Coordinates] are not parsed (highpass=0, lowpass=sfreq/2).

import numpy as np
from ._bytes import _read_file, _read_text, _arr_le
from .._info import Info
from .._containers import RawArray
from ..annotations import Annotations


def _unit_scale(u: str) -> float:
    if u == "V" or u == "C" or u == "°C" or u == "n/a":
        return 1.0
    if u == "µV" or u == "uV":
        return 1e-6
    if u == "mV":
        return 1e-3
    if u == "nV":
        return 1e-9
    return 1.0


def _ini_sections(txt: str) -> Dict[str, List[Tuple[str, str]]]:
    out = Dict[str, List[Tuple[str, str]]]()
    cur = ""
    for line0 in txt.split("\n"):
        line = line0.strip()
        if len(line) == 0 or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            cur = line[1:-1]
            out[cur] = List[Tuple[str, str]]()
            continue
        j = line.find("=")
        if j > 0 and cur != "":
            out[cur].append((line[:j].strip(), line[j + 1:].strip()))
    return out


def _get(sec: Dict[str, List[Tuple[str, str]]], name: str, key: str) -> str:
    for s, items in sec.items():
        if s.lower() == name.lower():
            for k, v in items:
                if k.lower() == key.lower():
                    return v
    return ""


def _dirname(p: str) -> str:
    j = p.rfind("/")
    return p[:j + 1] if j >= 0 else ""


def _read_vmrk(fname: str, sfreq: float, ignore_marker_types: bool) -> Annotations:
    txt = _read_text(fname)
    m = txt.find("[Marker Infos]")
    on = List[float]()
    du = List[float]()
    ty = List[str]()
    de = List[str]()
    if m >= 0:
        for line0 in txt[m + 14:].split("\n"):
            line = line0.rstrip("\r")
            if line.startswith("[") and line.strip().endswith("]"):
                break
            if not line.startswith("Mk"):
                continue
            j = line.find("=")
            if j < 0 or not line[2:j].isdigit():
                continue
            parts = line[j + 1:].split(",")
            mtype = parts[0].replace("\\1", ",")
            mdesc = parts[1].replace("\\1", ",")
            dur = int(parts[3]) if parts[3].isdigit() else 0
            on.append(float(int(parts[2]) - 1))
            du.append(float(dur))
            ty.append(mtype)
            de.append(mdesc)
    if len(ty) > 0 and ty[0] == "New Segment":
        on = on[1:]
        du = du[1:]
        ty = ty[1:]
        de = de[1:]
    on = [o / sfreq for o in on]
    du = [d / sfreq for d in du]
    if not ignore_marker_types:
        de = [f"{ty[i]}/{de[i]}" for i in range(len(de))]
    return Annotations(on, du, de)


def read_raw_brainvision(vhdr_fname, eog = ("HEOGL", "HEOGR", "VEOGb"), misc = "auto",
                         scale: float = 1.0, ignore_marker_types: bool = False,
                         preload = False, verbose = None) -> RawArray:
    hdr = str(vhdr_fname)
    if not hdr.endswith(".vhdr"):
        raise OSError(f"The header file must be given to read the data, not a file with extension '.{hdr.split('.')[-1]}'.")
    txt = _read_text(hdr)
    nl = txt.find("\n")
    first = txt[:nl].strip() if nl >= 0 else txt
    if not first.startswith("Brain Vision") and not first.startswith("BrainVision"):
        raise ValueError(f"Currently only support header versions: Brain Vision Data Exchange Header File Version 1.0/2.0, got {first}")
    body = txt[nl + 1:]
    c = body.find("[Comment]")
    if c >= 0:
        body = body[:c]
    cfg = _ini_sections(body)
    ci = "Common Infos"
    sfreq = 1e6 / float(_get(cfg, ci, "SamplingInterval"))
    order = _get(cfg, ci, "DataOrientation")
    if order != "MULTIPLEXED" and order != "VECTORIZED":
        raise NotImplementedError(f"Data Orientation {order} is not supported")
    if _get(cfg, ci, "DataFormat") != "BINARY":
        raise NotImplementedError("Typhon mne: ASCII BrainVision data is not supported")
    bfmt = _get(cfg, "Binary Infos", "BinaryFormat")
    code = ""
    bs = 0
    if bfmt == "INT_16":
        code = "i2"
        bs = 2
    elif bfmt == "INT_32":
        code = "i4"
        bs = 4
    elif bfmt == "IEEE_FLOAT_32":
        code = "f4"
        bs = 4
    else:
        raise NotImplementedError(f"Datatype {bfmt} is not supported")
    path = _dirname(hdr)
    data_fname = path + _get(cfg, ci, "DataFile")
    mrk = _get(cfg, ci, "MarkerFile")
    nchan = int(_get(cfg, ci, "NumberOfChannels"))
    names = ["" for _ in range(nchan)]
    cals = [np.nan for _ in range(nchan)]
    ranges = [1.0 for _ in range(nchan)]
    misc_auto = List[str]()
    for s, items in cfg.items():
        if s.lower() != "channel infos":
            continue
        for k, v in items:
            n = int(k[2:]) - 1
            if n >= nchan:
                continue
            props = v.split(",")
            if len(props) < 4:
                props.append("µV")
            elif props[3] == "":
                props[3] = "µV"
            name = props[0].replace("\\1", ",")
            res = props[2]
            unit = props[3].replace("\xc2", "")
            if res == "":
                res = "0.000001" if unit == "" else "1.0"
            names[n] = name
            cals[n] = float(res)
            ranges[n] = _unit_scale(unit) * scale
            if unit not in ("V", "mV", "µV", "uV", "nV"):
                misc_auto.append(name)
    eogl = [str(x) for x in eog]
    miscl = misc_auto if isinstance(misc, str) else [str(x) for x in misc]
    types = List[str]()
    for i in range(nchan):
        if names[i] in eogl:
            types.append("eog")
        elif names[i] in miscl:
            types.append("misc")
        elif names[i] == "STI 014":
            types.append("stim")
        else:
            types.append("eeg")
    info = Info(names, sfreq, types)
    for i in range(nchan):
        info.cals[i] = cals[i]
        info.ranges[i] = ranges[i]
        if types[i] != "eeg":
            info.coil_types[i] = 0
        if names[i] in miscl and names[i] in misc_auto:
            info.units[i] = -1
    b = _read_file(data_fname)
    n_samples = len(b) // (bs * nchan)
    flat = _arr_le(b, 0, n_samples * nchan, code)
    data = np.zeros((nchan, n_samples))
    if order == "MULTIPLEXED":
        m = flat.reshape((n_samples, nchan))
        for ch in range(nchan):
            data[ch, :] = m[:, ch] * (cals[ch] * ranges[ch])
    else:
        m = flat.reshape((nchan, n_samples))
        for ch in range(nchan):
            data[ch, :] = m[ch, :] * (cals[ch] * ranges[ch])
    raw = RawArray(data, info)
    raw._raw_kind = "RawBrainVision"
    if mrk != "":
        raw.set_annotations(_read_vmrk(path + mrk, sfreq, ignore_marker_types), False)
    return raw
