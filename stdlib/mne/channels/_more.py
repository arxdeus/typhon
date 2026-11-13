# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# read_custom_montage (.loc/.locs/.eloc/.sfp/.csd/.elc/.txt/.csv/.tsv/.xyz),
# DigMontage.__add__, make_1020_channel_selections, combine_channels.

import numpy as np
from .._info import Info, DigMontage, create_info
from .._containers import RawArray, EpochsArray, Evoked
from . import _read_theta_phi, _topomap_coords, HEAD_SIZE_DEFAULT


def _median(v: List[float]) -> float:
    s = sorted(v)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _norm3(p: List[float]) -> float:
    return np.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])


def _dedupe(names: List[str], pos: List[List[float]]) -> Tuple[List[str], List[List[float]]]:
    on = List[str]()
    op = List[List[float]]()
    seen = Dict[str, int]()
    for i in range(len(names)):
        if names[i] in seen:
            op[seen[names[i]]] = pos[i]
        else:
            seen[names[i]] = len(on)
            on.append(names[i])
            op.append(pos[i])
    return on, op


def _scale_pos(pos: List[List[float]], head_size: Optional[float]) -> float:
    if head_size is None:
        return 1.0
    return head_size.__val__() / _median([_norm3(p) for p in pos])


def _read_lines(fname: str) -> List[str]:
    out = List[str]()
    with open(fname) as f:
        for line in f:
            out.append(line)
    return out


def _read_eeglab_loc(fname: str, head_size: float) -> DigMontage:
    names = List[str]()
    pos = List[List[float]]()
    for line in _read_lines(fname):
        parts = line.split()
        if len(parts) < 4:
            continue
        theta = float(parts[1])
        rad = float(parts[2])
        az = -np.deg2rad(theta)
        pol = np.pi * rad
        z = np.cos(pol)
        xy = np.sin(pol)
        x0 = xy * np.cos(az)
        y0 = xy * np.sin(az)
        names.append(parts[3])
        pos.append([-y0, x0, z])
    sc = head_size / _median([_norm3(p) for p in pos])
    pos = [[p[0] * sc, p[1] * sc, p[2] * sc] for p in pos]
    n2, p2 = _dedupe(names, pos)
    return DigMontage(n2, p2, "head", None, None, None)


def _read_sfp(fname: str, head_size: Optional[float]) -> DigMontage:
    names = List[str]()
    pos = List[List[float]]()
    for line in _read_lines(fname):
        parts = line.split()
        if len(parts) < 4:
            continue
        if parts[0] == "headshape":
            continue
        names.append(parts[0])
        pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
    sc = _scale_pos(pos, head_size)
    n2, p2 = _dedupe(names, pos)
    fid = ["FidNz", "FidT9", "FidT10"]
    fp = List[Optional[List[float]]]()
    keep_n = List[str]()
    keep_p = List[List[float]]()
    for f in fid:
        r: Optional[List[float]] = None
        for i in range(len(n2)):
            if n2[i] == f:
                r = [p2[i][0] * sc, p2[i][1] * sc, p2[i][2] * sc]
        fp.append(r)
    for i in range(len(n2)):
        if n2[i] not in fid:
            keep_n.append(n2[i])
            keep_p.append([p2[i][0] * sc, p2[i][1] * sc, p2[i][2] * sc])
    return DigMontage(keep_n, keep_p, "unknown", fp[0], fp[1], fp[2])


def _read_csd(fname: str, head_size: Optional[float]) -> DigMontage:
    names = List[str]()
    pos = List[List[float]]()
    for line in _read_lines(fname):
        s = line.strip()
        if s == "" or s.startswith("//"):
            continue
        parts = s.split()
        if len(parts) < 8:
            continue
        names.append(parts[0])
        pos.append([float(parts[4]), float(parts[5]), float(parts[6])])
    sc = _scale_pos(pos, head_size)
    pos = [[p[0] * sc, p[1] * sc, p[2] * sc] for p in pos]
    n2, p2 = _dedupe(names, pos)
    return DigMontage(n2, p2, "unknown", None, None, None)


def _read_elc(fname: str, head_size: Optional[float]) -> DigMontage:
    lines = _read_lines(fname)
    i = 0
    scale = 1.0
    found = False
    while i < len(lines):
        if "UnitPosition" in lines[i]:
            u = lines[i].split()[1]
            if u == "m":
                scale = 1.0
            elif u == "mm":
                scale = 1e-3
            else:
                raise KeyError(u)
            found = True
            i += 1
            break
        i += 1
    if not found:
        raise RuntimeError(f"Could not detect units in file {fname}")
    while i < len(lines):
        if lines[i].endswith("Positions\n"):
            i += 1
            break
        i += 1
    pos = List[List[float]]()
    new_style = False
    while i < len(lines):
        line = lines[i]
        i += 1
        if line.endswith("Labels\n"):
            break
        if ":" in line:
            new_style = True
            pos.append([float(x) for x in line.split(":")[1].split()])
        else:
            pos.append([float(x) for x in line.split()])
    names = List[str]()
    while i < len(lines):
        line = lines[i]
        i += 1
        if line.strip(" ").strip("\n") == "":
            break
        if new_style:
            for t in line.strip(" ").strip("\n").split():
                names.append(t)
        else:
            names.append(line.strip(" ").strip("\n"))
    pos = [[p[0] * scale, p[1] * scale, p[2] * scale] for p in pos]
    sc = _scale_pos(pos, head_size)
    pos = [[p[0] * sc, p[1] * sc, p[2] * sc] for p in pos]
    n2, p2 = _dedupe(names, pos)
    fid = ["Nz", "LPA", "RPA"]
    fp = List[Optional[List[float]]]()
    kn = List[str]()
    kp = List[List[float]]()
    for f in fid:
        r: Optional[List[float]] = None
        for k in range(len(n2)):
            if n2[k] == f:
                r = p2[k]
        fp.append(r)
    for k in range(len(n2)):
        if n2[k] not in fid:
            kn.append(n2[k])
            kp.append(p2[k])
    return DigMontage(kn, kp, "unknown", fp[0], fp[1], fp[2])


def _read_xyz(fname: str) -> DigMontage:
    ext = fname[fname.rfind("."):].lower()
    lines = _read_lines(fname)
    names = List[str]()
    pos = List[List[float]]()
    start = 0 if ext == ".xyz" else 1
    for k in range(start, len(lines)):
        line = lines[k].rstrip("\n").rstrip("\r")
        if line == "":
            continue
        row = line.split(",") if ext == ".csv" else line.split("\t")
        if ext == ".xyz":
            names.append(row[4].strip())
            pos.append([float(row[1]), float(row[2]), float(row[3])])
        else:
            names.append(row[0])
            pos.append([float(row[1]), float(row[2]), float(row[3])])
    n2, p2 = _dedupe(names, pos)
    return DigMontage(n2, p2, "unknown", None, None, None)


def read_custom_montage(fname, head_size = 0.095, coord_frame = None) -> DigMontage:
    """mne.channels.read_custom_montage (eeglab/sfp/csd/elc/txt/xyz formats)."""
    f = str(fname)
    dot = f.rfind(".")
    ext = f[dot:] if dot >= 0 else ""
    hs: Optional[float] = None
    if not isinstance(head_size, NoneType):
        hs = float(head_size)
    m = DigMontage(List[str](), List[List[float]](), "unknown", None, None, None)
    if ext in (".loc", ".locs", ".eloc"):
        if hs is None:
            raise ValueError(f"``head_size`` cannot be None for '{ext}'")
        m = _read_eeglab_loc(f, hs.__val__())
    elif ext == ".sfp":
        m = _read_sfp(f, hs)
    elif ext == ".csd":
        m = _read_csd(f, hs)
    elif ext == ".elc":
        m = _read_elc(f, hs)
    elif ext == ".txt":
        if hs is None:
            raise ValueError(f"``head_size`` cannot be None for '{ext}'")
        m = _read_theta_phi(f, hs.__val__(), ["Nz", "LPA", "RPA"], False)
    elif ext in (".csv", ".tsv", ".xyz"):
        m = _read_xyz(f)
    else:
        raise ValueError(f"Invalid value for the 'fname' parameter. Allowed values are ['.loc', '.locs', '.eloc', '.sfp', '.csd', '.elc', '.txt', '.csv', '.tsv', '.xyz'], but got '{ext}' instead.")
    if not isinstance(coord_frame, NoneType):
        m.coord_frame = str(coord_frame)
    return m


@extend
class DigMontage:
    def __add__(self, other: DigMontage):
        for c in other.ch_names:
            if c in self.ch_names:
                raise RuntimeError(f"Cannot add two DigMontage objects if they contain duplicated channel names. Duplicated channel(s) found: '{c}'.")
        out = self.copy()
        s_fid = self.nasion is not None or self.lpa is not None or self.rpa is not None
        o_fid = other.nasion is not None or other.lpa is not None or other.rpa is not None
        if s_fid and o_fid:
            if self.coord_frame != other.coord_frame:
                raise RuntimeError("Cannot add two DigMontage objects if fiducial locations are not in the same coordinate system.")
            if self.nasion != other.nasion:
                raise RuntimeError("Cannot add two DigMontage objects if fiducial locations do not match (nasion)")
            if self.lpa != other.lpa:
                raise RuntimeError("Cannot add two DigMontage objects if fiducial locations do not match (lpa)")
            if self.rpa != other.rpa:
                raise RuntimeError("Cannot add two DigMontage objects if fiducial locations do not match (rpa)")
        elif o_fid:
            out.nasion = other.nasion
            out.lpa = other.lpa
            out.rpa = other.rpa
            out.coord_frame = other.coord_frame
        for i in range(len(other.ch_names)):
            out.ch_names.append(other.ch_names[i])
            out.pos.append(list(other.pos[i]))
        return out


def make_1020_channel_selections(info: Info, midline: str = "z", return_ch_names: Literal[bool] = False):
    """Left/Midline/Right EEG channel groupings (sorted front to back when positions exist)."""
    idx = Dict[str, List[int]]()
    idx["Left"] = List[int]()
    idx["Midline"] = List[int]()
    idx["Right"] = List[int]()
    mid = midline.lower()
    eeg = List[int]()
    has_pos = True
    for i in range(info.nchan):
        if info.ch_types[i] == "eeg":
            eeg.append(i)
            if not (np.isfinite(info.locs[i][0]) and np.isfinite(info.locs[i][1])):
                has_pos = False
    if len(eeg) == 0:
        has_pos = False
    names = [info.ch_names[i] for i in range(info.nchan)]
    src = list(range(info.nchan))
    if has_pos:
        src = eeg
    for i in src:
        last = names[i][-1:].lower()
        if last in mid and last != "":
            idx["Midline"].append(i)
        elif last.isdigit():
            idx["Left" if int(last) % 2 == 1 else "Right"].append(i)
    if has_pos:
        xy = _topomap_coords(info, list(range(info.nchan)))
        for k in ("Left", "Midline", "Right"):
            v = idx[k]
            order = sorted(range(len(v)), key=lambda j: xy[v[j], 1])
            idx[k] = [v[j] for j in order]
    if return_ch_names:
        out = Dict[str, List[str]]()
        for k in ("Left", "Midline", "Right"):
            out[k] = [names[j] for j in idx[k]]
        return out
    return idx


def _reduce(a: np.ndarray[float, 2], method: str) -> np.ndarray[float, 1]:
    if method == "mean":
        return a.mean(axis=0)
    elif method == "median":
        return np.median(a, axis=0)
    elif method == "std":
        return a.std(axis=0)
    raise ValueError(f'"method" must be a callable, or one of "mean", "median", or "std"; got "{method}".')


def _combine_rows(info: Info, groups, data2d_fn, keep_stim: bool, drop_bad: bool, method):
    pass


def combine_channels(inst, groups, method = "mean", keep_stim: bool = False, drop_bad: bool = False):
    """Combine channel groups (mean/median/std or callable) into a new instance."""
    info = inst.info
    names = List[str]()
    types = List[str]()
    keep_idx = List[int]()
    if keep_stim:
        for i in range(info.nchan):
            if info.ch_types[i] == "stim":
                keep_idx.append(i)
                names.append(info.ch_names[i])
                types.append("stim")
    gnames = List[str]()
    gpicks = List[List[int]]()
    for k, v in groups.items():
        gnames.append(str(k))
        picks = List[int]()
        for q in v:
            qi = int(q)
            if qi < 0 or qi >= info.nchan:
                raise ValueError("Some channel indices are out of bounds.")
            picks.append(qi)
        t0 = info.ch_types[picks[0]]
        for q in picks:
            if info.ch_types[q] != t0:
                raise ValueError(f'Cannot combine sensors of different types; "{k}" contains types {t0}, {info.ch_types[q]}.')
        good = List[int]()
        for q in picks:
            if not (drop_bad and info.ch_names[q] in info.bads):
                good.append(q)
        gpicks.append(good)
        names.append(str(k))
        types.append(t0)
    ninfo = create_info(names, info.sfreq, types)
    if hasattr(inst, "events"):
        n_ep = inst._data.shape[0]
        nt = inst._data.shape[2]
        out = np.zeros((n_ep, len(names), nt))
        for e in range(n_ep):
            for r in range(len(keep_idx)):
                out[e, r] = inst._data[e, keep_idx[r]]
            for g in range(len(gnames)):
                sel = np.array(gpicks[g], dtype=int)
                out[e, len(keep_idx) + g] = _reduce(inst._data[e][sel, :], method)
        res = EpochsArray(out, ninfo, inst.events, float(inst.times[0]), inst.event_id)
        return res
    elif hasattr(inst, "nave"):
        nt = inst.data.shape[1]
        out = np.zeros((len(names), nt))
        for r in range(len(keep_idx)):
            out[r] = inst.data[keep_idx[r]]
        for g in range(len(gnames)):
            sel = np.array(gpicks[g], dtype=int)
            out[len(keep_idx) + g] = _reduce(inst.data[sel, :], method)
        from ..epochs import EvokedArray
        return EvokedArray(out, ninfo, float(inst.times[0]))
    else:
        nt = inst._data.shape[1]
        out = np.zeros((len(names), nt))
        for r in range(len(keep_idx)):
            out[r] = inst._data[keep_idx[r]]
        for g in range(len(gnames)):
            sel = np.array(gpicks[g], dtype=int)
            out[len(keep_idx) + g] = _reduce(inst._data[sel, :], method)
        return RawArray(out, ninfo, inst.first_samp)
