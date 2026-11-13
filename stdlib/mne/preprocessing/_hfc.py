# Ported from mne 1.13.2 (BSD-3-Clause)
#
# compute_proj_hfc: homogeneous / harmonic field correction projectors, and
# read/write_fine_calibration (MaxFilter .dat text format).

import math
import numpy as np
from .._info import Info, Projection, _picks_to_idx
from ._maxwell import _prep_mf_coils, _basis


def compute_proj_hfc(info: Info, order: int = 1, picks = "meg", exclude = "bads", *,
                     accuracy: str = "accurate", verbose = None):
    """Projectors for homogeneous/harmonic field correction (mne 1.13.2)."""
    pk = List[int]()
    excl = List[str]()
    if isinstance(exclude, str):
        if exclude == "bads":
            excl = list(info.bads)
    else:
        for e in exclude:
            excl.append(str(e))
    if isinstance(picks, str) and picks == "meg":
        for i in range(info.nchan):
            if (info.ch_types[i] == "mag" or info.ch_types[i] == "grad") and info.ch_names[i] not in excl:
                pk.append(i)
    else:
        pk = _picks_to_idx(info, picks, "meg", exclude)
    n_chs = len(pk)
    for i in pk:
        if not (info.ch_types[i] == "mag" or info.ch_types[i] == "grad"):
            raise ValueError(f"Only {n_chs - 1}/{n_chs} picks could be interpreted as MEG channels.")
    coils = _prep_mf_coils(info, pk)
    S = _basis(coils, np.zeros(3), 0, order)
    bad = List[str]()
    for i in range(S.shape[0]):
        ok = True
        for j in range(S.shape[1]):
            if not (S[i, j] == S[i, j]) or abs(S[i, j]) == float("inf"):
                ok = False
        if not ok:
            bad.append(info.ch_names[pk[i]])
    if len(bad) > 0:
        raise ValueError("The following channel(s) generate non-finite projectors:\n    " + str(bad) + "\nPlease exclude from picks!")
    ncol = S.shape[1]
    for j in range(ncol):
        s = 0.0
        for i in range(S.shape[0]):
            s += S[i, j] * S[i, j]
        s = math.sqrt(s)
        for i in range(S.shape[0]):
            S[i, j] /= s
    names = [info.ch_names[i] for i in pk]
    projs = List[Projection]()
    j = 0
    for L in range(1, order + 1):
        for m in range(-L, L + 1):
            vec = np.empty((1, S.shape[0]))
            for i in range(S.shape[0]):
                vec[0, i] = S[i, j]
            projs.append(Projection(f"HFC: l={L} m={m}", 1, False, list(names), vec))
            j += 1
    return projs


def read_fine_calibration(fname):
    """Read a fine calibration .dat file: (ch_names, locs (n, 12), imb_cals) (mne 1.13.2)."""
    fn = str(fname)
    if not fn.endswith(".dat"):
        raise OSError("Fine- calibration filename should end with .dat")
    names = List[str]()
    locs = List[List[float]]()
    imb = List[List[float]]()
    with open(fn) as fid:
        for line in fid:
            if len(line) == 0 or line[0] == "#" or line[0] == "\n":
                continue
            vals = line.strip().split()
            if len(vals) != 14 and len(vals) != 16:
                raise RuntimeError("Error parsing fine calibration file, should have 14 or 16 entries per line but found " + str(len(vals)) + " on line:\n" + line)
            nm = vals[0]
            if len(nm) == 3 or len(nm) == 4:
                alld = True
                for ch in nm:
                    if not ch.isdigit():
                        alld = False
                if alld:
                    num = str(int(nm))
                    pad = "0" * (4 - len(num))
                    nm = "MEG" + pad + num
            names.append(nm)
            locs.append([float(vals[k]) for k in range(1, 13)])
            imb.append([float(vals[k]) for k in range(13, len(vals))])
    arr = np.empty((len(locs), 12))
    for i in range(len(locs)):
        for k in range(12):
            arr[i, k] = locs[i][k]
    return names, arr, imb


def write_fine_calibration(fname, ch_names, locs, imb_cals):
    """Write fine calibration information to a .dat file (mne 1.13.2)."""
    fn = str(fname)
    if not fn.endswith(".dat"):
        raise OSError("Fine- calibration filename should end with .dat")
    lc = np.asarray(locs, dtype = float)
    out = ""
    for i in range(len(ch_names)):
        nm = str(ch_names[i])
        if nm.startswith("MEG") and len(nm) == 7 and nm[3:].isdigit():
            nm = str(int(nm[3:]))
        vals = List[float]()
        for k in range(12):
            vals.append(float(lc[i, k]))
        for v in imb_cals[i]:
            vals.append(float(v))
        parts = List[str]()
        for v in vals:
            r = round(v, 6)
            parts.append(f"{r:0.6f}")
        out += nm + " " + " ".join(parts) + "\n"
    with open(fn, "w") as f:
        f.write(out)
