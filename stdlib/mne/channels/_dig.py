# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# mne.channels digitizer readers (dat, hpts, localite, polhemus isotrak/fastscan,
# captrak, egi), transform_to_head, compute_native_head_t, rename_channels,
# unify_bad_channels, get_builtin_montages, get_builtin_ch_adjacencies.
# DigMontage here carries no headshape/HPI points: they are parsed and dropped.

import numpy as np
from .._info import Info, DigMontage, _apply, _ras_to_neuromag


class Transform:
    """mne.transforms.Transform: from/to coordinate frames and a 4x4 matrix."""
    fro: str
    to: str
    trans: np.ndarray[float, 2]

    def __init__(self, fro: str, to: str, trans: np.ndarray[float, 2]):
        self.fro = fro
        self.to = to
        self.trans = trans

    def __getitem__(self, key: Literal[str]):
        if key == "from":
            return self.fro
        elif key == "to":
            return self.to
        elif key == "trans":
            return self.trans
        else:
            compile_error("unsupported Transform key")

    def __repr__(self) -> str:
        return f"<Transform | {self.fro}->{self.to}>"


def _read_text(fname: str) -> List[str]:
    out = List[str]()
    with open(fname) as f:
        for line in f:
            out.append(line)
    return out


def _unit_scale(unit: str) -> float:
    if unit == "m":
        return 1.0
    if unit == "cm":
        return 1e-2
    if unit == "mm":
        return 1e-3
    raise ValueError(f"Invalid value for the 'unit' parameter. Allowed values are ['m', 'cm', 'mm'], but got '{unit}' instead.")


def _mk(names: List[str], pos: List[List[float]], nas: Optional[List[float]],
        lpa: Optional[List[float]], rpa: Optional[List[float]], frame: str = "unknown") -> DigMontage:
    return DigMontage(names, pos, frame, nas, lpa, rpa)


def _dupes(names: List[str]):
    seen = Set[str]()
    for n in names:
        if n in seen:
            raise ValueError(f"Duplicate channel names found: {n}")
        seen.add(n)


def read_dig_dat(fname) -> DigMontage:
    """mne.channels.read_dig_dat (Compumedics .dat)."""
    names = List[str]()
    pos = List[List[float]]()
    nas: Optional[List[float]] = None
    lpa: Optional[List[float]] = None
    rpa: Optional[List[float]] = None
    lines = _read_text(str(fname))
    for i in range(len(lines)):
        items = lines[i].split()
        if len(items) == 0:
            continue
        if len(items) != 5:
            raise ValueError(f"Error reading {fname}, line {i} has unexpected number of entries:\n{lines[i].rstrip()}")
        num = items[1]
        if num == "67":
            continue
        p = [float(items[2]), float(items[3]), float(items[4])]
        if num == "78":
            nas = p
        elif num == "76":
            lpa = p
        elif num == "82":
            rpa = p
        else:
            names.append(items[0])
            pos.append(p)
    _dupes(names)
    return _mk(names, pos, nas, lpa, rpa)


def read_dig_hpts(fname, unit: str = "mm") -> DigMontage:
    """mne.channels.read_dig_hpts (eeg/cardinal points; hpi/extra are dropped)."""
    sc = _unit_scale(unit)
    names = List[str]()
    pos = List[List[float]]()
    nas: Optional[List[float]] = None
    lpa: Optional[List[float]] = None
    rpa: Optional[List[float]] = None
    for line in _read_text(str(fname)):
        s = line.strip()
        if len(s) == 0 or s.startswith("#"):
            continue
        it = s.split()
        kind = it[0].lower()
        p = [float(it[2]) * sc, float(it[3]) * sc, float(it[4]) * sc]
        if kind == "cardinal":
            if it[1] == "1":
                lpa = p
            elif it[1] == "2":
                nas = p
            elif it[1] == "3":
                rpa = p
        elif kind == "eeg":
            names.append(it[1])
            pos.append(p)
    return _mk(names, pos, nas, lpa, rpa)


def read_dig_localite(fname, nasion = None, lpa = None, rpa = None) -> DigMontage:
    """mne.channels.read_dig_localite."""
    names = List[str]()
    pos = List[List[float]]()
    first = True
    for line in _read_text(str(fname)):
        if first:
            first = False
            continue
        it = line.rstrip("\r\n").split(",")
        if len(it) != 5:
            raise ValueError("too many values to unpack")
        names.append(it[1])
        pos.append([float(it[2]) / 1000.0, float(it[3]) / 1000.0, float(it[4]) / 1000.0])
    fid = List[Optional[List[float]]]()
    for nm in (nasion, lpa, rpa):
        v: Optional[List[float]] = None
        if not isinstance(nm, NoneType):
            k = names.index(str(nm))
            v = pos[k]
            names.pop(k)
            pos.pop(k)
        fid.append(v)
    return _mk(names, pos, fid[0], fid[1], fid[2])


def _floats_in(s: str) -> List[float]:
    out = List[float]()
    for t in s.replace("\t", " ").split():
        out.append(float(t))
    return out


def read_dig_polhemus_isotrak(fname, ch_names = None, unit: str = "m") -> DigMontage:
    """mne.channels.read_dig_polhemus_isotrak (.hsp/.elp/.eeg)."""
    f = str(fname)
    dot = f.rfind(".")
    ext = f[dot:] if dot >= 0 else ""
    if ext not in (".hsp", ".elp", ".eeg"):
        raise ValueError(f"Invalid value for the 'fname' parameter. Allowed values are ('.hsp', '.elp', '.eeg'), but got '{ext}' instead.")
    sc = _unit_scale(unit)
    pts = List[List[float]]()
    lines = _read_text(f)
    if ext == ".elp":
        for ln in lines:
            t = ln.split()
            if len(t) < 3:
                continue
            v = List[float]()
            ok = True
            for x in t[len(t) - 3:]:
                try:
                    v.append(float(x))
                except:
                    ok = False
            if ok:
                pts.append(v)
    else:
        i = 0
        while i < len(lines) and "position of fiducials" not in lines[i].lower():
            i += 1
        i += 1
        fid = List[List[float]]()
        for k in range(3):
            fid.append(_floats_in(lines[i + k].replace("%F", "")))
        i += 4
        pts = fid
        if i < len(lines):
            hdr = _floats_in(lines[i])
            n_cols = int(hdr[1])
            vals = List[float]()
            for j in range(i + 1, len(lines)):
                for x in _floats_in(lines[j]):
                    vals.append(x)
            for r in range(len(vals) // n_cols):
                pts.append([vals[r * n_cols], vals[r * n_cols + 1], vals[r * n_cols + 2]])
    for r in range(len(pts)):
        pts[r] = [pts[r][0] * sc, pts[r][1] * sc, pts[r][2] * sc]
    nas = pts[0]
    lp = pts[1]
    rp = pts[2]
    names = List[str]()
    pos = List[List[float]]()
    if not isinstance(ch_names, NoneType):
        cn = [str(c) for c in ch_names]
        if len(pts) - 3 != len(cn):
            raise ValueError(f"Length of ``ch_names`` does not match the number of points in {f}. Expected ``ch_names`` length {len(pts) - 3}, given {len(cn)}")
        names = cn
        pos = [pts[3 + k] for k in range(len(cn))]
    return _mk(names, pos, nas, lp, rp)


def read_polhemus_fastscan(fname, unit: str = "mm", on_header_missing: str = "raise", *, verbose = None):
    """mne.channels.read_polhemus_fastscan: points array (n, 3)."""
    sc = _unit_scale(unit)
    hdr = ""
    rows = List[List[float]]()
    in_hdr = True
    for line in _read_text(str(fname)):
        if line.startswith("%"):
            if in_hdr:
                hdr += line
            continue
        in_hdr = False
        t = line.split()
        if len(t) == 0:
            continue
        rows.append([float(x) * sc for x in t])
    if "FastSCAN" not in hdr:
        msg = f"{fname} does not contain a valid Polhemus FastSCAN header"
        if on_header_missing == "raise":
            raise ValueError(msg)
        elif on_header_missing == "warn":
            print("RuntimeWarning: " + msg)
    out = np.zeros((len(rows), len(rows[0]) if len(rows) > 0 else 3))
    for i in range(len(rows)):
        for j in range(len(rows[i])):
            out[i, j] = rows[i][j]
    if out.shape[1] != 3:
        raise ValueError(f"Points must be of shape (n_points, 3), not {out.shape}")
    return out


def _xml_values(text: str, tag: str) -> List[str]:
    out = List[str]()
    op = "<" + tag + ">"
    cl = "</" + tag + ">"
    i = 0
    while True:
        a = text.find(op, i)
        if a < 0:
            break
        b = text.find(cl, a)
        out.append(text[a + len(op):b].strip())
        i = b + len(cl)
    return out


def _xml_blocks(text: str, tag: str) -> List[str]:
    return _xml_values(text, tag)


def read_dig_captrak(fname) -> DigMontage:
    """mne.channels.read_dig_captrak (.bvct)."""
    text = ""
    for ln in _read_text(str(fname)):
        text += ln
    names = List[str]()
    pos = List[List[float]]()
    nas: Optional[List[float]] = None
    lpa: Optional[List[float]] = None
    rpa: Optional[List[float]] = None
    for blk in _xml_blocks(text, "CapTrakElectrode"):
        nm = _xml_values(blk, "Name")[0]
        p = [float(_xml_values(blk, "X")[0]) * 1e-3, float(_xml_values(blk, "Y")[0]) * 1e-3,
             float(_xml_values(blk, "Z")[0]) * 1e-3]
        if nm == "Nasion":
            nas = p
        elif nm == "RPA":
            rpa = p
        elif nm == "LPA":
            lpa = p
        else:
            names.append(nm)
            pos.append(p)
    if nas is None or lpa is None or rpa is None:
        raise KeyError("nasion")
    return _mk(names, pos, nas, lpa, rpa)


def _pad3(n: int) -> str:
    s = str(n)
    while len(s) < 3:
        s = "0" + s
    return s


def read_dig_egi(fname) -> DigMontage:
    """mne.channels.read_dig_egi (MFF coordinates.xml)."""
    text = ""
    for ln in _read_text(str(fname)):
        text += ln
    names = List[str]()
    pos = List[List[float]]()
    nas: Optional[List[float]] = None
    lpa: Optional[List[float]] = None
    rpa: Optional[List[float]] = None
    for blk in _xml_blocks(text, "sensor"):
        nm = _xml_values(blk, "name")[0]
        number = int(_xml_values(blk, "number")[0])
        kind = int(_xml_values(blk, "type")[0])
        p = [float(_xml_values(blk, "x")[0]), float(_xml_values(blk, "y")[0]), float(_xml_values(blk, "z")[0])]
        if kind == 0:
            names.append("EEG " + _pad3(number))
            pos.append(p)
        elif kind == 1:
            names.append("EEG " + _pad3(len(names) + 1))
            pos.append(p)
        elif kind == 2:
            if nm == "Nasion":
                nas = p
            elif nm == "Right periauricular point":
                rpa = p
            elif nm == "Left periauricular point":
                lpa = p
            else:
                raise KeyError(nm)
        else:
            print(f"RuntimeWarning: Unknown sensor type {kind} detected. Skipping sensor...Proceed with caution!")
    if nas is None or lpa is None or rpa is None:
        raise KeyError("nasion")
    return _mk(names, pos, nas, lpa, rpa)


def compute_native_head_t(montage: DigMontage, *, on_missing: str = "warn", verbose = None) -> Transform:
    """mne.channels.compute_native_head_t."""
    frame = montage.coord_frame
    if frame == "head":
        return Transform("head", "head", np.eye(4))
    if montage.nasion is None or montage.lpa is None or montage.rpa is None:
        which = "nasion"
        if montage.nasion is not None:
            which = "lpa" if montage.lpa is None else "rpa"
        msg = f"Fiducial point {which} not found, assuming identity {frame} to head transformation"
        if on_missing == "raise":
            raise RuntimeError(msg)
        elif on_missing == "warn":
            print("RuntimeWarning: " + msg)
        return Transform(frame, "head", np.eye(4))
    t = _ras_to_neuromag(np.array(montage.nasion.__val__()), np.array(montage.lpa.__val__()),
                         np.array(montage.rpa.__val__()))
    return Transform(frame, "head", t)


def transform_to_head(montage: DigMontage) -> DigMontage:
    """mne.channels.transform_to_head."""
    return montage._to_head()


_MONTAGES = ["fsaverage_1005", "fsaverage_1010", "fsaverage_1020", "colin27_1005", "colin27_1020",
             "colin27_alphabetic", "colin27_postfixed", "colin27_prefixed", "colin27_primed",
             "biosemi16", "biosemi32", "biosemi64", "biosemi128", "biosemi160", "biosemi256",
             "easycap-M1", "easycap-M10", "easycap-M43", "EGI_256", "GSN-HydroCel-32",
             "GSN-HydroCel-64_1.0", "GSN-HydroCel-65_1.0", "GSN-HydroCel-128", "GSN-HydroCel-129",
             "GSN-HydroCel-256", "GSN-HydroCel-257", "mgh60", "mgh70", "artinis-octamon",
             "artinis-brite23", "brainproducts-RNP-BA-128", "spherical_1005", "spherical_1010",
             "spherical_1020"]

_ADJ = ["biosemi16", "biosemi32", "biosemi64", "bti148", "bti248", "bti248grad", "ctf151", "ctf275",
        "ctf64", "easycap128ch-avg", "easycap32ch-avg", "easycap64ch-avg", "easycapM1", "easycapM11",
        "easycapM14", "easycapM15", "ecog256", "ecog256bipolar", "eeg1010_neighb", "elec1005",
        "elec1010", "elec1020", "itab153", "itab28", "KIT-157", "KIT-208", "KIT-NYU-2019",
        "KIT-UMD-1", "KIT-UMD-2", "KIT-UMD-3", "KIT-UMD-4", "language29ch-avg", "mpi_59_channels",
        "neuromag122cmb", "neuromag306cmb", "neuromag306mag", "neuromag306planar", "yokogawa160",
        "yokogawa440"]


def get_builtin_montages(*, descriptions: Literal[bool] = False):
    """mne.channels.get_builtin_montages (names only)."""
    if descriptions:
        compile_error("descriptions=True is not supported")
    return list(_MONTAGES)


def get_builtin_ch_adjacencies(*, descriptions: Literal[bool] = False):
    """mne.channels.get_builtin_ch_adjacencies (names only)."""
    if descriptions:
        compile_error("descriptions=True is not supported")
    return list(_ADJ)


def rename_channels(info: Info, mapping, allow_duplicates: bool = False, *,
                    on_missing: str = "raise", verbose = None):
    """mne.channels.rename_channels (dict or callable mapping), in place."""
    new = list(info.ch_names)
    if isinstance(mapping, dict):
        for k in mapping.keys():
            ks = str(k)
            if ks not in info.ch_names:
                if on_missing == "raise":
                    raise ValueError(f"Invalid channel name(s) ['{ks}'] is not present in info")
                if on_missing == "warn":
                    print("RuntimeWarning: Channel rename map contains keys that are not present in the object to be renamed. These will be ignored.")
                continue
            new[info.ch_names.index(ks)] = str(mapping[k])
    else:
        for i in range(len(new)):
            new[i] = str(mapping(new[i]))
    if not allow_duplicates and len(set(new)) != len(new):
        raise ValueError("New channel names are not unique, renaming failed")
    bads = List[str]()
    for b in info.bads:
        bads.append(new[info.ch_names.index(b)])
    info.ch_names = new
    info.bads = bads
    return info


def unify_bad_channels(insts):
    """mne.channels.unify_bad_channels: union of info['bads'] across instances."""
    if len(insts) == 0:
        raise ValueError("insts must not be empty")
    base = insts[0].info.ch_names
    allb = List[str]()
    for inst in insts:
        if len(inst.info.ch_names) != len(base):
            raise ValueError("Channels do not match across the objects in insts. Consider calling equalize_channels before calling this function.")
        for i in range(len(base)):
            if inst.info.ch_names[i] != base[i]:
                if inst.info.ch_names[i] in base:
                    raise ValueError("Channel names are sorted differently across instances. Please use mne.channels.equalize_channels.")
                raise ValueError("Channels do not match across the objects in insts. Consider calling equalize_channels before calling this function.")
        for b in inst.info.bads:
            if b not in allb:
                allb.append(b)
    for inst in insts:
        inst.info.bads = list(allb)
    return insts
