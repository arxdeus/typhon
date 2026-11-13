# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# mne.io misc: get_channel_type_constants, anonymize_info, write_info,
# read/write_fiducials (FIF ISOTRAK block), read_raw (dispatch by extension).
# For 'chpi' the 'kind' entry holds only the first kind (700).

import numpy as np
from ._bytes import _i32_be, _f32_be
from ._fiff import (FifFile, FifWriter, FIFFB_MEAS, FIFF_BLOCK_ID, FIFFT_INT)
from .._info import Info

FIFFB_ISOTRAK = 107
FIFF_DIG_POINT = 213
FIFFT_DIG_POINT_STRUCT = 33
FIFF_MNE_COORD_FRAME = 3506

_CT_BASE = {
    "grad": {"kind": 1, "unit": 201},
    "mag": {"kind": 1, "unit": 112},
    "ref_meg": {"kind": 301},
    "eeg": {"kind": 2, "unit": 107, "coil_type": 1},
    "seeg": {"kind": 802, "unit": 107, "coil_type": 1},
    "dbs": {"kind": 803, "unit": 107, "coil_type": 1},
    "ecog": {"kind": 902, "unit": 107, "coil_type": 1},
    "eog": {"kind": 202, "unit": 107},
    "emg": {"kind": 302, "unit": 107},
    "ecg": {"kind": 402, "unit": 107},
    "resp": {"kind": 602, "unit": 107},
    "bio": {"kind": 102, "unit": 107},
    "misc": {"kind": 502, "unit": 107},
    "stim": {"kind": 3},
    "exci": {"kind": 920},
    "syst": {"kind": 900},
    "ias": {"kind": 910},
    "gof": {"kind": 1001},
    "dipole": {"kind": 1000},
    "chpi": {"kind": 700},
    "fnirs_cw_amplitude": {"kind": 1100, "unit": 107, "coil_type": 302},
    "fnirs_fd_ac_amplitude": {"kind": 1100, "unit": 107, "coil_type": 304},
    "fnirs_fd_phase": {"kind": 1100, "unit": 7, "coil_type": 305},
    "fnirs_td_gated_amplitude": {"kind": 1100, "unit": 0, "coil_type": 306},
    "fnirs_td_moments_intensity": {"kind": 1100, "unit": 0, "coil_type": 307},
    "fnirs_td_moments_mean": {"kind": 1100, "unit": 3, "coil_type": 308},
    "fnirs_td_moments_variance": {"kind": 1100, "unit": 118, "coil_type": 309},
    "fnirs_od": {"kind": 1100, "coil_type": 303},
    "hbo": {"kind": 1100, "unit": 6, "coil_type": 300},
    "hbr": {"kind": 1100, "unit": 6, "coil_type": 301},
    "csd": {"kind": 2, "unit": 117, "coil_type": 6},
    "temperature": {"kind": 1200, "unit": 114},
    "gsr": {"kind": 1300, "unit": 110},
    "eyegaze": {"kind": 1400, "coil_type": 400},
    "pupil": {"kind": 1400, "coil_type": 401},
}
_CT_DEFAULTS = {
    "grad": {"kind": 1, "unit": 201, "coil_type": 3012},
    "mag": {"kind": 1, "unit": 112, "coil_type": 3024},
    "ref_meg": {"kind": 301, "coil_type": 3024, "unit": 112},
    "eeg": {"kind": 2, "unit": 107, "coil_type": 1},
    "seeg": {"kind": 802, "unit": 107, "coil_type": 1},
    "dbs": {"kind": 803, "unit": 107, "coil_type": 1},
    "ecog": {"kind": 902, "unit": 107, "coil_type": 1},
    "eog": {"kind": 202, "unit": 107, "coil_type": 0},
    "emg": {"kind": 302, "unit": 107, "coil_type": 0},
    "ecg": {"kind": 402, "unit": 107, "coil_type": 0},
    "resp": {"kind": 602, "unit": 107},
    "bio": {"kind": 102, "unit": 107, "coil_type": 0},
    "misc": {"kind": 502, "unit": -1, "coil_type": 0},
    "stim": {"kind": 3, "unit": 107, "coil_type": 0},
    "exci": {"kind": 920},
    "syst": {"kind": 900},
    "ias": {"kind": 910},
    "gof": {"kind": 1001},
    "dipole": {"kind": 1000},
    "chpi": {"kind": 700},
    "fnirs_cw_amplitude": {"kind": 1100, "unit": 107, "coil_type": 302},
    "fnirs_fd_ac_amplitude": {"kind": 1100, "unit": 107, "coil_type": 304},
    "fnirs_fd_phase": {"kind": 1100, "unit": 7, "coil_type": 305},
    "fnirs_td_gated_amplitude": {"kind": 1100, "unit": 0, "coil_type": 306},
    "fnirs_td_moments_intensity": {"kind": 1100, "unit": 0, "coil_type": 307},
    "fnirs_td_moments_mean": {"kind": 1100, "unit": 3, "coil_type": 308},
    "fnirs_td_moments_variance": {"kind": 1100, "unit": 118, "coil_type": 309},
    "fnirs_od": {"kind": 1100, "coil_type": 303, "unit": -1},
    "hbo": {"kind": 1100, "unit": 6, "coil_type": 300},
    "hbr": {"kind": 1100, "unit": 6, "coil_type": 301},
    "csd": {"kind": 2, "unit": 117, "coil_type": 6},
    "temperature": {"kind": 1200, "unit": 114},
    "gsr": {"kind": 1300, "unit": 110},
    "eyegaze": {"kind": 1400, "coil_type": 400, "unit": 210},
    "pupil": {"kind": 1400, "coil_type": 401, "unit": -1},
}


def get_channel_type_constants(*, include_defaults: bool = False) -> Dict[str, Dict[str, int]]:
    """mne.io.get_channel_type_constants."""
    out = Dict[str, Dict[str, int]]()
    src = _CT_DEFAULTS if include_defaults else _CT_BASE
    for k, v in src.items():
        d = Dict[str, int]()
        for kk, vv in v.items():
            d[kk] = vv
        out[k] = d
    return out


def anonymize_info(info: Info, daysback = None, keep_his = False, verbose = None) -> Info:
    """mne.io.anonymize_info (meas_date shift, experimenter and description; modified in place).

    Subject info, proj/file ids and proc history are not part of this Info."""
    default_str = "mne_anonymize"
    default_desc = "Anonymized using a time shift to preserve age at acquisition"
    if info.meas_date is not None:
        md = info.meas_date.__val__()
        delta = md - 946684800.0
        if not isinstance(daysback, NoneType):
            delta = float(daysback) * 86400.0
        info.meas_date = md - delta
    elif not isinstance(daysback, NoneType):
        print('RuntimeWarning: Input info has "meas_date" set to None. Removing all information from time/date structures, *NOT* performing any time shifts!')
    info.experimenter = default_str
    info.description = default_desc
    return info


def write_info(fname, info: Info, data_type = None, reset_range: bool = True, *,
               overwrite: bool = False, verbose = None):
    """mne.io.write_info: measurement info only (FIF)."""
    import os
    f = str(fname)
    if not overwrite and os.path.exists(f):
        raise FileExistsError(f"Destination file exists. Please use option 'overwrite=True' to force overwriting.")
    w = FifWriter()
    w.start_block(FIFFB_MEAS)
    dt = 0
    if not isinstance(data_type, NoneType):
        dt = int(data_type)
    w.write_meas_info(info, dt)
    w.end_block(FIFFB_MEAS)
    w.end_file(f)


class DigPoint:
    """Digitizer point: kind, ident, r (3,), coord_frame."""
    kind: int
    ident: int
    r: np.ndarray[float, 1]
    coord_frame: int

    def __init__(self, kind: int, ident: int, r, coord_frame: int = 4):
        self.kind = kind
        self.ident = ident
        self.r = np.asarray(r, dtype=float)
        self.coord_frame = coord_frame

    def __getitem__(self, key: Literal[str]):
        if key == "kind":
            return self.kind
        elif key == "ident":
            return self.ident
        elif key == "r":
            return self.r
        elif key == "coord_frame":
            return self.coord_frame
        else:
            compile_error("unsupported DigPoint key")

    def __repr__(self) -> str:
        return f"<DigPoint | kind={self.kind}, ident={self.ident}, r={self.r}>"


def _frame_const(frame: str) -> int:
    names = ["unknown", "meg", "mri", "head", "isotrak", "hpi", "mni_tal", "ras", "fs_tal",
             "ctf_head", "ctf_meg", "mri_voxel"]
    consts = [0, 1, 5, 4, 2, 3, 2003, 2002, 2006, 1004, 1001, 2001]
    if frame not in names:
        raise ValueError(f"Invalid value for the 'frame' parameter. Allowed values are {names}, but got '{frame}' instead.")
    return consts[names.index(frame)]


def read_fiducials(fname, *, verbose = None):
    """mne.io.read_fiducials: (list of DigPoint, coord_frame int)."""
    f = FifFile(str(fname))
    nodes = f.tree.find(FIFFB_ISOTRAK)
    if len(nodes) == 0:
        raise IndexError("list index out of range")
    if len(nodes) > 1:
        print("RuntimeWarning: Multiple Isotrak found")
    frame = 4
    pts = List[DigPoint]()
    for t in nodes[0].entries:
        if t.kind == FIFF_DIG_POINT:
            p = t.pos + 16
            kind = _i32_be(f.b, p)
            ident = _i32_be(f.b, p + 4)
            r = [_f32_be(f.b, p + 8), _f32_be(f.b, p + 12), _f32_be(f.b, p + 16)]
            pts.append(DigPoint(kind, ident, r, 0))
        elif t.kind == FIFF_MNE_COORD_FRAME:
            frame = f.int(t)
    for pt in pts:
        pt.coord_frame = frame
    return pts, frame


def write_fiducials(fname, pts, coord_frame: str = "unknown", *, overwrite: bool = False,
                    verbose = None):
    """mne.io.write_fiducials (coord_frame given as a name)."""
    import os
    fn = str(fname)
    if not overwrite and os.path.exists(fn):
        raise FileExistsError(f"Destination file exists. Please use option 'overwrite=True' to force overwriting.")
    w = FifWriter()
    w.start_block(FIFFB_ISOTRAK)
    w.write_int(FIFF_MNE_COORD_FRAME, _frame_const(coord_frame))
    for pt in pts:
        w._head(FIFF_DIG_POINT, FIFFT_DIG_POINT_STRUCT, 20)
        w.w.i32(int(pt.kind))
        w.w.i32(int(pt.ident))
        w.w.f32(float(pt.r[0]))
        w.w.f32(float(pt.r[1]))
        w.w.f32(float(pt.r[2]))
    w.end_block(FIFFB_ISOTRAK)
    w.end_file(fn)


def read_raw(fname, *, preload = False, verbose = None, **kwargs):
    """mne.io.read_raw: dispatch on the file extension (fif, edf, bdf, gdf, vhdr, set)."""
    from .fiff import read_raw_fif
    from .edf import read_raw_edf, read_raw_bdf
    from .gdf import read_raw_gdf
    from .brainvision import read_raw_brainvision
    from .eeglab import read_raw_eeglab
    f = str(fname)
    low = f.lower()
    if low.endswith(".fif") or low.endswith(".fif.gz"):
        return read_raw_fif(f, False, preload)
    if low.endswith(".edf"):
        return read_raw_edf(f, preload = preload)
    if low.endswith(".bdf"):
        return read_raw_bdf(f, preload = preload)
    if low.endswith(".gdf"):
        return read_raw_gdf(f, preload = preload)
    if low.endswith(".vhdr"):
        return read_raw_brainvision(f, preload = preload)
    if low.endswith(".set"):
        return read_raw_eeglab(f, preload = preload)
    if low.endswith(".cnt"):
        from .cnt import read_raw_cnt
        return read_raw_cnt(f, preload = preload)
    if low.endswith(".raw") or low.endswith(".mff"):
        from .egi import read_raw_egi
        return read_raw_egi(f, preload = preload)
    if low.endswith(".dat"):
        from .bci2k import read_raw_bci2k
        return read_raw_bci2k(f, preload = preload)
    if low.endswith(".nxe"):
        from .eximia import read_raw_eximia
        return read_raw_eximia(f, preload = preload)
    if low.endswith(".nedf"):
        from .nedf import read_raw_nedf
        return read_raw_nedf(f, preload = preload)
    dot = f.rfind(".")
    ext = f[dot:] if dot >= 0 else ""
    raise ValueError(f"Unsupported file type ({ext}). Supported (Typhon mne): .fif, .edf, .bdf, .gdf, .vhdr, .set, .cnt, .raw, .mff, .dat, .nedf")
