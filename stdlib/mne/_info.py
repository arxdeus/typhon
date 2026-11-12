# Ported from mne 1.13.2 (BSD-3-Clause)

import numpy as np

_DATA_CH_TYPES_SPLIT = ("mag", "grad", "eeg", "csd", "seeg", "ecog", "dbs",
                        "hbo", "hbr", "fnirs_cw_amplitude", "fnirs_fd_ac_amplitude",
                        "fnirs_fd_phase", "fnirs_od")
_DATA_OR_ICA = ("mag", "grad", "eeg", "csd", "seeg", "ecog", "dbs", "hbo", "hbr",
                "fnirs_cw_amplitude", "fnirs_fd_ac_amplitude", "fnirs_fd_phase",
                "fnirs_od", "ref_meg")
_VALID_CH_TYPES = ("mag", "grad", "eeg", "csd", "seeg", "ecog", "dbs", "hbo", "hbr",
                   "fnirs_cw_amplitude", "fnirs_fd_ac_amplitude", "fnirs_fd_phase",
                   "fnirs_od", "ref_meg", "eog", "ecg", "emg", "misc", "stim", "resp",
                   "bio", "gsr", "temperature", "syst", "exci", "ias", "chpi",
                   "dipole", "gof", "eyegaze", "pupil")


def _scaling_cov_rank(ch_type: str) -> float:
    # mne.defaults scalings_cov_rank merged into _check_scaling_inputs defaults
    if ch_type == "mag":
        return 1e12
    if ch_type == "grad":
        return 1e11
    if ch_type == "eeg":
        return 1e5
    if ch_type == "seeg":
        return 1e1
    if ch_type in ("dbs", "ecog", "hbo", "hbr"):
        return 1e4
    return 1.0


class DigMontage:
    """Channel positions + fiducials (coord_frame 'unknown', 'mri' or 'head')."""
    ch_names: List[str]
    pos: List[List[float]]
    coord_frame: str
    nasion: Optional[List[float]]
    lpa: Optional[List[float]]
    rpa: Optional[List[float]]

    def __init__(self, ch_names: List[str], pos: List[List[float]], coord_frame: str,
                 nasion: Optional[List[float]], lpa: Optional[List[float]],
                 rpa: Optional[List[float]]):
        self.ch_names = ch_names
        self.pos = pos
        self.coord_frame = coord_frame
        self.nasion = nasion
        self.lpa = lpa
        self.rpa = rpa

    def copy(self):
        return DigMontage(list(self.ch_names), [list(p) for p in self.pos], self.coord_frame,
                          None if self.nasion is None else list(self.nasion.__val__()),
                          None if self.lpa is None else list(self.lpa.__val__()),
                          None if self.rpa is None else list(self.rpa.__val__()))

    def __repr__(self) -> str:
        nf = int(self.nasion is not None) + int(self.lpa is not None) + int(self.rpa is not None)
        return f"<DigMontage | 0 extras (headshape), 0 HPIs, {nf} fiducials, {len(self.ch_names)} channels>"

    def __len__(self) -> int:
        return len(self.ch_names)

    def get_positions(self):
        """dict-like: 'ch_pos' (name -> xyz), 'nasion'/'lpa'/'rpa' (Optional
        xyz), 'coord_frame'."""
        return _MontagePositions(self)

    def get_ch_pos(self) -> Dict[str, np.ndarray[float, 1]]:
        out = Dict[str, np.ndarray[float, 1]]()
        for i in range(len(self.ch_names)):
            out[self.ch_names[i]] = np.array(self.pos[i])
        return out

    def _to_head(self) -> DigMontage:
        """transform_to_head: apply the native->head transform from fiducials."""
        m = self.copy()
        if self.coord_frame == "head":
            return m
        if self.nasion is None or self.lpa is None or self.rpa is None:
            m.coord_frame = "head"
            return m
        t = _ras_to_neuromag(np.array(self.nasion.__val__()), np.array(self.lpa.__val__()),
                             np.array(self.rpa.__val__()))
        m.pos = [_apply(t, p) for p in m.pos]
        m.nasion = _apply(t, m.nasion.__val__())
        m.lpa = _apply(t, m.lpa.__val__())
        m.rpa = _apply(t, m.rpa.__val__())
        m.coord_frame = "head"
        return m

    def rename_channels(self, mapping: Dict[str, str]):
        self.ch_names = [mapping[c] if c in mapping else c for c in self.ch_names]
        return self


def _opt_arr(x: Optional[List[float]]) -> Optional[np.ndarray[float, 1]]:
    if x is None:
        return None
    return np.array(x.__val__())


class _MontagePositions:
    ch_pos: Dict[str, np.ndarray[float, 1]]
    nasion: Optional[np.ndarray[float, 1]]
    lpa: Optional[np.ndarray[float, 1]]
    rpa: Optional[np.ndarray[float, 1]]
    coord_frame: str

    def __init__(self, m: DigMontage):
        self.ch_pos = m.get_ch_pos()
        self.nasion = _opt_arr(m.nasion)
        self.lpa = _opt_arr(m.lpa)
        self.rpa = _opt_arr(m.rpa)
        self.coord_frame = m.coord_frame

    def __getitem__(self, key: Literal[str]):
        if key == "ch_pos":
            return self.ch_pos
        elif key == "nasion":
            return self.nasion
        elif key == "lpa":
            return self.lpa
        elif key == "rpa":
            return self.rpa
        elif key == "coord_frame":
            return self.coord_frame
        else:
            compile_error("unsupported montage positions key")


def _apply(t: np.ndarray[float, 2], p: List[float]) -> List[float]:
    return [t[i, 0] * p[0] + t[i, 1] * p[1] + t[i, 2] * p[2] + t[i, 3] for i in range(3)]


def _ras_to_neuromag(nasion64, lpa64, rpa64) -> np.ndarray[float, 2]:
    # mne stores dig fiducials as float32, so get_ras_to_neuromag_trans runs
    # its vector algebra in float32.
    nasion = nasion64.astype(np.float32)
    lpa = lpa64.astype(np.float32)
    rpa = rpa64.astype(np.float32)
    right = rpa - lpa
    ru = right / np.linalg.norm(right)
    origin = lpa + np.dot(nasion - lpa, ru) * ru
    ant = nasion - origin
    au = ant / np.linalg.norm(ant)
    su = np.cross(ru, au)
    rot = np.eye(4)
    for j in range(3):
        rot[0, j] = ru[j]
        rot[1, j] = au[j]
        rot[2, j] = su[j]
    tr = np.eye(4)
    tr[0, 3] = -origin[0]
    tr[1, 3] = -origin[1]
    tr[2, 3] = -origin[2]
    return np.dot(rot, tr)


class Info:
    """Measurement info (subset of mne.Info)."""
    ch_names: List[str]
    ch_types: List[str]
    sfreq: float
    bads: List[str]
    highpass: float
    lowpass: float
    description: Optional[str]
    cals: List[float]
    ranges: List[float]
    locs: List[List[float]]
    coil_types: List[int]
    units: List[int]
    line_freq: Optional[float]
    experimenter: Optional[str]
    _montage: Optional[DigMontage]

    def __init__(self, ch_names: List[str], sfreq: float, ch_types: List[str]):
        self.ch_names = ch_names
        self.ch_types = ch_types
        self.sfreq = sfreq
        self.bads = List[str]()
        self.highpass = 0.0
        self.lowpass = sfreq / 2.0
        self.description = None
        n = len(ch_names)
        self.cals = [1.0 for _ in range(n)]
        self.ranges = [1.0 for _ in range(n)]
        self.locs = [[np.nan for _ in range(12)] for _ in range(n)]
        self.coil_types = [_default_coil(t) for t in ch_types]
        self.units = [_default_unit(t) for t in ch_types]
        self.line_freq = None
        self.experimenter = None
        self._montage = None

    def _copy_extra_from(self, other: Info, sel: List[int]):
        self.cals = [other.cals[i] for i in sel]
        self.ranges = [other.ranges[i] for i in sel]
        self.locs = [list(other.locs[i]) for i in sel]
        self.coil_types = [other.coil_types[i] for i in sel]
        self.units = [other.units[i] for i in sel]
        self.line_freq = other.line_freq
        self.experimenter = other.experimenter
        if other._montage is not None:
            keep = [other.ch_names[i] for i in sel]
            m = other._montage.__val__()
            idx = [k for k in range(len(m.ch_names)) if m.ch_names[k] in keep]
            self._montage = DigMontage([m.ch_names[k] for k in idx], [list(m.pos[k]) for k in idx],
                                       m.coord_frame, m.nasion, m.lpa, m.rpa)

    def _append_ch(self, other: Info, i: int):
        self.ch_names.append(other.ch_names[i])
        self.ch_types.append(other.ch_types[i])
        self.cals.append(other.cals[i])
        self.ranges.append(other.ranges[i])
        self.locs.append(list(other.locs[i]))
        self.coil_types.append(other.coil_types[i])
        self.units.append(other.units[i])

    @property
    def nchan(self) -> int:
        return len(self.ch_names)

    def __getitem__(self, key: Literal[str]):
        if key == "sfreq":
            return self.sfreq
        elif key == "ch_names":
            return self.ch_names
        elif key == "nchan":
            return self.nchan
        elif key == "bads":
            return self.bads
        elif key == "highpass":
            return self.highpass
        elif key == "lowpass":
            return self.lowpass
        elif key == "description":
            return self.description
        else:
            compile_error("unsupported Info key")

    def __setitem__(self, key: Literal[str], value):
        if key == "bads":
            self.bads = value
        elif key == "highpass":
            self.highpass = float(value)
        elif key == "lowpass":
            self.lowpass = float(value)
        elif key == "description":
            self.description = value
        else:
            compile_error("unsupported or read-only Info key")

    def __contains__(self, key: str) -> bool:
        return key in ("sfreq", "ch_names", "nchan", "bads", "highpass", "lowpass", "description")

    def get_channel_types(self, picks=None, unique: bool = False, only_data_chs: bool = False) -> List[str]:
        idx = _picks_to_idx(self, picks, "all", ())
        out = List[str]()
        for i in idx:
            t = self.ch_types[i]
            if only_data_chs and t not in _DATA_OR_ICA:
                continue
            if unique and t in out:
                continue
            out.append(t)
        return out

    def copy(self):
        n = Info(list(self.ch_names), self.sfreq, list(self.ch_types))
        n.bads = list(self.bads)
        n.highpass = self.highpass
        n.lowpass = self.lowpass
        n.description = self.description
        n._copy_extra_from(self, list(range(self.nchan)))
        return n

    def __repr__(self) -> str:
        return f"<Info | {self.nchan} channels, sfreq: {self.sfreq} Hz>"

    def _pick(self, sel: List[int]):
        n = Info([self.ch_names[i] for i in sel], self.sfreq, [self.ch_types[i] for i in sel])
        n.bads = [b for b in self.bads if b in n.ch_names]
        n.highpass = self.highpass
        n.lowpass = self.lowpass
        n.description = self.description
        n._copy_extra_from(self, sel)
        return n

    def set_channel_types(self, mapping: Dict[str, str]):
        for k, v in mapping.items():
            if k not in self.ch_names:
                raise ValueError(f"This channel name ({k}) doesn't exist in info.")
            if v not in _VALID_CH_TYPES:
                raise ValueError(f"This function cannot change to this channel type: {v}.")
            i = self.ch_names.index(k)
            self.ch_types[i] = v
            self.coil_types[i] = _default_coil(v)
            self.units[i] = _default_unit(v)
        return self

    def rename_channels(self, mapping: Dict[str, str]):
        for k, v in mapping.items():
            if k not in self.ch_names:
                raise ValueError(f"Invalid channel name(s) {k} is not present in info")
        new = [mapping[c] if c in mapping else c for c in self.ch_names]
        if len(set(new)) != len(new):
            raise ValueError("New channel names are not unique, renaming failed")
        self.bads = [mapping[c] if c in mapping else c for c in self.bads]
        self.ch_names = new
        return self


def _default_coil(t: str) -> int:
    # FIFFV_COIL_EEG for EEG-like electrodes, else FIFFV_COIL_NONE
    if t in ("eeg", "seeg", "dbs", "ecog"):
        return 1
    if t == "mag":
        return 3024
    if t == "grad":
        return 3012
    return 0


def _default_unit(t: str) -> int:
    if t == "mag":
        return 112  # T
    if t == "grad":
        return 201  # T/m
    if t in ("stim", "ref_meg", "exci", "syst", "ias", "gof", "dipole", "chpi"):
        return -1
    return 107  # V


_KIND_OF_TYPE = (("mag", 1), ("grad", 1), ("eeg", 2), ("stim", 3), ("bio", 102),
                 ("eog", 202), ("ref_meg", 301), ("emg", 302), ("ecg", 402),
                 ("misc", 502), ("resp", 602), ("seeg", 802), ("dbs", 803),
                 ("syst", 900), ("ecog", 902), ("ias", 910), ("exci", 920),
                 ("temperature", 1200), ("gsr", 1300))


def _type_to_kind(t: str) -> int:
    for p in _KIND_OF_TYPE:
        if p[0] == t:
            return p[1]
    return 502


def _kind_to_type(kind: int, unit: int) -> str:
    if kind == 1:
        return "grad" if unit == 201 else "mag"
    for p in _KIND_OF_TYPE:
        if p[1] == kind:
            return p[0]
    return "misc"


def create_info(ch_names, sfreq: float, ch_types = "misc", verbose = None) -> Info:
    """Create a basic Info instance."""
    names = List[str]()
    if isinstance(ch_names, int):
        for i in range(ch_names):
            names.append(str(i))
    elif isinstance(ch_names, str):
        names.append(ch_names)
    else:
        for c in ch_names:
            names.append(str(c))
    if sfreq <= 0:
        raise ValueError("sfreq must be positive")
    types = List[str]()
    if isinstance(ch_types, str):
        for _ in range(len(names)):
            types.append(ch_types)
    else:
        for t in ch_types:
            types.append(str(t))
    if len(types) != len(names):
        raise ValueError(f"ch_types and ch_names must be the same length ({len(types)} != {len(names)}) for ch_types={types}")
    for t in types:
        if t not in _VALID_CH_TYPES:
            raise KeyError(f"kind must be one of {list(_VALID_CH_TYPES)}, not {t}")
    seen = Set[str]()
    for n in names:
        if n in seen:
            raise ValueError(f"Channel names must be unique, found duplicate {n}")
        seen.add(n)
    return Info(names, float(sfreq), types)


def _type_matches(t: str, kind: str) -> bool:
    if kind == "all":
        return True
    if kind == "data":
        return t in _DATA_OR_ICA
    if kind == "data_or_ica":
        return t in _DATA_OR_ICA
    if kind == "meg":
        return t == "mag" or t == "grad"
    if kind == "fnirs":
        return t.startswith("fnirs") or t == "hbo" or t == "hbr"
    return t == kind


def _picks_to_idx(info: Info, picks, none: str = "all", exclude = "bads",
                  allow_empty: bool = False) -> List[int]:
    """Convert picks (None | int | str | list | ndarray) to channel indices."""
    n = info.nchan
    excl = List[str]()
    if isinstance(exclude, str):
        if exclude == "bads":
            excl = list(info.bads)
    elif isinstance(exclude, NoneType):
        pass
    else:
        for e in exclude:
            excl.append(str(e))
    out = List[int]()
    if isinstance(picks, NoneType):
        for i in range(n):
            if _type_matches(info.ch_types[i], none) and info.ch_names[i] not in excl:
                out.append(i)
        if len(out) == 0 and not allow_empty:
            raise ValueError(f'picks (None, treated as "{none}") yielded no channels, consider passing picks explicitly')
        return out
    elif isinstance(picks, int):
        p = picks
        if p < 0:
            p += n
        if p < 0 or p >= n:
            raise IndexError(f"picks must be >= {-n} and < {n}")
        return [p]
    elif isinstance(picks, str):
        if picks in info.ch_names:
            return [info.ch_names.index(picks)]
        for i in range(n):
            if _type_matches(info.ch_types[i], picks) and info.ch_names[i] not in excl:
                out.append(i)
        if len(out) == 0 and not allow_empty:
            raise ValueError(f"picks ('{picks}') could not be interpreted as channel names (no channel matched) or a channel type")
        return out
    elif isinstance(picks, slice):
        return list(range(n))[picks]
    else:
        for p in picks:
            if isinstance(p, str):
                if p in info.ch_names:
                    out.append(info.ch_names.index(p))
                else:
                    found = False
                    for i in range(n):
                        if _type_matches(info.ch_types[i], p) and info.ch_names[i] not in excl:
                            out.append(i)
                            found = True
                    if not found:
                        raise ValueError(f"picks ({p}) could not be interpreted as channel names or types")
            elif isinstance(p, bool):
                compile_error("boolean picks are not supported")
            else:
                q = int(p)
                if q < 0:
                    q += n
                if q < 0 or q >= n:
                    raise IndexError(f"picks must be >= {-n} and < {n}")
                out.append(q)
        if len(out) == 0 and not allow_empty:
            raise ValueError("No appropriate channels found for the given picks")
        return out


def _picks_by_type(info: Info, meg_combined: bool = False, exclude_bads: bool = True) -> List[Tuple[str, List[int]]]:
    out = List[Tuple[str, List[int]]]()
    for t in _DATA_CH_TYPES_SPLIT:
        idx = List[int]()
        for k in range(info.nchan):
            if exclude_bads and info.ch_names[k] in info.bads:
                continue
            ct = info.ch_types[k]
            if ct == "ref_meg":
                continue
            if ct == t:
                idx.append(k)
        out.append((t, idx))
    if meg_combined and len(out[0][1]) > 0 and len(out[1][1]) > 0:
        meg = sorted(out[0][1] + out[1][1])
        out = [("meg", meg)] + out[2:]
    return [p for p in out if len(p[1]) > 0]


def _time_mask(times, tmin: Optional[float] = None, tmax: Optional[float] = None,
               sfreq: Optional[float] = None, raise_error: bool = True,
               include_tmax: bool = True):
    """Safely find sample boundaries (port of mne.utils._time_mask)."""
    tmn: float = -np.inf if tmin is None else tmin.__val__()
    tmx: float = np.inf if tmax is None else tmax.__val__()
    if not np.isfinite(tmn):
        tmn = float(times[0])
    inc = include_tmax
    if not np.isfinite(tmx):
        tmx = float(times[-1])
        inc = True
    if sfreq is not None:
        sf = float(sfreq)
        tmn = int(round(tmn * sf)) / sf - 0.5 / sf
        tmx = int(round(tmx * sf)) / sf
        tmx += (0.5 if inc else -0.5) / sf
    if raise_error and tmn > tmx:
        raise ValueError(f"tmin ({tmin}) must be less than or equal to tmax ({tmax})")
    mask = (times >= tmn) & (times <= tmx)
    if raise_error and not mask.any():
        raise ValueError(f"No samples remain when using tmin={tmin} and tmax={tmax}")
    return mask
