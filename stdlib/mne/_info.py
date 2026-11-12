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


class Info:
    """Measurement info (subset of mne.Info)."""
    ch_names: List[str]
    ch_types: List[str]
    sfreq: float
    bads: List[str]
    highpass: float
    lowpass: float
    description: Optional[str]

    def __init__(self, ch_names: List[str], sfreq: float, ch_types: List[str]):
        self.ch_names = ch_names
        self.ch_types = ch_types
        self.sfreq = sfreq
        self.bads = List[str]()
        self.highpass = 0.0
        self.lowpass = sfreq / 2.0
        self.description = None

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
        return n

    def __repr__(self) -> str:
        return f"<Info | {self.nchan} channels, sfreq: {self.sfreq} Hz>"

    def _pick(self, sel: List[int]):
        n = Info([self.ch_names[i] for i in sel], self.sfreq, [self.ch_types[i] for i in sel])
        n.bads = [b for b in self.bads if b in n.ch_names]
        n.highpass = self.highpass
        n.lowpass = self.lowpass
        n.description = self.description
        return n


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
