# Ported from mne 1.13.2 (BSD-3-Clause)
#
# Annotations (orig_time=None only) and events_from_annotations.

import numpy as np


def _us(x: float) -> float:
    # timedelta(seconds=x).total_seconds(): microsecond resolution, half-even
    return float(np.round(x * 1e6)) / 1e6


class Annotations:
    """Annotations with orig_time=None (onsets relative to the first sample)."""
    onset: List[float]
    duration: List[float]
    description: List[str]

    def __init__(self, onset, duration, description, orig_time = None, ch_names = None):
        if not isinstance(orig_time, NoneType):
            compile_error("Typhon mne: only Annotations(orig_time=None) is supported")
        self.onset = [float(v) for v in onset]
        n = len(self.onset)
        if isinstance(duration, int) or isinstance(duration, float):
            self.duration = [float(duration) for _ in range(n)]
        else:
            self.duration = [float(v) for v in duration]
        if isinstance(description, str):
            self.description = [description for _ in range(n)]
        else:
            self.description = [str(v) for v in description]
        if len(self.duration) != n or len(self.description) != n:
            raise ValueError("Onset, duration and description must be equal in sizes")

    @property
    def orig_time(self):
        return None

    def __len__(self) -> int:
        return len(self.onset)

    def copy(self):
        return Annotations(list(self.onset), list(self.duration), list(self.description))

    def __repr__(self) -> str:
        return f"<Annotations | {len(self)} segments>"

    def __getitem__(self, i: int):
        return (self.onset[i], self.duration[i], self.description[i])

    def append(self, onset, duration, description):
        self.onset.append(float(onset))
        self.duration.append(float(duration))
        self.description.append(str(description))
        return self

    def __add__(self, other: Annotations):
        a = self.copy()
        for i in range(len(other)):
            a.append(other.onset[i], other.duration[i], other.description[i])
        return a

    def crop(self, tmin: Optional[float] = None, tmax: Optional[float] = None,
             emit_warning: bool = False, use_orig_time: bool = True, verbose = None):
        if len(self) == 0:
            return self
        lo = 0.0
        hi = 0.0
        if tmin is None:
            lo = min(self.onset)
        else:
            lo = tmin
        if tmax is None:
            hi = max([self.onset[i] + self.duration[i] for i in range(len(self))])
        else:
            hi = tmax
        lo = _us(lo)
        hi = _us(hi)
        if lo > hi:
            raise ValueError(f"tmax should be greater than or equal to tmin ({lo} < {hi}).")
        on = List[float]()
        du = List[float]()
        de = List[str]()
        for i in range(len(self)):
            d = self.duration[i]
            if np.isnan(d):
                d = 0.0
            a = _us(self.onset[i])
            b = a + _us(d)
            if a > hi or b < lo:
                continue
            cl = a < lo
            if cl:
                a = lo
            cr = b > hi
            if cr:
                b = hi
            du.append(_us(b - a) if (cl or cr) else d)
            on.append(a)
            de.append(self.description[i])
        self.onset = on
        self.duration = du
        self.description = de
        return self


def _desc_ok(d: str) -> bool:
    # default regexp ^(?![Bb][Aa][Dd]|[Ee][Dd][Gg][Ee]).*$
    l = d.lower()
    return not (l.startswith("bad") or l.startswith("edge"))


def events_from_annotations(raw, event_id = "auto", regexp = "default",
                            use_rounding: bool = True, chunk_duration = None,
                            tol: float = 1e-8, verbose = None):
    """Port of mne.events_from_annotations for orig_time=None annotations.
    event_id may be a Dict[str, int] or None/"auto" (descriptions numbered
    from 1 in sorted order). Returns (events, event_id_)."""
    if not isinstance(chunk_duration, NoneType):
        compile_error("Typhon mne: chunk_duration is not supported")
    ann = raw.annotations
    out_id = Dict[str, int]()
    if len(ann) == 0:
        if isinstance(event_id, Dict):
            for k, v in event_id.items():
                out_id[k] = v
        return np.empty((0, 3), dtype=int), out_id
    use_re = not isinstance(regexp, NoneType)
    descs = sorted(ann.description)
    auto_n = 1
    for d in descs:
        if d in out_id:
            continue
        if use_re and not _desc_ok(d):
            continue
        if isinstance(event_id, Dict):
            if d in event_id:
                out_id[d] = event_id[d]
        else:
            out_id[d] = auto_n
            auto_n += 1
    sel = [i for i in range(len(ann)) if ann.description[i] in out_id]
    if len(sel) == 0 and use_re:
        raise ValueError("Could not find any of the events you specified.")
    sf = raw.info.sfreq
    ev = np.zeros((len(sel), 3), dtype=int)
    for k in range(len(sel)):
        i = sel[k]
        x = (ann.onset[i] - float(raw.times[0])) * sf
        ev[k, 0] = int(np.round(x)) if use_rounding else int(x)
        ev[k, 2] = out_id[ann.description[i]]
    return ev, out_id
