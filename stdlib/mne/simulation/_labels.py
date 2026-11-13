# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# Label (minimal), simulate_stc, SourceSimulator and add_eog / add_ecg.
# Deviations: add_eog/add_ecg take the sphere fit explicitly (r0, R) because
# fit_sphere_to_headshape is not available; head_pos is not supported.
# add_chpi is not ported (Typhon Info carries no cHPI information).

import numpy as np
from .._info import Info
from .._containers import RawArray
from ..bem import make_sphere_model
from ..forward import make_forward_solution, convert_forward_solution
from ..source_estimate import _STC
from ..source_space import SourceSpaces, setup_volume_source_space


class Label:
    """Minimal mne.Label: vertices, optional values, hemisphere."""
    vertices: np.ndarray[int, 1]
    values: np.ndarray[float, 1]
    hemi: str
    name: str

    def __init__(self, vertices, values = None, hemi: str = "lh", name: str = "", pos = None,
                 subject = None):
        self.vertices = np.asarray(vertices).astype(int)
        if isinstance(values, NoneType):
            self.values = np.ones(self.vertices.shape[0])
        else:
            self.values = np.asarray(values, dtype=float)
        self.hemi = hemi
        self.name = name


def _nearest(xhs, rr):
    out = np.zeros(rr.shape[0], dtype=int)
    for i in range(rr.shape[0]):
        best = np.inf
        for j in range(xhs.shape[0]):
            d = float(np.sum((xhs[j] - rr[i]) ** 2))
            if d < best:
                best = d
                out[i] = j
    return out


def _src_sel(src: SourceSpaces, hemi_ind: int, label: Label) -> List[int]:
    """np.intersect1d(src[hemi]['vertno'], label.vertices), or the nearest used
    vertices when empty."""
    s = src.spaces[hemi_ind]
    lv = Set[int]([int(v) for v in label.vertices])
    sel = sorted([int(v) for v in s.vertno if int(v) in lv])
    if len(sel) == 0:
        used = [int(v) for v in s.vertno]
        xhs = s.rr[np.array(used, dtype=int)]
        rr = s.rr[label.vertices]
        near = _nearest(xhs, rr)
        uniq = sorted(list(Set[int]([int(k) for k in near])))
        sel = [used[k] for k in uniq]
    return sel


def simulate_stc(src: SourceSpaces, labels: List[Label], stc_data, tmin: float, tstep: float,
                 value_fun = None, allow_overlap: bool = False):
    """mne.simulation.simulate_stc (two-hemisphere source spaces)."""
    sd = np.asarray(stc_data, dtype=float)
    if len(labels) != sd.shape[0]:
        raise ValueError("labels and stc_data must have the same length")
    nt = sd.shape[1]
    vert = [List[int](), List[int]()]
    rows = [List[np.ndarray[float, 1]](), List[np.ndarray[float, 1]]()]
    for i in range(len(labels)):
        lab = labels[i]
        h = 0 if lab.hemi == "lh" else 1
        sel = _src_sel(src, h, lab)
        block = List[np.ndarray[float, 1]]()
        for v in sel:
            if isinstance(value_fun, NoneType):
                block.append(sd[i].copy())
            else:
                k = int(np.searchsorted(lab.vertices, v))
                block.append(float(value_fun(lab.values[k])) * sd[i])
        for b in range(len(sel)):
            v = sel[b]
            found = -1
            if allow_overlap:
                for q in range(len(vert[h])):
                    if vert[h][q] == v:
                        found = q
            if found >= 0:
                rows[h][found] = rows[h][found] + block[b]
            else:
                vert[h].append(v)
                rows[h].append(block[b])
    for h in range(2):
        if len(vert[h]) != len(Set[int](vert[h])):
            raise RuntimeError("Labels had overlaps, they must be non-overlapping")
    vs = List[np.ndarray[int, 1]]()
    datas = List[np.ndarray[float, 1]]()
    for h in range(2):
        order = sorted(range(len(vert[h])), key=lambda k: vert[h][k])
        vs.append(np.array([vert[h][k] for k in order], dtype=int))
        for k in order:
            datas.append(rows[h][k])
    out = np.zeros((len(datas), nt))
    for k in range(len(datas)):
        out[k] = datas[k]
    return _STC[np.ndarray[float, 2]](out, vs, tmin, tstep, None, "surface")


class SourceSimulator:
    """mne.simulation.SourceSimulator."""
    src: SourceSpaces
    tstep: float
    first_samp: int
    labels: List[Label]
    waveforms: List[np.ndarray[float, 1]]
    events: List[List[int]]
    duration_: Optional[float]

    def __init__(self, src: SourceSpaces, tstep: float = 1e-3, duration = None, first_samp: int = 0):
        if not isinstance(duration, NoneType) and duration < tstep:
            raise ValueError("duration must be None or >= tstep.")
        self.src = src
        self.tstep = tstep
        self.first_samp = first_samp
        self.labels = List[Label]()
        self.waveforms = List[np.ndarray[float, 1]]()
        self.events = List[List[int]]()
        self.duration_ = None if isinstance(duration, NoneType) else float(duration)

    def _last(self, i: int) -> int:
        return self.events[i][0] + self.waveforms[i].shape[0] - 1

    @property
    def n_times(self) -> int:
        if self.duration_ is not None:
            return int(self.duration_.__val__() / self.tstep)
        ls = self.first_samp
        for i in range(len(self.events)):
            ls = max(ls, self._last(i))
        return ls - self.first_samp + 1

    @property
    def duration(self) -> float:
        if self.duration_ is not None:
            return self.duration_.__val__()
        return self.n_times * self.tstep

    @property
    def last_samp(self) -> int:
        return self.first_samp + self.n_times - 1

    def add_data(self, label: Label, waveform, events):
        ev = np.asarray(events).astype(int)
        w = np.asarray(waveform, dtype=float)
        n = ev.shape[0]
        for i in range(n):
            self.labels.append(label)
            self.events.append([int(ev[i, 0]), int(ev[i, 1]), int(ev[i, 2])])
            if w.ndim == 1:
                self.waveforms.append(w.copy())
            else:
                if w.shape[0] != n:
                    raise ValueError(f"Number of waveforms and events should match or there should be a single waveform ({w.shape[0]} != {n}).")
                self.waveforms.append(w[i].copy())

    def get_stim_channel(self, start_sample = None, stop_sample = None):
        a = self.first_samp if isinstance(start_sample, NoneType) else int(start_sample)
        b = a + self.n_times - 1 if isinstance(stop_sample, NoneType) else int(stop_sample)
        if b < a:
            raise ValueError("Argument start_sample must be >= stop_sample.")
        out = np.zeros(b - a + 1, dtype=int)
        for e in self.events:
            if e[0] >= a and e[0] < b:
                out[e[0] - a] = e[2]
        return out

    def get_stc(self, start_sample = None, stop_sample = None):
        if len(self.labels) == 0:
            raise ValueError("No simulation parameters were found. Please use function add_data to add simulation parameters.")
        a = self.first_samp if isinstance(start_sample, NoneType) else int(start_sample)
        b = a + self.n_times - 1 if isinstance(stop_sample, NoneType) else int(stop_sample)
        if b < a:
            raise ValueError("start_sample must be >= stop_sample.")
        n = b - a + 1
        data = np.zeros((len(self.labels), n))
        for i in range(len(self.labels)):
            ws = self.events[i][0]
            we = self._last(i)
            if we >= a and ws <= b:
                for s in range(max(ws, a), min(we, b) + 1):
                    data[i, s - a] += self.waveforms[i][s - ws]
        return simulate_stc(self.src, self.labels, data, (a - self.first_samp) * self.tstep,
                            self.tstep, None, True)


def _exg_forward(info: Info, rr, nn, r0, head_radius, picks: List[int]):
    bem = make_sphere_model(r0, head_radius, None, (0.97, 0.98, 0.99, 1.0), (0.33, 1.0, 0.004, 0.33))
    src = setup_volume_source_space(pos={"rr": rr, "nn": nn}, sphere_units="mm")
    inf = info._pick(picks)
    inf.bads = List[str]()
    fw = make_forward_solution(inf, None, src, bem, mindist=5.0)
    fw = convert_forward_solution(fw, surf_ori=True, force_fixed=True)
    return fw


def _add_exg(raw: RawArray, kind: str, r: float, r0, rng):
    info = raw.info
    sf = info.sfreq
    nt = raw._data.shape[1]
    times = raw.times
    meeg = [i for i in range(info.nchan) if info.ch_types[i] == "mag" or info.ch_types[i] == "grad"
            or info.ch_types[i] == "eeg"]
    meg = [i for i in meeg if info.ch_types[i] != "eeg"]
    exg_data = np.zeros((1, nt))
    picks = meeg
    ch_type = "eog"
    rr = np.zeros((1, 3))
    nn = np.zeros((1, 3))
    if kind == "blink":
        rr = np.zeros((2, 3))
        nn = np.zeros((2, 3))
        c = np.cos(np.pi / 3.0)
        s = np.sin(np.pi / 3.0)
        rr[0] = [c, s, 0.0]
        rr[1] = [-c, s, 0.0]
        for k in range(2):
            rr[k] = rr[k] / np.sqrt(np.sum(rr[k] ** 2)) * (0.96 * r) + np.array([r0[0], r0[1], r0[2]])
            nn[k] = [0.0, 0.0, 1.0]
        rate = (1 + np.cos(2 * np.pi * 1.0 / 60.0 * times)) / 2.0
        rate = rate * (12.5 / 60.0) + 4.5 / 60.0
        bd = (rng.uniform(size=nt) < rate / sf).astype(float)
        bd = bd * (rng.uniform(size=nt) + 0.5)
        kern = np.hanning(int(0.25 * sf))
        exg_data[0] = np.convolve(bd, kern, "same") * 1e-7
    else:
        ch_type = "ecg"
        picks = meg
        if len(meg) == 0:
            raise RuntimeError("Can only add ECG artifacts if MEG data channels are present")
        rr[0] = [-r, 0.0, -3 * r]
        nn[0] = [1.0, 0.0, 0.0]
        max_beats = int(np.ceil(times[nt - 1] * 80.0 / 60.0))
        cum = np.cumsum(rng.uniform(60.0 / 80.0, 60.0 / 40.0, max_beats) * sf).astype(int)
        cd = np.zeros(nt)
        for v in cum:
            if v < nt:
                cd[v] = 1.0
        kern = np.concatenate([2 * np.hanning(int(0.04 * sf)), -0.3 * np.hanning(int(0.05 * sf)),
                               0.2 * np.hanning(int(0.26 * sf))])
        exg_data[0] = np.convolve(cd, kern, "same") * 15e-8
    noise = rng.normal(scale=5e-6, size=nt)
    chs = [i for i in range(info.nchan) if info.ch_types[i] == ch_type]
    if len(chs) >= 1:
        raw._data[chs[len(chs) - 1]] = exg_data[0] * 1e3 + noise
    hr = r if kind == "blink" else None
    fw = _exg_forward(info, rr, nn, r0, hr, picks)
    g = fw.sol
    nsrc = g.shape[1]
    for k in range(len(picks)):
        for q in range(nsrc):
            raw._data[picks[k]] += g[k, q] * exg_data[0]
    return raw


def add_eog(raw: RawArray, head_pos = None, interp: str = "cos2", n_jobs = None, verbose = None, *,
            rng = None, random_state = None, r: float = 0.09, r0 = (0.0, 0.0, 0.04)):
    """mne.simulation.add_eog (blink); the sphere fit is given by r / r0."""
    from . import _get_rng
    return _add_exg(raw, "blink", r, r0, _get_rng(rng, random_state))


def add_ecg(raw: RawArray, head_pos = None, interp: str = "cos2", n_jobs = None, verbose = None, *,
            rng = None, random_state = None, r: float = 0.09, r0 = (0.0, 0.0, 0.04)):
    """mne.simulation.add_ecg; the sphere fit is given by r / r0."""
    from . import _get_rng
    return _add_exg(raw, "ecg", r, r0, _get_rng(rng, random_state))
