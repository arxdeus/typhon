# Ported from mne 1.13.2 (BSD-3-Clause)
#
# Xdawn spatial filtering for evoked-response enhancement: per-class
# prototype responses (plain averages, or least-squares evokeds with a
# Toeplitz design when epochs overlap), regularized covariances, generalized
# eigendecomposition, filters/patterns and apply() back-projection.

import numpy as np
import scipy.linalg
from .._info import Info, _picks_to_idx
from .._containers import EpochsArray
from .._cov import _regularized_covariance, _compute_rank_data, _restr_mat, pinv

def _construct_signal_from_epochs(epochs: np.ndarray[float, 3], events: np.ndarray[int, 2],
                                  sfreq: float, tmin: float):
    ne, nc, nt = epochs.shape
    tmax = tmin + nt / sfreq
    start = int(np.min(events[:, 0])) + int(tmin * sfreq)
    stop = int(np.max(events[:, 0])) + int(tmax * sfreq) + 1
    raw = np.zeros((nc, stop - start))
    for i in range(ne):
        on = int(events[i, 0] - events[0, 0])
        raw[:, on:on + nt] = epochs[i]
    return raw

def _least_square_evoked(epochs: np.ndarray[float, 3], events0: np.ndarray[int, 2], tmin: float,
                         sfreq: float):
    ne, nc, nt = epochs.shape
    tmax = tmin + nt / sfreq
    events = events0.copy()
    shift = int(events[0, 0]) + int(tmin * sfreq)
    for i in range(ne):
        events[i, 0] -= shift
    raw = _construct_signal_from_epochs(epochs, events, sfreq, tmin)
    n_min = int(tmin * sfreq)
    n_max = int(tmax * sfreq)
    window = n_max - n_min
    ns = raw.shape[1]
    classes = np.unique(events[:, 2])
    toeps = List[np.ndarray[float, 2]]()
    for ci in range(classes.shape[0]):
        trig = np.zeros(ns)
        for i in range(ne):
            if events[i, 2] == classes[ci]:
                trig[int(events[i, 0]) + n_min] = 1.0
        toeps.append(scipy.linalg.toeplitz(trig[0:window], trig))
    X = np.concatenate(toeps, axis=0)
    predictor = pinv(X @ X.T) @ X
    ev = predictor @ raw.T
    nk = classes.shape[0]
    evokeds = List[np.ndarray[float, 2]]()
    for k in range(nk):
        evokeds.append(np.ascontiguousarray(ev[k * window:(k + 1) * window].T))
    return evokeds, toeps

class Xdawn:
    n_components: int
    reg_kind: str
    reg_value: float
    correct_overlap: str
    rank_full: bool
    filters_: Dict[str, np.ndarray[float, 2]]
    patterns_: Dict[str, np.ndarray[float, 2]]
    evokeds_: Dict[str, np.ndarray[float, 2]]
    event_id_: Dict[str, int]
    correct_overlap_: bool
    info_: Info
    picks_: List[int]

    def __init__(self, n_components: int = 2, signal_cov = None, correct_overlap = "auto",
                 reg = None, *, rank = "full"):
        if signal_cov is not None:
            compile_error("Xdawn(signal_cov=...) is not supported in Typhon mne")
        self.n_components = n_components
        if isinstance(reg, NoneType):
            self.reg_kind = "empirical"
            self.reg_value = 0.0
        elif isinstance(reg, str):
            if reg not in ("empirical", "ledoit_wolf", "oas"):
                raise ValueError(f"reg={reg!r} is not supported in Typhon mne")
            self.reg_kind = reg
            self.reg_value = 0.0
        else:
            self.reg_kind = "shrinkage"
            self.reg_value = float(reg)
        if isinstance(correct_overlap, str):
            if correct_overlap != "auto":
                raise ValueError("Invalid value for the 'correct_overlap' parameter.")
            self.correct_overlap = "auto"
        else:
            self.correct_overlap = "true" if correct_overlap else "false"
        if isinstance(rank, str):
            if rank != "full" and rank != "info":
                raise ValueError(f"rank={rank!r} is not supported in Typhon mne")
            self.rank_full = True
        else:
            compile_error("Xdawn(rank=...) only supports 'full'/'info' in Typhon mne")
        self.filters_ = Dict[str, np.ndarray[float, 2]]()
        self.patterns_ = Dict[str, np.ndarray[float, 2]]()
        self.evokeds_ = Dict[str, np.ndarray[float, 2]]()
        self.event_id_ = Dict[str, int]()
        self.correct_overlap_ = False
        self.info_ = Info(List[str](), 1.0, List[str]())
        self.picks_ = List[int]()

    def fit(self, epochs, y = None):
        pk = _picks_to_idx(epochs.info, None, "data", ())
        info = epochs.info._pick(pk)
        X = np.ascontiguousarray(epochs._data[:, np.array(pk), :])
        yv = epochs.events[:, 2].copy()
        self.event_id_ = dict(epochs.event_id)
        ovl = False
        if self.correct_overlap == "auto":
            isi = np.diff(np.sort(epochs.events[:, 0]))
            window = int((epochs.tmax - epochs.tmin) * info.sfreq)
            ovl = int(isi.min()) < window
        else:
            ovl = self.correct_overlap == "true"
        if epochs.baseline is not None and ovl:
            raise ValueError("Cannot apply correct_overlap if epochs were baselined.")
        self.correct_overlap_ = ovl
        nch = X.shape[1]
        rank = _compute_rank_data(np.hstack([X[i] for i in range(X.shape[0])]), info, "full")
        signal_cov = _regularized_covariance(np.ascontiguousarray(np.hstack([X[i] for i in range(X.shape[0])])),
                                             self.reg_kind, self.reg_value, info, rank)
        classes = np.unique(yv)
        evokeds = List[np.ndarray[float, 2]]()
        toeps = List[np.ndarray[float, 2]]()
        use_toep = False
        if ovl:
            evokeds, toeps = _least_square_evoked(X, epochs.events, epochs.tmin, info.sfreq)
            use_toep = True
        else:
            for ci in range(classes.shape[0]):
                sel = List[int]()
                for i in range(yv.shape[0]):
                    if yv[i] == classes[ci]:
                        sel.append(i)
                evokeds.append(X[np.array(sel)].mean(axis=0))
        filts = List[np.ndarray[float, 2]]()
        pats = List[np.ndarray[float, 2]]()
        for k in range(len(evokeds)):
            evo = evokeds[k] @ toeps[k] if use_toep else evokeds[k]
            evo_cov = _regularized_covariance(np.ascontiguousarray(evo), self.reg_kind, self.reg_value, info, rank)
            evals, evecs = scipy.linalg.eigh(evo_cov, signal_cov)
            order = np.argsort(evals)[::-1]
            evecs = np.ascontiguousarray(evecs[:, order])
            for c in range(evecs.shape[1]):
                evecs[:, c] /= np.sqrt(float((evecs[:, c] ** 2).sum()))
            pat = pinv(np.ascontiguousarray(evecs.T))
            filts.append(np.ascontiguousarray(evecs[:, :nch].T))
            pats.append(np.ascontiguousarray(pat[:, :nch].T))
        # order by event_id values (dict iteration order follows event_id)
        names = List[str]()
        vals = List[int]()
        for kname, v in epochs.event_id.items():
            names.append(kname)
            vals.append(v)
        idx = np.argsort(np.array(vals))
        self.filters_ = Dict[str, np.ndarray[float, 2]]()
        self.patterns_ = Dict[str, np.ndarray[float, 2]]()
        self.evokeds_ = Dict[str, np.ndarray[float, 2]]()
        for q in range(len(names)):
            src = int(idx[q])
            self.filters_[names[q]] = filts[src]
            self.patterns_[names[q]] = pats[src]
            self.evokeds_[names[q]] = evokeds[src]
        self.info_ = info
        self.picks_ = pk
        return self

    def transform(self, inst):
        X = np.empty((0, 0, 0))
        if isinstance(inst, EpochsArray):
            X = inst._data
        else:
            X = np.asarray(inst, dtype=float)
        fl = List[np.ndarray[float, 2]]()
        for _, f in self.filters_.items():
            fl.append(f[:self.n_components])
        F = np.concatenate(fl, axis=0)
        out = np.empty((X.shape[0], F.shape[0], X.shape[2]))
        for e in range(X.shape[0]):
            out[e] = F @ X[e]
        return out

    def _pick_sources(self, data: np.ndarray[float, 2], include, exclude: List[int], eid: str):
        filters = self.filters_[eid]
        patterns = self.patterns_[eid]
        n = filters.shape[0]
        keep = List[int]()
        if include is not None:
            for i in include:
                keep.append(int(i))
        else:
            for i in range(n):
                if i not in exclude:
                    keep.append(i)
        ka = np.array(keep, dtype=int)
        sources = filters @ data
        return patterns[ka].T @ sources[ka]

    def apply(self, inst, event_id = None, include = None, exclude = None):
        n_filters = 0
        for _, f in self.filters_.items():
            n_filters = f.shape[0]
        excl = list(range(self.n_components, n_filters))
        if exclude is not None:
            for e in exclude:
                if int(e) not in excl:
                    excl.append(int(e))
        ids = List[str]()
        if event_id is None:
            for k, _ in self.event_id_.items():
                ids.append(k)
        else:
            for k, _ in event_id.items():
                ids.append(k)
        pa = np.array(self.picks_)
        ne = inst._data.shape[0]
        nt = inst._data.shape[2]
        d3 = inst._data[:, pa, :]
        data = np.ascontiguousarray(np.hstack([d3[i] for i in range(ne)]))
        out = Dict[str, EpochsArray]()
        for eid in ids:
            dr = self._pick_sources(data, include, excl, eid)
            ep = inst.copy()
            for i in range(ne):
                for c in range(len(self.picks_)):
                    ep._data[i, self.picks_[c], :] = dr[c, i * nt:(i + 1) * nt]
            out[eid] = ep
        return out
