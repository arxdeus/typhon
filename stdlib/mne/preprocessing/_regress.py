# Ported from mne 1.13.2 (BSD-3-Clause)
#
# EOGRegression / regress_artifact (least-squares artifact regression) and
# annotate_muscle_zscore (high-frequency envelope z-score detection).

import numpy as np
from .._info import Info, _picks_to_idx
from .._containers import RawArray, EpochsArray
from ..annotations import Annotations
from ..filter import filter_data, next_fast_len

def _data3(inst):
    if isinstance(inst, RawArray):
        return inst._data.reshape((1, inst._data.shape[0], inst._data.shape[1]))
    else:
        return inst._data

class EOGRegression:
    picks: Optional[List[int]]
    exclude: List[str]
    picks_artifact_kind: str
    picks_artifact: Optional[List[int]]
    proj: bool
    coef_: np.ndarray[float, 2]
    info_: Info

    def __init__(self, picks = None, exclude = "bads", picks_artifact = "eog", proj: bool = True):
        self.picks = None if picks is None else [int(p) for p in picks]
        self.exclude = List[str]()
        if not isinstance(exclude, str):
            for e in exclude:
                self.exclude.append(str(e))
        self.picks_artifact_kind = "eog"
        self.picks_artifact = None
        if isinstance(picks_artifact, str):
            self.picks_artifact_kind = picks_artifact
        else:
            self.picks_artifact = [int(p) for p in picks_artifact]
        self.proj = proj
        self.coef_ = np.empty((0, 0))
        self.info_ = Info(List[str](), 1.0, List[str]())

    def _picks(self, info: Info):
        excl = list(info.bads) + self.exclude
        if self.picks is None:
            p = _picks_to_idx(info, None, "data", excl)
        else:
            p = _picks_to_idx(info, self.picks.__val__(), "data", excl)
        if self.picks_artifact is None:
            pa = _picks_to_idx(info, self.picks_artifact_kind, "all", ())
        else:
            pa = _picks_to_idx(info, self.picks_artifact.__val__(), "all", ())
        return p, pa

    def fit(self, inst):
        picks, pa = self._picks(inst.info)
        d3 = _data3(inst)
        art = d3[:, np.array(pa), :]
        ref = art - art.mean(axis=-1, keepdims=True)
        # (n_art, n_epochs * n_times)
        ref2 = np.ascontiguousarray(ref.transpose((1, 0, 2))).reshape((len(pa), -1))
        cov_ref = ref2 @ ref2.T
        coef = np.zeros((len(picks), len(pa)))
        for pi in range(len(picks)):
            this = d3[:, picks[pi], :]
            cd = (this - this.mean(axis=-1, keepdims=True)).reshape((1, -1))
            coef[pi] = np.linalg.solve(cov_ref, ref2 @ cd.T).T[0]
        self.coef_ = coef
        self.info_ = inst.info.copy()
        return self

    def apply(self, inst, copy: bool = True):
        out = inst.copy() if copy else inst
        picks, pa = self._picks(out.info)
        d3 = _data3(out)
        art = d3[:, np.array(pa), :]
        ref = art - art.mean(axis=-1, keepdims=True)
        for pi in range(len(picks)):
            for e in range(d3.shape[0]):
                corr = self.coef_[pi] @ ref[e]
                if isinstance(out, RawArray):
                    out._data[picks[pi], :] -= corr
                else:
                    out._data[e, picks[pi], :] -= corr
        return out

    def __repr__(self) -> str:
        return f"<EOGRegression | fitted to {self.coef_.shape[1]} artifact channel{'s' if self.coef_.shape[1] != 1 else ''}>"

def regress_artifact(inst, picks = None, *, exclude = "bads", picks_artifact = "eog",
                     betas = None, proj: bool = True, copy: bool = True, verbose = None):
    model = EOGRegression(picks, exclude, picks_artifact, proj)
    if betas is None:
        model.fit(inst)
    else:
        b = np.asarray(betas, dtype=float)
        p, pa = model._picks(inst.info)
        if b.shape[0] != len(p) or b.shape[1] != len(pa):
            raise ValueError(f"Invalid value for the 'betas.shape' parameter. Allowed values are ({len(p)}, {len(pa)}), but got {b.shape} instead.")
        model.coef_ = b
        model.info_ = inst.info.copy()
    return model.apply(inst, copy=copy), model.coef_

def _annotations_from_mask(times, mask: np.ndarray[bool, 1], name: str) -> Annotations:
    on = List[float]()
    du = List[float]()
    n = mask.shape[0]
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            e = min(j, times.shape[0] - 1)
            on.append(float(times[i]))
            du.append(float(times[e]) - float(times[i]))
            i = j
        else:
            i += 1
    return Annotations(on, du, [name for _ in range(len(on))])

def annotate_muscle_zscore(raw, threshold: float = 4.0, ch_type: Optional[str] = None,
                           min_length_good: Optional[float] = 0.1, filter_freq = (110.0, 140.0),
                           n_jobs = None, verbose = None):
    import scipy.signal
    info = raw.info
    kind = ""
    if ch_type is None:
        for k in ("mag", "grad", "eeg"):
            if kind == "" and k in info.ch_types:
                kind = k
        if kind == "":
            raise ValueError("No M/EEG channel types found, please specify a 'ch_type' or provide M/EEG sensor data.")
    else:
        kind = ch_type.__val__()
        if kind not in ("mag", "grad", "eeg"):
            raise ValueError(f"Invalid value for the 'ch_type' parameter. Allowed values are 'mag', 'grad', and 'eeg', but got {kind!r} instead.")
    pk = List[int]()
    for i in range(info.nchan):
        if info.ch_types[i] == kind and info.ch_names[i] not in info.bads:
            pk.append(i)
    data = raw._data[np.array(pk), :].copy()
    sf = info.sfreq
    data = filter_data(data, sf, float(filter_freq[0]), float(filter_freq[1]), fir_design="firwin",
                       pad="reflect_limited")
    nt = data.shape[1]
    nfft = next_fast_len(nt)
    env = np.empty(data.shape)
    for r in range(data.shape[0]):
        h = scipy.signal.hilbert(data[r].copy(), N=nfft)
        env[r] = np.abs(h[:nt])
    nch = env.shape[0]
    z = np.empty(env.shape)
    for r in range(nch):
        mu = env[r].mean()
        sd = env[r].std()
        z[r] = (env[r] - mu) / sd
    sc = z.sum(axis=0) / np.sqrt(float(nch))
    sc = filter_data(sc, sf, None, 4.0)
    art = sc > threshold
    mlg = 0.0 if min_length_good is None else min_length_good.__val__()
    min_samps = mlg * sf
    i = 0
    while i < nt:
        if not art[i]:
            j = i
            while j < nt and not art[j]:
                j += 1
            if float(j - i) < min_samps:
                for k in range(i, j):
                    art[k] = True
            i = j
        else:
            i += 1
    times = np.arange(nt).astype(float) / sf
    return _annotations_from_mask(times, art, "BAD_muscle"), sc
