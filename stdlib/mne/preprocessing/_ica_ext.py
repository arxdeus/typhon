# Ported from mne 1.13.2 (BSD-3-Clause)
#
# ica_find_ecg_events, ica_find_eog_events, corrmap and get_score_funcs.

import numpy as np
from ._events import qrs_detector, _find_eog_events
from .ica import ICA, _find_outliers
from scipy.stats import pearsonr, kendalltau, pointbiserialr, spearmanrho, chatterjeexi, bws_test


def ica_find_ecg_events(raw, ecg_source, event_id: int = 999, tstart: float = 0.0, l_freq: float = 5,
                        h_freq: float = 35, qrs_threshold = "auto", verbose = None):
    """Find ECG peaks from one selected ICA source (mne 1.13.2)."""
    ev = qrs_detector(raw.info.sfreq, np.asarray(ecg_source, dtype=float).ravel(), tstart=tstart,
                      thresh_value=qrs_threshold, l_freq=l_freq, h_freq=h_freq)
    n = ev.size
    out = np.zeros((n, 3), dtype=int)
    for i in range(n):
        out[i, 0] = int(ev[i]) + raw.first_samp
        out[i, 2] = event_id
    return out


def ica_find_eog_events(raw, eog_source = None, event_id: int = 998, l_freq: float = 1,
                        h_freq: float = 10, verbose = None):
    """Locate EOG artifacts from one selected ICA source (mne 1.13.2)."""
    es = np.asarray(eog_source, dtype=float).ravel()
    e2 = es.reshape((1, es.shape[0])).copy()
    return _find_eog_events(e2, event_id, float(l_freq), float(h_freq), raw.info.sfreq, raw.first_samp)


class _ScoreFunc:
    name: str

    def __init__(self, name: str):
        self.name = name

    def __call__(self, x, y):
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float).ravel()
        out = np.empty(xa.shape[0])
        for i in range(xa.shape[0]):
            a = xa[i].copy()
            if self.name == "pearsonr":
                out[i] = float(pearsonr(a, ya).statistic)
            elif self.name == "kendalltau":
                out[i] = float(kendalltau(a, ya).statistic)
            elif self.name == "pointbiserialr":
                out[i] = float(pointbiserialr(a, ya).statistic)
            elif self.name == "spearmanrho":
                out[i] = float(spearmanrho(a, ya).statistic)
            elif self.name == "chatterjeexi":
                out[i] = float(chatterjeexi(a, ya).statistic)
            elif self.name == "bws_test":
                out[i] = float(bws_test(a, ya).statistic)
            else:
                raise ValueError(f"{self.name} is not a valid score_func.")
        return out


class ScoreFuncs:
    names: List[str]

    def __init__(self):
        self.names = ["bws_test", "chatterjeexi", "estimated_cdf", "kendalltau", "pearsonr",
                      "pointbiserialr", "spearmanrho"]

    def keys(self) -> List[str]:
        return sorted(self.names)

    def __contains__(self, k: str) -> bool:
        return k in self.names

    def __getitem__(self, k: str):
        if k not in self.names:
            raise KeyError(k)
        return _ScoreFunc(k)


def get_score_funcs():
    """Score functions usable by ICA.score_sources (mne 1.13.2)."""
    return ScoreFuncs()


def _compute_corr(x: np.ndarray[float, 1], Y: np.ndarray[float, 2]) -> np.ndarray[float, 1]:
    """Pearson correlation of vector x with each row of Y (mne compute_corr)."""
    n = x.shape[0]
    xm = x - x.mean()
    xs = float(xm.std(ddof=1))
    out = np.empty(Y.shape[0])
    for i in range(Y.shape[0]):
        yr = Y[i] - Y[i].mean()
        ys = float(yr.std(ddof=1))
        out[i] = (float(np.dot(xm, yr)) / float(n - 1)) / (xs * ys)
    return out


class _MaxCorr:
    newtarget: np.ndarray[float, 1]
    median: float
    sim: float
    idxs: List[List[int]]
    found: bool

    def __init__(self, newtarget, median: float, sim: float, idxs: List[List[int]], found: bool):
        self.newtarget = newtarget
        self.median = median
        self.sim = sim
        self.idxs = idxs
        self.found = found


def _find_max_corrs(all_maps: List[np.ndarray[float, 2]], target: np.ndarray[float, 1],
                    threshold: float) -> _MaxCorr:
    abs_corrs = List[np.ndarray[float, 1]]()
    pols = List[np.ndarray[float, 1]]()
    for m in all_maps:
        c = _compute_corr(target, m)
        abs_corrs.append(np.abs(c))
        pols.append(np.sign(c))
    idxs = List[List[int]]()
    for ac in abs_corrs:
        sel = List[int]()
        if threshold <= 1.0:
            for i in range(ac.shape[0]):
                if ac[i] > threshold:
                    sel.append(i)
        else:
            sel = _find_outliers(ac, threshold)
        idxs.append(sel)
    am = List[float]()
    for s in range(len(all_maps)):
        for i in idxs[s]:
            am.append(float(abs_corrs[s][i]))
    if len(am) == 0:
        return _MaxCorr(np.zeros(target.shape[0]), 0.0, 0.0, idxs, False)
    med = float(np.median(np.array(am)))
    nt = np.zeros(target.shape[0])
    for s in range(len(all_maps)):
        for i in idxs[s]:
            v = all_maps[s][i]
            nrm = float(np.sqrt((v * v).sum()))
            nt = nt + v * (pols[s][i] / nrm)
    nt = nt / float(len(am))
    a = target - target.mean()
    b = nt - nt.mean()
    sim = abs(float(np.dot(a, b)) / (float(np.sqrt((a * a).sum())) * float(np.sqrt((b * b).sum()))))
    return _MaxCorr(nt, med, sim, idxs, True)


def corrmap(icas, template, threshold = "auto", label = None, ch_type: str = "eeg", *, sensors = True,
            show_names = False, contours = 6, outlines = "head", sphere = None, image_interp = "cubic",
            extrapolate = "auto", border = "mean", cmap = None, plot = False, show = False, verbose = None):
    """Find similar independent components across subjects by map similarity (mne 1.13.2).

    Plotting is not supported (plot must be False). Returns None; labels are stored in ica.labels_.
    """
    if plot:
        raise NotImplementedError("corrmap plotting is not supported")
    for ica in icas:
        if ica.ch_names != icas[0].ch_names:
            raise ValueError("Not all ICA instances have the same channel names. Corrmap requires all instances to have the same montage. Consider interpolating bad channels before running ICA.")
    thr = List[float]()
    if isinstance(threshold, str):
        for v in range(60, 95):
            thr.append(float(v) / 100.0)
    elif isinstance(threshold, float) or isinstance(threshold, int):
        thr.append(float(threshold))
    else:
        for t in threshold:
            thr.append(float(t))
    all_maps = List[np.ndarray[float, 2]]()
    for ica in icas:
        all_maps.append(np.ascontiguousarray(ica.get_components().T))
    target = np.zeros(1)
    if isinstance(template, np.ndarray):
        if template.ndim == 1 and template.shape[0] == all_maps[0].shape[1]:
            target = template.astype(float)
        else:
            raise ValueError("`template` must be a length-2 tuple or an array the size of the ICA maps.")
    else:
        if len(template) != 2:
            raise ValueError("`template` must be a length-2 tuple or an array the size of the ICA maps.")
        target = all_maps[template[0]][template[1]].copy()
    err = "No component detected using when z-scoring threshold " + str(thr) + ", consider using a more lenient threshold"
    best = 0
    bs = -1.0
    paths = List[_MaxCorr]()
    for t in thr:
        paths.append(_find_max_corrs(all_maps, target, t))
    for k in range(len(paths)):
        if paths[k].sim > bs:
            bs = paths[k].sim
            best = k
    first = paths[best]
    if not first.found:
        raise RuntimeError(err)
    paths2 = List[_MaxCorr]()
    for t in thr:
        paths2.append(_find_max_corrs(all_maps, first.newtarget, t))
    best = 0
    bs = -1.0
    for k in range(len(paths2)):
        if paths2[k].median > bs:
            bs = paths2[k].median
            best = k
    final = paths2[best]
    for ii in range(len(icas)):
        mc = final.idxs[ii]
        if not isinstance(label, NoneType):
            lab = str(label)
            if len(mc) > 0:
                cur = icas[ii].labels_.get(lab, List[int]())
                merged = Set[int]()
                for v in mc:
                    merged.add(v)
                for v in cur:
                    merged.add(v)
                icas[ii].labels_[lab] = list(merged)
            else:
                if lab not in icas[ii].labels_:
                    icas[ii].labels_[lab] = List[int]()
    return None
