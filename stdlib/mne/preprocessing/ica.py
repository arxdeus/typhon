# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne.preprocessing.ICA: standardization pre-whitening, whitened PCA (mne's
# _PCA), FastICA (sklearn port, whiten=False) or Infomax unmixing, component
# sorting by explained variance, sources, apply(exclude) and correlation-based
# EOG/ECG component detection.

import numpy as np
import scipy.linalg as sl
from .._info import Info, _picks_to_idx, _picks_by_type
from .._containers import RawArray, EpochsArray
from ..filter import filter_data
from ._infomax import infomax

def _exp_var_ncomp(var, n: float) -> Tuple[int, float]:
    cvar = np.cumsum(np.asarray(var, dtype=float))
    cvar = cvar / cvar[-1]
    c = 0
    for i in range(cvar.shape[0]):
        if cvar[i] <= n:
            c += 1
    k = min(c + 1, cvar.shape[0])
    return k, float(cvar[k - 1])

def _svd_flip_u(U, V):
    k = U.shape[1]
    for j in range(k):
        best = 0
        for i in range(U.shape[0]):
            if abs(U[i, j]) > abs(U[best, j]):
                best = i
        s = 1.0 if U[best, j] >= 0.0 else -1.0
        if U[best, j] == 0.0:
            s = 0.0
        for i in range(U.shape[0]):
            U[i, j] *= s
        for i in range(V.shape[1]):
            V[j, i] *= s

def _zscore_masked(x: np.ndarray[float, 1], mask: np.ndarray[bool, 1]):
    s = 0.0
    c = 0
    for i in range(x.shape[0]):
        if not mask[i]:
            s += x[i]
            c += 1
    mu = s / c
    q = 0.0
    for i in range(x.shape[0]):
        if not mask[i]:
            q += (x[i] - mu) ** 2
    sd = np.sqrt(q / c)
    return (x - mu) / sd

def _find_outliers(X, threshold: float = 3.0, max_iter: int = 2, tail: int = 0) -> List[int]:
    x = np.asarray(X, dtype=float)
    n = x.shape[0]
    mask = np.zeros(n, bool)
    for _ in range(max_iter):
        z = _zscore_masked(x, mask)
        anyb = False
        for i in range(n):
            if mask[i]:
                continue
            v = z[i]
            if tail == 0:
                v = abs(v)
            elif tail == -1:
                v = -v
            if v > threshold:
                mask[i] = True
                anyb = True
        if not anyb:
            break
    out = List[int]()
    for i in range(n):
        if mask[i]:
            out.append(i)
    return out

def _pearson_rows(sources, target) -> np.ndarray[float, 1]:
    t = np.asarray(target, dtype=float).ravel()
    tm = t - t.mean()
    tn = np.sqrt((tm * tm).sum())
    k = sources.shape[0]
    out = np.empty(k)
    for i in range(k):
        s = sources[i] - sources[i].mean()
        r = float((s * tm).sum()) / (np.sqrt(float((s * s).sum())) * tn)
        out[i] = min(max(r, -1.0), 1.0)
    return out

class ICA:
    n_components: Optional[float]
    n_components_int: bool
    method: str
    random_state: Optional[int]
    max_iter: int
    fit_params: Dict[str, float]
    exclude: List[int]
    labels_: Dict[str, List[int]]
    ch_names: List[str]
    info: Info
    n_components_: int
    n_iter_: int
    n_samples_: int
    pre_whitener_: np.ndarray[float, 2]
    pca_mean_: np.ndarray[float, 1]
    pca_components_: np.ndarray[float, 2]
    pca_explained_variance_: np.ndarray[float, 1]
    unmixing_matrix_: np.ndarray[float, 2]
    mixing_matrix_: np.ndarray[float, 2]
    current_fit: str
    n_pca_components: Optional[float]

    def __init__(self, n_components = None, *, noise_cov = None, random_state: Optional[int] = None,
                 method: str = "fastica", fit_params: Optional[Dict[str, float]] = None,
                 max_iter = "auto", allow_ref_meg: bool = False, verbose = None):
        if method not in ("fastica", "infomax", "extended-infomax"):
            raise ValueError(f"Invalid value for the 'method' parameter. Allowed values are 'fastica', 'infomax', 'extended-infomax' (picard is not available), but got {method!r} instead.")
        if noise_cov is not None:
            compile_error("ICA(noise_cov=...) is not supported in Typhon mne")
        self.n_components_int = True
        if n_components is None:
            self.n_components = None
        elif isinstance(n_components, float):
            if not (0.0 < n_components < 1.0):
                raise ValueError(f"Selecting ICA components by explained variance needs values between 0.0 and 1.0 (exclusive), got n_components={n_components}")
            self.n_components = n_components
            self.n_components_int = False
        else:
            if n_components == 1:
                raise ValueError("Selecting one component with n_components=1 is not supported")
            self.n_components = float(n_components)
        self.method = method
        self.random_state = random_state
        if isinstance(max_iter, str):
            if max_iter != "auto":
                raise ValueError("max_iter must be 'auto' when str")
            self.max_iter = 1000 if method == "fastica" else 500
        else:
            self.max_iter = max_iter
        fp = Dict[str, float]()
        if fit_params is not None:
            for k, v in fit_params.__val__().items():
                fp[k] = v
        self.fit_params = fp
        self.exclude = List[int]()
        self.labels_ = Dict[str, List[int]]()
        self.ch_names = List[str]()
        self.info = Info(List[str](), 1.0, List[str]())
        self.n_components_ = 0
        self.n_iter_ = 0
        self.n_samples_ = 0
        self.pre_whitener_ = np.empty((0, 1))
        self.pca_mean_ = np.empty(0)
        self.pca_components_ = np.empty((0, 0))
        self.pca_explained_variance_ = np.empty(0)
        self.unmixing_matrix_ = np.empty((0, 0))
        self.mixing_matrix_ = np.empty((0, 0))
        self.current_fit = "unfitted"
        self.n_pca_components = None

    def __repr__(self) -> str:
        if self.current_fit == "unfitted":
            return f"<ICA | no decomposition, method: {self.method}>"
        return f"<ICA | {self.current_fit} decomposition, method: {self.method} (fit in {self.n_iter_} iterations on {self.n_samples_} samples), {self.n_components_} ICA components ({self.pca_components_.shape[0]} PCA components available), channel types: eeg, no sources marked for exclusion>"

    # ---------------------------------------------------------------
    def _compute_pre_whitener(self, data):
        pw = np.empty((data.shape[0], 1))
        for _, picks_ in _picks_by_type(self.info, False, False):
            pw[np.array(picks_), 0] = np.std(data[np.array(picks_)])
        for kind in ("ref_meg", "eog"):
            sel = List[int]()
            for i in range(self.info.nchan):
                if self.info.ch_types[i] == kind:
                    sel.append(i)
            if len(sel) > 0:
                pw[np.array(sel), 0] = np.std(data[np.array(sel)])
        self.pre_whitener_ = pw

    def _fit(self, data0: np.ndarray[float, 2], fit_type: str):
        if not np.isfinite(data0).all():
            raise ValueError("Input data contains non-finite values (NaN/Inf). ")
        self._compute_pre_whitener(data0)
        data = data0 / self.pre_whitener_
        # _PCA(n_components=None, whiten=True).fit_transform(data.T)
        X = np.ascontiguousarray(data.T)
        ns = X.shape[0]
        nf = X.shape[1]
        mean = X.mean(axis=0)
        X = X - mean
        U, S, V = sl.svd(X, full_matrices=False)
        U = np.ascontiguousarray(U)
        V = np.ascontiguousarray(V)
        _svd_flip_u(U, V)
        ncomp = min(ns, nf)
        ev = (S * S) / float(ns - 1)
        evr = ev / ev.sum()
        Xw = U[:, :ncomp] * np.sqrt(float(ns - 1))
        n_pca = ncomp
        if self.n_components is None:
            k, _ = _exp_var_ncomp(evr, 0.999999)
            self.n_components_ = min(n_pca, k)
        elif not self.n_components_int:
            k, evv = _exp_var_ncomp(evr, self.n_components.__val__())
            if k == 1:
                raise RuntimeError(f"One PCA component captures most of the explained variance ({100 * evv}%), your threshold results in 1 component. You should select a higher value.")
            self.n_components_ = k
        else:
            self.n_components_ = int(self.n_components.__val__())
        if self.n_components_ > n_pca:
            raise RuntimeError(f"n_components={self.n_components} requires {self.n_components_} PCA values but only {n_pca} are available")
        self.pca_mean_ = mean
        self.pca_components_ = V[:ncomp].copy()
        self.pca_explained_variance_ = ev[:ncomp].copy()
        nc = self.n_components_
        sub = np.ascontiguousarray(Xw[:, :nc])
        if self.method == "fastica":
            from sklearn.decomposition import FastICA
            if self.random_state is None:
                raise ValueError("ICA(method='fastica') needs an integer random_state in Typhon")
            fi = FastICA(whiten=False, random_state=self.random_state.__val__(),
                         max_iter=int(self.fit_params.get("max_iter", float(self.max_iter))),
                         tol=self.fit_params.get("tol", 1e-4))
            fi.fit(sub)
            self.unmixing_matrix_ = fi.components_
            self.n_iter_ = fi.n_iter_
        else:
            ext = self.method == "extended-infomax" or self.fit_params.get("extended", 0.0) != 0.0
            seed = 0 if self.random_state is None else self.random_state.__val__()
            W, it = infomax(sub, extended=ext, max_iter=int(self.fit_params.get("max_iter", float(self.max_iter))),
                            random_state=seed)
            self.unmixing_matrix_ = W
            self.n_iter_ = it
        norms = np.sqrt(self.pca_explained_variance_[:nc])
        for i in range(nc):
            if norms[i] == 0.0:
                norms[i] = 1.0
        self.unmixing_matrix_ = self.unmixing_matrix_ / norms
        self.mixing_matrix_ = sl.pinv(self.unmixing_matrix_)
        self.current_fit = fit_type

    def _get_picks(self, info: Info) -> List[int]:
        pk = List[int]()
        for name in self.ch_names:
            if name not in info.ch_names:
                raise RuntimeError(f"Data doesn't match fitted data: {len(self.ch_names)} channels fitted but channel {name} is missing.")
            pk.append(info.ch_names.index(name))
        return pk

    def fit(self, inst, picks = None, start: Optional[int] = None, stop: Optional[int] = None,
            decim: Optional[int] = None, reject = None, flat = None, tstep: float = 2.0,
            reject_by_annotation: bool = True, verbose = None):
        pk = _picks_to_idx(inst.info, picks, "data", "bads")
        if self.n_components is not None and self.n_components_int:
            if int(self.n_components.__val__()) > len(pk):
                raise ValueError(f"ica.n_components ({int(self.n_components.__val__())}) cannot be greater than len(picks) ({len(pk)})")
        self.info = inst.info._pick(pk)
        self.ch_names = list(self.info.ch_names)
        self.exclude = List[int]()
        self.labels_ = Dict[str, List[int]]()
        if isinstance(inst, RawArray):
            s = 0 if start is None else start.__val__()
            e = inst.n_times if stop is None else stop.__val__()
            data = inst._data[np.array(pk), s:e].copy()
            if decim is not None:
                data = np.ascontiguousarray(data[:, ::decim.__val__()])
            self.n_samples_ = data.shape[1]
            self._fit(data, "raw")
        else:
            d3 = inst._data[:, np.array(pk), :]
            if decim is not None:
                d3 = d3[:, :, ::decim.__val__()]
            data = np.ascontiguousarray(np.hstack([d3[i] for i in range(d3.shape[0])]))
            self.n_samples_ = data.shape[1]
            self._fit(data, "epochs")
        # sort components by explained variance (_ica_explained_variance)
        src = self._sources_2d(inst)
        nch = src.shape[0]
        nsm = src.shape[1]
        var = (self.mixing_matrix_ ** 2).sum(axis=0) * (src ** 2).sum(axis=1) / float(nch * nsm - 1)
        order = np.argsort(var, kind="stable")[::-1].copy()
        self.mixing_matrix_ = np.ascontiguousarray(self.mixing_matrix_[:, order])
        self.unmixing_matrix_ = np.ascontiguousarray(self.unmixing_matrix_[order, :])
        return self

    # ---------------------------------------------------------------
    def _transform(self, data: np.ndarray[float, 2]):
        d = data / self.pre_whitener_
        d = d - self.pca_mean_[:, None]
        pca_data = self.unmixing_matrix_ @ self.pca_components_[:self.n_components_]
        return pca_data @ d

    def _sources_2d(self, inst):
        pk = self._get_picks(inst.info)
        if isinstance(inst, RawArray):
            return self._transform(inst._data[np.array(pk), :].copy())
        else:
            d3 = inst._data[:, np.array(pk), :]
            return self._transform(np.ascontiguousarray(np.hstack([d3[i] for i in range(d3.shape[0])])))

    def _ica_names(self) -> List[str]:
        out = List[str]()
        for i in range(self.n_components_):
            out.append(f"ICA{i:03d}")
        return out

    def get_sources(self, inst, add_channels = None, start: Optional[int] = None,
                    stop: Optional[int] = None):
        if self.current_fit == "unfitted":
            raise RuntimeError("No fit available. Please fit ICA.")
        from .._info import create_info
        info = create_info(self._ica_names(), inst.info.sfreq, "misc")
        if isinstance(inst, RawArray):
            s = 0 if start is None else start.__val__()
            e = inst.n_times if stop is None else stop.__val__()
            pk = self._get_picks(inst.info)
            src = self._transform(inst._data[np.array(pk), s:e].copy())
            return RawArray(src, info, inst.first_samp + s)
        else:
            src = self._sources_2d(inst)
            ne = inst._data.shape[0]
            nt = inst._data.shape[2]
            out = np.empty((ne, self.n_components_, nt))
            for i in range(ne):
                out[i] = src[:, i * nt:(i + 1) * nt]
            return EpochsArray(out, info, inst.events, float(inst.times[0]), inst.event_id)

    def get_components(self):
        return (self.mixing_matrix_[:, :self.n_components_].T @ self.pca_components_[:self.n_components_]).T

    def _pick_sources(self, data: np.ndarray[float, 2], include, exclude, n_pca_components):
        d = data / self.pre_whitener_
        excl = list(self.exclude)
        if exclude is not None:
            for e in exclude:
                if int(e) not in excl:
                    excl.append(int(e))
        npc = self.pca_components_.shape[0]
        if n_pca_components is not None:
            npc = int(n_pca_components)
            if npc < self.n_components_:
                npc = self.n_components_
        nc = self.n_components_
        d = d - self.pca_mean_[:, None]
        keep = List[int]()
        if include is not None and len(include) > 0:
            for i in sorted(set([int(v) for v in include])):
                keep.append(i)
        else:
            for i in range(nc):
                if i not in excl:
                    keep.append(i)
        for i in range(nc, npc):
            keep.append(i)
        pcs = self.pca_components_[:npc]
        unm = np.eye(npc)
        unm[:nc, :nc] = self.unmixing_matrix_
        unm = unm @ pcs
        mix = np.eye(npc)
        mix[:nc, :nc] = self.mixing_matrix_
        mix = pcs.T @ mix
        ka = np.array(keep, dtype=int)
        proj = mix[:, ka] @ unm[ka, :]
        d = proj @ d
        d = d + self.pca_mean_[:, None]
        return d * self.pre_whitener_

    def apply(self, inst, include = None, exclude = None, n_pca_components = None,
              start: Optional[int] = None, stop: Optional[int] = None, *, on_baseline: str = "warn",
              verbose = None):
        pk = self._get_picks(inst.info)
        pa = np.array(pk)
        if isinstance(inst, RawArray):
            s = 0 if start is None else start.__val__()
            e = inst.n_times if stop is None else stop.__val__()
            d = self._pick_sources(inst._data[pa, s:e].copy(), include, exclude, n_pca_components)
            for k in range(len(pk)):
                inst._data[pk[k], s:e] = d[k]
        else:
            ne = inst._data.shape[0]
            nt = inst._data.shape[2]
            d3 = inst._data[:, pa, :]
            d = self._pick_sources(np.ascontiguousarray(np.hstack([d3[i] for i in range(ne)])),
                                   include, exclude, n_pca_components)
            for i in range(ne):
                for k in range(len(pk)):
                    inst._data[i, pk[k], :] = d[k, i * nt:(i + 1) * nt]
        return inst

    # ---------------------------------------------------------------
    def score_sources(self, inst, target = None, score_func: str = "pearsonr",
                      start: Optional[int] = None, stop: Optional[int] = None,
                      l_freq: Optional[float] = None, h_freq: Optional[float] = None,
                      reject_by_annotation: bool = True, verbose = None):
        if score_func != "pearsonr":
            raise ValueError(f"{score_func} is not a valid score_func.")
        src = self._sources_2d(inst)
        if isinstance(inst, RawArray):
            s = 0 if start is None else start.__val__()
            e = inst.n_times if stop is None else stop.__val__()
            src = np.ascontiguousarray(src[:, s:e])
        tg = self._target(inst, target, start, stop)
        if src.shape[1] != tg.shape[0]:
            raise ValueError("Sources and target do not have the same number of time slices.")
        if isinstance(inst, RawArray) and l_freq is not None and h_freq is not None:
            sf = inst.info.sfreq
            src = filter_data(src, sf, l_freq.__val__(), h_freq.__val__(), None, "10s", 0.5, 0.5,
                              phase="zero-double", fir_window="hann", fir_design="firwin2")
            tg = filter_data(tg, sf, l_freq.__val__(), h_freq.__val__(), None, "10s", 0.5, 0.5,
                             phase="zero-double", fir_window="hann", fir_design="firwin2")
        elif (l_freq is None) != (h_freq is None):
            raise ValueError("Must specify both pass bands")
        return _pearson_rows(src, tg)

    def _target(self, inst, target, start: Optional[int], stop: Optional[int]) -> np.ndarray[float, 1]:
        if isinstance(target, str):
            if target not in inst.info.ch_names:
                raise ValueError(f"{target} not in channel list ({inst.info.ch_names})")
            ci = inst.info.ch_names.index(target)
            if isinstance(inst, RawArray):
                s = 0 if start is None else start.__val__()
                e = inst.n_times if stop is None else stop.__val__()
                return inst._data[ci, s:e].copy()
            else:
                return np.ascontiguousarray(inst._data[:, ci, :]).ravel()
        else:
            return np.asarray(target, dtype=float).ravel()

    def _find_bads_ch(self, inst, chs: List[str], threshold: float, start: Optional[int],
                      stop: Optional[int], l_freq: Optional[float], h_freq: Optional[float],
                      prefix: str, measure: str):
        scores = List[np.ndarray[float, 1]]()
        idx = List[List[int]]()
        for ii in range(len(chs)):
            sc = self.score_sources(inst, chs[ii], "pearsonr", start, stop, l_freq, h_freq)
            scores.append(sc)
            this = List[int]()
            if measure == "zscore":
                this = _find_outliers(sc, threshold)
            elif measure == "correlation":
                for k in range(sc.shape[0]):
                    if abs(sc[k]) > threshold:
                        this.append(k)
            else:
                raise ValueError(f"Unknown measure {measure}")
            idx.append(this)
            self.labels_[f"{prefix}/{ii}/{chs[ii]}"] = list(this)
        cand = List[Tuple[float, int]]()
        for ii in range(len(idx)):
            for k in idx[ii]:
                cand.append((abs(float(scores[ii][k])), k))
        # sort by |score| descending (stable on ties like argsort()[::-1])
        absc = np.empty(len(cand))
        for q in range(len(cand)):
            absc[q] = cand[q][0]
        order = np.argsort(absc, kind="stable")[::-1]
        labels = List[int]()
        for o in order:
            k = cand[int(o)][1]
            if k not in labels:
                labels.append(k)
        return labels, scores

    def find_bads_eog(self, inst, ch_name = None, threshold = 3.0, start: Optional[int] = None,
                      stop: Optional[int] = None, l_freq: Optional[float] = 1.0,
                      h_freq: Optional[float] = 10.0, reject_by_annotation: bool = True,
                      measure: str = "zscore", verbose = None):
        if measure not in ("zscore", "correlation"):
            raise ValueError(f"Invalid value for the 'measure' parameter. Allowed values are 'zscore' and 'correlation', but got {measure!r} instead.")
        chs = _eog_names(inst.info, ch_name)
        th = 0.0
        if isinstance(threshold, str):
            th = 3.0 if measure == "zscore" else 0.9
        else:
            th = float(threshold)
        labels, scores = self._find_bads_ch(inst, chs, th, start, stop, l_freq, h_freq, "eog", measure)
        self.labels_["eog"] = labels
        return labels, scores

    def find_bads_ecg(self, inst, ch_name: Optional[str] = None, threshold = "auto",
                      start: Optional[int] = None, stop: Optional[int] = None,
                      l_freq: Optional[float] = 8.0, h_freq: Optional[float] = 16.0,
                      method: str = "correlation", reject_by_annotation: bool = True,
                      measure: str = "zscore", verbose = None):
        if method != "correlation":
            raise ValueError("find_bads_ecg: only method='correlation' is supported in Typhon mne")
        name = ""
        if ch_name is None:
            for i in range(inst.info.nchan):
                if inst.info.ch_types[i] == "ecg":
                    name = inst.info.ch_names[i]
                    break
            if name == "":
                raise RuntimeError("No ECG channel found (synthetic ECG is not supported)")
        else:
            name = ch_name.__val__()
        th = 0.0
        if isinstance(threshold, str):
            th = 3.0 if measure == "zscore" else 0.9
        else:
            th = float(threshold)
        labels, scores = self._find_bads_ch(inst, [name], th, start, stop, l_freq, h_freq, "ecg", measure)
        self.labels_["ecg"] = labels
        return labels, scores

    def copy(self):
        n = ICA(None, random_state=self.random_state, method=self.method)
        n.n_components = self.n_components
        n.n_components_int = self.n_components_int
        n.max_iter = self.max_iter
        n.fit_params = dict(self.fit_params)
        n.exclude = list(self.exclude)
        n.labels_ = {k: list(v) for k, v in self.labels_.items()}
        n.ch_names = list(self.ch_names)
        n.info = self.info.copy()
        n.n_components_ = self.n_components_
        n.n_iter_ = self.n_iter_
        n.n_samples_ = self.n_samples_
        n.pre_whitener_ = self.pre_whitener_.copy()
        n.pca_mean_ = self.pca_mean_.copy()
        n.pca_components_ = self.pca_components_.copy()
        n.pca_explained_variance_ = self.pca_explained_variance_.copy()
        n.unmixing_matrix_ = self.unmixing_matrix_.copy()
        n.mixing_matrix_ = self.mixing_matrix_.copy()
        n.current_fit = self.current_fit
        return n

def _eog_names(info: Info, ch_name) -> List[str]:
    out = List[str]()
    if ch_name is None:
        for i in range(info.nchan):
            if info.ch_types[i] == "eog" and info.ch_names[i] not in info.bads:
                out.append(info.ch_names[i])
        if len(out) == 0:
            raise RuntimeError("No EOG channel(s) found")
    elif isinstance(ch_name, str):
        out.append(ch_name)
    else:
        for c in ch_name:
            out.append(str(c))
    for c in out:
        if c not in info.ch_names:
            raise ValueError(f"The specified EOG channel cannot be found: {c}")
    return out
