# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne/decoding/ssd.py (SSD), with _ssd_estimate (_covs_ged.py), _ssd_mod /
# _get_spectral_ratio (_mod_ged.py) and the 'whitening' restriction matrix
# (mne.cov.compute_whitener(pca=True) for a single channel type).

import numpy as np
import scipy.linalg
from .._info import Info, create_info, _picks_to_idx, _time_mask
from .._cov import _regularized_covariance, _compute_rank_data, _restr_mat, pinv
from ..filter import filter_data
from ..time_frequency.psd import psd_array_welch


def _whitener_pca(C: np.ndarray[float, 2], rank: int) -> np.ndarray[float, 2]:
    """compute_whitener(Covariance(C), info, rank={'eeg': rank}, pca=True)[0]."""
    n = C.shape[0]
    e, v = np.linalg.eigh(C)
    vt = v.T.copy()
    nz = n - rank
    W = np.empty((rank, n))
    for i in range(rank):
        W[i] = vt[nz + i] / np.sqrt(e[nz + i])
    return W


class SSD:
    """Signal decomposition using the Spatio-Spectral Decomposition (mne.decoding.SSD).

    info: Info or a float sfreq; filt_params_signal/noise: dicts with
    l_freq, h_freq and optionally l_trans_bandwidth/h_trans_bandwidth (floats)."""
    info: Info
    sfreq_: float
    sig_l: float
    sig_h: float
    noise_l: float
    noise_h: float
    sig_ltb: float
    sig_htb: float
    noise_ltb: float
    noise_htb: float
    reg_kind: str
    reg_value: float
    n_components: Optional[int]
    picks_: List[int]
    sort_by_spectral_ratio: bool
    return_filtered: bool
    n_fft: Optional[int]
    n_fft_: int
    restr_type: Optional[str]
    filters_: np.ndarray[float, 2]
    patterns_: np.ndarray[float, 2]
    evals_: np.ndarray[float, 1]
    sorter_: np.ndarray[int, 1]
    _fitted: bool
    _info_given: bool
    _picks_arg: Optional[List[int]]

    def __init__(self, info, filt_params_signal: Dict[str, float], filt_params_noise: Dict[str, float],
                 reg = None, n_components: Optional[int] = None, picks = None,
                 sort_by_spectral_ratio: bool = True, return_filtered: bool = False,
                 n_fft: Optional[int] = None, cov_method_params = None, *,
                 restr_type: Optional[str] = "whitening", rank = None):
        if isinstance(info, float) or isinstance(info, int):
            self.sfreq_ = float(info)
            self.info = create_info(1, self.sfreq_, "eeg")
            self._info_given = False
        else:
            self.info = info
            self.sfreq_ = info.sfreq
            self._info_given = True
        for d, nm in ((filt_params_signal, "signal"), (filt_params_noise, "noise")):
            for k in ("l_freq", "h_freq"):
                if k not in d:
                    raise ValueError(f"{k} must be defined in filter parameters for {nm}")
        self.sig_l = filt_params_signal["l_freq"]
        self.sig_h = filt_params_signal["h_freq"]
        self.noise_l = filt_params_noise["l_freq"]
        self.noise_h = filt_params_noise["h_freq"]
        self.sig_ltb = filt_params_signal.get("l_trans_bandwidth", -1.0)
        self.sig_htb = filt_params_signal.get("h_trans_bandwidth", -1.0)
        self.noise_ltb = filt_params_noise.get("l_trans_bandwidth", -1.0)
        self.noise_htb = filt_params_noise.get("h_trans_bandwidth", -1.0)
        if self.noise_l > self.sig_l or self.sig_h > self.noise_h:
            raise ValueError("Wrongly specified frequency bands!\nThe signal band-pass must be within the noise band-pass!")
        if isinstance(reg, NoneType):
            self.reg_kind = "empirical"
            self.reg_value = 0.0
        elif isinstance(reg, str):
            self.reg_kind = reg
            self.reg_value = 0.0
        else:
            self.reg_kind = "shrinkage"
            self.reg_value = float(reg)
        if not isinstance(cov_method_params, NoneType):
            raise ValueError("cov_method_params is not supported in Typhon mne")
        if not isinstance(rank, NoneType):
            raise ValueError("SSD: only rank=None is supported in Typhon mne")
        self.n_components = n_components
        self._picks_arg = None if isinstance(picks, NoneType) else [int(p) for p in picks]
        self.picks_ = List[int]()
        self.sort_by_spectral_ratio = sort_by_spectral_ratio
        self.return_filtered = return_filtered
        self.n_fft = n_fft
        self.n_fft_ = 0
        self.restr_type = restr_type
        self.filters_ = np.zeros((0, 0))
        self.patterns_ = np.zeros((0, 0))
        self.evals_ = np.zeros(0)
        self.sorter_ = np.zeros(0, dtype=int)
        self._fitted = False

    def _filter(self, X, sig: bool):
        if sig:
            return _bp(X, self.sfreq_, self.sig_l, self.sig_h, self.sig_ltb, self.sig_htb)
        return _bp(X, self.sfreq_, self.noise_l, self.noise_h, self.noise_ltb, self.noise_htb)

    def fit(self, X, y = None):
        Xa = np.asarray(X, dtype=float)
        X3 = Xa.reshape((1, Xa.shape[0], Xa.shape[1])) if static.len(Xa.shape) == 2 else Xa
        n_ch_all = X3.shape[1]
        if self._info_given:
            if n_ch_all != self.info.nchan:
                raise ValueError(f"Info must match the input data.Found {n_ch_all} channels but expected {self.info.nchan}.")
            info = self.info
        else:
            info = create_info(n_ch_all, self.sfreq_, "eeg")
            self.info = info
        if self._picks_arg is None:
            self.picks_ = _picks_to_idx(info, None, "data", "bads")
        else:
            self.picks_ = list(self._picks_arg.__val__())
        nf = self.sfreq_ if self.n_fft is None else float(self.n_fft.__val__())
        self.n_fft_ = min(int(nf), X3.shape[2])
        pk = np.array(self.picks_)
        X_aux = np.ascontiguousarray(X3[:, pk, :])
        X_sig = self._filter(X_aux, True)
        X_noise = self._filter(X_aux, False)
        X_noise -= X_sig
        n_ep = X_aux.shape[0]
        n_c = X_aux.shape[1]
        Xs = np.concatenate([X_sig[i] for i in range(n_ep)], axis=1)
        Xn = np.concatenate([X_noise[i] for i in range(n_ep)], axis=1)
        pinfo = info._pick(self.picks_)
        full = Dict[str, int]()
        for t in pinfo.ch_types:
            full[t] = full.get(t, 0) + 1
        S = _regularized_covariance(Xs, self.reg_kind, self.reg_value, pinfo, full)
        R = _regularized_covariance(Xn, self.reg_kind, self.reg_value, pinfo, full)
        # mne keeps only the rank of the last covariance (R); full-rank R -> n
        r = 0
        for v in _compute_rank_data(Xn, pinfo, "estimated").values():
            r += v
        if self.restr_type is None:
            w, v = scipy.linalg.eigh(S, R)
            evecs = v
        else:
            if self.restr_type == "whitening":
                restr = _whitener_pca(S, r)
            else:
                rk = Dict[str, int]()
                rk[pinfo.ch_types[0]] = r
                restr = _restr_mat(S, pinfo, rk)
            S_r = restr @ S @ restr.T
            R_r = restr @ R @ restr.T
            w, vr = scipy.linalg.eigh(S_r, R_r)
            evecs = restr.T @ vr
        ix = np.argsort(w)[::-1].copy()
        evals = w[ix].copy()
        evecs = evecs[:, ix].copy()
        sorter = ix
        if self.sort_by_spectral_ratio:
            filt = evecs.T.copy()
            src = np.empty((n_ep, filt.shape[0], X3.shape[2]))
            for e in range(n_ep):
                src[e] = filt @ X_aux[e]
            _, sorter_spec = self._spectral_ratio3(src)
            evecs = evecs[:, sorter_spec].copy()
            evals = evals[sorter_spec].copy()
            sorter = sorter_spec
        self.evals_ = evals
        self.sorter_ = sorter
        self.filters_ = evecs.T.copy()
        self.patterns_ = pinv(evecs)
        self._fitted = True
        return self

    def _spectral_ratio3(self, src: np.ndarray[float, 3]):
        psd, freqs = psd_array_welch(src, sfreq=self.sfreq_, n_fft=self.n_fft_)
        sig = np.flatnonzero(_time_mask(freqs, self.sig_l, self.sig_h))
        noi = np.flatnonzero(_time_mask(freqs, self.noise_l, self.noise_h))
        ms = psd[:, :, sig].mean(axis=2).mean(axis=0)
        mn = psd[:, :, noi].mean(axis=2).mean(axis=0)
        ratio = ms / mn
        return ratio, ratio.argsort()[::-1].copy()

    def get_spectral_ratio(self, ssd_sources):
        a = np.asarray(ssd_sources, dtype=float)
        if static.len(a.shape) == 2:
            return self._spectral_ratio3(a.reshape((1, a.shape[0], a.shape[1])))
        else:
            return self._spectral_ratio3(a)

    def _n_comp(self) -> int:
        return self.filters_.shape[0] if self.n_components is None else self.n_components.__val__()

    def transform(self, X):
        """(n_epochs, n_components, n_times) (2-D input -> 2-D output)."""
        if not self._fitted:
            raise ValueError("This SSD instance is not fitted yet.")
        Xa = np.asarray(X, dtype=float)
        X3 = Xa.reshape((1, Xa.shape[0], Xa.shape[1])) if static.len(Xa.shape) == 2 else Xa
        X_aux = np.ascontiguousarray(X3[:, np.array(self.picks_), :])
        if self.return_filtered:
            X_aux = self._filter(X_aux, True)
        pick = self.filters_[:self._n_comp()]
        out = np.empty((X3.shape[0], pick.shape[0], X3.shape[2]))
        for e in range(X3.shape[0]):
            out[e] = pick @ X_aux[e]
        if static.len(Xa.shape) == 2:
            return out[0].copy()
        else:
            return out

    def fit_transform(self, X, y = None):
        return self.fit(X, y).transform(X)

    def apply(self, X):
        """Remove selected components from the signal (mne SSD.apply)."""
        Xs = self.transform(X)
        P = self.patterns_[:self._n_comp()].T
        if static.len(Xs.shape) == 2:
            return P @ Xs
        out = np.empty((Xs.shape[0], P.shape[0], Xs.shape[2]))
        for e in range(Xs.shape[0]):
            out[e] = P @ Xs[e]
        return out


def _bp(X, sfreq: float, lf: float, hf: float, ltb: float, htb: float):
    if ltb < 0:
        if htb < 0:
            return filter_data(X, sfreq, lf, hf)
        return filter_data(X, sfreq, lf, hf, h_trans_bandwidth=htb)
    if htb < 0:
        return filter_data(X, sfreq, lf, hf, l_trans_bandwidth=ltb)
    return filter_data(X, sfreq, lf, hf, l_trans_bandwidth=ltb, h_trans_bandwidth=htb)
