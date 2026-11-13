# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne/decoding/transformer.py: Scaler, Vectorizer, PSDEstimator,
# FilterEstimator, UnsupervisedSpatialFilter, TemporalFilter.

import numpy as np
from .._info import Info, _picks_to_idx, _picks_by_type
from ..filter import filter_data
from ..time_frequency.multitaper import psd_array_multitaper

_DEFAULT_SCALINGS = {"mag": 1e15, "grad": 1e13, "eeg": 1e6}


def _as3d(X) -> np.ndarray[float, 3]:
    a = np.asarray(X, dtype=float)
    if static.len(a.shape) == 3:
        return np.ascontiguousarray(a)
    elif static.len(a.shape) == 2:
        return np.ascontiguousarray(a.reshape((a.shape[0], a.shape[1], 1)))
    else:
        compile_error("X must be 2-D or 3-D")


def _reshape_2d(X: np.ndarray[float, 3]) -> np.ndarray[float, 2]:
    """(n_epochs, n_ch, n_t) -> (n_epochs * n_t, n_ch) like _sklearn_reshape_apply."""
    return np.ascontiguousarray(X.transpose(0, 2, 1)).reshape((-1, X.shape[1]))


def _unshape(X2: np.ndarray[float, 2], n_ep: int, n_ch: int, n_t: int) -> np.ndarray[float, 3]:
    return np.ascontiguousarray(X2.reshape((n_ep, n_t, n_ch)).transpose(0, 2, 1))


class Scaler:
    """Standardize channel data (mne.decoding.Scaler).

    scalings: None / dict (constant per channel type, needs info), 'mean'
    (sklearn StandardScaler) or 'median' (sklearn RobustScaler)."""
    info: Optional[Info]
    scalings_kind: str
    scalings_dict: Dict[str, float]
    with_mean: bool
    with_std: bool
    mean_: np.ndarray[float, 1]
    std_: np.ndarray[float, 1]
    n_features_in_: int
    _fitted: bool

    def __init__(self, info: Optional[Info] = None, scalings = None, with_mean: bool = True,
                 with_std: bool = True):
        self.info = info
        self.with_mean = with_mean
        self.with_std = with_std
        self.scalings_dict = dict(_DEFAULT_SCALINGS)
        if isinstance(scalings, NoneType):
            self.scalings_kind = "constant"
        elif isinstance(scalings, str):
            if scalings != "mean" and scalings != "median":
                raise ValueError(f"Invalid value for the 'scalings' parameter when str. Allowed values are 'mean' and 'median', but got '{scalings}' instead.")
            self.scalings_kind = scalings
        else:
            self.scalings_kind = "constant"
            for k, v in scalings.items():
                if k not in ("mag", "grad", "eeg"):
                    raise ValueError(f"Invalid value for the keys in `scalings` parameter. Allowed values are 'mag', 'grad', and 'eeg', but got '{k}' instead.")
                self.scalings_dict[k] = float(v)
        self.mean_ = np.zeros(0)
        self.std_ = np.zeros(0)
        self.n_features_in_ = 0
        self._fitted = False

    def fit(self, epochs_data, y = None):
        from sklearn.preprocessing import StandardScaler, RobustScaler
        X = _as3d(epochs_data)
        self.n_features_in_ = X.shape[1]
        X2 = _reshape_2d(X)
        nch = X.shape[1]
        if self.scalings_kind == "constant":
            if self.info is None:
                raise ValueError('Need to specify "info" if scalings is None or dict')
            info = self.info.__val__()
            std = np.ones(nch)
            pbt = _picks_by_type(info, False, False)
            n = 0
            for kind, picks in pbt:
                n += len(picks)
            if nch != n:
                raise ValueError(f"info had {n} data channels but X has {X.shape[0]} channels")
            if self.with_std:
                for kind, picks in pbt:
                    if kind not in self.scalings_dict:
                        raise KeyError(kind)
                    for p in picks:
                        std[p] = 1.0 / self.scalings_dict[kind]
            self.std_ = std
            self.mean_ = np.zeros(nch)
        elif self.scalings_kind == "mean":
            s = StandardScaler(with_mean=self.with_mean, with_std=self.with_std)
            s.fit(X2)
            self.mean_ = s.mean_.copy() if self.with_mean else np.zeros(nch)
            self.std_ = s.scale_.copy() if self.with_std else np.ones(nch)
        else:
            r = RobustScaler(with_centering=self.with_mean, with_scaling=self.with_std)
            r.fit(X2)
            self.mean_ = r.center_.copy() if self.with_mean else np.zeros(nch)
            self.std_ = r.scale_.copy() if self.with_std else np.ones(nch)
        self._fitted = True
        return self

    def _apply(self, epochs_data, inverse: bool):
        if not self._fitted:
            raise ValueError("This Scaler instance is not fitted yet.")
        X = _as3d(epochs_data)
        n_ep = X.shape[0]
        n_ch = X.shape[1]
        n_t = X.shape[2]
        X2 = _reshape_2d(X)
        if inverse:
            X2 = X2 * self.std_[None, :] + self.mean_[None, :]
        else:
            X2 = (X2 - self.mean_[None, :]) / self.std_[None, :]
        return _unshape(X2, n_ep, n_ch, n_t)

    def transform(self, epochs_data):
        a = np.asarray(epochs_data)
        if static.len(a.shape) == 2:
            return self._apply(a, False)
        else:
            return self._apply(a, False)

    def fit_transform(self, epochs_data, y = None):
        return self.fit(epochs_data, y).transform(epochs_data)

    def inverse_transform(self, epochs_data):
        a = np.asarray(epochs_data)
        out = self._apply(a, True)
        if static.len(a.shape) == 2:
            return out[:, :, 0].copy()
        else:
            return out


class Vectorizer:
    """Transform n-dimensional array into 2D array of n_samples by n_features (mne)."""
    features_shape_: List[int]
    n_features_in_: int

    def __init__(self):
        self.features_shape_ = List[int]()
        self.n_features_in_ = 0

    def fit(self, X, y = None):
        a = np.asarray(X)
        self.features_shape_ = [int(s) for s in a.shape[1:]]
        self.n_features_in_ = a.shape[1]
        return self

    def transform(self, X):
        a = np.asarray(X, dtype=float)
        if [int(s) for s in a.shape[1:]] != self.features_shape_:
            raise ValueError("Shape of X used in fit and transform must be same")
        return np.ascontiguousarray(a).reshape((a.shape[0], -1))

    def fit_transform(self, X, y = None):
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X) -> np.ndarray[float, 3]:
        """Inverse for (n_channels, n_times) feature shapes; returns a 3-D array."""
        a = np.ascontiguousarray(np.asarray(X, dtype=float))
        if len(self.features_shape_) != 2:
            raise ValueError("Typhon Vectorizer.inverse_transform supports 2-D feature shapes")
        n = self.features_shape_[0] * self.features_shape_[1]
        return a.reshape((a.size // n, self.features_shape_[0], self.features_shape_[1]))


class PSDEstimator:
    """Compute power spectral density (PSD) using a multi-taper method (mne)."""
    sfreq: float
    fmin: float
    fmax: float
    bandwidth: Optional[float]
    adaptive: bool
    low_bias: bool
    normalization: str

    def __init__(self, sfreq: float = 2 * np.pi, fmin: float = 0, fmax: float = np.inf,
                 bandwidth: Optional[float] = None, adaptive: bool = False,
                 low_bias: bool = True, n_jobs = None, normalization: str = "length"):
        self.sfreq = sfreq
        self.fmin = fmin
        self.fmax = fmax
        self.bandwidth = bandwidth
        self.adaptive = adaptive
        self.low_bias = low_bias
        self.normalization = normalization

    def fit(self, epochs_data, y = None):
        return self

    def transform(self, epochs_data):
        X = _as3d(epochs_data)
        if self.bandwidth is None:
            psd, _ = psd_array_multitaper(X, sfreq=self.sfreq, fmin=self.fmin, fmax=self.fmax,
                                          bandwidth=None, adaptive=self.adaptive,
                                          low_bias=self.low_bias, normalization=self.normalization)
            return psd
        psd2, _ = psd_array_multitaper(X, sfreq=self.sfreq, fmin=self.fmin, fmax=self.fmax,
                                       bandwidth=self.bandwidth.__val__(), adaptive=self.adaptive,
                                       low_bias=self.low_bias, normalization=self.normalization)
        return psd2

    def fit_transform(self, epochs_data, y = None):
        return self.fit(epochs_data, y).transform(epochs_data)


def _of(x) -> Optional[float]:
    if isinstance(x, NoneType):
        return None
    elif isinstance(x, Optional):
        return x
    else:
        return float(x)


def _filt(X, sfreq: float, lf: Optional[float], hf: Optional[float], picks: List[int],
          fl: str, ltb: str, htb: str, win: str, design: str):
    """filter_data with Optional band edges unwrapped into static None/float."""
    if lf is None:
        if hf is None:
            return filter_data(X, sfreq, None, None, picks, fl, ltb, htb, copy=False,
                               fir_window=win, fir_design=design)
        return filter_data(X, sfreq, None, hf.__val__(), picks, fl, ltb, htb, copy=False,
                           fir_window=win, fir_design=design)
    if hf is None:
        return filter_data(X, sfreq, lf.__val__(), None, picks, fl, ltb, htb, copy=False,
                           fir_window=win, fir_design=design)
    return filter_data(X, sfreq, lf.__val__(), hf.__val__(), picks, fl, ltb, htb, copy=False,
                       fir_window=win, fir_design=design)


class FilterEstimator:
    """Estimator to filter RtEpochs / epochs data (mne.decoding.FilterEstimator, FIR)."""
    info: Info
    l_freq: Optional[float]
    h_freq: Optional[float]
    picks_: List[int]
    _picks_arg: Optional[List[int]]
    filter_length: str
    l_trans_bandwidth: str
    h_trans_bandwidth: str
    fir_design: str

    def __init__(self, info: Info, l_freq, h_freq, picks = None, filter_length: str = "auto",
                 l_trans_bandwidth: str = "auto", h_trans_bandwidth: str = "auto", n_jobs = None,
                 method: str = "fir", iir_params = None, fir_design: str = "firwin"):
        if method != "fir":
            raise ValueError("FilterEstimator: only method='fir' is supported")
        self.info = info
        self.l_freq = _of(l_freq)
        self.h_freq = _of(h_freq)
        self._picks_arg = None if isinstance(picks, NoneType) else _picks_to_idx(info, picks)
        self.picks_ = List[int]()
        self.filter_length = filter_length
        self.l_trans_bandwidth = l_trans_bandwidth
        self.h_trans_bandwidth = h_trans_bandwidth
        self.fir_design = fir_design

    def fit(self, epochs_data, y = None):
        if self._picks_arg is None:
            self.picks_ = _picks_to_idx(self.info, None)
        else:
            self.picks_ = list(self._picks_arg.__val__())
        if self.l_freq is not None and self.l_freq.__val__() == 0:
            self.l_freq = None
        lf = self.l_freq
        hf = self.h_freq
        if hf is not None and (lf is None or lf.__val__() < hf.__val__()) and hf.__val__() < self.info.lowpass:
            self.info.lowpass = hf.__val__()
        if lf is not None and (hf is None or lf.__val__() < hf.__val__()) and lf.__val__() > self.info.highpass:
            self.info.highpass = lf.__val__()
        return self

    def transform(self, epochs_data):
        X = _as3d(epochs_data)
        return _filt(X, self.info.sfreq, self.l_freq, self.h_freq, self.picks_,
                     self.filter_length, self.l_trans_bandwidth, self.h_trans_bandwidth,
                     "hamming", self.fir_design)

    def fit_transform(self, epochs_data, y = None):
        return self.fit(epochs_data, y).transform(epochs_data)


class TemporalFilter[P = NoneType]:
    """Estimator to filter data array along the last dimension (mne, FIR)."""
    l_freq: Optional[float]
    h_freq: Optional[float]
    sfreq: float
    filter_length: str
    l_trans_bandwidth: str
    h_trans_bandwidth: str
    fir_window: str
    fir_design: str
    method: str
    iir_params: P

    def __init__(self, l_freq = None, h_freq = None, sfreq: float = 1.0,
                 filter_length: str = "auto", l_trans_bandwidth: str = "auto",
                 h_trans_bandwidth: str = "auto", n_jobs = None, method: str = "fir",
                 iir_params: P = None, fir_window: str = "hamming", fir_design: str = "firwin"):
        if method != "fir" and method != "iir":
            raise ValueError("TemporalFilter: method must be 'fir' or 'iir'")
        self.method = method
        self.iir_params = iir_params
        self.l_freq = _of(l_freq)
        self.h_freq = _of(h_freq)
        self.sfreq = sfreq
        self.filter_length = filter_length
        self.l_trans_bandwidth = l_trans_bandwidth
        self.h_trans_bandwidth = h_trans_bandwidth
        self.fir_window = fir_window
        self.fir_design = fir_design

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        a = np.asarray(X, dtype=float)
        X2 = np.ascontiguousarray(a).reshape((-1, a.shape[a.ndim - 1]))
        if self.method == "iir":
            from mne.filter import filter_data
            o2 = filter_data(X2, self.sfreq, self.l_freq, self.h_freq, method="iir",
                             iir_params=self.iir_params, copy=True)
            return o2.reshape(a.shape)
        out = _filt(X2, self.sfreq, self.l_freq, self.h_freq, list(range(X2.shape[0])),
                    self.filter_length, self.l_trans_bandwidth, self.h_trans_bandwidth,
                    self.fir_window, self.fir_design)
        return out.reshape(a.shape)

    def fit_transform(self, X, y = None):
        return self.fit(X, y).transform(X)


class UnsupervisedSpatialFilter[E]:
    """Use unsupervised spatial filtering across time and samples (mne).

    estimator: any sklearn-like transformer (e.g. sklearn.decomposition.PCA)."""
    estimator: E
    average: bool
    estimator_: E
    _fitted: bool

    def __init__(self, estimator: E, average: bool = False):
        self.estimator = estimator
        self.average = average
        self.estimator_ = estimator
        self._fitted = False

    def fit(self, X, y = None):
        from sklearn.base import clone
        Xa = _as3d(X)
        if self.average:
            X2 = np.ascontiguousarray(np.mean(Xa, axis=0).T)
        else:
            n_ep = Xa.shape[0]
            n_ch = Xa.shape[1]
            n_t = Xa.shape[2]
            X2 = np.ascontiguousarray(np.transpose(Xa, (1, 0, 2))).reshape((n_ch, n_ep * n_t)).T.copy()
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X2)
        self._fitted = True
        return self

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

    def _apply(self, X, inverse: bool):
        Xa = _as3d(X)
        n_ep = Xa.shape[0]
        n_ch = Xa.shape[1]
        n_t = Xa.shape[2]
        X2 = np.ascontiguousarray(np.transpose(Xa, (1, 0, 2))).reshape((n_ch, n_ep * n_t)).T.copy()
        if inverse:
            Y = np.asarray(self.estimator_.inverse_transform(X2), dtype=float)
        else:
            Y = np.asarray(self.estimator_.transform(X2), dtype=float)
        return np.ascontiguousarray(np.ascontiguousarray(Y.T).reshape((-1, n_ep, n_t)).transpose(1, 0, 2))

    def transform(self, X):
        if not self._fitted:
            raise ValueError("This UnsupervisedSpatialFilter instance is not fitted yet.")
        return self._apply(X, False)

    def inverse_transform(self, X):
        return self._apply(X, True)
