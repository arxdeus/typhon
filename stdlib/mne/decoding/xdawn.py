# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# mne.decoding: XdawnTransformer, SpatialFilter, get_spatial_filter_from_estimator,
# TimeFrequency.

import numpy as np
import scipy.linalg
from .._info import Info, create_info
from .._cov import _regularized_covariance, _compute_rank_data, _restr_mat, pinv
from ..time_frequency.tfr import _compute_tfr_c


class XdawnTransformer:
    """Xdawn spatial filters compatible with the fit/transform protocol.

    restr_type 'whitening' and cov_method_params are not supported.
    Fitted attributes: classes_, filters_ (n_classes, n_ch, n_ch),
    patterns_ (n_classes, n_ch, n_ch), evals_ (n_classes, n_ch), sorter_.
    """
    n_components: int
    reg_kind: str
    reg_value: float
    reg_repr: str
    signal_cov: Optional[np.ndarray[float, 2]]
    restr_type: Optional[str]
    info: Optional[Info]
    rank_dict: Optional[Dict[str, int]]
    rank_str: str
    classes_: np.ndarray[int, 1]
    filters_: np.ndarray[float, 3]
    patterns_: np.ndarray[float, 3]
    evals_: np.ndarray[float, 2]
    sorter_: np.ndarray[int, 2]
    n_features_in_: int
    _fitted: bool

    def __init__(self, n_components: int = 2, reg = None, signal_cov = None,
                 cov_method_params = None, *, restr_type: Optional[str] = None,
                 info: Optional[Info] = None, rank = "full"):
        self.n_components = n_components
        if isinstance(reg, NoneType):
            self.reg_kind = "empirical"
            self.reg_value = 0.0
            self.reg_repr = "None"
        elif isinstance(reg, str):
            if reg not in ("empirical", "ledoit_wolf", "oas"):
                raise ValueError(f"reg={reg!r} is not supported in Typhon mne (use None, float, 'empirical', 'ledoit_wolf' or 'oas')")
            self.reg_kind = reg
            self.reg_value = 0.0
            self.reg_repr = reg
        else:
            self.reg_kind = "shrinkage"
            self.reg_value = float(reg)
            self.reg_repr = str(reg)
        if not isinstance(cov_method_params, NoneType):
            raise ValueError("cov_method_params is not supported in Typhon mne")
        self.signal_cov = None
        if not isinstance(signal_cov, NoneType):
            self.signal_cov = np.asarray(signal_cov, dtype=float)
        if restr_type is not None:
            if restr_type.__val__() != "restricting":
                raise ValueError("restr_type='whitening' is not supported in Typhon mne (use 'restricting' or None)")
        self.restr_type = restr_type
        self.info = info
        self.rank_dict = None
        self.rank_str = "full"
        if isinstance(rank, NoneType):
            self.rank_str = "estimated"
        elif isinstance(rank, str):
            self.rank_str = rank
        else:
            d = Dict[str, int]()
            for k, v in rank.items():
                d[k] = v
            self.rank_dict = d
        self.classes_ = np.zeros(0, dtype=int)
        self.filters_ = np.zeros((0, 0, 0))
        self.patterns_ = np.zeros((0, 0, 0))
        self.evals_ = np.zeros((0, 0))
        self.sorter_ = np.zeros((0, 0), dtype=int)
        self.n_features_in_ = 0
        self._fitted = False

    def __repr__(self) -> str:
        return f"XdawnTransformer(n_components={self.n_components})"

    def fit(self, X, y = None):
        Xa = np.asarray(X).astype(float)
        if Xa.ndim != 3:
            raise ValueError(f"X must be 3-D (n_epochs, n_channels, n_times), got {Xa.ndim} dimensions")
        n_ep = Xa.shape[0]
        n_ch = Xa.shape[1]
        ya = np.ones(n_ep, dtype=int)
        if not isinstance(y, NoneType):
            ya = np.asarray(y)
            if ya.shape[0] != n_ep:
                raise ValueError(f"Found input variables with inconsistent numbers of samples: [{n_ep}, {ya.shape[0]}]")
        if self.n_components <= 0:
            raise ValueError("Invalid value for the 'n_components' parameter. Allowed are positive integers or None, but got a non-positive integer instead.")
        if self.signal_cov is not None:
            sc = self.signal_cov.__val__()
            if sc.shape[0] != n_ch or sc.shape[1] != n_ch:
                raise ValueError("signal_cov data should be of shape (n_channels, n_channels)")
        self.n_features_in_ = n_ch
        self.classes_ = np.unique(ya)
        info = Info(["0"], 1.0, ["mag"])
        rank = Dict[str, int]()
        given = self.rank_dict is not None
        if self.info is None:
            info = create_info(n_ch, 1000.0, "mag")
            if given:
                s = 0
                for v in self.rank_dict.__val__().values():
                    s += v
                rank["mag"] = s
        else:
            info = self.info.__val__()
            if given:
                rank = self.rank_dict.__val__()
        Xh = np.concatenate([Xa[i] for i in range(n_ep)], axis=1)
        if not given:
            rank = _compute_rank_data(Xh, info, self.rank_str)
        R = np.zeros((n_ch, n_ch))
        if self.signal_cov is None:
            R = _regularized_covariance(Xh, self.reg_kind, self.reg_value, info, rank)
        else:
            R = self.signal_cov.__val__().copy()
        n_cls = self.classes_.shape[0]
        covs = np.zeros((n_cls, n_ch, n_ch))
        for ci in range(n_cls):
            evo = np.zeros((n_ch, Xa.shape[2]))
            cnt = 0
            for i in range(n_ep):
                if ya[i] == self.classes_[ci]:
                    evo += Xa[i]
                    cnt += 1
            evo /= cnt
            covs[ci] = _regularized_covariance(evo, self.reg_kind, self.reg_value, info, rank)
        use_restr = self.restr_type is not None
        restr = np.eye(n_ch)
        if use_restr:
            restr = _restr_mat(R, info, rank)
        r = restr.shape[0]
        filters = np.zeros((n_cls, r if use_restr else n_ch, n_ch))
        patterns = np.zeros((n_cls, r if use_restr else n_ch, n_ch))
        evals = np.zeros((n_cls, r if use_restr else n_ch))
        sorter = np.zeros((n_cls, r if use_restr else n_ch), dtype=int)
        for ci in range(n_cls):
            S = covs[ci]
            w = np.zeros(0)
            v = np.zeros((0, 0))
            if use_restr:
                S_r = restr @ S @ restr.T
                R_r = restr @ R @ restr.T
                w, vr = scipy.linalg.eigh(S_r, R_r)
                v = restr.T @ vr
            else:
                w, v = scipy.linalg.eigh(S, R)
            ix = np.argsort(w)[::-1].copy()
            w = w[ix].copy()
            v = v[:, ix].copy()
            for j in range(v.shape[1]):
                nrm = np.linalg.norm(v[:, j])
                v[:, j] /= nrm
            evals[ci] = w
            sorter[ci] = ix
            filters[ci] = v.T
            patterns[ci] = pinv(v)
        self.evals_ = evals
        self.sorter_ = sorter
        self.filters_ = filters
        self.patterns_ = patterns
        self._fitted = True
        return self

    def _check(self):
        if not self._fitted:
            raise ValueError("This XdawnTransformer instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

    def _subset(self, patterns: bool) -> np.ndarray[float, 2]:
        src = self.patterns_ if patterns else self.filters_
        n_cls = src.shape[0]
        out = np.zeros((n_cls * self.n_components, src.shape[2]))
        for c in range(n_cls):
            for k in range(self.n_components):
                out[c * self.n_components + k] = src[c, k]
        return out

    def transform(self, X):
        self._check()
        Xa = np.asarray(X).astype(float)
        if Xa.ndim != 3:
            raise ValueError(f"X must be 3-D (n_epochs, n_channels, n_times), got {Xa.ndim} dimensions")
        if Xa.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xa.shape[1]} features, but XdawnTransformer is expecting {self.n_features_in_} features as input.")
        F = self._subset(False)
        out = np.zeros((Xa.shape[0], F.shape[0], Xa.shape[2]))
        for e in range(Xa.shape[0]):
            out[e] = F @ Xa[e]
        return out

    def fit_transform(self, X, y = None, **fit_params):
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X):
        self._check()
        Xa = np.asarray(X).astype(float)
        if Xa.ndim != 3:
            raise ValueError(f"X must be 3-D (n_epochs, n_components, n_times), got {Xa.ndim} dimensions")
        want = self.n_components * self.classes_.shape[0]
        if Xa.shape[1] != want:
            raise ValueError(f"X must have {want} components, got {Xa.shape[1]} instead.")
        P = self._subset(True)
        out = np.zeros((Xa.shape[0], P.shape[1], Xa.shape[2]))
        for e in range(Xa.shape[0]):
            out[e] = P.T @ Xa[e]
        return out


class SpatialFilter:
    """Container for spatial filters (n_components, n_channels), patterns and evals."""
    info: Info
    filters: np.ndarray[float, 2]
    patterns: np.ndarray[float, 2]
    evals: Optional[np.ndarray[float, 1]]
    patterns_method: str

    def __init__(self, info: Info, filters: np.ndarray[float, 2], *,
                 evals: Optional[np.ndarray[float, 1]] = None,
                 patterns: Optional[np.ndarray[float, 2]] = None,
                 patterns_method: str = "pinv"):
        if patterns_method != "pinv" and patterns_method != "haufe":
            raise ValueError(f"Invalid value for the 'patterns_method' parameter. Allowed values are 'pinv' and 'haufe', but got '{patterns_method}' instead.")
        self.info = info
        self.filters = filters
        self.evals = evals
        n_comps = self.filters.shape[0]
        n_chs = self.filters.shape[1]
        if patterns is None:
            self.patterns = np.linalg.pinv(self.filters.T)
            self.patterns_method = "pinv"
        else:
            self.patterns = patterns.__val__()
            self.patterns_method = patterns_method
        if patterns_method != "haufe" and n_comps > n_chs:
            raise ValueError("Number of components can't be greater than number of channels in filters, perhaps the provided matrix is transposed?")
        if self.filters.shape[0] != self.patterns.shape[0] or self.filters.shape[1] != self.patterns.shape[1]:
            raise ValueError(f"Shape mismatch between filters and patterns.Filters are {self.filters.shape},while patterns are {self.patterns.shape}")


def get_spatial_filter_from_estimator(estimator, info: Info, *, inverse_transform: bool = False,
                                      step_name = None, get_coefs = ("filters_", "patterns_", "evals_"),
                                      patterns_method = None, verbose = None):
    """Build a SpatialFilter from a fitted 2-D-filter estimator (CSP, SPoC, SSD).

    Multi-class estimators (XdawnTransformer, 3-D filters_) are not supported here.
    """
    for c in get_coefs:
        if c != "filters_" and c != "patterns_" and c != "evals_":
            raise ValueError(f"'get_coefs' can only include 'filters_', 'patterns_' and 'evals_', but got {c}.")
    return SpatialFilter(info, estimator.filters_, evals = estimator.evals_,
                         patterns = estimator.patterns_, patterns_method = "pinv")


class TimeFrequency:
    """Time-frequency transformer (morlet / multitaper) for 3-D (n, ch, times) data."""
    freqs: np.ndarray[float, 1]
    sfreq: float
    method: str
    n_cycles: float
    time_bandwidth: float
    use_fft: bool
    decim: int
    output: str
    fitted_: bool

    def __init__(self, freqs, sfreq: float = 1.0, method: str = "morlet", n_cycles: float = 7.0,
                 time_bandwidth = None, use_fft: bool = True, decim: int = 1,
                 output: str = "complex", n_jobs = 1, verbose = None):
        self.freqs = np.asarray(freqs, dtype=float)
        self.sfreq = float(sfreq)
        self.method = method
        self.n_cycles = float(n_cycles)
        self.time_bandwidth = 4.0 if isinstance(time_bandwidth, NoneType) else float(time_bandwidth)
        self.use_fft = use_fft
        self.decim = decim
        self.output = output
        self.fitted_ = False

    def fit(self, X, y = None):
        if self.output != "complex" and self.output != "power" and self.output != "phase":
            raise ValueError(f"Invalid value for the 'output' parameter. Allowed values are 'complex', 'power' and 'phase', but got '{self.output}' instead.")
        self.fitted_ = True
        return self

    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X):
        """Returns (n_samples, n_channels, n_freqs, n_times); power/phase are real."""
        if not self.fitted_:
            raise ValueError("This TimeFrequency instance is not fitted yet.")
        Xa = np.asarray(X).astype(float)
        if Xa.ndim != 3:
            raise ValueError("X must be 3-D (n_samples, n_channels, n_times)")
        out5, _ = _compute_tfr_c(Xa, self.freqs, self.sfreq, self.method, self.n_cycles, True,
                                 self.time_bandwidth, self.use_fft, self.decim, self.output)
        n_e = out5.shape[0]
        if self.output == "complex" and self.method == "multitaper":
            raise NotImplementedError("multitaper complex output has an extra taper axis; use morlet or output='power'")
        res = np.zeros((n_e, out5.shape[1], out5.shape[3], out5.shape[4]), dtype=complex)
        for e in range(n_e):
            res[e] = out5[e, :, 0]
        return res

    def transform_real(self, X):
        """Real-valued variant of transform() for output='power' or 'phase'."""
        return self.transform(X).real.copy()
