# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne/decoding/time_delaying_ridge.py (TimeDelayingRidge) and
# mne/decoding/receptive_field.py (ReceptiveField).
#
# The delayed auto/cross-correlations are accumulated by direct summation
# (mne uses FFT correlations followed by an edge correction; both compute
# the Gram matrix of the zero-padded delayed design, equal up to rounding).

import numpy as np
import scipy.linalg


def _as_tef(X) -> np.ndarray[float, 3]:
    """(n_times[, n_epochs], n_feats) -> (n_times, n_epochs, n_feats)."""
    a = np.asarray(X, dtype=float)
    if static.len(a.shape) == 2:
        return np.ascontiguousarray(a.reshape((a.shape[0], 1, a.shape[1])))
    elif static.len(a.shape) == 3:
        return np.ascontiguousarray(a)
    else:
        compile_error("X must be shape (n_times[, n_epochs], n_features)")


def _y_as_teo(y, X_ndim: Literal[int]) -> np.ndarray[float, 3]:
    a = np.asarray(y, dtype=float)
    if X_ndim == 2:
        if static.len(a.shape) == 1:
            return np.ascontiguousarray(a.reshape((a.shape[0], 1, 1)))
        elif static.len(a.shape) == 2:
            return np.ascontiguousarray(a.reshape((a.shape[0], 1, a.shape[1])))
        else:
            compile_error("y must be shape (n_times[, n_epochs][,n_outputs]")
    else:
        if static.len(a.shape) == 2:
            return np.ascontiguousarray(a.reshape((a.shape[0], a.shape[1], 1)))
        elif static.len(a.shape) == 3:
            return np.ascontiguousarray(a)
        else:
            compile_error("If X has 3 dimensions, y must have 2 or 3 dimensions")


def _compute_corrs(X: np.ndarray[float, 3], y: np.ndarray[float, 3], smin: int, smax: int,
                   fit_intercept: bool, edge_correction: bool):
    n_t = X.shape[0]
    n_ep = X.shape[1]
    n_x = X.shape[2]
    n_y = y.shape[2]
    X_offset = np.zeros(n_x)
    y_offset = np.zeros(n_y)
    Xc = X
    yc = y
    if fit_intercept:
        X_offset = X.reshape((n_t * n_ep, n_x)).mean(axis=0)
        y_offset = y.reshape((n_t * n_ep, n_y)).mean(axis=0)
        Xc = X - X_offset[None, None, :]
        yc = y - y_offset[None, None, :]
    L = smax - smin
    x_xt = np.zeros((n_x * L, n_x * L))
    x_y = np.zeros((n_x * L, n_y))
    for e in range(n_ep):
        for c0 in range(n_x):
            x0 = Xc[:, e, c0]
            for c1 in range(c0, n_x):
                x1 = Xc[:, e, c1]
                for i in range(L):
                    di = smin + i
                    for j in range(L):
                        dj = smin + j
                        s = 0.0
                        if edge_correction:
                            # sum_t x0[t - di] x1[t - dj], t, t-di, t-dj in [0, n_t)
                            lo = max(0, max(di, dj))
                            hi = min(n_t, min(n_t + di, n_t + dj))
                            for t in range(lo, hi):
                                s += x0[t - di] * x1[t - dj]
                        else:
                            lag = dj - di
                            lo = max(0, -lag)
                            hi = min(n_t, n_t - lag)
                            for t in range(lo, hi):
                                s += x0[t + lag] * x1[t]
                        x_xt[c0 * L + i, c1 * L + j] += s
                        if c0 != c1:
                            x_xt[c1 * L + j, c0 * L + i] += s
            for i in range(L):
                d = smin + i
                lo = max(0, d)
                hi = min(n_t, n_t + d)
                for o in range(n_y):
                    s = 0.0
                    for t in range(lo, hi):
                        s += x0[t - d] * yc[t, e, o]
                    x_y[c0 * L + i, o] += s
    return x_xt, x_y, X_offset, y_offset


def _compute_reg_neighbors(n_ch_x: int, n_delays: int, reg_type) -> np.ndarray[float, 2]:
    rt0 = ""
    rt1 = ""
    if isinstance(reg_type, str):
        rt0 = reg_type
        rt1 = reg_type
    else:
        if len(reg_type) != 2:
            raise ValueError(f"reg_type must have two elements, got {len(reg_type)}")
        rt0 = str(reg_type[0])
        rt1 = str(reg_type[1])
    for r in (rt0, rt1):
        if r != "ridge" and r != "laplacian":
            raise ValueError(f"reg_type entries must be one of ('ridge', 'laplacian'), got {r}")
    reg_time = rt0 == "laplacian" and n_delays > 1
    reg_chs = rt1 == "laplacian" and n_ch_x > 1
    N = n_ch_x * n_delays
    if not reg_time and not reg_chs:
        return np.eye(N)
    reg = np.zeros((N, N))
    if reg_time:
        for c in range(n_ch_x):
            b = c * n_delays
            for k in range(n_delays):
                reg[b + k, b + k] = 1.0 if (k == 0 or k == n_delays - 1) else 2.0
                if k + 1 < n_delays:
                    reg[b + k, b + k + 1] = -1.0
                    reg[b + k + 1, b + k] = -1.0
    if reg_chs:
        for r in range(N):
            if r + n_delays < N:
                reg[r, r + n_delays] -= 1.0
                reg[r + n_delays, r] -= 1.0
                reg[r, r] += 1.0
            if r >= n_delays:
                reg[r, r] += 1.0
    return reg


class TimeDelayingRidge:
    """Ridge regression of data with time delays (mne.decoding.TimeDelayingRidge)."""
    tmin: float
    tmax: float
    sfreq: float
    alpha: float
    reg_type_0: str
    reg_type_1: str
    fit_intercept: bool
    edge_correction: bool
    coef_: np.ndarray[float, 3]
    intercept_: np.ndarray[float, 1]
    cov_: np.ndarray[float, 2]
    _estimator_type: ClassVar[str] = "regressor"

    def __init__(self, tmin: float, tmax: float, sfreq: float, alpha: float = 0.0,
                 reg_type = "ridge", fit_intercept: bool = True, n_jobs = None,
                 edge_correction: bool = True):
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.sfreq = float(sfreq)
        self.alpha = float(alpha)
        if isinstance(reg_type, str):
            self.reg_type_0 = reg_type
            self.reg_type_1 = reg_type
        else:
            self.reg_type_0 = str(reg_type[0])
            self.reg_type_1 = str(reg_type[1])
        self.fit_intercept = fit_intercept
        self.edge_correction = edge_correction
        self.coef_ = np.zeros((0, 0, 0))
        self.intercept_ = np.zeros(0)
        self.cov_ = np.zeros((0, 0))

    @property
    def _smin(self) -> int:
        return int(round(self.tmin * self.sfreq))

    @property
    def _smax(self) -> int:
        return int(round(self.tmax * self.sfreq)) + 1

    def _fit3(self, X: np.ndarray[float, 3], y: np.ndarray[float, 3]):
        if self.tmin > self.tmax:
            raise ValueError(f"tmin must be <= tmax, got {self.tmin} and {self.tmax}")
        smin = self._smin
        smax = self._smax
        n_delays = smax - smin
        if X.shape[0] < (n_delays + 1) // 2:
            raise ValueError(f"Got n_samples = {X.shape[0]}, but at least {(n_delays + 1) // 2} are required to estimate {n_delays} delays.")
        x_xt, x_y, X_off, y_off = _compute_corrs(X, y, smin, smax, self.fit_intercept,
                                                 self.edge_correction)
        self.cov_ = x_xt
        n_x = X.shape[2]
        reg = _compute_reg_neighbors(n_x, n_delays, (self.reg_type_0, self.reg_type_1))
        mat = x_xt + self.alpha * reg
        w = scipy.linalg.solve(mat, x_y, assume_a="pos")
        n_out = x_y.shape[1]
        self.coef_ = np.ascontiguousarray(w.T).reshape((n_out, n_x, n_delays))
        if self.fit_intercept:
            self.intercept_ = y_off - X_off @ self.coef_.sum(axis=2).T
        else:
            self.intercept_ = np.zeros(n_out)
        return self

    def fit(self, X, y):
        Xa = np.asarray(X, dtype=float)
        if static.len(Xa.shape) == 3:
            return self._fit3(_as_tef(Xa), _y_as_teo(y, 3))
        else:
            return self._fit3(_as_tef(Xa), _y_as_teo(y, 2))

    def _predict3(self, X: np.ndarray[float, 3]) -> np.ndarray[float, 3]:
        n_t = X.shape[0]
        n_ep = X.shape[1]
        n_out = self.coef_.shape[0]
        out = np.zeros((n_t, n_ep, n_out))
        smin = self._smin
        offset = max(smin, 0)
        for ei in range(n_ep):
            for oi in range(n_out):
                for fi in range(self.coef_.shape[1]):
                    temp = np.convolve(X[:, ei, fi].copy(), self.coef_[oi, fi].copy())
                    temp = temp[max(-smin, 0):]
                    m = min(temp.shape[0], n_t - offset)
                    for k in range(m):
                        out[offset + k, ei, oi] += temp[k]
        out += self.intercept_[None, None, :]
        return out

    def predict(self, X):
        """Returns (n_times[, n_epochs], n_outputs) (mne additionally squeezes)."""
        Xa = np.asarray(X, dtype=float)
        o = self._predict3(_as_tef(Xa))
        if static.len(Xa.shape) == 2:
            return o[:, 0, :].copy()
        else:
            return o


def _times_to_delays(tmin: float, tmax: float, sfreq: float) -> np.ndarray[int, 1]:
    return np.arange(int(np.round(tmin * sfreq)), int(np.round(tmax * sfreq) + 1))


def _delay_time_series(X: np.ndarray[float, 3], delays: np.ndarray[int, 1],
                       fill_mean: bool) -> np.ndarray[float, 4]:
    n_t = X.shape[0]
    n_ep = X.shape[1]
    n_f = X.shape[2]
    nd = delays.shape[0]
    out = np.zeros((n_t, n_ep, n_f, nd))
    mean_value = X.reshape((n_t * n_ep, n_f)).mean(axis=0)
    for ii in range(nd):
        d = int(delays[ii])
        if fill_mean:
            out[:, :, :, ii] = mean_value[None, None, :]
        if d < 0:
            use = X[-d:]
            dst_lo = 0
        elif d > 0:
            use = X[:n_t - d]
            dst_lo = d
        else:
            use = X
            dst_lo = 0
        m = use.shape[0]
        out[dst_lo:dst_lo + m, :, :, ii] = use
        if fill_mean:
            um = use.reshape((m * n_ep, n_f)).mean(axis=0)
            out[dst_lo:dst_lo + m, :, :, ii] += (mean_value - um)[None, None, :]
    return out


def _reshape_for_est(Xd: np.ndarray[float, 4]) -> np.ndarray[float, 2]:
    """(n_t, n_ep, n_f, n_d) -> (n_t * n_ep, n_f * n_d) with Fortran sample order."""
    n_t = Xd.shape[0]
    n_ep = Xd.shape[1]
    nfd = Xd.shape[2] * Xd.shape[3]
    X3 = Xd.reshape((n_t, n_ep, nfd))
    out = np.empty((n_t * n_ep, nfd))
    for e in range(n_ep):
        out[e * n_t:(e + 1) * n_t] = X3[:, e, :]
    return out


def _r2_raw(yt: np.ndarray[float, 2], yp: np.ndarray[float, 2]) -> np.ndarray[float, 1]:
    n_out = yt.shape[1]
    out = np.empty(n_out)
    for o in range(n_out):
        m = yt[:, o].mean()
        num = np.sum((yt[:, o] - yp[:, o]) ** 2)
        den = np.sum((yt[:, o] - m) ** 2)
        if den != 0:
            out[o] = 1.0 - num / den
        elif num == 0:
            out[o] = 1.0
        else:
            out[o] = 0.0
    return out


def _corr_raw(yt: np.ndarray[float, 2], yp: np.ndarray[float, 2]) -> np.ndarray[float, 1]:
    from scipy.stats import pearsonr
    n_out = yt.shape[1]
    out = np.empty(n_out)
    for o in range(n_out):
        r = pearsonr(yt[:, o].copy(), yp[:, o].copy())
        out[o] = float(r[0])
    return out


class ReceptiveField[E]:
    """Fit a receptive field model (mne.decoding.ReceptiveField).

    estimator: a float (ridge alpha for TimeDelayingRidge, mne default 0.)
    or an sklearn regressor (e.g. Ridge)."""
    tmin: float
    tmax: float
    sfreq: float
    estimator: E
    fit_intercept: Optional[bool]
    fit_intercept_: bool
    scoring: str
    patterns: bool
    edge_correction: bool
    delays_: np.ndarray[int, 1]
    coef_: np.ndarray[float, 3]
    patterns_: np.ndarray[float, 3]
    _y_dim: int
    _tdr: TimeDelayingRidge
    _est: E
    _fitted: bool

    def __init__(self, tmin: float, tmax: float, sfreq: float, feature_names = None,
                 estimator: E = 0.0, fit_intercept: Optional[bool] = None, scoring: str = "r2",
                 patterns: bool = False, n_jobs = None, edge_correction: bool = True):
        if scoring != "r2" and scoring != "corrcoef":
            raise ValueError(f"scoring must be one of ['corrcoef', 'r2'], got {scoring}")
        if tmin > tmax:
            raise ValueError(f"tmin ({tmin}) must be at most tmax ({tmax})")
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.sfreq = float(sfreq)
        self.estimator = estimator
        self.fit_intercept = fit_intercept
        self.fit_intercept_ = True
        self.scoring = scoring
        self.patterns = patterns
        self.edge_correction = edge_correction
        self.delays_ = np.zeros(0, dtype=int)
        self.coef_ = np.zeros((0, 0, 0))
        self.patterns_ = np.zeros((0, 0, 0))
        self._y_dim = 0
        self._tdr = TimeDelayingRidge(tmin, tmax, sfreq)
        self._est = estimator
        self._fitted = False

    @property
    def valid_samples_(self) -> Tuple[int, int]:
        """(start, stop) of the valid-sample slice (mne returns a slice)."""
        lo = int(self.delays_[-1]) if self.delays_[-1] > 0 else 0
        return lo, 0 if self.delays_[0] >= 0 else int(self.delays_[0])

    def _is_tdr(self) -> Literal[bool]:
        return isinstance(self.estimator, float) or isinstance(self.estimator, int)

    def _fit3(self, X: np.ndarray[float, 3], y: np.ndarray[float, 3]):
        from sklearn.base import clone
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y do not have the same n_times\n{X.shape[0]} != {y.shape[0]}")
        if X.shape[1] != y.shape[1]:
            raise ValueError(f"X and y do not have the same n_epochs\n{X.shape[1]} != {y.shape[1]}")
        self.delays_ = _times_to_delays(self.tmin, self.tmax, self.sfreq)
        n_t = X.shape[0]
        n_ep = X.shape[1]
        n_f = X.shape[2]
        n_out = y.shape[2]
        nd = self.delays_.shape[0]
        fi = True if self.fit_intercept is None else self.fit_intercept.__val__()
        if self._is_tdr():
            self.fit_intercept_ = fi
            self._tdr = TimeDelayingRidge(self.tmin, self.tmax, self.sfreq,
                                          alpha=float(self.estimator), fit_intercept=fi,
                                          edge_correction=self.edge_correction)
            self._tdr._fit3(X, y)
            coef = self._tdr.coef_.copy()
        else:
            self._est = clone(self.estimator)
            if self.fit_intercept is not None and self._est.fit_intercept != self.fit_intercept.__val__():
                raise ValueError("Estimator fit_intercept != initialization fit_intercept")
            self.fit_intercept_ = self._est.fit_intercept
            Xd = _reshape_for_est(_delay_time_series(X, self.delays_, self.fit_intercept_))
            y2 = np.empty((n_t * n_ep, n_out))
            for e in range(n_ep):
                y2[e * n_t:(e + 1) * n_t] = y[:, e, :]
            if n_out == 1 and self._y_dim == 1:
                self._est.fit(Xd, y2[:, 0].copy())
            else:
                self._est.fit(Xd, y2)
            c = np.asarray(self._est.coef_, dtype=float)
            coef = np.ascontiguousarray(c).reshape((n_out, n_f, nd))
        self.coef_ = coef
        if self.patterns:
            n_total = n_t * n_ep
            if n_total < 2:
                raise ValueError(f"Cannot compute patterns with only one sample; got n_samples = {n_total}.")
            if self._is_tdr():
                cov_ = self._tdr.cov_ / float(n_total - 1)
            else:
                Xd2 = _reshape_for_est(_delay_time_series(X, self.delays_, self.fit_intercept_))
                Xc = Xd2 - Xd2.mean(0)[None, :]
                cov_ = (Xc.T @ Xc) / float(Xc.shape[0] - 1)
            cf = np.ascontiguousarray(coef.reshape((n_out, n_f * nd)).T)
            if n_out != 1:
                y2 = np.empty((n_total, n_out))
                for e in range(n_ep):
                    y2[e * n_t:(e + 1) * n_t] = y[:, e, :]
                yc = y2 - y2.mean(0)[None, :]
                inv_Y = np.linalg.pinv((yc.T @ yc) / float(n_total - 1))
                pat = cov_ @ (cf @ inv_Y)
            else:
                pat = cov_ @ (cf * (1.0 / float(n_total - 1)))
            self.patterns_ = np.ascontiguousarray(pat.T).reshape((n_out, n_f, nd))
        self._fitted = True
        return self

    def fit(self, X, y):
        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y, dtype=float)
        self._y_dim = static.len(ya.shape)
        if static.len(Xa.shape) == 3:
            return self._fit3(_as_tef(Xa), _y_as_teo(ya, 3))
        else:
            return self._fit3(_as_tef(Xa), _y_as_teo(ya, 2))

    def _predict3(self, X: np.ndarray[float, 3]) -> np.ndarray[float, 3]:
        if not self._fitted:
            raise ValueError("Estimator has not been fit yet.")
        if self._is_tdr():
            return self._tdr._predict3(X)
        n_t = X.shape[0]
        n_ep = X.shape[1]
        Xd = _reshape_for_est(_delay_time_series(X, self.delays_, self.fit_intercept_))
        p = np.asarray(self._est.predict(Xd), dtype=float)
        n_out = self.coef_.shape[0]
        p2 = np.ascontiguousarray(p).reshape((n_t * n_ep, n_out))
        out = np.empty((n_t, n_ep, n_out))
        for e in range(n_ep):
            out[:, e, :] = p2[e * n_t:(e + 1) * n_t]
        return out

    def predict(self, X):
        """Returns (n_times[, n_epochs], n_outputs)."""
        Xa = np.asarray(X, dtype=float)
        o = self._predict3(_as_tef(Xa))
        if static.len(Xa.shape) == 2:
            return o[:, 0, :].copy()
        else:
            return o

    def score(self, X, y) -> np.ndarray[float, 1]:
        Xa = np.asarray(X, dtype=float)
        if static.len(Xa.shape) == 3:
            yy = _y_as_teo(y, 3)
        else:
            yy = _y_as_teo(y, 2)
        yp = self._predict3(_as_tef(Xa))
        n_t = yy.shape[0]
        lo = int(self.delays_[-1]) if self.delays_[-1] > 0 else 0
        hi = n_t + int(self.delays_[0]) if self.delays_[0] < 0 else n_t
        nv = hi - lo
        n_ep = yy.shape[1]
        n_out = yy.shape[2]
        yt2 = np.empty((nv * n_ep, n_out))
        yp2 = np.empty((nv * n_ep, n_out))
        for e in range(n_ep):
            yt2[e * nv:(e + 1) * nv] = yy[lo:hi, e, :]
            yp2[e * nv:(e + 1) * nv] = yp[lo:hi, e, :]
        if self.scoring == "r2":
            return _r2_raw(yt2, yp2)
        return _corr_raw(yt2, yp2)
