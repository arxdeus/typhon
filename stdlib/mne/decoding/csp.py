# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne.decoding.CSP routed through the _GEDTransformer default path
# (mne/decoding/_ged.py, _covs_ged.py, _mod_ged.py, csp.py).

import numpy as np
import scipy.linalg
from .._info import Info, create_info
from .._cov import _regularized_covariance, _compute_rank_data, _restr_mat, pinv


def _ajd_pham(X, eps: float = 1e-6, max_iter: int = 15):
    """Approximate joint diagonalization based on Pham's algorithm."""
    n_epochs = X.shape[0]
    n_times = X.shape[1]
    A = np.concatenate([X[i] for i in range(n_epochs)], axis=0).T.copy()
    n_m = A.shape[1]
    V = np.eye(n_times)
    epsilon = n_times * (n_times - 1) * eps
    for it in range(max_iter):
        decr = 0.0
        for ii in range(1, n_times):
            for jj in range(ii):
                Ii = np.arange(ii, n_m, n_times)
                Ij = np.arange(jj, n_m, n_times)
                c1 = A[ii, Ii]
                c2 = A[jj, Ij]
                g12 = np.mean(A[ii, Ij] / c1)
                g21 = np.mean(A[ii, Ij] / c2)
                omega21 = np.mean(c1 / c2)
                omega12 = np.mean(c2 / c1)
                omega = np.sqrt(omega12 * omega21)
                tmp = np.sqrt(omega21 / omega12)
                tmp1 = (tmp * g12 + g21) / (omega + 1)
                tmp2 = (tmp * g12 - g21) / max(omega - 1, 1e-9)
                h12 = tmp1 + tmp2
                h21 = (tmp1 - tmp2) / tmp
                decr += n_epochs * (g12 * h12 + g21 * h21) / 2.0
                t = 1.0
                t = t + np.sqrt(complex(t * t - h12 * h21)).real
                t00 = 1.0
                t01 = -h12 / t
                t10 = -h21 / t
                t11 = 1.0
                # A[[ii, jj], :] = tau @ A[[ii, jj], :]
                for k in range(n_m):
                    ai = A[ii, k]
                    aj = A[jj, k]
                    A[ii, k] = t00 * ai + t01 * aj
                    A[jj, k] = t10 * ai + t11 * aj
                # column update: equivalent to the F-order reshape trick,
                # each pair (A[:, Ii[e]], A[:, Ij[e]]) is right-multiplied by tau.T
                for e in range(n_epochs):
                    ci = ii + e * n_times
                    cj = jj + e * n_times
                    for r in range(n_times):
                        ai = A[r, ci]
                        aj = A[r, cj]
                        A[r, ci] = ai * t00 + aj * t01
                        A[r, cj] = ai * t10 + aj * t11
                for k in range(n_times):
                    vi = V[ii, k]
                    vj = V[jj, k]
                    V[ii, k] = t00 * vi + t01 * vj
                    V[jj, k] = t10 * vi + t11 * vj
        if decr < epsilon:
            break
    D = np.reshape(A, (n_times, -1, n_times)).transpose(1, 0, 2)
    return V, D


def _normalize_eigenvectors(evecs, covs, sample_weights):
    w = sample_weights / sample_weights.sum()
    mean_cov = np.zeros((covs.shape[1], covs.shape[2]))
    for k in range(covs.shape[0]):
        mean_cov += w[k] * covs[k]
    for ii in range(evecs.shape[1]):
        v = evecs[:, ii].copy()
        tmp = np.dot(np.dot(v, mean_cov), v)
        evecs[:, ii] = v / np.sqrt(tmp)
    return evecs


def _compute_mutual_info(covs, sample_weights, evecs):
    class_probas = sample_weights / sample_weights.sum()
    out = np.zeros(evecs.shape[1])
    for jj in range(evecs.shape[1]):
        aa = 0.0
        bb = 0.0
        v = evecs[:, jj].copy()
        for k in range(covs.shape[0]):
            prob = class_probas[k]
            tmp = np.dot(np.dot(v, covs[k]), v)
            aa += prob * np.log(np.sqrt(tmp))
            bb += prob * (tmp ** 2 - 1)
        out[jj] = -(aa + 3.0 / 16 * bb ** 2)
    return out


class _CSP[TI: Literal[str]]:
    """M/EEG signal decomposition using the Common Spatial Patterns (CSP).

    TI is transform_into, fixed at compile time so transform() has a
    static return type: (n_epochs, n_components) for 'average_power',
    (n_epochs, n_components, n_times) for 'csp_space'.
    """
    n_components: int
    reg_kind: str
    reg_value: float
    reg_repr: str
    log: Optional[bool]
    cov_est: str
    transform_into: str
    norm_trace: bool
    restr_type: Optional[str]
    info: Optional[Info]
    rank_dict: Optional[Dict[str, int]]
    rank_str: Optional[str]
    component_order: str

    classes_: np.ndarray[int, 1]
    filters_: np.ndarray[float, 2]
    patterns_: np.ndarray[float, 2]
    evals_: Optional[np.ndarray[float, 1]]
    sorter_: np.ndarray[int, 1]
    mean_: np.ndarray[float, 1]
    std_: np.ndarray[float, 1]
    n_features_in_: int
    _fitted: bool

    def __init__(self, n_components: int = 4, reg = None, log: Optional[bool] = None,
                 cov_est: str = "concat", transform_into: str = "average_power",
                 norm_trace: bool = False, cov_method_params = None,
                 restr_type: Optional[str] = "restricting", info: Optional[Info] = None,
                 rank = None, component_order: str = "mutual_info"):
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
            if not (0 <= self.reg_value <= 1):
                raise ValueError(f"shrinkage must be between 0 and 1, got {self.reg_value}")
        if not isinstance(cov_method_params, NoneType):
            raise ValueError("cov_method_params is not supported in Typhon mne")
        self.log = log
        self.cov_est = cov_est
        self.transform_into = transform_into
        self.norm_trace = norm_trace
        self.restr_type = restr_type
        self.info = info
        self.rank_dict = None
        self.rank_str = None
        if isinstance(rank, NoneType):
            pass
        elif isinstance(rank, str):
            self.rank_str = rank
        else:
            d = Dict[str, int]()
            for k, v in rank.items():
                d[k] = v
            self.rank_dict = d
        self.component_order = component_order
        self.evals_ = None
        self.classes_ = np.zeros(0, dtype=int)
        self.filters_ = np.zeros((0, 0))
        self.patterns_ = np.zeros((0, 0))
        self.sorter_ = np.zeros(0, dtype=int)
        self.mean_ = np.zeros(0)
        self.std_ = np.zeros(0)
        self._fitted = False
        self.n_features_in_ = 0

    def __repr__(self) -> str:
        return f"CSP(n_components={self.n_components}, reg={self.reg_repr}, log={self.log}, transform_into='{self.transform_into}')"

    def _validate_params(self, y):
        if self.cov_est != "concat" and self.cov_est != "epoch":
            raise ValueError(f"Invalid value for the 'cov_est' parameter. Allowed values are 'concat' and 'epoch', but got '{self.cov_est}' instead.")
        if self.transform_into != "average_power" and self.transform_into != "csp_space":
            raise ValueError(f"Invalid value for the 'transform_into' parameter. Allowed values are 'average_power' and 'csp_space', but got '{self.transform_into}' instead.")
        if self.transform_into == "csp_space" and self.log is not None:
            raise TypeError("log must be an instance of None when transform_into is 'csp_space'")
        if self.component_order != "mutual_info" and self.component_order != "alternate":
            raise ValueError(f"Invalid value for the 'component_order' parameter. Allowed values are 'mutual_info' and 'alternate', but got '{self.component_order}' instead.")
        if self.restr_type is not None:
            if self.restr_type.__val__() != "restricting":
                raise ValueError("restr_type='whitening' is not supported in Typhon mne (use 'restricting' or None)")
        if self.n_components <= 0:
            raise ValueError("Invalid value for the 'n_components' parameter. Allowed are positive integers or None, but got a non-positive integer instead.")
        self.classes_ = np.unique(y)
        n_classes = self.classes_.shape[0]
        if n_classes < 2:
            raise ValueError(f"y should be a 1d array with more than two classes, but got {n_classes} class from {y}")
        elif n_classes > 2 and self.component_order == "alternate":
            raise ValueError(f"component_order='alternate' requires two classes, but data contains {n_classes} classes; use component_order='mutual_info' instead.")

    def _estimate(self, X, y):
        n_channels = X.shape[1]
        info = Info(["0"], 1.0, ["mag"])
        rank = Dict[str, int]()
        given_dict = self.rank_dict is not None
        if self.info is None:
            info = create_info(n_channels, 1000.0, "mag")
            if given_dict:
                s = 0
                for v in self.rank_dict.__val__().values():
                    s += v
                rank["mag"] = s
        else:
            info = self.info.__val__()
            if given_dict:
                rank = self.rank_dict.__val__()
        if not given_dict:
            Xh = np.concatenate([X[i] for i in range(X.shape[0])], axis=1)
            kind = "estimated" if self.rank_str is None else self.rank_str.__val__()
            rank = _compute_rank_data(Xh, info, kind)
        n_cls = self.classes_.shape[0]
        covs = np.zeros((n_cls, n_channels, n_channels))
        weights = np.zeros(n_cls)
        for ci in range(n_cls):
            sel = List[int]()
            for i in range(y.shape[0]):
                if y[i] == self.classes_[ci]:
                    sel.append(i)
            xc = X[np.array(sel)]
            cov = np.zeros((n_channels, n_channels))
            w = 0.0
            if self.cov_est == "concat":
                xcat = xc.transpose(1, 0, 2).reshape((n_channels, -1))
                cov = _regularized_covariance(xcat, self.reg_kind, self.reg_value, info, rank)
                w = float(n_channels)
            else:
                for k in range(xc.shape[0]):
                    cov += _regularized_covariance(xc[k], self.reg_kind, self.reg_value, info, rank)
                cov /= xc.shape[0]
                w = float(xc.shape[0])
            if self.norm_trace:
                cov /= np.trace(cov)
            covs[ci] = cov
            weights[ci] = w
        C_ref = covs.mean(axis=0)
        return covs, C_ref, info, rank, weights

    def fit(self, X, y):
        Xa = np.asarray(X).astype(float)
        if Xa.ndim != 3:
            compile_error("X must be 3-D (n_epochs, n_channels, n_times)")
        ya = np.asarray(y)
        if ya.shape[0] != Xa.shape[0]:
            raise ValueError(f"Found input variables with inconsistent numbers of samples: [{Xa.shape[0]}, {ya.shape[0]}]")
        self.n_features_in_ = Xa.shape[1]
        self._validate_params(ya)
        covs, C_ref, info, rank, weights = self._estimate(Xa, ya)
        n_ch = Xa.shape[1]
        restr = np.eye(n_ch)
        use_restr = self.restr_type is not None
        if use_restr:
            restr = _restr_mat(C_ref, info, rank)
        evals = np.zeros(0)
        evecs = np.zeros((0, 0))
        has_evals = True
        if covs.shape[0] > 2:
            has_evals = False
            if not use_restr:
                V, _ = _ajd_pham(covs)
                evecs = V
            else:
                rc = np.zeros((covs.shape[0], restr.shape[0], restr.shape[0]))
                for k in range(covs.shape[0]):
                    rc[k] = restr @ covs[k] @ restr.T
                V, _ = _ajd_pham(rc)
                ev = _normalize_eigenvectors(V.T.copy(), rc, weights)
                evecs = restr.T @ ev
        else:
            S = covs[0]
            R = covs[1]
            if not use_restr:
                w, v = scipy.linalg.eigh(S, R)
                evals = w
                evecs = v
            else:
                S_r = restr @ S @ restr.T
                R_r = restr @ R @ restr.T
                # CSP passes R_func=sum -> R_restr = S_restr + R_restr
                R_r = S_r + R_r
                w, v = scipy.linalg.eigh(S_r, R_r)
                evals = w
                evecs = restr.T @ v
        # _csp_mod
        n_classes = weights.shape[0]
        ix = np.zeros(0, dtype=int)
        if self.component_order == "mutual_info" and n_classes > 2:
            mi = _compute_mutual_info(covs, weights, evecs)
            ix = np.argsort(mi)[::-1].copy()
        elif self.component_order == "mutual_info" and n_classes == 2:
            ix = np.argsort(np.abs(evals - 0.5))[::-1].copy()
        else:
            i = np.argsort(evals)
            ix = np.empty_like(i)
            half = i.shape[0] // 2
            ix[1::2] = i[:half]
            ix[0::2] = i[half:][::-1]
        if has_evals:
            self.evals_ = evals[ix].copy()
        else:
            self.evals_ = None
        evecs = evecs[:, ix].copy()
        self.sorter_ = ix
        self.filters_ = evecs.T.copy()
        self.patterns_ = pinv(evecs)
        self._fitted = True
        pick = self.filters_[:self.n_components]
        n_ep = Xa.shape[0]
        nc = pick.shape[0]
        feats = np.zeros((n_ep, nc))
        for e in range(n_ep):
            proj = np.dot(pick, Xa[e])
            feats[e] = (proj ** 2).mean(axis=1)
        self.mean_ = feats.mean(axis=0)
        self.std_ = feats.std(axis=0)
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise ValueError("This CSP instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

    def _project(self, X):
        self._check_fitted()
        Xa = np.asarray(X).astype(float)
        if Xa.ndim != 3:
            compile_error("X must be 3-D (n_epochs, n_channels, n_times)")
        if Xa.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xa.shape[1]} features, but CSP is expecting {self.n_features_in_} features as input.")
        pick = self.filters_[:self.n_components]
        out = np.zeros((Xa.shape[0], pick.shape[0], Xa.shape[2]))
        for e in range(Xa.shape[0]):
            out[e] = pick @ Xa[e]
        return out

    def transform(self, X):
        """Average power features, or the CSP-space signals for 'csp_space'."""
        if TI == "csp_space":
            return self._project(X)
        else:
            Y = self._project(X)
            F = (Y ** 2).mean(axis=2)
            lg = True if self.log is None else self.log.__val__()
            if lg:
                F = np.log(F)
            else:
                F -= self.mean_
                F /= self.std_
            return F

    def transform_csp_space(self, X):
        return self._project(X)

    def fit_transform(self, X, y, **fit_params):
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X):
        if self.transform_into != "average_power":
            raise NotImplementedError("Can only inverse transform CSP features when transform_into is 'average_power'.")
        Xa = np.asarray(X).astype(float)
        if not (Xa.ndim == 2 and Xa.shape[1] == self.n_components):
            raise ValueError(f"X must be 2D with X[1]={self.n_components}, got X.shape={Xa.shape}")
        P = self.patterns_[:self.n_components].T
        out = np.zeros((Xa.shape[0], P.shape[0], P.shape[1]))
        for e in range(Xa.shape[0]):
            out[e] = Xa[e][None, :] * P
        return out

    def get_params(self, deep: bool = True) -> Dict[str, str]:
        d = Dict[str, str]()
        d["n_components"] = str(self.n_components)
        d["reg"] = self.reg_repr
        d["log"] = str(self.log)
        d["cov_est"] = self.cov_est
        d["transform_into"] = self.transform_into
        d["norm_trace"] = str(self.norm_trace)
        d["component_order"] = self.component_order
        return d


def CSP(n_components: int = 4, reg = None, log: Optional[bool] = None, cov_est: str = "concat",
        transform_into: Literal[str] = "average_power", norm_trace: bool = False,
        cov_method_params = None, *, restr_type: Optional[str] = "restricting",
        info: Optional[Info] = None, rank = None, component_order: str = "mutual_info"):
    """mne.decoding.CSP. transform_into must be a compile-time string literal."""
    if transform_into != "average_power" and transform_into != "csp_space":
        compile_error("transform_into must be 'average_power' or 'csp_space'")
    return _CSP[transform_into](n_components, reg, log, cov_est, transform_into, norm_trace,
                                cov_method_params, restr_type, info, rank, component_order)


class _SPoC[TI: Literal[str]]:
    """Source Power Comodulation (mne.decoding.SPoC)."""
    n_components: int
    reg_kind: str
    reg_value: float
    log: Optional[bool]
    restr_type: Optional[str]
    info: Optional[Info]
    rank_dict: Optional[Dict[str, int]]
    rank_str: Optional[str]
    filters_: np.ndarray[float, 2]
    patterns_: np.ndarray[float, 2]
    evals_: np.ndarray[float, 1]
    sorter_: np.ndarray[int, 1]
    mean_: np.ndarray[float, 1]
    std_: np.ndarray[float, 1]
    n_features_in_: int
    _fitted: bool
    _estimator_type: ClassVar[str] = "regressor"

    def __init__(self, n_components: int = 4, reg = None, log: Optional[bool] = None,
                 restr_type: Optional[str] = "restricting", info: Optional[Info] = None,
                 rank = None):
        c = _CSP[TI](n_components, reg, log, "epoch", TI, False, None, restr_type, info, rank,
                     "mutual_info")
        self.n_components = n_components
        self.reg_kind = c.reg_kind
        self.reg_value = c.reg_value
        self.log = log
        if TI == "csp_space" and log is not None:
            raise TypeError("log must be an instance of None when transform_into is 'csp_space'")
        self.restr_type = restr_type
        self.info = info
        self.rank_dict = c.rank_dict
        self.rank_str = c.rank_str
        self.filters_ = np.zeros((0, 0))
        self.patterns_ = np.zeros((0, 0))
        self.evals_ = np.zeros(0)
        self.sorter_ = np.zeros(0, dtype=int)
        self.mean_ = np.zeros(0)
        self.std_ = np.zeros(0)
        self.n_features_in_ = 0
        self._fitted = False

    def fit(self, X, y):
        Xa = np.asarray(X).astype(float)
        if Xa.ndim != 3:
            compile_error("X must be 3-D (n_epochs, n_channels, n_times)")
        target = np.asarray(y).astype(float)
        if target.shape[0] != Xa.shape[0]:
            raise ValueError(f"Found input variables with inconsistent numbers of samples: [{Xa.shape[0]}, {target.shape[0]}]")
        self.n_features_in_ = Xa.shape[1]
        n_ep = Xa.shape[0]
        n_ch = Xa.shape[1]
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
        if not given:
            Xh = np.concatenate([Xa[i] for i in range(n_ep)], axis=1)
            kind = "estimated" if self.rank_str is None else self.rank_str.__val__()
            rank = _compute_rank_data(Xh, info, kind)
        target -= target.mean()
        target /= target.std()
        S = np.zeros((n_ch, n_ch))
        R = np.zeros((n_ch, n_ch))
        for e in range(n_ep):
            cv = _regularized_covariance(Xa[e], self.reg_kind, self.reg_value, info, rank)
            S += cv * target[e]
            R += cv
        S /= n_ep
        R /= n_ep
        if self.restr_type is not None:
            restr = _restr_mat(R, info, rank)
            S_r = restr @ S @ restr.T
            R_r = restr @ R @ restr.T
            w, v = scipy.linalg.eigh(S_r, R_r)
            evecs = restr.T @ v
        else:
            w, evecs = scipy.linalg.eigh(S, R)
        ix = np.argsort(np.abs(w))[::-1].copy()
        self.evals_ = w[ix].copy()
        evecs = evecs[:, ix].copy()
        self.sorter_ = ix
        self.filters_ = evecs.T.copy()
        self.patterns_ = pinv(evecs)
        pick = self.filters_[:self.n_components]
        feats = np.zeros((n_ep, pick.shape[0]))
        for e in range(n_ep):
            proj = np.dot(pick, Xa[e])
            feats[e] = (proj ** 2).mean(axis=1)
        self.mean_ = feats.mean(axis=0)
        self.std_ = feats.std(axis=0)
        self._fitted = True
        return self

    def _project(self, X):
        if not self._fitted:
            raise ValueError("This SPoC instance is not fitted yet.")
        Xa = np.asarray(X).astype(float)
        pick = self.filters_[:self.n_components]
        out = np.zeros((Xa.shape[0], pick.shape[0], Xa.shape[2]))
        for e in range(Xa.shape[0]):
            out[e] = pick @ Xa[e]
        return out

    def transform(self, X):
        if TI == "csp_space":
            return self._project(X)
        else:
            Y = self._project(X)
            F = (Y ** 2).mean(axis=2)
            lg = True if self.log is None else self.log.__val__()
            if lg:
                F = np.log(F)
            else:
                F -= self.mean_
                F /= self.std_
            return F

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


def SPoC(n_components: int = 4, reg = None, log: Optional[bool] = None,
         transform_into: Literal[str] = "average_power", cov_method_params = None, *,
         restr_type: Optional[str] = "restricting", info: Optional[Info] = None, rank = None):
    """mne.decoding.SPoC. transform_into must be a compile-time string literal."""
    if transform_into != "average_power" and transform_into != "csp_space":
        compile_error("transform_into must be 'average_power' or 'csp_space'")
    if not isinstance(cov_method_params, NoneType):
        raise ValueError("cov_method_params is not supported in Typhon mne")
    return _SPoC[transform_into](n_components, reg, log, restr_type, info, rank)
