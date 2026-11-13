# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

from ..base import estimator_repr as _estimator_repr
import numpy as np
from scipy.special import gammaln, digamma, betaln
from ._base import (_init_resp, _estimate_gaussian_parameters, _compute_precision_cholesky,
                    _estimate_log_gaussian_prob, _precisions_from_chol, _from_canonical,
                    _logsumexp_rows, _fit_loop, _log_det_cholesky)

def _log_wishart_norm(dof, log_det_chol, d: int):
    # per component; dof and log_det_chol are 1-d (length m)
    m = dof.shape[0]
    out = np.empty(m)
    for c in range(m):
        s = 0.0
        for a in range(d):
            s += float(gammaln(0.5 * (dof[c] - a)))
        out[c] = -(dof[c] * log_det_chol[c] + dof[c] * d * 0.5 * np.log(2.0) + s)
    return out

class _BayesianGaussianMixture[CT: Literal[str]]:
    def __repr__(self) -> str:
        return _estimator_repr(self, False)

    n_components: int
    tol: float
    reg_covar: float
    max_iter: int
    n_init: int
    init_params: str
    random_state: Optional[int]
    warm_start: bool
    weight_concentration_prior_type: str
    weight_concentration_prior: Optional[float]
    mean_precision_prior: Optional[float]
    mean_prior: Optional[np.ndarray[float, 1]]
    degrees_of_freedom_prior: Optional[float]
    covariance_prior: Optional[np.ndarray[float, 2]]
    weight_concentration_prior_: float
    mean_precision_prior_: float
    mean_prior_: np.ndarray[float, 1]
    degrees_of_freedom_prior_: float
    _cov_prior: np.ndarray[float, 3]
    _wc0: np.ndarray[float, 1]
    _wc1: np.ndarray[float, 1]
    mean_precision_: np.ndarray[float, 1]
    means_: np.ndarray[float, 2]
    _dof: np.ndarray[float, 1]
    _cov: np.ndarray[float, 3]
    _pc: np.ndarray[float, 3]
    _prec: np.ndarray[float, 3]
    weights_: np.ndarray[float, 1]
    converged_: bool
    n_iter_: int
    lower_bound_: float
    lower_bounds_: List[float]
    _fitted: bool

    def __init__(self, n_components: int, tol: float, reg_covar: float, max_iter: int, n_init: int,
                 init_params: str, weight_concentration_prior_type: str,
                 weight_concentration_prior: Optional[float], mean_precision_prior: Optional[float],
                 mean_prior, degrees_of_freedom_prior: Optional[float], covariance_prior,
                 random_state: Optional[int], warm_start: bool):
        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.warm_start = warm_start
        if weight_concentration_prior_type != "dirichlet_process" and weight_concentration_prior_type != "dirichlet_distribution":
            raise ValueError(f"invalid weight_concentration_prior_type {weight_concentration_prior_type}")
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior = weight_concentration_prior
        self.mean_precision_prior = mean_precision_prior
        self.mean_prior = None
        if mean_prior is not None:
            self.mean_prior = np.array(mean_prior, dtype=float)
        self.degrees_of_freedom_prior = degrees_of_freedom_prior
        self.covariance_prior = None
        if covariance_prior is not None:
            self.covariance_prior = np.atleast_2d(np.array(covariance_prior, dtype=float))
        self.weight_concentration_prior_ = 0.0
        self.mean_precision_prior_ = 0.0
        self.mean_prior_ = np.empty(0)
        self.degrees_of_freedom_prior_ = 0.0
        self._cov_prior = np.empty((0, 0, 0))
        self._wc0 = np.empty(0)
        self._wc1 = np.empty(0)
        self.mean_precision_ = np.empty(0)
        self.means_ = np.empty((0, 0))
        self._dof = np.empty(0)
        self._cov = np.empty((0, 0, 0))
        self._pc = np.empty((0, 0, 0))
        self._prec = np.empty((0, 0, 0))
        self.weights_ = np.empty(0)
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = -np.inf
        self.lower_bounds_ = []
        self._fitted = False

    @property
    def covariances_(self):
        return _from_canonical(self._cov, CT)

    @property
    def precisions_cholesky_(self):
        return _from_canonical(self._pc, CT)

    @property
    def precisions_(self):
        return _from_canonical(self._prec, CT)

    @property
    def degrees_of_freedom_(self):
        if CT == "tied":
            return float(self._dof[0])
        else:
            return self._dof

    @property
    def weight_concentration_(self):
        # dirichlet_process: (alpha, beta); dirichlet_distribution: (conc, empty)
        return (self._wc0, self._wc1)

    @property
    def covariance_prior_(self):
        return _from_canonical(self._cov_prior, CT)

    def _check_parameters(self, X):
        n, d = X.shape
        self.weight_concentration_prior_ = (1.0 / self.n_components if self.weight_concentration_prior is None
                                            else self.weight_concentration_prior.__val__())
        self.mean_precision_prior_ = 1.0 if self.mean_precision_prior is None else self.mean_precision_prior.__val__()
        self.mean_prior_ = X.mean(axis=0) if self.mean_prior is None else self.mean_prior.__val__()
        if self.degrees_of_freedom_prior is None:
            self.degrees_of_freedom_prior_ = float(d)
        elif self.degrees_of_freedom_prior.__val__() > d - 1.0:
            self.degrees_of_freedom_prior_ = self.degrees_of_freedom_prior.__val__()
        else:
            raise ValueError(f"The parameter 'degrees_of_freedom_prior' should be greater than {d - 1}")
        if self.covariance_prior is None:
            if CT == "full" or CT == "tied":
                Xc = X - X.mean(axis=0)
                self._cov_prior = ((Xc.T @ Xc) / float(n - 1)).reshape((1, d, d))
            elif CT == "diag":
                self._cov_prior = np.var(X, axis=0, ddof=1).reshape((1, 1, d))
            else:
                self._cov_prior = np.full((1, 1, 1), float(np.var(X, axis=0, ddof=1).mean()))
        else:
            C = self.covariance_prior.__val__()
            if CT == "full" or CT == "tied":
                self._cov_prior = C.reshape((1, d, d))
            elif CT == "diag":
                self._cov_prior = C.reshape((1, 1, d))
            else:
                self._cov_prior = C.reshape((1, 1, 1))

    def _initialize_parameters(self, X, rs):
        self._update(X, _init_resp(X, self.n_components, self.init_params, rs))

    def _update(self, X, resp):
        nk, xk, sk = _estimate_gaussian_parameters(X, resp, self.reg_covar, CT)
        k = self.n_components
        d = X.shape[1]
        # weights
        if self.weight_concentration_prior_type == "dirichlet_process":
            self._wc0 = 1.0 + nk
            tail = np.zeros(k)
            # np.hstack((np.cumsum(nk[::-1])[-2::-1], 0))
            cs = np.cumsum(nk[::-1])
            for c in range(k - 1):
                tail[c] = cs[k - 2 - c]
            tail[k - 1] = 0.0
            self._wc1 = self.weight_concentration_prior_ + tail
        else:
            self._wc0 = self.weight_concentration_prior_ + nk
            self._wc1 = np.empty(0)
        # means
        self.mean_precision_ = self.mean_precision_prior_ + nk
        self.means_ = (self.mean_precision_prior_ * self.mean_prior_ + nk[:, None] * xk) / self.mean_precision_[:, None]
        # wishart
        mpp = self.mean_precision_prior_
        if CT == "full":
            self._dof = self.degrees_of_freedom_prior_ + nk
            cov = np.empty((k, d, d))
            for c in range(k):
                diff = xk[c] - self.mean_prior_
                cov[c, :, :] = (self._cov_prior[0] + nk[c] * sk[c]
                                + nk[c] * mpp / self.mean_precision_[c] * np.outer(diff, diff))
                cov[c, :, :] /= self._dof[c]
            self._cov = cov
        elif CT == "tied":
            dof = self.degrees_of_freedom_prior_ + float(nk.sum()) / k
            self._dof = np.full(1, dof)
            diff = xk - self.mean_prior_
            C = (self._cov_prior[0] + sk[0] * float(nk.sum()) / k
                 + mpp / k * ((nk / self.mean_precision_ * diff.T) @ diff))
            C /= dof
            self._cov = C.reshape((1, d, d))
        elif CT == "diag":
            self._dof = self.degrees_of_freedom_prior_ + nk
            diff = xk - self.mean_prior_
            C = self._cov_prior[0, 0][None, :] + nk[:, None] * (sk[0] + (mpp / self.mean_precision_)[:, None] * diff ** 2)
            C /= self._dof[:, None]
            self._cov = C.reshape((1, k, d))
        else:
            self._dof = self.degrees_of_freedom_prior_ + nk
            diff = xk - self.mean_prior_
            C = self._cov_prior[0, 0, 0] + nk * (sk[0, 0] + mpp / self.mean_precision_ * np.mean(diff ** 2, axis=1))
            C /= self._dof
            self._cov = C.reshape((1, 1, k))
        self._pc = _compute_precision_cholesky(self._cov, CT)

    def _m_step(self, X, log_resp):
        self._update(X, np.exp(log_resp))

    def _estimate_log_weights(self):
        if self.weight_concentration_prior_type == "dirichlet_process":
            dsum = digamma(self._wc0 + self._wc1)
            da = digamma(self._wc0)
            db = digamma(self._wc1)
            cs = np.cumsum(db - dsum)
            k = self._wc0.shape[0]
            shift = np.zeros(k)
            for c in range(1, k):
                shift[c] = cs[c - 1]
            return da - dsum + shift
        else:
            return digamma(self._wc0) - float(digamma(float(np.sum(self._wc0))))

    def _dof_per_comp(self):
        if CT == "tied":
            return np.full(self.n_components, self._dof[0])
        else:
            return self._dof

    def _estimate_weighted_log_prob(self, X):
        d = X.shape[1]
        dof = self._dof_per_comp()
        log_gauss = _estimate_log_gaussian_prob(X, self.means_, self._pc, CT) - 0.5 * d * np.log(dof)
        log_lambda = np.full(dof.shape[0], d * np.log(2.0))
        for a in range(d):
            log_lambda += digamma(0.5 * (dof - a))
        return log_gauss + 0.5 * (log_lambda - d / self.mean_precision_) + self._estimate_log_weights()

    def _estimate_log_prob_resp(self, X):
        w = self._estimate_weighted_log_prob(X)
        lpn = _logsumexp_rows(w)
        return lpn, w - lpn[:, None]

    def _e_step(self, X):
        lpn, lr = self._estimate_log_prob_resp(X)
        return float(np.mean(lpn)), lr

    def _compute_lower_bound(self, log_resp, log_prob_norm: float) -> float:
        d = self.mean_prior_.shape[0]
        k = self.n_components
        ld = _log_det_cholesky(self._pc, CT, k, d)
        log_wishart = 0.0
        if CT == "tied":
            ldc = np.full(1, ld[0] - 0.5 * d * np.log(self._dof[0]))
            log_wishart = k * float(_log_wishart_norm(self._dof, ldc, d)[0])
        else:
            ldc = ld - 0.5 * d * np.log(self._dof)
            log_wishart = float(np.sum(_log_wishart_norm(self._dof, ldc, d)))
        log_norm_weight = 0.0
        if self.weight_concentration_prior_type == "dirichlet_process":
            log_norm_weight = -float(np.sum(betaln(self._wc0, self._wc1)))
        else:
            log_norm_weight = float(gammaln(float(np.sum(self._wc0)))) - float(np.sum(gammaln(self._wc0)))
        R = np.exp(log_resp)
        return (-float(np.sum(R * log_resp)) - log_wishart - log_norm_weight
                - 0.5 * d * float(np.sum(np.log(self.mean_precision_))))

    def _get_parameters(self):
        return (self._wc0.copy(), self._wc1.copy(), self.mean_precision_.copy(), self.means_.copy(),
                self._dof.copy(), self._cov.copy(), self._pc.copy())

    def _set_parameters(self, params):
        self._wc0, self._wc1, self.mean_precision_, self.means_, self._dof, self._cov, self._pc = params
        if self.weight_concentration_prior_type == "dirichlet_process":
            s = self._wc0 + self._wc1
            tmp = self._wc1 / s
            k = s.shape[0]
            cp = np.ones(k)
            acc = 1.0
            for c in range(1, k):
                acc *= tmp[c - 1]
                cp[c] = acc
            w = self._wc0 / s * cp
            self.weights_ = w / float(np.sum(w))
        else:
            self.weights_ = self._wc0 / float(np.sum(self._wc0))
        self._prec = _precisions_from_chol(self._pc, CT)

    def fit_predict(self, X, y = None):
        Xa = np.asarray(X, dtype=float)
        self._check_parameters(Xa)
        do_init = not (self.warm_start and self._fitted)
        rs = np.random.RandomState(self.random_state)
        labels = _fit_loop(self, Xa, self.n_components, self.n_init, self.max_iter, self.tol, rs, do_init)
        self._fitted = True
        return labels

    def fit(self, X, y = None):
        self.fit_predict(X)
        return self

    def score_samples(self, X):
        return _logsumexp_rows(self._estimate_weighted_log_prob(np.asarray(X, dtype=float)))

    def score(self, X, y = None) -> float:
        return float(np.mean(self.score_samples(X)))

    def predict(self, X):
        return np.argmax(self._estimate_weighted_log_prob(np.asarray(X, dtype=float)), axis=1)

    def predict_proba(self, X):
        _, lr = self._estimate_log_prob_resp(np.asarray(X, dtype=float))
        return np.exp(lr)

def BayesianGaussianMixture(*, n_components: int = 1, covariance_type: Literal[str] = "full",
                            tol: float = 1e-3, reg_covar: float = 1e-6, max_iter: int = 100,
                            n_init: int = 1, init_params: str = "kmeans",
                            weight_concentration_prior_type: str = "dirichlet_process",
                            weight_concentration_prior: Optional[float] = None,
                            mean_precision_prior: Optional[float] = None, mean_prior = None,
                            degrees_of_freedom_prior: Optional[float] = None, covariance_prior = None,
                            random_state: Optional[int] = None, warm_start: bool = False,
                            verbose: int = 0, verbose_interval: int = 10):
    if covariance_type != "full" and covariance_type != "tied" and covariance_type != "diag" and covariance_type != "spherical":
        compile_error("covariance_type must be 'full', 'tied', 'diag' or 'spherical'")
    return _BayesianGaussianMixture[covariance_type](
        n_components, tol, reg_covar, max_iter, n_init, init_params, weight_concentration_prior_type,
        weight_concentration_prior, mean_precision_prior, mean_prior, degrees_of_freedom_prior,
        covariance_prior, random_state, warm_start)
