# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

from ..base import estimator_repr as _estimator_repr
import numpy as np
from ._base import (_init_resp, _estimate_gaussian_parameters, _compute_precision_cholesky,
                    _precision_cholesky_from_precisions, _to_canonical, _estimate_log_gaussian_prob,
                    _precisions_from_chol, _from_canonical, _logsumexp_rows, _fit_loop)

class _GaussianMixture[CT: Literal[str]]:
    def __repr__(self) -> str:
        return _estimator_repr(self, False)

    """GaussianMixture with the covariance_type CT fixed at compile time (so
    covariances_ / precisions_ have sklearn's per-type shapes)."""
    n_components: int
    covariance_type: str
    tol: float
    reg_covar: float
    max_iter: int
    n_init: int
    init_params: str
    random_state: Optional[int]
    warm_start: bool
    weights_init: Optional[np.ndarray[float, 1]]
    means_init: Optional[np.ndarray[float, 2]]
    precisions_init: Optional[np.ndarray[float, 3]]
    weights_: np.ndarray[float, 1]
    means_: np.ndarray[float, 2]
    _cov: np.ndarray[float, 3]
    _pc: np.ndarray[float, 3]
    _prec: np.ndarray[float, 3]
    converged_: bool
    n_iter_: int
    lower_bound_: float
    lower_bounds_: List[float]
    _fitted: bool

    def __init__(self, n_components: int, covariance_type: str, tol: float, reg_covar: float,
                 max_iter: int, n_init: int, init_params: str, weights_init, means_init,
                 precisions_init, random_state: Optional[int], warm_start: bool):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.warm_start = warm_start
        self.weights_init = None
        self.means_init = None
        self.precisions_init = None
        if weights_init is not None:
            self.weights_init = np.array(weights_init, dtype=float)
        if means_init is not None:
            self.means_init = np.array(means_init, dtype=float)
        if precisions_init is not None:
            P = np.asarray(precisions_init, dtype=float)
            d = P.shape[-1]
            if CT == "spherical":
                d = 1
            self.precisions_init = _to_canonical(P, CT, n_components, d)
        self.weights_ = np.empty(0)
        self.means_ = np.empty((0, 0))
        self._cov = np.empty((0, 0, 0))
        self._pc = np.empty((0, 0, 0))
        self._prec = np.empty((0, 0, 0))
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

    def _initialize_parameters(self, X, rs):
        if self.weights_init is None or self.means_init is None or self.precisions_init is None:
            self._initialize(X, _init_resp(X, self.n_components, self.init_params, rs), True)
        else:
            self._initialize(X, np.empty((0, 0)), False)

    def _initialize(self, X, resp, have_resp: bool):
        n = X.shape[0]
        weights = np.empty(0)
        means = np.empty((0, 0))
        cov = np.empty((0, 0, 0))
        if have_resp:
            weights, means, cov = _estimate_gaussian_parameters(X, resp, self.reg_covar, CT)
            if self.weights_init is None:
                weights /= float(n)
        self.weights_ = weights if self.weights_init is None else self.weights_init.__val__().copy()
        self.means_ = means if self.means_init is None else self.means_init.__val__().copy()
        if self.precisions_init is None:
            self._cov = cov
            self._pc = _compute_precision_cholesky(cov, CT)
        else:
            self._pc = _precision_cholesky_from_precisions(self.precisions_init.__val__(), CT)

    def _m_step(self, X, log_resp):
        self.weights_, self.means_, self._cov = _estimate_gaussian_parameters(
            X, np.exp(log_resp), self.reg_covar, CT)
        self.weights_ /= float(np.sum(self.weights_))
        self._pc = _compute_precision_cholesky(self._cov, CT)

    def _estimate_weighted_log_prob(self, X):
        return _estimate_log_gaussian_prob(X, self.means_, self._pc, CT) + np.log(self.weights_)

    def _estimate_log_prob_resp(self, X):
        w = self._estimate_weighted_log_prob(X)
        lpn = _logsumexp_rows(w)
        return lpn, w - lpn[:, None]

    def _e_step(self, X):
        lpn, lr = self._estimate_log_prob_resp(X)
        return float(np.mean(lpn)), lr

    def _compute_lower_bound(self, log_resp, log_prob_norm: float) -> float:
        return log_prob_norm

    def _get_parameters(self):
        return (self.weights_.copy(), self.means_.copy(), self._cov.copy(), self._pc.copy())

    def _set_parameters(self, params):
        self.weights_, self.means_, self._cov, self._pc = params
        self._prec = _precisions_from_chol(self._pc, CT)

    def fit_predict(self, X, y = None):
        Xa = np.asarray(X, dtype=float)
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

    def _n_parameters(self) -> int:
        d = self.means_.shape[1]
        k = self.n_components
        cov_params = 0.0
        if CT == "full":
            cov_params = k * d * (d + 1) / 2.0
        elif CT == "diag":
            cov_params = float(k * d)
        elif CT == "tied":
            cov_params = d * (d + 1) / 2.0
        else:
            cov_params = float(k)
        return int(cov_params + d * k + k - 1)

    def bic(self, X) -> float:
        Xa = np.asarray(X, dtype=float)
        n = Xa.shape[0]
        return -2 * self.score(Xa) * n + self._n_parameters() * np.log(float(n))

    def aic(self, X) -> float:
        Xa = np.asarray(X, dtype=float)
        return -2 * self.score(Xa) * Xa.shape[0] + 2 * self._n_parameters()

    def sample(self, n_samples: int = 1):
        if n_samples < 1:
            raise ValueError(f"Invalid value for 'n_samples': {n_samples} . The sampling requires at least one sample.")
        d = self.means_.shape[1]
        rng = np.random.RandomState(self.random_state)
        counts = rng.multinomial(n_samples, self.weights_)
        X = np.empty((n_samples, d))
        y = np.empty(n_samples, dtype=int)
        row = 0
        for c in range(self.n_components):
            m = int(counts[c])
            part = np.empty((m, d))
            if CT == "full":
                part = rng.multivariate_normal(self.means_[c], self._cov[c], m)
            elif CT == "tied":
                part = rng.multivariate_normal(self.means_[c], self._cov[0], m)
            elif CT == "diag":
                part = self.means_[c] + rng.standard_normal(size=(m, d)) * np.sqrt(self._cov[0, c])
            else:
                part = self.means_[c] + rng.standard_normal(size=(m, d)) * np.sqrt(self._cov[0, 0, c])
            for i in range(m):
                X[row + i] = part[i]
                y[row + i] = c
            row += m
        return X, y

def GaussianMixture(n_components: int = 1, *, covariance_type: Literal[str] = "full",
                    tol: float = 1e-3, reg_covar: float = 1e-6, max_iter: int = 100,
                    n_init: int = 1, init_params: str = "kmeans", weights_init = None,
                    means_init = None, precisions_init = None,
                    random_state: Optional[int] = None, warm_start: bool = False,
                    verbose: int = 0, verbose_interval: int = 10):
    if covariance_type != "full" and covariance_type != "tied" and covariance_type != "diag" and covariance_type != "spherical":
        compile_error("covariance_type must be 'full', 'tied', 'diag' or 'spherical'")
    return _GaussianMixture[covariance_type](n_components, covariance_type, tol, reg_covar, max_iter,
                                             n_init, init_params, weights_init, means_init,
                                             precisions_init, random_state, warm_start)
