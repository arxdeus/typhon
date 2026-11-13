# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# sklearn.decomposition.LatentDirichletAllocation (batch / online variational
# Bayes, dense float64 input, n_jobs ignored) with the Cython kernels of
# _online_lda_fast.pyx (Bernardo's AS 103 psi approximation).

from ..base import estimator_repr as _estimator_repr
import math
import numpy as np
from numpy.random import RandomState
from scipy.special import gammaln
from ._extmath import check_random_state

_EULER = 0.5772156649015329

def _psi(x0: float) -> float:
    x = x0
    if x <= 1e-6:
        return -_EULER - 1.0 / x
    result = 0.0
    while x < 6.0:
        result -= 1.0 / x
        x += 1.0
    r = 1.0 / x
    result += math.log(x) - 0.5 * r
    r = r * r
    result -= r * ((1.0 / 12.0) - r * ((1.0 / 120.0) - r * (1.0 / 252.0)))
    return result

def _dirichlet_expectation_2d(arr: np.ndarray[float, 2]) -> np.ndarray[float, 2]:
    n, m = arr.shape
    out = np.empty((n, m))
    for i in range(n):
        tot = 0.0
        for j in range(m):
            tot += arr[i, j]
        pt = _psi(tot)
        for j in range(m):
            out[i, j] = _psi(arr[i, j]) - pt
    return out

def _update_doc_distribution(X: np.ndarray[float, 2], exp_topic_word: np.ndarray[float, 2],
                             doc_topic_prior: float, max_doc_update_iter: int,
                             mean_change_tol: float, cal_sstats: bool, rs: RandomState,
                             random_init: bool):
    n_samples = X.shape[0]
    n_topics = exp_topic_word.shape[0]
    if random_init:
        doc_topic_distr = rs.gamma(100.0, 0.01, (n_samples, n_topics))
    else:
        doc_topic_distr = np.ones((n_samples, n_topics))
    exp_doc_topic = np.exp(_dirichlet_expectation_2d(doc_topic_distr))
    suff = np.zeros(exp_topic_word.shape)
    eps = 2.220446049250313e-16
    for d in range(n_samples):
        idx = List[int]()
        for w in range(X.shape[1]):
            if X[d, w] != 0.0:
                idx.append(w)
        ni = len(idx)
        cnts = np.empty(ni)
        etw = np.empty((n_topics, ni))
        for t in range(ni):
            cnts[t] = X[d, idx[t]]
            for k in range(n_topics):
                etw[k, t] = exp_topic_word[k, idx[t]]
        dt = np.ascontiguousarray(doc_topic_distr[d]).copy()
        edt = np.ascontiguousarray(exp_doc_topic[d]).copy()
        for _ in range(max_doc_update_iter):
            last = dt
            norm_phi = edt @ etw + eps
            dt = edt * (np.ascontiguousarray(cnts / norm_phi) @ np.ascontiguousarray(etw.T))
            total = 0.0
            for k in range(n_topics):
                dt[k] = dt[k] + doc_topic_prior
                total += dt[k]
            psi_total = _psi(total)
            for k in range(n_topics):
                edt[k] = math.exp(_psi(dt[k]) - psi_total)
            ch = 0.0
            for k in range(n_topics):
                ch += abs(last[k] - dt[k])
            if ch / float(n_topics) < mean_change_tol:
                break
        for k in range(n_topics):
            doc_topic_distr[d, k] = dt[k]
        if cal_sstats:
            norm_phi = edt @ etw + eps
            r = cnts / norm_phi
            for k in range(n_topics):
                for t in range(ni):
                    suff[k, idx[t]] += edt[k] * r[t]
    return doc_topic_distr, suff

def _loglikelihood(prior: float, distr: np.ndarray[float, 2], dirichlet_distr: np.ndarray[float, 2],
                   size: int) -> float:
    score = float(np.sum((prior - distr) * dirichlet_distr))
    score += float(np.sum(gammaln(distr) - gammaln(prior)))
    score += float(np.sum(gammaln(prior * float(size)) - gammaln(np.sum(distr, axis=1))))
    return score

class LatentDirichletAllocation:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    n_components: int
    doc_topic_prior: Optional[float]
    topic_word_prior: Optional[float]
    learning_method: str
    learning_decay: float
    learning_offset: float
    max_iter: int
    batch_size: int
    evaluate_every: int
    total_samples: float
    perp_tol: float
    mean_change_tol: float
    max_doc_update_iter: int
    random_state: Optional[int]
    components_: np.ndarray[float, 2]
    exp_dirichlet_component_: np.ndarray[float, 2]
    doc_topic_prior_: float
    topic_word_prior_: float
    n_batch_iter_: int
    n_iter_: int
    bound_: float
    random_state_: RandomState
    n_features_in_: int
    _fitted: bool

    def __init__(self, n_components: int = 10, *, doc_topic_prior: Optional[float] = None,
                 topic_word_prior: Optional[float] = None, learning_method: str = "batch",
                 learning_decay: float = 0.7, learning_offset: float = 10.0, max_iter: int = 10,
                 batch_size: int = 128, evaluate_every: int = -1, total_samples: float = 1e6,
                 perp_tol: float = 1e-1, mean_change_tol: float = 1e-3,
                 max_doc_update_iter: int = 100, n_jobs = None, verbose: int = 0,
                 random_state: Optional[int] = None):
        if learning_method != "batch" and learning_method != "online":
            raise ValueError(f"The 'learning_method' parameter of LatentDirichletAllocation must be a str among batch, online. Got {learning_method} instead.")
        self.n_components = n_components
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.learning_method = learning_method
        self.learning_decay = learning_decay
        self.learning_offset = learning_offset
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.evaluate_every = evaluate_every
        self.total_samples = total_samples
        self.perp_tol = perp_tol
        self.mean_change_tol = mean_change_tol
        self.max_doc_update_iter = max_doc_update_iter
        self.random_state = random_state
        self.components_ = np.empty((0, 0))
        self.exp_dirichlet_component_ = np.empty((0, 0))
        self.doc_topic_prior_ = 0.0
        self.topic_word_prior_ = 0.0
        self.n_batch_iter_ = 0
        self.n_iter_ = 0
        self.bound_ = 0.0
        self.random_state_ = RandomState(0)
        self.n_features_in_ = 0
        self._fitted = False

    def _init_latent_vars(self, n_features: int):
        self.random_state_ = check_random_state(self.random_state)
        self.n_batch_iter_ = 1
        self.n_iter_ = 0
        k = float(self.n_components)
        self.doc_topic_prior_ = 1.0 / k if self.doc_topic_prior is None else self.doc_topic_prior.__val__()
        self.topic_word_prior_ = 1.0 / k if self.topic_word_prior is None else self.topic_word_prior.__val__()
        self.components_ = self.random_state_.gamma(100.0, 1.0 / 100.0, (self.n_components, n_features))
        self.exp_dirichlet_component_ = np.exp(_dirichlet_expectation_2d(self.components_))

    def _e_step(self, X: np.ndarray[float, 2], cal_sstats: bool, random_init: bool):
        return _update_doc_distribution(X, self.exp_dirichlet_component_, self.doc_topic_prior_,
                                        self.max_doc_update_iter, self.mean_change_tol, cal_sstats,
                                        self.random_state_, random_init)

    def _em_step(self, X: np.ndarray[float, 2], total_samples: float, batch_update: bool):
        _, suff = self._e_step(X, True, True)
        suff = suff * self.exp_dirichlet_component_
        if batch_update:
            self.components_ = self.topic_word_prior_ + suff
        else:
            weight = (self.learning_offset + float(self.n_batch_iter_)) ** (-self.learning_decay)
            doc_ratio = total_samples / float(X.shape[0])
            self.components_ = self.components_ * (1.0 - weight)
            self.components_ += weight * (self.topic_word_prior_ + doc_ratio * suff)
        self.exp_dirichlet_component_ = np.exp(_dirichlet_expectation_2d(self.components_))
        self.n_batch_iter_ += 1

    def _check_nonneg(self, X, reset: bool, whom: str) -> np.ndarray[float, 2]:
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        if Xa.ndim != 2:
            raise ValueError(f"Expected 2D array, got {Xa.ndim}D array instead")
        if not reset and Xa.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xa.shape[1]} features, but LatentDirichletAllocation is expecting {self.n_features_in_} features as input")
        if reset:
            self.n_features_in_ = Xa.shape[1]
        if Xa.size > 0 and float(Xa.min()) < 0.0:
            raise ValueError(f"Negative values in data passed to {whom}.")
        return Xa

    def partial_fit(self, X, y = None):
        first = not self._fitted
        Xa = self._check_nonneg(X, first, "LatentDirichletAllocation.partial_fit")
        n_samples, n_features = Xa.shape
        if first:
            self._init_latent_vars(n_features)
            self._fitted = True
        if n_features != self.components_.shape[1]:
            raise ValueError(f"The provided data has {n_features} dimensions while the model was trained with feature size {self.components_.shape[1]}.")
        lo = 0
        while lo < n_samples:
            hi = min(n_samples, lo + self.batch_size)
            self._em_step(np.ascontiguousarray(Xa[lo:hi]), self.total_samples, False)
            lo = hi
        return self

    def fit(self, X, y = None):
        Xa = self._check_nonneg(X, True, "LatentDirichletAllocation.fit")
        n_samples, n_features = Xa.shape
        self._init_latent_vars(n_features)
        self._fitted = True
        last_bound = 0.0
        has_last = False
        for i in range(self.max_iter):
            if self.learning_method == "online":
                lo = 0
                while lo < n_samples:
                    hi = min(n_samples, lo + self.batch_size)
                    self._em_step(np.ascontiguousarray(Xa[lo:hi]), float(n_samples), False)
                    lo = hi
            else:
                self._em_step(Xa, float(n_samples), True)
            if self.evaluate_every > 0 and (i + 1) % self.evaluate_every == 0:
                dtd, _ = self._e_step(Xa, False, False)
                bound = self._perplexity_precomp(Xa, dtd, False)
                if has_last and last_bound != 0.0 and abs(last_bound - bound) < self.perp_tol:
                    break
                last_bound = bound
                has_last = True
            self.n_iter_ += 1
        dtd, _ = self._e_step(Xa, False, False)
        self.bound_ = self._perplexity_precomp(Xa, dtd, False)
        return self

    def _unnormalized_transform(self, X: np.ndarray[float, 2]) -> np.ndarray[float, 2]:
        dtd, _ = self._e_step(X, False, False)
        return dtd

    def transform(self, X, *, normalize: bool = True):
        Xa = self._check_nonneg(X, False, "LatentDirichletAllocation.transform")
        d = self._unnormalized_transform(Xa)
        if normalize:
            d = d / np.sum(d, axis=1).reshape((-1, 1))
        return d

    def fit_transform(self, X, y = None, *, normalize: bool = True):
        return self.fit(X, y).transform(X, normalize=normalize)

    def _approx_bound(self, X: np.ndarray[float, 2], doc_topic_distr: np.ndarray[float, 2],
                      sub_sampling: bool) -> float:
        n_samples = doc_topic_distr.shape[0]
        n_features = self.components_.shape[1]
        score = 0.0
        ddt = _dirichlet_expectation_2d(doc_topic_distr)
        dc = _dirichlet_expectation_2d(self.components_)
        K = self.n_components
        for d in range(n_samples):
            idx = List[int]()
            for w in range(X.shape[1]):
                if X[d, w] != 0.0:
                    idx.append(w)
            ni = len(idx)
            for t in range(ni):
                mx = -float("inf")
                for k in range(K):
                    v = ddt[d, k] + dc[k, idx[t]]
                    if v > mx:
                        mx = v
                s = 0.0
                for k in range(K):
                    s += math.exp(ddt[d, k] + dc[k, idx[t]] - mx)
                score += X[d, idx[t]] * (mx + math.log(s))
        score += _loglikelihood(self.doc_topic_prior_, doc_topic_distr, ddt, K)
        if sub_sampling:
            score *= self.total_samples / float(n_samples)
        score += _loglikelihood(self.topic_word_prior_, self.components_, dc, n_features)
        return score

    def score(self, X, y = None) -> float:
        Xa = self._check_nonneg(X, False, "LatentDirichletAllocation.score")
        d = self._unnormalized_transform(Xa)
        return self._approx_bound(Xa, d, False)

    def _perplexity_precomp(self, X: np.ndarray[float, 2], dtd: np.ndarray[float, 2],
                            sub_sampling: bool) -> float:
        bound = self._approx_bound(X, dtd, sub_sampling)
        word_cnt = float(np.sum(X))
        if sub_sampling:
            word_cnt *= self.total_samples / float(X.shape[0])
        return math.exp(-1.0 * (bound / word_cnt))

    def perplexity(self, X, sub_sampling: bool = False) -> float:
        Xa = self._check_nonneg(X, True, "LatentDirichletAllocation.perplexity")
        d = self._unnormalized_transform(Xa)
        return self._perplexity_precomp(Xa, d, sub_sampling)
