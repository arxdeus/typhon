# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# BernoulliRBM trained with persistent contrastive divergence (dense input),
# same RandomState stream as sklearn.

from ..base import estimator_repr as _estimator_repr
import numpy as np
from numpy.random.legacy import RandomState, check_random_state

def _expit(A):
    from scipy.special import expit
    return expit(A)

class BernoulliRBM:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    n_components: int
    learning_rate: float
    batch_size: int
    n_iter: int
    random_state: Optional[int]
    components_: np.ndarray[float, 2]
    intercept_hidden_: np.ndarray[float, 1]
    intercept_visible_: np.ndarray[float, 1]
    h_samples_: np.ndarray[float, 2]
    _rng: RandomState
    _has_rng: bool

    def __init__(self, n_components: int = 256, *, learning_rate: float = 0.1, batch_size: int = 10,
                 n_iter: int = 10, verbose: int = 0, random_state: Optional[int] = None):
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.random_state = random_state
        self.components_ = np.empty((0, 0))
        self.intercept_hidden_ = np.empty(0)
        self.intercept_visible_ = np.empty(0)
        self.h_samples_ = np.empty((0, 0))
        self._rng = RandomState(0)
        self._has_rng = False

    def _mean_hiddens(self, v):
        p = np.asarray(v, dtype=float) @ self.components_.T
        p += self.intercept_hidden_
        return _expit(p)

    def _sample_visibles(self, h, rng):
        p = np.dot(h, self.components_)
        p += self.intercept_visible_
        p = _expit(p)
        return (rng.uniform(size=p.shape) < p).astype(float)

    def _sample_hiddens(self, v, rng):
        p = self._mean_hiddens(v)
        return (rng.uniform(size=p.shape) < p).astype(float)

    def transform(self, X):
        return self._mean_hiddens(X)

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

    def _free_energy(self, v):
        a = v @ self.components_.T + self.intercept_hidden_
        return -(v @ self.intercept_visible_) - np.logaddexp(0, a).sum(axis=1)

    def _fit(self, v_pos, rng):
        h_pos = self._mean_hiddens(v_pos)
        v_neg = self._sample_visibles(self.h_samples_, rng)
        h_neg = self._mean_hiddens(v_neg)
        lr = float(self.learning_rate) / v_pos.shape[0]
        update = np.ascontiguousarray((v_pos.T @ h_pos).T)
        update -= np.dot(h_neg.T, v_neg)
        self.components_ += lr * update
        self.intercept_hidden_ += lr * (h_pos.sum(axis=0) - h_neg.sum(axis=0))
        self.intercept_visible_ += lr * (v_pos.sum(axis=0) - v_neg.sum(axis=0))
        u = rng.uniform(size=h_neg.shape)
        for i in range(h_neg.shape[0]):
            for j in range(h_neg.shape[1]):
                if u[i, j] < h_neg[i, j]:
                    h_neg[i, j] = 1.0
        self.h_samples_ = np.floor(h_neg)

    def fit(self, X, y = None):
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        n = Xa.shape[0]
        rng = check_random_state(self.random_state)
        self.components_ = rng.normal(0, 0.01, (self.n_components, Xa.shape[1]))
        self.intercept_hidden_ = np.zeros(self.n_components)
        self.intercept_visible_ = np.zeros(Xa.shape[1])
        self.h_samples_ = np.zeros((self.batch_size, self.n_components))
        nb = int(np.ceil(float(n) / self.batch_size))
        # gen_even_slices(nb * batch_size, nb, n_samples=n)
        slices = List[Tuple[int, int]]()
        tot = nb * self.batch_size
        start = 0
        for k in range(nb):
            size = tot // nb
            if k < tot % nb:
                size += 1
            if size > 0:
                end = start + size
                if end > n:
                    end = n
                slices.append((start, end))
                start = end
        for _ in range(self.n_iter):
            for a, b in slices:
                self._fit(np.ascontiguousarray(Xa[a:b]), rng)
        return self

    def partial_fit(self, X, y = None):
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        if not self._has_rng:
            self._rng = check_random_state(self.random_state)
            self._has_rng = True
        if self.components_.shape[0] == 0:
            self.components_ = self._rng.normal(0, 0.01, (self.n_components, Xa.shape[1]))
            self.intercept_hidden_ = np.zeros(self.n_components)
            self.intercept_visible_ = np.zeros(Xa.shape[1])
            self.h_samples_ = np.zeros((self.batch_size, self.n_components))
        self._fit(Xa, self._rng)
        return self

    def gibbs(self, v):
        if not self._has_rng:
            self._rng = check_random_state(self.random_state)
            self._has_rng = True
        h = self._sample_hiddens(v, self._rng)
        return self._sample_visibles(h, self._rng) > 0.5

    def score_samples(self, X):
        v = np.ascontiguousarray(np.asarray(X, dtype=float))
        rng = check_random_state(self.random_state)
        cols = rng.randint(0, v.shape[1], v.shape[0])
        v_ = v.copy()
        for i in range(v.shape[0]):
            v_[i, cols[i]] = 1 - v_[i, cols[i]]
        fe = self._free_energy(v)
        fe_ = self._free_energy(v_)
        return -v.shape[1] * np.logaddexp(0, -(fe_ - fe))
