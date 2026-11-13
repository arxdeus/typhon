# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Kernel feature-map approximations: RBFSampler, SkewedChi2Sampler,
# AdditiveChi2Sampler, Nystroem, PolynomialCountSketch (dense input).

from .base import estimator_repr as _estimator_repr
import numpy as np
from numpy.random.legacy import RandomState, check_random_state
from ._pairwise_k import kernel_matrix

class RBFSampler:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    gamma: str
    gamma_value: float
    _gamma: float
    n_components: int
    random_state: Optional[int]
    random_weights_: np.ndarray[float, 2]
    random_offset_: np.ndarray[float, 1]
    n_features_in_: int

    def __init__(self, *, gamma = 1.0, n_components: int = 100, random_state: Optional[int] = None):
        if isinstance(gamma, str):
            self.gamma = gamma
            self.gamma_value = 0.0
        else:
            self.gamma = ""
            self.gamma_value = float(gamma)
        self._gamma = 0.0
        self.n_components = n_components
        self.random_state = random_state
        self.random_weights_ = np.empty((0, 0))
        self.random_offset_ = np.empty(0)
        self.n_features_in_ = 0

    def fit(self, X, y = None):
        Xa = np.asarray(X, dtype=float)
        rs = check_random_state(self.random_state)
        p = Xa.shape[1]
        self.n_features_in_ = p
        if self.gamma == "scale":
            v = float(Xa.var())
            self._gamma = 1.0 / (float(p) * v) if v != 0 else 1.0
        else:
            self._gamma = self.gamma_value
        self.random_weights_ = (2.0 * self._gamma) ** 0.5 * rs.normal(size=(p, self.n_components))
        self.random_offset_ = rs.uniform(0, 2 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        P = np.asarray(X, dtype=float) @ self.random_weights_
        P += self.random_offset_
        np.cos(P, out=P)
        P *= (2.0 / float(self.n_components)) ** 0.5
        return P

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

class SkewedChi2Sampler:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    skewedness: float
    n_components: int
    random_state: Optional[int]
    random_weights_: np.ndarray[float, 2]
    random_offset_: np.ndarray[float, 1]

    def __init__(self, *, skewedness: float = 1.0, n_components: int = 100,
                 random_state: Optional[int] = None):
        self.skewedness = skewedness
        self.n_components = n_components
        self.random_state = random_state
        self.random_weights_ = np.empty((0, 0))
        self.random_offset_ = np.empty(0)

    def fit(self, X, y = None):
        Xa = np.asarray(X, dtype=float)
        rs = check_random_state(self.random_state)
        p = Xa.shape[1]
        u = rs.uniform(size=(p, self.n_components))
        self.random_weights_ = 1.0 / np.pi * np.log(np.tan(np.pi / 2.0 * u))
        self.random_offset_ = rs.uniform(0, 2 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        Xa = np.asarray(X, dtype=float).copy()
        if np.any(Xa <= -self.skewedness):
            raise ValueError("X may not contain entries smaller than -skewedness.")
        Xa += self.skewedness
        np.log(Xa, out=Xa)
        P = Xa @ self.random_weights_
        P += self.random_offset_
        np.cos(P, out=P)
        P *= np.sqrt(2.0) / np.sqrt(float(self.n_components))
        return P

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

class AdditiveChi2Sampler:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    sample_steps: int
    sample_interval: Optional[float]

    def __init__(self, *, sample_steps: int = 2, sample_interval: Optional[float] = None):
        self.sample_steps = sample_steps
        self.sample_interval = sample_interval

    def fit(self, X, y = None):
        if np.any(np.asarray(X, dtype=float) < 0):
            raise ValueError("Negative values in data passed to AdditiveChi2Sampler (input X)")
        if self.sample_interval is None and self.sample_steps not in (1, 2, 3):
            raise ValueError("If sample_steps is not in [1, 2, 3], you need to provide sample_interval")
        return self

    def transform(self, X):
        Xa = np.asarray(X, dtype=float)
        if np.any(Xa < 0):
            raise ValueError("Negative values in data passed to AdditiveChi2Sampler (input X)")
        si = 0.0
        if self.sample_interval is None:
            if self.sample_steps == 1:
                si = 0.8
            elif self.sample_steps == 2:
                si = 0.5
            elif self.sample_steps == 3:
                si = 0.4
            else:
                raise ValueError("If sample_steps is not in [1, 2, 3], you need to provide sample_interval")
        else:
            si = self.sample_interval.__val__()
        n, p = Xa.shape
        S = self.sample_steps
        out = np.zeros((n, p * (2 * S - 1)))
        for i in range(n):
            for k in range(p):
                x = Xa[i, k]
                if x == 0.0:
                    continue
                out[i, k] = np.sqrt(x * si)
                log_step = si * np.log(x)
                step = 2 * x * si
                for j in range(1, S):
                    f = np.sqrt(step / np.cosh(np.pi * j * si))
                    out[i, (2 * j - 1) * p + k] = f * np.cos(j * log_step)
                    out[i, 2 * j * p + k] = f * np.sin(j * log_step)
        return out

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

class Nystroem:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    kernel: str
    gamma: Optional[float]
    coef0: Optional[float]
    degree: Optional[float]
    n_components: int
    random_state: Optional[int]
    components_: np.ndarray[float, 2]
    component_indices_: np.ndarray[int, 1]
    normalization_: np.ndarray[float, 2]

    def __init__(self, kernel: str = "rbf", *, gamma: Optional[float] = None,
                 coef0: Optional[float] = None, degree: Optional[float] = None,
                 kernel_params = None, n_components: int = 100,
                 random_state: Optional[int] = None, n_jobs = None):
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.n_components = n_components
        self.random_state = random_state
        self.components_ = np.empty((0, 0))
        self.component_indices_ = np.empty(0, dtype=int)
        self.normalization_ = np.empty((0, 0))

    def _k(self, X, Y, same: bool = False):
        # unset params keep the kernel function defaults
        c0 = 1.0 if self.coef0 is None else self.coef0.__val__()
        dg = 3.0 if self.degree is None else self.degree.__val__()
        if self.kernel == "chi2":
            return kernel_matrix(X, Y, "chi2", 1.0 if self.gamma is None else self.gamma.__val__(), dg, c0, same)
        return kernel_matrix(X, Y, self.kernel, self.gamma, dg, c0, same)

    def fit(self, X, y = None):
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        rs = check_random_state(self.random_state)
        n = Xa.shape[0]
        nc = min(n, self.n_components)
        inds = rs.permutation(n)
        bi = inds[:nc].copy()
        basis = np.ascontiguousarray(Xa[bi])
        BK = self._k(basis, basis, True)
        U, S, V = np.linalg.svd(BK)
        S = np.clip(S, 1e-12, None)
        self.normalization_ = (U / np.sqrt(S)) @ V
        self.components_ = basis
        self.component_indices_ = bi
        return self

    def transform(self, X):
        E = self._k(X, self.components_)
        return E @ self.normalization_.T

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

class PolynomialCountSketch:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    gamma: float
    degree: int
    coef0: float
    n_components: int
    random_state: Optional[int]
    indexHash_: np.ndarray[int, 2]
    bitHash_: np.ndarray[int, 2]

    def __init__(self, *, gamma: float = 1.0, degree: int = 2, coef0: float = 0.0,
                 n_components: int = 100, random_state: Optional[int] = None):
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.n_components = n_components
        self.random_state = random_state
        self.indexHash_ = np.empty((0, 0), dtype=int)
        self.bitHash_ = np.empty((0, 0), dtype=int)

    def fit(self, X, y = None):
        Xa = np.asarray(X, dtype=float)
        rs = check_random_state(self.random_state)
        p = Xa.shape[1]
        if self.coef0 != 0:
            p += 1
        self.indexHash_ = rs.randint(0, high=self.n_components, size=(self.degree, p))
        self.bitHash_ = rs.choice(np.array([-1, 1]), size=(self.degree, p))
        return self

    def transform(self, X):
        from scipy.fft import fft, ifft
        Xa = np.sqrt(self.gamma) * np.asarray(X, dtype=float)
        n = Xa.shape[0]
        if self.coef0 != 0:
            Xa = np.hstack((Xa, np.sqrt(self.coef0) * np.ones((n, 1))))
        p = Xa.shape[1]
        if p != self.indexHash_.shape[1]:
            raise ValueError("Number of features of test samples does not match that of training samples.")
        cs = np.zeros((n, self.degree, self.n_components))
        for j in range(p):
            for d in range(self.degree):
                h = int(self.indexHash_[d, j])
                b = float(self.bitHash_[d, j])
                for i in range(n):
                    cs[i, d, h] += b * Xa[i, j]
        F = fft(cs, axis=2)
        Pr = np.prod(F, axis=1)
        return np.real(ifft(Pr))

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)
