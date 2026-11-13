# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# GaussianRandomProjection / SparseRandomProjection (dense components_;
# the sparse matrix is generated with the same RNG stream and densified),
# johnson_lindenstrauss_min_dim.

from .base import estimator_repr as _estimator_repr
import numpy as np
from numpy.random.legacy import RandomState, check_random_state

def johnson_lindenstrauss_min_dim(n_samples, *, eps = 0.1):
    e = np.asarray(eps, dtype=float)
    ns = np.asarray(n_samples, dtype=float)
    if np.any(e <= 0.0) or np.any(e >= 1):
        raise ValueError(f"The JL bound is defined for eps in ]0, 1[, got {eps}")
    if np.any(ns <= 0):
        raise ValueError(f"The JL bound is defined for n_samples greater than zero, got {n_samples}")
    den = e ** 2 / 2 - e ** 3 / 3
    return (4 * np.log(ns) / den).astype(int)

def _jl_scalar(n: int, eps: float) -> int:
    den = eps ** 2 / 2 - eps ** 3 / 3
    return int(4 * np.log(float(n)) / den)

def _check_input_size(n_components: int, n_features: int):
    if n_components <= 0:
        raise ValueError(f"n_components must be strictly positive, got {n_components}")
    if n_features <= 0:
        raise ValueError(f"n_features must be strictly positive, got {n_features}")

def _gaussian_random_matrix(n_components: int, n_features: int, rng):
    _check_input_size(n_components, n_features)
    return rng.normal(loc=0.0, scale=1.0 / np.sqrt(float(n_components)), size=(n_components, n_features))

def _sparse_random_matrix(n_components: int, n_features: int, density: float, rng):
    from .datasets._samples_generator import sample_without_replacement
    _check_input_size(n_components, n_features)
    if density == 1:
        comp = rng.binomial(1, 0.5, (n_components, n_features)) * 2 - 1
        return 1 / np.sqrt(float(n_components)) * comp.astype(float)
    rows = List[np.ndarray[int, 1]]()
    total = 0
    for _ in range(n_components):
        nnz = int(rng.binomial(n_features, density))
        rows.append(sample_without_replacement(n_features, nnz, random_state=rng))
        total += nnz
    data = rng.binomial(1, 0.5, size=total) * 2 - 1
    out = np.zeros((n_components, n_features))
    scale = np.sqrt(1 / density) / np.sqrt(float(n_components))
    t = 0
    for i in range(n_components):
        for c in rows[i]:
            # csr construction keeps duplicates summed; indices are distinct
            out[i, int(c)] += scale * float(data[t])
            t += 1
    return out

class _BaseRP:
    n_components: int
    _auto: bool
    eps: float
    compute_inverse_components: bool
    random_state: Optional[int]
    n_components_: int
    components_: np.ndarray[float, 2]
    inverse_components_: np.ndarray[float, 2]
    n_features_in_: int

    def _init_base(self, n_components, eps: float, compute_inverse_components: bool,
                   random_state: Optional[int]):
        if isinstance(n_components, str):
            self._auto = True
            self.n_components = 0
        else:
            self._auto = False
            self.n_components = int(n_components)
        self.eps = eps
        self.compute_inverse_components = compute_inverse_components
        self.random_state = random_state
        self.n_components_ = 0
        self.components_ = np.empty((0, 0))
        self.inverse_components_ = np.empty((0, 0))
        self.n_features_in_ = 0

    def _fit_dims(self, X) -> int:
        Xa = np.asarray(X, dtype=float)
        n, p = Xa.shape
        self.n_features_in_ = p
        if self._auto:
            self.n_components_ = _jl_scalar(n, self.eps)
            if self.n_components_ <= 0:
                raise ValueError(f"eps={self.eps} and n_samples={n} lead to a target dimension of {self.n_components_} which is invalid")
            elif self.n_components_ > p:
                raise ValueError(f"eps={self.eps} and n_samples={n} lead to a target dimension of {self.n_components_} which is larger than the original space with n_features={p}")
        else:
            self.n_components_ = self.n_components
        return p

    def _after_fit(self):
        if self.compute_inverse_components:
            import scipy.linalg as sl
            self.inverse_components_ = sl.pinv(self.components_, check_finite=False)

    def transform(self, X):
        return np.asarray(X, dtype=float) @ self.components_.T

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        import scipy.linalg as sl
        if self.compute_inverse_components:
            return np.asarray(X, dtype=float) @ self.inverse_components_.T
        return np.asarray(X, dtype=float) @ sl.pinv(self.components_, check_finite=False).T

class GaussianRandomProjection(_BaseRP):
    def __repr__(self) -> str:
        return _estimator_repr(self)

    def __init__(self, n_components = "auto", *, eps: float = 0.1,
                 compute_inverse_components: bool = False, random_state: Optional[int] = None):
        self._init_base(n_components, eps, compute_inverse_components, random_state)

    def fit(self, X, y = None):
        p = self._fit_dims(X)
        rs = check_random_state(self.random_state)
        self.components_ = _gaussian_random_matrix(self.n_components_, p, rs)
        self._after_fit()
        return self

class SparseRandomProjection(_BaseRP):
    def __repr__(self) -> str:
        return _estimator_repr(self)

    density: float
    _density_auto: bool
    density_: float
    dense_output: bool

    def __init__(self, n_components = "auto", *, density = "auto", eps: float = 0.1,
                 dense_output: bool = False, compute_inverse_components: bool = False,
                 random_state: Optional[int] = None):
        self._init_base(n_components, eps, compute_inverse_components, random_state)
        if isinstance(density, str):
            self._density_auto = True
            self.density = 0.0
        else:
            self._density_auto = False
            self.density = float(density)
        self.density_ = 0.0
        self.dense_output = dense_output

    def fit(self, X, y = None):
        p = self._fit_dims(X)
        rs = check_random_state(self.random_state)
        d = 1 / np.sqrt(float(p)) if self._density_auto else self.density
        if d <= 0 or d > 1:
            raise ValueError(f"Expected density in range ]0, 1], got: {d}")
        self.density_ = d
        self.components_ = _sparse_random_matrix(self.n_components_, p, d, rs)
        self._after_fit()
        return self
