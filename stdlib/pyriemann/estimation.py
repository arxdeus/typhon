# Ported from pyriemann 0.12 (BSD-3-Clause)

import numpy as np
from .geometry.covariance import (covariances, covariances_EP, block_covariances,
                                  cross_spectrum, coherence)
from .spatialfilters import Xdawn
from sklearn.covariance import shrunk_covariance

def _nextpow2(i: int) -> int:
    n = 1
    while n < i:
        n *= 2
    return n

class Covariances:
    estimator: str

    def __init__(self, estimator: str = "scm"):
        self.estimator = estimator

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        return covariances(X, estimator=self.estimator)

    def fit_transform(self, X, y = None):
        return self.fit(X, y).transform(X)

class ERPCovariances:
    estimator: str
    svd: int
    use_svd: bool
    P_: np.ndarray[float, 2]

    def __init__(self, estimator: str = "scm", svd = None):
        self.estimator = estimator
        self.use_svd = svd is not None
        self.svd = 0 if svd is None else int(svd)
        self.P_ = np.empty((0, 0))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        classes = np.unique(y)
        parts = []
        for c in classes:
            P = np.mean(X[y == c], axis=0)
            if self.use_svd:
                U, s, Vt = np.linalg.svd(P)
                P = U[:, 0:self.svd].T @ P
            parts.append(P)
        self.P_ = np.concatenate(parts, axis=0)
        return self

    def transform(self, X):
        return covariances_EP(X, self.P_, estimator=self.estimator)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

class XdawnCovariances:
    nfilter: int
    applyfilters: bool
    estimator: str
    xdawn_estimator: str
    Xd_: Xdawn
    P_: np.ndarray[float, 2]

    def __init__(self, nfilter: int = 4, applyfilters: bool = True,
                 estimator: str = "scm", xdawn_estimator: str = "scm"):
        self.nfilter = nfilter
        self.applyfilters = applyfilters
        self.estimator = estimator
        self.xdawn_estimator = xdawn_estimator
        self.Xd_ = Xdawn(nfilter, xdawn_estimator)
        self.P_ = np.empty((0, 0))

    def fit(self, X, y):
        self.Xd_ = Xdawn(self.nfilter, self.xdawn_estimator)
        self.Xd_.fit(X, y)
        self.P_ = self.Xd_.evokeds_
        return self

    def transform(self, X):
        Xa = np.asarray(X, dtype=float)
        if self.applyfilters:
            Xa = self.Xd_.transform(Xa)
        return covariances_EP(Xa, self.P_, estimator=self.estimator)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

class BlockCovariances:
    blocks: List[int]
    estimator: str
    _n: int
    _is_int: bool

    def __init__(self, block_size, estimator: str = "scm"):
        self.estimator = estimator
        if isinstance(block_size, int):
            self.blocks = [block_size]
            self._n = block_size
            self._is_int = True
        else:
            self.blocks = [int(b) for b in block_size]
            self._n = 0
            self._is_int = False

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        nch = X.shape[1]
        if self._is_int:
            if nch % self._n != 0:
                raise ValueError(f"Number of channels ({nch}) must be divisible by block size ({self._n})")
            blocks = [self._n for _ in range(nch // self._n)]
        else:
            tot = 0
            for b in self.blocks:
                tot += b
            if tot != nch:
                raise ValueError(f"Sum of individual block sizes must match number of channels ({nch})")
            blocks = self.blocks
        return block_covariances(X, blocks, self.estimator)

    def fit_transform(self, X, y = None):
        return self.transform(X)

class CrossSpectra:
    # transform returns (real, imag) parts, shape (n_matrices, n_ch, n_ch, n_freqs).
    window: int
    overlap: float
    fmin: Optional[float]
    fmax: Optional[float]
    fs: Optional[float]
    freqs_: np.ndarray[float, 1]

    def __init__(self, window: int = 128, overlap: float = 0.75, fmin = None,
                 fmax = None, fs = None):
        self.window = _nextpow2(window)
        self.overlap = overlap
        self.fmin = None if fmin is None else float(fmin)
        self.fmax = None if fmax is None else float(fmax)
        self.fs = None if fs is None else float(fs)
        self.freqs_ = np.empty(0)

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        re, im, f = cross_spectrum(X, self.window, self.overlap, self.fmin, self.fmax, self.fs)
        self.freqs_ = f
        return re, im

    def fit_transform(self, X, y = None):
        return self.transform(X)

class CoSpectra(CrossSpectra):
    def transform(self, X):
        re, im, f = cross_spectrum(X, self.window, self.overlap, self.fmin, self.fmax, self.fs)
        self.freqs_ = f
        return re

    def fit_transform(self, X, y = None):
        return self.transform(X)

class Coherences:
    window: int
    overlap: float
    fmin: Optional[float]
    fmax: Optional[float]
    fs: Optional[float]
    coh: str
    freqs_: np.ndarray[float, 1]

    def __init__(self, window: int = 128, overlap: float = 0.75, fmin = None,
                 fmax = None, fs = None, coh: str = "ordinary"):
        self.window = _nextpow2(window)
        self.overlap = overlap
        self.fmin = None if fmin is None else float(fmin)
        self.fmax = None if fmax is None else float(fmax)
        self.fs = None if fs is None else float(fs)
        self.coh = coh
        self.freqs_ = np.empty(0)

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        C, f = coherence(X, self.window, self.overlap, self.fmin, self.fmax, self.fs, self.coh)
        self.freqs_ = f
        return C

    def fit_transform(self, X, y = None):
        return self.transform(X)

class TimeDelayCovariances:
    delays: List[int]
    estimator: str

    def __init__(self, delays = 4, estimator: str = "scm"):
        if isinstance(delays, int):
            self.delays = [d for d in range(1, delays)]
        else:
            self.delays = [int(d) for d in delays]
        self.estimator = estimator

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        parts = [X]
        for d in self.delays:
            parts.append(np.roll(X, d, axis=-1))
        return covariances(np.concatenate(parts, axis=-2), self.estimator)

    def fit_transform(self, X, y = None):
        return self.transform(X)

class Kernels:
    metric: str

    def __init__(self, metric: str = "linear"):
        self.metric = metric

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        from sklearn.metrics.pairwise import pairwise_kernels
        X = np.asarray(X, dtype=float)
        out = np.empty((X.shape[0], X.shape[1], X.shape[1]))
        for i in range(X.shape[0]):
            out[i] = pairwise_kernels(X[i], None, metric=self.metric)
        return out

    def fit_transform(self, X, y = None):
        return self.transform(X)

class Shrinkage:
    shrinkage: float

    def __init__(self, shrinkage: float = 0.1):
        self.shrinkage = shrinkage

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        out = np.empty(X.shape)
        for i in range(X.shape[0]):
            out[i] = shrunk_covariance(X[i], self.shrinkage)
        return out

    def fit_transform(self, X, y = None):
        return self.transform(X)
