# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# QuantileTransformer / quantile_transform and KBinsDiscretizer.

import numpy as np
from ._data import _as2df

_BOUNDS_THRESHOLD = 1e-7

def _subsample_rows(n: int, size: int, random_state: Optional[int]):
    # utils.resample(replace=False): RandomState.shuffle(arange(n))[:size]
    from ..utils import _LegacyRS
    if random_state is None:
        raise ValueError("subsampling needs an integer random_state in Typhon")
    rs = _LegacyRS(random_state)
    idx = np.arange(n)
    rs.shuffle(idx)
    return idx[:size].copy()

class QuantileTransformer:
    n_quantiles: int
    output_distribution: str
    ignore_implicit_zeros: bool
    subsample: Optional[int]
    random_state: Optional[int]
    copy: bool
    n_quantiles_: int
    quantiles_: np.ndarray[float, 2]
    references_: np.ndarray[float, 1]
    n_features_in_: int

    def __init__(self, *, n_quantiles: int = 1000, output_distribution: str = "uniform",
                 ignore_implicit_zeros: bool = False, subsample: Optional[int] = 10000,
                 random_state: Optional[int] = None, copy: bool = True):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.ignore_implicit_zeros = ignore_implicit_zeros
        self.subsample = subsample
        self.random_state = random_state
        self.copy = copy
        self.n_quantiles_ = 0
        self.quantiles_ = np.empty((0, 0))
        self.references_ = np.empty(0)
        self.n_features_in_ = 0

    def fit(self, X, y = None):
        if self.subsample is not None and self.n_quantiles > self.subsample:
            raise ValueError(f"The number of quantiles cannot be greater than the number of samples used. Got {self.n_quantiles} quantiles and {self.subsample} samples.")
        Xa = _as2df(X)
        n = Xa.shape[0]
        self.n_features_in_ = Xa.shape[1]
        self.n_quantiles_ = max(1, min(self.n_quantiles, n))
        self.references_ = np.linspace(0.0, 1.0, self.n_quantiles_)
        refs = self.references_ * 100.0
        if self.subsample is not None and self.subsample < n:
            Xa = Xa[_subsample_rows(n, self.subsample, self.random_state)]
        self.quantiles_ = np.nanpercentile(Xa, refs, axis=0)
        # make quantiles monotonic (np.maximum.accumulate)
        for j in range(self.quantiles_.shape[1]):
            for i in range(1, self.quantiles_.shape[0]):
                if self.quantiles_[i, j] < self.quantiles_[i - 1, j]:
                    self.quantiles_[i, j] = self.quantiles_[i - 1, j]
        return self

    def _transform_col(self, col, q, inverse: bool):
        from scipy.special import ndtr, ndtri
        normal = self.output_distribution == "normal"
        x = col.copy()
        lbx = q[0]; ubx = q[q.shape[0] - 1]; lby = 0.0; uby = 1.0
        if inverse:
            lbx = 0.0; ubx = 1.0; lby = q[0]; uby = q[q.shape[0] - 1]
            if normal:
                x = ndtr(x)
        n = x.shape[0]
        lo = np.zeros(n, dtype=bool)
        hi = np.zeros(n, dtype=bool)
        for i in range(n):
            if normal:
                lo[i] = x[i] - _BOUNDS_THRESHOLD < lbx
                hi[i] = x[i] + _BOUNDS_THRESHOLD > ubx
            else:
                lo[i] = x[i] == lbx
                hi[i] = x[i] == ubx
        fin = ~np.isnan(x)
        xf = x[fin]
        refs = self.references_
        if not inverse:
            a = np.interp(xf, q, refs)
            b = np.interp(-xf, -q[::-1].copy(), -refs[::-1].copy())
            x[fin] = 0.5 * (a - b)
        else:
            x[fin] = np.interp(xf, refs, q)
        x[hi] = uby
        x[lo] = lby
        if not inverse and normal:
            x = ndtri(x)
            cmin = ndtri(_BOUNDS_THRESHOLD - 2.220446049250313e-16)
            cmax = ndtri(1.0 - (_BOUNDS_THRESHOLD - 2.220446049250313e-16))
            x = np.clip(x, cmin, cmax)
        return x

    def _transform(self, X, inverse: bool):
        Xa = _as2df(X).copy()
        if Xa.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xa.shape[1]} features, but QuantileTransformer is expecting {self.n_features_in_} features as input.")
        for j in range(Xa.shape[1]):
            Xa[:, j] = self._transform_col(np.ascontiguousarray(Xa[:, j]), np.ascontiguousarray(self.quantiles_[:, j]), inverse)
        return Xa

    def transform(self, X):
        return self._transform(X, False)

    def inverse_transform(self, X):
        return self._transform(X, True)

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

def quantile_transform(X, *, axis: int = 0, n_quantiles: int = 1000, output_distribution: str = "uniform",
                       ignore_implicit_zeros: bool = False, subsample: Optional[int] = 100000,
                       random_state: Optional[int] = None, copy: bool = True):
    qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution,
                             subsample=subsample, random_state=random_state)
    Xa = _as2df(X)
    if axis == 0:
        return qt.fit_transform(Xa)
    return np.ascontiguousarray(qt.fit_transform(np.ascontiguousarray(Xa.T)).T)

class KBinsDiscretizer:
    """encode 'onehot' returns a dense array (sklearn: sparse CSR)."""
    n_bins: List[int]
    _n_bins_scalar: int
    encode: str
    strategy: str
    quantile_method: str
    subsample: Optional[int]
    random_state: Optional[int]
    bin_edges_: List[np.ndarray[float, 1]]
    n_bins_: np.ndarray[int, 1]
    n_features_in_: int

    def __init__(self, n_bins = 5, *, encode: str = "onehot", strategy: str = "quantile",
                 quantile_method: str = "averaged_inverted_cdf", dtype = None,
                 subsample: Optional[int] = 200000, random_state: Optional[int] = None):
        self.n_bins = []
        self._n_bins_scalar = 0
        if isinstance(n_bins, int):
            self._n_bins_scalar = n_bins
        else:
            self.n_bins = [int(v) for v in n_bins]
        self.encode = encode
        self.strategy = strategy
        self.quantile_method = quantile_method
        self.subsample = subsample
        self.random_state = random_state
        self.bin_edges_ = []
        self.n_bins_ = np.empty(0, dtype=int)
        self.n_features_in_ = 0

    def fit(self, X, y = None, sample_weight = None):
        Xa = _as2df(X)
        n, p = Xa.shape
        if self.subsample is not None and n > self.subsample:
            raise NotImplementedError("KBinsDiscretizer subsampling (n_samples > subsample) is not supported in Typhon")
        nb = np.empty(p, dtype=int)
        for j in range(p):
            nb[j] = self._n_bins_scalar if len(self.n_bins) == 0 else self.n_bins[j]
            if nb[j] < 2:
                raise ValueError(f"KBinsDiscretizer received an invalid number of bins at indices [{j}]. Number of bins must be at least 2, and must be an int.")
        self.bin_edges_ = []
        for jj in range(p):
            col = np.ascontiguousarray(Xa[:, jj])
            cmin = float(np.min(col)); cmax = float(np.max(col))
            if cmin == cmax:
                nb[jj] = 1
                self.bin_edges_.append(np.array([-np.inf, np.inf]))
                continue
            e = np.empty(0)
            if self.strategy == "uniform":
                e = np.linspace(cmin, cmax, nb[jj] + 1)
            elif self.strategy == "quantile":
                lv = np.linspace(0.0, 100.0, nb[jj] + 1)
                e = np.asarray(np.percentile(col, lv, method=self.quantile_method), dtype=float)
            elif self.strategy == "kmeans":
                from ..cluster import KMeans
                ue = np.linspace(cmin, cmax, nb[jj] + 1)
                init = ((ue[1:] + ue[:-1]) * 0.5).reshape(nb[jj], 1)
                km = KMeans(n_clusters=nb[jj], init=init, n_init=1)
                centers = np.sort(np.ascontiguousarray(km.fit(col.reshape(n, 1), sample_weight=sample_weight).cluster_centers_[:, 0]))
                mid = (centers[1:] + centers[:-1]) * 0.5
                e = np.concatenate((np.array([cmin]), mid, np.array([cmax])))
            else:
                raise ValueError(f"Invalid strategy {self.strategy}")
            if self.strategy != "uniform":
                keep = List[float]()
                for k in range(e.shape[0]):
                    if k == 0 or e[k] - e[k - 1] > 1e-8:
                        keep.append(e[k])
                e = np.array(keep)
                nb[jj] = e.shape[0] - 1
            self.bin_edges_.append(e)
        self.n_bins_ = nb
        self.n_features_in_ = p
        return self

    def _ordinal(self, X):
        Xa = _as2df(X).copy()
        for jj in range(Xa.shape[1]):
            e = self.bin_edges_[jj]
            inner = e[1:e.shape[0] - 1].copy()
            col = np.ascontiguousarray(Xa[:, jj])
            Xa[:, jj] = np.searchsorted(inner, col, side="right").astype(float)
        return Xa

    def transform(self, X):
        Xt = self._ordinal(X)
        if self.encode == "ordinal":
            return Xt
        n, p = Xt.shape
        tot = int(np.sum(self.n_bins_))
        out = np.zeros((n, tot))
        off = 0
        for j in range(p):
            for i in range(n):
                out[i, off + int(Xt[i, j])] = 1.0
            off += self.n_bins_[j]
        return out

    def fit_transform(self, X, y = None, sample_weight = None):
        return self.fit(X, y, sample_weight).transform(X)

    def inverse_transform(self, X):
        Xa = np.asarray(X, dtype=float)
        p = self.n_bins_.shape[0]
        n = Xa.shape[0]
        Xo = np.empty((n, p))
        if self.encode != "ordinal":
            off = 0
            for j in range(p):
                for i in range(n):
                    best = 0
                    for k in range(self.n_bins_[j]):
                        if Xa[i, off + k] > Xa[i, off + best]:
                            best = k
                    Xo[i, j] = float(best)
                off += self.n_bins_[j]
        else:
            if Xa.shape[1] != p:
                raise ValueError(f"Incorrect number of features. Expecting {p}, received {Xa.shape[1]}.")
            Xo = Xa.copy()
        for j in range(p):
            e = self.bin_edges_[j]
            for i in range(n):
                b = int(Xo[i, j])
                Xo[i, j] = (e[b + 1] + e[b]) * 0.5
        return Xo
