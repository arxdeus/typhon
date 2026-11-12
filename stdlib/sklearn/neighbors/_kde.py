# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

import numpy as np
import math
from ._dist import _metric_code, _pairwise, _as2d

def _log_vn(n: int) -> float:
    return 0.5 * float(n) * math.log(math.pi) - math.lgamma(0.5 * float(n) + 1.0)

def _log_sn(n: int) -> float:
    return math.log(2.0 * math.pi) + _log_vn(n - 1)

def _kernel_code(kernel: str) -> int:
    if kernel == "gaussian":
        return 0
    if kernel == "tophat":
        return 1
    if kernel == "epanechnikov":
        return 2
    if kernel == "exponential":
        return 3
    if kernel == "linear":
        return 4
    if kernel == "cosine":
        return 5
    raise ValueError(f"invalid kernel: '{kernel}'")

def _log_kernel_norm(h: float, d: int, kc: int) -> float:
    factor = 0.0
    if kc == 0:
        factor = 0.5 * float(d) * math.log(2.0 * math.pi)
    elif kc == 1:
        factor = _log_vn(d)
    elif kc == 2:
        factor = _log_vn(d) + math.log(2.0 / (float(d) + 2.0))
    elif kc == 3:
        factor = _log_sn(d - 1) + math.lgamma(float(d))
    elif kc == 4:
        factor = _log_vn(d) - math.log(float(d) + 1.0)
    else:
        tmp = 2.0 / math.pi
        k = 1
        while k < d + 1:
            factor += tmp
            tmp *= -float((d - k) * (d - k - 1)) * (2.0 / math.pi) ** 2
            k += 2
        # scikit-learn yields NaN when this series is non-positive (even d)
        lf = math.log(factor) if factor > 0.0 else float("nan")
        factor = lf + _log_sn(d - 1)
    return -factor - float(d) * math.log(h)

def _log_kernel(dist: float, h: float, kc: int) -> float:
    NEG_INF = -float("inf")
    if kc == 0:
        return -0.5 * (dist * dist) / (h * h)
    if kc == 3:
        return -dist / h
    if dist >= h:
        return NEG_INF
    if kc == 1:
        return 0.0
    if kc == 2:
        return math.log(1.0 - (dist * dist) / (h * h))
    if kc == 4:
        return math.log(1.0 - dist / h)
    return math.log(math.cos(0.5 * math.pi * dist / h))

def kernel_norm(h: float, d: int, kernel: str, return_log: bool = False) -> float:
    r = _log_kernel_norm(h, d, _kernel_code(kernel))
    return r if return_log else math.exp(r)

class KernelDensity:
    """Exact (atol = rtol = 0 semantics) kernel density estimate.
    bandwidth: float, 'scott' or 'silverman'."""
    _bw_kind: str
    _bw_value: float
    algorithm: str
    kernel: str
    metric: str
    atol: float
    rtol: float
    breadth_first: bool
    leaf_size: int
    bandwidth_: float
    _fit_X: np.ndarray[float, 2]
    _w: np.ndarray[float, 1]
    _has_w: bool
    n_features_in_: int

    def __init__(self, *, bandwidth = 1.0, algorithm: str = "auto",
                 kernel: str = "gaussian", metric: str = "euclidean",
                 atol: float = 0.0, rtol: float = 0.0, breadth_first: bool = True,
                 leaf_size: int = 40, metric_params = None):
        self._bw_value = 0.0
        if isinstance(bandwidth, str):
            if bandwidth != "scott" and bandwidth != "silverman":
                raise ValueError("bandwidth must be a positive float, 'scott' or 'silverman'")
            self._bw_kind = bandwidth
        else:
            self._bw_kind = ""
            self._bw_value = float(bandwidth)
            if self._bw_value <= 0.0:
                raise ValueError("bandwidth must be positive")
        _kernel_code(kernel)
        _metric_code(metric, 2.0)
        self.algorithm = algorithm
        self.kernel = kernel
        self.metric = metric
        self.atol = atol
        self.rtol = rtol
        self.breadth_first = breadth_first
        self.leaf_size = leaf_size
        self.bandwidth_ = 0.0
        self._fit_X = np.empty((0, 0))
        self._w = np.empty(0)
        self._has_w = False
        self.n_features_in_ = 0

    def fit(self, X, y = None, sample_weight = None):
        Xa = _as2d(X)
        n = float(Xa.shape[0])
        d = float(Xa.shape[1])
        if self._bw_kind == "scott":
            self.bandwidth_ = n ** (-1.0 / (d + 4.0))
        elif self._bw_kind == "silverman":
            self.bandwidth_ = (n * (d + 2.0) / 4.0) ** (-1.0 / (d + 4.0))
        else:
            self.bandwidth_ = self._bw_value
        self._fit_X = Xa
        self.n_features_in_ = Xa.shape[1]
        if sample_weight is None:
            self._has_w = False
            self._w = np.ones(Xa.shape[0])
        else:
            self._has_w = True
            self._w = np.ascontiguousarray(np.asarray(sample_weight, dtype=float).ravel())
            if self._w.shape[0] != Xa.shape[0]:
                raise ValueError("sample_weight.shape must match X")
            if np.any(self._w < 0.0):
                raise ValueError("Negative values in data passed to sample_weight")
        return self

    def score_samples(self, X):
        Xq = _as2d(X)
        kc = _kernel_code(self.kernel)
        h = self.bandwidth_
        D = _pairwise(Xq, self._fit_X, _metric_code(self.metric, 2.0), 2.0)
        nq, nf = D.shape
        N = float(nf) if not self._has_w else float(self._w.sum())
        lnorm = _log_kernel_norm(h, self._fit_X.shape[1], kc)
        out = np.empty(nq)
        lk = np.empty(nf)
        for i in range(nq):
            m = -float("inf")
            for j in range(nf):
                v = _log_kernel(D[i, j], h, kc)
                if self._has_w:
                    v += math.log(self._w[j]) if self._w[j] > 0.0 else -float("inf")
                lk[j] = v
                if v > m:
                    m = v
            if m == -float("inf"):
                out[i] = -float("inf")
                continue
            s = 0.0
            for j in range(nf):
                s += math.exp(lk[j] - m)
            out[i] = m + math.log(s) + lnorm - math.log(N)
        return out

    def score(self, X, y = None) -> float:
        return float(np.sum(self.score_samples(X)))
