# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Losses used by HistGradientBoosting (sklearn/_loss/loss.py, _loss.pyx.tp,
# link.py). Raw predictions are stored column-major: raw[k * n + i].
# Gradients/hessians are computed in float64 and stored as float32
# (G_H_DTYPE) exactly like the Cython loops. The macOS arm64 wheel contracts
# a*b+c into fused multiply-adds inside single C expressions; those sites use
# _fma so results match bit for bit.

import math
import numpy as np
from ...tree._criterion import _fma
from .._gb import _avg, _weighted_percentile, _logit, _expit, _log1pexp

_EPS: float = 2.220446049250313e-16

def _mean(v: List[float]) -> float:
    # np.mean (pairwise sum / n)
    n = len(v)
    a = np.empty(n)
    for i in range(n):
        a[i] = v[i]
    return float(np.sum(a)) / float(n)

def _average(v: List[float], w: List[float], has_w: bool) -> float:
    # np.average(v, weights=w)
    if has_w:
        return _avg(v, w)
    return _mean(v)

def _median(v: List[float]) -> float:
    return float(np.median(np.array(v)))

def _percentile(v: List[float], q: float) -> float:
    return float(np.percentile(np.array(v), q))

class _HGBLoss:
    kind: str             # squared_error absolute_error quantile poisson gamma binomial multinomial
    quantile: float
    n_classes: int        # number of raw columns K
    constant_hessian: bool
    differentiable: bool

    def __init__(self, kind: str, quantile: float, n_classes: int, has_w: bool):
        self.kind = kind
        self.quantile = quantile
        self.n_classes = n_classes
        self.constant_hessian = (kind == "squared_error" or kind == "absolute_error"
                                 or kind == "quantile") and not has_w
        self.differentiable = not (kind == "absolute_error" or kind == "quantile")

    def fit_intercept_only(self, y: List[float], sw: List[float], has_w: bool) -> List[float]:
        kind = self.kind
        if kind == "absolute_error":
            if has_w:
                return [_weighted_percentile(y, sw, 50.0)]
            return [_median(y)]
        if kind == "quantile":
            if has_w:
                return [_weighted_percentile(y, sw, 100.0 * self.quantile)]
            return [_percentile(y, 100.0 * self.quantile)]
        if kind == "multinomial":
            K = self.n_classes
            p = [0.0] * K
            for k in range(K):
                ind = [1.0 if int(v) == k else 0.0 for v in y]
                pk = _average(ind, sw, has_w)
                p[k] = min(max(pk, _EPS), 1.0 - _EPS)
            # MultinomialLogit.link: log(p / gmean(p)), gmean = exp(mean(log p))
            lg = [math.log(v) for v in p]
            gm = math.exp(_mean(lg))
            return [math.log(p[k] / gm) for k in range(K)]
        yp = _average(y, sw, has_w)
        eps10 = 10.0 * _EPS
        if kind == "binomial":
            yp = min(max(yp, eps10), 1.0 - eps10)
            return [_logit(yp)]
        if kind == "poisson" or kind == "gamma":
            yp = max(yp, eps10)
            return [math.log(yp)]
        return [yp]

    def gradient_hessian(self, y: List[float], raw: List[float], sw: List[float], has_w: bool,
                         g: List[float32], h: List[float32]):
        """Fills g (and h unless the hessian is constant), column-major n*K."""
        n = len(y)
        kind = self.kind
        if kind == "multinomial":
            K = self.n_classes
            p = [0.0] * K
            for i in range(n):
                mx = raw[i]
                for k in range(1, K):
                    if mx < raw[k * n + i]:
                        mx = raw[k * n + i]
                s = 0.0
                for k in range(K):
                    p[k] = math.exp(raw[k * n + i] - mx)
                    s += p[k]
                yi = int(y[i])
                for k in range(K):
                    p[k] /= s
                    gk = p[k] - (1.0 if yi == k else 0.0)
                    hk = p[k] * (1.0 - p[k])
                    if has_w:
                        gk = gk * sw[i]
                        hk = hk * sw[i]
                    g[k * n + i] = float32(gk)
                    h[k * n + i] = float32(hk)
            return
        q = self.quantile
        const_h = self.constant_hessian
        for i in range(n):
            yi = y[i]
            r = raw[i]
            gv = 0.0
            hv = 1.0
            if kind == "squared_error":
                gv = r - yi
            elif kind == "absolute_error":
                gv = 1.0 if r > yi else -1.0
            elif kind == "quantile":
                gv = -q if yi >= r else 1.0 - q
            elif kind == "poisson":
                hv = math.exp(r)
                gv = hv - yi
            elif kind == "gamma":
                e = math.exp(-r)
                gv = _fma(-yi, e, 1.0)
                hv = e * yi
            else:  # binomial
                if r > -37.0:
                    e = math.exp(-r)
                    gv = _fma(-yi, e, 1.0 - yi) / (1.0 + e)
                    hv = e / ((1.0 + e) * (1.0 + e))
                else:
                    e = math.exp(r)
                    gv = e - yi
                    hv = e
            if has_w:
                gv = sw[i] * gv
                hv = sw[i] * hv
            g[i] = float32(gv)
            if not const_h:
                h[i] = float32(hv)

    def pointwise(self, y: List[float], raw: List[float], n: int) -> List[float]:
        out = List[float](n)
        kind = self.kind
        q = self.quantile
        for i in range(n):
            yi = y[i]
            if kind == "multinomial":
                K = self.n_classes
                mx = raw[i]
                for k in range(1, K):
                    if mx < raw[k * n + i]:
                        mx = raw[k * n + i]
                s = 0.0
                for k in range(K):
                    s += math.exp(raw[k * n + i] - mx)
                out.append(math.log(s) + mx - raw[int(yi) * n + i])
                continue
            r = raw[i]
            if kind == "squared_error":
                out.append(0.5 * (r - yi) * (r - yi))
            elif kind == "absolute_error":
                out.append(abs(r - yi))
            elif kind == "quantile":
                out.append(q * (yi - r) if yi >= r else (1.0 - q) * (r - yi))
            elif kind == "poisson":
                out.append(_fma(-yi, r, math.exp(r)))
            elif kind == "gamma":
                out.append(_fma(yi, math.exp(-r), r))
            else:
                out.append(_fma(-yi, r, _log1pexp(r)))
        return out

    def __call__(self, y: List[float], raw: List[float], sw: List[float], has_w: bool) -> float:
        """BaseLoss.__call__: np.average(loss, weights=sample_weight)."""
        return _average(self.pointwise(y, raw, len(y)), sw, has_w)

    def inverse_link(self, r: float) -> float:
        if self.kind == "poisson" or self.kind == "gamma":
            return math.exp(r)
        return r

    def predict_proba(self, raw):
        """raw: (n, K) ndarray -> (n, n_classes) probabilities."""
        n = raw.shape[0]
        if self.kind == "multinomial":
            K = raw.shape[1]
            out = np.empty((n, K))
            for i in range(n):
                mx = raw[i, 0]
                for k in range(1, K):
                    if raw[i, k] > mx:
                        mx = raw[i, k]
                s = 0.0
                for k in range(K):
                    out[i, k] = math.exp(raw[i, k] - mx)
                    s += out[i, k]
                for k in range(K):
                    out[i, k] /= s
            return out
        out = np.empty((n, 2))
        for i in range(n):
            p1 = _expit(raw[i, 0])
            out[i, 1] = p1
            out[i, 0] = 1.0 - p1
        return out
