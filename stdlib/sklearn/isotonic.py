# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# IsotonicRegression / isotonic_regression / check_increasing. The PAVA is
# scipy.optimize.isotonic_regression (Busing 2022, as scipy 1.18 pava_pybind)
# which sklearn calls on scipy >= 1.12; prediction is interp1d 'linear'.

from .base import estimator_repr as _estimator_repr
import math
import numpy as np

def _sw_or_ones(sample_weight, n: int) -> np.ndarray[float, 1]:
    if isinstance(sample_weight, NoneType):
        return np.ones(n)
    else:
        return np.asarray(sample_weight, dtype=float) * np.ones(n)

def _pava(x: np.ndarray[float, 1], w: np.ndarray[float, 1]):
    n = x.shape[0]
    if n == 0:
        return
    r = [-1] * (n + 1)
    r[0] = 0
    if n > 1:
        r[1] = 1
    b = 0
    xb_prev = x[b]
    wb_prev = w[b]
    i = 1
    while i < n:
        b += 1
        xb = x[i]
        wb = w[i]
        sb = 0.0
        if xb_prev >= xb:
            b -= 1
            sb = wb_prev * xb_prev + wb * xb
            wb += wb_prev
            xb = sb / wb
            while i < n - 1 and xb >= x[i + 1]:
                i += 1
                sb += w[i] * x[i]
                wb += w[i]
                xb = sb / wb
            while b > 0 and x[b - 1] >= xb:
                b -= 1
                sb += w[b] * x[b]
                wb += w[b]
                xb = sb / wb
        x[b] = xb
        xb_prev = xb
        w[b] = wb
        wb_prev = wb
        r[b + 1] = i + 1
        i += 1
    f = n - 1
    k = b
    while k >= 0:
        t = r[k]
        xk = x[k]
        j = f
        while j >= t:
            x[j] = xk
            j -= 1
        f = t - 1
        k -= 1

def _opt_f(v, default: float) -> float:
    if isinstance(v, NoneType):
        return default
    elif isinstance(v, Optional):
        return default if v is None else float(v.__val__())
    else:
        return float(v)

def isotonic_regression(y, *, sample_weight = None, y_min = None, y_max = None,
                        increasing: bool = True):
    ya = np.asarray(y, dtype=float)
    n = ya.shape[0]
    x = np.empty(n)
    w = np.empty(n)
    sw = _sw_or_ones(sample_weight, n)
    for i in range(n):
        k = i if increasing else n - 1 - i
        x[i] = ya[k]
        w[i] = sw[k]
    for i in range(n):
        if not w[i] > 0:
            raise ValueError("Weights w must be strictly positive.")
    _pava(x, w)
    out = np.empty(n)
    for i in range(n):
        out[i] = x[i] if increasing else x[n - 1 - i]
    lo = _opt_f(y_min, -np.inf)
    hi = _opt_f(y_max, np.inf)
    if lo != -np.inf or hi != np.inf:
        for i in range(n):
            out[i] = min(max(out[i], lo), hi)
    return out

def check_increasing(x, y) -> bool:
    from scipy.stats import spearmanr
    res = spearmanr(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    rho = float(res.statistic)
    return rho >= 0

class IsotonicRegression:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    y_min: Optional[float]
    y_max: Optional[float]
    increasing: str
    out_of_bounds: str
    increasing_: bool
    X_min_: float
    X_max_: float
    X_thresholds_: np.ndarray[float, 1]
    y_thresholds_: np.ndarray[float, 1]
    n_features_in_: int

    def __init__(self, *, y_min: Optional[float] = None, y_max: Optional[float] = None,
                 increasing = True, out_of_bounds: str = "nan"):
        self.y_min = y_min
        self.y_max = y_max
        if isinstance(increasing, str):
            if increasing != "auto":
                raise ValueError("increasing must be a bool or 'auto'")
            self.increasing = "auto"
        else:
            self.increasing = "true" if increasing else "false"
        if out_of_bounds != "nan" and out_of_bounds != "clip" and out_of_bounds != "raise":
            raise ValueError("The 'out_of_bounds' parameter of IsotonicRegression must be a str among {'nan', 'clip', 'raise'}.")
        self.out_of_bounds = out_of_bounds
        self.increasing_ = True
        self.X_min_ = 0.0
        self.X_max_ = 0.0
        self.X_thresholds_ = np.empty(0)
        self.y_thresholds_ = np.empty(0)
        self.n_features_in_ = 1

    def _flat(self, X) -> np.ndarray[float, 1]:
        Xa = np.asarray(X, dtype=float)
        if static.len(Xa.shape) == 2:
            if Xa.shape[1] != 1:
                raise ValueError("Isotonic regression input X should be a 1d array or 2d array with 1 feature")
            return np.ascontiguousarray(Xa[:, 0])
        else:
            return np.ascontiguousarray(Xa)

    def fit(self, X, y, sample_weight = None):
        Xf = self._flat(X)
        ya = np.asarray(y, dtype=float)
        n = Xf.shape[0]
        if ya.shape[0] != n:
            raise ValueError("Found input variables with inconsistent numbers of samples")
        if self.increasing == "auto":
            self.increasing_ = check_increasing(Xf, ya)
        else:
            self.increasing_ = self.increasing == "true"
        sw = _sw_or_ones(sample_weight, n)
        keep = List[int]()
        for i in range(n):
            if sw[i] > 0:
                keep.append(i)
        m = len(keep)
        Xk = np.empty(m); yk = np.empty(m); wk = np.empty(m)
        for t in range(m):
            Xk[t] = Xf[keep[t]]; yk[t] = ya[keep[t]]; wk[t] = sw[keep[t]]
        order = np.lexsort((yk, Xk))
        Xs = Xk[order]; ys = yk[order]; ws = wk[order]
        # _make_unique
        eps = 1e-15
        ux = List[float](); uy = List[float](); uw = List[float]()
        cur_x = Xs[0]; cur_y = 0.0; cur_w = 0.0
        for j in range(m):
            xv = Xs[j]
            if xv - cur_x >= eps:
                ux.append(cur_x); uw.append(cur_w); uy.append(cur_y / cur_w)
                cur_x = xv
                cur_w = ws[j]
                cur_y = ys[j] * ws[j]
            else:
                cur_w += ws[j]
                cur_y += ys[j] * ws[j]
        ux.append(cur_x); uw.append(cur_w); uy.append(cur_y / cur_w)
        Xu = np.array(ux)
        yu = isotonic_regression(np.array(uy), sample_weight=np.array(uw), y_min=self.y_min,
                                 y_max=self.y_max, increasing=self.increasing_)
        self.X_min_ = float(np.min(Xu))
        self.X_max_ = float(np.max(Xu))
        k = yu.shape[0]
        kx = List[float](); ky = List[float]()
        for i in range(k):
            keep_i = True
            if i > 0 and i < k - 1:
                keep_i = yu[i] != yu[i - 1] or yu[i] != yu[i + 1]
            if keep_i:
                kx.append(Xu[i]); ky.append(yu[i])
        self.X_thresholds_ = np.array(kx)
        self.y_thresholds_ = np.array(ky)
        return self

    def _transform(self, T):
        Tf = self._flat(T)
        n = Tf.shape[0]
        if self.out_of_bounds == "clip":
            for i in range(n):
                Tf[i] = min(max(Tf[i], self.X_min_), self.X_max_)
        xs = self.X_thresholds_
        ys = self.y_thresholds_
        m = xs.shape[0]
        out = np.empty(n)
        if m == 1:
            for i in range(n):
                out[i] = ys[0]
            return out
        for i in range(n):
            xn = Tf[i]
            if xn < xs[0] or xn > xs[m - 1] or math.isnan(xn):
                if self.out_of_bounds == "raise":
                    raise ValueError("A value in x_new is out of the interpolation range.")
                out[i] = np.nan
                continue
            # interp1d._call_linear
            hi = int(np.searchsorted(xs, xn))
            hi = min(max(hi, 1), m - 1)
            lo = hi - 1
            slope = (ys[hi] - ys[lo]) / (xs[hi] - xs[lo])
            out[i] = slope * (xn - xs[lo]) + ys[lo]
        return out

    def transform(self, T):
        return self._transform(T)

    def predict(self, T):
        return self._transform(T)

    def fit_transform(self, X, y, sample_weight = None):
        return self.fit(X, y, sample_weight).transform(X)

    def score(self, X, y, sample_weight = None) -> float:
        from .metrics import r2_score
        return r2_score(y, self.predict(X))
