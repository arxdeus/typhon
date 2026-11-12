# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# _BinMapper (sklearn/ensemble/_hist_gradient_boosting/binning.py and
# _binning.pyx). Binned data is stored column-major as List[int]:
# Xb[f * n + i]. The last bin (n_bins - 1) is reserved for missing values.

import math
import numpy as np

ALMOST_INF: float = 1e300

def _weighted_percentile_avg(a: List[float], w: List[float], q: float) -> float:
    """sklearn.utils.stats._weighted_percentile(a, w, q, average=True) for 1-D
    input without NaN (a is sorted ascending by the caller)."""
    n = len(a)
    cdf = [0.0] * n
    acc = 0.0
    for t in range(n):
        acc += w[t]
        cdf[t] = acc
    adj = q / 100.0 * cdf[n - 1]
    if adj == 0.0:
        adj = 5e-324
    lo = 0
    hi = n
    while lo < hi:
        mid = (lo + hi) >> 1
        if cdf[mid] < adj:
            lo = mid + 1
        else:
            hi = mid
    max_idx = n - 1
    pi = min(lo, max_idx)
    frac = cdf[pi] - adj
    if frac > 2.220446049250313e-16:
        return a[pi]
    p1 = min(pi + 1, max_idx)
    if w[p1] == 0.0:
        # searchsorted(cdf, cdf[pi], side="right")
        cv = cdf[pi]
        lo = 0
        hi = n
        while lo < hi:
            mid = (lo + hi) >> 1
            if cdf[mid] <= cv:
                lo = mid + 1
            else:
                hi = mid
        p1 = lo if lo <= max_idx else pi
    return (a[pi] + a[p1]) / 2.0

def _find_binning_thresholds(col: List[float], max_bins: int, sw: List[float],
                             has_w: bool) -> List[float]:
    vals = List[float](len(col))
    ws = List[float](len(col))
    for i in range(len(col)):
        v = col[i]
        if math.isnan(v):
            continue
        if has_w:
            if sw[i] == 0.0:
                continue
            ws.append(sw[i])
        vals.append(v)
    order = np.argsort(np.array(vals), kind="stable") if len(vals) > 0 else np.empty(0, dtype=int)
    sv = [vals[int(order[t])] for t in range(len(vals))]
    swv = [ws[int(order[t])] for t in range(len(vals))] if has_w else List[float]()
    distinct = List[float]()
    for v in sv:
        if len(distinct) == 0 or distinct[len(distinct) - 1] != v:
            distinct.append(v)
    if len(distinct) <= 1:
        return List[float]()
    th = List[float]()
    if len(distinct) <= max_bins:
        for t in range(len(distinct) - 1):
            th.append((distinct[t] + distinct[t + 1]) / 2.0)
    else:
        pct = np.linspace(0.0, 100.0, num=max_bins + 1)
        if not has_w:
            qs = np.percentile(np.array(sv), pct[1:max_bins], method="averaged_inverted_cdf")
            for t in range(max_bins - 1):
                th.append(float(qs[t]))
        else:
            for t in range(1, max_bins):
                th.append(_weighted_percentile_avg(sv, swv, float(pct[t])))
        # np.unique (sorted, deduplicated) when there are duplicates
        uq = sorted(th)
        ded = List[float]()
        for v in uq:
            if len(ded) == 0 or ded[len(ded) - 1] != v:
                ded.append(v)
        if len(ded) != len(th):
            th = ded
    for t in range(len(th)):
        if th[t] > ALMOST_INF:
            th[t] = ALMOST_INF
    return th

class _BinMapper:
    n_bins: int
    subsample: int
    random_state: int
    is_categorical_: List[bool]
    bin_thresholds_: List[List[float]]
    n_bins_non_missing_: List[int]
    missing_values_bin_idx_: int

    def __init__(self, n_bins: int = 256, random_state: int = 0):
        self.n_bins = n_bins
        self.subsample = 200000
        self.random_state = random_state
        self.is_categorical_ = List[bool]()
        self.bin_thresholds_ = List[List[float]]()
        self.n_bins_non_missing_ = List[int]()
        self.missing_values_bin_idx_ = n_bins - 1

    def fit(self, X: np.ndarray[float, 2], sw: List[float], has_w: bool,
            is_categorical: List[bool], known_categories: List[List[float]]):
        if not (3 <= self.n_bins <= 256):
            raise ValueError(f"n_bins={self.n_bins} should be no smaller than 3 and no larger than 256.")
        n = X.shape[0]
        p = X.shape[1]
        max_bins = self.n_bins - 1
        rows = List[int]()
        use_w = has_w
        if n > self.subsample:
            rs = np.random.RandomState(self.random_state)
            sub = np.empty(0, dtype=int)
            if has_w:
                tot = 0.0
                for v in sw:
                    tot += v
                probs = np.array([v / tot for v in sw])
                sub = rs.choice(n, self.subsample, replace=True, p=probs)
            else:
                sub = rs.choice(n, self.subsample, replace=True)
            for t in range(sub.shape[0]):
                rows.append(int(sub[t]))
            use_w = False
        else:
            rows = list(range(n))
        self.is_categorical_ = list(is_categorical) if len(is_categorical) == p else [False] * p
        self.missing_values_bin_idx_ = self.n_bins - 1
        self.bin_thresholds_ = List[List[float]]()
        self.n_bins_non_missing_ = List[int]()
        for f in range(p):
            if self.is_categorical_[f]:
                th = known_categories[f]
                self.bin_thresholds_.append(th)
                self.n_bins_non_missing_.append(len(th))
            else:
                col = [X[i, f] for i in rows]
                wsub = [sw[i] for i in rows] if use_w else List[float]()
                th = _find_binning_thresholds(col, max_bins, wsub, use_w)
                self.bin_thresholds_.append(th)
                self.n_bins_non_missing_.append(len(th) + 1)
        return self

    def transform(self, X: np.ndarray[float, 2]) -> List[int]:
        n = X.shape[0]
        p = X.shape[1]
        if p != len(self.n_bins_non_missing_):
            raise ValueError(f"This estimator was fitted with {len(self.n_bins_non_missing_)} features but {p} got passed to transform()")
        out = [0] * (n * p)
        mb = self.missing_values_bin_idx_
        for f in range(p):
            th = self.bin_thresholds_[f]
            cat = self.is_categorical_[f]
            nt = len(th)
            for i in range(n):
                v = X[i, f]
                if math.isnan(v) or (cat and v < 0.0):
                    out[f * n + i] = mb
                else:
                    left = 0
                    right = nt
                    while left < right:
                        middle = left + (right - left - 1) // 2
                        if v <= th[middle]:
                            right = middle
                        else:
                            left = middle + 1
                    out[f * n + i] = left
        return out

    def known_cat_bitsets(self) -> List[List[bool]]:
        """One 256-entry membership table per feature (empty for numeric)."""
        out = List[List[bool]]()
        for f in range(len(self.is_categorical_)):
            if self.is_categorical_[f]:
                b = [False] * 256
                for c in self.bin_thresholds_[f]:
                    b[int(c) & 255] = True
                out.append(b)
            else:
                out.append(List[bool]())
        return out
