# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# partial_dependence (sklearn/inspection/_partial_dependence.py).
# Deviations (static typing): `average` is always 2-D (n_outputs, n_grid_points)
# and `individual` 3-D (n_outputs, n_samples, n_grid_points), with the grid in
# sklearn's cartesian order (for one feature this is exactly sklearn's shape;
# for two features reshape to (n_outputs, len(v0), len(v1))).
# method="recursion" (and "auto" with sample_weight=None) is supported for
# HistGradientBoosting*; other estimators use "brute" (sklearn picks recursion
# for GradientBoosting / DecisionTreeRegressor / RandomForestRegressor).

import math
import numpy as np

class _PDResult:
    grid_values: List[np.ndarray[float, 1]]
    average: np.ndarray[float, 2]
    individual: np.ndarray[float, 3]

    def __init__(self, grid_values: List[np.ndarray[float, 1]], average: np.ndarray[float, 2],
                 individual: np.ndarray[float, 3]):
        self.grid_values = grid_values
        self.average = average
        self.individual = individual

def _mquantile(col: np.ndarray[float, 1], prob: float) -> float:
    # scipy.stats.mstats.mquantiles(alphap=0.4, betap=0.4)
    x = np.sort(col)
    n = x.shape[0]
    if n == 1:
        return float(x[0])
    m = 0.4 + prob * (1.0 - 0.4 - 0.4)
    aleph = float(n) * prob + m
    k = int(math.floor(min(max(aleph, 1.0), float(n - 1))))
    gamma = min(max(aleph - float(k), 0.0), 1.0)
    return (1.0 - gamma) * float(x[k - 1]) + gamma * float(x[k])

def _axis(col: np.ndarray[float, 1], is_cat: bool, lo_p: float, hi_p: float,
          grid_resolution: int) -> np.ndarray[float, 1]:
    u = np.unique(col)
    if is_cat or u.shape[0] < grid_resolution:
        return u
    e0 = _mquantile(col, lo_p)
    e1 = _mquantile(col, hi_p)
    if abs(e0 - e1) <= 1e-8 + 1e-5 * abs(e1):
        raise ValueError("percentiles are too close to each other, unable to build the grid. Please choose percentiles that are further apart.")
    return np.linspace(e0, e1, num=grid_resolution)

def _response(est, X) -> np.ndarray[float, 2]:
    """_get_response_values with response_method='auto' -> (n, n_outputs)."""
    if hasattr(est, "predict_proba"):
        pr = np.asarray(est.predict_proba(X), dtype=float)
        if pr.shape[1] == 2:
            return pr[:, 1:].copy()
        return pr
    elif hasattr(est, "decision_function"):
        d = np.asarray(est.decision_function(X), dtype=float)
        if d.ndim == 1:
            return d.reshape((d.shape[0], 1))
        else:
            return d.reshape((d.shape[0], d.shape[1]))
    else:
        p = np.asarray(est.predict(X), dtype=float)
        return p.reshape((p.shape[0], 1))

def _avg_col(pred: np.ndarray[float, 2], k: int, sw: List[float], has_w: bool) -> float:
    n = pred.shape[0]
    if pred.shape[1] == 1:
        # np.average of a 1-D array (pairwise sums)
        a = np.empty(n)
        for i in range(n):
            a[i] = pred[i, 0] * sw[i] if has_w else pred[i, 0]
        if has_w:
            return float(np.sum(a)) / float(np.sum(np.array(sw)))
        return float(np.sum(a)) / float(n)
    # axis=0 reduction of a C-contiguous 2-D array: sequential row sums
    s = 0.0
    ws = 0.0
    for i in range(n):
        if has_w:
            s += pred[i, k] * sw[i]
            ws += sw[i]
        else:
            s += pred[i, k]
    return s / (ws if has_w else float(n))

def _recursion(est, g: np.ndarray[float, 2], feats: List[int]) -> np.ndarray[float, 2]:
    if hasattr(est, "_compute_partial_dependence_recursion"):
        return est._compute_partial_dependence_recursion(g, feats)
    else:
        raise ValueError("method='recursion' is not supported for this estimator")
        return np.empty((0, 0))

def partial_dependence(estimator, X, features, *, sample_weight = None,
                       categorical_features = None, feature_names = None,
                       response_method: str = "auto", percentiles = (0.05, 0.95),
                       grid_resolution: int = 100, custom_values = None,
                       method: str = "auto", kind: str = "average"):
    Xa = np.asarray(X, dtype=float)
    n = Xa.shape[0]
    p = Xa.shape[1]
    feats = List[int]()
    if isinstance(features, int):
        feats.append(features)
    else:
        for f in features:
            feats.append(int(f))
    for f in feats:
        if f < 0 or f >= p:
            raise ValueError(f"all features must be in [0, {p - 1}]")
    if len(feats) > 2:
        raise ValueError("partial_dependence supports one or two features in Typhon")
    if response_method != "auto" and response_method != "predict_proba" and response_method != "decision_function":
        raise ValueError("response_method must be 'auto', 'predict_proba' or 'decision_function'")
    lo_p = float(percentiles[0])
    hi_p = float(percentiles[1])
    if not (0.0 <= lo_p <= 1.0 and 0.0 <= hi_p <= 1.0):
        raise ValueError("'percentiles' values must be in [0, 1].")
    if lo_p >= hi_p:
        raise ValueError("percentiles[0] must be strictly less than percentiles[1].")
    if grid_resolution <= 1:
        raise ValueError("'grid_resolution' must be strictly greater than 1.")
    is_cat = [False] * len(feats)
    if categorical_features is not None:
        cf = [v for v in categorical_features]
        for t in range(len(feats)):
            if isinstance(cf[0], bool):
                is_cat[t] = bool(cf[feats[t]])
            else:
                for c in cf:
                    if int(c) == feats[t]:
                        is_cat[t] = True
    values = List[np.ndarray[float, 1]]()
    for t in range(len(feats)):
        col = np.array([Xa[i, feats[t]] for i in range(n)])
        values.append(_axis(col, is_cat[t], lo_p, hi_p, grid_resolution))
    grid = List[List[float]]()
    if len(feats) == 1:
        for v in values[0]:
            grid.append([float(v)])
    else:
        for a in values[0]:
            for b in values[1]:
                grid.append([float(a), float(b)])
    has_w = sample_weight is not None
    sw = [1.0] * n
    if sample_weight is not None:
        swa = np.asarray(sample_weight, dtype=float)
        for i in range(n):
            sw[i] = float(swa[i])
    use_rec = False
    if hasattr(estimator, "_compute_partial_dependence_recursion"):
        if method == "recursion":
            use_rec = True
        elif method == "auto" and not has_w and kind == "average":
            use_rec = True
    elif method == "recursion":
        raise ValueError("method='recursion' is only supported for HistGradientBoosting* in Typhon; use method='brute'")
    if kind != "average" and kind != "individual" and kind != "both":
        raise ValueError("kind must be 'average', 'individual' or 'both'")
    if use_rec and kind != "average":
        raise ValueError("The 'recursion' method only applies when 'kind' is set to 'average'")
    if use_rec:
        if has_w:
            raise ValueError("The 'recursion' method can only be applied when sample_weight is None.")
        g = np.empty((len(grid), len(feats)))
        for s in range(len(grid)):
            for t in range(len(feats)):
                g[s, t] = grid[s][t]
        return _PDResult(values, _recursion(estimator, g, feats), np.empty((0, 0, 0)))
    Xe = Xa.copy()
    ng = len(grid)
    avg = np.empty((0, 0))
    ind = np.empty((0, 0, 0))
    for s in range(ng):
        for t in range(len(feats)):
            for i in range(n):
                Xe[i, feats[t]] = grid[s][t]
        pred = _response(estimator, Xe)
        K = pred.shape[1]
        if s == 0:
            avg = np.empty((K, ng))
            ind = np.empty((K, n, ng))
        for k in range(K):
            avg[k, s] = _avg_col(pred, k, sw, has_w)
            for i in range(n):
                ind[k, i, s] = pred[i, k]
    return _PDResult(values, avg, ind)
