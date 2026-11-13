# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# sklearn.inspection.permutation_importance
# (sklearn/inspection/_permutation_importance.py).
# Scoring: None (estimator.score: accuracy / r2), "accuracy", "r2",
# "neg_mean_squared_error", "neg_mean_absolute_error". Returns an object with
# importances_mean / importances_std / importances attributes.

import math
import numpy as np
from ..ensemble._base import sample_without_replacement

class _ImportancesBunch:
    """Result of permutation_importance (sklearn returns a Bunch)."""
    importances_mean: np.ndarray[float, 1]
    importances_std: np.ndarray[float, 1]
    importances: np.ndarray[float, 2]

    def __init__(self, importances_mean: np.ndarray[float, 1], importances_std: np.ndarray[float, 1],
                 importances: np.ndarray[float, 2]):
        self.importances_mean = importances_mean
        self.importances_std = importances_std
        self.importances = importances

def _is_clf(est) -> bool:
    if hasattr(est, "_estimator_type"):
        return est._estimator_type == "classifier"
    else:
        return hasattr(est, "classes_")

def _score(est, scoring: str, X, y, sw: List[float], has_w: bool) -> float:
    pred = est.predict(X)
    yt = np.asarray(y)
    n = yt.shape[0]
    w = sw if has_w else [1.0] * n
    sc = scoring
    if sc == "":
        sc = "accuracy" if _is_clf(est) else "r2"
    if sc == "accuracy":
        a = np.empty(n)
        for i in range(n):
            a[i] = 1.0 if float(pred[i]) == float(yt[i]) else 0.0
        if has_w:
            return float(np.average(a, weights=np.array(w)))
        return float(np.mean(a))
    yp = np.asarray(pred, dtype=float)
    yf = np.asarray(yt, dtype=float)
    if sc == "r2":
        wa = np.array(w)
        ybar = float(np.average(yf, weights=wa))
        num = float(np.sum(wa * (yf - yp) ** 2))
        den = float(np.sum(wa * (yf - ybar) ** 2))
        if den == 0.0:
            return 1.0 if num == 0.0 else 0.0
        return 1.0 - num / den
    if sc == "neg_mean_squared_error":
        return -float(np.average((yf - yp) ** 2, weights=np.array(w)))
    if sc == "neg_mean_absolute_error":
        return -float(np.average(np.abs(yf - yp), weights=np.array(w)))
    raise ValueError(f"scoring={scoring} is not supported")

def permutation_importance(estimator, X, y, *, scoring = None, n_repeats: int = 5,
                           n_jobs: Optional[int] = None, random_state: Optional[int] = None,
                           sample_weight = None, max_samples = 1.0):
    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y)
    n = Xa.shape[0]
    p = Xa.shape[1]
    sc = ""
    if scoring is not None:
        sc = scoring
    if random_state is None:
        raise ValueError("permutation_importance needs an integer random_state in Typhon")
    rs = np.random.RandomState(random_state)
    random_seed = int(rs.randint(2147483648))
    ms = 0
    if isinstance(max_samples, float):
        ms = int(max_samples * float(n))
    else:
        ms = max_samples
        if ms > n:
            raise ValueError("max_samples must be <= n_samples")
    has_w = sample_weight is not None
    sw = [1.0] * n
    if sample_weight is not None:
        swa = np.asarray(sample_weight, dtype=float)
        for i in range(n):
            sw[i] = float(swa[i])
    baseline = _score(estimator, sc, Xa, ya, sw, has_w)
    imps = np.empty((p, n_repeats))
    for col in range(p):
        r = np.random.RandomState(random_seed)
        Xp = Xa.copy()
        yp = ya
        swp = sw
        if ms < n:
            rows = sample_without_replacement(n, ms, r)
            Xp = Xa[rows]
            yp = ya[rows]
            swp = [sw[int(i)] for i in rows]
        idx = np.arange(Xp.shape[0])
        for t in range(n_repeats):
            r.shuffle(idx)
            colv = np.array([Xp[int(idx[i]), col] for i in range(Xp.shape[0])])
            for i in range(Xp.shape[0]):
                Xp[i, col] = colv[i]
            imps[col, t] = baseline - _score(estimator, sc, Xp, yp, swp, has_w)
    return _ImportancesBunch(np.mean(imps, axis=1), np.std(imps, axis=1), imps)
