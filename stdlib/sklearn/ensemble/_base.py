# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Shared ensemble helpers (sklearn/ensemble/_base.py, _bootstrap.py and the
# private bits of _forest.py / _bagging.py / utils._random).

import math
import numpy as np
from ..tree._classes import _check_rs, _to_f32_colmajor, _encode_classes, _r2

MAX_INT: Literal[int] = 2147483647

def _get_n_samples_bootstrap(n_samples: int, max_samples_kind: str, max_samples_value: float,
                             sw_sum: float, has_w: bool) -> int:
    # max_samples_kind: "none", "int", "float"
    if max_samples_kind == "none":
        return n_samples
    elif max_samples_kind == "int":
        return int(max_samples_value)
    weighted_n = sw_sum if has_w else float(n_samples)
    return max(int(max_samples_value * weighted_n), 1)

def _parse_max_samples(max_samples) -> Tuple[str, float]:
    if max_samples is None:
        return ("none", 0.0)
    elif isinstance(max_samples, int):
        return ("int", float(max_samples))
    elif isinstance(max_samples, str):
        return ("auto", 0.0)
    else:
        return ("float", float(max_samples))

def _generate_sample_indices(seed: int, n_samples: int, n_samples_bootstrap: int,
                             sw: List[float], has_w: bool):
    rs = np.random.RandomState(seed)
    if not has_w:
        return rs.randint(0, n_samples, n_samples_bootstrap)
    tot = 0.0
    for w in sw:
        tot += w
    p = np.array([w / tot for w in sw])
    return rs.choice(n_samples, n_samples_bootstrap, replace=True, p=p)

def _generate_unsampled_indices(seed: int, n_samples: int, n_samples_bootstrap: int,
                                sw: List[float], has_w: bool) -> List[int]:
    idx = _generate_sample_indices(seed, n_samples, n_samples_bootstrap, sw, has_w)
    cnt = [0] * n_samples
    for i in range(idx.shape[0]):
        cnt[int(idx[i])] += 1
    return [i for i in range(n_samples) if cnt[i] == 0]

def _bincount(idx, n: int) -> List[float]:
    c = [0.0] * n
    for i in range(idx.shape[0]):
        c[int(idx[i])] += 1.0
    return c

def sample_without_replacement(n_population: int, n_samples: int, rs):
    """sklearn.utils.random.sample_without_replacement (method='auto')."""
    ratio = float(n_samples) / float(n_population) if n_population != 0 else 1.0
    if ratio > 0.01 and ratio < 0.99:
        return rs.permutation(n_population)[:n_samples]
    out = np.empty(n_samples, dtype=int)
    if ratio < 0.2:
        selected = Set[int]()
        for i in range(n_samples):
            j = int(rs.randint(n_population))
            while j in selected:
                j = int(rs.randint(n_population))
            selected.add(j)
            out[i] = j
    else:
        for i in range(n_samples):
            out[i] = i
        for i in range(n_samples, n_population):
            j = int(rs.randint(0, i + 1))
            if j < n_samples:
                out[j] = i
    return out

def _mean_rows(rows: List[np.ndarray[float, 1]], n: int):
    # np.mean(list_of_arrays, axis=0): column-wise pairwise sums of a 2-d array
    m = len(rows)
    a = np.empty((m, n))
    for i in range(m):
        for j in range(n):
            a[i, j] = rows[i][j]
    return np.mean(a, axis=0)

def _accuracy(y_true, y_pred) -> float:
    from ..base import _accuracy as acc
    return acc(y_true, y_pred)
