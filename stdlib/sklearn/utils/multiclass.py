# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# sklearn.utils.multiclass: target type detection and label sets.

import numpy as np
from .validation import _is_float_t, _is_int_t

def _is_integral_float(y) -> bool:
    for v in y:
        if v != float(int(v)):
            return False
    return True

def is_multilabel(y) -> bool:
    """True for a 2-D (n, k>1) binary indicator matrix."""
    a = np.asarray(y)
    if static.len(a.shape) != 2:
        return False
    else:
        if a.shape[1] <= 1:
            return False
        labels = np.unique(a)
        if labels.shape[0] >= 3:
            return False
        if a.dtype is bool or _is_int_t(a.dtype):
            return True
        elif _is_float_t(a.dtype):
            return _is_integral_float(labels)
        return False

def type_of_target(y, input_name: str = "", raise_unknown: bool = False) -> str:
    """'continuous', 'continuous-multioutput', 'binary', 'multiclass',
    'multiclass-multioutput', 'multilabel-indicator' or 'unknown'."""
    a = np.asarray(y)
    nd: Literal[int] = static.len(a.shape)
    if is_multilabel(a):
        return "multilabel-indicator"
    if nd != 1 and nd != 2:
        if raise_unknown:
            raise ValueError(f"Unknown label type for {input_name if input_name else 'data'}")
        return "unknown"
    if a.size == 0 or min(a.shape) == 0:
        if nd == 1:
            return "binary"
        if raise_unknown:
            raise ValueError(f"Unknown label type for {input_name if input_name else 'data'}")
        return "unknown"
    suffix = ""
    ncols = 1
    if nd == 2:
        ncols = a.shape[1]
        if ncols > 1:
            suffix = "-multioutput"
    if _is_float_t(a.dtype):
        for v in a.ravel():
            if np.isnan(v) or np.isinf(v) or v != float(int(v)):
                from .validation import _assert_all_finite
                _assert_all_finite(a, False, input_name)
                return "continuous" + suffix
    if np.unique(a).shape[0] > 2 or (nd == 2 and ncols > 1):
        return "multiclass" + suffix
    return "binary"

def _unique_indicator(y):
    return np.arange(np.asarray(y).shape[1])

def unique_labels(*ys):
    """Sorted array of unique labels across the inputs (all of the same
    label type; multilabel inputs give column indices)."""
    types = List[str]()
    for y in ys:
        types.append(type_of_target(y))
    t0 = types[0]
    for t in types:
        if t != t0 and not ((t == "binary" or t == "multiclass") and (t0 == "binary" or t0 == "multiclass")):
            raise ValueError("Mix type of y not allowed, got types " + str(sorted(set(types))))
    if static.len(np.asarray(ys[0]).shape) == 2:
        if t0 != "multilabel-indicator":
            raise ValueError("Unknown label type: " + t0)
        return _ml_labels(*ys)
    else:
        if t0 != "binary" and t0 != "multiclass":
            raise ValueError("Unknown label type: " + t0)
        return _concat_unique(*ys)

def _ml_labels(*ys):
    k = np.asarray(ys[0]).shape[1]
    for y in ys:
        a = np.asarray(y)
        if static.len(a.shape) != 2:
            raise ValueError("Mix type of y not allowed")
        elif a.shape[1] != k:
            raise ValueError("Multi-label binary indicator input with different numbers of labels")
    return np.arange(k)

def _concat_unique(*ys):
    parts = tuple(np.asarray(y).ravel() for y in ys)
    return np.unique(np.concatenate(parts))

def check_classification_targets(y):
    t = type_of_target(y, input_name="y")
    if t not in ("binary", "multiclass", "multiclass-multioutput", "multilabel-indicator", "multilabel-sequences"):
        raise ValueError(f"Unknown label type: {t}. Maybe you are trying to fit a classifier, which expects discrete classes on a regression target with continuous values.")

def class_distribution(y, sample_weight = None):
    """Per-output (classes, n_classes, class_prior) for a 2-D y."""
    a = np.asarray(y)
    if static.len(a.shape) == 1:
        return class_distribution(a.reshape(-1, 1), sample_weight)
    else:
        n, m = a.shape
        sw = np.ones(n) if isinstance(sample_weight, NoneType) else np.asarray(sample_weight, dtype=float)
        classes = [np.unique(a[:, k]) for k in range(m)]
        n_classes = [c.shape[0] for c in classes]
        priors = List[np.ndarray[float, 1]]()
        for k in range(m):
            c = classes[k]
            p = np.zeros(c.shape[0])
            for i in range(n):
                p[int(np.searchsorted(c, a[i, k]))] += sw[i]
            priors.append(p / np.sum(p))
        return classes, n_classes, priors

def _ovr_decision_function(predictions, confidences, n_classes: int):
    """One-vs-one votes plus normalized confidences (OvO classifiers)."""
    P = np.asarray(predictions)
    C = np.asarray(confidences, dtype=float)
    n = P.shape[0]
    votes = np.zeros((n, n_classes))
    sum_conf = np.zeros((n, n_classes))
    k = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            for s in range(n):
                sum_conf[s, i] -= C[s, k]
                sum_conf[s, j] += C[s, k]
                if P[s, k] == 0:
                    votes[s, i] += 1
                else:
                    votes[s, j] += 1
            k += 1
    transformed = sum_conf / (3 * (np.abs(sum_conf) + 1))
    return votes + transformed
