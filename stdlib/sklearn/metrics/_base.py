# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Shared helpers for sklearn.metrics: target checking, label casting,
# weighted averages. Typhon is statically typed: 1-D targets are
# binary/multiclass, 2-D targets are multilabel indicator matrices, and
# `average=None` (NoneType) selects array returns at compile time.

import numpy as np
from ..utils.validation import _is_float_t, _is_int_t
from ..utils.multiclass import type_of_target

def _eq(a, b) -> bool:
    """a == b for labels, False when the types cannot compare."""
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    elif isinstance(a, str) or isinstance(b, str):
        return False
    elif isinstance(a, NoneType) or isinstance(b, NoneType):
        return False
    else:
        return a == b

def _cast_label(v, T: type):
    """v converted to label type T, or None when incompatible."""
    if isinstance(v, T):
        return Optional[T](v)
    elif isinstance(v, str) or T is str:
        return Optional[T]()
    elif isinstance(v, NoneType):
        return Optional[T]()
    else:
        return Optional[T](T(v))

def _common(yt, yp):
    """Bring the two label arrays to one dtype (numeric -> float)."""
    a = np.asarray(yt)
    b = np.asarray(yp)
    if a.dtype is b.dtype:
        return a, b
    elif a.dtype is str or b.dtype is str:
        compile_error("Mix of label input types (string and number)")
    else:
        return a.astype(float), b.astype(float)

def _zd(zero_division) -> float:
    if isinstance(zero_division, str):
        if zero_division != "warn":
            raise ValueError("zero_division must be 'warn', 0.0, 1.0 or np.nan")
        return 0.0
    else:
        z = float(zero_division)
        if z == 0.0 or z == 1.0:
            return z
        return np.nan

def _check_targets(y_true, y_pred):
    """(y_type, present_labels, y_true, y_pred) for 1-D (binary/multiclass)
    or 2-D (multilabel-indicator) targets."""
    a, b = _common(y_true, y_pred)
    if static.len(a.shape) != static.len(b.shape):
        compile_error("y_true and y_pred must have the same number of dimensions")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Found input variables with inconsistent numbers of samples: [{a.shape[0]}, {b.shape[0]}]")
    if a.shape[0] < 1:
        raise ValueError("Found empty input array (e.g., `y_true` or `y_pred`) while a minimum of 1 sample is required.")
    tt = type_of_target(a, input_name="y_true")
    tp = type_of_target(b, input_name="y_pred")
    t = tt
    if tt != tp:
        if (tt == "binary" and tp == "multiclass") or (tt == "multiclass" and tp == "binary"):
            t = "multiclass"
        else:
            raise ValueError(f"Classification metrics can't handle a mix of {tt} and {tp} targets")
    if t != "binary" and t != "multiclass" and t != "multilabel-indicator":
        raise ValueError(f"{t} is not supported")
    if static.len(a.shape) == 1:
        lab = np.unique(np.concatenate((a, b)))
        if t == "binary" and lab.shape[0] > 2:
            t = "multiclass"
        return t, lab, a, b
    else:
        if a.shape[1] != b.shape[1]:
            raise ValueError("Multi-label binary indicator input with different numbers of labels")
        return t, np.arange(a.shape[1]), a, b

def _sw_or_none(sample_weight, n: int):
    if isinstance(sample_weight, NoneType):
        return sample_weight
    else:
        sw = np.asarray(sample_weight, dtype=float)
        if sw.shape[0] != n:
            raise ValueError(f"Found input variables with inconsistent numbers of samples: [{n}, {sw.shape[0]}]")
        return sw

def _average(a, weights = None, normalize: bool = True) -> float:
    """sklearn.utils._array_api._average for 1-D numpy input."""
    A = np.asarray(a, dtype=float)
    if isinstance(weights, NoneType):
        if normalize:
            return float(np.average(A))
        return float(np.sum(A))
    else:
        W = np.asarray(weights, dtype=float)
        if normalize:
            return float(np.average(A, weights=W))
        return float(np.dot(A, W))

def _nanaverage(a, weights = None) -> float:
    A = np.asarray(a, dtype=float)
    if A.shape[0] == 0:
        return np.nan
    mask = np.isnan(A)
    if np.all(mask):
        return np.nan
    if isinstance(weights, NoneType):
        return float(np.nanmean(A))
    else:
        W = np.asarray(weights, dtype=float)
        a2 = A[~mask]
        w2 = W[~mask]
        if float(np.sum(w2)) == 0.0:
            return float(np.average(a2))
        return float(np.average(a2, weights=w2))

def _xlogy(x: float, y: float) -> float:
    if x == 0.0:
        return 0.0
    return x * np.log(y)

def _label_binarize_1d(y, classes):
    """One-hot (n, k) float matrix of y against sorted classes."""
    n = y.shape[0]
    k = classes.shape[0]
    T = np.zeros((n, k))
    for i in range(n):
        j = int(np.searchsorted(classes, y[i]))
        if j < k and _eq(classes[j], y[i]):
            T[i, j] = 1.0
    return T
