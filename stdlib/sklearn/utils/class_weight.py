# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# sklearn.utils.class_weight. class_weight is None, "balanced" or a dict
# {label: weight} (keys of the label type).

import numpy as np

def compute_class_weight(class_weight, *, classes, y, sample_weight = None):
    cls = np.asarray(classes)
    ya = np.asarray(y)
    uy = np.unique(ya)
    for v in uy:
        found = False
        for c in cls:
            if c == v:
                found = True
        if not found:
            raise ValueError("classes should include all valid labels that can be in y")
    k = cls.shape[0]
    if isinstance(class_weight, NoneType):
        return np.ones(k)
    elif isinstance(class_weight, str):
        if class_weight != "balanced":
            raise ValueError("class_weight must be dict, 'balanced', or None, got: " + repr(class_weight))
        # LabelEncoder on y: le.classes_ == uy
        for c in cls:
            ok = False
            for v in uy:
                if v == c:
                    ok = True
            if not ok:
                raise ValueError("classes should have valid labels that are in y")
        from .validation import _check_sample_weight
        sw = _check_sample_weight(sample_weight, ya)
        counts = np.zeros(uy.shape[0])
        for i in range(ya.shape[0]):
            counts[int(np.searchsorted(uy, ya[i]))] += sw[i]
        total = float(np.sum(counts))
        recip = total / (float(uy.shape[0]) * counts)
        w = np.empty(k)
        for i in range(k):
            w[i] = recip[int(np.searchsorted(uy, cls[i]))]
        return w
    elif isinstance(class_weight, Optional):
        if class_weight is None:
            return np.ones(k)
        return compute_class_weight(class_weight.__val__(), classes=classes, y=y, sample_weight=sample_weight)
    else:
        if len(class_weight) == 0:
            return np.ones(k)
        w = np.ones(k)
        unweighted = 0
        for i in range(k):
            c = cls[i]
            if c in class_weight:
                w[i] = float(class_weight[c])
            else:
                unweighted += 1
        if unweighted > 0 and k - unweighted != len(class_weight):
            missing = [cls[i] for i in range(k) if cls[i] not in class_weight]
            raise ValueError(f"The classes, {missing}, are not in class_weight")
        return w

def compute_sample_weight(class_weight, y, *, indices = None):
    """Per-sample weights from class weights (1-D y; 2-D y: product over
    outputs, class_weight 'balanced' or a list of dicts)."""
    ya = np.asarray(y)
    if static.len(ya.shape) == 1:
        return compute_sample_weight(class_weight, ya.reshape(-1, 1), indices=indices)
    else:
        n, m = ya.shape
        if not isinstance(indices, NoneType):
            if not isinstance(class_weight, str):
                raise ValueError(f"The only valid class_weight for subsampling is 'balanced'. Given {class_weight}.")
        out = np.ones(n)
        for k in range(m):
            y_full = np.ascontiguousarray(ya[:, k])
            classes_full = np.unique(y_full)
            if isinstance(indices, NoneType):
                wk = _cw_k(class_weight, k, m, classes_full, y_full)
                for i in range(n):
                    out[i] *= wk[int(np.searchsorted(classes_full, y_full[i]))]
            else:
                idx = np.asarray(indices)
                y_sub = y_full[idx]
                classes_sub = np.unique(y_sub)
                wsub = compute_class_weight(class_weight, classes=classes_sub, y=y_sub)
                for i in range(n):
                    j = int(np.searchsorted(classes_sub, y_full[i]))
                    jj = min(j, classes_sub.shape[0] - 1)
                    if classes_sub[jj] == y_full[i]:
                        out[i] *= wsub[jj]
                    else:
                        out[i] *= 0.0
        return out

def _cw_k(class_weight, k: int, m: int, classes_full, y_full):
    if isinstance(class_weight, List):
        if len(class_weight) != m:
            raise ValueError(f"For multi-output, number of elements in class_weight should match number of outputs. Got {len(class_weight)} element(s) while having {m} outputs.")
        return compute_class_weight(class_weight[k], classes=classes_full, y=y_full)
    else:
        if m > 1 and not isinstance(class_weight, str):
            raise ValueError("For multi-output, class_weight should be a list of dicts, or the string 'balanced'.")
        return compute_class_weight(class_weight, classes=classes_full, y=y_full)
