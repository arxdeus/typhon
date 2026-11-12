# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

import numpy as np

def _labels(y_true, y_pred):
    a = np.asarray(y_true)
    b = np.asarray(y_pred)
    return np.unique(np.concatenate((a, b))), a, b

def accuracy_score(y_true, y_pred, *, normalize: bool = True, sample_weight = None) -> float:
    a = np.asarray(y_true)
    b = np.asarray(y_pred)
    n = a.shape[0]
    c = 0
    for i in range(n):
        if a[i] == b[i]:
            c += 1
    return float(c) / float(n) if normalize else float(c)

def confusion_matrix(y_true, y_pred, *, labels = None, sample_weight = None, normalize = None):
    lab, a, b = _labels(y_true, y_pred)
    k = lab.shape[0]
    cm = np.zeros((k, k), dtype=int)
    for i in range(a.shape[0]):
        cm[int(np.searchsorted(lab, a[i])), int(np.searchsorted(lab, b[i]))] += 1
    return cm

def balanced_accuracy_score(y_true, y_pred, *, sample_weight = None, adjusted: bool = False) -> float:
    cm = confusion_matrix(y_true, y_pred)
    k = cm.shape[0]
    s = 0.0
    m = 0
    for i in range(k):
        r = int(np.sum(cm[i]))
        if r > 0:
            s += float(cm[i, i]) / float(r)
            m += 1
    score = s / float(m)
    if adjusted:
        chance = 1.0 / float(m)
        score = (score - chance) / (1 - chance)
    return score

def _prf(cm, i: int):
    tp = float(cm[i, i])
    fp = float(np.sum(cm[:, i])) - tp
    fn = float(np.sum(cm[i, :])) - tp
    return tp, fp, fn

def f1_score(y_true, y_pred, *, labels = None, pos_label: int = 1, average: str = "binary",
             sample_weight = None, zero_division = "warn") -> float:
    lab, a, b = _labels(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    k = lab.shape[0]
    def f1(tp: float, fp: float, fn: float) -> float:
        d = 2 * tp + fp + fn
        return 0.0 if d == 0 else 2 * tp / d
    if average == "binary":
        if k > 2:
            raise ValueError("Target is multiclass but average='binary'. Please choose another average setting.")
        i = int(np.searchsorted(lab, pos_label))
        if i >= k or lab[i] != pos_label:
            return 0.0
        tp, fp, fn = _prf(cm, i)
        return f1(tp, fp, fn)
    elif average == "micro":
        TP = 0.0; FP = 0.0; FN = 0.0
        for i in range(k):
            tp, fp, fn = _prf(cm, i)
            TP += tp; FP += fp; FN += fn
        return f1(TP, FP, FN)
    elif average == "macro" or average == "weighted":
        tot = 0.0
        wsum = 0.0
        for i in range(k):
            tp, fp, fn = _prf(cm, i)
            w = 1.0 if average == "macro" else tp + fn
            tot += w * f1(tp, fp, fn)
            wsum += w
        return tot / wsum
    else:
        raise ValueError(f"unsupported average {average}")

def _binary_auc(y, s) -> float:
    # Mann-Whitney U with average ranks for ties (== trapezoidal ROC area)
    n = s.shape[0]
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[order[j + 1]] == s[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = r
        i = j + 1
    npos = 0
    rs = 0.0
    for t in range(n):
        if y[t]:
            npos += 1
            rs += ranks[t]
    nneg = n - npos
    if npos == 0 or nneg == 0:
        raise ValueError("Only one class is present in y_true. ROC AUC score is not defined in that case.")
    return (rs - npos * (npos + 1) / 2.0) / (float(npos) * float(nneg))

def roc_auc_score(y_true, y_score, *, average: str = "macro", sample_weight = None,
                  max_fpr = None, multi_class: str = "raise", labels = None) -> float:
    yt = np.asarray(y_true)
    sc = np.asarray(y_score, dtype=float)
    if sc.ndim == 1:
        lab = np.unique(yt)
        if lab.shape[0] != 2:
            raise ValueError("binary roc_auc_score needs 2 classes")
        return _binary_auc(yt == lab[1], sc)
    else:
        if multi_class != "ovr":
            raise NotImplementedError("roc_auc_score: only multi_class='ovr' is ported")
        lab = np.unique(yt)
        k = lab.shape[0]
        tot = 0.0
        wsum = 0.0
        for i in range(k):
            yi = yt == lab[i]
            w = 1.0 if average == "macro" else float(np.sum(yi))
            tot += w * _binary_auc(yi, np.ascontiguousarray(sc[:, i]))
            wsum += w
        return tot / wsum
