# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Classification metrics: confusion matrices and set-wise scores
# (precision / recall / F-beta / Jaccard), accuracy, kappa, MCC, Hamming.
# `average=None` returns per-label arrays (static dispatch on NoneType).

import numpy as np
from ._base import (_check_targets, _sw_or_none, _average, _nanaverage, _zd, _eq,
                    _cast_label, _common)

def _labels_arr(labels, present):
    """User `labels` as an array of the target dtype (present labels when
    None) and whether labels were given."""
    if isinstance(labels, NoneType):
        return present.copy(), False
    else:
        L = np.asarray(labels)
        if L.dtype is present.dtype:
            return L.copy(), True
        elif L.dtype is str or present.dtype is str:
            compile_error("labels and targets must both be strings or both numbers")
        else:
            return L.astype(present.dtype), True

def _encode(y, sorted_labels):
    """Index of each y in sorted_labels (-1 when absent)."""
    n = y.shape[0]
    k = sorted_labels.shape[0]
    out = np.empty(n, dtype=int)
    for i in range(n):
        j = int(np.searchsorted(sorted_labels, y[i]))
        out[i] = j if j < k and sorted_labels[j] == y[i] else -1
    return out

def _mcm_counts(a, b, y_type: str, present, labs, n_labels: int, sw, samplewise: bool):
    """(tp_sum, pred_sum, true_sum, tn) as float arrays for the selected
    labels, following multilabel_confusion_matrix."""
    if static.len(a.shape) == 1:
        if samplewise:
            raise ValueError("Samplewise metrics are not available outside of multilabel classification.")
        full = np.concatenate((labs, np.setdiff1d(present, labs, assume_unique=True)))
        srt = np.unique(full)
        k = srt.shape[0]
        ei = _encode(a, srt)
        ep = _encode(b, srt)
        tp = np.zeros(k)
        ps = np.zeros(k)
        ts = np.zeros(k)
        for i in range(a.shape[0]):
            w = 1.0 if isinstance(sw, NoneType) else sw[i]
            if ei[i] == ep[i]:
                tp[ei[i]] += w
            ps[ep[i]] += w
            ts[ei[i]] += w
        idx = np.searchsorted(srt, labs[:n_labels])
        tp_s = tp[idx]
        ps_s = ps[idx]
        ts_s = ts[idx]
        total = float(a.shape[0]) if isinstance(sw, NoneType) else float(np.sum(sw))
        tn = total - tp_s - (ps_s - tp_s) - (ts_s - tp_s)
        return tp_s, ps_s, ts_s, tn
    else:
        n, m = a.shape
        sel = labs[:n_labels].astype(int)
        if n_labels > 0:
            if int(np.max(sel)) > m - 1:
                raise ValueError(f"All labels must be in [0, n labels) for multilabel targets. Got {int(np.max(sel))} > {m - 1}")
            if int(np.min(sel)) < 0:
                raise ValueError(f"All labels must be in [0, n labels) for multilabel targets. Got {int(np.min(sel))} < 0")
        L = sel.shape[0]
        if samplewise:
            tp = np.zeros(n); ps = np.zeros(n); ts = np.zeros(n)
            for i in range(n):
                w = 1.0 if isinstance(sw, NoneType) else sw[i]
                for jj in range(L):
                    j = sel[jj]
                    t = a[i, j] != 0
                    p = b[i, j] != 0
                    if t and p:
                        tp[i] += w
                    if p:
                        ps[i] += w
                    if t:
                        ts[i] += w
            if isinstance(sw, NoneType):
                tn = float(L) - tp - (ps - tp) - (ts - tp)
            else:
                tn = sw * float(L) - tp - (ps - tp) - (ts - tp)
            return tp, ps, ts, tn
        else:
            tp = np.zeros(L); ps = np.zeros(L); ts = np.zeros(L)
            for i in range(n):
                w = 1.0 if isinstance(sw, NoneType) else sw[i]
                for jj in range(L):
                    j = sel[jj]
                    t = a[i, j] != 0
                    p = b[i, j] != 0
                    if t and p:
                        tp[jj] += w
                    if p:
                        ps[jj] += w
                    if t:
                        ts[jj] += w
            total = float(n) if isinstance(sw, NoneType) else float(np.sum(sw))
            tn = total - tp - (ps - tp) - (ts - tp)
            return tp, ps, ts, tn

def multilabel_confusion_matrix(y_true, y_pred, *, sample_weight = None, labels = None,
                                samplewise: bool = False):
    """(n_outputs, 2, 2) matrices [[tn, fp], [fn, tp]] per label (or per
    sample with samplewise=True)."""
    y_type, present, a, b = _check_targets(y_true, y_pred)
    sw = _sw_or_none(sample_weight, a.shape[0])
    labs, given = _labels_arr(labels, present)
    tp, ps, ts, tn = _mcm_counts(a, b, y_type, present, labs, labs.shape[0], sw, samplewise)
    k = tp.shape[0]
    out = np.empty((k, 2, 2))
    for i in range(k):
        out[i, 0, 0] = tn[i]
        out[i, 0, 1] = ps[i] - tp[i]
        out[i, 1, 0] = ts[i] - tp[i]
        out[i, 1, 1] = tp[i]
    if isinstance(sample_weight, NoneType):
        return out.astype(int)
    else:
        return out

def confusion_matrix(y_true, y_pred, *, labels = None, sample_weight = None, normalize = None):
    """C[i, j] = number (or weight) of samples with true label i predicted
    as j. normalize in {None, 'true', 'pred', 'all'} (float result)."""
    a0, b0 = _common(y_true, y_pred)
    if static.len(a0.shape) != 1:
        compile_error("multilabel-indicator is not supported")
    if a0.shape[0] == 0:
        L0, g0 = _labels_arr(labels, np.unique(a0))
        if isinstance(normalize, NoneType) and isinstance(sample_weight, NoneType):
            return np.zeros((L0.shape[0], L0.shape[0]), dtype=int)
        else:
            return np.zeros((L0.shape[0], L0.shape[0]))
    y_type, present, a, b = _check_targets(a0, b0)
    sw = _sw_or_none(sample_weight, a.shape[0])
    labs, given = _labels_arr(labels, present)
    k = labs.shape[0]
    if given:
        if k == 0:
            raise ValueError("'labels' should contain at least one label.")
        hit = False
        for v in a:
            for l in labs:
                if v == l:
                    hit = True
        if not hit:
            raise ValueError("At least one label specified must be in y_true")
    # label -> first index (dict semantics: last duplicate wins, as sklearn)
    order = np.argsort(labs, kind="mergesort")
    srt = labs[order]
    cm = np.zeros((k, k))
    for i in range(a.shape[0]):
        it = _lab_index(srt, order, a[i])
        ip = _lab_index(srt, order, b[i])
        if it >= 0 and ip >= 0:
            cm[it, ip] += 1.0 if isinstance(sw, NoneType) else sw[i]
    if isinstance(normalize, NoneType):
        if isinstance(sample_weight, NoneType):
            return cm.astype(int)
        else:
            return cm
    else:
        if normalize == "true":
            cm = cm / np.sum(cm, axis=1, keepdims=True)
        elif normalize == "pred":
            cm = cm / np.sum(cm, axis=0, keepdims=True)
        elif normalize == "all":
            cm = cm / np.sum(cm)
        else:
            raise ValueError("normalize must be one of {'true', 'pred', 'all', None}")
        return np.nan_to_num(cm)

def _lab_index(srt, order, v) -> int:
    j = int(np.searchsorted(srt, v, side="right")) - 1
    if j >= 0 and srt[j] == v:
        return int(order[j])
    return -1

def accuracy_score(y_true, y_pred, *, normalize: bool = True, sample_weight = None) -> float:
    y_type, present, a, b = _check_targets(y_true, y_pred)
    sw = _sw_or_none(sample_weight, a.shape[0])
    n = a.shape[0]
    score = np.empty(n)
    for i in range(n):
        if static.len(a.shape) == 1:
            score[i] = 1.0 if a[i] == b[i] else 0.0
        else:
            ok = True
            for j in range(a.shape[1]):
                if a[i, j] != b[i, j]:
                    ok = False
            score[i] = 1.0 if ok else 0.0
    return _average(score, sw, normalize)

def zero_one_loss(y_true, y_pred, *, normalize: bool = True, sample_weight = None) -> float:
    score = accuracy_score(y_true, y_pred, normalize=normalize, sample_weight=sample_weight)
    if normalize:
        return 1 - score
    if isinstance(sample_weight, NoneType):
        return float(np.asarray(y_true).shape[0]) - score
    return float(np.sum(np.asarray(sample_weight, dtype=float))) - score

# ------------------------------------------------------------ set-wise

def _check_average(average):
    if not isinstance(average, NoneType):
        if average not in ("micro", "macro", "weighted", "samples", "binary"):
            raise ValueError("average has to be one of (None, 'micro', 'macro', 'weighted', 'samples')")

def _set_wise_labels(y_type: str, present, labels, average, pos_label):
    """_check_set_wise_labels: the label array to score."""
    _check_average(average)
    labs, given = _labels_arr(labels, present)
    if isinstance(average, NoneType):
        return labs
    else:
        if average == "binary":
            if y_type == "binary":
                pl = _cast_label(pos_label, present.dtype)
                found = False
                if pl is not None:
                    for v in present:
                        if v == pl.__val__():
                            found = True
                if not found and present.shape[0] >= 2:
                    raise ValueError(f"pos_label={pos_label} is not a valid label. It should be one of {present.tolist()}")
                if pl is None:
                    raise ValueError(f"pos_label={pos_label} is not a valid label.")
                out = np.empty(1, dtype=present.dtype)
                out[0] = pl.__val__()
                return out
            else:
                opts = "['micro', 'macro', 'weighted']" if y_type == "multiclass" else "[None, 'micro', 'macro', 'weighted', 'samples']"
                if y_type == "multiclass":
                    opts = "[None, 'micro', 'macro', 'weighted']"
                raise ValueError(f"Target is {y_type} but average='binary'. Please choose another average setting, one of {opts}.")
        return labs

def _prf_divide(num, den, zero_division) -> np.ndarray[float, 1]:
    out = np.empty(num.shape[0])
    zv = _zd(zero_division)
    for i in range(num.shape[0]):
        if den[i] == 0:
            out[i] = zv
        else:
            out[i] = num[i] / den[i]
    return out

def _prfs_core(y_true, y_pred, beta: float, labels, pos_label, average, sample_weight, zero_division):
    """(precision, recall, fscore, support) arrays and averaging weights."""
    if beta < 0:
        raise ValueError("beta should be >=0 in the F-beta score")
    _zd(zero_division)
    y_type, present, a, b = _check_targets(y_true, y_pred)
    sw = _sw_or_none(sample_weight, a.shape[0])
    labs = _set_wise_labels(y_type, present, labels, average, pos_label)
    samplewise = False
    if not isinstance(average, NoneType):
        samplewise = average == "samples"
    tp, ps, ts, tn = _mcm_counts(a, b, y_type, present, labs, labs.shape[0], sw, samplewise)
    if not isinstance(average, NoneType):
        if average == "micro":
            tp = np.array([float(np.sum(tp))])
            ps = np.array([float(np.sum(ps))])
            ts = np.array([float(np.sum(ts))])
    beta2 = beta * beta
    precision = _prf_divide(tp, ps, zero_division)
    recall = _prf_divide(tp, ts, zero_division)
    if np.isinf(beta):
        f = recall.copy()
    elif beta == 0:
        f = precision.copy()
    else:
        f = _prf_divide((1 + beta2) * tp, beta2 * ts + ps, zero_division)
    return precision, recall, f, ts, sw

def _avg_w(average, ts, sw):
    if average == "weighted":
        return Optional(ts)
    if not isinstance(sw, NoneType):
        if average == "samples":
            return Optional(np.asarray(sw, dtype=float))
    return Optional[np.ndarray[float, 1]]()

def _nanavg_opt(a, w: Optional[np.ndarray[float, 1]]) -> float:
    if w is None:
        return _nanaverage(a)
    return _nanaverage(a, w.__val__())

def precision_recall_fscore_support(y_true, y_pred, *, beta: float = 1.0, labels = None,
                                    pos_label = 1, average = None,
                                    warn_for = ("precision", "recall", "f-score"),
                                    sample_weight = None, zero_division = "warn"):
    p, r, f, ts, sw = _prfs_core(y_true, y_pred, beta, labels, pos_label, average,
                                 sample_weight, zero_division)
    if isinstance(average, NoneType):
        if isinstance(sample_weight, NoneType):
            return p, r, f, ts.astype(int)
        else:
            return p, r, f, ts
    else:
        w = _avg_w(average, ts, sw)
        return _nanavg_opt(p, w), _nanavg_opt(r, w), _nanavg_opt(f, w), None

def _one_score(which: Literal[int], y_true, y_pred, beta: float, labels, pos_label, average,
               sample_weight, zero_division):
    p, r, f, ts, sw = _prfs_core(y_true, y_pred, beta, labels, pos_label, average,
                                 sample_weight, zero_division)
    v = p if which == 0 else (r if which == 1 else f)
    if isinstance(average, NoneType):
        return v
    else:
        return _nanavg_opt(v, _avg_w(average, ts, sw))

def precision_score(y_true, y_pred, *, labels = None, pos_label = 1, average = "binary",
                    sample_weight = None, zero_division = "warn"):
    return _one_score(0, y_true, y_pred, 1.0, labels, pos_label, average, sample_weight, zero_division)

def recall_score(y_true, y_pred, *, labels = None, pos_label = 1, average = "binary",
                 sample_weight = None, zero_division = "warn"):
    return _one_score(1, y_true, y_pred, 1.0, labels, pos_label, average, sample_weight, zero_division)

def fbeta_score(y_true, y_pred, *, beta: float, labels = None, pos_label = 1, average = "binary",
                sample_weight = None, zero_division = "warn"):
    return _one_score(2, y_true, y_pred, float(beta), labels, pos_label, average, sample_weight, zero_division)

def f1_score(y_true, y_pred, *, labels = None, pos_label = 1, average = "binary",
             sample_weight = None, zero_division = "warn"):
    return _one_score(2, y_true, y_pred, 1.0, labels, pos_label, average, sample_weight, zero_division)

def jaccard_score(y_true, y_pred, *, labels = None, pos_label = 1, average = "binary",
                  sample_weight = None, zero_division = "warn"):
    y_type, present, a, b = _check_targets(y_true, y_pred)
    sw = _sw_or_none(sample_weight, a.shape[0])
    labs = _set_wise_labels(y_type, present, labels, average, pos_label)
    samplewise = False
    if not isinstance(average, NoneType):
        samplewise = average == "samples"
    tp, ps, ts, tn = _mcm_counts(a, b, y_type, present, labs, labs.shape[0], sw, samplewise)
    num = tp
    den = ps + ts - tp
    if not isinstance(average, NoneType):
        if average == "micro":
            num = np.array([float(np.sum(num))])
            den = np.array([float(np.sum(den))])
    jac = _prf_divide(num, den, zero_division)
    if isinstance(average, NoneType):
        return jac
    else:
        if average == "weighted":
            if float(np.sum(ts)) != 0.0:
                return _average(jac, ts)
            return _average(jac)
        elif average == "samples" and not isinstance(sw, NoneType):
            return _average(jac, sw)
        return _average(jac)

# ------------------------------------------------------------ report

class ClassificationReportDict:
    """output_dict=True result: report[label] -> {"precision", "recall",
    "f1-score", "support"}; report["accuracy"] -> float (static key)."""
    rows: Dict[str, Dict[str, float]]
    keys_: List[str]
    has_accuracy: bool
    accuracy: float

    def __init__(self):
        self.rows = {}
        self.keys_ = []
        self.has_accuracy = False
        self.accuracy = 0.0

    def _add(self, k: str, d: Dict[str, float]):
        self.rows[k] = d
        self.keys_.append(k)

    def __getitem__(self, k: str):
        return self.rows[k]

    @overload
    def __getitem__(self, k: Literal[str]):
        if k == "accuracy":
            if not self.has_accuracy:
                raise KeyError(k)
            return self.accuracy
        else:
            return self.rows[k]

    def __contains__(self, k: str) -> bool:
        return (k == "accuracy" and self.has_accuracy) or k in self.rows

    def keys(self) -> List[str]:
        ks = self.keys_.copy()
        if self.has_accuracy:
            # sklearn inserts "accuracy" after the per-class rows
            n = len(ks) - 2
            ks.insert(n, "accuracy")
        return ks

    def __len__(self) -> int:
        return len(self.keys())

def _row(p: float, r: float, f: float, s: float) -> Dict[str, float]:
    d = Dict[str, float]()
    d["precision"] = p
    d["recall"] = r
    d["f1-score"] = f
    d["support"] = s
    return d

def _rj(s: str, w: int) -> str:
    return s.rjust(w)

def _ff(v: float, digits: int) -> str:
    return ("%9." + str(digits) + "f") % v

def _sup(v) -> str:
    return str(v).rjust(9)

def classification_report(y_true, y_pred, *, labels = None, target_names = None,
                          sample_weight = None, digits: int = 2,
                          output_dict: Literal[bool] = False, zero_division = "warn"):
    """Text report (identical layout to sklearn) or ClassificationReportDict."""
    y_type, present, a, b = _check_targets(y_true, y_pred)
    labs, given = _labels_arr(labels, present)
    micro_is_accuracy = False
    if y_type == "multiclass" or y_type == "binary":
        if not given:
            micro_is_accuracy = True
        else:
            micro_is_accuracy = True
            for v in present:
                found = False
                for l in labs:
                    if l == v:
                        found = True
                if not found:
                    micro_is_accuracy = False
    names = List[str]()
    if isinstance(target_names, NoneType):
        for l in labs:
            names.append(str(l))
    else:
        for t in target_names:
            names.append(str(t))
        if len(names) != labs.shape[0]:
            if not given:
                raise ValueError(f"Number of classes, {labs.shape[0]}, does not match size of target_names, {len(names)}. Try specifying the labels parameter")
    p, r, f1, s = precision_recall_fscore_support(a, b, labels=labs, average=None,
                                                  sample_weight=sample_weight,
                                                  zero_division=zero_division)
    nrows = min(len(names), labs.shape[0])
    avg_opts = ["micro", "macro", "weighted"]
    if y_type.startswith("multilabel"):
        avg_opts.append("samples")
    headers = ["precision", "recall", "f1-score", "support"]
    total_s = np.sum(s)
    if output_dict:
        rep = ClassificationReportDict()
        for i in range(nrows):
            rep._add(names[i], _row(float(p[i]), float(r[i]), float(f1[i]), float(s[i])))
        for avg in avg_opts:
            ap, ar, af, _ = precision_recall_fscore_support(a, b, labels=labs, average=avg,
                                                            sample_weight=sample_weight,
                                                            zero_division=zero_division)
            if avg == "micro" and micro_is_accuracy:
                rep.has_accuracy = True
                rep.accuracy = ap
            else:
                rep._add(avg + " avg", _row(ap, ar, af, float(total_s)))
        return rep
    else:
        width = len("weighted avg")
        for nm in names:
            if len(nm) > width:
                width = len(nm)
        if digits > width:
            width = digits
        report = " " * width + " "
        for h in headers:
            report += " " + _rj(h, 9)
        report += "\n\n"
        for i in range(nrows):
            report += _rj(names[i], width) + " "
            report += " " + _ff(p[i], digits) + " " + _ff(r[i], digits) + " " + _ff(f1[i], digits)
            report += " " + _sup(s[i]) + "\n"
        report += "\n"
        for avg in avg_opts:
            ap, ar, af, _ = precision_recall_fscore_support(a, b, labels=labs, average=avg,
                                                            sample_weight=sample_weight,
                                                            zero_division=zero_division)
            if avg == "micro" and micro_is_accuracy:
                report += _rj("accuracy", width) + " "
                report += " " + " " * 9 + " " + " " * 9
                report += " " + _ff(af, digits) + " " + _sup(total_s) + "\n"
            else:
                report += _rj(avg + " avg", width) + " "
                report += " " + _ff(ap, digits) + " " + _ff(ar, digits) + " " + _ff(af, digits)
                report += " " + _sup(total_s) + "\n"
        return report

# ------------------------------------------------------------ agreement

def cohen_kappa_score(y1, y2, *, labels = None, weights = None, sample_weight = None,
                      replace_undefined_by: float = np.nan) -> float:
    try:
        C = confusion_matrix(y1, y2, labels=labels, sample_weight=sample_weight)
    except ValueError as e:
        if "At least one label specified must be in y_true" in str(e):
            raise ValueError("At least one label in `labels` must be present in `y1` (even though `cohen_kappa_score` is otherwise agnostic to the order of `y1` and `y2`).")
        raise e
    conf = C.astype(float)
    k = conf.shape[0]
    sum0 = np.sum(conf, axis=0)
    sum1 = np.sum(conf, axis=1)
    den0 = float(np.sum(sum0))
    if den0 == 0:
        return replace_undefined_by
    expected = np.outer(sum0, sum1) / den0
    w = np.ones((k, k))
    if isinstance(weights, NoneType):
        for i in range(k):
            w[i, i] = 0.0
    else:
        for i in range(k):
            for j in range(k):
                d = float(j - i)
                if weights == "linear":
                    w[i, j] = abs(d)
                elif weights == "quadratic":
                    w[i, j] = d * d
                else:
                    raise ValueError("weights must be None, 'linear' or 'quadratic'")
    num = float(np.sum(w * conf))
    den = float(np.sum(w * expected))
    if den == 0:
        return replace_undefined_by
    return float(1 - num / den)

def matthews_corrcoef(y_true, y_pred, *, sample_weight = None) -> float:
    y_type, present, a, b = _check_targets(y_true, y_pred)
    if static.len(a.shape) != 1:
        compile_error("multilabel-indicator is not supported")
    if y_type != "binary" and y_type != "multiclass":
        raise ValueError(f"{y_type} is not supported")
    C = confusion_matrix(_encode(a, present), _encode(b, present), sample_weight=sample_weight).astype(float)
    t_sum = np.sum(C, axis=1)
    p_sum = np.sum(C, axis=0)
    n_correct = float(np.trace(C))
    n_samples = float(np.sum(p_sum))
    cov_ytyp = n_correct * n_samples - float(np.dot(t_sum, p_sum))
    cov_ypyp = n_samples ** 2 - float(np.dot(p_sum, p_sum))
    cov_ytyt = n_samples ** 2 - float(np.dot(t_sum, t_sum))
    prod = cov_ypyp * cov_ytyt
    if prod == 0:
        return 0.0
    return float(cov_ytyp / np.sqrt(prod))

def balanced_accuracy_score(y_true, y_pred, *, sample_weight = None, adjusted: bool = False) -> float:
    C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight).astype(float)
    k = C.shape[0]
    per = List[float]()
    for i in range(k):
        r = float(np.sum(C[i]))
        v = C[i, i] / r if r != 0 else np.nan
        if not np.isnan(v):
            per.append(v)
    score = float(np.mean(np.array(per)))
    if adjusted:
        n_classes = len(per)
        chance = 1 / n_classes
        score -= chance
        score /= 1 - chance
    return score

def hamming_loss(y_true, y_pred, *, sample_weight = None) -> float:
    y_type, present, a, b = _check_targets(y_true, y_pred)
    sw = _sw_or_none(sample_weight, a.shape[0])
    if static.len(a.shape) == 2:
        wavg = 1.0 if isinstance(sw, NoneType) else float(np.mean(sw))
        nd = 0.0
        for i in range(a.shape[0]):
            w = 1.0 if isinstance(sw, NoneType) else sw[i]
            for j in range(a.shape[1]):
                if a[i, j] != b[i, j]:
                    nd += w
        return nd / (a.shape[0] * a.shape[1] * wavg)
    else:
        d = np.empty(a.shape[0])
        for i in range(a.shape[0]):
            d[i] = 1.0 if a[i] != b[i] else 0.0
        return _average(d, sw)

# ------------------------------------------------------------ probabilistic

_EPS64 = 2.220446049250313e-16

def _proba2d(y_proba):
    P = np.asarray(y_proba, dtype=float)
    if static.len(P.shape) == 1:
        Q = np.empty((P.shape[0], 2))
        for i in range(P.shape[0]):
            Q[i, 0] = 1 - P[i]
            Q[i, 1] = P[i]
        return Q
    else:
        if P.shape[1] == 1:
            Q = np.empty((P.shape[0], 2))
            for i in range(P.shape[0]):
                Q[i, 0] = 1 - P[i, 0]
                Q[i, 1] = P[i, 0]
            return Q
        return P.copy()

def _validate_multiclass_proba(y_true, y_proba, sample_weight, labels):
    yt = np.asarray(y_true)
    if static.len(yt.shape) != 1:
        compile_error("y_true must be 1-D")
    P0 = np.asarray(y_proba, dtype=float)
    if float(np.max(P0)) > 1:
        raise ValueError(f"y_prob contains values greater than 1: {float(np.max(P0))}")
    if float(np.min(P0)) < 0:
        raise ValueError(f"y_prob contains values lower than 0: {float(np.min(P0))}")
    if P0.shape[0] != yt.shape[0]:
        raise ValueError(f"Found input variables with inconsistent numbers of samples: [{P0.shape[0]}, {yt.shape[0]}]")
    if isinstance(labels, NoneType):
        classes = np.unique(yt)
    else:
        L = np.asarray(labels)
        classes = np.unique(L) if L.dtype is yt.dtype else np.unique(L.astype(yt.dtype))
        for v in yt:
            ok = False
            for c in classes:
                if c == v:
                    ok = True
            if not ok:
                raise ValueError(f"y_true contains values not belonging to the passed labels {L.tolist()}.")
    if classes.shape[0] == 1:
        if isinstance(labels, NoneType):
            raise ValueError(f"y_true contains only one label ({classes[0]}). Please provide the list of all expected class labels explicitly through the labels argument.")
        raise ValueError(f"The labels array needs to contain at least two labels, got {classes.tolist()}.")
    from ._base import _label_binarize_1d
    T = _label_binarize_1d(yt, classes)
    P = _proba2d(P0)
    if classes.shape[0] != P.shape[1]:
        if isinstance(labels, NoneType):
            raise ValueError(f"y_true and y_prob contain different number of classes: {classes.shape[0]} vs {P.shape[1]}. Please provide the true labels explicitly through the labels argument. Classes found in y_true: {classes.tolist()}")
        raise ValueError(f"The number of classes in labels is different from that in y_prob. Classes found in labels: {classes.tolist()}")
    return T, P

def _log_loss(T, P, normalize: bool, sw) -> float:
    n, k = P.shape
    loss = np.empty(n)
    for i in range(n):
        s = 0.0
        for j in range(k):
            p = min(max(P[i, j], _EPS64), 1 - _EPS64)
            if T[i, j] != 0.0:
                s += T[i, j] * np.log(p)
        loss[i] = -s
    return _average(loss, sw, normalize)

def log_loss(y_true, y_proba = None, *, normalize: bool = True, sample_weight = None,
             labels = None, y_pred = None) -> float:
    """Cross-entropy of probabilistic predictions (natural log)."""
    if isinstance(y_proba, NoneType):
        return log_loss(y_true, y_pred, normalize=normalize, sample_weight=sample_weight, labels=labels)
    else:
        T, P = _validate_multiclass_proba(y_true, y_proba, sample_weight, labels)
        sw = _sw_or_none(sample_weight, T.shape[0])
        return _log_loss(T, P, normalize, sw)

def _binary_pos_label(yt, pos_label):
    """_check_pos_label_consistency, falling back to the greater class."""
    classes = np.unique(yt)
    if isinstance(pos_label, NoneType):
        if yt.dtype is str:
            return classes[classes.shape[0] - 1]
        ok = classes.shape[0] <= 2
        for c in classes:
            if not (c == 0 or c == 1 or c == -1):
                ok = False
        if classes.shape[0] == 2 and not ((classes[0] == 0 and classes[1] == 1) or (classes[0] == -1 and classes[1] == 1)):
            ok = False
        if not ok:
            return classes[classes.shape[0] - 1]
        return type(classes[0])(1)
    else:
        pl = _cast_label(pos_label, yt.dtype)
        if pl is None:
            raise ValueError(f"pos_label={pos_label} is not a valid label.")
        return pl.__val__()

def _validate_binary_proba(y_true, y_proba, sample_weight, pos_label):
    yt = np.asarray(y_true)
    p = np.asarray(y_proba, dtype=float).reshape(-1)
    if p.shape[0] != yt.shape[0]:
        raise ValueError(f"Found input variables with inconsistent numbers of samples: [{p.shape[0]}, {yt.shape[0]}]")
    from ..utils.multiclass import type_of_target
    t = type_of_target(yt, input_name="y_true")
    if t != "binary":
        raise ValueError(f"The type of the target inferred from y_true is {t} but should be binary according to the shape of y_prob.")
    if float(np.max(p)) > 1:
        raise ValueError(f"y_prob contains values greater than 1: {float(np.max(p))}")
    if float(np.min(p)) < 0:
        raise ValueError(f"y_prob contains values less than 0: {float(np.min(p))}")
    pl = _binary_pos_label(yt, pos_label)
    n = yt.shape[0]
    T = np.zeros((n, 2))
    P = np.empty((n, 2))
    for i in range(n):
        if yt[i] == pl:
            T[i, 1] = 1.0
        else:
            T[i, 0] = 1.0
        P[i, 0] = 1 - p[i]
        P[i, 1] = p[i]
    return T, P

def _brier_tp(y_true, y_proba, sample_weight, pos_label, labels):
    P0 = np.asarray(y_proba, dtype=float)
    if static.len(P0.shape) == 1:
        return _validate_binary_proba(y_true, P0, sample_weight, pos_label)
    else:
        if P0.shape[1] == 1:
            return _validate_binary_proba(y_true, P0[:, 0], sample_weight, pos_label)
        return _validate_multiclass_proba(y_true, P0, sample_weight, labels)

def brier_score_loss(y_true, y_proba, *, sample_weight = None, pos_label = None, labels = None,
                     scale_by_half = "auto") -> float:
    T, P = _brier_tp(y_true, y_proba, sample_weight, pos_label, labels)
    sw = _sw_or_none(sample_weight, T.shape[0])
    d = np.sum((T - P) ** 2, axis=1)
    score = _average(d, sw)
    half = False
    if isinstance(scale_by_half, str):
        half = P.shape[1] < 3
    else:
        half = bool(scale_by_half)
    if half:
        score *= 0.5
    return float(score)

def d2_brier_score(y_true, y_proba, *, sample_weight = None, pos_label = None, labels = None) -> float:
    if np.asarray(y_proba).shape[0] < 2:
        return np.nan
    T, P = _brier_tp(y_true, y_proba, sample_weight, pos_label, labels)
    sw = _sw_or_none(sample_weight, T.shape[0])
    null = np.average(T, axis=0) if isinstance(sw, NoneType) else np.average(T, axis=0, weights=sw)
    num = _average(np.sum((T - P) ** 2, axis=1), sw)
    den = _average(np.sum((T - null[None, :]) ** 2, axis=1), sw)
    return float(1 - num / den)

def d2_log_loss_score(y_true, y_proba = None, *, sample_weight = None, labels = None,
                      y_pred = None) -> float:
    if isinstance(y_proba, NoneType):
        return d2_log_loss_score(y_true, y_pred, sample_weight=sample_weight, labels=labels)
    else:
        if np.asarray(y_proba).shape[0] < 2:
            return np.nan
        T, P = _validate_multiclass_proba(y_true, y_proba, sample_weight, labels)
        sw = _sw_or_none(sample_weight, T.shape[0])
        null = np.average(T, axis=0) if isinstance(sw, NoneType) else np.average(T, axis=0, weights=sw)
        Pn = np.tile(null, (T.shape[0], 1))
        num = _log_loss(T, P, False, sw)
        den = _log_loss(T, Pn, False, sw)
        return float(1 - num / den)

def hinge_loss(y_true, pred_decision, *, labels = None, sample_weight = None) -> float:
    yt = np.asarray(y_true)
    D = np.asarray(pred_decision, dtype=float)
    sw = _sw_or_none(sample_weight, yt.shape[0])
    uniq = np.unique(yt) if isinstance(labels, NoneType) else np.unique(np.asarray(labels))
    n = yt.shape[0]
    margin = np.empty(n)
    if static.len(D.shape) == 2 and uniq.shape[0] > 2:
        if uniq.shape[0] != D.shape[1]:
            if isinstance(labels, NoneType):
                raise ValueError("Please include all labels in y_true or pass labels as third argument")
            raise ValueError(f"The shape of pred_decision is not consistent with the number of classes. With a multiclass target, pred_decision shape must be (n_samples, n_classes), that is ({n}, {uniq.shape[0]}). Got: {D.shape}")
        for i in range(n):
            j = int(np.searchsorted(uniq, yt[i]))
            best = -np.inf
            for c in range(D.shape[1]):
                if c != j and D[i, c] > best:
                    best = D[i, c]
            margin[i] = D[i, j] - best
    else:
        if uniq.shape[0] > 2:
            raise ValueError(f"The shape of pred_decision cannot be 1d arraywith a multiclass target. pred_decision shape must be (n_samples, n_classes), that is ({n}, {uniq.shape[0]}). Got: {D.shape}")
        d = D.reshape(-1)
        # LabelBinarizer(neg_label=-1) fit on y_true: greater class -> +1
        cy = np.unique(yt)
        for i in range(n):
            s = 1.0 if cy.shape[0] == 2 and yt[i] == cy[1] else -1.0
            margin[i] = s * d[i]
    losses = np.maximum(1 - margin, 0.0)
    if isinstance(sw, NoneType):
        return float(np.average(losses))
    return float(np.average(losses, weights=sw))

def class_likelihood_ratios(y_true, y_pred, *, labels = None, sample_weight = None,
                            replace_undefined_by: float = np.nan):
    """(LR+, LR-) of a binary classification."""
    y_type, present, a, b = _check_targets(y_true, y_pred)
    if y_type != "binary":
        raise ValueError(f"class_likelihood_ratios only supports binary classification problems, got targets of type: {y_type}")
    C = confusion_matrix(a, b, labels=labels, sample_weight=sample_weight).astype(float)
    tn = C[0, 0]; fp = C[0, 1]; fn = C[1, 0]; tp = C[1, 1]
    pos_num = tp * (tn + fp)
    pos_den = fp * (tp + fn)
    neg_num = fn * (tn + fp)
    neg_den = tn * (tp + fn)
    lrp = replace_undefined_by
    lrn = replace_undefined_by
    if not (tp + fn == 0 or pos_den == 0):
        lrp = pos_num / pos_den
    if not (tp + fn == 0 or neg_den == 0):
        lrn = neg_num / neg_den
    return float(lrp), float(lrn)
