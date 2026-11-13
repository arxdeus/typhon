# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Ranking metrics: ROC / PR / DET curves, AUC scores, label ranking,
# DCG / NDCG and top-k accuracy. Binary curves work on a 0/1 float target
# built from pos_label, as sklearn's confusion_matrix_at_thresholds.

import numpy as np
from ..utils.multiclass import type_of_target
from ._base import _cast_label, _label_binarize_1d

# ------------------------------------------------------------ helpers

def _pos_label_mask(y_true, pos_label) -> np.ndarray[float, 1]:
    """_check_pos_label_consistency + (y_true == pos_label) as floats."""
    yt = np.asarray(y_true).reshape(-1)
    classes = np.unique(yt)
    n = yt.shape[0]
    out = np.zeros(n)
    if isinstance(pos_label, NoneType):
        if yt.dtype is str:
            compile_error("y_true takes string values: pos_label must be given")
        else:
            ok = True
            for c in classes:
                if not (c == 0 or c == 1 or c == -1):
                    ok = False
            if classes.shape[0] == 2 and classes[0] == -1 and classes[1] == 0:
                ok = False
            if classes.shape[0] > 2:
                ok = False
            if not ok:
                raise ValueError(f"y_true takes value in {{{', '.join(str(c) for c in classes)}}} and pos_label is not specified: either make y_true take value in {{0, 1}} or {{-1, 1}} or pass pos_label explicitly.")
            for i in range(n):
                out[i] = 1.0 if yt[i] == 1 else 0.0
            return out
    else:
        pl = _cast_label(pos_label, yt.dtype)
        if pl is not None:
            v = pl.__val__()
            for i in range(n):
                out[i] = 1.0 if yt[i] == v else 0.0
        return out

def _cm_at_thresholds(y01, y_score, sample_weight):
    """(fps, tps, thresholds) for a 0/1 float target (sklearn order:
    stable descending sort, zero weights dropped)."""
    s0 = np.asarray(y_score, dtype=float).reshape(-1)
    if s0.shape[0] != y01.shape[0]:
        raise ValueError(f"Found input variables with inconsistent numbers of samples: [{y01.shape[0]}, {s0.shape[0]}]")
    for v in s0:
        if not np.isfinite(v):
            raise ValueError("Input y_score contains NaN or infinity.")
    weighted = not isinstance(sample_weight, NoneType)
    yv = y01
    sv = s0
    wv = np.ones(y01.shape[0])
    if not isinstance(sample_weight, NoneType):
        w0 = np.asarray(sample_weight, dtype=float).reshape(-1)
        keep = w0 != 0
        yv = y01[keep]
        sv = s0[keep]
        wv = w0[keep]
    n = sv.shape[0]
    # stable descending sort == argsort(stable) of -score with ties kept in order
    order = np.argsort(-sv, kind="stable")
    ys = yv[order]
    ss = sv[order]
    ws = wv[order]
    idx = List[int]()
    for i in range(n - 1):
        if ss[i + 1] - ss[i] != 0:
            idx.append(i)
    idx.append(n - 1)
    m = len(idx)
    tps = np.empty(m)
    fps = np.empty(m)
    thr = np.empty(m)
    ctp = 0.0
    cfp = 0.0
    j = 0
    for i in range(n):
        ctp += ys[i] * ws[i]
        if weighted:
            cfp += (1 - ys[i]) * ws[i]
        if j < m and idx[j] == i:
            tps[j] = ctp
            fps[j] = cfp if weighted else 1 + float(i) - ctp
            thr[j] = ss[i]
            j += 1
    return fps, tps, thr

def _check_binary_target(y_true, pos_label):
    t = type_of_target(np.asarray(y_true), input_name="y_true")
    if not (t == "binary" or (t == "multiclass" and not isinstance(pos_label, NoneType))):
        raise ValueError(f"{t} format is not supported")

def confusion_matrix_at_thresholds(y_true, y_score, *, pos_label = None, sample_weight = None):
    _check_binary_target(y_true, pos_label)
    fps, tps, thr = _cm_at_thresholds(_pos_label_mask(y_true, pos_label), y_score, sample_weight)
    tns = fps[fps.shape[0] - 1] - fps
    fns = tps[tps.shape[0] - 1] - tps
    return tns, fps, fns, tps, thr

def metric_at_thresholds(y_true, y_score, metric_func, *, sample_weight = None, metric_params = None):
    """metric_func(y_true, y_pred) evaluated at each threshold (scalar-valued
    metric_func; inputs sorted by descending score, zero weights dropped).
    metric_params: a callable taking the weight-less metric (unused keys are
    not supported: pass a closure instead)."""
    yt = np.asarray(y_true).reshape(-1)
    sc = np.asarray(y_score, dtype=float).reshape(-1)
    if yt.shape[0] != sc.shape[0]:
        raise ValueError(f"Found input variables with inconsistent numbers of samples: [{yt.shape[0]}, {sc.shape[0]}]")
    for v in sc:
        if not np.isfinite(v):
            raise ValueError("Input contains NaN or infinity.")
    weighted: Literal[bool] = not isinstance(sample_weight, NoneType)
    w0 = np.ones(sc.shape[0])
    if weighted:
        w0 = np.asarray(sample_weight, dtype=float).reshape(-1)
        keep = w0 != 0
        yt = yt[keep]
        sc = sc[keep]
        w0 = w0[keep]
    order = np.argsort(-sc, kind="stable")
    yt = yt[order]
    sc = sc[order]
    n = sc.shape[0]
    idx = List[int]()
    for i in range(n - 1):
        if sc[i + 1] - sc[i] != 0:
            idx.append(i)
    idx.append(n - 1)
    m = len(idx)
    thr = np.empty(m)
    vals = np.empty(m)
    for t in range(m):
        thr[t] = sc[idx[t]]
        yp = np.empty(n, dtype=int)
        for i in range(n):
            yp[i] = 1 if sc[i] >= thr[t] else 0
        if weighted:
            ws = w0[order]
            vals[t] = float(metric_func(yt, yp, sample_weight=ws))
        else:
            vals[t] = float(metric_func(yt, yp))
    return vals, thr

def _keep_idx(keep: List[bool]):
    out = List[int]()
    for i in range(len(keep)):
        if keep[i]:
            out.append(i)
    return np.array(out, dtype=int)

# ------------------------------------------------------------ curves

def auc(x, y) -> float:
    X = np.asarray(x, dtype=float).reshape(-1)
    Y = np.asarray(y, dtype=float).reshape(-1)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Found input variables with inconsistent numbers of samples: [{X.shape[0]}, {Y.shape[0]}]")
    if X.shape[0] < 2:
        raise ValueError(f"At least 2 points are needed to compute area under curve, but x.shape = {X.shape[0]}")
    direction = 1.0
    dx = np.diff(X)
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1.0
        else:
            raise ValueError(f"x is neither increasing nor decreasing : {X}.")
    n = X.shape[0]
    return direction * float(np.sum(dx * (Y[1:] + Y[:n - 1]) / 2.0))

def _roc_from(fps0, tps0, thr0, drop_intermediate: bool):
    fps = fps0
    tps = tps0
    thr = thr0
    m = fps.shape[0]
    if drop_intermediate and m > 2:
        keep = [True] * m
        for i in range(1, m - 1):
            d2f = fps[i + 1] - 2 * fps[i] + fps[i - 1]
            d2t = tps[i + 1] - 2 * tps[i] + tps[i - 1]
            keep[i] = d2f != 0 or d2t != 0
        k = _keep_idx(keep)
        fps = fps[k]; tps = tps[k]; thr = thr[k]
    tps = np.concatenate((np.array([0.0]), tps))
    fps = np.concatenate((np.array([0.0]), fps))
    thr = np.concatenate((np.array([np.inf]), thr))
    L = fps.shape[0]
    if fps[L - 1] <= 0:
        fpr = np.full(L, np.nan)
    else:
        fpr = fps / fps[L - 1]
    if tps[L - 1] <= 0:
        tpr = np.full(L, np.nan)
    else:
        tpr = tps / tps[L - 1]
    return fpr, tpr, thr

def roc_curve(y_true, y_score, *, pos_label = None, sample_weight = None,
              drop_intermediate: bool = True):
    _check_binary_target(y_true, pos_label)
    fps, tps, thr = _cm_at_thresholds(_pos_label_mask(y_true, pos_label), y_score, sample_weight)
    return _roc_from(fps, tps, thr, drop_intermediate)

def _pr_from(fps0, tps0, thr0, drop_intermediate: bool):
    fps = fps0; tps = tps0; thr = thr0
    m = fps.shape[0]
    if drop_intermediate and m > 2:
        keep = [True] * m
        for i in range(1, m - 1):
            keep[i] = (tps[i] - tps[i - 1]) != 0 or (tps[i + 1] - tps[i]) != 0
        k = _keep_idx(keep)
        fps = fps[k]; tps = tps[k]; thr = thr[k]
    m = fps.shape[0]
    precision = np.empty(m)
    for i in range(m):
        ps = tps[i] + fps[i]
        precision[i] = tps[i] / ps if ps != 0 else 0.0
    if tps[m - 1] == 0:
        recall = np.full(m, 1.0)
    else:
        recall = tps / tps[m - 1]
    return (np.concatenate((precision[::-1], np.array([1.0]))),
            np.concatenate((recall[::-1], np.array([0.0]))), thr[::-1].copy())

def precision_recall_curve(y_true, y_score, *, pos_label = None, sample_weight = None,
                           drop_intermediate: bool = False):
    _check_binary_target(y_true, pos_label)
    fps, tps, thr = _cm_at_thresholds(_pos_label_mask(y_true, pos_label), y_score, sample_weight)
    return _pr_from(fps, tps, thr, drop_intermediate)

def det_curve(y_true, y_score, pos_label = None, sample_weight = None,
              drop_intermediate: bool = False):
    _check_binary_target(y_true, pos_label)
    fps, tps, thr = _cm_at_thresholds(_pos_label_mask(y_true, pos_label), y_score, sample_weight)
    tps = np.concatenate((np.array([0.0]), tps))
    fps = np.concatenate((np.array([0.0]), fps))
    thr = np.concatenate((np.array([np.inf]), thr))
    m = fps.shape[0]
    if drop_intermediate and m > 2:
        keep = [True] * m
        for i in range(1, m - 1):
            keep[i] = (tps[i] - tps[i - 1]) != 0 or (tps[i + 1] - tps[i]) != 0
        k = _keep_idx(keep)
        fps = fps[k]; tps = tps[k]; thr = thr[k]
    if np.unique(np.asarray(y_true)).shape[0] != 2:
        raise ValueError("Only one class is present in y_true. Detection error tradeoff curve is not defined in that case.")
    m = fps.shape[0]
    fns = tps[m - 1] - tps
    p_count = tps[m - 1]
    n_count = fps[m - 1]
    ss = int(np.searchsorted(fps, fps[0], side="right"))
    first = ss - 1 if ss > 0 else 0
    last = int(np.searchsorted(tps, tps[m - 1])) + 1
    a = fps[first:last][::-1] / n_count
    b = fns[first:last][::-1] / p_count
    return a, b, thr[first:last][::-1].copy()

# ------------------------------------------------------------ scores

class _RocAuc:
    max_fpr: Optional[float]
    def __init__(self, max_fpr: Optional[float]):
        self.max_fpr = max_fpr
    def __call__(self, y01, score, sw) -> float:
        n = y01.shape[0]
        has0 = False
        has1 = False
        for i in range(n):
            if y01[i] != 0:
                has1 = True
            else:
                has0 = True
        if not (has0 and has1):
            return np.nan
        fps, tps, thr = _cm_at_thresholds(y01, score, sw)
        fpr, tpr, _ = _roc_from(fps, tps, thr, True)
        mf = self.max_fpr
        if mf is None:
            return auc(fpr, tpr)
        max_fpr = mf.__val__()
        if max_fpr == 1:
            return auc(fpr, tpr)
        if max_fpr <= 0 or max_fpr > 1:
            raise ValueError(f"Expected max_fpr in range (0, 1], got: {max_fpr}")
        stop = int(np.searchsorted(fpr, max_fpr, side="right"))
        xi = np.array([fpr[stop - 1], fpr[stop]])
        yi = np.array([tpr[stop - 1], tpr[stop]])
        t2 = np.append(tpr[:stop], np.array([float(np.interp(max_fpr, xi, yi))]))
        f2 = np.append(fpr[:stop], np.array([max_fpr]))
        partial_auc = auc(f2, t2)
        min_area = 0.5 * max_fpr ** 2
        max_area = max_fpr
        return 0.5 * (1 + (partial_auc - min_area) / (max_area - min_area))

class _AvgPrec:
    def __call__(self, y01, score, sw) -> float:
        fps, tps, thr = _cm_at_thresholds(y01, score, sw)
        p, r, _ = _pr_from(fps, tps, thr, False)
        n = p.shape[0]
        return float(max(0.0, -float(np.sum(np.diff(r) * p[:n - 1]))))

def _average_binary_score(metric, Y, S, average, sample_weight):
    """Y: (n, k) 0/1 float matrix, S: (n, k) scores."""
    if not isinstance(average, NoneType):
        if average not in ("micro", "macro", "weighted", "samples"):
            raise ValueError("average has to be one of (None, 'micro', 'macro', 'weighted', 'samples')")
    n, k = Y.shape
    if isinstance(average, NoneType):
        out = np.empty(k)
        for c in range(k):
            out[c] = metric(np.ascontiguousarray(Y[:, c]), np.ascontiguousarray(S[:, c]), sample_weight)
        return out
    else:
        if average == "micro":
            if isinstance(sample_weight, NoneType):
                return float(metric(Y.reshape(-1), S.reshape(-1), sample_weight))
            else:
                sw = np.repeat(np.asarray(sample_weight, dtype=float), k)
                return float(metric(Y.reshape(-1), S.reshape(-1), sw))
        elif average == "samples":
            sc = np.empty(n)
            for i in range(n):
                sc[i] = metric(np.ascontiguousarray(Y[i]), np.ascontiguousarray(S[i]), None)
            if isinstance(sample_weight, NoneType):
                return float(np.average(sc))
            else:
                aw = np.asarray(sample_weight, dtype=float)
                for i in range(n):
                    if aw[i] == 0:
                        sc[i] = 0.0
                return float(np.average(sc, weights=aw))
        else:
            sc = np.empty(k)
            for c in range(k):
                sc[c] = metric(np.ascontiguousarray(Y[:, c]), np.ascontiguousarray(S[:, c]), sample_weight)
            if average == "weighted":
                aw = np.sum(Y, axis=0)
                if not isinstance(sample_weight, NoneType):
                    aw = np.sum(Y * np.asarray(sample_weight, dtype=float).reshape(-1, 1), axis=0)
                if np.isclose(float(np.sum(aw)), 0.0):
                    return 0.0
                for c in range(k):
                    if aw[c] == 0:
                        sc[c] = 0.0
                return float(np.average(sc, weights=aw))
            return float(np.average(sc))

def _opt_float(v) -> Optional[float]:
    if isinstance(v, NoneType):
        return Optional[float]()
    else:
        return Optional[float](float(v))

def _ovo(metric, yenc, S, k: int, average):
    n_pairs = k * (k - 1) // 2
    pair = np.empty(n_pairs)
    prev = np.empty(n_pairs)
    ix = 0
    for a in range(k):
        for b in range(a + 1, k):
            idx = List[int]()
            for i in range(yenc.shape[0]):
                if yenc[i] == a or yenc[i] == b:
                    idx.append(i)
            ii = np.array(idx, dtype=int)
            prev[ix] = float(len(idx)) / float(yenc.shape[0])
            at = (yenc[ii] == a).astype(float)
            bt = (yenc[ii] == b).astype(float)
            sa = metric(at, np.ascontiguousarray(S[ii, a]), None)
            sb = metric(bt, np.ascontiguousarray(S[ii, b]), None)
            pair[ix] = (sa + sb) / 2
            ix += 1
    if average == "weighted":
        return float(np.average(pair, weights=prev))
    return float(np.average(pair))

def roc_auc_score(y_true, y_score, *, average = "macro", sample_weight = None,
                  max_fpr = None, multi_class: str = "raise", labels = None):
    yt = np.asarray(y_true)
    S = np.asarray(y_score, dtype=float)
    met = _RocAuc(_opt_float(max_fpr))
    if static.len(yt.shape) == 2:
        if static.len(S.shape) != 2:
            compile_error("y_score must be 2-D for multilabel y_true")
        return _average_binary_score(met, yt.astype(float), S, average, sample_weight)
    else:
        y_type = type_of_target(yt, input_name="y_true")
        if static.len(S.shape) == 2:
            if y_type == "multiclass" or (y_type == "binary" and S.shape[1] > 2):
                if not isinstance(max_fpr, NoneType):
                    if float(max_fpr) != 1.0:
                        raise ValueError(f"Partial AUC computation not available in multiclass setting, 'max_fpr' must be set to `None`, received `max_fpr={max_fpr}` instead")
                if multi_class == "raise":
                    raise ValueError("multi_class must be in ('ovo', 'ovr')")
                if not np.allclose(1, np.sum(S, axis=1)):
                    raise ValueError("Target scores need to be probabilities for multiclass roc_auc, i.e. they should sum up to 1.0 over classes")
                if multi_class != "ovo" and multi_class != "ovr":
                    raise ValueError(f"multi_class='{multi_class}' is not supported for multiclass ROC AUC, multi_class must be in ('ovo', 'ovr')")
                if isinstance(labels, NoneType):
                    classes = np.unique(yt)
                    if classes.shape[0] != S.shape[1]:
                        raise ValueError("Number of classes in y_true not equal to the number of columns in 'y_score'")
                else:
                    L = np.asarray(labels)
                    classes = np.unique(L) if L.dtype is yt.dtype else np.unique(L.astype(yt.dtype))
                    if classes.shape[0] != L.shape[0]:
                        raise ValueError("Parameter 'labels' must be unique")
                    if classes.shape[0] != S.shape[1]:
                        raise ValueError(f"Number of given labels, {classes.shape[0]}, not equal to the number of columns in 'y_score', {S.shape[1]}")
                    if np.setdiff1d(yt, classes).shape[0] > 0:
                        raise ValueError("'y_true' contains labels not in parameter 'labels'")
                if multi_class == "ovo":
                    if not isinstance(sample_weight, NoneType):
                        raise ValueError("sample_weight is not supported for multiclass one-vs-one ROC AUC, 'sample_weight' must be None in this case.")
                    if isinstance(average, NoneType):
                        raise NotImplementedError("average=None is not implemented for multi_class='ovo'.")
                    else:
                        if average != "macro" and average != "weighted":
                            raise ValueError("average must be one of ('macro', 'weighted', None) for multiclass problems")
                        yenc = np.searchsorted(classes, yt)
                        return _ovo(met, yenc, S, classes.shape[0], average)
                else:
                    if not isinstance(average, NoneType):
                        if average == "samples":
                            raise ValueError("average must be one of ('micro', 'macro', 'weighted', None) for multiclass problems")
                    return _average_binary_score(met, _label_binarize_1d(yt, classes), S, average, sample_weight)
            S1 = S[:, S.shape[1] - 1].copy() if S.shape[1] == 2 else S[:, 0].copy()
            return roc_auc_score(yt, S1, average=average, sample_weight=sample_weight, max_fpr=max_fpr)
        else:
            if y_type != "binary":
                if y_type == "multiclass":
                    raise ValueError("y_score needs to be of shape (n_samples, n_classes) for a multiclass y_true")
                raise ValueError(f"{y_type} format is not supported")
            classes = np.unique(yt)
            y01 = np.zeros(yt.shape[0])
            for i in range(yt.shape[0]):
                y01[i] = 1.0 if classes.shape[0] == 2 and yt[i] == classes[1] else 0.0
            if classes.shape[0] == 1 and classes[0] == 1 and not (yt.dtype is str):
                y01 = np.ones(yt.shape[0])
            return float(met(y01, S, sample_weight))

def average_precision_score(y_true, y_score, *, average = "macro", pos_label = 1, sample_weight = None):
    yt = np.asarray(y_true)
    S = np.asarray(y_score, dtype=float)
    met = _AvgPrec()
    if static.len(yt.shape) == 2:
        if pos_label != 1:
            raise ValueError("Parameter pos_label is fixed to 1 for multilabel-indicator y_true. Do not set pos_label or set pos_label to 1.")
        return _average_binary_score(met, yt.astype(float), S, average, sample_weight)
    else:
        y_type = type_of_target(yt, input_name="y_true")
        present = np.unique(yt)
        if y_type == "binary":
            pl = _cast_label(pos_label, yt.dtype)
            found = False
            if pl is not None:
                for v in present:
                    if v == pl.__val__():
                        found = True
            if present.shape[0] == 2 and not found:
                raise ValueError(f"pos_label={pos_label} is not a valid label. It should be one of {present}")
            y01 = _pos_label_mask(yt, pos_label)
            return float(met(y01, S.reshape(-1), sample_weight))
        elif y_type == "multiclass":
            if pos_label != 1:
                raise ValueError("Parameter pos_label is fixed to 1 for multiclass y_true. Do not set pos_label or set pos_label to 1.")
            if static.len(S.shape) != 2:
                raise ValueError(f"`y_score` needs to be of shape `(n_samples, n_classes)`, since `y_true` contains multiple classes. Got `y_score.shape={S.shape}`.")
            else:
                if S.shape[1] != present.shape[0]:
                    raise ValueError(f"`y_score` needs to be of shape `(n_samples, n_classes)`, since `y_true` contains multiple classes. Got `y_score.shape={S.shape}`.")
                return _average_binary_score(met, _label_binarize_1d(yt, present), S, average, sample_weight)
        else:
            raise ValueError(f"{y_type} format is not supported")

# ------------------------------------------------------------ label ranking

def _check_ml(y_true, y_score):
    Y = np.asarray(y_true)
    S = np.asarray(y_score, dtype=float)
    if static.len(Y.shape) != 2 or static.len(S.shape) != 2:
        compile_error("multilabel-indicator y_true / 2-D y_score required")
    if Y.shape[0] != S.shape[0] or Y.shape[1] != S.shape[1]:
        raise ValueError("y_true and y_score have different shape")
    return Y, S

def _wavg(v, sample_weight) -> float:
    if isinstance(sample_weight, NoneType):
        return float(np.average(v))
    else:
        return float(np.average(v, weights=np.asarray(sample_weight, dtype=float)))

def label_ranking_average_precision_score(y_true, y_score, *, sample_weight = None) -> float:
    Y, S = _check_ml(y_true, y_score)
    t = type_of_target(Y, input_name="y_true")
    if t != "multilabel-indicator" and t != "binary":
        raise ValueError(f"{t} format is not supported")
    n, L = Y.shape
    out = 0.0
    for i in range(n):
        rel = List[int]()
        for j in range(L):
            if Y[i, j] != 0:
                rel.append(j)
        if len(rel) == 0 or len(rel) == L:
            aux = 1.0
        else:
            # rankdata(-score, 'max'): count of labels with -s <= -s_j, i.e. s >= s_j
            acc = 0.0
            for j in rel:
                rank = 0
                for q in range(L):
                    if S[i, q] >= S[i, j]:
                        rank += 1
                lr = 0
                for q in rel:
                    if S[i, q] >= S[i, j]:
                        lr += 1
                acc += float(lr) / float(rank)
            aux = acc / float(len(rel))
        if not isinstance(sample_weight, NoneType):
            aux = aux * float(np.asarray(sample_weight, dtype=float)[i])
        out += aux
    if isinstance(sample_weight, NoneType):
        out /= n
    else:
        out /= float(np.sum(np.asarray(sample_weight, dtype=float)))
    return float(out)

def coverage_error(y_true, y_score, *, sample_weight = None) -> float:
    Y, S = _check_ml(y_true, y_score)
    t = type_of_target(Y, input_name="y_true")
    if t != "multilabel-indicator":
        raise ValueError(f"{t} format is not supported")
    n, L = Y.shape
    cov = np.zeros(n)
    for i in range(n):
        mn = np.inf
        anyr = False
        for j in range(L):
            if Y[i, j] != 0:
                anyr = True
                if S[i, j] < mn:
                    mn = S[i, j]
        if anyr:
            c = 0
            for j in range(L):
                if S[i, j] >= mn:
                    c += 1
            cov[i] = float(c)
    return _wavg(cov, sample_weight)

def label_ranking_loss(y_true, y_score, *, sample_weight = None) -> float:
    Y, S = _check_ml(y_true, y_score)
    t = type_of_target(Y, input_name="y_true")
    if t != "multilabel-indicator":
        raise ValueError(f"{t} format is not supported")
    n, L = Y.shape
    loss = np.zeros(n)
    for i in range(n):
        npos = 0
        bad = 0.0
        for j in range(L):
            if Y[i, j] != 0:
                npos += 1
                for q in range(L):
                    if Y[i, q] == 0 and S[i, q] >= S[i, j]:
                        bad += 1.0
        if npos == 0 or npos == L:
            loss[i] = 0.0
        else:
            loss[i] = bad / float((L - npos) * npos)
    return _wavg(loss, sample_weight)

# ------------------------------------------------------------ DCG / NDCG

def _discount(m: int, k, log_base: float):
    d = 1 / (np.log(np.arange(m) + 2) / np.log(log_base))
    if not isinstance(k, NoneType):
        for i in range(int(k), m):
            d[i] = 0.0
    return d

def _tie_averaged_dcg(yt, ys, discount_cumsum) -> float:
    u, inv, counts = np.unique(-ys, return_inverse=True, return_counts=True)
    g = counts.shape[0]
    ranked = np.zeros(g)
    for i in range(yt.shape[0]):
        ranked[inv[i]] += yt[i]
    ranked = ranked / counts
    groups = np.cumsum(counts) - 1
    ds = np.empty(g)
    ds[0] = discount_cumsum[groups[0]]
    for i in range(1, g):
        ds[i] = discount_cumsum[groups[i]] - discount_cumsum[groups[i - 1]]
    return float(np.sum(ranked * ds))

def _dcg_sample_scores(Y, S, k, log_base: float, ignore_ties: bool):
    n, m = Y.shape
    disc = _discount(m, k, log_base)
    out = np.empty(n)
    if ignore_ties:
        for i in range(n):
            order = np.argsort(S[i])[::-1]
            out[i] = float(np.dot(disc, Y[i][order]))
    else:
        dc = np.cumsum(disc)
        for i in range(n):
            out[i] = _tie_averaged_dcg(np.ascontiguousarray(Y[i]), np.ascontiguousarray(S[i]), dc)
    return out

def _check_dcg_target(Y):
    t = type_of_target(Y, input_name="y_true")
    if t != "multilabel-indicator" and t != "continuous-multioutput" and t != "multiclass-multioutput":
        raise ValueError(f"Only ('multilabel-indicator', 'continuous-multioutput', 'multiclass-multioutput') formats are supported. Got {t} instead")

def dcg_score(y_true, y_score, *, k = None, log_base: float = 2, sample_weight = None,
              ignore_ties: bool = False) -> float:
    Y = np.asarray(y_true, dtype=float)
    S = np.asarray(y_score, dtype=float)
    _check_dcg_target(np.asarray(y_true))
    return _wavg(_dcg_sample_scores(Y, S, k, float(log_base), ignore_ties), sample_weight)

def ndcg_score(y_true, y_score, *, k = None, sample_weight = None, ignore_ties: bool = False) -> float:
    Y = np.asarray(y_true, dtype=float)
    S = np.asarray(y_score, dtype=float)
    if float(np.min(Y)) < 0:
        raise ValueError("ndcg_score should not be used on negative y_true values.")
    if Y.shape[1] <= 1:
        raise ValueError(f"Computing NDCG is only meaningful when there is more than 1 document. Got {Y.shape[1]} instead.")
    _check_dcg_target(np.asarray(y_true))
    gain = _dcg_sample_scores(Y, S, k, 2.0, ignore_ties)
    norm = _dcg_sample_scores(Y, Y, k, 2.0, True)
    for i in range(gain.shape[0]):
        if norm[i] == 0:
            gain[i] = 0.0
        else:
            gain[i] /= norm[i]
    return _wavg(gain, sample_weight)

# ------------------------------------------------------------ top-k

def top_k_accuracy_score(y_true, y_score, *, k: int = 2, normalize: bool = True,
                         sample_weight = None, labels = None) -> float:
    yt = np.asarray(y_true)
    S = np.asarray(y_score, dtype=float)
    y_type = type_of_target(yt, input_name="y_true")
    if not isinstance(labels, NoneType):
        if y_type == "binary" and len(labels) > 2:
            y_type = "multiclass"
    if y_type != "binary" and y_type != "multiclass":
        raise ValueError(f"y type must be 'binary' or 'multiclass', got '{y_type}' instead.")
    n = yt.shape[0]
    if static.len(S.shape) == 2:
        nsc = S.shape[1]
    else:
        nsc = 2
    if isinstance(labels, NoneType):
        classes = np.unique(yt)
        if classes.shape[0] != nsc:
            raise ValueError(f"Number of classes in 'y_true' ({classes.shape[0]}) not equal to the number of classes in 'y_score' ({nsc}).You can provide a list of all known classes by assigning it to the `labels` parameter.")
    else:
        L = np.asarray(labels)
        classes = np.unique(L) if L.dtype is yt.dtype else np.unique(L.astype(yt.dtype))
        if classes.shape[0] != L.shape[0]:
            raise ValueError("Parameter 'labels' must be unique.")
        if classes.shape[0] != nsc:
            raise ValueError(f"Number of given labels ({classes.shape[0]}) not equal to the number of classes in 'y_score' ({nsc}).")
        if np.setdiff1d(yt, classes).shape[0] > 0:
            raise ValueError("'y_true' contains labels not in parameter 'labels'.")
    enc = np.searchsorted(classes, yt)
    hits = np.zeros(n)
    if y_type == "binary":
        s1 = S.reshape(-1)
        if static.len(S.shape) == 2:
            if S.shape[1] != 1:
                raise ValueError(f"`y_true` is binary while y_score is 2d with {S.shape[1]} classes. If `y_true` does not contain all the labels, `labels` must be provided.")
        if k == 1:
            thr = 0.5 if float(np.min(s1)) >= 0 and float(np.max(s1)) <= 1 else 0.0
            for i in range(n):
                p = 1 if s1[i] > thr else 0
                hits[i] = 1.0 if p == enc[i] else 0.0
        else:
            hits = np.ones(n)
    else:
        if static.len(S.shape) != 2:
            raise ValueError(f"`y_score` needs to be of shape `(n_samples, n_classes)`, since `y_true` contains multiple classes. Got `y_score.shape={S.shape}`.")
        else:
            for i in range(n):
                order = np.argsort(S[i], kind="mergesort")[::-1]
                for q in range(min(k, order.shape[0])):
                    if order[q] == enc[i]:
                        hits[i] = 1.0
    if normalize:
        return _wavg(hits, sample_weight)
    elif isinstance(sample_weight, NoneType):
        return float(np.sum(hits))
    else:
        return float(np.dot(hits, np.asarray(sample_weight, dtype=float)))
