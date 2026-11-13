# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# String / callable scoring used by model_selection (sklearn.metrics._scorer
# subset). A scorer is evaluated as score(estimator, X, y) -> float, with the
# sign convention of sklearn (neg_* = negated loss).

import numpy as np

_SCORERS = ["accuracy", "balanced_accuracy", "f1", "f1_macro", "f1_micro", "f1_weighted",
            "precision", "precision_macro", "precision_micro", "precision_weighted",
            "recall", "recall_macro", "recall_micro", "recall_weighted", "jaccard",
            "roc_auc", "average_precision", "neg_log_loss", "r2", "explained_variance",
            "max_error", "neg_mean_squared_error", "neg_root_mean_squared_error",
            "neg_mean_absolute_error", "neg_median_absolute_error", "matthews_corrcoef"]

def _check_name(name: str):
    if name not in _SCORERS:
        raise ValueError(f"{name!r} is not a valid scoring value. Use sklearn.metrics.get_scorer_names() to get valid options.")

def _avg_kind(name: str, prefix: str) -> str:
    if name == prefix:
        return "binary"
    return name[len(prefix) + 1:]

def _pos_score(est, X):
    # binary positive-class continuous score: decision_function, else proba[:, 1]
    if hasattr(est, "decision_function"):
        d = np.asarray(est.decision_function(X), dtype=float)
        if static.len(d.shape) == 1:
            return d
        else:
            return np.ascontiguousarray(d[:, d.shape[1] - 1])
    else:
        p = np.asarray(est.predict_proba(X), dtype=float)
        return np.ascontiguousarray(p[:, 1])

def _score_name(name: str, est, X, y) -> float:
    from ..metrics import roc_auc_score, average_precision_score, log_loss
    if not hasattr(est, "decision_function") and not hasattr(est, "predict_proba"):
        if name == "roc_auc" or name == "average_precision" or name == "neg_log_loss":
            raise ValueError(f"scoring={name!r} needs an estimator with decision_function or predict_proba")
        return _metric_pred(name, y, est.predict(X))
    if name == "roc_auc":
        return float(roc_auc_score(y, _pos_score(est, X)))
    if name == "average_precision":
        return float(average_precision_score(y, _pos_score(est, X)))
    if name == "neg_log_loss":
        if hasattr(est, "predict_proba"):
            return -float(log_loss(y, est.predict_proba(X), labels=est.classes_))
        else:
            raise ValueError("neg_log_loss needs an estimator with predict_proba")
    return _metric_pred(name, y, est.predict(X))

def _metric_pred(name: str, y, yp) -> float:
    # generic only over the label/prediction types (shared by all estimators)
    from ..metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score,
                           recall_score, jaccard_score, r2_score, explained_variance_score,
                           max_error, mean_squared_error, root_mean_squared_error,
                           mean_absolute_error, median_absolute_error, matthews_corrcoef)
    if name == "accuracy":
        return float(accuracy_score(y, yp))
    if name == "balanced_accuracy":
        return float(balanced_accuracy_score(y, yp))
    if name == "matthews_corrcoef":
        return float(matthews_corrcoef(y, yp))
    if name.startswith("f1"):
        return float(f1_score(y, yp, average=_avg_kind(name, "f1")))
    if name.startswith("precision"):
        return float(precision_score(y, yp, average=_avg_kind(name, "precision")))
    if name.startswith("recall"):
        return float(recall_score(y, yp, average=_avg_kind(name, "recall")))
    if name == "jaccard":
        return float(jaccard_score(y, yp))
    yt = np.asarray(y, dtype=float)
    yf = np.asarray(yp, dtype=float)
    if name == "r2":
        return float(r2_score(yt, yf))
    if name == "explained_variance":
        return float(explained_variance_score(yt, yf))
    if name == "max_error":
        return -float(max_error(yt, yf))
    if name == "neg_mean_squared_error":
        return -float(mean_squared_error(yt, yf))
    if name == "neg_root_mean_squared_error":
        return -float(root_mean_squared_error(yt, yf))
    if name == "neg_mean_absolute_error":
        return -float(mean_absolute_error(yt, yf))
    if name == "neg_median_absolute_error":
        return -float(median_absolute_error(yt, yf))
    _check_name(name)
    return 0.0

def _score(scoring, est, X, y) -> float:
    """scoring: None (estimator.score), a scorer name, or callable(est, X, y)."""
    if scoring is None:
        return float(est.score(X, y))
    elif isinstance(scoring, str):
        return _score_name(scoring, est, X, y)
    elif isinstance(scoring, Optional[str]):
        if scoring is None:
            return float(est.score(X, y))
        return _score_name(scoring.__val__(), est, X, y)
    else:
        return float(scoring(est, X, y))

def _metric_names(scoring) -> List[str]:
    """Names of the metrics of a (possibly multi-metric) scoring parameter;
    single-metric -> ["score"]."""
    if isinstance(scoring, List[str]) or isinstance(scoring, Tuple):
        out = List[str]()
        for s in scoring:
            out.append(s)
        return out
    elif isinstance(scoring, Dict):
        return [k for k in scoring.keys()]
    else:
        return ["score"]

def _is_multi(scoring) -> Literal[bool]:
    return isinstance(scoring, List[str]) or isinstance(scoring, Tuple) or isinstance(scoring, Dict)

def _scores_all(scoring, est, X, y) -> List[float]:
    if isinstance(scoring, List[str]) or isinstance(scoring, Tuple):
        out = List[float]()
        for s in scoring:
            out.append(_score_name(s, est, X, y))
        return out
    elif isinstance(scoring, Dict):
        out = List[float]()
        for v in scoring.values():
            out.append(_score(v, est, X, y))
        return out
    else:
        return [_score(scoring, est, X, y)]

def get_scorer_names() -> List[str]:
    return sorted(_SCORERS)
