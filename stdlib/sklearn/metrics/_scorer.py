# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Scorer objects: scorer(estimator, X, y[, sample_weight]) -> float with the
# sklearn sign convention (greater is better). get_scorer(name) returns a
# _NamedScorer; make_scorer(metric, response_method=..., greater_is_better=...)
# wraps a metric callable(y_true, y_pred, **kwargs). Named scoring is shared
# with model_selection through model_selection._scorer_shim.

import numpy as np

class _NamedScorer:
    name: str
    def __init__(self, name: str):
        from ..model_selection._scorer_shim import _check_name
        _check_name(name)
        self.name = name
    def __call__(self, estimator, X, y_true) -> float:
        from ..model_selection._scorer_shim import _score_name
        return _score_name(self.name, estimator, X, y_true)
    def __repr__(self) -> str:
        return f"make_scorer({self.name})"

class _Scorer[F, K, RM: Literal[str]]:
    """make_scorer result. RM (static): "predict", "predict_proba",
    "decision_function" or "decision_function,predict_proba"."""
    score_func: F
    sign: float
    kwargs: K

    def __init__(self, score_func: F, sign: float, kwargs: K, RM: Literal[str]):
        self.score_func = score_func
        self.sign = sign
        self.kwargs = kwargs

    def _response(self, est, X):
        if RM == "predict_proba":
            return np.asarray(est.predict_proba(X), dtype=float)
        elif RM == "decision_function":
            return np.asarray(est.decision_function(X), dtype=float)
        elif RM == "decision_function,predict_proba":
            if hasattr(est, "decision_function"):
                return np.asarray(est.decision_function(X), dtype=float)
            else:
                return np.asarray(est.predict_proba(X), dtype=float)
        else:
            compile_error("unknown response_method")

    def __call__(self, estimator, X, y_true, sample_weight = None) -> float:
        if RM == "predict":
            yp = estimator.predict(X)
            return self.sign * float(self.score_func(y_true, yp, **self.kwargs))
        else:
            r = self._response(estimator, X)
            return self.sign * float(self.score_func(y_true, r, **self.kwargs))

    def __repr__(self) -> str:
        s = "make_scorer(" + self.score_func.__name__
        if self.sign < 0:
            s += ", greater_is_better=False"
        if RM != "predict":
            s += f", response_method='{RM}'"
        return s + ")"

def make_scorer(score_func, *, response_method: Literal[str] = "predict",
                greater_is_better: bool = True, **kwargs):
    sign = 1.0 if greater_is_better else -1.0
    return _Scorer[type(score_func), type(kwargs), response_method](score_func, sign, kwargs, response_method)

def get_scorer(scoring):
    if isinstance(scoring, str):
        return _NamedScorer(scoring)
    else:
        return scoring

def get_scorer_names() -> List[str]:
    from ..model_selection._scorer_shim import get_scorer_names as g
    return g()

class _PassthroughScorer:
    def __call__(self, estimator, X, y) -> float:
        return float(estimator.score(X, y))
    def __repr__(self) -> str:
        return "_PassthroughScorer()"

def check_scoring(estimator = None, scoring = None, *, allow_none: bool = False,
                  raise_exc: bool = True):
    """None -> estimator.score passthrough; str -> named scorer; callable ->
    itself."""
    if isinstance(scoring, NoneType):
        if hasattr(estimator, "score"):
            return _PassthroughScorer()
        else:
            compile_error("If no scoring is specified, the estimator passed should have a 'score' method.")
    elif isinstance(scoring, str):
        return _NamedScorer(scoring)
    else:
        return scoring
