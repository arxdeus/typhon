# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Estimator protocol. Typhon is statically typed, so estimators are plain
# classes used through generics (duck typing at compile time):
#   fit(X, y) -> self, predict(X), transform(X), fit_transform(X, y), score(X, y)
# clone() and set_params() work on ANY class through static field
# introspection; estimators that hold other estimators (Pipeline) define
# _sk_clone() to clone their children too.

import numpy as np

def _shallow_copy(est: T, T: type) -> T:
    n = T.__new__()
    for k, v in static.vars(est):
        setattr(n, k, v)
    return n

def clone(estimator, safe: bool = True):
    """Return an unfitted-equivalent copy of `estimator` (same parameters).

    Fields are copied by reference; every ported fit() rebinds its fitted
    arrays, so the clone and the original never share mutated state."""
    if isinstance(estimator, Union):
        compile_error("clone() of a Union value is not supported")
    elif hasattr(estimator, "_sk_clone"):
        return estimator._sk_clone()
    else:
        return _shallow_copy(estimator)

def set_params(estimator, **params):
    """Set public parameters by name (static check: unknown names are ignored
    at compile time only if they do not exist on the class)."""
    for k, v in static.vars(estimator):
        if hasattr(params, k):
            setattr(estimator, k, type(v)(getattr(params, k)))
    return estimator

def get_param(estimator, name: Literal[str]):
    return getattr(estimator, name)

def _accuracy(y_true, y_pred) -> float:
    a = np.asarray(y_true)
    b = np.asarray(y_pred)
    if a.shape[0] != b.shape[0]:
        raise ValueError("Found input variables with inconsistent numbers of samples")
    n = a.shape[0]
    c = 0
    for i in range(n):
        if a[i] == b[i]:
            c += 1
    return float(c) / float(n) if n > 0 else 0.0

class BaseEstimator:
    """Mixin base: inherit to get set_params() and __sklearn_clone__()."""
    def set_params(self, **params):
        for k, v in static.vars(self):
            if hasattr(params, k):
                setattr(self, k, type(v)(getattr(params, k)))
        return self

    def __sklearn_clone__(self):
        return _shallow_copy(self)

class ClassifierMixin:
    _estimator_type: Literal[str] = "classifier"

    def score(self, X, y, sample_weight = None):
        return _accuracy(y, self.predict(X))

class TransformerMixin:
    def fit_transform(self, X, y = None):
        return self.fit(X, y).transform(X)

def is_classifier(estimator) -> bool:
    return hasattr(estimator, "predict")
