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

def _is_param(k: str) -> bool:
    """Public constructor parameter: no leading/trailing underscore (fitted
    attributes end with '_' in sklearn)."""
    return not k.startswith("_") and not k.endswith("_")

def _param_repr(v) -> str:
    if isinstance(v, str):
        return repr(v)
    elif isinstance(v, NoneType):
        return "None"
    elif isinstance(v, Optional):
        if v is None:
            return "None"
        return _param_repr(v.__val__())
    elif isinstance(v, List):
        out = List[str]()
        for x in v:
            if isinstance(x, int) or isinstance(x, float) or isinstance(x, str) or isinstance(x, bool):
                out.append(_param_repr(x))
            else:
                return "[...]"
        return "[" + ", ".join(out) + "]"
    elif isinstance(v, Tuple):
        if static.len(v) == 2 and isinstance(v[0], str):
            return "(" + _param_repr(v[0]) + ", " + _param_repr(v[1]) + ")"
        out = List[str]()
        for x in v:
            out.append(_param_repr(x))
        return "[" + ", ".join(out) + "]"
    elif hasattr(v, "get_params") and hasattr(v, "fit"):
        return estimator_repr(v)
    else:
        return str(v)

def get_params(estimator, deep: bool = True) -> Dict[str, str]:
    """Parameter name -> repr string (Typhon has no heterogeneous dicts;
    values are the reprs sklearn prints)."""
    d = Dict[str, str]()
    for k, v in static.vars(estimator):
        if _is_param(k):
            d[k] = _param_repr(v)
    return d

def _default_instance(estimator):
    """Estimator built with all-default constructor arguments, or None-like
    marker via the second return value."""
    return (type(estimator)(), True)

def estimator_repr(estimator, defaults: Literal[bool] = True) -> str:
    """ClassName(param=value, ...) listing only the parameters that differ
    from the constructor defaults (like sklearn's print_changed_only)."""
    parts = List[str]()
    ok = defaults
    dflt = _default_instance(estimator)[0] if defaults else estimator
    names = List[str]()
    reps = List[str]()
    for k, v in static.vars(estimator):
        if _is_param(k):
            names.append(k)
            reps.append(_param_repr(v))
    dreps = List[str]()
    for k, v in static.vars(dflt):
        if _is_param(k):
            dreps.append(_param_repr(v))
    for i in range(len(names)):
        if (not ok and reps[i] != "None" and reps[i] != "False") or (ok and reps[i] != dreps[i]):
            parts.append(names[i] + "=" + reps[i])
    nm = estimator.__class__.__name__.lstrip("_")
    br = nm.find("[")
    if br >= 0:
        nm = nm[:br]
    return nm + "(" + ", ".join(parts) + ")"

class BaseEstimator:
    """Mixin base: inherit to get get_params()/set_params()/__repr__ and
    __sklearn_clone__()."""
    def get_params(self, deep: bool = True) -> Dict[str, str]:
        return get_params(self, deep)

    def set_params(self, **params):
        for k, v in static.vars(self):
            if hasattr(params, k):
                setattr(self, k, type(v)(getattr(params, k)))
        return self

    def __sklearn_clone__(self):
        return _shallow_copy(self)

    def __repr__(self) -> str:
        return estimator_repr(self)

def _r2(y_true, y_pred) -> float:
    from .metrics import r2_score
    return float(r2_score(y_true, y_pred))

class ClassifierMixin:
    def __repr__(self) -> str:
        return estimator_repr(self)

    _estimator_type: ClassVar[str] = "classifier"

    def score(self, X, y, sample_weight = None):
        if isinstance(sample_weight, NoneType):
            return _accuracy(y, self.predict(X))
        else:
            from .metrics import accuracy_score
            return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

class RegressorMixin:
    def __repr__(self) -> str:
        return estimator_repr(self)

    _estimator_type: ClassVar[str] = "regressor"

    def score(self, X, y, sample_weight = None):
        from .metrics import r2_score
        return float(r2_score(y, self.predict(X), sample_weight=sample_weight))

class TransformerMixin:
    def __repr__(self) -> str:
        return estimator_repr(self)

    def fit_transform(self, X, y = None):
        return self.fit(X, y).transform(X)

class ClusterMixin:
    def __repr__(self) -> str:
        return estimator_repr(self)

    _estimator_type: ClassVar[str] = "clusterer"

    def fit_predict(self, X, y = None):
        self.fit(X)
        return self.labels_

class BiclusterMixin:
    pass

class OutlierMixin:
    def __repr__(self) -> str:
        return estimator_repr(self)

    _estimator_type: ClassVar[str] = "outlier_detector"

    def fit_predict(self, X, y = None):
        return self.fit(X).predict(X)

class DensityMixin:
    _estimator_type: ClassVar[str] = "DensityEstimator"

    def score(self, X, y = None):
        return 0.0

class MetaEstimatorMixin:
    pass

class MultiOutputMixin:
    pass

class OneToOneFeatureMixin:
    def get_feature_names_out(self, input_features = None) -> List[str]:
        if isinstance(input_features, NoneType):
            return [f"x{i}" for i in range(self.n_features_in_)]
        else:
            return [str(f) for f in input_features]

class ClassNamePrefixFeaturesOutMixin:
    def get_feature_names_out(self, input_features = None) -> List[str]:
        nm = self.__class__.__name__.lower()
        return [f"{nm}{i}" for i in range(self._n_features_out)]

def _estimator_type_of(estimator) -> str:
    if hasattr(estimator, "_estimator_type"):
        return str(estimator._estimator_type)
    else:
        return ""

def is_classifier(estimator) -> bool:
    t = _estimator_type_of(estimator)
    if t != "":
        return t == "classifier"
    return hasattr(estimator, "predict_proba") or hasattr(estimator, "decision_function") and hasattr(estimator, "classes_")

def is_regressor(estimator) -> bool:
    return _estimator_type_of(estimator) == "regressor"

def is_clusterer(estimator) -> bool:
    return _estimator_type_of(estimator) == "clusterer"

def is_outlier_detector(estimator) -> bool:
    return _estimator_type_of(estimator) == "outlier_detector"
