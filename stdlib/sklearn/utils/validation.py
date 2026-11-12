# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# sklearn.utils.validation. Typhon arrays are statically typed, so
# `dtype` arguments are types (float, int, NoneType = keep) and the
# number of dimensions is a compile-time property.

import numpy as np
from numpy.random.legacy import RandomState, check_random_state

def _is_float_t(T: type) -> Literal[bool]:
    return T is float or T is float32 or T is float16

def _is_int_t(T: type) -> Literal[bool]:
    return isinstance(T, Int) or isinstance(T, UInt)

def _cls_name(est) -> str:
    # generic instantiations print as Name[args]; sklearn shows Name
    n = est.__class__.__name__
    i = n.find("[")
    return n if i < 0 else n[:i]

class NotFittedError(ValueError):
    def __init__(self, msg: str = ""):
        super().__init__(msg)

def _num_samples(x) -> int:
    if isinstance(x, NoneType):
        return 0
    elif isinstance(x, np.ndarray):
        if static.len(x.shape) == 0:
            raise TypeError("Input should have at least 1 dimension i.e. satisfy `len(x.shape) > 0`, got scalar `array(" + str(x) + ")` instead.")
        return x.shape[0]
    elif hasattr(x, "shape"):
        return x.shape[0]
    else:
        return len(x)

def _num_features(X) -> int:
    a = np.asarray(X)
    if static.len(a.shape) < 2:
        raise TypeError("X does not contain any features")
    return a.shape[1]

def check_consistent_length(*arrays):
    """Raise ValueError when the (non-None) arrays differ in first dimension."""
    lengths = List[int]()
    for a in arrays:
        if not isinstance(a, NoneType):
            if isinstance(a, Optional):
                if a is not None:
                    lengths.append(_num_samples(a.__val__()))
            else:
                lengths.append(_num_samples(a))
    for l in lengths:
        if l != lengths[0]:
            raise ValueError("Found input variables with inconsistent numbers of samples: " + str(lengths))

def _assert_all_finite(X, allow_nan: bool = False, input_name: str = ""):
    a = np.asarray(X)
    if not _is_float_t(a.dtype):
        return
    has_inf = False
    has_nan = False
    for v in a.ravel():
        if np.isnan(v):
            has_nan = True
        elif np.isinf(v):
            has_inf = True
    if has_inf or (has_nan and not allow_nan):
        pad = input_name + " " if input_name != "" else ""
        if has_nan and not allow_nan:
            raise ValueError(f"Input {pad}contains NaN.")
        raise ValueError(f"Input {pad}contains infinity or a value too large for dtype('float64').")

def assert_all_finite(X, *, allow_nan: bool = False, estimator_name = None, input_name: str = ""):
    _assert_all_finite(X, allow_nan, input_name)

def check_array(array, accept_sparse = False, *, accept_large_sparse: bool = True,
                dtype: type = float, order = None, copy: bool = False,
                force_writeable: bool = False, ensure_all_finite = True,
                ensure_non_negative: bool = False, ensure_2d: bool = True,
                allow_nd: bool = False, ensure_min_samples: int = 1,
                ensure_min_features: int = 1, estimator = None, input_name: str = ""):
    """Input validation on an array. dtype=NoneType keeps the input dtype;
    the default converts to float64 ("numeric" in sklearn)."""
    if dtype is NoneType:
        a = np.array(array, copy=copy) if copy else np.asarray(array)
        return _check_array_post(a, ensure_all_finite, ensure_non_negative, ensure_2d,
                                 allow_nd, ensure_min_samples, ensure_min_features, input_name)
    else:
        a = np.array(array, dtype=dtype, copy=True) if copy else np.asarray(array, dtype=dtype)
        return _check_array_post(a, ensure_all_finite, ensure_non_negative, ensure_2d,
                                 allow_nd, ensure_min_samples, ensure_min_features, input_name)

def _check_array_post(a, ensure_all_finite, ensure_non_negative: bool, ensure_2d: bool,
                      allow_nd: bool, ensure_min_samples: int, ensure_min_features: int,
                      input_name: str):
    nd: Literal[int] = static.len(a.shape)
    if ensure_2d and nd != 2:
        if nd == 0:
            raise ValueError("Expected 2D array, got scalar array instead:\narray=" + str(a) +
                             ".\nReshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.")
        if nd == 1:
            raise ValueError("Expected 2D array, got 1D array instead:\narray=" + str(a) +
                             ".\nReshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.")
    if not allow_nd and nd >= 3:
        raise ValueError(f"Found array with dim {nd}. None expected <= 2.")
    if isinstance(ensure_all_finite, str):
        if ensure_all_finite == "allow-nan":
            _assert_all_finite(a, True, input_name)
    elif ensure_all_finite:
        _assert_all_finite(a, False, input_name)
    if nd > 0 and ensure_min_samples > 0:
        n = a.shape[0]
        if n < ensure_min_samples:
            raise ValueError(f"Found array with {n} sample(s) (shape={a.shape}) while a minimum of {ensure_min_samples} is required.")
    if nd == 2 and ensure_min_features > 0:
        f = a.shape[1]
        if f < ensure_min_features:
            raise ValueError(f"Found array with {f} feature(s) (shape={a.shape}) while a minimum of {ensure_min_features} is required.")
    if ensure_non_negative:
        if a.size > 0 and np.min(a) < 0:
            raise ValueError("Negative values in data passed to " + input_name)
    return a

def as_float_array(X, *, copy: bool = True, ensure_all_finite = True):
    return check_array(X, dtype=float, copy=copy, ensure_all_finite=ensure_all_finite, ensure_2d=False)

def column_or_1d(y, *, dtype: type = NoneType, input_name: str = "y", warn: bool = False, device = None):
    """Ravel a column vector or 1d array, else raise ValueError."""
    a = np.asarray(y) if dtype is NoneType else np.asarray(y, dtype=dtype)
    nd: Literal[int] = static.len(a.shape)
    if nd == 1:
        return np.ascontiguousarray(a)
    elif nd == 2:
        if a.shape[1] == 1:
            return np.ascontiguousarray(a.reshape(-1))
        raise ValueError(f"y should be a 1d array, got an array of shape {a.shape} instead.")
    else:
        raise ValueError(f"y should be a 1d array, got an array of shape {a.shape} instead.")
        return np.ascontiguousarray(a.reshape(-1))

def check_X_y(X, y, accept_sparse = False, *, accept_large_sparse: bool = True,
              dtype: type = float, order = None, copy: bool = False,
              force_writeable: bool = False, ensure_all_finite = True,
              ensure_2d: bool = True, allow_nd: bool = False,
              multi_output: Literal[bool] = False, ensure_min_samples: int = 1,
              ensure_min_features: int = 1, y_numeric: Literal[bool] = False,
              estimator = None):
    if isinstance(y, NoneType):
        raise ValueError("estimator requires y to be passed, but the target y is None")
    Xa = check_array(X, dtype=dtype, copy=copy, ensure_all_finite=ensure_all_finite,
                     ensure_2d=ensure_2d, allow_nd=allow_nd, ensure_min_samples=ensure_min_samples,
                     ensure_min_features=ensure_min_features, input_name="X")
    ya = _check_y(y, multi_output=multi_output, y_numeric=y_numeric)
    check_consistent_length(Xa, ya)
    return Xa, ya

def _check_y(y, multi_output: Literal[bool] = False, y_numeric: Literal[bool] = False):
    if multi_output:
        a = check_array(y, dtype=NoneType, ensure_2d=False, input_name="y")
        if y_numeric:
            return np.asarray(a, dtype=float)
        return a
    else:
        a = column_or_1d(y, warn=True)
        _assert_all_finite(a, False, "y")
        if y_numeric:
            return np.asarray(a, dtype=float)
        return a

def _is_fitted(estimator, attributes = None) -> bool:
    if not isinstance(attributes, NoneType):
        return True
    if hasattr(estimator, "__sklearn_is_fitted__"):
        return estimator.__sklearn_is_fitted__()
    if hasattr(estimator, "_fitted"):
        return bool(estimator._fitted)
    # Typhon estimators declare their fitted attributes statically and
    # initialize them empty: an estimator is fitted when any trailing
    # underscore attribute holds data.
    any_attr = False
    fitted = False
    for k, v in static.vars(estimator):
        if k.endswith("_") and not k.startswith("__"):
            any_attr = True
            if _attr_set(v):
                fitted = True
    return fitted or not any_attr

def _attr_set(v) -> bool:
    if isinstance(v, NoneType):
        return False
    elif isinstance(v, Optional):
        return v is not None
    elif isinstance(v, np.ndarray):
        return v.size > 0
    elif isinstance(v, List) or isinstance(v, Dict) or isinstance(v, str):
        return len(v) > 0
    elif isinstance(v, int) or isinstance(v, float):
        return v != 0
    else:
        return True

def check_is_fitted(estimator, attributes = None, *, msg = None, all_or_any = None):
    if not hasattr(estimator, "fit"):
        raise TypeError(f"{_cls_name(estimator)} is not an estimator instance.")
    if not _is_fitted(estimator, attributes):
        name = _cls_name(estimator)
        if isinstance(msg, NoneType):
            raise NotFittedError(f"This {name} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        else:
            raise NotFittedError(str(msg).replace("%(name)s", name))

def check_scalar(x, name: str, target_type: type, *, min_val = None, max_val = None,
                 include_boundaries: str = "both"):
    """Validate scalar type and bounds (target_type is a single type)."""
    if not isinstance(x, target_type):
        raise TypeError(f"{name} must be an instance of {target_type.__name__}, not {x.__class__.__name__}.")
    if include_boundaries not in ("left", "right", "both", "neither"):
        raise ValueError(f"Unknown value for `include_boundaries`: {repr(include_boundaries)}. Possible values are: ('left', 'right', 'both', 'neither').")
    if isinstance(max_val, NoneType) and include_boundaries == "right":
        raise ValueError("`include_boundaries`='right' without specifying explicitly `max_val` is inconsistent.")
    if isinstance(min_val, NoneType) and include_boundaries == "left":
        raise ValueError("`include_boundaries`='left' without specifying explicitly `min_val` is inconsistent.")
    if not isinstance(min_val, NoneType):
        lo_incl = include_boundaries in ("left", "both")
        if (x < min_val) if lo_incl else (x <= min_val):
            raise ValueError(f"{name} == {x}, must be {'>=' if lo_incl else '>'} {min_val}.")
    if not isinstance(max_val, NoneType):
        hi_incl = include_boundaries in ("right", "both")
        if (x > max_val) if hi_incl else (x >= max_val):
            raise ValueError(f"{name} == {x}, must be {'<=' if hi_incl else '<'} {max_val}.")
    return x

def check_non_negative(X, whom: str):
    a = np.asarray(X)
    if a.size > 0 and np.min(a) < 0:
        raise ValueError(f"Negative values in data passed to {whom}")

def check_symmetric(array, *, tol: float = 1e-10, raise_warning: bool = True,
                    raise_exception: bool = False):
    a = np.asarray(array, dtype=float)
    if static.len(a.shape) != 2:
        compile_error("array must be 2-dimensional and square")
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"array must be 2-dimensional and square. shape = {a.shape}")
    if not np.allclose(a, a.T, atol=tol):
        if raise_exception:
            raise ValueError("Array must be symmetric")
        return 0.5 * (a + a.T)
    return a

def _check_sample_weight(sample_weight, X, *, dtype: type = float, ensure_non_negative: bool = False,
                         copy: bool = False, allow_all_zero_weights: bool = False):
    n = _num_samples(X)
    if isinstance(sample_weight, NoneType):
        return np.ones(n, dtype=dtype)
    elif isinstance(sample_weight, int) or isinstance(sample_weight, float):
        return np.full(n, dtype(sample_weight), dtype=dtype)
    elif isinstance(sample_weight, Optional):
        if sample_weight is None:
            return np.ones(n, dtype=dtype)
        return _check_sample_weight(sample_weight.__val__(), X, dtype=dtype,
                                    ensure_non_negative=ensure_non_negative,
                                    allow_all_zero_weights=allow_all_zero_weights)
    else:
        sw = np.array(sample_weight, dtype=dtype) if copy else np.asarray(sample_weight, dtype=dtype)
        if static.len(sw.shape) != 1:
            compile_error("Sample weights must be 1D array or scalar")
        if sw.shape[0] != n:
            raise ValueError(f"sample_weight.shape == ({sw.shape[0]},), expected ({n},)!")
        if not allow_all_zero_weights:
            allz = True
            for v in sw:
                if v != 0:
                    allz = False
            if allz:
                raise ValueError("Sample weights must contain at least one non-zero number.")
        if ensure_non_negative:
            check_non_negative(sw, "`sample_weight`")
        return sw

def has_fit_parameter(estimator, parameter: Literal[str]) -> bool:
    """True when estimator.fit accepts keyword `parameter` (static check)."""
    if hasattr(estimator, "fit"):
        for f in static.function.overloads(type(estimator), "fit"):
            for i in static.range(static.len(static.function.args(f))):
                if static.function.args(f)[i] == parameter:
                    return True
    return False

def _make_indexable(x):
    if isinstance(x, NoneType):
        return x
    elif isinstance(x, np.ndarray):
        return x
    elif hasattr(x, "tocsr"):
        return x.tocsr()
    else:
        return np.asarray(x)

def indexable(*iterables):
    """Make arrays indexable for cross-validation (lists -> ndarrays)."""
    res = tuple(_make_indexable(x) for x in iterables)
    check_consistent_length(*res)
    return res
