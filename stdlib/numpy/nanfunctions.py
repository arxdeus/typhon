# nanmax, nanmin, nanargmax, nanargmin implemented by masking NaNs.

import util

from .ndarray import ndarray
from .routines import asarray, where
from .ndmath import isnan
from .reductions import max as _max, min as _min, argmax as _argmax, \
                        argmin as _argmin, all as _all

def _is_float_dtype(T: type):
    return T is float or T is float32 or T is float16 or T is complex or T is complex64

def _fill(a, v):
    T = a.dtype
    return where(isnan(a), util.cast(v, T), a)

def _restore_nan(res, mask):
    if isinstance(res, ndarray):
        return where(mask, util.nan(res.dtype), res)
    else:
        return util.nan(type(res)) if mask else res

def nanmax(a, axis=None, out=None, keepdims: Literal[bool] = False):
    a = asarray(a)
    if _is_float_dtype(a.dtype):
        filled = _fill(a, -util.inf(a.dtype))
        res = _max(filled, axis=axis, keepdims=keepdims)
        allnan = _all(isnan(a), axis=axis, keepdims=keepdims)
        r = _restore_nan(res, allnan)
    else:
        r = _max(a, axis=axis, keepdims=keepdims)
    if out is not None:
        out[...] = r
        return out
    return r

def nanmin(a, axis=None, out=None, keepdims: Literal[bool] = False):
    a = asarray(a)
    if _is_float_dtype(a.dtype):
        filled = _fill(a, util.inf(a.dtype))
        res = _min(filled, axis=axis, keepdims=keepdims)
        allnan = _all(isnan(a), axis=axis, keepdims=keepdims)
        r = _restore_nan(res, allnan)
    else:
        r = _min(a, axis=axis, keepdims=keepdims)
    if out is not None:
        out[...] = r
        return out
    return r

def _any_true(x) -> bool:
    if isinstance(x, ndarray):
        for v in x.flat:
            if v:
                return True
        return False
    else:
        return bool(x)

def nanargmax(a, axis=None, out=None, keepdims: Literal[bool] = False):
    a = asarray(a)
    if _is_float_dtype(a.dtype):
        if _any_true(_all(isnan(a), axis=axis)):
            raise ValueError("All-NaN slice encountered")
        r = _argmax(_fill(a, -util.inf(a.dtype)), axis=axis, keepdims=keepdims)
    else:
        r = _argmax(a, axis=axis, keepdims=keepdims)
    if out is not None:
        out[...] = r
        return out
    return r

def nanargmin(a, axis=None, out=None, keepdims: Literal[bool] = False):
    a = asarray(a)
    if _is_float_dtype(a.dtype):
        if _any_true(_all(isnan(a), axis=axis)):
            raise ValueError("All-NaN slice encountered")
        r = _argmin(_fill(a, util.inf(a.dtype)), axis=axis, keepdims=keepdims)
    else:
        r = _argmin(a, axis=axis, keepdims=keepdims)
    if out is not None:
        out[...] = r
        return out
    return r

def _nan_to_num_scalar(x, nan: float, posinf, neginf):
    T = type(x)
    if T is complex or T is complex64:
        F = float if T is complex else float32
        return T(_nan_to_num_scalar(F(x.real), nan, posinf, neginf),
                 _nan_to_num_scalar(F(x.imag), nan, posinf, neginf))
    elif T is float or T is float32 or T is float16:
        if util.isnan(x):
            return T(nan)
        if util.isinf(x):
            if x > T(0):
                return util.maxnum(T) if posinf is None else T(posinf)
            else:
                return util.minnum(T) if neginf is None else T(neginf)
        return x
    else:
        return x

def nan_to_num(x, copy: bool = True, nan: float = 0.0, posinf = None, neginf = None):
    if isinstance(x, ndarray) or isinstance(x, List) or isinstance(x, Tuple):
        a = asarray(x)
        r = a.copy() if (copy or not isinstance(x, ndarray)) else a
        T = a.dtype
        if _is_float_dtype(T):
            for idx in util.multirange(r.shape):
                r[idx] = _nan_to_num_scalar(r[idx], nan, posinf, neginf)
        return r
    else:
        return _nan_to_num_scalar(x, nan, posinf, neginf)
