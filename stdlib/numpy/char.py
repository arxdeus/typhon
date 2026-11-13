# Ported from numpy 2.5.4 (BSD-3-Clause)
#
# numpy.char: legacy string operations (same kernels as numpy.strings) plus
# split/rsplit/splitlines/join returning arrays of lists.

import numpy as np
from .strings import upper, lower, swapcase, capitalize, title, str_len, \
    isalpha, isdigit, isalnum, isspace, islower, isupper, istitle, \
    isdecimal, isnumeric, strip, lstrip, rstrip, center, ljust, rjust, \
    zfill, expandtabs, replace, multiply, find, rfind, index, rindex, \
    count, startswith, endswith, partition, rpartition, add, equal, \
    not_equal, less, less_equal, greater, greater_equal, _sarr

def split(a, sep: Optional[str] = None, maxsplit: int = -1):
    x = _sarr(a)
    flat = x.ravel()
    out = np.empty(flat.shape, dtype=List[str])
    for i in range(flat.size):
        s = flat[i]
        out.data[i] = s.split(maxsplit=maxsplit) if sep is None else s.split(sep, maxsplit)
    return out.reshape(x.shape)

def rsplit(a, sep: Optional[str] = None, maxsplit: int = -1):
    x = _sarr(a)
    flat = x.ravel()
    out = np.empty(flat.shape, dtype=List[str])
    for i in range(flat.size):
        s = flat[i]
        out.data[i] = s.rsplit(maxsplit=maxsplit) if sep is None else s.rsplit(sep, maxsplit)
    return out.reshape(x.shape)

def splitlines(a, keepends: bool = False):
    x = _sarr(a)
    flat = x.ravel()
    out = np.empty(flat.shape, dtype=List[str])
    for i in range(flat.size):
        out.data[i] = flat[i].splitlines(keepends)
    return out.reshape(x.shape)

def join(sep, seq):
    x, y = np.broadcast_arrays(_sarr(sep), _sarr(seq))
    fx = x.ravel()
    fy = y.ravel()
    out = np.empty(fx.shape, dtype=str)
    for i in range(fx.size):
        s = fy[i]
        r = ""
        for j in range(len(s)):
            if j > 0:
                r += fx[i]
            r += s[j]
        out[i] = r
    return out.reshape(x.shape)

def compare_chararrays(a1, a2, cmp: str, rstrip: bool):
    x = _sarr(a1)
    y = _sarr(a2)
    if rstrip:
        x = globals_rstrip(x)
        y = globals_rstrip(y)
    if cmp == "==":
        return equal(x, y)
    elif cmp == "!=":
        return not_equal(x, y)
    elif cmp == "<":
        return less(x, y)
    elif cmp == "<=":
        return less_equal(x, y)
    elif cmp == ">":
        return greater(x, y)
    elif cmp == ">=":
        return greater_equal(x, y)
    raise ValueError("comparison must be '==', '!=', '<', '>', '<=', '>='")

def globals_rstrip(x):
    return rstrip(x)

def array(obj):
    return np.asarray(obj)

def asarray(obj):
    return np.asarray(obj)

from .strings import encode, decode, translate, mod
str_ = str
bytes_ = bytes
character = str
chararray = np.ndarray
ndarray = np.ndarray
narray = array
asnarray = asarray
strings_multiply = multiply
strings_partition = partition
strings_rpartition = rpartition
