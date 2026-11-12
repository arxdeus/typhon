# np.r_ and np.c_ index tricks.

import util
import internal.static as static

from .ndarray import ndarray
from .routines import asarray, array, arange, linspace, concatenate

def _slice_to_array(s: Slice):
    if s.V is complex:
        # complex step: number of points, endpoint inclusive (like linspace)
        start = float(s.start) if s.start is not None else 0.0
        stop = float(s.stop)
        num = int(abs(s.step))
        return linspace(start, stop, num=num)
    else:
        start = s.start if s.start is not None else 0
        step = s.step if s.step is not None else 1
        return arange(start, s.stop, step)

def _item_to_array(item, ndmin: Literal[int], trans_col: Literal[bool]):
    if isinstance(item, Slice):
        a = _slice_to_array(item)
    else:
        a = asarray(item)
    if static.len(a.shape) == 0:
        return a.reshape((1,) * ndmin)
    if static.len(a.shape) == 1 and ndmin == 2:
        if trans_col:
            return a.reshape(a.size, 1)
        else:
            return a.reshape(1, a.size)
    return a

def _parse_axis(directive: str) -> int:
    d = directive.strip()
    if d == "r":
        return 0
    if "," in d:
        parts = d.split(",")
        if len(parts) > 1 and parts[1].strip() != "":
            raise NotImplementedError("np.r_ string directive: only the axis field is supported")
        d = parts[0]
    return int(d)

def _convert_all(key, ndmin: Literal[int], trans_col: Literal[bool]):
    return tuple(_item_to_array(key[i], ndmin, trans_col) for i in static.range(static.len(key)))

@tuple
class RClass:
    def __getitem__(self, key):
        if isinstance(key, Tuple):
            if static.len(key) > 0 and isinstance(key[0], str):
                axis = _parse_axis(key[0])
                return concatenate(_convert_all(key[1:], 1, False), axis=axis)
            return concatenate(_convert_all(key, 1, False), axis=0)
        else:
            return _item_to_array(key, 1, False).copy()

    def __len__(self) -> int:
        return 0

@tuple
class CClass:
    def __getitem__(self, key):
        if isinstance(key, Tuple):
            return concatenate(_convert_all(key, 2, True), axis=-1)
        else:
            return _item_to_array(key, 2, True).copy()

    def __len__(self) -> int:
        return 0

r_ = RClass()
c_ = CClass()
