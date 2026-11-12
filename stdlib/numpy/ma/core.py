# Ported from numpy 2.5.4 (BSD-3-Clause)
#
# numpy.ma.core: the MaskedArray container.  A masked array always carries a
# materialized boolean mask of the data shape; `_nomask` records whether the
# mask is numpy's `nomask` (shown as `mask=False`).  Scalar results that are
# masked are returned as `masked` (None); unmasked scalars as Optional values.

import numpy as np
from numpy.util import tuple_set
from _fmt import ma_str, ma_repr
from _red import lanes, flat2, lane, _sum_type

nomask = False
masked = None
MaskError = ValueError
MAError = ValueError

def default_fill_value(T: type):
    if T is float or T is float32:
        return T(1e20)
    elif T is bool:
        return True
    elif T is complex:
        return complex(1e20, 0.0)
    elif T is str:
        return "N/A"
    else:
        return T(999999)

class MaskedArray[X, N: Literal[int]]:
    _data: np.ndarray[X, N]
    _mask: np.ndarray[bool, N]
    _nomask: bool
    _fill: X
    _hard: bool

    def __init__(self, data: np.ndarray[X, N], mask: np.ndarray[bool, N], is_nomask: bool,
                 fill: X, hard: bool = False):
        self._data = data
        self._mask = mask
        self._nomask = is_nomask
        self._fill = fill
        self._hard = hard

    def _wrap(self, d: np.ndarray[Y, M], m: np.ndarray[bool, M], nm: bool, Y: type, M: Literal[int]):
        if Y is X:
            return MaskedArray(d, m, nm, self._fill, False)
        else:
            return MaskedArray(d, m, nm, default_fill_value(Y), False)

    # ---- attributes -------------------------------------------------
    @property
    def data(self):
        return self._data

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        mm = np.broadcast_to(np.asarray(m, dtype=bool), self._data.shape).copy()
        if self._hard:
            self._mask = self._mask | mm
        else:
            self._mask = mm
        self._nomask = False

    @property
    def fill_value(self):
        return self._fill

    @fill_value.setter
    def fill_value(self, v):
        if v is None:
            self._fill = default_fill_value(X)
        else:
            self._fill = X(v)

    def set_fill_value(self, v = None):
        self.fill_value = v

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return N

    @property
    def size(self):
        return self._data.size

    @property
    def dtype(self):
        return X

    @property
    def hardmask(self):
        return self._hard

    @property
    def T(self):
        return self.transpose()

    def harden_mask(self):
        self._hard = True
        return self

    def soften_mask(self):
        self._hard = False
        return self

    def shrink_mask(self):
        if not self._mask.any():
            self._nomask = True
        return self

    def __len__(self):
        return self._data.shape[0]

    # ---- conversion -------------------------------------------------
    def filled(self, fill_value = None):
        r = self._data.copy()
        if fill_value is None:
            r[self._mask] = self._fill
        else:
            r[self._mask] = X(fill_value)
        return r

    def compressed(self):
        return self._data[~self._mask]

    def copy(self):
        return MaskedArray(self._data.copy(), self._mask.copy(), self._nomask, self._fill, self._hard)

    def astype(self, T: type):
        return MaskedArray(self._data.astype(T), self._mask.copy(), self._nomask, default_fill_value(T), self._hard)

    def ravel(self):
        return self._wrap(self._data.ravel(), self._mask.ravel(), self._nomask)

    def flatten(self):
        return self._wrap(self._data.flatten(), self._mask.flatten(), self._nomask)

    def reshape(self, *shape):
        if static.len(shape) == 1 and isinstance(shape[0], Tuple):
            return self._wrap(self._data.reshape(shape[0]), self._mask.reshape(shape[0]), self._nomask)
        else:
            return self._wrap(self._data.reshape(shape), self._mask.reshape(shape), self._nomask)

    def transpose(self):
        return self._wrap(self._data.transpose(), self._mask.transpose(), self._nomask)

    def squeeze(self):
        return self._wrap(np.squeeze(self._data), np.squeeze(self._mask), self._nomask)

    def tolist(self):
        r = List[Optional[X]]()
        fd = self._data.ravel()
        fm = self._mask.ravel()
        for i in range(fd.size):
            if fm[i]:
                r.append(None)
            else:
                r.append(fd[i])
        return r

    def count(self, axis = None):
        if axis is None:
            return int((~self._mask).sum())
        else:
            return (~self._mask).sum(axis=axis)

    # ---- printing ---------------------------------------------------
    def __str__(self):
        return ma_str(self._data, self._mask)

    def __repr__(self):
        return ma_repr(self._data, self._mask, self._nomask, str(self._fill))

    # ---- indexing ---------------------------------------------------
    def __getitem__(self, idx):
        r = self._data[idx]
        if isinstance(r, np.ndarray):
            return MaskedArray(r, self._mask[idx], self._nomask, self._fill, self._hard)
        else:
            if self._mask[idx]:
                return Optional[X]()
            return Optional[X](r)

    def __setitem__(self, idx, value):
        if value is None:
            self._mask[idx] = True
            self._nomask = False
        elif isinstance(value, MaskedArray):
            if self._hard:
                old = self._data.copy()
                self._data[idx] = value._data
                self._data = np.where(self._mask, old, self._data)
                self._mask[idx] = self._mask[idx] | value._mask
            else:
                self._data[idx] = value._data
                self._mask[idx] = value._mask
            self._nomask = False
        else:
            if self._hard:
                old = self._data.copy()
                self._data[idx] = value
                self._data = np.where(self._mask, old, self._data)
            else:
                self._data[idx] = value
                if not self._nomask:
                    self._mask[idx] = False

    # ---- reductions -------------------------------------------------
    def _red(self, kind: Literal[str], axis, p: float, R: type):
        if axis is None or N == 1:
            if axis is not None:
                if axis != 0 and axis != -1:
                    raise np.AxisError("axis out of bounds")
            d2, m2 = flat2(self._data, self._mask)
            v, ok = lane(d2, m2, 0, kind, p, R)
            if ok:
                return Optional[R](v)
            return Optional[R]()
        else:
            d2, m2, sh = lanes(self._data, self._mask, axis)
            L = d2.shape[0]
            out = np.empty(L, R)
            om = np.zeros(L, bool)
            for r in range(L):
                v, ok = lane(d2, m2, r, kind, p, R)
                out[r] = v
                om[r] = not ok
            if kind == "max" or kind == "min":
                for r in range(L):
                    if om[r]:
                        out[r] = R(self._fill)
            return MaskedArray(out.reshape(sh), om.reshape(sh), False, default_fill_value(R) if R is not X else R(self._fill), False)

    def sum(self, axis = None):
        return self._red("sum", axis, 0.0, type(_sum_type(X)))

    def prod(self, axis = None):
        return self._red("prod", axis, 0.0, type(_sum_type(X)))

    def product(self, axis = None):
        return self.prod(axis)

    def mean(self, axis = None):
        return self._red("mean", axis, 0.0, float)

    def var(self, axis = None, ddof: int = 0):
        return self._red("var", axis, float(ddof), float)

    def std(self, axis = None, ddof: int = 0):
        return self._red("std", axis, float(ddof), float)

    def min(self, axis = None):
        return self._red("min", axis, 0.0, X)

    def max(self, axis = None):
        return self._red("max", axis, 0.0, X)

    def ptp(self, axis = None):
        return self.max(axis) - self.min(axis)

    def any(self, axis = None):
        return self._red("any", axis, 0.0, bool)

    def all(self, axis = None):
        return self._red("all", axis, 0.0, bool)

    def _argred(self, kind: Literal[str], axis):
        if axis is None:
            d2, m2 = flat2(self._data, self._mask)
            v, ok = lane(d2, m2, 0, kind, 0.0, int)
            return v
        else:
            d2, m2, sh = lanes(self._data, self._mask, axis)
            L = d2.shape[0]
            out = np.empty(L, int)
            for r in range(L):
                v, ok = lane(d2, m2, r, kind, 0.0, int)
                out[r] = v
            return out.reshape(sh)

    def argmin(self, axis = None):
        return self._argred("argmin", axis)

    def argmax(self, axis = None):
        return self._argred("argmax", axis)

    def cumsum(self, axis = None):
        f = self.filled(X(0))
        if axis is None:
            return self._wrap(np.cumsum(f), self._mask.ravel().copy(), self._nomask)
        return self._wrap(np.cumsum(f, axis=axis), self._mask.copy(), self._nomask)

    def dot(self, b):
        return dot(self, b)

    # ---- arithmetic -------------------------------------------------
    def __add__(self, o):
        return _binop(self, o, "add")
    def __radd__(self, o):
        return _binop(o, self, "add")
    def __sub__(self, o):
        return _binop(self, o, "sub")
    def __rsub__(self, o):
        return _binop(o, self, "sub")
    def __mul__(self, o):
        return _binop(self, o, "mul")
    def __rmul__(self, o):
        return _binop(o, self, "mul")
    def __truediv__(self, o):
        return _binop(self, o, "div")
    def __rtruediv__(self, o):
        return _binop(o, self, "div")
    def __floordiv__(self, o):
        return _binop(self, o, "floordiv")
    def __rfloordiv__(self, o):
        return _binop(o, self, "floordiv")
    def __mod__(self, o):
        return _binop(self, o, "mod")
    def __rmod__(self, o):
        return _binop(o, self, "mod")
    def __pow__(self, o):
        return _binop(self, o, "pow")
    def __rpow__(self, o):
        return _binop(o, self, "pow")
    def __eq__(self, o):
        return _binop(self, o, "eq")
    def __ne__(self, o):
        return _binop(self, o, "ne")
    def __lt__(self, o):
        return _binop(self, o, "lt")
    def __le__(self, o):
        return _binop(self, o, "le")
    def __gt__(self, o):
        return _binop(self, o, "gt")
    def __ge__(self, o):
        return _binop(self, o, "ge")
    def __and__(self, o):
        return _binop(self, o, "and")
    def __or__(self, o):
        return _binop(self, o, "or")
    def __neg__(self):
        return self._wrap(-self._data, self._mask.copy(), self._nomask)
    def __pos__(self):
        return self.copy()
    def __abs__(self):
        return self._wrap(np.abs(self._data), self._mask.copy(), self._nomask)
    def __invert__(self):
        return self._wrap(~self._data, self._mask.copy(), self._nomask)

    def _inplace(self, r):
        if self._hard:
            keep = self._mask.copy()
            self._data = np.where(keep, self._data, r._data.astype(X))
            self._mask = keep | r._mask
        else:
            self._data = np.where(r._mask, self._data, r._data.astype(X))
            self._mask = r._mask
        self._nomask = r._nomask
        return self

    def __iadd__(self, o):
        return self._inplace(_binop(self, o, "add"))
    def __isub__(self, o):
        return self._inplace(_binop(self, o, "sub"))
    def __imul__(self, o):
        return self._inplace(_binop(self, o, "mul"))
    def __itruediv__(self, o):
        return self._inplace(_binop(self, o, "div"))
    def __ipow__(self, o):
        return self._inplace(_binop(self, o, "pow"))

# ---------------------------------------------------------------------
# helpers

def _dm(x):
    if isinstance(x, MaskedArray):
        return x._data, x._mask, x._nomask
    elif isinstance(x, np.ndarray):
        return x, np.zeros(x.shape, bool), True
    elif x is None:
        return np.asarray(0.0), np.asarray(True), False
    else:
        a = np.asarray(x)
        return a, np.zeros(a.shape, bool), True

def _fill_of(a, b, T: type):
    if isinstance(a, MaskedArray):
        if a.dtype is T:
            return a._fill
    if isinstance(b, MaskedArray):
        if b.dtype is T:
            return b._fill
    return default_fill_value(T)

def _op(op: Literal[str], da, db):
    if op == "add":
        return da + db
    elif op == "sub":
        return da - db
    elif op == "mul":
        return da * db
    elif op == "div":
        return np.true_divide(da, db)
    elif op == "floordiv":
        return np.floor_divide(da, db)
    elif op == "mod":
        return np.mod(da, db)
    elif op == "pow":
        return np.power(da, db)
    elif op == "eq":
        return da == db
    elif op == "ne":
        return da != db
    elif op == "lt":
        return da < db
    elif op == "le":
        return da <= db
    elif op == "gt":
        return da > db
    elif op == "ge":
        return da >= db
    elif op == "and":
        return da & db
    elif op == "or":
        return da | db
    elif op == "maximum":
        return np.maximum(da, db)
    elif op == "minimum":
        return np.minimum(da, db)
    elif op == "arctan2":
        return np.arctan2(da, db)
    elif op == "hypot":
        return np.hypot(da, db)
    else:
        compile_error("unknown masked binary op")

def _binop(a, b, op: Literal[str]):
    da, ma, na = _dm(a)
    db, mb, nb = _dm(b)
    res = np.asarray(_op(op, da, db))
    m = np.logical_or(ma, mb)
    nm = na and nb
    if op == "div" or op == "floordiv" or op == "mod":
        # _DomainSafeDivide: |a| * tiny >= |b|  plus invalid results
        tiny = np.finfo(float).tiny
        dom = np.abs(np.asarray(da).astype(float)) * tiny >= np.abs(np.asarray(db).astype(float))
        m = m | dom | ~np.isfinite(res)
        nm = False
    elif op == "pow":
        m = m | ~np.isfinite(res)
        nm = nm and not m.any()
    m = np.broadcast_to(m, res.shape).copy()
    if m.ndim == 0:
        return MaskedArray(res, m, nm, _fill_of(a, b, res.dtype), False)
    if not nm:
        dab = np.broadcast_to(np.asarray(da), res.shape)
        if res.dtype is type(dab.dtype()) and not (op == "eq" or op == "ne"):
            res = np.where(m, dab, res)
    return MaskedArray(res, m, nm, _fill_of(a, b, res.dtype), False)

def _unop(a, f, dom: Literal[str]):
    d, m, nm = _dm(a)
    res = np.asarray(f(d))
    if dom != "":
        df = np.asarray(d).astype(float)
        if dom == "sqrt":
            bad = df < 0.0
        elif dom == "log":
            bad = df <= 0.0
        elif dom == "arc":
            bad = np.abs(df) > 1.0
        elif dom == "tan":
            bad = np.abs(np.cos(df)) < 1e-35
        else:
            bad = df != df
        m = m | bad | ~np.isfinite(res)
        nm = False
        if res.dtype is type(d.dtype()):
            res = np.where(m, d, res)
    return MaskedArray(res, m.copy(), nm, default_fill_value(res.dtype), False)
