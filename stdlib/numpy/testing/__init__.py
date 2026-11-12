# Ported from numpy 2.5.4 (BSD-3-Clause)
#
# numpy.testing: array assertion helpers. Failures raise AssertionError with
# messages laid out like numpy's.

import numpy as np
import internal.static as static

def _fmt_arr(a) -> str:
    if a.ndim == 0:
        return "array(" + str(a.item()) + ")"
    return "array(" + str(a.tolist()) + ")"

def _header(msg: str, kind: str, extra: str) -> str:
    s = "\n" + kind
    if extra != "":
        s += " (" + extra + ")"
    if msg != "":
        s += "\n" + msg
    return s

def _shape_str(a) -> str:
    return str(a.shape)

def _bcast_check(x, y, msg: str, kind: str, strict: bool):
    if strict and _shape_str(x) != _shape_str(y):
        raise AssertionError(_header(msg, kind, "shapes " + _shape_str(x) + ", " + _shape_str(y) + " mismatch")
                             + "\n x: " + _fmt_arr(x) + "\n y: " + _fmt_arr(y))
    try:
        np.broadcast_shapes(x.shape, y.shape)
    except ValueError:
        raise AssertionError(_header(msg, kind, "shapes " + _shape_str(x) + ", " + _shape_str(y) + " mismatch")
                             + "\n x: " + _fmt_arr(x) + "\n y: " + _fmt_arr(y))

def _report(x, y, bad, total: int, msg: str, kind: str, extra: str, absd: bool):
    nbad = int(np.count_nonzero(np.atleast_1d(bad)))
    pct = 100.0 * nbad / total if total > 0 else 0.0
    s = _header(msg, kind, extra)
    s += "\nMismatched elements: " + str(nbad) + " / " + str(total) + " (" + f"{pct:.3g}" + "%)"
    if absd:
        bb = np.atleast_1d(bad).ravel()
        sh = np.atleast_1d(bad).shape
        xb = np.broadcast_to(np.atleast_1d(np.asarray(x, dtype=float)), sh).ravel()
        yb = np.broadcast_to(np.atleast_1d(np.asarray(y, dtype=float)), sh).ravel()
        mabs = 0.0
        mrel = 0.0
        for i in range(bb.size):
            if bb[i]:
                dv = abs(xb[i] - yb[i])
                if dv > mabs:
                    mabs = dv
                yv = abs(yb[i])
                rv = dv / yv if yv != 0.0 else np.inf
                if rv > mrel:
                    mrel = rv
        s += "\nMax absolute difference among violations: " + str(mabs)
        s += "\nMax relative difference among violations: " + str(mrel)
    s += "\n x: " + _fmt_arr(x) + "\n y: " + _fmt_arr(y)
    raise AssertionError(s)

def _nan_pos_ok(x, y, msg: str, kind: str):
    if x.dtype is float or x.dtype is float32 or y.dtype is float or y.dtype is float32:
        xn = np.isnan(np.asarray(x, dtype=float))
        yn = np.isnan(np.asarray(y, dtype=float))
        if bool(np.any(xn != yn)):
            raise AssertionError(_header(msg, kind, "") + "\nx and y nan location mismatch:"
                                 + "\n x: " + _fmt_arr(x) + "\n y: " + _fmt_arr(y))

def assert_allclose(actual, desired, rtol: float = 1e-07, atol: float = 0.0,
                    equal_nan: bool = True, err_msg: str = '', verbose: bool = True,
                    strict: bool = False):
    x = np.asarray(actual)
    y = np.asarray(desired)
    kind = "Not equal to tolerance rtol=" + f"{rtol:g}" + ", atol=" + f"{atol:g}"
    _bcast_check(x, y, err_msg, kind, strict)
    xf = np.asarray(x, dtype=float)
    yf = np.asarray(y, dtype=float)
    bad = np.abs(xf - yf) > atol + rtol * np.abs(yf)
    xn = np.isnan(xf)
    yn = np.isnan(yf)
    if equal_nan:
        bad = (bad | (xn != yn)) & ~(xn & yn)
    else:
        bad = bad | xn | yn
    xi = np.isinf(xf)
    yi = np.isinf(yf)
    bad = bad & ~(xi & yi & (xf == yf))
    if bool(np.any(bad)):
        _report(x, y, bad, int(bad.size), err_msg, kind, "", True)

def assert_array_equal(actual, desired, err_msg: str = '', verbose: bool = True,
                       strict: bool = False):
    x = np.asarray(actual)
    y = np.asarray(desired)
    kind = "Arrays are not equal"
    _bcast_check(x, y, err_msg, kind, strict)
    _nan_pos_ok(x, y, err_msg, kind)
    eq = x == y
    bad = ~eq
    if x.dtype is float or y.dtype is float:
        both = np.isnan(np.asarray(x, dtype=float)) & np.isnan(np.asarray(y, dtype=float))
        bad = bad & ~both
    if bool(np.any(bad)):
        num = (x.dtype is float or x.dtype is int or x.dtype is float32) and \
              (y.dtype is float or y.dtype is int or y.dtype is float32)
        if num:
            _report(x, y, bad, int(bad.size), err_msg, kind, "", True)
        else:
            _report(x, y, bad, int(bad.size), err_msg, kind, "", False)

def assert_array_almost_equal(actual, desired, decimal: int = 6, err_msg: str = '',
                              verbose: bool = True):
    x = np.asarray(actual)
    y = np.asarray(desired)
    kind = "Arrays are not almost equal to " + str(decimal) + " decimals"
    _bcast_check(x, y, err_msg, kind, False)
    _nan_pos_ok(x, y, err_msg, kind)
    xf = np.asarray(x, dtype=float)
    yf = np.asarray(y, dtype=float)
    thr = 1.5 * 10.0 ** (-decimal)
    bad = np.abs(xf - yf) >= thr
    both = np.isnan(xf) & np.isnan(yf)
    binf = np.isinf(xf) & np.isinf(yf) & (xf == yf)
    bad = bad & ~both & ~binf
    if bool(np.any(bad)):
        _report(x, y, bad, int(bad.size), err_msg, kind, "", True)

def assert_almost_equal(actual, desired, decimal: int = 7, err_msg: str = '',
                        verbose: bool = True):
    if isinstance(actual, np.ndarray) or isinstance(desired, np.ndarray) or \
       isinstance(actual, list) or isinstance(desired, list) or \
       isinstance(actual, tuple) or isinstance(desired, tuple):
        assert_array_almost_equal(actual, desired, decimal, err_msg)
    else:
        a = complex(actual)
        d = complex(desired)
        ok = abs(d - a) < 1.5 * 10.0 ** (-decimal)
        if np.isnan(a.real) and np.isnan(d.real):
            ok = True
        if a == d:
            ok = True
        if not ok:
            raise AssertionError(_header(err_msg, "Arrays are not almost equal to " + str(decimal) + " decimals", "")
                                 + "\n ACTUAL: " + repr(actual) + "\n DESIRED: " + repr(desired))

def assert_array_less(x, y, err_msg: str = '', verbose: bool = True, strict: bool = False):
    xa = np.asarray(x)
    ya = np.asarray(y)
    kind = "Arrays are not strictly ordered `x < y`"
    _bcast_check(xa, ya, err_msg, kind, strict)
    xf = np.asarray(xa, dtype=float)
    yf = np.asarray(ya, dtype=float)
    bad = ~(xf < yf)
    both = np.isnan(xf) & np.isnan(yf)
    bad = bad & ~both
    if bool(np.any(bad)):
        _report(xa, ya, bad, int(bad.size), err_msg, kind, "", True)

def assert_equal(actual, desired, err_msg: str = '', verbose: bool = True,
                 strict: bool = False):
    if isinstance(actual, np.ndarray) or isinstance(desired, np.ndarray):
        assert_array_equal(actual, desired, err_msg, verbose, strict)
    elif isinstance(actual, List) and isinstance(desired, List):
        assert_equal(len(actual), len(desired), err_msg)
        for k in range(len(actual)):
            assert_equal(actual[k], desired[k], "item=" + str(k) + "\n" + err_msg)
    elif isinstance(actual, Tuple) and isinstance(desired, Tuple):
        if static.len(actual) != static.len(desired):
            raise AssertionError(_header(err_msg, "Items are not equal:", "")
                                 + "\n ACTUAL: " + repr(actual) + "\n DESIRED: " + repr(desired))
        else:
            for i in static.range(static.len(actual)):
                assert_equal(actual[i], desired[i], err_msg)
    elif isinstance(actual, Dict) and isinstance(desired, Dict):
        assert_equal(len(actual), len(desired), err_msg)
        for k in desired:
            if k not in actual:
                raise AssertionError(repr(k))
            assert_equal(actual[k], desired[k], "key=" + repr(k) + "\n" + err_msg)
    else:
        ok = actual == desired
        if isinstance(actual, float) and isinstance(desired, float):
            if np.isnan(actual) and np.isnan(desired):
                ok = True
        if not ok:
            raise AssertionError(_header(err_msg, "Items are not equal:", "")
                                 + "\n ACTUAL: " + repr(actual) + "\n DESIRED: " + repr(desired))

def assert_(val, msg: str = ''):
    if not val:
        raise AssertionError(msg)

def assert_string_equal(actual: str, desired: str):
    if actual != desired:
        raise AssertionError("Differ:\n- " + actual + "\n+ " + desired)

def assert_array_max_ulp(a, b, maxulp: int = 1):
    af = np.asarray(a, dtype=float)
    bf = np.asarray(b, dtype=float)
    sp = np.spacing(np.maximum(np.abs(af), np.abs(bf)))
    ulp = np.abs(af - bf) / sp
    if bool(np.any(ulp > maxulp)):
        raise AssertionError("Arrays are not almost equal up to " + str(maxulp) + " ULP (max difference is "
                             + str(float(ulp.max())) + " ULP)")
    return ulp

class _AssertRaisesContext[E]:
    _name: str

    def __init__(self, name: str):
        self._name = name

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if ev is None:
            raise AssertionError(self._name + " not raised")
        return isinstance(ev, E)

def assert_raises(exc: type, func = None, *args, **kwargs):
    """
    assert_raises(Exc, f, *args, **kwargs) or ``with assert_raises(Exc):``.
    """
    if func is None:
        return _AssertRaisesContext[exc](exc.__name__)
    else:
        try:
            func(*args, **kwargs)
        except exc:
            return _AssertRaisesContext[exc](exc.__name__)
        raise AssertionError(exc.__name__ + " not raised by " + func.__name__)
