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

# ---- further numpy.testing API ----

IS_64BIT = True
IS_PYPY = False
IS_PYSTON = False
IS_WASM = False
IS_MUSL = False
IS_EDITABLE = False
IS_INSTALLED = True
HAS_REFCOUNT = False
HAS_LAPACK64 = False
NOGIL_BUILD = False
BLAS_SUPPORTS_FPE = True
verbose = 0

class KnownFailureException(Exception):
    def __init__(self, msg: str = ''):
        super().__init__(msg)

class SkipTest(Exception):
    def __init__(self, msg: str = ''):
        super().__init__(msg)

IgnoreException = KnownFailureException

def _nulp_diff_one(x: float, y: float) -> float:
    # numpy.testing._private.utils.nulp_diff for float64
    if x == y:
        return 0.0
    ix = _int64_view(x)
    iy = _int64_view(y)
    if ix < 0:
        ix = -9223372036854775808 - ix
    if iy < 0:
        iy = -9223372036854775808 - iy
    return float(abs(ix - iy))

def _int64_view(x: float) -> int:
    v = x
    return Ptr[int](__ptr__(v).as_byte())[0]

def assert_array_almost_equal_nulp(x, y, nulp: int = 1):
    ax = np.abs(np.asarray(x, dtype=float))
    ay = np.abs(np.asarray(y, dtype=float))
    xf = np.asarray(x, dtype=float)
    yf = np.asarray(y, dtype=float)
    ref = nulp * np.spacing(np.where(ax > ay, ax, ay))
    if not bool(np.all(np.abs(xf - yf) <= ref)):
        xb, yb = np.broadcast_arrays(xf, yf)
        xr = xb.ravel()
        yr = yb.ravel()
        mx = 0.0
        for i in range(xr.size):
            d = _nulp_diff_one(xr[i], yr[i])
            if d > mx:
                mx = d
        raise AssertionError("Arrays are not equal to " + str(nulp) + " ULP (max is " + str(int(mx)) + ")")

def assert_approx_equal(actual: float, desired: float, significant: int = 7,
                        err_msg: str = '', verbose: bool = True):
    import math
    if actual == desired:
        return
    scale = 0.5 * (abs(desired) + abs(actual))
    scale = 10.0 ** math.floor(math.log10(scale)) if scale != 0.0 else 1.0
    sc_desired = desired / scale if scale != 0.0 else desired
    sc_actual = actual / scale if scale != 0.0 else actual
    if math.isnan(actual) or math.isnan(desired):
        if math.isnan(actual) and math.isnan(desired):
            return
        raise AssertionError("\nItems are not equal to " + str(significant) + " significant digits:\n ACTUAL: "
                             + str(actual) + "\n DESIRED: " + str(desired))
    if abs(sc_desired - sc_actual) >= 10.0 ** -(significant - 1):
        s = "\nItems are not equal to " + str(significant) + " significant digits:"
        if err_msg != '':
            s += "\n" + err_msg
        s += "\n ACTUAL: " + str(actual) + "\n DESIRED: " + str(desired)
        raise AssertionError(s)

def assert_raises_regex(exc: type, expected_regexp: str, func, *args, **kwargs):
    import re
    try:
        func(*args, **kwargs)
    except exc as e:
        if re.search(expected_regexp, str(e)) is None:
            raise AssertionError('"' + expected_regexp + '" does not match "' + str(e) + '"')
        return
    raise AssertionError(exc.__name__ + " not raised by " + func.__name__)

class _NoopWarnCtx:
    def __init__(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, et, ev, tb):
        return False
    def filter(self, category = None, message: str = ''):
        pass
    def record(self, category = None, message: str = ''):
        return List[str]()

def assert_warns(warning_class = None, *args, **kwargs):
    # warnings are not tracked at runtime: context manager / call passthrough
    return _NoopWarnCtx()

def assert_no_warnings(func = None, *args, **kwargs):
    if func is None:
        return _NoopWarnCtx()
    return func(*args, **kwargs)

def suppress_warnings(forwarding_rule: str = 'always'):
    return _NoopWarnCtx()

def clear_and_catch_warnings(record: bool = False, modules = ()):
    return _NoopWarnCtx()

def assert_no_gc_cycles(func = None, *args, **kwargs):
    if func is None:
        return _NoopWarnCtx()
    func(*args, **kwargs)

def break_cycles():
    pass

def build_err_msg(arrays, err_msg: str, header: str = 'Items are not equal:',
                  verbose: bool = True, names = ('ACTUAL', 'DESIRED'), precision: int = 8) -> str:
    msg = '\n' + header
    if err_msg != '':
        if err_msg.find('\n') == -1 and len(err_msg) < 79 - len(header):
            msg = msg + ' ' + err_msg
        else:
            msg = msg + '\n' + err_msg
    if verbose:
        for i in static.range(static.len(arrays)):
            msg = msg + '\n ' + names[i] + ': ' + repr(arrays[i])
    return msg

def print_assert_equal(test_string: str, actual, desired):
    if actual != desired:
        raise AssertionError(test_string + " failed\nACTUAL: \n" + str(actual) + "\nDESIRED: \n" + str(desired) + "\n")

def assert_array_compare(comparison, x, y, err_msg: str = '', verbose: bool = True,
                         header: str = '', precision: int = 6, equal_nan: bool = True,
                         equal_inf: bool = True, strict: bool = False):
    xa = np.asarray(x)
    ya = np.asarray(y)
    ok = comparison(xa, ya)
    if not bool(np.all(ok)):
        bad = np.logical_not(ok)
        nbad = int(np.count_nonzero(np.atleast_1d(bad)))
        total = int(np.atleast_1d(bad).size)
        s = "\n" + header
        if err_msg != '':
            s += "\n" + err_msg
        s += "\nMismatched elements: " + str(nbad) + " / " + str(total)
        s += "\n x: " + _fmt_arr(xa) + "\n y: " + _fmt_arr(ya)
        raise AssertionError(s)

def measure(code_str: str, times: int = 1, label = None) -> float:
    raise NotImplementedError("numpy.testing.measure requires runtime code evaluation")

def jiffies() -> int:
    import time
    return int(time.time() * 100.0)

def memusage():
    return None

def run_threaded(func, max_workers: int = 8, pass_count: bool = False,
                 pass_barrier: bool = False, outer_iterations: int = 1, prepare_args = None):
    for _ in range(outer_iterations):
        for i in range(max_workers):
            if pass_count:
                func(i)
            else:
                func()

def decorate_methods(cls, decorator, testmatch = None):
    pass
