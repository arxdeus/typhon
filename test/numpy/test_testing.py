# Tests for numpy.testing assertion helpers (messages compared to numpy 2.5.4).
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_array_almost_equal, \
    assert_almost_equal, assert_equal, assert_array_less, assert_raises, assert_, \
    assert_string_equal, assert_array_max_ulp

def fail(what: str):
    print('TEST FAILED', what)

def expect_fail(name: str, f, *args, **kwargs):
    try:
        f(*args, **kwargs)
    except AssertionError as e:
        return str(e)
    fail(name + ' did not raise')
    return ''

# passing cases
assert_allclose(np.array([1.0, 2.0, np.nan]), [1.0 + 1e-9, 2.0, np.nan])
assert_allclose(1.0, 1.05, rtol=0.1)
assert_allclose(np.array([1.0, np.inf]), np.array([1.0, np.inf]))
assert_allclose(np.zeros((2, 3)), 0.0, atol=1e-12)
assert_array_equal(np.array([1, 2, 3]), [1, 2, 3])
assert_array_equal(np.array([[1.0, np.nan]]), np.array([[1.0, np.nan]]))
assert_array_almost_equal(np.array([1.0, 2.0]), np.array([1.0000001, 2.0]))
assert_array_almost_equal([1.0, 2.0], [1.004, 2.0], decimal=2)
assert_almost_equal(1.0, 1.00000001)
assert_almost_equal(np.array([0.5, 1.5]), [0.5, 1.5 + 1e-9])
assert_equal(3, 3)
assert_equal('a', 'a')
assert_equal([1, 2], [1, 2])
assert_equal((1, 'x'), (1, 'x'))
assert_equal({'a': 1}, {'a': 1})
assert_equal(np.nan, np.nan)
assert_equal(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
assert_array_less(np.array([1.0, 2.0]), np.array([2.0, 3.0]))
assert_array_less(np.array([1.0, 2.0]), 5.0)
assert_(1 + 1 == 2)
assert_string_equal('abc', 'abc')
assert_array_max_ulp(np.array([1.0]), np.array([1.0 + 2.220446049250313e-16]))

# failing cases and messages
m = expect_fail('allclose', assert_allclose, np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.5, 3.0]))
if 'Not equal to tolerance rtol=1e-07, atol=0' not in m:
    fail('allclose header: ' + m)
if 'Mismatched elements: 1 / 3 (33.3%)' not in m:
    fail('allclose count: ' + m)
if 'Max absolute difference among violations: 0.5' not in m:
    fail('allclose absdiff: ' + m)
if 'Max relative difference among violations: 0.2' not in m:
    fail('allclose reldiff: ' + m)
m = expect_fail('allclose shape', assert_allclose, np.zeros(2), np.zeros(3))
if '(shapes (2,), (3,) mismatch)' not in m:
    fail('allclose shape msg: ' + m)
m = expect_fail('allclose strict', assert_allclose, np.zeros(2), 0.0, strict=True)
if 'mismatch' not in m:
    fail('strict msg: ' + m)
m = expect_fail('allclose nan', assert_allclose, np.array([np.nan]), np.array([1.0]))
m = expect_fail('allclose nan2', assert_allclose, np.array([np.nan]), np.array([np.nan]), equal_nan=False)
m = expect_fail('array_equal', assert_array_equal, np.array([1, 2, 3]), np.array([1, 2, 4]), err_msg='custom')
if 'Arrays are not equal' not in m or 'custom' not in m or 'Mismatched elements: 1 / 3' not in m:
    fail('array_equal msg: ' + m)
m = expect_fail('array_equal nan', assert_array_equal, np.array([1.0, np.nan]), np.array([1.0, 2.0]))
if 'nan location mismatch' not in m:
    fail('nan loc: ' + m)
m = expect_fail('almost', assert_array_almost_equal, np.array([1.0, 2.0]), np.array([1.0, 2.1]))
if 'Arrays are not almost equal to 6 decimals' not in m:
    fail('almost msg: ' + m)
m = expect_fail('almost scalar', assert_almost_equal, 1.0, 1.1)
if 'ACTUAL: 1.0' not in m or 'DESIRED: 1.1' not in m:
    fail('almost scalar msg: ' + m)
m = expect_fail('equal', assert_equal, 1, 2)
if 'Items are not equal' not in m:
    fail('equal msg: ' + m)
expect_fail('equal list', assert_equal, [1, 2], [1, 3])
expect_fail('equal list len', assert_equal, [1, 2], [1, 2, 3])
expect_fail('equal dict', assert_equal, {'a': 1}, {'a': 2})
expect_fail('equal arr', assert_equal, np.array([1.0]), np.array([2.0]))
m = expect_fail('less', assert_array_less, np.array([1.0, 3.0]), np.array([2.0, 3.0]))
if 'Arrays are not strictly ordered `x < y`' not in m or 'Mismatched elements: 1 / 2 (50%)' not in m:
    fail('less msg: ' + m)
m = expect_fail('assert_', assert_, False, 'boom')
if m != 'boom':
    fail('assert_ msg: ' + m)
expect_fail('string', assert_string_equal, 'a', 'b')
expect_fail('ulp', assert_array_max_ulp, np.array([1.0]), np.array([1.001]))

# assert_raises: function form and context manager
def bad(x, y: int = 0):
    raise ValueError('bad')
def good(x):
    return x
assert_raises(ValueError, bad, 1, y=2)
try:
    assert_raises(ValueError, good, 1)
    fail('raises fn did not raise')
except AssertionError as e:
    if 'ValueError not raised' not in str(e):
        fail('raises msg: ' + str(e))
with assert_raises(ValueError):
    bad(1)
with assert_raises(ZeroDivisionError):
    _ = 1 // int(np.zeros(1)[0])
try:
    with assert_raises(KeyError):
        pass
    fail('ctx not raised')
except AssertionError as e:
    if 'KeyError not raised' not in str(e):
        fail('ctx msg: ' + str(e))
