# Empty-list array construction and floating-point error state (numpy 2.5.4).
import numpy as np

a = np.array([])
if a.shape != (0,) or a.dtype is not float or a.size != 0:
    print("TEST FAILED: np.array([])", a.shape)
b = np.array([[]])
if b.shape != (1, 0) or b.dtype is not float:
    print("TEST FAILED: np.array([[]])", b.shape)
c = np.array([[[]]])
if c.shape != (1, 1, 0):
    print("TEST FAILED: np.array([[[]]])", c.shape)
d = np.array(List[int]())
if d.shape != (0,) or d.dtype is not int:
    print("TEST FAILED: typed empty list", d.shape)
e = np.asarray(List[float]())
if e.shape != (0,) or e.sum() != 0.0:
    print("TEST FAILED: asarray empty")
f = np.array([[1.5, 2.0], [3.0, 4.0]])
if f.shape != (2, 2) or f[1, 0] != 3.0:
    print("TEST FAILED: regular nested list")
g = np.array([List[float](), List[float]()])
if g.shape != (2, 0):
    print("TEST FAILED: list of empty lists", g.shape)

d0 = np.geterr()
if d0["divide"] != "warn" or d0["over"] != "warn" or d0["under"] != "ignore" or d0["invalid"] != "warn":
    print("TEST FAILED: default geterr", d0)
old = np.seterr(all="ignore")
if old["divide"] != "warn" or np.geterr()["invalid"] != "ignore":
    print("TEST FAILED: seterr(all)")
np.seterr(divide=old["divide"], over=old["over"], under=old["under"], invalid=old["invalid"])
with np.errstate(divide="raise", invalid="ignore"):
    s = np.geterr()
    if s["divide"] != "raise" or s["invalid"] != "ignore" or s["over"] != "warn":
        print("TEST FAILED: errstate inside", s)
    r = np.array([1.0, 0.0]) / np.array([0.0, 0.0])
    if not np.isinf(r[0]) or not np.isnan(r[1]):
        print("TEST FAILED: division under errstate")
if np.geterr()["divide"] != "warn" or np.geterr()["invalid"] != "warn":
    print("TEST FAILED: errstate restore", np.geterr())
caught = False
try:
    np.seterr(divide="bogus")
except ValueError:
    caught = True
if not caught:
    print("TEST FAILED: invalid mode accepted")

@test
def test_array_of_arrays():
    x = np.array([2.1, 3.4])
    y = np.array([x, x])
    assert y.shape == (2, 2) and y[1, 1] == 3.4
    assert np.array([[x]]).shape == (1, 1, 2)
    e = List[np.ndarray[float, 1]]()
    assert np.array(e).shape == (0, 0)

test_array_of_arrays()

@test
def test_array_nested_promotion():
    t = np.array([(0, 0, 0), (0, 1, 0)])
    assert t.shape == (2, 3) and t[1, 1] == 1
    c = np.array([[1., 2.+1j], [3., 4.]])
    assert c.dtype is complex and c[0, 1] == 2.+1j and c[1, 0] == 3.
    f = np.array([[1, 2], [3.5, 4]])
    assert f.dtype is float and f[1, 0] == 3.5
    out = np.empty(3, dtype=List[int])
    for i in range(3):
        out[i] = [i, i + 1]
    assert out[2] == [2, 3]

test_array_nested_promotion()
