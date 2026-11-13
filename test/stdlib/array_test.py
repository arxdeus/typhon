# Ported from CPython 3.14 array (PSF)
from array import array

@test
def test_int_arrays():
    a = array("i", [1, 2, 3])
    a.append(4)
    a.extend([5, 6])
    assert len(a) == 6 and a[0] == 1 and a[-1] == 6
    a[1] = -7
    assert a.tolist() == [1, -7, 3, 4, 5, 6]
    assert a.itemsize == 4
    assert a.tobytes() == b"\x01\x00\x00\x00\xf9\xff\xff\xff\x03\x00\x00\x00\x04\x00\x00\x00\x05\x00\x00\x00\x06\x00\x00\x00"
    b = array[int]("i")
    b.frombytes(a.tobytes())
    assert b.tolist() == a.tolist()
    h = array("H", [1, 2])
    h.byteswap()
    assert h.tolist() == [256, 512]
    assert a.pop() == 6
    a.reverse()
    assert a.tolist() == [5, 4, 3, -7, 1]
    assert a.count(3) == 1 and a.index(4) == 1
    assert array("B", [255])[0] == 255
    try:
        array("b", [128])
        assert False
    except OverflowError:
        pass

@test
def test_float_arrays():
    d = array("d", [1.5, 2.5])
    assert d.tobytes() == b"\x00\x00\x00\x00\x00\x00\xf8?\x00\x00\x00\x00\x00\x00\x04@"
    f = array("f", [0.1])
    assert f[0] == 0.10000000149011612
    assert f.itemsize == 4 and d.itemsize == 8
    e = array[float]("d")
    e.frombytes(d.tobytes())
    assert e.tolist() == [1.5, 2.5]

test_int_arrays()
test_float_arrays()
