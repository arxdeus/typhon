# Ported from CPython 3.14 int/float methods (PSF)
@test
def test_int_methods():
    assert (7).bit_count() == 3
    assert (-255).bit_count() == 8
    assert (255).to_bytes(2, "big") == b"\x00\xff"
    assert (255).to_bytes(2, "little") == b"\xff\x00"
    assert int.from_bytes(b"\x01\x02", "big") == 258
    assert int.from_bytes(b"\x01\x02", "little") == 513
    assert (5).as_integer_ratio() == (5, 1)

@test
def test_float_methods():
    assert (3.5).as_integer_ratio() == (7, 2)
    assert (0.1).as_integer_ratio() == (3602879701896397, 36028797018963968)
    assert (3.5).hex() == "0x1.c000000000000p+1"
    assert (0.1).hex() == "0x1.999999999999ap-4"
    assert (-2.0).hex() == "-0x1.0000000000000p+1"
    assert float.fromhex("0x1.8p+1") == 3.0
    assert float.fromhex("-0x1.999999999999ap-4") == -0.1
    assert bytes([1, 2, 255]) == b"\x01\x02\xff"

test_int_methods()
test_float_methods()
