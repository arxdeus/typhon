# Ported from CPython 3.14 math (PSF)
import math

@test
def test_isqrt():
    assert math.isqrt(0) == 0
    assert math.isqrt(15) == 3
    assert math.isqrt(16) == 4
    assert math.isqrt(10**18 + 1) == 10**9

@test
def test_comb_perm():
    assert math.comb(10, 3) == 120
    assert math.comb(5, 7) == 0
    assert math.comb(60, 30) == 118264581564861424
    assert math.perm(10, 3) == 720
    assert math.perm(5) == 120

@test
def test_float_misc():
    assert math.cbrt(27.0) == 3.0
    assert math.exp2(10.0) == 1024.0
    assert math.nextafter(1.0, 2.0) == 1.0000000000000002
    assert math.ulp(1.0) == 2.220446049250313e-16
    assert math.fma(2.0, 3.0, 1.0) == 7.0
    assert math.dist([0.0, 0.0], [3.0, 4.0]) == 5.0

@test
def test_sumprod():
    assert math.sumprod([1, 2, 3], [4, 5, 6]) == 32
    assert math.sumprod([0.1, 0.2], [1.0, 1.0]) == 0.30000000000000004

test_isqrt()
test_comb_perm()
test_float_misc()
test_sumprod()
