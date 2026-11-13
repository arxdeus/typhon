# Ported from CPython 3.14 decimal (PSF); expected values from the reference interpreter
from decimal import Decimal, getcontext, ROUND_HALF_UP, ROUND_DOWN

@test
def test_basic():
    a = Decimal("1.1")
    b = Decimal("2.20")
    assert str(a + b) == "3.30"
    assert str(b - a) == "1.10"
    assert str(a * b) == "2.420"
    assert str(Decimal("1") / Decimal("3")) == "0.3333333333333333333333333333"
    assert str(Decimal("10") / Decimal("4")) == "2.5"
    assert str(Decimal("100") / Decimal("4")) == "25"
    assert str(Decimal("7") // Decimal("2")) == "3"
    assert str(Decimal("-7") % Decimal("2")) == "-1"
    assert repr(Decimal("0.1")) == "Decimal('0.1')"
    assert str(Decimal("1E+5")) == "1E+5"
    assert str(Decimal("0.0000001")) == "1E-7"
    assert str(Decimal(5)) == "5"

@test
def test_round_sqrt():
    assert str(Decimal("2").sqrt()) == "1.414213562373095048801688724"
    assert str(Decimal("3.14159").quantize(Decimal("0.01"))) == "3.14"
    assert str(Decimal("2.675").quantize(Decimal("0.01"))) == "2.68"
    assert str(Decimal("2.675").quantize(Decimal("0.01"), ROUND_DOWN)) == "2.67"
    assert str(Decimal("2.5").to_integral_value()) == "2"
    assert str(Decimal("3.5").to_integral_value()) == "4"
    assert str(Decimal("1.2345") ** 3) == "1.881365963625"
    assert Decimal("1.0") == Decimal("1.00")
    assert Decimal("1.5") > Decimal("1.49")
    assert Decimal("0.1") + Decimal("0.2") == Decimal("0.3")
    assert str(Decimal("123456789012345678901234567890") + Decimal("1")) == "1.234567890123456789012345679E+29"

test_basic()
test_round_sqrt()

@test
def test_transcendental():
    assert str(Decimal("2").ln()) == "0.6931471805599453094172321215"
    assert str(Decimal("1").exp()) == "2.718281828459045235360287471"
    assert str(Decimal("-1.5").exp()) == "0.2231301601484298289332804708"
    assert str(Decimal("100").log10()) == "2"
    assert str(Decimal("2").log10()) == "0.3010299956639811952137388947"
    assert str(Decimal("0.001").log10()) == "-3"
    assert str(Decimal("10").ln()) == "2.302585092994045684017991455"

test_transcendental()
