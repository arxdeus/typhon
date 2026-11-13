from decimal import Decimal, localcontext, getcontext, ROUND_DOWN, ROUND_UP, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_DOWN, ROUND_05UP

@test
def test_decimal_context():
    with localcontext(prec=5):
        assert str(Decimal(1) / Decimal(3)) == "0.33333"
    assert str(Decimal(1) / Decimal(3)) == "0.3333333333333333333333333333"
    q = Decimal("0.01")
    x = Decimal("2.675")
    assert str(x.quantize(q, ROUND_DOWN)) == "2.67"
    assert str(x.quantize(q, ROUND_UP)) == "2.68"
    assert str(x.quantize(q, ROUND_HALF_DOWN)) == "2.67"
    assert str(x.quantize(q, ROUND_CEILING)) == "2.68"
    assert str(x.quantize(q, ROUND_FLOOR)) == "2.67"
    assert str(Decimal("-2.675").quantize(q, ROUND_FLOOR)) == "-2.68"
    assert str(Decimal("2.601").quantize(q, ROUND_05UP)) == "2.60"
test_decimal_context()
