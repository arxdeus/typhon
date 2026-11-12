from fractions import Fraction

@test
def test_fractions():
    a = Fraction(1, 2)
    b = Fraction(3, 4)
    assert str(a + b) == "5/4"
    assert str(a - b) == "-1/4"
    assert str(a * b) == "3/8"
    assert str(a / b) == "2/3"
    assert repr(Fraction(6, -4)) == "Fraction(-3, 2)"
    assert str(Fraction(4, 2)) == "2"
    assert Fraction(1, 2) == Fraction(2, 4)
    assert a < b and b > a and a <= a and a >= a
    assert Fraction("3/6") == a
    assert str(Fraction("1.25")) == "5/4"
    assert str(Fraction("-1.5e2")) == "-150"
    assert str(Fraction(0.75)) == "3/4"
    assert float(b) == 0.75
    assert int(Fraction(-7, 2)) == -3
    assert Fraction(-7, 2).__floor__() == -4
    assert Fraction(-7, 2).__ceil__() == -3
    assert Fraction(5, 2).__round__() == 2
    assert Fraction(7, 2).__round__() == 4
    assert str(a ** 3) == "1/8"
    assert str(a ** -2) == "4"
    assert str(a + 1) == "3/2"
    assert str(1 - a) == "1/2"
    assert str(Fraction(3.141592653589793).limit_denominator(1000)) == "355/113"
    assert str(-b) == "-3/4"
    assert str(abs(Fraction(-1, 3))) == "1/3"
    assert Fraction(1, 3).as_integer_ratio() == (1, 3)
    try:
        Fraction(1, 0)
        assert False
    except ZeroDivisionError:
        pass
test_fractions()
