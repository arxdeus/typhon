# Ported from CPython 3.14 (PSF-2.0)

from math import frexp as _frexp

def _gcd(a: int, b: int) -> int:
    a = -a if a < 0 else a
    b = -b if b < 0 else b
    while b != 0:
        a, b = b, a % b
    return a

class Fraction:
    numerator: int
    denominator: int

    def __init__(self, numerator: int = 0, denominator: int = 1):
        if denominator == 0:
            raise ZeroDivisionError("Fraction(" + str(numerator) + ", 0)")
        g = _gcd(numerator, denominator)
        if denominator < 0:
            g = -g
        self.numerator = numerator // g if g != 0 else 0
        self.denominator = denominator // g if g != 0 else 1

    def __init__(self, value: float):
        n = 0
        d = 1
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("Cannot convert " + str(value) + " to Fraction")
        m, e = _frexp(value)
        k = 0
        while m != float(int(m)) and k < 1100:
            m *= 2.0
            k += 1
        n = int(m)
        e -= k
        d = 1
        if e >= 0:
            n *= 2 ** e
        else:
            d = 2 ** (-e)
        g = _gcd(n, d)
        self.numerator = n // g
        self.denominator = d // g

    def __init__(self, text: str):
        t = text.strip()
        if "/" in t:
            parts = t.split("/")
            n = int(parts[0])
            d = int(parts[1])
            if d == 0:
                raise ZeroDivisionError("Fraction(" + str(n) + ", 0)")
            g = _gcd(n, d)
            if d < 0:
                g = -g
            self.numerator = n // g
            self.denominator = d // g
        elif "." in t or "e" in t or "E" in t:
            neg = t.startswith("-")
            u = t.lstrip("+-")
            exp = 0
            if "e" in u or "E" in u:
                u = u.lower()
                p = u.split("e")
                u = p[0]
                exp = int(p[1])
            frac = 0
            if "." in u:
                q = u.split(".")
                frac = len(q[1])
                u = q[0] + q[1]
            n = int(u)
            d = 1
            e = exp - frac
            if e >= 0:
                n *= 10 ** e
            else:
                d = 10 ** (-e)
            if neg:
                n = -n
            g = _gcd(n, d)
            self.numerator = n // g
            self.denominator = d // g
        else:
            self.numerator = int(t)
            self.denominator = 1

    def __init__(self, other: Fraction):
        self.numerator = other.numerator
        self.denominator = other.denominator

    def __add__(self, o: Fraction) -> Fraction:
        return Fraction(self.numerator * o.denominator + o.numerator * self.denominator,
                        self.denominator * o.denominator)
    def __add__(self, o: int) -> Fraction:
        return Fraction(self.numerator + o * self.denominator, self.denominator)
    def __radd__(self, o: int) -> Fraction:
        return self + o
    def __sub__(self, o: Fraction) -> Fraction:
        return Fraction(self.numerator * o.denominator - o.numerator * self.denominator,
                        self.denominator * o.denominator)
    def __sub__(self, o: int) -> Fraction:
        return Fraction(self.numerator - o * self.denominator, self.denominator)
    def __rsub__(self, o: int) -> Fraction:
        return Fraction(o * self.denominator - self.numerator, self.denominator)
    def __mul__(self, o: Fraction) -> Fraction:
        return Fraction(self.numerator * o.numerator, self.denominator * o.denominator)
    def __mul__(self, o: int) -> Fraction:
        return Fraction(self.numerator * o, self.denominator)
    def __rmul__(self, o: int) -> Fraction:
        return self * o
    def __truediv__(self, o: Fraction) -> Fraction:
        if o.numerator == 0:
            raise ZeroDivisionError("Fraction(" + str(self.numerator * o.denominator) + ", 0)")
        return Fraction(self.numerator * o.denominator, self.denominator * o.numerator)
    def __truediv__(self, o: int) -> Fraction:
        return self / Fraction(o)
    def __rtruediv__(self, o: int) -> Fraction:
        return Fraction(o) / self
    def __floordiv__(self, o: Fraction) -> int:
        return (self.numerator * o.denominator) // (self.denominator * o.numerator)
    def __mod__(self, o: Fraction) -> Fraction:
        q = self // o
        return self - o * q
    def __pow__(self, n: int) -> Fraction:
        if n >= 0:
            return Fraction(self.numerator ** n, self.denominator ** n)
        if self.numerator == 0:
            raise ZeroDivisionError("Fraction(" + str(self.denominator ** (-n)) + ", 0)")
        return Fraction(self.denominator ** (-n), self.numerator ** (-n))
    def __neg__(self) -> Fraction:
        return Fraction(-self.numerator, self.denominator)
    def __pos__(self) -> Fraction:
        return self
    def __abs__(self) -> Fraction:
        return Fraction(abs(self.numerator), self.denominator)

    def __eq__(self, o: Fraction) -> bool:
        return self.numerator == o.numerator and self.denominator == o.denominator
    def __eq__(self, o: int) -> bool:
        return self.denominator == 1 and self.numerator == o
    def __ne__(self, o: Fraction) -> bool:
        return not (self == o)
    def __lt__(self, o: Fraction) -> bool:
        return self.numerator * o.denominator < o.numerator * self.denominator
    def __le__(self, o: Fraction) -> bool:
        return self.numerator * o.denominator <= o.numerator * self.denominator
    def __gt__(self, o: Fraction) -> bool:
        return o < self
    def __ge__(self, o: Fraction) -> bool:
        return o <= self
    def __lt__(self, o: int) -> bool:
        return self.numerator < o * self.denominator
    def __le__(self, o: int) -> bool:
        return self.numerator <= o * self.denominator
    def __gt__(self, o: int) -> bool:
        return self.numerator > o * self.denominator
    def __ge__(self, o: int) -> bool:
        return self.numerator >= o * self.denominator
    def __bool__(self) -> bool:
        return self.numerator != 0
    def __hash__(self) -> int:
        return hash(self.numerator) ^ (hash(self.denominator) * 31)
    def __float__(self) -> float:
        return self.numerator / self.denominator
    def __int__(self) -> int:
        if self.numerator < 0:
            return -((-self.numerator) // self.denominator)
        return self.numerator // self.denominator
    def __trunc__(self) -> int:
        return int(self)
    def __floor__(self) -> int:
        return self.numerator // self.denominator
    def __ceil__(self) -> int:
        return -((-self.numerator) // self.denominator)
    def __round__(self) -> int:
        n2 = 2 * self.numerator + self.denominator
        d2 = 2 * self.denominator
        q = n2 // d2
        if n2 % d2 == 0 and q % 2 != 0:
            q -= 1
        return q
    def __str__(self) -> str:
        if self.denominator == 1:
            return str(self.numerator)
        return str(self.numerator) + "/" + str(self.denominator)
    def __repr__(self) -> str:
        return "Fraction(" + str(self.numerator) + ", " + str(self.denominator) + ")"

    def as_integer_ratio(self) -> Tuple[int, int]:
        return (self.numerator, self.denominator)
    def is_integer(self) -> bool:
        return self.denominator == 1
    def limit_denominator(self, max_denominator: int = 1000000) -> Fraction:
        if max_denominator < 1:
            raise ValueError("max_denominator should be at least 1")
        if self.denominator <= max_denominator:
            return Fraction(self.numerator, self.denominator)
        p0, q0, p1, q1 = 0, 1, 1, 0
        n, d = self.numerator, self.denominator
        while True:
            a = n // d
            q2 = q0 + a * q1
            if q2 > max_denominator:
                break
            p0, q0, p1, q1 = p1, q1, p0 + a * p1, q2
            n, d = d, n - a * d
        k = (max_denominator - q0) // q1
        b1 = Fraction(p0 + k * p1, q0 + k * q1)
        b2 = Fraction(p1, q1)
        diff1 = abs(b1 - self)
        diff2 = abs(b2 - self)
        return b2 if diff2 <= diff1 else b1
