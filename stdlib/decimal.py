# Ported from CPython 3.14 _pydecimal (PSF license), reduced core:
# coefficients are Int[512] (<= ~150 digits), so context precision should stay <= 50.
# Supported: + - * / // % ** (int exponent), sqrt, quantize, to_integral*, compare, str/repr,
# int/float conversion, all rounding modes. Not supported: ln/exp/log10, signals/traps, Emin/Emax.

ROUND_DOWN = "ROUND_DOWN"
ROUND_HALF_UP = "ROUND_HALF_UP"
ROUND_HALF_EVEN = "ROUND_HALF_EVEN"
ROUND_CEILING = "ROUND_CEILING"
ROUND_FLOOR = "ROUND_FLOOR"
ROUND_UP = "ROUND_UP"
ROUND_HALF_DOWN = "ROUND_HALF_DOWN"
ROUND_05UP = "ROUND_05UP"

class DecimalException(ArithmeticError):
    pass

class InvalidOperation(DecimalException):
    pass

class DivisionByZero(DecimalException):
    pass

class Context:
    prec: int
    rounding: str

    def __init__(self, prec: int = 28, rounding: str = "ROUND_HALF_EVEN"):
        self.prec = prec
        self.rounding = rounding

_CTX = Context()

def getcontext() -> Context:
    return _CTX

def setcontext(c: Context):
    _CTX.prec = c.prec
    _CTX.rounding = c.rounding

def _p10(k: int) -> Int[512]:
    return Int[512](10) ** k

def _ndigits(c: Int[512]) -> int:
    if c == Int[512](0):
        return 1
    return len(str(c))

def _isqrt(n: Int[512]) -> Int[512]:
    if n <= Int[512](0):
        return Int[512](0)
    x = Int[512](1) << ((len(str(n)) * 4 + 3) // 1)
    while True:
        y = (x + n // x) >> 1
        if y >= x:
            return x
        x = y

def _round_coef(sign: int, coef: Int[512], exp: int, prec: int, rounding: str) -> Tuple[Int[512], int]:
    n = _ndigits(coef)
    if n <= prec:
        return (coef, exp)
    drop = n - prec
    d = _p10(drop)
    q = coef // d
    r = coef - q * d
    if r != Int[512](0):
        half = d // Int[512](2)
        inc = False
        if rounding == "ROUND_HALF_EVEN":
            inc = r > half or (r == half and (q % Int[512](2)) == Int[512](1))
        elif rounding == "ROUND_HALF_UP":
            inc = r >= half
        elif rounding == "ROUND_HALF_DOWN":
            inc = r > half
        elif rounding == "ROUND_UP":
            inc = True
        elif rounding == "ROUND_DOWN":
            inc = False
        elif rounding == "ROUND_CEILING":
            inc = sign == 0
        elif rounding == "ROUND_FLOOR":
            inc = sign == 1
        elif rounding == "ROUND_05UP":
            last = q % Int[512](10)
            inc = last == Int[512](0) or last == Int[512](5)
        if inc:
            q = q + Int[512](1)
            if _ndigits(q) > prec:
                q = q // Int[512](10)
                drop += 1
    return (q, exp + drop)

class Decimal:
    _sign: int
    _coef: Int[512]
    _exp: int
    _special: int  # 0 finite, 1 inf, 2 nan

    def __init__(self, sign: int, coef: Int[512], exp: int, special: int = 0):
        self._sign = sign
        self._coef = coef
        self._exp = exp
        self._special = special

    def __init__(self, v: str):
        d = Decimal._parse(v)
        self._sign = d._sign
        self._coef = d._coef
        self._exp = d._exp
        self._special = d._special

    def __init__(self, v: int):
        d = Decimal._from_int(v)
        self._sign = d._sign
        self._coef = d._coef
        self._exp = d._exp
        self._special = d._special

    def __init__(self, v: float):
        d = Decimal.from_float(v)
        self._sign = d._sign
        self._coef = d._coef
        self._exp = d._exp
        self._special = d._special

    @staticmethod
    def _from_int(v: int) -> Decimal:
        if v < 0:
            return Decimal(1, Int[512](-v), 0)
        return Decimal(0, Int[512](v), 0)

    @staticmethod
    def _parse(s: str) -> Decimal:
        t = s.strip().lower().replace("_", "")
        sign = 0
        if t.startswith("-"):
            sign = 1
            t = t[1:]
        elif t.startswith("+"):
            t = t[1:]
        if t == "inf" or t == "infinity":
            return Decimal(sign, Int[512](0), 0, 1)
        if t == "nan":
            return Decimal(sign, Int[512](0), 0, 2)
        e = 0
        pe = t.find("e")
        if pe >= 0:
            e = int(t[pe + 1:])
            t = t[:pe]
        dot = t.find(".")
        digits = t
        if dot >= 0:
            digits = t[:dot] + t[dot + 1:]
            e -= len(t) - dot - 1
        if len(digits) == 0:
            raise InvalidOperation("[<class 'decimal.ConversionSyntax'>]")
        for ch in digits:
            if not ch.isdigit():
                raise InvalidOperation("[<class 'decimal.ConversionSyntax'>]")
        return Decimal(sign, Int[512](int(digits)) if len(digits) < 19 else _big(digits), e)

    @staticmethod
    def from_str(s: str) -> Decimal:
        return Decimal._parse(s)

    @staticmethod
    def from_float(f: float) -> Decimal:
        if f != f:
            return Decimal(0, Int[512](0), 0, 2)
        if f == f * 2.0 and f != 0.0:
            return Decimal(0 if f > 0.0 else 1, Int[512](0), 0, 1)
        r = f.as_integer_ratio()
        num = r[0]
        den = r[1]
        sign = 0
        if num < 0 or (num == 0 and (1.0).cdiv(f) < 0.0 if f == 0.0 else False):
            sign = 1
            num = -num
        k = 0
        d = den
        while d > 1:
            d >>= 1
            k += 1
        # num / 2**k = num * 5**k / 10**k
        return Decimal(sign, Int[512](num) * (Int[512](5) ** k), -k)

    def _is_special(self) -> bool:
        return self._special != 0

    def is_nan(self) -> bool:
        return self._special == 2

    def is_infinite(self) -> bool:
        return self._special == 1

    def is_finite(self) -> bool:
        return self._special == 0

    def is_zero(self) -> bool:
        return self._special == 0 and self._coef == Int[512](0)

    def is_signed(self) -> bool:
        return self._sign == 1

    def adjusted(self) -> int:
        return self._exp + _ndigits(self._coef) - 1

    def copy_abs(self) -> Decimal:
        return Decimal(0, self._coef, self._exp, self._special)

    def copy_negate(self) -> Decimal:
        return Decimal(1 - self._sign, self._coef, self._exp, self._special)

    def as_tuple(self) -> Tuple[int, List[int], int]:
        ds = List[int]()
        for ch in str(self._coef):
            ds.append(ord(ch) - 48)
        return (self._sign, ds, self._exp)

    def _fix(self) -> Decimal:
        if self._special != 0:
            return self
        p = _CTX.prec
        r = _round_coef(self._sign, self._coef, self._exp, p, _CTX.rounding)
        return Decimal(self._sign, r[0], r[1])

    def __pos__(self) -> Decimal:
        return self._fix()

    def __neg__(self) -> Decimal:
        return Decimal(1 - self._sign, self._coef, self._exp, self._special)._fix()

    def __abs__(self) -> Decimal:
        return Decimal(0, self._coef, self._exp, self._special)._fix()

    def __bool__(self) -> bool:
        return not self.is_zero()

    def __str__(self) -> str:
        sg = "-" if self._sign == 1 else ""
        if self._special == 1:
            return sg + "Infinity"
        if self._special == 2:
            return "NaN"
        ds = str(self._coef)
        left = self._exp + len(ds)
        if self._exp <= 0 and left > -6:
            if self._exp == 0:
                return sg + ds
            if left > 0:
                return sg + ds[:left] + "." + ds[left:]
            return sg + "0." + ("0" * (-left)) + ds
        s = ds[0]
        if len(ds) > 1:
            s += "." + ds[1:]
        ex = left - 1
        return sg + s + "E" + ("+" if ex >= 0 else "-") + str(abs(ex))

    def __repr__(self) -> str:
        return "Decimal('" + self.__str__() + "')"

    def __int__(self) -> int:
        if self._special != 0:
            raise ValueError("cannot convert to integer")
        v = self._coef
        if self._exp >= 0:
            v = v * _p10(self._exp)
        else:
            v = v // _p10(-self._exp)
        r = int(v)
        return -r if self._sign == 1 else r

    def __float__(self) -> float:
        if self._special == 2:
            return (0.0).cdiv(0.0)
        if self._special == 1:
            return (-1.0 if self._sign == 1 else 1.0).cdiv(0.0)
        return float(str(self))

    def _aligned(self, o: Decimal) -> Tuple[Int[512], Int[512], int]:
        e = self._exp if self._exp < o._exp else o._exp
        a = self._coef * _p10(self._exp - e)
        b = o._coef * _p10(o._exp - e)
        return (a, b, e)

    def __add__(self, o: Decimal) -> Decimal:
        if self._special == 2 or o._special == 2:
            return Decimal(0, Int[512](0), 0, 2)
        if self._special == 1:
            if o._special == 1 and o._sign != self._sign:
                raise InvalidOperation("[<class 'decimal.InvalidOperation'>]")
            return self
        if o._special == 1:
            return o
        p = _CTX.prec
        x = self
        y = o
        # sticky handling for very different exponents
        if x._coef != Int[512](0) and y._coef != Int[512](0):
            hx = x.adjusted()
            hy = y.adjusted()
            if hx - hy > p + 3 and x._exp - y._exp > 90:
                y = Decimal(y._sign, Int[512](1), hx - p - 3 - 1 if hx - p - 4 < x._exp else x._exp - 1)
            elif hy - hx > p + 3 and y._exp - x._exp > 90:
                x = Decimal(x._sign, Int[512](1), hy - p - 3 - 1 if hy - p - 4 < y._exp else y._exp - 1)
        e = x._exp if x._exp < y._exp else y._exp
        if x._exp - e > 100 or y._exp - e > 100:
            if x._exp > y._exp:
                e = x._exp - 100
            else:
                e = y._exp - 100
        a = x._coef * _p10(x._exp - e)
        b = y._coef * _p10(y._exp - e)
        if x._sign == 1:
            a = -a
        if y._sign == 1:
            b = -b
        s = a + b
        sign = 0
        if s < Int[512](0):
            sign = 1
            s = -s
        elif s == Int[512](0):
            if x._sign == 1 and y._sign == 1:
                sign = 1
            elif _CTX.rounding == "ROUND_FLOOR" and x._sign != y._sign:
                sign = 1
        r = _round_coef(sign, s, e, p, _CTX.rounding)
        return Decimal(sign, r[0], r[1])

    def __sub__(self, o: Decimal) -> Decimal:
        return self.__add__(Decimal(1 - o._sign, o._coef, o._exp, o._special))

    def __mul__(self, o: Decimal) -> Decimal:
        sign = self._sign ^ o._sign
        if self._special == 2 or o._special == 2:
            return Decimal(0, Int[512](0), 0, 2)
        if self._special == 1 or o._special == 1:
            if self.is_zero() or o.is_zero():
                raise InvalidOperation("[<class 'decimal.InvalidOperation'>]")
            return Decimal(sign, Int[512](0), 0, 1)
        r = _round_coef(sign, self._coef * o._coef, self._exp + o._exp, _CTX.prec, _CTX.rounding)
        return Decimal(sign, r[0], r[1])

    def __truediv__(self, o: Decimal) -> Decimal:
        sign = self._sign ^ o._sign
        if self._special == 2 or o._special == 2:
            return Decimal(0, Int[512](0), 0, 2)
        if self._special == 1:
            if o._special == 1:
                raise InvalidOperation("[<class 'decimal.InvalidOperation'>]")
            return Decimal(sign, Int[512](0), 0, 1)
        if o._special == 1:
            return Decimal(sign, Int[512](0), 0)
        if o.is_zero():
            if self.is_zero():
                raise InvalidOperation("[<class 'decimal.DivisionUndefined'>]")
            raise DivisionByZero("[<class 'decimal.DivisionByZero'>]")
        p = _CTX.prec
        ideal = self._exp - o._exp
        if self.is_zero():
            return Decimal(sign, Int[512](0), ideal)
        shift = p + 3 + _ndigits(o._coef) - _ndigits(self._coef)
        if shift < 0:
            shift = 0
        num = self._coef * _p10(shift)
        q = num // o._coef
        rem = num - q * o._coef
        e = ideal - shift
        if rem != Int[512](0):
            q = q * Int[512](10) + Int[512](1)
            e -= 1
            r = _round_coef(sign, q, e, p, _CTX.rounding)
            return Decimal(sign, r[0], r[1])
        r = _round_coef(sign, q, e, p, _CTX.rounding)
        c = r[0]
        ex = r[1]
        while ex < ideal and c % Int[512](10) == Int[512](0):
            c = c // Int[512](10)
            ex += 1
        return Decimal(sign, c, ex)

    def _divint(self, o: Decimal) -> Tuple[Int[512], int, Decimal]:
        # truncated integer quotient (sign applied separately) and remainder
        if o.is_zero():
            raise DivisionByZero("[<class 'decimal.DivisionByZero'>]")
        sign = self._sign ^ o._sign
        a = self._coef
        b = o._coef
        e = self._exp if self._exp < o._exp else o._exp
        a = a * _p10(self._exp - e)
        b = b * _p10(o._exp - e)
        q = a // b
        if _ndigits(q) > _CTX.prec:
            raise InvalidOperation("[<class 'decimal.DivisionImpossible'>]")
        r = a - q * b
        return (q, sign, Decimal(self._sign, r, e))

    def __floordiv__(self, o: Decimal) -> Decimal:
        t = self._divint(o)
        return Decimal(t[1], t[0], 0)

    def __mod__(self, o: Decimal) -> Decimal:
        t = self._divint(o)
        r = t[2]
        if r.is_zero():
            return Decimal(self._sign, Int[512](0), r._exp)
        return r

    def __divmod__(self, o: Decimal) -> Tuple[Decimal, Decimal]:
        return (self.__floordiv__(o), self.__mod__(o))

    def __pow__(self, n: int) -> Decimal:
        if n == 0:
            return Decimal(0, Int[512](1), 0)
        neg = n < 0
        m = -n if neg else n
        sign = self._sign if m % 2 == 1 else 0
        if _ndigits(self._coef) * m <= 140:
            c = self._coef ** m
            r = _round_coef(sign, c, self._exp * m, _CTX.prec, _CTX.rounding)
            res = Decimal(sign, r[0], r[1])
        else:
            saved = _CTX.prec
            _CTX.prec = saved + 6
            res = Decimal(0, Int[512](1), 0)
            base = self
            k = m
            while k > 0:
                if k & 1:
                    res = res * base
                base = base * base
                k >>= 1
            _CTX.prec = saved
            res = res._fix()
        if neg:
            return Decimal(0, Int[512](1), 0).__truediv__(res)
        return res

    def sqrt(self) -> Decimal:
        if self._special == 2:
            return self
        if self._special == 1:
            if self._sign == 1:
                raise InvalidOperation("[<class 'decimal.InvalidOperation'>]")
            return self
        if self.is_zero():
            return Decimal(self._sign, Int[512](0), self._exp // 2)
        if self._sign == 1:
            raise InvalidOperation("[<class 'decimal.InvalidOperation'>]")
        p = _CTX.prec
        c = self._coef
        e = self._exp
        if e % 2 != 0:
            c = c * Int[512](10)
            e -= 1
        s = 0
        nd = _ndigits(c)
        need = 2 * (p + 2)
        if nd < need:
            s = (need - nd + 1) // 2
            c = c * _p10(2 * s)
            e -= 2 * s
        r = _isqrt(c)
        ee = e // 2
        ideal = self._exp // 2
        if r * r != c:
            r = r * Int[512](10) + Int[512](1)
            ee -= 1
            q = _round_coef(0, r, ee, p, _CTX.rounding)
            return Decimal(0, q[0], q[1])
        q = _round_coef(0, r, ee, p, _CTX.rounding)
        co = q[0]
        ex = q[1]
        while ex < ideal and co % Int[512](10) == Int[512](0):
            co = co // Int[512](10)
            ex += 1
        return Decimal(0, co, ex)

    def _cmp(self, o: Decimal) -> int:
        # -1, 0, 1 ; NaN not handled
        if self.is_zero() and o.is_zero():
            return 0
        sa = -1 if self._sign == 1 else 1
        sb = -1 if o._sign == 1 else 1
        if self._special == 1 or o._special == 1:
            va = 0 if self._special == 0 else sa
            vb = 0 if o._special == 0 else sb
            if self._special == 1 and o._special == 1:
                return 0 if sa == sb else (1 if sa > sb else -1)
            if self._special == 1:
                return sa
            return -sb
        if self.is_zero():
            return -sb
        if o.is_zero():
            return sa
        if sa != sb:
            return 1 if sa > sb else -1
        ha = self.adjusted()
        hb = o.adjusted()
        if ha != hb:
            r = 1 if ha > hb else -1
            return r * sa
        t = self._aligned(o)
        if t[0] == t[1]:
            return 0
        r = 1 if t[0] > t[1] else -1
        return r * sa

    def compare(self, o: Decimal) -> Decimal:
        if self._special == 2 or o._special == 2:
            return Decimal(0, Int[512](0), 0, 2)
        c = self._cmp(o)
        return Decimal(1 if c < 0 else 0, Int[512](0 if c == 0 else 1), 0)

    def __eq__(self, o: Decimal) -> bool:
        if self._special == 2 or o._special == 2:
            return False
        return self._cmp(o) == 0

    def __ne__(self, o: Decimal) -> bool:
        return not self.__eq__(o)

    def __lt__(self, o: Decimal) -> bool:
        if self._special == 2 or o._special == 2:
            return False
        return self._cmp(o) < 0

    def __le__(self, o: Decimal) -> bool:
        if self._special == 2 or o._special == 2:
            return False
        return self._cmp(o) <= 0

    def __gt__(self, o: Decimal) -> bool:
        if self._special == 2 or o._special == 2:
            return False
        return self._cmp(o) > 0

    def __ge__(self, o: Decimal) -> bool:
        if self._special == 2 or o._special == 2:
            return False
        return self._cmp(o) >= 0

    def __add__(self, o: int) -> Decimal:
        return self.__add__(Decimal._from_int(o))

    def __radd__(self, o: int) -> Decimal:
        return Decimal._from_int(o).__add__(self)

    def __sub__(self, o: int) -> Decimal:
        return self.__sub__(Decimal._from_int(o))

    def __rsub__(self, o: int) -> Decimal:
        return Decimal._from_int(o).__sub__(self)

    def __mul__(self, o: int) -> Decimal:
        return self.__mul__(Decimal._from_int(o))

    def __rmul__(self, o: int) -> Decimal:
        return Decimal._from_int(o).__mul__(self)

    def __truediv__(self, o: int) -> Decimal:
        return self.__truediv__(Decimal._from_int(o))

    def __rtruediv__(self, o: int) -> Decimal:
        return Decimal._from_int(o).__truediv__(self)

    def __eq__(self, o: int) -> bool:
        return self.__eq__(Decimal._from_int(o))

    def __lt__(self, o: int) -> bool:
        return self.__lt__(Decimal._from_int(o))

    def __gt__(self, o: int) -> bool:
        return self.__gt__(Decimal._from_int(o))

    def _to_integral(self, rounding: str) -> Decimal:
        if self._special != 0 or self._exp >= 0:
            return self
        drop = -self._exp
        nd = _ndigits(self._coef)
        d = _p10(drop) if drop <= 150 else Int[512](0)
        if drop > nd + 1:
            q = Int[512](0)
            r = self._coef
            inexact = r != Int[512](0)
            inc = False
            if inexact:
                if rounding == "ROUND_UP" or (rounding == "ROUND_CEILING" and self._sign == 0) or (rounding == "ROUND_FLOOR" and self._sign == 1):
                    inc = True
            return Decimal(self._sign, Int[512](1 if inc else 0), 0)
        q = self._coef // d
        r = self._coef - q * d
        if r != Int[512](0):
            half = d // Int[512](2)
            inc = False
            if rounding == "ROUND_HALF_EVEN":
                inc = r > half or (r == half and (q % Int[512](2)) == Int[512](1))
            elif rounding == "ROUND_HALF_UP":
                inc = r >= half
            elif rounding == "ROUND_HALF_DOWN":
                inc = r > half
            elif rounding == "ROUND_UP":
                inc = True
            elif rounding == "ROUND_CEILING":
                inc = self._sign == 0
            elif rounding == "ROUND_FLOOR":
                inc = self._sign == 1
            elif rounding == "ROUND_05UP":
                last = q % Int[512](10)
                inc = last == Int[512](0) or last == Int[512](5)
            if inc:
                q = q + Int[512](1)
        return Decimal(self._sign, q, 0)

    def to_integral_value(self, rounding: str = "") -> Decimal:
        return self._to_integral(rounding if rounding != "" else _CTX.rounding)

    def to_integral(self, rounding: str = "") -> Decimal:
        return self.to_integral_value(rounding)

    def to_integral_exact(self, rounding: str = "") -> Decimal:
        return self.to_integral_value(rounding)

    def quantize(self, exp: Decimal, rounding: str = "") -> Decimal:
        rm = rounding if rounding != "" else _CTX.rounding
        target = exp._exp
        if self._special != 0:
            return self
        if self._exp >= target:
            sh = self._exp - target
            c = self._coef * _p10(sh)
            if _ndigits(c) > _CTX.prec:
                raise InvalidOperation("[<class 'decimal.InvalidOperation'>]")
            return Decimal(self._sign, c, target)
        t = Decimal(self._sign, self._coef, self._exp - target)
        r = t._to_integral(rm)
        if _ndigits(r._coef) > _CTX.prec:
            raise InvalidOperation("[<class 'decimal.InvalidOperation'>]")
        return Decimal(self._sign, r._coef, target)

    def normalize(self) -> Decimal:
        if self._special != 0:
            return self
        f = self._fix()
        if f._coef == Int[512](0):
            return Decimal(f._sign, Int[512](0), 0)
        c = f._coef
        e = f._exp
        while c % Int[512](10) == Int[512](0):
            c = c // Int[512](10)
            e += 1
        return Decimal(f._sign, c, e)

def _big(digits: str) -> Int[512]:
    r = Int[512](0)
    for ch in digits:
        r = r * Int[512](10) + Int[512](ord(ch) - 48)
    return r

def D(s: str) -> Decimal:
    return Decimal._parse(s)

# ---- transcendental functions (correctly rounded via guard digits + Ziv-style retry) ----

def _scaled(d: Decimal, digits: int) -> Int[512]:
    # d * 10**digits as integer (truncated)
    e = d._exp + digits
    if e >= 0:
        return d._coef * _p10(e)
    return d._coef // _p10(-e)

def _ln_fixed(x: Int[512], scale: int) -> Int[512]:
    # ln(x / 10**scale) * 10**scale, x > 0. Reduce to [0.5, 2) by powers of 2? Use atanh series on m=(y-1)/(y+1).
    one = _p10(scale)
    k = 0
    # bring x into [1, 10) * 10**k to use ln(10) multiples
    ln10 = _ln10_fixed(scale)
    while x >= one * Int[512](10):
        x = x // Int[512](10)
        k += 1
    while x < one:
        x = x * Int[512](10)
        k -= 1
    # now x in [1,10); reduce further by sqrt-free halving: divide by 2 until < 1.5
    twos = 0
    two = Int[512](2)
    while x >= one + one // two:
        x = x // two
        twos += 1
    ln2 = _ln2_fixed(scale)
    # atanh series: ln(y) = 2 * sum m^(2i+1)/(2i+1), m=(y-1)/(y+1)
    m = (x - one) * one // (x + one)
    msign = False
    if m < Int[512](0):
        # x < one: use |m| = (one - x) / (one + x)
        m = (one - x) * one // (x + one)
        msign = True
    m2 = m * m // one
    term = m
    s = Int[512](0)
    i = 1
    while term != Int[512](0):
        s += term // Int[512](i)
        term = term * m2 // one
        i += 2
    if msign:
        s = -s
    r = s * two + Int[512](twos) * ln2 + Int[512](k) * ln10
    return r

def _atanh_inv(n: int, scale: int) -> Int[512]:
    # atanh(1/n) * 10**scale
    one = _p10(scale)
    x = one // Int[512](n)
    n2 = Int[512](n * n)
    s = Int[512](0)
    i = 1
    while x != Int[512](0):
        s += x // Int[512](i)
        x = x // n2
        i += 2
    return s

def _ln2_fixed(scale: int) -> Int[512]:
    # ln2 = 2*atanh(1/3)
    return _atanh_inv(3, scale) * Int[512](2)

def _ln10_fixed(scale: int) -> Int[512]:
    # ln10 = 3 ln2 + ln(1.25) ; ln(1.25) = 2*atanh(1/9)
    return _ln2_fixed(scale) * Int[512](3) + _atanh_inv(9, scale) * Int[512](2)

def _exp_fixed(x: Int[512], scale: int, neg: bool) -> Int[512]:
    # exp(+-x/10**scale) * 10**scale ; reduce argument by halving k times then squaring
    one = _p10(scale)
    k = 0
    while x > one // Int[512](8):
        x = x // Int[512](2)
        k += 1
    s = one
    term = one
    i = 1
    while term != Int[512](0):
        term = term * x // one // Int[512](i)
        s += term
        i += 1
    for _ in range(k):
        s = s * s // one
    if neg:
        return one * one // s
    return s

def _round_to_prec(sign: int, num: Int[512], scale: int) -> Decimal:
    r = _round_coef(sign, num, -scale, _CTX.prec, _CTX.rounding)
    return Decimal(sign, r[0], r[1])

def _trailing_trim(d: Decimal, ideal: int) -> Decimal:
    c = d._coef
    e = d._exp
    while e < ideal and c % Int[512](10) == Int[512](0) and c != Int[512](0):
        c = c // Int[512](10)
        e += 1
    return Decimal(d._sign, c, e)

def _dec_ln(x: Decimal) -> Decimal:
    if x._special == 2:
        return x
    if x._sign == 1 and not x.is_zero():
        raise InvalidOperation("[<class 'decimal.InvalidOperation'>]")
    if x.is_zero():
        return Decimal(1, Int[512](0), 0, 1)
    if x._special == 1:
        return x
    if x._coef == Int[512](1) and x._exp == 0:
        return Decimal(0, Int[512](0), 0)
    p = _CTX.prec
    guard = 6
    for _ in range(4):
        scale = p + guard + 4 + (len(str(abs(x.adjusted()))) if x.adjusted() != 0 else 0)
        # x = coef * 10**exp ; ln x = ln(coef/10**(nd-1)) + (adjusted)*ln10
        nd = _ndigits(x._coef)
        xs = x._coef * _p10(scale) // _p10(nd - 1)
        r = _ln_fixed(xs, scale) + Int[512](x.adjusted()) * _ln10_fixed(scale)
        sign = 0
        if r < Int[512](0):
            sign = 1
            r = -r
        # check rounding stability: evaluate at precision p with +-error band
        err = Int[512](10) ** 3
        lo = _round_to_prec(sign, r - err if r > err else Int[512](0), scale)
        hi = _round_to_prec(sign, r + err, scale)
        if lo._coef == hi._coef and lo._exp == hi._exp:
            res = _round_to_prec(sign, r, scale)
            return res
        guard += 6
    return _round_to_prec(sign, r, scale)

def _dec_exp(x: Decimal) -> Decimal:
    if x._special == 2:
        return x
    if x._special == 1:
        if x._sign == 1:
            return Decimal(0, Int[512](0), 0)
        return x
    if x.is_zero():
        return Decimal(0, Int[512](1), 0)
    p = _CTX.prec
    guard = 6
    for _ in range(4):
        adj = x.adjusted()
        extra = adj + 1 if adj >= 0 else 0
        scale = p + guard + 4 + extra
        xs = _scaled(x, scale)
        if xs < Int[512](0):
            xs = -xs
        r = _exp_fixed(xs, scale, x._sign == 1)
        err = Int[512](10) ** 3
        lo = _round_to_prec(0, r - err, scale)
        hi = _round_to_prec(0, r + err, scale)
        if lo._coef == hi._coef and lo._exp == hi._exp:
            return _round_to_prec(0, r, scale)
        guard += 6
    return _round_to_prec(0, r, scale)

def _dec_log10(x: Decimal) -> Decimal:
    if x._special == 2:
        return x
    if x._sign == 1 and not x.is_zero():
        raise InvalidOperation("[<class 'decimal.InvalidOperation'>]")
    if x.is_zero():
        return Decimal(1, Int[512](0), 0, 1)
    if x._special == 1:
        return x
    # exact power of ten
    c = x._coef
    e = x._exp
    while c % Int[512](10) == Int[512](0):
        c = c // Int[512](10)
        e += 1
    if c == Int[512](1):
        v = Decimal._from_int(e)
        return v._fix()
    saved = _CTX.prec
    _CTX.prec = saved + 8
    l = _dec_ln(x)
    t = _dec_ln(Decimal(0, Int[512](10), 0))
    q = l.__truediv__(t)
    _CTX.prec = saved
    return q._fix()

@extend
class Decimal:
    def ln(self) -> Decimal:
        return _dec_ln(self)

    def exp(self) -> Decimal:
        return _dec_exp(self)

    def log10(self) -> Decimal:
        return _dec_log10(self)

@extend
class Decimal:
    def __pow__(self, y: Decimal) -> Decimal:
        # integral exponent: exact repeated multiplication path
        if y._special == 0:
            yi = y.to_integral_value("ROUND_DOWN")
            if yi == y and _ndigits(yi._coef) + (yi._exp if yi._exp > 0 else 0) <= 15:
                return self.__pow__(int(yi))
        if self._special != 0 or y._special != 0:
            raise InvalidOperation("[<class 'decimal.InvalidOperation'>]")
        if self._sign == 1 and not self.is_zero():
            raise InvalidOperation("[<class 'decimal.InvalidOperation'>]")
        if self.is_zero():
            return Decimal(0, Int[512](0), 0)
        saved = _CTX.prec
        _CTX.prec = saved + 12
        r = (y * _dec_ln(self)).exp()
        _CTX.prec = saved
        return r._fix()
