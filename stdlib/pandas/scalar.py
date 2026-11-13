# Ported from pandas 3.1.1 (BSD-3-Clause)

# Dynamically typed cell value. pandas columns have a runtime dtype, so a
# single element read from a Series/DataFrame is a Scalar that remembers
# whether it came from a float64, int64, bool or str column.

K_FLOAT: Literal[int] = 0
K_INT: Literal[int] = 1
K_BOOL: Literal[int] = 2
K_STR: Literal[int] = 3
K_NONE: Literal[int] = 4
K_DT: Literal[int] = 5     # datetime64: int64 nanoseconds since the epoch (NaT = int64 min)
K_TD: Literal[int] = 6     # timedelta64: int64 nanoseconds

NA_STR = "\x00<NA>"

from ._dt import NAT, Timestamp, Timedelta, fmt_ts_full, fmt_td

def _isnan(x: float) -> bool:
    return x != x

@tuple
class Scalar:
    kind: int
    f: float
    i: int
    s: str

    @staticmethod
    def none() -> Scalar:
        return Scalar(K_NONE, 0.0, 0, "")

    @staticmethod
    def of(x) -> Scalar:
        if isinstance(x, Scalar):
            return x
        elif x is None:
            return Scalar.none()
        elif isinstance(x, bool):
            return Scalar(K_BOOL, 0.0, 1 if x else 0, "")
        elif isinstance(x, int):
            return Scalar(K_INT, 0.0, x, "")
        elif isinstance(x, float):
            return Scalar(K_FLOAT, x, 0, "")
        elif isinstance(x, str):
            return Scalar(K_STR, 0.0, 0, x)
        elif isinstance(x, Timestamp):
            return Scalar(K_DT, 0.0, x.value, "")
        elif isinstance(x, Timedelta):
            return Scalar(K_TD, 0.0, x.value, "")
        elif isinstance(x, Optional):
            if x is None:
                return Scalar.none()
            return Scalar.of(x.__val__())
        elif hasattr(x, "__index__"):
            return Scalar(K_INT, 0.0, int(x), "")
        elif hasattr(x, "__float__"):
            return Scalar(K_FLOAT, float(x), 0, "")
        else:
            compile_error("pandas: unsupported scalar type")

    @property
    def is_none(self) -> bool:
        return self.kind == K_NONE

    @property
    def is_num(self) -> bool:
        return self.kind == K_FLOAT or self.kind == K_INT or self.kind == K_BOOL

    def isna(self) -> bool:
        if self.kind == K_DT or self.kind == K_TD:
            return self.i == NAT
        if self.kind == K_FLOAT:
            return _isnan(self.f)
        if self.kind == K_STR:
            return self.s == NA_STR
        return self.kind == K_NONE

    def num(self) -> float:
        if self.kind == K_DT or self.kind == K_TD:
            return float("nan") if self.i == NAT else float(self.i)
        if self.kind == K_FLOAT:
            return self.f
        if self.kind == K_INT or self.kind == K_BOOL:
            return float(self.i)
        if self.kind == K_NONE:
            return float("nan")
        raise TypeError(f"could not convert string to float: '{self.s}'")

    def __float__(self) -> float:
        if self.kind == K_STR:
            return float(self.s)
        return self.num()

    def __int__(self) -> int:
        if self.kind == K_FLOAT:
            return int(self.f)
        if self.kind == K_STR:
            return int(self.s)
        return self.i

    def __index__(self) -> int:
        return self.__int__()

    def __bool__(self) -> bool:
        if self.kind == K_FLOAT:
            return self.f != 0.0
        if self.kind == K_STR:
            return len(self.s) > 0
        return self.i != 0

    def __str__(self) -> str:
        if self.kind == K_DT:
            return "NaT" if self.i == NAT else fmt_ts_full(self.i)
        if self.kind == K_TD:
            return "NaT" if self.i == NAT else fmt_td(self.i)
        if self.kind == K_FLOAT:
            return str(self.f)
        if self.kind == K_INT:
            return str(self.i)
        if self.kind == K_BOOL:
            return "True" if self.i else "False"
        if self.kind == K_STR:
            return "nan" if self.s == NA_STR else self.s
        return "None"

    def __repr__(self) -> str:
        if self.kind == K_DT and self.i != NAT:
            return "Timestamp('" + fmt_ts_full(self.i) + "')"
        if self.kind == K_TD and self.i != NAT:
            return "Timedelta('" + fmt_td(self.i) + "')"
        if self.kind == K_STR and self.s != NA_STR:
            return repr(self.s)
        return self.__str__()

    def __format__(self, spec: str) -> str:
        if self.kind == K_FLOAT:
            return self.f.__format__(spec)
        if self.kind == K_INT:
            return self.i.__format__(spec)
        return self.__str__().__format__(spec)

    def __hash__(self) -> int:
        if self.kind == K_STR:
            return hash(self.s)
        if self.kind == K_NONE:
            return 0
        if self.kind == K_DT or self.kind == K_TD:
            return hash(self.i)
        if self.kind == K_FLOAT:
            if self.f != self.f:
                return 0x7ff8000000000000  # float.__hash__(nan) does not terminate
            if self.f == float(int(self.f)) if self.f == self.f and abs(self.f) < 9e18 else False:
                return hash(int(self.f))
            return hash(self.f)
        return hash(self.i)

    def _cmp(self, o: Scalar) -> int:
        # -2 means unordered (NaN or type mismatch)
        if self.kind == K_STR or o.kind == K_STR:
            if self.kind != o.kind:
                return -2
            return 0 if self.s == o.s else (-1 if self.s < o.s else 1)
        if self.kind == K_NONE or o.kind == K_NONE:
            return 0 if self.kind == o.kind else -2
        if (self.kind == K_DT or self.kind == K_TD) and self.kind == o.kind:
            if self.i == NAT or o.i == NAT:
                return -2
            return 0 if self.i == o.i else (-1 if self.i < o.i else 1)
        if self.kind != K_FLOAT and o.kind != K_FLOAT:
            return 0 if self.i == o.i else (-1 if self.i < o.i else 1)
        a = self.num()
        b = o.num()
        if a == b:
            return 0
        if a < b:
            return -1
        if a > b:
            return 1
        return -2

    def __eq__(self, o) -> bool:
        return self._cmp(Scalar.of(o)) == 0
    def __ne__(self, o) -> bool:
        return self._cmp(Scalar.of(o)) != 0
    def __lt__(self, o) -> bool:
        return self._cmp(Scalar.of(o)) == -1
    def __le__(self, o) -> bool:
        c = self._cmp(Scalar.of(o))
        return c == -1 or c == 0
    def __gt__(self, o) -> bool:
        return self._cmp(Scalar.of(o)) == 1
    def __ge__(self, o) -> bool:
        c = self._cmp(Scalar.of(o))
        return c == 1 or c == 0

    # arithmetic: with a Python float the result is a float; otherwise
    # the result keeps pandas/numpy promotion rules (int op int -> int)
    def _arith(self, o: Scalar, op: int) -> Scalar:
        if self.kind == K_DT or self.kind == K_TD or o.kind == K_DT or o.kind == K_TD:
            if self.i == NAT or o.i == NAT:
                return Scalar(self.kind if self.kind == K_DT or self.kind == K_TD else o.kind, 0.0, NAT, "")
            if self.kind == K_DT and o.kind == K_TD and (op == 0 or op == 1):
                return Scalar(K_DT, 0.0, self.i + o.i if op == 0 else self.i - o.i, "")
            if self.kind == K_TD and o.kind == K_DT and op == 0:
                return Scalar(K_DT, 0.0, self.i + o.i, "")
            if self.kind == K_DT and o.kind == K_DT and op == 1:
                return Scalar(K_TD, 0.0, self.i - o.i, "")
            if self.kind == K_TD and o.kind == K_TD and (op == 0 or op == 1):
                return Scalar(K_TD, 0.0, self.i + o.i if op == 0 else self.i - o.i, "")
            if self.kind == K_TD and o.kind == K_TD and op == 3:
                return Scalar(K_FLOAT, float(self.i) / float(o.i), 0, "")
            if self.kind == K_TD and (o.kind == K_INT or o.kind == K_FLOAT) and (op == 2 or op == 3):
                return Scalar(K_TD, 0.0, int(float(self.i) * o.num()) if op == 2 else int(float(self.i) / o.num()), "")
            raise TypeError("unsupported operand type(s) for datetime scalar")
        if self.kind == K_STR or o.kind == K_STR:
            if op == 0 and self.kind == o.kind:
                return Scalar(K_STR, 0.0, 0, self.s + o.s)
            raise TypeError("unsupported operand type(s) for str scalar")
        intres = self.kind != K_FLOAT and o.kind != K_FLOAT
        if intres and op != 3 and op != 6:
            a = self.i
            b = o.i
            r = 0
            if op == 0:
                r = a + b
            elif op == 1:
                r = a - b
            elif op == 2:
                r = a * b
            elif op == 4:
                r = a // b
            elif op == 5:
                r = a % b
            return Scalar(K_INT, 0.0, r, "")
        x = self.num()
        y = o.num()
        r = 0.0
        if op == 0:
            r = x + y
        elif op == 1:
            r = x - y
        elif op == 2:
            r = x * y
        elif op == 3:
            r = x / y if y != 0.0 else (float("nan") if x == 0.0 or x != x else (float("inf") if x > 0 else -float("inf")))
        elif op == 4:
            r = x // y
        elif op == 5:
            r = x % y
        elif op == 6:
            r = x ** y
        return Scalar(K_FLOAT, r, 0, "")

    def __add__(self, o: float) -> float:
        return self.num() + o
    def __radd__(self, o: float) -> float:
        return o + self.num()
    def __sub__(self, o: float) -> float:
        return self.num() - o
    def __rsub__(self, o: float) -> float:
        return o - self.num()
    def __mul__(self, o: float) -> float:
        return self.num() * o
    def __rmul__(self, o: float) -> float:
        return o * self.num()
    def __truediv__(self, o: float) -> float:
        return self.num() / o
    def __rtruediv__(self, o: float) -> float:
        return o / self.num()
    def __pow__(self, o: float) -> float:
        return self.num() ** o

    def __add__(self, o) -> Scalar:
        return self._arith(Scalar.of(o), 0)
    def __radd__(self, o) -> Scalar:
        return Scalar.of(o)._arith(self, 0)
    def __sub__(self, o) -> Scalar:
        return self._arith(Scalar.of(o), 1)
    def __rsub__(self, o) -> Scalar:
        return Scalar.of(o)._arith(self, 1)
    def __mul__(self, o) -> Scalar:
        return self._arith(Scalar.of(o), 2)
    def __rmul__(self, o) -> Scalar:
        return Scalar.of(o)._arith(self, 2)
    def __truediv__(self, o) -> Scalar:
        return self._arith(Scalar.of(o), 3)
    def __rtruediv__(self, o) -> Scalar:
        return Scalar.of(o)._arith(self, 3)
    def __floordiv__(self, o) -> Scalar:
        return self._arith(Scalar.of(o), 4)
    def __mod__(self, o) -> Scalar:
        return self._arith(Scalar.of(o), 5)
    def __pow__(self, o) -> Scalar:
        return self._arith(Scalar.of(o), 6)

    def __neg__(self) -> Scalar:
        if self.kind == K_FLOAT:
            return Scalar(K_FLOAT, -self.f, 0, "")
        return Scalar(K_INT, 0.0, -self.i, "")

    def __abs__(self) -> Scalar:
        if self.kind == K_FLOAT:
            return Scalar(K_FLOAT, abs(self.f), 0, "")
        return Scalar(K_INT, 0.0, abs(self.i), "")
