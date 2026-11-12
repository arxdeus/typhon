# Ported from numpy 2.5.4 (BSD-3-Clause)
#
# Generic series class shared by Polynomial, Chebyshev, Legendre, Hermite,
# HermiteE and Laguerre (numpy's ABCPolyBase). The family is the static
# parameter K (see polyutils).

import numpy as np
from .polyutils import _add, _sub, _mul, _div, _pow, _val, _der, _int, _fit, \
    _roots, _fromroots, _line, mapparms, getdomain, trimcoef, as_series

_SUB = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]
_SUP = ["⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹"]

def _map_digits(s: str, table: List[str]) -> str:
    out = ""
    for ch in s:
        out += table[ord(ch) - 48]
    return out

def _fmt(x: float, parens: bool = False) -> str:
    if x == float(int(x)) and abs(x) < 1e16:
        s = str(float(int(x)))
    else:
        s = np.format_float_positional(x, precision=8, unique=True, trim='-') if abs(x) >= 1e-4 and abs(x) < 1e16 else str(x)
    if parens and (s.startswith("-") or "e" in s):
        s = "(" + s + ")"
    return s

def _arr_repr(a) -> str:
    return repr(a)

def _default_domain(K: Literal[int]):
    if K == 5:
        return np.array([0.0, 1.0])
    return np.array([-1.0, 1.0])

class ABCPolyBase[K: Literal[int]]:
    coef: np.ndarray[float, 1]
    domain: np.ndarray[float, 1]
    window: np.ndarray[float, 1]
    symbol: str

    def __init__(self, coef, domain = None, window = None, symbol: str = 'x'):
        self.coef = as_series(coef)
        self.domain = _default_domain(K) if domain is None else np.asarray(domain, dtype=float).copy()
        self.window = _default_domain(K) if window is None else np.asarray(window, dtype=float).copy()
        if self.domain.size != 2:
            raise ValueError("Domain has wrong number of elements.")
        if self.window.size != 2:
            raise ValueError("Window has wrong number of elements.")
        self.symbol = symbol

    @property
    def basis_name(self) -> str:
        if K == 0:
            return ""
        elif K == 1:
            return "T"
        elif K == 2:
            return "P"
        elif K == 3:
            return "H"
        elif K == 4:
            return "He"
        else:
            return "L"

    @property
    def _name(self) -> str:
        if K == 0:
            return "Polynomial"
        elif K == 1:
            return "Chebyshev"
        elif K == 2:
            return "Legendre"
        elif K == 3:
            return "Hermite"
        elif K == 4:
            return "HermiteE"
        else:
            return "Laguerre"

    def _new(self, coef):
        return ABCPolyBase[K](coef, self.domain, self.window, self.symbol)

    def has_samecoef(self, other) -> bool:
        return self.coef.size == other.coef.size and bool(np.all(self.coef == other.coef))

    def has_samedomain(self, other) -> bool:
        return bool(np.all(self.domain == other.domain))

    def has_samewindow(self, other) -> bool:
        return bool(np.all(self.window == other.window))

    def _get_coefficients(self, other):
        return as_series(other)

    def _get_coefficients(self, other: ABCPolyBase[K]):
        if not self.has_samedomain(other):
            raise TypeError("Domains differ")
        if not self.has_samewindow(other):
            raise TypeError("Windows differ")
        if self.symbol != other.symbol:
            raise ValueError("Polynomial symbols differ")
        return other.coef

    def __len__(self) -> int:
        return self.coef.size

    def __eq__(self, other: ABCPolyBase[K]) -> bool:
        return (self.has_samedomain(other) and self.has_samewindow(other)
                and self.has_samecoef(other) and self.symbol == other.symbol)

    def __ne__(self, other: ABCPolyBase[K]) -> bool:
        return not (self == other)

    def __neg__(self):
        return self._new(-self.coef)

    def __pos__(self):
        return self

    def __add__(self, other):
        return self._new(_add(self.coef, self._get_coefficients(other)))

    def __sub__(self, other):
        return self._new(_sub(self.coef, self._get_coefficients(other)))

    def __mul__(self, other):
        return self._new(_mul(K, self.coef, self._get_coefficients(other)))

    def __truediv__(self, other):
        if isinstance(other, ABCPolyBase):
            raise TypeError("unsupported types for true division")
        return self._new(self.coef / float(other))

    def __floordiv__(self, other):
        return self._new(_div(K, self.coef, self._get_coefficients(other))[0])

    def __mod__(self, other):
        return self._new(_div(K, self.coef, self._get_coefficients(other))[1])

    def __divmod__(self, other):
        q, r = _div(K, self.coef, self._get_coefficients(other))
        return self._new(q), self._new(r)

    def __pow__(self, n: int):
        return self._new(_pow(K, self.coef, n))

    def __radd__(self, other):
        return self._new(_add(as_series(other), self.coef))

    def __rsub__(self, other):
        return self._new(_sub(as_series(other), self.coef))

    def __rmul__(self, other):
        return self._new(_mul(K, as_series(other), self.coef))

    def __rfloordiv__(self, other):
        return self._new(_div(K, as_series(other), self.coef)[0])

    def __rmod__(self, other):
        return self._new(_div(K, as_series(other), self.coef)[1])

    def mapparms(self):
        return mapparms(self.domain, self.window)

    def __call__(self, arg):
        off, scl = self.mapparms()
        if isinstance(arg, list) or isinstance(arg, tuple):
            return _val(K, off + scl * np.asarray(arg, dtype=float), self.coef)
        else:
            return _val(K, off + scl * arg, self.coef)

    def degree(self) -> int:
        return self.coef.size - 1

    def copy(self):
        return self._new(self.coef.copy())

    def cutdeg(self, deg: int):
        return self.truncate(deg + 1)

    def trim(self, tol: float = 0.0):
        return self._new(trimcoef(self.coef, tol))

    def truncate(self, size: int):
        if size < 1:
            raise ValueError("size must be a positive integer")
        if size >= self.coef.size:
            return self._new(self.coef.copy())
        return self._new(self.coef[:size].copy())

    def deriv(self, m: int = 1):
        off, scl = self.mapparms()
        return self._new(_der(K, self.coef, m, scl))

    def integ(self, m: int = 1, k = None, lbnd = None):
        off, scl = self.mapparms()
        lb = 0.0
        if lbnd is not None:
            lb = off + scl * float(lbnd)
        return self._new(_int(K, self.coef, m, k, lb, 1.0 / scl))

    def roots(self):
        r = _roots(K, self.coef)
        off, scl = mapparms(self.window, self.domain)
        return off + scl * r

    def linspace(self, n: int = 100, domain = None):
        d = self.domain if domain is None else np.asarray(domain, dtype=float)
        x = np.linspace(d[0], d[1], n)
        return x, self(x)

    def convert(self, domain = None, kind: type = NoneType, window = None):
        if kind is NoneType:
            return self.convert(domain, ABCPolyBase[K], window)
        else:
            if domain is None:
                return self(kind.identity(kind._default_domain(), window=window, symbol=self.symbol))
            return self(kind.identity(domain, window=window, symbol=self.symbol))

    @staticmethod
    def _default_domain():
        return _default_domain(K)

    @staticmethod
    def fit(x, y, deg, domain = None, rcond = None, full: Literal[bool] = False, w = None,
            window = None, symbol: str = 'x'):
        dom = getdomain(x) if domain is None else np.asarray(domain, dtype=float)
        win = _default_domain(K) if window is None else np.asarray(window, dtype=float)
        off, scl = mapparms(dom, win)
        xn = off + scl * np.asarray(x, dtype=float)
        if full:
            c, info = _fit(K, xn, y, deg, rcond, True, w)
            return ABCPolyBase[K](c, dom, win, symbol), info
        else:
            return ABCPolyBase[K](_fit(K, xn, y, deg, rcond, False, w), dom, win, symbol)

    @staticmethod
    def fromroots(roots, domain = None, window = None, symbol: str = 'x'):
        r = np.atleast_1d(np.asarray(roots, dtype=float))
        dom = _default_domain(K) if domain is None else np.asarray(domain, dtype=float)
        win = _default_domain(K) if window is None else np.asarray(window, dtype=float)
        off, scl = mapparms(dom, win)
        rnew = off + scl * r
        coef = _fromroots(K, rnew) / (scl ** r.size)
        return ABCPolyBase[K](coef, dom, win, symbol)

    @staticmethod
    def identity(domain = None, window = None, symbol: str = 'x'):
        dom = _default_domain(K) if domain is None else np.asarray(domain, dtype=float)
        win = _default_domain(K) if window is None else np.asarray(window, dtype=float)
        off, scl = mapparms(win, dom)
        return ABCPolyBase[K](_line(K, off, scl), dom, win, symbol)

    @staticmethod
    def basis(deg: int, domain = None, window = None, symbol: str = 'x'):
        if deg < 0:
            raise ValueError("deg must be non-negative integer")
        c = np.zeros(deg + 1)
        c[deg] = 1.0
        return ABCPolyBase[K](c, domain, window, symbol)

    @staticmethod
    def cast(series, domain = None, window = None):
        return series.convert(domain, ABCPolyBase[K], window)

    def _term(self, i: int, arg: str) -> str:
        if K == 0:
            return "·" + arg + _map_digits(str(i), _SUP) if i > 1 else "·" + arg
        return "·" + self.basis_name + _map_digits(str(i), _SUB) + "(" + arg + ")"

    def __str__(self) -> str:
        linewidth = 75
        out = _fmt(self.coef[0])
        off, scale = self.mapparms()
        sym = self.symbol
        parens = True
        if off == 0.0 and scale == 1.0:
            parens = False
        elif scale == 1.0:
            sym = _fmt(off) + " + " + sym
        elif off == 0.0:
            sym = _fmt(scale) + sym
        else:
            sym = _fmt(off) + " + " + _fmt(scale) + sym
        if parens:
            sym = "(" + sym + ")"
        n = self.coef.size
        for i in range(1, n):
            out += " "
            cf = self.coef[i]
            nxt = ("+ " + _fmt(cf, True)) if cf >= 0 else ("- " + _fmt(-cf, True))
            nxt += self._term(i, sym)
            last = out.split("\n")[-1]
            ll = len(last) + len(nxt)
            if i < n - 1:
                ll += 2
            if ll >= linewidth:
                nxt = nxt.replace(" ", "\n", 1)
            out += nxt
        return out

    def __repr__(self) -> str:
        return (self._name + "(" + str(self.coef.tolist()) + ", domain=" + str(self.domain.tolist())
                + ", window=" + str(self.window.tolist()) + ", symbol='" + self.symbol + "')")
