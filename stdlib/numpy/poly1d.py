# Ported from numpy 2.5.4 (BSD-3-Clause)
#
# numpy.poly1d (float coefficients), print options and small top-level aliases.

import util
from .ndarray import ndarray
from .routines import asarray, array, zeros, atleast_1d, trim_zeros
from .polynomial_legacy import polyval, polyadd, polysub, polymul, polyder, polyint, roots
from .ndmath import invert, left_shift, right_shift

bitwise_invert = invert
bitwise_not = invert
bitwise_left_shift = left_shift
bitwise_right_shift = right_shift

def astype(x, dtype: type, copy: bool = True):
    return asarray(x).astype(dtype, copy=copy)

class poly1d:
    coeffs: ndarray[float, 1]

    def __init__(self, c_or_r, r: bool = False):
        c = asarray(c_or_r).astype(float)
        if r:
            from .polynomial_legacy import poly
            c = poly(c).astype(float)
        c = atleast_1d(c)
        c = trim_zeros(c, 'f')
        if c.size == 0:
            c = zeros(1, float)
        self.coeffs = c

    @property
    def c(self):
        return self.coeffs

    @property
    def coef(self):
        return self.coeffs

    @property
    def coefficients(self):
        return self.coeffs

    @property
    def order(self):
        return self.coeffs.size - 1

    @property
    def o(self):
        return self.coeffs.size - 1

    @property
    def roots(self):
        return roots(self.coeffs)

    @property
    def r(self):
        return roots(self.coeffs)

    def __len__(self):
        return self.coeffs.size - 1

    def __call__(self, val):
        return polyval(self.coeffs, val)

    def __getitem__(self, val: int):
        ind = self.order - val
        if val < 0:
            return 0.0
        if ind < 0:
            return 0.0
        return self.coeffs[ind]

    def __neg__(self):
        return poly1d(-self.coeffs)

    def __pos__(self):
        return self

    def __add__(self, other):
        if isinstance(other, poly1d):
            return poly1d(polyadd(self.coeffs, other.coeffs))
        return poly1d(polyadd(self.coeffs, asarray(other).astype(float)))

    def __radd__(self, other):
        return poly1d(polyadd(asarray(other).astype(float), self.coeffs))

    def __sub__(self, other):
        if isinstance(other, poly1d):
            return poly1d(polysub(self.coeffs, other.coeffs))
        return poly1d(polysub(self.coeffs, asarray(other).astype(float)))

    def __rsub__(self, other):
        return poly1d(polysub(asarray(other).astype(float), self.coeffs))

    def __mul__(self, other):
        if isinstance(other, poly1d):
            return poly1d(polymul(self.coeffs, other.coeffs))
        return poly1d(self.coeffs * float(other))

    def __rmul__(self, other):
        return poly1d(self.coeffs * float(other))

    def __pow__(self, val: int):
        if val < 0:
            raise ValueError("Power to non-negative integers only.")
        res = poly1d([1.0])
        for _ in range(val):
            res = res * self
        return res

    def __eq__(self, other: poly1d):
        if self.coeffs.shape != other.coeffs.shape:
            return False
        return bool((self.coeffs == other.coeffs).all())

    def __ne__(self, other: poly1d):
        return not self.__eq__(other)

    def deriv(self, m: int = 1):
        return poly1d(polyder(self.coeffs, m))

    def integ(self, m: int = 1):
        return poly1d(polyint(self.coeffs, m))

    def __repr__(self):
        return "poly1d(" + repr(self.coeffs.tolist()) + ")"

    def __str__(self):
        n = self.coeffs.size
        s = ''
        for i in range(n):
            c = self.coeffs[i]
            p = n - 1 - i
            if c == 0.0 and n > 1:
                continue
            term = str(abs(c))
            if p == 1:
                term += ' x'
            elif p > 1:
                term += ' x^' + str(p)
            if s == '':
                s = ('-' if c < 0 else '') + term
            else:
                s += (' - ' if c < 0 else ' + ') + term
        return s

# ---- print options (accepted and stored; array formatting itself is fixed) ----
class _PrintOptions:
    precision: int
    threshold: int
    edgeitems: int
    linewidth: int
    suppress: bool

    def __init__(self):
        self.precision = 8
        self.threshold = 1000
        self.edgeitems = 3
        self.linewidth = 75
        self.suppress = False

_popts = _PrintOptions()

def set_printoptions(precision: int = -1, threshold: int = -1, edgeitems: int = -1,
                     linewidth: int = -1, suppress: bool = False):
    if precision >= 0:
        _popts.precision = precision
    if threshold >= 0:
        _popts.threshold = threshold
    if edgeitems >= 0:
        _popts.edgeitems = edgeitems
    if linewidth >= 0:
        _popts.linewidth = linewidth
    _popts.suppress = suppress

def get_printoptions():
    return {'precision': _popts.precision, 'threshold': _popts.threshold,
            'edgeitems': _popts.edgeitems, 'linewidth': _popts.linewidth,
            'suppress': _popts.suppress}

def array_str(a):
    return str(asarray(a))

def array_repr(a):
    return repr(asarray(a))

def array2string(a):
    return str(asarray(a))

class printoptions:
    precision: int
    threshold: int
    edgeitems: int
    linewidth: int
    suppress: bool
    _o: Tuple[int, int, int, int, bool]

    def __init__(self, precision: int = -1, threshold: int = -1, edgeitems: int = -1,
                 linewidth: int = -1, suppress: bool = False):
        self.precision = precision
        self.threshold = threshold
        self.edgeitems = edgeitems
        self.linewidth = linewidth
        self.suppress = suppress
        self._o = (_popts.precision, _popts.threshold, _popts.edgeitems,
                   _popts.linewidth, _popts.suppress)

    def __enter__(self):
        self._o = (_popts.precision, _popts.threshold, _popts.edgeitems,
                   _popts.linewidth, _popts.suppress)
        set_printoptions(self.precision, self.threshold, self.edgeitems,
                         self.linewidth, self.suppress)
        return get_printoptions()

    def __exit__(self):
        _popts.precision = self._o[0]
        _popts.threshold = self._o[1]
        _popts.edgeitems = self._o[2]
        _popts.linewidth = self._o[3]
        _popts.suppress = self._o[4]

def get_include() -> str:
    return ''
