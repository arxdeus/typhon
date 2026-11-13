# Ported from scipy 1.18.1 (BSD-3-Clause)
# Algorithms from xsf (BSD-3) cpu/legendre.h, cpu/sph_harm.h, cpu/recur.h:
# legendre_p, assoc_legendre_p (unnormalized / normalized, branch cuts 2 and 3),
# sph_legendre_p, sph_harm_y and the *_all variants.  Derivatives (diff_n <= 2)
# use second order dual numbers, as xsf does.

import cmath
from C import sqrt(float) -> float
from C import sin(float) -> float
from C import cos(float) -> float

INF = float('inf')
NAN = float('nan')
M_PI = 3.14159265358979323846

class Dual[T]:
    """Second order dual number: value, first and second derivatives."""
    v: T
    d1: T
    d2: T

    def __init__(self, v: T, d1: T, d2: T):
        self.v = v
        self.d1 = d1
        self.d2 = d2

    def __add__(self, o: Dual[T]) -> Dual[T]:
        return Dual[T](self.v + o.v, self.d1 + o.d1, self.d2 + o.d2)

    def __sub__(self, o: Dual[T]) -> Dual[T]:
        return Dual[T](self.v - o.v, self.d1 - o.d1, self.d2 - o.d2)

    def __mul__(self, o: Dual[T]) -> Dual[T]:
        return Dual[T](self.v * o.v, self.v * o.d1 + self.d1 * o.v,
                       self.v * o.d2 + T(2.0) * self.d1 * o.d1 + self.d2 * o.v)

    def scale(self, c: T) -> Dual[T]:
        return Dual[T](c * self.v, c * self.d1, c * self.d2)

    def __neg__(self) -> Dual[T]:
        return Dual[T](-self.v, -self.d1, -self.d2)

def _const(c: T, T: type) -> Dual[T]:
    return Dual[T](c, T(0.0), T(0.0))

def _var(z: T, T: type) -> Dual[T]:
    return Dual[T](z, T(1.0), T(0.0))

def _sqrt_t(x: T, T: type) -> T:
    if T is complex:
        return cmath.sqrt(x)
    else:
        return sqrt(x)

def _dsqrt(x: Dual[T], T: type) -> Dual[T]:
    s = _sqrt_t(x.v, T)
    d1 = x.d1 / (T(2.0) * s)
    d2 = x.d2 / (T(2.0) * s) - x.d1 * x.d1 / (T(4.0) * s * s * s)
    return Dual[T](s, d1, d2)

def _real(z: T, T: type) -> float:
    if T is complex:
        return z.real
    else:
        return z

def _imag(z: T, T: type) -> float:
    if T is complex:
        return z.imag
    else:
        return 0.0

def _ipow(z: T, k: int, T: type) -> T:
    r = T(1.0)
    if k >= 0:
        for i in range(k):
            r = r * z
    else:
        for i in range(-k):
            r = r / z
    return r

# ---------------------------------------------------------------- legendre_p
def legendre_p_d(n: int, z: T, T: type) -> Dual[T]:
    zz = _var(z, T)
    if n < 0:
        return _const(T(0.0), T)
    p0 = _const(T(1.0), T)
    p1 = zz
    if n == 0:
        return p0
    for j in range(2, n + 1):
        fac0 = -float(j - 1) / float(j)
        fac1 = float(2 * j - 1) / float(j)
        p2 = p0.scale(T(fac0)) + (zz * p1).scale(T(fac1))
        p0 = p1
        p1 = p2
    return p1

# ---------------------------------------------------------------- assoc_legendre_p
def _alp_m_abs_m(norm: bool, m: int, zz: Dual[T], branch_cut: int, T: type) -> Dual[T]:
    """P_{|m|}^{m}(z) via the diagonal recurrence."""
    m_signbit = m < 0
    one = _const(T(1.0), T)
    w = one
    if branch_cut == 3:
        w = _dsqrt(zz - one, T) * _dsqrt(zz + one, T)
    else:
        w = -_dsqrt(one - zz * zz, T)
        if m_signbit:
            w = -w
    r0 = one
    r1 = w
    if norm:
        r0 = _const(T(1.0 / sqrt(2.0)), T)
        r1 = w.scale(T(sqrt(3.0) / 2.0))
    elif m_signbit:
        r1 = w.scale(T(0.5))
    bcs = -1.0 if branch_cut == 3 else 1.0
    ma = m if m >= 0 else -m
    if ma == 0:
        return r0
    if ma == 1:
        return r1
    # rotate twice: res = [r0, r1]; iterate from |m| = 2
    for k in range(2, ma + 1):
        fac = 0.0
        if norm:
            fac = bcs * sqrt(float((2 * k + 1) * (2 * k - 1)) / float(4 * k * (k - 1)))
        elif m < 0:
            fac = bcs / float((2 * k) * (2 * k - 2))
        else:
            fac = bcs * float((2 * k - 1) * (2 * k - 3))
        t = (one - zz * zz).scale(T(fac)) * r0
        r0 = r1
        r1 = t
    return r1

def _alp_pm1(norm: bool, n: int, m: int, z: T, branch_cut: int, T: type) -> Dual[T]:
    v = T(0.0)
    if m == 0:
        if _real(z, T) >= 0:
            v = T(1.0)
        else:
            v = T(1.0 if n % 2 == 0 else -1.0)
        if norm:
            v = v * T(sqrt(float(2 * n + 1) / 2.0))
    bcs = T(-1.0 if branch_cut == 3 else 1.0)
    d1 = T(0.0)
    ma = m if m >= 0 else -m
    tinf = T(INF)
    if ma > n:
        d1 = T(0.0)
    elif m == 0:
        d1 = T(float(n) * float(n + 1)) * _ipow(z, n + 1, T) / T(2.0)
    elif m == 1:
        d1 = _ipow(z, n, T) * tinf
    elif m == 2:
        d1 = -bcs * T(float(n + 2) * float(n + 1) * float(n) * float(n - 1)) * _ipow(z, n + 1, T) / T(4.0)
    elif m == -2:
        d1 = -bcs * _ipow(z, n + 1, T) / T(4.0)
    elif m == -1:
        d1 = -_ipow(z, n, T) * tinf
    d2 = T(0.0)
    if ma > n:
        d2 = T(0.0)
    elif m == 0:
        d2 = T(float(n + 2) * float(n + 1) * float(n) * float(n - 1) / 8.0)
    elif m == 1:
        d2 = tinf
    elif m == 2:
        d2 = -T(float((n + 1) * n - 3) * float(n + 2) * float(n + 1) * float(n) * float(n - 1) / 12.0)
    elif m == 3:
        d2 = tinf
    elif m == 4:
        d2 = T(float(n + 4) * float(n + 3) * float(n + 2) * float(n + 1) * float(n) * float(n - 1)
               * float(n - 2) * float(n - 3) / 48.0)
    elif m == -3:
        d2 = -tinf
    elif m == -2:
        d2 = T(-0.25)
    elif m == -1:
        d2 = -tinf
    return Dual[T](v, d1, d2)

def _alp_n_from(norm: bool, n: int, m: int, zz: Dual[T], branch_cut: int, pmm: Dual[T], T: type) -> Dual[T]:
    """Given P_{|m|}^m, recur up in degree to P_n^m."""
    zero = _const(T(0.0), T)
    ma = m if m >= 0 else -m
    if ma > n:
        return zero
    z = zz.v
    if abs(_real(z, T)) == 1.0 and _imag(z, T) == 0.0:
        return _alp_pm1(norm, n, m, z, branch_cut, T)
    r0 = pmm
    r1 = zero
    if norm:
        r1 = (zz * pmm).scale(T(sqrt(float(2 * ma + 3))))
    else:
        r1 = (zz * pmm).scale(T(float(2 * (ma + 1) - 1) / float(ma + 1 - m)))
    if n == ma:
        return r0
    for j in range(ma + 2, n + 1):
        fac0 = 0.0
        fac1 = 0.0
        if norm:
            fac0 = -sqrt(float((2 * j + 1) * ((j - 1) * (j - 1) - m * m)) / float((2 * j - 3) * (j * j - m * m)))
            fac1 = sqrt(float((2 * j + 1) * (4 * (j - 1) * (j - 1) - 1)) / float((2 * j - 3) * (j * j - m * m)))
        else:
            fac0 = -float(j + m - 1) / float(j - m)
            fac1 = float(2 * j - 1) / float(j - m)
        t = r0.scale(T(fac0)) + (zz * r1).scale(T(fac1))
        r0 = r1
        r1 = t
    return r1

def assoc_legendre_p_d(norm: bool, n: int, m: int, z: T, branch_cut: int, T: type) -> Dual[T]:
    zz = _var(z, T)
    if n < 0:
        return _const(T(0.0), T)
    ma = m if m >= 0 else -m
    if ma <= n and abs(_real(z, T)) == 1.0 and _imag(z, T) == 0.0:
        return _alp_pm1(norm, n, m, z, branch_cut, T)
    pmm = _alp_m_abs_m(norm, m, zz, branch_cut, T)
    return _alp_n_from(norm, n, m, zz, branch_cut, pmm, T)

# ---------------------------------------------------------------- sph_legendre_p / sph_harm_y
def _sph_m_abs_m(m: int, th: Dual[float]) -> Dual[float]:
    st = _dsin(th)
    ast = st if st.v >= 0 else -st
    r0 = _const(1.0 / (2.0 * sqrt(M_PI)), float)
    fac1 = -sqrt(3.0) / (2.0 * sqrt(2.0 * M_PI))
    if m < 0:
        fac1 = -fac1
    r1 = ast.scale(fac1)
    ma = m if m >= 0 else -m
    if ma == 0:
        return r0
    if ma == 1:
        return r1
    for k in range(2, ma + 1):
        fac = sqrt(float((2 * k + 1) * (2 * k - 1)) / float(4 * k * (k - 1)))
        t = (st * st).scale(fac) * r0
        r0 = r1
        r1 = t
    return r1

def _dsin(th: Dual[float]) -> Dual[float]:
    s = sin(th.v)
    c = cos(th.v)
    return Dual[float](s, c * th.d1, c * th.d2 - s * th.d1 * th.d1)

def _dcos(th: Dual[float]) -> Dual[float]:
    s = sin(th.v)
    c = cos(th.v)
    return Dual[float](c, -s * th.d1, -s * th.d2 - c * th.d1 * th.d1)

def _sph_n_from(n: int, m: int, th: Dual[float], pmm: Dual[float]) -> Dual[float]:
    ma = m if m >= 0 else -m
    if ma > n:
        return _const(0.0, float)
    ct = _dcos(th)
    r0 = pmm
    r1 = (ct * pmm).scale(sqrt(float(2 * ma + 3)))
    if n == ma:
        return r0
    for j in range(ma + 2, n + 1):
        fac0 = -sqrt(float((2 * j + 1) * ((j - 1) * (j - 1) - m * m)) / float((2 * j - 3) * (j * j - m * m)))
        fac1 = sqrt(float((2 * j + 1) * (4 * (j - 1) * (j - 1) - 1)) / float((2 * j - 3) * (j * j - m * m)))
        t = r0.scale(fac0) + (ct * r1).scale(fac1)
        r0 = r1
        r1 = t
    return r1

def sph_legendre_p_d(n: int, m: int, theta: float) -> Dual[float]:
    if n < 0:
        return _const(0.0, float)
    th = _var(theta, float)
    return _sph_n_from(n, m, th, _sph_m_abs_m(m, th))
