# Ported from scipy 1.18.1 (BSD-3-Clause)
# Algorithms from xsf (BSD-3) cephes/dd_real.h (double-double arithmetic,
# after the QD library by Hida, Li & Bailey, BSD).
#
# A double-double value is a tuple (hi, lo).

from C import fma(float, float, float) -> float
from C import frexp(float, Ptr[i32]) -> float
from C import ldexp(float, i32) -> float as c_ldexp
from . import _kernels as K

INF = float('inf')
NAN = float('nan')

def quick_two_sum(a: float, b: float):
    s = a + b
    c = s - a
    return (s, b - c)

def two_sum(a: float, b: float):
    s = a + b
    c = s - a
    d = b - c
    e = s - c
    return (s, (a - e) + d)

def two_prod(a: float, b: float):
    p = a * b
    return (p, fma(a, b, -p))

def dd(x: float):
    return (x, 0.0)

def neg(x):
    return (-x[0], -x[1])

def add(l, r):
    s = two_sum(l[0], r[0])
    t = two_sum(l[1], r[1])
    s2 = s[1] + t[0]
    u = quick_two_sum(s[0], s2)
    s2 = u[1] + t[1]
    return quick_two_sum(u[0], s2)

def add_d(l, r: float):
    s = two_sum(l[0], r)
    s2 = s[1] + l[1]
    return quick_two_sum(s[0], s2)

def sub(l, r):
    return add(l, neg(r))

def sub_d(l, r: float):
    s = two_sum(l[0], -r)
    s2 = s[1] + l[1]
    return quick_two_sum(s[0], s2)

def d_sub(l: float, r):
    s = two_sum(l, -r[0])
    s2 = s[1] - r[1]
    return quick_two_sum(s[0], s2)

def mul(l, r):
    p = two_prod(l[0], r[0])
    p2 = p[1] + (l[0] * r[1] + l[1] * r[0])
    return quick_two_sum(p[0], p2)

def mul_d(l, r: float):
    p1 = two_prod(l[0], r)
    p2 = two_prod(l[1], r)
    return quick_two_sum(p1[0], p2[1] + p2[0] + p1[1])

def div(l, r):
    q1 = l[0] / r[0]
    rr = sub(l, mul_d(r, q1))
    q2 = rr[0] / r[0]
    rr = sub(rr, mul_d(r, q2))
    q3 = rr[0] / r[0]
    q = quick_two_sum(q1, q2)
    return add_d(q, q3)

def div_d(l, r: float):
    return div(l, (r, 0.0))

def d_div(l: float, r):
    return div((l, 0.0), r)

def floor(a):
    hi = K.floor(a[0])
    lo = 0.0
    if hi == a[0]:
        lo = K.floor(a[1])
        return quick_two_sum(hi, lo)
    return (hi, lo)

def ldexp(a, e: int):
    return (c_ldexp(a[0], i32(e)), c_ldexp(a[1], i32(e)))

def frexp_dd(a):
    """Returns (mantissa_dd, exponent)."""
    ex = i32(0)
    man = frexp(a[0], __ptr__(ex))
    exponent = int(ex)
    b1 = c_ldexp(a[1], i32(-exponent))
    if K.fabs(man) == 0.5 and man * b1 < 0:
        man *= 2
        b1 *= 2
        exponent -= 1
    return ((man, b1), exponent)

def isfinite(a) -> bool:
    x = a[0]
    return x == x and x != INF and x != -INF

def mul_pwr2(a, b: float):
    return (a[0] * b, a[1] * b)

def square(a):
    p = two_prod(a[0], a[0])
    p2 = p[1] + 2.0 * a[0] * a[1]
    p2 += a[1] * a[1]
    return quick_two_sum(p[0], p2)

_INV_FACT = ((1.66666666666666657e-01, 9.25185853854297066e-18),
             (4.16666666666666644e-02, 2.31296463463574266e-18),
             (8.33333333333333322e-03, 1.15648231731787138e-19),
             (1.38888888888888894e-03, -5.30054395437357706e-20),
             (1.98412698412698413e-04, 1.72095582934207053e-22),
             (2.48015873015873016e-05, 2.15119478667758816e-23),
             (2.75573192239858925e-06, -1.85839327404647208e-22),
             (2.75573192239858883e-07, 2.37677146222502973e-23),
             (2.50521083854417202e-08, -1.44881407093591197e-24),
             (2.08767569878681002e-09, -1.20734505911325997e-25),
             (1.60590438368216133e-10, 1.25852945887520981e-26),
             (1.14707455977297245e-11, 2.06555127528307454e-28),
             (7.64716373181981641e-13, 7.03872877733453001e-30),
             (4.77947733238738525e-14, 4.39920548583408126e-31),
             (2.81145725434552060e-15, 1.65088427308614326e-31))
_E = (2.718281828459045091e+00, 1.445646891729250158e-16)
_LOG2 = (6.931471805599452862e-01, 2.319046813846299558e-17)
_DD_EPS = 4.93038065763132e-32

def exp(a):
    k = 512.0
    inv_k = 1.0 / k
    if a[0] <= -709.0:
        return (0.0, 0.0)
    if a[0] >= 709.0:
        return (INF, INF)
    if a[0] == 0.0 and a[1] == 0.0:
        return (1.0, 0.0)
    if a[0] == 1.0 and a[1] == 0.0:
        return _E
    m = K.floor(a[0] / _LOG2[0] + 0.5)
    r = mul_pwr2(sub(a, mul_d(_LOG2, m)), inv_k)
    p = square(r)
    s = add(r, mul_pwr2(p, 0.5))
    p = mul(p, r)
    t = mul(p, _INV_FACT[0])
    i = 0
    while True:
        s = add(s, t)
        p = mul(p, r)
        i += 1
        t = mul(p, _INV_FACT[i])
        if not (K.fabs(t[0]) > inv_k * _DD_EPS and i < 5):
            break
    s = add(s, t)
    for j in range(9):
        s = add(mul_pwr2(s, 2.0), square(s))
    s = add_d(s, 1.0)
    return ldexp(s, int(m))

def log(a):
    if a[0] == 1.0 and a[1] == 0.0:
        return (0.0, 0.0)
    if a[0] <= 0.0:
        return (NAN, NAN)
    x = (K.log(a[0]), 0.0)
    x = sub_d(add(x, mul(a, exp(neg(x)))), 1.0)
    return x

def log1p(a):
    if a[0] <= -1.0:
        return (-INF, -INF)
    la = K.log1p(a[0])
    elam1 = K.expm1(la)
    ll = K.log1p(a[1] / (1 + a[0]))
    if a[0] > 0:
        ll -= (elam1 - a[0]) / (elam1 + 1)
    return add_d((la, 0.0), ll)
