# Ported from scipy 1.18.1 (BSD-3-Clause)
# Algorithms from Cephes (BSD-3) / xsf (BSD-3)
#
# Real hypergeometric functions: hyp2f1 (cephes, as xsf::hyp2f1 for real
# arguments).  Internal helpers return (value, loss) tuples.

from . import _kernels as K
from ._kernels import exp, log, sqrt, fabs, pow
from . import _bessel as B

from C import round(float) -> float as c_round

NAN = float('nan')
INF = float('inf')
MACHEP = 1.1102230246251565404236316680908203125E-16
H_EPS = 1.0e-13
ETHRESH = 1.0e-12
MAXITER = 10000

def _lgam_sgn(x: float):
    return (K.gammaln(x), K.gammasgn(x))

def _hys2f1(a: float, b: float, c: float, x: float):
    intflag = False
    if fabs(b) > fabs(a):
        f = b
        b = a
        a = f
    ib = c_round(b)
    if fabs(b - ib) < H_EPS and ib <= 0 and fabs(b) < fabs(a):
        f = b
        b = a
        a = f
        intflag = True
    if (fabs(a) > fabs(c) + 1 or intflag) and fabs(c - a) > 2 and fabs(a) > 2:
        return _hyp2f1ra(a, b, c, x)
    i = 0
    umax = 0.0
    f = a
    g = b
    h = c
    s = 1.0
    u = 1.0
    k = 0.0
    while True:
        if fabs(h) < H_EPS:
            return (INF, 1.0)
        m = k + 1.0
        u = u * ((f + k) * (g + k) * x / ((h + k) * m))
        s += u
        k = fabs(u)
        if k > umax:
            umax = k
        k = m
        i += 1
        if i > MAXITER:
            return (s, 1.0)
        if not (s == 0 or fabs(u / s) > MACHEP):
            break
    return (s, (MACHEP * umax) / fabs(s) + (MACHEP * i))

def _hyp2f1ra(a: float, b: float, c: float, x: float):
    da = 0.0
    if (c < 0 and a <= c) or (c >= 0 and a >= c):
        da = c_round(a - c)
    else:
        da = c_round(a)
    t = a - da
    loss = 0.0
    if fabs(da) > MAXITER:
        return (NAN, 1.0)
    f0 = 0.0
    f1 = 0.0
    f2 = 0.0
    if da < 0:
        r1 = _hys2f1(t, b, c, x)
        f1 = r1[0]
        loss += r1[1]
        r0 = _hys2f1(t - 1, b, c, x)
        f0 = r0[0]
        loss += r0[1]
        t -= 1
        n = 1
        while n < -da:
            f2 = f1
            f1 = f0
            f0 = -(2 * t - c - t * x + b * x) / (c - t) * f1 - t * (x - 1) / (c - t) * f2
            t -= 1
            n += 1
    else:
        r1 = _hys2f1(t, b, c, x)
        f1 = r1[0]
        loss += r1[1]
        r0 = _hys2f1(t + 1, b, c, x)
        f0 = r0[0]
        loss += r0[1]
        t += 1
        n = 1
        while n < da:
            f2 = f1
            f1 = f0
            f0 = -((2 * t - c - t * x + b * x) * f1 + (c - t) * f2) / (t * (x - 1))
            t += 1
            n += 1
    return (f0, loss)

def _hyt2f1(a: float, b: float, c: float, x: float):
    neg_int_a = False
    neg_int_b = False
    ia = c_round(a)
    ib = c_round(b)
    if a <= 0 and fabs(a - ia) < H_EPS:
        neg_int_a = True
    if b <= 0 and fabs(b - ib) < H_EPS:
        neg_int_b = True
    err = 0.0
    s = 1.0 - x
    y = 0.0
    if x < -0.5 and not (neg_int_a or neg_int_b):
        if b > a:
            r = _hys2f1(a, c - b, c, -x / s)
            return (pow(s, -a) * r[0], r[1])
        r = _hys2f1(c - a, b, c, -x / s)
        return (pow(s, -b) * r[0], r[1])
    d = c - a - b
    id = c_round(d)
    if x > 0.9 and not (neg_int_a or neg_int_b):
        if fabs(d - id) > MACHEP:
            r0 = _hys2f1(a, b, c, x)
            y = r0[0]
            err = r0[1]
            if err < ETHRESH:
                return (y, err)
            rq = _hys2f1(a, b, 1.0 - d, s)
            q = rq[0]
            err = rq[1]
            sign = 1.0
            lg = _lgam_sgn(d)
            w = lg[0]
            sign *= lg[1]
            lg = _lgam_sgn(c - a)
            w -= lg[0]
            sign *= lg[1]
            lg = _lgam_sgn(c - b)
            w -= lg[0]
            sign *= lg[1]
            q *= sign * exp(w)
            rr = _hys2f1(c - a, c - b, d + 1.0, s)
            r = pow(s, d) * rr[0]
            err1 = rr[1]
            sign = 1.0
            lg = _lgam_sgn(-d)
            w = lg[0]
            sign *= lg[1]
            lg = _lgam_sgn(a)
            w -= lg[0]
            sign *= lg[1]
            lg = _lgam_sgn(b)
            w -= lg[0]
            sign *= lg[1]
            r *= sign * exp(w)
            y = q + r
            q = fabs(q)
            r = fabs(r)
            if q > r:
                r = q
            err += err1 + (MACHEP * r) / y
            y *= K.gamma(c)
            return (y, err)
        else:
            e = 0.0
            d1 = 0.0
            d2 = 0.0
            aid = 0
            if id >= 0.0:
                e = d
                d1 = d
                d2 = 0.0
                aid = int(id)
            else:
                e = -d
                d1 = 0.0
                d2 = d
                aid = int(-id)
            ax = log(s)
            psi = K.digamma
            y = psi(1.0) + psi(1.0 + e) - psi(a + d1) - psi(b + d1) - ax
            y *= B.rgamma(e + 1.0)
            p = (a + d1) * (b + d1) * s * B.rgamma(e + 2.0)
            t = 1.0
            while True:
                r = psi(1.0 + t) + psi(1.0 + t + e) - psi(a + t + d1) - psi(b + t + d1) - ax
                q = p * r
                y += q
                p *= s * (a + t + d1) / (t + 1.0)
                p *= (b + t + d1) / (t + 1.0 + e)
                t += 1.0
                if t > MAXITER:
                    return (NAN, 1.0)
                if not (y == 0 or fabs(q / y) > H_EPS):
                    break
            if id == 0.0:
                y *= K.gamma(c) / (K.gamma(a) * K.gamma(b))
                return (y, err)
            y1 = 1.0
            if aid != 1:
                t = 0.0
                p = 1.0
                for i in range(1, aid):
                    r = 1.0 - e + t
                    p *= s * (a + t + d2) * (b + t + d2) / r
                    t += 1.0
                    p /= t
                    y1 += p
            p = K.gamma(c)
            y1 *= K.gamma(e) * p * (B.rgamma(a + d1) * B.rgamma(b + d1))
            y *= p * (B.rgamma(a + d2) * B.rgamma(b + d2))
            if (aid & 1) != 0:
                y = -y
            q = pow(s, id)
            if id > 0.0:
                y *= q
            else:
                y1 *= q
            y += y1
            return (y, err)
    return _hys2f1(a, b, c, x)

def _hyp2f1_neg_c_equal_bc(a: float, b: float, x: float) -> float:
    collector = 1.0
    sm = 1.0
    collector_max = 1.0
    if not (fabs(b) < 1e5):
        return NAN
    k = 1.0
    while k <= -b:
        collector *= (a + k - 1) * x / k
        collector_max = max(fabs(collector), collector_max)
        sm += collector
        k += 1.0
    if 1e-16 * (1 + collector_max / fabs(sm)) > 1e-7:
        return NAN
    return sm

def hyp2f1(a: float, b: float, c: float, x: float) -> float:
    err = 0.0
    ax = fabs(x)
    s = 1.0 - x
    ia = c_round(a)
    ib = c_round(b)
    neg_int_a = False
    neg_int_b = False
    neg_int_ca_or_cb = False
    if x == 0.0:
        return 1.0
    d = c - a - b
    id = c_round(d)
    if (a == 0 or b == 0) and c != 0:
        return 1.0
    if a == -1 and c != 0:
        result = 1 - (b / c) * x
        if fabs(result) < 1e-15:
            return 0.0
        return result
    if a <= 0 and fabs(a - ia) < H_EPS:
        neg_int_a = True
    if b <= 0 and fabs(b - ib) < H_EPS:
        neg_int_b = True
    if d <= -1 and not (fabs(d - id) > H_EPS and s < 0) and not (neg_int_a or neg_int_b):
        return pow(s, d) * hyp2f1(c - a, c - b, c, x)
    if d <= 0 and x == 1 and not (neg_int_a or neg_int_b):
        return INF
    if ax < 1.0 or x == -1.0:
        if fabs(b - c) < H_EPS:
            if neg_int_b:
                return _hyp2f1_neg_c_equal_bc(a, b, x)
            return pow(s, -a)
        if fabs(a - c) < H_EPS:
            return pow(s, -b)
    hypok = False
    if c <= 0.0:
        ic = c_round(c)
        if fabs(c - ic) < H_EPS:
            if neg_int_a and ia > ic:
                hypok = True
            elif neg_int_b and ib > ic:
                hypok = True
            else:
                return INF
    if not hypok and (neg_int_a or neg_int_b):
        hypok = True
    if not hypok:
        t1 = fabs(b - a)
        if x < -2.0 and fabs(t1 - c_round(t1)) > H_EPS:
            p = hyp2f1(a, 1 - c + a, 1 - b + a, 1.0 / x)
            q = hyp2f1(b, 1 - c + b, 1 - a + b, 1.0 / x)
            p *= pow(-x, -a)
            q *= pow(-x, -b)
            t1 = K.gamma(c)
            s = t1 * K.gamma(b - a) * (B.rgamma(b) * B.rgamma(c - a))
            y = t1 * K.gamma(a - b) * (B.rgamma(a) * B.rgamma(c - b))
            return s * p + y * q
        elif x < -1.0:
            if fabs(a) < fabs(b):
                return pow(s, -a) * hyp2f1(a, c - b, c, x / (x - 1))
            return pow(s, -b) * hyp2f1(b, c - a, c, x / (x - 1))
        if ax > 1.0:
            return INF
        p = c - a
        ia = c_round(p)
        if ia <= 0.0 and fabs(p - ia) < H_EPS:
            neg_int_ca_or_cb = True
        r = c - b
        ib = c_round(r)
        if ib <= 0.0 and fabs(r - ib) < H_EPS:
            neg_int_ca_or_cb = True
        id = c_round(d)
        if fabs(ax - 1.0) < H_EPS:
            if x > 0.0:
                if neg_int_ca_or_cb:
                    if d >= 0.0:
                        return pow(s, d) * _hys2f1(c - a, c - b, c, x)[0]
                    return INF
                if d <= 0.0:
                    return INF
                return K.gamma(c) * K.gamma(d) * (B.rgamma(p) * B.rgamma(r))
            if d <= -1.0:
                return INF
        if d < 0.0:
            ry = _hyt2f1(a, b, c, x)
            if ry[1] < ETHRESH:
                return ry[0]
            aid = int(2 - id)
            e = c + aid
            d2 = hyp2f1(a, b, e, x)
            d1 = hyp2f1(a, b, e + 1.0, x)
            q = a + b + 1.0
            y = 0.0
            for i in range(aid):
                r = e - 1.0
                y = (e * (r - (2.0 * e - q) * x) * d2 + (e - a) * (e - b) * x * d1) / (e * r * s)
                e = r
                d1 = d2
                d2 = y
            return y
        if neg_int_ca_or_cb:
            return pow(s, d) * _hys2f1(c - a, c - b, c, x)[0]
    return _hyt2f1(a, b, c, x)[0]

# ------------------------------------------------------------------ hyp0f1 (scipy _hyp0f1.pxd)
DBL_MAX = 1.7976931348623157e308
DBL_MIN = 2.2250738585072014e-308
PI = 3.14159265358979323846

def _xlogy(x: float, y: float) -> float:
    return K.xlogy(x, y)

def _hyp0f1_asy(v: float, z: float) -> float:
    arg = sqrt(z)
    v1 = fabs(v - 1)
    x = 2.0 * arg / v1
    p1 = sqrt(1.0 + x * x)
    eta = p1 + log(x) - K.log1p(p1)
    arg_exp_i = -0.5 * log(p1)
    arg_exp_i -= 0.5 * log(2.0 * PI * v1)
    arg_exp_i += K.gammaln(v)
    gs = K.gammasgn(v)
    arg_exp_k = arg_exp_i
    arg_exp_i += v1 * eta
    arg_exp_k -= v1 * eta
    pp = 1.0 / p1
    p2 = pp * pp
    p4 = p2 * p2
    p6 = p4 * p2
    u1 = (3.0 - 5.0 * p2) * pp / 24.0
    u2 = (81.0 - 462.0 * p2 + 385.0 * p4) * p2 / 1152.0
    u3 = (30375.0 - 369603.0 * p2 + 765765.0 * p4 - 425425.0 * p6) * pp * p2 / 414720.0
    u_corr_i = 1.0 + u1 / v1 + u2 / (v1 * v1) + u3 / (v1 * v1 * v1)
    result = exp(arg_exp_i - _xlogy(v1, arg)) * gs * u_corr_i
    if v - 1 < 0:
        u_corr_k = 1.0 - u1 / v1 + u2 / (v1 * v1) - u3 / (v1 * v1 * v1)
        result += exp(arg_exp_k + _xlogy(v1, arg)) * gs * 2.0 * B.sinpi(v1) * u_corr_k
    return result

def hyp0f1(v: float, z: float) -> float:
    if v <= 0.0 and v == K.floor(v):
        return NAN
    if z == 0.0 and v != 0.0:
        return 1.0
    if fabs(z) < 1e-6 * (1.0 + fabs(v)):
        return 1.0 + z / v + z * z / (2.0 * v * (v + 1.0))
    if z > 0:
        arg = sqrt(z)
        arg_exp = _xlogy(1.0 - v, arg) + K.gammaln(v)
        bess_val = B.iv(v - 1, 2.0 * arg)
        if arg_exp > log(DBL_MAX) or bess_val == 0 or arg_exp < log(DBL_MIN) or bess_val == INF or bess_val == -INF:
            return _hyp0f1_asy(v, z)
        return exp(arg_exp) * K.gammasgn(v) * bess_val
    arg = sqrt(-z)
    return pow(arg, 1.0 - v) * K.gamma(v) * B.jv_cephes(v - 1, 2 * arg)

# ------------------------------------------------------------------ hyp1f1 (Boost hypergeometric_1F1 core paths)
# Ported paths: parameter checks, a == -1, b == a, b - a == -1, (1, 2),
# large-|z| asymptotic 2F0 (A&S 13.5.1), Boost's rational (Luke) and Pade
# approximants, Kummer reflection and the log-scaled direct series.  The
# Bessel/Tricomi/recurrence fallbacks of Boost for divergent regions fall back
# to the scaled direct series here.
_EPS = 2.220446049250313e-16
_LOG_MAX = 709.782712893384
_MAX_SERIES = 1000000

def _llt(x: float) -> int:
    return int(x)

def _1f1_asym_region(a: float, b: float, z: float) -> bool:
    half_digits = 26
    in_region = False
    if fabs(a) < 0.001:
        return False
    if z > 0:
        one_minus_a = 1 - a
        b_minus_a = b - a
        if fabs((one_minus_a + half_digits) * (b_minus_a + half_digits) / (half_digits * z)) < 0.7:
            in_region = True
            if one_minus_a < 0 or b_minus_a < 0:
                if fabs(one_minus_a * b_minus_a / z) > 0.5:
                    in_region = False
    elif fabs((1 - (b - a) + half_digits) * (a + half_digits) / (half_digits * z)) < 0.7:
        if K.floor(b - a) == (b - a) and b - a < 0:
            return False
        in_region = True
        a1 = 1 - (b - a)
        if a1 < 0 or a < 0:
            if fabs(a1 * a / z) > 0.5:
                in_region = False
    if in_region:
        if a < 0 and K.floor(a) == a:
            in_region = False
        if b < 0 and K.floor(b) == b:
            in_region = False
        if fabs(z) < 40:
            in_region = False
    return in_region

def _1f1_asym_large_z(a: float, b: float, z: float):
    """Returns (value, log_scaling, ok)."""
    ls = 0
    prefix = 1.0
    if z < 0:
        a = b - a
        z = -z
        prefix = 1.0
    else:
        e = _llt(z)
        ls += e
        prefix = exp(z - e)
    if fabs(a) < 10 and fabs(b) < 10:
        prefix *= pow(z, a) * pow(z, -b) * K.gamma(b) / K.gamma(a)
    else:
        t = log(z) * (a - b)
        e = _llt(t)
        ls += e
        prefix *= exp(t - e)
        t = K.gammaln(b)
        s = K.gammasgn(b)
        e = _llt(t)
        ls += e
        prefix *= s * exp(t - e)
        t = K.gammaln(a)
        s = K.gammasgn(a)
        e = _llt(t)
        ls -= e
        prefix /= s * exp(t - e)
    k = 0
    a1_poch = 1 - a
    a2_poch = b - a
    z_mult = 1 / z
    sm = 0.0
    abs_sum = 0.0
    term = 1.0
    last_term = 0.0
    while True:
        sm += term
        last_term = term
        abs_sum += fabs(sm)
        term *= a1_poch * a2_poch * z_mult
        k += 1
        term /= k
        a1_poch += 1
        a2_poch += 1
        if fabs(sm) * _EPS > fabs(term):
            break
        if fabs(sm) / abs_sum < _EPS:
            return (prefix * sm, ls, False)
        if k > _MAX_SERIES:
            return (prefix * sm, ls, False)
        if k > 10 and fabs(term) > fabs(last_term):
            return (prefix * sm, ls, False)
    return (prefix * sm, ls, True)

def _1f1_rational(ap: float, cp: float, zp: float) -> float:
    z = -zp
    z2 = z / 2.0
    ct1 = ap * (z / cp)
    ct2 = z2 / (1.0 + cp)
    xn3 = 0.0
    xn2 = 1.0
    xn1 = 2.0
    xn0 = 3.0
    b1 = 1.0
    a1 = 1.0
    b2 = 1.0 + ((1.0 + ap) * (z2 / cp))
    a2 = b2 - ct1
    b3 = 1.0 + ((2.0 + b2) * (((2.0 + ap) / 3.0) * ct2))
    a3 = b3 - ((1.0 + ct2) * ct1)
    ct1 = 3.0
    result = 0.0
    prev_result = a3 / b3
    for k in range(2, _MAX_SERIES):
        ct2 = (z2 / ct1) / (cp + xn1)
        g1 = 1.0 + (ct2 * (xn2 - ap))
        ct2 *= ((ap + xn1) / (cp + xn2))
        g2 = ct2 * ((cp - xn1) + (((ap + xn0) / (ct1 + 2.0)) * z2))
        g3 = ((ct2 * z2) * (((z2 / ct1) / (ct1 - 2.0)) * ((ap + xn2)) / (cp + xn3))) * (ap - xn2)
        b4 = (g1 * b3) + (g2 * b2) + (g3 * b1)
        a4 = (g1 * a3) + (g2 * a2) + (g3 * a1)
        prev_result = result
        result = a4 / b4
        if (fabs(result) * _EPS) > fabs(result - prev_result) / fabs(result):
            break
        b1 = b2
        b2 = b3
        b3 = b4
        a1 = a2
        a2 = a3
        a3 = a4
        xn3 = xn2
        xn2 = xn1
        xn1 = xn0
        xn0 += 1
        ct1 += 2.0
    return result

def _1f1_pade(cp: float, zp: float) -> float:
    z = -zp
    zz = z * z
    b0 = 1.0
    a0 = 1.0
    xi1 = 1.0
    ct1 = cp + 1.0
    cp1 = cp - 1.0
    b1 = 1.0 + (z / ct1)
    a1 = b1 - (z / cp)
    b2 = 0.0
    a2 = 0.0
    result = 0.0
    for k in range(1, _MAX_SERIES):
        ct2 = ct1 * ct1
        g1 = 1.0 + ((cp1 / (ct2 + ct1 + ct1)) * z)
        g2 = ((xi1 / (ct2 - 1.0)) * ((xi1 + cp1) / ct2)) * zz
        b2 = (g1 * b1) + (g2 * b0)
        a2 = (g1 * a1) + (g2 * a0)
        prev_result = result
        result = a2 / b2
        if (fabs(result) * _EPS) > fabs(result - prev_result):
            break
        b0 = b1
        b1 = b2
        a0 = a1
        a1 = a2
        ct1 += 2
        xi1 += 1
    return a2 / b2

def _1f1_generic_series(a: float, b: float, z: float):
    """Returns (sum, log_scaling)."""
    ls = 0
    sm = 0.0
    term = 1.0
    upper_limit = sqrt(1.7976931348623157e308)
    lower_limit = 1 / upper_limit
    n = 0
    lsf = int(_LOG_MAX) - 2
    scaling_factor = exp(float(lsf))
    small_a = fabs(a) < 0.25
    while True:
        sm += term
        if fabs(sm) >= upper_limit:
            sm /= scaling_factor
            term /= scaling_factor
            ls += lsf
        if fabs(sm) < lower_limit:
            sm *= scaling_factor
            term *= scaling_factor
            ls -= lsf
        term_m1 = term
        term *= (((a + n) / ((b + n) * (n + 1))) * z)
        if n > _MAX_SERIES:
            return (NAN, 0)
        n += 1
        diff = fabs(term / sm)
        if not (diff > _EPS or fabs(term_m1) < fabs(term) or (small_a and n < 10)):
            break
    return (sm, ls)

def _1f1_imp(a: float, b: float, z: float, depth: int):
    """Returns (value, log_scaling)."""
    if z == 0 or a == 0:
        return (1.0, 0)
    if b <= 0 and b == K.floor(b):
        if a >= 0 or a < b or a != K.floor(a):
            return (NAN, 0)
    if a == -1:
        r = 1 - (z / b)
        if fabs(r) < 0.5:
            r = (b - z) / b
        return (r, 0)
    b_minus_a = b - a
    if b_minus_a == 0:
        if a < 0 and K.floor(a) == a:
            if a < -20 and z > 0 and z < 1:
                return (exp(z) * K.gammaincc(1 - a, z), 0)
            return _1f1_generic_series(a, b, z)
        scale = _llt(z)
        return (exp(z - scale), scale)
    if b_minus_a == -1 and fabs(a) > 0.5:
        if a < 0 and a == K.ceil(a) and a > -50:
            return _1f1_generic_series(a, b, z)
        ls = _llt(K.floor(z))
        return ((b + z) * exp(z - ls) / b, ls)
    if a == 1 and b == 2:
        return (K.expm1(z) / z, 0)
    if b - a == b and fabs(z / b) < _EPS:
        return (1.0, 0)
    if _1f1_asym_region(a, b, z):
        r = _1f1_asym_large_z(a, b, z)
        if r[2]:
            return (r[0], r[1])
    if fabs(a * z / b) < 3.5 and fabs(z * 100) < fabs(b) and (fabs(a) > 1e-2 or b < -5):
        return (_1f1_rational(a, b, z), 0)
    need_kummer = False
    if z > 0:
        need_kummer = False
    elif z < -1:
        need_kummer = True
    elif a > 0:
        need_kummer = (fabs((a + 10) * z / (10 * (b + 10))) < 1) if b > 0 else True
    else:
        need_kummer = False if b > 0 else True
    if need_kummer and depth < 2:
        if a == 1:
            return (_1f1_pade(b, z), 0)
        if b < 0 and K.floor(b) == b:
            return _1f1_generic_series(a, b, z)
        scaling = _llt(z)
        r = _1f1_imp(b_minus_a, b, -z, depth + 1)
        return (exp(z - scaling) * r[0], r[1] + scaling)
    return _1f1_generic_series(a, b, z)

def hyp1f1(a: float, b: float, x: float) -> float:
    if _isnan_h(a) or _isnan_h(b) or _isnan_h(x):
        return NAN
    if b <= 0 and K.floor(b) == b:
        if b != 0 and a == b:
            return _1f1_generic_series(a, b, x)[0]
        if not (a < 0 and K.floor(a) == a and a >= b):
            return INF
    r = _1f1_imp(a, b, x, 0)
    result = r[0]
    log_scaling = r[1]
    max_scaling = int(_LOG_MAX) - 2
    max_scale_factor = exp(float(max_scaling))
    while log_scaling > max_scaling:
        result *= max_scale_factor
        log_scaling -= max_scaling
    while log_scaling < -max_scaling:
        result /= max_scale_factor
        log_scaling += max_scaling
    if log_scaling != 0:
        result *= exp(float(log_scaling))
    return result

def _isnan_h(x: float) -> bool:
    return x != x

# ------------------------------------------------------------------ hyperu (scipy _hypergeometric.pxd)
from . import _specfun as SF
from . import _gamma2 as G2

def hyperu(a: float, b: float, x: float) -> float:
    if _isnan_h(a) or _isnan_h(b) or _isnan_h(x):
        return NAN
    if x < 0.0:
        return NAN
    if x == 0.0:
        if b > 1.0:
            return INF
        return G2.poch(1.0 - b + a, -a)
    if b == 1 and x < 1 and -0.25 < a and a < 0.3:
        return (x + 1 + 2 * a) * SF.hypu(a + 1, 1.0, x) - (a + 1) * (a + 1) * SF.hypu(a + 2, 1.0, x)
    return SF.hypu(a, b, x)
