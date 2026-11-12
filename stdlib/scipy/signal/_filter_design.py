# Ported from scipy 1.18.1 (BSD-3-Clause)
#
# IIR filter design. Zeros and poles are always returned as complex arrays and
# transfer-function coefficients as float arrays (static typing; scipy may
# return a float zero array for e.g. Butterworth lowpass designs).

import numpy as np
import numpy.linalg as _alg
import math
import cmath

_EPS = 2.220446049250313e-16

def _carr(x) -> np.ndarray[complex, 1]:
    if isinstance(x, List[complex]):
        a = np.empty(len(x), dtype=complex)
        for i in range(len(x)):
            a[i] = x[i]
        return a
    else:
        a0 = np.atleast_1d(np.asarray(x))
        a = np.empty(a0.size, dtype=complex)
        for i in range(a0.size):
            a[i] = complex(a0[i])
        return a

def _clist(x) -> List[complex]:
    a = _carr(x)
    return [a[i] for i in range(a.size)]

def _farr(x: List[float]):
    a = np.empty(len(x), dtype=float)
    for i in range(len(x)):
        a[i] = x[i]
    return a

def _flist(x) -> List[float]:
    a0 = np.atleast_1d(np.asarray(x))
    return [float(a0[i]) for i in range(a0.size)]

def _isreal(c: complex) -> bool:
    return c.imag == 0.0

# ---------------------------------------------------------------- polynomials

def _poly_c(zs: List[complex]) -> List[complex]:
    a = [complex(1.0)]
    for z in zs:
        b = [complex(0.0)] * (len(a) + 1)
        for i in range(len(a)):
            b[i] += a[i]
            b[i + 1] -= a[i] * z
        a = b
    return a

def _poly_real(zs: List[complex]) -> List[float]:
    return [c.real for c in _poly_c(zs)]

def _polymul(a: List[float], b: List[float]) -> List[float]:
    out = [0.0] * (len(a) + len(b) - 1)
    for i in range(len(a)):
        for j in range(len(b)):
            out[i + j] += a[i] * b[j]
    return out

def _roots(p: List[float]) -> List[complex]:
    # numpy.roots: strip leading/trailing zeros, companion matrix eigenvalues
    nz = [i for i in range(len(p)) if p[i] != 0.0]
    if len(nz) == 0:
        return []
    trailing = len(p) - nz[-1] - 1
    q = p[nz[0]:nz[-1] + 1]
    N = len(q)
    out: List[complex] = []
    if N > 1:
        A = np.zeros((N - 1, N - 1), dtype=float)
        for i in range(1, N - 1):
            A[i, i - 1] = 1.0
        for j in range(N - 1):
            A[0, j] = -q[j + 1] / q[0]
        ev = _alg.eigvals(A)
        for i in range(ev.size):
            out.append(complex(ev[i]))
    for _ in range(trailing):
        out.append(complex(0.0))
    return out

# ----------------------------------------------------------- analog prototypes

def buttap(N: int):
    if N < 0:
        raise ValueError('Filter order must be a nonnegative integer')
    p: List[complex] = []
    for i in range(N):
        m = float(-N + 1 + 2 * i)
        p.append(-cmath.exp(1j * math.pi * m / (2 * N)))
    return _carr(List[complex]()), _carr(p), 1.0

def cheb1ap(N: int, rp: float):
    if N < 0:
        raise ValueError('Filter order must be a nonnegative integer')
    elif N == 0:
        return _carr(List[complex]()), _carr(List[complex]()), 10 ** (-rp / 20)
    eps = math.sqrt(10 ** (0.1 * rp) - 1.0)
    mu = 1.0 / N * math.asinh(1 / eps)
    p: List[complex] = []
    for i in range(N):
        theta = math.pi * float(-N + 1 + 2 * i) / (2 * N)
        p.append(-cmath.sinh(complex(mu, theta)))
    prod = complex(1.0)
    for c in p:
        prod *= -c
    k = prod.real
    if N % 2 == 0:
        k = k / math.sqrt(1 + eps * eps)
    return _carr(List[complex]()), _carr(p), k

def cheb2ap(N: int, rs: float):
    if N < 0:
        raise ValueError('Filter order must be a nonnegative integer')
    elif N == 0:
        return _carr(List[complex]()), _carr(List[complex]()), 1.0
    de = 1.0 / math.sqrt(10 ** (0.1 * rs) - 1)
    mu = math.asinh(1.0 / de) / N
    m: List[float] = []
    if N % 2:
        for v in range(-N + 1, 0, 2):
            m.append(float(v))
        for v in range(2, N, 2):
            m.append(float(v))
    else:
        for v in range(-N + 1, N, 2):
            m.append(float(v))
    z = [1j / math.sin(mm * math.pi / (2 * N)) for mm in m]
    p: List[complex] = []
    for i in range(N):
        theta1 = math.pi * float(-N + 1 + 2 * i) / (2 * N)
        p.append(-1 / cmath.sinh(complex(mu, theta1)))
    pp = complex(1.0)
    for c in p:
        pp *= -c
    pz = complex(1.0)
    for c in z:
        pz *= -c
    k = (pp / pz).real
    return _carr(z), _carr(p), k

# Bessel: zeros of the reverse Bessel polynomial via companion eigenvalues
# polished with Newton iterations on the exact integer-coefficient polynomial.
def _falling_factorial(x: int, n: int) -> float:
    val = 1.0
    for a in range(x - n + 1, x + 1):
        val *= float(a)
    return val

def _bessel_poly_coeffs(n: int) -> List[float]:
    # coefficients of theta_n(s) with highest power first
    out: List[float] = []
    for k in range(n + 1):
        num = _falling_factorial(2 * n - k, n)
        den = (2.0 ** (n - k)) * math.factorial(k)
        out.append(num / den)
    # out[k] is coefficient a_k of s^k in the reverse Bessel polynomial
    return out[::-1]

def _polyval_c(c: List[float], x: complex) -> complex:
    r = complex(0.0)
    for v in c:
        r = r * x + v
    return r

def _bessel_zeros(N: int) -> List[complex]:
    # roots of reverse Bessel polynomial theta_N; scipy returns zeros of
    # K_{N+1/2}(1/x) which equal 1/roots(theta_N).
    c = _bessel_poly_coeffs(N)
    r = _roots(c)
    dc = [c[i] * (len(c) - 1 - i) for i in range(len(c) - 1)]
    for i in range(len(r)):
        x = r[i]
        for _ in range(50):
            f = _polyval_c(c, x)
            fp = _polyval_c(dc, x)
            if fp == 0:
                break
            dx = f / fp
            x -= dx
            if abs(dx) <= 1e-16 * abs(x):
                break
        r[i] = x
    # reciprocal: zeros of K(1/x)
    x = [1 / v for v in r]
    # symmetrize conjugates (sorted by imag part like scipy's mean with conj reverse)
    x.sort(key=lambda v: (v.imag, v.real))
    n = len(x)
    y = [(x[i] + x[n - 1 - i].conjugate()) / 2 for i in range(n)]
    return y

def besselap(N: int, norm: str = 'phase'):
    if N < 0:
        raise ValueError('Filter order must be a nonnegative integer')
    if N == 0:
        return _carr(List[complex]()), _carr(List[complex]()), 1.0
    p = [1 / v for v in _bessel_zeros(N)]
    a_last = _falling_factorial(2 * N, N) / (2.0 ** N)
    a_last = float(round(a_last))
    k = 1.0
    if norm == 'delay':
        k = a_last
    elif norm == 'mag':
        # find w where |k/prod(jw - p)| = 1/sqrt(2) by Newton (secant) iteration
        def G(w: float):
            prod = complex(1.0)
            for pp in p:
                prod *= complex(0, w) - pp
            return abs(a_last / prod) - 1 / math.sqrt(2)
        x0 = 1.5
        x1 = x0 * (1 + 1e-4) + 1e-4
        q0 = G(x0)
        q1 = G(x1)
        for _ in range(50):
            if q1 == q0:
                break
            x2 = x1 - q1 * (x1 - x0) / (q1 - q0)
            if abs(x2 - x1) < 1.48e-8:
                x1 = x2
                break
            x0, q0 = x1, q1
            x1 = x2
            q1 = G(x1)
        nf = x1
        p = [v / nf for v in p]
        k = nf ** (-N) * a_last
    elif norm == 'phase':
        f = 10 ** (-math.log10(a_last) / N)
        p = [v * f for v in p]
        k = 1.0
    else:
        raise ValueError('normalization not understood')
    return _carr(List[complex]()), _carr(p), k

# ------------------------------------------------------------- transformations

def _relative_degree(z: List[complex], p: List[complex]) -> int:
    d = len(p) - len(z)
    if d < 0:
        raise ValueError('Improper transfer function. Must have at least as many poles as zeros.')
    return d

def lp2lp_zpk(z, p, k: float, wo: float = 1.0):
    zl = _clist(z)
    pl = _clist(p)
    degree = _relative_degree(zl, pl)
    return _carr([wo * v for v in zl]), _carr([wo * v for v in pl]), k * wo ** degree

def lp2hp_zpk(z, p, k: float, wo: float = 1.0):
    zl = _clist(z)
    pl = _clist(p)
    degree = _relative_degree(zl, pl)
    zh = [wo / v for v in zl] + [complex(0.0)] * degree
    ph = [wo / v for v in pl]
    nz = complex(1.0)
    for v in zl:
        nz *= -v
    npp = complex(1.0)
    for v in pl:
        npp *= -v
    return _carr(zh), _carr(ph), k * (nz / npp).real

def lp2bp_zpk(z, p, k: float, wo: float = 1.0, bw: float = 1.0):
    zl = _clist(z)
    pl = _clist(p)
    degree = _relative_degree(zl, pl)
    zlp = [v * bw / 2 for v in zl]
    plp = [v * bw / 2 for v in pl]
    zb = [v + cmath.sqrt(v ** 2 - wo ** 2) for v in zlp] + [v - cmath.sqrt(v ** 2 - wo ** 2) for v in zlp]
    pb = [v + cmath.sqrt(v ** 2 - wo ** 2) for v in plp] + [v - cmath.sqrt(v ** 2 - wo ** 2) for v in plp]
    zb += [complex(0.0)] * degree
    return _carr(zb), _carr(pb), k * bw ** degree

def lp2bs_zpk(z, p, k: float, wo: float = 1.0, bw: float = 1.0):
    zl = _clist(z)
    pl = _clist(p)
    degree = _relative_degree(zl, pl)
    zh = [bw / 2 / v for v in zl]
    ph = [bw / 2 / v for v in pl]
    zb = [v + cmath.sqrt(v ** 2 - wo ** 2) for v in zh] + [v - cmath.sqrt(v ** 2 - wo ** 2) for v in zh]
    pb = [v + cmath.sqrt(v ** 2 - wo ** 2) for v in ph] + [v - cmath.sqrt(v ** 2 - wo ** 2) for v in ph]
    zb += [complex(0, wo)] * degree
    zb += [complex(0, -wo)] * degree
    nz = complex(1.0)
    for v in zl:
        nz *= -v
    npp = complex(1.0)
    for v in pl:
        npp *= -v
    return _carr(zb), _carr(pb), k * (nz / npp).real

def bilinear_zpk(z, p, k: float, fs: float):
    zl = _clist(z)
    pl = _clist(p)
    degree = _relative_degree(zl, pl)
    fs2 = 2.0 * fs
    zz = [(fs2 + v) / (fs2 - v) for v in zl] + [complex(-1.0)] * degree
    pz = [(fs2 + v) / (fs2 - v) for v in pl]
    a = complex(1.0)
    for v in zl:
        a *= fs2 - v
    b = complex(1.0)
    for v in pl:
        b *= fs2 - v
    return _carr(zz), _carr(pz), k * (a / b).real

# ------------------------------------------------------------- representations

def zpk2tf(z, p, k: float):
    b = [k * c for c in _poly_real(_clist(z))]
    a = _poly_real(_clist(p))
    return _farr(b), _farr(a)

def normalize(b, a):
    num = _flist(b)
    den = _flist(a)
    while len(den) > 1 and den[0] == 0.0:
        den = den[1:]
    if den[0] == 0.0:
        raise ValueError('Denominator must have at least on nonzero element.')
    d0 = den[0]
    num = [v / d0 for v in num]
    den = [v / d0 for v in den]
    lz = 0
    for v in num:
        if abs(v) <= 1e-14:
            lz += 1
        else:
            break
    if lz == len(num):
        lz -= 1
    return _farr(num[lz:]), _farr(den)

def tf2zpk(b, a):
    bn, an = normalize(b, a)
    bl = _flist(bn)
    al = _flist(an)
    k = bl[0]
    bl = [v / k for v in bl]
    return _carr(_roots(bl)), _carr(_roots(al)), k

def sos2tf(sos):
    s = np.asarray(sos)
    b = [1.0]
    a = [1.0]
    for i in range(s.shape[0]):
        b = _polymul(b, [float(s[i, 0]), float(s[i, 1]), float(s[i, 2])])
        a = _polymul(a, [float(s[i, 3]), float(s[i, 4]), float(s[i, 5])])
    return _farr(b), _farr(a)

def sos2zpk(sos):
    s = np.asarray(sos)
    n = s.shape[0]
    z = np.zeros(2 * n, dtype=complex)
    p = np.zeros(2 * n, dtype=complex)
    k = 1.0
    for i in range(n):
        zz, pp, kk = tf2zpk([float(s[i, 0]), float(s[i, 1]), float(s[i, 2])],
                            [float(s[i, 3]), float(s[i, 4]), float(s[i, 5])])
        for j in range(zz.size):
            z[2 * i + j] = zz[j]
        for j in range(pp.size):
            p[2 * i + j] = pp[j]
        k *= kk
    return z, p, k

def _cplxreal(zin: List[complex]):
    tol = 100 * _EPS
    if len(zin) == 0:
        return List[complex](), List[float]()
    z = sorted(zin, key=lambda v: (v.real, abs(v.imag)))
    zr: List[float] = []
    zc_all: List[complex] = []
    for v in z:
        if abs(v.imag) <= tol * abs(v):
            zr.append(v.real)
        else:
            zc_all.append(v)
    zp = [v for v in zc_all if v.imag > 0]
    zn = [v for v in zc_all if v.imag < 0]
    if len(zp) != len(zn):
        raise ValueError('Array contains complex value with no matching conjugate.')
    # sort runs of (approximately) equal real part by |imag|
    i = 0
    while i < len(zp):
        j = i
        while j + 1 < len(zp) and zp[j + 1].real - zp[j].real <= tol * abs(zp[j]):
            j += 1
        if j > i:
            a = sorted(zp[i:j + 1], key=lambda v: abs(v.imag))
            b = sorted(zn[i:j + 1], key=lambda v: abs(v.imag))
            for t in range(j - i + 1):
                zp[i + t] = a[t]
                zn[i + t] = b[t]
        i = j + 1
    for t in range(len(zp)):
        if abs(zp[t] - zn[t].conjugate()) > tol * abs(zn[t]):
            raise ValueError('Array contains complex value with no matching conjugate.')
    zc = [(zp[t] + zn[t].conjugate()) / 2 for t in range(len(zp))]
    return zc, zr

def _argsort_stable(key: List[float]) -> List[int]:
    idx = list(range(len(key)))
    idx.sort(key=lambda i: key[i])
    return idx

def _nearest_idx(fro: List[complex], to: complex, which: str) -> int:
    order = _argsort_stable([abs(v - to) for v in fro])
    if which == 'any':
        return order[0]
    for i in order:
        r = _isreal(fro[i])
        if (which == 'real' and r) or (which == 'complex' and not r):
            return i
    raise ValueError('no matching element')

def _single_zpksos(z: List[complex], p: List[complex], k: float) -> List[float]:
    sos = [0.0] * 6
    b = [k * c for c in _poly_real(z)]
    a = _poly_real(p)
    for i in range(len(b)):
        sos[3 - len(b) + i] = b[i]
    for i in range(len(a)):
        sos[6 - len(a) + i] = a[i]
    return sos

def zpk2sos(z, p, k: float, pairing: Optional[str] = None, analog: bool = False):
    zl = _clist(z)
    pl = _clist(p)
    pr: str = ('minimal' if analog else 'nearest') if pairing is None else pairing
    if pr not in ('nearest', 'keep_odd', 'minimal'):
        raise ValueError(f"pairing must be one of ['nearest', 'keep_odd', 'minimal'], not {pr}")
    if analog and pr != 'minimal':
        raise ValueError('for analog zpk2sos conversion, pairing must be "minimal"')
    if len(zl) == 0 and len(pl) == 0:
        out = np.zeros((1, 6), dtype=float)
        if not analog:
            out[0, 0] = k
            out[0, 3] = 1.0
        else:
            out[0, 2] = k
            out[0, 5] = 1.0
        return out
    n_sections = 0
    if pr != 'minimal':
        lp0 = len(pl)
        lz0 = len(zl)
        pl += [complex(0.0)] * max(lz0 - lp0, 0)
        zl += [complex(0.0)] * max(lp0 - lz0, 0)
        n_sections = (max(len(pl), len(zl)) + 1) // 2
        if len(pl) % 2 == 1 and pr == 'nearest':
            pl.append(complex(0.0))
            zl.append(complex(0.0))
    else:
        if len(pl) < len(zl):
            raise ValueError('for analog zpk2sos conversion, must have len(p)>=len(z)')
        n_sections = (len(pl) + 1) // 2

    zc, zr = _cplxreal(zl)
    zl = zc + [complex(v) for v in zr]
    pc, prr = _cplxreal(pl)
    pl = pc + [complex(v) for v in prr]

    def idx_worst(v: List[complex]) -> int:
        best = 0
        bv = 0.0
        for i in range(len(v)):
            m = abs(1 - abs(v[i])) if not analog else abs(v[i].real)
            if i == 0 or m < bv:
                bv = m
                best = i
        return best

    def nreal(v: List[complex]) -> int:
        c = 0
        for x in v:
            if _isreal(x):
                c += 1
        return c

    sos = np.zeros((n_sections, 6), dtype=float)
    for si in range(n_sections - 1, -1, -1):
        p1_idx = idx_worst(pl)
        p1 = pl.pop(p1_idx)
        row: List[float]
        if _isreal(p1) and nreal(pl) == 0:
            if pr != 'minimal':
                zi = _nearest_idx(zl, p1, 'real')
                z1 = zl.pop(zi)
                row = _single_zpksos([z1, complex(0.0)], [p1, complex(0.0)], 1.0)
            elif len(zl) > 0:
                zi = _nearest_idx(zl, p1, 'real')
                z1 = zl.pop(zi)
                row = _single_zpksos([z1], [p1], 1.0)
            else:
                row = _single_zpksos(List[complex](), [p1], 1.0)
        elif (len(pl) + 1 == len(zl) and not _isreal(p1) and nreal(pl) == 1 and nreal(zl) == 1):
            zi = _nearest_idx(zl, p1, 'complex')
            z1 = zl.pop(zi)
            row = _single_zpksos([z1, z1.conjugate()], [p1, p1.conjugate()], 1.0)
        else:
            p2 = complex(0.0)
            if _isreal(p1):
                ridx = [i for i in range(len(pl)) if _isreal(pl[i])]
                sub = [pl[i] for i in ridx]
                p2 = pl.pop(ridx[idx_worst(sub)])
            else:
                p2 = p1.conjugate()
            if len(zl) > 0:
                zi = _nearest_idx(zl, p1, 'any')
                z1 = zl.pop(zi)
                if not _isreal(z1):
                    row = _single_zpksos([z1, z1.conjugate()], [p1, p2], 1.0)
                else:
                    if len(zl) > 0:
                        zi2 = _nearest_idx(zl, p1, 'real')
                        z2 = zl.pop(zi2)
                        row = _single_zpksos([z1, z2], [p1, p2], 1.0)
                    else:
                        row = _single_zpksos([z1], [p1, p2], 1.0)
            else:
                row = _single_zpksos(List[complex](), [p1, p2], 1.0)
        for j in range(6):
            sos[si, j] = row[j]
    for j in range(3):
        sos[0, j] *= k
    return sos

def tf2sos(b, a, pairing: Optional[str] = None, analog: bool = False):
    z, p, k = tf2zpk(b, a)
    return zpk2sos(z, p, k, pairing, analog)

# ------------------------------------------------------------------ iirfilter

def _band(btype: str) -> str:
    if btype in ('band', 'bandpass', 'pass', 'bp'):
        return 'bandpass'
    if btype in ('bs', 'bandstop', 'bands', 'stop'):
        return 'bandstop'
    if btype in ('l', 'low', 'lowpass', 'lp'):
        return 'lowpass'
    if btype in ('high', 'highpass', 'h', 'hp'):
        return 'highpass'
    raise ValueError(f"'{btype}' is an invalid bandtype for filter.")

def _iir_zpk(N: int, Wn, rp: Optional[float], rs: Optional[float], btype: str,
             analog: bool, ftype: str, fs: Optional[float]):
    W = _flist(Wn)
    if fs is not None:
        if analog:
            raise ValueError('fs cannot be specified for an analog filter')
        W = [w / (fs / 2) for w in W]
    for w in W:
        if w <= 0:
            raise ValueError('filter critical frequencies must be greater than 0')
    if len(W) > 1 and not W[0] < W[1]:
        raise ValueError('Wn[0] must be less than Wn[1]')
    bt = _band(btype.lower())
    ft = ftype.lower()
    if rp is not None and rp < 0:
        raise ValueError('passband ripple (rp) must be positive')
    if rs is not None and rs < 0:
        raise ValueError('stopband attenuation (rs) must be positive')
    if ft in ('butter', 'butterworth'):
        z, p, k = buttap(N)
    elif ft in ('bessel', 'bessel_phase'):
        z, p, k = besselap(N, 'phase')
    elif ft == 'bessel_delay':
        z, p, k = besselap(N, 'delay')
    elif ft == 'bessel_mag':
        z, p, k = besselap(N, 'mag')
    elif ft in ('cheby1', 'chebyshev1', 'chebyshevi'):
        if rp is None:
            raise ValueError('passband ripple (rp) must be provided to design a Chebyshev I filter.')
        z, p, k = cheb1ap(N, rp)
    elif ft in ('cheby2', 'chebyshev2', 'chebyshevii'):
        if rs is None:
            raise ValueError('stopband attenuation (rs) must be provided to design a Chebyshev II filter.')
        z, p, k = cheb2ap(N, rs)
    elif ft in ('ellip', 'elliptic', 'cauer'):
        if rs is None or rp is None:
            raise ValueError('Both rp and rs must be provided to design an elliptic filter.')
        z, p, k = ellipap(N, rp, rs)
    else:
        raise ValueError(f"'{ftype}' is not a valid basic IIR filter.")
    fsd = 2.0
    warped: List[float]
    if not analog:
        for w in W:
            if w <= 0 or w >= 1:
                if fs is not None:
                    raise ValueError(f'Digital filter critical frequencies must be 0 < Wn < fs/2 (fs={fs} -> fs/2={fs / 2})')
                raise ValueError('Digital filter critical frequencies must be 0 < Wn < 1')
        warped = [2 * fsd * math.tan(math.pi * w / fsd) for w in W]
    else:
        warped = W
    if bt == 'lowpass' or bt == 'highpass':
        if len(W) != 1:
            raise ValueError('Must specify a single critical frequency Wn for lowpass or highpass filter')
        if bt == 'lowpass':
            z, p, k = lp2lp_zpk(z, p, k, warped[0])
        else:
            z, p, k = lp2hp_zpk(z, p, k, warped[0])
    else:
        if len(W) < 2:
            raise ValueError('Wn must specify start and stop frequencies for bandpass or bandstop filter')
        bw = warped[1] - warped[0]
        wo = math.sqrt(warped[0] * warped[1])
        if bt == 'bandpass':
            z, p, k = lp2bp_zpk(z, p, k, wo, bw)
        else:
            z, p, k = lp2bs_zpk(z, p, k, wo, bw)
    if not analog:
        z, p, k = bilinear_zpk(z, p, k, fsd)
    return z, p, k

def _output(z, p, k: float, output: Literal[str], analog: bool):
    if output == 'zpk':
        return z, p, k
    elif output == 'ba':
        return zpk2tf(z, p, k)
    elif output == 'sos':
        return zpk2sos(z, p, k, None, analog)
    else:
        compile_error("output must be 'ba', 'zpk' or 'sos'")

def iirfilter(N: int, Wn, rp: Optional[float] = None, rs: Optional[float] = None,
              btype: str = 'band', analog: bool = False, ftype: str = 'butter',
              output: Literal[str] = 'ba', fs: Optional[float] = None):
    z, p, k = _iir_zpk(N, Wn, rp, rs, btype, analog, ftype, fs)
    return _output(z, p, k, output, analog)

def butter(N: int, Wn, btype: str = 'low', analog: bool = False,
           output: Literal[str] = 'ba', fs: Optional[float] = None):
    return iirfilter(N, Wn, btype=btype, analog=analog, output=output, ftype='butter', fs=fs)

def cheby1(N: int, rp: float, Wn, btype: str = 'low', analog: bool = False,
           output: Literal[str] = 'ba', fs: Optional[float] = None):
    return iirfilter(N, Wn, rp=rp, btype=btype, analog=analog, output=output, ftype='cheby1', fs=fs)

def cheby2(N: int, rs: float, Wn, btype: str = 'low', analog: bool = False,
           output: Literal[str] = 'ba', fs: Optional[float] = None):
    return iirfilter(N, Wn, rs=rs, btype=btype, analog=analog, output=output, ftype='cheby2', fs=fs)

def bessel(N: int, Wn, btype: str = 'low', analog: bool = False,
           output: Literal[str] = 'ba', norm: str = 'phase', fs: Optional[float] = None):
    return iirfilter(N, Wn, btype=btype, analog=analog, output=output, ftype='bessel_' + norm, fs=fs)

def _notch_peak(w0: float, Q: float, ftype: str, fs: float):
    w0 = 2 * w0 / fs
    if w0 > 1.0 or w0 < 0.0:
        raise ValueError('w0 should be such that 0 < w0 < 1')
    bw = w0 / Q * math.pi
    w0 = w0 * math.pi
    beta = math.tan(bw / 2.0)
    gain = 1.0 / (1.0 + beta)
    if ftype == 'notch':
        b = [gain, gain * -2.0 * math.cos(w0), gain]
    else:
        b = [(1.0 - gain), 0.0, -(1.0 - gain)]
    a = [1.0, -2.0 * gain * math.cos(w0), 2.0 * gain - 1.0]
    return _farr(b), _farr(a)

def iirnotch(w0: float, Q: float, fs: float = 2.0):
    return _notch_peak(w0, Q, 'notch', fs)

def iirpeak(w0: float, Q: float, fs: float = 2.0):
    return _notch_peak(w0, Q, 'peak', fs)

# ------------------------------------------------------------ frequency response

def _freq_grid(worN, whole: bool, fs: float, include_nyquist: bool):
    if isinstance(worN, int):
        N = worN
        last = 2 * math.pi if whole else math.pi
        endpoint = include_nyquist and not whole
        w = np.empty(N, dtype=float)
        for i in range(N):
            if endpoint:
                w[i] = last * i / (N - 1) if N > 1 else 0.0
            else:
                w[i] = last * i / N
        return w
    else:
        wa = _flist(worN)
        w = np.empty(len(wa), dtype=float)
        for i in range(len(wa)):
            w[i] = 2 * math.pi * wa[i] / fs
        return w

def _polyval_desc(c: List[float], x: complex) -> complex:
    # np.polynomial.polynomial.polyval with ascending coefficients b[0] + b[1] z + ...
    r = complex(0.0)
    for i in range(len(c) - 1, -1, -1):
        r = r * x + c[i]
    return r

def freqz(b, a = 1.0, worN = 512, whole: bool = False, fs: float = 2 * math.pi,
          include_nyquist: bool = False):
    # Evaluated by direct polynomial evaluation at exp(-jw) (scipy uses an FFT
    # shortcut for FIR + integer worN; results agree to rounding).
    bl = _flist(b)
    al = _flist(a)
    w = _freq_grid(worN, whole, fs, include_nyquist)
    h = np.empty(w.size, dtype=complex)
    for i in range(w.size):
        zm1 = cmath.exp(complex(0, -w[i]))
        h[i] = _polyval_desc(bl, zm1) / _polyval_desc(al, zm1)
    return w * (fs / (2 * math.pi)), h

def freqz_sos(sos, worN = 512, whole: bool = False, fs: float = 2 * math.pi):
    s = np.asarray(sos)
    w = _freq_grid(worN, whole, fs, False)
    h = np.ones(w.size, dtype=complex)
    for r in range(s.shape[0]):
        bl = [float(s[r, 0]), float(s[r, 1]), float(s[r, 2])]
        al = [float(s[r, 3]), float(s[r, 4]), float(s[r, 5])]
        for i in range(w.size):
            zm1 = cmath.exp(complex(0, -w[i]))
            h[i] *= _polyval_desc(bl, zm1) / _polyval_desc(al, zm1)
    return w * (fs / (2 * math.pi)), h

def sosfreqz(sos, worN = 512, whole: bool = False, fs: float = 2 * math.pi):
    return freqz_sos(sos, worN, whole, fs)

def freqz_zpk(z, p, k: float, worN = 512, whole: bool = False, fs: float = 2 * math.pi):
    zl = _clist(z)
    pl = _clist(p)
    w = _freq_grid(worN, whole, fs, False)
    h = np.empty(w.size, dtype=complex)
    for i in range(w.size):
        zm = cmath.exp(complex(0, w[i]))
        num = complex(k)
        for v in zl:
            num *= zm - v
        den = complex(1.0)
        for v in pl:
            den *= zm - v
        h[i] = num / den
    return w * (fs / (2 * math.pi)), h

# ------------------------------------------------------- elliptic functions
# Private complete elliptic integral / Jacobi elliptic helpers (AGM, as in
# cephes ellpj). scipy.special does not provide ellipk/ellipj natively yet.

def _agm(a: float, b: float) -> float:
    for _ in range(100):
        an = 0.5 * (a + b)
        bn = math.sqrt(a * b)
        if abs(an - bn) <= 1e-16 * an:
            return an
        a, b = an, bn
    return 0.5 * (a + b)

def _ellipk(m: float) -> float:
    # complete elliptic integral of the first kind K(m)
    if m == 1.0:
        return math.inf
    return math.pi / (2.0 * _agm(1.0, math.sqrt(1.0 - m)))

def _ellipkm1(p: float) -> float:
    # K(1 - p), accurate for small p
    if p == 0.0:
        return math.inf
    return math.pi / (2.0 * _agm(1.0, math.sqrt(p)))

def _ellipj(u: float, m: float):
    MACHEP = 1.11022302462515654042e-16
    if m < 0.0 or m > 1.0:
        return math.nan, math.nan, math.nan, math.nan
    if m < 1.0e-9:
        t = math.sin(u)
        b = math.cos(u)
        ai = 0.25 * m * (u - t * b)
        return t - ai * b, b + ai * t, 1.0 - 0.5 * m * t * t, u - ai
    if m >= 0.9999999999:
        ai = 0.25 * (1.0 - m)
        b = math.cosh(u)
        t = math.tanh(u)
        phi = 1.0 / b
        twon = b * math.sinh(u)
        sn = t + ai * (twon - u) / (b * b)
        ph = 2.0 * math.atan(math.exp(u)) - math.pi / 2 + ai * (twon - u) / b
        ai *= t * phi
        cn = phi - ai * (twon - u)
        dn = phi + ai * (twon + u)
        return sn, cn, dn, ph
    a = [0.0] * 9
    c = [0.0] * 9
    a[0] = 1.0
    b = math.sqrt(1.0 - m)
    c[0] = math.sqrt(m)
    twon = 1.0
    i = 0
    while abs(c[i] / a[i]) > MACHEP:
        if i > 7:
            break
        ai = a[i]
        i += 1
        c[i] = (ai - b) / 2.0
        t = math.sqrt(ai * b)
        a[i] = (ai + b) / 2.0
        b = t
        twon *= 2.0
    phi = twon * a[i] * u
    while True:
        t = c[i] * math.sin(phi) / a[i]
        b = phi
        phi = (math.asin(t) + phi) / 2.0
        i -= 1
        if i == 0:
            break
    t = math.cos(phi)
    return math.sin(phi), t, t / math.cos(phi - b), phi

def _pow10m1(x: float) -> float:
    return math.expm1(2.302585092994046 * x)

def _ellipdeg(n: int, m1: float) -> float:
    K1 = _ellipk(m1)
    K1p = _ellipkm1(m1)
    q1 = math.exp(-math.pi * K1p / K1)
    q = q1 ** (1.0 / n)
    num = 0.0
    for mm in range(8):
        num += q ** (mm * (mm + 1))
    den = 0.0
    for mm in range(1, 9):
        den += q ** (mm * mm)
    den = 1 + 2 * den
    return 16 * q * (num / den) ** 4

def _arc_jac_sn(w: complex, m: float) -> complex:
    def comp(kx: complex) -> complex:
        return cmath.sqrt((1 - kx) * (1 + kx))
    k = m ** 0.5
    if k > 1:
        return complex(math.nan, 0.0)
    elif k == 1:
        return cmath.atanh(w)
    ks = [k]
    niter = 0
    while ks[-1] != 0:
        k_ = ks[-1]
        k_p = math.sqrt((1 - k_) * (1 + k_))
        ks.append((1 - k_p) / (1 + k_p))
        niter += 1
        if niter > 10:
            raise ValueError('Landen transformation not converging')
    K = 1.0
    for i in range(1, len(ks)):
        K *= 1 + ks[i]
    K *= math.pi / 2
    wn = w
    for i in range(len(ks) - 1):
        kn = ks[i]
        knext = ks[i + 1]
        wn = 2 * wn / ((1 + knext) * (1 + comp(kn * wn)))
    u = 2 / math.pi * cmath.asin(wn)
    return K * u

def _arc_jac_sc1(w: float, m: float) -> float:
    z = _arc_jac_sn(complex(0.0, w), m)
    if abs(z.real) > 1e-14:
        raise ValueError('arc_jac_sc1 failed')
    return z.imag

def ellipap(N: int, rp: float, rs: float):
    EPS = 2e-16
    if N < 0:
        raise ValueError('Filter order must be a nonnegative integer')
    elif N == 0:
        return _carr(List[complex]()), _carr(List[complex]()), 10 ** (-rp / 20)
    elif N == 1:
        p1 = -math.sqrt(1.0 / _pow10m1(0.1 * rp))
        return _carr(List[complex]()), _carr([complex(p1)]), -p1
    eps_sq = _pow10m1(0.1 * rp)
    eps = math.sqrt(eps_sq)
    ck1_sq = eps_sq / _pow10m1(0.1 * rs)
    if ck1_sq == 0:
        raise ValueError('Cannot design a filter with given rp and rs specifications.')
    val0 = _ellipk(ck1_sq)
    m = _ellipdeg(N, ck1_sq)
    capk = _ellipk(m)
    js = list(range(1 - N % 2, N, 2))
    s: List[float] = []
    c: List[float] = []
    d: List[float] = []
    for j in js:
        a1, a2, a3, _ = _ellipj(j * capk / N, m)
        s.append(a1)
        c.append(a2)
        d.append(a3)
    z: List[complex] = []
    for v in s:
        if abs(v) > EPS:
            z.append(1j * (1.0 / (math.sqrt(m) * v)))
    z = z + [v.conjugate() for v in z]
    r = _arc_jac_sc1(1.0 / eps, ck1_sq)
    v0 = capk * r / (N * val0)
    sv, cv, dv, _ = _ellipj(v0, 1 - m)
    p: List[complex] = []
    for i in range(len(js)):
        p.append(-(c[i] * d[i] * sv * cv + 1j * s[i] * dv) / (1 - (d[i] * sv) ** 2.0))
    if N % 2:
        tot = 0.0
        for v in p:
            tot += (v * v.conjugate()).real
        thr = EPS * math.sqrt(tot)
        newp = [v for v in p if abs(v.imag) > thr]
        p = p + [v.conjugate() for v in newp]
    else:
        p = p + [v.conjugate() for v in p]
    pp = complex(1.0)
    for v in p:
        pp *= -v
    pz = complex(1.0)
    for v in z:
        pz *= -v
    k = (pp / pz).real
    if N % 2 == 0:
        k = k / math.sqrt(1 + eps_sq)
    return _carr(z), _carr(p), k

def ellip(N: int, rp: float, rs: float, Wn, btype: str = 'low', analog: bool = False,
          output: Literal[str] = 'ba', fs: Optional[float] = None):
    return iirfilter(N, Wn, rp=rp, rs=rs, btype=btype, analog=analog, output=output, ftype='elliptic', fs=fs)

# --------------------------------------------------------- order selection

def _validate_gpass_gstop(gpass: float, gstop: float):
    if gpass <= 0.0:
        raise ValueError('gpass should be larger than 0.0')
    elif gstop <= 0.0:
        raise ValueError('gstop should be larger than 0.0')
    elif gpass > gstop:
        raise ValueError('gpass should be smaller than gstop')

def _band_stop_obj(wp: float, ind: int, passb: List[float], stopb: List[float],
                   gpass: float, gstop: float, kind: str) -> float:
    pc = list(passb)
    pc[ind] = wp
    nat = math.inf
    for sb in stopb:
        v = abs(sb * (pc[0] - pc[1]) / (sb ** 2 - pc[0] * pc[1]))
        if v < nat:
            nat = v
    if kind == 'butter':
        GSTOP = 10 ** (0.1 * abs(gstop))
        GPASS = 10 ** (0.1 * abs(gpass))
        return math.log10((GSTOP - 1.0) / (GPASS - 1.0)) / (2 * math.log10(nat))
    elif kind == 'cheby':
        GSTOP = 10 ** (0.1 * abs(gstop))
        GPASS = 10 ** (0.1 * abs(gpass))
        return math.acosh(math.sqrt((GSTOP - 1.0) / (GPASS - 1.0))) / math.acosh(nat)
    else:
        GSTOP = 10 ** (0.1 * gstop)
        GPASS = 10 ** (0.1 * gpass)
        arg1 = math.sqrt((GPASS - 1.0) / (GSTOP - 1.0))
        arg0 = 1.0 / nat
        return _ellipk(arg0 ** 2) * _ellipk(1 - arg1 ** 2) / (_ellipk(1 - arg0 ** 2) * _ellipk(arg1 ** 2))

def _fminbound(func, x1: float, x2: float, xatol: float = 1e-5, maxfun: int = 500) -> float:
    # scipy.optimize._minimize_scalar_bounded (Brent)
    sqrt_eps = math.sqrt(2.2e-16)
    golden_mean = 0.5 * (3.0 - math.sqrt(5.0))
    a = x1
    b = x2
    fulc = a + golden_mean * (b - a)
    nfc = fulc
    xf = fulc
    rat = 0.0
    e = 0.0
    x = xf
    fx = func(x)
    num = 1
    ffulc = fx
    fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1
    while abs(xf - xm) > tol2 - 0.5 * (b - a):
        golden = True
        if abs(e) > tol1:
            golden = False
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            r = e
            e = rat
            if abs(p) < abs(0.5 * q * r) and p > q * (a - xf) and p < q * (b - xf):
                rat = (p + 0.0) / q
                x = xf + rat
                if x - a < tol2 or b - x < tol2:
                    si = (1.0 if xm - xf > 0 else (-1.0 if xm - xf < 0 else 0.0)) + (1.0 if xm - xf == 0 else 0.0)
                    rat = tol1 * si
            else:
                golden = True
        if golden:
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean * e
        si = (1.0 if rat > 0 else (-1.0 if rat < 0 else 0.0)) + (1.0 if rat == 0 else 0.0)
        x = xf + si * max(abs(rat), tol1)
        fu = func(x)
        num += 1
        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if fu <= fnfc or nfc == xf:
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif fu <= ffulc or fulc == xf or fulc == nfc:
                fulc, ffulc = x, fu
        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1
        if num >= maxfun:
            break
    return xf

def _ord_setup(wp, ws, gpass: float, gstop: float, analog: bool, fs: Optional[float], kind: str):
    _validate_gpass_gstop(gpass, gstop)
    wpl = _flist(wp)
    wsl = _flist(ws)
    if fs is not None:
        if analog:
            raise ValueError('fs cannot be specified for an analog filter')
        wpl = [2 * v / fs for v in wpl]
        wsl = [2 * v / fs for v in wsl]
    ftype = 2 * (len(wpl) - 1) + 1
    if wpl[0] >= wsl[0]:
        ftype += 1
    if not analog:
        passb = [math.tan(math.pi * v / 2.0) for v in wpl]
        stopb = [math.tan(math.pi * v / 2.0) for v in wsl]
    else:
        passb = wpl
        stopb = wsl
    nats: List[float] = []
    if ftype == 1:
        nats = [stopb[i] / passb[i] for i in range(len(stopb))]
    elif ftype == 2:
        nats = [passb[i] / stopb[i] for i in range(len(stopb))]
    elif ftype == 3:
        p0 = passb
        s0 = stopb
        wp0 = _fminbound(lambda w: _band_stop_obj(w, 0, p0, s0, gpass, gstop, kind), passb[0], stopb[0] - 1e-12)
        wp1 = _fminbound(lambda w: _band_stop_obj(w, 1, p0, s0, gpass, gstop, kind), stopb[1] + 1e-12, passb[1])
        passb = [wp0, wp1]
        nats = [sb * (passb[0] - passb[1]) / (sb ** 2 - passb[0] * passb[1]) for sb in stopb]
    else:
        nats = [(sb ** 2 - passb[0] * passb[1]) / (sb * (passb[0] - passb[1])) for sb in stopb]
    nat = min([abs(v) for v in nats])
    return ftype, passb, nat

def _post_wn(WN: List[float], analog: bool, fs: Optional[float], like):
    wn = WN if analog else [math.atan(v) * 2.0 / math.pi for v in WN]
    if fs is not None:
        wn = [v * fs / 2 for v in wn]
    if isinstance(like, float) or isinstance(like, int):
        return wn[0]
    else:
        return _farr(wn)

def buttord(wp, ws, gpass: float, gstop: float, analog: bool = False, fs: Optional[float] = None):
    # Returns (ord, wn); wn is a float for scalar wp, an ndarray otherwise.
    ftype, passb, nat = _ord_setup(wp, ws, gpass, gstop, analog, fs, 'butter')
    GSTOP = 10 ** (0.1 * abs(gstop))
    GPASS = 10 ** (0.1 * abs(gpass))
    ord = int(math.ceil(math.log10((GSTOP - 1.0) / (GPASS - 1.0)) / (2 * math.log10(nat))))
    W0 = (GPASS - 1.0) ** (-1.0 / (2.0 * ord)) if ord != 0 else 1.0
    WN: List[float]
    if ftype == 1:
        WN = [W0 * v for v in passb]
    elif ftype == 2:
        WN = [v / W0 for v in passb]
    elif ftype == 3:
        discr = math.sqrt((passb[1] - passb[0]) ** 2 + 4 * W0 ** 2 * passb[0] * passb[1])
        WN = sorted([abs((passb[1] - passb[0] + discr) / (2 * W0)), abs((passb[1] - passb[0] - discr) / (2 * W0))])
    else:
        WN = sorted([abs(-w0 * (passb[1] - passb[0]) / 2.0 + math.sqrt(w0 ** 2 / 4.0 * (passb[1] - passb[0]) ** 2 + passb[0] * passb[1])) for w0 in (-W0, W0)])
    return ord, _post_wn(WN, analog, fs, wp)

def cheb1ord(wp, ws, gpass: float, gstop: float, analog: bool = False, fs: Optional[float] = None):
    ftype, passb, nat = _ord_setup(wp, ws, gpass, gstop, analog, fs, 'cheby')
    GSTOP = 10 ** (0.1 * abs(gstop))
    GPASS = 10 ** (0.1 * abs(gpass))
    v_pass_stop = math.acosh(math.sqrt((GSTOP - 1.0) / (GPASS - 1.0)))
    ord = int(math.ceil(v_pass_stop / math.acosh(nat)))
    return ord, _post_wn(passb, analog, fs, wp)

def ellipord(wp, ws, gpass: float, gstop: float, analog: bool = False, fs: Optional[float] = None):
    ftype, passb, nat = _ord_setup(wp, ws, gpass, gstop, analog, fs, 'ellip')
    arg1_sq = _pow10m1(0.1 * gpass) / _pow10m1(0.1 * gstop)
    arg0 = 1.0 / nat
    d00 = _ellipk(arg0 ** 2)
    d01 = _ellipkm1(arg0 ** 2)
    d10 = _ellipk(arg1_sq)
    d11 = _ellipkm1(arg1_sq)
    ord = int(math.ceil(d00 * d11 / (d01 * d10)))
    return ord, _post_wn(passb, analog, fs, wp)

# -------------------------------------------------------------- group delay

def group_delay(system, w = 512, whole: bool = False, fs: float = 2 * math.pi):
    b = _flist(system[0])
    a = _flist(system[1])
    wv: List[float] = []
    if isinstance(w, int):
        last = 2 * math.pi if whole else math.pi
        wv = [last * i / w for i in range(w)]
    else:
        wv = [2 * math.pi * v / fs for v in _flist(w)]
    ar = a[::-1]
    c = [0.0] * (len(b) + len(ar) - 1)
    for i in range(len(b)):
        for j in range(len(ar)):
            c[i + j] += b[i] * ar[j]
    cr = [c[i] * i for i in range(len(c))]
    n = len(wv)
    gd = np.empty(n, dtype=float)
    wo = np.empty(n, dtype=float)
    for i in range(n):
        z = cmath.exp(complex(0, -wv[i]))
        num = _polyval_desc(cr, z)
        den = _polyval_desc(c, z)
        g = (num / den).real - len(a) + 1
        if math.isnan(g) or math.isinf(g):
            g = 0.0
        gd[i] = g
        wo[i] = wv[i] * (fs / (2 * math.pi))
    return wo, gd
