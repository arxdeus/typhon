# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# scipy.stats.qmc: Sobol (Joe-Kuo direction numbers, LMS + digital shift
# scrambling), Halton (scrambled van der Corput), LatinHypercube, discrepancy,
# update_discrepancy, scale, van_der_corput, primes.
#
# Engines take ``rng`` (int seed or numpy Generator) and, as in scipy, draw
# from a child generator spawned from it, so sample streams are identical.

import numpy as np
from numpy.random import Generator, PCG64
import scipy.special._kernels as K
from ._sobol_dn import SOBOL_DN, SOBOL_MAXDIM

# ------------------------------------------------------------------ helpers

def _child_rng(rng) -> Generator[PCG64]:
    # scipy: rng normalized with np.random.default_rng, then one child spawned
    if rng is None:
        return np.random.default_rng().spawn(1)[0]
    elif isinstance(rng, int):
        return np.random.default_rng(rng).spawn(1)[0]
    else:
        return rng.spawn(1)[0]

def primes_from_2_to(n: int):
    out = []
    for c in range(2, n):
        isp = True
        q = 2
        while q * q <= c:
            if c % q == 0:
                isp = False
                break
            q += 1
        if isp:
            out.append(c)
    return np.array(out, dtype=int)

def n_primes(n: int) -> List[int]:
    out = []
    c = 2
    while len(out) < n:
        isp = True
        q = 2
        while q * q <= c:
            if c % q == 0:
                isp = False
                break
            q += 1
        if isp:
            out.append(c)
        c += 1
    return out

def _ensure_in_unit_hypercube(sample):
    s = np.ascontiguousarray(np.asarray(sample).astype(float))
    if s.ndim != 2:
        raise ValueError("Sample is not a 2D array")
    if float(np.max(s)) > 1. or float(np.min(s)) < 0.:
        raise ValueError("Sample is not in unit hypercube")
    return s

def scale(sample, l_bounds, u_bounds, *, reverse: bool = False):
    s = np.asarray(sample).astype(float)
    if s.ndim != 2:
        raise ValueError('Sample is not a 2D array')
    d = s.shape[1]
    lower = np.broadcast_to(np.asarray(l_bounds).astype(float), (d,))
    upper = np.broadcast_to(np.asarray(u_bounds).astype(float), (d,))
    for i in range(d):
        if not (lower[i] < upper[i]):
            raise ValueError("Bounds are not consistent 'l_bounds' < 'u_bounds'")
    if not reverse:
        if float(np.max(s)) > 1. or float(np.min(s)) < 0.:
            raise ValueError('Sample is not in unit hypercube')
        return s * (upper - lower) + lower
    else:
        for idx in np.ndindex(s.shape):
            v = s[idx]
            if not (v >= lower[idx[1]] and v <= upper[idx[1]]):
                raise ValueError('Sample is out of bounds')
        return (s - lower) / (upper - lower)

# ------------------------------------------------------------------ discrepancy

def _centered(s, iterative: bool) -> float:
    n = s.shape[0]
    d = s.shape[1]
    disc1 = 0.0
    for i in range(n):
        prod = 1.0
        for j in range(d):
            a = abs(s[i, j] - 0.5)
            prod *= 1 + 0.5 * a - 0.5 * a ** 2
        disc1 += prod
    disc2 = 0.0
    for i in range(n):
        for j in range(n):
            prod = 1.0
            for k in range(d):
                prod *= (1 + 0.5 * abs(s[i, k] - 0.5) + 0.5 * abs(s[j, k] - 0.5)
                         - 0.5 * abs(s[i, k] - s[j, k]))
            disc2 += prod
    nn = float(n + 1 if iterative else n)
    return (13.0 / 12.0) ** d - 2.0 / nn * disc1 + 1.0 / (nn ** 2) * disc2

def _wrap(s, iterative: bool) -> float:
    n = s.shape[0]
    d = s.shape[1]
    disc = 0.0
    for i in range(n):
        for j in range(n):
            prod = 1.0
            for k in range(d):
                x = abs(s[i, k] - s[j, k])
                prod *= 3.0 / 2.0 - x + x ** 2
            disc += prod
    nn = float(n + 1 if iterative else n)
    return -(4.0 / 3.0) ** d + 1.0 / (nn ** 2) * disc

def _mixture(s, iterative: bool) -> float:
    n = s.shape[0]
    d = s.shape[1]
    disc1 = 0.0
    for i in range(n):
        prod = 1.0
        for j in range(d):
            a = abs(s[i, j] - 0.5)
            prod *= 5.0 / 3.0 - 0.25 * a - 0.25 * a ** 2
        disc1 += prod
    disc2 = 0.0
    for i in range(n):
        for j in range(n):
            prod = 1.0
            for k in range(d):
                x = abs(s[i, k] - s[j, k])
                prod *= (15.0 / 8.0 - 0.25 * abs(s[i, k] - 0.5) - 0.25 * abs(s[j, k] - 0.5)
                         - 3.0 / 4.0 * x + 0.5 * x ** 2)
            disc2 += prod
    nn = float(n + 1 if iterative else n)
    return (19.0 / 12.0) ** d - 2.0 / nn * disc1 + 1.0 / (nn ** 2) * disc2

def _l2star(s, iterative: bool) -> float:
    n = s.shape[0]
    d = s.shape[1]
    disc1 = 0.0
    for i in range(n):
        prod = 1.0
        for j in range(d):
            prod *= 1 - s[i, j] ** 2
        disc1 += prod
    disc2 = 0.0
    for i in range(n):
        tmp = 0.0
        for j in range(n):
            prod = 1.0
            for k in range(d):
                prod *= 1 - max(s[i, k], s[j, k])
            tmp += prod
        disc2 += tmp
    nn = float(n + 1 if iterative else n)
    one_div_n = 1.0 / nn
    return K.sqrt(3.0 ** (-d) - one_div_n * 2.0 ** (1 - d) * disc1 + 1 / nn ** 2 * disc2)

def discrepancy(sample, *, iterative: bool = False, method: str = "CD", workers: int = 1) -> float:
    s = _ensure_in_unit_hypercube(sample)
    if method == "CD":
        return _centered(s, iterative)
    elif method == "WD":
        return _wrap(s, iterative)
    elif method == "MD":
        return _mixture(s, iterative)
    elif method == "L2-star":
        return _l2star(s, iterative)
    else:
        raise ValueError(f"{method!r} is not a valid method. It must be one of {{'CD', 'WD', 'MD', 'L2-star'}}")

def update_discrepancy(x_new, sample, initial_disc: float) -> float:
    s = _ensure_in_unit_hypercube(sample)
    x = np.asarray(x_new).astype(float)
    n = s.shape[0] + 1
    d = s.shape[1]
    abs_ = np.empty(d, dtype=float)
    prod = 1.0
    for i in range(d):
        abs_[i] = abs(x[i] - 0.5)
        prod *= 1 + 0.5 * abs_[i] - 0.5 * abs_[i] ** 2
    disc1 = (-2 / float(n)) * prod
    disc2 = 0.0
    for i in range(n - 1):
        prod = 1.0
        for j in range(d):
            prod *= 1 + 0.5 * abs_[j] + 0.5 * abs(s[i, j] - 0.5) - 0.5 * abs(x[j] - s[i, j])
        disc2 += prod
    disc2 *= 2 / float(n) ** 2
    prod = 1.0
    for i in range(d):
        prod *= 1 + abs_[i]
    disc3 = 1 / float(n) ** 2 * prod
    return initial_disc + disc1 + disc2 + disc3

# ------------------------------------------------------------------ van der Corput / Halton

def _vdc_permutations(base: int, rng):
    count = int(K.ceil(54 / K.log2(float(base)))) - 1
    perms = np.empty((count, base), dtype=int)
    for c in range(count):
        row = np.arange(base)
        rng.shuffle(row)
        for j in range(base):
            perms[c, j] = row[j]
    return perms

def _vdc(n: int, base: int, start_index: int):
    seq = np.zeros(n, dtype=float)
    for i in range(n):
        q = start_index + i
        b2r = 1.0 / base
        acc = 0.0
        while q > 0:
            r = q % base
            acc += r * b2r
            b2r /= base
            q //= base
        seq[i] = acc
    return seq

def _vdc_scrambled(n: int, base: int, start_index: int, perms):
    seq = np.zeros(n, dtype=float)
    for i in range(n):
        q = start_index + i
        b2r = 1.0 / base
        acc = 0.0
        for j in range(perms.shape[0]):
            r = perms[j, q % base]
            acc += r * b2r
            b2r /= base
            q //= base
        seq[i] = acc
    return seq

def van_der_corput(n: int, base: int = 2, *, start_index: int = 0, scramble: bool = False,
                   permutations = None, rng = None, workers: int = 1):
    if base < 2:
        raise ValueError("'base' must be at least 2")
    if scramble:
        if permutations is None:
            r = np.random.default_rng(rng) if rng is None or isinstance(rng, int) else rng
            return _vdc_scrambled(n, base, start_index, _vdc_permutations(base, r))
        else:
            return _vdc_scrambled(n, base, start_index, np.asarray(permutations).astype(int))
    return _vdc(n, base, start_index)

class Halton:
    d: int
    scramble: bool
    base: List[int]
    rng: Generator[PCG64]
    num_generated: int
    _perms: List[np.ndarray[int, 2]]

    def __init__(self, d: int, *, scramble: bool = True, optimization = None, rng = None):
        self.d = d
        self.scramble = scramble
        self.rng = _child_rng(rng)
        self.num_generated = 0
        self.base = n_primes(d)
        self._perms = []
        if scramble:
            for b in self.base:
                self._perms.append(_vdc_permutations(b, self.rng))

    def random(self, n: int = 1, *, workers: int = 1):
        out = np.empty((n, self.d), dtype=float)
        for i in range(self.d):
            col = (_vdc_scrambled(n, self.base[i], self.num_generated, self._perms[i])
                   if self.scramble else _vdc(n, self.base[i], self.num_generated))
            for r in range(n):
                out[r, i] = col[r]
        self.num_generated += n
        return out

    def fast_forward(self, n: int):
        self.random(n)
        return self

    def integers(self, l_bounds, *, u_bounds = None, n: int = 1, endpoint: bool = False, workers: int = 1):
        return _integers(self, l_bounds, u_bounds, n, endpoint)

def _integers(eng, l_bounds, u_bounds, n: int, endpoint: bool):
    lo = np.zeros(eng.d, dtype=int)
    hi = np.zeros(eng.d, dtype=int)
    if u_bounds is None:
        hi = np.broadcast_to(np.atleast_1d(np.asarray(l_bounds)), (eng.d,)).astype(int)
    else:
        lo = np.broadcast_to(np.atleast_1d(np.asarray(l_bounds)), (eng.d,)).astype(int)
        hi = np.broadcast_to(np.atleast_1d(np.asarray(u_bounds)), (eng.d,)).astype(int)
    if endpoint:
        hi = hi + 1
    s = scale(eng.random(n), lo, hi)
    return np.floor(s).astype(int)

# ------------------------------------------------------------------ Latin hypercube

class LatinHypercube:
    d: int
    scramble: bool
    strength: int
    rng: Generator[PCG64]
    num_generated: int

    def __init__(self, d: int, *, scramble: bool = True, strength: int = 1,
                 optimization = None, rng = None):
        if strength != 1:
            raise ValueError(f"{strength!r} is not a valid strength. Only strength=1 is supported")
        self.d = d
        self.scramble = scramble
        self.strength = strength
        self.rng = _child_rng(rng)
        self.num_generated = 0

    def random(self, n: int = 1, *, workers: int = 1):
        samples = np.full((n, self.d), 0.5)
        if self.scramble:
            samples = self.rng.uniform(size=(n, self.d))
        perms = np.empty((self.d, n), dtype=int)
        for i in range(self.d):
            row = np.arange(1, n + 1)
            self.rng.shuffle(row)
            for j in range(n):
                perms[i, j] = row[j]
        out = np.empty((n, self.d), dtype=float)
        for r in range(n):
            for c in range(self.d):
                out[r, c] = (perms[c, r] - samples[r, c]) / n
        self.num_generated += n
        return out

    def integers(self, l_bounds, *, u_bounds = None, n: int = 1, endpoint: bool = False, workers: int = 1):
        return _integers(self, l_bounds, u_bounds, n, endpoint)

# ------------------------------------------------------------------ Sobol

def _bit_length(n: int) -> int:
    b = 0
    while n != 0:
        n >>= 1
        b += 1
    return b

def _low_0_bit(x: int) -> int:
    i = 1
    while x & 1:
        x >>= 1
        i += 1
    return i

def _direction_numbers(dim: int):
    if dim > SOBOL_MAXDIM:
        raise ValueError(f"Maximum supported dimensionality is {SOBOL_MAXDIM}.")
    s = ''
    for part in SOBOL_DN:
        s += part
    recs = s.split(';')
    polys = []
    vinits = []
    for d in range(dim):
        fields = recs[d].split(' ')
        polys.append(int(fields[0]))
        vinits.append([int(f) for f in fields[1:]])
    return polys, vinits

def _initialize_v(dim: int, bits: int):
    v = np.zeros((dim, bits), dtype=int)
    if dim == 0:
        return v
    polys, vinits = _direction_numbers(dim)
    mask = (1 << bits) - 1 if bits < 64 else -1
    for i in range(bits):
        v[0, i] = 1
    for d in range(1, dim):
        p = polys[d]
        m = _bit_length(p) - 1
        for j in range(m):
            v[d, j] = vinits[d][j]
        for j in range(m, bits):
            newv = v[d, j - m]
            pow2 = 1
            for k in range(m):
                pow2 = pow2 << 1
                if (p >> (m - 1 - k)) & 1:
                    newv = newv ^ (pow2 * v[d, j - k - 1])
            v[d, j] = newv & mask
    pow2 = 1
    for d in range(bits):
        for i in range(dim):
            v[i, bits - 1 - d] = (v[i, bits - 1 - d] * pow2) & mask
        pow2 = pow2 << 1
    return v

class Sobol:
    MAXDIM: ClassVar[int] = 1111
    d: int
    scramble: bool
    bits: int
    maxn: int
    rng: Generator[PCG64]
    num_generated: int
    _sv: np.ndarray[int, 2]
    _shift: np.ndarray[int, 1]
    _quasi: np.ndarray[int, 1]
    _scale: float

    def __init__(self, d: int, *, scramble: bool = True, bits: Optional[int] = None,
                 rng = None, optimization = None):
        self.d = d
        self.rng = _child_rng(rng)
        self.num_generated = 0
        self.scramble = scramble
        b = 30
        if bits is not None:
            b = bits
        if b > 64:
            raise ValueError("Maximum supported 'bits' is 64")
        if b > 62:
            raise ValueError("bits > 62 is not supported")
        self.bits = b
        self.maxn = 1 << b
        self._sv = _initialize_v(d, b)
        self._shift = np.zeros(d, dtype=int)
        if scramble:
            self._do_scramble()
        self._quasi = self._shift.copy()
        self._scale = 1.0 / float(1 << b)

    def _do_scramble(self):
        d = self.d
        bits = self.bits
        # scipy: rng_integers(rng, 2, size=..., dtype=uint32/uint64)
        if bits <= 32:
            sh = self.rng.integers(2, size=(d, bits), dtype=np.uint32)
            ltm = self.rng.integers(2, size=(d, bits, bits), dtype=np.uint32)
            self._finish_scramble(sh.astype(int), ltm.astype(int))
        else:
            sh = self.rng.integers(2, size=(d, bits), dtype=np.uint64)
            ltm = self.rng.integers(2, size=(d, bits, bits), dtype=np.uint64)
            self._finish_scramble(sh.astype(int), ltm.astype(int))

    def _finish_scramble(self, sh, ltm):
        d = self.d
        bits = self.bits
        for i in range(d):
            acc = 0
            for j in range(bits):
                acc += sh[i, j] * (1 << j)
            self._shift[i] = acc
        # lower-triangular with unit diagonal
        for dd in range(d):
            for i in range(bits):
                for j in range(bits):
                    if j > i:
                        ltm[dd, i, j] = 0
                ltm[dd, i, i] = 1
        for dd in range(d):
            for j in range(bits):
                vdj = self._sv[dd, j]
                l = 1
                t2 = 0
                for p in range(bits - 1, -1, -1):
                    lsmdp = 0
                    pw = 1
                    for i in range(bits):
                        lsmdp += ltm[dd, p, bits - 1 - i] * pw
                        pw *= 2
                    t1 = 0
                    for k in range(bits):
                        t1 += ((lsmdp >> k) & 1) * ((vdj >> k) & 1)
                    t1 = t1 % 2
                    t2 = t2 + t1 * l
                    l = 2 * l
                self._sv[dd, j] = t2

    def _draw(self, n: int, num_gen: int, out, row0: int):
        g = num_gen
        for i in range(n):
            l = _low_0_bit(g)
            for j in range(self.d):
                q = self._quasi[j] ^ self._sv[j, l - 1]
                self._quasi[j] = q
                out[row0 + i, j] = float(q) * self._scale
            g += 1

    def random(self, n: int = 1, *, workers: int = 1):
        sample = np.empty((n, self.d), dtype=float)
        if n == 0:
            return sample
        total_n = self.num_generated + n
        if total_n > self.maxn:
            raise ValueError(f"At most 2**{self.bits}={self.maxn} distinct points can be generated.")
        if self.num_generated == 0:
            for j in range(self.d):
                sample[0, j] = float(self._shift[j]) * self._scale
            if n > 1:
                self._draw(n - 1, 0, sample, 1)
        else:
            self._draw(n, self.num_generated - 1, sample, 0)
        self.num_generated += n
        return sample

    def random_base2(self, m: int):
        n = 1 << m
        total_n = self.num_generated + n
        if total_n & (total_n - 1) != 0:
            raise ValueError("The balance properties of Sobol' points require n to be a power of 2.")
        return self.random(n)

    def reset(self):
        self._quasi = self._shift.copy()
        self.num_generated = 0
        return self

    def fast_forward(self, n: int):
        g = self.num_generated - 1 if self.num_generated else 0
        cnt = n if self.num_generated else n - 1
        for i in range(cnt):
            l = _low_0_bit(g)
            for j in range(self.d):
                self._quasi[j] = self._quasi[j] ^ self._sv[j, l - 1]
            g += 1
        self.num_generated += n
        return self

    def integers(self, l_bounds, *, u_bounds = None, n: int = 1, endpoint: bool = False, workers: int = 1):
        return _integers(self, l_bounds, u_bounds, n, endpoint)
