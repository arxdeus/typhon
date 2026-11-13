# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Multivariate distributions (scipy.stats._multivariate, _qmvnt).

import numpy as np
import internal.static as static
import scipy.special._kernels as K
import scipy.linalg as LA
from scipy.special import multigammaln
from ._distn_infrastructure import check_random_state, _npsum, NAN, INF

from C import asin(float) -> float
from C import sin(float) -> float
from C import fmod(float, float) -> float
from C import atan2(float, float) -> float

_LOG_2PI = 1.8378770664093453
_LOG_2 = 0.6931471805599453
_LOG_PI = 1.1447298858494002
_PI = 3.141592653589793

def _sq1(out, size: Literal[int]):
    # scipy _squeeze_output on (size, ...) samples: drop the leading axis when size == 1
    if size == 1:
        return out[0]
    else:
        return out

def _ndsel(out: np.ndarray[float, 1], x, nd: Literal[int]):
    # scalar when the quantile has exactly ``nd`` dims (a single point), else array
    if isinstance(x, np.ndarray):
        if static.len(x.shape) <= nd:
            return out[0]
        else:
            return out
    elif isinstance(x, List[List[float]]) or isinstance(x, List[List[int]]):
        if nd >= 2:
            return out[0]
        else:
            return out
    else:
        return out[0]

def _pts(out: np.ndarray[float, 1], x, dim: int):
    # scipy _squeeze_output on per-point results, with a static result type:
    # scalars and 1-D quantiles are a single point (float); >= 2-D quantiles
    # give one value per row. (scipy also accepts a 1-D batch of points when
    # dim == 1; here pass it as an (n, 1) array.)
    if isinstance(x, np.ndarray):
        if static.len(x.shape) <= 1:
            if out.size != 1:
                raise ValueError("for dim == 1 pass several points as an (n, 1) array")
            return out[0]
        else:
            return out
    elif isinstance(x, List[List[float]]) or isinstance(x, List[List[int]]):
        return out
    else:
        if out.size != 1:
            raise ValueError("for dim == 1 pass several points as an (n, 1) array")
        return out[0]

# ------------------------------------------------------------------- _PSD

class _PSD:
    _M: np.ndarray[float, 2]
    U: np.ndarray[float, 2]
    V: np.ndarray[float, 2]
    eps: float
    rank: int
    log_pdet: float

    def __init__(self, M: np.ndarray[float, 2], allow_singular: bool = True):
        self._M = M
        s, u = LA.eigh(M, lower=True)
        # _eigvalsh_to_eps: 1e6 * eps(float64) * max|s|
        eps = 1e6 * 2.220446049250313e-16 * float(np.max(np.abs(s)))
        if float(np.min(s)) < -eps:
            raise ValueError("The input matrix must be symmetric positive semidefinite.")
        d = List[float]()
        for v in s:
            if v > eps:
                d.append(v)
        if len(d) < len(s) and not allow_singular:
            raise np.linalg.LinAlgError("When `allow_singular is False`, the input matrix must be "
                                        "symmetric positive definite.")
        n = s.size
        s_pinv = np.empty(n, dtype=float)
        for i in range(n):
            s_pinv[i] = 0.0 if abs(s[i]) <= eps else 1.0 / s[i]
        self.U = u * np.sqrt(s_pinv)
        self.eps = 1e3 * eps
        nv = n - len(d)
        V = np.empty((n, nv), dtype=float)
        j = 0
        for c in range(n):
            if s[c] <= eps:
                for r in range(n):
                    V[r, j] = u[r, c]
                j += 1
        self.V = V
        self.rank = len(d)
        ld = np.empty(len(d), dtype=float)
        for i in range(len(d)):
            ld[i] = K.log(d[i])
        self.log_pdet = float(np.sum(ld))

    def in_support(self, dev: np.ndarray[float, 1]) -> bool:
        if self.V.shape[1] == 0:
            return True
        r = dev @ self.V
        return float(np.linalg.norm(r)) < self.eps

def _as_cov(cov, dim: int) -> np.ndarray[float, 2]:
    c = np.asarray(cov).astype(float)
    if c.ndim == 0:
        return float(c.item()) * np.eye(dim)
    elif c.ndim == 1:
        if dim == 1:
            return c.reshape(1, 1)
        return np.diag(c)
    else:
        c2 = c.reshape(c.shape[0], c.shape[1])
        if c2.shape[0] != c2.shape[1]:
            raise ValueError("Array 'cov' must be square if it is two dimensional, "
                             f"but cov.shape = {c2.shape}.")
        if c2.shape[0] != dim:
            raise ValueError(f"Dimension mismatch: array 'cov' is of shape {c2.shape}, "
                             f"but 'mean' is a vector of length {dim}.")
        return c2

def _mvn_params(mean, cov):
    dim = 1
    if mean is None:
        c = np.asarray(cov).astype(float)
        dim = 1 if c.ndim < 2 else c.shape[0]
        m = np.zeros(dim)
    else:
        m = np.asarray(mean).astype(float).ravel()
        dim = m.size
    return dim, m, _as_cov(cov, dim)

def _quantiles(x, dim: int) -> np.ndarray[float, 2]:
    a = np.asarray(x).astype(float)
    if a.ndim == 0:
        return a.reshape(1, 1)
    elif a.ndim == 1:
        if dim == 1:
            return a.reshape(a.size, 1)
        return a.reshape(1, a.size)
    else:
        return a.reshape(-1, dim)

def _out_shape(x, dim: int):
    a = np.asarray(x)
    if a.ndim <= 1:
        return () if (dim > 1 or a.ndim == 0) else (a.size,)
    return a.shape[:-1]

# --------------------------------------------- multivariate normal (Genz)

def _bvn_cdf(dh: float, dk: float, r: float) -> float:
    # xsf bivariate_normal_cdf: P(X > dh, Y > dk) (Genz)
    if dh == INF or dk == INF:
        return 0.0
    if dh == -INF:
        return 1.0 if dk == -INF else K.ndtr(-dk)
    if dk == -INF:
        return K.ndtr(-dh)
    if r == 0.0:
        return K.ndtr(-dh) * K.ndtr(-dk)
    w6 = (0.1713244923791705, 0.3607615730481384, 0.4679139345726904)
    x6 = (0.9324695142031522, 0.6612093864662647, 0.2386191860831970)
    w12 = (0.04717533638651177, 0.1069393259953183, 0.1600783285433464,
           0.2031674267230659, 0.2334925365383547, 0.2491470458134029)
    x12 = (0.9815606342467191, 0.9041172563704750, 0.7699026741943050,
           0.5873179542866171, 0.3678314989981802, 0.1252334085114692)
    w20 = (0.01761400713915212, 0.04060142980038694, 0.06267204833410906, 0.08327674157670475,
           0.1019301198172404, 0.1181945319615184, 0.1316886384491766, 0.1420961093183821,
           0.1491729864726037, 0.1527533871307259)
    x20 = (0.9931285991850949, 0.9639719272779138, 0.9122344282513259, 0.8391169718222188,
           0.7463319064601508, 0.6360536807265150, 0.5108670019508271, 0.3737060887154196,
           0.2277858511416451, 0.07652652113349733)
    w = List[float]()
    x = List[float]()
    if abs(r) < 0.3:
        w = list(w6)
        x = list(x6)
    elif abs(r) < 0.75:
        w = list(w12)
        x = list(x12)
    else:
        w = list(w20)
        x = list(x20)
    n = len(w)
    tp = 2 * _PI
    h = dh
    k = dk
    hk = h * k
    bvn = 0.0
    if abs(r) < 0.925:
        hs = (h * h + k * k) / 2.0
        asr = asin(r) / 2.0
        for i in range(n):
            for xi in (1.0 - x[i], 1.0 + x[i]):
                sn = sin(asr * xi)
                bvn += K.exp((sn * hk - hs) / (1.0 - sn * sn)) * w[i]
        bvn = bvn * asr / tp + K.ndtr(-h) * K.ndtr(-k)
    else:
        if r < 0.0:
            k = -k
            hk = -hk
        if abs(r) < 1.0:
            as_ = 1.0 - r * r
            a = K.sqrt(as_)
            bs = (h - k) * (h - k)
            asr = -(bs / as_ + hk) / 2.0
            c = (4.0 - hk) / 8.0
            d = (12.0 - hk) / 80.0
            if asr > -100.0:
                bvn = a * K.exp(asr) * (1.0 - c * (bs - as_) * (1.0 - d * bs) / 3.0
                                        + c * d * as_ * as_)
            if hk > -100.0:
                b = K.sqrt(bs)
                sp = K.sqrt(tp) * K.ndtr(-b / a)
                bvn = bvn - K.exp(-hk / 2.0) * sp * b * (1.0 - c * bs * (1.0 - d * bs) / 3.0)
            a = a / 2.0
            tmp = 0.0
            for i in range(n):
                for xi in (1.0 - x[i], 1.0 + x[i]):
                    xs_i = (a * xi) * (a * xi)
                    asr_i = -(bs / xs_i + hk) / 2.0
                    if asr_i <= -100.0:
                        continue
                    sp = 1.0 + c * xs_i * (1.0 + 5.0 * d * xs_i)
                    rs = K.sqrt(1.0 - xs_i)
                    ep = K.exp(-(hk / 2.0) * xs_i / ((1.0 + rs) * (1.0 + rs))) / rs
                    tmp += w[i] * K.exp(asr_i) * (sp - ep)
            bvn = (a * tmp - bvn) / tp
        if r > 0.0:
            bvn = bvn + K.ndtr(-(h if h > k else k))
        elif h >= k:
            bvn = -bvn
        else:
            L = K.ndtr(k) - K.ndtr(h) if h < 0.0 else K.ndtr(-h) - K.ndtr(-k)
            bvn = L - bvn
    return max(0.0, min(1.0, bvn))

def _bvn(a: np.ndarray[float, 1], b: np.ndarray[float, 1], A: np.ndarray[float, 2]) -> float:
    s1 = K.sqrt(A[0, 0])
    s2 = K.sqrt(A[1, 1])
    r = A[0, 1] / (s1 * s2)
    xl = a[0] / s1
    xu = b[0] / s1
    yl = a[1] / s2
    yu = b[1] / s2
    p = (_bvn_cdf(xl, yl, r) - _bvn_cdf(xu, yl, r) - _bvn_cdf(xl, yu, r)
         + _bvn_cdf(xu, yu, r))
    return max(0., min(p, 1.))

def _primes_to(n: int) -> List[int]:
    # all primes < n (sieve)
    out = List[int]()
    if n <= 2:
        return out
    sieve = [True] * n
    for i in range(2, n):
        if sieve[i]:
            out.append(i)
            j = i * i
            while j < n:
                sieve[j] = False
                j += i
    return out

def _factorize_int(n: int) -> List[int]:
    factors = Set[int]()
    for p in _primes_to(int(K.sqrt(float(n))) + 1):
        while not (n % p):
            factors.add(p)
            n //= p
        if n == 1:
            break
    if n != 1:
        factors.add(n)
    return sorted(list(factors))

def _primitive_root(p: int) -> int:
    pm = p - 1
    factors = _factorize_int(pm)
    n = len(factors)
    r = 2
    k = 0
    while k < n:
        d = pm // factors[k]
        if pow(r, d, p) == 1:
            r += 1
            k = 0
        else:
            k += 1
    return r

def _cbc_lattice(n_dim: int, n_qmc_samples: int):
    primes = _primes_to(n_qmc_samples + 1)
    n_qmc_samples = primes[len(primes) - 1]
    bt = np.ones(n_dim)
    gm = np.ones(n_dim)
    for i in range(1, n_dim):
        gm[i] = 0.8 ** float(i - 1)
    q = np.ones(1)
    w = 0
    z = np.arange(1, n_dim + 1)
    m = (n_qmc_samples - 1) // 2
    g = _primitive_root(n_qmc_samples)
    perm = np.ones(m, dtype=int)
    for j in range(m - 1):
        perm[j + 1] = (g * perm[j]) % n_qmc_samples
    perm = np.minimum(n_qmc_samples - perm, perm)
    pn = perm / n_qmc_samples
    c = pn * pn - pn + 1.0 / 6
    fc = np.fft.fft(c)
    for s in range(1, n_dim):
        reordered = np.hstack((c[:w + 1][::-1], c[w + 1:m][::-1]))
        q = q * (bt[s - 1] + gm[s - 1] * reordered)
        w = int(np.argmin(np.fft.ifft(fc * np.fft.fft(q)).real))
        z[s] = perm[w]
    return z / n_qmc_samples, n_qmc_samples

def _permuted_cholesky(covar: np.ndarray[float, 2], low: np.ndarray[float, 1],
                       high: np.ndarray[float, 1], tol: float = 1e-10):
    cho = covar.copy()
    new_lo = low.copy()
    new_hi = high.copy()
    n = cho.shape[0]
    dc = np.sqrt(np.maximum(np.diag(cho), 0.0))
    for i in range(n):
        if dc[i] == 0.0:
            dc[i] = 1.0
    new_lo /= dc
    new_hi /= dc
    for i in range(n):
        for j in range(n):
            cho[i, j] = cho[i, j] / dc[j] / dc[i]
    y = np.zeros(n)
    sqtp = K.sqrt(2 * _PI)
    for k in range(n):
        epk = (k + 1) * tol
        im = k
        ck = 0.0
        dem = 1.0
        s = 0.0
        lo_m = 0.0
        hi_m = 0.0
        for i in range(k, n):
            if cho[i, i] > tol:
                ci = K.sqrt(cho[i, i])
                if i > 0:
                    s = 0.0
                    for t in range(k):
                        s += cho[i, t] * y[t]
                lo_i = (new_lo[i] - s) / ci
                hi_i = (new_hi[i] - s) / ci
                de = K.ndtr(hi_i) - K.ndtr(lo_i)
                if de <= dem:
                    ck = ci
                    dem = de
                    lo_m = lo_i
                    hi_m = hi_i
                    im = i
        if im > k:
            cho[im, im] = cho[k, k]
            for t in range(k):
                tmp = cho[im, t]
                cho[im, t] = cho[k, t]
                cho[k, t] = tmp
            for t in range(im + 1, n):
                tmp = cho[t, im]
                cho[t, im] = cho[t, k]
                cho[t, k] = tmp
            for t in range(k + 1, im):
                tmp = cho[t, k]
                cho[t, k] = cho[im, t]
                cho[im, t] = tmp
            tmp = new_lo[k]
            new_lo[k] = new_lo[im]
            new_lo[im] = tmp
            tmp = new_hi[k]
            new_hi[k] = new_hi[im]
            new_hi[im] = tmp
        if ck > epk:
            cho[k, k] = ck
            for t in range(k + 1, n):
                cho[k, t] = 0.0
            for i in range(k + 1, n):
                cho[i, k] /= ck
                for t in range(k + 1, i + 1):
                    cho[i, t] -= cho[i, k] * cho[t, k]
            if abs(dem) > tol:
                y[k] = ((K.exp(-lo_m * lo_m / 2) - K.exp(-hi_m * hi_m / 2)) / (sqtp * dem))
            else:
                y[k] = (lo_m + hi_m) / 2
                if lo_m < -10:
                    y[k] = hi_m
                elif hi_m > 10:
                    y[k] = lo_m
            for t in range(k + 1):
                cho[k, t] /= ck
            new_lo[k] /= ck
            new_hi[k] /= ck
        else:
            for t in range(k, n):
                cho[t, k] = 0.0
            y[k] = (new_lo[k] + new_hi[k]) / 2
    return cho, new_lo, new_hi

def _qmvn(m: int, covar: np.ndarray[float, 2], low: np.ndarray[float, 1],
          high: np.ndarray[float, 1], rng, n_batches: int = 10):
    cho, lo, hi = _permuted_cholesky(covar, low, high)
    n = cho.shape[0]
    q, n_qmc_samples = _cbc_lattice(n - 1, max(m // n_batches, 1))
    rndm = rng.random(size=(n_batches, n))
    # _qmvn_inner
    ct = cho[0, 0]
    ci = K.ndtr(lo[0] / ct)
    dci = K.ndtr(hi[0] / ct) - K.ndtr(lo[0] / ct)
    prob = 0.0
    error_var = 0.0
    y = np.zeros((n - 1, n_qmc_samples))
    for j in range(n_batches):
        c = np.full(n_qmc_samples, ci)
        dc = np.full(n_qmc_samples, dci)
        pv = dc.copy()
        for i in range(1, n):
            r = rndm[j, i]
            qq = q[i - 1]
            ct = cho[i, i]
            lo_i = lo[i]
            hi_i = hi[i]
            for k in range(n_qmc_samples):
                z_k = qq * (k + 1) + r
                z_k = fmod(z_k, 1.0)
                x_k = abs(2 * z_k - 1)
                res_k = K.ndtri(c[k] + x_k * dc[k])
                if res_k == INF:
                    res_k = 1e100
                elif res_k == -INF:
                    res_k = -1e100
                y[i - 1, k] = res_k
                s_k = 0.0
                for ii in range(i):
                    s_k += cho[i, ii] * y[ii, k]
                ck = K.ndtr((lo_i - s_k) / ct)
                dk = K.ndtr((hi_i - s_k) / ct)
                c[k] = ck
                dc[k] = dk - ck
                pv[k] = pv[k] * dc[k]
        dd = (float(np.mean(pv)) - prob) / (j + 1)
        prob += dd
        error_var = (j - 1) * error_var / (j + 1) + dd * dd
    return prob, 3 * K.sqrt(error_var), n_qmc_samples * n_batches

def _qauto(covar: np.ndarray[float, 2], low: np.ndarray[float, 1], high: np.ndarray[float, 1],
           rng, error: float, limit: int) -> float:
    n = covar.shape[0]
    if n == 1:
        s = K.sqrt(covar[0, 0])
        return K.ndtr(high[0] / s) - K.ndtr(low[0] / s)
    elif n == 2:
        return _bvn(low, high, covar)
    mi = min(limit, n * 1000)
    prob = 0.0
    est_error = 1.0
    n_samples = 0
    while est_error > error and n_samples < limit:
        mi = int(round(K.sqrt(2.0) * mi))
        pi, ei, ni = _qmvn(mi, covar, low, high, rng)
        n_samples += ni
        wt = 1.0 / (1 + (ei / est_error) ** 2)
        prob += wt * (pi - prob)
        est_error = K.sqrt(wt) * ei
    return prob

class multivariate_normal_frozen:
    dim: int
    mean: np.ndarray[float, 1]
    cov: np.ndarray[float, 2]
    psd: _PSD
    allow_singular: bool
    maxpts: int
    abseps: float
    releps: float

    def __init__(self, mean = None, cov = 1, allow_singular: bool = False, seed = None,
                 maxpts = None, abseps: float = 1e-5, releps: float = 1e-5):
        self.dim, self.mean, self.cov = _mvn_params(mean, cov)
        self.psd = _PSD(self.cov, allow_singular=allow_singular)
        self.allow_singular = allow_singular
        self.maxpts = 1000000 * self.dim if maxpts is None else int(maxpts)
        self.abseps = abseps
        self.releps = releps

    @property
    def cov_info(self): return self.psd

    def logpdf(self, x):
        X = _quantiles(x, self.dim)
        n = X.shape[0]
        out = np.empty(n, dtype=float)
        for i in range(n):
            dev = X[i] - self.mean
            wv = dev @ self.psd.U
            maha = float(np.sum(wv * wv))
            v = -0.5 * (self.psd.rank * _LOG_2PI + self.psd.log_pdet + maha)
            if self.psd.rank < self.dim and not self.psd.in_support(dev):
                v = -INF
            out[i] = v
        return _pts(out, x, self.dim)
    def pdf(self, x):
        return np.exp(self.logpdf(x))
    def logcdf(self, x, lower_limit = None, rng = None):
        return np.log(self.cdf(x, lower_limit=lower_limit, rng=rng))
    def cdf(self, x, lower_limit = None, rng = None):
        X = _quantiles(x, self.dim)
        rs = check_random_state(rng) if rng is not None else np.random.default_rng()
        n = X.shape[0]
        out = np.empty(n, dtype=float)
        lower = np.full(self.dim, -INF)
        if lower_limit is not None:
            lower = np.broadcast_to(np.asarray(lower_limit).astype(float), (self.dim,)).copy()
        for i in range(n):
            b = X[i] - self.mean
            a = lower - self.mean
            sign = 1.0
            for j in range(self.dim):
                if b[j] < a[j]:
                    t = a[j]
                    a[j] = b[j]
                    b[j] = t
                    sign = -sign
            out[i] = _qauto(self.cov, a, b, rs, self.abseps, self.maxpts) * sign
        return _pts(out, x, self.dim)
    def rvs(self, size: Literal[int] = 1, random_state = None):
        rs = check_random_state(random_state)
        return _sq1(rs.multivariate_normal(self.mean, self.cov, size), size)
    def entropy(self) -> float:
        return 0.5 * (self.psd.rank * (_LOG_2PI + 1) + self.psd.log_pdet)

class multivariate_normal_gen:
    def __call__(self, mean = None, cov = 1, allow_singular: bool = False, seed = None,
                 maxpts = None, abseps: float = 1e-5, releps: float = 1e-5):
        return multivariate_normal_frozen(mean, cov, allow_singular, seed, maxpts, abseps, releps)
    def logpdf(self, x, mean = None, cov = 1, allow_singular: bool = False):
        return multivariate_normal_frozen(mean, cov, allow_singular).logpdf(x)
    def pdf(self, x, mean = None, cov = 1, allow_singular: bool = False):
        return multivariate_normal_frozen(mean, cov, allow_singular).pdf(x)
    def cdf(self, x, mean = None, cov = 1, allow_singular: bool = False, maxpts = None,
            abseps: float = 1e-5, releps: float = 1e-5, lower_limit = None, rng = None):
        return multivariate_normal_frozen(mean, cov, allow_singular, None, maxpts, abseps,
                                          releps).cdf(x, lower_limit=lower_limit, rng=rng)
    def logcdf(self, x, mean = None, cov = 1, allow_singular: bool = False, maxpts = None,
               abseps: float = 1e-5, releps: float = 1e-5, lower_limit = None, rng = None):
        return np.log(self.cdf(x, mean, cov, allow_singular, maxpts, abseps, releps,
                               lower_limit, rng))
    def rvs(self, mean = None, cov = 1, size: Literal[int] = 1, random_state = None):
        return multivariate_normal_frozen(mean, cov, True).rvs(size, random_state)
    def entropy(self, mean = None, cov = 1) -> float:
        return multivariate_normal_frozen(mean, cov, True).entropy()
    def fit(self, x, fix_mean = None, fix_cov = None):
        X = np.asarray(x).astype(float)
        if X.ndim != 2:
            raise ValueError("`x` must be two-dimensional.")
        n, dim = X.shape
        mean = X.mean(axis=0) if fix_mean is None else np.asarray(fix_mean).astype(float).ravel()
        if fix_cov is None:
            centered = X - mean
            cov = (centered.T @ centered) / n
        else:
            cov = np.asarray(fix_cov).astype(float).reshape(dim, dim)
        return mean, cov

multivariate_normal = multivariate_normal_gen()

# -------------------------------------------------------- multivariate t

def _mvt_params(loc, shape, df):
    dim = 1
    if loc is None:
        s = np.asarray(shape).astype(float)
        dim = 1 if s.ndim < 2 else s.shape[0]
        l = np.zeros(dim)
    else:
        l = np.asarray(loc).astype(float).ravel()
        dim = l.size
    d = float(df)
    if d <= 0:
        raise ValueError("'df' must be greater than zero.")
    if d != d:
        raise ValueError("'df' is 'nan' but must be greater than zero or 'np.inf'.")
    return dim, l, _as_cov(shape, dim), d

class multivariate_t_frozen:
    dim: int
    loc: np.ndarray[float, 1]
    shape: np.ndarray[float, 2]
    df: float
    psd: _PSD

    def __init__(self, loc = None, shape = 1, df = 1, allow_singular: bool = False, seed = None):
        self.dim, self.loc, self.shape, self.df = _mvt_params(loc, shape, df)
        self.psd = _PSD(self.shape, allow_singular=allow_singular)

    def logpdf(self, x):
        X = _quantiles(x, self.dim)
        n = X.shape[0]
        out = np.empty(n, dtype=float)
        df = self.df
        dim = self.dim
        for i in range(n):
            dev = X[i] - self.loc
            wv = dev @ self.psd.U
            maha = float(np.sum(wv * wv))
            if df == INF:
                out[i] = -0.5 * (self.psd.rank * _LOG_2PI + self.psd.log_pdet + maha)
            else:
                t = 0.5 * (df + dim)
                A = K.gammaln(t)
                B = K.gammaln(0.5 * df)
                C = dim / 2. * K.log(df * _PI)
                D = 0.5 * self.psd.log_pdet
                E = -t * K.log(1 + (1. / df) * maha)
                out[i] = A - B - C - D + E
        return _pts(out, x, self.dim)
    def pdf(self, x):
        return np.exp(self.logpdf(x))
    def rvs(self, size: Literal[int] = 1, random_state = None):
        rs = check_random_state(random_state)
        if self.df == INF:
            xs = np.ones(size)
        else:
            xs = rs.chisquare(self.df, size=size) / self.df
        z = rs.multivariate_normal(np.zeros(self.dim), self.shape, size=size)
        samples = self.loc + z / np.sqrt(xs)[..., None]
        return _sq1(samples, size)
    def entropy(self) -> float:
        dim = float(self.dim)
        df = self.df
        if df == INF:
            return 0.5 * (self.psd.rank * (_LOG_2PI + 1) + self.psd.log_pdet)
        shape_term = 0.5 * self.psd.log_pdet
        threshold = dim * 100 * 4 / (K.log(dim) + 1)
        if df >= threshold:
            norm_ent = 0.5 * K.log(2 * _PI) + 0.5
            return (dim * norm_ent + dim / df
                    - dim * (dim - 2) * df ** -2.0 / 4
                    + dim ** 2 * (dim - 2) * df ** -3.0 / 6
                    + dim * (-3 * dim ** 3 + 8 * dim ** 2 - 8) * df ** -4.0 / 24
                    + dim ** 2 * (3 * dim ** 3 - 10 * dim ** 2 + 16) * df ** -5.0 / 30
                    + shape_term)
        halfsum = 0.5 * (dim + df)
        half_df = 0.5 * df
        return (-K.gammaln(halfsum) + K.gammaln(half_df)
                + 0.5 * dim * K.log(df * _PI) + halfsum
                * (K.digamma(halfsum) - K.digamma(half_df)) + shape_term)

class multivariate_t_gen:
    def __call__(self, loc = None, shape = 1, df = 1, allow_singular: bool = False, seed = None):
        return multivariate_t_frozen(loc, shape, df, allow_singular, seed)
    def logpdf(self, x, loc = None, shape = 1, df = 1):
        return multivariate_t_frozen(loc, shape, df).logpdf(x)
    def pdf(self, x, loc = None, shape = 1, df = 1, allow_singular: bool = False):
        return multivariate_t_frozen(loc, shape, df, allow_singular).pdf(x)
    def rvs(self, loc = None, shape = 1, df = 1, size: Literal[int] = 1, random_state = None):
        return multivariate_t_frozen(loc, shape, df, True).rvs(size, random_state)
    def entropy(self, loc = None, shape = 1, df = 1) -> float:
        return multivariate_t_frozen(loc, shape, df, True).entropy()

multivariate_t = multivariate_t_gen()

# -------------------------------------------------------------- dirichlet

def _dir_alpha(alpha) -> np.ndarray[float, 1]:
    a = np.asarray(alpha).astype(float)
    if float(np.min(a)) <= 0:
        raise ValueError("All parameters must be greater than 0")
    if a.ndim != 1:
        raise ValueError("Parameter vector 'a' must be one dimensional, "
                         f"but a.shape = {a.shape}.")
    return a.reshape(a.size)

def _lnB(alpha: np.ndarray[float, 1]) -> float:
    g = np.empty(alpha.size, dtype=float)
    for i in range(alpha.size):
        g[i] = K.gammaln(alpha[i])
    return float(np.sum(g)) - K.gammaln(float(np.sum(alpha)))

def _dir_x(alpha: np.ndarray[float, 1], x) -> np.ndarray[float, 2]:
    # columns are points; returns (K, npoints)
    X = np.asarray(x).astype(float)
    k = alpha.size
    if X.ndim == 1:
        X2 = X.reshape(X.size, 1)
    else:
        X2 = X.reshape(X.shape[0], -1)
    if X2.shape[0] + 1 != k and X2.shape[0] != k:
        raise ValueError("Vector 'x' must have either the same number "
                         "of entries as, or one entry fewer than, "
                         f"parameter vector 'a', but alpha.shape = {alpha.shape} "
                         f"and x.shape = {X.shape}.")
    if X2.shape[0] != k:
        last = 1 - np.sum(X2, axis=0)
        X2 = np.vstack((X2, last.reshape(1, last.size)))
    if float(np.min(X2)) < 0:
        raise ValueError("Each entry in 'x' must be greater than or equal to zero.")
    if float(np.max(X2)) > 1:
        raise ValueError("Each entry in 'x' must be smaller or equal one.")
    for c in range(X2.shape[1]):
        for r in range(k):
            if X2[r, c] == 0 and alpha[r] < 1:
                raise ValueError("Each entry in 'x' must be greater than zero if its "
                                 "alpha is less than one.")
    if (np.abs(np.sum(X2, axis=0) - 1.0) > 10e-10).any():
        raise ValueError("The input vector 'x' must lie within the normal simplex.")
    return X2

class dirichlet_frozen:
    alpha: np.ndarray[float, 1]
    def __init__(self, alpha, seed = None):
        self.alpha = _dir_alpha(alpha)
    def logpdf(self, x):
        X = _dir_x(self.alpha, x)
        lnB = _lnB(self.alpha)
        n = X.shape[1]
        out = np.empty(n, dtype=float)
        for c in range(n):
            v = List[float](self.alpha.size)
            for r in range(self.alpha.size):
                v.append(K.xlogy(self.alpha[r] - 1, X[r, c]))
            out[c] = -lnB + _npsum(v)
        return _ndsel(out, x, 1)
    def pdf(self, x):
        return np.exp(self.logpdf(x))
    def mean(self):
        return self.alpha / float(np.sum(self.alpha))
    def var(self):
        a0 = float(np.sum(self.alpha))
        return (self.alpha * (a0 - self.alpha)) / ((a0 * a0) * (a0 + 1))
    def cov(self):
        a0 = float(np.sum(self.alpha))
        a = self.alpha / a0
        return (np.diag(a) - np.outer(a, a)) / (a0 + 1)
    def entropy(self) -> float:
        a0 = float(np.sum(self.alpha))
        k = self.alpha.size
        s = np.empty(k, dtype=float)
        for i in range(k):
            s[i] = (self.alpha[i] - 1) * K.digamma(self.alpha[i])
        return _lnB(self.alpha) + (a0 - k) * K.digamma(a0) - float(np.sum(s))
    def rvs(self, size = 1, random_state = None):
        rs = check_random_state(random_state)
        return rs.dirichlet(self.alpha, size=size)

class dirichlet_gen:
    def __call__(self, alpha, seed = None):
        return dirichlet_frozen(alpha, seed)
    def logpdf(self, x, alpha): return dirichlet_frozen(alpha).logpdf(x)
    def pdf(self, x, alpha): return dirichlet_frozen(alpha).pdf(x)
    def mean(self, alpha): return dirichlet_frozen(alpha).mean()
    def var(self, alpha): return dirichlet_frozen(alpha).var()
    def cov(self, alpha): return dirichlet_frozen(alpha).cov()
    def entropy(self, alpha) -> float: return dirichlet_frozen(alpha).entropy()
    def rvs(self, alpha, size = 1, random_state = None):
        return dirichlet_frozen(alpha).rvs(size, random_state)

dirichlet = dirichlet_gen()

# ------------------------------------------------------------ multinomial

class multinomial_frozen:
    n: int
    p: np.ndarray[float, 1]
    bad: bool
    def __init__(self, n, p, seed = None):
        self.p = np.asarray(p).astype(float).ravel()
        self.n = int(n)
        eps = 2.220446049250313e-16 * 10
        bad = abs(1 - float(np.sum(self.p))) > eps
        bad = bad or bool((self.p < 0).any()) or bool((self.p > 1).any())
        self.bad = bad or self.n < 0
    def logpmf(self, x):
        X = np.asarray(x).astype(float)
        k = self.p.size
        X2 = X.reshape(-1, k)
        m = X2.shape[0]
        out = np.empty(m, dtype=float)
        for i in range(m):
            v = List[float](k)
            tot = 0.0
            badx = False
            for j in range(k):
                xv = X2[i, j]
                if xv != K.floor(xv) or xv < 0:
                    badx = True
                tot += K.floor(xv)
                v.append(K.xlogy(K.floor(xv), self.p[j]) - K.gammaln(K.floor(xv) + 1))
            r = K.gammaln(self.n + 1.0) + _npsum(v)
            if badx or tot != self.n:
                r = -INF
            if self.bad:
                r = NAN
            out[i] = r
        if X.ndim <= 1:
            return out[0]
        return out.reshape(X.shape[:-1])
    def pmf(self, x):
        return np.exp(self.logpmf(x))
    def mean(self):
        return self.n * self.p
    def cov(self):
        k = self.p.size
        out = np.empty((k, k), dtype=float)
        for i in range(k):
            for j in range(k):
                out[i, j] = self.n * (-self.p[i] * self.p[j])
            out[i, i] += self.n * self.p[i]
        return out
    def entropy(self) -> float:
        k = self.p.size
        e = List[float](k)
        for j in range(k):
            e.append(K.entr(self.p[j]))
        term1 = self.n * _npsum(e) - K.gammaln(self.n + 1.0)
        # sum over x=1..n and over the k components of binom.pmf(x,n,p)*gammaln(x+1)
        vals = List[float]()
        for xi in range(1, self.n + 1):
            for j in range(k):
                pj = self.p[j]
                lp = (K.gammaln(self.n + 1.0) - K.gammaln(xi + 1.0) - K.gammaln(self.n - xi + 1.0)
                      + K.xlogy(float(xi), pj) + K.xlog1py(float(self.n - xi), -pj))
                vals.append(K.exp(lp) * K.gammaln(xi + 1.0))
        return term1 + float(np.sum(np.array(vals)))
    def rvs(self, size = None, random_state = None):
        if self.bad:
            raise ValueError("`multinomial.rvs` requires `n > 0`, `(p > 0).all()`, and"
                             "`p.sum() == 1`.")
        rs = check_random_state(random_state)
        return rs.multinomial(self.n, self.p, size)

class multinomial_gen:
    def __call__(self, n, p, seed = None):
        return multinomial_frozen(n, p, seed)
    def logpmf(self, x, n, p): return multinomial_frozen(n, p).logpmf(x)
    def pmf(self, x, n, p): return multinomial_frozen(n, p).pmf(x)
    def mean(self, n, p): return multinomial_frozen(n, p).mean()
    def cov(self, n, p): return multinomial_frozen(n, p).cov()
    def entropy(self, n, p) -> float: return multinomial_frozen(n, p).entropy()
    def rvs(self, n, p, size = None, random_state = None):
        return multinomial_frozen(n, p).rvs(size, random_state)

multinomial = multinomial_gen()

# ------------------------------------------------------ wishart / invwishart

def _wishart_params(df, scale):
    s = np.asarray(1.0 if scale is None else scale).astype(float)
    if s.ndim == 0:
        S = s.reshape(1, 1)
    elif s.ndim == 1:
        S = np.diag(s)
    else:
        S = s.reshape(s.shape[0], s.shape[1])
        if S.shape[0] != S.shape[1]:
            raise ValueError("Array 'scale' must be square if it is two dimensional,"
                             f" but scale.scale = {S.shape}.")
    dim = S.shape[0]
    d = float(dim) if df is None else float(df)
    if d <= dim - 1:
        raise ValueError("Degrees of freedom must be greater than the "
                         "dimension of scale matrix minus 1.")
    return dim, d, S

def _wishart_x(x, dim: int):
    # returns list of (dim, dim) matrices and whether the input was a single matrix
    a = np.asarray(x).astype(float)
    out = List[np.ndarray[float, 2]]()
    if a.ndim == 0:
        out.append(float(a.item()) * np.eye(dim))
    elif a.ndim == 1:
        if dim == 1:
            for v in a:
                out.append(np.full((1, 1), v))
        else:
            out.append(np.diag(a))
    elif a.ndim == 2:
        out.append(a.reshape(a.shape[0], a.shape[1]))
    else:
        a3 = a.reshape(a.shape[0], a.shape[1], -1)
        for i in range(a3.shape[2]):
            out.append(a3[:, :, i].copy())
    for m in out:
        if m.shape[0] != dim or m.shape[1] != dim:
            raise ValueError('Quantiles have incompatible dimensions')
    return out

def _chol_logdet(m: np.ndarray[float, 2]):
    c = LA.cholesky(m, lower=True)
    ld = 0.0
    lv = np.empty(m.shape[0], dtype=float)
    for i in range(m.shape[0]):
        lv[i] = K.log(c[i, i])
    return c, 2 * float(np.sum(lv))

def _wishart_size(size):
    s = np.asarray(size).astype(int)
    shape = (int(s.item()),) if s.ndim == 0 else tuple(int(v) for v in s)
    n = 1
    for v in shape:
        n *= v
    return n, shape

def _wishart_standard_rvs(n: int, shape, dim: int, df: float, rs) -> np.ndarray[float, 3]:
    n_tril = dim * (dim - 1) // 2
    cov = rs.normal(size=n * n_tril)
    # variances: chisquare(df - i, size=n) ** 0.5 for i in range(dim), shape (dim, n) -> transposed
    var = np.empty((dim, n), dtype=float)
    for i in range(dim):
        ch = rs.chisquare(df - (i + 1) + 1, size=n)
        for j in range(n):
            var[i, j] = ch[j] ** 0.5
    # A[k] lower triangular, k in C order over shape
    A = np.zeros((n, dim, dim))
    # variances reshape((dim,) + shape[::-1]).T -> element [k][i] = var[i, flat_rev(k)]
    rshape = shape[::-1]
    for k in range(n):
        # index of k in C order of shape, mapped into C order of reversed shape
        idx = List[int]()
        r = k
        for d in range(len(shape) - 1, -1, -1):
            idx.append(r % shape[d])
            r //= shape[d]
        # idx is reversed multi-index, i.e. index into rshape in C order
        kr = 0
        for d in range(len(rshape)):
            kr = kr * rshape[d] + idx[d]
        t = 0
        for a in range(1, dim):
            for b in range(a):
                A[k, a, b] = cov[k * n_tril + t]
                t += 1
        for a in range(dim):
            A[k, a, a] = var[a, kr]
    return A

class wishart_frozen:
    dim: int
    df: float
    scale: np.ndarray[float, 2]
    C: np.ndarray[float, 2]
    log_det_scale: float

    def __init__(self, df = None, scale = None, seed = None):
        self.dim, self.df, self.scale = _wishart_params(df, scale)
        self.C, self.log_det_scale = _chol_logdet(self.scale)

    def logpdf(self, x):
        xs = _wishart_x(x, self.dim)
        dim = self.dim
        df = self.df
        out = np.empty(len(xs), dtype=float)
        for i in range(len(xs)):
            _, log_det_x = _chol_logdet(xs[i])
            sx = LA.cho_solve((self.C, True), xs[i])
            tr = float(np.trace(sx))
            out[i] = ((0.5 * (df - dim - 1) * log_det_x - 0.5 * tr) -
                      (0.5 * df * dim * _LOG_2 + 0.5 * df * self.log_det_scale +
                       float(multigammaln(0.5 * df, dim))))
        return _ndsel(out, x, 2)
    def pdf(self, x):
        return np.exp(self.logpdf(x))
    def mean(self):
        return self.df * self.scale
    def mode(self):
        if self.df >= self.dim + 1:
            return (self.df - self.dim - 1) * self.scale
        raise ValueError("mode undefined")
    def var(self):
        var = self.scale ** 2
        diag = np.diag(self.scale)
        var = var + np.outer(diag, diag)
        return var * self.df
    def rvs(self, size: Literal[int] = 1, random_state = None):
        n, shape = _wishart_size(size)
        rs = check_random_state(random_state)
        A = _wishart_standard_rvs(n, shape, self.dim, self.df, rs)
        for k in range(n):
            CA = self.C @ A[k]
            A[k] = CA @ CA.T
        return _sq1(A, size)
    def entropy(self) -> float:
        dim = self.dim
        df = self.df
        ps = np.empty(dim, dtype=float)
        for i in range(dim):
            ps[i] = K.digamma(0.5 * (df + 1 - (i + 1)))
        return (0.5 * (dim + 1) * self.log_det_scale + 0.5 * dim * (dim + 1) * _LOG_2
                + float(multigammaln(0.5 * df, dim))
                - 0.5 * (df - dim - 1) * float(np.sum(ps)) + 0.5 * df * dim)

class wishart_gen:
    def __call__(self, df = None, scale = None, seed = None):
        return wishart_frozen(df, scale, seed)
    def logpdf(self, x, df, scale): return wishart_frozen(df, scale).logpdf(x)
    def pdf(self, x, df, scale): return wishart_frozen(df, scale).pdf(x)
    def mean(self, df, scale): return wishart_frozen(df, scale).mean()
    def mode(self, df, scale): return wishart_frozen(df, scale).mode()
    def var(self, df, scale): return wishart_frozen(df, scale).var()
    def entropy(self, df, scale) -> float: return wishart_frozen(df, scale).entropy()
    def rvs(self, df, scale, size: Literal[int] = 1, random_state = None):
        return wishart_frozen(df, scale).rvs(size, random_state)

wishart = wishart_gen()

class invwishart_frozen:
    dim: int
    df: float
    scale: np.ndarray[float, 2]
    C: np.ndarray[float, 2]
    log_det_scale: float

    def __init__(self, df = None, scale = None, seed = None):
        self.dim, self.df, self.scale = _wishart_params(df, scale)
        self.C, self.log_det_scale = _chol_logdet(self.scale)

    def logpdf(self, x):
        xs = _wishart_x(x, self.dim)
        dim = self.dim
        df = self.df
        out = np.empty(len(xs), dtype=float)
        for i in range(len(xs)):
            log_det_x = 0.0
            tr = 0.0
            if dim > 1:
                Cx, log_det_x = _chol_logdet(xs[i])
                A = LA.solve_triangular(Cx, self.C, lower=True)
                tr = float(np.linalg.norm(A)) ** 2
            else:
                log_det_x = K.log(xs[i][0, 0])
                tr = self.C[0, 0] ** 2 / xs[i][0, 0]
            out[i] = ((0.5 * df * self.log_det_scale - 0.5 * tr) -
                      (0.5 * df * dim * _LOG_2 + 0.5 * (df + dim + 1) * log_det_x) -
                      float(multigammaln(0.5 * df, dim)))
        return _ndsel(out, x, 2)
    def pdf(self, x):
        return np.exp(self.logpdf(x))
    def mean(self):
        if self.df > self.dim + 1:
            return self.scale / (self.df - self.dim - 1)
        raise ValueError("mean undefined")
    def mode(self):
        return self.scale / (self.df + self.dim + 1)
    def var(self):
        df = self.df
        dim = self.dim
        if df > dim + 3:
            var = (df - dim + 1) * self.scale ** 2
            diag = np.diag(self.scale)
            var = var + (df - dim - 1) * np.outer(diag, diag)
            var = var / ((df - dim) * (df - dim - 1) ** 2 * (df - dim - 3))
            return var
        raise ValueError("var undefined")
    def rvs(self, size: Literal[int] = 1, random_state = None):
        n, shape = _wishart_size(size)
        rs = check_random_state(random_state)
        dim = self.dim
        n_tril = dim * (dim - 1) // 2
        A = np.zeros((n, dim, dim))
        nrm = rs.normal(size=shape + (n_tril,)).reshape(n, n_tril)
        dfs = np.empty(dim, dtype=float)
        for i in range(dim):
            dfs[i] = (self.df - dim + 1) + i
        ch = rs.chisquare(df=dfs, size=shape + (dim,)).reshape(n, dim)
        for k in range(n):
            t = 0
            for a in range(1, dim):
                for b in range(a):
                    A[k, a, b] = nrm[k, t]
                    t += 1
            for a in range(dim):
                A[k, a, a] = ch[k, a] ** 0.5
        for k in range(n):
            if dim > 1:
                # CA = C @ inv(A) (trsm side=1, lower); out = CA^T... trmm(CA, CA, side=1, lower, trans_a)
                CA = LA.solve_triangular(A[k].T, self.C.T, lower=False).T
                A[k] = CA @ CA.T
            else:
                A[k, 0, 0] = (self.C[0, 0] / A[k, 0, 0]) ** 2
        return _sq1(A, size)
    def entropy(self) -> float:
        dim = self.dim
        df = self.df
        ps = np.empty(dim, dtype=float)
        for i in range(1, dim + 1):
            ps[i - 1] = K.digamma(0.5 * (df - dim + i))
        return (float(multigammaln(0.5 * df, dim)) + 0.5 * dim * df
                + 0.5 * (dim + 1) * (self.log_det_scale - _LOG_2)
                - 0.5 * (df + dim + 1) * float(np.sum(ps)))

class invwishart_gen:
    def __call__(self, df = None, scale = None, seed = None):
        return invwishart_frozen(df, scale, seed)
    def logpdf(self, x, df, scale): return invwishart_frozen(df, scale).logpdf(x)
    def pdf(self, x, df, scale): return invwishart_frozen(df, scale).pdf(x)
    def mean(self, df, scale): return invwishart_frozen(df, scale).mean()
    def mode(self, df, scale): return invwishart_frozen(df, scale).mode()
    def var(self, df, scale): return invwishart_frozen(df, scale).var()
    def entropy(self, df, scale) -> float: return invwishart_frozen(df, scale).entropy()
    def rvs(self, df, scale, size: Literal[int] = 1, random_state = None):
        return invwishart_frozen(df, scale).rvs(size, random_state)

invwishart = invwishart_gen()

# ---------------------------------------------------------- matrix normal

def _mn_cov(c, n: int) -> np.ndarray[float, 2]:
    a = np.asarray(c).astype(float)
    if a.ndim == 0:
        return float(a.item()) * np.eye(n)
    elif a.ndim == 1:
        return np.diag(a)
    else:
        m = a.reshape(a.shape[0], a.shape[1])
        if m.shape[0] != m.shape[1]:
            raise ValueError("Array `rowcov` must be square.")
        return m

class matrix_normal_frozen:
    mean: np.ndarray[float, 2]
    rowcov: np.ndarray[float, 2]
    colcov: np.ndarray[float, 2]
    rowpsd: _PSD
    colpsd: _PSD

    def __init__(self, mean = None, rowcov = 1, colcov = 1, seed = None):
        if mean is None:
            r = np.asarray(rowcov)
            c = np.asarray(colcov)
            nr = 1 if r.ndim == 0 else r.shape[0]
            nc = 1 if c.ndim == 0 else c.shape[0]
            self.mean = np.zeros((nr, nc))
        else:
            m = np.asarray(mean).astype(float)
            if m.ndim != 2:
                raise ValueError("Array `mean` must be two dimensional.")
            self.mean = m.reshape(m.shape[0], m.shape[1])
        self.rowcov = _mn_cov(rowcov, self.mean.shape[0])
        self.colcov = _mn_cov(colcov, self.mean.shape[1])
        if self.rowcov.shape[0] != self.mean.shape[0]:
            raise ValueError("Arrays `mean` and `rowcov` must have the same number of rows.")
        if self.colcov.shape[0] != self.mean.shape[1]:
            raise ValueError("Arrays `mean` and `colcov` must have the same number of columns.")
        self.rowpsd = _PSD(self.rowcov, allow_singular=False)
        self.colpsd = _PSD(self.colcov, allow_singular=False)

    def logpdf(self, X):
        a = np.asarray(X).astype(float)
        nr = self.mean.shape[0]
        nc = self.mean.shape[1]
        a3 = a.reshape(-1, nr, nc)
        out = np.empty(a3.shape[0], dtype=float)
        for i in range(a3.shape[0]):
            dev = a3[i] - self.mean
            sd = self.rowpsd.U.T @ dev @ self.colpsd.U
            maha = float(np.sum(sd * sd))
            out[i] = -0.5 * (nr * nc * _LOG_2PI + nc * self.rowpsd.log_pdet
                             + nr * self.colpsd.log_pdet + maha)
        return _ndsel(out, X, 2)
    def pdf(self, X):
        return np.exp(self.logpdf(X))
    def rvs(self, size: Literal[int] = 1, random_state = None):
        nr = self.mean.shape[0]
        nc = self.mean.shape[1]
        rowchol = LA.cholesky(self.rowcov, lower=True)
        colchol = LA.cholesky(self.colcov, lower=True)
        rs = check_random_state(random_state)
        std = rs.standard_normal(size=(nc, size, nr))
        out = np.empty((size, nr, nc), dtype=float)
        for i in range(size):
            Z = std[:, i, :].T.copy()
            out[i] = self.mean + rowchol @ Z @ colchol.T
        if size == 1:
            return out.reshape(nr, nc)
        return out
    def entropy(self) -> float:
        n = float(self.mean.shape[0])
        p = float(self.mean.shape[1])
        return (0.5 * n * p * (1 + _LOG_2PI) + 0.5 * p * self.rowpsd.log_pdet +
                0.5 * n * self.colpsd.log_pdet)

class matrix_normal_gen:
    def __call__(self, mean = None, rowcov = 1, colcov = 1, seed = None):
        return matrix_normal_frozen(mean, rowcov, colcov, seed)
    def logpdf(self, X, mean = None, rowcov = 1, colcov = 1):
        return matrix_normal_frozen(mean, rowcov, colcov).logpdf(X)
    def pdf(self, X, mean = None, rowcov = 1, colcov = 1):
        return matrix_normal_frozen(mean, rowcov, colcov).pdf(X)
    def rvs(self, mean = None, rowcov = 1, colcov = 1, size: Literal[int] = 1, random_state = None):
        return matrix_normal_frozen(mean, rowcov, colcov).rvs(size, random_state)
    def entropy(self, rowcov = 1, colcov = 1) -> float:
        r = np.asarray(rowcov)
        c = np.asarray(colcov)
        return matrix_normal_frozen(np.zeros((r.shape[0], c.shape[0])), rowcov, colcov).entropy()

matrix_normal = matrix_normal_gen()

# ----------------------------------------------------- orthogonal groups

def _ortho_rvs(dim: int, size: int, rs) -> np.ndarray[float, 3]:
    n = size if size > 1 else 1
    z = rs.normal(size=n * dim * dim).reshape(n, dim, dim)
    out = np.empty((n, dim, dim), dtype=float)
    for k in range(n):
        q, r = np.linalg.qr(z[k])
        for j in range(dim):
            d = r[j, j]
            sg = d / abs(d)
            for i in range(dim):
                q[i, j] *= sg
        out[k] = q
    return out

class ortho_group_gen:
    def rvs(self, dim: int, size: Literal[int] = 1, random_state = None):
        rs = check_random_state(random_state)
        q = _ortho_rvs(dim, size, rs)
        if size > 1:
            return q
        else:
            return q[0].copy()

class special_ortho_group_gen:
    def rvs(self, dim: int, size: Literal[int] = 1, random_state = None):
        rs = check_random_state(random_state)
        q = _ortho_rvs(dim, size, rs)
        for k in range(q.shape[0]):
            det = float(np.linalg.det(q[k]))
            if dim:
                for j in range(dim):
                    q[k, 0, j] /= det
        if size > 1:
            return q
        else:
            return q[0].copy()

ortho_group = ortho_group_gen()
special_ortho_group = special_ortho_group_gen()

class random_correlation_gen:
    def _givens_to_1(self, aii: float, ajj: float, aij: float):
        aiid = aii - 1.
        ajjd = ajj - 1.
        if ajjd == 0:
            return 0., 1.
        dd = K.sqrt(max(aij ** 2 - aiid * ajjd, 0.0))
        t = (aij + K._copysign(dd, aij)) / ajjd
        c = 1. / K.sqrt(1. + t * t)
        s = 1.0 if c == 0 else c * t
        return c, s
    def _to_corr(self, m: np.ndarray[float, 2]) -> np.ndarray[float, 2]:
        d = m.shape[0]
        for i in range(d - 1):
            if m[i, i] == 1:
                continue
            j = i + 1
            if m[i, i] > 1:
                for jj in range(i + 1, d):
                    j = jj
                    if m[jj, jj] < 1:
                        break
            else:
                for jj in range(i + 1, d):
                    j = jj
                    if m[jj, jj] > 1:
                        break
            c, s = self._givens_to_1(m[i, i], m[j, j], m[i, j])
            # drot on rows i, j: x' = c x + s' y, y' = c y - s' x with s' = -s
            for k in range(d):
                x = m[i, k]
                y = m[j, k]
                m[i, k] = c * x - s * y
                m[j, k] = c * y + s * x
            for k in range(d):
                x = m[k, i]
                y = m[k, j]
                m[k, i] = c * x - s * y
                m[k, j] = c * y + s * x
        return m
    def rvs(self, eigs, random_state = None, tol: float = 1e-13, diag_tol: float = 1e-7):
        e = np.asarray(eigs).astype(float).ravel()
        dim = e.size
        if dim <= 1:
            raise ValueError("Array 'eigs' must be a vector of length greater than 1.")
        if abs(float(np.sum(e)) - dim) > tol:
            raise ValueError("Sum of eigenvalues must equal dimensionality.")
        for x in e:
            if x < -tol:
                raise ValueError("All eigenvalues must be non-negative.")
        rs = check_random_state(random_state)
        m = ortho_group.rvs(dim, random_state=rs)
        m = (m @ np.diag(e)) @ m.T
        m = self._to_corr(m)
        if float(np.max(np.abs(np.diag(m) - 1))) > diag_tol:
            raise RuntimeError("Failed to generate a valid correlation matrix")
        return m

random_correlation = random_correlation_gen()

# ------------------------------------------------------ uniform_direction

def _sample_uniform_direction(dim: int, n: int, rs) -> np.ndarray[float, 2]:
    s = rs.standard_normal((n, dim))
    for i in range(n):
        nrm = float(np.linalg.norm(s[i]))
        for j in range(dim):
            s[i, j] /= nrm
    return s

class uniform_direction_frozen:
    dim: int
    def __init__(self, dim: int, seed = None):
        if dim < 1:
            raise ValueError("Dimension of vector must be specified, and must be an "
                             "integer greater than 0.")
        self.dim = dim
    def rvs(self, size = None, random_state = None):
        rs = check_random_state(random_state)
        if size is None:
            return _sample_uniform_direction(self.dim, 1, rs)[0].copy()
        else:
            n = int(size)
            return _sample_uniform_direction(self.dim, n, rs)

class uniform_direction_gen:
    def __call__(self, dim: int, seed = None):
        return uniform_direction_frozen(dim, seed)
    def rvs(self, dim: int, size = None, random_state = None):
        return uniform_direction_frozen(dim).rvs(size, random_state)

uniform_direction = uniform_direction_gen()

# -------------------------------------------------- dirichlet_multinomial

class dirichlet_multinomial_frozen:
    alpha: np.ndarray[float, 1]
    n: int
    def __init__(self, alpha, n, seed = None):
        self.alpha = np.asarray(alpha).astype(float).ravel()
        if (self.alpha <= 0).any():
            raise ValueError("`alpha` must contain only positive values.")
        nf = float(n)
        if nf < 0 or nf != K.floor(nf):
            raise ValueError("`n` must be a non-negative integer.")
        self.n = int(nf)
    def logpmf(self, x) -> float:
        a = self.alpha
        xa = np.asarray(x).astype(float).ravel()
        if (xa < 0).any() or (xa != np.floor(xa)).any():
            raise ValueError("`x` must contain only non-negative integers.")
        Sa = float(np.sum(a))
        n = float(self.n)
        out = K.gammaln(Sa) + K.gammaln(n + 1) - K.gammaln(n + Sa)
        v = np.empty(a.size, dtype=float)
        for i in range(a.size):
            v[i] = K.gammaln(xa[i] + a[i]) - (K.gammaln(a[i]) + K.gammaln(xa[i] + 1))
        out += float(np.sum(v))
        if float(np.sum(xa)) != n:
            return -INF
        return out
    def pmf(self, x) -> float:
        return K.exp(self.logpmf(x))
    def mean(self):
        Sa = float(np.sum(self.alpha))
        return self.n * self.alpha / Sa
    def var(self):
        a = self.alpha
        Sa = float(np.sum(a))
        n = float(self.n)
        return n * a / Sa * (1 - a / Sa) * (n + Sa) / (1 + Sa)
    def cov(self):
        a = self.alpha
        Sa = float(np.sum(a))
        n = float(self.n)
        var = self.var()
        k = a.size
        out = np.empty((k, k), dtype=float)
        for i in range(k):
            for j in range(k):
                out[i, j] = -n * (a[i] * a[j]) / Sa ** 2 * (n + Sa) / (1 + Sa)
            out[i, i] = var[i]
        return out

class dirichlet_multinomial_gen:
    def __call__(self, alpha, n, seed = None):
        return dirichlet_multinomial_frozen(alpha, n, seed)
    def logpmf(self, x, alpha, n) -> float: return dirichlet_multinomial_frozen(alpha, n).logpmf(x)
    def pmf(self, x, alpha, n) -> float: return dirichlet_multinomial_frozen(alpha, n).pmf(x)
    def mean(self, alpha, n): return dirichlet_multinomial_frozen(alpha, n).mean()
    def var(self, alpha, n): return dirichlet_multinomial_frozen(alpha, n).var()
    def cov(self, alpha, n): return dirichlet_multinomial_frozen(alpha, n).cov()

dirichlet_multinomial = dirichlet_multinomial_gen()

# -------------------------------------------------------- vonmises_fisher

import scipy.special._bessel as _B

def _vmf_log_norm(dim: int, kappa: float) -> float:
    halfdim = 0.5 * dim
    return (0.5 * (dim - 2) * K.log(kappa) - halfdim * _LOG_2PI
            - K.log(_B.ive(halfdim - 1, kappa)) - kappa)

class vonmises_fisher_frozen:
    dim: int
    mu: np.ndarray[float, 1]
    kappa: float
    def __init__(self, mu, kappa = 1.0, seed = None):
        m = np.asarray(mu).astype(float)
        if m.ndim > 1:
            raise ValueError("'mu' must have one-dimensional shape.")
        self.mu = m.ravel()
        if not np.allclose(np.linalg.norm(self.mu), 1.):
            raise ValueError("'mu' must be a unit vector of norm 1.")
        if not self.mu.size > 1:
            raise ValueError("'mu' must have at least two entries.")
        k = float(kappa)
        if k < 0:
            raise ValueError("'kappa' must be a positive scalar.")
        if k == 0.:
            raise ValueError("For 'kappa=0' the von Mises-Fisher distribution "
                             "becomes the uniform distribution on the sphere "
                             "surface. Consider using "
                             "'scipy.stats.uniform_direction' instead.")
        self.kappa = k
        self.dim = self.mu.size
    def logpdf(self, x):
        X = np.asarray(x).astype(float)
        X2 = X.reshape(-1, self.dim)
        out = np.empty(X2.shape[0], dtype=float)
        lnf = _vmf_log_norm(self.dim, self.kappa)
        for i in range(X2.shape[0]):
            out[i] = lnf + self.kappa * float(np.dot(self.mu, X2[i]))
        if X.ndim <= 1:
            return out[0]
        return out.reshape(X.shape[:-1])
    def pdf(self, x):
        return np.exp(self.logpdf(x))
    def entropy(self) -> float:
        halfdim = 0.5 * self.dim
        return (-_vmf_log_norm(self.dim, self.kappa)
                - self.kappa * _B.ive(halfdim, self.kappa) / _B.ive(halfdim - 1, self.kappa))
    def _rotate(self, samples: np.ndarray[float, 2]) -> np.ndarray[float, 2]:
        dim = self.dim
        embedded = np.zeros((dim, dim))
        for j in range(dim):
            embedded[0, j] = self.mu[j]
        rot, _ = np.linalg.qr(embedded.T.copy())
        first = rot[:, 0].copy()
        rotsign = 1.0 if np.allclose(first, self.mu) else -1.0
        return (samples @ rot.T) * rotsign
    def rvs(self, size: Literal[int] = 1, random_state = None):
        rs = check_random_state(random_state)
        dim = self.dim
        kappa = self.kappa
        n = size
        if dim == 2:
            mean_angle = atan2(self.mu[1], self.mu[0])
            ang = rs.vonmises(mean_angle, kappa, size=n)
            out = np.empty((n, 2), dtype=float)
            for i in range(n):
                out[i, 0] = K.cos(ang[i])
                out[i, 1] = K.sin(ang[i])
            if size == 1:
                return out[0].copy()
            else:
                return out
        samples = np.empty((n, dim), dtype=float)
        if dim == 3:
            xs = rs.random(n)
            for i in range(n):
                xs[i] = 1. + K.log(xs[i] + (1. - xs[i]) * K.exp(-2 * kappa)) / kappa
            uc = _sample_uniform_direction(2, n, rs)
            for i in range(n):
                t = K.sqrt(1. - xs[i] * xs[i])
                samples[i, 0] = xs[i]
                samples[i, 1] = t * uc[i, 0]
                samples[i, 2] = t * uc[i, 1]
        else:
            dm1 = float(dim - 1)
            sq = K.sqrt(4 * kappa ** 2. + dm1 ** 2)
            env = (-2 * kappa + sq) / dm1
            if env == 0:
                env = dm1 / 4 * kappa ** -1. - dm1 ** 3 / 64 * kappa ** -3.
            node = (1. - env) / (1. + env)
            corr = kappa * node + dm1 * (K.log(4.0) + K.log(env) - 2 * K.log1p(env))
            xs = np.zeros(n)
            na = 0
            halfdim = 0.5 * dm1
            while na < n:
                m = n - na
                sb = rs.beta(halfdim, halfdim, size=m)
                cx = (1 - (1 + env) * sb) / (1 - (1 - env) * sb)
                at = rs.random(m)
                for i in range(m):
                    crit = (kappa * cx[i] + dm1 * (K.log((1 + env - cx[i] + cx[i] * env) / (1 + env)))
                            - corr) > K.log(at[i])
                    if crit:
                        xs[na] = cx[i]
                        na += 1
            rest = _sample_uniform_direction(dim - 1, na, rs)
            for i in range(n):
                t = K.sqrt(1 - xs[i] ** 2)
                samples[i, 0] = xs[i]
                for j in range(dim - 1):
                    samples[i, j + 1] = t * rest[i, j]
        out = self._rotate(samples)
        if size == 1:
            return out[0].copy()
        else:
            return out

class vonmises_fisher_gen:
    def __call__(self, mu, kappa = 1.0, seed = None):
        return vonmises_fisher_frozen(mu, kappa, seed)
    def logpdf(self, x, mu, kappa = 1.0): return vonmises_fisher_frozen(mu, kappa).logpdf(x)
    def pdf(self, x, mu, kappa = 1.0): return vonmises_fisher_frozen(mu, kappa).pdf(x)
    def entropy(self, mu, kappa = 1.0) -> float: return vonmises_fisher_frozen(mu, kappa).entropy()
    def rvs(self, mu, kappa = 1.0, size: Literal[int] = 1, random_state = None):
        return vonmises_fisher_frozen(mu, kappa).rvs(size, random_state)

vonmises_fisher = vonmises_fisher_gen()

# ------------------------------------------------- multivariate_hypergeom

class multivariate_hypergeom_frozen:
    m: np.ndarray[int, 1]
    n: int
    M: int
    bad: bool
    def __init__(self, m, n, seed = None):
        self.m = np.asarray(m).astype(int).ravel()
        self.n = int(n)
        self.M = int(np.sum(self.m))
        self.bad = bool((self.m < 0).any()) or self.n < 0 or self.n > self.M
    def logpmf(self, x) -> float:
        xa = np.asarray(x).astype(int).ravel()
        if xa.size != self.m.size:
            raise ValueError("Size of each quantile must be size of 'm'")
        if self.bad:
            return NAN
        tot = 0
        for i in range(xa.size):
            if xa[i] < 0 or xa[i] > self.m[i]:
                return -INF
            tot += xa[i]
        if tot != self.n:
            return -INF
        num = np.empty(xa.size, dtype=float)
        for i in range(xa.size):
            mi = float(self.m[i])
            xi = float(xa[i])
            num[i] = K.betaln(mi + 1, 1) - K.betaln(xi + 1, mi - xi + 1)
        Mf = float(self.M)
        nf = float(self.n)
        den = K.betaln(Mf + 1, 1) - K.betaln(nf + 1, Mf - nf + 1)
        return float(np.sum(num)) - den
    def pmf(self, x) -> float:
        return K.exp(self.logpmf(x))
    def mean(self):
        Mf = float(self.M)
        return self.n * (self.m / Mf)
    def var(self):
        Mf = float(self.M)
        mf = self.m.astype(float)
        return self.n * mf / Mf * (Mf - mf) / Mf * (Mf - self.n) / (Mf - 1)
    def cov(self):
        Mf = float(self.M)
        nf = float(self.n)
        mf = self.m.astype(float)
        k = mf.size
        out = np.empty((k, k), dtype=float)
        for i in range(k):
            for j in range(k):
                out[i, j] = -nf * (Mf - nf) / (Mf - 1) * (mf[i] * mf[j]) / (Mf ** 2)
            out[i, i] = (nf * (Mf - nf) * mf[i] * (Mf - mf[i])) / (Mf - 1) / (Mf ** 2)
        return out
    def rvs(self, size = None, random_state = None):
        rs = check_random_state(random_state)
        k = self.m.size
        if size is None:
            out = np.empty(k, dtype=int)
            rem = self.M
            n = self.n
            for c in range(k - 1):
                rem = rem - self.m[c]
                n0 = 1 if n == 0 else 0
                v = (1 - n0) * int(rs.hypergeometric(int(self.m[c]), rem + n0, n + n0))
                out[c] = v
                n = n - v
            out[k - 1] = n
            return out.reshape(1, k)
        else:
            cnt = int(size)
            out = np.empty((cnt, k), dtype=int)
            rem = self.M
            nv = np.full(cnt, self.n, dtype=int)
            for c in range(k - 1):
                rem = rem - self.m[c]
                n0 = (nv == 0).astype(int)
                hv = rs.hypergeometric(np.full(cnt, int(self.m[c]), dtype=int), rem + n0, nv + n0,
                                       size=cnt)
                for i in range(cnt):
                    out[i, c] = (1 - n0[i]) * hv[i]
                    nv[i] = nv[i] - out[i, c]
            for i in range(cnt):
                out[i, k - 1] = nv[i]
            return out

class multivariate_hypergeom_gen:
    def __call__(self, m, n, seed = None):
        return multivariate_hypergeom_frozen(m, n, seed)
    def logpmf(self, x, m, n) -> float: return multivariate_hypergeom_frozen(m, n).logpmf(x)
    def pmf(self, x, m, n) -> float: return multivariate_hypergeom_frozen(m, n).pmf(x)
    def mean(self, m, n): return multivariate_hypergeom_frozen(m, n).mean()
    def var(self, m, n): return multivariate_hypergeom_frozen(m, n).var()
    def cov(self, m, n): return multivariate_hypergeom_frozen(m, n).cov()
    def rvs(self, m, n, size = None, random_state = None):
        return multivariate_hypergeom_frozen(m, n).rvs(size, random_state)

multivariate_hypergeom = multivariate_hypergeom_gen()
