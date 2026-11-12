# Ported from scipy 1.18.1 (BSD-3-Clause)

# scipy.special: elementwise special functions. Scalars in -> float out;
# any ndarray / list argument -> float64 ndarray broadcast over all inputs.

import numpy as np
import internal.static as static
from . import _kernels as K
from . import _cdf as C
from . import _bessel as B
from . import _gamma2 as G2
from . import _noncentral as NC
from . import _kolmogorov as KS
from . import _hyp as H
from . import _misc as M
from . import _ellip as E
from . import _orth as O

def _is_arr(x) -> Literal[bool]:
    return isinstance(x, np.ndarray) or isinstance(x, List) or isinstance(x, Tuple)

def _bshape(*xs):
    return np.broadcast_shapes(*tuple(np.asarray(x).shape for x in xs))

def _map1(f, x):
    if _is_arr(x):
        a = np.asarray(x)
        out = np.empty(a.shape, dtype=float)
        for idx in np.ndindex(a.shape):
            out[idx] = f(float(a[idx]))
        return out
    else:
        return f(float(x))

def _map2(f, x, y):
    if _is_arr(x) or _is_arr(y):
        shp = _bshape(x, y)
        a = np.broadcast_to(np.asarray(x), shp)
        b = np.broadcast_to(np.asarray(y), shp)
        out = np.empty(shp, dtype=float)
        for idx in np.ndindex(shp):
            out[idx] = f(float(a[idx]), float(b[idx]))
        return out
    else:
        return f(float(x), float(y))

def _map3(f, x, y, z):
    if _is_arr(x) or _is_arr(y) or _is_arr(z):
        shp = _bshape(x, y, z)
        a = np.broadcast_to(np.asarray(x), shp)
        b = np.broadcast_to(np.asarray(y), shp)
        c = np.broadcast_to(np.asarray(z), shp)
        out = np.empty(shp, dtype=float)
        for idx in np.ndindex(shp):
            out[idx] = f(float(a[idx]), float(b[idx]), float(c[idx]))
        return out
    else:
        return f(float(x), float(y), float(z))

# gamma family
def gamma(x): return _map1(K.gamma, x)
def gammaln(x): return _map1(K.gammaln, x)
def loggamma(x):
    # real input only: log(gamma(x)) for x >= 0, nan for x < 0 (scipy real loggamma)
    return _map1(_loggamma_real, x)
def _loggamma_real(x: float) -> float:
    if x < 0:
        return K.NAN
    return K.gammaln(x)
def gammasgn(x): return _map1(K.gammasgn, x)
def digamma(x): return _map1(K.digamma, x)
psi = digamma
def beta(a, b): return _map2(K.beta, a, b)
def betaln(a, b): return _map2(K.betaln, a, b)
def betainc(a, b, x): return _map3(K.betainc, a, b, x)
def betaincc(a, b, x): return _map3(K.betaincc, a, b, x)
def betaincinv(a, b, y): return _map3(K.betaincinv, a, b, y)
def gammainc(a, x): return _map2(K.gammainc, a, x)
def gammaincc(a, x): return _map2(K.gammaincc, a, x)
def gammaincinv(a, y): return _map2(K.gammaincinv, a, y)
def gammainccinv(a, y): return _map2(K.gammainccinv, a, y)

# error functions / normal
def erf(x): return _map1(K.erf, x)
def erfc(x): return _map1(K.erfc, x)
def erfcx(x): return _map1(K.erfcx, x)
def erfinv(y): return _map1(K.erfinv, y)
def erfcinv(y): return _map1(K.erfcinv, y)
def ndtr(x): return _map1(K.ndtr, x)
def ndtri(y): return _map1(K.ndtri, y)
def log_ndtr(x): return _map1(K.log_ndtr, x)

# distribution functions
def stdtr(df, t): return _map2(K.stdtr, df, t)
def stdtrit(df, p): return _map2(K.stdtrit, df, p)
def fdtr(dfn, dfd, x): return _map3(K.fdtr, dfn, dfd, x)
def fdtrc(dfn, dfd, x): return _map3(K.fdtrc, dfn, dfd, x)
def fdtri(dfn, dfd, p): return _map3(K.fdtri, dfn, dfd, p)
def chdtr(v, x): return _map2(K.chdtr, v, x)
def chdtrc(v, x): return _map2(K.chdtrc, v, x)
def chdtri(v, p): return _map2(K.chdtri, v, p)
def bdtr(k, n, p): return _map3(C.bdtr, k, n, p)

# discrete / gamma distribution functions (_cdf)
def pdtr(k, m): return _map2(C.pdtr, k, m)
def pdtrc(k, m): return _map2(C.pdtrc, k, m)
def pdtri(k, y): return _map2(C.pdtri, k, y)
def pdtrik(p, m): return _map2(C.pdtrik, p, m)
def bdtrc(k, n, p): return _map3(C.bdtrc, k, n, p)
def bdtri(k, n, y): return _map3(C.bdtri, k, n, y)
def nbdtr(k, n, p): return _map3(C.nbdtr, k, n, p)
def nbdtrc(k, n, p): return _map3(C.nbdtrc, k, n, p)
def nbdtri(k, n, y): return _map3(C.nbdtri, k, n, y)
def gdtr(a, b, x): return _map3(C.gdtr, a, b, x)
def gdtrc(a, b, x): return _map3(C.gdtrc, a, b, x)
def gdtria(p, b, x): return _map3(C.gdtria, p, b, x)
def gdtrib(a, p, x): return _map3(C.gdtrib, a, p, x)
def gdtrix(a, b, p): return _map3(C.gdtrix, a, b, p)
def ndtri_exp(y): return _map1(C.ndtri_exp, y)
def btdtria(p, b, x): return _map3(C.btdtria, p, b, x)
def btdtrib(a, p, x): return _map3(C.btdtrib, a, p, x)
def chdtriv(p, x): return _map2(C.chdtriv, p, x)
def bdtrik(y, n, p): return _map3(C.bdtrik, y, n, p)
def bdtrin(k, y, p): return _map3(C.bdtrin, k, y, p)
def nbdtrik(y, n, p): return _map3(C.nbdtrik, y, n, p)
def nbdtrin(k, y, p): return _map3(C.nbdtrin, k, y, p)
def stdtridf(p, t): return _map2(C.stdtridf, p, t)
def fdtridfd(dfn, p, x): return _map3(C.fdtridfd, dfn, p, x)

# elementary
def xlogy(x, y): return _map2(K.xlogy, x, y)
def xlog1py(x, y): return _map2(K.xlog1py, x, y)
def entr(x): return _map1(K.entr, x)
def rel_entr(x, y): return _map2(K.rel_entr, x, y)
def kl_div(x, y): return _map2(K.kl_div, x, y)
def expit(x): return _map1(K.expit, x)
def logit(x): return _map1(K.logit, x)
def log_expit(x): return _map1(K.log_expit, x)
def boxcox(x, lmbda): return _map2(K.boxcox, x, lmbda)
def log1p(x): return _map1(K.log1p, x)
def expm1(x): return _map1(K.expm1, x)

# Bessel
def i0(x): return _map1(K.i0, x)
def i0e(x): return _map1(K.i0e, x)
def i1(x): return _map1(K.i1, x)
def i1e(x): return _map1(K.i1e, x)

def j0(x): return _map1(B.j0, x)
def j1(x): return _map1(B.j1, x)
def y0(x): return _map1(B.y0, x)
def y1(x): return _map1(B.y1, x)
def k0(x): return _map1(B.k0, x)
def k1(x): return _map1(B.k1, x)
def k0e(x): return _map1(B.k0e, x)
def k1e(x): return _map1(B.k1e, x)
def jv(v, z): return _map2(B.jv, v, z)
jn = jv
def yv(v, z): return _map2(B.yv, v, z)
def yn(n, x): return _map2(B.yn, n, x)
def iv(v, z): return _map2(B.iv, v, z)
def ive(v, z): return _map2(B.ive, v, z)
def kv(v, z): return _map2(B.kv, v, z)
def kve(v, z): return _map2(B.kve, v, z)
def kn(n, x): return _map2(B.kn, n, x)
def rgamma(x): return _map1(B.rgamma, x)
def _sph_d(f, fd, n, z, derivative: Literal[bool]):
    if derivative:
        return _map2(fd, n, z)
    else:
        return _map2(f, n, z)
def spherical_jn(n, z, derivative: Literal[bool] = False): return _sph_d(B.spherical_jn, B.spherical_jn_d, n, z, derivative)
def spherical_yn(n, z, derivative: Literal[bool] = False): return _sph_d(B.spherical_yn, B.spherical_yn_d, n, z, derivative)
def spherical_in(n, z, derivative: Literal[bool] = False): return _sph_d(B.spherical_in, B.spherical_in_d, n, z, derivative)
def spherical_kn(n, z, derivative: Literal[bool] = False): return _sph_d(B.spherical_kn, B.spherical_kn_d, n, z, derivative)
def _airy_k(i: int, x: float) -> float:
    return B.airy(x)[i]
def _ai(x: float) -> float: return B.airy(x)[0]
def _aip(x: float) -> float: return B.airy(x)[1]
def _bi(x: float) -> float: return B.airy(x)[2]
def _bip(x: float) -> float: return B.airy(x)[3]
def airy(z):
    """Real-argument Airy functions: (Ai, Ai', Bi, Bi')."""
    return (_map1(_ai, z), _map1(_aip, z), _map1(_bi, z), _map1(_bip, z))

# gamma-family extras (_gamma2)
def zeta(x, q = None):
    if q is None:
        return _map1(G2.riemann_zeta, x)
    else:
        return _map2(G2.zeta, x, q)
def zetac(x): return _map1(G2.zetac, x)
def exp1(x): return _map1(G2.exp1, x)
def expi(x): return _map1(G2.expi, x)
def expn(n, x): return _map2(G2.expn, n, x)
def spence(x): return _map1(G2.spence, x)
def poch(a, m): return _map2(G2.poch, a, m)
def polygamma(n, x): return _map2(G2.polygamma, n, x)
def owens_t(h, a): return _map2(G2.owens_t, h, a)
def _scaled_exp1(x): return _map1(G2.scaled_exp1, x)

# noncentral distributions (_noncentral)
def chndtr(x, df, nc): return _map3(NC.chndtr, x, df, nc)
def chndtrix(p, df, nc): return _map3(NC.chndtrix, p, df, nc)
def nctdtr(df, nc, t): return _map3(NC.nctdtr, df, nc, t)
def nctdtrit(df, nc, p): return _map3(NC.nctdtrit, df, nc, p)
def _map4(f, a, b, c, d):
    if _is_arr(a) or _is_arr(b) or _is_arr(c) or _is_arr(d):
        shp = _bshape(a, b, c, d)
        A = np.broadcast_to(np.asarray(a), shp)
        Bb = np.broadcast_to(np.asarray(b), shp)
        Cc = np.broadcast_to(np.asarray(c), shp)
        D = np.broadcast_to(np.asarray(d), shp)
        out = np.empty(shp, dtype=float)
        for idx in np.ndindex(shp):
            out[idx] = f(float(A[idx]), float(Bb[idx]), float(Cc[idx]), float(D[idx]))
        return out
    else:
        return f(float(a), float(b), float(c), float(d))
def ncfdtr(dfn, dfd, nc, f): return _map4(NC.ncfdtr, dfn, dfd, nc, f)
def ncfdtri(dfn, dfd, nc, p): return _map4(NC.ncfdtri, dfn, dfd, nc, p)

# Kolmogorov-Smirnov (_kolmogorov)
def kolmogorov(y): return _map1(KS.kolmogorov, y)
def kolmogi(p): return _map1(KS.kolmogi, p)
def smirnov(n, d): return _map2(KS.smirnov, n, d)
def smirnovi(n, p): return _map2(KS.smirnovi, n, p)
def _kolmogc(x): return _map1(KS.kolmogc, x)
def _kolmogci(p): return _map1(KS.kolmogci, p)
def _kolmogp(x): return _map1(KS.kolmogp, x)
def _smirnovc(n, d): return _map2(KS.smirnovc, n, d)
def _smirnovci(n, p): return _map2(KS.smirnovci, n, p)
def _smirnovp(n, d): return _map2(KS.smirnovp, n, d)

# hypergeometric (_hyp)
def hyp2f1(a, b, c, z): return _map4(H.hyp2f1, a, b, c, z)
def hyp0f1(v, z): return _map2(H.hyp0f1, v, z)
def hyp1f1(a, b, x): return _map3(H.hyp1f1, a, b, x)
def hyperu(a, b, x): return _map3(H.hyperu, a, b, x)

# misc (_misc)
def boxcox1p(x, lmbda): return _map2(M.boxcox1p, x, lmbda)
def inv_boxcox(y, lmbda): return _map2(M.inv_boxcox, y, lmbda)
def inv_boxcox1p(y, lmbda): return _map2(M.inv_boxcox1p, y, lmbda)
def betainccinv(a, b, y): return _map3(M.betainccinv, a, b, y)
def binom(n, k): return _map2(M.binom, n, k)
def sindg(x): return _map1(M.sindg, x)
def cosdg(x): return _map1(M.cosdg, x)
def tandg(x): return _map1(M.tandg, x)
def cotdg(x): return _map1(M.cotdg, x)
def radian(d, m, s): return _map3(M.radian, d, m, s)
def cosm1(x): return _map1(M.cosm1, x)
def round(x): return _map1(M.cround, x)
def exp2(x): return _map1(M.exp2, x)
def exp10(x): return _map1(M.exp10, x)
def cbrt(x): return _map1(M.cbrt, x)
def huber(delta, r): return _map2(M.huber, delta, r)
def pseudo_huber(delta, r): return _map2(M.pseudo_huber, delta, r)

# elliptic (_ellip)
def ellipk(m): return _map1(E.ellipk, m)
def ellipkm1(p): return _map1(E.ellipkm1, p)
def ellipe(m): return _map1(E.ellipe, m)
def ellipkinc(phi, m): return _map2(E.ellipkinc, phi, m)
def ellipeinc(phi, m): return _map2(E.ellipeinc, phi, m)
def _ej0(u: float, m: float) -> float: return E.ellipj(u, m)[0]
def _ej1(u: float, m: float) -> float: return E.ellipj(u, m)[1]
def _ej2(u: float, m: float) -> float: return E.ellipj(u, m)[2]
def _ej3(u: float, m: float) -> float: return E.ellipj(u, m)[3]
def ellipj(u, m):
    """Jacobi elliptic functions: (sn, cn, dn, ph)."""
    return (_map2(_ej0, u, m), _map2(_ej1, u, m), _map2(_ej2, u, m), _map2(_ej3, u, m))
def agm(a, b): return _map2(E.agm, a, b)
def elliprc(x, y): return _map2(E.elliprc, x, y)
def elliprd(x, y, z): return _map3(E.elliprd, x, y, z)
def elliprf(x, y, z): return _map3(E.elliprf, x, y, z)
def elliprg(x, y, z): return _map3(E.elliprg, x, y, z)
def elliprj(x, y, z, p): return _map4(E.elliprj, x, y, z, p)

# orthogonal polynomials (_orth): integer-typed n uses scipy's recurrence
# ('p' loops), float n uses the hypergeometric representation.
def _ncall1(fl, fi, v, x: float) -> float:
    if isinstance(v, int) or isinstance(v, bool):
        return fi(int(v), x)
    else:
        return fl(float(v), x)
def _ncall2(fl, fi, v, a: float, x: float) -> float:
    if isinstance(v, int) or isinstance(v, bool):
        return fi(int(v), a, x)
    else:
        return fl(float(v), a, x)
def _ncall3(fl, fi, v, a: float, b: float, x: float) -> float:
    if isinstance(v, int) or isinstance(v, bool):
        return fi(int(v), a, b, x)
    else:
        return fl(float(v), a, b, x)
def _mapn1(fl, fi, n, x):
    if _is_arr(n) or _is_arr(x):
        shp = _bshape(n, x)
        A = np.broadcast_to(np.asarray(n), shp)
        X = np.broadcast_to(np.asarray(x), shp)
        out = np.empty(shp, dtype=float)
        for idx in np.ndindex(shp):
            out[idx] = _ncall1(fl, fi, A[idx], float(X[idx]))
        return out
    else:
        return _ncall1(fl, fi, n, float(x))
def _mapn2(fl, fi, n, a, x):
    if _is_arr(n) or _is_arr(a) or _is_arr(x):
        shp = _bshape(n, a, x)
        A = np.broadcast_to(np.asarray(n), shp)
        P = np.broadcast_to(np.asarray(a), shp)
        X = np.broadcast_to(np.asarray(x), shp)
        out = np.empty(shp, dtype=float)
        for idx in np.ndindex(shp):
            out[idx] = _ncall2(fl, fi, A[idx], float(P[idx]), float(X[idx]))
        return out
    else:
        return _ncall2(fl, fi, n, float(a), float(x))
def _mapn3(fl, fi, n, a, b, x):
    if _is_arr(n) or _is_arr(a) or _is_arr(b) or _is_arr(x):
        shp = _bshape(n, a, b, x)
        A = np.broadcast_to(np.asarray(n), shp)
        P = np.broadcast_to(np.asarray(a), shp)
        Q = np.broadcast_to(np.asarray(b), shp)
        X = np.broadcast_to(np.asarray(x), shp)
        out = np.empty(shp, dtype=float)
        for idx in np.ndindex(shp):
            out[idx] = _ncall3(fl, fi, A[idx], float(P[idx]), float(Q[idx]), float(X[idx]))
        return out
    else:
        return _ncall3(fl, fi, n, float(a), float(b), float(x))
def _herm_f(n: float, x: float) -> float: return O.eval_hermite(int(n), x)
def _hermn_f(n: float, x: float) -> float: return O.eval_hermitenorm(int(n), x)
def eval_jacobi(n, alpha, beta, x): return _mapn3(O.eval_jacobi, O.eval_jacobi_l, n, alpha, beta, x)
def eval_sh_jacobi(n, p, q, x): return _mapn3(O.eval_sh_jacobi, O.eval_sh_jacobi_l, n, p, q, x)
def eval_gegenbauer(n, alpha, x): return _mapn2(O.eval_gegenbauer, O.eval_gegenbauer_l, n, alpha, x)
def eval_chebyt(n, x): return _mapn1(O.eval_chebyt, O.eval_chebyt_l, n, x)
def eval_chebyu(n, x): return _mapn1(O.eval_chebyu, O.eval_chebyu_l, n, x)
def eval_chebys(n, x): return _mapn1(O.eval_chebys, O.eval_chebys_l, n, x)
def eval_chebyc(n, x): return _mapn1(O.eval_chebyc, O.eval_chebyc_l, n, x)
def eval_sh_chebyt(n, x): return _mapn1(O.eval_sh_chebyt, O.eval_sh_chebyt_l, n, x)
def eval_sh_chebyu(n, x): return _mapn1(O.eval_sh_chebyu, O.eval_sh_chebyu_l, n, x)
def eval_legendre(n, x): return _mapn1(O.eval_legendre, O.eval_legendre_l, n, x)
def eval_sh_legendre(n, x): return _mapn1(O.eval_sh_legendre, O.eval_sh_legendre_l, n, x)
def eval_genlaguerre(n, alpha, x): return _mapn2(O.eval_genlaguerre, O.eval_genlaguerre_l, n, alpha, x)
def eval_laguerre(n, x): return _mapn1(O.eval_laguerre, O.eval_laguerre_l, n, x)
def eval_hermite(n, x): return _mapn1(_herm_f, O.eval_hermite, n, x)
def eval_hermitenorm(n, x): return _mapn1(_hermn_f, O.eval_hermitenorm, n, x)

# combinatorics
def _comb_exact(N: int, k: int) -> int:
    if k > N or N < 0 or k < 0:
        return 0
    k = min(k, N - k)
    r = 1
    for i in range(1, k + 1):
        r = r * (N - k + i) // i
    return r

def _comb_f(N: float, k: float) -> float:
    if N < 0 or k < 0 or k > N:
        return 0.0
    if k == 0:
        return 1.0
    # binom via beta (xsf binom for non-negative arguments)
    if k > 1e8 * N or N < 0:
        return 0.0
    kx = K.floor(k)
    if k == kx and (N > 1e10 or k < 20):
        nx = K.floor(N)
        if N == nx:
            # product formula
            kk = kx
            if nx - kx < kx:
                kk = nx - kx
            if kk >= 0 and kk < 20:
                num = 1.0
                den = 1.0
                for i in range(1, int(kk) + 1):
                    num *= i + N - kk
                    den *= i
                    if K.fabs(num) > 1e50:
                        num /= den
                        den = 1.0
                return num / den
    if N >= 1e10 * k and k > 0:
        return K.exp(-K.betaln(1 + N - k, 1 + k) - K.log(N + 1))
    if k > 1e8 * K.fabs(N):
        num = K.gamma(1 + N) / K.fabs(k) + K.gamma(1 + N) * N / (2 * k * k)
        num /= K.pi * K.fabs(k) ** N
        return num
    return 1.0 / (N + 1) / K.beta(1 + N - k, 1 + k)

def comb(N, k, *, exact: Literal[bool] = False, repetition: bool = False):
    if exact:
        n = int(N)
        kk = int(k)
        if repetition:
            return _comb_exact(n + kk - 1, kk)
        return _comb_exact(n, kk)
    else:
        if repetition:
            return _map2(lambda a, b: _comb_f(a + b - 1.0, b), N, k)
        return _map2(_comb_f, N, k)

def perm(N, k, exact: Literal[bool] = False):
    if exact:
        n = int(N)
        kk = int(k)
        if kk > n or n < 0 or kk < 0:
            return 0
        r = 1
        for i in range(n - kk + 1, n + 1):
            r *= i
        return r
    else:
        return _map2(_perm_f, N, k)

def _perm_f(N: float, k: float) -> float:
    if k > N or N < 0 or k < 0:
        return 0.0
    return K.exp(K.gammaln(N + 1) - K.gammaln(N - k + 1)) if N > 170 else K.gamma(N + 1) / K.gamma(N - k + 1)

def factorial(n, exact: Literal[bool] = False):
    if exact:
        m = int(n)
        if m < 0:
            return 0
        r = 1
        for i in range(2, m + 1):
            r *= i
        return r
    else:
        return _map1(_fact_f, n)

def _fact_f(n: float) -> float:
    if n < 0 or not (n == n):
        return 0.0 if n < 0 else K.NAN
    return K.gamma(n + 1.0)

# reductions over arrays
def _amax_axis(a, axis):
    return np.max(a, axis=axis, keepdims=True)

def logsumexp(a, axis = None, b = None, keepdims: Literal[bool] = False,
              return_sign: Literal[bool] = False):
    """log(sum(b * exp(a))) computed stably. ``b`` weights supported; with
    ``return_sign`` returns ``(res, sign)``."""
    x = np.asarray(a).astype(float)
    if axis is None:
        mx = -K.INF
        for v in x.flat:
            if v > mx:
                mx = v
        if not (mx > -K.INF and mx < K.INF):
            mx = 0.0 if not (mx == mx) or mx == -K.INF else mx
        s = 0.0
        if b is None:
            for v in x.flat:
                s += K.exp(v - mx)
        else:
            bb = np.broadcast_to(np.asarray(b).astype(float), x.shape)
            for idx in np.ndindex(x.shape):
                s += bb[idx] * K.exp(x[idx] - mx)
        sgn = 1.0 if s >= 0 else -1.0
        r = K.log(K.fabs(s)) + mx
        if return_sign:
            if keepdims:
                return (np.full((1,) * x.ndim, r), np.full((1,) * x.ndim, sgn))
            return (r, sgn)
        else:
            if keepdims:
                return np.full((1,) * x.ndim, r)
            return r
    else:
        mx = np.max(x, axis=axis, keepdims=True)
        mx2 = np.where(np.isfinite(mx), mx, 0.0)
        if b is None:
            e = np.exp(x - mx2)
        else:
            e = np.asarray(b).astype(float) * np.exp(x - mx2)
        s = np.sum(e, axis=axis, keepdims=True)
        sgn = np.sign(s)
        r = np.log(np.abs(s)) + mx2
        if keepdims:
            if return_sign:
                return (r, sgn)
            else:
                return r
        else:
            r2 = np.squeeze(r, axis=axis)
            sgn2 = np.squeeze(sgn, axis=axis)
            if return_sign:
                return (r2, sgn2)
            else:
                return r2

def softmax(x, axis = None):
    a = np.asarray(x).astype(float)
    if axis is None:
        mx = np.max(a)
        e = np.exp(a - mx)
        return e / np.sum(e)
    else:
        mx = np.max(a, axis=axis, keepdims=True)
        e = np.exp(a - mx)
        return e / np.sum(e, axis=axis, keepdims=True)

def log_softmax(x, axis = None):
    a = np.asarray(x).astype(float)
    if axis is None:
        mx = np.max(a)
        mx = mx if (mx == mx and mx > -K.INF and mx < K.INF) else 0.0
        tmp = a - mx
        return tmp - np.log(np.sum(np.exp(tmp)))
    else:
        mx = np.max(a, axis=axis, keepdims=True)
        mx = np.where(np.isfinite(mx), mx, 0.0)
        tmp = a - mx
        return tmp - np.log(np.sum(np.exp(tmp), axis=axis, keepdims=True))
