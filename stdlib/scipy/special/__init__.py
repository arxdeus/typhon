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
from . import _roots as R
from . import _specfun as SPF
from . import _faddeeva as FD
from . import _struve as ST
from . import _lambertw as LW
from . import _kelvin as KV
from . import _parcyl as PC
from . import _legendre as LG
from . import _legendre_p as LP
import cmath
from C import log1p(float) -> float as _log1p

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

def _sici0(x: float) -> float: return G2.sici(x)[0]
def _sici1(x: float) -> float: return G2.sici(x)[1]
def _shichi0(x: float) -> float: return G2.shichi(x)[0]
def _shichi1(x: float) -> float: return G2.shichi(x)[1]
def _fres0(x: float) -> float: return G2.fresnel(x)[0]
def _fres1(x: float) -> float: return G2.fresnel(x)[1]
def struve(v, x): return _map2(ST.struve, v, x)
def modstruve(v, x): return _map2(ST.modstruve, v, x)
def erfi(x): return _map1(FD.erfi, x)
def dawsn(x): return _map1(FD.dawsn, x)
def sici(x): return (_map1(_sici0, x), _map1(_sici1, x))
def shichi(x): return (_map1(_shichi0, x), _map1(_shichi1, x))
def fresnel(z): return (_map1(_fres0, z), _map1(_fres1, z))

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

# Gauss quadrature roots/weights (_roots)
def _rw(t, mu: Literal[bool]):
    if mu:
        return t
    else:
        return (t[0], t[1])
def roots_legendre(n, mu: Literal[bool] = False): return _rw(R.roots_legendre_mu(n), mu)
def roots_chebyt(n, mu: Literal[bool] = False): return _rw(R.roots_chebyt_mu(n), mu)
def roots_chebyu(n, mu: Literal[bool] = False): return _rw(R.roots_chebyu_mu(n), mu)
def roots_gegenbauer(n, alpha, mu: Literal[bool] = False): return _rw(R.roots_gegenbauer_mu(n, float(alpha)), mu)
def roots_jacobi(n, alpha, beta, mu: Literal[bool] = False): return _rw(R.roots_jacobi_mu(n, float(alpha), float(beta)), mu)
def roots_genlaguerre(n, alpha, mu: Literal[bool] = False): return _rw(R.roots_genlaguerre_mu(n, float(alpha)), mu)
def roots_laguerre(n, mu: Literal[bool] = False): return _rw(R.roots_genlaguerre_mu(n, 0.0), mu)
def roots_hermite(n, mu: Literal[bool] = False): return _rw(R.roots_hermite_mu(n), mu)
def roots_hermitenorm(n, mu: Literal[bool] = False): return _rw(R.roots_hermitenorm_mu(n), mu)
def _roots_scale(t, sx: float, ox: float, sw: float):
    return (t[0] * sx + ox, t[1] * sw, t[2] * sw)
def roots_chebyc(n, mu: Literal[bool] = False): return _rw(_roots_scale(R.roots_chebyt_mu(n), 2.0, 0.0, 2.0), mu)
def roots_chebys(n, mu: Literal[bool] = False): return _rw(_roots_scale(R.roots_chebyu_mu(n), 2.0, 0.0, 2.0), mu)
def roots_sh_chebyt(n, mu: Literal[bool] = False):
    t = R.roots_chebyt_mu(n)
    return _rw(((t[0] + 1) / 2, t[1], t[2]), mu)
def roots_sh_chebyu(n, mu: Literal[bool] = False):
    t = R.roots_chebyu_mu(n)
    m_us = K.beta(1.5, 1.5)
    return _rw(((t[0] + 1) / 2, t[1] * (m_us / t[2]), m_us), mu)
def roots_sh_legendre(n, mu: Literal[bool] = False):
    t = R.roots_legendre_mu(n)
    return _rw(((t[0] + 1) / 2, t[1] / 2, 1.0), mu)
def roots_sh_jacobi(n, p1, q1, mu: Literal[bool] = False):
    if (p1 - q1) <= -1 or q1 <= 0:
        raise ValueError("(p - q) must be greater than -1, and q must be greater than 0.")
    t = R.roots_jacobi_mu(n, float(p1 - q1), float(q1 - 1))
    scale = 2.0 ** float(p1)
    return _rw(((t[0] + 1) / 2, t[1] / scale, t[2] / scale), mu)
p_roots = roots_legendre

# Bessel zeros (specfun jyzo)
def _jyzo(n, nt, which: int):
    if K.floor(float(n)) != float(n) or K.floor(float(nt)) != float(nt):
        raise ValueError("Arguments must be integers.")
    if nt <= 0:
        raise ValueError("nt > 0")
    r = SPF.jyzo(abs(int(n)), int(nt))
    return np.array(r[which])
def jnyn_zeros(n, nt):
    r = SPF.jyzo(abs(int(n)), int(nt))
    return (np.array(r[0]), np.array(r[1]), np.array(r[2]), np.array(r[3]))
def jn_zeros(n, nt): return _jyzo(n, nt, 0)
def jnp_zeros(n, nt): return _jyzo(n, nt, 1)
def yn_zeros(n, nt): return _jyzo(n, nt, 2)
def ynp_zeros(n, nt): return _jyzo(n, nt, 3)

# Lambert W / Wright omega (_lambertw)
def _lw1(z, k: int, tol: float) -> complex:
    return LW.lambertw(complex(z), k, tol)
def lambertw(z, k = 0, tol: float = 1e-8):
    """Lambert W function (complex result), branch k."""
    if _is_arr(z) or _is_arr(k):
        shp = _bshape(z, k)
        Z = np.broadcast_to(np.asarray(z), shp)
        Kk = np.broadcast_to(np.asarray(k), shp)
        out = np.empty(shp, dtype=complex)
        for idx in np.ndindex(shp):
            out[idx] = _lw1(Z[idx], int(Kk[idx]), tol)
        return out
    else:
        return _lw1(z, int(k), tol)
def wrightomega(x): return _map1(LW.wrightomega_real, x)

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

# Kelvin, parabolic cylinder, Faddeeva, Legendre, misc (_kelvin, _parcyl, _legendre, _legendre_p)
def ber(x): return _map1(KV.ber, x)
def bei(x): return _map1(KV.bei, x)
def ker(x): return _map1(KV.ker, x)
def kei(x): return _map1(KV.kei, x)
def berp(x): return _map1(KV.berp, x)
def beip(x): return _map1(KV.beip, x)
def kerp(x): return _map1(KV.kerp, x)
def keip(x): return _map1(KV.keip, x)
def _kelvin_i(x: float, i: int) -> complex: return KV.kelvin(x)[i]
def kelvin(x):
    """(Be, Ke, Bep, Kep) complex Kelvin function values."""
    if _is_arr(x):
        a = np.asarray(x)
        o0 = np.empty(a.shape, dtype=complex)
        o1 = np.empty(a.shape, dtype=complex)
        o2 = np.empty(a.shape, dtype=complex)
        o3 = np.empty(a.shape, dtype=complex)
        for idx in np.ndindex(a.shape):
            t = KV.kelvin(float(a[idx]))
            o0[idx] = t[0]
            o1[idx] = t[1]
            o2[idx] = t[2]
            o3[idx] = t[3]
        return (o0, o1, o2, o3)
    else:
        return KV.kelvin(float(x))
def _klvnzo(nt: int, kd: int):
    if nt <= 0 or nt != int(nt):
        raise ValueError("nt must be positive integer scalar.")
    return np.array(KV.klvnzo(nt, kd))
def ber_zeros(nt): return _klvnzo(int(nt), 1)
def bei_zeros(nt): return _klvnzo(int(nt), 2)
def ker_zeros(nt): return _klvnzo(int(nt), 3)
def kei_zeros(nt): return _klvnzo(int(nt), 4)
def berp_zeros(nt): return _klvnzo(int(nt), 5)
def beip_zeros(nt): return _klvnzo(int(nt), 6)
def kerp_zeros(nt): return _klvnzo(int(nt), 7)
def keip_zeros(nt): return _klvnzo(int(nt), 8)
def kelvin_zeros(nt):
    n = int(nt)
    return (_klvnzo(n, 1), _klvnzo(n, 2), _klvnzo(n, 3), _klvnzo(n, 4),
            _klvnzo(n, 5), _klvnzo(n, 6), _klvnzo(n, 7), _klvnzo(n, 8))

def _pbdv0(v: float, x: float) -> float: return PC.pbdv(v, x)[0]
def _pbdv1(v: float, x: float) -> float: return PC.pbdv(v, x)[1]
def _pbvv0(v: float, x: float) -> float: return PC.pbvv(v, x)[0]
def _pbvv1(v: float, x: float) -> float: return PC.pbvv(v, x)[1]
def _pbwa0(a: float, x: float) -> float: return PC.pbwa(a, x)[0]
def _pbwa1(a: float, x: float) -> float: return PC.pbwa(a, x)[1]
def pbdv(v, x):
    """Parabolic cylinder function D_v(x) and its derivative: (d, dp)."""
    return (_map2(_pbdv0, v, x), _map2(_pbdv1, v, x))
def pbvv(v, x):
    """Parabolic cylinder function V_v(x) and its derivative: (v, vp)."""
    return (_map2(_pbvv0, v, x), _map2(_pbvv1, v, x))
def pbwa(a, x):
    """Parabolic cylinder function W(a, x) and its derivative: (w, wp)."""
    return (_map2(_pbwa0, a, x), _map2(_pbwa1, a, x))

def wofz(z):
    """Faddeeva function w(z) = exp(-z**2) erfc(-iz)."""
    if _is_arr(z):
        a = np.asarray(z)
        out = np.empty(a.shape, dtype=complex)
        for idx in np.ndindex(a.shape):
            out[idx] = FD.w(complex(a[idx]))
        return out
    else:
        return FD.w(complex(z))
def voigt_profile(x, sigma, gamma): return _map3(FD.voigt_profile, x, sigma, gamma)

def lpmv(m, v, x): return _map3(LG.lpmv, m, v, x)
def tklmbda(x, lmbda): return _map2(LG.tklmbda, x, lmbda)

def _softplus1(x: float) -> float:
    # np.logaddexp(0, x)
    if x != x:
        return x
    if x > 0:
        return x + _log1p(K.exp(-x))
    elif x <= 0:
        return _log1p(K.exp(x))
    return x
def softplus(x): return _map1(_softplus1, x)

def multigammaln(a, d):
    """log of the multivariate gamma function of dimension d."""
    dd = int(d)
    if float(dd) != float(d):
        raise ValueError("d should be a positive integer (dimension)")
    A = np.asarray(a).astype(float)
    for v in A.flat:
        if v <= 0.5 * (dd - 1):
            raise ValueError("condition a > 0.5 * (d-1) not met")
    if _is_arr(a):
        out = np.empty(A.shape, dtype=float)
        for idx in np.ndindex(A.shape):
            r = (dd * (dd - 1) * 0.25) * K.log(3.141592653589793)
            s = 0.0
            for j in range(1, dd + 1):
                s += K.gammaln(A[idx] - (j - 1.0) / 2)
            out[idx] = r + s
        return out
    else:
        av = float(a)
        r = (dd * (dd - 1) * 0.25) * K.log(3.141592653589793)
        s = 0.0
        for j in range(1, dd + 1):
            s += K.gammaln(av - (j - 1.0) / 2)
        return r + s

# ---- 1.18 Legendre API (legendre_p, assoc_legendre_p, sph_legendre_p, sph_harm_y)
def _is_cplx(z) -> Literal[bool]:
    if isinstance(z, complex):
        return True
    elif isinstance(z, np.ndarray):
        return z.dtype is complex
    else:
        return False

def _dual_k(d, k: int):
    if k == 0:
        return d.v
    elif k == 1:
        return d.d1
    else:
        return d.d2

def _lp_core(kind: int, n, m, z, norm: bool, branch_cut: int, diff_n: int, T: type):
    if diff_n < 0 or diff_n > 2:
        raise ValueError("diff_n must be 0, 1 or 2")
    shp = _bshape(n, m, z)
    N = np.broadcast_to(np.asarray(n), shp)
    M = np.broadcast_to(np.asarray(m), shp)
    Z = np.broadcast_to(np.asarray(z), shp)
    out = np.empty((diff_n + 1,) + shp, dtype=T)
    for idx in np.ndindex(shp):
        d = LP.legendre_p_d(0, T(0.0), T)
        if kind == 0:
            d = LP.legendre_p_d(int(N[idx]), T(Z[idx]), T)
        else:
            d = LP.assoc_legendre_p_d(norm, int(N[idx]), int(M[idx]), T(Z[idx]), branch_cut, T)
        for k in range(diff_n + 1):
            out[(k,) + idx] = _dual_k(d, k)
    return out

def legendre_p(n, z, *, diff_n: int = 0):
    """Legendre polynomial P_n(z); result has leading axis of size diff_n + 1."""
    if _is_cplx(z):
        return _lp_core(0, n, 0, z, False, 2, diff_n, complex)
    else:
        return _lp_core(0, n, 0, z, False, 2, diff_n, float)

def assoc_legendre_p(n, m, z, *, branch_cut: int = 2, norm: bool = False, diff_n: int = 0):
    """Associated Legendre P_n^m(z) (1.18 API)."""
    if _is_cplx(z):
        return _lp_core(1, n, m, z, norm, branch_cut, diff_n, complex)
    else:
        return _lp_core(1, n, m, z, norm, branch_cut, diff_n, float)

def sph_legendre_p(n, m, theta, *, diff_n: int = 0):
    """Spherical Legendre P_n^m(cos theta) normalized as in sph_harm_y."""
    shp = _bshape(n, m, theta)
    N = np.broadcast_to(np.asarray(n), shp)
    M = np.broadcast_to(np.asarray(m), shp)
    TH = np.broadcast_to(np.asarray(theta), shp)
    out = np.empty((diff_n + 1,) + shp, dtype=float)
    for idx in np.ndindex(shp):
        d = LP.sph_legendre_p_d(int(N[idx]), int(M[idx]), float(TH[idx]))
        for k in range(diff_n + 1):
            out[(k,) + idx] = _dual_k(d, k)
    return out

def _sphy_parts(n: int, m: int, th: float, ph: float):
    d = LP.sph_legendre_p_d(n, m, th)
    e = cmath.exp(complex(0.0, float(m) * ph))
    return (d, e)

def sph_harm_y(n, m, theta, phi, *, diff_n: Literal[int] = 0):
    """Spherical harmonic Y_n^m(theta, phi) (theta polar, phi azimuthal)."""
    shp = _bshape(n, m, theta, phi)
    N = np.broadcast_to(np.asarray(n), shp)
    M = np.broadcast_to(np.asarray(m), shp)
    TH = np.broadcast_to(np.asarray(theta), shp)
    PH = np.broadcast_to(np.asarray(phi), shp)
    y = np.empty(shp, dtype=complex)
    if diff_n == 0 and not (_is_arr(n) or _is_arr(m) or _is_arr(theta) or _is_arr(phi)):
        t0 = _sphy_parts(int(n), int(m), float(theta), float(phi))
        return t0[0].v * t0[1]
    elif diff_n == 0:
        for idx in np.ndindex(shp):
            t = _sphy_parts(int(N[idx]), int(M[idx]), float(TH[idx]), float(PH[idx]))
            y[idx] = t[0].v * t[1]
        return y
    elif diff_n == 1:
        g = np.empty((2,) + shp, dtype=complex)
        for idx in np.ndindex(shp):
            mm = int(M[idx])
            t = _sphy_parts(int(N[idx]), mm, float(TH[idx]), float(PH[idx]))
            y[idx] = t[0].v * t[1]
            g[(0,) + idx] = t[0].d1 * t[1]
            g[(1,) + idx] = complex(0.0, float(mm)) * t[0].v * t[1]
        return (y, g)
    else:
        g = np.empty((2,) + shp, dtype=complex)
        h = np.empty((2, 2) + shp, dtype=complex)
        for idx in np.ndindex(shp):
            mm = int(M[idx])
            t = _sphy_parts(int(N[idx]), mm, float(TH[idx]), float(PH[idx]))
            im = complex(0.0, float(mm))
            y[idx] = t[0].v * t[1]
            g[(0,) + idx] = t[0].d1 * t[1]
            g[(1,) + idx] = im * t[0].v * t[1]
            h[(0, 0) + idx] = t[0].d2 * t[1]
            h[(0, 1) + idx] = im * t[0].d1 * t[1]
            h[(1, 0) + idx] = im * t[0].d1 * t[1]
            h[(1, 1) + idx] = -float(mm * mm) * t[0].v * t[1]
        return (y, g, h)

def legendre_p_all(n, z, *, diff_n: int = 0):
    """All P_j(z), 0 <= j <= n; shape (diff_n + 1, n + 1, *z.shape)."""
    nn = int(n)
    zs = np.asarray(z).shape
    T = complex if _is_cplx(z) else float
    out = np.empty((diff_n + 1, nn + 1) + zs, dtype=T)
    Z = np.asarray(z)
    for idx in np.ndindex(zs):
        for j in range(nn + 1):
            d = LP.legendre_p_d(j, T(Z[idx]), T)
            for k in range(diff_n + 1):
                out[(k, j) + idx] = _dual_k(d, k)
    return out

def assoc_legendre_p_all(n, m, z, *, branch_cut: int = 2, norm: bool = False, diff_n: int = 0):
    """All P_j^i(z), 0 <= j <= n, -m <= i <= m; shape (diff_n+1, n+1, 2m+1, *z.shape)."""
    nn = int(n)
    mm = int(m)
    zs = np.asarray(z).shape
    T = complex if _is_cplx(z) else float
    out = np.empty((diff_n + 1, nn + 1, 2 * mm + 1) + zs, dtype=T)
    Z = np.asarray(z)
    for idx in np.ndindex(zs):
        for j in range(nn + 1):
            for i in range(-mm, mm + 1):
                d = LP.assoc_legendre_p_d(norm, j, i, T(Z[idx]), branch_cut, T)
                col = i if i >= 0 else i + 2 * mm + 1
                for k in range(diff_n + 1):
                    out[(k, j, col) + idx] = _dual_k(d, k)
    return out

def sph_legendre_p_all(n, m, theta, *, diff_n: int = 0):
    nn = int(n)
    mm = int(m)
    zs = np.asarray(theta).shape
    out = np.empty((diff_n + 1, nn + 1, 2 * mm + 1) + zs, dtype=float)
    TH = np.asarray(theta)
    for idx in np.ndindex(zs):
        for j in range(nn + 1):
            for i in range(-mm, mm + 1):
                d = LP.sph_legendre_p_d(j, i, float(TH[idx]))
                col = i if i >= 0 else i + 2 * mm + 1
                for k in range(diff_n + 1):
                    out[(k, j, col) + idx] = _dual_k(d, k)
    return out

def sph_harm_y_all(n, m, theta, phi):
    """All Y_j^i(theta, phi), 0 <= j <= n, -m <= i <= m; shape (n+1, 2m+1, *shape)."""
    nn = int(n)
    mm = int(m)
    shp = _bshape(theta, phi)
    TH = np.broadcast_to(np.asarray(theta), shp)
    PH = np.broadcast_to(np.asarray(phi), shp)
    out = np.empty((nn + 1, 2 * mm + 1) + shp, dtype=complex)
    for idx in np.ndindex(shp):
        for j in range(nn + 1):
            for i in range(-mm, mm + 1):
                t = _sphy_parts(j, i, float(TH[idx]), float(PH[idx]))
                col = i if i >= 0 else i + 2 * mm + 1
                out[(j, col) + idx] = t[0].v * t[1]
    return out

# orthogonal polynomial objects (_orthogonal / _orthopoly)
from ._orthopoly import orthopoly1d
from . import _orthogonal as OG
def jacobi(n, alpha, beta, monic: bool = False): return OG.jacobi(int(n), float(alpha), float(beta), monic)
def sh_jacobi(n, p, q, monic: bool = False): return OG.sh_jacobi(int(n), float(p), float(q), monic)
def genlaguerre(n, alpha, monic: bool = False): return OG.genlaguerre(int(n), float(alpha), monic)
def laguerre(n, monic: bool = False): return OG.laguerre(int(n), monic)
def hermite(n, monic: bool = False): return OG.hermite(int(n), monic)
def hermitenorm(n, monic: bool = False): return OG.hermitenorm(int(n), monic)
def gegenbauer(n, alpha, monic: bool = False): return OG.gegenbauer(int(n), float(alpha), monic)
def chebyt(n, monic: bool = False): return OG.chebyt(int(n), monic)
def chebyu(n, monic: bool = False): return OG.chebyu(int(n), monic)
def chebyc(n, monic: bool = False): return OG.chebyc(int(n), monic)
def chebys(n, monic: bool = False): return OG.chebys(int(n), monic)
def sh_chebyt(n, monic: bool = False): return OG.sh_chebyt(int(n), monic)
def sh_chebyu(n, monic: bool = False): return OG.sh_chebyu(int(n), monic)
def legendre(n, monic: bool = False): return OG.legendre(int(n), monic)
def sh_legendre(n, monic: bool = False): return OG.sh_legendre(int(n), monic)
c_roots = roots_chebyc
cg_roots = roots_gegenbauer
h_roots = roots_hermite
he_roots = roots_hermitenorm
j_roots = roots_jacobi
js_roots = roots_sh_jacobi
l_roots = roots_laguerre
la_roots = roots_genlaguerre
s_roots = roots_chebys
t_roots = roots_chebyt
ts_roots = roots_sh_chebyt
u_roots = roots_chebyu
us_roots = roots_sh_chebyu
ps_roots = roots_sh_legendre

# ---------------------------------------------------------------------------
# sequences, zeros, integrals (_specfun2) and python-level helpers of
# scipy.special._basic
from . import _specfun2 as S2

def _nonneg_int(n, name: str) -> int:
    if n != int(n) or n < 0:
        raise ValueError(name + " must be a non-negative integer.")
    return int(n)

def _arr(l: List[float], k: int):
    out = np.empty(k, dtype=float)
    for i in range(k):
        out[i] = l[i]
    return out

def ai_zeros(nt):
    if nt != int(nt) or nt <= 0:
        raise ValueError("nt must be a positive integer scalar.")
    t = S2.airyzo(int(nt), 1)
    return (np.array(t[0]), np.array(t[1]), np.array(t[2]), np.array(t[3]))

def bi_zeros(nt):
    if nt != int(nt) or nt <= 0:
        raise ValueError("nt must be a positive integer scalar.")
    t = S2.airyzo(int(nt), 2)
    return (np.array(t[0]), np.array(t[1]), np.array(t[2]), np.array(t[3]))

def _czeros(l: List[complex]):
    out = np.empty(len(l), dtype=complex)
    for i in range(len(l)):
        out[i] = l[i]
    return out

def erf_zeros(nt):
    if nt != int(nt) or nt <= 0:
        raise ValueError("Argument must be positive scalar integer.")
    return _czeros(S2.cerzo(int(nt)))

def fresnel_zeros(nt):
    if nt != int(nt) or nt <= 0:
        raise ValueError("Argument must be positive scalar integer.")
    return (_czeros(S2.fcszo(2, int(nt))), _czeros(S2.fcszo(1, int(nt))))

def fresnelc_zeros(nt):
    if nt != int(nt) or nt <= 0:
        raise ValueError("Argument must be positive scalar integer.")
    return _czeros(S2.fcszo(1, int(nt)))

def fresnels_zeros(nt):
    if nt != int(nt) or nt <= 0:
        raise ValueError("Argument must be positive scalar integer.")
    return _czeros(S2.fcszo(2, int(nt)))

def _cyz_c(nt, kf: int):
    if nt != int(nt) or nt <= 0:
        raise ValueError("Arguments must be scalar positive integer.")
    t = S2.cyzo(int(nt), kf, 0)
    return (_czeros(t[0]), _czeros(t[1]))
def _cyz_r(nt, kf: int):
    if nt != int(nt) or nt <= 0:
        raise ValueError("Arguments must be scalar positive integer.")
    n = int(nt)
    t = S2.cyzo(n, kf, 1)
    zr = np.empty(n, dtype=float)
    vr = np.empty(n, dtype=float)
    for i in range(n):
        zr[i] = t[0][i].real
        vr[i] = t[1][i].real
    return (zr, vr)

def y0_zeros(nt, complex: Literal[bool] = False):
    if complex:
        return _cyz_c(nt, 0)
    else:
        return _cyz_r(nt, 0)
def y1_zeros(nt, complex: Literal[bool] = False):
    if complex:
        return _cyz_c(nt, 1)
    else:
        return _cyz_r(nt, 1)
def y1p_zeros(nt, complex: Literal[bool] = False):
    if complex:
        return _cyz_c(nt, 2)
    else:
        return _cyz_r(nt, 2)

# derivatives of Bessel functions via A&S 9.1.31
def _asv(v):
    if _is_arr(v):
        return np.asarray(v).astype(float)
    else:
        return float(v)
def _bdiff_jv(v, z, n: int, phase: float):
    p = 1.0
    s = jv(_asv(v) - n, z)
    for i in range(1, n + 1):
        p = phase * (p * (n - i + 1)) / i
        s = s + p * jv(_asv(v) - n + i * 2, z)
    return s / (2.0 ** n)
def _bdiff_yv(v, z, n: int, phase: float):
    p = 1.0
    s = yv(_asv(v) - n, z)
    for i in range(1, n + 1):
        p = phase * (p * (n - i + 1)) / i
        s = s + p * yv(_asv(v) - n + i * 2, z)
    return s / (2.0 ** n)
def _bdiff_iv(v, z, n: int, phase: float):
    p = 1.0
    s = iv(_asv(v) - n, z)
    for i in range(1, n + 1):
        p = phase * (p * (n - i + 1)) / i
        s = s + p * iv(_asv(v) - n + i * 2, z)
    return s / (2.0 ** n)
def _bdiff_kv(v, z, n: int, phase: float):
    p = 1.0
    s = kv(_asv(v) - n, z)
    for i in range(1, n + 1):
        p = phase * (p * (n - i + 1)) / i
        s = s + p * kv(_asv(v) - n + i * 2, z)
    return s / (2.0 ** n)

def jvp(v, z, n = 1):
    m = _nonneg_int(n, 'n')
    if m == 0:
        return jv(v, z)
    return _bdiff_jv(v, z, m, -1.0)
def yvp(v, z, n = 1):
    m = _nonneg_int(n, 'n')
    if m == 0:
        return yv(v, z)
    return _bdiff_yv(v, z, m, -1.0)
def ivp(v, z, n = 1):
    m = _nonneg_int(n, 'n')
    if m == 0:
        return iv(v, z)
    return _bdiff_iv(v, z, m, 1.0)
def kvp(v, z, n = 1):
    m = _nonneg_int(n, 'n')
    if m == 0:
        return kv(v, z)
    sg = -1.0 if m % 2 == 1 else 1.0
    return sg * _bdiff_kv(v, z, m, 1.0)

def _hk1(v: float, z: float) -> complex:
    return complex(B.jv(v, z), B.yv(v, z))
def _hk2(v: float, z: float) -> complex:
    return complex(B.jv(v, z), -B.yv(v, z))
def _cmap2(f, x, y):
    if _is_arr(x) or _is_arr(y):
        shp = _bshape(x, y)
        a = np.broadcast_to(np.asarray(x), shp)
        b = np.broadcast_to(np.asarray(y), shp)
        out = np.empty(shp, dtype=complex)
        for idx in np.ndindex(shp):
            out[idx] = f(float(a[idx]), float(b[idx]))
        return out
    else:
        return f(float(x), float(y))
def hankel1(v, z): return _cmap2(_hk1, v, z)
def hankel2(v, z): return _cmap2(_hk2, v, z)
def _hk1e(v: float, z: float) -> complex:
    return _hk1(v, z) * cmath.exp(complex(0.0, -z))
def _hk2e(v: float, z: float) -> complex:
    return _hk2(v, z) * cmath.exp(complex(0.0, z))
def hankel1e(v, z): return _cmap2(_hk1e, v, z)
def hankel2e(v, z): return _cmap2(_hk2e, v, z)
def h1vp(v, z, n = 1):
    m = _nonneg_int(n, 'n')
    if m == 0:
        return hankel1(v, z)
    p = 1.0
    vv = _asv(v)
    s = hankel1(vv - m, z)
    for i in range(1, m + 1):
        p = -1.0 * (p * (m - i + 1)) / i
        s = s + p * hankel1(vv - m + i * 2, z)
    return s / (2.0 ** m)
def h2vp(v, z, n = 1):
    m = _nonneg_int(n, 'n')
    if m == 0:
        return hankel2(v, z)
    p = 1.0
    vv = _asv(v)
    s = hankel2(vv - m, z)
    for i in range(1, m + 1):
        p = -1.0 * (p * (m - i + 1)) / i
        s = s + p * hankel2(vv - m + i * 2, z)
    return s / (2.0 ** m)

def riccati_jn(n, x):
    m = _nonneg_int(n, 'n')
    n1 = 1 if m == 0 else m
    t = S2.rctj(float(x), n1)
    return (_arr(t[0], m + 1), _arr(t[1], m + 1))
def riccati_yn(n, x):
    m = _nonneg_int(n, 'n')
    n1 = 1 if m == 0 else m
    t = S2.rcty(float(x), n1)
    return (_arr(t[0], m + 1), _arr(t[1], m + 1))

def lmbda(v, x):
    if v < 0:
        raise ValueError("argument must be > 0.")
    n = int(v)
    v0 = v - n
    n1 = 1 if n < 1 else n
    v1 = n1 + v0
    if v != K.floor(float(v)):
        t = S2.lamv(float(v1), float(x))
        return (_arr(t[0], n + 1), _arr(t[1], n + 1))
    else:
        t = S2.lamn(int(v1), float(x))
        return (_arr(t[0], n + 1), _arr(t[1], n + 1))

def lqn(n, z):
    m = _nonneg_int(n, 'n')
    n1 = 1 if m < 1 else m
    t = S2.lqn(float(z), n1)
    return (_arr(t[0], m + 1), _arr(t[1], m + 1))

def lqmn(m, n, z):
    if m < 0:
        raise ValueError("m must be a non-negative integer.")
    if n < 0:
        raise ValueError("n must be a non-negative integer.")
    mi = int(m)
    ni = int(n)
    mm = max(1, mi)
    nn = max(1, ni)
    t = S2.lqmn(float(z), mm, nn)
    w = nn + 1
    q = np.empty((mi + 1, ni + 1), dtype=float)
    qd = np.empty((mi + 1, ni + 1), dtype=float)
    for i in range(mi + 1):
        for j in range(ni + 1):
            q[i, j] = t[0][i * w + j]
            qd[i, j] = t[1][i * w + j]
    return (q, qd)

def bernoulli(n):
    if n < 0:
        raise ValueError("n must be a non-negative integer.")
    m = int(n)
    n1 = 2 if m < 2 else m
    return _arr(S2.bernob(n1), m + 1)
def euler(n):
    if n < 0:
        raise ValueError("n must be a non-negative integer.")
    m = int(n)
    n1 = 2 if m < 2 else m
    return _arr(S2.eulerb(n1), m + 1)

def pbdv_seq(v, x):
    t = PC.pbdv_seq(float(v), float(x))
    return (np.array(t[0]), np.array(t[1]))
def pbvv_seq(v, x):
    t = PC.pbvv_seq(float(v), float(x))
    return (np.array(t[0]), np.array(t[1]))

def _i1(x: float, i: int) -> float:
    if i == 0:
        return S2.it1j0y0(x)[0]
    return S2.it1j0y0(x)[1]
def _i2(x: float, i: int) -> float:
    if i == 0:
        return S2.it2j0y0(x)[0]
    return S2.it2j0y0(x)[1]
def _i3(x: float, i: int) -> float:
    if i == 0:
        return S2.it1i0k0(x)[0]
    return S2.it1i0k0(x)[1]
def _i4(x: float, i: int) -> float:
    if i == 0:
        return S2.it2i0k0(x)[0]
    return S2.it2i0k0(x)[1]
def _itj0(x: float) -> float: return _i1(x, 0)
def _ity0(x: float) -> float: return _i1(x, 1)
def _it2j0(x: float) -> float: return _i2(x, 0)
def _it2y0(x: float) -> float: return _i2(x, 1)
def _iti0(x: float) -> float: return _i3(x, 0)
def _itk0(x: float) -> float: return _i3(x, 1)
def _it2i0(x: float) -> float: return _i4(x, 0)
def _it2k0(x: float) -> float: return _i4(x, 1)
def itj0y0(x): return (_map1(_itj0, x), _map1(_ity0, x))
def it2j0y0(x): return (_map1(_it2j0, x), _map1(_it2y0, x))
def iti0k0(x): return (_map1(_iti0, x), _map1(_itk0, x))
def it2i0k0(x): return (_map1(_it2i0, x), _map1(_it2k0, x))
def itstruve0(x): return _map1(S2.itstruve0, x)
def it2struve0(x): return _map1(S2.it2struve0, x)
def itmodstruve0(x): return _map1(S2.itmodstruve0, x)
def _ita(x: float, i: int) -> float:
    return S2.itairy_signed(x)[i]
def _ita0(x: float) -> float: return _ita(x, 0)
def _ita1(x: float) -> float: return _ita(x, 1)
def _ita2(x: float) -> float: return _ita(x, 2)
def _ita3(x: float) -> float: return _ita(x, 3)
def itairy(x): return (_map1(_ita0, x), _map1(_ita1, x), _map1(_ita2, x), _map1(_ita3, x))

def _mfp(x: float, i: int, j: int) -> complex:
    return S2.modified_fresnel(0, x)[i] if j == 0 else S2.modified_fresnel(1, x)[i]
def _mfp0(x: float) -> complex: return S2.modified_fresnel(0, x)[0]
def _mfp1(x: float) -> complex: return S2.modified_fresnel(0, x)[1]
def _mfm0(x: float) -> complex: return S2.modified_fresnel(1, x)[0]
def _mfm1(x: float) -> complex: return S2.modified_fresnel(1, x)[1]
def _cmap1(f, x):
    if _is_arr(x):
        a = np.asarray(x)
        out = np.empty(a.shape, dtype=complex)
        for idx in np.ndindex(a.shape):
            out[idx] = f(float(a[idx]))
        return out
    else:
        return f(float(x))
def modfresnelp(x): return (_cmap1(_mfp0, x), _cmap1(_mfp1, x))
def modfresnelm(x): return (_cmap1(_mfm0, x), _cmap1(_mfm1, x))

def besselpoly(a, lmb, nu): return _map3(S2.besselpoly, a, lmb, nu)
def exprel(x): return _map1(_exprel_f, x)
def _exprel_f(x: float) -> float:
    if fabs_(x) < 1e-16:
        return 1.0
    elif x > 717.0:
        return float('inf')
    return K.expm1(x) / x
def fabs_(x: float) -> float:
    return -x if x < 0 else x
def powm1(x, y): return _map2(_powm1_f, x, y)
def _powm1_f(x: float, y: float) -> float:
    if x != x or y != y:
        return float('nan')
    if y == 0.0 or x == 1.0:
        return 0.0
    if x > 0:
        if fabs_(y * (x - 1.0)) < 0.5 or fabs_(y) < 0.2:
            l = y * K.log(x)
            if l < 0.5:
                return K.expm1(l)
    elif y != K.floor(y):
        return float('nan')
    return x ** y - 1.0
def sinc(x): return np.sinc(x)
def assoc_laguerre(x, n, k = 0.0): return eval_genlaguerre(n, k, x)
def diric(x, n): return _map2(_diric_f, x, n)
def _rint(v: float) -> float:
    f = K.floor(v)
    d = v - f
    if d > 0.5 or (d == 0.5 and (int(f) % 2 != 0)):
        return f + 1.0
    return f
def _diric_f(x: float, n: float) -> float:
    if n <= 0 or n != K.floor(n):
        return float('nan')
    h = x / 2
    d = K.sin(h)
    if fabs_(d) < 1e-7:
        z = _rint(h / 3.141592653589793)
        return (-1.0) ** (z * (n - 1))
    return K.sin(n * h) / (n * d)

def nrdtrimn(p, std, x):
    return x - std * ndtri(p)
def nrdtrisd(mn, p, x):
    return (x - mn) / ndtri(p)

def stirling2(N, K_, *, exact: Literal[bool] = False):
    if exact:
        n = int(N)
        k = int(K_)
        if n < 0 or k < 0 or k > n:
            return 0
        if n == 0 and k == 0:
            return 1
        row = [0] * (k + 2)
        row[0] = 1
        for i in range(1, n + 1):
            j = min(i, k)
            while j >= 1:
                row[j] = row[j] * j + row[j - 1]
                j -= 1
            row[0] = 0
        return row[k]
    else:
        return _map2(_st2_f, N, K_)
def _st2_f(n: float, k: float) -> float:
    if n < 0 or k < 0 or k > n:
        return 0.0
    ni = int(n)
    ki = int(k)
    if ni == 0 and ki == 0:
        return 1.0
    row = [0.0] * (ki + 2)
    row[0] = 1.0
    for i in range(1, ni + 1):
        j = min(i, ki)
        while j >= 1:
            row[j] = row[j] * j + row[j - 1]
            j -= 1
        row[0] = 0.0
    return row[ki]

def factorialk(n, k, exact: Literal[bool] = False, extend: str = "zero"):
    kk = int(k)
    if kk < 1:
        raise ValueError("For `extend='zero'`, k must be a positive integer, received: " + str(k))
    if exact:
        m = int(n)
        if m < 0:
            return 0
        r = 1
        while m > 1:
            r *= m
            m -= kk
        return r
    else:
        return _map1(_fk_gen(kk), n)
class _fk_gen:
    k: int
    def __init__(self, k: int):
        self.k = k
    def __call__(self, x: float) -> float:
        if x < 0:
            return 0.0
        m = int(x)
        r = 1.0
        while m > 1:
            r *= m
            m -= self.k
        return r
def factorial2(n, exact: Literal[bool] = False, extend: str = "zero"):
    return factorialk(n, 2, exact, extend)

# error-handling API (no state: special functions return nan on error)
class SpecialFunctionWarning(Warning):
    pass
class SpecialFunctionError(Exception):
    pass
def seterr(**kwargs):
    return {'singular': 'ignore', 'underflow': 'ignore', 'overflow': 'ignore', 'slow': 'ignore', 'loss': 'ignore', 'no_result': 'ignore', 'domain': 'ignore', 'arg': 'ignore', 'other': 'ignore'}
def geterr():
    return {'singular': 'ignore', 'underflow': 'ignore', 'overflow': 'ignore', 'slow': 'ignore', 'loss': 'ignore', 'no_result': 'ignore', 'domain': 'ignore', 'arg': 'ignore', 'other': 'ignore'}

# Mathieu functions (xsf specfun port, see _mathieu.py)
from . import _mathieu as MT
def mathieu_a(m, q): return _map2(MT.cem_cva, m, q)
def mathieu_b(m, q): return _map2(MT.sem_cva, m, q)
def _mt0(m: float, q: float, x: float) -> float: return MT.cem(m, q, x)[0]
def _mt1(m: float, q: float, x: float) -> float: return MT.cem(m, q, x)[1]
def _mt2(m: float, q: float, x: float) -> float: return MT.sem(m, q, x)[0]
def _mt3(m: float, q: float, x: float) -> float: return MT.sem(m, q, x)[1]
def _mt4(m: float, q: float, x: float) -> float: return MT.mcm1(m, q, x)[0]
def _mt5(m: float, q: float, x: float) -> float: return MT.mcm1(m, q, x)[1]
def _mt6(m: float, q: float, x: float) -> float: return MT.msm1(m, q, x)[0]
def _mt7(m: float, q: float, x: float) -> float: return MT.msm1(m, q, x)[1]
def _mt8(m: float, q: float, x: float) -> float: return MT.mcm2(m, q, x)[0]
def _mt9(m: float, q: float, x: float) -> float: return MT.mcm2(m, q, x)[1]
def _mt10(m: float, q: float, x: float) -> float: return MT.msm2(m, q, x)[0]
def _mt11(m: float, q: float, x: float) -> float: return MT.msm2(m, q, x)[1]
def mathieu_cem(m, q, x): return (_map3(_mt0, m, q, x), _map3(_mt1, m, q, x))
def mathieu_sem(m, q, x): return (_map3(_mt2, m, q, x), _map3(_mt3, m, q, x))
def mathieu_modcem1(m, q, x): return (_map3(_mt4, m, q, x), _map3(_mt5, m, q, x))
def mathieu_modsem1(m, q, x): return (_map3(_mt6, m, q, x), _map3(_mt7, m, q, x))
def mathieu_modcem2(m, q, x): return (_map3(_mt8, m, q, x), _map3(_mt9, m, q, x))
def mathieu_modsem2(m, q, x): return (_map3(_mt10, m, q, x), _map3(_mt11, m, q, x))
def mathieu_even_coef(m, q):
    if q < 0:
        raise ValueError("q >=0")
    if float(m) != float(int(m)) or m < 0:
        raise ValueError("m must be an integer >=0.")
    return np.array(MT.even_coef(int(m), float(q)))
def mathieu_odd_coef(m, q):
    if q < 0:
        raise ValueError("q >=0")
    if float(m) != float(int(m)) or m <= 0:
        raise ValueError("m must be an integer > 0")
    return np.array(MT.odd_coef(int(m), float(q)))

from . import _wright as WR
def wright_bessel(a, b, x): return _map3(WR.wright_bessel, a, b, x)
def log_wright_bessel(a, b, x): return _map3(WR.log_wright_bessel, a, b, x)

from . import _ellip_harm as EH
def _eh_call(h2: float, k2: float, n: float, p: float, s: float, sm: float, sn: float) -> float:
    return EH.ellip_harm(h2, k2, int(n), int(p), s, sm, sn)
def ellip_harm(h2, k2, n, p, s, signm = 1, signn = 1):
    if _is_arr(h2) or _is_arr(k2) or _is_arr(n) or _is_arr(p) or _is_arr(s):
        shp = _bshape(h2, k2, n, p, s)
        a0 = np.broadcast_to(np.asarray(h2), shp)
        a1 = np.broadcast_to(np.asarray(k2), shp)
        a2 = np.broadcast_to(np.asarray(n), shp)
        a3 = np.broadcast_to(np.asarray(p), shp)
        a4 = np.broadcast_to(np.asarray(s), shp)
        out = np.empty(shp, dtype=float)
        for idx in np.ndindex(shp):
            out[idx] = _eh_call(float(a0[idx]), float(a1[idx]), float(a2[idx]), float(a3[idx]), float(a4[idx]), float(signm), float(signn))
        return out
    return _eh_call(float(h2), float(k2), float(n), float(p), float(s), float(signm), float(signn))
def _eh2_call(h2: float, k2: float, n: float, p: float, s: float) -> float:
    return EH.ellipsoid(h2, k2, int(n), int(p), s)
def ellip_harm_2(h2, k2, n, p, s):
    if _is_arr(h2) or _is_arr(k2) or _is_arr(n) or _is_arr(p) or _is_arr(s):
        shp = _bshape(h2, k2, n, p, s)
        a0 = np.broadcast_to(np.asarray(h2), shp)
        a1 = np.broadcast_to(np.asarray(k2), shp)
        a2 = np.broadcast_to(np.asarray(n), shp)
        a3 = np.broadcast_to(np.asarray(p), shp)
        a4 = np.broadcast_to(np.asarray(s), shp)
        out = np.empty(shp, dtype=float)
        for idx in np.ndindex(shp):
            out[idx] = _eh2_call(float(a0[idx]), float(a1[idx]), float(a2[idx]), float(a3[idx]), float(a4[idx]))
        return out
    return _eh2_call(float(h2), float(k2), float(n), float(p), float(s))
def _en_call(h2: float, k2: float, n: float, p: float) -> float:
    return EH.ellipsoid_norm(h2, k2, int(n), int(p))
def ellip_normal(h2, k2, n, p):
    if _is_arr(h2) or _is_arr(k2) or _is_arr(n) or _is_arr(p):
        shp = _bshape(h2, k2, n, p)
        a0 = np.broadcast_to(np.asarray(h2), shp)
        a1 = np.broadcast_to(np.asarray(k2), shp)
        a2 = np.broadcast_to(np.asarray(n), shp)
        a3 = np.broadcast_to(np.asarray(p), shp)
        out = np.empty(shp, dtype=float)
        for idx in np.ndindex(shp):
            out[idx] = _en_call(float(a0[idx]), float(a1[idx]), float(a2[idx]), float(a3[idx]))
        return out
    return _en_call(float(h2), float(k2), float(n), float(p))

# noncentral inverses wrt df / nc (_ncinv)
from . import _ncinv as NI
def chndtridf(x, p, nc): return _map3(NI.chndtridf, x, p, nc)
def chndtrinc(x, df, p): return _map3(NI.chndtrinc, x, df, p)
def nctdtridf(p, nc, t): return _map3(NI.nctdtridf, p, nc, t)
def nctdtrinc(df, p, t): return _map3(NI.nctdtrinc, df, p, t)
def ncfdtridfd(dfn, p, nc, f): return _map4(NI.ncfdtridfd, dfn, p, nc, f)
def ncfdtridfn(p, dfd, nc, f): return _map4(NI.ncfdtridfn, p, dfd, nc, f)
def ncfdtrinc(dfn, dfd, p, f): return _map4(NI.ncfdtrinc, dfn, dfd, p, f)

# scaled Airy functions (real argument)
def _eai(x: float) -> float: return B.airye(x)[0]
def _eaip(x: float) -> float: return B.airye(x)[1]
def _ebi(x: float) -> float: return B.airye(x)[2]
def _ebip(x: float) -> float: return B.airye(x)[3]
def airye(z):
    """Exponentially scaled Airy functions (real argument)."""
    return (_map1(_eai, z), _map1(_eaip, z), _map1(_ebi, z), _map1(_ebip, z))

# zeros of Jn and Jn' sorted (_jdzo)
from . import _jdzo as JD
def jnjnp_zeros(nt):
    if K.floor(float(nt)) != float(nt) or nt > 1200:
        raise ValueError("Number must be integer <= 1200.")
    r = JD.jdzo(int(nt))
    return (np.array(r[0]), np.array(r[1]), np.array(r[2]), np.array(r[3]))

# parabolic cylinder Dn(z), complex z (_cpbdn)
from . import _cpbdn as CPB
def pbdn_seq(n, z):
    if K.floor(float(n)) != float(n):
        raise ValueError("n must be an integer.")
    n1 = 1 if abs(int(n)) <= 1 else int(n)
    t = CPB.cpbdn(n1, complex(z))
    return (np.array(t[0][:n1 + 1]), np.array(t[1][:n1 + 1]))

# spheroidal wave functions (_spheroidal)
from . import _spheroidal as SW
def pro_cv(m, n, c): return _map3(_pro_cv1, m, n, c)
def obl_cv(m, n, c): return _map3(_obl_cv1, m, n, c)
def _pro_cv1(m: float, n: float, c: float) -> float: return SW.segv_k(m, n, c, 1)
def _obl_cv1(m: float, n: float, c: float) -> float: return SW.segv_k(m, n, c, -1)
def _cv_seq_check(m, n, c):
    if K.floor(float(n)) != float(n) or K.floor(float(m)) != float(m):
        raise ValueError("Modes must be integers.")
    if n - m > 199:
        raise ValueError("Difference between n and m is too large.")
def pro_cv_seq(m, n, c):
    _cv_seq_check(m, n, c)
    return np.array(SW.cv_seq(int(m), int(n), float(c), 1))
def obl_cv_seq(m, n, c):
    _cv_seq_check(m, n, c)
    return np.array(SW.cv_seq(int(m), int(n), float(c), -1))

def _a_pro(m: float, n: float, c: float, x: float, i: int) -> float: return SW.ang1(m, n, c, 0.0, x, 1, True)[i]
def _a_obl(m: float, n: float, c: float, x: float, i: int) -> float: return SW.ang1(m, n, c, 0.0, x, -1, True)[i]
def _a_prov(m: float, n: float, c: float, cv: float, x: float, i: int) -> float: return SW.ang1(m, n, c, cv, x, 1, False)[i]
def _a_oblv(m: float, n: float, c: float, cv: float, x: float, i: int) -> float: return SW.ang1(m, n, c, cv, x, -1, False)[i]
def _r_nocv(m: float, n: float, c: float, x: float, kd: int, kf: int, i: int) -> float: return SW.rad(m, n, c, 0.0, x, kd, kf, True)[i]
def _r_cv(m: float, n: float, c: float, cv: float, x: float, kd: int, kf: int, i: int) -> float: return SW.rad(m, n, c, cv, x, kd, kf, False)[i]

def _pair4(f, m, n, c, x):
    return (_map4(lambda a, b, d, e: f(a, b, d, e, 0), m, n, c, x), _map4(lambda a, b, d, e: f(a, b, d, e, 1), m, n, c, x))
def pro_ang1(m, n, c, x): return _pair4(_a_pro, m, n, c, x)
def obl_ang1(m, n, c, x): return _pair4(_a_obl, m, n, c, x)
def pro_rad1(m, n, c, x): return _pair4(lambda a, b, d, e, i: _r_nocv(a, b, d, e, 1, 1, i), m, n, c, x)
def pro_rad2(m, n, c, x): return _pair4(lambda a, b, d, e, i: _r_nocv(a, b, d, e, 1, 2, i), m, n, c, x)
def obl_rad1(m, n, c, x): return _pair4(lambda a, b, d, e, i: _r_nocv(a, b, d, e, -1, 1, i), m, n, c, x)
def obl_rad2(m, n, c, x): return _pair4(lambda a, b, d, e, i: _r_nocv(a, b, d, e, -1, 2, i), m, n, c, x)

def _map5(f, a, b, c, d, e):
    if _is_arr(a) or _is_arr(b) or _is_arr(c) or _is_arr(d) or _is_arr(e):
        shp = _bshape(a, b, c, d, e)
        A = np.broadcast_to(np.asarray(a), shp)
        Bb = np.broadcast_to(np.asarray(b), shp)
        Cc = np.broadcast_to(np.asarray(c), shp)
        D = np.broadcast_to(np.asarray(d), shp)
        E = np.broadcast_to(np.asarray(e), shp)
        out = np.empty(shp, dtype=float)
        for idx in np.ndindex(shp):
            out[idx] = f(float(A[idx]), float(Bb[idx]), float(Cc[idx]), float(D[idx]), float(E[idx]))
        return out
    else:
        return f(float(a), float(b), float(c), float(d), float(e))
def _pair5(f, m, n, c, cv, x):
    return (_map5(lambda a, b, d, e, g: f(a, b, d, e, g, 0), m, n, c, cv, x), _map5(lambda a, b, d, e, g: f(a, b, d, e, g, 1), m, n, c, cv, x))
def pro_ang1_cv(m, n, c, cv, x): return _pair5(_a_prov, m, n, c, cv, x)
def obl_ang1_cv(m, n, c, cv, x): return _pair5(_a_oblv, m, n, c, cv, x)
def pro_rad1_cv(m, n, c, cv, x): return _pair5(lambda a, b, d, e, g, i: _r_cv(a, b, d, e, g, 1, 1, i), m, n, c, cv, x)
def pro_rad2_cv(m, n, c, cv, x): return _pair5(lambda a, b, d, e, g, i: _r_cv(a, b, d, e, g, 1, 2, i), m, n, c, cv, x)
def obl_rad1_cv(m, n, c, cv, x): return _pair5(lambda a, b, d, e, g, i: _r_cv(a, b, d, e, g, -1, 1, i), m, n, c, cv, x)
def obl_rad2_cv(m, n, c, cv, x): return _pair5(lambda a, b, d, e, g, i: _r_cv(a, b, d, e, g, -1, 2, i), m, n, c, cv, x)

class errstate:
    """No-op context manager (special functions never raise)."""
    kwargs: Dict[str, str]
    def __init__(self, **kwargs):
        self.kwargs = {}
    def __enter__(self):
        pass
    def __exit__(self, a, b, c):
        pass
