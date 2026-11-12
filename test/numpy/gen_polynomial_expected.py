# Generates test/numpy/test_polynomial.py from BODY by evaluating it with
# real numpy (2.5.4): run with the reference venv python, redirect to the test.
import numpy as np
import numpy.polynomial as npp
from numpy.polynomial import polynomial as P, chebyshev as C, legendre as G, \
    hermite as H, hermite_e as E, laguerre as L
from numpy.polynomial import Polynomial, Chebyshev, Legendre, Hermite, HermiteE, Laguerre

BODY = r'''
FAMS = [P, C, G, H, E, L]
c1 = [1.0, -2.0, 3.5, 0.25]
c2 = [0.5, 1.5, -1.0]
xs = np.array([-0.7, 0.0, 0.3, 1.1, 2.5])
R('polyval', P.polyval(xs, c1))
R('chebval', C.chebval(xs, c1))
R('legval', G.legval(xs, c1))
R('hermval', H.hermval(xs, c1))
R('hermeval', E.hermeval(xs, c1))
R('lagval', L.lagval(xs, c1))
R('polyval_s', P.polyval(1.3, c1))
R('chebval_s', C.chebval(0.4, c2))
R('polyadd', P.polyadd(c1, c2))
R('chebsub', C.chebsub(c1, c2))
R('polymul', P.polymul(c1, c2))
R('chebmul', C.chebmul(c1, c2))
R('legmul', G.legmul(c1, c2))
R('hermmul', H.hermmul(c1, c2))
R('hermemul', E.hermemul(c1, c2))
R('lagmul', L.lagmul(c1, c2))
R('polymulx', P.polymulx(c1))
R('chebmulx', C.chebmulx(c1))
R('legmulx', G.legmulx(c1))
R('hermmulx', H.hermmulx(c1))
R('hermemulx', E.hermemulx(c1))
R('lagmulx', L.lagmulx(c1))
R('polydiv_q', P.polydiv(c1, c2)[0])
R('polydiv_r', P.polydiv(c1, c2)[1])
R('chebdiv_q', C.chebdiv(c1, c2)[0])
R('chebdiv_r', C.chebdiv(c1, c2)[1])
R('lagdiv_q', L.lagdiv(c1, c2)[0])
R('hermdiv_r', H.hermdiv(c1, c2)[1])
R('polypow', P.polypow(c2, 3))
R('chebpow', C.chebpow(c2, 3))
R('legpow', G.legpow(c2, 2))
R('polyder', P.polyder(c1))
R('polyder2', P.polyder(c1, 2, 3.0))
R('chebder', C.chebder(c1))
R('chebder2', C.chebder(c1, 2))
R('legder', G.legder(c1))
R('hermder', H.hermder(c1))
R('hermeder', E.hermeder(c1))
R('lagder', L.lagder(c1, 2))
R('polyint', P.polyint(c1))
R('polyint2', P.polyint(c1, 2, [1.0, 2.0], -1.0, 0.5))
R('chebint', C.chebint(c1, 1, [3.0], 0.5))
R('legint', G.legint(c1, 2))
R('hermint', H.hermint(c1, 1, [1.0], 1.0))
R('hermeint', E.hermeint(c1))
R('lagint', L.lagint(c1, 1, [2.0]))
R('polyvander', P.polyvander(xs, 3))
R('chebvander', C.chebvander(xs, 4))
R('legvander', G.legvander(xs, 3))
R('hermvander', H.hermvander(xs, 3))
R('hermevander', E.hermevander(xs, 3))
R('lagvander', L.lagvander(xs, 3))
R('polyfromroots', P.polyfromroots([1.0, -2.0, 0.5]))
R('chebfromroots', C.chebfromroots([1.0, -2.0, 0.5]))
R('legfromroots', G.legfromroots([0.3, -0.4]))
R('hermfromroots', H.hermfromroots([1.0, 2.0]))
R('hermefromroots', E.hermefromroots([1.0, 2.0]))
R('lagfromroots', L.lagfromroots([1.0, 2.0, 3.0]))
R('polycompanion', P.polycompanion(c1))
R('chebcompanion', C.chebcompanion(c1))
R('legcompanion', G.legcompanion(c1))
R('hermcompanion', H.hermcompanion(c1))
R('hermecompanion', E.hermecompanion(c1))
R('lagcompanion', L.lagcompanion(c1))
R('polyroots', np.sort(np.real(P.polyroots([-6.0, 11.0, -6.0, 1.0]))))
R('chebroots', np.sort(np.real(C.chebroots([0.1, 0.2, 0.3, 0.4]))))
R('legroots', np.sort(np.real(G.legroots([0.1, -0.5, 0.3, 1.0]))))
R('hermroots', np.sort(np.real(H.hermroots([-1.0, 0.5, 0.25]))))
R('hermeroots', np.sort(np.real(E.hermeroots([-1.0, 0.5, 0.25]))))
R('lagroots', np.sort(np.real(L.lagroots([1.0, 2.0, 3.0]))))
R('polyroots_c', np.imag(P.polyroots([1.0, 0.0, 1.0])))
R('leggauss_x', G.leggauss(5)[0])
R('leggauss_w', G.leggauss(5)[1])
R('hermgauss_x', H.hermgauss(6)[0])
R('hermgauss_w', H.hermgauss(6)[1])
R('hermegauss_x', E.hermegauss(4)[0])
R('hermegauss_w', E.hermegauss(4)[1])
R('laggauss_x', L.laggauss(5)[0])
R('laggauss_w', L.laggauss(5)[1])
R('chebgauss_x', C.chebgauss(4)[0])
R('chebgauss_w', C.chebgauss(4)[1])
R('chebpts1', C.chebpts1(5))
R('chebpts2', C.chebpts2(5))
R('chebinterpolate', C.chebinterpolate(np.exp, 6))
R('chebweight', C.chebweight(np.array([0.0, 0.5])))
R('lagweight', L.lagweight(np.array([0.0, 0.5])))
R('cheb2poly', C.cheb2poly(c1))
R('poly2cheb', C.poly2cheb(c1))
R('leg2poly', G.leg2poly(c1))
R('poly2leg', G.poly2leg(c1))
R('herm2poly', H.herm2poly(c1))
R('poly2herm', H.poly2herm(c1))
R('herme2poly', E.herme2poly(c1))
R('poly2herme', E.poly2herme(c1))
R('lag2poly', L.lag2poly(c1))
R('poly2lag', L.poly2lag(c1))
R('chebline', C.chebline(3.0, 2.0))
R('hermline', H.hermline(3.0, 2.0))
R('lagline', L.lagline(3.0, 2.0))
R('polytrim', P.polytrim([1.0, 1e-10, 0.0, 1e-12], 1e-9))
fx = np.linspace(0.0, 3.0, 12)
fy = np.sin(fx) + 0.1 * fx * fx
R('polyfit', P.polyfit(fx, fy, 3))
R('chebfit', C.chebfit(fx, fy, 4))
R('legfit_w', G.legfit(fx, fy, 2, w=np.linspace(1.0, 2.0, 12)))
R('lagfit_deglist', L.lagfit(fx, fy, [0, 2, 3]))
R('hermfit', H.hermfit(fx, fy, 3))
R('mapdomain', npp.polyutils.mapdomain(np.array([0.0, 1.5, 3.0]), [0.0, 3.0], [-1.0, 1.0]))
R('mapparms', np.array(npp.polyutils.mapparms([0.0, 3.0], [-1.0, 1.0])))
# classes
p = Polynomial.fit(fx, fy, 3)
R('Pfit_coef', p.coef)
R('Pfit_domain', p.domain)
R('Pfit_call', p(np.array([0.5, 2.0])))
R('Pfit_convert', p.convert().coef)
R('Pfit_deriv', p.deriv().coef)
R('Pfit_deriv_conv', p.deriv().convert().coef)
R('Pfit_integ', p.integ(1, [1.0], 0.5).coef)
R('Pfit_integ_val', np.array([p.integ(1, [1.0], 0.5)(0.5), p.integ(2)(2.0)]))
R('Pfit_roots', np.sort(np.real(Polynomial.fit([0.0, 1.0, 2.0, 3.0], [-1.0, 1.0, -1.0, 3.0], 2).roots())))
ch = Chebyshev.fit(fx, fy, 5)
R('Cfit_coef', ch.coef)
R('Cfit_call', ch(1.7))
R('Cfit_toP', ch.convert(kind=Polynomial).coef)
R('Cfit_toP_dom', ch.convert(domain=[0.0, 3.0], kind=Polynomial).coef)
R('Cfit_cutdeg', ch.cutdeg(2).coef)
lg = Legendre([1.0, 2.0, 3.0], domain=[0.0, 2.0])
R('Leg_call', lg(np.array([0.0, 0.5, 2.0])))
R('Leg_roots', np.sort(np.real(lg.roots())))
R('Leg_mul', (lg * lg).coef)
R('Leg_add_s', (lg + 2.0).coef)
R('Leg_rsub', (2.0 - lg).coef)
R('Leg_pow', (lg ** 2).coef)
R('Leg_floordiv', (lg // Legendre([1.0, 1.0], domain=[0.0, 2.0])).coef)
R('Leg_mod', (lg % Legendre([1.0, 1.0], domain=[0.0, 2.0])).coef)
R('Leg_truediv', (lg / 2.0).coef)
R('Leg_neg', (-lg).coef)
R('Leg_linspace', lg.linspace(5)[1])
R('Herm_fromroots', Hermite.fromroots([1.0, 2.0, 4.0]).coef)
R('Herm_fromroots_dom', Hermite.fromroots([1.0, 2.0], domain=[0.0, 4.0]).coef)
R('HermE_identity', HermiteE.identity().coef)
R('Lag_identity_dom', Laguerre.identity(domain=[1.0, 3.0]).coef)
R('Lag_basis', Laguerre.basis(3)(np.array([0.5, 1.0])))
R('Lag_basis_conv', Laguerre.basis(3).convert(kind=Polynomial).coef)
R('Cheb_basis_dom', Chebyshev.basis(2, domain=[0.0, 4.0])(np.array([1.0, 3.0])))
R('P_trim', Polynomial([1.0, 2.0, 1e-12, 0.0]).trim(1e-9).coef)
R('P_degree', np.array([float(Polynomial([1.0, 2.0, 3.0]).degree())]))
R('P_mul_P', (Polynomial([1.0, 1.0]) * Polynomial([1.0, -1.0])).coef)
R('P_rmul', (3.0 * Polynomial([1.0, 1.0])).coef)
R('Cheb_cast', Chebyshev.cast(Polynomial([0.0, 0.0, 1.0])).coef)
R('He_deriv_dom', HermiteE([1.0, 2.0, 3.0], domain=[0.0, 2.0]).deriv(1).coef)
R('Lag_integ', Laguerre([1.0, 2.0, 3.0]).integ(1, [0.0], 0.5).coef)
'''

def fmt(v):
    a = np.atleast_1d(np.asarray(v, dtype=float))
    return repr(a.ravel().tolist())

out = []
def R(name, v):
    out.append(f"    ({name!r}, {fmt(v)}),")

exec(BODY)

print('''# Generated by test/numpy/gen_polynomial_expected.py (numpy 2.5.4). Do not edit.
import numpy as np
import numpy.polynomial as npp
from numpy.polynomial import polynomial as P, chebyshev as C, legendre as G, \\
    hermite as H, hermite_e as E, laguerre as L
from numpy.polynomial import Polynomial, Chebyshev, Legendre, Hermite, HermiteE, Laguerre

EXPECTED = [''')
print('\n'.join(out))
print(''']

_idx = [0]
_fail = [0]

def R(name: str, v):
    exp_name, exp = EXPECTED[_idx[0]]
    _idx[0] += 1
    a = np.atleast_1d(np.asarray(v, dtype=float)).ravel()
    if exp_name != name or a.size != len(exp):
        print('TEST FAILED', name, 'size', a.size, len(exp))
        _fail[0] += 1
        return
    for i in range(a.size):
        if abs(a[i] - exp[i]) > 1e-9 + 1e-9 * abs(exp[i]):
            print('TEST FAILED', name, i, a[i], exp[i])
            _fail[0] += 1
            return
''')
print(BODY)
print('''
def _str_checks():
    p = Polynomial([1.0, -2.0, 3.5])
    s = str(p)
    if s != '1.0 - 2.0·x + 3.5·x²':
        print('TEST FAILED str', s)
    c = Chebyshev([1.0, -2.0, 3.5])
    if str(c) != '1.0 - 2.0·T₁(x) + 3.5·T₂(x)':
        print('TEST FAILED str cheb', str(c))
    if str(HermiteE([1.0, 2.0])) != '1.0 + 2.0·He₁(x)':
        print('TEST FAILED str herme', str(HermiteE([1.0, 2.0])))
    q = Polynomial([1.0, 2.0], domain=[0.0, 4.0])
    if str(q) != '1.0 + 2.0·(-1.0 + 0.5x)':
        print('TEST FAILED str dom', str(q))
    if not (Polynomial([1.0, 2.0]) == Polynomial([1.0, 2.0])) or Polynomial([1.0]) == Polynomial([2.0]):
        print('TEST FAILED eq')
    if len(Polynomial([1.0, 2.0, 3.0])) != 3:
        print('TEST FAILED len')
    try:
        Polynomial([1.0]) + Polynomial([1.0], domain=[0.0, 1.0])
        print('TEST FAILED domain mismatch not raised')
    except TypeError:
        pass
    try:
        H.hermgauss(0)
        print('TEST FAILED gauss deg')
    except ValueError:
        pass

_str_checks()
assert _idx[0] == len(EXPECTED)
''')
