# Generates expected literals for test_integrate.py (scipy 1.18.1).
import warnings
warnings.simplefilter("ignore")
# Run: /Users/mind/.jcode/scratch/.venv/bin/python test/scipy/gen_integrate_expected.py
import math
import numpy as np
from scipy import integrate as I

def lit(v):
    return repr(np.asarray(v, dtype=float).tolist())


Q = {
 'poly': I.quad(lambda x: x*x, 0, 4),
 'expinf': I.quad(lambda x: math.exp(-x), 0, np.inf),
 'gauss': I.quad(lambda x: math.exp(-x*x), -np.inf, np.inf),
 'sqrt': I.quad(lambda x: 1/math.sqrt(x), 0, 1),
 'log': I.quad(lambda x: math.log(x), 0, 1),
 'args': I.quad(lambda x, a: math.sin(a*x), 0, 3, args=(2.0,)),
 'points': I.quad(lambda x: abs(x-0.3), 0, 1, points=[0.3]),
 'points2': I.quad(lambda x: 1.0 if x < 0.5 else 2.0, 0, 2, points=[0.5, 1.7, 5.0]),
 'neginf': I.quad(lambda x: 1/(1+x*x), -np.inf, 0),
 'flip': I.quad(lambda x: x**3, 2, -1),
 'limit3': I.quad(lambda x: math.cos(50*x), 0, 1, limit=3),
 'tol': I.quad(lambda x: math.exp(x), 0, 1, epsabs=1e-3, epsrel=1e-3),
 'dbl': I.dblquad(lambda y, x: x*y, 0, 2, 0, 1),
 'dblf': I.dblquad(lambda y, x: x*y, 0, 1, lambda x: 0.0, lambda x: 1-2*x),
 'dbla': I.dblquad(lambda y, x, k: k*x + y, 0, 1, 0, 2, args=(3.0,)),
 'tpl': I.tplquad(lambda z, y, x: x*y*z, 0, 1, 0, 2, 0, 3),
 'tplf': I.tplquad(lambda z, y, x: x + y + z, 0, 1, lambda x: 0.0, lambda x: x, lambda x, y: 0.0, lambda x, y: x*y),
}
print('QUAD = {')
for k, v in Q.items():
    print(f'    {k!r}: ({v[0]!r}, {v[1]!r}),')
print('}')
W = {
 'cauchy': I.quad(lambda x: 1.0 / (1.0 + x*x), -1.0, 2.0, weight='cauchy', wvar=0.3),
 'alg': I.quad(lambda x: math.exp(x), 0.0, 1.0, weight='alg', wvar=(-0.5, 0.3)),
 'alga': I.quad(lambda x: 1.0 + x, 0.0, 1.0, weight='alg-loga', wvar=(0.2, -0.4)),
 'algb': I.quad(lambda x: 1.0 + x, 0.0, 1.0, weight='alg-logb', wvar=(0.0, 0.5)),
 'alglog': I.quad(lambda x: math.cos(x), 0.0, 2.0, weight='alg-log', wvar=(0.5, 0.5)),
 'cos': I.quad(lambda x: math.exp(-x), 0.0, 10.0, weight='cos', wvar=20.0),
 'sin': I.quad(lambda x: x*x, 0.0, 3.0, weight='sin', wvar=1.5),
 'cosinf': I.quad(lambda x: math.exp(-x), 0.0, np.inf, weight='cos', wvar=2.0),
 'sininf': I.quad(lambda x: 1.0/(1.0+x*x), 1.0, np.inf, weight='sin', wvar=3.0),
 'sinninf': I.quad(lambda x: math.exp(x), -np.inf, 0.0, weight='sin', wvar=2.0),
}
print('WQUAD = {')
for k, v in W.items():
    print(f'    {k!r}: ({v[0]!r}, {v[1]!r}),')
print('}')
r = I.quad(lambda x: math.cos(50*x), 0, 1, limit=3, full_output=1)
print('FULL1 =', (r[2]['neval'], r[2]['last'], r[2]['iord'].tolist(), r[2]['alist'].tolist(), r[2]['blist'].tolist(), r[2]['rlist'].tolist(), r[2]['elist'].tolist(), r[3][:45]))
r = I.quad(lambda x: x*x, 0, 4, full_output=1, points=[1.0])
print('FULLP =', (r[2]['neval'], r[2]['last'], r[2]['iord'].tolist()[:2], r[2]['level'].tolist()[:2], r[2]['pts'].tolist(), r[2]['ndin'].tolist()))
r = I.quad(lambda x: math.exp(-x), 0.0, np.inf, weight='cos', wvar=2.0, full_output=1)
print('FULLF =', (r[2]['lst'], r[2]['neval'], r[2]['rslst'].tolist()[:5], r[2]['erlst'].tolist()[:5]))
r = I.quad(lambda x: x*x, 0.0, 3.0, weight='sin', wvar=50.0, full_output=1)
print('FULLO =', (r[0], r[2]['neval'], r[2]['last'], r[2]['momcom'], r[2]['chebmo'][:3, 0].tolist()))
r = I.quad(lambda x: complex(math.cos(x), math.sin(2*x)), 0.0, 1.0, complex_func=True)
print('CPLX =', (r[0].real, r[0].imag, r[1].real, r[1].imag))
r = I.quad_vec(lambda x: np.array([x, x*x, math.sin(x)]), 0.0, 2.0, full_output=True)
print('QV1 =', (r[0].tolist(), r[1], r[2].neval, r[2].status, r[2].intervals.shape[0]))
print('QV2 =', I.quad_vec(lambda x: math.exp(-x*x), -np.inf, np.inf))
r = I.quad_vec(lambda x: np.array([math.exp(-x), 1.0/(1.0+x*x)]), 0.0, np.inf, norm='max', full_output=True)
print('QV3 =', (r[0].tolist(), r[1], r[2].neval))
r = I.quad_vec(lambda x: math.sqrt(x), 0.0, 1.0, quadrature='trapezoid', epsrel=1e-6, full_output=True)
print('QV4 =', (r[0], r[1], r[2].neval))
r = I.quad_vec(lambda x: np.array([abs(x - 0.3), x]), 0.0, 1.0, points=[0.3], quadrature='gk21', full_output=True)
print('QV5 =', (r[0].tolist(), r[2].neval))
print('NQ2 =', I.nquad(lambda x, y: x*y, [[0.0, 1.0], [0.0, 2.0]]))
r = I.nquad(lambda x, y, z: x + y*z, [[0.0, 1.0], lambda z: (0.0, z), [0.0, 2.0]], full_output=True)
print('NQ3 =', (r[0], r[1], r[2]['neval']))
print('NQA =', I.nquad(lambda x, y, a: a*x + y, [[0.0, 1.0], [0.0, 1.0]], args=(2.0,), opts={'epsrel': 1e-10}))
print('NC =', ([I.newton_cotes(n)[0].tolist() for n in (1, 2, 4)], [I.newton_cotes(n)[1] for n in (1, 2, 4, 7)]))
an, B = I.newton_cotes(np.array([0, 1, 2.5, 3]))
print('NCX =', (an.tolist(), float(B)))
an, B = I.newton_cotes(16)
print('NC16 =', (an.tolist(), float(B)))
print('FIXED =', [float(I.fixed_quad(np.cos, 0.0, 1.0, n=n)[0]) for n in (1, 2, 4, 5, 10, 30)])
print('FIXED_ARGS =', I.fixed_quad(lambda x, p: x**p, 0, 2, args=(3,), n=2)[0])

y = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 7.0])
x = np.array([0.0, 0.5, 1.3, 2.0, 2.2, 3.5])
y2 = np.arange(24.0).reshape(2, 3, 4) ** 1.5
print('TRAP =', lit([I.trapezoid(y), I.trapezoid(y, x), I.trapezoid(y, dx=0.3)]))
print('TRAP2 =', lit(I.trapezoid(y2, axis=1)))
print('SIMP =', lit([I.simpson(y), I.simpson(y, x=x), I.simpson(y[:5]), I.simpson(y[:5], x=x[:5]),
                 I.simpson(y[:2], x=x[:2]), I.simpson(y, dx=0.25), I.simpson(y[:4])]))
print('SIMP2 =', lit(I.simpson(y2, axis=0)))
print('SIMP3 =', lit(I.simpson(y2, axis=2)))
print('CTRAP =', lit(I.cumulative_trapezoid(y, x)))
print('CTRAP0 =', lit(I.cumulative_trapezoid(y, dx=2.0, initial=0)))
print('CTRAP2 =', lit(I.cumulative_trapezoid(y2, axis=1, initial=0)))
print('CSIMP =', lit(I.cumulative_simpson(y)))
print('CSIMPX =', lit(I.cumulative_simpson(y, x=x, initial=1.5)))
print('CSIMP5 =', lit(I.cumulative_simpson(y[:5], dx=0.5)))
print('CSIMP2 =', lit(I.cumulative_simpson(y[:2], initial=0)))
print('CSIMPA =', lit(I.cumulative_simpson(y2, axis=1)))
r = np.exp(-np.linspace(0, 2, 17))
print('ROMB =', lit([I.romb(r, dx=2/16), I.romb(r[:9]), I.romb(r[:2])]))
print('ROMB2 =', lit(I.romb(np.vstack([r, r**2]), dx=0.125)))

