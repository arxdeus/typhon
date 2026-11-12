# Generates expected literals for test_integrate.py (scipy 1.18.1).
import warnings
warnings.simplefilter("ignore")
# Run: /Users/mind/.jcode/scratch/.venv/bin/python test/scipy/gen_integrate_expected.py
import math
import numpy as np
from scipy import integrate as I

def lit(v):
    return repr(np.asarray(v, dtype=float).tolist())

def decay(t, y): return -0.5 * y
def osc(t, y): return [y[1], -y[0]]
def lv(t, z, a, b, c, d): return [a*z[0] - b*z[0]*z[1], -c*z[1] + d*z[0]*z[1]]
def cannon(t, y): return [y[1], -0.5]
def hit(t, y): return y[0]
hit.terminal = True; hit.direction = -1
def apex(t, y): return y[1]

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

print('IVP0 = [')
for m in ['RK45', 'RK23', 'DOP853']:
    s = I.solve_ivp(decay, [0, 10], [2, 4, 8], method=m)
    print(f'    ({m!r}, {s.nfev}, {lit(s.t)}, {lit(s.y)}),')
print(']')
print('IVP1 = [')
for m in ['RK45', 'RK23', 'DOP853']:
    s = I.solve_ivp(osc, (0.0, 20.0), [1.0, 0.0], method=m, rtol=1e-8, atol=1e-10)
    print(f'    ({m!r}, {s.nfev}, {lit(s.t[-3:])}, {lit(s.y[:, -1])}),')
print(']')
s = I.solve_ivp(lv, [0, 15], [10, 5], args=(1.5, 1, 3, 1), dense_output=True)
print('LV =', (s.nfev, s.t.size, s.y[:, -1].tolist(), s.sol(7.3).tolist(), s.sol([0.5, 14.0]).tolist()))
s = I.solve_ivp(decay, [0, 10], [2.0], t_eval=[0, 1, 2.5, 10])
print('TEVAL =', (s.nfev, s.t.tolist(), s.y.tolist()))
s = I.solve_ivp(cannon, [0, 100], [0, 10], events=(hit, apex))
print('EVENTS =', (s.status, s.nfev, float(s.t[-1]), s.y[:, -1].tolist(), s.t_events[0].tolist(), s.t_events[1].tolist(), s.y_events[1].tolist()))
s = I.solve_ivp(decay, [10, 0], [2.0], method='DOP853', dense_output=True, t_eval=[9.0, 3.0])
print('BACK =', (s.nfev, s.t.tolist(), s.y.tolist(), s.sol(4.0).tolist()))
s = I.solve_ivp(osc, [0, 5], [1.0, 0.0], method='RK23', dense_output=True, max_step=0.1, atol=[1e-6, 1e-8], first_step=0.01)
print('OPTS =', (s.nfev, s.t.size, s.y[:, -1].tolist(), s.sol(2.345).tolist()))
s = I.solve_ivp(osc, [0, 5], [1.0, 0.0], method='DOP853', dense_output=True)
print('DOPDENSE =', (s.nfev, s.sol(1.234).tolist(), s.sol(4.9).tolist()))
s = I.solve_ivp(decay, [0, 0], [1.0])
print('ZERO =', (s.nfev, s.t.tolist(), s.y.tolist(), s.status))
