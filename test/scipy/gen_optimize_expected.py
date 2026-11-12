# Generates the expected literals for test/scipy/test_optimize.py.
# Run with the reference venv: python test/scipy/gen_optimize_expected.py
import numpy as np
from scipy import optimize as so

def L(v):
    return repr(np.asarray(v, float).tolist())

def f1(x):
    return (x - 2.0) * x * (x + 2.0) ** 2

def g(x):
    return x ** 3 - 2 * x - 5

def F(x):
    return np.array([x[0] + 0.5 * (x[0] - x[1]) ** 3 - 1.0, 0.5 * (x[1] - x[0]) ** 3 + x[1]])

def res(x):
    return np.array([10 * (x[1] - x[0] ** 2), (1 - x[0])])

def quad(x, a):
    return float(np.sum((x - a) ** 2)) + 0.5 * x[0] * x[1]

def model(x, a, b, c):
    return a * np.exp(-b * x) + c

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
print('# minimize on rosen: (method, x, fun, nit, nfev, njev)')
for m in ['Nelder-Mead', 'BFGS', 'CG', 'Powell', 'L-BFGS-B']:
    r = so.minimize(so.rosen, x0, method=m)
    print(f"    ('{m}', {L(r.x)}, {float(r.fun)!r}, {r.nit}, {r.nfev}, {r.get('njev', 0)}),")
r = so.minimize(so.rosen, x0, jac=so.rosen_der)
print('bfgs_jac', L(r.x), float(r.fun), r.nit, r.nfev, r.njev)
r = so.minimize(so.rosen, x0, method='CG', jac=so.rosen_der)
print('cg_jac', L(r.x), float(r.fun), r.nit, r.nfev, r.njev)
r = so.minimize(so.rosen, x0, method='L-BFGS-B', jac=so.rosen_der)
print('lbfgsb_jac', L(r.x), float(r.fun), r.nit, r.nfev, r.njev)
a = np.array([1.0, -2.0])
r = so.minimize(quad, np.zeros(2), args=(a,), tol=1e-10)
print('quad_args', L(r.x), float(r.fun), r.nit)
r = so.minimize(so.rosen, np.array([3.0, 3.0]), method='L-BFGS-B', bounds=[(-1, 2), (0.5, 1.5)])
print('lbb', L(r.x), float(r.fun), r.nit, r.nfev, r.message)
r = so.minimize(so.rosen, np.array([3.0, 3.0]), bounds=[(None, 2), (0.5, None)])
print('lbb2', L(r.x), float(r.fun), r.nit, r.nfev)
r = so.minimize(so.rosen, np.array([1.5, 1.5]), method='Nelder-Mead', bounds=[(-1, 2), (0.5, 1.5)])
print('nmb', L(r.x), float(r.fun), r.nit, r.nfev)
r = so.minimize(so.rosen, np.array([1.5, 1.5]), method='Powell', bounds=[(-1, 2), (0.5, 1.5)])
print('pwb', L(r.x), float(r.fun), r.nit, r.nfev)
r = so.minimize(so.rosen, x0, method='Nelder-Mead', options={'maxiter': 50})
print('nm_maxiter', L(r.x), float(r.fun), r.nit, r.nfev, r.status, r.success)
r = so.minimize(so.rosen, x0, method='Nelder-Mead', options={'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-8})
print('nm_adaptive', L(r.x), float(r.fun), r.nit, r.nfev)
r = so.minimize(so.rosen, x0, method='BFGS', options={'gtol': 1e-8, 'maxiter': 10})
print('bfgs_maxiter', float(r.fun), r.nit, r.status, r.success)

print('# scalar')
r = so.minimize_scalar(f1)
print('brent', float(r.x), float(r.fun), r.nit, r.nfev)
r = so.minimize_scalar(f1, bracket=(0, 1, 2))
print('brent3', float(r.x), float(r.fun), r.nit, r.nfev)
r = so.minimize_scalar(f1, bounds=(-3, -1), method='bounded')
print('bounded', float(r.x), float(r.fun), r.nfev)
r = so.minimize_scalar(f1, method='golden')
print('golden', float(r.x), float(r.fun), r.nit, r.nfev)
print('roots', so.brentq(g, 2, 3), so.brenth(g, 2, 3), so.bisect(g, 2, 3), so.ridder(g, 2, 3))
print('brentq_args', so.brentq(lambda x, c: x * x - c, 0, 3, args=(2.0,)))
print('newton', so.newton(g, 2.0), so.newton(g, 2.0, fprime=lambda x: 3 * x ** 2 - 2),
      so.newton(g, 2.0, fprime=lambda x: 3 * x ** 2 - 2, fprime2=lambda x: 6 * x))
s = so.root_scalar(g, bracket=[2, 3]); print('rs_brentq', s.root, s.iterations, s.function_calls)
s = so.root_scalar(g, x0=2.0, fprime=lambda x: 3 * x ** 2 - 2); print('rs_newton', s.root, s.iterations, s.function_calls)
s = so.root_scalar(g, x0=2.0, x1=3.0); print('rs_secant', s.root, s.iterations, s.function_calls)
s = so.root_scalar(g, method='bisect', bracket=[2, 3], xtol=1e-6); print('rs_bisect', s.root, s.iterations, s.function_calls)

print('# systems / least squares')
print('fsolve', L(so.fsolve(F, [0.0, 0.0])))
rr = so.root(F, [0.0, 0.0]); print('root', L(rr.x), rr.nfev, rr.status)
ls = so.least_squares(res, np.array([2.0, 2.0])); print('trf', L(ls.x), ls.cost, ls.nfev, ls.njev, ls.status, ls.optimality)
ls = so.least_squares(res, np.array([2.0, 2.0]), method='lm'); print('lm', L(ls.x), ls.cost, ls.nfev, ls.status)
ls = so.least_squares(res, np.array([2.0, 2.0]), bounds=([-np.inf, 1.5], np.inf)); print('trfb', L(ls.x), ls.cost, ls.nfev, ls.status, ls.active_mask.tolist())
xd = np.linspace(0, 4, 50)
yd = 2.5 * np.exp(-1.3 * xd) + 0.5 + 0.01 * np.sin(7 * xd)
popt, pcov = so.curve_fit(model, xd, yd); print('cf', L(popt), L(pcov))
popt, pcov = so.curve_fit(model, xd, yd, p0=[2.0, 1.0, 0.0], sigma=np.full(50, 0.1), absolute_sigma=True); print('cf_sigma', L(popt), L(pcov))
popt, pcov = so.curve_fit(model, xd, yd, bounds=(0, [3., 1., 0.5])); print('cfb', L(popt), L(pcov))

print('# lsap / numdiff')
C = np.array([[4., 1, 3], [2, 0, 5], [3, 2, 2]])
print('lsap', so.linear_sum_assignment(C))
print('lsap_max', so.linear_sum_assignment(np.array([[4., 1, 3, 9], [2, 0, 5, 1]]).T, maximize=True))
rng = np.random.default_rng(0)
C2 = rng.random((6, 8))
r, c = so.linear_sum_assignment(C2); print('lsap_rect', r.tolist(), c.tolist(), C2[r, c].sum())
print('C2', repr(C2.tolist()))
x = np.array([1.5, 1.9])
print('fprime', L(so.approx_fprime(x, so.rosen)), so.check_grad(so.rosen, so.rosen_der, x))
rr = so.root(F, [0.0, 0.0], method='lm'); print('root_lm', L(rr.x), rr.nfev, rr.status)
