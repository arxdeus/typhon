# Generates the expected literals for test/scipy/test_optimize_legacy.py.
# Run with the reference venv: python test/scipy/gen_optimize_legacy.py
import numpy as np
from scipy import optimize as so

def L(v):
    return repr(np.asarray(v, float).tolist())

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
for name, r in [('fmin', so.fmin(so.rosen, x0, full_output=True, disp=False)),
                ('powell', so.fmin_powell(so.rosen, x0, full_output=True, disp=False)),
                ('bfgs', so.fmin_bfgs(so.rosen, x0, fprime=so.rosen_der, full_output=True, disp=False)),
                ('cg', so.fmin_cg(so.rosen, x0, fprime=so.rosen_der, full_output=True, disp=False)),
                ('ncg', so.fmin_ncg(so.rosen, x0, so.rosen_der, fhess=so.rosen_hess, full_output=True, disp=False))]:
    print(name, L(r[0]), float(r[1]), [int(v) for v in r[2:] if np.ndim(v) == 0])
x, f, d = so.fmin_l_bfgs_b(so.rosen, x0, fprime=so.rosen_der)
print('lbfgsb', L(x), float(f), d['funcalls'], d['nit'], d['warnflag'])
x, f, d = so.fmin_l_bfgs_b(so.rosen, x0, approx_grad=True, bounds=[(1.0, 2.0)] * 5)
print('lbfgsb_b', L(x), float(f), d['funcalls'], d['nit'], d['warnflag'])
x, nf, rc = so.fmin_tnc(so.rosen, x0, fprime=so.rosen_der, disp=0)
print('tnc', L(x), nf, rc)
def eqc(x): return np.array([x[0] + x[1] - 1.0])
def iqc(x): return np.array([x[0] - 0.2])
r = so.fmin_slsqp(lambda x: x[0] ** 2 + x[1] ** 2, [1.0, 1.0], f_eqcons=eqc, f_ieqcons=iqc, full_output=True, iprint=0)
print('slsqp', L(r[0]), float(r[1]), r[2], r[3])
x = so.fmin_cobyla(lambda x: x[0] ** 2 + x[1] ** 2, [1.0, 1.0], [lambda x: x[0] + x[1] - 1.0], disp=0)
print('cobyla', L(x))
def res(p, t, y): return p[0] * np.exp(-p[1] * t) - y
t = np.linspace(0, 3, 8); y = 2.5 * np.exp(-1.3 * t) + np.array([0.01, -0.02, 0.015, -0.01, 0.005, 0.02, -0.015, 0.0])
print('t', L(t)); print('y', L(y))
r = so.leastsq(res, [1.0, 1.0], args=(t, y), full_output=True)
print('leastsq', L(r[0]), L(r[1]), L(r[2]['fvec']), r[2]['nfev'], r[4])
r = so.leastsq(res, [1.0, 1.0], args=(t, y))
print('leastsq1', L(r[0]), r[1])
r = so.isotonic_regression([1, 3, 2, 4, 3, 5, 5, 4, 6.0], weights=[1, 2, 1, 1, 3, 1, 1, 2, 1])
print('iso', L(r.x), L(r.weights), r.blocks.tolist())
r = so.isotonic_regression([1, 3, 2, 4, 3, 5, 5, 4, 6.0], increasing=False)
print('iso_dec', L(r.x), L(r.weights), r.blocks.tolist())
print('rhp', L(so.rosen_hess_prod(x0, np.array([1., 2, 3, 4, 5]))))
def f(x): return x ** 3 - 2 * x - 5
for k in (1, 2, 3):
    r = so.toms748(f, 2, 3, k=k, full_output=True)
    print('toms', k, repr(float(r[0])), r[1].iterations, r[1].function_calls)
r = so.toms748(lambda x: np.cos(x) - x, 0, 1, full_output=True, xtol=1e-14)
print('toms2', repr(float(r[0])), r[1].iterations, r[1].function_calls)
