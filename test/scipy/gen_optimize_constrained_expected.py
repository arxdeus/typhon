# Generates the expected literals for test/scipy/test_optimize_constrained.py
# and test/scipy/test_optimize_local2.py.
# Run with the reference venv: python test/scipy/gen_optimize_constrained_expected.py
import numpy as np
from scipy import optimize as so
from scipy.optimize import LinearConstraint, NonlinearConstraint, Bounds

def L(v):
    return repr(np.asarray(v, float).tolist())

X0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

def hp(x, p):
    return so.rosen_hess(x) @ p

print('# ---- local2: (method, x, fun, nit, nfev, njev, nhev)')
for m in ['Newton-CG', 'trust-ncg', 'trust-krylov', 'trust-exact', 'dogleg']:
    r = so.minimize(so.rosen, X0, method=m, jac=so.rosen_der, hess=so.rosen_hess)
    print(f"    ('{m}', {L(r.x)}, {float(r.fun)!r}, {r.nit}, {r.nfev}, {r.njev}, {r.get('nhev', 0)}, {r.status}),")
for m in ['Newton-CG', 'trust-ncg', 'trust-krylov']:
    r = so.minimize(so.rosen, X0, method=m, jac=so.rosen_der, hessp=hp)
    print(f"    hessp ('{m}', {L(r.x)}, {r.nit}, {r.nfev}, {r.njev}, {r.nhev}),")
r = so.minimize(so.rosen, X0, method='trust-ncg', jac=so.rosen_der, hess=so.BFGS())
print('tncg_bfgs', L(r.x), r.nit, r.nfev, r.njev)
r = so.minimize(so.rosen, X0, method='trust-ncg', jac=so.rosen_der, hess=so.SR1())
print('tncg_sr1', L(r.x), r.nit, r.nfev, r.njev)
r = so.minimize(so.rosen, X0, method='Newton-CG', jac=so.rosen_der, hess='2-point')
print('ncg_2p', L(r.x), r.nit, r.nfev, r.njev, r.nhev)
x12 = np.linspace(-1.2, 1.5, 12)
for m in ['trust-krylov', 'trust-exact']:
    r = so.minimize(so.rosen, x12, method=m, jac=so.rosen_der, hess=so.rosen_hess)
    print(m, '12', L(r.x), r.nit, r.nfev, r.njev, r.nhev)
r = so.minimize(so.rosen, X0, method='TNC', jac=so.rosen_der)
print('tnc', L(r.x), float(r.fun), r.nit, r.nfev, r.status)
r = so.minimize(so.rosen, np.array([3.0, 3.0]), method='TNC', jac=so.rosen_der, bounds=[(-1.0, 2.0), (0.5, 1.5)])
print('tncb', L(r.x), float(r.fun), r.nit, r.nfev, r.status)

t = np.linspace(0.0, 10.0, 30)
y = 3.0 * np.exp(-0.4 * t) + 0.5 + 0.05 * np.sin(7.0 * t)
y[5] += 2.0
y[17] -= 1.5
def res(p):
    return p[0] * np.exp(-p[1] * t) + p[2] - y
def jac(p):
    J = np.empty((t.size, 3))
    J[:, 0] = np.exp(-p[1] * t)
    J[:, 1] = -p[0] * t * np.exp(-p[1] * t)
    J[:, 2] = 1.0
    return J
p0 = np.array([1.0, 0.1, 0.0])
pb = np.array([1.0, 0.1, 0.7])
B = ([0.0, 0.0, 0.6], [10.0, 1.0, 2.0])
print('# least_squares: (method, loss, x, cost, nfev, njev, status)')
for meth in ['trf', 'dogbox']:
    for loss in ['linear', 'soft_l1', 'huber', 'cauchy', 'arctan']:
        r = so.least_squares(res, p0, jac=jac, method=meth, loss=loss, f_scale=0.3)
        print(f"    ('{meth}', '{loss}', {L(r.x)}, {r.cost!r}, {r.nfev}, {r.njev}, {r.status}),")
for meth in ['trf', 'dogbox']:
    for loss in ['linear', 'soft_l1', 'huber']:
        r = so.least_squares(res, pb, jac=jac, method=meth, loss=loss, bounds=B)
        print(f"    bnd ('{meth}', '{loss}', {L(r.x)}, {r.cost!r}, {r.nfev}, {L(r.active_mask)}),")
r = so.least_squares(res, p0, jac=jac, tr_solver='lsmr')
print('lsmr', L(r.x), r.cost, r.nfev)
r = so.least_squares(res, pb, jac=jac, tr_solver='lsmr', bounds=B)
print('lsmrb', L(r.x), r.cost, r.nfev)
A = np.array([[0.8, 0.3, -0.2, 0.5], [0.1, 0.9, 0.4, -0.3], [-0.5, 0.2, 1.1, 0.6], [0.3, -0.7, 0.2, 0.9], [1.2, 0.4, 0.5, -0.1], [0.2, 0.3, -0.6, 0.4]])
b = np.array([1.0, -2.0, 0.5, 3.0, -1.0, 2.0])
for meth in ['trf', 'bvls']:
    r = so.lsq_linear(A, b, bounds=(-0.5, 1.0), method=meth)
    print('lsqlin', meth, L(r.x), r.cost, r.nit, r.status)
x, rn = so.nnls(A, b)
print('nnls', L(x), rn)

print('# ---- constrained')
def f2(x):
    return (x[0] - 1.0) ** 2 + (x[1] - 2.5) ** 2
cons3 = ({'type': 'ineq', 'fun': lambda x: x[0] - 2 * x[1] + 2},
         {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
         {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
r = so.minimize(f2, np.array([2.0, 0.0]), method='SLSQP', bounds=((0.0, None), (0.0, None)), constraints=cons3)
print('slsqp_a', L(r.x), float(r.fun), r.nit, r.nfev, r.njev)
def hs(x):
    return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]
def hsg(x):
    return np.array([x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]), x[0] * x[3], x[0] * x[3] + 1.0, x[0] * (x[0] + x[1] + x[2])])
c1 = {'type': 'ineq', 'fun': lambda x: np.array([x[0] * x[1] * x[2] * x[3] - 25.0])}
c2 = {'type': 'eq', 'fun': lambda x: np.array([x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 - 40.0])}
HSB = Bounds(np.ones(4), 5 * np.ones(4))
r = so.minimize(hs, np.array([1.0, 5.0, 5.0, 1.0]), jac=hsg, bounds=HSB, constraints=(c1, c2))
print('hs071_slsqp', L(r.x), float(r.fun), r.nit, r.nfev, r.njev)
r = so.minimize(hs, np.array([1.0, 5.0, 5.0, 1.0]), method='trust-constr', jac=hsg, bounds=HSB, constraints=(c1, c2))
print('hs071_tc', L(r.x), float(r.fun), r.nit, r.nfev, r.status)
r = so.minimize(hs, np.array([1.0, 5.0, 5.0, 1.0]), method='COBYLA', bounds=HSB, constraints=(c1, c2))
print('hs071_cobyla', L(r.x), float(r.fun), r.nfev, r.status)
lc = LinearConstraint([[1.0, 2.0], [2.0, 1.0]], [-np.inf, 1.0], [1.0, 1.0])
def cf(x):
    return np.array([x[0] ** 2 + x[1], x[0] ** 2 - x[1]])
def cj(x):
    return np.array([[2 * x[0], 1.0], [2 * x[0], -1.0]])
def ch(x, v):
    return v[0] * np.array([[2.0, 0.0], [0.0, 0.0]]) + v[1] * np.array([[2.0, 0.0], [0.0, 0.0]])
nc = NonlinearConstraint(cf, -np.inf, 1.0, jac=cj, hess=ch)
RB = Bounds([0.0, -0.5], [1.0, 2.0])
x0 = np.array([0.5, 0.0])
r = so.minimize(so.rosen, x0, method='trust-constr', jac=so.rosen_der, hess=so.rosen_hess, constraints=(lc, nc), bounds=RB)
print('tc_doc', L(r.x), float(r.fun), r.nit, r.nfev, r.njev, r.cg_niter, r.status)
r = so.minimize(so.rosen, x0, method='trust-constr', jac=so.rosen_der, hess=so.SR1(), constraints=(lc, NonlinearConstraint(cf, -np.inf, 1.0)), bounds=RB)
print('tc_sr1', L(r.x), float(r.fun), r.nit, r.nfev, r.njev, r.status)
r = so.minimize(so.rosen, x0, method='trust-constr', jac=so.rosen_der, hess=so.rosen_hess)
print('tc_unc', L(r.x), r.nit, r.nfev, r.cg_niter)
ceq = {'type': 'eq', 'fun': lambda x: np.array([x[0] + x[1] - 1.0])}
r = so.minimize(so.rosen, x0, method='trust-constr', jac=so.rosen_der, hess=so.rosen_hess, constraints=(ceq,))
print('tc_eq', L(r.x), r.nit, r.nfev)
r = so.minimize(so.rosen, x0, method='SLSQP', jac=so.rosen_der, constraints=(lc, nc), bounds=RB)
print('slsqp_doc', L(r.x), float(r.fun), r.nit, r.nfev, r.njev)
r = so.minimize(so.rosen, x0, method='COBYLA', constraints=(lc, nc), bounds=RB)
print('cobyla_doc', L(r.x), float(r.fun), r.nfev)
r = so.minimize(f2, np.array([2.0, 0.0]), method='COBYLA', constraints=cons3)
print('cobyla_a', L(r.x), float(r.fun), r.nfev, r.status)
r = so.minimize(f2, np.array([2.0, 0.0]), method='trust-constr', constraints=cons3)
print('tc_a', L(r.x), float(r.fun), r.nit, r.status)
