# Generates the expected values embedded in test_optimize_global.py and
# test_linprog.py (scipy 1.18.1, numpy 2.5.4). Run with the reference venv.
import numpy as np
from scipy.optimize import (differential_evolution, basinhopping, dual_annealing,
                            direct, brute, shgo, rosen, linprog, milp,
                            LinearConstraint, Bounds)

def ack(x):
    return -20.0*np.exp(-0.2*np.sqrt(0.5*(x[0]**2+x[1]**2))) - np.exp(0.5*(np.cos(2*np.pi*x[0])+np.cos(2*np.pi*x[1]))) + np.e + 20
def styb(x):
    return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x)
def ras(x):
    return np.sum(x * x - 10 * np.cos(2 * np.pi * x)) + 10 * np.size(x)
def bhf(x):
    return np.cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] + 0.2) * x[0]
def vr(x):
    return np.sum(x**2, axis=0)

def p(name, r, extra=''):
    print(name, list(map(float, r.x)), float(r.fun), r.nit, r.nfev, extra)

for st in ['best1bin', 'rand1exp', 'randtobest1bin', 'best2exp', 'rand2bin', 'currenttobest1exp']:
    p('de_' + st, differential_evolution(ack, [(-5.0, 5.0)] * 2, strategy=st, rng=1, maxiter=200))
p('de_deferred', differential_evolution(rosen, [(0.0, 2.0)] * 3, rng=2, updating='deferred', polish=False))
p('de_seed_random', differential_evolution(rosen, [(0.0, 2.0)] * 3, seed=5, init='random', mutation=0.6))
p('de_vec', differential_evolution(vr, [(-3.0, 3.0)] * 2, rng=4, vectorized=True))
p('de_int', differential_evolution(rosen, [(-2.0, 2.0)] * 3, rng=6, integrality=[True, False, False], x0=[1.0, 1.0, 0.5]))
p('bh', basinhopping(bhf, [1.0, 1.0], niter=60, rng=7))
p('bh_ns', basinhopping(bhf, [1.0, 1.0], niter=40, seed=3, niter_success=5, T=0.5))
p('da', dual_annealing(ras, [(-5.12, 5.12)] * 3, rng=42, maxiter=200))
p('da_nols', dual_annealing(ras, [(-5.12, 5.12)] * 3, seed=1, maxiter=150, no_local_search=True))
for lb in [True, False]:
    r = direct(styb, [(-4.0, 4.0)] * 3, locally_biased=lb)
    p('direct_' + str(lb), r, r.status)
r = direct(rosen, [(-2.0, 2.0)] * 4, maxfun=3000, vol_tol=1e-30, eps=-1e-3); p('direct_eps', r, r.status)
r = direct(rosen, [(-2.0, 3.0)] * 2, maxiter=20, f_min=0.0, f_min_rtol=1e-2); p('direct_fmin', r, r.status)
x0, fv, g, J = brute(styb, (slice(-4.0, 4.0, 0.25), slice(-4.0, 4.0, 0.25)), full_output=True, finish=None)
print('brute_nofinish', list(x0), fv)
x0, fv, g, J = brute(styb, ((-4.0, 4.0), (-3.0, 3.0)), Ns=15, full_output=True)
print('brute_fmin', list(x0), fv)
r = shgo(rosen, [(0.0, 2.0)] * 3); print('shgo_rosen', list(r.x), r.fun, r.nit, r.nfev - r.nlfev, len(r.xl))
r = shgo(styb, [(-4.0, 4.0)] * 3, n=30, iters=2); print('shgo_styb', list(r.x), r.fun, r.nit, r.nfev - r.nlfev, len(r.xl), list(r.funl))
r = shgo(styb, [(-4.0, 4.0)] * 2, n=None, iters=3); print('shgo_none', list(r.x), r.fun, r.nit, r.nfev - r.nlfev, len(r.xl))
