# Generates the expected literals for test/scipy/test_optimize_root.py.
# Run with the reference venv: python test/scipy/gen_optimize_root.py
import numpy as np
from scipy import optimize as so

def L(v):
    return repr(np.asarray(v, float).tolist())

def F(x):
    return np.array([x[0] + 0.5 * (x[0] - x[1]) ** 3 - 1.0, 0.5 * (x[1] - x[0]) ** 3 + x[1]])

def G(x):
    return np.array([x[0] ** 2 + x[1] ** 2 - 4.0, x[0] - x[1] - 1.0, 0.0 * x[2] + x[2] - 0.5][:3])

def H(x):
    n = x.size
    y = np.empty(n)
    for i in range(n):
        l = x[i - 1] if i > 0 else 0.0
        r = x[i + 1] if i < n - 1 else 0.0
        y[i] = 3 * x[i] - l - r + 0.2 * x[i] ** 3 - 1.0
    return y

x0 = np.array([1.0, 1.0])
h0 = np.zeros(6)
print('# (method, x, success, nit, nfev) on F from [1,1]')
for m in ['broyden1', 'broyden2', 'anderson', 'diagbroyden', 'df-sane']:
    r = so.root(F, x0, method=m)
    print(f"    ('{m}', {L(r.x)}, {bool(r.success)}, {r.nit}, {r.nfev}),")
print('# H problem')
for m in ['broyden1', 'broyden2', 'anderson', 'diagbroyden', 'krylov', 'df-sane']:
    r = so.root(H, h0, method=m)
    print(f"    ('{m}', {L(r.x)}, {bool(r.success)}, {r.nit}, {r.nfev}),")
for m in ['broyden1', 'broyden2', 'anderson', 'diagbroyden', 'krylov']:
    r = so.root(H, h0, method=m, tol=1e-9)
    print('tol', m, L(r.x), r.nit)
r = so.root(H, h0, method='broyden1', options=dict(line_search=None, maxiter=3))
print('maxiter', L(r.x), r.success, r.status, r.nit, r.message)
print('lm_neg', L(so.linearmixing(lambda x: -H(x), h0, alpha=0.3, maxiter=500)))
print('lm', L(so.linearmixing(H, h0, alpha=-0.3, maxiter=500, line_search=None)))
print('em', L(so.excitingmixing(lambda x: -H(x), h0, alpha=0.2, maxiter=500, line_search=None)))
for meth in ['lgmres', 'gmres', 'bicgstab', 'cgs', 'minres', 'tfqmr']:
    print('nk', meth, L(so.newton_krylov(H, h0, method=meth)))
print('b1svd', L(so.broyden1(H, h0, reduction_method='svd', max_rank=3)))
print('b1simple', L(so.broyden1(H, h0, reduction_method='simple', max_rank=2)))
print('b1restart', L(so.broyden1(H, h0, max_rank=2)))
print('b1wolfe', L(so.broyden1(H, h0, line_search='wolfe')))
print('b1nols', L(so.broyden1(H, h0, line_search=None)))
r = so.root(H, h0, method='df-sane', options=dict(line_search='cheng'))
print('cheng', L(r.x), r.nit, r.nfev)
print('fp', L(so.fixed_point(lambda x, a: np.sqrt(a / (x + 3)), [1.2, 1.3], args=(np.array([10, 12.]),))))
print('fp1', float(so.fixed_point(lambda x: np.cos(x), 0.5)), float(so.fixed_point(lambda x: np.cos(x), 0.5, method='iteration')))
def f(x): return x[0] ** 2 + 3 * x[1] ** 2
def g(x): return np.array([2 * x[0], 6 * x[1]])
xk = np.array([1.8, 1.7]); pk = -g(xk)
a, fc, gc, nf, of, ns = so.line_search(f, g, xk, pk)
print('ls', a, nf, of, ns)
