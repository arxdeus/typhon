# Ported from scipy 1.18.1 (BSD-3-Clause)
# Generates test/scipy/test_integrate_tanhsinh.py (tanhsinh, nsum).
import numpy as np
from scipy.integrate import tanhsinh, nsum
from _gen_common import Gen
ns = {'np': np}
g = Gen(ns, tol=1e-9)
def fl(a): return '[' + ', '.join('float("nan")' if v != v else repr(float(v)) for v in np.atleast_1d(a).ravel()) + ']'
def chk(call, r, fields):
    g.raw(f'    r = {call}')
    for f in fields:
        v = getattr(r, f)
        tol = 1e-6 if f == 'error' else 1e-9
        g.raw(f'    assert closec(np.atleast_1d(r.{f}).astype(float), {fl(v)}, {tol!r})')
g.test('test_tanhsinh')
F = ['integral', 'error', 'nfev', 'status', 'maxlevel']
chk('tanhsinh(lambda x: np.exp(-x**2 / 2), -np.inf, np.inf)', tanhsinh(lambda x: np.exp(-x**2 / 2), -np.inf, np.inf), F)
chk('tanhsinh(lambda x, c: x ** c, 0.0, 1.0, args=(np.array([-0.5, 2.0, 3.5]),))', tanhsinh(lambda x, c: x ** c, 0.0, 1.0, args=(np.array([-0.5, 2.0, 3.5]),)), F)
chk('tanhsinh(np.cos, 3.0, 0.5)', tanhsinh(np.cos, 3.0, 0.5), F)
chk('tanhsinh(lambda x: 1.0 / (1.0 + x * x), 0.0, np.inf, atol=1e-10, rtol=1e-12)', tanhsinh(lambda x: 1.0 / (1.0 + x * x), 0.0, np.inf, atol=1e-10, rtol=1e-12), F)
chk('tanhsinh(lambda x: np.exp(x), -np.inf, 2.0)', tanhsinh(lambda x: np.exp(x), -np.inf, 2.0), F)
chk('tanhsinh(lambda x: 1 / x, 0.0, 1.0, maxlevel=5)', tanhsinh(lambda x: 1 / x, 0.0, 1.0, maxlevel=5), ['integral', 'nfev', 'status'])
chk('tanhsinh(lambda x: x * x, 1.0, 1.0)', tanhsinh(lambda x: x * x, 1.0, 1.0), ['integral', 'status', 'nfev'])
chk('tanhsinh(lambda x: np.sin(x) ** 2, np.array([0.0, 1.0]), np.array([3.0, 2.0]), minlevel=0, maxlevel=6)', tanhsinh(lambda x: np.sin(x) ** 2, np.array([0.0, 1.0]), np.array([3.0, 2.0]), minlevel=0, maxlevel=6), F)
g.test('test_nsum')
F2 = ['sum', 'error', 'nfev', 'status']
def n(call, **kw):
    r = eval(call.replace('nsum(', 'nsum(', 1), {'np': np, 'nsum': nsum})
    chk(call, r, F2)
n('nsum(lambda k: 1.0 / k ** 2, 1.0, np.inf)')
n('nsum(lambda k: 1.0 / k ** 2, 1.0, 100.0)')
n('nsum(lambda k, x: x ** k / 2.0 ** k, 0.0, np.inf, args=(np.array([0.3, 0.5]),))')
n('nsum(lambda k: np.exp(-np.abs(k)), -np.inf, np.inf)')
n('nsum(lambda k: np.exp(k), -np.inf, 5.0)')
n('nsum(lambda k: k, 1.0, 10.0, step=3.0)')
n('nsum(lambda k: 1.0 / k ** 3, 1.0, 5000000.0, maxterms=1000)')
n('nsum(lambda k: k, 5.0, 1.0)')
g.write('test/scipy/test_integrate_tanhsinh.py', 'gen_integrate_tanhsinh.py', ['import numpy as np', 'from scipy.integrate import tanhsinh, nsum'], [])
