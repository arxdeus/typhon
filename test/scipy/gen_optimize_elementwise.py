# Ported from scipy 1.18.1 (BSD-3-Clause)
# Generates test/scipy/test_optimize_elementwise.py (find_root, find_minimum, bracket_root, bracket_minimum).
import numpy as np
from scipy.optimize import elementwise as ew
from _gen_common import Gen
ns = {'np': np}
g = Gen(ns, tol=1e-9)
def flatlit(a):
    return '[' + ', '.join('float("nan")' if v != v else repr(float(v)) for v in a) + ']'
def F(r):  # flatten helper
    return np.concatenate([np.atleast_1d(np.asarray(v, dtype=float)).ravel() for v in r])
g.test('test_find_root')
g.raw('    def f(x, c):\n        return x**3 - 2*x - c')
for c in [5.0, np.array([3.0, 4.0, 5.0, -1.0])]:
    r = ew.find_root(lambda x, c: x**3 - 2*x - c, (0.0, 3.0), args=(c,))
    cs = repr(c.tolist()) if isinstance(c, np.ndarray) else repr(c)
    ca = f'np.array({cs})' if isinstance(c, np.ndarray) else cs
    g.raw(f'    r = find_root(f, (0.0, 3.0), args=({ca},))')
    flat = F([r.x, r.f_x, r.nit, r.nfev, r.status, r.bracket[0], r.bracket[1], r.f_bracket[0], r.f_bracket[1]])
    g.raw(f'    assert closec(np.concatenate((np.atleast_1d(r.x), np.atleast_1d(r.f_x), np.atleast_1d(r.nit).astype(float), np.atleast_1d(r.nfev).astype(float), np.atleast_1d(r.status).astype(float), np.atleast_1d(r.bracket[0]), np.atleast_1d(r.bracket[1]), np.atleast_1d(r.f_bracket[0]), np.atleast_1d(r.f_bracket[1]))), {flatlit(flat)}, 1e-9)')
r = ew.find_root(lambda x: np.cos(x) - x, (0.0, 1.0), tolerances={'xatol': 1e-3, 'frtol': 1e-3}, maxiter=20)
g.raw("    r = find_root(lambda x: np.cos(x) - x, (0.0, 1.0), tolerances={'xatol': 1e-3, 'frtol': 1e-3}, maxiter=20)")
g.raw(f"    assert closec(np.array([r.x, float(r.nit), float(r.nfev)]), {flatlit([r.x, r.nit, r.nfev])}, 1e-9)")
r = ew.find_root(lambda x: x*x + 1.0, (-1.0, 2.0))
g.raw("    r = find_root(lambda x: x*x + 1.0, (-1.0, 2.0))")
g.raw(f"    assert closec(np.array([float(r.status), float(r.success)]), {flatlit([r.status, r.success])}, 1e-12)")
g.test('test_find_minimum')
g.raw('    def f(x, c):\n        return (x - c)**2 + 1')
for c in [1.5, np.array([1.0, 2.0, -3.0])]:
    r = ew.find_minimum(lambda x, c: (x - c)**2 + 1, (-5.0, 0.0, 9.0), args=(c,))
    cs = repr(c.tolist()) if isinstance(c, np.ndarray) else repr(c)
    ca = f'np.array({cs})' if isinstance(c, np.ndarray) else cs
    g.raw(f'    r = find_minimum(f, (-5.0, 0.0, 9.0), args=({ca},))')
    flat = F([r.x, r.f_x, r.nit, r.nfev, r.status, r.bracket[0], r.bracket[1], r.bracket[2], r.f_bracket[0], r.f_bracket[2]])
    g.raw(f'    assert closec(np.concatenate((np.atleast_1d(r.x), np.atleast_1d(r.f_x), np.atleast_1d(r.nit).astype(float), np.atleast_1d(r.nfev).astype(float), np.atleast_1d(r.status).astype(float), np.atleast_1d(r.bracket[0]), np.atleast_1d(r.bracket[1]), np.atleast_1d(r.bracket[2]), np.atleast_1d(r.f_bracket[0]), np.atleast_1d(r.f_bracket[2]))), {flatlit(flat)}, 1e-9)')
g.test('test_brackets')
for c in [np.array([5.0, -7.0, 100.0])]:
    r = ew.bracket_root(lambda x, c: x - c, 0.0, args=(c,))
    g.raw(f'    r = bracket_root(lambda x, c: x - c, 0.0, args=(np.array({c.tolist()!r}),))')
    flat = F([r.bracket[0], r.bracket[1], r.f_bracket[0], r.f_bracket[1], r.nit, r.nfev, r.status])
    g.raw(f'    assert closec(np.concatenate((r.bracket[0], r.bracket[1], r.f_bracket[0], r.f_bracket[1], r.nit.astype(float), r.nfev.astype(float), r.status.astype(float))), {flatlit(flat)}, 1e-9)')
r = ew.bracket_root(lambda x: x*x - 5.0, 0.0, xmin=-1.0, xmax=4.0)
g.raw("    r = bracket_root(lambda x: x*x - 5.0, 0.0, xmin=-1.0, xmax=4.0)")
flat = F([r.bracket[0], r.bracket[1], r.f_bracket[0], r.f_bracket[1], r.nit, r.nfev, r.status])
g.raw(f"    assert closec(np.array([r.bracket[0], r.bracket[1], r.f_bracket[0], r.f_bracket[1], float(r.nit), float(r.nfev), float(r.status)]), {flatlit(flat)}, 1e-9)")
r = ew.bracket_root(lambda x: x*x + 5.0, 0.0, xmax=10.0, factor=1.5, maxiter=6)
g.raw("    r = bracket_root(lambda x: x*x + 5.0, 0.0, xmax=10.0, factor=1.5, maxiter=6)")
flat = F([r.bracket[0], r.bracket[1], r.nit, r.nfev, r.status])
g.raw(f"    assert closec(np.array([r.bracket[0], r.bracket[1], float(r.nit), float(r.nfev), float(r.status)]), {flatlit(flat)}, 1e-9)")
r = ew.bracket_minimum(lambda x, c: (x - c)**2, 0.0, args=(np.array([3.0, -9.0]),))
g.raw("    r = bracket_minimum(lambda x, c: (x - c)**2, 0.0, args=(np.array([3.0, -9.0]),))")
flat = F([r.bracket[0], r.bracket[1], r.bracket[2], r.f_bracket[0], r.f_bracket[1], r.f_bracket[2], r.nit, r.nfev, r.status])
g.raw(f"    assert closec(np.concatenate((r.bracket[0], r.bracket[1], r.bracket[2], r.f_bracket[0], r.f_bracket[1], r.f_bracket[2], r.nit.astype(float), r.nfev.astype(float), r.status.astype(float))), {flatlit(flat)}, 1e-9)")
r = ew.bracket_minimum(lambda x: np.abs(x - 7.0), 0.0, xmax=5.0)
g.raw("    r = bracket_minimum(lambda x: np.abs(x - 7.0), 0.0, xmax=5.0)")
flat = F([r.bracket[0], r.bracket[1], r.bracket[2], r.nit, r.nfev, r.status])
g.raw(f"    assert closec(np.array([r.bracket[0], r.bracket[1], r.bracket[2], float(r.nit), float(r.nfev), float(r.status)]), {flatlit(flat)}, 1e-9)")
g.write('test/scipy/test_optimize_elementwise.py', 'gen_optimize_elementwise.py', ['import numpy as np', 'from scipy.optimize import find_root, find_minimum, bracket_root, bracket_minimum'], [])
