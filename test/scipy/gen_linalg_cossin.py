# Ported from scipy 1.18.1 (BSD-3-Clause)
# Generates test/scipy/test_linalg_cossin.py (cossin via LAPACK ?orcsd / ?uncsd).
import numpy as np
from scipy.linalg import cossin, qr
from _gen_common import Gen
rng = np.random.default_rng(0)
X = qr(rng.standard_normal((6, 6)))[0]
Xc = qr(rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5)))[0]
ns = {'np': np, 'cossin': cossin, 'X': X, 'Xc': Xc}
g = Gen(ns, tol=1e-8)
g.test('test_cossin')
g.raw(f'    X = np.array({X.tolist()!r})')
g.raw('    Xc = np.array([[complex(a, b) for a, b in zip(r1, r2)] for r1, r2 in zip(' + repr(Xc.real.tolist()) + ', ' + repr(Xc.imag.tolist()) + ')])')
for p, q in [(2, 3), (3, 3), (4, 2), (1, 5)]:
    g.raw(f'    u, cs, vh = cossin(X, p={p}, q={q})')
    ns['u'], ns['cs'], ns['vh'] = cossin(X, p=p, q=q)
    g.E('cs.ravel()', 'cs.ravel()')
    g.raw('    assert closec((u @ cs @ vh).ravel(), X.ravel().tolist(), 1e-12)')
    g.raw(f'    (u1, u2), th, (v1h, v2h) = cossin(X, p={p}, q={q}, separate=True)')
    (ns['u1'], ns['u2']), ns['th'], (ns['v1h'], ns['v2h']) = cossin(X, p=p, q=q, separate=True)
    g.E('th', 'th')
    g.E('np.abs(u1).ravel()', 'np.abs(u1).ravel()')
    g.E('np.abs(v2h).ravel()', 'np.abs(v2h).ravel()')
g.raw('    u, cs, vh = cossin(X, p=2, q=3, swap_sign=True)')
ns['u'], ns['cs'], ns['vh'] = cossin(X, p=2, q=3, swap_sign=True)
g.E('cs.ravel()', 'cs.ravel()')
g.raw('    assert closec((u @ cs @ vh).ravel(), X.ravel().tolist(), 1e-12)')
g.raw('    u, cs, vh = cossin((X[:2, :3], X[:2, 3:], X[2:, :3], X[2:, 3:]))')
g.raw('    assert closec((u @ cs @ vh).ravel(), X.ravel().tolist(), 1e-12)')
g.test('test_cossin_complex')
g.raw('    Xc = np.array([[complex(a, b) for a, b in zip(r1, r2)] for r1, r2 in zip(' + repr(Xc.real.tolist()) + ', ' + repr(Xc.imag.tolist()) + ')])')
for p, q in [(2, 3), (3, 1)]:
    ns['u'], ns['cs'], ns['vh'] = cossin(Xc, p=p, q=q)
    g.raw(f'    u, cs, vh = cossin(Xc, p={p}, q={q})')
    g.E('np.abs(cs).ravel()', 'np.abs(cs).ravel()')
    g.raw('    R = u @ cs @ vh')
    g.raw('    assert closec(np.abs(R - Xc).ravel(), [0.0] * 25, 1e-10)')
g.write('test/scipy/test_linalg_cossin.py', 'gen_linalg_cossin.py', ['import numpy as np', 'from scipy.linalg import cossin'], [])
