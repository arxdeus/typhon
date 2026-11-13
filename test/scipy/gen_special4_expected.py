# Ported from scipy 1.18.1 (BSD-3-Clause)
# Generates test/scipy/test_special_orthopoly.py (orthopoly1d factories).
import numpy as np, scipy.special as sp
from _gen_common import Gen
ns = {'np': np, 'sp': sp}
g = Gen(ns, tol=1e-9)
g.test('test_orthopoly')
for e in ['legendre(4)', 'legendre(3, monic=True)', 'chebyt(5)', 'chebyu(4)', 'chebyc(4)', 'chebys(3)', 'hermite(4)', 'hermitenorm(4)', 'laguerre(3)',
          'genlaguerre(3, 0.5)', 'jacobi(3, 0.5, 0.2)', 'gegenbauer(3, 1.5)', 'sh_legendre(3)', 'sh_chebyt(3)', 'sh_chebyu(3)', 'sh_jacobi(3, 2.0, 1.5)',
          'legendre(0)', 'hermite(0)', 'chebyt(0)', 'chebyc(0)']:
    g.raw(f'    p = sp.{e}')
    ns['p'] = eval('sp.' + e)
    p = ns['p']
    g.E('p.coeffs', 'p.coeffs', tol=1e-8)
    g.E('p(0.3)', 'p(0.3)')
    g.E('p(np.array([-0.5, 0.1, 0.9]))', 'p(np.array([-0.5, 0.1, 0.9]))')
    g.E('np.array([p.normcoef, p.limits[0] if abs(p.limits[0]) < 1e300 else 0.0, p.order])', 'np.array([p.normcoef, p.limits[0] if abs(p.limits[0]) < 1e300 else 0.0, p.order])')
g.E('np.array([1.0])', 'np.array([1.0])')
g.write('test/scipy/test_special_orthopoly.py', 'gen_special4_expected.py', ['import numpy as np', 'import scipy.special as sp'], [])
