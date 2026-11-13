# Ported from scipy 1.18.1 (BSD-3-Clause)
# Generates test/scipy/test_special_wright.py
import numpy as np, scipy.special as sp
from _gen_common import Gen
ns = {'np': np, 'sp': sp}
g = Gen(ns, tol=1e-9)
g.test('test_wright')
cases = [(0.0, 1.5, 2.0), (1e-6, 2.0, 0.5), (1e-5, 3.0, 50.0), (1e-4, 1.0, 5.0), (5e-4, 0.0005, 3.0), (1e-3, 20.0, 8.0),
         (0.1, 1.0, 0.5), (0.5, 1.0, 1.5), (1.0, 1.0, 1.9), (2.0, 0.5, 1.0), (0.3, 2.0, 100.0), (1.5, 0.5, 30.0),
         (0.5, 1.0, 10.0), (1.0, 2.0, 5.0), (3.0, 1.0, 20.0), (0.2, 3.0, 40.0), (5.0, 1.0, 10.0), (7.0, 2.0, 1e5),
         (12.0, 1.0, 1e3), (1.0, 1.0, 0.0), (2.5, 0.5, 3.0), (0.7, 5.0, 12.0), (4.0, 3.0, 200.0), (0.9, 9.0, 4.0)]
for a, b, x in cases:
    g.E(f'np.array([sp.wright_bessel({a!r}, {b!r}, {x!r})])', tol=1e-8)
    if x > 0:
        g.E(f'np.array([sp.log_wright_bessel({a!r}, {b!r}, {x!r})])', tol=1e-8)
g.E('sp.wright_bessel(np.array([0.5, 1.0, 2.0]), 1.0, np.array([1.0, 3.0, 5.0]))', tol=1e-8)
g.write('test/scipy/test_special_wright.py', 'gen_special6_expected.py', ['import numpy as np', 'import scipy.special as sp'], [])
