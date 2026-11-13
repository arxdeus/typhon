# Ported from scipy 1.18.1 (BSD-3-Clause)
# Generates test/scipy/test_special_ellip_harm.py
import numpy as np, scipy.special as sp
from _gen_common import Gen
ns = {'np': np, 'sp': sp}
g = Gen(ns, tol=1e-8)
g.test('test_ellip_harm')
for n, p in [(0, 1), (1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (2, 5), (3, 1), (3, 4), (3, 7), (4, 2), (4, 6), (4, 9), (5, 11)]:
    g.E(f'np.array([sp.ellip_harm(5.0, 8.0, {n}, {p}, 2.5), sp.ellip_harm(5.0, 8.0, {n}, {p}, 3.1, -1, 1), sp.ellip_harm(0.5, 2.0, {n}, {p}, 2.2, 1, -1)])')
for n, p in [(0, 1), (1, 1), (1, 3), (2, 2), (3, 5), (4, 8)]:
    g.E(f'np.array([sp.ellip_harm_2(5.0, 8.0, {n}, {p}, 3.1), sp.ellip_harm_2(0.5, 2.0, {n}, {p}, 2.2)])', tol=1e-7)
for n, p in [(0, 1), (1, 1), (1, 2), (2, 3), (3, 4), (3, 7), (4, 5)]:
    g.E(f'np.array([sp.ellip_normal(5.0, 8.0, {n}, {p}), sp.ellip_normal(0.5, 2.0, {n}, {p})])', tol=1e-7)
g.E('sp.ellip_harm(5.0, 8.0, 2, 3, np.array([2.0, 3.0, 4.0]))')
g.E('np.array([sp.ellip_harm(5.0, 8.0, -1, 1, 2.0), sp.ellip_harm(5.0, 8.0, 2, 6, 2.0)])')
g.write('test/scipy/test_special_ellip_harm.py', 'gen_special7_expected.py', ['import numpy as np', 'import scipy.special as sp'], [])
