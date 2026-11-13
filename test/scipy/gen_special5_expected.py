# Ported from scipy 1.18.1 (BSD-3-Clause)
# Generates test/scipy/test_special_mathieu.py
import numpy as np, scipy.special as sp
from _gen_common import Gen
ns = {'np': np, 'sp': sp}
g = Gen(ns, tol=1e-9)
g.test('test_mathieu')
for m, q in [(0, 0.0), (0, 0.5), (1, 1.0), (2, 3.5), (3, 2.0), (4, 10.0), (5, 7.0), (7, 30.0), (9, 20.0), (15, 5.0), (3, -2.0), (2, -4.0), (14, 100.0)]:
    g.E(f'np.array([sp.mathieu_a({m}, {q!r})])')
    if m > 0:
        g.E(f'np.array([sp.mathieu_b({m}, {q!r})])')
for m, q, x in [(0, 0.5, 10.0), (1, 1.0, 45.0), (2, 3.5, 20.0), (3, 2.0, 20.0), (4, 10.0, 70.0), (5, 7.0, 33.0), (3, -2.0, 25.0), (2, -1.0, 30.0), (8, 12.0, 60.0)]:
    g.E(f'np.array(sp.mathieu_cem({m}, {q!r}, {x!r}))')
    if m > 0:
        g.E(f'np.array(sp.mathieu_sem({m}, {q!r}, {x!r}))')
for m, q, x in [(0, 0.5, 1.2), (1, 1.0, 0.4), (2, 1.5, 0.7), (3, 2.5, 0.4), (4, 8.0, 1.0), (5, 3.0, 2.0)]:
    g.E(f'np.array(sp.mathieu_modcem1({m}, {q!r}, {x!r}))')
    g.E(f'np.array(sp.mathieu_modcem2({m}, {q!r}, {x!r}))')
    if m > 0:
        g.E(f'np.array(sp.mathieu_modsem1({m}, {q!r}, {x!r}))')
        g.E(f'np.array(sp.mathieu_modsem2({m}, {q!r}, {x!r}))')
g.E('sp.mathieu_a(np.array([0.0, 1.0, 2.0, 3.0]), 2.0)')
g.E('sp.mathieu_even_coef(2, 1.5)')
g.E('sp.mathieu_even_coef(3, 0.5)')
g.E('sp.mathieu_odd_coef(3, 2.5)')
g.E('sp.mathieu_odd_coef(2, 4.0)')
g.write('test/scipy/test_special_mathieu.py', 'gen_special5_expected.py', ['import numpy as np', 'import scipy.special as sp'], [])
