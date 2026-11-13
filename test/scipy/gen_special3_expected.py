# Ported from scipy 1.18.1 (BSD-3-Clause)
# Generates test/scipy/test_special_seq.py and test_special_orthopoly.py.
# Run: /Users/mind/.jcode/scratch/.venv/bin/python test/scipy/gen_special3_expected.py
import numpy as np, scipy.special as sp
from _gen_common import Gen
ns = {'np': np, 'sp': sp}

g = Gen(ns, tol=1e-9)
g.test('test_zeros')
for f in ['ai_zeros(4)', 'bi_zeros(4)']:
    for i in range(4):
        g.E(f'sp.{f}[{i}]')
for i in range(2):
    g.E(f'sp.fresnel_zeros(3)[{i}]')
g.E('sp.fresnelc_zeros(3)'); g.E('sp.fresnels_zeros(3)'); g.E('sp.erf_zeros(3)')
for f in ['y0_zeros', 'y1_zeros', 'y1p_zeros']:
    for i in range(2):
        g.E(f'sp.{f}(3)[{i}]')
        g.E(f'sp.{f}(3, True)[{i}]', tol=1e-8)
g.test('test_bessel_derivs')
for e in ['sp.jvp(1.5, 2.0)', 'sp.jvp(1.5, 2.0, 2)', 'sp.yvp(1.5, 2.0)', 'sp.yvp(0.5, 3.0, 3)', 'sp.ivp(0.5, 1.2, 2)', 'sp.kvp(0.5, 1.2, 3)', 'sp.kvp(1.5, 2.2)',
          'sp.hankel1(0.5, 2.0)', 'sp.hankel2(1.5, 3.0)', 'sp.hankel1e(1.0, 2.0)', 'sp.hankel2e(1.0, 2.0)', 'sp.h1vp(1.0, 2.0)', 'sp.h2vp(0.5, 2.0, 2)',
          'sp.jvp(np.array([0.5, 1.0]), 2.0, 2)']:
    g.E(e)
g.test('test_riccati_lamb')
for i in range(2):
    g.E(f'sp.riccati_jn(4, 2.5)[{i}]'); g.E(f'sp.riccati_yn(4, 2.5)[{i}]'); g.E(f'sp.riccati_jn(0, 2.5)[{i}]')
    g.E(f'sp.lmbda(3, 2.5)[{i}]'); g.E(f'sp.lmbda(3.5, 15.0)[{i}]'); g.E(f'sp.lmbda(5, 20.0)[{i}]'); g.E(f'sp.lmbda(2.5, 3.0)[{i}]')
g.test('test_legendre_q')
for i in range(2):
    g.E(f'sp.lqn(4, 0.3)[{i}]'); g.E(f'sp.lqn(3, 2.5)[{i}]'); g.E(f'sp.lqn(0, 0.5)[{i}]')
    g.E(f'sp.lqmn(2, 3, 0.4)[{i}]'); g.E(f'sp.lqmn(2, 3, 1.7)[{i}]'); g.E(f'sp.lqmn(3, 2, 1.01)[{i}]')
g.test('test_numbers')
g.E('sp.bernoulli(8)'); g.E('sp.euler(8)'); g.E('sp.bernoulli(1)'); g.E('sp.euler(0)')
g.raw('    assert sp.stirling2(10, 4, exact=True) == 34105')
g.E('sp.stirling2(np.array([6, 8]), np.array([3, 4]))')
g.raw('    assert sp.factorial2(9, exact=True) == 945')
g.raw('    assert sp.factorialk(10, 3, exact=True) == 280')
g.E('sp.factorial2(np.array([5, 6]))'); g.E('sp.factorialk(np.array([10, 7]), 3)')
g.test('test_parcyl_seq')
for i in range(2):
    g.E(f'sp.pbdv_seq(3.3, 1.2)[{i}]'); g.E(f'sp.pbvv_seq(2.2, 0.7)[{i}]')
g.test('test_integrals')
for i in range(2):
    g.E(f'sp.itj0y0(np.array([0.5, 5.0, 25.0, -2.0]))[{i}]'); g.E(f'sp.it2j0y0(np.array([0.5, 5.0, 25.0]))[{i}]')
    g.E(f'sp.iti0k0(np.array([0.5, 5.0, 25.0]))[{i}]'); g.E(f'sp.it2i0k0(np.array([0.5, 5.0, 25.0, 50.0]))[{i}]')
g.E('sp.itstruve0(np.array([0.5, 5.0, 35.0]))'); g.E('sp.it2struve0(np.array([0.5, 30.0, -3.0]))'); g.E('sp.itmodstruve0(np.array([0.5, 5.0, 35.0]))')
for i in range(4):
    g.E(f'sp.itairy(np.array([0.5, 5.0, 12.0, -3.0, -12.0]))[{i}]', tol=1e-8)
for i in range(2):
    g.E(f'sp.modfresnelp(np.array([0.5, 3.0, 8.0, -1.0]))[{i}]'); g.E(f'sp.modfresnelm(np.array([0.5, 3.0, 8.0, -1.0]))[{i}]')
g.test('test_misc_funcs')
for e in ['sp.besselpoly(1.2, 0.5, 1.0)', 'sp.besselpoly(0.7, 1.5, -2.0)', 'sp.exprel(np.array([1e-20, 0.5, 20.0]))', 'sp.powm1(1.0001, 3.0)', 'sp.powm1(2.0, 0.1)',
          'sp.diric(np.array([0.5, 1e-9, 6.2831853]), 4)', 'sp.assoc_laguerre(0.7, 3, 1.5)', 'sp.nrdtrimn(0.3, 2.0, 1.0)', 'sp.nrdtrisd(0.5, 0.3, 1.0)', 'sp.sinc(0.5)']:
    g.E(e)
g.write('test/scipy/test_special_seq.py', 'gen_special3_expected.py', ['import numpy as np', 'import scipy.special as sp'], [])
