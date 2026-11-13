# Generates test/scipy/test_interpolate3.py from the reference scipy 1.18.1.
import warnings; warnings.simplefilter('ignore')
import numpy as np, scipy.interpolate as I
from _gen_common import Gen
setup = ['x = np.linspace(-1.0, 3.0, 40)', 'y = np.tanh(3 * x) + 1.0 / (x + 1.5)', 'q = np.array([-0.95, 0.123, 1.0, 2.71, 3.0])']
ns = {'np': np, 'I': I}
for l in setup: exec(l, ns)
g = Gen(ns, tol=1e-9)
g.test('test_aaa')
g.raw('    a = I.AAA(x, y, clean_up=False)'); ns['a'] = I.AAA(ns['x'], ns['y'], clean_up=False)
g.E('a(q)'); g.raw(f'    assert a.support_points.size == {ns["a"].support_points.size}'); g.E('np.sort(a.support_points)')
g.E('a.errors', tol=1e-6)
g.raw('    a2 = I.AAA(x, np.exp(x), rtol=1e-8)'); ns['a2'] = I.AAA(ns['x'], np.exp(ns['x']), rtol=1e-8)
g.E('a2(q)'); g.raw(f'    assert a2.support_points.size == {ns["a2"].support_points.size}')
g.E('I.AAA(x, np.sin(4 * x), clean_up=False, max_terms=6)(q)')
g.test('test_floater_hormann')
for d in [0, 1, 3, 5]:
    g.E(f'I.FloaterHormannInterpolator(x, y, d={d})(q)')
g.E('I.FloaterHormannInterpolator(x, y).weights')
g.E('I.FloaterHormannInterpolator(x, np.stack([y, y * 2], axis=1), d=1)(q)')
g.write('test_interpolate3.py', 'gen_interpolate3_expected.py', ['import numpy as np', 'import scipy.interpolate as I'], setup)
