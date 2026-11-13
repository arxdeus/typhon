# Generates test/scipy/test_fitpack.py from the reference scipy 1.18.1.
import numpy as np, scipy.interpolate as I
from _gen_common import Gen
setup = ['x = np.linspace(0.0, 10.0, 40)', 'y = np.sin(x) + 0.1 * np.cos(7.3 * x)', 'w = 1.0 + 0.5 * np.cos(x)',
         'xe = np.linspace(-0.5, 10.5, 23)',
         'P = np.array([np.cos(x / 3.0), np.sin(x / 2.0), x / 10.0])',
         'x2 = np.array([0.0, 0.3, 0.9, 1.0, 1.8, 2.5, 2.6, 3.9, 4.4, 5.0, 6.1, 6.2, 7.7, 8.0, 9.9])',
         'y2 = np.exp(-x2 / 3.0) * np.cos(2.0 * x2)']
ns = {'np': np, 'I': I}
for l in setup: exec(l, ns)
g = Gen(ns, tol=1e-10)
g.test('test_splrep_smooth')
for s in [0.0, 0.05, 0.3, 2.0, 50.0]:
    for k in [1, 2, 3, 5]:
        g.raw(f'    tck = I.splrep(x, y, s={s}, k={k})')
        ns['tck'] = I.splrep(ns['x'], ns['y'], s=s, k=k)
        g.E('tck[0]'); g.E('tck[1]'); g.E('I.splev(xe, tck)')
g.raw('    tck = I.splrep(x, y, w=w)'); ns['tck'] = I.splrep(ns['x'], ns['y'], w=ns['w'])
g.E('tck[0]'); g.E('tck[1]')
for args in ['x2, y2, s=0.01', 'x2, y2, s=0.1, k=4', 'x2, y2, w=np.ones(15) * 3.0', 'x2, y2, t=np.array([2.0, 4.0, 6.0, 8.0])',
             'x, y, t=np.array([1.0, 2.5, 3.0, 5.0, 7.5]), k=2', 'x, y, s=0.5, xb=-1.0, xe=11.0']:
    g.raw(f'    tck = I.splrep({args})'); ns['tck'] = eval(f'I.splrep({args})', ns)
    g.E('tck[0]'); g.E('tck[1]'); g.E('I.splev(xe, tck)')
g.raw('    r = I.splrep(x, y, s=0.3, full_output=1)')
r = I.splrep(ns['x'], ns['y'], s=0.3, full_output=1)
g.raw(f'    assert abs(r[1] - {r[1]!r}) < 1e-10 and r[2] == {r[2]}')
g.test('test_tools')
ns['tck'] = I.splrep(ns['x'], ns['y'], s=0.3); g.raw('    tck = I.splrep(x, y, s=0.3)')
g.E('I.sproot(tck)'); g.E('I.sproot(I.splrep(x, y))')
g.E('I.spalde(2.5, tck)'); g.E('I.spalde(0.0, tck)'); g.E('np.array(I.spalde(np.array([1.0, 7.25, 10.0]), tck))')
for xi, m in [(3.3, 1), (3.3, 2), (0.5, 3), (9.99, 1)]:
    g.E(f'I.insert({xi}, tck, {m})[0]'); g.E(f'I.insert({xi}, tck, {m})[1]')
g.test('test_splprep')
for s in [0.0, 0.01, 0.5]:
    g.raw(f'    tcku = I.splprep(P, s={s})'); ns['tcku'] = I.splprep(ns['P'], s=s)
    g.E('tcku[0][0]'); g.E('tcku[0][1][0]'); g.E('tcku[0][1][2]'); g.E('tcku[1]')
    g.E('np.array(I.splev(np.array([0.1, 0.5, 0.77]), tcku[0]))')
g.raw('    tcku = I.splprep(P[:2], u=x, s=0.2, k=2)'); ns['tcku'] = I.splprep(ns['P'][:2], u=ns['x'], s=0.2, k=2)
g.E('tcku[0][0]'); g.E('tcku[0][1][1]'); g.E('tcku[1]')
g.raw('    tcku = I.splprep(P, w=w, s=0.3)'); ns['tcku'] = I.splprep(ns['P'], w=ns['w'], s=0.3)
g.E('tcku[0][0]'); g.E('tcku[0][1][1]')
g.write('test/scipy/test_fitpack.py', 'gen_fitpack_expected.py', ['import numpy as np', 'import scipy.interpolate as I'], setup)
