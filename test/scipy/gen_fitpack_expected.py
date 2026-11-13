# Generates test/scipy/test_fitpack.py from the reference scipy 1.18.1.
import numpy as np, scipy.interpolate as I
from _gen_common import Gen
setup = ['gx = np.linspace(0.0, 4.0, 9)', 'gy = np.linspace(-1.0, 2.0, 7)', 'Z = np.sin(gx).reshape((-1, 1)) * np.cos(gy).reshape((1, -1)) + 0.05 * np.cos(np.arange(63.0) ** 1.3).reshape(9, 7)', 'xq = np.array([0.3, 1.7, 3.9])', 'sx = np.cos(np.arange(80.0) ** 1.3) * 2.0', 'sy = np.sin(np.arange(80.0) ** 1.1) * 1.5', 'sz = np.exp(-(sx ** 2 + sy ** 2) / 2.0) + 0.02 * np.cos(np.arange(80.0) * 3.1)', 'yq = np.array([-0.5, 0.2, 1.9])', 'x = np.linspace(0.0, 10.0, 40)', 'y = np.sin(x) + 0.1 * np.cos(7.3 * x)', 'w = 1.0 + 0.5 * np.cos(x)',
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
g.test('test_rect_bivariate')
for args in ['s=0.0', 's=0.01', 's=0.1', 's=2.0', 's=0.01, kx=2, ky=4', 's=0.3, kx=1, ky=5', 's=0.002, kx=5, ky=2']:
    g.raw(f'    r = I.RectBivariateSpline(gx, gy, Z, {args})'); ns['r'] = eval(f'I.RectBivariateSpline(gx, gy, Z, {args})', ns)
    g.E('r.get_knots()[0]'); g.E('r.get_knots()[1]'); g.E('r.get_coeffs()'); g.E('[r.get_residual()]', tol=1e-8)
g.raw('    r = I.RectBivariateSpline(gx, gy, Z, s=0.05)'); ns['r'] = I.RectBivariateSpline(ns['gx'], ns['gy'], ns['Z'], s=0.05)
g.E('r(xq, yq)'); g.E('r(xq, yq, dx=1)'); g.E('r(xq, yq, dy=2)'); g.E('r.ev(xq, yq)'); g.E('r.ev(xq, yq, dx=1, dy=1)')
g.E('[r.integral(0.2, 3.1, -0.4, 1.5)]'); g.E('r.partial_derivative(1, 0)(xq, yq)')
g.E('I.bisplev(xq, yq, (r.get_knots()[0], r.get_knots()[1], r.get_coeffs(), 3, 3))')
g.E('I.bisplev(xq, yq, (r.get_knots()[0], r.get_knots()[1], r.get_coeffs(), 3, 3), dx=1, dy=2)')
g.test('test_surfit')
import warnings; warnings.simplefilter('ignore')
for args in ['', 's=0.5', 's=0.05', 'kx=2, ky=4, s=0.1', 'w=1.0 + 0.5 * np.cos(sx), s=0.2', 'kx=1, ky=1, s=0.3']:
    g.raw(f'    sp = I.SmoothBivariateSpline(sx, sy, sz, {args})'); ns['sp'] = eval(f'I.SmoothBivariateSpline(sx, sy, sz, {args})', ns)
    g.E('sp.get_knots()[0]'); g.E('sp.get_knots()[1]'); g.E('sp.get_coeffs()', tol=1e-8); g.E('[sp.get_residual()]', tol=1e-8)
    g.E('sp(xq * 0.3, yq * 0.5)', tol=1e-8)
g.raw('    l = I.LSQBivariateSpline(sx, sy, sz, np.array([-1.0, 0.0, 1.0]), np.array([-0.5, 0.5]))')
ns['l'] = I.LSQBivariateSpline(ns['sx'], ns['sy'], ns['sz'], np.array([-1.0, 0.0, 1.0]), np.array([-0.5, 0.5]))
g.E('l.get_knots()[0]'); g.E('l.get_coeffs()'); g.E('[l.get_residual()]'); g.E('l.ev(xq * 0.3, yq * 0.5)')
g.raw('    tck = I.bisplrep(sx, sy, sz, s=0.3)'); ns['tck'] = I.bisplrep(ns['sx'], ns['sy'], ns['sz'], s=0.3)
g.E('tck[0]'); g.E('tck[1]'); g.E('tck[2]', tol=1e-8); g.E('I.bisplev(np.array([0.1, 0.5]), np.array([-0.3, 0.2]), tck)', tol=1e-8)
g.raw('    tck = I.bisplrep(sx, sy, sz, kx=2, ky=2)'); ns['tck'] = I.bisplrep(ns['sx'], ns['sy'], ns['sz'], kx=2, ky=2)
g.E('tck[0]'); g.E('tck[1]'); g.E('tck[2]', tol=1e-8)
g.write('test/scipy/test_fitpack.py', 'gen_fitpack_expected.py', ['import numpy as np', 'import scipy.interpolate as I'], setup)
