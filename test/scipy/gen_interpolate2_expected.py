# Generates test/scipy/test_interpolate2.py from the reference scipy 1.18.1.
import warnings; warnings.simplefilter('ignore')
import numpy as np, scipy.interpolate as I
from _gen_common import Gen
setup = ['x = np.linspace(0.0, 10.0, 40)', 'y = np.sin(x) + 0.1 * np.cos(7.3 * x)', 'xe = np.linspace(-0.5, 10.5, 17)',
         'xi = np.array([0.0, 0.5, 1.0, 1.5, 2.3, 3.0])', 'yi = np.sin(xi)', 'Y2 = np.stack([yi, np.cos(xi)], axis=1)',
         'q = np.array([0.2, 1.7, 2.9, 1.0, -0.3])',
         'tk = np.array([0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 10.0, 10.0, 10.0])',
         'P = np.array([np.cos(x / 3.0), np.sin(x / 2.0), x / 10.0])']
ns = {'np': np, 'I': I}
for l in setup: exec(l, ns)
g = Gen(ns, tol=1e-9)
g.test('test_univariate')
g.raw('    s = I.UnivariateSpline(x, y)'); ns['s'] = I.UnivariateSpline(ns['x'], ns['y'])
g.E('s.get_knots()'); g.E('s.get_coeffs()'); g.E('[s.get_residual()]'); g.E('s(xe)')
for sf in [0.2, 1.0, 0.02]:
    g.raw(f'    s.set_smoothing_factor({sf})'); ns['s'].set_smoothing_factor(sf)
    g.E('s.get_knots()'); g.E('s.get_coeffs()'); g.E('s(xe)')
g.E('s.roots()'); g.E('[s.integral(1.0, 7.0)]'); g.E('s.derivatives(2.0)')
g.E('s.derivative()(xe)'); g.E('s.antiderivative(2)(xe)')
for args in ['x, y, s=0.5, k=2', 'x, y, w=1.0 + x * 0.1, s=3.0', 'x, y, k=5, s=0.1', 'x, y, bbox=(-1.0, 11.0), s=0.4']:
    g.raw(f'    s = I.UnivariateSpline({args})'); ns['s'] = eval(f'I.UnivariateSpline({args})', ns)
    g.E('s.get_knots()'); g.E('s(xe)')
for e in [0, 1, 3]:
    g.E(f'I.InterpolatedUnivariateSpline(x, y, k=4, ext={e})(xe)')
g.E("I.InterpolatedUnivariateSpline(x, y, ext='const')(xe)")
g.raw('    l = I.LSQUnivariateSpline(x, y, np.array([2.0, 5.0, 7.0]), k=2)'); ns['l'] = I.LSQUnivariateSpline(ns['x'], ns['y'], np.array([2.0, 5.0, 7.0]), k=2)
g.E('l.get_coeffs()'); g.E('[l.get_residual()]'); g.E('l(xe)')
g.test('test_polyint')
g.raw('    K = I.KroghInterpolator(xi, yi)'); ns['K'] = I.KroghInterpolator(ns['xi'], ns['yi'])
g.E('K(q)'); g.E('K.derivative(q, 2)'); g.E('K.derivatives(q, 3)'); g.E('K.derivatives(q)')
g.E('I.KroghInterpolator(np.array([0.0, 0.0, 1.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0, 4.0, 0.5]))(q)')
g.E('I.KroghInterpolator(xi, Y2)(q)'); g.E('I.krogh_interpolate(xi, yi, q, der=1)')
g.raw('    B = I.BarycentricInterpolator(xi, yi)'); ns['B'] = I.BarycentricInterpolator(ns['xi'], ns['yi'])
g.E('B(q)'); g.E('B.derivative(q, 2)'); g.E('B.derivatives(q, 3)')
g.E('I.barycentric_interpolate(xi, Y2, q)'); g.E('I.barycentric_interpolate(xi, yi, q, der=1)')
g.E('I.lagrange(np.array([0.0, 1.0, 2.0, 3.5]), np.array([1.0, 3.0, 2.0, -1.0])).coeffs')
g.raw('    pp, qq = I.pade(np.array([1.0, 1.0, 0.5, 1.0 / 6, 1.0 / 24, 1.0 / 120]), 2)')
ns['pp'], ns['qq'] = I.pade(np.array([1.0, 1.0, 0.5, 1.0 / 6, 1.0 / 24, 1.0 / 120]), 2)
g.E('pp.coeffs'); g.E('qq.coeffs')
g.E('I.approximate_taylor_polynomial(np.exp, 0.5, 3, 0.2).coeffs', tol=1e-7)
g.E('I.approximate_taylor_polynomial(np.sin, 0.3, 4, 0.5, order=6).coeffs', tol=1e-7)
g.test('test_lsq_smoothing')
g.E('I.make_lsq_spline(x, y, tk).c'); g.E('I.make_lsq_spline(x, np.stack([y, y * y], axis=1), tk, w=1.0 + x).c')
g.E('I.make_lsq_spline(x, y, np.array([0.0, 0.0, 0.0, 3.0, 7.0, 10.0, 10.0, 10.0]), k=2)(xe)')
g.E('I.make_smoothing_spline(x, y).c', tol=1e-7); g.E('I.make_smoothing_spline(x, y)(xe)', tol=1e-7)
g.E('I.make_smoothing_spline(x, y, lam=0.5).c'); g.E('I.make_smoothing_spline(x, y, w=1.0 + 0.5 * np.cos(x), lam=0.05)(xe)')
g.test('test_make_splrep')
for s in [0.05, 0.3, 2.0, 50.0]:
    g.raw(f'    b = I.make_splrep(x, y, s={s})'); ns['b'] = I.make_splrep(ns['x'], ns['y'], s=s)
    g.E('b.t'); g.E('b.c'); g.E('b(xe)')
g.raw('    b = I.make_splrep(x, y, s=0.5, w=1.0 + x * 0.1, k=2)'); ns['b'] = I.make_splrep(ns['x'], ns['y'], s=0.5, w=1.0 + ns['x'] * 0.1, k=2)
g.E('b.t'); g.E('b.c')
g.raw('    b = I.make_splrep(x, y, s=0.3, t=np.array([0.0, 0.0, 0.0, 0.0, 3.0, 6.0, 10.0, 10.0, 10.0, 10.0]))')
ns['b'] = I.make_splrep(ns['x'], ns['y'], s=0.3, t=np.array([0.0, 0.0, 0.0, 0.0, 3.0, 6.0, 10.0, 10.0, 10.0, 10.0]))
g.E('b.c')
kn = ns['kn'] = list(I.generate_knots(ns['x'], ns['y'], s=0.3))
g.raw('    kn = I.generate_knots(x, y, s=0.3)')
g.raw(f'    assert len(kn) == {len(kn)}')
for i, k in enumerate(kn):
    g.E(f'kn[{i}]')
g.raw('    sp, u = I.make_splprep(P, s=0.01)'); ns['sp'], ns['u'] = I.make_splprep(list(ns['P']), s=0.01)
g.E('sp.t'); g.E('sp.c'); g.E('u'); g.E('sp(np.array([0.1, 0.5, 0.9]))')
g.test('test_cubic_classes')
g.raw('    cs = I.CubicSpline(xi, yi, bc_type="natural")')
ns['cs'] = I.CubicSpline(ns['xi'], ns['yi'], bc_type='natural')
g.raw('    assert isinstance(cs, I.PPoly)')
g.E('cs(q)'); g.E('cs.c'); g.E('cs.derivative()(q)'); g.E('[cs.integrate(0.0, 2.0)]')
g.E('I.PchipInterpolator(xi, yi)(q)'); g.E("I.Akima1DInterpolator(xi, yi, method='makima')(q)"); g.E('I.CubicHermiteSpline(xi, yi, np.cos(xi))(q)')
g.write('test_interpolate2.py', 'gen_interpolate2_expected.py', ['import numpy as np', 'import scipy.interpolate as I'], setup)
