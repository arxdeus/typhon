# Generates test/scipy/test_rbf_griddata.py from the reference scipy 1.18.1.
import numpy as np, scipy.interpolate as I
from _gen_common import Gen
setup = ['y = np.cos(np.arange(60.0).reshape(30, 2) ** 1.37) * 1.5',
         'd = np.sin(y[:, 0] * 2.0) * y[:, 1] + y[:, 0]',
         'x = np.array([[0.1, 0.2], [0.5, -0.3], [-0.8, 0.4], [3.0, 3.0], [0.0, 0.0], [1.2, -1.1]])',
         'y3 = np.cos(np.arange(60.0).reshape(20, 3) ** 1.3)', 'd3 = y3[:, 0] * y3[:, 1] - y3[:, 2] ** 2',
         'x3 = np.array([[0.1, 0.2, 0.0], [0.5, -0.3, 0.4], [-0.2, 0.1, -0.3]])',
         'gx, gy = np.meshgrid(np.linspace(-1.0, 1.0, 4), np.linspace(-0.5, 0.5, 3))']
ns = {'np': np, 'I': I}
for l in setup: exec(l, ns)
g = Gen(ns, tol=1e-8)
g.test('test_rbf')
for kern, eps in [('thin_plate_spline', 1.0), ('linear', 1.0), ('cubic', 1.0), ('quintic', 1.0), ('multiquadric', 2.0), ('inverse_multiquadric', 2.0), ('inverse_quadratic', 2.0), ('gaussian', 3.0)]:
    g.E(f"I.RBFInterpolator(y, d, kernel='{kern}', epsilon={eps})(x)")
g.E('I.RBFInterpolator(y, d, smoothing=0.01, degree=2)(x)')
g.E('I.RBFInterpolator(y, d, neighbors=8)(x)')
g.E("I.RBFInterpolator(y, np.stack([d, d * 2], axis=1), kernel='gaussian', epsilon=2.0, degree=-1)(x)")
g.E('I.RBFInterpolator(y3, d3)(x3)')
g.E("I.RBFInterpolator(y3, d3, kernel='multiquadric', epsilon=1.5, smoothing=np.full(20, 0.1))(x3)")
g.test('test_griddata')
for m in ['nearest', 'linear', 'cubic']:
    g.E(f"I.griddata(y, d, x, method='{m}')")
    g.E(f"I.griddata(y, d, (gx, gy), method='{m}')")
g.E("I.griddata(y, d, x, method='linear', fill_value=-7.0)")
g.E("I.griddata(y, d, x, method='cubic', rescale=True)")
g.E("I.griddata(y3, d3, x3, method='linear')")
g.E("I.griddata(y3, d3, x3, method='nearest')")
g.E('I.LinearNDInterpolator(y, np.stack([d, 2 * d], axis=1), fill_value=0.0)(x)')
g.E('I.CloughTocher2DInterpolator(y, d, rescale=True)(x)')
g.E('I.NearestNDInterpolator(y, d)(x)')
g.test('test_rbf_legacy')
for f in ['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate']:
    g.E(f"I.Rbf(y[:, 0], y[:, 1], d, function='{f}')(x[:, 0], x[:, 1])")
g.E("I.Rbf(y[:, 0], y[:, 1], d, epsilon=0.7, smooth=0.01)(x[:, 0], x[:, 1])")
g.E("I.Rbf(y3[:, 0], y3[:, 1], y3[:, 2], d3, function='cubic')(x3[:, 0], x3[:, 1], x3[:, 2])")
g.E("I.Rbf(y[:, 0], d, function='linear')(np.array([0.1, 0.5, -0.2]))")
g.E("I.Rbf(y[:, 0], y[:, 1], d, norm='cityblock')(gx, gy)")
g.write('test_rbf_griddata.py', 'gen_rbf_griddata_expected.py', ['import numpy as np', 'import scipy.interpolate as I'], setup)
