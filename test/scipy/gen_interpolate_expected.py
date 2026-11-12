# Generates the expected values used by test_interpolate.py (scipy 1.18.1).
# Run with a CPython that has scipy installed; paste the output into the test.
import numpy as np
from scipy import interpolate as I

X = np.array([0., .5, 1.3, 2., 2.2, 3.1, 4., 5.5])
Y = np.sin(X)
Y2 = np.vstack([np.sin(X), np.cos(X) * 2.0, X ** 2])  # shape (3, 8): interp along axis=1
XQ = np.array([-0.7, 0.0, 0.25, 1.0, 2.1, 3.1, 4.9, 5.5, 6.3])
XIN = np.array([0.0, 0.25, 1.0, 2.1, 3.1, 4.9, 5.5])

out = {}
def put(name, v):
    out[name] = np.asarray(v, dtype=float).ravel().tolist()

for kind in ['linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next']:
    f = I.interp1d(X, Y, kind=kind, bounds_error=False, fill_value='extrapolate')
    put(f'i1d_{kind}_ext', f(XQ))
    put(f'i1d_{kind}_in', I.interp1d(X, Y, kind=kind)(XIN))
put('i1d_fill', I.interp1d(X, Y, bounds_error=False, fill_value=-9.0)(XQ))
put('i1d_fill2', I.interp1d(X, Y, bounds_error=False, fill_value=(-1.0, 7.0))(XQ))
put('i1d_nan', I.interp1d(X, Y, bounds_error=False)(XQ))
put('i1d_axis', I.interp1d(X, Y2, kind='cubic', axis=1)(XIN))
put('i1d_axis0', I.interp1d(X, Y2.T, axis=0)(XIN))
put('i1d_unsorted', I.interp1d(X[::-1], Y[::-1])(XIN))

for bc in ['not-a-knot', 'natural', 'clamped']:
    cs = I.CubicSpline(X, Y, bc_type=bc)
    put(f'cs_{bc}', cs(XQ))
    put(f'cs_{bc}_d1', cs(XQ, 1))
    put(f'cs_{bc}_d2', cs(XQ, 2))
    put(f'cs_{bc}_int', [cs.integrate(-0.5, 4.4), cs.integrate(5.0, 1.0)])
cs = I.CubicSpline(X, Y, bc_type=((1, 0.3), (2, -1.0)))
put('cs_custom', cs(XQ))
XP = np.linspace(0, 2 * np.pi, 9); YP = np.sin(XP); YP[-1] = YP[0]
csp = I.CubicSpline(XP, YP, bc_type='periodic')
put('cs_periodic', csp(np.array([-1.0, 0.3, 2.0, 7.5, 13.0])))
put('cs_periodic_int', [csp.integrate(0.5, 11.0)])
put('cs_noext', I.CubicSpline(X, Y, extrapolate=False)(XQ))
put('cs_2d', I.CubicSpline(X, Y2, axis=1)(XIN))
put('cs_3pt', I.CubicSpline(X[:3], Y[:3])(XQ))
put('cs_2pt', I.CubicSpline(X[:2], Y[:2])(XQ))
cs = I.CubicSpline(X, Y)
put('cs_deriv', cs.derivative(2)(XQ))
put('cs_anti', cs.antiderivative()(XQ))
put('cs_anti2', cs.antiderivative(2)(XQ))
put('cs_roots', cs.roots())
put('cs_solve', cs.solve(0.5))
put('cs_c', cs.c)

put('pchip', I.PchipInterpolator(X, Y)(XQ))
put('pchip_d1', I.PchipInterpolator(X, Y)(XQ, 1))
YM = np.array([0., 1., 1., 3., 2., 2.5, 2.5, 0.])
put('pchip_mono', I.PchipInterpolator(X, YM)(XQ))
put('pchip_2d', I.PchipInterpolator(X, Y2, axis=1)(XIN))
put('pchip_interp', I.pchip_interpolate(X, YM, XIN, der=1))
put('akima', I.Akima1DInterpolator(X, YM)(XQ))
put('makima', I.Akima1DInterpolator(X, YM, method='makima')(XQ))
put('akima_ext', I.Akima1DInterpolator(X, YM, extrapolate=True)(XQ))
put('akima_2d', I.Akima1DInterpolator(X, Y2, axis=1)(XIN))
put('hermite', I.CubicHermiteSpline(X, Y, np.cos(X))(XQ))

pp = I.PPoly(np.array([[1., 2.], [0., -1.], [3., 0.5]]), np.array([0., 1., 3.]))
put('ppoly', pp(XQ))
put('ppoly_int', [pp.integrate(-1, 2.5), pp.integrate(0.5, 4.0, extrapolate=False)])
put('ppoly_noext', I.PPoly(pp.c, pp.x, extrapolate=False)(XQ))

for k in [1, 2, 3, 4, 5]:
    b = I.make_interp_spline(X, Y, k=k)
    put(f'mis_{k}', b(XQ))
    put(f'mis_{k}_t', b.t)
    put(f'mis_{k}_c', b.c)
    put(f'mis_{k}_d1', b(XQ, nu=1))
    put(f'mis_{k}_int', [b.integrate(-0.5, 4.4), b.integrate(-0.5, 4.4, extrapolate=False)])
put('mis_0', I.make_interp_spline(X, Y, k=0)(XIN))
put('mis_nat', I.make_interp_spline(X, Y, bc_type='natural')(XQ))
put('mis_clamp', I.make_interp_spline(X, Y, bc_type='clamped')(XQ))
put('mis_bc', I.make_interp_spline(X, Y, k=5, bc_type=([(1, 0.5), (2, 0.0)], [(1, -1.0), (2, 1.0)]))(XQ))
put('mis_2d', I.make_interp_spline(X, Y2, axis=1)(XIN))
put('mis_per', I.make_interp_spline(XP, YP, bc_type='periodic')(np.array([-1.0, 0.3, 2.0, 7.5])))
put('mis_per5', I.make_interp_spline(XP, YP, k=5, bc_type='periodic')(np.array([-1.0, 0.3, 2.0, 7.5])))
b = I.make_interp_spline(X, Y)
put('bs_der2', b.derivative(2)(XQ))
put('bs_anti', b.antiderivative()(XQ))
put('bs_anti_t', b.antiderivative().t)
put('bs_noext', b(XQ, extrapolate=False))
be = I.BSpline.basis_element([0., 1., 2., 4.])
put('bs_basis', be(np.array([-0.5, 0.0, 0.5, 1.5, 3.0, 4.0])))
put('bs_basis_noext', I.BSpline.basis_element([0., 1., 2., 4.], extrapolate=False)(np.array([-0.5, 0.5, 3.0, 4.5])))
bs = I.BSpline(np.array([0., 0., 0., 1., 2., 3., 3., 3.]), np.array([1., -1., 2., 0.5, 3.]), 2)
put('bs_manual', bs(XQ))
put('bs_manual_int', [bs.integrate(0.2, 2.7), bs.integrate(-1.0, 5.0)])
put('bs_design', I.BSpline.design_matrix(np.array([0.0, 0.7, 1.5, 3.0]), bs.t, 2).toarray())

for k in [1, 2, 3, 5]:
    tck = I.splrep(X, Y, k=k)
    put(f'splrep_{k}_t', tck[0])
    put(f'splrep_{k}_c', tck[1])
    put(f'splev_{k}', I.splev(XQ, tck))
    put(f'splev_{k}_d1', I.splev(XQ, tck, der=1))
tck = I.splrep(X, Y)
for e in [1, 3]:
    put(f'splev_ext{e}', I.splev(XQ, tck, ext=e))
put('splint', [I.splint(0.3, 4.2, tck), I.splint(-1.0, 9.0, tck), I.splint(4.0, 1.0, tck)])
t2, c2, k2 = I.splder(tck)
put('splder_t', t2); put('splder_c', c2)
t3, c3, k3 = I.splantider(tck, 2)
put('splantider_c', c3)

g1 = np.array([0., 1., 2.5, 4.]); g2 = np.array([-1., 0., 0.5, 2., 3.])
V = np.sin(g1)[:, None] * np.cos(g2)[None, :] + g1[:, None] * g2[None, :]
pts = np.array([[0.3, -0.5], [2.5, 0.25], [3.9, 2.9], [1.0, 0.0], [0.0, 3.0]])
put('rgi_lin', I.RegularGridInterpolator((g1, g2), V)(pts))
put('rgi_near', I.RegularGridInterpolator((g1, g2), V, method='nearest')(pts))
ptsx = np.array([[-0.5, 0.0], [1.0, 3.5], [2.0, 1.0]])
put('rgi_fill', I.RegularGridInterpolator((g1, g2), V, bounds_error=False, fill_value=-5.0)(ptsx))
put('rgi_extrap', I.RegularGridInterpolator((g1, g2), V, bounds_error=False, fill_value=None)(ptsx))
put('rgi_desc', I.RegularGridInterpolator((g1[::-1], g2), V[::-1])(pts))
g3 = np.array([0., 1., 3.])
V3 = np.arange(4 * 5 * 3, dtype=float).reshape(4, 5, 3) ** 1.5
put('rgi_3d', I.RegularGridInterpolator((g1, g2, g3), V3)(np.array([[0.3, -0.5, 2.0], [3.0, 2.2, 0.4]])))
put('rgi_trail', I.RegularGridInterpolator((g1, g2), V3)(pts))
put('interpn', I.interpn((g1, g2), V, pts))
put('interpn_near', I.interpn((g1, g2), V, pts, method='nearest'))
put('rgi_mesh', I.RegularGridInterpolator((g1, g2), V)((np.array([[0.5], [1.5]]), np.array([[0.0, 1.0, 2.5]]))))

for k, v in out.items():
    print(f"    '{k}': {v!r},")
