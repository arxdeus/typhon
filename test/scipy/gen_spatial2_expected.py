# Generates test/scipy/test_spatial2.py from the reference scipy 1.18.1.
import numpy as np, scipy.spatial as S
from scipy.spatial import distance as D
from _gen_common import Gen
setup = ['B = np.array([[1.0, 0., 1., 1., 0., 1.], [0.0, 1., 1., 0., 0., 1.], [1.0, 1., 0., 0., 1., 0.], [1.0, 0., 0., 1., 1., 1.]])',
         'U = np.cos(np.arange(40.0).reshape(20, 2) * 1.7)', 'V = np.sin(np.arange(30.0).reshape(15, 2) * 0.9)',
         'P = np.cos(np.arange(36.0).reshape(12, 3) ** 1.3)',
         'a = np.array([[1.0, 3.], [1., 2.], [1., 1.], [2., 1.]])', 'b = np.array([[4.0, -2.], [4., -4.], [4., -6.], [2., -6.]])']
ns = {'np': np, 'S': S, 'D': D}
for l in setup: exec(l, ns)
g = Gen(ns)
g.test('test_distance_new')
for m in ['rogerstanimoto', 'russellrao', 'sokalsneath', 'yule', 'dice', 'jensenshannon']:
    off = ' + 0.5' if m == 'jensenshannon' else ''
    g.E(f"D.pdist(B{off}, '{m}')"); g.E(f"D.cdist(B{off}, B[:2]{off}, '{m}')")
    g.E(f"[D.{m}(B[0]{off}, B[3]{off})]")
g.E("[D.jensenshannon(np.array([0.2, 0.3, 0.5]), np.array([0.1, 0.6, 0.3]))]")
g.E("[D.jensenshannon(np.array([0.2, 0.3, 0.5]), np.array([0.1, 0.6, 0.3]), 2.0)]")
g.E("D.jensenshannon(B + 0.5, B[::-1] + 0.5, axis=1)")
g.E("D.jensenshannon(B + 0.5, B[::-1] + 0.5)")
for args in ['U, V', 'V, U', 'U, V, 3', 'P, P[::2] + 0.1', 'P[::2], P, 7']:
    r = eval(f'D.directed_hausdorff({args})', ns)
    g.raw(f'    h = D.directed_hausdorff({args})')
    g.raw(f'    assert abs(h[0] - {r[0]!r}) < 1e-13 and h[1] == {r[1]} and h[2] == {r[2]}')
g.test('test_procrustes_slerp')
for k in range(3):
    g.E(f"S.procrustes(a, b * 1.3 + 0.2)[{k}]" if k < 2 else "[S.procrustes(a, b * 1.3 + 0.2)[2]]", tol=1e-10)
    g.E(f"S.procrustes(P, np.sin(P) * 2.0)[{k}]" if k < 2 else "[S.procrustes(P, np.sin(P) * 2.0)[2]]", tol=1e-10)
g.E("S.geometric_slerp(np.array([1.0, 0., 0.]), np.array([0.0, 0.6, 0.8]), np.linspace(0.0, 1.0, 5))")
g.E("S.geometric_slerp(np.array([1.0, 0.]), np.array([0.0, 1.]), 0.3)")
g.E("S.geometric_slerp(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0.0, 0.6, 0.8, 0.]), np.array([0.25, 1.5, -0.3]))")
g.test('test_spherical_voronoi')
ns['sph'] = S.SphericalVoronoi(ns['P'] / np.linalg.norm(ns['P'], axis=1)[:, None])
g.raw('    sph = S.SphericalVoronoi(P / np.linalg.norm(P, axis=1).reshape((-1, 1)))')
g.E('np.sort(sph.calculate_areas())')
g.E('[float(np.sum(sph.calculate_areas()))]')
vv = ns['sph'].vertices
rows = sorted([[round(float(c), 9) for c in v] for v in vv])
g.raw('    rows = [[round(float(c), 9) for c in sph.vertices[i]] for i in range(sph.vertices.shape[0])]')
g.raw('    rows.sort()')
g.raw('    flat = [c for r in rows for c in r]')
g.raw(f'    assert closec(np.array(flat), {[c for r in rows for c in r]!r}, 1e-8)')
ns['sp2'] = S.SphericalVoronoi(np.array([[1.0, 0.0], [0.0, 1.0], [-0.6, -0.8], [0.6, -0.8]]) * 2.0, radius=2.0)
g.raw('    sp2 = S.SphericalVoronoi(np.array([[1.0, 0.0], [0.0, 1.0], [-0.6, -0.8], [0.6, -0.8]]) * 2.0, radius=2.0)')
g.E('np.sort(sp2.calculate_areas())')
g.write('test/scipy/test_spatial2.py', 'gen_spatial2_expected.py', ['import numpy as np', 'import scipy.spatial as S', 'from scipy.spatial import distance as D'], setup)
