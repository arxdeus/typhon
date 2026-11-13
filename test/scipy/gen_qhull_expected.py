# Generates test/scipy/test_qhull.py from the reference scipy 1.18.1.
# Simplex/facet order differs from Qhull, so results are canonicalised.
import numpy as np, scipy.spatial as S
from _gen_common import Gen
setup = ['rng = np.random.default_rng(0)',
         'p = np.cos(np.arange(24.0).reshape(12, 2) * 1.37) * 1.5 + np.array([0.3, -0.2])',
         'p3 = np.cos(np.arange(36.0).reshape(12, 3) ** 1.3)',
         'p4 = np.cos(np.arange(60.0).reshape(15, 4) ** 1.21) + 0.3 * np.sin(np.arange(60.0).reshape(15, 4) ** 1.07)',
         'q = np.array([[0.2, 0.3], [5.0, 5.0], [0.9, 0.1], [-0.4, 0.0], [1.0, -0.5]])',
         'hs = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [2.0, 1.0, -4.0], [-0.5, 1.0, -2.0]])']
helpers = '''
def canon(S):
    a = np.sort(np.asarray(S), axis=1)
    rows = [[float(a[i, j]) for j in range(a.shape[1])] for i in range(a.shape[0])]
    rows.sort()
    out = List[float]()
    for r in rows:
        for v in r:
            out.append(v)
    return out

def canon_eq(E):
    a = np.asarray(E)
    rows = [[round(float(a[i, j]), 9) for j in range(a.shape[1])] for i in range(a.shape[0])]
    rows.sort()
    out = List[float]()
    for r in rows:
        for v in r:
            out.append(v)
    return out

def region_coords(v, reg):
    rows = [[round(float(c), 9) for c in v.vertices[j]] for j in reg if j >= 0]
    rows.sort()
    out = List[float]()
    for r in rows:
        for c in r:
            out.append(c)
    return out

def contains(tri, xs, idx):
    # point xs[k] lies in simplex idx[k] (or idx == -1 and outside all)
    T = tri.transform
    for k in range(xs.shape[0]):
        s = int(idx[k])
        if s < 0:
            continue
        D = xs.shape[1]
        c = T[s, :D].dot(xs[k] - T[s, D])
        if not (np.all(c >= -1e-10) and 1 - c.sum() >= -1e-10):
            return False
    return True
'''
ns = {'np': np, 'S': S}
for l in setup: exec(l, ns)
exec(helpers.replace('List[float]()', '[]'), ns)
g = Gen(ns, tol=1e-9)
g.test('test_delaunay')
for P in ['p', 'p3', 'p4']:
    g.raw(f'    d = S.Delaunay({P})'); ns['d'] = S.Delaunay(ns[P])
    g.E('canon(d.simplices)'); g.E('[d.paraboloid_scale, d.paraboloid_shift]')
    g.E('canon_eq(np.sort(np.abs(d.equations), axis=1))')
    g.E('canon(d.convex_hull)')
    g.raw(f'    assert d.nsimplex == {ns["d"].nsimplex}')
    g.E('np.sort(d.vertex_neighbor_vertices[1][d.vertex_neighbor_vertices[0][0]:d.vertex_neighbor_vertices[0][1]])')
    g.E('d.vertex_neighbor_vertices[0]')
g.raw('    d = S.Delaunay(p)')
ns['d'] = S.Delaunay(ns['p'])
g.raw(f'    r = d.find_simplex(q)')
r = ns['d'].find_simplex(ns['q'])
g.raw(f'    assert [int(v) < 0 for v in r] == {[bool(v < 0) for v in r]}')
g.raw('    assert contains(d, q, r)')
g.raw('    r = S.tsearch(d, q)')
g.raw('    assert contains(d, q, r)')
g.E('np.sort(d.plane_distance(q).ravel())')
g.E('d.lift_points(q)')
g.test('test_hull')
for P in ['p', 'p3', 'p4']:
    g.raw(f'    h = S.ConvexHull({P})'); ns['h'] = S.ConvexHull(ns[P])
    g.E('canon(h.simplices)'); g.E('[h.volume, h.area]'); g.E('np.sort(h.vertices)')
    g.E('canon_eq(h.equations)')
g.raw('    h = S.ConvexHull(p)')
g.raw('    v = h.vertices')
g.raw('    for i in range(len(v)):')
g.raw('        a = p[v[(i + 1) % len(v)]] - p[v[i]]')
g.raw('        b = p[v[(i + 2) % len(v)]] - p[v[(i + 1) % len(v)]]')
g.raw('        assert a[0] * b[1] - a[1] * b[0] > 0')
g.test('test_voronoi')
ns['v'] = S.Voronoi(ns['p']); g.raw('    v = S.Voronoi(p)')
g.E('canon_eq(v.vertices)')
g.E('canon(v.ridge_points)')
# regions as sets of rounded vertex coordinates per input point
vv = ns['v']
for i in range(ns['p'].shape[0]):
    reg = vv.regions[vv.point_region[i]]
    g.raw(f'    reg = v.regions[v.point_region[{i}]]')
    g.raw(f'    assert (-1 in reg) == {(-1 in reg)} and len(reg) == {len(reg)}')
    coords = sorted([tuple(round(float(c), 9) for c in vv.vertices[j]) for j in reg if j >= 0])
    flat = [c for t in coords for c in t]
    g.raw(f'    assert closec(np.array(region_coords(v, reg)), {[float(x) for x in flat]!r}, 1e-9)')
g.test('test_halfspace')
ns['hi'] = S.HalfspaceIntersection(ns['hs'], np.array([0.5, 0.5])); g.raw('    hi = S.HalfspaceIntersection(hs, np.array([0.5, 0.5]))')
g.E('canon_eq(hi.intersections)'); g.E('[hi.dual_volume, hi.dual_area]'); g.E('np.sort(hi.dual_vertices)')
g.write('test_qhull.py', 'gen_qhull_expected.py', ['import numpy as np', 'import scipy.spatial as S'], setup + helpers.split('\n'))
