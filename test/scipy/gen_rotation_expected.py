# Generates test/scipy/test_rotation.py from the reference scipy 1.18.1.
import warnings; warnings.simplefilter('ignore')
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from _gen_common import Gen
setup = ['E = np.array([[30.0, 20.0, 10.0], [-50.0, 80.0, 170.0], [0.0, 90.0, 0.0], [120.0, -45.0, 33.0]])',
         'r = R.from_euler("zyx", E, degrees=True)', 'qs = np.array([0.1, 0.2, 0.3, 0.9])', 's = R.from_quat(qs)',
         'ax = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])',
         'ax2 = np.array([[0.0, 0.0, 1.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])',
         'a = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.3, 0.2, 0.9], [-0.5, 0.4, 0.1]])',
         'V = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.0], [0.0, 0.0, 1.0], [2.0, -2.0, 1.0]])']
ns = {'np': np, 'R': R, 'Slerp': Slerp}
for l in setup: exec(l, ns)
ns['b'] = ns['r'][0].apply(ns['a']) + 0.01 * np.cos(np.arange(12.0)).reshape(4, 3)
setup.append('b = r[0].apply(a) + 0.01 * np.cos(np.arange(12.0)).reshape(4, 3)')
g = Gen(ns, tol=1e-10)
g.test('test_conversions')
g.E('r.as_quat()'); g.E('r.as_matrix()'); g.E('r.as_rotvec()'); g.E('r.as_rotvec(degrees=True)'); g.E('r.as_mrp()')
for seq in ['xyz', 'zyx', 'ZXZ', 'xzx', 'YZX', 'zyz']:
    g.E(f"r.as_euler('{seq}')"); g.E(f"s.as_euler('{seq}', degrees=True)")
g.E('s.as_quat()'); g.E('s.as_matrix()'); g.E('[s.magnitude()]'); g.E('r.magnitude()')
g.E('s.as_quat(canonical=True, scalar_first=True)'); g.E('R.from_quat(np.array([0.9, 0.1, 0.2, 0.3]), scalar_first=True).as_quat()')
g.E('R.from_quat(-qs).as_quat(canonical=True)')
g.E('R.from_rotvec(np.array([0.1, -0.2, 0.3])).as_quat()'); g.E('R.from_rotvec(np.array([1e-4, 2e-4, 0.0])).as_quat()')
g.E('R.from_rotvec(np.array([[10.0, 20.0, 30.0], [0.0, 0.0, 180.0]]), degrees=True).as_quat()')
g.E('R.from_mrp(np.array([0.1, 0.2, -0.1])).as_quat()'); g.E('R.from_mrp(r.as_mrp()).as_quat()')
g.E('R.from_matrix(r.as_matrix()).as_quat()'); g.E('R.from_matrix(np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])).as_quat()')
g.E("R.from_euler('x', 0.4).as_quat()"); g.E("R.from_euler('XY', np.array([0.4, -1.2])).as_quat()")
g.E("R.from_davenport(ax, 'e', np.array([0.1, 0.2, 0.3])).as_quat()"); g.E("R.from_davenport(ax2, 'i', np.array([[0.1, 0.2, 0.3], [1.0, -0.5, 2.0]])).as_quat()")
g.E("r.as_davenport(ax, 'i')"); g.E("r.as_davenport(ax2, 'e', degrees=True)")
g.test('test_ops')
g.E('s.apply(np.array([1.0, 2.0, 3.0]))'); g.E('r.apply(np.array([1.0, 0.0, 0.0]))'); g.E('r.apply(V)'); g.E('r.apply(V, inverse=True)')
g.E('(s * r).as_quat()'); g.E('(r * s).as_quat()'); g.E('(r * r).as_quat()'); g.E('s.inv().as_quat()'); g.E('r.inv().as_quat()')
g.E('(s ** 0.3).as_quat()'); g.E('(r ** 2.5).as_quat()'); g.E('(r ** -1).as_quat()'); g.E('(s ** 0).as_quat()')
g.E('r.mean().as_quat()'); g.E('r.mean(weights=np.array([1.0, 2.0, 0.5, 0.0])).as_quat()')
g.raw('    assert len(r) == 4')
g.E('r[1].as_quat()'); g.E('r[-1].as_quat()'); g.E('r[1:3].as_quat()')
g.E('R.identity(2).as_quat()'); g.E('R.identity().as_quat()')
g.E('R.concatenate([r, R.from_quat(np.array([[0.0, 0.0, 0.0, 1.0]]))]).as_quat()')
g.E('R.random(3, rng=7).as_quat()'); g.E('R.random(rng=11).as_quat()')
g.raw('    assert list(r.approx_equal(r)) == [True, True, True, True]')
g.raw('    assert not s.approx_equal(r[0])')
g.test('test_align_slerp')
g.raw('    est, rssd = R.align_vectors(a, b)'); est, rssd = R.align_vectors(ns['a'], ns['b']); ns['est'] = est; ns['rssd'] = rssd
g.E('est.as_quat()'); g.E('[rssd]')
g.raw('    est, rssd, sens = R.align_vectors(a, b, return_sensitivity=True)'); ns['sens'] = R.align_vectors(ns['a'], ns['b'], return_sensitivity=True)[2]
g.E('sens')
g.raw('    est, rssd = R.align_vectors(a, b, weights=np.array([np.inf, 1.0, 1.0, 2.0]))'); est, rssd = R.align_vectors(ns['a'], ns['b'], weights=np.array([np.inf, 1.0, 1.0, 2.0])); ns['est'] = est; ns['rssd'] = rssd
g.E('est.as_quat()'); g.E('[rssd]')
g.raw('    est, rssd = R.align_vectors(a[0], b[1])'); est, rssd = R.align_vectors(ns['a'][0], ns['b'][1]); ns['est'] = est; ns['rssd'] = rssd
g.E('est.as_quat()'); g.E('[rssd]')
g.raw('    est, rssd = R.align_vectors(a[0], -a[0])'); est, rssd = R.align_vectors(ns['a'][0], -ns['a'][0]); ns['est'] = est
g.E('est.as_quat()')
g.raw('    sl = Slerp(np.array([0.0, 1.0, 3.0, 4.5]), r)'); ns['sl'] = Slerp(np.array([0.0, 1.0, 3.0, 4.5]), ns['r'])
g.E('sl(np.array([0.0, 0.5, 2.0, 3.0, 4.2, 4.5])).as_quat()'); g.E('sl(1.7).as_quat()')
g.write('test_rotation.py', 'gen_rotation_expected.py', ['import numpy as np', 'from scipy.spatial.transform import Rotation as R, Slerp'], setup)
