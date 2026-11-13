# Tests for numpy 2.x top-level additions (expected values from numpy 2.5.4).
import numpy as np

def check(name: str, got, exp):
    if got != exp:
        print('TEST FAILED', name, got, exp)

x = np.array([3, 1, 3, 2, 1, 5])
check('unique_values', sorted(np.unique_values(x).tolist()), [1, 2, 3, 5])
uc = np.unique_counts(x)
check('unique_counts', [uc.values.tolist(), uc.counts.tolist()], [[1, 2, 3, 5], [2, 1, 2, 1]])
ui = np.unique_inverse(x)
check('unique_inverse', ui.inverse_indices.tolist(), [2, 0, 2, 1, 0, 3])
ua = np.unique_all(x)
check('unique_all', [ua.indices.tolist(), ua.counts.tolist()], [[1, 3, 0, 5], [2, 1, 2, 1]])

o = np.ogrid[0:2, 0:3]
check('ogrid0', o[0].shape, (2, 1))
check('ogrid1', o[1].shape, (1, 3))
check('ogrid2', o[1].tolist(), [[0, 1, 2]])
m = np.mgrid[0:2, 0:3]
check('mgrid0', m.shape, (2, 2, 3))
check('mgrid1', [m[0].tolist(), m[1].tolist()], [[[0, 0, 0], [1, 1, 1]], [[0, 1, 2], [0, 1, 2]]])
mc = np.mgrid[0:1:3j]
check('mgrid_c', mc.tolist(), [0.0, 0.5, 1.0])
a = np.arange(10)
check('s_', a[np.s_[2:8:3]].tolist(), [2, 5])

check('binary_repr', [np.binary_repr(-5, 8), np.binary_repr(5), np.binary_repr(-5), np.binary_repr(3, 5)], ['11111011', '101', '-101', '00011'])
check('base_repr', [np.base_repr(-255, 16, 2), np.base_repr(10, 3), np.base_repr(0)], ['-00FF', '101', '0'])
check('fix', np.fix(np.array([-1.5, 2.7])).tolist(), [-1.0, 2.0])
check('float_power', np.float_power(np.array([2, 3]), 2).tolist(), [4.0, 9.0])
q, r = np.divmod(np.array([7, -7]), 3)
check('divmod', [q.tolist(), r.tolist()], [[2, -3], [1, 2]])
check('bitwise_count', np.bitwise_count(np.array([7, -1, 255])).astype(int).tolist(), [3, 1, 8])

z = np.zeros((2, 3, 4))
check('rollaxis', [np.rollaxis(z, 2).shape, np.rollaxis(z, 0, 3).shape, np.rollaxis(z, 1, 1).shape], [(4, 2, 3), (3, 4, 2), (2, 3, 4)])
check('permute_dims', np.permute_dims(z, (2, 0, 1)).shape, (4, 2, 3))
check('concat', np.concat((np.array([1, 2]), np.array([3]))).tolist(), [1, 2, 3])
u = np.unstack(np.array([[1, 2], [3, 4]]), axis=1)
check('unstack', [u[0].tolist(), u[1].tolist()], [[1, 3], [2, 4]])
check('cumulative_prod', np.cumulative_prod(np.array([1, 2, 3]), include_initial=True).tolist(), [1, 1, 2, 6])

check('vecdot', np.vecdot(np.array([[1, 2], [3, 4]]), np.array([1, 1])).tolist(), [3, 7])
check('matvec', np.matvec(np.array([[1, 2], [3, 4]]), np.array([1, 10])).tolist(), [21, 43])
check('vecmat', np.vecmat(np.array([1, 10]), np.array([[1, 2], [3, 4]])).tolist(), [31, 42])

h, ex, ey = np.histogram2d(np.array([0., 1, 2, 3]), np.array([0., 1, 1, 3]), bins=2)
check('histogram2d', h.tolist(), [[2.0, 0.0], [1.0, 1.0]])
check('histogram2d_e', [ex.tolist(), ey.tolist()], [[0.0, 1.5, 3.0], [0.0, 1.5, 3.0]])
hd = np.histogram2d(np.array([0., 1, 2, 3]), np.array([0., 1, 1, 3]), bins=2, density=True)[0]
check('histogram2d_density', hd.tolist(), [[0.2222222222222222, 0.0], [0.1111111111111111, 0.1111111111111111]])

check('lin.vector_norm', np.linalg.vector_norm(np.array([[3., 4.], [1., 1.]]), axis=1).tolist(), [5.0, 1.4142135623730951])
check('lin.vector_norm_all', np.linalg.vector_norm(np.array([[3., 4.], [0., 0.]])), 5.0)
check('lin.matrix_norm', np.linalg.matrix_norm(np.eye(2)), 1.4142135623730951)
check('lin.vecdot', np.linalg.vecdot(np.array([1., 2.]), np.array([3., 4.])), 11.0)
check('lin.cross', np.linalg.cross(np.array([1., 0., 0.]), np.array([0., 1., 0.])).tolist(), [0.0, 0.0, 1.0])
check('lin.diagonal', np.linalg.diagonal(np.arange(9).reshape(3, 3), offset=1).tolist(), [1, 5])
print('lib_more done')
