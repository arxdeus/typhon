# Tests for numpy additions (index tricks, einsum, polynomial helpers, ...).
# Expected values from real numpy, see test/numpy/gen_gaps_expected.py.
import numpy as np

def _eq(a, b) -> bool:
    return np.array_equal(np.asarray(a), np.asarray(b))

def _close(a, b) -> bool:
    return np.allclose(np.asarray(a), np.asarray(b), rtol=1e-10, atol=1e-12)

# ---- expected values (gen_gaps_expected.py) ----
einsum_ijjk = [[70.0, 76.0, 82.0, 88.0, 94.0], [190.0, 212.0, 234.0, 256.0, 278.0], [310.0, 348.0, 386.0, 424.0, 462.0]]
einsum_ell = [[[14.0, 38.0, 62.0], [38.0, 126.0, 214.0], [62.0, 214.0, 366.0]], [[734.0, 950.0, 1166.0], [950.0, 1230.0, 1510.0], [1166.0, 1510.0, 1854.0]]]
einsum_3op = [[880.0, 2930.0, 4980.0, 7030.0], [2560.0, 8410.0, 14260.0, 20110.0], [4240.0, 13890.0, 23540.0, 33190.0]]
einsum_cjk = [[[80.0, 92.0, 104.0, 116.0], [92.0, 107.0, 122.0, 137.0], [104.0, 122.0, 140.0, 158.0], [116.0, 137.0, 158.0, 179.0]], [[800.0, 848.0, 896.0, 944.0], [848.0, 899.0, 950.0, 1001.0], [896.0, 950.0, 1004.0, 1058.0], [944.0, 1001.0, 1058.0, 1115.0]]]
polyfit2 = [2.995846784242106, -1.9998660647480917, 0.501458842058525]
polyfit_cov = [[0.0007701419598584215, -0.0007701419598584212, 0.00011552129397876312], [-0.0007701419598584216, 0.0008302130327273779, -0.00014555683041324137], [0.00011552129397876328, -0.00014555683041324148, 3.835306960094923e-05]]
polyfit_w = [1.323976862244389, -0.14384443329681948]
unwrap = [-3.141592653589793, -1.841592653589793, -0.541592653589793, 0.7584073464102072, 2.058407346410207, 3.358407346410207, 4.658407346410208, 5.9584073464102065, 7.258407346410207]
unwrap_ax0 = [[0.0, 3.0, 6.0], [1.0, 0.7168146928204138, 8.283185307179586]]

@test
def test_r_c_():
    a = np.arange(3)
    assert _eq(np.r_[a, 0, 0:2], [0, 1, 2, 0, 0, 1])
    assert _eq(np.r_[2:5], [2, 3, 4])
    assert _close(np.r_[0:1:5j], [0.0, 0.25, 0.5, 0.75, 1.0])
    assert _close(np.c_[a, a * 1.5], [[0.0, 0.0], [1.0, 1.5], [2.0, 3.0]])
    assert _close(np.c_[np.ones((3, 2)), a], [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 2.0]])
    assert _close(np.r_[np.ones((2, 2)), np.zeros((1, 2))], [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    assert _close(np.r_["-1", np.ones((2, 2)), np.zeros((2, 1))], [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    assert _eq(np.r_[3], [3])
    assert _eq(np.c_[1, 2], [[1, 2]])
    y = np.array([1., 0., 1., 1.])
    X = np.zeros((4, 2, 5))
    e = np.c_[np.arange(len(y)) * X.shape[2], np.zeros(len(y), int), y.astype(int)]
    assert e.dtype is int and e.shape == (4, 3)
    assert _eq(e, [[0, 0, 1], [5, 0, 0], [10, 0, 1], [15, 0, 1]])

test_r_c_()

@test
def test_einsum():
    A = np.arange(12.).reshape(3, 4)
    B = np.arange(20.).reshape(4, 5)
    C = np.arange(9).reshape(3, 3)
    X = np.arange(24.).reshape(2, 3, 4)
    v = np.arange(4.)
    assert _close(np.einsum('ij,jk->ik', A, B), einsum_ijjk)
    assert _close(np.einsum('ij,jk', A, B), einsum_ijjk)
    assert np.einsum('ii', C) == 12
    assert _eq(np.einsum('ii->i', C), [0, 4, 8])
    assert _eq(np.einsum('ij->ji', C), C.T)
    assert np.einsum('i,i', v, v) == 14.0
    assert np.einsum('i,j->ij', v, v).shape == (4, 4)
    assert _close(np.einsum('...ij,...jk->...ik', X, X.transpose(0, 2, 1)), einsum_ell)
    assert _close(np.einsum('bij,bjk->bik', X, X.transpose(0, 2, 1)), einsum_ell)
    assert _close(np.einsum('ij,jk,kl->il', A, B, B.T), einsum_3op)
    assert _close(np.einsum('ij,ij->i', A, A), [14., 126., 366.])
    assert np.einsum('ijk->', X) == 276.0
    assert _close(np.einsum('cij,cik->cjk', X, X), einsum_cjk)
    assert _eq(np.einsum('ii,i->i', C, C[0]), [0, 4, 16])
    r = np.einsum('ij,jk->ik', C, C)
    assert r.dtype is int and _eq(r, [[15, 18, 21], [42, 54, 66], [69, 90, 111]])
    assert _close(np.einsum('ij,jk->ik', A, B, optimize=True), einsum_ijjk)

@test
def test_poly():
    assert np.polyval([1, 2, 3], 2) == 11
    assert _close(np.polyval(np.array([1., 2., 3.]), np.array([1., 2.])), [6., 11.])
    x = np.linspace(0., 1., 11)
    y = 3 * x * x - 2 * x + 0.5 + 0.01 * np.sin(17 * x)
    assert np.allclose(np.polyfit(x, y, 2), polyfit2, rtol=1e-9)
    c, V = np.polyfit(x, y, 2, cov=True)
    assert np.allclose(V, np.array(polyfit_cov), rtol=1e-8)
    assert np.allclose(np.polyfit(x, y, 1, w=x + 1.0), polyfit_w, rtol=1e-9)
    r = np.roots([1., -6., 11., -6.])
    assert np.allclose(np.sort(r.real), [1., 2., 3.]) and np.allclose(r.imag, 0.)
    assert _close(np.sort_complex(np.roots([1, 0, 1])), [-1j, 1j])
    assert _close(np.roots([1, 2, 0, 0]), [-2., 0., 0.])
    assert _close(np.poly([1., 2., 3.]), [1., -6., 11., -6.])
    assert _eq(np.polyder([1, 2, 3, 4]), [3, 4, 3])
    assert _close(np.polyder([1., 2, 3, 4], 2), [6., 4.])
    assert _close(np.polyint([3, 2, 1]), [1., 1., 1., 0.])
    assert _close(np.polyint([1., 1.], 2, k=[1., 2.]), [1. / 6, 0.5, 1., 2.])

@test
def test_misc_gaps():
    assert np.angle(1j) == np.pi / 2
    assert _close(np.angle([1 + 1j, -1 + 0j], deg=True), [45., 180.])
    assert np.angle(-2.0) == np.pi
    assert _close(np.real_if_close(np.array([2.1 + 4e-15j, 5.2 + 3e-15j])), [2.1, 5.2])
    ph = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8.]) * 1.3 % (2 * np.pi) - np.pi
    assert _close(np.unwrap(ph), unwrap)
    assert _close(np.unwrap(np.array([[0., 3., 6.], [1., 7., 2.]]), axis=0), unwrap_ax0)
    assert _close(np.unwrap(np.array([0, 5, 10, 15, 20]), period=6), [0, -1, -2, -3, -4])
    assert np.trapezoid([1., 2., 3.]) == 4.0
    assert _eq(np.cumulative_sum(np.array([1, 2, 3]), include_initial=True), [0, 1, 3, 6])
    assert _eq(np.cumulative_sum(np.array([1, 2, 3])), [1, 3, 6])

test_einsum()
test_poly()
test_misc_gaps()

@test
def test_audit_fixes():
    Y = np.array([[1., np.nan, 3.], [4., 5., np.nan]])
    assert _close(np.nanmax(Y, axis=1), [3., 5.])
    assert _close(np.nanmin(Y, axis=0), [1., 5., 3.])
    assert np.nanmax(Y) == 5.0
    assert _eq(np.nanargmax(Y, axis=1), [2, 1])
    assert _eq(np.nanargmin(Y, axis=1), [0, 0])
    assert np.isnan(np.nanmax(np.array([np.nan, np.nan])))
    assert _close(np.nan_to_num(Y), [[1., 0., 3.], [4., 5., 0.]])
    assert np.nan_to_num(np.inf) == 1.7976931348623157e308
    assert _eq(np.diff(np.array([1, 4, 9]), prepend=0), [1, 3, 5])
    assert _close(np.diff(np.array([1., 4., 9.]), append=np.array([0.])), [3., 5., -9.])
    assert _eq(np.diff(np.array([[1, 2], [4, 8]]), axis=1, append=0), [[1, -2], [4, -8]])

test_audit_fixes()

@test
def test_pyriemann_fixes():
    w = np.array([1., 2.])
    X = np.arange(18.).reshape(2, 3, 3)
    r = np.tensordot(w, X, axes=((0,), (0,)))
    assert r.shape == (3, 3) and _close(r, X[0] + 2 * X[1])
    assert np.tensordot(w, X, axes=(0, 0)).shape == (3, 3)
    assert X.mT[0, 0, 1] == 3.0 and X.mT.shape == (2, 3, 3)
    o = np.ones_like(np.zeros(()))
    assert o.ndim == 0 and o.item() == 1.0
    assert _close(X[..., np.array([0, 1]), np.array([1, 2])], [[1., 5.], [10., 14.]])

test_pyriemann_fixes()

@test
def test_followups():
    w = np.array([1., 2.])
    X = np.arange(18.).reshape(2, 3, 3)
    assert np.tensordot(w, X, axes=([0], [0])).shape == (3, 3)
    assert _close(np.tensordot(w, X, axes=[[0], [0]]), X[0] + 2 * X[1])
    assert np.tensordot(X, X, axes=[(0, 1), (0, 1)]).shape == (3, 3)
    assert np.tensordot(X, X, axes=([1], [2])).shape == (2, 3, 2, 3)
    x: float = np.interp(2.5, [1., 2., 3.], [0., 10., 20.])
    assert x == 15.0
    assert np.interp(np.array([[1.5]]), [1., 2.], [0., 1.]).shape == (1, 1)
    y: float = np.i0(1.0)
    assert abs(y - 1.2660658777520082) < 1e-14
    A = np.arange(24.).reshape(2, 3, 4)
    assert np.einsum('...j,...j->...', A, A).shape == (2, 3)
    assert np.einsum('i...->...', A).shape == (3, 4)
    assert _close(np.einsum('...ij,jk,k->...i', A, np.arange(8.).reshape(4, 2), np.ones(2))[1], [398., 510., 622.])
    assert np.einsum('ij,jk,ki->', np.ones((2, 3)), np.ones((3, 4)), np.ones((4, 2))) == 24.0
    assert np.einsum('i,j,k->ijk', w, w, w).shape == (2, 2, 2)
    assert _close(np.einsum('...ij,j->...i', A, np.ones(4))[1], [54., 70., 86.])
    assert np.einsum('ij...,j...->i...', np.ones((2, 3, 4)), np.ones((3, 1))).shape == (2, 4)

test_followups()

@test
def test_lazy_submodule_attrs():
    # np.<subpackage> without an explicit `import numpy.<subpackage>`
    assert np.polynomial.Polynomial([1.0, 2.0])(2.0) == 5.0
    np.testing.assert_allclose(np.ones(2), np.ones(2))
    np.testing.assert_raises(ValueError, int, 'x')

test_lazy_submodule_attrs()

def _rs_either(s: Optional[int]):
    if s is None:
        return np.random.RandomState()
    return np.random.RandomState(s)

@test
def test_lazy_submodule_init_on_every_branch():
    # regression: the lazy np.random loader was emitted only in the first
    # typechecked branch, so the other branch saw uninitialized module globals
    a = _rs_either(3).randint(0, 100)
    assert a == _rs_either(3).randint(0, 100)
    assert 0 <= _rs_either(None).randint(0, 100) < 100

test_lazy_submodule_init_on_every_branch()
