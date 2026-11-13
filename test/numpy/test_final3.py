import numpy as np

@test
def test_ma_leftovers():
    m = np.ma.masked_array([1., 2., 3., 4.], mask=[False, True, False, False])
    r = np.ma.append(m, [5., 6.])
    assert r.filled(-1.0).tolist() == [1.0, -1.0, 3.0, 4.0, 5.0, 6.0]
    assert np.ma.in1d(m, [1., 4.]).filled(False).tolist() == [True, False, False, True]
    q = np.ma.round_(np.ma.masked_array([1.26, 2.5], mask=[False, True]), 1)
    assert q.filled(0.0).tolist() == [1.3, 0.0]

test_ma_leftovers()

@test
def test_polynomial_printstyle():
    np.polynomial.set_default_printstyle('unicode')
    np.polynomial.set_default_printstyle('ascii')

test_polynomial_printstyle()
