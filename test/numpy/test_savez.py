import numpy as np
import os

@test
def test_savez():
    a = np.arange(6.0).reshape(2, 3)
    b = np.array([1, 2, 3])
    np.savez('/tmp/typhon_t_savez.npz', a, x=b)
    np.savez_compressed('/tmp/typhon_t_savez_c.npz', a, x=b)
    for p in ('/tmp/typhon_t_savez.npz', '/tmp/typhon_t_savez_c.npz'):
        with open(p, 'rb') as f:
            d = f.read()
        assert d[:4] == b'PK\x03\x04'
        assert d.find(b'arr_0.npy') > 0 and d.find(b'x.npy') > 0
        assert d.find(b'PK\x05\x06') > 0
        os.remove(p)

test_savez()

@test
def test_printoptions():
    with np.printoptions(precision=3):
        assert np.get_printoptions()['precision'] == 3
    assert np.get_printoptions()['precision'] == 8

test_printoptions()
