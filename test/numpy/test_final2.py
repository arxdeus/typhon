import numpy as np

@test
def test_npz_roundtrip():
    a = np.arange(6.0).reshape(2, 3)
    np.savez('/tmp/typhon_f2.npz', a, x=np.array([1.0, 2.0]))
    z = np.load_npz('/tmp/typhon_f2.npz')
    assert z.files == ['arr_0', 'x']
    assert z.get('arr_0', ndim=2).tolist() == a.tolist()
    assert z.get('x').tolist() == [1.0, 2.0]

test_npz_roundtrip()

@test
def test_strings_codec():
    x = np.array(['ab', 'c'])
    assert np.strings.decode(np.strings.encode(x)).tolist() == ['ab', 'c']
    assert np.char.str_len(x).tolist() == [2, 1]

test_strings_codec()
