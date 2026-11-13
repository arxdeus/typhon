import numpy as np

@test
def test_empty_fancy_index():
    X = np.arange(12.0).reshape(3, 4)
    e = np.array([], dtype=int)
    assert X[:, e].shape == (3, 0)
    assert X[e].shape == (0, 4)
    assert X[e, :].shape == (0, 4)
    assert X[np.zeros(3, dtype=bool)].shape == (0, 4)
    assert X[:, np.zeros(4, dtype=bool)].shape == (3, 0)
    assert X[..., e].shape == (3, 0)
    assert X[1, e].shape == (0,)
    Y = np.zeros((0, 4))
    assert Y[:, np.array([1, 2])].shape == (0, 2)

test_empty_fancy_index()
