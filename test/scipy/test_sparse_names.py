# Ported from scipy 1.18.1 (BSD-3-Clause)
import numpy as np
import scipy.sparse as sp

@test
def test_sparse_names():
    A = sp.csr_array(np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]]))
    assert sp.matrix_transpose(A).shape == (3, 2)
    assert sp.permute_dims(A, (1, 0)).shape == (3, 2)
    assert sp.permute_dims(A, (-1, -2)).shape == (3, 2)
    assert sp.swapaxes(A, 0, 1).shape == (3, 2)
    assert sp.swapaxes(A, -2, -1).toarray()[2, 0] == 2.0
    w = sp.SparseEfficiencyWarning("x")
    assert isinstance(w, sp.SparseWarning) and isinstance(w, Warning)
    t = sp.get_index_dtype(maxval=10)

test_sparse_names()
