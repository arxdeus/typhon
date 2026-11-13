# Generates expected literals for test/sklearn/test_final_utils.py (scikit-learn 1.9.1).
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import scipy.sparse as sp
from sklearn.utils.sparsefuncs import incr_mean_variance_axis, mean_variance_axis
from sklearn.utils.sparsefuncs_fast import csr_row_norms

def L(name, a):
    print(f"{name} = {[float(v) for v in np.asarray(a, dtype=float).ravel()]!r}")

n = 12
A = np.zeros((n, 4))
for i in range(n):
    for j in range(4):
        v = (i * 7919 + j * 104729 + i * j * 13) % 7
        A[i, j] = float(v) if v > 2 else 0.0
S1 = sp.csr_matrix(A[:5]); S2 = sp.csr_matrix(A[5:]); S3 = sp.csc_matrix(A[5:])
m, v, c = incr_mean_variance_axis(S1, axis=0, last_mean=np.zeros(4), last_var=np.zeros(4), last_n=0)
L("a_m", m); L("a_v", v); L("a_n", c)
m, v, c = incr_mean_variance_axis(S2, axis=0, last_mean=m, last_var=v, last_n=c)
L("b_m", m); L("b_v", v); L("b_n", c)
m, v, c = incr_mean_variance_axis(S3, axis=0, last_mean=m, last_var=v, last_n=c)
L("c_m", m); L("c_v", v); L("c_n", c)
m, v, c = incr_mean_variance_axis(S2, axis=1, last_mean=np.arange(7.0), last_var=np.ones(7), last_n=3)
L("r_m", m); L("r_v", v); L("r_n", c)
L("rn", csr_row_norms(S1))
