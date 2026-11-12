# Copyright (C) 2026 the Typhon authors
# Prints the expected literals used by test_sparse_linalg2.py (run with a
# CPython that has scipy 1.18.1).
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sl

def L(x):
    return repr(np.asarray(x).tolist())

def out(name, v):
    print(f"{name} = {L(v)}")

n = 30
A = sp.diags_array([np.full(n, 4.), np.full(n - 1, -1.), np.full(n - 1, -2.)], offsets=[0, 1, -1]).tocsr()
b = np.cos(np.arange(n))
for name, f in [("qmr", sl.qmr), ("tfqmr", sl.tfqmr), ("lgmres", sl.lgmres), ("gcrotmk", sl.gcrotmk)]:
    x, info = f(A, b, rtol=1e-10)
    out(f"x_{name}", x); print(f"info_{name} = {info}")
    it = [0]
    def cb(xk):
        it[0] += 1
    f(A, b, rtol=1e-10, callback=cb)
    print(f"it_{name} = {it[0]}")
M = sp.diags_array([1. / np.full(n, 4.)], offsets=[0])
x, info = sl.tfqmr(A, b, rtol=1e-10, M=M); out("x_tfqmr_M", x)
x, info = sl.lgmres(A, b, rtol=1e-10, M=M, inner_m=5, outer_k=2); out("x_lgmres_M", x)
x, info = sl.gcrotmk(A, b, rtol=1e-10, m=5, k=3); out("x_gcrotmk_mk", x)
x, info = sl.qmr(A, b, rtol=1e-10, maxiter=5); out("x_qmr5", x); print(f"info_qmr5 = {info}")
x, info = sl.lgmres(A, b, rtol=1e-10, inner_m=4, maxiter=3); out("x_lgmres3", x); print(f"info_lgmres3 = {info}")
# expm on sparse input
E = sp.csc_array(np.array([[0.1, 0.2, 0.], [0., -0.3, 0.1], [0.05, 0., 0.2]]))
out("expm_small", sl.expm(E).toarray())
E2 = sp.csc_array(np.array([[1., 2., 0., 0.], [0., -1., 3., 0.], [0., 0., 0.5, 1.], [0., 0., 0., 2.]]))
out("expm_upper", sl.expm(E2).toarray())
E3 = (sp.random_array((12, 12), density=0.3, rng=np.random.default_rng(5)) * 3.0).tocsc()
out("E3_d", E3.toarray())
out("expm_big", sl.expm(E3).toarray())
print(f"expm_fmt = {sl.expm(sp.csr_array(E)).format!r}")
# onenormest
np.random.seed(1234)
G = sp.csr_array(np.random.default_rng(3).standard_normal((40, 40)))
out("G_d", G.toarray())
np.random.seed(7)
print(f"one_est = {sl.onenormest(G)!r}")
np.random.seed(7)
e, v, w = sl.onenormest(G, t=3, compute_v=True, compute_w=True)
print(f"one_est3 = {e!r}"); out("one_v3", v); out("one_w3", w)
print(f"one_exact = {sl.onenormest(sp.csr_array(np.eye(3) * 2.), t=3)!r}")
