# Copyright (C) 2026 the Typhon authors
# Prints the expected literals used by test_sparse_lu.py (run with a CPython
# that has scipy 1.18.1).
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sl

def L(x):
    return repr(np.asarray(x).tolist())

def out(name, v):
    print(f"{name} = {L(v)}")

def poisson2d(k):
    T = sp.diags_array([4. * np.ones(k), -np.ones(k - 1), -np.ones(k - 1)], offsets=[0, 1, -1])
    E = sp.diags_array([-np.ones(k - 1), -np.ones(k - 1)], offsets=[1, -1])
    return (sp.kron(sp.eye_array(k), T) + sp.kron(E, sp.eye_array(k))).tocsc()

def randmat(n, seed):
    rng = np.random.default_rng(seed)
    A = sp.random_array((n, n), density=0.15, rng=rng, format='csc')
    return (A + sp.diags_array(rng.uniform(0.0, 0.5, n))).tocsc()

def pivmat():
    # needs row interchanges (small diagonal)
    D = np.array([[1e-3, 2., 0., 0., 1.], [3., 1e-3, 1., 0., 0.], [0., 4., 2., 5., 0.],
                  [1., 0., 0., 1e-3, 2.], [0., 0., 3., 1., 1e-3]])
    return sp.csc_array(D)

mats = {"P": poisson2d(5), "R": randmat(30, 7), "V": pivmat()}
R = mats["R"]
out("R_data", R.data); out("R_ind", R.indices); out("R_ptr", R.indptr)
for name, A in mats.items():
    b = np.arange(1., A.shape[0] + 1)
    for spec in ["COLAMD", "NATURAL", "MMD_ATA", "MMD_AT_PLUS_A"]:
        lu = sl.splu(A, permc_spec=spec)
        tag = f"{name}_{spec}"
        out(f"pc_{tag}", lu.perm_c); out(f"pr_{tag}", lu.perm_r)
        print(f"lnnz_{tag} = {lu.L.nnz}"); print(f"unnz_{tag} = {lu.U.nnz}")
        out(f"x_{tag}", lu.solve(b))
    lu = sl.splu(A)
    out(f"Ld_{name}", lu.L.toarray()); out(f"Ud_{name}", lu.U.toarray())
    lu = sl.splu(A, diag_pivot_thresh=0.1)
    out(f"pr_th_{name}", lu.perm_r)
    lu = sl.splu(A, options=dict(SymmetricMode=True))
    out(f"pc_sym_{name}", lu.perm_c); out(f"pr_sym_{name}", lu.perm_r)
    out(f"xs_{name}", sl.spsolve(A, b, permc_spec="MMD_AT_PLUS_A"))
A = mats["P"]
print(f"tri_P = {sl.is_sptriangular(A)!r}"); print(f"bw_P = {sl.spbandwidth(A)!r}")
T = sp.tril(A, format='csr')
print(f"tri_T = {sl.is_sptriangular(T)!r}"); print(f"bw_T = {sl.spbandwidth(T)!r}")
