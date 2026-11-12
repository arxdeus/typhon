# Copyright (C) 2026 the Typhon authors
# Prints the expected literals used by test_sparse_eigen.py (run with a
# CPython that has scipy 1.18.1).
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sl

def L(x):
    a = np.asarray(x)
    if np.iscomplexobj(a):
        return repr([complex(v) for v in a.ravel()])
    return repr(a.tolist())

def out(name, v):
    print(f"{name} = {L(v)}")

n = 30
T = sp.diags_array([np.arange(1., n + 1), -np.ones(n - 1), -np.ones(n - 1)], offsets=[0, 1, -1]).tocsr()
for w in ['LM', 'SM', 'LA', 'SA', 'BE']:
    out(f"sh_{w}", sl.eigsh(T, k=4, which=w)[0])
    out(f"shn_{w}", sl.eigsh(T, k=4, which=w, return_eigenvectors=False))
out("sh_sig", sl.eigsh(T, k=3, sigma=10.2)[0])
out("shn_sig", sl.eigsh(T, k=3, sigma=10.2, return_eigenvectors=False))
rng = np.random.default_rng(0)
Bm = (sp.random_array((n, n), density=0.2, rng=rng) + sp.diags_array(np.arange(1., n + 1))).tocsr()
out("B_data", Bm.data); out("B_ind", Bm.indices); out("B_ptr", Bm.indptr)
for w in ['LM', 'SM', 'LR', 'SR']:
    out(f"e_{w}", sl.eigs(Bm, k=4, which=w)[0])
    out(f"en_{w}", sl.eigs(Bm, k=4, which=w, return_eigenvectors=False))
# complex pair ordering
R = sp.csr_array(np.array([[0., -2., 0., 0., 0., 0., 0., 0.], [2., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 3., 0., 0., 0., 0., 0.], [0., 0., 0., 1., -1., 0., 0., 0.],
                           [0., 0., 0., 1., 1., 0., 0., 0.], [0., 0., 0., 0., 0., 0.5, 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0.25, 0.], [0., 0., 0., 0., 0., 0., 0., 0.1]]))
out("rot_LM", sl.eigs(R, k=3, which='LM')[0])
# lobpcg
A = sp.diags_array([np.arange(1., 101.)], offsets=[0]).tocsr()
X = np.random.default_rng(1).standard_normal((100, 3))
out("X0", X)
lam, V = sl.lobpcg(A, X, largest=False, tol=1e-8, maxiter=200)
out("lob_small", lam)
lam, V = sl.lobpcg(A, X, largest=True, tol=1e-8, maxiter=200)
out("lob_large", lam)
lam, V, lh, rh = sl.lobpcg(A, X, largest=False, maxiter=5, retLambdaHistory=True, retResidualNormsHistory=True)
out("lob5", lam); print(f"lob5_nh = {len(lh)}"); out("lob5_h1", lh[1]); out("lob5_r2", rh[2])
Mp = sp.diags_array([1. / np.arange(1., 101.)], offsets=[0])
lam, V = sl.lobpcg(A, X, M=Mp, largest=False, tol=1e-8, maxiter=50)
out("lob_prec", lam)
Bg = sp.diags_array([np.full(100, 2.)], offsets=[0])
lam, V = sl.lobpcg(A, X, B=Bg, largest=False, tol=1e-8, maxiter=200)
out("lob_gen", lam)
Yc = np.eye(100, 2)
lam, V = sl.lobpcg(A, X, Y=Yc, largest=False, tol=1e-8, maxiter=200)
out("lob_con", lam)
S = sp.diags_array([np.arange(1., 11.)], offsets=[0])
lam, V = sl.lobpcg(S, np.random.default_rng(2).standard_normal((10, 3)), largest=True)
out("lob_dense", lam)
