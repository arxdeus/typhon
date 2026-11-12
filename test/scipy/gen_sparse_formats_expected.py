# Copyright (C) 2026 the Typhon authors
# Prints the expected literals used by test_sparse_formats.py (run with a
# CPython that has scipy 1.18.1).
import numpy as np
import scipy.sparse as sp

def L(x):
    return repr(np.asarray(x).tolist())

def out(name, v):
    print(f"{name} = {L(v)}")

A = np.array([[1., 0., 2., 0.], [0., 0., 3., 0.], [4., 5., 0., 0.], [0., 0., 0., 6.]])
S = sp.csr_array(A)
D = S.todia()
out("dia_data", D.data); out("dia_off", D.offsets); print(f"dia_nnz = {D.nnz}")
out("dia_T_data", D.T.data); out("dia_T_off", D.T.offsets)
D2 = sp.diags_array([np.array([1., 2., 3.]), np.array([4., 5.])], offsets=[0, -1])
out("diags_data", D2.data); out("diags_off", D2.offsets)
DS = D + D2[0:4, 0:4] if False else D + sp.dia_array(np.eye(4))
out("dia_add_data", DS.data); out("dia_add_off", DS.offsets)
DM = D @ D
out("dia_mm_data", DM.data); out("dia_mm_off", DM.offsets)
out("dia_mv", D @ np.array([1., 2., 3., 4.]))
out("dia_sum0", D.sum(axis=0)); out("dia_sum1", D.sum(axis=1))
E = sp.eye(3, 4, k=1)
print(f"eye_fmt = {E.format!r}"); out("eye_data", E.data); out("eye_off", E.offsets)
B = S.tobsr((2, 2))
out("bsr_data", B.data); out("bsr_ind", B.indices); out("bsr_ptr", B.indptr)
print(f"bsr_est = {S.tobsr().blocksize!r}")
G = sp.csr_array(np.kron(np.eye(3), np.array([[1., 2.], [3., 4.]])) + np.diag(np.arange(1., 7.)))
print(f"bsr_est2 = {G.tobsr().blocksize!r}")
BT = B.T
out("bsr_T_data", BT.data); out("bsr_T_ind", BT.indices); out("bsr_T_ptr", BT.indptr)
BB = B @ B
out("bsr_mm_data", BB.data); out("bsr_mm_ind", BB.indices); out("bsr_mm_ptr", BB.indptr)
BC = B @ S
print(f"bsr_csr_bs = {BC.blocksize!r}"); out("bsr_csr_mm", BC.toarray())
out("bsr_add_data", (B + B).data)
out("bsr_mv", B @ np.array([1., 2., 3., 4.]))
out("bsr_mm2", B @ np.arange(8.).reshape(4, 2))
out("bsr_diag1", B.diagonal(1)); out("bsr_diagm1", B.diagonal(-1))
Bc = B.tocsr()
out("bsr_csr_ind", Bc.indices); out("bsr_csr_data", Bc.data)
K = sp.kron(sp.csr_array(np.array([[1., 2.], [0., 3.]])), np.array([[0., 1.], [1., 0.]]))
print(f"kron_fmt = {K.format!r}"); out("kron_data", K.data)
print(f"kron_coo_fmt = {sp.kron(sp.eye(2), sp.eye(2)).format!r}")
Li = S.tolil()
Li[0, 1] = 7.
Li[1:3, 2] = 9.
Li[3, [0, 1]] = np.array([1., 2.])
Li[2, 0] = 0.
print(f"lil_rows = {[list(r) for r in Li.rows]!r}"); print(f"lil_data = {[[float(v) for v in r] for r in Li.data]!r}")
out("lil_fancy", Li[[3, 0]].toarray())
out("lil_slice", Li[1:4, 0:3].toarray())
Li2 = sp.lil_array((3, 3))
Li2[:, 1] = 2.
Li2[0, :] = np.array([1., 0., 3.])
out("lil2", Li2.toarray())
Dk = sp.dok_array((3, 4))
Dk[1, 2] = 5.
Dk[0, 3] = 1.
Dk[2, :] = 2.
Dk[0, 3] = 0.
out("dok_d", Dk.toarray()); print(f"dok_nnz = {Dk.nnz}")
c = Dk.tocoo(); out("dok_row", c.row); out("dok_col", c.col)
out("dok_mv", Dk @ np.array([1., 2., 3., 4.]))
R = S.reshape((2, 8)); out("reshape_d", R.toarray())
RF = S.reshape((8, 2), order='F'); out("reshapeF_d", RF.toarray())
for f in ['csr', 'csc', 'coo', 'dia', 'bsr', 'lil', 'dok']:
    X = S.copy().asformat(f)
    X.resize((5, 3))
    out(f"resize_{f}", X.toarray())
    if f == 'bsr':
        continue  # scipy 1.18.1 bsr setdiag raises AttributeError (_swap)
    Y = S.copy().asformat(f)
    Y.setdiag(np.array([9., 8.]), -1)
    out(f"setdiag_{f}", Y.toarray())
print(f"diag_fmt = {sp.diags([1., 2.]).format!r}")
print(f"spdiag_fmt = {sp.spdiags(np.ones((1, 3)), [0], 3, 3).format!r}")
print(f"bdiag_fmt = {sp.block_diag([sp.eye(2)]).format!r}")
print(f"hstack_csc = {sp.hstack([sp.csc_array(A)] * 2).format!r}")
print(f"hstack_csr = {sp.hstack([S] * 2).format!r}")
print(f"vstack_coo = {sp.vstack([sp.coo_array(A)] * 2).format!r}")
print(f"rand_dia = {sp.random(4, 4, density=0.5, format='dia', rng=1).format!r}")
out("rand_lil", sp.random(4, 5, density=0.5, format='lil', rng=2).toarray())
