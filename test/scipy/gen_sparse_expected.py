# Copyright (C) 2026 the Typhon authors
# Prints the expected literals used by test_sparse.py, test_sparse_linalg.py
# and test_csgraph.py (run with a CPython that has scipy 1.18.1).
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sl
from scipy.sparse import csgraph as cg

def L(x):
    a = np.asarray(x)
    if np.iscomplexobj(a):
        return repr([complex(v) for v in a.ravel()]) if a.ndim <= 1 else \
            repr([[complex(v) for v in r] for r in a])
    return repr(a.tolist())

def out(name, v):
    print(f"{name} = {L(v)}")

# ---------------------------------------------------------------- sparse
A = np.array([[1., 0., 2., 0.], [0., 0., 3., 0.], [4., 5., 0., 0.], [0., 0., 0., 6.]])
S = sp.csr_array(A)
B = sp.csr_array(np.array([[0., 1., 0., 0.], [2., 0., 0., 1.], [0., 0., 7., 0.], [1., 0., 0., 1.]]))
print("# sparse")
out("ss_mm", (S @ B).toarray())
out("ss_mm_ind", (S @ B).indices)
out("ss_mm_ptr", (S @ B).indptr)
out("ss_add", (S + B).toarray())
out("ss_sub_data", (S - B).data)
out("ss_mul", S.multiply(B).toarray())
out("ss_mv", S @ np.array([1., 2., 3., 4.]))
out("ss_rmv", np.array([1., 2., 3., 4.]) @ S)
out("ss_sum0", S.sum(axis=0))
out("ss_sum1", S.sum(axis=1))
out("ss_diag1", S.diagonal(1))
out("ss_csc_ind", S.tocsc().indices)
out("ss_csc_ptr", S.tocsc().indptr)
U = sp.csr_array((np.array([1., 2., 3., 4.]), np.array([2, 0, 1, 0]), np.array([0, 2, 4])), shape=(2, 3))
V = sp.csr_array((np.array([5., 6., 7.]), np.array([1, 0, 2]), np.array([0, 2, 3])), shape=(2, 3))
out("un_add_ind", (U + V).indices)
out("un_add_data", (U + V).data)
out("un_mm_ind", (U @ V.T).indices)
C = sp.coo_array((np.array([1., 2., 3., 4.]), (np.array([0, 0, 2, 0]), np.array([1, 1, 0, 3]))), shape=(3, 4))
out("coo_dense", C.toarray())
out("coo_csr_data", C.tocsr().data)
out("eye_k1", sp.eye(4, 5, k=1).toarray())
out("diags_d", sp.diags_array([np.array([1., 2., 3.]), np.array([4., 5.])], offsets=[0, -1]).toarray())
out("spdiags_d", sp.spdiags(np.array([[1., 2., 3.], [4., 5., 6.]]), [0, 1], 3, 3).toarray())
out("kron_d", sp.kron(sp.csr_array(np.array([[1., 2.], [0., 3.]])), sp.csr_array(np.array([[0., 1.], [1., 0.]]))).toarray())
out("hstack_d", sp.hstack([S, B]).toarray())
out("vstack_shape", list(sp.vstack([S, B]).shape))
out("bdiag_d", sp.block_diag([sp.csr_array(np.array([[1., 2.]])), sp.csr_array(np.array([[3.], [4.]]))]).toarray())
r, c, v = sp.find(S)
out("find_r", r); out("find_c", c); out("find_v", v)
out("tril_d", sp.tril(S, -1).toarray())
out("triu_d", sp.triu(S, 1).toarray())
R = sp.random(5, 6, density=0.4, rng=3)
out("rand_d", R.toarray())
X = sp.csr_array(np.array([[1 + 2j, 0j], [3j, 4 + 0j]]))
out("cx_mv", X @ np.array([1 + 1j, 2 + 0j]))
out("cx_H", X.conj().T.toarray())
out("int_sum", sp.csr_array(np.array([[1, 0], [2, 3]])).sum(axis=0))
print()

# ---------------------------------------------------------------- linalg
print("# linalg")
n = 30
d0 = np.arange(1., n + 1)
T = sp.diags_array([d0 + 2, -np.ones(n - 1), -np.ones(n - 1)], offsets=[0, 1, -1]).tocsr()
N = sp.diags_array([d0 + 2, -2 * np.ones(n - 1), np.ones(n - 1), 0.5 * np.ones(n - 3)], offsets=[0, 1, -1, 3]).tocsr()
b = np.sin(np.arange(n) + 1.0)
for name, f, M in [("cg", sl.cg, T), ("bicgstab", sl.bicgstab, N), ("gmres", sl.gmres, N),
                   ("minres", sl.minres, T), ("cgs", sl.cgs, N), ("bicg", sl.bicg, N)]:
    x, info = f(M, b, rtol=1e-10)
    out(f"x_{name}", x)
x, info = sl.gmres(N, b, rtol=1e-10, restart=5)
out("x_gmres5", x)
out("x_spsolve", sl.spsolve(N, b))
res = sl.lsqr(N, b, atol=1e-12, btol=1e-12)
out("x_lsqr", res[0]); print("lsqr_itn =", res[2])
res = sl.lsmr(N, b, atol=1e-12, btol=1e-12)
out("x_lsmr", res[0]); print("lsmr_itn =", res[2])
for w in ['LM', 'SM', 'LA', 'SA', 'BE']:
    out(f"eigsh_{w}", np.sort(sl.eigsh(T, k=4, which=w)[0]))
out("eigsh_sigma", np.sort(sl.eigsh(T, k=3, sigma=10.2)[0]))
E = sp.diags_array([d0, -2 * np.ones(n - 1), np.ones(n - 1)], offsets=[0, 1, -1]).tocsr()
ev = sl.eigs(E, k=4, which='LM')[0]
out("eigs_LM", sorted(ev, key=lambda z: (round(z.real, 8), round(z.imag, 8))))
ev = sl.eigs(E, k=3, which='SR')[0]
out("eigs_SR", sorted(ev, key=lambda z: (round(z.real, 8), round(z.imag, 8))))
R = sp.csr_array(np.array([[(i * 7 + j * 3) % 11 - 5.0 if (i + j) % 3 else 0. for j in range(12)] for i in range(20)]))
out("svds_s", np.sort(sl.svds(R, k=3)[1]))
out("norm_fro", sl.norm(N)); out("norm_1", sl.norm(N, 1)); out("norm_inf", sl.norm(N, np.inf))
out("norm_ax0", sl.norm(N, axis=0)[:5])
out("expm_mul", sl.expm_multiply(-0.1 * T, b))
lu = sl.splu(sp.csc_array(N))
out("splu_x", lu.solve(b)); out("splu_xT", lu.solve(b, trans='T'))
out("inv_d", sl.inv(sp.csc_array(np.array([[4., 1.], [2., 3.]]))).toarray())
Lt = sp.csr_array(np.tril(np.array([[2., 0, 0], [1, 3, 0], [4, 5, 6]])))
out("tri_x", sl.spsolve_triangular(Lt, np.array([1., 2., 3.])))
print()

# ---------------------------------------------------------------- csgraph
print("# csgraph")
rng = np.random.default_rng(7)
Gd = np.zeros((9, 9))
for i in range(9):
    for j in range(9):
        if i != j and (i * 5 + j * 3) % 7 < 3:
            Gd[i, j] = float((i + 2 * j) % 5 + 1)
Gs = sp.csr_array(Gd)
for m in ['D', 'BF', 'FW', 'J']:
    for directed in [True, False]:
        d, p = cg.shortest_path(Gs, method=m, directed=directed, return_predecessors=True)
        out(f"sp_{m}_{int(directed)}_d", d)
        out(f"sp_{m}_{int(directed)}_p", p)
d, p = cg.shortest_path(Gs, unweighted=True, return_predecessors=True)
out("sp_unw_d", d); out("sp_unw_p", p)
d, p, s = cg.dijkstra(Gs, indices=[0, 4], min_only=True, return_predecessors=True)
out("dj_min_d", d); out("dj_min_p", p); out("dj_min_s", s)
out("dj_lim", cg.dijkstra(Gs, indices=2, limit=4.0))
Neg = Gd.copy(); Neg[1, 0] = -1.0; Neg[3, 2] = -0.5
NegS = sp.csr_array(Neg)
d, p = cg.johnson(NegS, return_predecessors=True)
out("jo_d", d); out("jo_p", p)
d, p = cg.bellman_ford(NegS, indices=[1, 3], return_predecessors=True)
out("bf_d", d); out("bf_p", p)
Cg = np.zeros((10, 10))
for (i, j) in [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3), (6, 7), (8, 8), (7, 6), (9, 6)]:
    Cg[i, j] = 1.0
Cs = sp.csr_array(Cg)
n_, lab = cg.connected_components(Cs, directed=True, connection='strong')
print("scc_n =", n_); out("scc_l", lab)
n_, lab = cg.connected_components(Cs, directed=True, connection='weak')
print("wcc_n =", n_); out("wcc_l", lab)
for directed in [True, False]:
    o, p = cg.breadth_first_order(Gs, 3, directed=directed)
    out(f"bfo_{int(directed)}_o", o); out(f"bfo_{int(directed)}_p", p)
    o, p = cg.depth_first_order(Gs, 3, directed=directed)
    out(f"dfo_{int(directed)}_o", o); out(f"dfo_{int(directed)}_p", p)
out("bft_d", cg.breadth_first_tree(Gs, 0, directed=False).toarray())
out("dft_d", cg.depth_first_tree(Gs, 0, directed=True).toarray())
out("mst_d", cg.minimum_spanning_tree(Gs).toarray())
out("rcm", cg.reverse_cuthill_mckee(Gs))
out("rcm_sym", cg.reverse_cuthill_mckee(sp.csr_array(Gd + Gd.T), symmetric_mode=True))
out("lap_d", cg.laplacian(Gd))
out("lap_n", cg.laplacian(Gd, normed=True))
L_, dg = cg.laplacian(Gs, normed=True, return_diag=True)
out("lap_sn", L_.toarray()); out("lap_sdiag", dg)
out("lap_out", cg.laplacian(Gd, use_out_degree=True))
Fc = sp.csr_array((Gd * 2).astype(np.int32))
r = cg.maximum_flow(Fc, 0, 8)
print("mf_v =", r.flow_value); out("mf_f", r.flow.toarray())
r = cg.maximum_flow(Fc, 0, 8, method='edmonds_karp')
out("mf_ek", r.flow.toarray())
out("cdm", cg.construct_dist_matrix(Gs, cg.shortest_path(Gs, return_predecessors=True)[1]))
print("srank =", cg.structural_rank(Cs))
