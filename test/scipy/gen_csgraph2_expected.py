# Copyright (C) 2026 the Typhon authors
# Prints the expected literals used by test_csgraph2.py (run with a CPython
# that has scipy 1.18.1).
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph as cg

def L(x):
    return repr(np.asarray(x).tolist())

def out(name, v):
    print(f"{name} = {L(v)}")

B = sp.csr_array(np.array([[1, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 0, 1],
                           [0, 0, 1, 1, 0], [0, 0, 0, 1, 1], [1, 0, 0, 0, 0]], dtype=float))
out("mbm_row", cg.maximum_bipartite_matching(B, perm_type='row'))
out("mbm_col", cg.maximum_bipartite_matching(B, perm_type='column'))
R = sp.random_array((12, 15), density=0.3, rng=np.random.default_rng(4)).tocsr()
out("R_d", R.toarray())
out("mbm_R", cg.maximum_bipartite_matching(R))
W = sp.csr_array(np.array([[5., 9., 1., 0.], [10., 3., 2., 7.], [8., 7., 4., 2.], [1., 6., 3., 8.]]))
r, c = cg.min_weight_full_bipartite_matching(W); out("mw_r", r); out("mw_c", c)
r, c = cg.min_weight_full_bipartite_matching(W, maximize=True); out("mwx_r", r); out("mwx_c", c)
Wt = sp.csr_array(np.array([[5., 9., 1.], [10., 3., 2.], [8., 7., 4.], [1., 6., 3.]]))
r, c = cg.min_weight_full_bipartite_matching(Wt); out("mwt_r", r); out("mwt_c", c)
Wr = sp.csr_array(np.random.default_rng(9).uniform(1, 10, (8, 11)))
out("Wr_d", Wr.toarray())
r, c = cg.min_weight_full_bipartite_matching(Wr); out("mwr_c", c)
G = sp.csr_array(np.array([[0, 3, 2, 0, 0, 0], [0, 0, 0, 4, 0, 0], [0, 1, 0, 2, 3, 0],
                           [0, 0, 0, 0, 2, 1], [0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0]], dtype=float))
d, p = cg.yen(G, 0, 5, 4, return_predecessors=True); out("yen_d", d); out("yen_p", p)
out("yen_u", cg.yen(G, 0, 5, 3, directed=False))
out("yen_uw", cg.yen(G, 0, 5, 5, unweighted=True))
Gn = G.copy(); Gn[2, 1] = -1.0
out("yen_neg", cg.yen(Gn, 0, 5, 3))
Gd = np.array([[0, 1., np.inf], [2, 0, 3], [np.nan, 4, 0]])
m = cg.csgraph_masked_from_dense(Gd)
out("mk_mask", m.mask); out("mk_from", cg.csgraph_from_masked(m).toarray())
t = cg.csgraph_to_masked(cg.csgraph_from_dense(Gd)); out("tm_mask", t.mask)
