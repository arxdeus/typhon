# Generates expected literals used in test/numpy/test_gaps.py (numpy 2.x).
import numpy as np
A = np.arange(12.).reshape(3,4); B = np.arange(20.).reshape(4,5); C = np.arange(9).reshape(3,3)
X = np.arange(24.).reshape(2,3,4)
x = np.linspace(0., 1., 11); y = 3*x*x - 2*x + 0.5 + 0.01*np.sin(17*x)
ph = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8.]) * 1.3 % (2*np.pi) - np.pi
out = {
 "einsum_ijjk": np.einsum('ij,jk->ik', A, B).tolist(),
 "einsum_ell": np.einsum('...ij,...jk->...ik', X, X.transpose(0,2,1)).tolist(),
 "einsum_3op": np.einsum('ij,jk,kl->il', A, B, B.T).tolist(),
 "einsum_cjk": np.einsum('cij,cik->cjk', X, X).tolist(),
 "polyfit2": np.polyfit(x, y, 2).tolist(),
 "polyfit_cov": np.polyfit(x, y, 2, cov=True)[1].tolist(),
 "polyfit_w": np.polyfit(x, y, 1, w=x+1.0).tolist(),
 "unwrap": np.unwrap(ph).tolist(),
 "unwrap_ax0": np.unwrap(np.array([[0., 3., 6.],[1., 7., 2.]]), axis=0).tolist(),
}
for k, v in out.items():
    print(f"{k} = {v!r}")
