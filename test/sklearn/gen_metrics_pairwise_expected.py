# Expected literals for test/sklearn/test_metrics_pairwise.py (scikit-learn 1.9.1).
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import pairwise as PW
from sklearn import metrics as M

def L(name, a):
    a = np.asarray(a, dtype=float).ravel()
    print(f"{name} = {[float(v) for v in a]}")

X = np.array([[((i * 7 + j * 13) % 11) / 3.0 - 1.0 for j in range(4)] for i in range(7)])
Y = np.array([[((i * 5 + j * 3) % 7) / 2.0 - 0.5 for j in range(4)] for i in range(5)])
P = np.abs(X) + 0.1; Q = np.abs(Y) + 0.2
L("eu", PW.euclidean_distances(X, Y)); L("eu_xx", PW.euclidean_distances(X)); L("eu_sq", PW.euclidean_distances(X, Y, squared=True))
Xn = X.copy(); Xn[1, 2] = np.nan; Xn[4, 0] = np.nan; Yn = Y.copy(); Yn[0, 1] = np.nan
L("nan_eu", PW.nan_euclidean_distances(Xn, Yn)); L("nan_eu_xx", PW.nan_euclidean_distances(Xn))
L("man", PW.manhattan_distances(X, Y)); L("cos_s", PW.cosine_similarity(X, Y)); L("cos_d", PW.cosine_distances(X))
H = np.array([[0.1, 0.2], [0.5, -1.0], [-0.3, 2.0]]); L("hav", PW.haversine_distances(H))
L("p_eu", PW.paired_euclidean_distances(X[:5], Y)); L("p_man", PW.paired_distances(X[:5], Y, metric="manhattan")); L("p_cos", PW.paired_cosine_distances(X[:5], Y))
for m in ["cityblock", "cosine", "euclidean", "l1", "chebyshev", "minkowski", "canberra", "braycurtis", "sqeuclidean", "correlation", "hamming", "jaccard"]:
    L("pd_" + m, PW.pairwise_distances(X, Y, metric=m))
L("pd_mink3", PW.pairwise_distances(X, metric="minkowski", p=3)); L("pd_eu_xx", PW.pairwise_distances(X))
for k in ["linear", "poly", "rbf", "laplacian", "sigmoid", "cosine"]:
    L("k_" + k, PW.pairwise_kernels(X, Y, metric=k))
L("k_rbf_g", PW.rbf_kernel(X, gamma=0.3)); L("k_poly2", PW.polynomial_kernel(X, Y, degree=2, gamma=0.5, coef0=0.0))
L("k_chi2", PW.chi2_kernel(P, Q, gamma=0.7)); L("k_achi2", PW.additive_chi2_kernel(P, Q))
a, b = PW.pairwise_distances_argmin_min(X, Y); L("amin_i", a); L("amin_v", b)
L("amin0", PW.pairwise_distances_argmin(X, Y, axis=0, metric="manhattan"))
