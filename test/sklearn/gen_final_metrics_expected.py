# Generates expected literals for test/sklearn/test_final_metrics.py (scikit-learn 1.9.1).
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import DistanceMetric as DM
from sklearn.metrics import pairwise as PW

def L(name, a):
    a = np.asarray(a, dtype=float).ravel()
    print(f"{name} = {[float(v) for v in a]}")

X = np.array([[((i * 7 + j * 13) % 11) / 3.0 - 1.0 for j in range(4)] for i in range(6)])
Y = np.array([[((i * 5 + j * 3) % 7) / 2.0 - 0.5 for j in range(4)] for i in range(4)])
B = (X > 0).astype(float); C = (Y > 0.2).astype(float)
H = np.array([[0.1, 0.2], [0.5, -1.0], [-0.3, 2.0], [0.9, 0.4]])
V = np.array([1.0, 2.0, 0.5, 3.0])
S = np.cov(X.T) + np.eye(4)
w = np.array([1.0, 0.5, 2.0, 1.5])
for m in ["euclidean", "manhattan", "chebyshev", "canberra", "braycurtis"]:
    L("dm_" + m, DM.get_metric(m).pairwise(X, Y))
L("dm_mink3", DM.get_metric("minkowski", p=3).pairwise(X, Y))
L("dm_minkw", DM.get_metric("minkowski", p=1.5, w=w).pairwise(X))
L("dm_seu", DM.get_metric("seuclidean", V=V).pairwise(X, Y))
L("dm_mah", DM.get_metric("mahalanobis", V=S).pairwise(X, Y))
L("dm_mah_vi", DM.get_metric("mahalanobis", VI=np.linalg.inv(S)).pairwise(X))
L("dm_hav", DM.get_metric("haversine").pairwise(H))
for m in ["hamming", "jaccard", "dice", "matching", "kulsinski", "rogerstanimoto", "russellrao", "sokalmichener", "sokalsneath"]:
    L("dm_" + m, DM.get_metric(m).pairwise(B, C))
d = DM.get_metric("minkowski", p=3)
L("r2d", d.rdist_to_dist(np.array([1.0, 8.0, 27.0]))); L("d2r", d.dist_to_rdist(np.array([1.0, 2.0, 3.0])))
h = DM.get_metric("haversine")
L("h_r2d", h.rdist_to_dist(np.array([0.1, 0.5]))); L("h_d2r", h.dist_to_rdist(np.array([0.3, 1.2])))
e = DM.get_metric("euclidean")
L("e_r2d", e.rdist_to_dist(np.array([4.0, 9.0]))); L("e_d2r", e.dist_to_rdist(np.array([2.0, 3.0])))
Xs = sp.csr_matrix(X * (X > 0)); Ys = sp.csr_matrix(Y * (Y > 0.3))
L("sp_pwd_eu", PW.pairwise_distances(Xs, Ys)); L("sp_eu", PW.euclidean_distances(Xs))
L("sp_man", PW.manhattan_distances(Xs, Ys)); L("sp_cos", PW.cosine_similarity(Xs, Ys))
L("sp_cosd", PW.cosine_distances(Xs)); L("sp_rbf", PW.rbf_kernel(Xs, Ys, gamma=0.4))
L("sp_lin", PW.linear_kernel(Xs, Ys)); L("sp_poly", PW.polynomial_kernel(Xs, Ys, degree=2))
L("sp_dm", DM.get_metric("manhattan").pairwise(Xs, Ys))
