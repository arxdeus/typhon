# Generates the expected literals for test/sklearn/test_datasets.py (scikit-learn 1.9.1).
#   <venv>/bin/python test/sklearn/gen_datasets_expected.py
import numpy as np
from sklearn import datasets as D

def L(name, a):
    print(f"{name} = {[float(v) for v in np.asarray(a, dtype=float).ravel()]!r}")

X, y = D.make_classification(n_samples=40, n_features=6, n_informative=3, n_redundant=1, n_repeated=1, n_classes=3, n_clusters_per_class=1, random_state=0)
L("cls_X", X); L("cls_y", y)
X, y = D.make_classification(n_samples=30, n_features=5, weights=[0.3], flip_y=0.2, hypercube=False, shift=None, scale=None, random_state=3)
L("cls2_X", X); L("cls2_y", y)
X, y, c = D.make_regression(n_samples=20, n_features=5, n_informative=3, noise=0.5, bias=1.0, coef=True, random_state=1)
L("reg_X", X); L("reg_y", y); L("reg_c", c)
X, y = D.make_regression(n_samples=15, n_features=6, effective_rank=2, random_state=2)
L("reg2_X", X); L("reg2_y", y)
X, y = D.make_blobs(n_samples=25, centers=3, n_features=2, random_state=4); L("blob_X", X); L("blob_y", y)
X, y = D.make_blobs(n_samples=[3, 5], centers=[[0.0, 1.0], [2.0, 2.0]], cluster_std=[0.5, 1.5], random_state=5); L("blob2_X", X); L("blob2_y", y)
X, y = D.make_moons(n_samples=21, noise=0.1, random_state=6); L("moon_X", X); L("moon_y", y)
X, y = D.make_circles(n_samples=20, noise=0.05, factor=0.5, random_state=7); L("circ_X", X); L("circ_y", y)
X, t = D.make_s_curve(n_samples=12, noise=0.1, random_state=8); L("sc_X", X); L("sc_t", t)
X, t = D.make_swiss_roll(n_samples=12, noise=0.1, random_state=9); L("sw_X", X); L("sw_t", t)
X, t = D.make_swiss_roll(n_samples=10, hole=True, random_state=9); L("swh_X", X); L("swh_t", t)
for k, f in enumerate((D.make_friedman1, D.make_friedman2, D.make_friedman3)):
    X, y = f(n_samples=10, noise=0.3, random_state=10 + k); L(f"fr{k+1}_X", X); L(f"fr{k+1}_y", y)
L("lr", D.make_low_rank_matrix(n_samples=6, n_features=5, effective_rank=2, random_state=11))
L("spd", D.make_spd_matrix(4, random_state=12))
L("sspd", D.make_sparse_spd_matrix(6, alpha=0.7, random_state=13))
L("sspd2", D.make_sparse_spd_matrix(5, alpha=0.5, norm_diag=True, random_state=14))
X, Y = D.make_multilabel_classification(n_samples=8, n_features=6, n_classes=3, random_state=15); L("ml_X", X); L("ml_Y", Y)
X, y = D.make_gaussian_quantiles(n_samples=15, n_features=2, n_classes=3, random_state=16); L("gq_X", X); L("gq_y", y)
X, y = D.make_hastie_10_2(n_samples=6, random_state=17); L("ha_X", X); L("ha_y", y)
r, rows, cols = D.make_biclusters((6, 5), 2, noise=0.5, random_state=18); L("bi_r", r); L("bi_rows", rows); L("bi_cols", cols)
r, rows, cols = D.make_checkerboard((6, 4), 2, random_state=19); L("cb_r", r); L("cb_rows", rows)
X, y = D.make_sparse_uncorrelated(n_samples=6, n_features=4, random_state=20); L("su_X", X); L("su_y", y)
X, y = D.load_iris(return_X_y=True); print("iris_sum =", float(X.sum()), "; iris_ysum =", int(y.sum()))
X, y = D.load_wine(return_X_y=True); print("wine_sum =", float(X.sum()), "; wine_ysum =", int(y.sum()))
X, y = D.load_breast_cancer(return_X_y=True); print("bc_sum =", float(X.sum()), "; bc_ysum =", int(y.sum()))
X, y = D.load_digits(return_X_y=True); print("dg_sum =", float(X.sum()), "; dg_ysum =", int(y.sum()), "; dg_n =", X.shape[0])
X, y = D.load_digits(n_class=3, return_X_y=True); print("dg3_n =", X.shape[0])
X, y = D.load_diabetes(return_X_y=True); L("db_X0", X[:3]); print("db_ysum =", float(y.sum()))
