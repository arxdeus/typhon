# Generates the expected literals for test/sklearn/test_pls_semi.py (scikit-learn 1.9.1).
#   <venv>/bin/python test/sklearn/gen_pls_semi_expected.py
import warnings
import numpy as np
from sklearn.cross_decomposition import PLSRegression, PLSCanonical, CCA, PLSSVD
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier
from sklearn.naive_bayes import GaussianNB
warnings.filterwarnings("ignore")

def data(n, p, nc):
    X = np.zeros((n, p)); y = np.zeros(n, dtype=int)
    for i in range(n):
        y[i] = i % nc
        for j in range(p):
            v = (i * 7919 + j * 104729 + i * j * 13 + j * j * 3) % 101
            X[i, j] = (v - 50.0) / 25.0 + (0.8 * y[i] if (j + y[i]) % 2 == 0 else -0.3 * y[i])
    return X, y

def L(name, a):
    print(f"{name} = {[float(v) for v in np.asarray(a, dtype=float).ravel()]!r}")

X, y = data(40, 5, 3)
Y = np.column_stack([X[:, 0] - X[:, 1] + 0.1 * np.sin(np.arange(40.0)), X[:, 2] * 0.5 + X[:, 3], np.cos(np.arange(40.0))])
Q = X[:6] + 0.1
m = PLSRegression(n_components=3).fit(X, Y)
L("plsr_xw", m.x_weights_); L("plsr_yl", m.y_loadings_); L("plsr_coef", m.coef_); L("plsr_pred", m.predict(Q)); L("plsr_tr", m.transform(Q)); L("plsr_xs", m.x_scores_[:5]); L("plsr_it", m.n_iter_)
m = PLSRegression(n_components=2, scale=False).fit(X, Y[:, 0]); L("plsr1_pred", m.predict(Q)); L("plsr1_coef", m.coef_)
m = PLSCanonical(n_components=2).fit(X, Y); L("plsc_xw", m.x_weights_); L("plsc_yw", m.y_weights_); L("plsc_pred", m.predict(Q)); L("plsc_inv", m.inverse_transform(m.transform(Q)))
m = PLSCanonical(n_components=2, algorithm="svd").fit(X, Y); L("plscs_xr", m.x_rotations_)
m = CCA(n_components=2).fit(X, Y); L("cca_xw", m.x_weights_); L("cca_yr", m.y_rotations_); L("cca_pred", m.predict(Q))
m = PLSSVD(n_components=2).fit(X, Y); L("plssvd_xw", m.x_weights_); L("plssvd_yw", m.y_weights_); L("plssvd_tr", m.transform(Q))
yu = y.copy(); yu[::3] = -1; yu[1::5] = -1
m = LabelPropagation(gamma=0.5).fit(X, yu); L("lp_ld", m.label_distributions_[:8]); L("lp_tr", m.transduction_); L("lp_proba", m.predict_proba(Q)); print("lp_it =", m.n_iter_)
m = LabelSpreading(gamma=0.5, alpha=0.3).fit(X, yu); L("ls_ld", m.label_distributions_[:8]); L("ls_tr", m.transduction_); L("ls_proba", m.predict_proba(Q)); print("ls_it =", m.n_iter_)
m = LabelSpreading(kernel="knn", n_neighbors=5).fit(X, yu); L("lsk_tr", m.transduction_); L("lsk_proba", m.predict_proba(Q))
m = LabelPropagation(kernel="knn", n_neighbors=4).fit(X, yu); L("lpk_tr", m.transduction_)
m = SelfTrainingClassifier(GaussianNB(), threshold=0.9).fit(X, yu); L("st_tr", m.transduction_); L("st_li", m.labeled_iter_); print("st_it =", m.n_iter_); print("st_term =", repr(m.termination_condition_)); L("st_pred", m.predict(Q))
m = SelfTrainingClassifier(GaussianNB(), criterion="k_best", k_best=3, max_iter=2).fit(X, yu); L("stk_li", m.labeled_iter_); print("stk_term =", repr(m.termination_condition_))
