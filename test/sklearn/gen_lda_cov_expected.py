# Generates the expected literals for test/sklearn/test_lda_cov.py (scikit-learn 1.9.1).
#   <venv>/bin/python test/sklearn/gen_lda_cov_expected.py
import warnings
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.covariance import MinCovDet, EllipticEnvelope, GraphicalLasso, GraphicalLassoCV, graphical_lasso
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

X, y = data(60, 4, 3)
Q = X[:6] + 0.1
for k, kw in (("svd", dict()), ("eig", dict(solver="eigen", reg_param=0.0)), ("reg", dict(reg_param=0.3, store_covariance=True)),
              ("shr", dict(solver="eigen", shrinkage="auto")), ("shf", dict(solver="eigen", shrinkage=0.2, priors=[0.2, 0.3, 0.5]))):
    q = QDA(**kw).fit(X, y)
    L(f"qda_{k}_dec", q.decision_function(Q)); L(f"qda_{k}_proba", q.predict_proba(Q)); L(f"qda_{k}_pred", q.predict(X))
    L(f"qda_{k}_means", q.means_)
L("qda_reg_cov0", QDA(reg_param=0.3, store_covariance=True).fit(X, y).covariance_[0])
q = QDA().fit(X[y < 2], y[y < 2]); L("qda_bin_dec", q.decision_function(Q))
l = LDA().fit(X, y); L("lda_logp", l.predict_log_proba(Q))

rng = np.random.RandomState(0)
Z = rng.standard_normal((80, 3)) @ np.array([[1.0, 0.3, 0.0], [0.0, 1.0, 0.5], [0.0, 0.0, 0.7]])
Z[:6] += 6.0
m = MinCovDet(random_state=1).fit(Z)
L("mcd_rloc", m.raw_location_); L("mcd_rcov", m.raw_covariance_); L("mcd_rsup", m.raw_support_)
L("mcd_loc", m.location_); L("mcd_cov", m.covariance_); L("mcd_sup", m.support_); L("mcd_dist", m.dist_)
m = MinCovDet(random_state=2, support_fraction=0.8).fit(Z[:, :1]); L("mcd1_loc", m.location_); L("mcd1_cov", m.covariance_)
e = EllipticEnvelope(random_state=3, contamination=0.1).fit(Z)
print("ee_off =", e.offset_); L("ee_dec", e.decision_function(Z[:10])); L("ee_pred", e.predict(Z))
g = GraphicalLasso(alpha=0.1).fit(Z)
L("gl_cov", g.covariance_); L("gl_prec", g.precision_); print("gl_it =", g.n_iter_)
c, p = graphical_lasso(np.cov(Z.T, bias=True), alpha=0.3); L("glf_cov", c); L("glf_prec", p)
g = GraphicalLassoCV(alphas=3, n_refinements=2, cv=3).fit(Z)
print("glcv_alpha =", g.alpha_); L("glcv_prec", g.precision_)
