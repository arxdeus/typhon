# Generates expected values for test_linear_basic.py, test_linear_cd.py and
# test_linear_lars.py (scikit-learn 1.9.1 venv): python gen_linear_expected.py basic|cd|lars
import sys
import numpy as np
import warnings
warnings.simplefilter("ignore")
from sklearn.linear_model import (LinearRegression, Ridge, RidgeClassifier, RidgeCV,
                                  RidgeClassifierCV, ridge_regression, Lasso, ElasticNet,
                                  MultiTaskLasso, MultiTaskElasticNet, LassoCV, ElasticNetCV,
                                  enet_path, lasso_path, Lars, LassoLars, LarsCV, LassoLarsCV,
                                  LassoLarsIC, lars_path, OrthogonalMatchingPursuit,
                                  OrthogonalMatchingPursuitCV, orthogonal_mp)

rng = np.random.RandomState(0)
X = rng.randn(50, 6)
w = np.array([1.5, -2.0, 0.0, 0.5, 0.0, 3.0])
y = X @ w + 0.5 * rng.randn(50)
Y2 = np.column_stack([y, X @ w[::-1] + 0.3 * rng.randn(50)])
sw = rng.uniform(0.5, 2.0, 50)
yc = np.digitize(X[:, 0] + X[:, 1], [-0.7, 0.7])

def L(v):
    return [float(x) for x in np.asarray(v, dtype=float).ravel()]

out = []
part = sys.argv[1]
if part == "basic":
    m = LinearRegression().fit(X, y)
    out.append(("lr", L(m.coef_) + [float(m.intercept_)]))
    m = LinearRegression().fit(X, y, sample_weight=sw)
    out.append(("lr_sw", L(m.coef_) + [float(m.intercept_)]))
    m = LinearRegression(positive=True).fit(X, y)
    out.append(("lr_pos", L(m.coef_) + [float(m.intercept_)]))
    for s in ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"]:
        kw = dict(positive=True) if s == "lbfgs" else {}
        m = Ridge(alpha=2.0, solver=s, random_state=0, tol=1e-10, max_iter=100000, **kw).fit(X, y)
        out.append(("ridge_" + s, L(m.coef_) + [float(m.intercept_)]))
    m = Ridge(alpha=[1.0, 3.0]).fit(X, Y2, sample_weight=sw)
    out.append(("ridge_multi", L(m.coef_) + L(m.intercept_)))
    out.append(("ridge_reg", L(ridge_regression(X, y, alpha=1.5))))
    m = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(X, y)
    out.append(("ridgecv", L(m.coef_) + [float(m.intercept_), float(m.alpha_), float(m.best_score_)]))
    m = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5).fit(X, y)
    out.append(("ridgecv5", L(m.coef_) + [float(m.intercept_), float(m.alpha_)]))
    m = RidgeClassifier(alpha=1.0).fit(X, yc)
    out.append(("ridgeclf", L(m.coef_) + L(m.intercept_) + L(m.predict(X[:10]))))
    m = RidgeClassifierCV(alphas=[0.1, 1.0, 10.0]).fit(X, yc)
    out.append(("ridgeclfcv", L(m.coef_) + L(m.intercept_) + [float(m.alpha_)]))
elif part == "cd":
    m = Lasso(alpha=0.1).fit(X, y)
    out.append(("lasso", L(m.coef_) + [float(m.intercept_), float(m.n_iter_), float(m.dual_gap_)]))
    m = Lasso(alpha=0.05, positive=True, selection="random", random_state=0).fit(X, y)
    out.append(("lasso_pos", L(m.coef_) + [float(m.intercept_), float(m.n_iter_)]))
    m = ElasticNet(alpha=0.2, l1_ratio=0.3).fit(X, y, sample_weight=sw)
    out.append(("enet_sw", L(m.coef_) + [float(m.intercept_), float(m.n_iter_)]))
    m = MultiTaskLasso(alpha=0.2).fit(X, Y2)
    out.append(("mtl", L(m.coef_) + L(m.intercept_) + [float(m.n_iter_)]))
    m = MultiTaskElasticNet(alpha=0.2, l1_ratio=0.6).fit(X, Y2)
    out.append(("mten", L(m.coef_) + L(m.intercept_) + [float(m.n_iter_)]))
    m = LassoCV(cv=4, alphas=20).fit(X, y)
    out.append(("lassocv", L(m.coef_) + [float(m.intercept_), float(m.alpha_)] + L(m.mse_path_)))
    m = ElasticNetCV(cv=3, l1_ratio=[0.2, 0.8], alphas=10).fit(X, y)
    out.append(("enetcv", L(m.coef_) + [float(m.intercept_), float(m.alpha_), float(m.l1_ratio_)]))
    a, c, g = enet_path(X, y, l1_ratio=0.5, alphas=8)
    out.append(("enet_path", L(a) + L(c) + L(g)))
    a, c, g = lasso_path(X, y, alphas=[1.0, 0.3, 0.1])
    out.append(("lasso_path", L(a) + L(c)))
else:
    m = Lars().fit(X, y)
    out.append(("lars", L(m.coef_) + [float(m.intercept_)] + L(m.alphas_) + L(m.active_)))
    m = Lars(n_nonzero_coefs=3).fit(X, y)
    out.append(("lars3", L(m.coef_) + [float(m.intercept_)]))
    m = LassoLars(alpha=0.05).fit(X, y)
    out.append(("lassolars", L(m.coef_) + [float(m.intercept_), float(m.n_iter_)]))
    a, act, c = lars_path(X, y, method="lasso")
    out.append(("lars_path", L(a) + L(act) + L(c)))
    m = LarsCV(cv=4).fit(X, y)
    out.append(("larscv", L(m.coef_) + [float(m.intercept_), float(m.alpha_)]))
    m = LassoLarsCV(cv=4).fit(X, y)
    out.append(("lassolarscv", L(m.coef_) + [float(m.intercept_), float(m.alpha_)]))
    for crit in ["aic", "bic"]:
        m = LassoLarsIC(criterion=crit).fit(X, y)
        out.append(("llic_" + crit, L(m.coef_) + [float(m.intercept_), float(m.alpha_)] + L(m.criterion_)))
    m = OrthogonalMatchingPursuit(n_nonzero_coefs=3).fit(X, y)
    out.append(("omp", L(m.coef_) + [float(m.intercept_)]))
    out.append(("omp_fn", L(orthogonal_mp(X, y, tol=10.0))))
    m = OrthogonalMatchingPursuitCV(cv=4).fit(X, y)
    out.append(("ompcv", L(m.coef_) + [float(m.intercept_), float(m.n_nonzero_coefs_)]))

for name, vals in out:
    print(f"    ({name!r}, {vals!r}),")
