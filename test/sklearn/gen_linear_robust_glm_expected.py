# Generates expected values for test_linear_robust_glm.py (scikit-learn 1.9.1 venv).
import numpy as np
import warnings
warnings.simplefilter("ignore")
from sklearn.linear_model import (PoissonRegressor, GammaRegressor, TweedieRegressor, HuberRegressor,
                                  QuantileRegressor, RANSACRegressor, TheilSenRegressor, ARDRegression)

rng = np.random.RandomState(0)
X = rng.randn(60, 3)
eta = 0.3 * X[:, 0] - 0.5 * X[:, 1] + 0.2
yp = rng.poisson(np.exp(eta)).astype(float)
yg = rng.gamma(2.0, np.exp(eta) / 2.0)
yr = X @ np.array([1.0, -2.0, 0.5]) + 0.3 * rng.randn(60)
yo = yr.copy()
yo[:6] += 8.0
sw = rng.uniform(0.5, 2.0, 60)
XW = rng.randn(8, 12)
yw = XW[:, 0] - XW[:, 3] + 0.1 * rng.randn(8)

def L(v):
    return [float(x) for x in np.asarray(v, dtype=float).ravel()]

out = []
def g(name, m):
    out.append((name, L(m.coef_) + [float(m.intercept_), float(m.n_iter_)]))

g("pois", PoissonRegressor(alpha=0.1).fit(X, yp))
g("pois_nc", PoissonRegressor(alpha=0.1, solver="newton-cholesky").fit(X, yp, sample_weight=sw))
g("gam", GammaRegressor(alpha=0.05).fit(X, yg))
g("gam_nc", GammaRegressor(alpha=0.05, solver="newton-cholesky").fit(X, yg))
g("tw15", TweedieRegressor(power=1.5, alpha=0.1).fit(X, yp))
g("tw0", TweedieRegressor(power=0.0, alpha=0.1, solver="newton-cholesky").fit(X, yr))
g("tw3id", TweedieRegressor(power=3.0, link="identity", alpha=0.01).fit(X, yg + 1.0))
mp = PoissonRegressor(alpha=0.1).fit(X, yp)
out.append(("pois_score", [mp.score(X, yp)] + L(mp.predict(X[:4]))))
h = HuberRegressor().fit(X, yo)
out.append(("huber", L(h.coef_) + [float(h.intercept_), float(h.scale_), float(h.n_iter_)] + L(h.outliers_)))
h2 = HuberRegressor(epsilon=1.8, alpha=0.01, fit_intercept=False).fit(X, yo, sample_weight=sw)
out.append(("huber2", L(h2.coef_) + [float(h2.scale_), float(h2.n_iter_)]))
q = QuantileRegressor(quantile=0.3, alpha=0.01).fit(X, yo)
out.append(("quant", L(q.coef_) + [float(q.intercept_)]))
q2 = QuantileRegressor(quantile=0.7, alpha=0.0, fit_intercept=False).fit(X, yo, sample_weight=sw)
out.append(("quant2", L(q2.coef_)))
r = RANSACRegressor(random_state=0).fit(X, yo)
out.append(("ransac", L(r.estimator_.coef_) + [float(r.estimator_.intercept_), float(r.n_trials_)] + L(r.inlier_mask_)))
r2 = RANSACRegressor(min_samples=0.3, loss="squared_error", residual_threshold=1.0, random_state=3).fit(X, yo)
out.append(("ransac2", L(r2.estimator_.coef_) + [float(r2.n_trials_)]))
t = TheilSenRegressor(random_state=0).fit(X[:20], yo[:20])
out.append(("theil", L(t.coef_) + [float(t.intercept_), float(t.n_iter_), float(t.breakdown_)]))
t2 = TheilSenRegressor(random_state=0, max_subpopulation=50).fit(X, yo)
out.append(("theil2", L(t2.coef_) + [float(t2.intercept_), float(t2.n_iter_), float(t2.n_subpopulation_)]))
a = ARDRegression().fit(X, yr)
out.append(("ard", L(a.coef_) + [float(a.intercept_), float(a.alpha_), float(a.n_iter_)] + L(a.lambda_) + L(a.sigma_)))
a2 = ARDRegression(compute_score=True).fit(XW, yw)
out.append(("ard2", L(a2.coef_) + [float(a2.intercept_), float(a2.alpha_), float(a2.n_iter_)] + L(a2.scores_)))
mu, sd = a.predict(X[:3], return_std=True)
out.append(("ard_std", L(mu) + L(sd)))

for name, vals in out:
    print(f"    ({name!r}, {vals!r}),")
