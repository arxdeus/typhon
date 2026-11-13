# Generates expected values for test_logistic.py (scikit-learn 1.9.1 venv).
import numpy as np
import warnings
warnings.simplefilter("ignore")
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

rng = np.random.RandomState(0)
X = rng.randn(80, 4)
yb = np.empty(80, dtype=int)
y3 = np.empty(80, dtype=int)
for i in range(80):
    d = X[i, 0] - X[i, 1] + 0.5 * X[i, 2] + 0.5 * rng.randn()
    yb[i] = 1 if d > 0 else 0
    y3[i] = 0 if d < -0.6 else (1 if d < 0.6 else 2)
sw = rng.uniform(0.5, 2.0, 80)

def L(v):
    return [float(x) for x in np.asarray(v, dtype=float).ravel()]

out = []
def lr(name, m):
    out.append((name, L(m.coef_) + L(m.intercept_) + L(m.n_iter_)))

for s in ["lbfgs", "newton-cg", "newton-cholesky", "sag", "saga", "liblinear"]:
    lr(s + "_b", LogisticRegression(solver=s, random_state=0, max_iter=2000).fit(X, yb))
for s in ["lbfgs", "newton-cg", "newton-cholesky", "sag", "saga"]:
    lr(s + "_3", LogisticRegression(solver=s, random_state=0, max_iter=2000).fit(X, y3))
lr("lbfgs_sw_bal", LogisticRegression(class_weight="balanced").fit(X, y3, sample_weight=sw))
lr("nch_sw", LogisticRegression(solver="newton-cholesky", C=0.3).fit(X, yb, sample_weight=sw))
lr("lbfgs_inf", LogisticRegression(C=np.inf).fit(X, yb))
lr("nch_inf3", LogisticRegression(C=np.inf, solver="newton-cholesky").fit(X, y3))
lr("ll_l1", LogisticRegression(l1_ratio=1.0, solver="liblinear", C=0.3, random_state=1).fit(X, yb))
lr("ll_dual", LogisticRegression(dual=True, solver="liblinear", random_state=1).fit(X, yb))
lr("ll_cw", LogisticRegression(solver="liblinear", class_weight={0: 2.0, 1: 0.5}, intercept_scaling=3.0).fit(X, yb))
lr("saga_en", LogisticRegression(l1_ratio=0.4, solver="saga", C=0.5, random_state=0, max_iter=3000).fit(X, y3))
lr("saga_l1", LogisticRegression(l1_ratio=1.0, solver="saga", C=0.5, random_state=0, max_iter=3000).fit(X, yb))
lr("noint", LogisticRegression(fit_intercept=False, C=2.0).fit(X, y3))
m = LogisticRegression().fit(X, y3)
out.append(("proba", L(m.predict_proba(X[:3]))))
out.append(("dec", L(LogisticRegression().fit(X, yb).decision_function(X[:4]))))

m = LogisticRegressionCV(Cs=5, cv=4, use_legacy_attributes=True).fit(X, yb)
out.append(("cvb", L(m.coef_) + L(m.intercept_) + L(m.C_) + L(m.n_iter_) + L(m.scores_[1])))
m = LogisticRegressionCV(Cs=[0.01, 0.1, 1.0, 10.0], cv=3, use_legacy_attributes=True).fit(X, y3)
out.append(("cv3", L(m.coef_) + L(m.intercept_) + L(m.C_) + L(m.coefs_paths_[2])))
m = LogisticRegressionCV(Cs=4, cv=3, solver="saga", l1_ratios=[0.2, 0.8], random_state=0, max_iter=500,
                         use_legacy_attributes=True).fit(X, yb)
out.append(("cven", L(m.coef_) + L(m.intercept_) + L(m.C_) + L(m.l1_ratio_) + L(m.n_iter_)))
m = LogisticRegressionCV(Cs=3, cv=3, solver="liblinear", refit=False, random_state=0,
                         use_legacy_attributes=True).fit(X, yb)
out.append(("cvll", L(m.coef_) + L(m.intercept_) + L(m.C_)))

for name, vals in out:
    print(f"    ({name!r}, {vals!r}),")
