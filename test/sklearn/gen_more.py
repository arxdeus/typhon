# Expected literals for test/sklearn/test_more.py (scikit-learn 1.9.1, pyriemann 0.12).
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.covariance import LedoitWolf, OAS, EmpiricalCovariance, ShrunkCovariance, ledoit_wolf, oas
from sklearn.preprocessing import MinMaxScaler, RobustScaler, FunctionTransformer, StandardScaler
from sklearn.decomposition import FastICA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, GridSearchCV
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM

def data(n, p, nc):
    X = np.zeros((n, p)); y = np.zeros(n, dtype=int)
    for i in range(n):
        y[i] = i % nc
        for j in range(p):
            v = (i * 7919 + j * 104729 + i * j * 13 + j * j * 3) % 101
            X[i, j] = (v - 50.0) / 25.0 + (0.8 * y[i] if (j + y[i]) % 2 == 0 else -0.3 * y[i])
    return X, y

def signals(n_trials, n_ch, n_t):
    X = np.zeros((n_trials, n_ch, n_t)); y = np.zeros(n_trials, dtype=int)
    for i in range(n_trials):
        y[i] = i % 2
        for c in range(n_ch):
            for t in range(n_t):
                v = (i * 1009 + c * 7919 + t * 104729 + i * c * t * 13 + t * t * (c + 1) * 3 + c * c * t * 7) % 101
                X[i, c, t] = (v - 50.0) / 25.0 + ((0.9 * y[i]) if (c + t) % 3 == 0 else 0.0) * (1 + c % 2)
    return X, y

def L(name, a):
    print(f"{name} = {[float(v) for v in np.asarray(a, dtype=float).ravel()]!r}")

X, y = data(60, 5, 3)
for nm, est in (("emp", EmpiricalCovariance()), ("lw", LedoitWolf()), ("oas", OAS()), ("shr", ShrunkCovariance(shrinkage=0.2))):
    e = est.fit(X)
    L(f"cov_{nm}", e.covariance_); L(f"prec_{nm}", e.precision_)
    if hasattr(e, "shrinkage_"): print(f"sh_{nm} = {float(e.shrinkage_)!r}")
    L(f"maha_{nm}", e.mahalanobis(X[:5])); print(f"score_{nm} = {float(e.score(X[:20]))!r}")
L("lwf", ledoit_wolf(X[:, :3])[0]); L("oasf", oas(X[:, 1:])[0])
L("mm_t", MinMaxScaler().fit_transform(X)[:6]); L("mm_t2", MinMaxScaler((-1, 2)).fit(X).transform(X[:4] * 1.5))
L("rb_t", RobustScaler().fit_transform(X)[:6]); L("rb_c", RobustScaler().fit(X).center_)
L("rb_s", RobustScaler(quantile_range=(10.0, 90.0), unit_variance=True).fit(X).scale_)
L("ft_t", FunctionTransformer(np.log1p, np.expm1).fit_transform(np.abs(X[:4])))
S = np.zeros((300, 3))
for t in range(300):
    S[t, 0] = np.sin(t * 0.07); S[t, 1] = 1.0 if (t // 13) % 2 == 0 else -1.0; S[t, 2] = ((t * 37) % 41) / 20.5 - 1.0
A = np.array([[1.0, 0.5, 0.3], [0.4, 1.2, 0.2], [0.3, 0.6, 0.9]])
M = S @ A.T
for nm, kw in (("par", {}), ("def", {"algorithm": "deflation"}), ("exp", {"fun": "exp"}), ("cube", {"fun": "cube", "whiten": "arbitrary-variance"}), ("eigh", {"whiten_solver": "eigh"})):
    ica = FastICA(n_components=3, random_state=0, max_iter=400, **kw).fit(M)
    L(f"ica_comp_{nm}", ica.components_); L(f"ica_mix_{nm}", ica.mixing_); print(f"ica_iter_{nm} = {ica.n_iter_}")
    L(f"ica_src_{nm}", ica.transform(M[:5]))
ica2 = FastICA(n_components=2, random_state=3).fit(M)
L("ica2_comp", ica2.components_)
XS, ys = signals(24, 4, 64)
cv = KFold(4)
p1 = make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"), LogisticRegression(max_iter=1000))
L("pr_cvs", cross_val_score(p1, XS, ys, cv=cv))
print(f"pr_cvp = {[int(v) for v in cross_val_predict(p1, XS, ys, cv=cv)]!r}")
p2 = make_pipeline(Covariances("lwf"), MDM(metric="riemann"))
L("pr_cvs_mdm", cross_val_score(p2, XS, ys, cv=cv))
g = GridSearchCV(make_pipeline(Covariances("scm"), TangentSpace(), SVC()), {"svc__C": [0.01, 1.0, 100.0]}, cv=cv).fit(XS, ys)
L("pr_gs", g.cv_results_["mean_test_score"]); print(f"pr_gs_best = {g.best_index_}")
lr = make_pipeline(Covariances("oas"), TangentSpace(), StandardScaler(), LogisticRegression(max_iter=1000)).fit(XS, ys)
L("pr_coef", lr.steps[-1][1].coef_); L("pr_proba", lr.predict_proba(XS[:6]))
