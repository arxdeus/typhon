# Generates the expected literals for test/sklearn/test_misc_estimators.py (scikit-learn 1.9.1).
#   <venv>/bin/python test/sklearn/gen_misc_estimators_expected.py
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.isotonic import IsotonicRegression, isotonic_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.kernel_approximation import RBFSampler, Nystroem, AdditiveChi2Sampler, SkewedChi2Sampler, PolynomialCountSketch
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection, johnson_lindenstrauss_min_dim

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

X, y = data(30, 4, 3)
y[:4] = 1
yr = X[:, 0] * 2.0 - X[:, 1] + 0.5 * y
Y2 = np.column_stack([yr, X[:, 2] ** 2])
sw = 1.0 + (np.arange(30) % 4)
for s in ("most_frequent", "prior", "stratified", "uniform"):
    d = DummyClassifier(strategy=s, random_state=3).fit(X, y)
    L(f"dc_{s}_pred", d.predict(X[:10])); L(f"dc_{s}_proba", d.predict_proba(X[:5]))
d = DummyClassifier(strategy="constant", constant=2).fit(X, y); L("dc_const_pred", d.predict(X[:3]))
L("dc_prior_sw", DummyClassifier().fit(X, y, sample_weight=sw).class_prior_)
L("dr_mean", DummyRegressor().fit(X, yr).predict(X[:2]))
L("dr_median", DummyRegressor(strategy="median").fit(X, yr).predict(X[:2]))
L("dr_q", DummyRegressor(strategy="quantile", quantile=0.3).fit(X, yr, sample_weight=sw).predict(X[:2]))
L("dr_mean_sw", DummyRegressor().fit(X, yr, sample_weight=sw).predict(X[:2]))
L("dr_2d", DummyRegressor(strategy="median").fit(X, Y2).predict(X[:2]))
print("dr_score =", DummyRegressor().fit(X, yr).score(X, yr))

xi = np.array([3.0, 1.0, 2.0, 2.0, 5.0, 4.0, 4.0, 7.0, 6.0, 8.0, 0.5, 9.0])
yi = np.array([1.0, 0.5, 2.0, 1.0, 3.0, 2.5, 2.0, 2.2, 4.0, 3.9, 0.1, 5.0])
L("iso_fn", isotonic_regression(yi))
L("iso_fn_dec", isotonic_regression(yi, increasing=False, y_min=1.0, y_max=4.0))
L("iso_fn_w", isotonic_regression(yi, sample_weight=1 + np.arange(12.0) % 3))
ir = IsotonicRegression().fit(xi, yi)
L("iso_xt", ir.X_thresholds_); L("iso_yt", ir.y_thresholds_)
L("iso_pred", ir.predict([0.0, 0.7, 2.5, 4.0, 6.3, 9.0, 10.0]))
ir = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(xi, -yi, sample_weight=1 + np.arange(12.0) % 3)
L("iso2_pred", ir.predict([0.0, 0.7, 2.5, 4.0, 6.3, 9.0, 10.0])); print("iso2_inc =", bool(ir.increasing_))

for k, kw in (("lin", dict(kernel="linear", alpha=0.5)), ("rbf", dict(kernel="rbf", gamma=0.3)),
              ("poly", dict(kernel="poly", degree=2, coef0=1.0)), ("lap", dict(kernel="laplacian"))):
    kr = KernelRidge(**kw).fit(X, yr)
    L(f"kr_{k}_dual", kr.dual_coef_); L(f"kr_{k}_pred", kr.predict(X[:6] + 0.1))
kr = KernelRidge(kernel="rbf", alpha=[0.5, 2.0]).fit(X, Y2, sample_weight=sw); L("kr2_dual", kr.dual_coef_); L("kr2_pred", kr.predict(X[:3]))

L("rbfs", RBFSampler(gamma=0.5, n_components=7, random_state=1).fit_transform(X[:5]))
L("rbfs_scale", RBFSampler(gamma="scale", n_components=5, random_state=2).fit_transform(X[:4]))
Xp = np.abs(X)
L("skc", SkewedChi2Sampler(skewedness=2.0, n_components=6, random_state=3).fit_transform(Xp[:4]))
L("ac2", AdditiveChi2Sampler().fit_transform(Xp[:3]))
L("ac3", AdditiveChi2Sampler(sample_steps=3).fit_transform(Xp[:2]))
ny = Nystroem(gamma=0.2, n_components=8, random_state=4).fit(X)
L("ny_idx", ny.component_indices_); L("ny_tr", ny.transform(X[:4]))
ny = Nystroem(kernel="poly", degree=2, n_components=5, random_state=5).fit(X); L("nyp_tr", ny.transform(X[:3]))
L("pcs", PolynomialCountSketch(degree=2, n_components=8, random_state=6).fit_transform(X[:3]))
L("pcs2", PolynomialCountSketch(degree=3, coef0=1.0, gamma=0.5, n_components=6, random_state=7).fit_transform(X[:3]))

print("jl =", int(johnson_lindenstrauss_min_dim(1000, eps=0.3)))
Xw = np.hstack([X] * 25)
g = GaussianRandomProjection(n_components=5, random_state=8).fit(Xw)
L("grp_comp", g.components_[:2]); L("grp_tr", g.transform(Xw[:3]))
g = GaussianRandomProjection(eps=0.9, random_state=8).fit(Xw); print("grp_auto =", g.n_components_)
s = SparseRandomProjection(n_components=6, random_state=9).fit(Xw)
L("srp_comp", s.components_.toarray()); L("srp_tr", s.transform(Xw[:3])); print("srp_dens =", s.density_)
s = SparseRandomProjection(n_components=4, density=1.0, random_state=10, compute_inverse_components=True).fit(X)
L("srp1_comp", s.components_); L("srp1_inv", s.inverse_components_)
