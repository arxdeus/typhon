# Expected literals for test/sklearn/test_preprocessing_more.py (scikit-learn 1.9.1).
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.preprocessing import (PolynomialFeatures, SplineTransformer, QuantileTransformer,
    KBinsDiscretizer, PowerTransformer, TargetEncoder, quantile_transform, power_transform)

def data(n, p):
    X = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            v = (i * 7919 + j * 104729 + i * j * 13 + j * j * 3) % 101
            X[i, j] = (v - 50.0) / 25.0
    return X

def L(name, a):
    print(f"{name} = {[float(v) for v in np.asarray(a, dtype=float).ravel()]!r}")

X = data(30, 3)
L("poly2", PolynomialFeatures(2).fit_transform(X))
L("poly_int", PolynomialFeatures((2, 3), interaction_only=True, include_bias=False).fit_transform(X))
print(f"poly_names = {list(PolynomialFeatures(2).fit(X).get_feature_names_out())!r}")
L("spl", SplineTransformer(n_knots=4, degree=3).fit_transform(X))
L("spl_q", SplineTransformer(n_knots=3, degree=2, knots="quantile", extrapolation="linear").fit(X).transform(X * 1.5))
L("spl_per", SplineTransformer(n_knots=4, degree=3, extrapolation="periodic").fit_transform(X))
qt = QuantileTransformer(n_quantiles=10).fit(X)
L("qt_u", qt.transform(X)); L("qt_inv", qt.inverse_transform(qt.transform(X)))
L("qt_n", QuantileTransformer(n_quantiles=15, output_distribution="normal").fit_transform(X))
L("qt_fn", quantile_transform(X, n_quantiles=12))
for s in ("uniform", "quantile", "kmeans"):
    k = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy=s).fit(X)
    L(f"kb_{s}", k.transform(X)); L(f"kb_{s}_edges", np.concatenate(k.bin_edges_))
L("kb_oh", KBinsDiscretizer(n_bins=3, encode="onehot-dense").fit_transform(X))
pt = PowerTransformer().fit(X)
L("pt_yj_l", pt.lambdas_); L("pt_yj", pt.transform(X)); L("pt_yj_inv", pt.inverse_transform(pt.transform(X)))
Xp = np.abs(X) + 0.5
pb = PowerTransformer(method="box-cox").fit(Xp)
L("pt_bc_l", pb.lambdas_); L("pt_bc", pb.transform(Xp))
L("pt_fn", power_transform(X, standardize=False))
Xc = np.array([[i % 3, (i * 7) % 4] for i in range(40)], dtype=float)
yr = np.array([((i * 37) % 11) / 3.0 + (i % 3) for i in range(40)])
te = TargetEncoder(random_state=0)
L("te_ft", te.fit_transform(Xc, yr)); L("te_tr", te.transform(Xc)); L("te_enc", np.concatenate(te.encodings_))
te2 = TargetEncoder(smooth=2.0, cv=3, shuffle=False)
L("te2_ft", te2.fit_transform(Xc, yr))
