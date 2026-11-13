# Expected literals for test/sklearn/test_pipeline_compose.py (scikit-learn 1.9.1).
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion, make_union
from sklearn.compose import ColumnTransformer, make_column_transformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV

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

X, y = data(30, 4, 2)
yr = X[:, 0] * 2.0 + X[:, 2] ** 2 + 5.0
p = Pipeline([("sc", StandardScaler()), ("skip", "passthrough"), ("pca", PCA(n_components=2)), ("lr", LogisticRegression(C=0.5))]).fit(X, y)
L("p_proba", p.predict_proba(X)); L("p_tr", p[:-1].transform(X))
p.set_params(lr__C=0.05).fit(X, y); L("p_coef2", p.named_steps["lr"].coef_)
pi = make_pipeline(StandardScaler(), MinMaxScaler()).fit(X)
L("pi_inv", pi.inverse_transform(pi.transform(X) * 0.5))
fu = FeatureUnion([("pca", PCA(n_components=2)), ("mm", MinMaxScaler())], transformer_weights={"pca": 2.0}).fit(X)
L("fu", fu.transform(X))
L("mu", make_union(StandardScaler(), PolynomialFeatures(2)).fit_transform(X[:, :2]))
ct = ColumnTransformer([("s", StandardScaler(), [0, 2]), ("m", MinMaxScaler(), np.array([False, True, False, False]))], remainder="passthrough").fit(X)
L("ct", ct.transform(X)); print(f"ct_oi = {ct.output_indices_['remainder']!r}")
ct2 = ColumnTransformer([("p", PCA(n_components=1), slice(0, 3)), ("d", "drop", [3])], remainder=StandardScaler()).fit(X)
L("ct2", ct2.transform(X))
ct3 = make_column_transformer((StandardScaler(), [1]), (PolynomialFeatures(2), [0, 3]), remainder="drop")
L("ct3", ct3.fit_transform(X))
tt = TransformedTargetRegressor(Ridge(), func=np.log, inverse_func=np.exp).fit(X, yr)
L("tt_pred", tt.predict(X)); print(f"tt_score = {tt.score(X, yr)!r}")
tt2 = TransformedTargetRegressor(Ridge(alpha=0.5), transformer=StandardScaler()).fit(X, yr)
L("tt2_pred", tt2.predict(X))
gs = GridSearchCV(Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression())]), {"lr__C": [0.01, 1.0], "sc__with_mean": [True, False]}, cv=3).fit(X, y)
L("gs_mean", gs.cv_results_["mean_test_score"]); print(f"gs_best = {gs.best_index_}")
