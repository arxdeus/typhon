# Generates the expected literals for test/sklearn/test_mlp.py (scikit-learn 1.9.1).
#   <venv>/bin/python test/sklearn/gen_mlp_expected.py
import warnings
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor, BernoulliRBM
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
yr = X[:, 0] * 1.5 - np.tanh(X[:, 1]) + 0.2 * X[:, 2] * X[:, 3]
Q = X[:6] + 0.1
m = MLPClassifier((8,), solver="lbfgs", random_state=0, max_iter=60).fit(X, y)
L("cl_c0", m.coefs_[0]); L("cl_proba", m.predict_proba(Q)); print("cl_it =", m.n_iter_); print("cl_loss =", m.loss_)
m = MLPClassifier((6, 5), activation="tanh", solver="adam", random_state=1, max_iter=30).fit(X, y)
L("ca_proba", m.predict_proba(Q)); L("ca_curve", m.loss_curve_); print("ca_it =", m.n_iter_)
m = MLPClassifier((5,), activation="logistic", solver="sgd", learning_rate="invscaling", random_state=2, max_iter=25, batch_size=16).fit(X, y < 1)
L("cs_proba", m.predict_proba(Q)); L("cs_curve", m.loss_curve_)
m = MLPClassifier((7,), solver="sgd", learning_rate="adaptive", random_state=3, max_iter=40, early_stopping=True, n_iter_no_change=3, learning_rate_init=0.05).fit(X, y)
L("ce_vs", m.validation_scores_); L("ce_proba", m.predict_proba(Q)); print("ce_it =", m.n_iter_)
sw = 1.0 + (np.arange(60) % 3)
m = MLPRegressor(hidden_layer_sizes=(10,), solver="lbfgs", random_state=4, max_iter=80, alpha=0.01).fit(X, yr, sample_weight=sw)
L("rl_pred", m.predict(Q)); print("rl_loss =", m.loss_)
m = MLPRegressor(hidden_layer_sizes=(8, 4), solver="adam", random_state=5, max_iter=20, batch_size=10).fit(X, np.column_stack([yr, -yr]))
L("ra_pred", m.predict(Q)); L("ra_curve", m.loss_curve_)
m = MLPRegressor(hidden_layer_sizes=(6,), solver="sgd", random_state=6, max_iter=50, early_stopping=True, learning_rate_init=0.01).fit(X, yr)
L("re_vs", m.validation_scores_); L("re_pred", m.predict(Q))
Xb = (X > 0).astype(float)
r = BernoulliRBM(n_components=5, n_iter=4, batch_size=7, random_state=7, learning_rate=0.05).fit(Xb)
L("rbm_comp", r.components_); L("rbm_ih", r.intercept_hidden_); L("rbm_tr", r.transform(Xb[:5])); L("rbm_ss", r.score_samples(Xb[:5]))
