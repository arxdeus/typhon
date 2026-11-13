# Generates expected literals for test/sklearn/test_final_chain.py (scikit-learn 1.9.1).
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.frozen import FrozenEstimator

def L(name, a):
    print(f"{name} = {[float(v) for v in np.asarray(a, dtype=float).ravel()]!r}")

n = 40
X = np.zeros((n, 3))
for i in range(n):
    for j in range(3):
        X[i, j] = ((i * 7919 + j * 104729 + i * j * 13) % 101 - 50.0) / 25.0
Y = np.column_stack([(X[:, 0] + 0.5 * X[:, 1] > 0.1), (X[:, 1] - X[:, 2] > 0.0), (X[:, 0] * X[:, 2] > 0.2)]).astype(int)
Q = X[:6] + 0.07
lr = lambda: LogisticRegression(C=2.0)
for tag, kw in [("pp", dict(chain_method="predict_proba")), ("df", dict(chain_method="decision_function")),
                ("dfcv", dict(chain_method="decision_function", cv=3, order=[2, 0, 1])),
                ("ppcv", dict(chain_method="predict_proba", cv=3))]:
    m = ClassifierChain(lr(), **kw).fit(X, Y)
    L(tag + "_pred", m.predict(Q)); L(tag + "_proba", m.predict_proba(Q)); L(tag + "_dec", m.decision_function(Q))
    L(tag + "_logp", m.predict_log_proba(Q))
f = FrozenEstimator(LogisticRegression(C=2.0).fit(X, Y[:, 0]))
L("fz_pred", f.predict(Q)); L("fz_proba", f.predict_proba(Q)); L("fz_dec", f.decision_function(Q))
f.fit(X, 1 - Y[:, 0]); L("fz_pred2", f.predict(Q))
