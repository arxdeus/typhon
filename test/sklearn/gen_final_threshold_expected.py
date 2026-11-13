# Generates expected literals for test/sklearn/test_final_threshold.py (scikit-learn 1.9.1).
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import FixedThresholdClassifier, TunedThresholdClassifierCV

def L(name, a):
    print(f"{name} = {[float(v) for v in np.asarray(a, dtype=float).ravel()]!r}")

n = 60
X = np.zeros((n, 3))
for i in range(n):
    for j in range(3):
        X[i, j] = ((i * 7919 + j * 104729 + i * j * 13) % 101 - 50.0) / 25.0
y = ((X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + 0.3 * np.sin(np.arange(n))) > 0.4).astype(int)
Q = X[:12] + 0.05
f = FixedThresholdClassifier(LogisticRegression(), threshold=0.3).fit(X, y); L("fx_pred", f.predict(Q))
f = FixedThresholdClassifier(LogisticRegression(), threshold=0.5, response_method="decision_function").fit(X, y); L("fx2_pred", f.predict(Q))
f = FixedThresholdClassifier(LogisticRegression()).fit(X, y); L("fx3_pred", f.predict(Q))
for tag, kw in [("t1", dict()), ("t2", dict(scoring="f1", cv=3)), ("t3", dict(scoring="accuracy", cv=0.3, random_state=0, thresholds=20)),
                ("t4", dict(scoring="recall", thresholds=[0.1, 0.3, 0.5, 0.7], response_method="predict_proba"))]:
    m = TunedThresholdClassifierCV(LogisticRegression(), **kw).fit(X, y)
    print(f"{tag}_th = {float(m.best_threshold_)!r}"); print(f"{tag}_bs = {float(m.best_score_)!r}"); L(tag + "_pred", m.predict(Q))
m = TunedThresholdClassifierCV(LogisticRegression().fit(X, y), cv="prefit", refit=False, scoring="f1").fit(X, y)
print(f"t5_th = {float(m.best_threshold_)!r}"); L("t5_pred", m.predict(Q))
