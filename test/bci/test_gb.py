import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier

# expected values from gen_gb.py (reference scikit-learn)
H_P0 = [0.9287411572311914, 0.03080507095156273, 0.04045377181724581]
H_P7 = [0.027961690322790122, 0.8893813281619121, 0.08265698151529793]
H_LAB = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
H_PSUM = 482.6870606436839
G_P0 = [0.8891692440515871, 0.055476179961847734, 0.05535457598656499]
G_P7 = [0.06802777503566376, 0.8041551457285623, 0.12781707923577398]
G_LAB = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
G_PSUM = 478.9374373785349

def make(n: int, d: int):
    X = np.empty((n, d)); y = np.empty(n, dtype=int)
    for i in range(n):
        y[i] = i % 3
        for j in range(d):
            X[i, j] = float((y[i] + j) % 3) * 0.8 + 1.5 * np.sin(0.37 * i * (j + 1) + 0.11 * j) + np.cos(1.7 * i + j * j)
    return X, y

def check(P, lab, p0, p7, labs, psum: float):
    n = P.shape[0]
    for k in range(3):
        assert abs(P[0, k] - p0[k]) < 1e-9
        assert abs(P[7, k] - p7[k]) < 1e-9
    s = 0.0
    for i in range(n):
        assert int(lab[i]) == labs[i]
        for k in range(3):
            s += P[i, k] * float((i * 3 + k) % 7 + 1)
    assert abs(s - psum) < 1e-8

X, y = make(120, 12)
h = HistGradientBoostingClassifier(max_iter=20)
h.fit(X, y)
check(h.predict_proba(X), h.predict(X), H_P0, H_P7, H_LAB, H_PSUM)
g = GradientBoostingClassifier(n_estimators=20)
g.fit(X, y)
check(g.predict_proba(X), g.predict(X), G_P0, G_P7, G_LAB, G_PSUM)
print("ok")
