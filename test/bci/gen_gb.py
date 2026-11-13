# Generates expected values for test_gb.py with the reference sklearn.
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as H, GradientBoostingClassifier as G

def make(n, d):
    X = np.empty((n, d)); y = np.empty(n, dtype=int)
    for i in range(n):
        y[i] = i % 3
        for j in range(d):
            X[i, j] = float((y[i] + j) % 3) * 0.8 + 1.5 * np.sin(0.37 * i * (j + 1) + 0.11 * j) + np.cos(1.7 * i + j * j)
    return X, y

X, y = make(120, 12)
for nm, m in (("H", H(max_iter=20)), ("G", G(n_estimators=20))):
    m.fit(X, y)
    P = m.predict_proba(X)
    print(nm + "_P0 =", [float(v) for v in P[0]])
    print(nm + "_P7 =", [float(v) for v in P[7]])
    print(nm + "_LAB =", [int(v) for v in m.predict(X)])
    print(nm + "_PSUM =", float(sum(P[i, k] * ((i * 3 + k) % 7 + 1) for i in range(120) for k in range(3))))
