# Generates expected literals for test/sklearn/test_final_svm.py (scikit-learn 1.9.1).
import warnings
import numpy as np
from sklearn.svm import LinearSVC
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

X, y = data(48, 4, 3)
sw = 1.0 + (np.arange(48) % 3)
Q = X[:7] + 0.05
for tag, kw, w in [("cs", dict(C=1.0), None), ("cs2", dict(C=0.3, fit_intercept=False), None),
                   ("cs3", dict(C=2.0, class_weight={0: 1.0, 1: 2.0, 2: 0.5}), sw)]:
    m = LinearSVC(multi_class="crammer_singer", random_state=0, **kw).fit(X, y, sample_weight=w)
    L(tag + "_coef", m.coef_); L(tag + "_icpt", m.intercept_)
    print(f"{tag}_it = {int(m.n_iter_)}")
    L(tag + "_dec", m.decision_function(Q))
