# Generates expected values for test_linear_sgd.py (scikit-learn 1.9.1 venv).
import numpy as np
import warnings
warnings.simplefilter("ignore")
from sklearn.linear_model import (SGDClassifier, SGDRegressor, SGDOneClassSVM, Perceptron,
                                  PassiveAggressiveClassifier, PassiveAggressiveRegressor)

rng = np.random.RandomState(0)
X = rng.randn(60, 4)
w = np.array([1.5, -2.0, 0.5, 0.0])
yr = X @ w + 0.3 * rng.randn(60)
yb = (X[:, 0] + X[:, 1] > 0).astype(int)
y3 = np.digitize(X[:, 0] - X[:, 2], [-0.5, 0.5])

def L(v):
    return [float(x) for x in np.asarray(v, dtype=float).ravel()]

out = []
def clf(name, m, y):
    m.fit(X, y)
    out.append((name + "_coef", L(m.coef_)))
    out.append((name + "_int", L(m.intercept_)))
    out.append((name + "_meta", [float(m.n_iter_), float(m.t_)]))
    out.append((name + "_pred", L(m.predict(X[:10]))))

def reg(name, m):
    m.fit(X, yr)
    out.append((name + "_coef", L(m.coef_)))
    out.append((name + "_int", L(m.intercept_)))
    out.append((name + "_meta", [float(m.n_iter_), float(m.t_)]))

clf("hinge", SGDClassifier(random_state=0), yb)
clf("log3", SGDClassifier(loss="log_loss", penalty="elasticnet", random_state=1, max_iter=50), y3)
clf("mh", SGDClassifier(loss="modified_huber", penalty="l1", random_state=2, average=True), yb)
clf("sqh_bal", SGDClassifier(loss="squared_hinge", learning_rate="invscaling", eta0=0.05,
                             class_weight="balanced", random_state=3), y3)
clf("es", SGDClassifier(early_stopping=True, learning_rate="adaptive", eta0=0.1, random_state=4), yb)
clf("perc", Perceptron(random_state=0), y3)
clf("pac", PassiveAggressiveClassifier(random_state=0), yb)
m = SGDClassifier(loss="log_loss", random_state=0, max_iter=50).fit(X, yb)
out.append(("log_proba", L(m.predict_proba(X[:5]))))

reg("sq", SGDRegressor(random_state=0))
reg("hub", SGDRegressor(loss="huber", penalty="elasticnet", random_state=1, average=10))
reg("eps", SGDRegressor(loss="epsilon_insensitive", learning_rate="constant", eta0=0.01, random_state=2))
reg("sqeps_es", SGDRegressor(loss="squared_epsilon_insensitive", early_stopping=True, random_state=3))
reg("par", PassiveAggressiveRegressor(random_state=0))

m = SGDRegressor(random_state=0)
for _ in range(3):
    m.partial_fit(X, yr)
out.append(("pf_coef", L(m.coef_) + L(m.intercept_) + [float(m.t_)]))

oc = SGDOneClassSVM(nu=0.2, random_state=0).fit(X)
out.append(("oc_coef", L(oc.coef_) + L(oc.offset_) + [float(oc.n_iter_)]))
out.append(("oc_pred", L(oc.predict(X[:10]))))

for name, vals in out:
    print(f"    ({name!r}, {vals!r}),")
