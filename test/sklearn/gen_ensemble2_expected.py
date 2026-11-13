# Generates the expected literals of test_ensemble2.py (run with scikit-learn 1.9.1):
#   python gen_ensemble2_expected.py > expected.txt
import math
import numpy as np
from sklearn.ensemble import (StackingClassifier, StackingRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor, AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV, Ridge, LinearRegression
from sklearn.inspection import permutation_importance, partial_dependence

def data(n, p, nc):
    X = np.zeros((n, p)); y = np.zeros(n, dtype=int)
    for i in range(n):
        y[i] = i % nc
        for j in range(p):
            v = (i * 7919 + j * 104729 + i * j * 13 + j * j * 3) % 101
            X[i, j] = (v - 50.0) / 25.0 + (0.8 * y[i] if (j + y[i]) % 2 == 0 else -0.3 * y[i])
    return X, y

def out(name, v):
    a = np.asarray(v, dtype=float).ravel()
    print(f"{name} = [{', '.join(repr(float(x)) for x in a)}]")

X, y = data(120, 4, 3)
yb = (y == 1).astype(int)
yr = np.array([math.sin(X[i, 0]) * 2.0 + X[i, 1] * X[i, 2] + 0.1 * (i % 7) for i in range(120)])
Xt = X[::7]

# stacking
sr = StackingRegressor((("t", DecisionTreeRegressor(max_depth=3, random_state=0)), ("e", DecisionTreeRegressor(max_depth=5, random_state=1))),
                       final_estimator=DecisionTreeRegressor(max_depth=3, random_state=0)).fit(X, yr)
out("st_reg", sr.predict(Xt))
out("st_reg_tr", sr.transform(Xt))
srp = StackingRegressor((("t", DecisionTreeRegressor(max_depth=2, random_state=0)),),
                        final_estimator=DecisionTreeRegressor(max_depth=4, random_state=0), cv=3, passthrough=True).fit(X, yr)
out("st_reg_pt", srp.predict(Xt))
sc = StackingClassifier((("t", DecisionTreeClassifier(max_depth=3, random_state=0)),
                         ("g", GradientBoostingClassifier(n_estimators=5, random_state=0))),
                        final_estimator=DecisionTreeClassifier(max_depth=3, random_state=0)).fit(X, y)
out("st_clf_tr", sc.transform(Xt))
out("st_clf", sc.predict(Xt))
scb = StackingClassifier((("t", DecisionTreeClassifier(max_depth=3, random_state=0)),),
                         final_estimator=DecisionTreeClassifier(max_depth=2, random_state=0),
                         stack_method="predict").fit(X, yb)
out("st_clfb_tr", scb.transform(Xt))

# GB extras
g = GradientBoostingRegressor(n_estimators=200, n_iter_no_change=3, validation_fraction=0.2,
                              random_state=0, max_depth=2).fit(X, yr)
print(f"gb_es_n = {g.n_estimators_}")
out("gb_es", g.predict(Xt))
g2 = GradientBoostingRegressor(n_estimators=10, warm_start=True, random_state=0, max_depth=2).fit(X, yr)
g2.set_params(n_estimators=20); g2.fit(X, yr)
out("gb_ws", g2.predict(Xt))
g3 = GradientBoostingRegressor(n_estimators=10, init="zero", random_state=0, max_depth=2).fit(X, yr)
out("gb_zero", g3.predict(Xt))
g4 = GradientBoostingRegressor(n_estimators=10, init=DecisionTreeRegressor(max_depth=2, random_state=0),
                               random_state=0, max_depth=2).fit(X, yr)
out("gb_init", g4.predict(Xt))
gc = GradientBoostingClassifier(n_estimators=10, random_state=0, max_depth=2).fit(X, yb)
out("gb_dec1d", gc.decision_function(Xt))
gce = GradientBoostingClassifier(n_estimators=100, n_iter_no_change=2, random_state=0, max_depth=2).fit(X, y)
print(f"gbc_es_n = {gce.n_estimators_}")
out("gbc_es", gce.predict_proba(Xt))
ab = AdaBoostClassifier(n_estimators=10, random_state=0).fit(X, yb)
out("ab_dec1d", ab.decision_function(Xt))

# inspection
t = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, yr)
pi = permutation_importance(t, X, yr, n_repeats=3, random_state=0)
out("pi_mean", pi.importances_mean)
out("pi_all", pi.importances)
tc = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
pic = permutation_importance(tc, X, y, n_repeats=4, random_state=1, max_samples=0.5)
out("pic_all", pic.importances)
pd = partial_dependence(t, X, [1], grid_resolution=10, method="brute", kind="both")
out("pd_grid", pd["grid_values"][0])
out("pd_avg", pd["average"])
out("pd_ind", pd["individual"][:, ::10, :])
pd2 = partial_dependence(tc, X, [0, 2], grid_resolution=5, method="brute")
out("pd2_avg", pd2["average"])
