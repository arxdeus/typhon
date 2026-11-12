# Generates the expected literals of test_histgb.py (run with scikit-learn 1.9.1):
#   python gen_histgb_expected.py > expected.txt
import math
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HC, HistGradientBoostingRegressor as HR

def data(n, p, nc):
    X = np.zeros((n, p)); y = np.zeros(n, dtype=int)
    for i in range(n):
        y[i] = i % nc
        for j in range(p):
            v = (i * 7919 + j * 104729 + i * j * 13 + j * j * 3) % 101
            X[i, j] = (v - 50.0) / 25.0 + (0.8 * y[i] if (j + y[i]) % 2 == 0 else -0.3 * y[i])
        X[i, p - 1] = math.sin(i * 1.7) * 3.0 + 0.01 * i
    return X, y

def out(name, v):
    a = np.asarray(v, dtype=float).ravel()
    print(f"{name} = [{', '.join(repr(float(x)) for x in a)}]")

X, y = data(600, 5, 3)
yr = np.array([math.sin(X[i, 0]) * 2.0 + X[i, 1] * X[i, 2] + 0.1 * (i % 7) for i in range(600)])
yb = (y == 1).astype(int)
Xt = X[::37]
Xn = X.copy()
for i in range(600):
    if i % 7 == 0:
        Xn[i, 1] = np.nan
    if i % 11 == 3:
        Xn[i, 3] = np.nan
Xtn = Xn[::37]
sw = np.array([1.0 + (i % 5) * 0.5 for i in range(600)])
# categorical column with 6 levels (raw values 0..5 shifted) + missing
Xc = X.copy()
for i in range(600):
    Xc[i, 2] = float((i * 7) % 6) * 2.0 + 1.0
    if i % 13 == 0:
        Xc[i, 2] = np.nan
yc = yr + np.array([1.5 if Xc[i, 2] in (3.0, 9.0) else 0.0 for i in range(600)])
Xtc = Xc[::37]

r = HR(max_iter=15, random_state=0).fit(X, yr)
out("r_pred", r.predict(Xt)); out("r_train", r.train_score_)
out("r_th4", r._bin_mapper.bin_thresholds_[4]); out("r_th0", r._bin_mapper.bin_thresholds_[0])
r16 = HR(max_iter=8, max_bins=16, max_leaf_nodes=8, l2_regularization=0.5, random_state=0).fit(X, yr, sample_weight=sw)
out("r16_pred", r16.predict(Xt)); out("r16_th4", r16._bin_mapper.bin_thresholds_[4])
for loss in ["absolute_error", "poisson", "gamma"]:
    yy = yr if loss == "absolute_error" else np.exp(yr / 3.0)
    m = HR(loss=loss, max_iter=10, max_depth=3, random_state=0).fit(X, yy)
    out(f"r_{loss}", m.predict(Xt))
mq = HR(loss="quantile", quantile=0.8, max_iter=10, min_samples_leaf=10, random_state=0).fit(X, yr)
out("r_quantile", mq.predict(Xt))
mn = HR(max_iter=10, random_state=0).fit(Xn, yr)
out("r_nan", mn.predict(Xtn))
mc = HR(max_iter=10, categorical_features=[2], random_state=0).fit(Xc, yc)
out("r_cat", mc.predict(Xtc))
mm = HR(max_iter=10, monotonic_cst=[1, 0, -1, 0, 0], random_state=0).fit(X, yr)
out("r_mono", mm.predict(Xt))
mi = HR(max_iter=10, interaction_cst=[[0, 1], [2, 3, 4]], random_state=0).fit(X, yr)
out("r_ic", mi.predict(Xt))
mf = HR(max_iter=10, max_features=0.5, random_state=3).fit(X, yr)
out("r_mf", mf.predict(Xt))
me = HR(max_iter=200, early_stopping=True, n_iter_no_change=3, random_state=1).fit(X, yr)
print(f"r_es_n = {me.n_iter_}"); out("r_es_val", me.validation_score_); out("r_es_pred", me.predict(Xt))
me2 = HR(max_iter=200, early_stopping=True, scoring=None, validation_fraction=None, n_iter_no_change=2, tol=1e-3, random_state=1).fit(X, yr)
print(f"r_es2_n = {me2.n_iter_}"); out("r_es2_train", me2.train_score_)
w = HR(max_iter=5, warm_start=True, random_state=0).fit(X, yr)
w.set_params(max_iter=12); w.fit(X, yr)
out("r_warm", w.predict(Xt))
st = list(HR(max_iter=4, random_state=0).fit(X, yr).staged_predict(Xt[:3]))
out("r_staged", np.array(st))

c = HC(max_iter=10, random_state=0).fit(X, y)
out("c_proba", c.predict_proba(Xt))
cb = HC(max_iter=10, random_state=0, class_weight="balanced").fit(X, yb)
out("cb_proba", cb.predict_proba(Xt)); out("cb_dec", cb.decision_function(Xt))
ces = HC(max_iter=100, early_stopping=True, n_iter_no_change=2, scoring=None, random_state=2).fit(X, y)
print(f"c_es_n = {ces.n_iter_}"); out("c_es_val", ces.validation_score_); out("c_es_train", ces.train_score_)
ces2 = HC(max_iter=100, early_stopping=True, n_iter_no_change=3, random_state=5).fit(Xn, yb, sample_weight=sw)
print(f"c_es2_n = {ces2.n_iter_}"); out("c_es2_val", ces2.validation_score_); out("c_es2_proba", ces2.predict_proba(Xtn))
cc = HC(max_iter=8, categorical_features=[False, False, True, False, False], random_state=0).fit(Xc, y)
out("c_cat", cc.predict_proba(Xtc))
cs = [p[:, 1] for p in HC(max_iter=3, random_state=0).fit(X, yb).staged_predict_proba(Xt[:4])]
out("c_staged", np.array(cs))
