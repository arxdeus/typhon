# Expected literals for test/sklearn/test_model_selection_more.py (scikit-learn 1.9.1).
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import uniform, expon
from sklearn.model_selection import (GroupKFold, StratifiedGroupKFold, RepeatedKFold,
    RepeatedStratifiedKFold, ShuffleSplit, StratifiedShuffleSplit, GroupShuffleSplit,
    LeaveOneOut, LeavePOut, LeaveOneGroupOut, LeavePGroupsOut, PredefinedSplit, TimeSeriesSplit,
    train_test_split, cross_validate, cross_val_predict, cross_val_score, GridSearchCV,
    RandomizedSearchCV, ParameterSampler, learning_curve, validation_curve, permutation_test_score)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier

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

def S(name, cv, X, y=None, groups=None):
    tr = []; te = []
    for a, b in cv.split(X, y, groups):
        tr.extend(list(a) + [-1]); te.extend(list(b) + [-1])
    L(name + "_tr", tr); L(name + "_te", te)

X, y = data(24, 3, 3)
g = np.array([i % 6 for i in range(24)]) + np.array([i // 12 for i in range(24)]) * 0
yr = X[:, 0] * 1.5 - X[:, 1] + 0.1 * np.arange(24)
S("gkf", GroupKFold(3), X, y, g)
S("gkfs", GroupKFold(3, shuffle=True, random_state=1), X, y, g)
S("sgkf", StratifiedGroupKFold(3), X, y, np.arange(24) // 3)
S("rkf", RepeatedKFold(n_splits=3, n_repeats=2, random_state=0), X)
S("rskf", RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=3), X, y)
S("ss", ShuffleSplit(3, test_size=0.25, random_state=0), X)
S("ss2", ShuffleSplit(2, test_size=5, train_size=10, random_state=4), X)
S("sss", StratifiedShuffleSplit(3, test_size=0.25, random_state=0), X, y)
S("gss", GroupShuffleSplit(2, test_size=0.34, random_state=0), X, y, g)
S("loo", LeaveOneOut(), X[:4])
S("lpo", LeavePOut(2), X[:4])
S("logo", LeaveOneGroupOut(), X, y, g)
S("lpgo", LeavePGroupsOut(2), X[:12], None, g[:12] % 4)
S("ps", PredefinedSplit([0, 1, -1, 0, 1, 2, 2, -1]), X[:8])
S("tss", TimeSeriesSplit(3), X)
S("tss2", TimeSeriesSplit(3, max_train_size=6, test_size=3, gap=1), X)
a, b, c, d = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
L("tts_a", a); L("tts_d", d)
r = cross_validate(LogisticRegression(), X, y, cv=3, scoring=["accuracy", "f1_macro"], return_train_score=True, return_estimator=True)
L("cv_acc", r["test_accuracy"]); L("cv_f1", r["test_f1_macro"]); L("cv_tr_acc", r["train_accuracy"])
L("cv_coef0", r["estimator"][0].coef_)
r2 = cross_validate(Ridge(), X, yr, cv=4, scoring="neg_mean_squared_error")
L("cv_r", r2["test_score"])
L("cvs_r2", cross_val_score(Ridge(alpha=2.0), X, yr, cv=3))
L("cvp_proba", cross_val_predict(LogisticRegression(), X, y, cv=3, method="predict_proba"))
L("cvp_dec", cross_val_predict(LogisticRegression(), X, (y == 1).astype(int), cv=3, method="decision_function"))
gs = GridSearchCV(LogisticRegression(), {"C": [0.01, 0.1, 1.0], "fit_intercept": [True, False]}, cv=3, return_train_score=True).fit(X, y)
L("gs_mean", gs.cv_results_["mean_test_score"]); L("gs_rank", gs.cv_results_["rank_test_score"])
L("gs_std", gs.cv_results_["std_test_score"]); L("gs_trmean", gs.cv_results_["mean_train_score"])
print(f"gs_best = {gs.best_index_}"); L("gs_coef", gs.best_estimator_.coef_)
gk = GridSearchCV(KNeighborsClassifier(), [{"n_neighbors": [1, 3, 5], "weights": ["uniform", "distance"]}, {"n_neighbors": [7]}], cv=4, scoring={"acc": "accuracy", "bal": "balanced_accuracy"}, refit="bal").fit(X, y)
L("gk_acc", gk.cv_results_["mean_test_acc"]); L("gk_bal", gk.cv_results_["mean_test_bal"]); L("gk_rank", gk.cv_results_["rank_test_bal"])
print(f"gk_best = {gk.best_index_}"); print(f"gk_bs = {gk.best_score_!r}")
ps = list(ParameterSampler({"a": [1, 2, 3, 4], "b": [10.0, 20.0, 30.0]}, n_iter=5, random_state=0))
L("ps_a", [p["a"] for p in ps]); L("ps_b", [p["b"] for p in ps])
ps2 = list(ParameterSampler({"C": uniform(0.1, 5.0), "k": [1, 2, 3], "s": expon(scale=2.0)}, n_iter=4, random_state=2))
L("ps2_C", [p["C"] for p in ps2]); L("ps2_k", [p["k"] for p in ps2]); L("ps2_s", [p["s"] for p in ps2])
rs = RandomizedSearchCV(Ridge(), {"alpha": uniform(0.1, 10.0)}, n_iter=4, cv=3, random_state=0, scoring="r2").fit(X, yr)
L("rs_alpha", [p["alpha"] for p in rs.cv_results_["params"]]); L("rs_mean", rs.cv_results_["mean_test_score"])
print(f"rs_best = {rs.best_index_}")
sz, trs, tes = learning_curve(LogisticRegression(), X, y, cv=3, train_sizes=[0.4, 0.7, 1.0])
L("lc_sz", sz); L("lc_tr", trs); L("lc_te", tes)
sz2, trs2, tes2 = learning_curve(Ridge(), X, yr, cv=3, train_sizes=[5, 10], shuffle=True, random_state=0)
L("lc2_sz", sz2); L("lc2_te", tes2)
vtr, vte = validation_curve(Ridge(), X, yr, param_name="alpha", param_range=[0.1, 1.0, 10.0], cv=3)
L("vc_tr", vtr); L("vc_te", vte)
s0, perms, pv = permutation_test_score(LogisticRegression(), X, y, cv=3, n_permutations=5, random_state=0)
print(f"pt_s = {s0!r}"); L("pt_perm", perms); print(f"pt_pv = {pv!r}")
