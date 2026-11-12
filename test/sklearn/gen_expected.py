# Generates the expected literals for test/sklearn/*.py (scikit-learn 1.9.1).
#   /Users/mind/orca/projects/typhon-test/.venv/bin/python test/sklearn/gen_expected.py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline

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

def LI(name, a):
    print(f"{name} = {[int(v) for v in np.asarray(a).ravel()]!r}")

for nc in (2, 3):
    X, y = data(60, 5, nc)
    print(f"# nc={nc}")
    s = StandardScaler().fit(X)
    L(f"sc_mean_{nc}", s.mean_); L(f"sc_scale_{nc}", s.scale_); L(f"sc_t_{nc}", s.transform(X[:4]))
    lda = LinearDiscriminantAnalysis().fit(X, y)
    L(f"lda_coef_{nc}", lda.coef_); L(f"lda_int_{nc}", lda.intercept_)
    L(f"lda_dec_{nc}", lda.decision_function(X[:6])); LI(f"lda_pred_{nc}", lda.predict(X))
    L(f"lda_tr_{nc}", lda.transform(X[:4]))
    lr = LogisticRegression(max_iter=1000, random_state=0, tol=1e-12).fit(X, y)
    L(f"lr_coef_{nc}", lr.coef_); L(f"lr_int_{nc}", lr.intercept_)
    LI(f"lr_pred_{nc}", lr.predict(X)); L(f"lr_proba_{nc}", lr.predict_proba(X[:4]))
    sv = SVC(kernel="rbf", random_state=0).fit(X, y)
    LI(f"svc_support_{nc}", sv.support_); LI(f"svc_nsup_{nc}", sv.n_support_)
    L(f"svc_dual_{nc}", sv.dual_coef_); L(f"svc_int_{nc}", sv.intercept_)
    L(f"svc_dec_{nc}", sv.decision_function(X[:6])); LI(f"svc_pred_{nc}", sv.predict(X))
    cv = KFold(5)
    for nm, m in (("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
                  ("lda", make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())),
                  ("svc", make_pipeline(StandardScaler(), SVC()))):
        L(f"cvs_{nm}_{nc}", cross_val_score(m, X, y, cv=cv))
        LI(f"cvp_{nm}_{nc}", cross_val_predict(m, X, y, cv=cv))
for seed in (0, 42):
    print(f"kf_shuf_{seed} = {[[int(v) for v in te] for _, te in KFold(4, shuffle=True, random_state=seed).split(np.zeros(23))]!r}")

# ---- extended coverage (test_extended.py)
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix, f1_score
from sklearn.decomposition import PCA
print("# ---- extended")
for nc in (2, 3):
    X, y = data(60, 5, nc)
    for solver, shr in (("lsqr", None), ("lsqr", "auto"), ("eigen", None), ("eigen", "auto"), ("lsqr", 0.3)):
        tag = f"{solver}_{'none' if shr is None else str(shr).replace('.', 'p')}_{nc}"
        m = LinearDiscriminantAnalysis(solver=solver, shrinkage=shr).fit(X, y)
        L(f"ldx_coef_{tag}", m.coef_); L(f"ldx_int_{tag}", m.intercept_); LI(f"ldx_pred_{tag}", m.predict(X))
        L(f"ldx_cov_{tag}", m.covariance_)
    lr = LogisticRegression(max_iter=1000, tol=1e-12, C=0.5).fit(X, y)
    L(f"lrx_proba_{nc}", lr.predict_proba(X[:8])); L(f"lrx_coef_{nc}", lr.coef_)
    for k in ("linear", "poly", "sigmoid"):
        sv = SVC(kernel=k, gamma=0.1 if k == "sigmoid" else "scale", coef0=0.5 if k != "linear" else 0.0).fit(X, y)
        LI(f"svk_sup_{k}_{nc}", sv.support_); L(f"svk_dual_{k}_{nc}", sv.dual_coef_)
        L(f"svk_int_{k}_{nc}", sv.intercept_); LI(f"svk_pred_{k}_{nc}", sv.predict(X))
    sp = SVC(probability=True, random_state=0).fit(X, y)
    L(f"svp_A_{nc}", sp.probA_); L(f"svp_B_{nc}", sp.probB_); L(f"svp_proba_{nc}", sp.predict_proba(X[:8]))
    LI(f"svp_pred_{nc}", sp.predict(X))
    pca = PCA(n_components=3).fit(X)
    L(f"pca_comp_{nc}", pca.components_); L(f"pca_evr_{nc}", pca.explained_variance_ratio_)
    L(f"pca_ev_{nc}", pca.explained_variance_); L(f"pca_t_{nc}", pca.transform(X[:5]))
    pf = PCA().fit(X)
    L(f"pcaf_ev_{nc}", pf.explained_variance_)
    gs = GridSearchCV(make_pipeline(StandardScaler(), SVC()), {"svc__C": [0.1, 1.0, 10.0]}, cv=KFold(4)).fit(X, y)
    L(f"gs_mean_{nc}", gs.cv_results_["mean_test_score"]); print(f"gs_best_{nc} = {gs.best_index_}")
    gl = GridSearchCV(LogisticRegression(max_iter=500), {"C": [0.01, 1.0]}, cv=3).fit(X, y)
    L(f"gl_mean_{nc}", gl.cv_results_["mean_test_score"]); print(f"gl_best_{nc} = {gl.best_index_}")
ys = np.array([0, 1, 1, 0, 2, 2, 1, 0, 0, 2, 1, 1, 0, 2, 0, 1, 2, 0, 1, 1, 0, 0, 2])
for sh, rs in ((False, None), (True, 0), (True, 7)):
    print(f"skf_{sh}_{rs} = {[[int(v) for v in te] for _, te in StratifiedKFold(3, shuffle=sh, random_state=rs).split(np.zeros(23), ys)]!r}")
Xs = np.arange(46.0).reshape(23, 2)
for rs, ts in ((0, 0.25), (3, 0.3), (5, 5)):
    a, b, c, d = train_test_split(Xs, ys, test_size=ts, random_state=rs)
    print(f"tts_{rs} = {[[int(v) for v in a[:, 0] // 2], [int(v) for v in b[:, 0] // 2]]!r}")
a, b = train_test_split(Xs, test_size=0.25, random_state=1, stratify=ys)
print(f"tts_strat = {[[int(v) for v in a[:, 0] // 2], [int(v) for v in b[:, 0] // 2]]!r}")
yt = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1]); yp = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1])
sc = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.6, 0.3, 0.7, 0.4, 0.95])
print(f"m_acc = {accuracy_score(yt, yp)!r}"); print(f"m_bacc = {balanced_accuracy_score(yt, yp)!r}")
print(f"m_auc = {roc_auc_score(yt, sc)!r}"); LI("m_cm", confusion_matrix(yt, yp))
print(f"m_f1 = {f1_score(yt, yp)!r}")
y3t = np.array([0, 1, 2, 2, 1, 0, 2, 1]); y3p = np.array([0, 2, 2, 1, 1, 0, 2, 0])
LI("m_cm3", confusion_matrix(y3t, y3p)); print(f"m_bacc3 = {balanced_accuracy_score(y3t, y3p)!r}")
for av in ("macro", "micro", "weighted"):
    print(f"m_f1_{av} = {f1_score(y3t, y3p, average=av)!r}")
P3 = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.2, 0.2, 0.6], [0.3, 0.4, 0.3], [0.2, 0.6, 0.2], [0.5, 0.3, 0.2], [0.1, 0.1, 0.8], [0.4, 0.4, 0.2]])
print(f"m_auc_ovr = {roc_auc_score(y3t, P3, multi_class='ovr')!r}")
