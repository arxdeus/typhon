# Generates the expected literals for test/sklearn/test_svm.py and test_svm2.py (scikit-learn 1.9.1).
#   <venv>/bin/python test/sklearn/gen_svm_expected.py svm|svm2
import sys, warnings
import numpy as np
from sklearn.svm import SVC, NuSVC, SVR, NuSVR, OneClassSVM, LinearSVC, LinearSVR, l1_min_c
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
X2, y2 = data(40, 3, 2)
yr = X[:, 0] * 1.5 - X[:, 1] + 0.3 * np.sin(np.arange(48.0))
sw = 1.0 + (np.arange(48) % 3)
Q = X[:7] + 0.05
part = sys.argv[1]

def dump(tag, m, cls=True, Q=Q):
    L(tag + "_sup", m.support_); L(tag + "_dual", m.dual_coef_); L(tag + "_icpt", m.intercept_)
    if cls:
        L(tag + "_nsup", m.n_support_)
    L(tag + "_dec", m.decision_function(Q) if hasattr(m, "decision_function") else m.predict(Q))
    L(tag + "_pred", m.predict(Q))

if part == "svm":
    dump("nusvc", NuSVC(nu=0.3).fit(X, y))
    dump("nusvc2", NuSVC(nu=0.4, kernel="linear").fit(X2, y2), Q=X2[:7] + 0.05)
    dump("nusvc_ovo", NuSVC(nu=0.3, decision_function_shape="ovo", kernel="poly", degree=2).fit(X, y))
    dump("svr", SVR(C=2.0, epsilon=0.2).fit(X, yr), False)
    dump("svr_lin", SVR(kernel="linear", C=0.5).fit(X, yr, sample_weight=sw), False)
    dump("nusvr", NuSVR(nu=0.4, C=3.0).fit(X, yr), False)
    oc = OneClassSVM(nu=0.2, gamma=0.3).fit(X)
    dump("oc", oc, False); L("oc_ss", oc.score_samples(Q)); L("oc_off", oc.offset_)
    dump("svc_cw", SVC(class_weight={0: 2.0, 2: 0.5}).fit(X, y, sample_weight=sw))
    dump("svc_bal", SVC(class_weight="balanced", kernel="sigmoid", gamma=0.1).fit(X[:40], y[:40]))
    m = SVC(break_ties=True).fit(X, y); L("svc_bt_pred", m.predict(Q))
    m = SVC(kernel="linear", C=0.3).fit(X, y); L("svc_coef", m.coef_)
    m = SVC(kernel="precomputed").fit(X @ X.T, y); L("svc_pre_dual", m.dual_coef_); L("svc_pre_dec", m.decision_function(Q @ X.T))
    m = NuSVC(nu=0.3, probability=True, random_state=1).fit(X, y); L("nusvc_pA", m.probA_); L("nusvc_pB", m.probB_); L("nusvc_proba", m.predict_proba(Q))
else:
    for k, kw in (("l2sq", dict()), ("l2hinge", dict(loss="hinge", dual=True)),
                  ("primal", dict(dual=False)), ("l1", dict(penalty="l1", dual=False, C=0.5))):
        m = LinearSVC(random_state=0, **kw).fit(X, y)
        L(f"lsvc_{k}_coef", m.coef_); L(f"lsvc_{k}_icpt", m.intercept_); print(f"lsvc_{k}_it =", m.n_iter_)
        L(f"lsvc_{k}_dec", m.decision_function(Q))
    m = LinearSVC(random_state=3, class_weight="balanced", C=2.0).fit(X2, y2, sample_weight=sw[:40])
    L("lsvc_b_coef", m.coef_); L("lsvc_b_icpt", m.intercept_); L("lsvc_b_pred", m.predict(X2[:9]))
    m = LinearSVC(random_state=0, fit_intercept=False, max_iter=50).fit(X, y); L("lsvc_ni_coef", m.coef_)
    for k, kw in (("eps", dict()), ("sqd", dict(loss="squared_epsilon_insensitive", dual=True)),
                  ("sqp", dict(loss="squared_epsilon_insensitive", dual=False))):
        m = LinearSVR(random_state=0, epsilon=0.1, **kw).fit(X, yr)
        L(f"lsvr_{k}_coef", m.coef_); L(f"lsvr_{k}_icpt", m.intercept_); L(f"lsvr_{k}_pred", m.predict(Q))
    print("l1c =", l1_min_c(X, y)); print("l1c2 =", l1_min_c(X2, y2, loss="log", intercept_scaling=3.0))
