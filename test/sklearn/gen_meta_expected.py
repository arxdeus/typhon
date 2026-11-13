# Generates the expected literals for test/sklearn/test_multiclass.py, test_multioutput.py, test_calibration.py (scikit-learn 1.9.1).
#   <venv>/bin/python test/sklearn/gen_meta_expected.py multiclass|multioutput|calibration
import sys
import warnings
import numpy as np
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier, RegressorChain, ClassifierChain
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.kernel_ridge import KernelRidge
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

X, y = data(60, 4, 3)
Q = X[:8] + 0.1
if sys.argv[1] == "multiclass":
    m = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y); L("ovr_dec", m.decision_function(Q)); L("ovr_pred", m.predict(X))
    m = OneVsRestClassifier(GaussianNB()).fit(X, y); L("ovr_nb_proba", m.predict_proba(Q)); L("ovr_nb_pred", m.predict(X))
    m = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X[y < 2], y[y < 2]); L("ovr_bin_pred", m.predict(Q))
    m = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, y); L("ovo_dec", m.decision_function(Q)); L("ovo_pred", m.predict(X))
    m = OneVsOneClassifier(GaussianNB()).fit(X, y); L("ovo_nb_dec", m.decision_function(Q))
    m = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2.0, random_state=4).fit(X, y); L("occ_cb", m.code_book_); L("occ_pred", m.predict(X))
    m = OutputCodeClassifier(GaussianNB(), random_state=5).fit(X, y); L("occ_nb_cb", m.code_book_); L("occ_nb_pred", m.predict(Q))
if sys.argv[1] == "multioutput":
    Y = np.column_stack([X[:, 0] * 2 - X[:, 1], X[:, 2] ** 2, X[:, 3] - X[:, 0]])
    m = MultiOutputRegressor(KernelRidge(alpha=0.5, kernel="rbf", gamma=0.2)).fit(X, Y); L("mor_pred", m.predict(Q))
    Yc = np.column_stack([y, (X[:, 0] > 0).astype(int), (y + (X[:, 1] > 0)) % 3])
    m = MultiOutputClassifier(GaussianNB()).fit(X, Yc); L("moc_pred", m.predict(Q)); L("moc_proba1", m.predict_proba(Q)[1]); print("moc_score =", m.score(X, Yc))
    m = RegressorChain(KernelRidge(alpha=0.5), order=[2, 0, 1]).fit(X, Y); L("rc_pred", m.predict(Q))
    m = RegressorChain(KernelRidge(alpha=0.5), order="random", random_state=3, cv=3).fit(X, Y); L("rc_order", m.order_); L("rc2_pred", m.predict(Q))
    m = ClassifierChain(GaussianNB(), order=[1, 0, 2]).fit(X, Yc); L("cc_pred", m.predict(Q)); 
    Yb = np.column_stack([(y > 0).astype(int), (X[:, 0] > 0).astype(int)])
    m = ClassifierChain(GaussianNB()).fit(X, Yb); L("ccb_proba", m.predict_proba(Q))
    m = ClassifierChain(GaussianNB(), cv=3).fit(X, Yc); L("cc2_pred", m.predict(Q))
if sys.argv[1] != "calibration":
    sys.exit(0)
m = CalibratedClassifierCV(LinearSVC(random_state=0), cv=3).fit(X, y); L("cal_sig_proba", m.predict_proba(Q))
m = CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=4).fit(X, y); L("cal_iso_proba", m.predict_proba(Q))
m = CalibratedClassifierCV(SVC(), cv=3, ensemble=False).fit(X[y < 2], y[y < 2]); L("cal_ne_proba", m.predict_proba(Q))
sw = 1.0 + (np.arange(60) % 3)
m = CalibratedClassifierCV(GaussianNB(), cv=3).fit(X, y, sample_weight=sw); L("cal_sw_proba", m.predict_proba(Q))
pt, pp = calibration_curve((y > 0).astype(int), np.abs(np.sin(np.arange(60.0))), n_bins=4); L("cc_pt", pt); L("cc_pp", pp)
pt, pp = calibration_curve((y > 0).astype(int), np.abs(np.sin(np.arange(60.0))), n_bins=3, strategy="quantile"); L("ccq_pt", pt); L("ccq_pp", pp)
