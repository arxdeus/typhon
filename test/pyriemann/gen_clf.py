# Generates test_classif.py (FGDA, FgMDM, TSClassifier, KNN, SVC, MeanField) from real pyriemann 0.12.
import numpy as np, warnings, os
warnings.simplefilter("ignore")
from pyriemann.geometry import covariance as C
from pyriemann import classification as K, tangentspace as T
D = os.path.dirname(__file__)
def signals(n_trials, n_ch, n_t):
    X = np.zeros((n_trials, n_ch, n_t))
    for i in range(n_trials):
        for c in range(n_ch):
            for t in range(n_t):
                v = (i * 1009 + c * 7919 + t * 104729 + i * c * t * 13
                     + t * t * (c + 1) * 3 + c * c * t * 7) % 101
                X[i, c, t] = (v - 50.0) / 25.0 + (0.5 * (i % 2)) * ((c + t) % 3 == 0)
    return X
def lit(a): return repr(np.asarray(a, dtype=float).ravel().tolist())
head = open(os.path.join(D, "test_pipeline.py")).read().split("P = spd(5, 4)")[0]
head = head.replace("import numpy as np\n", "import numpy as np\nfrom pyriemann.geometry.covariance import covariances\nfrom pyriemann.tangentspace import FGDA\nfrom pyriemann.classification import FgMDM, TSClassifier, KNearestNeighbor, SVC, MeanField\n", 1)
X = signals(16, 4, 40)
# make classes separable-ish: scale channel 0 for class 1
y = np.array([(i // 2) % 2 for i in range(16)])
X[y == 1, 0, :] *= 1.8
cov = C.covariances(X, "scm")
Xtr, Xte, ytr, yte = cov[:12], cov[12:], y[:12], y[12:]
L = []; Tt = []
def add(name, arr, call, atol=1e-8):
    L.append(f"E_{name} = {lit(arr)}")
    Tt.append(f"    assert close({call}, E_{name}, atol={atol})")
f = T.FGDA().fit(Xtr, ytr)
add("fgda", f.transform(Xte), "FGDA().fit(Xtr, ytr).transform(Xte)")
add("fgda_ft", T.FGDA().fit_transform(Xtr, ytr), "FGDA().fit_transform(Xtr, ytr)")
m = K.FgMDM().fit(Xtr, ytr)
add("fg_pred", m.predict(Xte), "FgMDM().fit(Xtr, ytr).predict(Xte)")
add("fg_tr", m.transform(Xte), "FgMDM().fit(Xtr, ytr).transform(Xte)")
add("fg_pr", m.predict_proba(Xte), "FgMDM().fit(Xtr, ytr).predict_proba(Xte)")
t = K.TSClassifier().fit(Xtr, ytr)
add("ts_pr", t.predict_proba(Xte), "TSClassifier().fit(Xtr, ytr).predict_proba(Xte)", 1e-6)
add("ts_pred", t.predict(Xte), "TSClassifier().fit(Xtr, ytr).predict(Xte)")
kn = K.KNearestNeighbor(n_neighbors=3).fit(Xtr, ytr)
add("knn_pred", kn.predict(Xte), "KNearestNeighbor(3).fit(Xtr, ytr).predict(Xte)")
add("knn_pr", kn.predict_proba(Xte), "KNearestNeighbor(3).fit(Xtr, ytr).predict_proba(Xte)")
s = K.SVC(C=2.0).fit(Xtr, ytr)
add("svc_df", s.decision_function(Xte), "SVC(\"riemann\", None, 2.0).fit(Xtr, ytr).decision_function(Xte)", 1e-6)
add("svc_pred", s.predict(Xte), "SVC(\"riemann\", None, 2.0).fit(Xtr, ytr).predict(Xte)")
mf = K.MeanField(power_list=[-1, 0, 1]).fit(Xtr, ytr)
add("mf_tr", mf.transform(Xte), "MeanField().fit(Xtr, ytr).transform(Xte)", 1e-7)
add("mf_pred", mf.predict(Xte), "MeanField().fit(Xtr, ytr).predict(Xte)")
mf2 = K.MeanField(power_list=[-0.5, 0.5], method_combination="inf_means").fit(Xtr, ytr)
add("mf2_tr", mf2.transform(Xte), "MeanField([-0.5, 0.5], \"inf_means\").fit(Xtr, ytr).transform(Xte)", 1e-7)
setup = '''X = signals(16, 4, 40)
y = np.array([(i // 2) % 2 for i in range(16)])
for i in range(16):
    if y[i] == 1:
        X[i, 0, :] = X[i, 0, :] * 1.8
cov = covariances(X, "scm")
Xtr = cov[:12]
Xte = cov[12:]
ytr = y[:12]
'''
open(os.path.join(D, "test_classif.py"), "w").write(head + setup + "\n".join(L) + "\n\n@test\ndef test_classif():\n" + "\n".join(Tt) + "\ntest_classif()\n")
