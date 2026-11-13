# Generates test_potato.py (Potato, PotatoField) from real pyriemann 0.12.
import numpy as np, warnings, os
warnings.simplefilter("ignore")
from pyriemann.geometry import covariance as C
from pyriemann.artifact_detection import Potato, PotatoField
D = os.path.dirname(__file__)
src = open(os.path.join(D, "test_clust2.py")).read()
head = src.split("X = signals(30, 3, 40)")[0]
head = head.replace("from pyriemann.clustering import MeanShift, GaussianMixture\n", "from pyriemann.artifact_detection import Potato, PotatoField\n")
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
X = signals(24, 4, 40)
X[5] *= 6.0; X[17, 1] *= 9.0
cov = C.covariances(X, "scm")
c2 = cov[:, :3, :3]
L = []; T = []
def add(name, arr, call, atol=1e-8):
    L.append(f"E_{name} = {lit(arr)}")
    T.append(f"    assert close({call}, E_{name}, atol={atol})")
p = Potato(threshold=2.0).fit(cov)
add("pt_z", p.transform(cov), 'Potato("riemann", 2.0).fit(cov).transform(cov)', 1e-7)
add("pt_pred", p.predict(cov), 'Potato("riemann", 2.0).fit(cov).predict(cov)')
add("pt_pr", p.predict_proba(cov), 'Potato("riemann", 2.0).fit(cov).predict_proba(cov)', 1e-7)
add("pt_cm", p.covmean_, 'Potato("riemann", 2.0).fit(cov).covmean_', 1e-7)
p2 = Potato(metric="logeuclid", threshold=1.5).fit(cov[:12], np.array([1] * 10 + [0, 0])); p2.partial_fit(cov[12:], alpha=0.3)
add("pt_pf", p2.transform(cov), 'pf_potato(cov)', 1e-7)
pf = PotatoField(n_potatoes=2, p_threshold=0.05, z_threshold=2.0).fit([cov, c2])
add("pf_z", pf.transform([cov, c2]), 'PotatoField(2, 0.05, 2.0).fit([cov, c2]).transform([cov, c2])', 1e-7)
add("pf_pr", pf.predict_proba([cov, c2]), 'PotatoField(2, 0.05, 2.0).fit([cov, c2]).predict_proba([cov, c2])', 1e-7)
add("pf_pred", pf.predict([cov, c2]), 'PotatoField(2, 0.05, 2.0).fit([cov, c2]).predict([cov, c2])')
pf2 = PotatoField(n_potatoes=2, p_threshold=0.05, z_threshold=2.0, method_combination="stouffer").fit([cov, c2])
add("pf2_pr", pf2.predict_proba([cov, c2]), 'PotatoField(2, 0.05, 2.0, "riemann", 10, 1, 0, "stouffer").fit([cov, c2]).predict_proba([cov, c2])', 1e-7)
setup = '''X = signals(24, 4, 40)
X[5] = X[5] * 6.0
X[17, 1] = X[17, 1] * 9.0
cov = covariances(X, "scm")
c2 = cov[:, 0:3, 0:3].copy()
def pf_potato(c):
    p = Potato("logeuclid", 1.5)
    p.fit(c[:12], np.array([1] * 10 + [0, 0]))
    p.partial_fit(c[12:], None, None, 0.3)
    return p.transform(c)

'''
open(os.path.join(D, "test_potato.py"), "w").write(head + setup + "\n".join(L) + "\n\n@test\ndef test_potato():\n" + "\n".join(T) + "\ntest_potato()\n")
