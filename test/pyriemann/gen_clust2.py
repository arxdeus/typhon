# Generates test_clust2.py (MeanShift, GaussianMixture) from real pyriemann 0.12.
import numpy as np, warnings, os
warnings.simplefilter("ignore")
from pyriemann.geometry import covariance as C
from pyriemann.clustering import MeanShift, GaussianMixture
D = os.path.dirname(__file__)
src = open(os.path.join(D, "test_tl2.py")).read()
head = src.split("X = signals(18, 4, 60)")[0]
head = head.replace("from pyriemann.estimation import Covariances\nfrom pyriemann.transfer import TLRotate\n", "from pyriemann.clustering import MeanShift, GaussianMixture\n")
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
X = signals(30, 3, 40)
for i in range(30):
    if (i // 5) % 2 == 1:
        X[i, 0, :] *= 2.5
cov = C.covariances(X, "scm")
L = []; T = []
def add(name, arr, call, atol=1e-8):
    L.append(f"E_{name} = {lit(arr)}")
    T.append(f"    assert close({call}, E_{name}, atol={atol})")
ms = MeanShift().fit(cov)
add("ms_modes", ms.modes_, 'MeanShift().fit(cov).modes_', 1e-6)
add("ms_lab", ms.labels_, 'MeanShift().fit(cov).labels_')
ms2 = MeanShift(kernel="normal").fit(cov)
add("ms2_modes", ms2.modes_, 'MeanShift("normal").fit(cov).modes_', 1e-6)
g = GaussianMixture(n_components=2, random_state=1).fit(cov)
add("gm_w", g.weights_, 'GaussianMixture(2, "riemann", None, None, 1e-5, 100, 1).fit(cov).weights_', 1e-6)
add("gm_means", g.means_, 'GaussianMixture(2, "riemann", None, None, 1e-5, 100, 1).fit(cov).means_', 1e-5)
add("gm_pr", g.predict_proba(cov), 'GaussianMixture(2, "riemann", None, None, 1e-5, 100, 1).fit(cov).predict_proba(cov)', 1e-5)
add("gm_sc", np.array([g.score(cov)]), 'GaussianMixture(2, "riemann", None, None, 1e-5, 100, 1).fit(cov).score(cov)', 1e-5)
setup = '''X = signals(30, 3, 40)
for i in range(30):
    if (i // 5) % 2 == 1:
        X[i, 0, :] = X[i, 0, :] * 2.5
cov = covariances(X, "scm")

'''
open(os.path.join(D, "test_clust2.py"), "w").write(head + setup + "\n".join(L) + "\n\n@test\ndef test_clust2():\n" + "\n".join(T) + "\ntest_clust2()\n")
