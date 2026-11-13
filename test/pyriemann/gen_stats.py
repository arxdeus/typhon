# Generates test_stats.py (PermutationDistance, PermutationModel) from real pyriemann 0.12.
import numpy as np, warnings, os
warnings.simplefilter("ignore")
from pyriemann.geometry import covariance as C
from pyriemann.stats import PermutationDistance, PermutationModel
D = os.path.dirname(__file__)
src = open(os.path.join(D, "test_nch.py")).read()
head = src.split("X = signals(12, 4, 60)")[0]
head = head.replace("from pyriemann.classification import NearestConvexHull\n", "from pyriemann.stats import PermutationDistance, PermutationModel\n")
def signals(n_trials, n_ch, n_t):
    X = np.zeros((n_trials, n_ch, n_t))
    for i in range(n_trials):
        for c in range(n_ch):
            for t in range(n_t):
                v = (i * 1009 + c * 7919 + t * 104729 + i * c * t * 13
                     + t * t * (c + 1) * 3 + c * c * t * 7) % 101
                X[i, c, t] = (v - 50.0) / 25.0 + ((0.5 * (i % 2)) if (c + t) % 3 == 0 else 0.0)
    return X
def lit(a): return repr(np.asarray(a, dtype=float).ravel().tolist())
X = signals(12, 4, 60)
cov = C.covariances(X, "scm")
y = np.array([i % 3 for i in range(12)])
L = []; T = []
def add(name, arr, call, atol=1e-7):
    L.append(f"E_{name} = {lit(arr)}")
    T.append(f"    assert close({call}, E_{name}, atol={atol})")
for mode in ("pairwise", "ttest", "ftest"):
    p, s = PermutationDistance(20, mode=mode).test(cov, y, verbose=False)
    add(f"pd_{mode}", s, f'PermutationDistance(20, "riemann", "{mode}").test(cov, y, None, False)[1]')
    L.append(f"E_pv_{mode} = {float(p)!r}")
    T.append(f'    assert abs(PermutationDistance(20, "riemann", "{mode}").test(cov, y, None, False)[0] - E_pv_{mode}) < 1e-12')
p, s = PermutationDistance(15, "logeuclid", "ttest").test(cov, y, verbose=False)
add("pd_le", s, 'PermutationDistance(15, "logeuclid", "ttest").test(cov, y, None, False)[1]')
yb = np.array([i % 2 for i in range(12)])
p, s = PermutationModel(8, cv=3).test(cov, yb, verbose=False)
add("pm", s, 'PermutationModel(8, MDM(), 3).test(cov, yb, None, False)[1]')
setup = '''X = signals(12, 4, 60)
cov = covariances(X, "scm")
y = np.array([i % 3 for i in range(12)])
yb = np.array([i % 2 for i in range(12)])

'''
head = head.replace("from pyriemann.stats import", "from pyriemann.classification import MDM\nfrom pyriemann.stats import")
open(os.path.join(D, "test_stats.py"), "w").write(head + setup + "\n".join(L) + "\n\n@test\ndef test_stats():\n" + "\n".join(T) + "\ntest_stats()\n")
