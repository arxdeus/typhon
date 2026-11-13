# Generates test_nch.py (NearestConvexHull) from real pyriemann 0.12.
import numpy as np, warnings, os
warnings.simplefilter("ignore")
from pyriemann.geometry import covariance as C
from pyriemann.classification import NearestConvexHull
D = os.path.dirname(__file__)
src = open(os.path.join(D, "test_embed.py")).read()
head = src.split("X = signals(16, 4, 60)")[0]
head = head.replace("from pyriemann.embedding import SpectralEmbedding, LocallyLinearEmbedding\n", "from pyriemann.classification import NearestConvexHull\n")
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
y = np.array([i % 2 for i in range(12)])
L = []; T = []
def add(name, arr, call, atol=1e-6):
    L.append(f"E_{name} = {lit(arr)}")
    T.append(f"    assert close({call}, E_{name}, atol={atol})")
for met in ("logeuclid", "euclid"):
    m = NearestConvexHull(met).fit(cov[:8], y[:8])
    add(f"d_{met}", m.transform(cov[8:]), f'NearestConvexHull("{met}").fit(cov[:8], y[:8]).transform(cov[8:])')
    add(f"p_{met}", m.predict(cov[8:]), f'NearestConvexHull("{met}").fit(cov[:8], y[:8]).predict(cov[8:])')
    add(f"pp_{met}", m.predict_proba(cov[8:]), f'NearestConvexHull("{met}").fit(cov[:8], y[:8]).predict_proba(cov[8:])')
setup = '''X = signals(12, 4, 60)
cov = covariances(X, "scm")
y = np.array([i % 2 for i in range(12)])

'''
open(os.path.join(D, "test_nch.py"), "w").write(head + setup + "\n".join(L) + "\n\n@test\ndef test_nch():\n" + "\n".join(T) + "\ntest_nch()\n")
