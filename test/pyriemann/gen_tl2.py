# Generates test_tl2.py (Covariances mcd, TLRotate) from real pyriemann 0.12.
import numpy as np, warnings, os
warnings.simplefilter("ignore")
from pyriemann.geometry import covariance as C
from pyriemann.estimation import Covariances
from pyriemann.transfer import TLRotate, encode_domains
D = os.path.dirname(__file__)
src = open(os.path.join(D, "test_misc.py")).read()
head = src.split("X = signals(18, 5, 40)")[0]
head = head.replace("from pyriemann.clustering import Kmeans\n", "from pyriemann.estimation import Covariances\nfrom pyriemann.transfer import TLRotate\n")
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
X = signals(18, 4, 60)
cov = C.covariances(X, "scm")
y = np.array([i % 3 for i in range(18)])
dom = ["src" if i < 9 else "tgt" for i in range(18)]
Xe, ye = encode_domains(cov.copy(), y, dom)
# make source a rotated version of itself to give TLRotate something to find
th = 0.7
R0 = np.eye(4); R0[:2, :2] = [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
cs = cov.copy(); cs[:9] = R0 @ cs[:9] @ R0.T
Xe2, ye2 = encode_domains(cs, y, dom)
L = []; T = []
def add(name, arr, call, atol=1e-8):
    L.append(f"E_{name} = {lit(arr)}")
    T.append(f"    assert close({call}, E_{name}, atol={atol})")
add("mcd", Covariances("mcd", random_state=5).fit_transform(X[:3]), 'covariances(X[:3], "mcd", 5)', 1e-7)
add("rot_e", TLRotate("tgt").fit_transform(Xe.copy(), ye), 'TLRotate("tgt").fit_transform(Xe, ye)', 1e-6)
add("rot_r", TLRotate("tgt", metric="riemann").fit_transform(Xe2.copy(), ye2), 'TLRotate("tgt", None, "riemann").fit_transform(Xe2, ye2)', 1e-6)
setup = '''X = signals(18, 4, 60)
cov = covariances(X, "scm")
y = np.array([i % 3 for i in range(18)])
dom = ["src" if i < 9 else "tgt" for i in range(18)]
Xe, ye = encode_domains(cov, y, dom)
th = 0.7
R0 = np.eye(4)
R0[0, 0] = np.cos(th)
R0[0, 1] = -np.sin(th)
R0[1, 0] = np.sin(th)
R0[1, 1] = np.cos(th)
cs = cov.copy()
for i in range(9):
    cs[i] = R0 @ cov[i] @ R0.T
Xe2, ye2 = encode_domains(cs, y, dom)

'''
open(os.path.join(D, "test_tl2.py"), "w").write(head + setup + "\n".join(L) + "\n\n@test\ndef test_tl2():\n" + "\n".join(T) + "\ntest_tl2()\n")
