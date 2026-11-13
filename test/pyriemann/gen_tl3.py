# Generates test_tl3.py (MDWM, TLClassifier) from real pyriemann 0.12.
import numpy as np, warnings, os
warnings.simplefilter("ignore")
from pyriemann.geometry import covariance as C
from pyriemann.transfer import encode_domains, TLClassifier, MDWM
from pyriemann.classification import MDM
D = os.path.dirname(__file__)
src = open(os.path.join(D, "test_tl2.py")).read()
head = src.split("X = signals(18, 4, 60)")[0]
head = head.replace("from pyriemann.transfer import TLRotate\n", "from pyriemann.transfer import TLClassifier, MDWM\n")
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
X = signals(18, 4, 60)
cov = C.covariances(X, "scm")
y = np.array([i % 3 for i in range(18)])
dom = ["src" if i < 12 else "tgt" for i in range(18)]
Xe, ye = encode_domains(cov, y, dom)
L = []; T = []
def add(name, arr, call, atol=1e-8):
    L.append(f"E_{name} = {lit(arr)}")
    T.append(f"    assert close({call}, E_{name}, atol={atol})")
for tr in (0.0, 0.4, 1.0):
    m = MDWM(tr, "tgt").fit(Xe, ye)
    n = str(tr).replace(".", "")
    add(f"mdwm_means_{n}", m.covmeans_, f"MDWM({tr}, \"tgt\").fit(Xe, ye).covmeans_", 1e-7)
    add(f"mdwm_pred_{n}", m.predict(Xe), f"MDWM({tr}, \"tgt\").fit(Xe, ye).predict(Xe)")
m = MDWM(0.5, "tgt", metric="logeuclid").fit(Xe, ye)
add("mdwm_le", m.covmeans_, 'MDWM(0.5, "tgt", "logeuclid").fit(Xe, ye).covmeans_', 1e-7)
L.append(f"E_mdwm_score = {m.score(Xe, ye)!r}")
T.append('    assert abs(MDWM(0.5, "tgt", "logeuclid").fit(Xe, ye).score(Xe, ye) - E_mdwm_score) < 1e-12')
c = TLClassifier("tgt", MDM()).fit(Xe, ye)
add("tlc_pred", c.predict(Xe), 'TLClassifier("tgt", MDM()).fit(Xe, ye).predict(Xe)')
add("tlc_proba", c.predict_proba(Xe), 'TLClassifier("tgt", MDM()).fit(Xe, ye).predict_proba(Xe)', 1e-8)
L.append(f"E_tlc_score = {c.score(Xe, ye)!r}")
T.append('    assert abs(TLClassifier("tgt", MDM()).fit(Xe, ye).score(Xe, ye) - E_tlc_score) < 1e-12')
dw = {"src": 0.3, "tgt": 1.0}
c = TLClassifier("tgt", MDM(), domain_weight=dw).fit(Xe, ye)
add("tlc_w", c.estimator.covmeans_, 'TLClassifier("tgt", MDM(), {"src": 0.3, "tgt": 1.0}).fit(Xe, ye).estimator.covmeans_', 1e-7)
setup = '''X = signals(18, 4, 60)
cov = covariances(X, "scm")
y = np.array([i % 3 for i in range(18)])
dom = ["src" if i < 12 else "tgt" for i in range(18)]
Xe, ye = encode_domains(cov, y, dom)

'''
open(os.path.join(D, "test_tl3.py"), "w").write(head + setup + "\n".join(L) + "\n\n@test\ndef test_tl3():\n" + "\n".join(T) + "\ntest_tl3()\n")
