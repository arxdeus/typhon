# Generates test_embed.py (SpectralEmbedding, LocallyLinearEmbedding) from real pyriemann 0.12.
# Eigenvector signs are arbitrary, so absolute values are compared.
import numpy as np, warnings, os
warnings.simplefilter("ignore")
from pyriemann.geometry import covariance as C
from pyriemann.embedding import SpectralEmbedding, LocallyLinearEmbedding
D = os.path.dirname(__file__)
src = open(os.path.join(D, "test_tl3.py")).read()
head = src.split("X = signals(18, 4, 60)")[0]
head = head.replace("from pyriemann.transfer import TLClassifier, MDWM\n", "from pyriemann.embedding import SpectralEmbedding, LocallyLinearEmbedding\n")
def signals(n_trials, n_ch, n_t):
    X = np.zeros((n_trials, n_ch, n_t))
    for i in range(n_trials):
        for c in range(n_ch):
            for t in range(n_t):
                v = (i * 1009 + c * 7919 + t * 104729 + i * c * t * 13
                     + t * t * (c + 1) * 3 + c * c * t * 7) % 101
                X[i, c, t] = (v - 50.0) / 25.0 + ((0.5 * (i % 2)) if (c + t) % 3 == 0 else 0.0)
    return X
def lit(a): return repr(np.abs(np.asarray(a, dtype=float)).ravel().tolist())
X = signals(16, 4, 60)
cov = C.covariances(X, "scm")
L = []; T = []
def add(name, arr, call, atol=1e-6):
    L.append(f"E_{name} = {lit(arr)}")
    T.append(f"    assert close(np.abs({call}), E_{name}, atol={atol})")
add("se", SpectralEmbedding(2).fit_transform(cov), "SpectralEmbedding(2).fit_transform(cov)")
add("se_le", SpectralEmbedding(3, "logeuclid", 0.5).fit_transform(cov), 'SpectralEmbedding(3, "logeuclid", 0.5).fit_transform(cov)')
l = LocallyLinearEmbedding(2, 4).fit(cov)
add("lle", l.embedding_, "LocallyLinearEmbedding(2, 4).fit_transform(cov)")
L.append(f"E_lle_err = {l.error_!r}")
T.append("    assert abs(LocallyLinearEmbedding(2, 4).fit(cov).error_ - E_lle_err) < 1e-8")
add("lle_tr", l.transform(cov[:5]), "LocallyLinearEmbedding(2, 4).fit(cov).transform(cov[:5])")
setup = '''X = signals(16, 4, 60)
cov = covariances(X, "scm")

'''
open(os.path.join(D, "test_embed.py"), "w").write(head + setup + "\n".join(L) + "\n\n@test\ndef test_embed():\n" + "\n".join(T) + "\ntest_embed()\n")
