# Generates test_ajd.py from real pyriemann 0.12 (venv python).
import numpy as np, warnings
warnings.simplefilter("ignore")
from pyriemann.utils.ajd import ajd_pham, rjd, uwedge, ajd
def spd(n_mats, n):
    out = np.zeros((n_mats, n, n))
    for k in range(n_mats):
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                A[i, j] = ((i + 1) * (j + 2) * (k + 3) % 7 - 3.0) / 4.0
        out[k] = A @ A.T + (k + 1.0) * np.eye(n)
    return out
P = spd(5, 4)
def lit(a): return repr(np.asarray(a).ravel().tolist())
L = []
for name, f in (("pham", ajd_pham), ("rjd", rjd), ("uwedge", uwedge)):
    V, D = f(P.copy())
    L.append(f"E_{name}_V = {lit(V)}\nE_{name}_D = {lit(D)}")
V, D = ajd_pham(P.copy(), sample_weight=np.array([1., 2, 3, 4, 5]))
L.append(f"E_phamw_V = {lit(V)}")
V, D = ajd(P.copy(), method="rjd")
L.append(f"E_ajd_V = {lit(V)}")
head = open(__file__.replace("gen_ajd.py", "test_pipeline.py")).read().split("P = spd(5, 4)")[0]
body = '''P = spd(5, 4)
''' + "\n".join(L) + '''

@test
def test_ajd():
    for nm in ("pham", "rjd", "uwedge"):
        pass
    V, D = ajd_pham(P)
    assert close(V, E_pham_V, atol=1e-9) and close(D, E_pham_D, atol=1e-9)
    V, D = rjd(P)
    assert close(V, E_rjd_V, atol=1e-9) and close(D, E_rjd_D, atol=1e-9)
    V, D = uwedge(P)
    assert close(V, E_uwedge_V, atol=1e-8) and close(D, E_uwedge_D, atol=1e-8)
    V, D = ajd_pham(P, sample_weight=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert close(V, E_phamw_V, atol=1e-9)
    V, D = ajd(P, method="rjd")
    assert close(V, E_ajd_V, atol=1e-9)
test_ajd()
'''
head = head.replace("import numpy as np\n", "import numpy as np\nfrom pyriemann.geometry.ajd import ajd, ajd_pham, rjd, uwedge\n", 1)
open(__file__.replace("gen_ajd.py", "test_ajd.py"), "w").write(head + body)
