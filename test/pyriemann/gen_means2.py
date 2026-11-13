# Generates test_means2.py (ale/alm/bmp/cheap/thompson/logchol means, logchol dist/geodesic) from pyriemann 0.12.
import numpy as np, warnings, os
warnings.simplefilter("ignore")
from pyriemann.geometry import mean as M, distance as Di, geodesic as G
D = os.path.dirname(__file__)
def spd(n_mats, n):
    out = np.zeros((n_mats, n, n))
    for k in range(n_mats):
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                A[i, j] = ((i + 1) * (j + 2) * (k + 3) % 7 - 3.0) / 4.0
        out[k] = A @ A.T + (k + 1.0) * np.eye(n)
    return out
def lit(a): return repr(np.asarray(a, dtype=float).ravel().tolist())
head = open(os.path.join(D, "test_pipeline.py")).read().split("P = spd(5, 4)")[0]
head = head.replace("import numpy as np\n", "import numpy as np\nfrom pyriemann.geometry.mean import gmean, mean_ale, mean_alm, mean_bmp, mean_cheap, mean_thompson, mean_logchol\nfrom pyriemann.geometry.distance import distance, pairwise_distance\nfrom pyriemann.geometry.geodesic import geodesic\n", 1)
P = spd(5, 4); w = np.array([1., 2, 3, 4, 5])
L = []; T = []
def add(name, arr, call, atol=1e-8):
    L.append(f"E_{name} = {lit(arr)}")
    T.append(f"    assert close({call}, E_{name}, atol={atol})")
for nm in ("ale", "alm", "bmp", "cheap", "thompson", "logchol"):
    add("m_"+nm, M.gmean(P[:4] if nm in ("alm","bmp","cheap") else P, metric=nm), f'gmean({"P[:4]" if nm in ("alm","bmp","cheap") else "P"}, "{nm}")', 1e-7)
add("m_logchol_w", M.gmean(P, metric="logchol", sample_weight=w), 'gmean(P, "logchol", w)')
add("m_ale_w", M.gmean(P, metric="ale", sample_weight=w), 'gmean(P, "ale", w)', 1e-7)
add("d_logchol", Di.distance(P[0], P[1], metric="logchol"), 'distance(P[0], P[1], "logchol")')
add("d_logchol_sq", Di.distance(P[:3], P[1], metric="logchol", squared=True), 'distance(P[:3], P[1], "logchol", True)')
add("pd_logchol", Di.pairwise_distance(P, metric="logchol"), 'pairwise_distance(P, None, "logchol")')
add("g_logchol", G.geodesic(P[0], P[1], 0.3, metric="logchol"), 'geodesic(P[0], P[1], 0.3, "logchol")')
setup = "P = spd(5, 4)\nw = np.array([1.0, 2.0, 3.0, 4.0, 5.0])\n"
open(os.path.join(D, "test_means2.py"), "w").write(head + setup + "\n".join(L) + "\n\n@test\ndef test_means2():\n" + "\n".join(T) + "\ntest_means2()\n")
