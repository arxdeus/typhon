# Generates expected values for test_cluster_bisect_bicluster.py (scikit-learn 1.9.1 venv).
import numpy as np
import warnings
warnings.simplefilter("ignore")
from sklearn.cluster import (BisectingKMeans, SpectralCoclustering, SpectralBiclustering, k_means,
                             affinity_propagation, mean_shift, spectral_clustering)

rng = np.random.RandomState(0)
X = np.vstack([rng.randn(30, 2) * 0.4 + [0, 0], rng.randn(30, 2) * 0.4 + [4, 4],
               rng.randn(30, 2) * 0.4 + [0, 5], rng.randn(20, 2) * 0.4 + [-4, 2]])
rb = np.random.RandomState(1)
B = rb.rand(24, 18) * 0.5
B[:8, :6] += 3.0
B[8:16, 6:12] += 5.0
B[16:, 12:] += 2.0

def L(v):
    return [float(x) for x in np.asarray(v, dtype=float).ravel()]

out = []
for name, kw in [("bi", dict(n_clusters=4, random_state=0)),
                 ("bl", dict(n_clusters=5, random_state=2, bisecting_strategy="largest_cluster")),
                 ("bk", dict(n_clusters=4, random_state=3, init="k-means++", n_init=3))]:
    m = BisectingKMeans(**kw).fit(X)
    out.append((name + "_centers", L(m.cluster_centers_)))
    out.append((name + "_labels", L(m.labels_)))
    out.append((name + "_inertia", [m.inertia_]))
    out.append((name + "_pred", L(m.predict(np.array([[0.1, 0.2], [3.9, 4.2], [0.3, 4.6], [-3.0, 2.0]])))))
c = SpectralCoclustering(n_clusters=3, random_state=0).fit(B)
out.append(("co_rows", L(c.row_labels_)))
out.append(("co_cols", L(c.column_labels_)))
for meth in ["bistochastic", "log", "scale"]:
    s = SpectralBiclustering(n_clusters=(3, 3), method=meth, random_state=0, n_components=4, n_best=2).fit(B)
    out.append(("bc_" + meth + "_rows", L(s.row_labels_)))
    out.append(("bc_" + meth + "_cols", L(s.column_labels_)))
ce, la, inr = k_means(X, 4, random_state=0, n_init=3)
out.append(("km_func", L(ce) + L(la) + [inr]))
S = -((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
idx, lab, nit = affinity_propagation(S, random_state=0, return_n_iter=True)
out.append(("ap_func", L(idx) + L(lab) + [float(nit)]))
mc, ml = mean_shift(X, bandwidth=1.2)
out.append(("ms_func", L(mc) + L(ml)))
A = np.exp(-0.5 * (-S))
sl = spectral_clustering(A, n_clusters=4, random_state=0, assign_labels="discretize")
out.append(("sc_func", L(sl)))
for name, vals in out:
    print(f"    ({name!r}, {vals!r}),")
