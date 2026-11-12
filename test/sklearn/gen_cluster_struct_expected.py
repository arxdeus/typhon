# Generates expected values for test_cluster_struct.py (scikit-learn 1.9.1 venv).
import numpy as np
import warnings
warnings.simplefilter("ignore")
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration, ward_tree, linkage_tree

rng = np.random.RandomState(2)
X = np.vstack([rng.randn(12, 2) * 0.4, rng.randn(12, 2) * 0.4 + [3, 0], rng.randn(12, 2) * 0.4 + [0, 3]])
n = X.shape[0]
# chain + kNN-like dense connectivity, with two components (fixed internally)
D = ((X[:, None] - X[None]) ** 2).sum(-1)
C = np.zeros((n, n))
for i in range(n):
    for j in np.argsort(D[i], kind="stable")[1:4]:
        C[i, j] = 1
C[:, 30:] = 0
C[30:, :] = 0
for i in range(30, n - 1):
    C[i, i + 1] = 1

def L(v):
    return [float(x) for x in np.asarray(v, dtype=float).ravel()]

out = []
ch, ncc, nl, par, d = ward_tree(X, connectivity=C, return_distance=True)
out.append(("ward_children", L(ch)))
out.append(("ward_dist", L(d)))
out.append(("ward_ncc", [float(ncc)]))
for lk in ["complete", "average", "single"]:
    ch, ncc, nl, par, d = linkage_tree(X, connectivity=C, linkage=lk, return_distance=True)
    out.append((f"{lk}_children", L(ch)))
    out.append((f"{lk}_dist", L(d)))
for lk in ["ward", "complete", "average", "single"]:
    a = AgglomerativeClustering(n_clusters=3, connectivity=C, linkage=lk).fit(X)
    out.append((f"agg_{lk}", L(a.labels_)))
a = AgglomerativeClustering(n_clusters=4, connectivity=C, linkage="ward", compute_full_tree=False).fit(X)
out.append(("agg_partial", L(a.labels_)))
a = AgglomerativeClustering(n_clusters=None, distance_threshold=2.0, connectivity=C, linkage="average").fit(X)
out.append(("agg_thresh", L(a.labels_) + [float(a.n_clusters_)]))
F = rng.randn(10, 8)
F[:, 4:] += F[:, :4] * 2
fa = FeatureAgglomeration(n_clusters=3).fit(F)
out.append(("fa_labels", L(fa.labels_)))
out.append(("fa_transform", L(fa.transform(F)[:3])))
out.append(("fa_inverse", L(fa.inverse_transform(fa.transform(F))[:2])))
for name, vals in out:
    print(f"    ({name!r}, {vals!r}),")
