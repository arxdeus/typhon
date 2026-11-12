# Generates expected values for test_cluster_density.py (scikit-learn 1.9.1 venv).
import numpy as np
import warnings
warnings.simplefilter("ignore")
from sklearn.cluster import OPTICS, HDBSCAN, cluster_optics_dbscan

rng = np.random.RandomState(0)
X = np.vstack([rng.randn(25, 2) * 0.3 + [0, 0], rng.randn(25, 2) * 0.5 + [3, 3],
               rng.randn(25, 2) * 0.2 + [0, 4], rng.uniform(-2, 5, (8, 2))])

def L(v):
    return [float(x) for x in np.asarray(v, dtype=float).ravel()]

out = []
o = OPTICS(min_samples=5).fit(X)
out.append(("op_order", L(o.ordering_)))
out.append(("op_core", L(o.core_distances_)))
out.append(("op_reach", L(np.where(np.isinf(o.reachability_), -1, o.reachability_))))
out.append(("op_pred", L(o.predecessor_)))
out.append(("op_labels", L(o.labels_)))
out.append(("op_hier", L(o.cluster_hierarchy_)))
o2 = OPTICS(min_samples=4, xi=0.1, min_cluster_size=0.1, metric="manhattan").fit(X)
out.append(("op2_labels", L(o2.labels_)))
o3 = OPTICS(min_samples=6, cluster_method="dbscan", eps=0.6).fit(X)
out.append(("op3_labels", L(o3.labels_)))
out.append(("op_dbscan_fn", L(cluster_optics_dbscan(reachability=o.reachability_, core_distances=o.core_distances_, ordering=o.ordering_, eps=0.4))))

h = HDBSCAN(min_cluster_size=5, copy=False).fit(X)
out.append(("hd_labels", L(h.labels_)))
out.append(("hd_probs", L(h.probabilities_)))
h2 = HDBSCAN(min_cluster_size=6, min_samples=3, cluster_selection_method="leaf", copy=False).fit(X)
out.append(("hd2_labels", L(h2.labels_)))
out.append(("hd2_probs", L(h2.probabilities_)))
h3 = HDBSCAN(min_cluster_size=5, cluster_selection_epsilon=1.5, store_centers="both", copy=False).fit(X)
out.append(("hd3_labels", L(h3.labels_)))
out.append(("hd3_centroids", L(h3.centroids_)))
out.append(("hd3_medoids", L(h3.medoids_)))
out.append(("hd_cut", L(h.dbscan_clustering(0.5, min_cluster_size=4))))
for name, vals in out:
    print(f"    ({name!r}, {vals!r}),")
