# Generates expected values for test_cluster_more.py (scikit-learn 1.9.1 venv).
import numpy as np
import warnings
warnings.simplefilter("ignore")
from sklearn.cluster import (MiniBatchKMeans, MeanShift, estimate_bandwidth, AffinityPropagation,
                             SpectralClustering)

rng = np.random.RandomState(0)
X = np.vstack([rng.randn(30, 2) * 0.4 + [0, 0], rng.randn(30, 2) * 0.4 + [4, 4],
               rng.randn(30, 2) * 0.4 + [0, 5]])

def L(v):
    return [float(x) for x in np.asarray(v, dtype=float).ravel()]

out = []
m = MiniBatchKMeans(n_clusters=3, random_state=0, batch_size=20).fit(X)
out.append(("mbk_centers", L(m.cluster_centers_)))
out.append(("mbk_labels", L(m.labels_)))
out.append(("mbk_meta", [m.inertia_, float(m.n_iter_), float(m.n_steps_)]))
m2 = MiniBatchKMeans(n_clusters=4, init="random", random_state=3, batch_size=16, n_init=2).fit(X)
out.append(("mbk2_centers", L(m2.cluster_centers_)))
out.append(("mbk2_meta", [m2.inertia_, float(m2.n_iter_), float(m2.n_steps_)]))

bw = estimate_bandwidth(X, quantile=0.2)
out.append(("bandwidth", [float(bw)]))
ms = MeanShift(bandwidth=bw).fit(X)
out.append(("ms_centers", L(ms.cluster_centers_)))
out.append(("ms_labels", L(ms.labels_)))
ms2 = MeanShift(bandwidth=1.5, bin_seeding=True).fit(X)
out.append(("ms2_centers", L(ms2.cluster_centers_)))
out.append(("ms2_predict", L(ms2.predict(np.array([[0.1, 0.2], [3.9, 4.2], [0.3, 4.6]])))))

ap = AffinityPropagation(random_state=0).fit(X)
out.append(("ap_centers_idx", L(ap.cluster_centers_indices_)))
out.append(("ap_labels", L(ap.labels_)))
out.append(("ap_niter", [float(ap.n_iter_)]))
ap2 = AffinityPropagation(random_state=1, preference=-50.0, damping=0.7).fit(X)
out.append(("ap2_centers_idx", L(ap2.cluster_centers_indices_)))
out.append(("ap2_niter", [float(ap2.n_iter_)]))

for assign in ["kmeans", "discretize", "cluster_qr"]:
    sc = SpectralClustering(n_clusters=3, random_state=0, gamma=0.5, assign_labels=assign).fit(X)
    out.append((f"sc_{assign}", L(sc.labels_)))
sc = SpectralClustering(n_clusters=3, random_state=0, affinity="nearest_neighbors", n_neighbors=8,
                        assign_labels="discretize").fit(X)
out.append(("sc_nn", L(sc.labels_)))

for name, vals in out:
    print(f"    ({name!r}, {vals!r}),")
