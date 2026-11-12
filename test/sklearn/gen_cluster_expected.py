# Expected literals for test/sklearn/test_cluster.py (scikit-learn 1.9.1).
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus, DBSCAN, AgglomerativeClustering
from sklearn import metrics as M
from sklearn.metrics import cluster as MC

def blobs(n, p, k):
    X = np.zeros((n, p)); y = np.zeros(n, dtype=int)
    for i in range(n):
        y[i] = i % k
        for j in range(p):
            v = (i * 7919 + j * 104729 + i * j * 13 + j * j * 3) % 101
            X[i, j] = (v - 50.0) / 30.0 + 1.2 * ((y[i] * (j + 2)) % 5)
    return X, y

def L(name, a):
    a = np.asarray(a).ravel()
    if a.dtype.kind in "iub":
        print(f"{name} = {[int(v) for v in a]}")
    else:
        print(f"{name} = {[float(v) for v in a]}")

def S(name, v):
    print(f"{name} = {float(v)!r}")

X, y = blobs(60, 3, 4)
c, idx = kmeans_plusplus(X, 4, random_state=0); L("pp_idx", idx)
for tag, kw in [("a", dict(n_clusters=4, random_state=0)),
                ("b", dict(n_clusters=3, init="random", n_init=5, random_state=42)),
                ("c", dict(n_clusters=5, random_state=1, algorithm="elkan", max_iter=3)),
                ("d", dict(n_clusters=4, n_init=3, random_state=7, tol=0.0))]:
    km = KMeans(**kw).fit(X)
    L(f"km_{tag}_centers", km.cluster_centers_); L(f"km_{tag}_labels", km.labels_)
    S(f"km_{tag}_inertia", km.inertia_); print(f"km_{tag}_iter = {km.n_iter_}")
km = KMeans(n_clusters=4, random_state=0).fit(X)
L("km_pred", km.predict(X[:7] + 0.3)); L("km_tr", km.transform(X[:5])); S("km_score", km.score(X[:10]))
init = X[[0, 1, 2, 3]]
km = KMeans(n_clusters=4, init=init, n_init=1).fit(X); L("km_arr_centers", km.cluster_centers_); print(f"km_arr_iter = {km.n_iter_}")
sw = np.array([1.0 + (i % 3) for i in range(60)])
km = KMeans(n_clusters=3, random_state=3).fit(X, sample_weight=sw); L("km_sw_centers", km.cluster_centers_); S("km_sw_inertia", km.inertia_)

for tag, kw in [("e", dict(eps=1.3, min_samples=4)), ("m", dict(eps=2.2, min_samples=4, metric="manhattan")),
                ("c", dict(eps=1.0, min_samples=4, metric="chebyshev")), ("k", dict(eps=1.2, min_samples=4, metric="minkowski", p=3))]:
    d = DBSCAN(**kw).fit(X); L(f"db_{tag}_labels", d.labels_); L(f"db_{tag}_core", d.core_sample_indices_)

X2, _ = blobs(40, 2, 3)
for lk in ["ward", "complete", "average", "single"]:
    a = AgglomerativeClustering(n_clusters=3, linkage=lk, compute_distances=True).fit(X2)
    L(f"ag_{lk}_labels", a.labels_); L(f"ag_{lk}_children", a.children_); L(f"ag_{lk}_dist", a.distances_)
a = AgglomerativeClustering(n_clusters=None, distance_threshold=2.0, linkage="average", metric="manhattan").fit(X2)
L("ag_thr_labels", a.labels_); print(f"ag_thr_n = {a.n_clusters_}")

lt = [i % 4 for i in range(60)]
lp = list(KMeans(n_clusters=5, random_state=0).fit(X).labels_)
lp2 = [(i * 7) % 3 for i in range(60)]
for tag, b in [("1", lp), ("2", lp2)]:
    S(f"ari{tag}", M.adjusted_rand_score(lt, b)); S(f"ri{tag}", M.rand_score(lt, b))
    S(f"mi{tag}", M.mutual_info_score(lt, b)); S(f"ami{tag}", M.adjusted_mutual_info_score(lt, b))
    S(f"nmi{tag}", M.normalized_mutual_info_score(lt, b)); S(f"nmig{tag}", M.normalized_mutual_info_score(lt, b, average_method="geometric"))
    L(f"hcv{tag}", M.homogeneity_completeness_v_measure(lt, b)); S(f"fm{tag}", M.fowlkes_mallows_score(lt, b))
    L(f"cm{tag}", MC.contingency_matrix(lt, b))
    S(f"sil{tag}", M.silhouette_score(X, b)); S(f"silm{tag}", M.silhouette_score(X, b, metric="manhattan"))
    S(f"ch{tag}", M.calinski_harabasz_score(X, b)); S(f"db{tag}", M.davies_bouldin_score(X, b))
L("sils", M.silhouette_samples(X, lp)[:12])
