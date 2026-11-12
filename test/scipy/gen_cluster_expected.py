# Expected values for test/scipy/test_cluster.py (scipy 1.18.1).
import warnings
import numpy as np
from scipy.cluster import hierarchy as H
from scipy.cluster import vq as V
warnings.simplefilter('ignore')
rng = np.random.default_rng(11)
X = np.round(rng.normal(size=(12, 3)), 2)
X[5] = X[2]                          # exact tie
Xt = np.round(rng.random((9, 2)) * 4)  # many ties
def L(a): return repr(np.asarray(a).tolist())
print('X =', L(X)); print('Xt =', L(Xt))
for m in ['single','complete','average','weighted','centroid','median','ward']:
    print(f'Z_{m} =', L(H.linkage(X, m)))
    print(f'Zt_{m} =', L(H.linkage(Xt, m)))
print('Z_cb =', L(H.linkage(X, 'average', metric='cityblock')))
Z = H.linkage(X, 'average')
print('FC_max3 =', L(H.fcluster(Z, 3, 'maxclust')))
print('FC_dist =', L(H.fcluster(Z, 1.5, 'distance')))
print('FC_inc =', L(H.fcluster(Z, 0.8)))
print('FC_inc3 =', L(H.fcluster(Z, 1.0, depth=3)))
print('FCD =', L(H.fclusterdata(X, 1.2, criterion='distance', method='complete')))
print('INC =', L(H.inconsistent(Z)))
print('INC3 =', L(H.inconsistent(Z, 3)))
c, coph = H.cophenet(Z, __import__('scipy.spatial.distance').spatial.distance.pdist(X))
print('COPH =', L(coph)); print('COPHC =', repr(float(c)))
print('LL =', L(H.leaves_list(Z)))
print('MD =', L(H.maxdists(Z)))
print('CUT =', L(H.cut_tree(Z, n_clusters=[2, 4])))
print('CUTH =', L(H.cut_tree(Z, height=[1.0])))
print('CUTALL =', L(H.cut_tree(Z)))
d = H.dendrogram(Z, no_plot=True)
for k in ['ivl','leaves','icoord','dcoord','color_list','leaves_color_list']: print(f'D_{k} =', repr(d[k]))
d = H.dendrogram(Z, no_plot=True, truncate_mode='lastp', p=4, count_sort=True)
for k in ['ivl','leaves','icoord','dcoord']: print(f'DT_{k} =', repr(d[k]))
d = H.dendrogram(Z, no_plot=True, truncate_mode='level', p=2, distance_sort='descending', labels=[f'p{i}' for i in range(12)])
for k in ['ivl','leaves','icoord','dcoord']: print(f'DL_{k} =', repr(d[k]))
T = H.fcluster(Z, 3, 'maxclust'); l, m = H.leaders(Z, T)
print('LEAD =', L(l), '; LEADM =', L(m))
r = H.to_tree(Z); print('TREE =', repr((r.id, r.count, float(r.dist), r.pre_order())))
# vq
W = V.whiten(X); print('WH =', L(W))
book = X[[0, 4, 8]]
c, dd = V.vq(X, book); print('VQC =', L(c)); print('VQD =', L(dd))
cb, dist = V.kmeans(X, book.copy()); print('KMG =', L(cb)); print('KMGD =', repr(float(dist)))
cb, dist = V.kmeans(X, 3, rng=5); print('KMR =', L(cb)); print('KMRD =', repr(float(dist)))
cb, lab = V.kmeans2(X, book.copy(), minit='matrix'); print('K2M =', L(cb)); print('K2ML =', L(lab))
for mi in ['random', 'points', '++']:
    cb, lab = V.kmeans2(X, 3, minit=mi, rng=3); print(f'K2_{mi[0] if mi!="++" else "pp"} =', L(cb)); print(f'K2L_{mi[0] if mi!="++" else "pp"} =', L(lab))
y1 = X[:, 0]
cb, lab = V.kmeans2(y1, 2, minit='random', rng=4); print('K21 =', L(cb)); print('K21L =', L(lab))
