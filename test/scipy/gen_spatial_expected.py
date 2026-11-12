# Expected values for test/scipy/test_spatial.py (scipy 1.18.1).
import warnings
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial import distance as D
warnings.simplefilter('ignore')
rng = np.random.default_rng(7)
X = np.round(rng.normal(size=(7, 3)), 3)
Y = np.round(rng.normal(size=(4, 3)), 3)
B = (rng.random((5, 6)) > 0.5).astype(float)
Q = np.round(rng.normal(size=(5, 3)), 3)
P = np.round(rng.random((40, 2)) * 10, 3)
def L(a): return repr(np.asarray(a).tolist())
print('X =', L(X)); print('Y =', L(Y)); print('B =', L(B)); print('Q =', L(Q)); print('P =', L(P))
for m in ['euclidean','sqeuclidean','cityblock','chebyshev','cosine','correlation','canberra','braycurtis','seuclidean','mahalanobis']:
    print(f"PD_{m} =", L(D.pdist(X, m))); print(f"CD_{m} =", L(D.cdist(X, Y, m)))
print('PD_mink3 =', L(D.pdist(X, 'minkowski', p=3.0)))
print('PD_seV =', L(D.pdist(X, 'seuclidean', V=[0.5, 1.0, 2.0])))
for m in ['hamming','jaccard','dice']:
    print(f"PB_{m} =", L(D.pdist(B, m)))
print('SINGLE =', L([D.euclidean(X[0],X[1]), D.sqeuclidean(X[0],X[1]), D.cityblock(X[0],X[1]), D.chebyshev(X[0],X[1]), D.minkowski(X[0],X[1],1.5), D.cosine(X[0],X[1]), D.correlation(X[0],X[1]), D.canberra(X[0],X[1]), D.braycurtis(X[0],X[1]), D.seuclidean(X[0],X[1],[1.,2.,3.]), D.mahalanobis(X[0],X[1],np.eye(3)*2), D.hamming(B[0],B[1]), D.jaccard(B[0],B[1]), D.dice(B[0],B[1])]))
t = KDTree(P, leafsize=4)
for k in (1, 3):
    d, i = t.query(Q[:, :2] * 3 + 5, k=k); print(f'KQ{k}_d =', L(d)); print(f'KQ{k}_i =', L(i))
d, i = t.query(Q[:, :2] * 3 + 5, k=4, p=1.0, distance_upper_bound=1.2); print('KQub_d =', L(d)); print('KQub_i =', L(i))
d, i = t.query(Q[:, :2] * 3 + 5, k=2, p=np.inf); print('KQinf_d =', L(d)); print('KQinf_i =', L(i))
print('KB =', repr([sorted(x) for x in t.query_ball_point(Q[:, :2] * 3 + 5, 1.5)]))
print('KP =', repr(sorted(t.query_pairs(0.8))))
print('KC =', repr(t.count_neighbors(KDTree(Q[:, :2]*3+5), 2.0)))
from scipy.spatial import distance_matrix
print('DM =', L(distance_matrix(X, Y, p=1.5)))
