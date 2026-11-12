# Generates expected values for test_birch.py (scikit-learn 1.9.1 venv).
import numpy as np
import warnings
warnings.simplefilter("ignore")
from sklearn.cluster import Birch

rng = np.random.RandomState(4)
X = np.vstack([rng.randn(40, 2) * 0.3 + [0, 0], rng.randn(40, 2) * 0.3 + [3, 1], rng.randn(40, 2) * 0.3 + [1, 4]])
rng.shuffle(X)

def L(v):
    return [float(x) for x in np.asarray(v, dtype=float).ravel()]

out = []
b = Birch(threshold=0.4, branching_factor=5, n_clusters=3).fit(X)
out.append(("b_centers", L(b.subcluster_centers_)))
out.append(("b_sublabels", L(b.subcluster_labels_)))
out.append(("b_labels", L(b.labels_)))
out.append(("b_transform", L(b.transform(X[:3]))))
b2 = Birch(threshold=0.8, n_clusters=None).fit(X)
out.append(("b2_centers", L(b2.subcluster_centers_)))
out.append(("b2_predict", L(b2.predict(X[:10]))))
b3 = Birch(threshold=0.3, branching_factor=3, n_clusters=3)
b3.partial_fit(X[:60]); b3.partial_fit(X[60:])
out.append(("b3_centers", L(b3.subcluster_centers_)))
out.append(("b3_labels", L(b3.labels_)))
for name, vals in out:
    print(f"    ({name!r}, {vals!r}),")
