# Generates expected literals for test/sklearn/test_final_metrics2.py (scikit-learn 1.9.1).
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import consensus_score, metric_at_thresholds, accuracy_score, f1_score

def L(name, a):
    print(f"{name} = {[float(v) for v in np.asarray(a, dtype=float).ravel()]!r}")

a = (np.array([[1, 1, 0, 0, 1], [0, 0, 1, 1, 0], [1, 0, 1, 0, 0]], dtype=float), np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]], dtype=float))
b = (np.array([[0, 0, 1, 1, 0], [1, 1, 0, 0, 0]], dtype=float), np.array([[0, 1, 0, 1], [1, 0, 1, 1]], dtype=float))
print("cs =", float(consensus_score(a, b)))
yt = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
ys = np.array([0.1, 0.8, 0.35, 0.4, 0.8, 0.2, 0.9, 0.5, 0.35, 0.05])
sw = np.array([1.0, 2, 1, 0, 1, 3, 1, 1, 2, 1])
v, t = metric_at_thresholds(yt, ys, accuracy_score); L("acc_v", v); L("acc_t", t)
v, t = metric_at_thresholds(yt, ys, f1_score, sample_weight=sw); L("f1w_v", v); L("f1w_t", t)
