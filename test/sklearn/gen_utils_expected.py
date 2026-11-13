# Expected literals for test/sklearn/test_utils.py (scikit-learn 1.9.1).
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn import utils as U
from sklearn.utils import extmath as E, class_weight as CW, multiclass as MC
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.base import clone, is_classifier, is_regressor

def L(name, a):
    a = np.asarray(a, dtype=float).ravel()
    print(f"{name} = {[float(v) for v in a]}")

y = np.array([0, 1, 1, 2, 2, 2, 0, 1, 2, 2])
L("cw_bal", CW.compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y))
L("sw_bal", CW.compute_sample_weight("balanced", y))
print("tot_bin =", repr(MC.type_of_target(np.array([0, 1, 1]))))
print("tot_mc =", repr(MC.type_of_target(y)))
print("tot_cont =", repr(MC.type_of_target(np.array([0.5, 1.5]))))
print("tot_ml =", repr(MC.type_of_target(np.array([[0, 1], [1, 1]]))))
L("ul", MC.unique_labels(np.array([3, 1]), np.array([2, 1, 5])))
X = np.array([[((i * 7 + j * 13) % 11) / 3.0 - 1.0 for j in range(3)] for i in range(6)])
L("rn", E.row_norms(X)); L("rn2", E.row_norms(X, squared=True))
L("smax", E.softmax(X)); L("cart", E.cartesian([np.array([1, 2]), np.array([3, 4, 5])]))
L("scs", E.stable_cumsum(np.arange(1.0, 6.0)))
m, s = E.weighted_mode(np.array([1, 2, 2, 3, 3]), np.array([1.0, 1.0, 0.5, 2.0, 0.1])); L("wm", m); L("wmw", s)
L("bat", [b.start for b in U.gen_batches(10, 3)])
L("evs", [b.stop for b in U.gen_even_slices(10, 3)])
L("rs", U.resample(np.arange(10), random_state=0)); L("sh", U.shuffle(np.arange(8), random_state=1))
L("rs_s", U.resample(np.arange(10), y, n_samples=6, random_state=3, stratify=y)[0])
print("mm =", U.murmurhash3_32("hello", seed=3), U.murmurhash3_32(42, positive=True))
print("rep_ridge =", repr(repr(Ridge(alpha=0.5))))
print("is_c =", is_classifier(LogisticRegression()), is_regressor(Ridge()), is_classifier(Ridge()))
