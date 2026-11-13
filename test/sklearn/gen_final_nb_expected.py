# Generates expected literals for test/sklearn/test_final_nb.py (scikit-learn 1.9.1).
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.naive_bayes import BernoulliNB, ComplementNB, CategoricalNB, MultinomialNB

def L(name, a):
    print(f"{name} = {[float(v) for v in np.asarray(a, dtype=float).ravel()]!r}")

n = 40
C = np.zeros((n, 5))
for i in range(n):
    for j in range(5):
        C[i, j] = (i * 7919 + j * 104729 + i * j * 13) % 6
y = np.array([(i * 5 + (1 if C[i, 0] > 2 else 0)) % 3 for i in range(n)])
sw = 0.5 + (np.arange(n) % 4)
Q = C[:8]
def dump(tag, m, cat=False):
    L(tag + "_cc", m.class_count_); L(tag + "_clp", m.class_log_prior_)
    if cat:
        for k, a in enumerate(m.category_count_): L(f"{tag}_cat{k}", a)
        for k, a in enumerate(m.feature_log_prob_): L(f"{tag}_flp{k}", a)
        L(tag + "_ncat", m.n_categories_)
    else:
        L(tag + "_fc", m.feature_count_); L(tag + "_flp", m.feature_log_prob_)
    if tag.startswith("cnb"): L(tag + "_fa", m.feature_all_)
    L(tag + "_pred", m.predict(Q)); L(tag + "_proba", m.predict_proba(Q)); L(tag + "_jll", m.predict_joint_log_proba(Q))
for tag, mk in [("bnb", lambda: BernoulliNB(alpha=0.7, binarize=2.0)), ("cnb", lambda: ComplementNB(alpha=0.9)),
                ("cnbn", lambda: ComplementNB(alpha=0.9, norm=True)), ("cat", lambda: CategoricalNB(alpha=0.8, min_categories=7)),
                ("mnb", lambda: MultinomialNB(alpha=0.5, fit_prior=False))]:
    m = mk(); m.partial_fit(C[:15], y[:15], classes=[0, 1, 2], sample_weight=sw[:15]) if tag != "cat" else m.partial_fit(C[:15], y[:15], classes=[0, 1, 2])
    m.partial_fit(C[15:], y[15:], sample_weight=sw[15:]) if tag != "cat" else m.partial_fit(C[15:], y[15:])
    dump(tag, m, tag == "cat")
