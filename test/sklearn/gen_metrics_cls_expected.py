# Expected literals for test/sklearn/test_metrics_cls.py (scikit-learn 1.9.1).
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn import metrics as M

def L(name, a):
    a = np.asarray(a, dtype=float).ravel()
    print(f"{name} = {[float(v) for v in a]}")

def S(name, v):
    print(f"{name} = {float(v)!r}")

yt = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1]); yp = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1])
y3t = np.array([0, 1, 2, 2, 1, 0, 2, 1, 3, 3, 0, 2]); y3p = np.array([0, 2, 2, 1, 1, 0, 2, 0, 3, 1, 0, 3])
sw = np.array([1.0 + (i % 3) * 0.5 for i in range(12)])
mlt = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0]])
mlp = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0]])
st = np.array(["cat", "dog", "cat", "bird", "dog", "bird", "cat"]); sp = np.array(["cat", "cat", "cat", "bird", "dog", "dog", "bird"])

S("acc", M.accuracy_score(yt, yp)); S("acc_n", M.accuracy_score(y3t, y3p, normalize=False))
S("acc_w", M.accuracy_score(y3t, y3p, sample_weight=sw)); S("acc_ml", M.accuracy_score(mlt, mlp))
S("zo", M.zero_one_loss(y3t, y3p)); S("zo_n", M.zero_one_loss(y3t, y3p, normalize=False))
S("bacc", M.balanced_accuracy_score(y3t, y3p)); S("bacc_adj", M.balanced_accuracy_score(y3t, y3p, adjusted=True))
S("bacc_w", M.balanced_accuracy_score(y3t, y3p, sample_weight=sw))
L("cm", M.confusion_matrix(y3t, y3p)); L("cm_w", M.confusion_matrix(y3t, y3p, sample_weight=sw))
L("cm_true", M.confusion_matrix(y3t, y3p, normalize="true")); L("cm_lab", M.confusion_matrix(y3t, y3p, labels=[3, 1, 0]))
L("cm_str", M.confusion_matrix(st, sp))
L("mcm", M.multilabel_confusion_matrix(y3t, y3p)); L("mcm_ml", M.multilabel_confusion_matrix(mlt, mlp))
L("mcm_sw", M.multilabel_confusion_matrix(mlt, mlp, samplewise=True))
S("f1b", M.f1_score(yt, yp)); S("pb", M.precision_score(yt, yp)); S("rb", M.recall_score(yt, yp, pos_label=0))
for avg in ["micro", "macro", "weighted"]:
    S(f"f1_{avg}", M.f1_score(y3t, y3p, average=avg)); S(f"p_{avg}", M.precision_score(y3t, y3p, average=avg))
    S(f"r_{avg}", M.recall_score(y3t, y3p, average=avg, sample_weight=sw)); S(f"fb_{avg}", M.fbeta_score(y3t, y3p, beta=0.5, average=avg))
    S(f"j_{avg}", M.jaccard_score(y3t, y3p, average=avg))
    S(f"ml_f1_{avg}", M.f1_score(mlt, mlp, average=avg))
S("ml_f1_samples", M.f1_score(mlt, mlp, average="samples")); S("ml_j_samples", M.jaccard_score(mlt, mlp, average="samples"))
p, r, f, s = M.precision_recall_fscore_support(y3t, y3p, beta=2.0); L("prfs_p", p); L("prfs_r", r); L("prfs_f", f); L("prfs_s", s)
L("f1_none_lab", M.f1_score(y3t, y3p, average=None, labels=[0, 3, 5]))
L("p_zd1", M.precision_score([0, 1, 1, 2], [0, 0, 0, 0], average=None, zero_division=1.0))
L("j_none", M.jaccard_score(y3t, y3p, average=None))
S("kappa", M.cohen_kappa_score(y3t, y3p)); S("kappa_lin", M.cohen_kappa_score(y3t, y3p, weights="linear"))
S("kappa_q", M.cohen_kappa_score(y3t, y3p, weights="quadratic", sample_weight=sw))
S("mcc", M.matthews_corrcoef(y3t, y3p)); S("mcc_b", M.matthews_corrcoef(yt, yp)); S("mcc_w", M.matthews_corrcoef(y3t, y3p, sample_weight=sw))
S("ham", M.hamming_loss(y3t, y3p)); S("ham_ml", M.hamming_loss(mlt, mlp))
P = np.array([[0.7, 0.2, 0.05, 0.05], [0.1, 0.3, 0.5, 0.1], [0.2, 0.2, 0.5, 0.1], [0.3, 0.4, 0.2, 0.1], [0.2, 0.6, 0.1, 0.1], [0.5, 0.3, 0.1, 0.1],
              [0.1, 0.1, 0.7, 0.1], [0.4, 0.4, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7], [0.2, 0.3, 0.1, 0.4], [0.6, 0.2, 0.1, 0.1], [0.0, 0.2, 0.3, 0.5]])
pb = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.6, 0.3, 0.7, 0.4, 1.0])
S("ll", M.log_loss(y3t, P)); S("ll_w", M.log_loss(y3t, P, sample_weight=sw, normalize=False)); S("ll_b", M.log_loss(yt, pb))
S("brier", M.brier_score_loss(yt, pb)); S("brier_m", M.brier_score_loss(y3t, P)); S("brier_pl", M.brier_score_loss(yt, pb, pos_label=0))
S("d2ll", M.d2_log_loss_score(y3t, P)); S("d2b", M.d2_brier_score(yt, pb))
D = P * 2 - 0.5
S("hinge_m", M.hinge_loss(y3t, D)); S("hinge_b", M.hinge_loss(yt, pb - 0.5))
lr = M.class_likelihood_ratios(yt, yp); S("lrp", lr[0]); S("lrn", lr[1])
print("rep1 = " + repr(M.classification_report(y3t, y3p)))
print("rep2 = " + repr(M.classification_report(y3t, y3p, target_names=["zero", "one", "two", "three"], digits=4)))
print("rep3 = " + repr(M.classification_report(mlt, mlp)))
print("rep4 = " + repr(M.classification_report(st, sp, sample_weight=sw[:7])))
d = M.classification_report(y3t, y3p, output_dict=True)
S("repd_acc", d["accuracy"]); S("repd_mf1", d["macro avg"]["f1-score"]); S("repd_2r", d["2"]["recall"])
