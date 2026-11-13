# Expected literals for test/sklearn/test_metrics_rank_reg.py (scikit-learn 1.9.1).
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn import metrics as M

def L(name, a):
    a = np.asarray(a, dtype=float).ravel()
    print(f"{name} = {[float(v) for v in a]}")

def S(name, v):
    print(f"{name} = {float(v)!r}")

yt = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1])
sc = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.6, 0.3, 0.7, 0.4, 0.95, 0.4, 0.3])
sw = np.array([1.0 + (i % 3) * 0.5 for i in range(12)])
y3 = np.array([0, 1, 2, 2, 1, 0, 2, 1, 0, 2, 1, 0])
P3 = np.array([[(i * 7 + j * 3) % 5 + 1.0 for j in range(3)] for i in range(12)]); P3 = P3 / P3.sum(axis=1, keepdims=True)
mlt = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1]])
mls = np.array([[0.9, 0.2, 0.4], [0.1, 0.8, 0.3], [0.6, 0.6, 0.1], [0.3, 0.2, 0.7], [0.5, 0.4, 0.4], [0.2, 0.3, 0.9]])

f, t, th = M.roc_curve(yt, sc); L("roc_f", f); L("roc_t", t); L("roc_th", th)
f, t, th = M.roc_curve(yt, sc, drop_intermediate=False, sample_weight=sw); L("roc_f2", f); L("roc_t2", t); L("roc_th2", th)
p, r, th = M.precision_recall_curve(yt, sc); L("pr_p", p); L("pr_r", r); L("pr_th", th)
p, r, th = M.precision_recall_curve(yt, sc, pos_label=0, drop_intermediate=True); L("pr_p2", p); L("pr_r2", r); L("pr_th2", th)
a, b, th = M.det_curve(yt, sc); L("det_a", a); L("det_b", b); L("det_th", th)
S("auc_xy", M.auc([0.0, 0.5, 1.0, 2.0], [0.0, 0.7, 0.8, 1.0]))
S("rauc", M.roc_auc_score(yt, sc)); S("rauc_w", M.roc_auc_score(yt, sc, sample_weight=sw)); S("rauc_mf", M.roc_auc_score(yt, sc, max_fpr=0.3))
S("rauc_ovr", M.roc_auc_score(y3, P3, multi_class="ovr")); S("rauc_ovr_w", M.roc_auc_score(y3, P3, multi_class="ovr", average="weighted"))
S("rauc_ovo", M.roc_auc_score(y3, P3, multi_class="ovo")); S("rauc_ovo_w", M.roc_auc_score(y3, P3, multi_class="ovo", average="weighted"))
S("rauc_ml", M.roc_auc_score(mlt, mls)); S("rauc_ml_mi", M.roc_auc_score(mlt, mls, average="micro"))
L("rauc_ml_none", M.roc_auc_score(mlt, mls, average=None))
S("ap", M.average_precision_score(yt, sc)); S("ap_w", M.average_precision_score(yt, sc, sample_weight=sw))
S("ap_ml", M.average_precision_score(mlt, mls)); S("ap_ml_s", M.average_precision_score(mlt, mls, average="samples"))
S("ap_mc", M.average_precision_score(y3, P3))
S("lrap", M.label_ranking_average_precision_score(mlt, mls)); S("cov", M.coverage_error(mlt, mls)); S("lrl", M.label_ranking_loss(mlt, mls))
rel = np.array([[3, 2, 0, 1, 0], [0, 1, 2, 3, 1], [1, 1, 0, 0, 2]]); rs = np.array([[0.5, 0.4, 0.4, 0.1, 0.2], [0.1, 0.2, 0.3, 0.4, 0.5], [0.3, 0.3, 0.2, 0.9, 0.1]])
S("dcg", M.dcg_score(rel, rs)); S("dcg_k", M.dcg_score(rel, rs, k=2)); S("ndcg", M.ndcg_score(rel, rs)); S("ndcg_k", M.ndcg_score(rel, rs, k=3, ignore_ties=True))
S("topk", M.top_k_accuracy_score(y3, P3, k=2)); S("topk1", M.top_k_accuracy_score(y3, P3, k=1, normalize=False))
# regression
rt = np.array([3.0, -0.5, 2.0, 7.0, 4.2, 1.1, 0.3, 5.5]); rp = np.array([2.5, 0.0, 2.1, 7.8, 3.9, 1.0, 0.9, 4.0])
w8 = sw[:8]
Rt = np.stack([rt, rt * 0.5 + 1.0], axis=1); Rp = np.stack([rp, rp * 0.4 + 1.3], axis=1)
S("mse", M.mean_squared_error(rt, rp)); S("mse_w", M.mean_squared_error(rt, rp, sample_weight=w8)); S("rmse", M.root_mean_squared_error(rt, rp))
S("mae", M.mean_absolute_error(rt, rp)); L("mae_raw", M.mean_absolute_error(Rt, Rp, multioutput="raw_values"))
S("mae_mo", M.mean_absolute_error(Rt, Rp, multioutput=[0.3, 0.7]))
S("medae", M.median_absolute_error(rt, rp)); S("medae_w", M.median_absolute_error(rt, rp, sample_weight=w8))
S("maxe", M.max_error(rt, rp)); S("r2", M.r2_score(rt, rp)); S("r2_w", M.r2_score(rt, rp, sample_weight=w8))
S("r2_vw", M.r2_score(Rt, Rp, multioutput="variance_weighted")); L("r2_raw", M.r2_score(Rt, Rp, multioutput="raw_values"))
S("ev", M.explained_variance_score(rt, rp)); S("ev_w", M.explained_variance_score(Rt, Rp, sample_weight=w8))
S("mape", M.mean_absolute_percentage_error(rt, rp))
pt = np.abs(rt) + 0.1; pp = np.abs(rp) + 0.2
S("msle", M.mean_squared_log_error(pt, pp)); S("rmsle", M.root_mean_squared_log_error(pt, pp))
S("pois", M.mean_poisson_deviance(pt, pp)); S("gam", M.mean_gamma_deviance(pt, pp, sample_weight=w8))
S("tw15", M.mean_tweedie_deviance(pt, pp, power=1.5)); S("tw3", M.mean_tweedie_deviance(pt, pp, power=3)); S("twm", M.mean_tweedie_deviance(rt, pp, power=-1))
S("d2tw", M.d2_tweedie_score(pt, pp, power=1.5)); S("d2pin", M.d2_pinball_score(rt, rp, alpha=0.3)); S("d2ae", M.d2_absolute_error_score(rt, rp, sample_weight=w8))
S("pin", M.mean_pinball_loss(rt, rp, alpha=0.9))
