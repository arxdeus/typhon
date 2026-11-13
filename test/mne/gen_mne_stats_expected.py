# Generates test/mne/test_mne_stats.py from templates/test_mne_stats.tpl using
# reference values from real mne 1.13.2.
# Run: /path/to/venv/bin/python test/mne/gen_mne_stats_expected.py
import os
import numpy as np
import mne
from mne import stats as S
from scipy import sparse

mne.set_log_level("ERROR")
HERE = os.path.dirname(os.path.abspath(__file__))


def lst(a):
    a = np.asarray(a, dtype=float).ravel()
    return "[" + ", ".join(repr(float(v)) for v in a) + "]"


def ilst(a):
    return "[" + ", ".join(str(int(v)) for v in np.ravel(a)) + "]"


def data(n, m, k=0.0, off=0.0):
    i = np.arange(n)[:, None]
    j = np.arange(m)[None, :]
    return np.sin(0.7 * i * (j + 1) + 0.3 * j + k) + 0.4 * np.cos(1.9 * i + 0.5 * j * j + k) + off


def cl_desc(res, out_type="indices"):
    T, clusters, pv, H0 = res
    E = {"T": lst(T), "pv": lst(pv), "H0": lst(H0), "n": str(len(clusters))}
    if out_type == "indices":
        E["c0"] = ilst(np.concatenate([np.ravel(np.asarray(c)) for c in clusters[0]])) if len(clusters) else "[]"
    else:
        if len(clusters):
            m = np.zeros(np.shape(T), bool)
            m[clusters[0]] = True
            E["c0"] = ilst(np.flatnonzero(m))
        else:
            E["c0"] = "[]"
    return E


E = {}
X1 = data(12, 30, off=0.35)
E["tt1"] = lst(S.ttest_1samp_no_p(X1))
E["tt1s"] = lst(S.ttest_1samp_no_p(X1, sigma=1e-3))
E["tt1a"] = lst(S.ttest_1samp_no_p(X1, sigma=0.1, method="absolute"))
A = data(10, 20, 0.0, 0.3)
B = data(14, 20, 1.0)
E["tti"] = lst(S.ttest_ind_no_p(A, B))
E["ttiu"] = lst(S.ttest_ind_no_p(A, B, equal_var=False, sigma=0.01))
C3 = data(9, 20, 2.0, -0.2)
E["f1"] = lst(S.f_oneway(A, B, C3))
E["f1s"] = lst(S.f_oneway(A, B, C3, sigma=0.1))
p = np.abs(np.sin(np.arange(25) * 1.7)) * 0.08
E["fdr_r"] = ilst(S.fdr_correction(p)[0])
E["fdr_p"] = lst(S.fdr_correction(p)[1])
E["fdrn_r"] = ilst(S.fdr_correction(p, 0.1, "negcorr")[0])
E["fdrn_p"] = lst(S.fdr_correction(p, 0.1, "negcorr")[1])
E["bon_r"] = ilst(S.bonferroni_correction(p * 0.05)[0])
E["bon_p"] = lst(S.bonferroni_correction(p * 0.05)[1])
# repeated-measures ANOVA: 8 subjects, 2x3 conditions, 5 tests
D = np.stack([data(8, 5, k, 0.1 * k) for k in range(6)], axis=1)
f, pv = S.f_mway_rm(D, [2, 3])
E["mw_f"] = lst(f)
E["mw_p"] = lst(pv)
f, pv = S.f_mway_rm(D, [2, 3], effects="A:B", correction=True)
E["mwc_f"] = lst(f)
E["mwc_p"] = lst(pv)
E["fth"] = lst(S.f_threshold_mway_rm(8, [2, 3]))
E["fth1"] = lst([S.f_threshold_mway_rm(8, [2, 3], effects="B", pvalue=0.01)])
# permutation t test
T, pv, H0 = S.permutation_t_test(X1[:, :8], n_permutations=200, seed=3)
E["pt_T"] = lst(T); E["pt_p"] = lst(pv); E["pt_H0"] = lst(H0)
T, pv, H0 = S.permutation_t_test(X1[:6, :8], n_permutations=1000, tail=1)
E["pte_p"] = lst(pv); E["pte_H0"] = lst(H0)
T, pv, H0 = S.permutation_t_test(X1[:, :8], n_permutations=100, rng=7, tail=-1)
E["ptg_p"] = lst(pv); E["ptg_H0"] = lst(H0)
X25 = data(24, 6, 0.5, 0.2)
T, pv, H0 = S.permutation_t_test(X25, n_permutations=50, seed=1)
E["ptb_p"] = lst(pv); E["ptb_H0"] = lst(H0)
# bootstrap
E["bs"] = lst(S.bootstrap_confidence_interval(X1[:, :4], n_bootstraps=300, random_state=0))
E["bsm"] = lst(S.bootstrap_confidence_interval(X1[:, :4], ci=0.9, n_bootstraps=200, stat_fun="median", rng=5))
# cluster tests: 1-D time series
for key, kw in {"c1": dict(seed=0), "c1t": dict(rng=2, tail=1), "c1n": dict(threshold=1.5, seed=4, tail=0)}.items():
    r = S.permutation_cluster_1samp_test(X1, n_permutations=128, **kw)
    for k2, v in cl_desc(r).items():
        E[key + "_" + k2] = v
# 2-D (time x freq) with ndimage lattice
X2 = np.stack([data(10, 12, 0.2 * k, 0.3) for k in range(5)], axis=2)  # (10, 12, 5)
r = S.permutation_cluster_1samp_test(X2, threshold=1.0, n_permutations=64, seed=1)
for k2, v in cl_desc(r).items():
    E["c2_" + k2] = v
# TFCE
r = S.permutation_cluster_1samp_test(X1[:, :12], threshold=dict(start=0.5, step=0.5), n_permutations=40, seed=2)
for k2, v in cl_desc(r).items():
    E["tf_" + k2] = v
# between-groups F test
r = S.permutation_cluster_test([A, B, C3], n_permutations=60, seed=0)
for k2, v in cl_desc(r).items():
    E["cb_" + k2] = v
r = S.permutation_cluster_test([A, B], threshold=2.0, n_permutations=60, rng=3, out_type="mask",
                               stat_fun=S.ttest_ind_no_p, tail=0)
for k2, v in cl_desc(r, "mask").items():
    E["cbt_" + k2] = v
# spatio-temporal with adjacency: (n_obs, n_times=6, n_space=5)
adj = sparse.coo_array(np.array([[0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0],
                                 [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]], float))
XS = np.stack([data(11, 5, 0.3 * t, 0.25) for t in range(6)], axis=1)
r = S.spatio_temporal_cluster_1samp_test(XS, threshold=1.2, adjacency=adj, n_permutations=50, seed=5)
for k2, v in cl_desc(r).items():
    E["st_" + k2] = v
r = S.spatio_temporal_cluster_1samp_test(XS, threshold=1.2, adjacency=adj, n_permutations=50,
                                         seed=5, spatial_exclude=[1], step_down_p=0.3)
for k2, v in cl_desc(r).items():
    E["ste_" + k2] = v
XS2 = np.stack([data(9, 5, 0.3 * t + 1.0) for t in range(6)], axis=1)
r = S.spatio_temporal_cluster_test([XS, XS2], adjacency=adj, n_permutations=40, seed=8, threshold=0.5)
for k2, v in cl_desc(r).items():
    E["stb_" + k2] = v
ca = S.combine_adjacency(3, adj)
E["ca_r"] = ilst(ca.row); E["ca_c"] = ilst(ca.col); E["ca_n"] = str(ca.shape[0])
# linear regression on epochs
info = mne.create_info(["a", "b", "c"], 100.0, "eeg")
ep = mne.EpochsArray(np.stack([data(3, 20, 0.1 * e, 0.0) * 1e-6 for e in range(15)]), info, tmin=-0.05)
dm = np.column_stack([np.ones(15), np.arange(15) % 3, np.sin(np.arange(15.0))])
lm = S.linear_regression(ep, dm, names=["icpt", "cond", "s"])
E["lr_b"] = lst(lm["cond"].beta.data); E["lr_se"] = lst(lm["cond"].stderr.data)
E["lr_t"] = lst(lm["s"].t_val.data); E["lr_p"] = lst(lm["s"].p_val.data)
E["lr_l"] = lst(lm["icpt"].mlog10_p_val.data)

src = open(os.path.join(HERE, "templates", "test_mne_stats.tpl")).read()
for k, v in E.items():
    src = src.replace("@@" + k + "@@", v)
assert "@@" not in src, src[src.index("@@"):src.index("@@") + 40]
open(os.path.join(HERE, "test_mne_stats.py"), "w").write(src)
print("wrote test_mne_stats.py with", len(E), "values")
