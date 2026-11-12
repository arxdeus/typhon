# Generates expected values for test/pyriemann/*.py using real pyriemann 0.12.
# Run: /path/to/venv/bin/python test/pyriemann/gen_expected.py
# Data generators below are mirrored exactly in the .py tests.
import numpy as np
import warnings
warnings.simplefilter("ignore")
from pyriemann.geometry import base, distance as D, mean as M, geodesic as G
from pyriemann.geometry import tangentspace as TS, covariance as C
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM

np.set_printoptions(precision=17)

def signals(n_trials, n_ch, n_t):
    # Deterministic pseudo-signals from integer arithmetic.
    X = np.zeros((n_trials, n_ch, n_t))
    for i in range(n_trials):
        for c in range(n_ch):
            for t in range(n_t):
                v = (i * 1009 + c * 7919 + t * 104729 + i * c * t * 13
                     + t * t * (c + 1) * 3 + c * c * t * 7) % 101
                X[i, c, t] = (v - 50.0) / 25.0 + (0.5 * (i % 2)) * ((c + t) % 3 == 0)
    return X

def spd(n_mats, n):
    out = np.zeros((n_mats, n, n))
    for k in range(n_mats):
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                A[i, j] = ((i + 1) * (j + 2) * (k + 3) % 7 - 3.0) / 4.0
        out[k] = A @ A.T + (k + 1.0) * np.eye(n)
    return out

def lit(a):
    return repr(np.asarray(a).ravel().tolist())

P = spd(5, 4)
A, B = P[0], P[1]
print("# base")
for f in ["sqrtm", "invsqrtm", "logm", "expm"]:
    print(f, lit(getattr(base, f)(A / 10 if f == "expm" else A)))
print("powm", lit(base.powm(A, 0.3)))
print("# distance A,B")
for m in ["riemann", "logeuclid", "euclid", "logdet", "kullback", "kullback_sym",
          "wasserstein", "harmonic", "thompson", "chol"]:
    print(m, repr(float(D.distance(A, B, metric=m))))
print("dist3d_riemann", lit(D.distance(P, B, metric="riemann")))
for m in ["riemann", "logeuclid", "euclid", "harmonic", "logdet", "kullback"]:
    print("pair_" + m, lit(D.pairwise_distance(P, metric=m)))
print("pairXY_riemann", lit(D.pairwise_distance(P, P[:2], metric="riemann")))
print("# mean")
w = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
for m in ["riemann", "logeuclid", "euclid", "logdet", "harmonic", "kullback_sym",
          "wasserstein"]:
    print("mean_" + m, lit(M.gmean(P, metric=m)))
print("mean_riemann_w", lit(M.mean_riemann(P, sample_weight=w)))
print("mean_power_05", lit(M.mean_power(P, 0.5)))
print("# geodesic")
for m in ["riemann", "logeuclid", "euclid", "wasserstein"]:
    print("geo_" + m, lit(G.geodesic(A, B, 0.3, metric=m)))
print("# tangentspace")
Cref = M.mean_riemann(P)
for m in ["riemann", "logeuclid", "euclid", "wasserstein"]:
    T = TS.tangent_space(P, Cref, metric=m)
    print("ts_" + m, lit(T))
    print("unts_" + m, lit(TS.untangent_space(T, Cref, metric=m)))
print("upper", lit(TS.upper(A)))
print("transport_riemann", lit(TS.transport(A - B, A, Cref)))
print("# covariance")
X = signals(6, 4, 50)
for e in ["cov", "scm", "lwf", "oas", "sch", "corr"]:
    print("cov_" + e, lit(C.covariances(X, estimator=e)))
print("norm_trace", lit(C.normalize(P, "trace")))
print("norm_corr", lit(C.normalize(P, "corr")))
print("norm_det", lit(C.normalize(P, "determinant")))
print("nondiag", lit(C.get_nondiag_weight(P)))
print("nspd", lit(base.nearest_sym_pos_def(A - 6 * np.eye(4))))
print("# pipeline (acceptance shapes)")
X = signals(40, 8, 256)
y = np.array([i % 2 for i in range(40)])
scm = Covariances("cov").fit_transform(X)
oasc = Covariances("oas").fit_transform(X)
print("pipe_scm0", lit(scm[0]))
print("pipe_oas0", lit(oasc[0]))
pm = M.mean_logeuclid(oasc)
print("pipe_pm", lit(pm))
T = TangentSpace(metric="logeuclid").fit(scm).transform(scm)
print("pipe_ts_shape", T.shape)
print("pipe_ts_rows02", lit(T[:2]))
print("pipe_ts_sum", repr(float(T.sum())))
ts2 = TS.tangent_space(oasc, pm, metric="riemann")
print("pipe_prts_rows01", lit(ts2[:2]))
print("pipe_prts_sum", repr(float(ts2.sum())))
preds = np.zeros(40, dtype=int)
for k in range(5):
    te = np.arange(k * 8, (k + 1) * 8)
    tr = np.setdiff1d(np.arange(40), te)
    clf = MDM(metric="logeuclid").fit(scm[tr], y[tr])
    preds[te] = clf.predict(scm[te])
    if k == 0:
        print("mdm_fold0_proba", lit(clf.predict_proba(scm[te])))
        print("mdm_fold0_transform", lit(clf.transform(scm[te])))
        print("mdm_fold0_covmeans", lit(clf.covmeans_))
        print("mdm_fold0_score", repr(clf.score(scm[te], y[te])))
print("mdm_preds", lit(preds))
clf = MDM().fit(scm, y)
print("mdm_riemann_transform03", lit(clf.transform(scm[:3])))
