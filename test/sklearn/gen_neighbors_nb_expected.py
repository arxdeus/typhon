# Generates the expected literals for test/sklearn/test_neighbors.py and
# test/sklearn/test_naive_bayes.py (scikit-learn 1.9.1).
#   <venv>/bin/python test/sklearn/gen_neighbors_nb_expected.py
import numpy as np
from sklearn.neighbors import (NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor,
                               RadiusNeighborsClassifier, RadiusNeighborsRegressor,
                               NearestCentroid, KernelDensity)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

def data(n, p, nc):
    X = np.zeros((n, p)); y = np.zeros(n, dtype=int)
    for i in range(n):
        y[i] = i % nc
        for j in range(p):
            v = (i * 7919 + j * 104729 + i * j * 13 + j * j * 3) % 101
            X[i, j] = (v - 50.0) / 25.0 + (0.8 * y[i] if (j + y[i]) % 2 == 0 else -0.3 * y[i])
    return X, y

def counts(n, p, nc):
    X = np.zeros((n, p)); y = np.zeros(n, dtype=int)
    for i in range(n):
        y[i] = i % nc
        for j in range(p):
            X[i, j] = float((i * 31 + j * 17 + i * j * 7 + y[i] * (j + 1)) % 6)
    return X, y

def L(name, a):
    print(f"{name} = {[float(v) for v in np.asarray(a, dtype=float).ravel()]!r}")

def LI(name, a):
    print(f"{name} = {[int(v) for v in np.asarray(a).ravel()]!r}")

X, y = data(60, 4, 3)
Xn = X + 0.01 * np.sin(np.arange(240.0).reshape((60, 4)))  # tie-free distances
Q = Xn[:7] + 0.13
yr = X[:, 0] * 2.0 - X[:, 1] + 0.5 * y
X0 = X
X = Xn
print("# neighbors")
for m, p in (("minkowski", 2), ("minkowski", 1), ("minkowski", np.inf), ("minkowski", 3), ("cosine", 2)):
    tag = f"{m}_{p}".replace("inf", "inf")
    nn = NearestNeighbors(n_neighbors=4, metric=m, p=p, algorithm="brute").fit(X)
    d, i = nn.kneighbors(Q)
    L(f"nn_d_{tag}", d); LI(f"nn_i_{tag}", i)
nn = NearestNeighbors(n_neighbors=3, algorithm="kd_tree").fit(X)
d, i = nn.kneighbors()
L("nn_self_d", d[:5]); LI("nn_self_i", i[:5])
G = nn.kneighbors_graph(Q[:3], mode="distance").toarray()
L("nn_graph", G)
rd, ri = nn.radius_neighbors(Q[:3], radius=1.2, sort_results=True)
for t in range(3):
    L(f"rn_d{t}", rd[t]); LI(f"rn_i{t}", ri[t])
for w in ("uniform", "distance"):
    c = KNeighborsClassifier(n_neighbors=5, weights=w).fit(X, y)
    L(f"knc_proba_{w}", c.predict_proba(Q)); LI(f"knc_pred_{w}", c.predict(Q))
    print(f"knc_score_{w} = {c.score(X, y)!r}")
    r = KNeighborsRegressor(n_neighbors=4, weights=w).fit(X, yr)
    L(f"knr_pred_{w}", r.predict(Q)); print(f"knr_score_{w} = {r.score(X, yr)!r}")
rc = RadiusNeighborsClassifier(radius=1.6, weights="distance", outlier_label=1).fit(X, y)
LI("rnc_pred", rc.predict(Q)); L("rnc_proba", rc.predict_proba(Q))
rr = RadiusNeighborsRegressor(radius=1.6).fit(X, yr)
L("rnr_pred", rr.predict(Q))
for m in ("euclidean", "manhattan"):
    nc = NearestCentroid(metric=m).fit(X, y)
    L(f"nc_cent_{m}", nc.centroids_); LI(f"nc_pred_{m}", nc.predict(Q))
nc = NearestCentroid(shrink_threshold=0.3, priors="empirical").fit(X[:50], y[:50])
L("ncs_cent", nc.centroids_); LI("ncs_pred", nc.predict(Q)); L("ncs_proba", nc.predict_proba(Q))
for k in ("gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"):
    kd = KernelDensity(kernel=k, bandwidth=1.3).fit(X)
    L(f"kde_{k}", kd.score_samples(Q))
kd = KernelDensity(kernel="cosine", bandwidth=1.3).fit(X[:, :3])
L("kde_cosine3", kd.score_samples(Q[:, :3]))
kd = KernelDensity(bandwidth="scott").fit(X)
print(f"kde_scott_bw = {kd.bandwidth_!r}"); L("kde_scott", kd.score_samples(Q[:3]))
kd = KernelDensity(bandwidth="silverman", kernel="epanechnikov").fit(X, sample_weight=np.arange(60) % 4 + 0.5)
L("kde_silv_w", kd.score_samples(Q[:3]))
L("cvs_knn", cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)), X, y))

print("# naive_bayes")
X = X0
Q = X[:7] + 0.13
g = GaussianNB().fit(X, y)
L("gnb_theta", g.theta_); L("gnb_var", g.var_); L("gnb_prior", g.class_prior_)
print(f"gnb_eps = {float(g.epsilon_)!r}")
L("gnb_proba", g.predict_proba(Q)); L("gnb_logp", g.predict_log_proba(Q)); LI("gnb_pred", g.predict(X))
g2 = GaussianNB(priors=[0.2, 0.3, 0.5], var_smoothing=1e-3)
g2.partial_fit(X[:25], y[:25], classes=[0, 1, 2]); g2.partial_fit(X[25:], y[25:])
L("gnb_pf_theta", g2.theta_); L("gnb_pf_var", g2.var_); L("gnb_pf_proba", g2.predict_proba(Q))
sw = (np.arange(60) % 5) + 1.0
g3 = GaussianNB().fit(X, y, sample_weight=sw)
L("gnb_sw_theta", g3.theta_); L("gnb_sw_var", g3.var_)
L("cvs_gnb", cross_val_score(GaussianNB(), X, y))
C, yc = counts(50, 6, 3)
Cq = C[:6][::-1] + 1.0
m = MultinomialNB(alpha=0.5).fit(C, yc)
L("mnb_flp", m.feature_log_prob_); L("mnb_clp", m.class_log_prior_); L("mnb_proba", m.predict_proba(Cq)); LI("mnb_pred", m.predict(C))
m2 = MultinomialNB(fit_prior=False); m2.partial_fit(C[:20], yc[:20], classes=[0, 1, 2]); m2.partial_fit(C[20:], yc[20:], sample_weight=sw[:30])
L("mnb_pf_flp", m2.feature_log_prob_); L("mnb_pf_proba", m2.predict_proba(Cq))
cn = ComplementNB().fit(C, yc); L("cnb_flp", cn.feature_log_prob_); L("cnb_proba", cn.predict_proba(Cq)); LI("cnb_pred", cn.predict(C))
cn2 = ComplementNB(norm=True, alpha=0.3).fit(C, yc); L("cnb_norm_flp", cn2.feature_log_prob_); LI("cnb_norm_pred", cn2.predict(Cq))
b = BernoulliNB(binarize=2.0).fit(C, yc); L("bnb_flp", b.feature_log_prob_); L("bnb_proba", b.predict_proba(Cq)); LI("bnb_pred", b.predict(C))
ca = CategoricalNB(alpha=0.7).fit(C, yc)
L("catnb_proba", ca.predict_proba(C[:6][::-1])); LI("catnb_pred", ca.predict(C)); LI("catnb_ncat", ca.n_categories_)
L("catnb_flp0", ca.feature_log_prob_[0])
