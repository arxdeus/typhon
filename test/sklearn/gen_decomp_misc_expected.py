# Prints expected literals for test_decomposition / test_manifold / test_mixture /
# test_feature_selection / test_impute (scikit-learn 1.9.1).
import sys
import numpy as np

def data(n, p, nc):
    X = np.zeros((n, p)); y = np.zeros(n, dtype=int)
    for i in range(n):
        y[i] = i % nc
        for j in range(p):
            v = (i * 7919 + j * 104729 + i * j * 13 + j * j * 3) % 101
            X[i, j] = (v - 50.0) / 25.0 + (0.8 * y[i] if (j + y[i]) % 2 == 0 else -0.3 * y[i])
    return X, y

def L(name, a):
    a = np.asarray(a, dtype=float).ravel()
    print(f"{name} = [" + ", ".join(repr(float(v)) for v in a) + "]")

def S(name, v):
    print(f"{name} = {v!r}")

def decomposition():
    from sklearn.decomposition import PCA, TruncatedSVD, NMF, KernelPCA, IncrementalPCA, FactorAnalysis
    X, y = data(30, 6, 3)
    Xp = np.abs(X) + 0.1
    t = TruncatedSVD(3, algorithm="randomized", random_state=0).fit(X)
    L("tsvd_comp", t.components_); L("tsvd_evr", t.explained_variance_ratio_); L("tsvd_sv", t.singular_values_)
    L("tsvd_tr", t.transform(X[:4]))
    t = TruncatedSVD(2, algorithm="arpack").fit(X)
    L("tsvda_comp", t.components_); L("tsvda_ev", t.explained_variance_)
    p = PCA(3, svd_solver="randomized", random_state=1, whiten=True).fit(X)
    L("pcar_comp", p.components_); L("pcar_ev", p.explained_variance_); S("pcar_noise", p.noise_variance_)
    L("pcar_tr", p.transform(X[:3])); L("pcar_ss", p.score_samples(X[:5]))
    p = PCA(2, svd_solver="full").fit(X)
    L("pcaf_ss", p.score_samples(X[:5])); L("pcaf_inv", p.inverse_transform(p.transform(X[:2])))
    for init in ["nndsvd", "nndsvda", "nndsvdar", "random"]:
        m = NMF(3, init=init, random_state=0, max_iter=400)
        W = m.fit_transform(Xp)
        L(f"nmf_cd_{init}_H", m.components_); L(f"nmf_cd_{init}_W", W[:5]); S(f"nmf_cd_{init}_err", m.reconstruction_err_); S(f"nmf_cd_{init}_it", m.n_iter_)
    m = NMF(3, init="nndsvda", solver="mu", beta_loss="kullback-leibler", random_state=0, max_iter=300)
    W = m.fit_transform(Xp)
    L("nmf_kl_H", m.components_); S("nmf_kl_err", m.reconstruction_err_); S("nmf_kl_it", m.n_iter_)
    L("nmf_kl_tr", m.transform(Xp[:3]))
    m = NMF(3, init="nndsvd", solver="mu", random_state=0, max_iter=200, alpha_W=0.01, l1_ratio=0.5)
    W = m.fit_transform(Xp)
    L("nmf_mu_H", m.components_); S("nmf_mu_err", m.reconstruction_err_)
    m = NMF(2, init="random", solver="cd", random_state=3, shuffle=True, alpha_W=0.02, alpha_H=0.01, l1_ratio=0.3)
    W = m.fit_transform(Xp)
    L("nmf_sh_H", m.components_); L("nmf_sh_tr", m.transform(Xp[:3]))
    for k in ["linear", "poly", "rbf", "sigmoid", "cosine"]:
        kp = KernelPCA(3, kernel=k, gamma=0.1 if k != "linear" else None, fit_inverse_transform=(k == "rbf"))
        Z = kp.fit_transform(X)
        L(f"kpca_{k}_eig", kp.eigenvalues_); L(f"kpca_{k}_Z", Z[:4]); L(f"kpca_{k}_tr", kp.transform(X[:3] + 0.1))
        if k == "rbf":
            L("kpca_rbf_inv", kp.inverse_transform(Z[:3]))
    ip = IncrementalPCA(3, batch_size=8).fit(X)
    L("ipca_comp", ip.components_); L("ipca_ev", ip.explained_variance_); L("ipca_evr", ip.explained_variance_ratio_)
    S("ipca_noise", ip.noise_variance_); L("ipca_var", ip.var_)
    fa = FactorAnalysis(2).fit(X)
    L("fa_comp", fa.components_); L("fa_noise", fa.noise_variance_); S("fa_it", fa.n_iter_)
    L("fa_tr", fa.transform(X[:3])); L("fa_ss", fa.score_samples(X[:4]))
    fa = FactorAnalysis(2, rotation="varimax").fit(X)
    L("fav_comp", fa.components_)

for sec in sys.argv[1:]:
    globals()[sec]()
