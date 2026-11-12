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
    if isinstance(v, np.ndarray):
        v = float(v.ravel()[0])  # e.g. tied BayesianGaussianMixture lower bound has shape (1,)
    elif isinstance(v, (np.floating, np.integer)):
        v = v.item()
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

def mixture():
    from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
    X, y = data(60, 3, 3)
    for ct in ["full", "tied", "diag", "spherical"]:
        for ini in ["kmeans", "k-means++", "random", "random_from_data"]:
            g = GaussianMixture(3, covariance_type=ct, init_params=ini, random_state=0, max_iter=200).fit(X)
            tag = f"gm_{ct}_{ini.replace('-', '').replace('+', 'p')}"
            L(tag + "_w", g.weights_); L(tag + "_m", g.means_); L(tag + "_c", g.covariances_)
            L(tag + "_p", g.precisions_); S(tag + "_it", g.n_iter_); S(tag + "_lb", g.lower_bound_)
            L(tag + "_ss", g.score_samples(X[:5])); L(tag + "_pp", g.predict_proba(X[:3]))
            S(tag + "_bic", g.bic(X)); S(tag + "_aic", g.aic(X))
    g = GaussianMixture(2, means_init=[[0, 0, 0], [1, 1, 1]], weights_init=[0.3, 0.7], random_state=1, n_init=2).fit(X)
    L("gm_mi_m", g.means_); L("gm_mi_w", g.weights_); L("gm_mi_pred", g.predict(X[:10]))
    Xs, ys = g.sample(7)
    L("gm_sample_X", Xs); L("gm_sample_y", ys)
    for ct in ["full", "tied", "diag", "spherical"]:
        for wt in ["dirichlet_process", "dirichlet_distribution"]:
            b = BayesianGaussianMixture(n_components=4, covariance_type=ct, weight_concentration_prior_type=wt, random_state=0, max_iter=300).fit(X)
            tag = f"bgm_{ct}_{wt[10:]}"
            L(tag + "_w", b.weights_); L(tag + "_m", b.means_); L(tag + "_c", b.covariances_)
            S(tag + "_it", b.n_iter_); S(tag + "_lb", b.lower_bound_); L(tag + "_ss", b.score_samples(X[:4]))
            L(tag + "_dof", b.degrees_of_freedom_)

def feature_selection():
    from sklearn.feature_selection import (f_classif, f_regression, r_regression, chi2, mutual_info_classif,
        mutual_info_regression, SelectKBest, SelectPercentile, SelectFpr, SelectFdr, SelectFwe,
        GenericUnivariateSelect, VarianceThreshold, SelectFromModel, RFE, RFECV, SequentialFeatureSelector)
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    X, y = data(40, 8, 3)
    yr = X[:, 0] * 2.0 - X[:, 3] + 0.1 * np.arange(40) % 7
    Xp = np.abs(X)
    f, p = f_classif(X, y); L("fc_f", f); L("fc_p", p)
    f, p = f_regression(X, yr); L("fr_f", f); L("fr_p", p)
    L("rr", r_regression(X, yr)); L("rr_nc", r_regression(X, yr, center=False))
    c, p = chi2(Xp, y); L("chi_c", c); L("chi_p", p)
    L("mic", mutual_info_classif(X, y, random_state=0))
    L("mir", mutual_info_regression(X, yr, random_state=0))
    Xd = np.floor(np.abs(X[:, :3]))
    L("mic_d", mutual_info_classif(Xd, y, discrete_features=True, random_state=1))
    L("mir_d", mutual_info_regression(np.hstack([Xd, X[:, 3:5]]), yr, discrete_features=[0, 1, 2], random_state=2))
    L("kb", SelectKBest(f_classif, k=3).fit(X, y).get_support())
    L("kb_t", SelectKBest(chi2, k=2).fit_transform(Xp, y)[:3])
    L("pc", SelectPercentile(f_regression, percentile=40).fit(X, yr).get_support())
    L("fpr", SelectFpr(f_classif, alpha=0.3).fit(X, y).get_support())
    L("fdr", SelectFdr(f_classif, alpha=0.5).fit(X, y).get_support())
    L("fwe", SelectFwe(f_classif, alpha=0.5).fit(X, y).get_support())
    L("gus", GenericUnivariateSelect(f_classif, mode="k_best", param=4).fit(X, y).get_support())
    Xv = X.copy(); Xv[:, 2] = 1.0; Xv[:, 5] *= 0.1
    vt = VarianceThreshold(0.05).fit(Xv); L("vt_var", vt.variances_); L("vt_sup", vt.get_support())
    vt0 = VarianceThreshold().fit(Xv); L("vt0_sup", vt0.get_support())
    sfm = SelectFromModel(LogisticRegression(max_iter=500)).fit(X, y)
    L("sfm_sup", sfm.get_support()); S("sfm_th", sfm.threshold_)
    sfm2 = SelectFromModel(LinearDiscriminantAnalysis(), threshold="1.2*median", max_features=3).fit(X, y)
    L("sfm2_sup", sfm2.get_support())
    rfe = RFE(LogisticRegression(max_iter=500), n_features_to_select=3, step=2).fit(X, y)
    L("rfe_sup", rfe.support_); L("rfe_rank", rfe.ranking_)
    rcv = RFECV(LinearDiscriminantAnalysis(), cv=4).fit(X, y)
    L("rcv_sup", rcv.support_); L("rcv_rank", rcv.ranking_); L("rcv_mean", rcv.cv_results_["mean_test_score"])
    sfs = SequentialFeatureSelector(LinearDiscriminantAnalysis(), n_features_to_select=3, cv=3).fit(X, y)
    L("sfs_f", sfs.get_support())
    sfs = SequentialFeatureSelector(LinearDiscriminantAnalysis(), n_features_to_select=5, direction="backward", cv=3).fit(X, y)
    L("sfs_b", sfs.get_support())

def nan_data():
    X, y = data(25, 5, 3)
    for i in range(25):
        for j in range(5):
            if (i * 3 + j * 7) % 6 == 0:
                X[i, j] = np.nan
    X[2, 4] = 1.5; X[7, 4] = 1.5
    return X, y

def impute():
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer, MissingIndicator
    from sklearn.linear_model import BayesianRidge
    X, y = nan_data()
    Xr = np.round(X)
    for st in ["mean", "median", "most_frequent", "constant"]:
        im = SimpleImputer(strategy=st, fill_value=-1.0)
        L(f"si_{st}", im.fit_transform(Xr if st == "most_frequent" else X)); L(f"si_{st}_st", im.statistics_)
    im = SimpleImputer(add_indicator=True).fit(X); L("si_ind", im.transform(X[:6]))
    L("mi", MissingIndicator().fit_transform(X)); L("mi_all", MissingIndicator(features="all").fit(X).transform(X[:4]))
    L("knn", KNNImputer(n_neighbors=3).fit_transform(X))
    L("knn_d", KNNImputer(n_neighbors=4, weights="distance").fit(X).transform(X[:8]))
    Xc = X[:, :4].copy(); Xc[np.isnan(Xc)] = 0.0
    yb = Xc @ np.array([1.0, -2.0, 0.5, 0.3]) + 0.1 * np.sin(np.arange(25))
    br = BayesianRidge(compute_score=True).fit(Xc, yb)
    L("br_coef", br.coef_); S("br_int", br.intercept_); S("br_alpha", br.alpha_); S("br_lambda", br.lambda_)
    S("br_it", br.n_iter_); L("br_scores", br.scores_)
    m, sd = br.predict(Xc[:4], return_std=True); L("br_m", m); L("br_sd", sd)
    br2 = BayesianRidge().fit(Xc[:3], yb[:3]); L("br2_coef", br2.coef_)
    it = IterativeImputer(random_state=0)
    L("it", it.fit_transform(X)); S("it_n", it.n_iter_); L("it_tr", it.transform(X[:5] + 0.5))
    it2 = IterativeImputer(random_state=0, n_nearest_features=2, imputation_order="random", max_iter=5)
    L("it2", it2.fit_transform(X))
    it3 = IterativeImputer(initial_strategy="median", imputation_order="descending", min_value=-1.0, max_value=2.0, add_indicator=True)
    L("it3", it3.fit_transform(X))

def manifold_tsne():
    from sklearn.manifold import TSNE
    X, y = data(40, 5, 3)
    t = TSNE(method="exact", perplexity=8, random_state=0, max_iter=400, init="random")
    L("tsne_ex_r", t.fit_transform(X)); S("tsne_ex_r_kl", t.kl_divergence_); S("tsne_ex_r_it", t.n_iter_)
    t = TSNE(method="exact", perplexity=10, random_state=1, max_iter=300, init="pca", learning_rate=100.0)
    L("tsne_ex_p", t.fit_transform(X)); S("tsne_ex_p_kl", t.kl_divergence_)
    t = TSNE(perplexity=8, random_state=0, max_iter=400, init="pca")
    L("tsne_bh_p", t.fit_transform(X)); S("tsne_bh_p_kl", t.kl_divergence_); S("tsne_bh_p_it", t.n_iter_)
    t = TSNE(n_components=3, perplexity=5, random_state=2, max_iter=300, init="random", angle=0.3)
    L("tsne_bh3", t.fit_transform(X)); S("tsne_bh3_kl", t.kl_divergence_)

def manifold():
    from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, SpectralEmbedding, trustworthiness
    from sklearn.manifold import ClassicalMDS
    X, y = data(40, 5, 3)
    m = MDS(2, random_state=0, n_init=2, init="random", max_iter=200)
    L("mds_m", m.fit_transform(X)); S("mds_m_stress", m.stress_); S("mds_m_it", m.n_iter_)
    m = MDS(2, metric_mds=False, random_state=1, n_init=1, init="random", max_iter=100)
    L("mds_nm", m.fit_transform(X)); S("mds_nm_stress", m.stress_); S("mds_nm_it", m.n_iter_)
    m = MDS(2, init="classical_mds", max_iter=100)
    L("mds_c", m.fit_transform(X)); S("mds_c_stress", m.stress_)
    L("cmds", ClassicalMDS(3).fit_transform(X))
    im = Isomap(n_neighbors=6, n_components=2).fit(X)
    L("iso_emb", im.embedding_); L("iso_dist", im.dist_matrix_[:3]); L("iso_tr", im.transform(X[:4] + 0.05))
    S("iso_err", im.reconstruction_error())
    for meth, k in [("standard", 8), ("modified", 8), ("ltsa", 8), ("hessian", 9)]:
        e = LocallyLinearEmbedding(n_neighbors=k, n_components=2, method=meth, eigen_solver="dense").fit(X)
        L(f"lle_{meth}", e.embedding_); S(f"lle_{meth}_err", e.reconstruction_error_)
    e = LocallyLinearEmbedding(n_neighbors=8, n_components=2, eigen_solver="dense").fit(X)
    L("lle_tr", e.transform(X[:3] + 0.05))
    L("se_nn", SpectralEmbedding(n_components=2, random_state=0, n_neighbors=8).fit_transform(X))
    L("se_rbf", SpectralEmbedding(n_components=2, random_state=0, affinity="rbf", gamma=0.2).fit_transform(X))
    Z = X[:, :2] + 0.3 * X[:, 2:4]
    S("tw", trustworthiness(X, Z, n_neighbors=5)); S("tw3", trustworthiness(X, X[:, [0, 2, 4]], n_neighbors=3))

for sec in sys.argv[1:]:
    globals()[sec]()
