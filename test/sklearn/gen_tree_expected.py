# Generates the expected literals of test_tree.py / test_ensemble.py
# (run with scikit-learn 1.9.1): python gen_tree_expected.py tree|ensemble
import sys
import warnings
import numpy as np
warnings.filterwarnings("ignore")

def data(n, p, nc):
    X = np.zeros((n, p)); y = np.zeros(n, dtype=int)
    for i in range(n):
        y[i] = i % nc
        for j in range(p):
            v = (i * 7919 + j * 104729 + i * j * 13 + j * j * 3) % 101
            X[i, j] = (v - 50.0) / 25.0 + (0.8 * y[i] if (j + y[i]) % 2 == 0 else -0.3 * y[i])
    return X, y

def reg_target(X):
    n = X.shape[0]
    return np.array([np.sin(X[i, 0]) * 2.0 + X[i, 1] * X[i, 2] + 0.1 * (i % 7) for i in range(n)])

def out(name, v):
    a = np.asarray(v, dtype=float).ravel()
    print(f"{name} = [{', '.join(repr(float(x)) for x in a)}]")

def outi(name, v):
    print(f"{name} = {repr(v)}")

def tree_cases():
    from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                              ExtraTreeClassifier, ExtraTreeRegressor, export_text)
    X, y = data(90, 5, 3)
    Xt, _ = data(20, 5, 3)
    Xt = Xt + 0.37
    yr = reg_target(X)
    yp = np.abs(yr) + 0.5
    cfgs = [
        ("c_gini", DecisionTreeClassifier(random_state=0)),
        ("c_ent_d3", DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=1)),
        ("c_sqrt", DecisionTreeClassifier(max_features="sqrt", random_state=3, min_samples_leaf=2)),
        ("c_leaf", DecisionTreeClassifier(max_leaf_nodes=7, random_state=0)),
        ("c_rand", DecisionTreeClassifier(splitter="random", random_state=5, max_features=0.6)),
        ("c_bal", DecisionTreeClassifier(class_weight="balanced", random_state=0, max_depth=4, min_samples_split=6)),
        ("c_ccp", DecisionTreeClassifier(ccp_alpha=0.02, random_state=0)),
        ("c_mid", DecisionTreeClassifier(min_impurity_decrease=0.01, min_weight_fraction_leaf=0.02, random_state=2)),
        ("c_extra", ExtraTreeClassifier(random_state=7)),
    ]
    for name, est in cfgs:
        yy = y if name != "c_bal" else np.where(y == 2, 1, y)
        est.fit(X, yy)
        t = est.tree_
        outi(name + "_nodes", int(t.node_count))
        out(name + "_feat", t.feature)
        out(name + "_thr", t.threshold)
        out(name + "_imp", t.impurity)
        out(name + "_val", t.value)
        out(name + "_fi", est.feature_importances_)
        out(name + "_proba", est.predict_proba(Xt))
        out(name + "_apply", est.apply(Xt))
        outi(name + "_depth", int(est.get_depth()))
        outi(name + "_leaves", int(est.get_n_leaves()))
    rcfgs = [
        ("r_mse", DecisionTreeRegressor(random_state=0)),
        ("r_d4", DecisionTreeRegressor(max_depth=4, random_state=0, min_samples_leaf=3)),
        ("r_mae", DecisionTreeRegressor(criterion="absolute_error", max_depth=4, random_state=0)),
        ("r_pois", DecisionTreeRegressor(criterion="poisson", max_depth=4, random_state=0)),
        ("r_leaf", DecisionTreeRegressor(max_leaf_nodes=9, max_features=3, random_state=4)),
        ("r_rand", DecisionTreeRegressor(splitter="random", random_state=2, max_depth=6)),
        ("r_ccp", DecisionTreeRegressor(ccp_alpha=0.05, random_state=0)),
        ("r_extra", ExtraTreeRegressor(random_state=3, max_depth=5)),
        ("r_mono", DecisionTreeRegressor(monotonic_cst=[1, 0, -1, 0, 0], max_depth=4, random_state=0)),
    ]
    for name, est in rcfgs:
        est.fit(X, yp if name == "r_pois" else yr)
        t = est.tree_
        outi(name + "_nodes", int(t.node_count))
        out(name + "_feat", t.feature)
        out(name + "_thr", t.threshold)
        out(name + "_imp", t.impurity)
        out(name + "_val", t.value)
        out(name + "_fi", est.feature_importances_)
        out(name + "_pred", est.predict(Xt))
        outi(name + "_depth", int(est.get_depth()))
    sw = np.array([1.0 + (i % 4) * 0.5 for i in range(90)])
    est = DecisionTreeClassifier(random_state=0, max_depth=5).fit(X, y, sample_weight=sw)
    out("c_sw_thr", est.tree_.threshold); out("c_sw_proba", est.predict_proba(Xt))
    est = DecisionTreeRegressor(random_state=0, max_depth=5, criterion="absolute_error").fit(X, yr, sample_weight=sw)
    out("r_sw_thr", est.tree_.threshold); out("r_sw_pred", est.predict(Xt))
    est = DecisionTreeClassifier(random_state=0, max_depth=2).fit(X, y)
    out("dpath", est.decision_path(Xt[:5]).toarray())
    outi("etext", export_text(est))
    est = DecisionTreeRegressor(random_state=0, max_depth=2).fit(X, yr)
    outi("etext_r", export_text(est, feature_names=["a", "b", "c", "d", "e"], decimals=3))
    path = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X, y)
    out("ccp_alphas", path.ccp_alphas); out("ccp_imps", path.impurities)
    Xn = X.copy()
    for i in range(0, 90, 7):
        Xn[i, i % 5] = np.nan
    est = DecisionTreeClassifier(random_state=0, max_depth=4).fit(Xn, y)
    out("nan_thr", est.tree_.threshold); out("nan_mgl", est.tree_.missing_go_to_left)
    out("nan_pred", est.predict(Xn[:20]))
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out("cvs", cross_val_score(make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=0, max_depth=4)), X, y, cv=KFold(5)))

def ensemble_cases():
    from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier,
        ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier,
        AdaBoostRegressor, BaggingClassifier, BaggingRegressor, IsolationForest, VotingClassifier,
        VotingRegressor)
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    X, y = data(90, 5, 3)
    Xt, _ = data(20, 5, 3)
    Xt = Xt + 0.37
    yr = reg_target(X)
    yb = np.where(y == 2, 1, y)
    m = RandomForestClassifier(n_estimators=6, random_state=0, oob_score=True).fit(X, y)
    out("rfc_proba", m.predict_proba(Xt)); out("rfc_fi", m.feature_importances_); out("rfc_oob", [m.oob_score_])
    m = RandomForestClassifier(n_estimators=4, random_state=3, class_weight="balanced_subsample", max_samples=0.6, max_depth=4).fit(X, y)
    out("rfc2_proba", m.predict_proba(Xt))
    m = RandomForestRegressor(n_estimators=5, random_state=1, max_depth=5, oob_score=True).fit(X, yr)
    out("rfr_pred", m.predict(Xt)); out("rfr_fi", m.feature_importances_); out("rfr_oob", [m.oob_score_])
    m = ExtraTreesClassifier(n_estimators=5, random_state=2).fit(X, y)
    out("etc_proba", m.predict_proba(Xt)); out("etc_fi", m.feature_importances_)
    m = ExtraTreesRegressor(n_estimators=4, random_state=2, bootstrap=True, max_samples=0.5).fit(X, yr)
    out("etr_pred", m.predict(Xt))
    m = GradientBoostingClassifier(n_estimators=10, random_state=0, subsample=0.8, max_features=3).fit(X, yb)
    out("gbb_proba", m.predict_proba(Xt)); out("gbb_train", m.train_score_); out("gbb_oob", m.oob_scores_)
    m = GradientBoostingClassifier(loss="exponential", n_estimators=8, random_state=1).fit(X, yb)
    out("gbe_dec", m.decision_function(Xt))
    m = GradientBoostingClassifier(n_estimators=8, random_state=0, learning_rate=0.2).fit(X, y)
    out("gbm_proba", m.predict_proba(Xt)); out("gbm_fi", m.feature_importances_)
    out("gbm_staged", [p[0] for p in m.staged_predict(Xt[:1])])
    for loss in ["squared_error", "absolute_error", "huber", "quantile"]:
        m = GradientBoostingRegressor(loss=loss, n_estimators=10, random_state=1, subsample=0.7, max_depth=2).fit(X, yr)
        out("gbr_%s_pred" % loss, m.predict(Xt)); out("gbr_%s_fi" % loss, m.feature_importances_)
    m = AdaBoostClassifier(n_estimators=10, random_state=0).fit(X, y)
    out("abc_proba", m.predict_proba(Xt)); out("abc_w", m.estimator_weights_)
    m = AdaBoostRegressor(n_estimators=8, random_state=0, loss="square").fit(X, yr)
    out("abr_pred", m.predict(Xt))
    m = BaggingClassifier(n_estimators=5, random_state=0, max_samples=0.7, max_features=0.8).fit(X, y)
    out("bgc_proba", m.predict_proba(Xt))
    m = BaggingRegressor(n_estimators=5, random_state=2, bootstrap_features=True).fit(X, yr)
    out("bgr_pred", m.predict(Xt))
    m = IsolationForest(n_estimators=10, random_state=0).fit(X)
    out("iso_score", m.score_samples(Xt))
    m = IsolationForest(n_estimators=6, random_state=1, contamination=0.1, max_samples=40).fit(X)
    out("iso2_dec", m.decision_function(Xt))
    m = VotingClassifier([("a", DecisionTreeClassifier(random_state=0, max_depth=3)), ("b", RandomForestClassifier(n_estimators=4, random_state=0))], voting="soft", weights=[1, 2]).fit(X, y)
    out("vc_proba", m.predict_proba(Xt))
    m = VotingRegressor([("a", DecisionTreeRegressor(random_state=0, max_depth=3)), ("b", GradientBoostingRegressor(n_estimators=5, random_state=0))]).fit(X, yr)
    out("vr_pred", m.predict(Xt))
    from sklearn.model_selection import KFold, cross_val_score
    out("cvs_rf", cross_val_score(RandomForestClassifier(n_estimators=5, random_state=0), X, y, cv=KFold(4)))

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "tree"
    if which == "tree":
        tree_cases()
    else:
        ensemble_cases()
