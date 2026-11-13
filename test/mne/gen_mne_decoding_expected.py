# Generates test/mne/test_decoding.py and test_decoding_sl.py from templates/*.tpl using
# reference values from real mne 1.13.2 / scikit-learn 1.9.1.
# Run: /path/to/venv/bin/python test/mne/gen_mne_decoding_expected.py
import os
import numpy as np
import mne
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from mne.decoding import (Scaler, Vectorizer, PSDEstimator, FilterEstimator, TemporalFilter,
                          UnsupervisedSpatialFilter, SPoC, EMS, compute_ems, LinearModel,
                          get_coef, SlidingEstimator, GeneralizingEstimator,
                          cross_val_multiscore)

mne.set_log_level("ERROR")
HERE = os.path.dirname(os.path.abspath(__file__))


def lst(a):
    a = np.asarray(a, dtype=float).ravel()
    return "[" + ", ".join(repr(float(v)) for v in a) + "]"


def make(n_ep=24, n_ch=4, n_t=40):
    e = np.arange(n_ep)[:, None, None]
    c = np.arange(n_ch)[None, :, None]
    t = np.arange(n_t)[None, None, :]
    y = np.arange(n_ep) % 2
    X = (np.sin(0.31 * t * (c + 1) + 0.17 * e) + 0.6 * np.cos(0.13 * t * (e % 5 + 1) + c)
         + (y[:, None, None] * 0.9) * np.sin(0.5 * t) * (c % 2 == 0) + 0.1 * c)
    return X, y


E = {}
X, y = make()
info = mne.create_info(["a", "b", "c", "d"], 100.0, ["eeg", "eeg", "mag", "mag"])
for nm, sc in [("sc_n", None), ("sc_m", "mean"), ("sc_md", "median")]:
    s = Scaler(info=info if sc is None else None, scalings=sc)
    Z = s.fit_transform(X * 1e-6, y)
    E[nm] = lst(Z[:3, :, ::7])
    E[nm + "_i"] = lst(s.inverse_transform(Z)[:2, :, ::9])
s = Scaler(info=info, scalings=dict(eeg=2e5, mag=3e14))
E["sc_d"] = lst(s.fit_transform(X * 1e-6, y)[:2, :, ::9])
s = Scaler(scalings="mean", with_mean=False)
E["sc_nm"] = lst(s.fit_transform(X, y)[:2, :, ::9])
v = Vectorizer()
V = v.fit_transform(X)
E["vec"] = lst(V[:2, ::11]); E["vec_shape"] = str(list(V.shape))
E["vec_i"] = lst(v.inverse_transform(V)[1, :, ::13])
E["psd"] = lst(PSDEstimator(sfreq=100.0, fmin=2, fmax=30).fit_transform(X)[:2, :, ::3])
E["psdb"] = lst(PSDEstimator(sfreq=100.0, bandwidth=8.0, adaptive=True).fit_transform(X)[:1, :, ::5])
info2 = mne.create_info(4, 100.0, "eeg")
XL, _ = make(6, 4, 400)
fe = FilterEstimator(info2, 5.0, 30.0)
E["fe"] = lst(fe.fit(XL, np.arange(6) % 2).transform(XL)[:2, :, ::37])
E["tf"] = lst(TemporalFilter(l_freq=None, h_freq=20.0, sfreq=100.0).fit_transform(XL)[:2, :, ::37])
usf = UnsupervisedSpatialFilter(PCA(3), average=False)
U = usf.fit_transform(X)
E["usf"] = lst(np.abs(U[:2, :, ::9]))
usf2 = UnsupervisedSpatialFilter(PCA(2), average=True)
E["usfa"] = lst(np.abs(usf2.fit_transform(X)[:2, :, ::9]))
# SPoC (continuous target)
yc = np.sin(np.arange(24) * 0.7) + 0.1 * np.arange(24)
Xs = X * (1.0 + 0.5 * yc[:, None, None] * (np.arange(4)[None, :, None] == 1))
sp = SPoC(n_components=2)
F = sp.fit_transform(Xs, yc)
E["spoc_f"] = lst(F[:5]); E["spoc_ev"] = lst(sp.evals_)
E["spoc_fl"] = lst(np.sort(np.abs(sp.filters_), axis=1))
sp2 = SPoC(n_components=2, log=False, reg=0.1)
E["spoc2"] = lst(sp2.fit_transform(Xs, yc)[:5])
# EMS
ems = EMS()
T = ems.fit_transform(X, y)
E["ems_f"] = lst(ems.filters_[:, ::7]); E["ems_t"] = lst(T[:3, ::7])
ev = np.column_stack([np.arange(24) * 50, np.zeros(24, int), y + 1])
ep = mne.EpochsArray(X * 1e-6, info2, events=ev, event_id=dict(a=1, b=2))
sur, filt, cond = compute_ems(ep)
E["cems_s"] = lst(sur[:4, ::5]); E["cems_f"] = lst(filt[:, ::9])
# LinearModel / get_coef
Xf = X.reshape(24, -1)[:, :20]
lm = LinearModel(LogisticRegression(solver="liblinear", random_state=0))
lm.fit(Xf, y)
E["lm_f"] = lst(lm.filters_); E["lm_p"] = lst(lm.patterns_)
pipe = make_pipeline(StandardScaler(), LinearModel(Ridge(alpha=1.0)))
pipe.fit(Xf, yc)
E["gc_p"] = lst(get_coef(pipe, "patterns_", inverse_transform=True))
E["gc_f"] = lst(get_coef(pipe, "filters_", inverse_transform=False))
# Sliding / Generalizing
sl = SlidingEstimator(LogisticRegression(solver="liblinear", random_state=0))
sl.fit(X[:, :, :10], y)
E["sl_pred"] = lst(sl.predict(X[:4, :, :10]))
E["sl_df"] = lst(sl.decision_function(X[:4, :, :10]))
E["sl_sc"] = lst(sl.score(X[:, :, :10], y))
sl2 = SlidingEstimator(LogisticRegression(solver="liblinear", random_state=0), scoring="roc_auc")
sl2.fit(X[:, :, :10], y)
E["sl_auc"] = lst(sl2.score(X[:, :, :10], y))
gl = GeneralizingEstimator(LogisticRegression(solver="liblinear", random_state=0))
gl.fit(X[:, :, :6], y)
E["gl_sc"] = lst(gl.score(X[:, :, :6], y))
E["gl_df"] = lst(gl.decision_function(X[:3, :, :6]))
cvs = cross_val_multiscore(SlidingEstimator(LogisticRegression(solver="liblinear", random_state=0)), X[:, :, :8], y, cv=3)
E["cvm_sl"] = lst(cvs)
cvl = cross_val_multiscore(LogisticRegression(solver="liblinear", random_state=0), Xf, y, cv=StratifiedKFold(4), scoring="roc_auc")
E["cvm_lr"] = lst(cvl)

for name in ["test_decoding", "test_decoding_spoc", "test_decoding_lm", "test_decoding_sl", "test_decoding_gl"]:
    src = open(os.path.join(HERE, "templates", name + ".tpl")).read()
    for k, vv in E.items():
        src = src.replace("@@" + k + "@@", vv)
    assert "@@" not in src, src[src.index("@@"):src.index("@@") + 40]
    open(os.path.join(HERE, name + ".py"), "w").write(src)
    print("wrote", name)
