# Generates test/mne/test_cov.py from templates/test_cov.tpl using reference
# values from real mne 1.13.2.
# Run: /path/to/venv/bin/python test/mne/gen_cov_expected.py
import os
import numpy as np
import mne

mne.set_log_level("ERROR")
HERE = os.path.dirname(os.path.abspath(__file__))


def lst(a):
    a = np.asarray(a, dtype=float).ravel()
    return "[" + ", ".join(repr(float(v)) for v in a) + "]"


def make_data(n_ep, n_t, seed):
    rng = np.random.RandomState(seed)
    mix = rng.randn(6, 6) * 0.3 + np.eye(6)
    Z = rng.randn(n_ep, 6, n_t)
    X = np.array([np.dot(mix, Z[e]) for e in range(n_ep)])
    X[:, :2] *= 1e-12   # mag
    X[:, 2:4] *= 1e-11  # grad
    X[:, 4:] *= 1e-5    # eeg
    return X


E = {}
names = ["M1", "M2", "G1", "G2", "E1", "E2"]
types = ["mag", "mag", "grad", "grad", "eeg", "eeg"]
info = mne.create_info(names, 100.0, types)
X = make_data(12, 30, 0)
ev = np.c_[np.arange(12) * 40, np.zeros(12, int), np.array([1, 2] * 6)]
ep = mne.EpochsArray(X.copy(), info, ev, tmin=-0.1, event_id={"a": 1, "b": 2}, baseline=(None, 0))
for m in ["empirical", "diagonal_fixed", "ledoit_wolf", "oas", "shrinkage", "shrunk"]:
    c = mne.compute_covariance(ep, method=m)
    E["c_" + m] = lst(c.data)
c = mne.compute_covariance(ep, keep_sample_mean=False)
E["c_nokeep"] = lst(c.data)
c = mne.compute_covariance(ep, tmin=-0.05, tmax=0.1)
E["c_tw"] = lst(c.data)
E["c_tw_nfree"] = lst([c["nfree"]])
cs = mne.compute_covariance(ep, method="auto", return_estimators=True)
E["auto_methods"] = repr([x["method"] for x in cs])
E["auto_logliks"] = lst([x["loglik"] for x in cs])
E["auto_best"] = lst(cs[0].data)
cs = mne.compute_covariance(ep, method=["empirical", "oas"], return_estimators=True, cv=4)
E["two_logliks"] = lst([x["loglik"] for x in cs])
# rank-deficient: average reference projection on eeg + dict params
ep2 = ep.copy().set_eeg_reference(projection=True)
c2 = mne.compute_covariance(ep2, method="ledoit_wolf")
E["c_proj_lw"] = lst(c2.data)
c3 = mne.compute_covariance(ep, method="shrinkage", method_params=dict(shrinkage=dict(shrinkage=0.3)))
E["c_shr03"] = lst(c3.data)
c4 = mne.compute_covariance(ep, method="diagonal_fixed", method_params=dict(diagonal_fixed=dict(eeg=0.2, mag=0.05)))
E["c_diagfix2"] = lst(c4.data)

raw = mne.io.RawArray(np.hstack(list(X)), info)
rc = mne.compute_raw_covariance(raw)
E["raw_emp"] = lst(rc.data)
E["raw_nfree"] = lst([rc["nfree"]])
rc = mne.compute_raw_covariance(raw, tmin=0.5, tmax=3.0, tstep=0.5, method="oas")
E["raw_oas"] = lst(rc.data)

ah = mne.make_ad_hoc_cov(info)
E["adhoc"] = lst(ah.data)
ah2 = mne.make_ad_hoc_cov(info, std=dict(eeg=1e-6, grad=1e-12, mag=1e-14))
E["adhoc2"] = lst(ah2.data)

emp = mne.compute_covariance(ep)
r = mne.cov.regularize(emp, info)
E["reg_default"] = lst(r.data)
r = mne.cov.regularize(emp, info, mag=0.2, grad=0.05, eeg=0.3, rank="full")
E["reg_full"] = lst(r.data)
info2 = ep2.info
emp2 = mne.compute_covariance(ep2)
r = mne.cov.regularize(emp2, info2, rank="full")
E["reg_full_proj"] = lst(r.data)
r = mne.cov.regularize(emp2, info2)
E["reg_proj"] = lst(r.data)

W, chn = mne.cov.compute_whitener(emp, info)
E["W"] = lst(W)
W, chn, rk, C = mne.cov.compute_whitener(emp2, info2, return_rank=True, return_colorer=True)
E["W2"] = lst(W)
E["W2_rank"] = lst([rk])
E["C2"] = lst(C)
W, _ = mne.cov.compute_whitener(emp2, info2, pca=True)
E["W2_pca"] = lst(np.abs(W))
W, _ = mne.cov.compute_whitener(ah, info)
E["W_adhoc"] = lst(W)
W, _ = mne.cov.compute_whitener(emp, info, rank=dict(eeg=1))
E["W_rank1"] = lst(np.abs(W))
evk = ep.average()
we = mne.whiten_evoked(evk, emp)
E["white_ev"] = lst(we.data[:, ::5])
s = emp + mne.compute_covariance(ep, tmax=0.0)
E["cov_add"] = lst(s.data)
E["cov_add_nfree"] = lst([s["nfree"]])

src = open(os.path.join(HERE, "templates", "test_cov.tpl")).read()
for k, v in E.items():
    src = src.replace("@@" + k + "@@", v)
assert "@@" not in src, src[src.index("@@"):src.index("@@") + 40]
open(os.path.join(HERE, "test_cov.py"), "w").write(src)
print("wrote test_cov.py")
