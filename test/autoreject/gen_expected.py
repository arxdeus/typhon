# Generates the expected values of test_autoreject.py from autoreject 0.5.2
# (run with the reference venv; rewrites the BEGIN/END EXPECTED block).
import os
import warnings
import numpy as np
import mne
warnings.simplefilter("ignore")
mne.set_log_level("error")
from autoreject import (AutoReject, Ransac, compute_thresholds, get_rejection_threshold,
                        validation_curve, _GlobalAutoReject)
from autoreject.bayesopt import _GaussianProcess, expected_improvement, bayes_opt
from autoreject.utils import clean_by_interp

NAMES = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3',
         'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9',
         'O1', 'Oz', 'O2', 'PO10', 'AF7', 'AF8']

def make_data(n_ep, n_ch, n_t):
    X = np.zeros((n_ep, n_ch, n_t))
    for e in range(n_ep):
        for c in range(n_ch):
            for t in range(n_t):
                v = (e * 1009 + c * 7919 + t * 104729 + e * c * t * 13 + t * t * (c + 1) * 3) % 101
                X[e, c, t] = ((v - 50.0) / 25.0 + np.sin(0.3 * t + 0.1 * c)) * 1e-5
    # artifacts: some epochs with many bad channels, some with few
    X[3] *= 8.0
    X[11, :20] *= 6.0
    X[17, 4] *= 12.0
    X[17, 9] *= 10.0
    X[25, 2:6] *= 9.0
    X[30, 13] += 3e-4
    X[34, 0:3] *= 7.0
    X[:, 21] *= 1.0
    return X

def make_epochs(X):
    info = mne.create_info(NAMES, 100., 'eeg')
    ep = mne.EpochsArray(X, info, verbose=False)
    ep.set_montage('standard_1020')
    return ep

def lit(a):
    return repr([float(x) for x in np.asarray(a, dtype=float).ravel()])

out = []
def add(name, val):
    out.append(f"E_{name} = {lit(val)}")

X = make_data(40, 32, 60)
ep = make_epochs(X)

# bayesopt
gp = _GaussianProcess().fit(np.array([[0.1], [0.5], [1.3], [2.0]]), np.array([1.0, 0.2, 0.7, 1.5]))
m, s = gp.predict(np.array([[0.3], [1.0], [1.7]]), return_std=True)
add("gp_mean", m); add("gp_std", s)
add("ei", expected_improvement(gp, 0.2, np.array([0.3, 1.0, 1.7])))
f = lambda x: (x - 0.73) ** 2
bx, bf = bayes_opt(f, np.array([0.0, 0.5, 1.0, 2.0]), np.linspace(0, 2, 41), expected_improvement, max_iter=10)
add("bo", [bx, bf])

# clean_by_interp
ci = clean_by_interp(ep, verbose=False).get_data()
add("cbi", ci[[0, 3, 17]][:, [0, 4, 21, 31]][:, :, :5])

# compute_thresholds
th = compute_thresholds(ep, random_state=42, verbose=False)
add("th", [th[n] for n in NAMES])
th2 = compute_thresholds(ep, random_state=7, augment=False, picks=list(range(10)), verbose=False)
add("th2", [th2[n] for n in NAMES[:10]])
th3 = compute_thresholds(ep, method='random_search', random_state=3, picks=[0, 1, 2], verbose=False)
add("th3", [th3[n] for n in NAMES[:3]])

# get_rejection_threshold
rj = get_rejection_threshold(ep, verbose=False)
add("rj", [rj['eeg']])
rj2 = get_rejection_threshold(ep, decim=2, cv=3, verbose=False)
add("rj2", [rj2['eeg']])

# validation_curve
tr, te, pr = validation_curve(ep, param_range=None, cv=4, return_param_range=True)
add("vc_tr", tr); add("vc_te", te); add("vc_pr", pr)

# _GlobalAutoReject
g = _GlobalAutoReject(32, 60, thresh=2e-4).fit(X.reshape(40, -1))
add("gar", [g.score(X.reshape(40, -1))])

# AutoReject
ar = AutoReject(n_interpolate=np.array([1, 4, 8]), consensus=np.linspace(0, 1, 6),
                cv=4, random_state=42, verbose=False)
ar.fit(ep)
add("ar_th", [ar.threshes_[n] for n in NAMES])
add("ar_ci", [ar.consensus_['eeg'], ar.n_interpolate_['eeg']])
add("ar_loss", np.where(np.isinf(ar.loss_['eeg']), -1.0, ar.loss_['eeg']))
clean, log = ar.transform(ep, return_log=True)
add("ar_bad", log.bad_epochs)
add("ar_labels", log.labels)
add("ar_clean_shape", list(clean.get_data().shape))
add("ar_clean", clean.get_data()[:, [2, 4, 9, 13]][:, :, :4])
add("ar_mean", ar.local_reject_['eeg'].mean_[[0, 9, 31]][:, :6])

# _AutoReject directly
from autoreject import _AutoReject
la = _AutoReject(n_interpolate=4, consensus=0.3, picks=list(range(16)), verbose=False)
la.thresh_func = __import__('functools').partial(la.thresh_func, random_state=5)
la.fit(ep)
llog = la.get_reject_log(ep)
add("la_bad", llog.bad_epochs)
add("la_labels", np.nan_to_num(llog.labels, nan=-1.0))
add("la_mean", la.mean_[[0, 15]][:, :5])

# Ransac
Xr = make_data(40, 32, 60)[:20].copy()
Xr[:, 7] = np.cos(np.arange(60) * 1.7)[None, :] * 3e-5 + Xr[:, 7] * 0.2
Xr[:, 15] = np.roll(Xr[:, 15], 7, axis=-1) * 4.0
epr = make_epochs(Xr)
rs = Ransac(n_resample=20, min_channels=0.25, min_corr=0.75, unbroken_time=0.4,
            random_state=435656, verbose=False)
rs.fit(epr)
out.append(f"E_rs_bad = {rs.bad_chs_!r}")
add("rs_corr", rs.corr_[:3])
add("rs_sub0", [NAMES.index(c) for c in rs.ch_subsets_[0]])
rt = rs.transform(epr).get_data()
add("rs_tr", rt[[0, 5]][:, [7, 15]][:, :, :5])
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_autoreject.py")
src = open(path).read()
head = src.split("# BEGIN EXPECTED\n")[0]
tail = src.split("# END EXPECTED\n")[1]
body = "\n".join(out).replace("-inf", "-1e999").replace("inf", "1e999")
open(path, "w").write(head + "# BEGIN EXPECTED\n" + body + "\n# END EXPECTED\n" + tail)
