# Generates expected values for test_xdawn_ctps.py (mne 1.13.2 venv).
import numpy as np
import mne
from mne.preprocessing import Xdawn, create_ecg_epochs, create_eog_epochs, ICA
from mne.preprocessing.ctps_ import ctps
mne.set_log_level("error")

def L(v):
    return [float(x) for x in np.asarray(v, dtype=float).ravel()]

out = []
rng = np.random.RandomState(1)
n_ep, n_ch, n_t = 24, 4, 40
X = rng.randn(n_ep, n_ch, n_t)
y = np.array([1, 2] * 12)
tt = np.arange(n_t)
X[y == 1, 0] += 3 * np.sin(tt / 4.)
X[y == 2, 1] += 2 * np.cos(tt / 5.)
events = np.c_[np.arange(n_ep) * 100, np.zeros(n_ep, int), y]
info = mne.create_info(n_ch, 100., 'eeg')
ep = mne.EpochsArray(X, info, events, tmin=0., event_id={'a': 1, 'b': 2})
xd = Xdawn(n_components=2).fit(ep)
out.append(("xd_filters_a", L(np.abs(xd.filters_['a']))))
out.append(("xd_patterns_b", L(np.abs(xd.patterns_['b']))))
out.append(("xd_transform", L(np.abs(xd.transform(ep)[3, :, :5]))))
ap = xd.apply(ep)
out.append(("xd_apply_a", L(ap['a'].get_data()[2, :, :5])))
# overlapping epochs -> least-squares evokeds
events2 = np.c_[np.arange(n_ep) * 30, np.zeros(n_ep, int), y]
ep2 = mne.EpochsArray(X, info, events2, tmin=0., event_id={'a': 1, 'b': 2})
xd2 = Xdawn(n_components=1, reg=0.1).fit(ep2)
out.append(("xd2_overlap", [float(xd2.correct_overlap_)]))
out.append(("xd2_filters_b", L(np.abs(xd2.filters_['b']))))

ph = rng.rand(10, 2, 30)
ks, pk, _ = ctps(ph, is_raw=False)
out.append(("ctps_ks", L(ks[:, :6])))
out.append(("ctps_pk", L(pk[:, :6])))
x3 = rng.randn(12, 2, 64)
x3[:, 0] += np.sin(np.arange(64) / 3.)
ks, pk, _ = ctps(x3)
out.append(("ctps_raw_pk", L(pk[:, ::8])))

sf = 250.; n = 5000; rng = np.random.RandomState(3)
ecg = np.zeros(n)
for c in range(100, n, 227):
    ecg += np.exp(-0.5 * ((np.arange(n) - c) / 3.) ** 2)
ecg += 0.02 * rng.randn(n)
mix = rng.randn(3, 2)
eeg = mix @ np.vstack([ecg, rng.randn(n)]) * 1e-5 + rng.randn(3, n) * 1e-6
d2 = np.vstack([eeg, ecg[None] * 1e-4])
raw2 = mne.io.RawArray(d2, mne.create_info(['A', 'B', 'C', 'ECG'], sf, ['eeg'] * 3 + ['ecg']))
ee = create_ecg_epochs(raw2, tmin=-0.2, tmax=0.2)
out.append(("ecg_epochs_shape", [float(v) for v in ee.get_data().shape]))
out.append(("ecg_epochs_events", L(ee.events[:, 0])))
out.append(("ecg_epochs_val", L(ee.get_data()[1, :, 50])))
ica = ICA(n_components=2, random_state=0).fit(raw2)
inds, scores = ica.find_bads_ecg(raw2)
out.append(("ctps_inds", L(inds)))
out.append(("ctps_scores", L(scores)))

for name, vals in out:
    print(f"    ({name!r}, {vals!r}),")
