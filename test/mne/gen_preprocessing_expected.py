# Generates expected values for test_preprocessing.py (mne 1.13.2 venv).
import numpy as np
import mne
from mne.preprocessing import (ICA, find_eog_events, find_ecg_events, peak_finder,
                               regress_artifact, infomax, annotate_muscle_zscore)
mne.set_log_level("error")

def make_raw():
    rng = np.random.RandomState(0)
    sf = 100.; n = 2000; t = np.arange(n) / sf
    s1 = np.sin(2 * np.pi * 1.3 * t); s2 = np.sign(np.sin(2 * np.pi * 0.7 * t)); s3 = rng.laplace(size=n)
    blink = np.zeros(n)
    for c in [300, 800, 1300, 1700]:
        blink += 5 * np.exp(-0.5 * ((np.arange(n) - c) / 8.) ** 2)
    S = np.vstack([s1, s2, s3, blink])
    A = rng.randn(5, 4)
    X = A @ S + 0.01 * rng.randn(5, n)
    eog = blink + 0.05 * rng.randn(n)
    data = np.vstack([X, eog[None]]) * 1e-5
    info = mne.create_info(['E1', 'E2', 'E3', 'E4', 'E5', 'EOG'], sf, ['eeg'] * 5 + ['eog'])
    return mne.io.RawArray(data, info)

def L(v):
    return [float(x) for x in np.asarray(v, dtype=float).ravel()]

out = []
raw = make_raw()
ica = ICA(n_components=4, random_state=42, method='fastica')
ica.fit(raw)
out.append(("n_iter", [float(ica.n_iter_), float(ica.n_components_)]))
out.append(("pre_whitener", L(ica.pre_whitener_)))
out.append(("pca_ev", L(ica.pca_explained_variance_)))
out.append(("unmixing", L(ica.unmixing_matrix_)))
out.append(("components", L(ica.get_components())))
out.append(("sources_head", L(ica.get_sources(raw).get_data()[:, :5])))
inds, scores = ica.find_bads_eog(raw, measure='correlation', threshold=0.5)
out.append(("eog_corr_inds", L(inds)))
out.append(("eog_scores", L(scores)))
inds2, _ = ica.find_bads_eog(raw)
out.append(("eog_z_inds", L(inds2)))
r2 = ica.apply(raw.copy(), exclude=[3])
out.append(("apply_col300", L(r2.get_data()[:, 300])))
out.append(("apply_col1000", L(r2.get_data()[:, 1000])))
ev = find_eog_events(raw)
out.append(("eog_events", L(ev[:, 0])))

ica2 = ICA(n_components=0.95, random_state=42, method='infomax')
ica2.fit(raw)
out.append(("infomax_ncomp", [float(ica2.n_components_), float(ica2.n_iter_)]))
out.append(("infomax_unmixing", L(ica2.unmixing_matrix_)))

ep = mne.EpochsArray(raw.get_data().reshape(6, 10, 200).transpose(1, 0, 2).copy(), raw.info)
ica3 = ICA(n_components=3, random_state=7, method='fastica').fit(ep)
out.append(("epochs_unmixing", L(ica3.unmixing_matrix_)))
out.append(("epochs_sources", L(ica3.get_sources(ep).get_data()[2, :, :4])))

x = np.sin(np.linspace(0, 6 * np.pi, 200)) + np.linspace(0, 1, 200)
pi, pm = peak_finder(x)
out.append(("peak_inds", L(pi)))
out.append(("peak_mags", L(pm)))
pi, pm = peak_finder(x, thresh=0.1, extrema=-1)
out.append(("valley_inds", L(pi)))
out.append(("valley_mags", L(pm)))

raw_r = raw.copy().set_eeg_reference(projection=True).apply_proj()
raw_c, betas = regress_artifact(raw_r)
out.append(("betas", L(betas)))
out.append(("regressed_col300", L(raw_c.get_data()[:, 300])))

# ECG: 1.1 Hz spike train at 250 Hz
sf = 250.; n = 5000; rng = np.random.RandomState(3)
ecg = np.zeros(n)
for c in range(100, n, 227):
    ecg += np.exp(-0.5 * ((np.arange(n) - c) / 3.) ** 2)
ecg += 0.02 * rng.randn(n)
d2 = np.vstack([rng.randn(2, n) * 1e-6, ecg[None] * 1e-4])
info2 = mne.create_info(['A', 'B', 'ECG'], sf, ['eeg', 'eeg', 'ecg'])
raw2 = mne.io.RawArray(d2, info2)
ev, idx, pulse = find_ecg_events(raw2)
out.append(("ecg_events", L(ev[:, 0])))
out.append(("ecg_pulse", [float(pulse), float(idx)]))

# muscle: 500 Hz, burst of 125 Hz activity
sf = 500.; n = 3000; rng = np.random.RandomState(5)
d3 = rng.randn(3, n) * 1e-6
tt = np.arange(n) / sf
d3[:, 1200:1500] += 2e-5 * np.sin(2 * np.pi * 125 * tt[1200:1500])
raw3 = mne.io.RawArray(d3, mne.create_info(3, sf, 'eeg'))
ann, sc = annotate_muscle_zscore(raw3, threshold=3.0)
out.append(("muscle_onset", L(ann.onset)))
out.append(("muscle_duration", L(ann.duration)))
out.append(("muscle_scores", L(sc[::250])))

for name, vals in out:
    print(f"    ({name!r}, {vals!r}),")
