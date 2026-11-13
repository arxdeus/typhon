# Generates expected values for test_ica_picard.py (mne 1.13.2 + python-picard 0.8.2).
import numpy as np
import mne
import warnings
warnings.simplefilter("ignore")
mne.set_log_level("ERROR")
from mne.preprocessing import ICA

def make_raw():
    rng = np.random.RandomState(0)
    sf = 100.
    n = 2000
    t = np.arange(n) / sf
    s1 = np.sin(2 * np.pi * 1.3 * t)
    s2 = np.sign(np.sin(2 * np.pi * 0.7 * t))
    s3 = rng.laplace(size=n)
    blink = np.zeros(n)
    for c in [300, 800, 1300, 1700]:
        blink += 5 * np.exp(-0.5 * ((np.arange(n) - c) / 8.) ** 2)
    S = np.vstack([s1, s2, s3, blink])
    A = rng.randn(5, 4)
    X = A @ S + 0.01 * rng.randn(5, n)
    eog = blink + 0.05 * rng.randn(n)
    data = np.vstack([X, eog.reshape((1, n))]) * 1e-5
    info = mne.create_info(['E1', 'E2', 'E3', 'E4', 'E5', 'EOG'], sf, ['eeg'] * 5 + ['eog'])
    return mne.io.RawArray(data, info)

def L(v):
    return [float(x) for x in np.asarray(v, dtype=float).ravel()]

raw = make_raw()
out = []
cfgs = [("po", dict(), "tanh"),
        ("pstd", dict(ortho=False, extended=False), "tanh"),
        ("pext", dict(ortho=False, extended=True), "tanh"),
        ("pexp", dict(fun="exp"), "exp"),
        ("pcube", dict(fun="cube", ortho=False, extended=False), "cube")]
for name, fp, _ in cfgs:
    ica = ICA(n_components=4, random_state=42, method="picard", fit_params=fp, max_iter=300)
    ica.fit(raw)
    out.append((name + "_meta", [float(ica.n_iter_), float(ica.n_components_)]))
    out.append((name + "_unmixing", L(ica.unmixing_matrix_)))
    if name == "po":
        out.append((name + "_sources", L(ica.get_sources(raw).get_data()[:, :5])))
        ev = ica.get_explained_variance_ratio(raw)
        out.append(("po_evr", [float(ev["eeg"])]))
        ev1 = ica.get_explained_variance_ratio(raw, components=[1, 2])
        out.append(("po_evr12", [float(ev1["eeg"])]))
        out.append(("po_score", L(ica.score_sources(raw, target="EOG"))))
ica = ICA(n_components=4, random_state=7, method="picard", fit_params=dict(fastica_it=5, tol=1e-6)).fit(raw)
out.append(("pfi_unmixing", L(ica.unmixing_matrix_)))
out.append(("pfi_niter", [float(ica.n_iter_)]))
for name, vals in out:
    print(f"    ({name!r}, {vals!r}),")
