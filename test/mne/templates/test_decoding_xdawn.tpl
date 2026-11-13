import numpy as np
import mne
from mne.decoding import XdawnTransformer, CSP, SpatialFilter, get_spatial_filter_from_estimator, TimeFrequency

#EXPECTED#

def check(name: str, got, rtol: float = 1e-7, atol: float = 1e-10):
    want = List[float]()
    for n, v in EXPECTED:
        if n == name:
            want = v
    g = np.asarray(got, dtype=float).ravel()
    ok = g.shape[0] == len(want)
    if ok:
        for i in range(len(want)):
            if abs(g[i] - want[i]) > atol + rtol * abs(want[i]):
                ok = False
    if not ok:
        print("TEST FAILED:", name, g.shape[0], len(want))

rng = np.random.RandomState(3)
n_ep = 30
n_ch = 5
n_t = 60
y = np.arange(n_ep) % 3
X = rng.randn(n_ep, n_ch, n_t)
mix = rng.randn(n_ch, n_ch)
for e in range(n_ep):
    X[e] = mix @ X[e]
    X[e, :, 10:20] += (y[e] + 1) * np.outer(rng.randn(n_ch) * 0 + np.arange(1, n_ch + 1), np.hanning(10))

t = XdawnTransformer().fit(X, y)
check('x1_f', t.filters_); check('x1_p', t.patterns_); check('x1_e', t.evals_)
Z = t.transform(X)
check('x1_t', Z[:2]); check('x1_inv', t.inverse_transform(Z)[:2])
t = XdawnTransformer(n_components=3, reg=0.1).fit(X, y)
check('x2_f', t.filters_); check('x2_p', t.patterns_); check('x2_e', t.evals_)
Z = t.transform(X)
check('x2_t', Z[:2]); check('x2_inv', t.inverse_transform(Z)[:2])
t = XdawnTransformer(n_components=2, restr_type='restricting').fit(X, y)
check('x3_f', t.filters_); check('x3_p', t.patterns_); check('x3_e', t.evals_)
Z = t.transform(X)
check('x3_t', Z[:2]); check('x3_inv', t.inverse_transform(Z)[:2])
check('xnoy_f', XdawnTransformer(2).fit(X).filters_)

y2 = np.arange(n_ep) % 2
info = mne.create_info(n_ch, 100., 'eeg')
csp = CSP(n_components=3, reg=None, log=True).fit(X, y2)
sf = get_spatial_filter_from_estimator(csp, info)
check('sf_f', sf.filters); check('sf_p', sf.patterns); check('sf_e', sf.evals.__val__())
sf2 = SpatialFilter(info, csp.filters_[:3])
check('sf2_p', sf2.patterns)

print("done")
