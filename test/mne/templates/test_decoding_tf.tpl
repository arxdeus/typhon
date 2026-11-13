import numpy as np
import mne
from mne.decoding import TimeFrequency

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

tf = TimeFrequency(freqs=[8., 12., 20.], sfreq=100., method='morlet', n_cycles=3., output='power')
check('tf_pow', tf.fit_transform(X[:3]).real)
tf = TimeFrequency(freqs=[8., 12.], sfreq=100., method='morlet', n_cycles=3., output='complex', decim=2)
r = tf.fit_transform(X[:2])
check('tf_re', r.real); check('tf_im', r.imag)
tf = TimeFrequency(freqs=[8., 12.], sfreq=100., method='multitaper', n_cycles=3., output='power', time_bandwidth=4.)
check('tf_mt', tf.fit_transform(X[:2]).real)
print("done")
