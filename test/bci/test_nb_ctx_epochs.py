# Ported-from: bci-research-lab notebooks (synthetic 3-class data); expected values from the reference libs
import numpy as np
import mne

def make_X(n_ep: int = 24, n_ch: int = 6, n_t: int = 500):
    X = np.zeros((n_ep, n_ch, n_t))
    y = np.arange(n_ep) % 3
    for e in range(n_ep):
        for c in range(n_ch):
            for t in range(n_t):
                v = (np.sin(0.37 * t * (c + 1) + 0.11 * e) + 0.5 * np.cos(0.071 * t * (e % 7 + 1) + c)
                     + 0.3 * np.sin(1.3 * t + 0.7 * e * c))
                if c % 3 == int(y[e]):
                    v += 1.5 * np.sin(2 * np.pi * 11.0 / 125.0 * t + e)
                if c == 5 - int(y[e]):
                    v += 1.2 * np.sin(2 * np.pi * 20.0 / 125.0 * t + 0.3 * e)
                X[e, c, t] = 20.0 * v + 3.0 * c
    return X, y


E = Dict[str, Tuple[int, List[float]]]()
S = Dict[str, float]()
def load():
    E['m'] = (18, [1.8215932524393879, 3.6139134664791204, 5.606321064495172, 0.9900396732166388, 3.1164264680417775, 5.398018173871952])
    E['raw'] = (300, [9.999999999999999e-06, 3.8743702237062674e-05, 5.3293358485410335e-05, 5.3445396381060916e-05, 4.826332180486869e-05, 4.093642545676673e-05])
    E['ep'] = (1800, [9.999999999999999e-06, 3.8743702237062674e-05, 5.3293358485410335e-05, 5.3445396381060916e-05, 4.826332180486869e-05, 4.093642545676673e-05])
    pass

def show(name: str, a):
    x = np.asarray(a, dtype=float).ravel()
    n, ex = E[name]
    assert x.shape[0] == n
    scale = 1e-300
    for i in range(len(ex)):
        scale = max(scale, abs(ex[i]))
    for i in range(len(ex)):
        if abs(x[i] - ex[i]) > 1e-6 * max(abs(ex[i]), 1e-3 * scale):
            print("mismatch", name, i, x[i], ex[i])
            assert False

def score(name: str, v: float):
    if abs(v - S[name]) > 1e-9:
        print("mismatch", name, v, S[name])
        assert False

@test
def test_ctx_epochs():
    load()
    X, y = make_X(6, 3, 100)
    with mne.use_log_level('WARNING'):
            show('m', np.mean(X, axis=-1))
    info = mne.create_info(['a', 'b', 'c'], sfreq=125.0, ch_types='eeg')
    raw = mne.io.RawArray(X[0] * 1e-6, info, verbose=False)
    show('raw', raw.get_data())
    ep = mne.EpochsArray(X * 1e-6, info, verbose=False)
    ep2 = mne.concatenate_epochs([ep[:3], ep[3:]])
    show('ep', ep2.get_data()); print(len(ep2))

test_ctx_epochs()
