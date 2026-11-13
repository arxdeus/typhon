# Ported-from: bci-research-lab notebooks (synthetic 3-class data); expected values from the reference libs
import numpy as np
from mne.filter import construct_iir_filter, IIRParams
from mne.time_frequency import psd_array_welch
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import sosfilt, sosfilt_zi
from sklearn.utils.class_weight import compute_sample_weight

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
    E['sos'] = (24, [0.031044876645809256, 0.06208975329161851, 0.031044876645809256, 1.0, -0.4445863351728787, 0.20695688918803046])
    E['sosfilt'] = (3000, [3.3881317890172014e-21, 8.923446902934856e-07, 4.46874069076617e-06, 8.121325831639072e-06, 3.5720405102032265e-06, -1.0117074988871857e-05])
    E['psd'] = (5760, [3.379642378418631e-11, 2.1332830972610045e-11, 2.0087289819048823e-13, 7.094371617424877e-15, 2.2216162205026167e-15, 4.639950060606209e-12])
    E['freqs'] = (40, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    E['sw'] = (20, [0.9523809523809523, 0.9523809523809523, 1.1111111111111112, 0.9523809523809523, 0.9523809523809523, 1.1111111111111112])
    E['ptp'] = (437, [0.00010476768821173358, 0.00010476768821173358, 0.00010476768821173358, 0.00010476768821173358, 9.958561363554135e-05, 9.225871728743939e-05])
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
def test_iir_psd_misc():
    load()
    
    X, y = make_X()
    V = X * 1e-6
    p = construct_iir_filter(IIRParams(order=4, ftype="butter", output="sos"), [8.0, 30.0], None, 125.0, "bandpass", return_copy=False)
    sos = p['sos']
    show('sos', sos)
    x = V[0]
    zi = sosfilt_zi(sos)[:, None, :] * x[None, :, :1]
    f, _ = sosfilt(sos, x, axis=-1, zi=zi)
    show('sosfilt', f)
    psd, freqs = psd_array_welch(V, 125.0, fmin=1.0, fmax=40.0, n_fft=125, verbose=False)
    show('psd', psd); show('freqs', freqs)
    show('sw', compute_sample_weight('balanced', y[:20]))
    show('ptp', np.ptp(sliding_window_view(V[0, 0], 64), axis=-1))

test_iir_psd_misc()
