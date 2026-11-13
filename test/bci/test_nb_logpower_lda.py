# Ported-from: bci-research-lab notebooks (synthetic 3-class data); expected values from the reference libs
import numpy as np
from mne.filter import filter_data, IIRParams
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

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
    E['feat'] = (288, [-21.52391684425836, -27.119871003642487, -27.777254026733303, -21.53061146019906, -32.348384938134686, -32.8775876043745])
    E['coef'] = (36, [-164.33554530106895, -315.1640187102606, 84.78808741498203, 180.18126992901455, 155.87813508495353, -115.48178350258767])
    E['pred'] = (24, [0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
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
def test_logpower_lda():
    load()
    
    def logp(X):
        p = np.mean(np.square(X), axis=-1)
        return np.log(np.maximum(p, np.finfo(np.float64).tiny))
    
    X, y = make_X()
    V = X * 1e-6
    feats = []
    for lo, hi in [(8.0, 13.0), (13.0, 30.0)]:
        F = filter_data(V, 125.0, lo, hi, method='iir', iir_params=IIRParams(order=4, ftype="butter", output="sos"), phase='zero', copy=True, verbose=False)
        feats.append(logp(F))
    Z = np.hstack(feats)
    show('feat', Z)
    lda = LDA().fit(Z, y)
    show('coef', lda.coef_)
    show('pred', lda.predict(Z))
    print('score', float(lda.score(Z, y)))

test_logpower_lda()
