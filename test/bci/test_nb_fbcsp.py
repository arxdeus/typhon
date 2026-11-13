# Ported-from: bci-research-lab notebooks (synthetic 3-class data); expected values from the reference libs
import numpy as np
from mne.decoding import CSP, TemporalFilter
from mne.filter import IIRParams
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer

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
    E['fb_feat'] = (288, [1.4105423530710555, -0.9094693303668179, -0.9741893795061146, -1.554616256368809, 1.3317728735264596, -0.8669181937950977])
    E['fb_coef'] = (36, [601.9326636379836, -71.56788577592626, -230.22800666918644, -83.77315241952098, 238.36359988856069, 46.227431161987546])
    E['fb_pred'] = (24, [0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    S['fb_score'] = 1.0
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
def test_fbcsp():
    load()
    
    X, y = make_X()
    V = X * 1e-6
    branches = []
    for i, (lo, hi) in enumerate([(8.0, 13.0), (13.0, 20.0), (20.0, 30.0)]):
        branches.append((f'band_{i}', make_pipeline(
            FunctionTransformer(np.copy, validate=False),
            TemporalFilter(l_freq=lo, h_freq=hi, sfreq=125.0, method='iir', iir_params=IIRParams(order=4, ftype="butter", output="sos")),
            CSP())))
    m = make_pipeline(FeatureUnion(branches), StandardScaler(), LDA()).fit(V, y)
    show('fb_feat', m.steps[1][1].transform(m.steps[0][1].transform(V))); show('fb_coef', m.steps[2][1].coef_)
    show('fb_pred', m.predict(V)); score('fb_score', float(m.score(V, y)))

test_fbcsp()
