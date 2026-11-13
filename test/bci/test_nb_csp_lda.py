# Ported-from: bci-research-lab notebooks (synthetic 3-class data); expected values from the reference libs
import numpy as np
from mne.decoding import CSP, Scaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    E['csp_feat'] = (96, [-0.7662128365700108, 0.6946012373349828, -0.4903955525754925, -0.18837031367106383, 0.7249315175201534, -0.8448628348720881])
    E['csp_coef'] = (12, [-3954.0538970117514, 6475.288566201736, 1463.607426633202, -421.7748136592811, 7893.6743283732885, -6374.6196021127125])
    E['csp_pred'] = (24, [0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    E['sc_feat'] = (96, [-0.8887282049999177, 0.7124540716042896, -0.4875500361946879, -0.21865277026532645, 0.7968482297589516, -0.9306241399917068])
    E['sc_coef'] = (12, [-85974.48108440508, 30595.514632451093, -6885.3672174293315, -2934.497045701117, 149207.33402132997, -36584.8070321651])
    E['sc_pred'] = (24, [0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    S['csp_score'] = 1.0
    S['sc_score'] = 1.0
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
def test_csp_lda():
    load()
    
    X, y = make_X()
    V = X * 1e-6
    m1 = make_pipeline(CSP(n_components=4), StandardScaler(), LDA())
    m1.fit(V, y)
    show('csp_feat', m1.steps[0][1].transform(V))
    show('csp_coef', m1.steps[2][1].coef_)
    show('csp_pred', m1.predict(V))
    score('csp_score', float(m1.score(V, y)))
    m2 = make_pipeline(Scaler(scalings="mean"), CSP(n_components=4), LDA())
    m2.fit(V, y)
    show('sc_feat', m2.steps[1][1].transform(m2.steps[0][1].transform(V)))
    show('sc_coef', m2.steps[2][1].coef_)
    show('sc_pred', m2.predict(V))
    score('sc_score', float(m2.score(V, y)))

test_csp_lda()
