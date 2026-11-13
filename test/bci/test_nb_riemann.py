# Ported-from: bci-research-lab notebooks (synthetic 3-class data); expected values from the reference libs
import numpy as np
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
    E['mdm_pred'] = (24, [0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    E['mdm_proba'] = (72, [0.9753133453665269, 0.0013615415229668772, 0.02332511311050609, 0.001511171837874398, 0.997289784687761, 0.0011990434743645766])
    E['ts_feat'] = (504, [0.5252302967023795, 0.03799607722935276, 0.02580088975760864, 0.7887314170858271, 0.030288851390829274, 0.017677552462149568])
    E['ts_lda_coef'] = (63, [78118960.97100998, -8768848.776765328, 7067903.248029784, 54315460.24374528, 2260864.266252902, -1250477.7084375338])
    E['ts_svc_dec'] = (72, [2.2222365555352703, -0.17479167256129519, 0.8423002852155199, -0.1801327233203186, 2.225557391542861, 0.8405458589974772])
    E['ts_svc_pred'] = (24, [0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    E['ts_lr_coef'] = (63, [0.48253413502363873, -0.044671812856628704, -0.05091718537664128, 0.4758643661596065, -0.05793563236834383, -0.13339032614032187])
    E['ts_lr_pred'] = (24, [0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    S['mdm_score'] = 1.0
    S['ts_lda_score'] = 1.0
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
def test_riemann():
    load()
    
    X, y = make_X()
    V = X * 1e-6
    m = make_pipeline(Covariances(), MDM())
    m.fit(V, y)
    show('mdm_pred', m.predict(V)); score('mdm_score', float(m.score(V, y)))
    show('mdm_proba', m.predict_proba(V))
    ts = make_pipeline(Covariances(estimator='lwf'), TangentSpace())
    F = ts.fit_transform(V, y)
    show('ts_feat', F)
    m = make_pipeline(Covariances(), TangentSpace(), StandardScaler(), LDA()).fit(V, y)
    show('ts_lda_coef', m[-1].coef_); score('ts_lda_score', float(m.score(V, y)))
    m = make_pipeline(Covariances(), TangentSpace(), StandardScaler(), SVC()).fit(V, y)
    show('ts_svc_dec', m.decision_function(V)); show('ts_svc_pred', m.predict(V))
    m = make_pipeline(Covariances(), TangentSpace(), StandardScaler(), LogisticRegression()).fit(V, y)
    show('ts_lr_coef', m[-1].coef_); show('ts_lr_pred', m.predict(V))

test_riemann()
