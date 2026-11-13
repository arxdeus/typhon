import numpy as np
import mne
from mne.filter import IIRParams, filter_data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import FunctionTransformer
from functools import partial

# expected values from gen_logpower_lda.py (reference numpy/scipy/mne/sklearn)
F0 = [-24.182576521007174, -23.067601587620008, -22.354136967262026, -24.18656681522035, -23.065289819305598, -22.354483721928535, -24.812092096651206, -23.93845264848234, -23.202461701604765, -25.114396667163188, -23.936145964123895, -23.15347269190785]
F7 = [-23.04256450938206, -22.343390544147294, -24.1766835851653, -23.07010115929212, -22.349956534170417, -24.16646972731116, -23.13470551384674, -25.108159740187688, -23.938678066154104, -23.186402000083444, -25.049371657582963, -23.834027851641935]
COEF = [-18726.298331726677, -2748.161099543408, 10144.833894986301, -21260.5124070004, -1230.5027253627868, 10128.079467239128, -3714.668578165809, -878.4647123137654, 11362.05962919272, -14023.48307624249, -4434.9615892423135, 2069.7248580812156, 3146.5955931473422, 5800.280693877945, -9367.56416557403, 6732.388577315256, 5555.829317755008, -8519.53834825436, 1697.741567734225, -9797.792609418111, -2714.379050175711, 8299.807904752495, -2667.9012547801585, 1088.0192270451664, 15579.702738579314, -3052.119594334545, -777.2697294122518, 14528.123829685112, -4325.326592392227, -1608.5411189847482, 2016.9270104315783, 10676.257321731886, -8647.680579016993, 5723.675171489971, 7102.862844022469, -3157.744085126381]
ICPT = [-823166.2528620649, -46162.69385988792, 771907.2886064502]
PRED = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
P0 = [1.0, 0.0, 0.0]

def make_data(n: int = 60):
    X = np.zeros((n, 6, 500))
    y = np.arange(n) % 3
    for e in range(n):
        for c in range(6):
            for t in range(500):
                a10 = 8.0 + 6.0 * float((y[e] + c) % 3)
                a20 = 5.0 + 4.0 * float((2 * y[e] + c) % 3)
                X[e, c, t] = (a10 * np.sin(2 * np.pi * 10.0 / 125.0 * t + 0.3 * e)
                              + a20 * np.sin(2 * np.pi * 20.0 / 125.0 * t + 0.17 * e * (c + 1))
                              + 3.0 * np.sin(0.91 * t * (c + 1) + 0.37 * e) + 2.0 * np.cos(2.3 * t + e * c))
    return X, y

def two_band_log_power(X):
    X = np.asarray(X, dtype=np.float64)
    power = np.mean(np.square(X), axis=-1)
    return np.log(np.maximum(power, np.finfo(np.float64).tiny))

def make_two_band_lda(sfreq: float):
    volts = FunctionTransformer(partial(np.multiply, 1e-6), validate=False)
    branches = []
    for name, lo, hi in (('band_8_13', 8.0, 13.0), ('band_13_30', 13.0, 30.0)):
        bandpass = FunctionTransformer(
            partial(filter_data, sfreq=sfreq, l_freq=lo, h_freq=hi, method='iir',
                    iir_params=IIRParams(order=4, ftype='butter', output='sos'),
                    phase='zero', copy=True, verbose=False), validate=False)
        branches.append((name, make_pipeline(bandpass, FunctionTransformer(two_band_log_power, validate=False))))
    return make_pipeline(volts, FeatureUnion(branches), LDA())

def relclose(a, b, rtol: float) -> bool:
    x = np.asarray(a).ravel()
    for i in range(x.shape[0]):
        if abs(x[i] - b[i]) > rtol * max(abs(b[i]), 1e-12):
            print("mismatch", i, x[i], b[i])
            return False
    return True

X, y = make_data()
m = make_two_band_lda(125.0)
m.fit(X, y)
F = m.head().transform(X)
assert F.shape == (60, 12)
assert relclose(F[0], F0, 1e-10)
assert relclose(F[7], F7, 1e-10)
lda = m[-1]
assert relclose(lda.coef_, COEF, 1e-8)
assert relclose(lda.intercept_, ICPT, 1e-8)
pred = m.predict(X)
for i in range(60):
    assert int(pred[i]) == PRED[i]
assert relclose(m.predict_proba(X)[0], P0, 1e-8)
print("ok")
