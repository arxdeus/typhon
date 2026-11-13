"""Prints expected values for test_logpower_lda.py (run with the reference venv)."""
from functools import partial
import numpy as np, mne
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import FunctionTransformer

def make_data(n=60):
    X = np.zeros((n, 6, 500)); y = np.arange(n) % 3
    for e in range(n):
        for c in range(6):
            t = np.arange(500)
            amp10 = 8.0 + 6.0 * ((y[e] + c) % 3)
            amp20 = 5.0 + 4.0 * ((2 * y[e] + c) % 3)
            X[e, c] = (amp10 * np.sin(2 * np.pi * 10.0 / 125.0 * t + 0.3 * e)
                       + amp20 * np.sin(2 * np.pi * 20.0 / 125.0 * t + 0.17 * e * (c + 1))
                       + 3.0 * np.sin(0.91 * t * (c + 1) + 0.37 * e) + 2.0 * np.cos(2.3 * t + e * c))
    return X, y

def tp(x):
    p = np.mean(np.square(np.asarray(x, dtype=np.float64)), axis=-1)
    return np.log(np.maximum(p, np.finfo(np.float64).tiny))

def build():
    volts = FunctionTransformer(partial(np.multiply, 1e-6), validate=False)
    br = []
    for nm, lo, hi in (('a', 8.0, 13.0), ('b', 13.0, 30.0)):
        bp = FunctionTransformer(partial(mne.filter.filter_data, sfreq=125.0, l_freq=lo, h_freq=hi,
            method='iir', iir_params=dict(order=4, ftype='butter', output='sos'),
            phase='zero', copy=True, verbose=False), validate=False)
        br.append((nm, make_pipeline(bp, FunctionTransformer(tp, validate=False))))
    return make_pipeline(volts, FeatureUnion(br), LDA())

def fmt(a): return "[" + ", ".join(repr(float(v)) for v in np.asarray(a).ravel()) + "]"

if __name__ == "__main__":
    X, y = make_data()
    m = build(); m.fit(X, y)
    F = m[:-1].transform(X)
    lda = m[-1]
    print("F0 =", fmt(F[0])); print("F7 =", fmt(F[7]))
    print("COEF =", fmt(lda.coef_)); print("ICPT =", fmt(lda.intercept_))
    print("PRED =", list(map(int, m.predict(X))))
    print("PROBA =", fmt(m.predict_proba(X)[:3])); print("PROBA_LAST =", fmt(m.predict_proba(X)[-2:]))
