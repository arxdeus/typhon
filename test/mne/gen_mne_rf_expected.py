# Generates test/mne/test_decoding_rf.py from templates/test_decoding_rf.tpl using
# reference values from real mne 1.13.2 / scikit-learn 1.9.1.
# Run: /path/to/venv/bin/python test/mne/gen_mne_rf_expected.py
import os
import numpy as np
import mne
from sklearn.linear_model import Ridge
from mne.decoding import ReceptiveField, TimeDelayingRidge

mne.set_log_level("ERROR")
HERE = os.path.dirname(os.path.abspath(__file__))


def lst(a):
    a = np.asarray(a, dtype=float).ravel()
    return "[" + ", ".join(repr(float(v)) for v in a) + "]"


def stim(n_t, n_f):
    t = np.arange(n_t)[:, None]
    f = np.arange(n_f)[None, :]
    return np.sin(0.37 * t * (f + 1) + f) + 0.5 * np.cos(0.11 * t + 2.0 * f)


E = {}
X = stim(200, 3)
w = np.array([[0.5, -0.2, 0.1, 0.3], [0.0, 0.4, -0.3, 0.2], [0.1, 0.1, 0.2, -0.5]])
y = np.zeros(200)
for f in range(3):
    y += np.convolve(X[:, f], w[f])[:200]
y += 0.05 * np.sin(np.arange(200) * 1.3)
rf = ReceptiveField(-0.01, 0.02, 100.0, estimator=1.0, patterns=True)
rf.fit(X, y)
E["rf_coef"] = lst(rf.coef_); E["rf_pat"] = lst(rf.patterns_)
E["rf_pred"] = lst(rf.predict(X)[::17]); E["rf_score"] = lst(rf.score(X, y))
E["rf_delays"] = lst(rf.delays_)
rfc = ReceptiveField(0.0, 0.03, 100.0, estimator=0.5, scoring="corrcoef", fit_intercept=False)
rfc.fit(X, y)
E["rfc_coef"] = lst(rfc.coef_); E["rfc_score"] = lst(rfc.score(X, y))
rfr = ReceptiveField(-0.02, 0.01, 100.0, estimator=Ridge(alpha=2.0), patterns=True)
rfr.fit(X, y)
E["rfr_coef"] = lst(rfr.coef_); E["rfr_pat"] = lst(rfr.patterns_)
E["rfr_pred"] = lst(rfr.predict(X)[::19]); E["rfr_score"] = lst(rfr.score(X, y))
# multi-epoch, multi-output, laplacian reg
X3 = np.stack([stim(120, 2) * (1 + 0.1 * e) for e in range(3)], axis=1)  # (120, 3, 2)
Y3 = np.stack([X3[:, :, 0] * 0.7 - 0.3 * np.roll(X3[:, :, 1], 1, axis=0), X3[:, :, 1] * 0.2], axis=2)
tdr = TimeDelayingRidge(0.0, 0.02, 100.0, alpha=3.0, reg_type="laplacian")
tdr.fit(X3, Y3)
E["tdr_coef"] = lst(tdr.coef_); E["tdr_int"] = lst(tdr.intercept_)
E["tdr_pred"] = lst(tdr.predict(X3)[::13])
tdr2 = TimeDelayingRidge(-0.02, 0.01, 100.0, alpha=1.0, reg_type=("ridge", "laplacian"), fit_intercept=False)
tdr2.fit(X3, Y3)
E["tdr2_coef"] = lst(tdr2.coef_)

src = open(os.path.join(HERE, "templates", "test_decoding_rf.tpl")).read()
for k, v in E.items():
    src = src.replace("@@" + k + "@@", v)
assert "@@" not in src, src[src.index("@@"):src.index("@@") + 40]
open(os.path.join(HERE, "test_decoding_rf.py"), "w").write(src)
print("wrote test_decoding_rf.py with", len(E), "values")
