# Ported from mne 1.13.2 / numpy 2.5.4 (BSD-3-Clause): user log_power_lda_model/preprocessing.py
# (included byte-for-byte between the markers below, guarded by check_verbatim.py). Expected values from the reference venv.
# --- BEGIN verbatim preprocessing.py (bci-research-lab d617e8b) ---
"""Stateless Butterworth/log-power preprocessing from two_band_logpower.ipynb.

Input: complete EEG trials (N, 6, 500) in microvolts, sampled at 125 Hz.
Output: float64 log-power features (N, 12), ready for the LDA ONNX model.
Channel order: 2, 3, 1, 10, 7, 13 (Fz, F4, F3, C3, Cz, C4).
"""
from __future__ import annotations

import mne
import numpy as np

TWO_BAND_RANGES = (
    ('band_8_13', 8.0, 13.0),
    ('band_13_30', 13.0, 30.0),
)


def butterworth_filter(X, low_hz, high_hz, sfreq=125.0):
    """Apply the notebook's fourth-order zero-phase SOS bandpass to volts."""
    return mne.filter.filter_data(
        X,
        sfreq=sfreq, l_freq=low_hz, h_freq=high_hz,
        method='iir',
        iir_params=dict(order=4, ftype='butter', output='sos'),
        phase='zero', copy=True, verbose=False,
    )


def two_band_log_power(X):
    """Compute the exact notebook log(mean(X**2)) along the time axis."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError('Ожидается массив (пробы, каналы, отсчёты).')
    power = np.mean(np.square(X), axis=-1)
    return np.log(np.maximum(power, np.finfo(np.float64).tiny))


def preprocess(X):
    """Convert raw microvolt trials to 12 features without fitting parameters.

    The first six features are 8–13 Hz, followed by six at 13–30 Hz.
    Filtering uses the complete four-second trial, as in the notebook.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3 or X.shape[0] == 0 or X.shape[1:] != (6, 500):
        raise ValueError('Ожидается непустой массив (N, 6, 500).')
    if not np.isfinite(X).all():
        raise ValueError('Сигнал должен содержать конечные значения в мкВ.')
    volts = np.multiply(1e-6, X)
    return np.concatenate([
        two_band_log_power(butterworth_filter(volts, low_hz, high_hz))
        for _, low_hz, high_hz in TWO_BAND_RANGES
    ], axis=1)
# --- END verbatim ---


F0 = [-23.748217065105933, -23.556119017579345, -23.39139185592984, -23.222906306340054, -23.105822060294443, -22.965396201809828, -25.557345561913333, -25.55816574252022, -25.335760012454074, -25.536076423278846, -25.54549139783193, -25.543744456632755]
F1 = [-23.73598166335382, -23.545760907239917, -23.372462405188372, -23.19620517927191, -23.063149750410627, -22.921954505039704, -25.111001356541003, -25.109345648435205, -24.952619057689848, -25.09681705484916, -25.098358457826887, -25.098195400728308]
F2 = [-23.73296812203022, -23.546439430090558, -23.373008329918672, -23.168893784722812, -23.05784624795592, -22.915441505356085, -24.73426324182871, -24.730003953657683, -24.61433809416145, -24.720623132845734, -24.71332573515926, -24.711840511610138]

def make(n: int):
    X = np.zeros((n, 6, 500))
    t = np.arange(500)
    for e in range(n):
        for c in range(6):
            X[e, c] = ((10.0 + c) * np.sin(2 * np.pi * (9.0 + e) / 125.0 * t + 0.4 * c)
                       + (4.0 + e) * np.sin(2 * np.pi * 21.0 / 125.0 * t + 0.1 * e * c)
                       + 2.0 * np.cos(1.7 * t * (c + 1) + e))
    return X

def maxrel(a, b) -> float:
    m = 0.0
    for j in range(12):
        m = max(m, abs(a[j] - b[j]) / abs(b[j]))
    return m

X = make(3)
F = preprocess(X)
print(F.shape)  # EXPECT: (3, 12)
print(maxrel(F[0], F0) < 1e-14, maxrel(F[1], F1) < 1e-14, maxrel(F[2], F2) < 1e-14)  # EXPECT: True True True
F1x = preprocess(X[1:2])
print(F1x.shape, maxrel(F1x[0], F1) < 1e-14)  # EXPECT: (1, 12) True

def err(X) -> str:
    try:
        preprocess(X)
        return "ok"
    except ValueError as e:
        return str(e)

print(err(np.zeros((1, 5, 500))))  # EXPECT: Ожидается непустой массив (N, 6, 500).
print(err(np.zeros((0, 6, 500))))  # EXPECT: Ожидается непустой массив (N, 6, 500).
Xn = X.copy()
Xn[2, 4, 10] = np.nan
print(err(Xn))  # EXPECT: Сигнал должен содержать конечные значения в мкВ.
Xn[2, 4, 10] = np.inf
print(err(Xn))  # EXPECT: Сигнал должен содержать конечные значения в мкВ.
