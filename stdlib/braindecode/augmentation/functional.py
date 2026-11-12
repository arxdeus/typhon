# Ported from braindecode 2.0.0 (BSD-3-Clause)
#
# Functional augmentations on batches X: ndarray[float32, 3] (batch, ch, time)
# and y: ndarray[int, 1]. Random draws go through the legacy numpy
# RandomState in the same order as braindecode, so outputs match it.

import numpy as np
from mne.filter import notch_filter
from .._random_state import RandomState, check_random_state

F32 = np.float32


def _sig32(x: float) -> F32:
    # torch.sigmoid on float32
    v = F32(x)
    return F32(1.0) / (F32(1.0) + F32(np.exp(float(-v))))


def identity(X, y):
    return X, y


def time_reverse(X, y):
    return np.ascontiguousarray(X[:, :, ::-1]), y


def sign_flip(X, y):
    return -X, y


def ft_surrogate(X, y, phase_noise_magnitude: float, channel_indep: bool, random_state = None):
    if not (0 <= phase_noise_magnitude <= 1):
        raise AssertionError(f"eps must be a float between 0 and 1. Got {phase_noise_magnitude}.")
    rng = check_random_state(random_state)
    B, C, n = X.shape
    f = np.fft.fft(X.astype(float), axis=-1)
    cc = C if channel_indep else 1
    half = (n - 1) // 2 if n % 2 == 1 else n // 2 - 1
    rp = rng.random((B, cc, half)) * (2.0 * np.pi)
    phase = np.zeros((B, cc, n))
    for b in range(B):
        for c in range(cc):
            for k in range(half):
                phase[b, c, 1 + k] = rp[b, c, k]
                phase[b, c, n - 1 - k] = -rp[b, c, k]
    out = np.empty((B, C, n), dtype=F32)
    for b in range(B):
        for c in range(C):
            pc = c if channel_indep else 0
            row = np.empty(n, dtype=complex)
            for k in range(n):
                row[k] = f[b, c, k] * np.exp(complex(0.0, phase_noise_magnitude * phase[b, pc, k]))
            r = np.fft.ifft(row)
            for k in range(n):
                out[b, c, k] = F32(r[k].real)
    return out, y


def _pick_channels_randomly(X, p_pick: float, rng: RandomState):
    B, C, _ = X.shape
    u = rng.uniform(0, 1, size=(B, C))
    m = np.empty((B, C), dtype=F32)
    for b in range(B):
        for c in range(C):
            m[b, c] = _sig32(float(F32(1000) * (F32(u[b, c]) - F32(p_pick))))
    return m


def channels_dropout(X, y, p_drop: float, random_state = None):
    rng = check_random_state(random_state)
    m = _pick_channels_randomly(X, p_drop, rng)
    out = X.copy()
    B, C, T = X.shape
    for b in range(B):
        for c in range(C):
            for t in range(T):
                out[b, c, t] = X[b, c, t] * m[b, c]
    return out, y


def channels_shuffle(X, y, p_shuffle: float, random_state = None):
    if p_shuffle == 0:
        return X, y
    rng = check_random_state(random_state)
    m = _pick_channels_randomly(X, 1 - p_shuffle, rng)
    B, C, T = X.shape
    out = X.copy()
    for b in range(B):
        sel = [c for c in range(C) if np.round(float(m[b, c])) != 0.0]
        perm = list(range(C))
        ps = rng.permutation(np.array(sel, dtype=int))
        for k in range(len(sel)):
            perm[sel[k]] = int(ps[k])
        for c in range(C):
            out[b, c, :] = X[b, perm[c], :]
    return out, y


def gaussian_noise(X, y, std: float, random_state = None):
    rng = check_random_state(random_state)
    noise = rng.normal(loc=np.zeros(X.shape), scale=1)
    out = X.copy()
    for i in range(X.size):
        out.data[i] = X.data[i] + F32(noise.data[i]) * F32(std)
    return out, y


def channels_permute(X, y, permutation: List[int]):
    return X[:, np.array(permutation, dtype=int), :].copy(), y


def smooth_time_mask(X, y, mask_start_per_sample, mask_len_samples: int):
    B, C, T = X.shape
    s = F32(1000.0 / T)
    out = X.copy()
    for b in range(B):
        ms = float(mask_start_per_sample[b])
        for t in range(T):
            # torch: float32 t minus float64 tensor -> float64 then sigmoid in float64
            a = 1.0 / (1.0 + np.exp(-(float(s) * -(t - ms))))
            c = 1.0 / (1.0 + np.exp(-(float(s) * (t - ms - mask_len_samples))))
            mk = F32(a + c)
            for ch in range(C):
                out[b, ch, t] = X[b, ch, t] * mk
    return out, y


def bandstop_filter(X, y, sfreq: float, bandwidth: float, freqs_to_notch):
    if bandwidth == 0:
        return X, y
    out = X.copy()
    for b in range(X.shape[0]):
        s = X[b].astype(float)
        r = notch_filter(s, Fs=sfreq, freqs=float(freqs_to_notch[b]), method="fir",
                         notch_widths=bandwidth)
        out[b] = r.astype(F32)
    return out, y


def _analytic(x):
    N = x.shape[-1]
    f = np.fft.fft(x, axis=-1)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = 1.0
        h[N // 2] = 1.0
        for k in range(1, N // 2):
            h[k] = 2.0
    else:
        h[0] = 1.0
        for k in range(1, (N + 1) // 2):
            h[k] = 2.0
    return np.fft.ifft(f * h, axis=-1)


def frequency_shift(X, y, delta_freq, sfreq: float):
    """delta_freq: one shift per example (sequence of floats)."""
    B, C, N = X.shape
    Np = 1 << int(np.ceil(np.log2(float(N))))
    pad = np.zeros((B, C, Np), dtype=F32)
    pad[:, :, :N] = X
    # torch computes the analytic signal in complex64 for float32 input
    an = _analytic(pad.astype(float))
    out = np.empty((B, C, N), dtype=F32)
    for b in range(B):
        fs = float(F32(float(delta_freq[b])))
        for c in range(C):
            for t in range(N):
                tt = float(F32(float(t) / sfreq))
                ph = complex(0.0, 2.0 * np.pi) * fs * tt
                out[b, c, t] = F32((an[b, c, t] * np.exp(ph)).real)
    return out, y


def mixup(X, y, lam, idx_perm):
    B, C, T = X.shape
    out = np.empty((B, C, T), dtype=F32)
    for b in range(B):
        l = F32(float(lam[b]))
        j = int(idx_perm[b])
        for c in range(C):
            for t in range(T):
                out[b, c, t] = l * X[b, c, t] + (F32(1) - l) * X[j, c, t]
    yb = np.array([y[int(idx_perm[b])] for b in range(B)])
    lam32 = np.array([F32(float(lam[b])) for b in range(B)], dtype=F32)
    return out, (y.copy(), yb, lam32)
