# Ported from mne 1.13.2 (BSD-3-Clause)
#
# apply_pca_obs: PCA optimal basis sets removal of the heartbeat artefact.

import math
import numpy as np
from scipy.interpolate import PchipInterpolator
from .._info import _picks_to_idx
from .._containers import RawArray


def _fit_ecg_template(data: np.ndarray[float, 1], pca_template: np.ndarray[float, 2], a_peak_idx: int,
                      peak_range: int, pre_range: int, post_range: int, mid_p: int,
                      fitted_art: np.ndarray[float, 1], post_idx_prev: int, has_prev: bool,
                      n_samples_fit: int) -> int:
    n_t = 2 * peak_range + 1
    template = pca_template[0:n_t, :]
    seg = data[a_peak_idx - peak_range:a_peak_idx + peak_range + 1]
    det = seg - float(seg.mean())
    coef = np.linalg.lstsq(template, det, rcond=None)[0]
    pad_fit = np.dot(template, coef)
    s0 = a_peak_idx - pre_range - 1
    for k in range(pre_range + post_range + 1):
        fitted_art[s0 + k] = pad_fit[mid_p - pre_range - 1 + k]
    if not has_prev:
        return a_peak_idx + post_range
    w0 = post_idx_prev
    w1 = a_peak_idx - pre_range
    if w0 < w1:
        nx = w1 - w0 + 1
        x_interpol = np.empty(nx)
        for k in range(nx):
            x_interpol[k] = float(w0 + k)
        n1 = n_samples_fit + 1
        xf = List[float]()
        yf = List[float]()
        for k in range(n1):
            xf.append(float(w0 - n_samples_fit + k))
            yf.append(fitted_art[w0 - n_samples_fit + k])
        for k in range(n1):
            xf.append(float(w1 + k))
            yf.append(fitted_art[w1 + k])
        xfa = np.array(xf)
        yfa = np.array(yf)
        yi = PchipInterpolator(xfa, yfa)(x_interpol)
        for k in range(nx):
            fitted_art[post_idx_prev + k] = yi[k]
    return a_peak_idx + post_range


def _pca_obs_row(data0: np.ndarray[float, 1], qrs: np.ndarray[int, 1], n_components: int) -> np.ndarray[float, 1]:
    data = data0 - float(data0.mean())
    n = data.shape[0]
    fitted_art = np.zeros(n)
    pk = List[int]()
    for q in qrs:
        if q < n:
            pk.append(int(q))
    peak_count = len(pk)
    diffs = np.empty(peak_count - 1)
    for i in range(peak_count - 1):
        diffs[i] = float(pk[i + 1] - pk[i])
    mRR = float(np.median(diffs))
    peak_range = int(round(mRR / 2.0))
    mid_p = peak_range + 1
    n_samples_fit = int(round(peak_range / 8.0))
    while pk[peak_count - 1] + peak_range > n:
        peak_count -= 1
    L = 2 * peak_range + 1
    pcamat = np.zeros((peak_count - 1, L))
    for p in range(1, peak_count):
        row = data[pk[p] - peak_range:pk[p] + peak_range + 1]
        pcamat[p - 1, :] = row - float(row.mean())
    # constant detrend along axis 1 (already zero-mean per row)
    for p in range(pcamat.shape[0]):
        pcamat[p, :] = pcamat[p, :] - float(pcamat[p, :].mean())
    mean_effect = np.empty(L)
    for t in range(L):
        s = 0.0
        for p in range(pcamat.shape[0]):
            s += pcamat[p, t]
        mean_effect[t] = s / pcamat.shape[0]
    dp = pcamat.copy()
    for p in range(dp.shape[0]):
        dp[p, :] = dp[p, :] - float(dp[p, :].mean())
    # _PCA.fit: center columns, SVD, svd_flip (u based)
    nsmp = dp.shape[0]
    X = dp.copy()
    for t in range(L):
        mu = 0.0
        for p in range(nsmp):
            mu += X[p, t]
        mu /= nsmp
        for p in range(nsmp):
            X[p, t] -= mu
    U, S, V = np.linalg.svd(X, full_matrices=False)
    k = S.shape[0]
    for j in range(k):
        best = 0
        for i in range(U.shape[0]):
            if abs(U[i, j]) > abs(U[best, j]):
                best = i
        sg = 0.0
        if U[best, j] > 0.0:
            sg = 1.0
        elif U[best, j] < 0.0:
            sg = -1.0
        for i in range(U.shape[0]):
            U[i, j] *= sg
        for i in range(V.shape[1]):
            V[j, i] *= sg
    nc = min(n_components, k)
    pca_template = np.zeros((L, 1 + n_components))
    for t in range(L):
        pca_template[t, 0] = mean_effect[t]
    for j in range(nc):
        ev = S[j] * S[j] / float(nsmp - 1)
        sq = math.sqrt(ev)
        for t in range(L):
            pca_template[t, 1 + j] = V[j, t] * sq
    post_prev = 0
    has_prev = False
    for p in range(peak_count):
        if pk[p] - peak_range < 0:
            continue
        pre_range = peak_range
        post_range = peak_range
        if p == 0:
            post_range = (pk[p + 1] - pk[p]) // 2
            if post_range > peak_range:
                post_range = peak_range
        elif p == peak_count - 1:
            pre_range = (pk[p] - pk[p - 1]) // 2
            if pre_range > peak_range:
                pre_range = peak_range
        else:
            pre_range = (pk[p] - pk[p - 1]) // 2
            post_range = (pk[p + 1] - pk[p]) // 2
            if pre_range >= peak_range:
                pre_range = peak_range
            if post_range > peak_range:
                post_range = peak_range
        nxt = _fit_ecg_template(data, pca_template, pk[p], peak_range, pre_range, post_range, mid_p,
                                fitted_art, post_prev, has_prev, n_samples_fit)
        if p != peak_count - 1:
            post_prev = nxt
            has_prev = True
    return data - fitted_art


def apply_pca_obs(raw, picks, *, qrs_times, n_components: int = 4, n_jobs = None, copy: bool = True,
                  verbose = None):
    """Remove the heartbeat artefact with PCA optimal basis sets (mne 1.13.2)."""
    q = np.asarray(qrs_times, dtype=float)
    if q.ndim > 1:
        raise ValueError("qrs_times must be a 1d array")
    for v in q:
        if v < 0.0:
            raise ValueError("qrs_times must be strictly positive")
    out = raw.copy() if copy else raw
    qi = np.empty(q.shape[0], dtype=int)
    for i in range(q.shape[0]):
        qi[i] = int(q[i] * raw.info.sfreq)
    pk = _picks_to_idx(out.info, picks, "all", ())
    for ci in pk:
        row = out._data[ci, :].copy()
        out._data[ci, :] = _pca_obs_row(row, qi, n_components)
    return out
