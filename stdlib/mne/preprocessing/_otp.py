# Ported from mne 1.13.2 (BSD-3-Clause)
#
# oversampled_temporal_projection (leave-one-out temporal projection) with
# constant-overlap-add (hann) windowing.

import math
import numpy as np
from .._info import _picks_to_idx
from .._containers import RawArray


def _svd_basis(cov: np.ndarray[float, 2], data: np.ndarray[float, 2]) -> np.ndarray[float, 2]:
    w, u = np.linalg.eigh(cov)
    n = w.shape[0]
    last = w[n - 1]
    eps = 2.220446049250313e-16
    keep = List[int]()
    nrm = np.ones(n)
    s = np.sqrt(np.maximum(w, 0.0))
    for i in range(n):
        if w[i] > eps * last:
            keep.append(i)
            nrm[i] = 1.0 / s[i]
    un = u * nrm
    ut = un.T
    v = np.zeros((len(keep), data.shape[1]))
    for k in range(len(keep)):
        v[k, :] = ut[keep[k], :] @ data
    return v


def _otp(data: np.ndarray[float, 2], good: List[int], bad: List[int]) -> np.ndarray[float, 2]:
    for v in data.flat:
        if not np.isfinite(v):
            raise RuntimeError("non-finite data (inf or nan) found in raw instance")
    nc, nt = data.shape
    means = np.zeros(nc)
    for c in range(nc):
        means[c] = data[c].mean()
        data[c] -= means[c]
    ng = len(good)
    dg = np.zeros((ng, nt))
    norms = np.zeros(ng)
    for i in range(ng):
        r = data[good[i]]
        nr = float(np.sqrt((r * r).sum()))
        norms[i] = nr
        if nr > 0:
            dg[i] = r / nr
        else:
            dg[i] = r
    cov = dg @ dg.T
    full = np.zeros((0, nt))
    if len(bad) > 0:
        full = _svd_basis(cov, dg)
    out = data.copy()
    for mi in range(ng):
        idx = [j for j in range(ng) if j != mi]
        sub_cov = cov[np.array(idx)][:, np.array(idx)]
        sub_dat = dg[np.array(idx)]
        tb = _svd_basis(sub_cov, sub_dat)
        x = (dg[mi] @ tb.T) @ tb
        x = x * norms[mi]
        out[good[mi]] = x + means[good[mi]]
    for p in bad:
        out[p] = (data[p] @ full.T) @ full + means[p]
    return out


def oversampled_temporal_projection(raw, duration = 10.0, picks = None, verbose = None):
    """mne.preprocessing.oversampled_temporal_projection."""
    pk = _picks_to_idx(raw.info, picks, "all", ())
    good = List[int]()
    bad = List[int]()
    gi = List[int]()
    bi = List[int]()
    for ii in range(len(pk)):
        if raw.info.ch_names[pk[ii]] in raw.info.bads:
            bad.append(ii)
            bi.append(pk[ii])
        else:
            good.append(ii)
            gi.append(pk[ii])
    n = int(np.round(float(duration) * raw.info.sfreq))
    if n < len(good) - 1:
        raise ValueError(f"duration ({float(n) / raw.info.sfreq}) yielded {n} samples, which is fewer than the number of channels -1 ({len(good) - 1})")
    out = raw.copy()
    nt = out._data.shape[1]
    ov = n // 2
    step = n - ov
    if nt < n:
        raise ValueError(f"Number of samples per window ({n}) must be at most the total number of samples ({nt})")
    # hann window (scipy get_window, fftbins when n even)
    win = np.zeros(n)
    den = float(n) if (n - 1) % 2 == 1 else float(n - 1)
    for k in range(n):
        win[k] = 0.5 - 0.5 * math.cos(2.0 * math.pi * float(k) / den)
    # COLA normalisation: median of binsums
    nb = n // step
    bins = np.zeros(step)
    for ii in range(nb):
        bins += win[ii * step:(ii + 1) * step]
    if n % step != 0:
        r = n % step
        bins[:r] += win[n - r:]
    const = float(np.median(bins))
    win = win / const
    starts = List[int]()
    s0 = 0
    while s0 <= nt - n:
        starts.append(s0)
        s0 += step
    acc = np.zeros(out._data.shape)
    np_ = len(pk)
    pka = np.array(pk)
    for si in range(len(starts)):
        start = starts[si]
        stop = start + n
        if si == len(starts) - 1:
            stop = nt
        ln = stop - start
        w = np.zeros(ln)
        w[:n] = win
        if si == len(starts) - 1:
            sh = step
            while sh < ln:
                nu = ln - sh
                w[sh:] += win[:nu]
                sh += step
        if si == 0:
            nu = n - step
            while nu > 0:
                w[:nu] += win[n - nu:]
                nu -= step
        seg = out._data[pka][:, start:stop].copy()
        res = _otp(seg, good, bad)
        for r in range(np_):
            acc[pk[r], start:stop] += res[r] * w
    for r in range(np_):
        out._data[pk[r]] = acc[pk[r]]
    return out

