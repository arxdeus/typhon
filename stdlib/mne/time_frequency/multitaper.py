# Ported from mne 1.13.2 (BSD-3-Clause)

import math
import numpy as np
from scipy.signal.windows import dpss as _sp_dpss
from scipy.signal import get_window as _sp_get_window
from scipy.fft import rfft as _rfft, rfftfreq as _rfftfreq


def dpss_windows(N: int, half_nbw: float, Kmax: int, *, sym: bool = True, norm = None,
                 low_bias: bool = True):
    dp = np.ones((1, 1))
    ev = np.ones(1)
    if N > 1:
        dp, ev = _sp_dpss(N, half_nbw, Kmax, sym=sym, norm=norm, return_ratios=True)
    if low_bias:
        keep = List[int]()
        for i in range(ev.shape[0]):
            if ev[i] > 0.9:
                keep.append(i)
        if len(keep) == 0:
            keep.append(int(np.argmax(ev)))
        ka = np.array(keep)
        dp = dp[ka].copy()
        ev = ev[ka].copy()
    return dp, ev


def _compute_mt_params(n_times: int, sfreq: float, bandwidth, low_bias: bool, adaptive: bool):
    if isinstance(bandwidth, str):
        w = _sp_get_window(bandwidth, n_times).reshape((1, n_times))
        return w, np.ones(1), False
    half_nbw = 4.0
    if not isinstance(bandwidth, NoneType):
        half_nbw = float(bandwidth) * n_times / (2.0 * sfreq)
    if half_nbw < 0.5:
        raise ValueError(f"bandwidth value {bandwidth} yields a normalized half-bandwidth of {half_nbw} < 0.5, use a value of at least {sfreq / n_times}")
    n_tapers_max = int(2 * half_nbw)
    w, ev = dpss_windows(n_times, half_nbw, n_tapers_max, sym=False, low_bias=low_bias)
    ad = adaptive
    if ad and ev.shape[0] < 3:
        ad = False
    return w, ev, ad


def _mt_spectra_row(sig, dpss, n_fft: int, remove_dc: bool):
    s = sig
    if remove_dc:
        s = sig - sig.mean()
    nt = dpss.shape[0]
    nf = n_fft // 2 + 1
    out = np.zeros((nt, nf), dtype=complex)
    for k in range(nt):
        out[k] = _rfft(s * dpss[k], n_fft)
    r2 = math.sqrt(2.0)
    for k in range(nt):
        out[k, 0] /= r2
        if n_fft % 2 == 0:
            out[k, nf - 1] /= r2
    return out


def _psd_from_mt_row(x_mt, w):
    # x_mt (n_tapers, n_freqs) complex, w (n_tapers,) real
    nt = x_mt.shape[0]
    nf = x_mt.shape[1]
    den = 0.0
    for k in range(nt):
        den += w[k] * w[k]
    psd = np.zeros(nf)
    for f in range(nf):
        acc = 0.0
        for k in range(nt):
            v = w[k] * x_mt[k, f]
            acc += (v * v.conjugate()).real
        psd[f] = acc * (2 / den)
    return psd


def _psd_from_mt_w2(x_mt, d):
    # weights d (n_tapers, n_freqs)
    nt = x_mt.shape[0]
    nf = x_mt.shape[1]
    psd = np.zeros(nf)
    for f in range(nf):
        acc = 0.0
        den = 0.0
        for k in range(nt):
            v = d[k, f] * x_mt[k, f]
            acc += (v * v.conjugate()).real
            den += d[k, f] * d[k, f]
        psd[f] = acc * (2 / den)
    return psd


def _trapezoid_dx(y, dx: float) -> float:
    s = 0.0
    for i in range(y.shape[0] - 1):
        s += (y[i] + y[i + 1]) * 0.5
    return s * dx


def _psd_adaptive_row(x_mt_full, eigvals, lo: int, hi: int, max_iter: int):
    nt = x_mt_full.shape[0]
    nfull = x_mt_full.shape[1]
    rt = np.sqrt(eigvals)
    psd_est = _psd_from_mt_row(x_mt_full, rt)
    var = _trapezoid_dx(psd_est, np.pi / nfull) / (2 * np.pi)
    xk = x_mt_full[:, lo:hi].copy()
    nf = hi - lo
    psd_iter = _psd_from_mt_row(xk[:2, :], rt[:2])
    err = np.zeros((nt, nf))
    d_k = np.zeros((nt, nf))
    for n in range(max_iter):
        for k in range(nt):
            for f in range(nf):
                d = psd_iter[f] / (eigvals[k] * psd_iter[f] + (1 - eigvals[k]) * var)
                d_k[k, f] = d * rt[k]
        err -= d_k
        mx = 0.0
        for f in range(nf):
            m = 0.0
            for k in range(nt):
                m += err[k, f] ** 2
            m /= nt
            if m > mx:
                mx = m
        if mx < 1e-10:
            break
        psd_iter = _psd_from_mt_w2(xk, d_k)
        err = d_k.copy()
    return psd_iter


def _psd_mt_2d(x2, sfreq: float, fmin: float, fmax: float, bandwidth, adaptive: bool,
               low_bias: bool, normalization: str, remove_dc: bool, max_iter: int):
    if normalization != "length" and normalization != "full":
        raise ValueError(f"Invalid value for the 'normalization' parameter. Allowed values are 'length' and 'full', but got '{normalization}' instead.")
    n_times = x2.shape[1]
    dp, ev, ad = _compute_mt_params(n_times, sfreq, bandwidth, low_bias, adaptive)
    w = np.sqrt(ev)
    freqs_all = _rfftfreq(n_times, 1.0 / sfreq)
    lo = -1
    hi = -1
    for i in range(freqs_all.shape[0]):
        if freqs_all[i] >= fmin and freqs_all[i] <= fmax:
            if lo < 0:
                lo = i
            hi = i + 1
    if lo < 0:
        lo = 0
        hi = 0
    freqs = freqs_all[lo:hi].copy()
    psd = np.zeros((x2.shape[0], hi - lo))
    for r in range(x2.shape[0]):
        xm = _mt_spectra_row(x2[r], dp, n_times, remove_dc)
        if not ad:
            psd[r] = _psd_from_mt_row(xm[:, lo:hi], w)
        else:
            psd[r] = _psd_adaptive_row(xm, ev, lo, hi, max_iter)
    if normalization == "full":
        psd /= sfreq
    return psd, freqs


def psd_array_multitaper(x, sfreq: float, fmin: float = 0.0, fmax: float = np.inf,
                         bandwidth = None, adaptive: bool = False, low_bias: bool = True,
                         normalization: str = "length", remove_dc: bool = True,
                         output: Literal[str] = "power", n_jobs = None, *,
                         max_iter: int = 150, verbose = None):
    """Compute power spectral density (PSD) using a multi-taper method."""
    if output != "power":
        compile_error("only output='power' is supported in Codon mne")
    xf = np.asarray(x).astype(float)
    n_times = xf.shape[xf.ndim - 1]
    x2 = xf.reshape((-1, n_times))
    psd, freqs = _psd_mt_2d(x2, float(sfreq), float(fmin), float(fmax), bandwidth, adaptive,
                            low_bias, normalization, remove_dc, max_iter)
    return psd.reshape(xf.shape[:-1] + (freqs.shape[0],)), freqs
