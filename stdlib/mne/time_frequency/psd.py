# Ported from mne 1.13.2 (BSD-3-Clause)

import numpy as np
from scipy.signal import spectrogram as _spectrogram


def _check_nfft(n: int, n_fft: int, n_per_seg: Optional[int], n_overlap: int):
    if n_per_seg is None and n_fft > n:
        raise ValueError(f"If n_per_seg is None n_fft is not allowed to be > n_times. If you want zero-padding, you have to set n_per_seg to relevant length. Got n_fft of {n_fft} while signal length is {n}.")
    nps = n_fft if (n_per_seg is None or n_per_seg > n_fft) else n_per_seg
    nps = n if nps > n else nps
    if n_overlap >= nps:
        raise ValueError(f"n_overlap cannot be greater than n_per_seg (or n_fft). Got n_overlap of {n_overlap} while n_per_seg is {nps}.")
    return n_fft, nps, n_overlap


def _median_biases(n: int):
    biases = np.ones(n + 1)
    if n >= 3:
        ii_2 = 2 * np.arange(1, (n - 1) // 2 + 1)
        sums = 1 + np.cumsum(1.0 / (ii_2 + 1) - 1.0 / ii_2)
        idx = np.arange(2, n) // 2 - 1
        biases[3:] = sums[idx]
    return biases


def _psd_welch_2d(x, sfreq: float, fmin: float, fmax: float, n_fft: int, n_overlap: int,
                  n_per_seg: Optional[int], average: Optional[str], window, remove_dc: bool):
    """x: (n_rows, n_times). Returns (psds (n_rows, n_freqs[, n_seg]) flattened as 3-D, freqs)."""
    n_times = x.shape[1]
    nf, nps, nov = _check_nfft(n_times, n_fft, n_per_seg, n_overlap)
    freqs_all = np.arange(nf // 2 + 1, dtype=float) * (sfreq / nf)
    lo = -1
    hi = -1
    for i in range(freqs_all.shape[0]):
        if freqs_all[i] >= fmin and freqs_all[i] <= fmax:
            if lo < 0:
                lo = i
            hi = i
    if lo < 0:
        raise ValueError(f"No frequencies found between fmin={fmin} and fmax={fmax}")
    freqs = freqs_all[lo:hi + 1].copy()
    # non-finite channel handling: mne zeros them, then sets PSD to NaN
    bad = np.zeros(x.shape[0], dtype=bool)
    step = max(nps - nov, 1)
    analyzed_end = 0
    if n_times >= nps:
        n_segments = 1 + (n_times - nps) // step
        analyzed_end = step * (n_segments - 1) + nps
    xx = x
    any_bad = False
    for r in range(x.shape[0]):
        for t in range(analyzed_end):
            if not np.isfinite(x[r, t]):
                bad[r] = True
                any_bad = True
                break
    if any_bad:
        xx = x.copy()
        for r in range(x.shape[0]):
            if bad[r]:
                xx[r, :] = 0.0
    S = np.zeros((1, 1, 1))
    if remove_dc:
        _, _, S = _spectrogram(xx, detrend="constant", noverlap=nov, nperseg=nps, nfft=nf,
                               fs=sfreq, window=window, mode="psd")
    else:
        _, _, S = _spectrogram(xx, detrend=False, noverlap=nov, nperseg=nps, nfft=nf,
                               fs=sfreq, window=window, mode="psd")
    S = S[:, lo:hi + 1, :]
    out = np.zeros((1, 1, 1))
    if average is None:
        out = S.copy()
    elif average == "mean":
        out = np.nanmean(S, axis=-1).reshape((S.shape[0], S.shape[1], 1))
    elif average == "median":
        biases = _median_biases(S.shape[2])
        med = np.nanmedian(S, axis=-1)
        out = np.zeros((S.shape[0], S.shape[1], 1))
        for r in range(S.shape[0]):
            for k in range(S.shape[1]):
                cnt = 0
                for s in range(S.shape[2]):
                    if not np.isnan(S[r, k, s]):
                        cnt += 1
                out[r, k, 0] = med[r, k] / biases[cnt]
    else:
        raise ValueError(f"Invalid value for the 'average' parameter. Allowed values are None, False, 'mean', and 'median', but got {average} instead.")
    if any_bad:
        for r in range(out.shape[0]):
            if bad[r]:
                out[r, :, :] = np.nan
    return out, freqs


def psd_array_welch(x, sfreq: float, fmin: float = 0, fmax: float = np.inf, n_fft: int = 256,
                    n_overlap: int = 0, n_per_seg: Optional[int] = None, n_jobs = None,
                    average = "mean", window = "hamming", remove_dc: bool = True, *,
                    output: Literal[str] = "power", verbose = None):
    """Compute power spectral density (PSD) using Welch's method.

    Returns (psds, freqs); psds has shape x.shape[:-1] + (n_freqs,) for
    average='mean'|'median', or x.shape[:-1] + (n_freqs, n_segments) for
    average=None (average=False is treated as None like mne).
    """
    if output != "power":
        compile_error("only output='power' is supported in Typhon mne")
    xa = np.asarray(x)
    xf = xa.astype(float)
    nps: Optional[int] = None if n_per_seg is None else int(n_per_seg)
    n_times = xf.shape[-1]
    x2 = xf.reshape((-1, n_times))
    dshape = xf.shape[:-1]
    if isinstance(average, str):
        ps, freqs = _psd_welch_2d(x2, float(sfreq), float(fmin), float(fmax), int(n_fft),
                                  int(n_overlap), nps, average, window, remove_dc)
        return ps.reshape(dshape + (freqs.shape[0],)), freqs
    elif isinstance(average, NoneType) or isinstance(average, bool):
        if isinstance(average, bool):
            if average:
                raise ValueError("average must be None, False, 'mean' or 'median'")
        ps, freqs = _psd_welch_2d(x2, float(sfreq), float(fmin), float(fmax), int(n_fft),
                                  int(n_overlap), nps, None, window, remove_dc)
        return ps.reshape(dshape + (freqs.shape[0], ps.shape[2])), freqs
    else:
        compile_error("average must be None, False, 'mean' or 'median'")
