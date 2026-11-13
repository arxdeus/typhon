# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne.minimum_norm.time_frequency (whole source space, pick_ori None/normal):
# source_induced_power, compute_source_psd_epochs, compute_source_psd.

import numpy as np
from .._info import Info
from ..source_estimate import _STC
from ..time_frequency.tfr import morlet, cwt
from ..time_frequency.multitaper import _compute_mt_params, _mt_spectra_row, _psd_from_mt_row
from ._op import InverseOperator, combine_xyz, _check_method, _pick_channels_inverse_operator
from ._inverse import _assemble_kernel, _check_or_prepare, _check_ori, FIFFV_MNE_FREE_ORI


def _source_params(epochs, inv0: InverseOperator, lambda2: float, method: str, nave: float,
                   pca: bool, pick_ori: str, prepared: bool, method_params):
    inv = _check_or_prepare(inv0, nave, lambda2, method, method_params, prepared)
    sel = _pick_channels_inverse_operator(epochs.info.ch_names, inv)
    K, nn, has_nn, vertno, _ = _assemble_kernel(inv, method, pick_ori)
    Vh = np.eye(K.shape[1])
    if pca:
        U, s, Vt = np.linalg.svd(K, full_matrices=False)
        rank = int(np.sum(s > 1e-8 * s[0]))
        K = s[:rank][None, :] * U[:, :rank]
        Vh = Vt[:rank].copy()
    return K, sel, Vh, vertno, nn, has_nn, inv


def source_induced_power(epochs, inverse_operator: InverseOperator, freqs, label = None,
                         lambda2: float = 1.0 / 9.0, method: str = "dSPM", nave: float = 1.0,
                         n_cycles = 5, decim: int = 1, use_fft: bool = False,
                         pick_ori: str = "", baseline = None, baseline_mode: str = "logratio",
                         pca: bool = True, n_jobs = None, *, return_plv: Literal[bool] = True,
                         zero_mean: bool = False, prepared: bool = False, method_params = None,
                         use_cps: bool = True, verbose = None):
    """mne.minimum_norm.source_induced_power -> (power[, plv]) arrays
    (n_sources, n_freqs, n_times). Baseline correction is not supported."""
    if not isinstance(label, NoneType) or not isinstance(baseline, NoneType):
        compile_error("source_induced_power: label/baseline are not supported")
    _check_method(method)
    po = "normal" if pick_ori == "" else pick_ori
    _check_ori(po, inverse_operator)
    K, sel, Vh, vertno, nn, has_nn, inv = _source_params(epochs, inverse_operator, lambda2, method,
                                                         nave, pca, po, prepared, method_params)
    free = inv.source_ori == FIFFV_MNE_FREE_ORI and pick_ori == ""
    sfreq = epochs.info.sfreq
    Ws = morlet(sfreq, freqs, n_cycles, None, zero_mean)
    sa = np.array(sel, dtype=int)
    n_ep = epochs._data.shape[0]
    n_src = K.shape[0] // 3 if free else K.shape[0]
    n_f = len(Ws)
    n_t = len(range(0, epochs._data.shape[2], decim))
    power = np.zeros((n_src, n_f, n_t))
    plv = np.zeros((n_src, n_f, n_t), dtype=complex)
    for e in range(n_ep):
        x = np.dot(Vh, epochs._data[e][sa])
        for f in range(n_f):
            tfr = cwt(x, [Ws[f]], use_fft, "same", decim)
            tf2 = tfr.reshape((x.shape[0], -1))
            for part in range(2):
                t = np.real(tf2) if part == 0 else np.imag(tf2)
                sol = np.dot(K, t)
                pick = sol[2::3] if free else sol
                for s in range(n_src):
                    for k in range(n_t):
                        if part == 0:
                            plv[s, f, k] += pick[s, k]
                        else:
                            plv[s, f, k] += 1j * pick[s, k]
                if free:
                    sol = combine_xyz(sol, True)
                else:
                    sol = sol * sol
                power[:, f, :] += sol
    # plv is normalised per epoch in mne: recompute per epoch
    power = power / float(n_ep)
    if has_nn:
        power = power * (nn ** 2)[:, :, None]
    if return_plv:
        return power, np.abs(plv) / float(n_ep)
    return power


def compute_source_psd_epochs(epochs, inverse_operator: InverseOperator, lambda2: float = 1.0 / 9.0,
                              method: str = "dSPM", fmin: float = 0.0, fmax: float = 200.0,
                              pick_ori: str = "", label = None, nave: float = 1.0,
                              pca: bool = True, inv_split = None, bandwidth = 4.0,
                              adaptive: bool = False, low_bias: bool = True,
                              return_generator: bool = False, n_jobs = None,
                              prepared: bool = False, method_params = None, use_cps: bool = True,
                              verbose = None):
    """mne.minimum_norm.compute_source_psd_epochs: list of SourceEstimate (non-adaptive)."""
    if not isinstance(label, NoneType) or adaptive:
        compile_error("compute_source_psd_epochs: label/adaptive are not supported")
    _check_method(method)
    po = "normal" if pick_ori == "" else pick_ori
    K, sel, Vh, vertno, nn, has_nn, inv = _source_params(epochs, inverse_operator, lambda2, method,
                                                         nave, pca, po, prepared, method_params)
    free = inv.source_ori == FIFFV_MNE_FREE_ORI and pick_ori == ""
    n_times = epochs._data.shape[2]
    sfreq = epochs.info.sfreq
    dpss, eigvals, _ = _compute_mt_params(n_times, sfreq, bandwidth, low_bias, False)
    nt = dpss.shape[0]
    wts = np.sqrt(eigvals)
    nf = n_times // 2 + 1
    freqs = np.arange(nf).astype(float) * sfreq / float(n_times)
    mask = np.array([freqs[i] >= fmin and freqs[i] <= fmax for i in range(nf)])
    fsel = np.array([i for i in range(nf) if mask[i]], dtype=int)
    fstep = float(np.mean(np.diff(freqs)))
    sa = np.array(sel, dtype=int)
    out = []
    for e in range(epochs._data.shape[0]):
        data = np.dot(Vh, epochs._data[e][sa])
        nr = data.shape[0]
        xmt = np.empty((nr, nt, nf), dtype=complex)
        for c in range(nr):
            xmt[c] = _mt_spectra_row(data[c], dpss, n_times, True)
        psd = np.empty((K.shape[0], fsel.shape[0]))
        for s in range(K.shape[0]):
            xs = np.zeros((nt, nf), dtype=complex)
            for i in range(nt):
                for c in range(nr):
                    xs[i] = xs[i] + K[s, c] * xmt[c, i]
            psd[s] = _psd_from_mt_row(xs[:, fsel], wts)
        if free:
            psd = combine_xyz(psd)
        if has_nn:
            psd = psd * nn ** 2
        out.append(_STC[np.ndarray[float, 2]](psd, [v.copy() for v in vertno], float(freqs[fsel[0]]),
                                              fstep, None, inv.src.kind))
    return out


def compute_source_psd(raw, inverse_operator: InverseOperator, lambda2: float = 1.0 / 9.0,
                       method: str = "dSPM", tmin: float = 0.0, tmax = None, fmin: float = 0.0,
                       fmax: float = 200.0, n_fft: int = 2048, overlap: float = 0.5,
                       pick_ori: str = "", label = None, nave: float = 1.0, pca: bool = True,
                       prepared: bool = False, method_params = None, use_cps: bool = True,
                       dB: bool = False, verbose = None):
    """mne.minimum_norm.compute_source_psd (hann window): mean source PSD."""
    from ..epochs import EpochsArray
    from ..event import make_fixed_length_events
    sf = raw.info.sfreq
    duration = ((1.0 - overlap) * n_fft) / sf
    ev = make_fixed_length_events(raw, 1, tmin, tmax, duration)
    n = raw._data.shape[1]
    segs = List[np.ndarray[float, 2]]()
    for i in range(ev.shape[0]):
        s0 = int(ev[i, 0]) - raw.first_samp
        if s0 + n_fft <= n and s0 >= 0:
            segs.append(raw._data[:, s0:s0 + n_fft].copy())
    if len(segs) == 0:
        raise RuntimeError("No segments")
    arr = np.empty((len(segs), raw._data.shape[0], n_fft))
    for i in range(len(segs)):
        arr[i] = segs[i]
    ep = EpochsArray(arr, raw.info.copy())
    stcs = compute_source_psd_epochs(ep, inverse_operator, lambda2, method, fmin, fmax, pick_ori,
                                     None, nave, pca, None, "hann", False, True, False, None,
                                     prepared, method_params)
    acc = stcs[0].data * 0.0
    for s in stcs:
        acc = acc + s.data
    acc = acc / float(len(stcs))
    if dB:
        acc = 10.0 * np.log10(acc)
    r = stcs[0]
    return _STC[np.ndarray[float, 2]](acc, r.vertices, r._tmin, r._tstep, None, r.kind)
