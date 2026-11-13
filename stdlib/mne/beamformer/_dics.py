# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# DICS beamformer: make_dics, apply_dics (single frequency; apply_dics_multi
# returns one STC per frequency), apply_dics_epochs, apply_dics_csd.

import numpy as np
from .._info import Info
from .._containers import Evoked, EpochsArray
from ..cov import Covariance, _check_rank, _make_scalings, _Rank, pick_channels_cov
from ..forward import Forward, pick_channels_forward
from ..source_estimate import _STC
from ..time_frequency.csd import CrossSpectralDensity
from ._compute import (_prepare_beamformer_input, _compute_beamformer, _compute_power,
                       _check_info_inv, _n_ch_types, _check_channels_spatial_filter, _opt_str,
                       _sum_rank, _rank_matrix, _herm, _like, _check_option, _ad_hoc_unit)
from ._lcmv import (Beamformer, _make_stc_sc, _make_stc_vec, _stc_kind, _proj_whiten_data,
                    _combine_xyz, _get_src_type)


def _common_channels(info: Info, fwd: Forward, csd: CrossSpectralDensity) -> List[str]:
    """mne.equalize_channels([info, forward, csd]) on good channels (template order)."""
    fs = Set[str](fwd.ch_names)
    cs = Set[str](csd.ch_names)
    out = List[str]()
    for c in info.ch_names:
        if c not in info.bads and c in fs and c in cs:
            out.append(c)
    return out


def _csd_matrix(csd: CrossSpectralDensity, i: int, names: List[str]) -> np.ndarray[complex, 2]:
    full = csd.get_data(None, i)
    idx = [csd.ch_names.index(c) for c in names]
    n = len(idx)
    out = np.zeros((n, n), dtype=complex)
    for a in range(n):
        for b in range(n):
            out[a, b] = full[idx[a], idx[b]]
    return out


def _real_of(C: np.ndarray[complex, 2]):
    n = C.shape[0]
    out = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            out[a, b] = C[a, b].real
    return out


def make_dics(info: Info, forward: Forward, csd: CrossSpectralDensity, reg: float = 0.05,
              noise_csd = None, label = None, pick_ori = None, rank = None, weight_norm = None,
              reduce_rank: bool = False, depth = 1.0, real_filter: bool = True,
              inversion: str = "matrix", *, verbose = None) -> Beamformer:
    """mne.beamformer.make_dics (volume / discrete source spaces)."""
    if not isinstance(label, NoneType):
        compile_error("label restriction is not supported")
    rk = _check_rank(rank)
    po = _opt_str(pick_ori)
    wn = _opt_str(weight_norm)
    _check_option("pick_ori", po, ["", "vector", "normal", "max-power"])
    _check_option("inversion", inversion, ["single", "matrix"])
    freqs = csd.frequencies
    n_freqs = len(freqs)
    has_noise = not isinstance(noise_csd, NoneType)
    # _check_one_ch_type for CSD input: no channel pre-selection
    if not has_noise:
        allow_mismatch = True
    else:
        allow_mismatch = False
    common = _common_channels(info, forward, csd)
    if has_noise:
        ns = Set[str](noise_csd.ch_names)
        common = [c for c in common if c in ns]
    info_e = info._pick([info.ch_names.index(c) for c in common])
    info_e.bads = List[str]()
    fwd_e = pick_channels_forward(forward, common, (), True, True)
    # noise csd -> real covariance (mean over frequencies)
    nc = _ad_hoc_unit(info_e)
    if has_noise:
        ncsd = noise_csd
        if len(ncsd) > 1:
            ncsd = ncsd.mean()
        nd = _real_of(_csd_matrix(ncsd, 0, common))
        nc = Covariance(nd, common, List[str](), info_e.projs, ncsd.n_fft)
    cxyz = "none" if inversion == "single" else "fro"
    bi = _prepare_beamformer_input(info_e, fwd_e, po, nc, rk, _depth_exp(depth), cxyz, "whiten", True)
    # ranks
    ch_names = list(bi.info.ch_names)
    csd_rank = List[int]()
    nrank = Dict[str, int]()
    if not allow_mismatch:
        nrank = _rank_matrix(nc._C, bi.info, rk, ch_names)
    for i in range(n_freqs):
        Ci = _csd_matrix(csd, i, ch_names)
        r = _rank_matrix(Ci, bi.info, rk, ch_names)
        if not allow_mismatch:
            for k, v in r.items():
                if k not in nrank or nrank[k] != v:
                    raise ValueError(f"{k} data rank ({v}) did not match the noise rank ({nrank.get(k, -1)})")
        csd_rank.append(_sum_rank(r))
    n_orient = 3 if bi.is_free_ori else 1
    f = Beamformer("DICS", ch_names)
    f.is_complex = not real_filter
    W3 = List[np.ndarray[float, 2]]()
    W3c = List[np.ndarray[complex, 2]]()
    mpos = List[np.ndarray[float, 2]]()
    for i in range(n_freqs):
        Cm = _csd_matrix(csd, i, ch_names)
        if real_filter:
            W, mpo = _compute_beamformer(bi.gain, _real_of(Cm), reg, n_orient, wn, po, reduce_rank,
                                         csd_rank[i], inversion, bi.nn, bi.orient_std, bi.whitener)
            W3.append(W)
            mpos.append(mpo)
        else:
            Wc, mpo = _compute_beamformer(bi.gain, Cm, reg, n_orient, wn, po, reduce_rank,
                                          csd_rank[i], inversion, bi.nn, bi.orient_std, bi.whitener)
            W3c.append(Wc)
            mpos.append(mpo)
    nr = mpos[0].shape[0]
    if real_filter:
        r0 = W3[0].shape[0]
        c0 = W3[0].shape[1]
        w = np.zeros((n_freqs, r0, c0))
        for i in range(n_freqs):
            w[i] = W3[i]
        f.weights3 = w
    else:
        r0 = W3c[0].shape[0]
        c0 = W3c[0].shape[1]
        wc = np.zeros((n_freqs, r0, c0), dtype=complex)
        for i in range(n_freqs):
            wc[i] = W3c[i]
        f.weights3_c = wc
    if po == "max-power":
        mp = np.zeros((n_freqs, nr, 3))
        for i in range(n_freqs):
            mp[i] = mpos[i]
        f.max_power_ori = mp
    f.csd = csd
    f.whitener = bi.whitener
    f.proj = bi.proj
    f.vertices = bi.vertno
    f.n_sources = bi.gain.shape[1] // n_orient
    f.pick_ori = None if po == "" else po
    f.inversion = inversion
    f.weight_norm = None if wn == "" else wn
    f.src_type = _get_src_type(forward)
    f.source_nn = forward.source_nn.copy()
    f.is_free_ori = bi.is_free_ori if (po == "" or po == "vector") else False
    return f


def _depth_exp(depth) -> Optional[float]:
    if isinstance(depth, NoneType):
        return None
    if isinstance(depth, Dict):
        e: Optional[float] = None
        for k, v in depth.items():
            if k == "exp":
                e = float(v)
        return e
    return float(depth)


def _apply_dics_one(M, f: Beamformer, projs, W, tmin: float, sfreq: float):
    """Apply one frequency's weights to (n_channels, n_times) data -> STC."""
    Mw = _proj_whiten_data(M, projs, f)
    sol = np.dot(W, Mw)
    vec = f.pick_ori is not None and f.pick_ori.__val__() == "vector"
    if f.is_free_ori and not vec:
        sol = _combine_xyz(sol)
    if vec:
        return _make_stc_vec(sol, f.vertices, tmin, 1.0 / sfreq, f.subject, f.source_nn, _stc_kind(f.src_type))
    return _make_stc_sc(sol, f.vertices, tmin, 1.0 / sfreq, f.subject, _stc_kind(f.src_type))


def apply_dics_multi(evoked: Evoked, filters: Beamformer, *, verbose = None):
    """apply_dics returning one STC per frequency (scalar data only)."""
    if filters.is_complex:
        raise ValueError("complex filters (real_filter=False) can only be applied to a CSD")
    sel = _check_channels_spatial_filter(evoked.ch_names, filters.ch_names)
    data = evoked.data[np.array(sel, dtype=int)]
    out = List[_STC[np.ndarray[float, 2]]]()
    for i in range(filters.weights3.shape[0]):
        out.append(_apply_dics_scalar(data, filters, evoked.info.projs, filters.weights3[i],
                                      float(evoked.times[0]), evoked.info.sfreq))
    return out


def _apply_dics_scalar(M, f: Beamformer, projs, W, tmin: float, sfreq: float):
    Mw = _proj_whiten_data(M, projs, f)
    sol = np.dot(W, Mw)
    if f.is_free_ori and not (f.pick_ori is not None and f.pick_ori.__val__() == "vector"):
        sol = _combine_xyz(sol)
    return _make_stc_sc(sol, f.vertices, tmin, 1.0 / sfreq, f.subject, _stc_kind(f.src_type))


def apply_dics(evoked: Evoked, filters: Beamformer, *, verbose = None):
    """mne.beamformer.apply_dics (single-frequency filters)."""
    r = apply_dics_multi(evoked, filters)
    if len(r) != 1:
        raise ValueError("filters contain several frequencies; use apply_dics_multi")
    return r[0]


def apply_dics_epochs(epochs: EpochsArray, filters: Beamformer, return_generator: bool = False,
                      *, verbose = None):
    """mne.beamformer.apply_dics_epochs -> list of STC."""
    if filters.weights3.shape[0] > 1:
        raise ValueError("This function only works on DICS beamformer weights that have been computed for a single frequency. When calling make_dics(), make sure to use a CSD object with only a single frequency (or frequency-bin) defined.")
    sel = _check_channels_spatial_filter(epochs.ch_names, filters.ch_names)
    data = epochs.get_data(sel)
    out = List[_STC[np.ndarray[float, 2]]]()
    for e in range(data.shape[0]):
        out.append(_apply_dics_scalar(data[e].copy(), filters, epochs.info.projs, filters.weights3[0],
                                      float(epochs.times[0]), epochs.info.sfreq))
    return out


def apply_dics_csd(csd: CrossSpectralDensity, filters: Beamformer, *, verbose = None):
    """mne.beamformer.apply_dics_csd -> (stc of source power, frequencies)."""
    names = filters.ch_names
    n_orient = 3 if filters.is_free_ori else 1
    freqs = csd.frequencies
    n_freqs = len(freqs)
    nsrc = filters.n_sources
    power = np.zeros((nsrc, n_freqs))
    wh = filters.whitener
    for i in range(n_freqs):
        Cm = _csd_matrix(csd, i, names)
        whc = _like(wh, Cm)
        Cw = np.dot(whc, np.dot(Cm, _herm(whc)))
        if filters.is_complex:
            Wc = filters.weights3_c[i]
            p = _compute_power(Cw, Wc, n_orient)
        else:
            W = filters.weights3[i]
            Wc = _like(W, Cw)
            p = _compute_power(Cw, Wc, n_orient)
        for s in range(nsrc):
            power[s, i] = p[s]
    stc = _make_stc_sc(power, filters.vertices, 0.0, 1.0, filters.subject, _stc_kind(filters.src_type))
    return stc, freqs
