# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne.minimum_norm.inverse: make_inverse_operator, apply_inverse (+ raw,
# epochs, cov), estimate_snr. pick_ori is a compile-time string ("" for
# None, "normal", "vector") because the estimate type depends on it.

import numpy as np
import scipy.special
from .._info import Info
from .._proj import make_projector, _has_eeg_average_ref_proj
from ..cov import Covariance, prepare_noise_cov, compute_whitener
from ..forward import (Forward, convert_forward_solution, pick_channels_forward, is_fixed_orient,
                       compute_depth_prior, compute_orient_prior, _triage_loose, _normal_orth,
                       FIFFV_MNE_FREE_ORI, FIFFV_MNE_FIXED_ORI)
from ..source_estimate import _STC
from ._op import (InverseOperator, INVERSE_METHODS, combine_xyz, compute_rank_inverse,
                  prepare_inverse_operator, _check_method, _pick_channels_inverse_operator,
                  _check_ch_names)


def _prepare_forward(forward: Forward, info: Info, noise_cov: Covariance, fixed, loose, rank,
                     pca, exp, limit_depth_chs, combine_xyz_mode, allow_fixed_depth: bool, limit):
    fwd = forward.copy()
    lo = _triage_loose(fwd.src, loose, fixed)
    fixed_inverse = lo == 0.0
    constrained = lo < 1.0
    if fixed_inverse and fwd.src.kind == "volume":
        raise ValueError("Computing inverse solutions for volume source spaces with fixed orientations is not supported.")
    if fwd.src.kind == "volume" and lo < 1.0:
        raise ValueError("Computing inverse solutions with restricted orientations (loose < 1) is not supported for volume source spaces.")
    ex: Optional[float] = None
    if not isinstance(exp, NoneType):
        e = float(exp)
        if e < 0:
            raise ValueError(f"depth exponent should be greater than or equal to 0, got {e}")
        if e != 0.0:
            ex = e
    if fixed_inverse:
        if not is_fixed_orient(fwd):
            if allow_fixed_depth:
                fwd = convert_forward_solution(fwd, force_fixed=True, copy=False)
        elif ex is not None and not allow_fixed_depth:
            raise ValueError("For a fixed orientation inverse solution with depth weighting, the forward solution must be free-orientation and in surface orientation")
    else:
        if is_fixed_orient(fwd):
            raise ValueError("Forward operator has fixed orientation and can only be used to make a fixed-orientation inverse operator.")
        if constrained and not fwd.surf_ori:
            fwd = convert_forward_solution(fwd, surf_ori=True, copy=False)
    # _select_orient_forward
    bads = Set[str](info.bads)
    for b in noise_cov.bads:
        bads.add(b)
    names = List[str]()
    for c in info.ch_names:
        if c not in bads and c in fwd.ch_names and c in noise_cov.names:
            names.append(c)
    fwd = pick_channels_forward(fwd, names, (), True, False)
    info_picked = info._pick([info.ch_names.index(n) for n in names])
    depth_prior: Optional[np.ndarray[float, 1]] = None
    if ex is not None:
        depth_prior = compute_depth_prior(fwd, info_picked, ex, limit, limit_depth_chs,
                                          combine_xyz_mode, noise_cov, rank)
    orient_prior: Optional[np.ndarray[float, 1]] = None
    if fixed_inverse:
        if not is_fixed_orient(fwd):
            if depth_prior is not None:
                depth_prior = depth_prior.__val__()[2::3].copy()
            fwd = convert_forward_solution(fwd, surf_ori=True, force_fixed=True, copy=False)
    else:
        orient_prior = compute_orient_prior(fwd, lo)
    nc = prepare_noise_cov(noise_cov, info, names, rank)
    nc.diag = False  # mne stores the prepared covariance as a full matrix
    W, _ = compute_whitener(nc, info, names, rank, None, False, pca)
    gain = np.dot(W, fwd.sol)
    source_std = np.ones(gain.shape[1])
    if depth_prior is not None:
        source_std = source_std * depth_prior.__val__()
    if orient_prior is not None:
        source_std = source_std * orient_prior.__val__()
    source_std = np.sqrt(source_std)
    gain = gain * source_std[None, :]
    trace_GRGT = float(np.sum(gain * gain))
    n_nzero = int(np.sum(nc.eig.__val__() > 0))
    scale = np.sqrt(float(n_nzero) / trace_GRGT)
    source_std = source_std * scale
    gain = gain * scale
    return fwd, info_picked, gain, depth_prior, orient_prior, source_std, nc


def make_inverse_operator(info: Info, forward: Forward, noise_cov: Covariance, loose = "auto",
                          depth = 0.8, fixed = "auto", rank = None, use_cps: bool = True,
                          *, verbose = None) -> InverseOperator:
    """mne.minimum_norm.make_inverse_operator (depth: float | None)."""
    fwd, gain_info, gain, depth_prior, orient_prior, source_std, nc = _prepare_forward(
        forward, info, noise_cov, fixed, loose, rank, "white", depth, True, "spectral", False, 10.0)
    ef, sing, vh = np.linalg.svd(gain, full_matrices=False)
    ginfo = fwd.info.copy()
    ginfo.bads = [b for b in info.bads if b in ginfo.ch_names]
    inv = InverseOperator(ef.T.copy(), vh.T.copy(), sing, source_std * source_std, orient_prior,
                          depth_prior, nc, [p.copy() for p in gain_info.projs], ginfo, fwd.src.copy(),
                          fwd.source_nn.copy(), fwd.source_ori, fwd.nsource, 1.0, False,
                          fwd.coord_frame, list(gain_info.ch_names))
    return inv


def _check_reference(info: Info, ch_names: List[str]):
    sel = [i for i in range(info.nchan) if info.ch_names[i] in ch_names]
    has_eeg = False
    for i in sel:
        if info.ch_types[i] == "eeg":
            has_eeg = True
    if has_eeg and not _has_eeg_average_ref_proj(info):
        raise ValueError("EEG average reference (using a projector) is mandatory for modeling, use the method set_eeg_reference(projection=True)")


def _check_ori(pick_ori: str, inv: InverseOperator):
    if pick_ori != "" and pick_ori != "normal" and pick_ori != "vector":
        raise ValueError(f"Invalid value for the 'pick_ori' parameter. Allowed values are None, 'normal' and 'vector', but got '{pick_ori}' instead.")
    k = inv.src.kind
    if pick_ori == "normal" and k != "surface" and k != "discrete":
        raise RuntimeError(f"Normal source orientation is supported only for surface or discrete SourceSpaces, got type {k}")


def _check_or_prepare(inv: InverseOperator, nave: float, lambda2: float, method: str, method_params,
                      prepared: bool):
    if not prepared:
        return prepare_inverse_operator(inv, nave, lambda2, method, method_params, "non-src")
    if not inv.prepared:
        raise ValueError("inverse operator has not been prepared, but got argument prepared=True. Either pass prepared=False or use prepare_inverse_operator.")
    return inv


def _vertno(inv: InverseOperator) -> List[np.ndarray[int, 1]]:
    return [s.vertno.copy() for s in inv.src.spaces]


def _assemble_kernel(inv: InverseOperator, method: str, pick_ori: str):
    """Returns (K, noise_norm (n, 1), has_noise_norm, vertno, source_nn)."""
    eigen_leads = inv.eigen_leads
    source_cov = inv.source_cov
    source_nn = inv.source_nn
    vertno = _vertno(inv)
    has_nn = method == "dSPM" or method == "sLORETA"
    nn = np.zeros((0, 1))
    if has_nn:
        nn = inv.noisenorm.reshape((-1, 1))
    all_surf = True
    for s in inv.src.spaces:
        if s.type != "surf":
            all_surf = False
    if pick_ori == "normal" and all_surf:
        isid = True
        sn = inv.source_nn
        for i in range(sn.shape[0]):
            for j in range(3):
                d = 1.0 if (i % 3) == j else 0.0
                if abs(sn[i, j] - d) > 1e-6 + 1e-5 * d:
                    isid = False
        if isid:
            eigen_leads = eigen_leads.copy()
            source_nn = source_nn.copy()
            offset = 0
            for si in range(len(inv.src.spaces)):
                s = inv.src.spaces[si]
                for k in range(vertno[si].shape[0]):
                    o = _normal_orth(s.nn[int(vertno[si][k])])
                    r = offset + 3 * k
                    source_nn[r:r + 3] = o
                    eigen_leads[r:r + 3] = np.dot(o, eigen_leads[r:r + 3])
                offset += 3 * vertno[si].shape[0]
    if pick_ori == "normal":
        if inv.source_ori != FIFFV_MNE_FREE_ORI:
            raise ValueError("Picking normal orientation can only be done with a free orientation inverse operator.")
        po = inv.orient_prior
        if po is None or not (0.0 < po.__val__()[0] <= 1.0):
            raise ValueError("Picking normal orientation can only be done when working with loose orientations.")
    trans = np.dot(inv.eigen_fields, np.dot(inv.whitener, inv.proj))
    trans = trans * inv.reginv[:, None]
    K = np.dot(eigen_leads, trans)
    if not inv.eigen_leads_weighted:
        K = K * np.sqrt(source_cov)[:, None]
    if pick_ori == "normal":
        K = K[2::3].copy()
    return K, nn, has_nn, vertno, source_nn


def _is_free(inv: InverseOperator, pick_ori: str) -> bool:
    return inv.source_ori == FIFFV_MNE_FREE_ORI and pick_ori != "normal"


def _sol_from_K(K, nn, has_nn: bool, data, inv: InverseOperator, pick_ori: str):
    """Apply the kernel to data (n_sel_chan, n_times) -> 2-D solution."""
    sol = np.dot(K, data)
    free = _is_free(inv, pick_ori)
    if free and pick_ori != "vector":
        sol = combine_xyz(sol)
    if has_nn:
        nrm = nn
        if free and pick_ori == "vector":
            nrm = np.repeat(nn, 3, axis=0)
        sol = sol * nrm
    return sol


def _vec_data(sol, source_nn, nv: int):
    """Rotate vector solutions back into the head frame (mne._make_stc, vector)."""
    nt = sol.shape[1]
    if sol.shape[0] == nv:
        out = np.empty((nv, 3, nt))
        for v in range(nv):
            for c in range(3):
                out[v, c, :] = sol[v] * source_nn[v, c]
        return out
    d3 = sol.reshape((nv, 3, nt))
    out = np.empty((nv, 3, nt))
    for v in range(nv):
        out[v] = np.dot(source_nn[3 * v:3 * v + 3].T, d3[v])
    return out


def _make_stc(sol, vertno: List[np.ndarray[int, 1]], source_nn, tmin: float, tstep: float,
              inv: InverseOperator, pick_ori: Literal[str]):
    nv = 0
    for v in vertno:
        nv += v.shape[0]
    if pick_ori == "vector":
        return _STC[np.ndarray[float, 3]](_vec_data(sol, source_nn, nv), vertno, tmin, tstep,
                                          None, inv.src.kind)
    else:
        return _STC[np.ndarray[float, 2]](sol, vertno, tmin, tstep, None, inv.src.kind)


def _explained(data_w, est_w) -> float:
    r = data_w - est_w
    return 100.0 * (1.0 - float(np.sum(r * r)) / float(np.sum(data_w * data_w)))


def _data_est(inv: InverseOperator, data):
    Pi = inv.sing * inv.reginv
    data_w = np.dot(inv.whitener, np.dot(inv.proj, data))
    w_t = np.dot(inv.eigen_fields, data_w)
    est = np.dot(inv.colorer, np.dot(inv.eigen_fields.T, Pi[:, None] * w_t))
    return est, data_w


def apply_inverse(evoked, inverse_operator: InverseOperator, lambda2: float = 1.0 / 9.0,
                  method: Literal[str] = "dSPM", pick_ori: Literal[str] = "",
                  prepared: bool = False, label = None, method_params = None,
                  return_residual: Literal[bool] = False, use_cps: bool = True, verbose = None):
    """mne.minimum_norm.apply_inverse (pick_ori: "" | "normal" | "vector")."""
    if not isinstance(label, NoneType):
        compile_error("apply_inverse: label restriction is not supported")
    _check_method(method)
    _check_ori(pick_ori, inverse_operator)
    _check_reference(evoked.info, inverse_operator.ch_names)
    _check_ch_names(inverse_operator, evoked.info)
    inv = _check_or_prepare(inverse_operator, float(evoked.nave), lambda2, method, method_params, prepared)
    sel = _pick_channels_inverse_operator(evoked.info.ch_names, inv)
    sa = np.array(sel, dtype=int)
    data = evoked.data[sa]
    K, nn, has_nn, vertno, source_nn = _assemble_kernel(inv, method, pick_ori)
    sol = _sol_from_K(K, nn, has_nn, data, inv, pick_ori)
    tmin = float(evoked.times[0])
    stc = _make_stc(sol, vertno, source_nn, tmin, 1.0 / evoked.info.sfreq, inv, pick_ori)
    if return_residual:
        est, _ = _data_est(inv, data)
        residual = evoked.copy()
        for i in range(len(sel)):
            residual.data[sel[i]] = residual.data[sel[i]] - est[i]
        return stc, residual
    else:
        return stc


def apply_inverse_raw(raw, inverse_operator: InverseOperator, lambda2: float,
                      method: Literal[str] = "dSPM", label = None, start = None, stop = None,
                      nave: float = 1.0, time_func = None, pick_ori: Literal[str] = "",
                      buffer_size = None, prepared: bool = False, method_params = None,
                      use_cps: bool = True, verbose = None):
    """mne.minimum_norm.apply_inverse_raw."""
    if not isinstance(label, NoneType):
        compile_error("apply_inverse_raw: label restriction is not supported")
    _check_method(method)
    _check_ori(pick_ori, inverse_operator)
    _check_reference(raw.info, inverse_operator.ch_names)
    _check_ch_names(inverse_operator, raw.info)
    inv = _check_or_prepare(inverse_operator, float(nave), lambda2, method, method_params, prepared)
    sel = _pick_channels_inverse_operator(raw.info.ch_names, inv)
    n = raw._data.shape[1]
    s0 = 0 if isinstance(start, NoneType) else int(start)
    s1 = n if isinstance(stop, NoneType) else int(stop)
    sa = np.array(sel, dtype=int)
    data = raw._data[sa, s0:s1].copy()
    if not isinstance(time_func, NoneType):
        data = np.asarray(time_func(data), dtype=float)
    K, nn, has_nn, vertno, source_nn = _assemble_kernel(inv, method, pick_ori)
    sol = _sol_from_K(K, nn, has_nn, data, inv, pick_ori)
    tmin = float(raw.times[s0])
    return _make_stc(sol, vertno, source_nn, tmin, 1.0 / raw.info.sfreq, inv, pick_ori)


def apply_inverse_epochs(epochs, inverse_operator: InverseOperator, lambda2: float,
                         method: Literal[str] = "dSPM", label = None, nave: float = 1.0,
                         pick_ori: Literal[str] = "", return_generator: bool = False,
                         prepared: bool = False, method_params = None, use_cps: bool = True,
                         verbose = None):
    """mne.minimum_norm.apply_inverse_epochs (returns a list of estimates)."""
    if not isinstance(label, NoneType):
        compile_error("apply_inverse_epochs: label restriction is not supported")
    _check_method(method)
    _check_ori(pick_ori, inverse_operator)
    _check_reference(epochs.info, inverse_operator.ch_names)
    _check_ch_names(inverse_operator, epochs.info)
    inv = _check_or_prepare(inverse_operator, float(nave), lambda2, method, method_params, prepared)
    sel = _pick_channels_inverse_operator(epochs.info.ch_names, inv)
    sa = np.array(sel, dtype=int)
    K, nn, has_nn, vertno, source_nn = _assemble_kernel(inv, method, pick_ori)
    tstep = 1.0 / epochs.info.sfreq
    tmin = float(epochs.times[0])
    out = []
    for e in range(epochs._data.shape[0]):
        data = epochs._data[e][sa]
        sol = _sol_from_K(K, nn, has_nn, data, inv, pick_ori)
        out.append(_make_stc(sol, [v.copy() for v in vertno], source_nn, tmin, tstep, inv, pick_ori))
    return out


def apply_inverse_cov(cov: Covariance, info: Info, inverse_operator: InverseOperator,
                      nave: float = 1.0, lambda2: float = 1.0 / 9.0, method: str = "dSPM",
                      pick_ori = None, prepared: bool = False, label = None, method_params = None,
                      use_cps: bool = True, verbose = None):
    """mne.minimum_norm.apply_inverse_cov: source power (n_sources, 1) SourceEstimate."""
    if not isinstance(label, NoneType):
        compile_error("apply_inverse_cov: label restriction is not supported")
    _check_method(method)
    inv0 = inverse_operator
    sel = _pick_channels_inverse_operator(cov.names, inv0)
    use_names = [cov.names[i] for i in sel]
    inv = _check_or_prepare(inv0, float(nave), lambda2, method, method_params, prepared)
    isel = _pick_channels_inverse_operator(use_names, inv)
    nch = len(use_names)
    free = inv.source_ori == FIFFV_MNE_FREE_ORI
    normal = False
    if not isinstance(pick_ori, NoneType):
        if str(pick_ori) != "normal":
            raise ValueError(f"Invalid value for the 'pick_ori' parameter. Allowed values are None and 'normal', but got '{pick_ori}' instead.")
        normal = True
    _check_ori("normal" if normal else "", inv)
    po = "normal" if normal else "vector"
    K, nn, has_nn, vertno, source_nn = _assemble_kernel(inv, method, po)
    eye = np.eye(nch)[np.array(isel, dtype=int)]
    sol = _sol_from_K(K, nn, has_nn, eye, inv, po)
    nv = 0
    for v in vertno:
        nv += v.shape[0]
    Kf = sol
    if free and not normal:
        Kf = _vec_data(sol, source_nn, nv).reshape((-1, nch))
    elif (not free) and not normal:
        Kf = sol
    sa = np.array(sel, dtype=int)
    C = cov._C[np.ix_(sa, sa)]
    tmp = np.dot(C, Kf.T)
    diag = np.sum(Kf * tmp.T, axis=1)
    res = diag.reshape((-1, 1))
    if free and not normal:
        r3 = diag.reshape((nv, 3))
        res = np.sum(r3, axis=1).reshape((-1, 1))
    return _STC[np.ndarray[float, 2]](res, vertno, 0.0, 1.0, None, inv.src.kind)


def estimate_snr(evoked, inv: InverseOperator, verbose = None):
    """mne.minimum_norm.estimate_snr -> (snr, snr_est)."""
    _check_reference(evoked.info, inv.ch_names)
    _check_ch_names(inv, evoked.info)
    pinv = prepare_inverse_operator(inv, float(evoked.nave), 1.0 / 9.0, "MNE", None, "non-src")
    sel = _pick_channels_inverse_operator(evoked.info.ch_names, pinv)
    sa = np.array(sel, dtype=int)
    data_white = np.dot(pinv.whitener, np.dot(pinv.proj, evoked.data[sa]))
    data_white_ef = np.dot(pinv.eigen_fields, data_white)
    n_times = data_white.shape[1]
    n_ch_eff = compute_rank_inverse(pinv)
    signal = np.sum(data_white ** 2, axis=0)
    snr = signal / float(n_ch_eff)
    lambda2_est = np.full(n_times, 10.0)
    remaining = np.ones(n_times, dtype=bool)
    for t in range(n_times):
        if snr[t] <= 1.0:
            lambda2_est[t] = np.inf
            remaining[t] = False
    lambda_mult = 0.99
    sing2 = pinv.sing * pinv.sing
    val = float(scipy.special.chdtri(float(n_ch_eff), 1e-3))
    for n_iter in range(1000):
        any_rem = False
        for t in range(n_times):
            if not remaining[t]:
                continue
            err = 0.0
            for k in range(sing2.shape[0]):
                f = 0.0
                if pinv.sing[k] != 0.0:
                    f = sing2[k] / (sing2[k] + lambda2_est[t])
                ew = data_white_ef[k, t] * (1.0 - f)
                err += ew * ew
            if err < val:
                remaining[t] = False
            else:
                any_rem = True
        if not any_rem:
            break
        for t in range(n_times):
            if remaining[t]:
                lambda2_est[t] = lambda2_est[t] * lambda_mult
    snr_est = 1.0 / np.sqrt(lambda2_est)
    return np.sqrt(snr), snr_est
