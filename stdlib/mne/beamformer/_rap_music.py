# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# RAP-MUSIC and TRAP-MUSIC dipole scanning (mne.beamformer.rap_music / trap_music).

import numpy as np
from .._info import Info
from .._containers import Evoked
from ..epochs import EvokedArray
from ..cov import Covariance
from ..forward import Forward, convert_forward_solution, pick_channels_forward, is_fixed_orient
from ._compute import _prepare_beamformer_input, _check_info_inv


class Dipole:
    """Fixed-position dipole time course (mne.Dipole subset)."""
    times: np.ndarray[float, 1]
    pos: np.ndarray[float, 2]
    amplitude: np.ndarray[float, 1]
    ori: np.ndarray[float, 2]
    gof: np.ndarray[float, 1]

    def __init__(self, times, pos, amplitude, ori, gof):
        self.times = times
        self.pos = pos
        self.amplitude = amplitude
        self.ori = ori
        self.gof = gof

    def __len__(self) -> int:
        return self.times.shape[0]


def _split_gof(M, X, gain):
    """mne.inverse_sparse.mxne_inverse._split_gof."""
    nt = M.shape[1]
    norm = np.zeros(nt)
    for t in range(nt):
        norm[t] = float(np.sum(M[:, t] * M[:, t]))
        if norm[t] == 0.0:
            norm[t] = np.inf
    M_est = np.dot(gain, X)
    res = M - M_est
    U, s, _ = np.linalg.svd(gain, full_matrices=False)
    keep = [i for i in range(s.shape[0]) if s[i] >= s[0] * 1e-6]
    U = U[:, np.array(keep, dtype=int)].copy()
    fit_orth = np.dot(U.T, M)
    res_orth = np.dot(U.T, res)
    UG = np.dot(U.T, gain)
    nk = U.shape[1]
    nx = X.shape[0]
    gof = np.zeros((nx, nt))
    for j in range(nx):
        w = np.zeros((nk, nt))
        for k in range(nk):
            w[k] = UG[k, j] * X[j]
        for t in range(nt):
            wn = float(np.sqrt(np.sum(w[:, t] * w[:, t])))
            if wn == 0.0:
                wn = 1.0
            fb = 0.0
            rb = 0.0
            for k in range(nk):
                wk = w[k, t] / wn
                fb += (fit_orth[k, t] * wk) ** 2
                rb += (res_orth[k, t] * wk) ** 2
            gof[j, t] = 100.0 * (fb - rb) / norm[t]
    return gof


def _compute_subcorr(G, phi_sig):
    Ug, Sg, Vg = np.linalg.svd(G, full_matrices=False)
    rank = 0
    for i in range(Sg.shape[0]):
        if Sg[i] > Sg[0] * 1e-6:
            rank += 1
    if rank == 0:
        return 0.0, np.zeros(G.shape[0])
    rank = max(rank, 2)
    rank = min(rank, Sg.shape[0])
    Ug = Ug[:, :rank]
    Sg = Sg[:rank]
    Vg = Vg[:rank]
    tmp = np.dot(Ug.T, phi_sig)
    Uc, Sc, _ = np.linalg.svd(tmp, full_matrices=False)
    v = np.zeros(rank)
    for i in range(rank):
        v[i] = Uc[i, 0] / Sg[i]
    X = np.dot(Vg.T, v)
    return float(Sc[0]), X / np.linalg.norm(X)


def _compute_proj(A):
    U, _, _ = np.linalg.svd(A, full_matrices=False)
    return np.identity(A.shape[0]) - np.dot(U, U.T)


def _apply_rap_music(data, info: Info, times, forward: Forward, noise_cov: Covariance,
                     n_dipoles: int, use_trap: bool):
    fwd = forward
    align = forward.source_nn.copy()
    if fwd.surf_ori and not is_fixed_orient(fwd):
        fwd = convert_forward_solution(fwd, surf_ori=False)
    bi = _prepare_beamformer_input(info, fwd, "", noise_cov, None)
    G0 = bi.gain
    fwd = pick_channels_forward(fwd, bi.info.ch_names, (), True, True)
    M = np.dot(bi.whitener, data)
    _, eigv = np.linalg.eigh(np.dot(M, M.T))
    nc = eigv.shape[1]
    phi_sig = eigv[:, nc - n_dipoles:].copy()
    n_orient = 3 if bi.is_free_ori else 1
    n_src = G0.shape[1] // n_orient
    n_channels = G0.shape[0]
    gain = fwd.sol.copy()
    A = np.zeros((n_channels, n_dipoles))
    oris = np.zeros((n_dipoles, 3))
    poss = np.zeros((n_dipoles, 3))
    idxs = List[int]()
    G_proj = G0.copy()
    phi_proj = phi_sig.copy()
    for k in range(n_dipoles):
        subcorr_max = -1.0
        source_idx = 0
        source_ori = np.zeros(3)
        source_pos = np.zeros(3)
        for i_source in range(n_src):
            Gk = G_proj[:, i_source * n_orient:(i_source + 1) * n_orient].copy()
            sub, ori = _compute_subcorr(Gk, phi_proj)
            if sub > subcorr_max:
                subcorr_max = sub
                source_idx = i_source
                so = np.zeros(3)
                for q in range(ori.shape[0]):
                    so[q] = ori[q]
                source_ori = so
                source_pos = fwd.source_rr[i_source].copy()
                if n_orient == 3:
                    sg = float(np.dot(source_ori, fwd.source_nn[3 * i_source + 2]))
                    sign = 1.0
                    if sg > 0:
                        sign = 1.0
                    elif sg < 0:
                        sign = -1.0
                    source_ori = source_ori * sign
                else:
                    source_ori = fwd.source_nn[i_source].copy()
        idxs.append(source_idx)
        if n_orient == 3:
            A[:, k] = np.dot(G0[:, source_idx * 3:source_idx * 3 + 3], source_ori)
        else:
            A[:, k] = G0[:, source_idx]
        oris[k] = source_ori
        poss[k] = source_pos
        proj = _compute_proj(A[:, :k + 1])
        G_proj = np.dot(proj, G0)
        phi_proj = np.dot(proj, phi_sig)
        if use_trap:
            phi_proj = phi_proj[:, phi_proj.shape[1] - (n_dipoles - k):].copy()
    sol = np.linalg.lstsq(A, M)[0]
    nt = M.shape[1]
    X = np.zeros((0, nt))
    if n_orient == 3:
        X = np.zeros((n_dipoles * 3, nt))
        for d in range(n_dipoles):
            for q in range(3):
                X[3 * d + q] = sol[d] * oris[d, q]
    else:
        X = sol.copy()
    gain_dip = np.zeros((n_channels, n_dipoles))
    gain_active = np.zeros((n_channels, 0))
    if n_orient == 3:
        ga = np.zeros((n_channels, n_dipoles * 3))
        for d in range(n_dipoles):
            si = idxs[d]
            for q in range(3):
                ga[:, 3 * d + q] = gain[:, 3 * si + q]
                gain_dip[:, d] += gain[:, 3 * si + q] * oris[d, q]
        gain_active = np.dot(bi.whitener, ga)
    else:
        ga = np.zeros((n_channels, n_dipoles))
        for d in range(n_dipoles):
            ga[:, d] = gain[:, idxs[d]]
            gain_dip[:, d] = gain[:, idxs[d]]
        gain_active = np.dot(bi.whitener, ga)
    explained = np.dot(gain_dip, sol)
    tstep = 1.0
    if times.shape[0] > 1:
        dd = np.diff(times)
        tstep = float(np.median(dd))
    gof_split = _split_gof(M, X, gain_active)
    n_per = 3 if n_orient == 3 else 1
    dipoles = List[Dipole]()
    for d in range(n_dipoles):
        gofd = np.zeros(nt)
        for q in range(n_per):
            gofd = gofd + gof_split[d * n_per + q]
        pos = np.zeros((nt, 3))
        for t in range(nt):
            pos[t] = fwd.source_rr[idxs[d]]
        Xd = X[d * n_per:(d + 1) * n_per]
        amp = np.zeros(nt)
        ori = np.zeros((nt, 3))
        if n_per == 1:
            amp = Xd[0].copy()
            for t in range(nt):
                ori[t] = fwd.source_nn[idxs[d]]
        else:
            if fwd.surf_ori:
                Xd = np.dot(fwd.source_nn[idxs[d] * 3:idxs[d] * 3 + 3].T, Xd)
            for t in range(nt):
                a = float(np.sqrt(np.sum(Xd[:, t] ** 2)))
                amp[t] = a
                if a > 0.0:
                    for q in range(3):
                        ori[t, q] = Xd[q, t] / a
        # sign alignment with the found orientation
        for t in range(nt):
            dsum = ori[t, 0] * oris[d, 0] + ori[t, 1] * oris[d, 1] + ori[t, 2] * oris[d, 2]
            sg = 1.0
            if dsum > 0:
                sg = 1.0
            elif dsum < 0:
                sg = -1.0
            for q in range(3):
                ori[t, q] = ori[t, q] * sg
            amp[t] = amp[t] * sg
        tt = float(times[0]) + tstep * np.arange(nt)
        dipoles.append(Dipole(tt, pos, amp, ori, gofd))
    return dipoles, explained, bi.info


def _rap_music(evoked: Evoked, forward: Forward, noise_cov: Covariance, n_dipoles: int, use_trap: bool):
    info = evoked.info
    picks = _check_info_inv(info, forward.ch_names, [noise_cov.names], [noise_cov.bads])
    data = evoked.data[np.array(picks, dtype=int)].copy()
    info2 = info._pick(picks)
    dipoles, explained, _ = _apply_rap_music(data, info2, evoked.times, forward, noise_cov, n_dipoles, use_trap)
    return dipoles


def rap_music(evoked: Evoked, forward: Forward, noise_cov: Covariance, n_dipoles: int = 5,
              return_residual: Literal[bool] = False, *, verbose = None):
    """mne.beamformer.rap_music (list of Dipole; residual via rap_music_residual)."""
    return _rap_music(evoked, forward, noise_cov, n_dipoles, False)


def trap_music(evoked: Evoked, forward: Forward, noise_cov: Covariance, n_dipoles: int = 5,
               return_residual: Literal[bool] = False, *, verbose = None):
    """mne.beamformer.trap_music."""
    return _rap_music(evoked, forward, noise_cov, n_dipoles, True)


def rap_music_residual(evoked: Evoked, forward: Forward, noise_cov: Covariance, n_dipoles: int = 5,
                       use_trap: bool = False):
    """(dipoles, residual Evoked) as rap_music(..., return_residual=True)."""
    info = evoked.info
    picks = _check_info_inv(info, forward.ch_names, [noise_cov.names], [noise_cov.bads])
    data = evoked.data[np.array(picks, dtype=int)].copy()
    info2 = info._pick(picks)
    dipoles, explained, _ = _apply_rap_music(data, info2, evoked.times, forward, noise_cov, n_dipoles, use_trap)
    res = EvokedArray(data - explained, info2, float(evoked.times[0]), "", evoked.nave)
    return dipoles, res
