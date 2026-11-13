# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne.minimum_norm.resolution_matrix: resolution matrix, PSF and CTF.

import numpy as np
from ..forward import Forward, convert_forward_solution, pick_channels_forward
from ..source_estimate import _STC
from ._op import InverseOperator, FIFFV_MNE_FREE_ORI
from ._inverse import apply_inverse, _vertno


def _inverse_matrix(inv: InverseOperator, fwd: Forward, method: Literal[str], lambda2: float):
    """_get_matrix_from_inverse_operator: (n_dipoles, n_good_channels)."""
    from .._containers import Evoked
    names = list(inv.info.ch_names)
    n = len(names)
    info = inv.info.copy()
    info.sfreq = 1000.0
    info.projs = [p.copy() for p in inv.projs]
    times = np.arange(n).astype(float) / 1000.0
    ev = Evoked(np.eye(n), info, times, 1, "")
    bad_idx = [names.index(b) for b in inv.info.bads]
    keep = np.array([i for i in range(n) if i not in bad_idx], dtype=int)
    if inv.source_ori != FIFFV_MNE_FREE_ORI:
        stc = apply_inverse(ev, inv, lambda2, method, "")
        return stc.data[:, keep]
    stc = apply_inverse(ev, inv, lambda2, method, "vector")
    d = stc.data[:, :, keep]
    return d.reshape((d.shape[0] * d.shape[1], d.shape[2]))


def make_inverse_resolution_matrix(forward: Forward, inverse_operator: InverseOperator,
                                   method: Literal[str] = "dSPM", lambda2: float = 1.0 / 9.0, verbose = None):
    """mne.minimum_norm.make_inverse_resolution_matrix."""
    inv = inverse_operator
    fwd = forward
    if inv.source_ori == FIFFV_MNE_FREE_ORI and fwd.source_ori == FIFFV_MNE_FREE_ORI:
        if inv._is_surf_ori != fwd.surf_ori:
            fwd = convert_forward_solution(fwd, surf_ori=inv._is_surf_ori, force_fixed=False)
    good = [c for c in inv.info.ch_names if c not in inv.info.bads]
    fwd = pick_channels_forward(fwd, good, (), True)
    invmat = _inverse_matrix(inv, fwd, method, lambda2)
    return np.dot(invmat, fwd.sol)


def _normalise(funcs, norm):
    if norm == "max":
        mx = max(-float(np.min(funcs)), float(np.max(funcs)))
        return funcs / mx
    elif norm == "norm":
        nr = np.sqrt(np.sum(funcs * funcs, axis=0))
        return funcs / float(np.max(nr))
    return funcs


def _summarise(funcs, mode: str, n_comp: int):
    nc = funcs.shape[1]
    if mode == "max" or mode == "maxval":
        score = np.zeros(nc)
        for j in range(nc):
            if mode == "maxval":
                score[j] = max(-float(np.min(funcs[:, j])), float(np.max(funcs[:, j])))
            else:
                score[j] = np.sqrt(np.sum(funcs[:, j] ** 2))
        if n_comp > 1:
            idx = np.argsort(score)[-n_comp:]
        else:
            idx = np.array([int(np.argmax(score))], dtype=int)
        return funcs[:, idx].copy(), np.zeros(0)
    elif mode == "sum":
        return np.sum(funcs, axis=1).reshape((-1, 1)), np.zeros(0)
    elif mode == "mean":
        return np.mean(funcs, axis=1).reshape((-1, 1)), np.zeros(0)
    elif mode == "svd":
        u, s, _ = np.linalg.svd(funcs, full_matrices=False)
        s2 = s * s
        k = n_comp if n_comp > 1 else 1
        return u[:, :k].copy(), 100.0 * s2[:k] / float(np.sum(s2))
    raise ValueError(f"Invalid mode {mode}")


def _psf_ctf(resmat, src, source_nn, vertno: List[np.ndarray[int, 1]], kind: str,
             idx: List[List[int]], func: str, mode: str, n_comp: int, norm: str):
    nv = 0
    for v in vertno:
        nv += v.shape[0]
    rm = resmat.T.copy() if func == "ctf" else resmat
    n_r = rm.shape[0]
    n_c = rm.shape[1]
    if (nv != n_r and n_r != 3 * nv) or (nv != n_c and n_c != 3 * nv):
        raise ValueError(f"Number of vertices ({nv}) and corresponding dimension of resolution matrix ({resmat.shape[0]}, {resmat.shape[1]}) do not match")
    out = List[Tuple[np.ndarray[float, 2], np.ndarray[float, 1]]]()
    for verts in idx:
        cols = List[int]()
        for v in verts:
            if nv != n_c:
                cols.extend([3 * v, 3 * v + 1, 3 * v + 2])
            else:
                cols.append(v)
        funcs = rm[:, np.array(cols, dtype=int)].copy()
        funcs = _normalise(funcs, norm)
        pv = np.zeros(0)
        if mode != "":
            funcs, pv = _summarise(funcs, mode, n_comp)
        if nv != n_r:
            comb = np.empty((n_r // 3, funcs.shape[1]))
            for i in range(nv):
                comb[i] = np.sqrt(np.sum(funcs[3 * i:3 * i + 3] ** 2, axis=0))
            funcs = comb
        out.append((funcs, pv))
    return out


def get_point_spread(resmat, inv_or_fwd_src, idx, mode: str = "", *, n_comp: int = 1,
                     norm: str = "", verbose = None):
    """mne.minimum_norm.get_point_spread (scalar estimates).

    inv_or_fwd_src is an InverseOperator or Forward; idx is a list of int
    (one function set) or a list of lists. Returns one SourceEstimate, or a
    list when several index groups are given."""
    return _get_psf_ctf_api(resmat, inv_or_fwd_src, idx, "psf", mode, n_comp, norm)


def get_cross_talk(resmat, inv_or_fwd_src, idx, mode: str = "", *, n_comp: int = 1,
                   norm: str = "", verbose = None):
    """mne.minimum_norm.get_cross_talk (scalar estimates)."""
    return _get_psf_ctf_api(resmat, inv_or_fwd_src, idx, "ctf", mode, n_comp, norm)


def _groups(idx) -> List[List[int]]:
    g = List[List[int]]()
    if isinstance(idx, List[int]):
        g.append([int(i) for i in idx])
    else:
        for grp in idx:
            g.append([int(i) for i in grp])
    return g


def _get_psf_ctf_api(resmat, srcobj, idx, func: str, mode: str, n_comp: int, norm: str):
    if mode == "pca":
        mode = "svd"
    elif mode == "maxnorm":
        mode = "max"
    if (mode == "" or mode == "mean" or mode == "sum") and n_comp > 1:
        raise ValueError(f"n_comp must be 1 for mode={mode}.")
    src = srcobj.src
    vertno = [s.vertno.copy() for s in src.spaces]
    res = _psf_ctf(np.asarray(resmat, dtype=float), src, srcobj.source_nn, vertno, src.kind,
                   _groups(idx), func, mode, n_comp, norm)
    stcs = [_STC[np.ndarray[float, 2]](r[0], [v.copy() for v in vertno], 0.0, 1.0, None, src.kind)
            for r in res]
    if isinstance(idx, List[int]):
        return stcs[0]
    return stcs
