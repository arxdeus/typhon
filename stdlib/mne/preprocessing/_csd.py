# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# compute_current_source_density: spherical spline surface Laplacian.

import numpy as np
from .._info import Info
from ..channels import _calc_g, _calc_h, _fit_sphere


def _csd_matrix(info: Info, picks: List[int], sphere, lambda2: float, stiffness: float,
                n_legendre_terms: int) -> np.ndarray[float, 2]:
    if lambda2 < 0 or lambda2 >= 1:
        raise ValueError(f"lambda2 must be between 0 and 1, got {lambda2}")
    if stiffness < 0:
        raise ValueError(f"stiffness must be non-negative got {stiffness}")
    if n_legendre_terms < 1:
        raise ValueError(f"n_legendre_terms must be greater than 0, got {n_legendre_terms}")
    n = len(picks)
    raw_pos = np.zeros((n, 3))
    for k in range(n):
        for j in range(3):
            raw_pos[k, j] = info.locs[picks[k]][j]
    ox = 0.0
    oy = 0.0
    oz = 0.0
    radius = 0.0
    if isinstance(sphere, str):
        if info._montage is None:
            raise RuntimeError('Cannot fit headshape without digitization, info["dig"] is None')
        m = info._montage.__val__()
        hsp = List[List[float]]()
        for p in m.pos:
            if not (p[2] < -1e-6 and p[1] > 1e-6):
                hsp.append(p)
        r, o = _fit_sphere(np.array(hsp))
        radius = r
    else:
        s = np.array([float(v) for v in sphere])
        ox = s[0]
        oy = s[1]
        oz = s[2]
        radius = s[3]
    if radius <= 0:
        raise ValueError(f"sphere radius must be greater than 0, got {radius}")
    pos = np.zeros((n, 3))
    for k in range(n):
        for j in range(3):
            if not np.isfinite(raw_pos[k, j]):
                raise ValueError("Zero or infinite position found in chs")
        if np.isclose(raw_pos[k, 0], 0.0) and np.isclose(raw_pos[k, 1], 0.0) and np.isclose(raw_pos[k, 2], 0.0):
            raise ValueError("Zero or infinite position found in chs")
        pos[k, 0] = raw_pos[k, 0] - ox
        pos[k, 1] = raw_pos[k, 1] - oy
        pos[k, 2] = raw_pos[k, 2] - oz
        nr = np.sqrt(pos[k, 0] ** 2 + pos[k, 1] ** 2 + pos[k, 2] ** 2)
        for j in range(3):
            pos[k, j] /= nr
    cos_dist = np.clip(np.dot(pos, pos.T), -1.0, 1.0)
    G = _calc_g(cos_dist, stiffness, n_legendre_terms)
    H = _calc_h(cos_dist, stiffness, n_legendre_terms)
    for i in range(n):
        G[i, i] += lambda2
    Gi = np.linalg.inv(G)
    TC = Gi.sum(axis=0)
    sgi = TC.sum()
    Z = np.eye(n) - (1.0 / n)
    Cp2 = np.dot(Gi, Z)
    c02 = Cp2.sum(axis=0) / sgi
    C2 = Cp2 - np.dot(TC.reshape((n, 1)), c02.reshape((1, n)))
    return np.dot(C2.T, H).T / (radius ** 2)


def _csd_picks(info: Info) -> List[int]:
    out = List[int]()
    for i in range(info.nchan):
        if info.ch_types[i] == "eeg" and info.ch_names[i] not in info.bads:
            out.append(i)
    return out


def _csd_apply_info(info: Info, picks: List[int]):
    for p in picks:
        info.ch_types[p] = "csd"
        info.coil_types[p] = 8
        info.units[p] = 20


def compute_current_source_density(inst, sphere = "auto", lambda2: float = 1e-5, stiffness: float = 4.0,
                                   n_legendre_terms: int = 50, copy: bool = True, *, verbose = None):
    """Surface Laplacian (CSD) of Raw/Epochs/Evoked EEG data (mne 1.13.2)."""
    out = inst.copy() if copy else inst
    picks = _csd_picks(out.info)
    if len(picks) == 0:
        raise ValueError("No EEG channels found.")
    T = _csd_matrix(out.info, picks, sphere, float(lambda2), float(stiffness), n_legendre_terms)
    pa = np.array(picks, dtype=int)
    if hasattr(out, "events"):
        for e in range(out._data.shape[0]):
            sub = np.dot(T, out._data[e][pa, :])
            for k in range(len(picks)):
                out._data[e, picks[k], :] = sub[k]
    elif hasattr(out, "nave"):
        sub = np.dot(T, out.data[pa, :])
        for k in range(len(picks)):
            out.data[picks[k], :] = sub[k]
    else:
        sub = np.dot(T, out._data[pa, :])
        for k in range(len(picks)):
            out._data[picks[k], :] = sub[k]
    _csd_apply_info(out.info, picks)
    return out
