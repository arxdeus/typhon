# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# average_forward_solutions, pick_types_forward, sensitivity_map.

import numpy as np
from .forward import Forward, convert_forward_solution, is_fixed_orient, pick_channels_forward
from .source_estimate import _STC
from ._proj import make_projector, make_eeg_average_ref_proj, _has_eeg_average_ref_proj
from ._info import Projection


def average_forward_solutions(fwds, weights = None, verbose = None) -> Forward:
    """mne.average_forward_solutions."""
    if len(fwds) == 0:
        raise ValueError("fwds must not be empty")
    w = np.ones(len(fwds)) if isinstance(weights, NoneType) else np.asarray(weights, dtype=float).copy()
    if not (w >= 0).all():
        raise ValueError("weights must be non-negative")
    if w.shape[0] != len(fwds):
        raise ValueError("weights must be None or the same length as fwds")
    s = float(np.sum(w))
    if not s > 0:
        raise ValueError("weights cannot all be zero")
    w = w / s
    f0 = fwds[0]
    for k in range(1, len(fwds)):
        f = fwds[k]
        if f.sol.shape[0] != f0.sol.shape[0] or f.sol.shape[1] != f0.sol.shape[1]:
            raise ValueError("Forward solutions have incompatible dimensions")
        if f.source_ori != f0.source_ori or f.surf_ori != f0.surf_ori or f.coord_frame != f0.coord_frame:
            raise ValueError("Forward solutions have incompatible orientations")
    out = f0.copy()
    out.sol = f0.sol * w[0]
    out._orig_sol = f0._orig_sol * w[0]
    for k in range(1, len(fwds)):
        out.sol = out.sol + w[k] * fwds[k].sol
        out._orig_sol = out._orig_sol + w[k] * fwds[k]._orig_sol
    return out


def pick_types_forward(orig: Forward, meg = False, eeg: bool = False, ref_meg: bool = True, seeg: bool = False,
                       ecog: bool = False, dbs: bool = False, include = (), exclude = ()) -> Forward:
    """mne.pick_types_forward (meg: bool | 'mag' | 'grad')."""
    info = orig.info
    sel = List[str]()
    ex = List[str]()
    if isinstance(exclude, str):
        if exclude == "bads":
            ex = list(info.bads)
    else:
        for e in exclude:
            ex.append(str(e))
    for i in range(info.nchan):
        t = info.ch_types[i]
        n = info.ch_names[i]
        ok = False
        if isinstance(meg, bool):
            ok = (meg and (t == "mag" or t == "grad"))
        else:
            ok = (t == str(meg))
        if eeg and t == "eeg":
            ok = True
        if seeg and t == "seeg":
            ok = True
        if ecog and t == "ecog":
            ok = True
        if dbs and t == "dbs":
            ok = True
        if ok and n not in ex:
            sel.append(n)
    for n in include:
        if str(n) not in sel:
            sel.append(str(n))
    if len(sel) == 0:
        raise ValueError("No valid channels found")
    return pick_channels_forward(orig, sel)


def sensitivity_map(fwd: Forward, projs = None, ch_type: str = "grad", mode: str = "fixed", exclude = (),
                    verbose = None):
    """mne.sensitivity_map: returns a SourceEstimate (n_sources, 1)."""
    if ch_type != "eeg" and ch_type != "grad" and ch_type != "mag":
        raise ValueError(f"ch_type must be one of eeg, grad, mag, got {ch_type}")
    if mode not in ("free", "fixed", "ratio", "radiality", "angle", "remaining", "dampening"):
        raise ValueError(f"mode must be one of free, fixed, ratio, radiality, angle, remaining, dampening, got {mode}")
    if is_fixed_orient(fwd, True):
        raise ValueError("fwd should must be computed with free orientation")
    f2 = pick_types_forward(fwd, meg=(False if ch_type == "eeg" else ch_type), eeg=(ch_type == "eeg"), exclude=exclude)
    f2 = convert_forward_solution(f2, surf_ori=True, force_fixed=False, copy=False)
    gain = f2.sol
    pr = List[Projection]()
    have_proj = False
    if not isinstance(projs, NoneType):
        for p in projs:
            pr.append(p)
        have_proj = True
    if ch_type == "eeg":
        have_ref = False
        for p in pr:
            if p.kind == 10:
                have_ref = True
        if not have_ref:
            pr.append(make_eeg_average_ref_proj(f2.info))
        have_proj = True
    residual = mode == "angle" or mode == "remaining" or mode == "dampening"
    proj = np.eye(gain.shape[0])
    U = np.zeros((gain.shape[0], 0))
    ncomp = 0
    if have_proj:
        proj, ncomp, U = make_projector(pr, f2.ch_names, (), True)
        if not residual:
            gain = np.dot(proj, gain)
        elif ncomp == 0:
            raise RuntimeError(f"No valid projectors found for channel type {ch_type}, cannot compute {mode}")
    elif residual:
        raise ValueError(f"No projectors used, cannot compute {mode}")
    nloc = gain.shape[1] // 3
    sm = np.empty(nloc)
    for k in range(nloc):
        gg = gain[:, 3 * k:3 * (k + 1)]
        s = np.zeros(1)
        if mode != "fixed":
            s = np.linalg.svd(gg, compute_uv=False)
        gz = float(np.linalg.norm(gg[:, 2]))
        if mode == "free":
            sm[k] = s[0]
        elif mode == "fixed":
            sm[k] = gz
        elif mode == "ratio":
            sm[k] = gz / s[0]
        elif mode == "radiality":
            sm[k] = 1.0 - gz / s[0]
        elif mode == "angle":
            co = float(np.linalg.norm(np.dot(gg[:, 2], U)))
            sm[k] = co / gz
        else:
            p = float(np.linalg.norm(np.dot(proj, gg[:, 2])))
            if mode == "remaining":
                sm[k] = p / gz
            else:
                sm[k] = 1.0 - p / gz
    if mode == "fixed" or mode == "free":
        sm = sm / float(np.max(sm))
    verts = List[np.ndarray[int, 1]]()
    for sp in f2.src.spaces:
        verts.append(sp.vertno)
    kind = "surface" if f2.src.kind == "surface" else "volume"
    return _STC[np.ndarray[float, 2]](sm.reshape((nloc, 1)), verts, 0.0, 1.0, None, kind)
