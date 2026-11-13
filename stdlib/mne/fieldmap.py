# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# make_field_map (sphere field interpolation, mne.forward._field_interpolation and _lead_dots).
# Deviation: the target surfaces are passed explicitly (surfaces={'meg': (rr, nn), 'eeg': (rr, nn)} in
# head coordinates) because the helmet / head surfaces come from data files; trans must be None.

import numpy as np
from ._info import Info
from ._containers import Evoked
from ._proj import make_projector, _has_eeg_average_ref_proj
from .cov import make_ad_hoc_cov
from .forward import _meg_coils, _eeg_els, _Coils


class _Leg:
    """Legendre (derivative) series state for one channel type."""
    ch_type: str
    n_coeff: int
    n_fact: np.ndarray[float, 2]

    def __init__(self, ch_type: str, n_coeff: int):
        self.ch_type = ch_type
        self.n_coeff = n_coeff
        nf = np.arange(1, n_coeff).astype(float)
        if ch_type == "meg":
            f0 = nf / (2.0 * nf + 1.0)
            f1 = f0 / (nf + 1.0)
            f2 = f0 * (nf + 1.0)
            m = np.zeros((n_coeff - 1, 4))
            m[:, 0] = f2
            m[:, 1] = f0
            m[:, 2] = f1
            m[:, 3] = f1
            self.n_fact = m
        else:
            e = (2.0 * nf + 1.0) * (2.0 * nf + 1.0) / nf
            self.n_fact = e.reshape((n_coeff - 1, 1))

    def sums_meg(self, beta: np.ndarray[float, 1], ct: np.ndarray[float, 1]) -> np.ndarray[float, 2]:
        n = beta.shape[0]
        nc = self.n_coeff
        out = np.zeros((4, n))
        for i in range(n):
            x = float(ct[i])
            p0 = np.zeros(nc)
            p0d = np.zeros(nc)
            p0dd = np.zeros(nc)
            p0[0] = 1.0
            p0[1] = x
            p0d[1] = 1.0
            for k in range(2, nc):
                p0[k] = ((2 * k - 1) * x * p0[k - 1] - (k - 1) * p0[k - 2]) / k
                p0d[k] = k * p0[k - 1] + x * p0d[k - 1]
                p0dd[k] = (k + 1) * p0d[k - 1] + x * p0dd[k - 1]
            b = float(beta[i])
            bp = b
            for j in range(nc - 1):
                kk = j + 1
                out[0, i] += bp * self.n_fact[j, 0] * p0[kk]
                out[1, i] += bp * self.n_fact[j, 1] * p0d[kk]
                out[2, i] += bp * self.n_fact[j, 2] * p0d[kk]
                out[3, i] += bp * self.n_fact[j, 3] * p0dd[kk]
                bp *= b
        # series columns: P_n, P_n', P_n', P_n'' (as mne's _eval_legen_meg)
        return out

    def sum_eeg(self, beta: np.ndarray[float, 1], ct: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
        n = beta.shape[0]
        nc = self.n_coeff
        out = np.zeros(n)
        for i in range(n):
            x = float(ct[i])
            p0 = np.zeros(nc)
            p0[0] = 1.0
            p0[1] = x
            for k in range(2, nc):
                p0[k] = ((2 * k - 1) * x * p0[k - 1] - (k - 1) * p0[k - 2]) / k
            b = float(beta[i])
            bp = b
            s = 0.0
            for j in range(nc - 1):
                s += bp * self.n_fact[j, 0] * p0[j + 1]
                bp *= b
            out[i] = s
        return out


_MEG_CONST: float = 4e-14 * np.pi
_EEG_CONST: float = 1.0 / (4.0 * np.pi)


def _norm_rows(rm: np.ndarray[float, 2], r0: np.ndarray[float, 1]):
    d = rm - r0[None, :]
    ln = np.sqrt(np.sum(d * d, axis=1))
    return d / ln[:, None], ln


def _sphere_dot(r: float, rr1: np.ndarray[float, 2], lr1: np.ndarray[float, 1], cos1: np.ndarray[float, 2],
                w1: Optional[np.ndarray[float, 1]], rr2: np.ndarray[float, 2], lr2: np.ndarray[float, 1],
                cos2: np.ndarray[float, 2], w2: np.ndarray[float, 1], leg: _Leg) -> np.ndarray[float, 2]:
    """result (n1, n2) of _fast_sphere_dot_r0 before the per-coil reduction (w1 handled by caller)."""
    n1 = rr1.shape[0]
    n2 = rr2.shape[0]
    ct = np.clip(np.dot(rr1, rr2.T), -1.0, 1.0)
    lr1lr2 = lr1[:, None] * lr2[None, :]
    beta = (r * r) / lr1lr2
    bf = beta.reshape(n1 * n2)
    cf = ct.reshape(n1 * n2)
    res = np.zeros((n1, n2))
    if leg.ch_type == "meg":
        sums = leg.sums_meg(bf, cf)
        s0 = sums[0].reshape((n1, n2))
        s1 = sums[1].reshape((n1, n2))
        s2 = sums[2].reshape((n1, n2))
        s3 = sums[3].reshape((n1, n2))
        # a[i, j] = cosmag1_i . rr1_i (broadcast over j)
        a = np.sum(cos1 * rr1, axis=1)[:, None] * np.ones((1, n2))
        n1c2 = np.dot(cos1, rr2.T)
        n2c1 = np.dot(rr1, cos2.T)
        n2c2 = (np.sum(cos2 * rr2, axis=1))[None, :] * np.ones((n1, 1))
        n1n2 = np.dot(cos1, cos2.T)
        part1 = ct * a * n2c2
        part2 = a * n2c1 + n1c2 * n2c2
        res = (a * n2c2 * s0 + (2.0 * part1 - part2) * s1 + (n1n2 + part1 - part2) * s2 +
               (n1c2 - ct * a) * (n2c1 - ct * n2c2) * s3)
        res = res * _MEG_CONST / lr1lr2
    else:
        s = leg.sum_eeg(bf, cf).reshape((n1, n2))
        res = s * _EEG_CONST / lr1lr2
    res = res * w2[None, :]
    if w1 is not None:
        res = res * w1.__val__()[:, None]
    return res


def _coil_parts(c: _Coils, r0: np.ndarray[float, 1]):
    """Per-coil lists of (unit rmag, length, cosmag, w)."""
    rl = List[np.ndarray[float, 2]]()
    ll = List[np.ndarray[float, 1]]()
    cl = List[np.ndarray[float, 2]]()
    wl = List[np.ndarray[float, 1]]()
    for k in range(c.n):
        idx = [i for i in range(c.bins.shape[0]) if int(c.bins[i]) == k]
        ia = np.array(idx, dtype=int)
        rn, ln = _norm_rows(c.rmags[ia], r0)
        rl.append(rn)
        ll.append(ln)
        cl.append(c.cosmags[ia])
        wl.append(c.ws[ia])
    return rl, ll, cl, wl


def _self_dots(r: float, c: _Coils, r0, leg: _Leg) -> np.ndarray[float, 2]:
    rl, ll, cl, wl = _coil_parts(c, r0)
    n = c.n
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            m = _sphere_dot(r, rl[i], ll[i], cl[i], wl[i], rl[j], ll[j], cl[j], wl[j], leg)
            v = float(np.sum(m))
            out[i, j] = v
            out[j, i] = v
    return out


def _surface_dots(r: float, c: _Coils, rr_s: np.ndarray[float, 2], nn_s: np.ndarray[float, 2], r0,
                  leg: _Leg) -> np.ndarray[float, 2]:
    rl, ll, cl, wl = _coil_parts(c, r0)
    rs, ls = _norm_rows(rr_s, r0)
    out = np.zeros((rr_s.shape[0], c.n))
    for j in range(c.n):
        m = _sphere_dot(r, rs, ls, nn_s, None, rl[j], ll[j], cl[j], wl[j], leg)
        for i in range(rr_s.shape[0]):
            out[i, j] = float(np.sum(m[i]))
    return out


def _reg_pinv_tsvd(x: np.ndarray[float, 2], rank: int):
    U, s, Vh = np.linalg.svd(x, hermitian=True)
    n = s.shape[0]
    sinv = np.zeros(n)
    for i in range(n):
        if i < rank and s[i] > 0:
            sinv[i] = 1.0 / s[i]
    return np.dot(U * sinv[None, :], Vh)


def make_field_map(evoked: Evoked, trans = None, subject = None, subjects_dir = None, ch_type = None,
                   mode: str = "fast", meg_surf: str = "helmet", origin = (0.0, 0.0, 0.04), n_jobs = None,
                   surfaces = None, verbose = None):
    """mne.make_field_map. Returns a list of dicts {'kind', 'ch_names', 'data' (n_vertices, n_sensors),
    'origin'} (one for 'meg' and/or 'eeg'). `surfaces` = {'meg': (rr, nn), 'eeg': (rr, nn)}."""
    if surfaces is None:
        raise NotImplementedError("make_field_map needs surfaces={'meg': (rr, nn), ...} (no helmet / head surface files)")
    if not isinstance(trans, NoneType):
        raise NotImplementedError("make_field_map: only trans=None is supported")
    info = evoked.info
    types = List[str]()
    if isinstance(ch_type, NoneType):
        has_meg = False
        has_eeg = False
        for t in info.ch_types:
            if t == "mag" or t == "grad":
                has_meg = True
            if t == "eeg":
                has_eeg = True
        if has_meg:
            types.append("meg")
        if has_eeg:
            types.append("eeg")
    else:
        types.append(str(ch_type))
    if "eeg" in types:
        types.remove("eeg")
    r0 = np.array([float(o) for o in origin])
    out = List[Dict[str, np.ndarray[float, 2]]]()
    for ty in types:
        picks = List[int]()
        for i in range(info.nchan):
            t = info.ch_types[i]
            if ty == "meg" and (t == "mag" or t == "grad"):
                picks.append(i)
            elif ty == "eeg" and t == "eeg":
                picks.append(i)
        if len(picks) == 0:
            raise RuntimeError("cannot map, no channels found")
        rr_s = np.asarray(surfaces[ty][0], dtype=float)
        nn_s = np.asarray(surfaces[ty][1], dtype=float)
        coils = _meg_coils(info, picks) if ty == "meg" else _eeg_els(info, picks)
        leg = _Leg(ty, 50 if mode == "fast" else 100)
        miss = 1e-4 if ty == "meg" else 1e-3
        sd = _self_dots(0.06, coils, r0, leg)
        fd = _surface_dots(0.06, coils, rr_s, nn_s, r0, leg)
        names = [info.ch_names[p] for p in picks]
        sub = info._pick(picks)
        proj_op = make_projector(sub.projs, names)[0]
        proj_dots = np.dot(proj_op.T, np.dot(sd, proj_op))
        std = np.zeros(len(picks))
        for k in range(len(picks)):
            t = info.ch_types[picks[k]]
            std[k] = 20e-15 if t == "mag" else (5e-13 if t == "grad" else 1e-6)
        W = np.diag(1.0 / std)
        wd = np.dot(W.T, np.dot(proj_dots, W))
        s = np.linalg.svd(wd, compute_uv=False, hermitian=True)
        varexp = np.cumsum(s)
        varexp = varexp / varexp[-1]
        rank = 1
        for i in range(varexp.shape[0]):
            if varexp[i] >= 1.0 - miss:
                rank = i + 1
                break
        inv = _reg_pinv_tsvd(wd, rank)
        inv_w = np.dot(W.T, np.dot(inv, W))
        inv_wp = np.dot(proj_op.T, inv_w)
        mapping = np.dot(fd, inv_wp)
        if ty == "eeg" and _has_eeg_average_ref_proj(sub):
            mapping = mapping - np.mean(mapping, axis=0)
        d = Dict[str, np.ndarray[float, 2]]()
        d["data"] = mapping
        d["origin"] = r0.reshape((1, 3))
        out.append(d)
    return out
