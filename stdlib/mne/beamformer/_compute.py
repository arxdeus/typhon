# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# Shared beamformer machinery (mne.beamformer._compute_beamformer): input
# preparation, regularized pseudo-inverse, LCMV/DICS weights, source power
# and the Beamformer container. Sources are processed one at a time with 2-D
# linear algebra (mne uses stacked 3-D arrays); results are identical.

import numpy as np
from .._info import Info
from .._proj import make_projector
from ..cov import (Covariance, prepare_noise_cov, compute_whitener, _check_rank, _make_scalings,
                   _rank_of_cov, _Rank)
from ..forward import (Forward, convert_forward_solution, pick_channels_forward, is_fixed_orient,
                       compute_depth_prior, compute_orient_prior, _triage_loose)
from ..source_space import SourceSpaces
from ..time_frequency.csd import CrossSpectralDensity


def _herm[T](a: np.ndarray[T, 2]):
    """Conjugate transpose (plain transpose for real arrays)."""
    if isinstance(a, np.ndarray[complex, 2]):
        return a.conj().T.copy()
    else:
        return a.T.copy()


def _like[T](a, ref: np.ndarray[T, 2]):
    """Copy of the 2-D real array `a` with the element type of `ref`."""
    out = np.zeros(a.shape, dtype=T)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            out[i, j] = a[i, j]
    return out


def _cols_scaled[T](u: np.ndarray[T, 2], s: np.ndarray[float, 1]):
    out = u.copy()
    for j in range(u.shape[1]):
        out[:, j] = u[:, j] * s[j]
    return out


# ---------------------------------------------------------------------------
# option helpers

def _opt_str(x) -> str:
    """None -> '' else str(x) (static string options)."""
    if isinstance(x, NoneType):
        return ""
    return str(x)


def _check_option(name: str, value: str, allowed: List[str]):
    if value not in allowed:
        raise ValueError(f"Invalid value for the '{name}' parameter. Allowed values are {allowed}, but got {repr(value)} instead.")


def _ad_hoc_unit(info: Info) -> Covariance:
    """make_ad_hoc_cov(info, std=1.0)."""
    names = List[str]()
    vals = List[float]()
    for i in range(info.nchan):
        t = info.ch_types[i]
        if t == "mag" or t == "grad" or t == "eeg":
            names.append(info.ch_names[i])
            vals.append(0.0 if info.ch_names[i] in info.bads else 1.0)
    return Covariance(np.array(vals), names, list(info.bads), info.projs, 0)


def _check_info_inv(info: Info, fwd_names: List[str], ex_names: List[List[str]],
                    ex_bads: List[List[str]]) -> List[int]:
    """Good channels common to forward, info and covariances (indices into info)."""
    fset = Set[str](fwd_names)
    names = List[str]()
    for c in info.ch_names:
        if c not in info.bads and c in fset:
            names.append(c)
    refs = Set[str]()
    for i in range(info.nchan):
        if info.ch_types[i] == "ref_meg":
            refs.add(info.ch_names[i])
    names = [c for c in names if c not in refs]
    for k in range(len(ex_names)):
        nset = Set[str](ex_names[k])
        bset = Set[str](ex_bads[k])
        names = [c for c in names if c not in bset and c in nset]
    return [info.ch_names.index(c) for c in names]


def _n_ch_types(info: Info) -> int:
    seen = Set[str]()
    for t in info.ch_types:
        if t == "mag" or t == "grad" or t == "eeg":
            seen.add(t)
    return len(seen)


def _check_channels_spatial_filter(ch_names: List[str], filt_names: List[str]) -> List[int]:
    for c in filt_names:
        if c not in ch_names:
            raise ValueError(f"The spatial filter was computed with channel {c} which is not present in the data. You should compute a new spatial filter restricted to the good data channels.")
    fs = Set[str](filt_names)
    return [i for i in range(len(ch_names)) if ch_names[i] in fs]


# ---------------------------------------------------------------------------
# rank of a (possibly complex) covariance-like matrix

def _rank_matrix[T](C: np.ndarray[T, 2], info: Info, rank: _Rank, ch_names: List[str]) -> Dict[str, int]:
    """mne.rank._compute_rank for a square matrix whose channels are
    `ch_names` (all present in info); complex input supported."""
    sc = _make_scalings(None, "cov_rank")
    out = Dict[str, int]()
    for k, v in rank.given.items():
        out[k] = v
    sel = [info.ch_names.index(n) for n in ch_names]
    sub = info._pick(sel)
    sub.bads = List[str]()
    for ct_picks in _picks_by_type_meg(sub):
        ch_type = ct_picks[0]
        picks = ct_picks[1]
        if ch_type in out:
            continue
        names = [sub.ch_names[p] for p in picks]
        proj_op, n_proj, _ = make_projector(sub.projs, names)
        this_rank = 0
        if rank.kind == "info" or rank.kind == "full":
            this_rank = len(picks)
            if rank.kind != "full":
                this_rank -= n_proj
        else:
            n = len(picks)
            data = np.zeros((n, n), dtype=T)
            for i in range(n):
                for j in range(n):
                    data[i, j] = C[picks[i], picks[j]]
            P = _like(proj_op, data)
            data = np.dot(np.dot(P, data), _herm(P))
            sv = np.array([sc.get(sub.ch_types[p]) for p in picks])
            for i in range(n):
                for j in range(n):
                    data[i, j] = data[i, j] * (sv[i] * sv[j])
            s = np.linalg.svd(data, compute_uv=False)
            mx = float(np.amax(s))
            tol = s.shape[0] * mx * 2.220446049250313e-16
            this_rank = int(np.sum(s > tol))
        out[ch_type] = this_rank
    return out


def _picks_by_type_meg(info: Info) -> List[Tuple[str, List[int]]]:
    from .._info import _picks_by_type
    return _picks_by_type(info, meg_combined=True, exclude_bads=False)


def _sum_rank(r: Dict[str, int]) -> int:
    n = 0
    for k, v in r.items():
        n += v
    return n


# ---------------------------------------------------------------------------
# forward preparation

class _BfInput:
    is_free_ori: bool
    info: Info
    proj: np.ndarray[float, 2]
    vertno: List[np.ndarray[int, 1]]
    gain: np.ndarray[float, 2]
    whitener: np.ndarray[float, 2]
    nn: np.ndarray[float, 2]
    orient_std: np.ndarray[float, 1]
    fwd: Forward

    def __init__(self, is_free_ori: bool, info: Info, proj, vertno: List[np.ndarray[int, 1]], gain,
                 whitener, nn, orient_std, fwd: Forward):
        self.is_free_ori = is_free_ori
        self.info = info
        self.proj = proj
        self.vertno = vertno
        self.gain = gain
        self.whitener = whitener
        self.nn = nn
        self.orient_std = orient_std
        self.fwd = fwd


def _prepare_beamformer_input(info: Info, forward: Forward, pick_ori: str, noise_cov: Covariance,
                              rank, exp: Optional[float] = None, combine_xyz: str = "fro",
                              limit_depth_chs: str = "whiten", allow_fixed_depth: bool = True,
                              loose: float = -1.0, use_cps: bool = True) -> _BfInput:
    """Input preparation common for LCMV, DICS and RAP-MUSIC (pca=False)."""
    _check_option("pick_ori", pick_ori, ["", "normal", "max-power", "vector"])
    fwd = forward.copy()
    lo0 = loose
    if lo0 < 0.0:
        lo0 = 0.0 if is_fixed_orient(fwd) else 1.0
    lo = _triage_loose(fwd.src, lo0, "auto")
    fixed_inverse = lo == 0.0
    constrained = lo < 1.0
    has_exp = False
    expv = 0.0
    if exp is not None:
        expv = exp.__val__()
        if expv < 0:
            raise ValueError(f"depth exponent should be greater than or equal to 0, got {expv}")
        has_exp = expv != 0.0
    if fixed_inverse:
        if not is_fixed_orient(fwd):
            if allow_fixed_depth:
                fwd = convert_forward_solution(fwd, force_fixed=True, use_cps=True, copy=False)
        elif has_exp and not allow_fixed_depth:
            raise ValueError("For a fixed orientation inverse solution with depth weighting, the forward solution must be free-orientation and in surface orientation")
    else:
        if is_fixed_orient(fwd):
            raise ValueError("Forward operator has fixed orientation and can only be used to make a fixed-orientation inverse operator.")
        if constrained and not fwd.surf_ori:
            fwd = convert_forward_solution(fwd, surf_ori=True, use_cps=use_cps, copy=False)
    # _select_orient_forward
    bad = Set[str](info.bads)
    for b in noise_cov.bads:
        bad.add(b)
    fnames = Set[str](fwd.ch_names)
    nnames = Set[str](noise_cov.names)
    ch_names = List[str]()
    for c in info.ch_names:
        if c not in bad and c in fnames and c in nnames:
            ch_names.append(c)
    fwd = pick_channels_forward(fwd, ch_names, (), True, True)
    info_picked = info._pick([info.ch_names.index(c) for c in ch_names])
    depth_prior = np.ones(0)
    if has_exp:
        depth_prior = compute_depth_prior(fwd, info_picked, expv, None, limit_depth_chs, combine_xyz,
                                          noise_cov, rank)
    orient_prior = np.ones(0)
    has_orient = False
    if fixed_inverse:
        if not is_fixed_orient(fwd):
            if has_exp:
                nd = depth_prior.shape[0] // 3
                dp = np.zeros(nd)
                for i in range(nd):
                    dp[i] = depth_prior[3 * i + 2]
                depth_prior = dp
            fwd = convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=use_cps, copy=False)
    else:
        orient_prior = compute_orient_prior(fwd, lo)
        has_orient = True
    nc = prepare_noise_cov(noise_cov, info, ch_names, rank)
    whitener, _ = compute_whitener(nc, info, ch_names, rank=rank, pca=False)
    gain = np.dot(whitener, fwd.sol)
    source_std = np.ones(gain.shape[1])
    if has_exp:
        source_std = source_std * depth_prior
    if has_orient:
        source_std = source_std * orient_prior
    source_std = np.sqrt(source_std)
    for j in range(gain.shape[1]):
        gain[:, j] = gain[:, j] * source_std[j]
    trace_GRGT = float(np.linalg.norm(gain, "fro")) ** 2
    n_nzero = int(np.sum(nc.eig.__val__() > 0))
    scale = np.sqrt(n_nzero / trace_GRGT)
    gain = gain * scale
    # undo the scaling that MNE prefers
    gain = gain / np.sqrt(n_nzero / trace_GRGT)
    is_free = not is_fixed_orient(fwd)
    nsrc = fwd.nsource
    nn = np.zeros((nsrc, 3))
    for i in range(nsrc):
        for q in range(3):
            nn[i, q] = fwd.source_nn[3 * i + 2, q] if is_free else fwd.source_nn[i, q]
    if fwd.surf_ori:
        for i in range(nsrc):
            nn[i, 0] = 0.0
            nn[i, 1] = 0.0
            nn[i, 2] = 1.0
    vertno = List[np.ndarray[int, 1]]()
    for s in fwd.src.spaces:
        vertno.append(s.vertno.copy())
    if pick_ori != "" and not is_free:
        raise ValueError(f"Normal or max-power orientation (got {repr(pick_ori)}) can only be picked when a forward operator with free orientation is used.")
    if pick_ori == "normal" and not fwd.surf_ori:
        raise ValueError("Normal orientation can only be picked when a forward operator oriented in surface coordinates is used.")
    if pick_ori == "normal" and fwd.src.kind != "surface" and fwd.src.kind != "discrete":
        raise RuntimeError(f"Normal source orientation is supported only for surface or discrete SourceSpaces, got type {fwd.src.kind}")
    orient_std = np.ones(gain.shape[1])
    if has_orient:
        orient_std = np.sqrt(orient_prior)
    proj, _, _ = make_projector(info_picked.projs, info_picked.ch_names)
    return _BfInput(is_free, info_picked, proj, vertno, gain, whitener, nn, orient_std, fwd)


# ---------------------------------------------------------------------------
# linear algebra helpers

def _sym_mat_pow[T](A: np.ndarray[T, 2], power: float, reduce_rank: bool = False):
    """Exponentiate a Hermitian matrix (power in -1, 0.5, -0.5)."""
    w, u = np.linalg.eigh(A)
    s = w.copy()
    limit = s[s.shape[0] - 1] * 1e-7
    for i in range(s.shape[0]):
        if s[i] < -limit:
            raise ValueError("Matrix is not positive semi-definite")
    for i in range(s.shape[0]):
        if s[i] <= limit:
            s[i] = np.inf if power < 0 else 0.0
    if reduce_rank:
        s[0] = np.inf
    if power == -0.5 or power == 0.5:
        s = np.sqrt(s)
    use_s = np.zeros(s.shape[0])
    for i in range(s.shape[0]):
        use_s[i] = 1.0 / s[i] if power < 0 else s[i]
    return np.dot(_cols_scaled(u, use_s), _herm(u))


def _reg_pinv[T](x: np.ndarray[T, 2], reg: float, rank: int):
    """Regularized pseudo-inverse of a Hermitian matrix (integer rank).
    Returns (x_inv, loading_factor)."""
    n = x.shape[0]
    if rank > n:
        raise ValueError(f"Invalid value for the rank parameter ({rank}) given the shape of the input matrix ({n} x {n}).")
    u, s, vh = np.linalg.svd(x, hermitian=True)
    loading = reg * float(np.mean(s))
    if reg != 0.0:
        xr = x.copy()
        for i in range(n):
            xr[i, i] = xr[i, i] + loading
        u, s, vh = np.linalg.svd(xr, hermitian=True)
    s_inv = np.zeros(n)
    for i in range(n):
        if i < rank and s[i] > 0:
            s_inv[i] = 1.0 / s[i]
    return np.dot(_cols_scaled(u, s_inv), vh), loading


def _sym_inv_sm[T](x: np.ndarray[T, 2], reduce_rank: bool, inversion: str, sk: np.ndarray[float, 1]):
    """Symmetric inversion with single- or matrix-style inversion (one source)."""
    k = x.shape[0]
    out = np.zeros((k, k), dtype=T)
    if k == 1:
        v = x[0, 0]
        if v == 0:
            out[0, 0] = 1.0
        else:
            out[0, 0] = 1.0 / v
        return out
    if inversion == "matrix":
        out = _sym_mat_pow(x, -1.0, reduce_rank)
        for i in range(k):
            for j in range(k):
                out[i, j] = out[i, j] * sk[i] * sk[j]
    else:
        for i in range(k):
            out[i, i] = (1.0 / x[i, i]) * (sk[i] * sk[i])
    return out


def _compute_beamformer[T](G: np.ndarray[float, 2], Cm0: np.ndarray[T, 2], reg: float, n_orient0: int,
                           weight_norm: str, pick_ori: str, reduce_rank: bool, rank: int,
                           inversion: str, nn: np.ndarray[float, 2], orient_std: np.ndarray[float, 1],
                           whitener: np.ndarray[float, 2]):
    """Compute a spatial beamformer filter (LCMV or DICS).

    Returns (W (n_sources * n_orient, n_channels), max_power_ori (n_sources, 3) or
    zeros(0, 3)). weight_norm / pick_ori use '' for None."""
    _check_option("weight_norm", weight_norm, ["unit-noise-gain-invariant", "unit-noise-gain", "nai", ""])
    wh = _like(whitener, Cm0)
    Cm = np.dot(np.dot(wh, Cm0), _herm(wh))
    Cm = (Cm + _herm(Cm)) / 2.0
    n_channels = G.shape[0]
    Cm_inv, loading_factor = _reg_pinv(Cm, reg, rank)
    n_orient = n_orient0
    n_sources = G.shape[1] // n_orient
    _check_option("reduce_rank", "x", ["x"])
    _check_option("inversion", inversion, ["matrix", "single"])
    if inversion == "single" and n_orient > 1 and pick_ori == "vector" and weight_norm == "unit-noise-gain-invariant":
        raise ValueError('Cannot use pick_ori="vector" with inversion="single" and weight_norm="unit-noise-gain-invariant"')
    if reduce_rank and inversion == "single":
        raise ValueError('reduce_rank cannot be used with inversion="single"; consider using inversion="matrix" if you have a rank-deficient forward model (i.e., from a sphere model with MEG channels), otherwise consider using reduce_rank=False')
    CmInv2 = np.dot(Cm_inv, Cm_inv)
    n_out = 1 if (pick_ori == "max-power" or pick_ori == "normal") else n_orient0
    outW = np.zeros((n_sources * n_out, n_channels), dtype=T)
    mpo = np.zeros((n_sources, 3))
    for src in range(n_sources):
        Gk = G[:, src * n_orient0:(src + 1) * n_orient0].copy()
        sk = orient_std[src * n_orient0:(src + 1) * n_orient0].copy()
        no = n_orient0
        if no > 1:
            sv = np.linalg.svd(Gk, compute_uv=False)
            if (not reduce_rank) and sv[0] > 1e6 * sv[2]:
                raise ValueError("Singular matrix detected when estimating spatial filters. Consider reducing the rank of the forward operator by using reduce_rank=True.")
        if reduce_rank:
            u, s, vh = np.linalg.svd(Gk, full_matrices=False)
            k = s.shape[0] - 1
            Gk = np.dot(u[:, :k], s[:k].reshape((k, 1)) * vh[:k, :])
        # 2. reorient lead field to max-power direction or normal
        if pick_ori == "max-power":
            GkT = _like(Gk, Cm)
            GkH = _herm(GkT)
            numer0 = np.dot(GkH, Cm_inv)
            denom0 = np.dot(numer0, GkT)
            ori_numer = np.eye(no)
            ori_denom = denom0
            if weight_norm != "":
                ori_numer = denom0.copy()
                ori_denom = np.dot(np.dot(GkH, CmInv2), GkT)
            ori_denom_inv = _sym_inv_sm(ori_denom, reduce_rank, inversion, sk)
            if weight_norm == "":
                ori_pick = ori_denom_inv.copy()
            else:
                ori_pick = np.dot(ori_denom_inv, ori_numer)
            ev, evec = np.linalg.eig(ori_pick.real)
            best = 0
            bv = -1.0
            for i in range(no):
                a = abs(ev[i])
                if a >= bv:
                    bv = a
                    best = i
            ori = np.zeros(no)
            for i in range(no):
                ori[i] = float(evec[i, best].real)
            sg = 0.0
            for i in range(no):
                sg += ori[i] * nn[src, i]
            sign = 1.0
            if sg > 0:
                sign = 1.0
            elif sg < 0:
                sign = -1.0
            for i in range(no):
                ori[i] = ori[i] * sign
                mpo[src, i] = ori[i]
            Gk = np.dot(Gk, ori.reshape((no, 1)))
            no = 1
        elif pick_ori == "normal":
            Gk = Gk[:, 2:3].copy()
            no = 1
        GkT = _like(Gk, Cm)
        bf_numer = np.dot(_herm(GkT), Cm_inv)
        bf_denom = np.dot(bf_numer, GkT)
        bf_inv = _sym_inv_sm(bf_denom, reduce_rank, inversion, sk)
        W = np.dot(bf_inv, bf_numer)
        if weight_norm != "":
            if weight_norm == "unit-noise-gain" or weight_norm == "nai":
                for i in range(no):
                    ss = 0.0
                    for c in range(n_channels):
                        ss += abs(W[i, c]) ** 2
                    nrm = np.sqrt(ss)
                    if nrm == 0.0:
                        nrm = np.inf
                    for c in range(n_channels):
                        W[i, c] = W[i, c] / nrm
            else:
                inner = np.dot(bf_numer, _herm(bf_numer))
                W = np.dot(_sym_mat_pow(inner, -0.5, False), bf_numer)
            if weight_norm == "nai":
                noise = 0.0
                if rank > n_channels:
                    if loading_factor == 0:
                        raise RuntimeError("Cannot compute noise subspace with a full-rank covariance matrix and no regularization. Try manually specifying the rank of the covariance matrix or using regularization.")
                    noise = loading_factor
                else:
                    nev, _ = np.linalg.eigh(Cm)
                    noise = float(nev[n_channels - rank])
                    noise = max(noise, loading_factor)
                sq = np.sqrt(noise)
                for i in range(no):
                    for c in range(n_channels):
                        W[i, c] = W[i, c] / sq
        for i in range(no):
            for c in range(n_channels):
                outW[src * n_out + i, c] = W[i, c]
    return outW, mpo


def _compute_power[T](Cm: np.ndarray[T, 2], W: np.ndarray[T, 2], n_orient: int):
    """Source power from beamformer filters and a (whitened) covariance / CSD."""
    n_sources = W.shape[0] // n_orient
    out = np.zeros(n_sources)
    for s in range(n_sources):
        Wk = W[s * n_orient:(s + 1) * n_orient].copy()
        M = np.dot(np.dot(Wk, Cm), _herm(Wk))
        t = 0.0
        for i in range(n_orient):
            t += float(M[i, i].real)
        out[s] = t
    return out
