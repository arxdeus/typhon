# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne.forward subset: sphere-model forward solutions (EEG Berg-Scherg
# equivalent dipoles, MEG Sarvas field), Forward container,
# convert_forward_solution, apply_forward / apply_forward_raw,
# compute_depth_prior / compute_orient_prior.
#
# Typhon Info carries no dev_head_t: MEG coils are taken to be in head
# coordinates (identity device->head transform).

import numpy as np
from ._info import Info, _picks_to_idx
from .bem import ConductorModel
from .source_space import SourceSpaces, SourceSpace, FIFFV_COORD_HEAD

FIFFV_MNE_FIXED_ORI: int = 1
FIFFV_MNE_FREE_ORI: int = 2
_MAG_FACTOR = 1e-7


def _coil_def(ct: int):
    """Accurate (accuracy=2) coil definitions from mne's coil_def.dat:
    (coil_class, w, rmag) in the coil frame (cosmag is always +z)."""
    if ct == 2000:
        return 1, np.array([1.0]), np.zeros((1, 3))
    if ct == 3012 or ct == 3013 or ct == 3014:
        xs = (0.01079, 0.005891)
        y = 0.006713
        rm = np.empty((8, 3))
        w = np.empty(8)
        k = 0
        for sx in (1.0, -1.0):
            for (x, yy) in ((xs[0], y), (xs[1], y), (xs[1], -y), (xs[0], -y)):
                rm[k, 0] = sx * x
                rm[k, 1] = yy
                rm[k, 2] = 0.0003
                w[k] = 14.9858 * sx
                k += 1
        return 3, w, rm
    if ct == 3022 or ct == 3023 or ct == 3024:
        a = 0.009675 if ct != 3024 else 0.007875
        b = 0.003225 if ct != 3024 else 0.002625
        vals = (-a, -b, b, a)
        rm = np.empty((16, 3))
        k = 0
        for x in vals:
            for y in vals:
                rm[k, 0] = x
                rm[k, 1] = y
                rm[k, 2] = 0.0003
                k += 1
        return 1, np.full(16, 0.0625), rm
    raise RuntimeError(f"Desired coil definition not found (type = {ct} acc = 2)")


class _Coils:
    """Concatenated integration points (_concatenate_coils)."""
    rmags: np.ndarray[float, 2]
    cosmags: np.ndarray[float, 2]
    ws: np.ndarray[float, 1]
    bins: np.ndarray[int, 1]
    n: int
    r0s: np.ndarray[float, 2]

    def __init__(self, rmags, cosmags, ws, bins, n: int, r0s):
        self.rmags = rmags
        self.cosmags = cosmags
        self.ws = ws
        self.bins = bins
        self.n = n
        self.r0s = r0s


def _meg_coils(info: Info, picks: List[int]) -> _Coils:
    rl = List[np.ndarray[float, 2]]()
    cl = List[np.ndarray[float, 2]]()
    wl = List[np.ndarray[float, 1]]()
    bl = List[int]()
    r0s = np.zeros((len(picks), 3))
    for k in range(len(picks)):
        loc = info.locs[picks[k]]
        T = np.eye(4)
        for i in range(3):
            T[i, 3] = loc[i]
        # loc[3:] holds ex, ey, ez; coil_trans[:3,:3] = reshape(...).T
        for c in range(3):
            for i in range(3):
                T[i, c] = loc[3 + 3 * c + i]
        _, w, rm = _coil_def(info.coil_types[picks[k]])
        R = T[:3, :3]
        rr = np.dot(rm, R.T) + T[:3, 3][None, :]
        cz = np.zeros((rm.shape[0], 3))
        cz[:, 2] = 1.0
        cc = np.dot(cz, R.T)
        rl.append(rr)
        cl.append(cc)
        wl.append(w)
        for _ in range(rm.shape[0]):
            bl.append(k)
        r0s[k] = T[:3, 3]
    return _Coils(np.concatenate(rl), np.concatenate(cl), np.concatenate(wl),
                  np.array(bl, dtype=int), len(picks), r0s)


def _eeg_els(info: Info, picks: List[int]) -> _Coils:
    rl = List[np.ndarray[float, 2]]()
    wl = List[np.ndarray[float, 1]]()
    bl = List[int]()
    for k in range(len(picks)):
        loc = info.locs[picks[k]]
        if not (np.isfinite(loc[0]) and np.isfinite(loc[1]) and np.isfinite(loc[2])):
            raise RuntimeError("Missing EEG channel location")
        has_ref = np.isfinite(loc[3]) and np.isfinite(loc[4]) and np.isfinite(loc[5]) and \
            (loc[3] != 0 or loc[4] != 0 or loc[5] != 0)
        if has_ref:
            rl.append(np.array([[loc[0], loc[1], loc[2]], [loc[3], loc[4], loc[5]]]))
            wl.append(np.array([1.0, -1.0]))
            bl.append(k)
            bl.append(k)
        else:
            rl.append(np.array([[loc[0], loc[1], loc[2]]]))
            wl.append(np.array([1.0]))
            bl.append(k)
    rm = np.concatenate(rl)
    return _Coils(rm, rm.copy(), np.concatenate(wl), np.array(bl, dtype=int), len(picks),
                  np.zeros((len(picks), 3)))


def _bincount_rows(B, row: int, bins, x, n: int):
    for i in range(bins.shape[0]):
        B[row, bins[i]] += x[i]


def _sphere_field(rrs, c: _Coils, r0):
    """_do_sphere_field (Sarvas)."""
    n_coils = c.n
    rrs = rrs - r0[None, :]
    this_poss = c.rmags - r0[None, :]
    r = np.sqrt(np.sum(this_poss * this_poss, axis=1))
    B = np.zeros((3 * rrs.shape[0], n_coils))
    npt = this_poss.shape[0]
    for ri in range(rrs.shape[0]):
        rr = rrs[ri]
        if np.sqrt(np.dot(rr, rr)) <= 1e-10:
            continue
        a_vec = this_poss - rr[None, :]
        a = np.sqrt(np.sum(a_vec * a_vec, axis=1))
        rr0 = np.sum(this_poss * rr[None, :], axis=1)
        ar = (r * r) - rr0
        ar0 = ar / a
        F = a * (r * a + ar)
        gr = (a * a) / r + ar0 + 2.0 * (a + r)
        g0 = a + 2 * r + ar0
        re = np.sum(this_poss * c.cosmags, axis=1)
        r0e = np.sum(rr[None, :] * c.cosmags, axis=1)
        g = (g0 * r0e - gr * re) / (F * F)
        for p in range(npt):
            good = (a[p] > 0) or (r[p] > 0) or ((a[p] * r[p]) + 1 > 1e-5)
            if not good:
                continue
            cm = c.cosmags[p]
            tp = this_poss[p]
            v1x = rr[1] * cm[2] - rr[2] * cm[1]
            v1y = rr[2] * cm[0] - rr[0] * cm[2]
            v1z = rr[0] * cm[1] - rr[1] * cm[0]
            v2x = rr[1] * tp[2] - rr[2] * tp[1]
            v2y = rr[2] * tp[0] - rr[0] * tp[2]
            v2z = rr[0] * tp[1] - rr[1] * tp[0]
            w = c.ws[p]
            b = c.bins[p]
            B[3 * ri, b] += w * (v1x / F[p] + v2x * g[p])
            B[3 * ri + 1, b] += w * (v1y / F[p] + v2y * g[p])
            B[3 * ri + 2, b] += w * (v1z / F[p] + v2z * g[p])
    B *= _MAG_FACTOR
    return B


def _eeg_spherepot(rrs, c: _Coils, sphere: ConductorModel):
    """_do_eeg_spherepot_coil."""
    n_coils = c.n
    r0 = sphere.r0
    rad = sphere.layer_rad[0]
    mu = sphere.mu
    lams = sphere.lambda_
    rrs = rrs - r0[None, :]
    this_pos = c.rmags - r0[None, :]
    r2 = np.sum(this_pos * this_pos, axis=1)
    r = np.sqrt(r2)
    B = np.zeros((3 * rrs.shape[0], n_coils))
    npt = this_pos.shape[0]
    for ri in range(rrs.shape[0]):
        rr = rrs[ri]
        if np.sqrt(np.dot(rr, rr)) >= rad:
            continue
        vval = np.zeros((npt, 3))
        for eq in range(mu.shape[0]):
            rd = mu[eq] * rr
            rd2 = np.sum(rd * rd)
            rd2_inv = 1.0 / rd2
            a_vec = this_pos - rd[None, :]
            a = np.sqrt(np.sum(a_vec * a_vec, axis=1))
            a3 = 2.0 / (a * a * a)
            rrd = np.sum(this_pos * rd[None, :], axis=1)
            ra = r2 - rrd
            rda = rrd - rd2
            F = a * (r * a + ra)
            c1 = a3 * rda + 1.0 / a - 1.0 / r
            c2 = a3 + (a + r) / (r * F)
            m1 = c1 - c2 * rrd
            m2 = c2 * rd2
            vval += lams[eq] * rd2_inv * (m1[:, None] * rd[None, :] + m2[:, None] * this_pos)
        for p in range(npt):
            b = c.bins[p]
            w = c.ws[p]
            for jj in range(3):
                B[3 * ri + jj, b] += vval[p, jj] * w
    B *= 0.25 / np.pi
    return B


class _Sol:
    """fwd["sol"]: dict-like with data and row_names."""
    data: np.ndarray[float, 2]
    row_names: List[str]

    def __init__(self, data: np.ndarray[float, 2], row_names: List[str]):
        self.data = data
        self.row_names = row_names

    def __getitem__(self, key: Literal[str]):
        if key == "data":
            return self.data
        elif key == "row_names":
            return self.row_names
        else:
            compile_error("unsupported sol key")


class Forward:
    """Forward solution (mne.Forward); gain rows follow `ch_names`."""
    sol: np.ndarray[float, 2]
    _orig_sol: np.ndarray[float, 2]
    ch_names: List[str]
    source_ori: int
    _orig_source_ori: int
    surf_ori: bool
    nsource: int
    source_nn: np.ndarray[float, 2]
    source_rr: np.ndarray[float, 2]
    src: SourceSpaces
    info: Info
    coord_frame: int

    def __init__(self, sol, ch_names: List[str], src: SourceSpaces, info: Info, rr):
        self.sol = sol
        self._orig_sol = sol.copy()
        self.ch_names = ch_names
        self.source_ori = FIFFV_MNE_FREE_ORI
        self._orig_source_ori = FIFFV_MNE_FREE_ORI
        self.surf_ori = False
        self.nsource = rr.shape[0]
        self.source_rr = rr
        self.source_nn = np.tile(np.eye(3), (rr.shape[0], 1))
        self.src = src
        self.info = info
        self.coord_frame = FIFFV_COORD_HEAD

    def copy(self):
        f = Forward(self.sol.copy(), list(self.ch_names), self.src.copy(), self.info.copy(),
                    self.source_rr.copy())
        f._orig_sol = self._orig_sol.copy()
        f.source_ori = self.source_ori
        f._orig_source_ori = self._orig_source_ori
        f.surf_ori = self.surf_ori
        f.source_nn = self.source_nn.copy()
        return f

    @property
    def nchan(self) -> int:
        return len(self.ch_names)

    def __getitem__(self, key: Literal[str]):
        if key == "sol":
            return _Sol(self.sol, self.ch_names)
        elif key == "nsource":
            return self.nsource
        elif key == "nchan":
            return self.nchan
        elif key == "source_ori":
            return self.source_ori
        elif key == "surf_ori":
            return self.surf_ori
        elif key == "source_nn":
            return self.source_nn
        elif key == "source_rr":
            return self.source_rr
        elif key == "src":
            return self.src
        elif key == "info":
            return self.info
        elif key == "coord_frame":
            return self.coord_frame
        else:
            compile_error("unsupported Forward key")

    def __repr__(self) -> str:
        ori = "Free" if self.source_ori == FIFFV_MNE_FREE_ORI else "Fixed"
        return f"<Forward | MEG/EEG channels: {self.nchan}, Source space: {self.src.kind} with {self.nsource} vertices, Source orientation: {ori}>"


def is_fixed_orient(forward: Forward, orig: bool = False) -> bool:
    if orig:
        return forward._orig_source_ori == FIFFV_MNE_FIXED_ORI
    return forward.source_ori == FIFFV_MNE_FIXED_ORI


def make_forward_solution(info: Info, trans, src, bem: ConductorModel, meg: bool = True,
                          eeg: bool = True, *, mindist: float = 0.0, ignore_ref: bool = False,
                          n_jobs = None, on_inside: str = "raise", verbose = None) -> Forward:
    """mne.make_forward_solution with a spherical conductor model (trans must be
    None/'identity'; MEG coils are assumed to be given in head coordinates)."""
    if not isinstance(trans, NoneType):
        if not (isinstance(trans, str) and (trans == "identity" or trans == "fsaverage")):
            compile_error("make_forward_solution: only trans=None is supported in Typhon mne")
    src = src.copy()
    meg_picks = [i for i in range(info.nchan) if info.ch_types[i] in ("mag", "grad")]
    eeg_picks = [i for i in range(info.nchan) if info.ch_types[i] == "eeg"]
    use_meg = meg and len(meg_picks) > 0
    use_eeg = eeg and len(eeg_picks) > 0
    if not use_meg and not use_eeg:
        raise RuntimeError("No MEG or EEG channels found.")
    if use_eeg and bem.nlayers == 0:
        raise RuntimeError("Spherical model has zero shells, cannot use with EEG data")
    inner: Optional[float] = None if bem.nlayers == 0 else bem.layer_rad[0]
    for s in src.spaces:
        s.coord_frame = FIFFV_COORD_HEAD
        if inner is not None:
            from .source_space import _filter_sphere
            _filter_sphere(s, bem.r0, inner, mindist)
    rl = List[np.ndarray[float, 2]]()
    for s in src.spaces:
        rl.append(s.rr[s.vertno])
    rr = np.concatenate(rl)
    if rr.shape[0] < 1:
        raise RuntimeError("No points left in source space after excluding points close to inner skull.")
    blocks = List[np.ndarray[float, 2]]()
    names = List[str]()
    if use_meg:
        c = _meg_coils(info, meg_picks)
        blocks.append(_sphere_field(rr, c, bem.r0).T.copy())
        names.extend([info.ch_names[p] for p in meg_picks])
    if use_eeg:
        c = _eeg_els(info, eeg_picks)
        blocks.append(_eeg_spherepot(rr, c, bem).T.copy())
        names.extend([info.ch_names[p] for p in eeg_picks])
    sol = np.concatenate(blocks, axis=0)
    keep = [i for i in range(info.nchan) if (use_meg and i in meg_picks) or (use_eeg and i in eeg_picks)]
    finfo = info._pick(keep)
    return Forward(sol, names, src, finfo, rr)


def _normal_orth(nn):
    """mne.surface._normal_orth for one normal: rows (ex, ey, ez=nn)."""
    prod = np.outer(nn, nn)
    _, u = np.linalg.eigh(np.eye(3) - prod)
    u = u[:, ::-1].copy()
    s = float(np.dot(nn, u[:, 2]))
    sign = 1.0 if s >= 0 else -1.0
    if s == 0:
        sign = 1.0
    u *= sign
    return u.T.copy()


def _block_diag_mult(sol, nn, n: int):
    """sol @ _block_diag(nn.T, n) for nn of shape (n_blocks * n, 3)."""
    nb = nn.shape[0] // n
    out = np.zeros((sol.shape[0], nb * n))
    for b in range(nb):
        blk = sol[:, 3 * b:3 * b + 3]
        out[:, n * b:n * b + n] = np.dot(blk, nn[n * b:n * b + n].T)
    return out


def convert_forward_solution(fwd: Forward, surf_ori: bool = False, force_fixed: bool = False,
                             copy: bool = True, use_cps: bool = True, *, verbose = None) -> Forward:
    """mne.convert_forward_solution (discrete/volume source spaces)."""
    fwd = fwd.copy() if copy else fwd
    if force_fixed:
        surf_ori = True
    for s in fwd.src.spaces:
        if s.type == "vol" and force_fixed:
            raise ValueError("Forward operator was generated with sources from a volume source space. Conversion to fixed orientation is not possible. Consider using a discrete source space if you have meaningful normal orientations.")
    if is_fixed_orient(fwd, True) or force_fixed:
        nl = List[np.ndarray[float, 2]]()
        for s in fwd.src.spaces:
            nl.append(s.nn[s.vertno])
        fwd.source_nn = np.concatenate(nl)
        if not is_fixed_orient(fwd, True):
            fwd.sol = _block_diag_mult(fwd._orig_sol, fwd.source_nn, 1).astype(np.float32).astype(float)
        fwd.source_ori = FIFFV_MNE_FIXED_ORI
        fwd.surf_ori = True
    elif surf_ori:
        nn3 = np.tile(np.eye(3), (fwd.nsource, 1))
        pp = 0
        for s in fwd.src.spaces:
            if s.type == "surf" or s.type == "discrete":
                for v in range(s.nuse):
                    o = _normal_orth(s.nn[int(s.vertno[v])])
                    nn3[pp:pp + 3] = o
                    pp += 3
            else:
                pp += 3 * s.nuse
        fwd.source_nn = nn3
        fwd.sol = _block_diag_mult(fwd._orig_sol, nn3, 3)
        fwd.source_ori = FIFFV_MNE_FREE_ORI
        fwd.surf_ori = True
    else:
        fwd.source_nn = np.tile(np.eye(3), (fwd.nsource, 1))
        fwd.sol = fwd._orig_sol.copy()
        fwd.source_ori = FIFFV_MNE_FREE_ORI
        fwd.surf_ori = False
    return fwd


def pick_channels_forward(orig: Forward, include = (), exclude = (), ordered: bool = True,
                          copy: bool = True, *, verbose = None) -> Forward:
    """mne.pick_channels_forward."""
    from .cov import _pick_names
    fwd = orig.copy() if copy else orig
    sel = _pick_names(orig.ch_names, [str(c) for c in include], [str(c) for c in exclude], ordered)
    sa = np.array(sel, dtype=int)
    fwd.sol = fwd.sol[sa].copy()
    fwd._orig_sol = fwd._orig_sol[sa].copy()
    fwd.ch_names = [orig.ch_names[k] for k in sel]
    inames = [fwd.ch_names[k] for k in range(len(sel))]
    keep = [i for i in range(fwd.info.nchan) if fwd.info.ch_names[i] in inames]
    fwd.info = fwd.info._pick(keep)
    return fwd


def _triage_loose(src: SourceSpaces, loose, fixed) -> float:
    """_triage_loose for single-kind (discrete/volume/surface) source spaces."""
    fx = 0
    if isinstance(fixed, str):
        if fixed != "auto":
            raise ValueError(f"fixed must be True, False or 'auto', got {fixed}")
        fx = 0
    elif fixed:
        fx = 1
    else:
        fx = 2
    lo = 0.0
    if isinstance(loose, str):
        if loose != "auto":
            raise ValueError(f"Invalid value for the 'loose' parameter. Allowed values are 'auto', but got {loose!r} instead.")
        if fx == 1:
            lo = 0.0
        else:
            lo = 0.2 if src.kind == "surface" else 1.0
    else:
        lo = float(loose)
    if fx == 1 and lo != 0.0:
        raise ValueError(f'When using fixed=True, loose must be 0. or "auto", got {loose}')
    if fx == 2 and lo == 0.0:
        raise ValueError('If loose==0., then fixed must be True or "auto", got False')
    kind = src.kind
    if kind != "surface" and kind != "discrete" and lo != 1:
        raise ValueError(f'loose parameter has to be 1 or "auto" for non-surface/discrete source spaces, got loose["{kind}"] = {lo}')
    if not (0 <= lo <= 1):
        raise ValueError(f"loose ({kind}) must be between 0 and 1, got {lo}")
    return lo


def compute_orient_prior(forward: Forward, loose = "auto", verbose = None):
    n_sources = forward.sol.shape[1]
    lo = _triage_loose(forward.src, loose, "auto")
    prior = np.ones(n_sources)
    if is_fixed_orient(forward):
        if lo > 0.0:
            raise ValueError(f"loose must be 0. with forward operator with fixed orientation, got {lo}")
        return prior
    if lo == 1.0:
        return prior
    if not forward.surf_ori:
        raise ValueError(f"Forward operator is not oriented in surface coordinates. loose parameter should be 1. not {lo}.")
    start = 0
    for s in forward.src.spaces:
        stop = start + 3 * s.nuse
        for i in range(start, stop, 3):
            prior[i] *= lo
            prior[i + 1] *= lo
        start = stop
    return prior


def _restrict_gain_matrix(G, info: Info):
    for kind in ("grad", "mag", "eeg"):
        sel = [i for i in range(info.nchan) if info.ch_types[i] == kind]
        if len(sel) > 0:
            return G[np.array(sel, dtype=int)].copy()
    return G


def compute_depth_prior(forward: Forward, info: Info, exp: float = 0.8, limit = 10.0,
                        limit_depth_chs = False, combine_xyz = "spectral", noise_cov = None,
                        rank = None, verbose = None):
    """mne.forward.compute_depth_prior."""
    is_fixed_ori = is_fixed_orient(forward)
    G = forward.sol
    whiten = False
    restrict = False
    if isinstance(limit_depth_chs, str):
        if limit_depth_chs != "whiten":
            raise ValueError(f'limit_depth_chs, if str, must be "whiten", got {limit_depth_chs}')
        if isinstance(noise_cov, NoneType):
            raise ValueError('With limit_depth_chs="whiten", noise_cov must be a Covariance, got NoneType')
        whiten = True
    else:
        restrict = limit_depth_chs
    cxyz = "spectral"
    if isinstance(combine_xyz, str):
        cxyz = combine_xyz
    elif not combine_xyz:
        cxyz = "none"
    if restrict:
        G = _restrict_gain_matrix(G, info)
    elif whiten:
        from .cov import compute_whitener
        if not isinstance(noise_cov, NoneType):
            W, _ = compute_whitener(noise_cov, info, pca=True, rank=rank)
            G = np.dot(W, G)
    d = np.zeros(0)
    if is_fixed_ori or cxyz == "fro" or cxyz == "none":
        d = np.sum(G ** 2, axis=0)
        if not (is_fixed_ori or cxyz == "none"):
            d = d.reshape(-1, 3).sum(axis=1)
        mn = np.inf
        for v in d:
            if v != 0.0 and v < mn:
                mn = v
        for i in range(d.shape[0]):
            if d[i] == 0.0:
                d[i] = mn
    else:
        nv = G.shape[1] // 3
        d = np.empty(nv)
        for v in range(nv):
            Gv = G[:, 3 * v:3 * v + 3]
            M = np.dot(Gv.T, Gv)
            d[v] = float(np.linalg.norm(M, 2))
    w = 1.0 / d
    if not isinstance(limit, NoneType):
        lim = float(limit)
        ws = np.sort(w)
        weight_limit = lim ** 2
        lv = 0.0
        if not restrict and not whiten:
            lv = ws[0] * weight_limit
        else:
            lv = ws[-1]
            if ws[-1] > weight_limit * ws[0]:
                for i in range(ws.shape[0]):
                    if ws[i] > weight_limit * ws[0]:
                        lv = ws[i]
                        break
        w = np.minimum(w / lv, 1.0)
    depth_prior = w ** exp
    if not (is_fixed_ori or cxyz == "none"):
        depth_prior = np.repeat(depth_prior, 3)
    return depth_prior


def _stc_src_sel(src: SourceSpaces, vertices: List[np.ndarray[int, 1]]):
    src_sel = List[int]()
    stc_sel = List[int]()
    so = 0
    to = 0
    for si in range(len(src.spaces)):
        s = src.spaces[si]
        v = vertices[si]
        vl = [int(x) for x in s.vertno]
        tl = [int(x) for x in v]
        joint = sorted([x for x in tl if x in vl])
        for x in joint:
            src_sel.append(vl.index(x) + so)
            stc_sel.append(tl.index(x) + to)
        so += len(vl)
        to += len(tl)
    if len(src_sel) != to:
        raise ValueError(f"Only {len(src_sel)} of {to} SourceEstimate vertices found in source space, likely due to forward calculations")
    return src_sel, stc_sel


def _apply_forward(fwd: Forward, stc, start, stop):
    vector = stc.data.ndim == 3
    if vector:
        fwd = convert_forward_solution(fwd, force_fixed=False, surf_ori=False)
    elif not is_fixed_orient(fwd):
        fwd = convert_forward_solution(fwd, force_fixed=True)
    src_sel, stc_sel = _stc_src_sel(fwd.src, stc.vertices)
    gain = fwd.sol
    n_t = stc.data.shape[-1]
    a = 0 if isinstance(start, NoneType) else int(start)
    b = n_t if isinstance(stop, NoneType) else int(stop)
    if a < 0:
        a += n_t
    if b < 0:
        b += n_t
    times = stc.times[a:b].copy()
    sa = np.array(stc_sel, dtype=int)
    sd = stc.data[sa][..., a:b].reshape(-1, b - a)
    g = np.zeros((gain.shape[0], 0))
    if vector:
        cols = List[int]()
        for s in src_sel:
            cols.extend([3 * s, 3 * s + 1, 3 * s + 2])
        g = gain[:, np.array(cols, dtype=int)]
    else:
        g = gain[:, np.array(src_sel, dtype=int)]
    return np.dot(g, sd), times


def _fill_info(info: Info, fwd: Forward, sfreq: float, data):
    sel = [i for i in range(info.nchan) if info.ch_names[i] in fwd.ch_names]
    out = info._pick(sel)
    out.bads = List[str]()
    out.highpass = 0.0
    out.lowpass = sfreq / 2.0
    out.sfreq = sfreq
    out.projs = []
    order = [fwd.ch_names.index(n) for n in out.ch_names]
    return out, data[np.array(order, dtype=int)].copy()


def apply_forward(fwd: Forward, stc, info: Info, start = None, stop = None,
                  use_cps: bool = True, on_missing: str = "raise", verbose = None):
    """mne.apply_forward -> Evoked."""
    from .epochs import EvokedArray
    for n in fwd.ch_names:
        if n not in info.ch_names:
            raise ValueError(f"Channel {n} of forward operator not present in evoked_template.")
    data, times = _apply_forward(fwd, stc, start, stop)
    sfreq = float(1.0 / stc.tstep)
    inf, d = _fill_info(info, fwd, sfreq, data)
    ev = EvokedArray(d, inf, float(times[0]), nave=1)
    ev.times = times
    return ev


def apply_forward_raw(fwd: Forward, stc, info: Info, start = None, stop = None,
                      on_missing: str = "raise", use_cps: bool = True, verbose = None):
    """mne.apply_forward_raw -> RawArray."""
    from ._containers import RawArray
    for n in fwd.ch_names:
        if n not in info.ch_names:
            raise ValueError(f"Channel {n} of forward operator not present in info.")
    data, times = _apply_forward(fwd, stc, start, stop)
    sfreq = 1.0 / stc.tstep
    inf, d = _fill_info(info, fwd, sfreq, data)
    return RawArray(d, inf, first_samp=int(np.round(float(times[0]) * sfreq)))
