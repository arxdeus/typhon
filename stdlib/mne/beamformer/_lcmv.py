# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# LCMV beamformer: Beamformer container, make_lcmv, apply_lcmv(_epochs/_raw/_cov),
# make_lcmv_resolution_matrix.

import numpy as np
from .._info import Info
from .._proj import make_projector
from .._containers import Evoked, EpochsArray, RawArray
from ..cov import (Covariance, _check_rank, _rank_of_cov, _make_scalings, pick_channels_cov,
                   _Rank, compute_whitener)
from ..forward import Forward, pick_channels_forward
from ..epochs import EvokedArray
from ..source_estimate import _STC
from ..time_frequency.csd import CrossSpectralDensity
from ._compute import (_BfInput, _prepare_beamformer_input, _compute_beamformer, _compute_power,
                       _ad_hoc_unit, _check_info_inv, _n_ch_types, _check_channels_spatial_filter,
                       _opt_str, _sum_rank, _rank_matrix, _herm, _like, _check_option)


class Beamformer:
    """A computed beamformer (mne.beamformer.Beamformer).

    Weights are stored 3-D (n_freqs, n_sources * n_orient, n_channels); LCMV
    filters have n_freqs == 1. Complex DICS filters (real_filter=False) use
    `weights_c`."""
    kind: str
    weights3: np.ndarray[float, 3]
    weights3_c: np.ndarray[complex, 3]
    is_complex: bool
    data_cov: Optional[Covariance]
    noise_cov: Optional[Covariance]
    csd: Optional[CrossSpectralDensity]
    whitener: np.ndarray[float, 2]
    weight_norm: Optional[str]
    pick_ori: Optional[str]
    ch_names: List[str]
    proj: np.ndarray[float, 2]
    is_ssp: bool
    vertices: List[np.ndarray[int, 1]]
    is_free_ori: bool
    n_sources: int
    src_type: str
    source_nn: np.ndarray[float, 2]
    subject: Optional[str]
    rank: int
    max_power_ori: np.ndarray[float, 3]
    inversion: str

    def __init__(self, kind: str, ch_names: List[str]):
        self.kind = kind
        self.weights3 = np.zeros((0, 0, 0))
        self.weights3_c = np.zeros((0, 0, 0), dtype=complex)
        self.is_complex = False
        self.data_cov = None
        self.noise_cov = None
        self.csd = None
        self.whitener = np.zeros((0, 0))
        self.weight_norm = None
        self.pick_ori = None
        self.ch_names = ch_names
        self.proj = np.zeros((0, 0))
        self.is_ssp = False
        self.vertices = List[np.ndarray[int, 1]]()
        self.is_free_ori = False
        self.n_sources = 0
        self.src_type = "volume"
        self.source_nn = np.zeros((0, 3))
        self.subject = None
        self.rank = 0
        self.max_power_ori = np.zeros((0, 0, 0))
        self.inversion = "matrix"

    @property
    def weights(self):
        """LCMV weights, shape (n_sources * n_orient, n_channels)."""
        return self.weights3[0]

    def copy(self):
        b = Beamformer(self.kind, list(self.ch_names))
        b.weights3 = self.weights3.copy()
        b.weights3_c = self.weights3_c.copy()
        b.is_complex = self.is_complex
        if self.data_cov is not None:
            b.data_cov = self.data_cov.__val__().copy()
        if self.noise_cov is not None:
            b.noise_cov = self.noise_cov.__val__().copy()
        if self.csd is not None:
            b.csd = self.csd.__val__().copy()
        b.whitener = self.whitener.copy()
        b.weight_norm = self.weight_norm
        b.pick_ori = self.pick_ori
        b.proj = self.proj.copy()
        b.is_ssp = self.is_ssp
        b.vertices = [v.copy() for v in self.vertices]
        b.is_free_ori = self.is_free_ori
        b.n_sources = self.n_sources
        b.src_type = self.src_type
        b.source_nn = self.source_nn.copy()
        b.subject = self.subject
        b.rank = self.rank
        b.max_power_ori = self.max_power_ori.copy()
        b.inversion = self.inversion
        return b

    def __getitem__(self, key: Literal[str]):
        if key == "kind":
            return self.kind
        elif key == "weights":
            return self.weights3[0]
        elif key == "ch_names":
            return self.ch_names
        elif key == "vertices":
            return self.vertices
        elif key == "n_sources":
            return self.n_sources
        elif key == "is_free_ori":
            return self.is_free_ori
        elif key == "pick_ori":
            return self.pick_ori
        elif key == "weight_norm":
            return self.weight_norm
        elif key == "inversion":
            return self.inversion
        elif key == "src_type":
            return self.src_type
        elif key == "source_nn":
            return self.source_nn
        elif key == "subject":
            return self.subject
        elif key == "rank":
            return self.rank
        elif key == "proj":
            return self.proj
        elif key == "whitener":
            return self.whitener
        elif key == "is_ssp":
            return self.is_ssp
        elif key == "max_power_ori":
            return self.max_power_ori
        else:
            compile_error("unsupported Beamformer key")

    def __repr__(self) -> str:
        nv = 0
        for v in self.vertices:
            nv += v.shape[0]
        subj = "unknown" if self.subject is None else f'"{self.subject.__val__()}"'
        out = f"<Beamformer | {self.kind}, subject {subj}, {nv} vert, {len(self.ch_names)} ch"
        if self.pick_ori is not None:
            out += f", {self.pick_ori.__val__()} ori"
        if self.weight_norm is not None:
            out += f", {self.weight_norm.__val__()} norm"
        out += f", {self.inversion} inversion"
        if self.kind == "LCMV":
            out += f", rank {self.rank}"
        out += ">"
        return out


def _get_src_type(f: Forward) -> str:
    return f.src.kind


def _make_stc_vec(sol, vertices: List[np.ndarray[int, 1]], tmin: float, tstep: float,
                  subject: Optional[str], source_nn: np.ndarray[float, 2], src_type: str):
    """Vector _STC (n_vertices, 3, n_times) from a (n_rows, n_times) solution."""
    nv = 0
    for v in vertices:
        nv += v.shape[0]
    nt = sol.shape[1]
    out = np.zeros((nv, 3, nt))
    if sol.shape[0] == nv:
        for i in range(nv):
            for q in range(3):
                out[i, q, :] = sol[i, :] * source_nn[i, q]
    else:
        for i in range(nv):
            for q in range(3):
                acc = np.zeros(nt)
                for p in range(3):
                    acc = acc + source_nn[3 * i + p, q] * sol[3 * i + p, :]
                out[i, q, :] = acc
    return _STC[np.ndarray[float, 3]](out, vertices, tmin, tstep, subject, src_type)


def _make_stc_sc(sol, vertices: List[np.ndarray[int, 1]], tmin: float, tstep: float,
                 subject: Optional[str], src_type: str):
    return _STC[np.ndarray[float, 2]](sol, vertices, tmin, tstep, subject, src_type)


def _stc_kind(src_type: str) -> str:
    if src_type == "discrete":
        return "volume"
    return src_type


def make_lcmv(info: Info, forward: Forward, data_cov: Covariance, reg: float = 0.05,
              noise_cov = None, label = None, pick_ori = None, rank = "info",
              weight_norm = "unit-noise-gain-invariant", reduce_rank: bool = False,
              depth = None, inversion: str = "matrix", *, verbose = None) -> Beamformer:
    """mne.beamformer.make_lcmv (volume / discrete source spaces)."""
    if not isinstance(label, NoneType):
        compile_error("label restriction is not supported")
    po = _opt_str(pick_ori)
    wn = _opt_str(weight_norm)
    rk = _check_rank(rank)
    has_noise = not isinstance(noise_cov, NoneType)
    # _check_one_ch_type
    nc_in = _ad_hoc_unit(info)
    allow_mismatch = True
    if has_noise:
        nc_in = noise_cov.copy()
        allow_mismatch = False
    names_l = List[List[str]]()
    bads_l = List[List[str]]()
    names_l.append(data_cov.names)
    bads_l.append(data_cov.bads)
    if has_noise:
        names_l.append(nc_in.names)
        bads_l.append(nc_in.bads)
    pre = _check_info_inv(info, forward.ch_names, names_l, bads_l)
    info_pre = info._pick(pre)
    if _n_ch_types(info_pre) > 1 and not has_noise:
        raise ValueError("Source reconstruction with several sensor types requires a noise covariance matrix to be able to apply whitening.")
    if not has_noise:
        nc_in = _ad_hoc_unit(info_pre)
    picks = _check_info_inv(info, forward.ch_names, names_l, bads_l)
    info2 = info._pick(picks)
    data_rank = _rank_of_cov(data_cov, info2, rk, _make_scalings(None, "cov_rank"))
    noise_rank = _rank_of_cov(nc_in, info2, rk, _make_scalings(None, "cov_rank"))
    for k, v in data_rank.items():
        if (k not in noise_rank or noise_rank[k] != v) and not allow_mismatch:
            raise ValueError(f"{k} data rank ({v}) did not match the noise rank ({noise_rank.get(k, -1)})")
    rank_int = _sum_rank(data_rank)
    # depth (depth_sparse defaults)
    exp: Optional[float] = 0.8
    if not isinstance(depth, NoneType):
        if isinstance(depth, Dict):
            for kk, vv in depth.items():
                if kk == "exp":
                    exp = float(vv)
        else:
            exp = float(depth)
    else:
        exp = None
    cxyz = "fro"
    if inversion == "single":
        cxyz = "none"
    bi = _prepare_beamformer_input(info2, forward, po, nc_in, data_rank_to_arg(data_rank, rk), exp,
                                   cxyz, "whiten", True)
    ch_names = list(bi.info.ch_names)
    dc = pick_channels_cov(data_cov, ch_names)
    Cm = dc._get_square()
    n_orient = 3 if bi.is_free_ori else 1
    W, mpo = _compute_beamformer(bi.gain, Cm, reg, n_orient, wn, po, reduce_rank, rank_int,
                                 inversion, bi.nn, bi.orient_std, bi.whitener)
    f = Beamformer("LCMV", ch_names)
    f.weights3 = W.reshape((1, W.shape[0], W.shape[1]))
    f.data_cov = dc
    f.noise_cov = nc_in
    f.whitener = bi.whitener
    f.weight_norm = None if wn == "" else wn
    f.pick_ori = None if po == "" else po
    f.proj = bi.proj
    f.is_ssp = len(bi.info.projs) > 0
    f.vertices = bi.vertno
    f.is_free_ori = bi.is_free_ori if (po == "" or po == "vector") else False
    f.n_sources = forward.nsource
    f.src_type = _get_src_type(forward)
    f.source_nn = forward.source_nn.copy()
    f.subject = None
    f.rank = rank_int
    if po == "max-power":
        f.max_power_ori = mpo.reshape((1, mpo.shape[0], 3))
    f.inversion = inversion
    return f


def data_rank_to_arg(r: Dict[str, int], rk: _Rank) -> _Rank:
    return _Rank(rk.kind, r)


def _proj_whiten_data(M, projs, f: Beamformer):
    """Apply SSP + whitening to (n_channels, n_times) data."""
    if f.is_ssp:
        P, _, _ = make_projector(projs, f.ch_names)
        if P.shape[0] != f.proj.shape[0] or not np.allclose(P, f.proj, rtol=1e-13, atol=2.220446049250313e-16):
            raise ValueError("The SSP projections present in the data do not match the projections used when calculating the spatial filter.")
        if f.whitener.shape[0] == 0:
            return np.dot(f.proj, M)
    return np.dot(f.whitener, M)


def _combine_xyz(sol):
    n = sol.shape[0] // 3
    out = np.zeros((n, sol.shape[1]))
    for i in range(n):
        out[i] = np.sqrt(sol[3 * i] ** 2 + sol[3 * i + 1] ** 2 + sol[3 * i + 2] ** 2)
    return out


def _apply_lcmv_one(M0, f: Beamformer, projs, tmin: float, sfreq: float):
    """One data matrix -> _STC (scalar, or vector when pick_ori == 'vector')."""
    M = _proj_whiten_data(M0, projs, f)
    W = f.weights
    sol = np.dot(W, M)
    vec = f.pick_ori is not None and f.pick_ori.__val__() == "vector"
    if f.is_free_ori and not vec:
        sol = _combine_xyz(sol)
    return sol, vec


def apply_lcmv(evoked: Evoked, filters: Beamformer, *, verbose = None):
    """mne.beamformer.apply_lcmv. Returns an _STC (scalar data); use
    apply_lcmv_vector for pick_ori='vector' filters."""
    sel = _check_channels_spatial_filter(evoked.ch_names, filters.ch_names)
    data = evoked.data[np.array(sel, dtype=int)]
    sol, vec = _apply_lcmv_one(data, filters, evoked.info.projs, float(evoked.times[0]), evoked.info.sfreq)
    if vec:
        raise ValueError("pick_ori='vector' filters must be applied with apply_lcmv_vector")
    return _make_stc_sc(sol, filters.vertices, float(evoked.times[0]), 1.0 / evoked.info.sfreq,
                        filters.subject, _stc_kind(filters.src_type))


def apply_lcmv_vector(evoked: Evoked, filters: Beamformer, *, verbose = None):
    """apply_lcmv for pick_ori='vector' filters (VolVectorSourceEstimate)."""
    sel = _check_channels_spatial_filter(evoked.ch_names, filters.ch_names)
    data = evoked.data[np.array(sel, dtype=int)]
    sol, vec = _apply_lcmv_one(data, filters, evoked.info.projs, float(evoked.times[0]), evoked.info.sfreq)
    return _make_stc_vec(sol, filters.vertices, float(evoked.times[0]), 1.0 / evoked.info.sfreq,
                         filters.subject, filters.source_nn, _stc_kind(filters.src_type))


def apply_lcmv_epochs(epochs: EpochsArray, filters: Beamformer, *, return_generator: bool = False,
                      verbose = None):
    """mne.beamformer.apply_lcmv_epochs -> list of STC (returned as a list)."""
    sel = _check_channels_spatial_filter(epochs.ch_names, filters.ch_names)
    data = epochs.get_data(sel)
    out = List[_STC[np.ndarray[float, 2]]]()
    for e in range(data.shape[0]):
        M = data[e].copy()
        sol, vec = _apply_lcmv_one(M, filters, epochs.info.projs, float(epochs.times[0]), epochs.info.sfreq)
        out.append(_make_stc_sc(sol, filters.vertices, float(epochs.times[0]), 1.0 / epochs.info.sfreq,
                                filters.subject, _stc_kind(filters.src_type)))
    return out


def apply_lcmv_raw(raw: RawArray, filters: Beamformer, start = None, stop = None, *, verbose = None):
    """mne.beamformer.apply_lcmv_raw."""
    sel = _check_channels_spatial_filter(raw.ch_names, filters.ch_names)
    n = raw._data.shape[1]
    a = 0 if isinstance(start, NoneType) else int(start)
    b = n if isinstance(stop, NoneType) else int(stop)
    data = raw._data[np.array(sel, dtype=int), a:b].copy()
    tmin = float(raw.times[a])
    sol, vec = _apply_lcmv_one(data, filters, raw.info.projs, tmin, raw.info.sfreq)
    return _make_stc_sc(sol, filters.vertices, tmin, 1.0 / raw.info.sfreq, filters.subject,
                        _stc_kind(filters.src_type))


def apply_lcmv_cov(data_cov: Covariance, filters: Beamformer, *, verbose = None):
    """mne.beamformer.apply_lcmv_cov: source power estimate."""
    sel = _check_channels_spatial_filter(data_cov.names, filters.ch_names)
    names = [data_cov.names[i] for i in sel]
    dc = pick_channels_cov(data_cov, names)
    W = filters.weights
    n_orient = W.shape[0] // filters.n_sources
    C = dc._get_square()
    if filters.is_ssp:
        P, _, _ = make_projector(dc.projs, filters.ch_names)
        if filters.whitener.shape[0] == 0:
            C = np.dot(np.dot(filters.proj, C), filters.proj.T)
    if filters.whitener.shape[0] > 0:
        C = np.dot(np.dot(filters.whitener, C), filters.whitener.T)
    power = _compute_power(C, W, n_orient)
    pw = power.reshape((power.shape[0], 1))
    return _make_stc_sc(pw, filters.vertices, 0.0, 1.0, filters.subject, _stc_kind(filters.src_type))


def make_lcmv_resolution_matrix(filters: Beamformer, forward: Forward, info: Info):
    """mne.beamformer.make_lcmv_resolution_matrix."""
    bads = filters.noise_cov.__val__().bads
    names = [c for c in filters.noise_cov.__val__().names if c not in bads]
    fwd = pick_channels_forward(forward, names, (), True, True)
    leadfield = fwd.sol
    sel = [info.ch_names.index(c) for c in filters.ch_names]
    inf2 = info._pick(sel)
    n = len(sel)
    ev = EvokedArray(np.eye(n), inf2, 0.0)
    stc = apply_lcmv(ev, filters)
    return np.dot(stc.data, leadfield)
