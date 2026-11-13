# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne.minimum_norm core: InverseOperator container, prepare_inverse_operator,
# eLORETA weight fitting, kernel assembly.

import numpy as np
from .._info import Info, Projection
from .._proj import make_projector
from ..cov import Covariance, _whitener_from_prepared
from ..source_space import SourceSpaces
from ..forward import _normal_orth, FIFFV_MNE_FREE_ORI, FIFFV_MNE_FIXED_ORI

INVERSE_METHODS = ("MNE", "dSPM", "sLORETA", "eLORETA")


class InverseOperator:
    """Inverse operator (mne.minimum_norm.InverseOperator).

    Matrices follow mne's `inv[key]["data"]` layout: `eigen_fields` is
    (n_comp, n_chan), `eigen_leads` is (n_sources * n_orient, n_comp)."""
    eigen_fields: np.ndarray[float, 2]
    eigen_leads: np.ndarray[float, 2]
    sing: np.ndarray[float, 1]
    source_cov: np.ndarray[float, 1]
    orient_prior: Optional[np.ndarray[float, 1]]
    depth_prior: Optional[np.ndarray[float, 1]]
    noise_cov: Covariance
    projs: List[Projection]
    info: Info
    src: SourceSpaces
    source_nn: np.ndarray[float, 2]
    source_ori: int
    nsource: int
    nave: float
    eigen_leads_weighted: bool
    coord_frame: int
    ch_names: List[str]
    prepared: bool
    reginv: np.ndarray[float, 1]
    proj: np.ndarray[float, 2]
    whitener: np.ndarray[float, 2]
    colorer: np.ndarray[float, 2]
    noisenorm: np.ndarray[float, 1]

    def __init__(self, eigen_fields, eigen_leads, sing, source_cov, orient_prior, depth_prior,
                 noise_cov: Covariance, projs: List[Projection], info: Info, src: SourceSpaces,
                 source_nn, source_ori: int, nsource: int, nave: float,
                 eigen_leads_weighted: bool, coord_frame: int, ch_names: List[str]):
        self.eigen_fields = eigen_fields
        self.eigen_leads = eigen_leads
        self.sing = sing
        self.source_cov = source_cov
        self.orient_prior = orient_prior
        self.depth_prior = depth_prior
        self.noise_cov = noise_cov
        self.projs = projs
        self.info = info
        self.src = src
        self.source_nn = source_nn
        self.source_ori = source_ori
        self.nsource = nsource
        self.nave = nave
        self.eigen_leads_weighted = eigen_leads_weighted
        self.coord_frame = coord_frame
        self.ch_names = ch_names
        self.prepared = False
        self.reginv = np.zeros(0)
        self.proj = np.zeros((0, 0))
        self.whitener = np.zeros((0, 0))
        self.colorer = np.zeros((0, 0))
        self.noisenorm = np.zeros(0)

    def _copy(self, share_src: bool):
        op = None if self.orient_prior is None else self.orient_prior.__val__().copy()
        dp = None if self.depth_prior is None else self.depth_prior.__val__().copy()
        src = self.src if share_src else self.src.copy()
        out = InverseOperator(self.eigen_fields.copy(), self.eigen_leads.copy(), self.sing.copy(),
                              self.source_cov.copy(), op, dp, self.noise_cov.copy(),
                              [p.copy() for p in self.projs], self.info.copy(), src,
                              self.source_nn.copy(), self.source_ori, self.nsource, self.nave,
                              self.eigen_leads_weighted, self.coord_frame, list(self.ch_names))
        out.prepared = self.prepared
        out.reginv = self.reginv.copy()
        out.proj = self.proj.copy()
        out.whitener = self.whitener.copy()
        out.colorer = self.colorer.copy()
        out.noisenorm = self.noisenorm.copy()
        return out

    def copy(self):
        return self._copy(False)

    @property
    def _is_surf_ori(self) -> bool:
        if self.orient_prior is None:
            return False
        p = self.orient_prior.__val__()
        for i in range(p.shape[0]):
            if abs(p[i] - 1.0) > 1e-8 + 1e-5:
                return True
        return False

    @property
    def nchan(self) -> int:
        return len(self.ch_names)

    def __getitem__(self, key: Literal[str]):
        if key == "nsource":
            return self.nsource
        elif key == "source_ori":
            return self.source_ori
        elif key == "nave":
            return self.nave
        elif key == "sing":
            return self.sing
        elif key == "source_nn":
            return self.source_nn
        elif key == "info":
            return self.info
        elif key == "src":
            return self.src
        elif key == "noise_cov":
            return self.noise_cov
        elif key == "projs":
            return self.projs
        elif key == "coord_frame":
            return self.coord_frame
        elif key == "eigen_leads_weighted":
            return self.eigen_leads_weighted
        elif key == "reginv":
            return self.reginv
        elif key == "whitener":
            return self.whitener
        elif key == "colorer":
            return self.colorer
        elif key == "proj":
            return self.proj
        elif key == "noisenorm":
            return self.noisenorm
        else:
            compile_error("unsupported InverseOperator key")

    def __repr__(self) -> str:
        ori = "Free" if self.source_ori == FIFFV_MNE_FREE_ORI else "Fixed"
        if ori == "Free" and self._is_surf_ori:
            lo = float(np.min(self.orient_prior.__val__()))
            ori = f"Loose ({lo})"
        return f"<InverseOperator | channels: {len(self.ch_names)}, source space: {self.src.kind} with {self.nsource} sources, source orientation: {ori}>"


def _check_method(method: str):
    if method not in INVERSE_METHODS:
        raise ValueError(f"Invalid value for the 'method' parameter. Allowed values are 'MNE', 'dSPM', 'sLORETA' and 'eLORETA', but got '{method}' instead.")


def combine_xyz(vec, square: bool = False):
    v = np.asarray(vec)
    if v.ndim != 2:
        raise ValueError("Input must be 2D")
    if v.shape[0] % 3 != 0:
        raise ValueError("Input must have 3N rows")
    comb = v[0::3] ** 2
    comb = comb + v[1::3] ** 2
    comb = comb + v[2::3] ** 2
    if not square:
        comb = np.sqrt(comb)
    return comb


def compute_rank_inverse(inv: InverseOperator) -> int:
    """mne.minimum_norm.compute_rank_inverse."""
    if not inv.noise_cov.diag:
        eig = inv.noise_cov.eig.__val__()
        return int(np.sum(eig > 0))
    _, ncomp, _ = make_projector(inv.projs, inv.noise_cov.names)
    return inv.noise_cov.dim - ncomp


def _compute_reginv(inv: InverseOperator, lambda2: float):
    sing = inv.sing.copy()
    reginv = np.zeros(sing.shape[0])
    n_nzero = compute_rank_inverse(inv)
    for i in range(min(n_nzero, sing.shape[0])):
        if sing[i] > 0:
            reginv[i] = sing[i] / (sing[i] ** 2 + lambda2)
    return reginv


def _pick_channels_inverse_operator(ch_names: List[str], inv: InverseOperator) -> List[int]:
    sel = List[int]()
    for name in inv.noise_cov.names:
        if name not in ch_names:
            raise ValueError(f"The inverse operator was computed with channel {name} which is not present in the data. You should compute a new inverse operator restricted to the good data channels.")
        sel.append(ch_names.index(name))
    return sel


def _check_ch_names(inv: InverseOperator, info: Info):
    if inv.noise_cov.names != inv.ch_names:
        raise ValueError("Channels in inverse operator eigen fields do not match noise covariance channels.")
    missing = sorted([c for c in inv.ch_names if c not in info.ch_names])
    if len(missing) > 0:
        raise ValueError(f"{len(missing)} channels in inverse operator are not present in the data ({missing})")


def _eloreta_opts(mp):
    eps = 1e-6
    max_iter = 20
    force_equal = False
    if not isinstance(mp, NoneType):
        for k, v in mp.items():
            key = str(k)
            if key == "eps":
                eps = float(v)
            elif key == "max_iter":
                max_iter = int(v)
            elif key == "force_equal":
                force_equal = float(v) != 0.0
            else:
                raise ValueError(f"Invalid eLORETA option {key}")
    return eps, max_iter, force_equal


def _sqrtm_sym(A, inv: Literal[bool] = False):
    """mne.utils.sqrtm_sym for one symmetric matrix: (A^(+-1/2), sqrt eigenvalues)."""
    s, u = np.linalg.eigh(A)
    n = s.shape[0]
    limit = s[n - 1] * 1e-7
    for i in range(n):
        if s[i] < -limit:
            raise ValueError("Matrix is not positive semi-definite")
    sq = np.empty(n)
    use = np.empty(n)
    for i in range(n):
        v = s[i]
        if v <= limit:
            v = np.inf if inv else 0.0
        sq[i] = np.sqrt(v)
        use[i] = 1.0 / sq[i] if inv else sq[i]
    out = np.dot(u * use[None, :], u.T)
    return out, sq


def _eloreta_rgt(G, Rd, Rb, block: bool, n_src: int):
    """G @ (R @ G.T) for diagonal (Rd) or 3x3-block (Rb) R."""
    n_chan = G.shape[0]
    if not block:
        return np.dot(G, Rd[:, None] * G.T)
    RGt = np.empty((n_src * 3, n_chan))
    for s in range(n_src):
        RGt[3 * s:3 * s + 3] = np.dot(Rb[s], G[:, 3 * s:3 * s + 3].T)
    return np.dot(G, RGt)


def _eloreta_normalize(G, Rd, Rb, block: bool, n_src: int, n_nzero: int):
    """_normalize_R: rescale R in place so that trace(G R G') = n_nzero."""
    GRG = _eloreta_rgt(G, Rd, Rb, block, n_src)
    norm = float(np.trace(GRG)) / float(n_nzero)
    GRG = GRG / norm
    if block:
        Rb /= norm
    else:
        Rd /= norm
    return GRG


def _eloreta_R_sqrt_mult(other, Rd, Rb, block: bool, n_src: int):
    if not block:
        return Rd[None, :] * other
    out = np.empty(other.shape)
    for s in range(n_src):
        out[:, 3 * s:3 * s + 3] = np.dot(Rb[s], other[:, 3 * s:3 * s + 3].T).T
    return out


def _compute_eloreta(inv: InverseOperator, lambda2: float, eps: float, max_iter: int, force_equal: bool):
    """mne.minimum_norm._eloreta._compute_eloreta."""
    if inv.eigen_leads_weighted:
        raise RuntimeError("eLORETA cannot be computed with weighted eigen leads")
    G = np.dot(inv.eigen_fields.T * inv.sing[None, :], inv.eigen_leads.T)
    n_nzero = compute_rank_inverse(inv)
    n_src = inv.nsource
    n_chan = G.shape[0]
    n_orient = G.shape[1] // n_src
    for j in range(G.shape[1]):
        G[:, j] = G[:, j] / np.sqrt(inv.source_cov[j])
    source_std = np.ones(G.shape[1])
    if inv.orient_prior is not None:
        source_std = source_std * np.sqrt(inv.orient_prior.__val__())
    G = G * source_std[None, :]
    block = n_orient != 1 and not force_equal
    Rd = np.ones(0)
    Rb = np.zeros((0, 3, 3))
    Pd = np.ones(0)
    Pb = np.zeros((0, 3, 3))
    if block:
        Pb = np.empty((n_src, 3, 3))
        Rb = np.empty((n_src, 3, 3))
        for s in range(n_src):
            for i in range(3):
                for j in range(3):
                    Pb[s, i, j] = source_std[3 * s + i] * source_std[3 * s + j]
                    Rb[s, i, j] = Pb[s, i, j] if i == j else 0.0
                    # R starts as eye * prior
            for i in range(3):
                for j in range(3):
                    if i != j:
                        Rb[s, i, j] = 0.0
    else:
        Pd = source_std ** 2
        Rd = Pd.copy()
    GRG = _eloreta_normalize(G, Rd, Rb, block, n_src, n_nzero)
    for kk in range(max_iter):
        s, u = np.linalg.eigh(GRG)
        s = np.abs(s)
        sidx = np.argsort(s)[::-1][:n_nzero]
        ss = np.empty(n_nzero)
        uu = np.empty((GRG.shape[0], n_nzero))
        for i in range(n_nzero):
            v = s[sidx[i]]
            ss[i] = 1.0 / (v + lambda2) if v > 0 else 0.0
            uu[:, i] = u[:, sidx[i]]
        N = np.dot(uu * ss[None, :], uu.T)
        Rd_last = Rd.copy()
        Rb_last = Rb.copy()
        if n_orient == 1:
            NG = np.dot(N, G)
            for j in range(G.shape[1]):
                Rd[j] = 1.0 / np.sqrt(np.sum(NG[:, j] * G[:, j]))
            Rd = Rd * Pd
        else:
            for sI in range(n_src):
                G3 = G[:, 3 * sI:3 * sI + 3].T
                M = np.dot(G3, np.dot(N, G3.T))
                R3, sv = _sqrtm_sym(M, True)
                if force_equal:
                    m = 1.0 / ((sv[0] + sv[1] + sv[2]) / 3.0)
                    Rd[3 * sI] = m
                    Rd[3 * sI + 1] = m
                    Rd[3 * sI + 2] = m
                else:
                    Rb[sI] = R3
            if force_equal:
                Rd = Rd * Pd
            else:
                Rb = Rb * Pb
        GRG = _eloreta_normalize(G, Rd, Rb, block, n_src, n_nzero)
        num = 0.0
        den = 0.0
        if block:
            num = float(np.sum((Rb - Rb_last) ** 2))
            den = float(np.sum(Rb_last ** 2))
        else:
            num = float(np.sum((Rd - Rd_last) ** 2))
            den = float(np.sum(Rd_last ** 2))
        if np.sqrt(num) / np.sqrt(den) < eps:
            break
    G = G / source_std[None, :]
    _eloreta_normalize(G, Rd, Rb, block, n_src, n_nzero)
    Rsd = np.ones(0)
    Rsb = np.zeros((0, 3, 3))
    if block:
        Rsb = np.empty((n_src, 3, 3))
        for s in range(n_src):
            Rsb[s], _ = _sqrtm_sym(Rb[s], False)
    else:
        Rsd = np.sqrt(Rd)
    A = _eloreta_R_sqrt_mult(G, Rsd, Rsb, block, n_src)
    ef, sing, vh = np.linalg.svd(A, full_matrices=False)
    inv.sing = sing
    inv.reginv = _compute_reginv(inv, lambda2)
    inv.eigen_leads_weighted = True
    inv.eigen_leads = _eloreta_R_sqrt_mult(vh, Rsd, Rsb, block, n_src).T.copy()
    inv.eigen_fields = ef.T.copy()
    inv.source_cov = np.full(inv.source_cov.shape[0], np.nan)


def prepare_inverse_operator(orig: InverseOperator, nave: float, lambda2: float,
                             method: str = "dSPM", method_params = None, copy = True,
                             verbose = None) -> InverseOperator:
    """mne.minimum_norm.prepare_inverse_operator."""
    if nave <= 0:
        raise ValueError("The number of averages should be positive")
    _check_method(method)
    inv = orig._copy(not (isinstance(copy, bool) and copy)) if copy else orig
    scale = float(inv.nave) / float(nave)
    nc = inv.noise_cov
    nc._C = nc._C * scale
    if nc.diag:
        # operator covariances are stored as full matrices: eig is scaled once
        nc.eig = np.diag(nc._C).copy()
        nc.eigvec = np.eye(nc._C.shape[0])
    else:
        nc.eig = nc.eig.__val__() * scale
    inv.source_cov = inv.source_cov * scale
    if inv.eigen_leads_weighted:
        inv.eigen_leads = inv.eigen_leads * np.sqrt(scale)
    inv.nave = float(nave)
    inv.reginv = _compute_reginv(inv, lambda2)
    proj, ncomp, _ = make_projector(inv.projs, nc.names)
    inv.proj = proj
    W, Cc, _ = _whitener_from_prepared(nc, "white")
    inv.whitener = W
    inv.colorer = Cc
    inv.noisenorm = np.zeros(0)
    if method == "eLORETA":
        eps, max_iter, force_equal = _eloreta_opts(method_params)
        _compute_eloreta(inv, lambda2, eps, max_iter, force_equal)
    elif method != "MNE":
        nw = inv.reginv.copy()
        if method == "sLORETA":
            nw = inv.reginv * np.sqrt(1.0 + inv.sing ** 2 / lambda2)
        nrow = inv.eigen_leads.shape[0]
        nn = np.zeros(nrow)
        for k in range(nrow):
            if inv.eigen_leads_weighted:
                one = inv.eigen_leads[k, :] * nw
            else:
                one = np.sqrt(inv.source_cov[k]) * inv.eigen_leads[k, :] * nw
            nn[k] = np.sqrt(np.sum(one * one))
        if inv.source_ori == FIFFV_MNE_FREE_ORI:
            nn = combine_xyz(nn.reshape((-1, 1))).ravel()
        inv.noisenorm = 1.0 / np.abs(nn)
    inv.prepared = True
    return inv
