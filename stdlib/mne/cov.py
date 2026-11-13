# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne.cov: Covariance container, make_ad_hoc_cov, compute_covariance,
# compute_raw_covariance, regularize, prepare_noise_cov, compute_whitener,
# whiten_evoked and the rank helpers they rely on (mne.rank._compute_rank).

import numpy as np
from ._info import Info, Projection, _picks_by_type, _picks_to_idx, _time_mask
from ._proj import make_projector

_EPS64 = 2.220446049250313e-16
_LOG_2PI = 1.8378770664093453


def _scal_default(ch_type: str) -> float:
    """mne.defaults DEFAULTS['scalings'] (used by compute_covariance)."""
    if ch_type == "mag":
        return 1e15
    if ch_type == "grad":
        return 1e13
    if ch_type in ("eeg", "dbs", "ecog", "hbo", "hbr", "eog", "emg", "ecg", "bio"):
        return 1e6
    if ch_type in ("seeg", "csd", "pupil"):
        return 1e3
    if ch_type == "fnirs_td_moments_mean":
        return 1e12
    if ch_type == "fnirs_td_moments_variance":
        return 1e24
    return 1.0


def _scal_cov_rank(ch_type: str) -> float:
    """mne.defaults DEFAULTS['scalings_cov_rank'] merged with the
    _check_scaling_inputs fallback dict(mag=1e15, grad=1e13, eeg=1e6)."""
    if ch_type == "mag":
        return 1e12
    if ch_type == "grad":
        return 1e11
    if ch_type == "eeg":
        return 1e5
    if ch_type == "seeg":
        return 1e1
    if ch_type in ("dbs", "ecog", "hbo", "hbr"):
        return 1e4
    return 1.0


class _Scalings:
    """Per-channel-type scaling dict (user values override a default table)."""
    kind: str      # 'data' (DEFAULTS['scalings']) or 'cov_rank'
    user: Dict[str, float]

    def __init__(self, kind: str, user: Dict[str, float]):
        self.kind = kind
        self.user = user

    def get(self, ch_type: str) -> float:
        if ch_type in self.user:
            return self.user[ch_type]
        if self.kind == "data":
            return _scal_default(ch_type)
        return _scal_cov_rank(ch_type)


def _make_scalings(scalings, kind: str) -> _Scalings:
    u = Dict[str, float]()
    if isinstance(scalings, NoneType):
        pass
    elif isinstance(scalings, Dict):
        for k, v in scalings.items():
            u[str(k)] = float(v)
    else:
        compile_error("scalings must be a dict or None")
    return _Scalings(kind, u)


def _scales_vec(n: int, picks_list: List[Tuple[str, List[int]]], sc: _Scalings):
    s = np.ones(n)
    for ct, picks in picks_list:
        v = sc.get(ct)
        for p in picks:
            s[p] = v
    return s


class _Rank:
    """Normalized `rank` argument: kind in 'estimated' | 'info' | 'full'."""
    kind: str
    given: Dict[str, int]

    def __init__(self, kind: str, given: Dict[str, int]):
        self.kind = kind
        self.given = given


def _check_rank(rank) -> _Rank:
    if isinstance(rank, NoneType):
        return _Rank("estimated", Dict[str, int]())
    elif isinstance(rank, _Rank):
        return rank
    elif isinstance(rank, str):
        if rank == "info" or rank == "full":
            return _Rank(rank, Dict[str, int]())
        if rank == "auto":
            return _Rank("estimated", Dict[str, int]())
        raise ValueError(f"rank, if str, must be 'info' or 'full', got {rank}")
    elif isinstance(rank, Dict):
        g = Dict[str, int]()
        for k, v in rank.items():
            g[str(k)] = int(v)
        return _Rank("estimated", g)
    else:
        compile_error("rank must be None, 'info', 'full' or a dict")


def _estimate_rank_from_s(s) -> int:
    max_s = float(np.amax(s))
    tol = s.shape[0] * max_s * _EPS64
    return int(np.sum(s > tol))


def _rank_lookup(rank: Dict[str, int], ch_type: str) -> int:
    if ch_type not in rank and (ch_type == "mag" or ch_type == "grad"):
        return rank["meg"]
    return rank[ch_type]


def _compute_rank_cov(C, info: Info, rank: _Rank, scalings: _Scalings,
                      diag: bool = False) -> Dict[str, int]:
    """mne.rank._compute_rank for a (square) covariance whose channels match
    info.ch_names (bads already removed); proj from info.projs."""
    out = Dict[str, int]()
    for k, v in rank.given.items():
        out[k] = v
    for ch_type, picks in _picks_by_type(info, meg_combined=True, exclude_bads=False):
        if ch_type in out:
            if out[ch_type] > len(picks):
                raise ValueError(f"rank[{repr(ch_type)}]={out[ch_type]} exceeds the number of channels ({len(picks)})")
            continue
        names = [info.ch_names[p] for p in picks]
        proj_op, n_proj, _ = make_projector(info.projs, names)
        this_rank = 0
        if rank.kind == "info" or rank.kind == "full":
            this_rank = len(picks)
            if rank.kind != "full":
                this_rank -= n_proj
        elif diag:
            this_rank = -n_proj
            for p in picks:
                if C[p, p] > 0:
                    this_rank += 1
        else:
            pa = np.array(picks)
            data = C[np.ix_(pa, pa)]
            data = np.dot(np.dot(proj_op, data), proj_op.T)
            sub_types = [info.ch_types[p] for p in picks]
            sv = np.array([scalings.get(t) for t in sub_types])
            data = data * (sv[None, :] * sv[:, None])
            s = np.linalg.svd(data, compute_uv=False)
            this_rank = _estimate_rank_from_s(s)
        out[ch_type] = this_rank
    return out


def _compute_rank_data(data, info: Info, rank: _Rank, scalings: _Scalings) -> Dict[str, int]:
    """mne.rank._compute_rank for (n_channels, n_samples) data (Raw/Epochs path)."""
    out = Dict[str, int]()
    for k, v in rank.given.items():
        out[k] = v
    for ch_type, picks in _picks_by_type(info, meg_combined=True):
        if ch_type in out:
            if out[ch_type] > len(picks):
                raise ValueError(f"rank[{repr(ch_type)}]={out[ch_type]} exceeds the number of channels ({len(picks)})")
            continue
        names = [info.ch_names[p] for p in picks]
        proj_op, n_proj, _ = make_projector(info.projs, names)
        this_rank = 0
        if rank.kind == "info" or rank.kind == "full":
            this_rank = len(picks)
            if rank.kind != "full":
                this_rank -= n_proj
        else:
            pa = np.array(picks)
            sub = np.dot(proj_op, data[pa, :])
            for ii in range(len(picks)):
                sub[ii, :] *= scalings.get(info.ch_types[picks[ii]])
            s = np.linalg.svd(sub, compute_uv=False)
            this_rank = _estimate_rank_from_s(s)
        out[ch_type] = this_rank
    return out


def _get_ch_whitener(A, rank: int):
    eig, eigvec = np.linalg.eigh(A)
    eigvec = eigvec.T.copy()
    n = eig.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n - rank):
        eig[i] = 0.0
        mask[i] = False
    return eig, eigvec, mask


def _eigvec_subspace(eigvec, mask):
    n = eigvec.shape[0]
    bad = eigvec[~mask]
    P = np.eye(n) - np.dot(bad.T, bad)
    eig, ev = np.linalg.eigh(P)
    return eig, ev.T.copy()


def _smart_eigh(C, info: Info, rank: _Rank, scalings: _Scalings, projs: List[Projection],
                proj_subspace: bool = False, do_compute_rank: bool = True):
    """mne.cov._smart_eigh (info channels == C channels)."""
    n_chan = C.shape[0]
    proj, ncomp, _ = make_projector(projs, info.ch_names)
    if ncomp > 0:
        C = np.dot(proj, np.dot(C, proj.T))
    rk = Dict[str, int]()
    if do_compute_rank:
        # mne _compute_rank(Covariance, ...) uses info['projs'] only
        info2 = info.copy()
        info2.bads = List[str]()
        rk = _compute_rank_cov(C, info2, rank, scalings)
    else:
        rk = rank.given
    tot = 0
    for v in rk.values():
        tot += v
    if proj_subspace and tot == n_chan:
        return np.ones(n_chan), np.eye(n_chan), np.ones(n_chan, dtype=bool)
    eig = np.zeros(n_chan)
    eigvec = np.zeros((n_chan, n_chan))
    mask = np.zeros(n_chan, dtype=bool)
    for ch_type, picks in _picks_by_type(info, meg_combined=True, exclude_bads=False):
        if len(picks) == 0:
            continue
        pa = np.array(picks)
        this_C = C[np.ix_(pa, pa)]
        this_rank = _rank_lookup(rk, ch_type)
        e, ev, m = _get_ch_whitener(this_C, this_rank)
        if proj_subspace:
            e, ev = _eigvec_subspace(ev, m)
        for i in range(len(picks)):
            eig[picks[i]] = e[i]
            mask[picks[i]] = m[i]
            for j in range(len(picks)):
                eigvec[picks[i], picks[j]] = ev[i, j]
    return eig, eigvec, mask


class Covariance:
    """Noise covariance matrix (mne.Covariance).

    The matrix is always stored square in `_C`; `diag` marks diagonal
    covariances (whose mne `.data` is the 1-D diagonal, see `data`).
    """
    _C: np.ndarray[float, 2]
    names: List[str]
    bads: List[str]
    projs: List[Projection]
    nfree: int
    diag: bool
    eig: Optional[np.ndarray[float, 1]]
    eigvec: Optional[np.ndarray[float, 2]]
    method: Optional[str]
    loglik: Optional[float]

    def __init__(self, data, names, bads: List[str], projs: List[Projection], nfree: int,
                 eig: Optional[np.ndarray[float, 1]] = None,
                 eigvec: Optional[np.ndarray[float, 2]] = None,
                 method: Optional[str] = None, loglik: Optional[float] = None, *, verbose = None):
        a = np.asarray(data, dtype=float)
        if a.ndim == 1:
            self._C = np.diag(a)
            self.diag = True
        else:
            self._C = a.copy()
            self.diag = False
        self.names = [str(n) for n in names]
        self.bads = [str(b) for b in bads]
        self.projs = [p.copy() for p in projs]
        self.nfree = nfree
        self.eig = eig
        self.eigvec = eigvec
        self.method = method
        self.loglik = loglik

    @property
    def data(self):
        """Square covariance (diagonal covariances: use `diag_data`)."""
        return self._C

    @property
    def diag_data(self):
        return np.diag(self._C).copy()

    @property
    def ch_names(self) -> List[str]:
        return self.names

    @property
    def dim(self) -> int:
        return len(self.names)

    def copy(self):
        c = Covariance(self._C.copy(), list(self.names), list(self.bads), self.projs, self.nfree,
                       method=self.method, loglik=self.loglik)
        c.diag = self.diag
        if self.eig is not None:
            c.eig = self.eig.__val__().copy()
        if self.eigvec is not None:
            c.eigvec = self.eigvec.__val__().copy()
        return c

    def as_diag(self):
        if self.diag:
            return self
        self.diag = True
        self._C = np.diag(np.diag(self._C).copy())
        self.eig = None
        self.eigvec = None
        return self

    def _get_square(self):
        return self._C.copy()

    def __repr__(self) -> str:
        kind = "diagonal" if self.diag else "full"
        return f"<Covariance | kind : {kind}, shape {self.dim} × {self.dim}, n_samples : {self.nfree}>"

    def __add__(self, cov: Covariance):
        if self.names != cov.names:
            raise ValueError("Both Covariance do not have the same list of channels.")
        if self.diag != cov.diag:
            raise ValueError("Covariances must both be diagonal or both be full.")
        out = cov.copy()
        out._C = (cov._C * cov.nfree + self._C * self.nfree) / (self.nfree + cov.nfree)
        out.nfree += self.nfree
        b = Set[str](cov.bads)
        for x in self.bads:
            b.add(x)
        out.bads = sorted(list(b))
        return out

    def __getitem__(self, key: Literal[str]):
        if key == "data":
            return self.data
        elif key == "names":
            return self.names
        elif key == "bads":
            return self.bads
        elif key == "nfree":
            return self.nfree
        elif key == "diag":
            return self.diag
        elif key == "dim":
            return self.dim
        elif key == "projs":
            return self.projs
        elif key == "eig":
            return self.eig
        elif key == "eigvec":
            return self.eigvec
        elif key == "method":
            return self.method
        elif key == "loglik":
            return self.loglik
        else:
            compile_error("unsupported Covariance key")

    def pick_channels(self, ch_names, ordered: bool = True, *, verbose = None):
        return pick_channels_cov(self, ch_names, (), ordered, False)


def _pick_names(names: List[str], include: List[str], exclude: List[str], ordered: bool) -> List[int]:
    """mne.pick_channels (include empty -> all)."""
    pos = Dict[str, int]()
    for i in range(len(names)):
        pos[names[i]] = i
    inc = include if len(include) > 0 else names
    sel = List[int]()
    missing = List[str]()
    for n in inc:
        if n not in pos:
            missing.append(n)
        elif n not in exclude:
            sel.append(pos[n])
    if len(missing) > 0 and ordered:
        raise ValueError(f"Missing channels from ch_names required by include:\n{missing}")
    if not ordered:
        sel = sorted(list(Set[int](sel)))
    return sel


def pick_channels_cov(orig: Covariance, include = (), exclude = "bads", ordered: bool = False,
                      copy: bool = True, *, verbose = None) -> Covariance:
    """mne.pick_channels_cov."""
    c = orig.copy() if copy else orig
    ex = List[str]()
    if isinstance(exclude, str):
        if exclude == "bads":
            ex = list(orig.bads)
        else:
            ex = [exclude]
    else:
        ex = [str(e) for e in exclude]
    sel = _pick_names(orig.names, [str(i) for i in include], ex, ordered)
    sa = np.array(sel, dtype=int)
    c._C = c._C[np.ix_(sa, sa)].copy()
    c.names = [orig.names[k] for k in sel]
    c.bads = [n for n in orig.bads if n in orig.names]
    return c


def make_ad_hoc_cov(info: Info, std = None, *, verbose = None) -> Covariance:
    """mne.make_ad_hoc_cov (defaults: grad 5 fT/cm, mag 20 fT, eeg 0.2 uV)."""
    sg = 5e-13
    sm = 20e-15
    se = 0.2e-6
    if not isinstance(std, NoneType):
        if isinstance(std, Dict):
            for k, v in std.items():
                if k == "grad":
                    sg = float(v)
                elif k == "mag":
                    sm = float(v)
                elif k == "eeg":
                    se = float(v)
                else:
                    raise ValueError(f"Unknown std key {k}")
        else:
            compile_error("std must be a dict or None")
    picks = List[int]()
    vals = List[float]()
    for i in range(info.nchan):
        t = info.ch_types[i]
        if t == "grad" or t == "mag" or t == "eeg":
            picks.append(i)
            bad = info.ch_names[i] in info.bads
            v = sg if t == "grad" else (sm if t == "mag" else se)
            vals.append(0.0 if bad else v * v)
    names = [info.ch_names[p] for p in picks]
    return Covariance(np.array(vals), names, list(info.bads), info.projs, 0)


# ---------------------------------------------------------------------------
# estimators (sklearn.covariance subset used by mne.cov, assume_centered=True)

def _emp_cov(X):
    return np.dot(X.T, X) / X.shape[0]


def _pinvh(a):
    """scipy.linalg.pinvh (atol=0, rtol=n*eps)."""
    s, u = np.linalg.eigh(a)
    n = a.shape[0]
    maxS = float(np.max(np.abs(s)))
    val = maxS * n * _EPS64
    keep = np.abs(s) > val
    U = u[:, keep]
    psig = 1.0 / s[keep]
    return np.dot(U * psig, U.T)


def _logdet_psd(A) -> float:
    """mne.fixes._logdet."""
    vals = np.linalg.eigvalsh(A)
    tol = float(np.max(vals)) * vals.shape[0] * _EPS64
    out = 0.0
    for v in vals:
        out += np.log(v if v > tol else tol)
    return out


def _fast_logdet(A) -> float:
    sign, ld = np.linalg.slogdet(A)
    if not sign > 0:
        return -np.inf
    return float(ld)


def _sk_log_likelihood(emp_cov, precision) -> float:
    p = precision.shape[0]
    ll = -float(np.sum(emp_cov * precision)) + _fast_logdet(precision)
    ll -= p * np.log(2 * np.pi)
    return ll / 2.0


def _gaussian_loglik(cov, X) -> float:
    """mne.cov._gaussian_loglik_scorer with precision = pinvh(cov)."""
    precision = _pinvh(cov)
    n_features = X.shape[1]
    ll = -0.5 * np.sum(X * np.dot(X, precision), axis=1)
    ll -= 0.5 * (n_features * _LOG_2PI - _logdet_psd(precision))
    return float(np.mean(ll))


def _kfold(n: int, k: int) -> List[Tuple[np.ndarray[int, 1], np.ndarray[int, 1]]]:
    """sklearn KFold(k) (no shuffle) train/test indices."""
    if k > n:
        raise ValueError(f"Cannot have number of splits n_splits={k} greater than the number of samples: n_samples={n}.")
    out = List[Tuple[np.ndarray[int, 1], np.ndarray[int, 1]]]()
    cur = 0
    for f in range(k):
        size = n // k + (1 if f < n % k else 0)
        test = np.arange(cur, cur + size)
        train = np.concatenate((np.arange(0, cur), np.arange(cur + size, n)))
        out.append((train, test))
        cur += size
    return out


class _Shrinks:
    items: List[Tuple[str, float, List[int]]]

    def __init__(self, items: List[Tuple[str, float, List[int]]]):
        self.items = items


def _shrunk_cov_fit(X, shr: _Shrinks):
    """mne.cov._ShrunkCovariance.fit."""
    from ._cov import _shrunk_blocks
    return _shrunk_blocks(X, shr.items)


def _sk_shrunk(emp, s: float):
    n = emp.shape[0]
    out = (1.0 - s) * emp
    mu = np.trace(emp) / n
    out += s * mu * np.eye(n)
    return out


class _CovSpec:
    """A covariance estimator recipe (re-fittable on CV folds)."""
    method: str
    shr: _Shrinks
    regs: Dict[str, float]
    info: Info

    def __init__(self, method: str, shr: _Shrinks, regs: Dict[str, float], info: Info):
        self.method = method
        self.shr = shr
        self.regs = regs
        self.info = info

    def fit(self, X, cv_clone: bool = False):
        if self.method == "empirical":
            return _emp_cov(X)
        elif self.method == "diagonal_fixed":
            # sklearn.clone of mne's _RegCovariance drops its keyword-only
            # params, so CV refits use assume_centered=False.
            c = _emp_cov(X - X.mean(axis=0)[None, :]) if cv_clone else _emp_cov(X)
            c = 0.5 * (c + c.T)
            return _reg_full_noproj(c, self.info, self.regs)
        else:
            return _shrunk_cov_fit(X, self.shr)


def _reg_full_noproj(C, info: Info, regs: Dict[str, float]):
    """regularize(cov, info, proj=False, rank='full') on a cov matching info."""
    C = C.copy()
    for ct, picks in _picks_by_type(info, meg_combined=False):
        reg = regs[ct] if ct in regs else 0.1
        if reg == 0.0 or len(picks) == 0:
            continue
        pa = np.array(picks)
        this_C = C[np.ix_(pa, pa)]
        sigma = float(np.mean(np.diag(this_C)))
        for i in range(len(picks)):
            this_C[i, i] += reg * sigma
        for i in range(len(picks)):
            for j in range(len(picks)):
                C[picks[i], picks[j]] = this_C[i, j]
    return C


def _cross_val(data, spec: _CovSpec, cv: int) -> float:
    tot = 0.0
    folds = _kfold(data.shape[0], cv)
    for tr, te in folds:
        c = spec.fit(data[tr], True)
        tot += _gaussian_loglik(c, data[te])
    return tot / len(folds)


def _grid_shrinkage(X, cv: int, grid: List[float]) -> float:
    """GridSearchCV(ShrunkCovariance(assume_centered=True), shrinkage grid, cv)."""
    folds = _kfold(X.shape[0], cv)
    best = -np.inf
    best_s = grid[0]
    for s in grid:
        tot = 0.0
        for tr, te in folds:
            c = _sk_shrunk(_emp_cov(X[tr]), s)
            tot += _sk_log_likelihood(_emp_cov(X[te]), _pinvh(c))
        sc = tot / len(folds)
        if sc > best:
            best = sc
            best_s = s
    return best_s


_COV_METHODS = ("empirical", "diagonal_fixed", "ledoit_wolf", "oas", "shrunk", "shrinkage")


def _method_list(method) -> List[str]:
    out = List[str]()
    if isinstance(method, NoneType):
        out.append("empirical")
    elif isinstance(method, str):
        if method == "auto":
            out.extend(["shrunk", "diagonal_fixed", "empirical"])
        else:
            out.append(method)
    else:
        for m in method:
            out.append(str(m))
    for m in out:
        if m not in _COV_METHODS:
            if m == "pca" or m == "factor_analysis":
                raise ValueError(f"{m} can so far only be used with rank=\"full\" (not supported in Typhon mne)")
            raise ValueError(f"Invalid method ({out}). Accepted values (individually or in a list) are any of '{_COV_METHODS}' or None.")
    return out


class _MethodParams:
    shrinkage: float
    shrunk_grid: List[float]
    regs: Dict[str, float]

    def __init__(self):
        self.shrinkage = 0.1
        g = np.logspace(-4, 0, 30)
        self.shrunk_grid = [float(g[i]) for i in range(30)]
        self.regs = Dict[str, float]()


def _method_params(method_params) -> _MethodParams:
    mp = _MethodParams()
    if isinstance(method_params, NoneType):
        return mp
    for key, sub in method_params.items():
        for k2, v in sub.items():
            if key == "shrinkage" and k2 == "shrinkage":
                mp.shrinkage = float(v)
            elif key == "shrunk" and k2 == "shrinkage":
                if isinstance(v, float) or isinstance(v, int):
                    mp.shrunk_grid = [float(v)]
                else:
                    mp.shrunk_grid = [float(x) for x in v]
            elif key == "diagonal_fixed" and k2 not in ("store_precision", "assume_centered"):
                mp.regs[k2] = float(v)
    if not (0 <= mp.shrinkage <= 1):
        raise ValueError(f"shrinkage must be between 0 and 1, got {mp.shrinkage}")
    return mp


def _compute_covariance_auto(data, methods: List[str], info: Info, mp: _MethodParams, cv: int,
                             scalings: _Scalings, rank: _Rank) -> List[Tuple[str, np.ndarray[float, 2], Optional[float]]]:
    """mne.cov._compute_covariance_auto; data is (n_samples, n_channels)."""
    n_ch = data.shape[1]
    picks_list = _picks_by_type(info)
    # _compute_rank gets the full DEFAULTS['scalings'] dict here
    rk = _compute_rank_data(data.T, info, rank, scalings)
    scales = _scales_vec(n_ch, picks_list, scalings)
    data = data * scales[None, :]
    C = np.dot(data.T, data)
    _, eigvec, mask = _smart_eigh(C, info, _Rank("estimated", rk), scalings, List[Projection](),
                                  proj_subspace=True, do_compute_rank=False)
    eigvec = eigvec[mask]
    data = np.dot(data, eigvec.T)
    used = [i for i in range(n_ch) if mask[i]]
    sub_picks = List[Tuple[str, List[int]]]()
    for ct, picks in picks_list:
        sp = List[int]()
        for p in picks:
            k = 0
            while k < len(used) and used[k] < p:
                k += 1
            sp.append(k)
        sub_picks.append((ct, sp))
    sub_info = info._pick(used) if len(used) != n_ch else info
    specs = List[_CovSpec]()
    for m in methods:
        shr = List[Tuple[str, float, List[int]]]()
        if m == "ledoit_wolf" or m == "oas":
            from ._cov import _ledoit_wolf_shrinkage, _oas_shrinkage
            for ct, sp in sub_picks:
                sub = data[:, np.array(sp)].copy()
                s = _ledoit_wolf_shrinkage(sub) if m == "ledoit_wolf" else _oas_shrinkage(sub)
                shr.append((ct, s, sp))
        elif m == "shrinkage":
            shr.append(("all", mp.shrinkage, list(range(data.shape[1]))))
        elif m == "shrunk":
            for ct, sp in sub_picks:
                sub = data[:, np.array(sp)].copy()
                shr.append((ct, _grid_shrinkage(sub, cv, mp.shrunk_grid), sp))
        specs.append(_CovSpec(m, _Shrinks(shr), mp.regs, sub_info))
    out = List[Tuple[str, np.ndarray[float, 2], Optional[float]]]()
    for ei in range(len(methods)):
        cov = specs[ei].fit(data)
        loglik: Optional[float] = None
        if len(methods) > 1:
            loglik = _cross_val(data, specs[ei], cv)
        cov = np.dot(eigvec.T, np.dot(cov, eigvec))
        ns = data.shape[0]
        cov *= ns / max(ns - 1, 1)
        inv = 1.0 / scales
        cov *= inv[None, :] * inv[:, None]
        out.append((methods[ei], cov, loglik))
    return out


def _epochs_list(inst) -> List[np.ndarray[float, 3]]:
    """_unpack_covariance_inputs: per-event-type data blocks (Evoked -> 1 epoch)."""
    out = List[np.ndarray[float, 3]]()
    if isinstance(inst, List):
        for e in inst:
            out.extend(_epochs_list(e))
    elif hasattr(inst, "nave"):
        out.append(inst.data[None, :, :].copy())
    else:
        if len(inst.event_id) > 1:
            for k in inst.event_id.keys():
                code = inst.event_id[k]
                sel = [i for i in range(inst.events.shape[0]) if int(inst.events[i, 2]) == code]
                out.append(inst._data[np.array(sel, dtype=int)].copy())
        else:
            out.append(inst._data)
    return out


def _first(inst):
    if isinstance(inst, List):
        return inst[0]
    else:
        return inst


def _tslice(times, sfreq: float, tmin: Optional[float], tmax: Optional[float]) -> Tuple[int, int]:
    """mne.cov._get_tslice."""
    n = times.shape[0]
    if tmin is None and tmax is None:
        return 0, n
    mask = _time_mask(times, tmin, tmax, sfreq)
    idx = [i for i in range(n) if mask[i]]
    a = idx[0] if tmin is not None else 0
    b = idx[-1] + 1 if tmax is not None else n
    return a, b


def compute_covariance(inst = None, keep_sample_mean: bool = True, tmin = None, tmax = None,
                       projs = None, method = "empirical", method_params = None, cv: int = 3,
                       scalings = None, n_jobs = None, return_estimators: Literal[bool] = False,
                       on_mismatch: str = "raise", rank = None, on_few_samples: str = "warn",
                       verbose = None):
    """mne.compute_covariance (Epochs / list of Epochs / Evoked)."""
    sc = _make_scalings(scalings, "data")
    methods = _method_list(method)
    rk = _check_rank(rank)
    mp = _method_params(method_params)
    if not keep_sample_mean:
        if len(methods) != 1 or methods[0] != "empirical":
            raise ValueError('`keep_sample_mean=False` is only supported with method="empirical"')
    inst0 = _first(inst)
    if hasattr(inst0, "nave") and not keep_sample_mean:
        raise ValueError("`keep_sample_mean=False` cannot be used when `inst` is an Evoked instance")
    info0 = inst0.info
    use_projs = list(info0.projs)
    if not isinstance(projs, NoneType):
        use_projs = [p.copy() for p in projs]
    blocks = _epochs_list(inst)
    pl = _picks_by_type(info0)
    pm = List[int]()
    for _, p in pl:
        pm.extend(p)
    pm = sorted(pm)
    ch_names = [info0.ch_names[k] for k in pm]
    info = info0._pick(pm)
    pa = np.array(pm, dtype=int)
    tmn: Optional[float] = None if isinstance(tmin, NoneType) else float(tmin)
    tmx: Optional[float] = None if isinstance(tmax, NoneType) else float(tmax)
    a, b = _tslice(inst0.times, info0.sfreq, tmn, tmx)
    data_mean = List[np.ndarray[float, 2]]()
    norm_const = 0
    if not keep_sample_mean:
        for blk in blocks:
            ne = blk.shape[0]
            s = np.sum(blk[:, pa, a:b], axis=0)
            data_mean.append(np.dot(s, s.T) / ne)
            norm_const += (b - a) * (ne - 1)
    cols = List[np.ndarray[float, 2]]()
    for blk in blocks:
        for e in range(blk.shape[0]):
            cols.append(blk[e][pa, a:b])
    X = np.hstack(cols)
    n_tot = X.shape[1]
    if n_tot <= 0:
        raise ValueError("No samples found to compute the covariance matrix")
    res = _compute_covariance_auto(X.T.copy(), methods, info, mp, cv, sc, rk)
    covs = List[Covariance]()
    for name, c, ll in res:
        if not keep_sample_mean:
            c = c * (n_tot - 1)
            for m in data_mean:
                c -= m
            c /= norm_const
        covs.append(Covariance(c, ch_names, list(info.bads), use_projs, n_tot - 1,
                               method=name, loglik=ll))
    if len(covs) > 1:
        covs.sort(key=lambda c: -c.loglik.__val__())
    if return_estimators:
        return covs
    else:
        return covs[0]


def compute_raw_covariance(raw, tmin: float = 0, tmax = None, tstep = 0.2, reject = None,
                           flat = None, picks = None, *, on_few_samples: str = "warn",
                           method = "empirical", method_params = None, cv: int = 3,
                           scalings = None, n_jobs = None, return_estimators: Literal[bool] = False,
                           reject_by_annotation: bool = True, rank = None, verbose = None):
    """mne.compute_raw_covariance (no rejection)."""
    if not isinstance(reject, NoneType) or not isinstance(flat, NoneType):
        compile_error("compute_raw_covariance(reject/flat) is not supported in Typhon mne")
    from .event import make_fixed_length_events
    from .epochs import Epochs
    tmn = float(tmin)
    dt = 1.0 / raw.info.sfreq
    tmx = float(raw.times[-1]) + dt if isinstance(tmax, NoneType) else float(tmax)
    ts = tmx - tmn if isinstance(tstep, NoneType) else float(tstep)
    events = make_fixed_length_events(raw, 1, tmn, tmx, ts)
    info = raw.info
    pk = List[int]()
    data_mask = List[bool]()
    if isinstance(picks, NoneType):
        for i in range(info.nchan):
            pk.append(i)
            t = info.ch_types[i]
            data_mask.append(t in ("mag", "grad", "eeg", "csd", "seeg", "ecog", "dbs", "hbo", "hbr")
                             and info.ch_names[i] not in info.bads)
    else:
        pk = _picks_to_idx(info, picks, "all", "bads")
        data_mask = [True for _ in pk]
    epochs = Epochs(raw, events, 1, 0.0, ts - dt, baseline=None, picks=pk, proj=False,
                    reject_by_annotation=reject_by_annotation)
    ms = _method_list(method)
    if len(ms) == 1 and ms[0] == "empirical":
        sel = [k for k in range(len(pk)) if data_mask[k]]
        sa = np.array(sel, dtype=int)
        nch = len(sel)
        data = np.zeros((nch, nch))
        mu = np.zeros(nch)
        n_samples = 0
        d = epochs._data
        for e in range(d.shape[0]):
            seg = d[e][sa]
            mu += seg.sum(axis=1)
            data += np.dot(seg, seg.T)
            n_samples += seg.shape[1]
        data -= mu[:, None] * (mu[None, :] / n_samples)
        data /= n_samples - 1.0
        ch_names = [info.ch_names[pk[k]] for k in sel]
        bads = [b for b in info.bads if b in ch_names]
        c = Covariance(data, ch_names, bads, info.projs, n_samples - 1)
        if return_estimators:
            return [c]
        else:
            return c
    d = epochs._data
    ch_means = d.mean(axis=0).mean(axis=1)
    epochs._data = d - ch_means[None, :, None]
    epochs.baseline = (0.0, 0.0)
    return compute_covariance(epochs, keep_sample_mean=True, method=method,
                              method_params=method_params, cv=cv, scalings=scalings,
                              return_estimators=return_estimators, rank=rank)


def _rank_of_cov(cov: Covariance, info: Info, rank: _Rank, scalings: _Scalings) -> Dict[str, int]:
    """mne.rank._compute_rank(Covariance, rank, scalings, info)."""
    names_in = Set[str](info.ch_names)
    inc = [n for n in cov.names if n in names_in]
    sub = pick_channels_cov(cov, inc, list(cov.bads), False)
    sel = [info.ch_names.index(n) for n in sub.names]
    inf2 = info._pick(sel)
    inf2.bads = List[str]()
    return _compute_rank_cov(sub._C, inf2, rank, scalings, sub.diag)


def _reg_value(ct: str, mag: float, grad: float, eeg: float, other: Dict[str, float]) -> float:
    if ct == "mag":
        return mag
    if ct == "grad":
        return grad
    if ct == "eeg":
        return eeg
    if ct in other:
        return other[ct]
    return 0.1


def regularize(cov: Covariance, info: Info, mag: float = 0.1, grad: float = 0.1, eeg: float = 0.1,
               exclude = "bads", proj: bool = True, *, seeg: float = 0.1, ecog: float = 0.1,
               hbo: float = 0.1, hbr: float = 0.1, csd: float = 0.1, dbs: float = 0.1,
               rank = None, scalings = None, verbose = None) -> Covariance:
    """mne.cov.regularize."""
    cov = cov.copy()
    sc = _make_scalings(scalings, "cov_rank")
    rk = _check_rank(rank)
    other = {"seeg": seeg, "ecog": ecog, "hbo": hbo, "hbr": hbr, "csd": csd, "dbs": dbs}
    ex = List[str]()
    if isinstance(exclude, str):
        if exclude != "bads":
            raise ValueError('exclude must be a list of strings or "bads"')
        ex = list(info.bads) + list(cov.bads)
    else:
        ex = [str(e) for e in exclude]
    full = rk.kind == "full"
    rank_d = Dict[str, int]()
    if not full:
        rank_d = _rank_of_cov(cov, info, rk, sc)
    by_type = Dict[str, List[str]]()
    order = List[str]()
    for i in range(info.nchan):
        t = info.ch_types[i]
        if info.ch_names[i] in ex or t == "ref_meg":
            continue
        if t not in by_type:
            by_type[t] = List[str]()
        by_type[t].append(info.ch_names[i])
    good = pick_channels_cov(cov, info.ch_names, ex, False)
    ch_names = good.names
    C = good._C
    projs = list(info.projs) + list(good.projs)
    for ct in by_type.keys():
        if ct not in ("mag", "grad", "eeg", "seeg", "ecog", "dbs", "hbo", "hbr", "csd"):
            continue
        idx = [i for i in range(len(ch_names)) if ch_names[i] in by_type[ct]]
        if len(idx) == 0:
            continue
        reg = _reg_value(ct, mag, grad, eeg, other)
        if reg == 0.0:
            continue
        ia = np.array(idx, dtype=int)
        this_C = C[np.ix_(ia, ia)]
        n = len(idx)
        U = np.eye(n)
        names = [ch_names[k] for k in idx]
        if full:
            if proj:
                P, ncomp, _ = make_projector(projs, names)
                if ncomp > 0:
                    u, _, _ = np.linalg.svd(P, full_matrices=False)
                    U = u[:, :n - ncomp].copy()
        else:
            this_info = info._pick([info.ch_names.index(nm) for nm in names])
            _, eigvec, mask = _smart_eigh(this_C, this_info, _Rank("estimated", rank_d), sc,
                                          this_info.projs)
            U = eigvec[mask].T.copy()
        this_C = np.dot(U.T, np.dot(this_C, U))
        sigma = float(np.mean(np.diag(this_C)))
        for i in range(this_C.shape[0]):
            this_C[i, i] += reg * sigma
        this_C = np.dot(U, np.dot(this_C, U.T))
        for i in range(n):
            for j in range(n):
                C[idx[i], idx[j]] = this_C[i, j]
    sel = _pick_names(cov.names, info.ch_names, ex, False)
    for i in range(len(sel)):
        for j in range(len(sel)):
            cov._C[sel[i], sel[j]] = C[i, j]
    return cov


def prepare_noise_cov(noise_cov: Covariance, info: Info, ch_names = None, rank = None,
                      scalings = None, on_rank_mismatch: str = "ignore", *, verbose = None) -> Covariance:
    """mne.cov.prepare_noise_cov."""
    names = list(info.ch_names) if isinstance(ch_names, NoneType) else [str(c) for c in ch_names]
    idx = List[int]()
    missing = List[str]()
    for c in names:
        if c in noise_cov.names:
            idx.append(noise_cov.names.index(c))
        else:
            missing.append(c)
    if len(missing) > 0:
        raise RuntimeError(f"Not all channels present in noise covariance:\n{missing}")
    ia = np.array(idx, dtype=int)
    C = noise_cov._C[np.ix_(ia, ia)].copy()
    info2 = info._pick([info.ch_names.index(c) for c in names])
    projs = list(info.projs) + list(noise_cov.projs)
    out = Covariance(C, names, list(noise_cov.bads), noise_cov.projs, noise_cov.nfree,
                     method=noise_cov.method, loglik=noise_cov.loglik)
    out.diag = noise_cov.diag
    eig, eigvec, _ = _smart_eigh(C, info2, _check_rank(rank), _make_scalings(scalings, "cov_rank"), projs)
    out.eig = eig
    out.eigvec = eigvec
    return out


def _whitener_from_prepared(noise_cov: Covariance, pca):
    eig = noise_cov.eig.__val__().copy()
    ev = noise_cov.eigvec.__val__()
    n = eig.shape[0]
    nz = eig > 0
    w = np.zeros(n)
    for i in range(n):
        if nz[i]:
            w[i] = 1.0 / np.sqrt(eig[i])
        else:
            eig[i] = 0.0
    W = w[:, None] * ev
    Cc = np.sqrt(eig) * ev.T
    n_nz = int(np.sum(nz))
    if isinstance(pca, bool):
        if pca:
            return W[nz].copy(), Cc[:, nz].copy(), n_nz
        else:
            return np.dot(ev.T, W), np.dot(Cc, ev), n_nz
    else:
        if pca != "white":
            raise ValueError(f"space must be one of (True, 'white', False), got {pca}")
        return W, Cc, n_nz


def compute_whitener(noise_cov: Covariance, info = None, picks = None, rank = None,
                     scalings = None, return_rank: Literal[bool] = False, pca = False,
                     return_colorer: Literal[bool] = False, on_rank_mismatch: str = "warn",
                     verbose = None):
    """mne.cov.compute_whitener."""
    nc = noise_cov
    ch_names = List[str]()
    if isinstance(info, NoneType):
        if nc.eig is None:
            raise ValueError("info can only be None if the noise cov has already been prepared with prepare_noise_cov")
        ch_names = list(nc.names)
    else:
        pk = _picks_to_idx(info, picks, "data", "bads")
        ch_names = [info.ch_names[k] for k in pk]
        nc = prepare_noise_cov(noise_cov, info, ch_names, rank, scalings)
    W, Cc, n_nz = _whitener_from_prepared(nc, pca)
    if return_rank and return_colorer:
        return W, ch_names, n_nz, Cc
    elif return_rank:
        return W, ch_names, n_nz
    elif return_colorer:
        return W, ch_names, Cc
    else:
        return W, ch_names


def whiten_evoked(evoked, noise_cov: Covariance, picks = None, diag = None, rank = None,
                  scalings = None, verbose = None):
    """mne.whiten_evoked."""
    evoked = evoked.copy()
    pk = _picks_to_idx(evoked.info, picks, "data", "bads")
    nc = noise_cov
    if not isinstance(diag, NoneType):
        if diag:
            nc = noise_cov.copy().as_diag()
    W, _ = compute_whitener(nc, evoked.info, pk, rank, scalings)
    pa = np.array(pk, dtype=int)
    evoked.data[pa] = np.sqrt(float(evoked.nave)) * np.dot(W, evoked.data[pa])
    return evoked
