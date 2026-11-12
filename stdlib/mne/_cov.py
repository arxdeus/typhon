# Ported from mne 1.13.2 (BSD-3-Clause)
#
# Covariance and rank helpers used by the decoding module: mne.rank
# (_compute_rank / estimate_rank), mne.cov (_regularized_covariance,
# _compute_covariance_auto, _smart_eigh) and the sklearn covariance
# estimators those paths rely on (EmpiricalCovariance, LedoitWolf, OAS,
# shrunk_covariance), restricted to the cases reachable from CSP.

import numpy as np
from ._info import Info, _picks_by_type, _scaling_cov_rank

_EPS = 2.220446049250313e-16


def _svdvals(a):
    return np.linalg.svd(a, compute_uv=False)


def _compute_rank_data(data, info: Info, kind: str = "estimated",
                       given: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """mne.cov._compute_rank_raw_array on (n_channels, n_samples) data.

    kind: 'estimated' (rank=None), 'info' or 'full'; given: rank dict entries.
    """
    if kind != "estimated" and kind != "info" and kind != "full":
        raise ValueError(f"rank, if str, must be 'info' or 'full', got {kind}")
    out = Dict[str, int]()
    if given is not None:
        for k, v in given.__val__().items():
            out[k] = v
    for ch_type, picks in _picks_by_type(info, meg_combined=True):
        if ch_type in out:
            if out[ch_type] > len(picks):
                raise ValueError(f"rank[{repr(ch_type)}]={out[ch_type]} exceeds the number of channels ({len(picks)})")
            continue
        this_rank = 0
        if kind == "info" or kind == "full":
            this_rank = len(picks)
        else:
            sub = data[np.array(picks), :].copy()
            # _scaled_array with scalings_cov_rank (meg combined uses per-type)
            for ii in range(len(picks)):
                sc = _scaling_cov_rank(info.ch_types[picks[ii]])
                sub[ii, :] *= sc
            s = _svdvals(sub)
            max_s = float(np.amax(s))
            tol = s.shape[0] * max_s * _EPS
            this_rank = int(np.sum(s > tol))
        out[ch_type] = this_rank
    return out


def _rank_for(rank: Dict[str, int], ch_type: str) -> int:
    if ch_type not in rank and (ch_type == "mag" or ch_type == "grad"):
        return rank["meg"]
    return rank[ch_type]


def _smart_eigh_subspace(C, info: Info, rank: Dict[str, int]):
    """_smart_eigh(..., proj_subspace=True, do_compute_rank=False) returning (eigvec, mask)."""
    n_chan = C.shape[0]
    total = 0
    for v in rank.values():
        total += v
    if total == n_chan:
        return np.eye(n_chan), np.ones(n_chan, dtype=bool)
    eigvec = np.zeros((n_chan, n_chan))
    mask = np.zeros(n_chan, dtype=bool)
    for ch_type, picks in _picks_by_type(info, meg_combined=True, exclude_bads=False):
        if len(picks) == 0:
            continue
        pa = np.array(picks)
        this_C = C[np.ix_(pa, pa)]
        this_rank = _rank_for(rank, ch_type)
        # _get_ch_whitener (pca=False)
        e, ev = np.linalg.eigh(this_C)
        ev = ev.T.copy()
        n = e.shape[0]
        m = np.ones(n, dtype=bool)
        for i in range(n - this_rank):
            m[i] = False
        # _eigvec_subspace
        bad = ev[~m]
        P = np.eye(n) - np.dot(bad.T, bad)
        _, ev2 = np.linalg.eigh(P)
        ev2 = ev2.T
        for i in range(n):
            mask[picks[i]] = m[i]
            for j in range(n):
                eigvec[picks[i], picks[j]] = ev2[i, j]
    return eigvec, mask


def _empirical_cov(X):
    # sklearn.covariance.empirical_covariance(X, assume_centered=True)
    return np.dot(X.T, X) / X.shape[0]


def _shrunk_covariance(emp_cov, shrinkage: float):
    n_features = emp_cov.shape[-1]
    shrunk_cov = (1.0 - shrinkage) * emp_cov
    mu = np.trace(emp_cov) / n_features
    shrunk_cov += shrinkage * mu * np.eye(n_features)
    return shrunk_cov


def _ledoit_wolf_shrinkage(X) -> float:
    # sklearn ledoit_wolf_shrinkage(assume_centered=True, block_size=1000)
    n_samples = X.shape[0]
    n_features = X.shape[1]
    if n_features == 1:
        return 0.0
    block_size = 1000
    n_splits = int(n_features / block_size)
    X2 = X ** 2
    emp_cov_trace = np.sum(X2, axis=0) / n_samples
    mu = np.sum(emp_cov_trace) / n_features
    beta_ = 0.0
    delta_ = 0.0
    bs = block_size
    for i in range(n_splits):
        for j in range(n_splits):
            beta_ += np.sum(np.dot(X2.T[bs * i:bs * (i + 1)], X2[:, bs * j:bs * (j + 1)]))
            delta_ += np.sum(np.dot(X.T[bs * i:bs * (i + 1)], X[:, bs * j:bs * (j + 1)]) ** 2)
        beta_ += np.sum(np.dot(X2.T[bs * i:bs * (i + 1)], X2[:, bs * n_splits:]))
        delta_ += np.sum(np.dot(X.T[bs * i:bs * (i + 1)], X[:, bs * n_splits:]) ** 2)
    for j in range(n_splits):
        beta_ += np.sum(np.dot(X2.T[bs * n_splits:], X2[:, bs * j:bs * (j + 1)]))
        delta_ += np.sum(np.dot(X.T[bs * n_splits:], X[:, bs * j:bs * (j + 1)]) ** 2)
    delta_ += np.sum(np.dot(X.T[bs * n_splits:], X[:, bs * n_splits:]) ** 2)
    delta_ /= n_samples ** 2
    beta_ += np.sum(np.dot(X2.T[bs * n_splits:], X2[:, bs * n_splits:]))
    beta = 1.0 / (n_features * n_samples) * (beta_ / n_samples - delta_)
    delta = delta_ - 2.0 * mu * emp_cov_trace.sum() + n_features * mu ** 2
    delta /= n_features
    beta = min(beta, delta)
    return 0.0 if beta == 0 else beta / delta


def _oas_shrinkage(X) -> float:
    n_samples = X.shape[0]
    n_features = X.shape[1]
    if n_features == 1:
        return 0.0
    emp_cov = _empirical_cov(X)
    alpha = np.mean(emp_cov ** 2)
    mu = np.trace(emp_cov) / n_features
    mu_squared = mu ** 2
    num = alpha + mu_squared
    den = (n_samples + 1) * (alpha - mu_squared / n_features)
    return 1.0 if den == 0 else min(num / den, 1.0)


def _shrunk_blocks(data, shrinkages: List[Tuple[str, float, List[int]]]):
    # mne.cov._ShrunkCovariance.fit with assume_centered=True
    cov = _empirical_cov(data)
    n = cov.shape[0]
    zero_cross = np.zeros((n, n), dtype=bool)
    for a in range(len(shrinkages)):
        for b in range(a + 1, len(shrinkages)):
            if shrinkages[a][0] == "eeg" or shrinkages[b][0] == "eeg":
                for i in shrinkages[a][2]:
                    for j in shrinkages[b][2]:
                        zero_cross[i, j] = True
                        zero_cross[j, i] = True
    for _, c, picks in shrinkages:
        pa = np.array(picks)
        sub = cov[np.ix_(pa, pa)]
        sh = _shrunk_covariance(sub, c)
        for i in range(len(picks)):
            for j in range(len(picks)):
                cov[picks[i], picks[j]] = sh[i, j]
    for a in range(len(shrinkages)):
        for b in range(a + 1, len(shrinkages)):
            c_ij = np.sqrt((1.0 - shrinkages[a][1]) * (1.0 - shrinkages[b][1]))
            for i in shrinkages[a][2]:
                for j in shrinkages[b][2]:
                    cov[i, j] *= c_ij
                    cov[j, i] *= c_ij
    for i in range(n):
        for j in range(n):
            if zero_cross[i, j]:
                cov[i, j] = 0.0
    return cov


def _searchsorted_list(used: List[int], picks: List[int]) -> List[int]:
    out = List[int]()
    for p in picks:
        k = 0
        while k < len(used) and used[k] < p:
            k += 1
        out.append(k)
    return out


def _regularized_covariance(data, reg_kind: str, shrink: float, info: Info,
                            rank: Dict[str, int]):
    """mne.cov._regularized_covariance for data (n_channels, n_samples).

    reg_kind: 'empirical' | 'shrinkage' (float reg) | 'ledoit_wolf' | 'oas'.
    """
    n_ch = data.shape[0]
    picks_list = _picks_by_type(info)
    rank2 = _compute_rank_data(data, info, "estimated", rank)
    # _scaled_array on data.T (n_samples, n_channels) -> scale channel rows
    scales = np.ones(n_ch)
    for ct, picks in picks_list:
        for p in picks:
            scales[p] = _scaling_cov_rank(ct)
    d = data.T.copy()
    for p in range(n_ch):
        d[:, p] *= scales[p]
    C = np.dot(d.T, d)
    eigvec, mask = _smart_eigh_subspace(C, info, rank2)
    eigvec = eigvec[mask]
    d = np.dot(d, eigvec.T)
    used = List[int]()
    for i in range(mask.shape[0]):
        if mask[i]:
            used.append(i)
    cov = np.zeros((1, 1))
    if reg_kind == "empirical":
        cov = _empirical_cov(d)
    elif reg_kind == "shrinkage":
        cov = _shrunk_blocks(d, [("all", shrink, list(range(d.shape[1])))])
    elif reg_kind == "ledoit_wolf" or reg_kind == "oas":
        shr = List[Tuple[str, float, List[int]]]()
        for ct, picks in picks_list:
            sp = _searchsorted_list(used, picks)
            sub = d[:, np.array(sp)].copy()
            s = _ledoit_wolf_shrinkage(sub) if reg_kind == "ledoit_wolf" else _oas_shrinkage(sub)
            shr.append((ct, s, sp))
        cov = _shrunk_blocks(d, shr)
    else:
        raise ValueError(f"Invalid reg ({reg_kind}). Supported in Typhon mne: None, float, 'empirical', 'ledoit_wolf', 'oas'")
    cov = np.dot(eigvec.T, np.dot(cov, eigvec))
    ns = d.shape[0]
    cov *= ns / max(ns - 1, 1)
    # _undo_scaling_cov
    inv = 1.0 / scales
    cov *= inv[None, :] * inv[:, None]
    return cov


def _restr_mat(C_ref, info: Info, rank: Dict[str, int]):
    eigvec, mask = _smart_eigh_subspace(C_ref, info, rank)
    return eigvec[mask]


def pinv(a, rtol: Optional[float] = None):
    """mne.utils.pinv."""
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    maxS = float(np.max(s))
    rt = float(max(max(vh.shape[0], vh.shape[1]), max(u.shape[0], u.shape[1]))) * _EPS
    if rtol is not None:
        rt = rtol
    rk = int(np.sum(s > maxS * rt))
    uu = u[:, :rk].copy()
    uu /= s[:rk]
    return (uu @ vh[:rk]).T.copy()
