# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne/stats/cluster_level.py and mne/stats/_adjacency.py.
#
# Cluster formation is implemented on an explicit neighbor graph (CSR arrays
# over the flattened test space).  Clusters are labelled in order of their
# smallest flat index, which is the ordering produced by both
# scipy.ndimage.label (adjacency=None) and scipy.sparse.csgraph
# .connected_components (sparse adjacency) in the reference implementation,
# so cluster lists, sums, p-values and H0 match mne exactly.

import numpy as np
from scipy import stats as _sst
from .parametric import ttest_1samp_no_p, _f_oneway_list
from .permutations import _check_rng


class _Graph:
    indptr: np.ndarray[int, 1]
    indices: np.ndarray[int, 1]
    isolated: bool

    def __init__(self, indptr: np.ndarray[int, 1], indices: np.ndarray[int, 1], isolated: bool):
        self.indptr = indptr
        self.indices = indices
        self.isolated = isolated


def _graph_from_lists(nb: List[List[int]]) -> _Graph:
    n = len(nb)
    indptr = np.zeros(n + 1, dtype=int)
    tot = 0
    for i in range(n):
        tot += len(nb[i])
        indptr[i + 1] = tot
    indices = np.empty(tot, dtype=int)
    k = 0
    for i in range(n):
        for j in nb[i]:
            indices[k] = j
            k += 1
    return _Graph(indptr, indices, False)


def _grid_graph(shape: List[int]) -> _Graph:
    """Face-connected lattice (scipy.ndimage.label default structure)."""
    nd = len(shape)
    n = 1
    for s in shape:
        n *= s
    strides = [1] * nd
    for d in range(nd - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]
    nb = [List[int]() for _ in range(n)]
    for i in range(n):
        for d in range(nd):
            c = (i // strides[d]) % shape[d]
            if c > 0:
                nb[i].append(i - strides[d])
            if c < shape[d] - 1:
                nb[i].append(i + strides[d])
    return _graph_from_lists(nb)


def _edge_arrays(adjacency) -> Tuple[np.ndarray[int, 1], np.ndarray[int, 1], int]:
    coo = adjacency.tocoo()
    row = np.asarray(coo.row).astype(int)
    col = np.asarray(coo.col).astype(int)
    return row, col, int(adjacency.shape[0])


def _adjacency_graph(adjacency, n_tests: int, n_times: int, max_step: int) -> _Graph:
    row, col, n_src = _edge_arrays(adjacency)
    if n_src == n_tests:
        nb = [List[int]() for _ in range(n_tests)]
        for k in range(len(row)):
            r = row[k]
            c = col[k]
            if r != c:
                nb[r].append(c)
                nb[c].append(r)
        return _graph_from_lists(nb)
    got_times = n_tests // n_src
    if got_times != n_times or n_tests % n_src != 0:
        raise ValueError(f"adjacency (len {n_src}) must be of the correct size, i.e. be equal to or evenly divide the number of tests ({n_tests}).")
    sp = [List[int]() for _ in range(n_src)]
    for k in range(len(row)):
        r = row[k]
        c = col[k]
        if r != c:
            sp[r].append(c)
            sp[c].append(r)
    nb = [List[int]() for _ in range(n_tests)]
    for t in range(n_times):
        for s in range(n_src):
            i = t * n_src + s
            for j in sp[s]:
                nb[i].append(t * n_src + j)
            for step in range(1, max_step + 1):
                if t >= step:
                    nb[i].append(i - step * n_src)
                if t + step < n_times:
                    nb[i].append(i + step * n_src)
    return _graph_from_lists(nb)


def _components(x_in: np.ndarray[bool, 1], g: _Graph) -> List[np.ndarray[int, 1]]:
    n = x_in.shape[0]
    out = List[np.ndarray[int, 1]]()
    if g.isolated:
        for i in range(n):
            if x_in[i]:
                a = np.empty(1, dtype=int)
                a[0] = i
                out.append(a)
        return out
    seen = np.zeros(n, dtype=bool)
    stack = List[int]()
    for i in range(n):
        if x_in[i] and not seen[i]:
            members = List[int]()
            seen[i] = True
            stack.append(i)
            while len(stack) > 0:
                v = stack.pop()
                members.append(v)
                for k in range(g.indptr[v], g.indptr[v + 1]):
                    w = g.indices[k]
                    if x_in[w] and not seen[w]:
                        seen[w] = True
                        stack.append(w)
            members.sort()
            out.append(np.array(members, dtype=int))
    return out


def _cluster_sum(x: np.ndarray[float, 1], c: np.ndarray[int, 1], t_power: float) -> float:
    s = 0.0
    if t_power == 1:
        for i in c:
            s += x[i]
    else:
        for i in c:
            v = x[i]
            sg = 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)
            s += sg * abs(v) ** t_power
    return s


class _Thresh:
    tfce: bool
    thresholds: List[float]
    h_power: float
    e_power: float
    start: float
    step: float

    def __init__(self):
        self.tfce = False
        self.thresholds = List[float]()
        self.h_power = 2.0
        self.e_power = 0.5
        self.start = 0.0
        self.step = 0.0


def _make_thresh(threshold, x: np.ndarray[float, 1], tail: int) -> _Thresh:
    th = _Thresh()
    if isinstance(threshold, Dict):
        if "start" not in threshold or "step" not in threshold:
            raise KeyError('threshold, if dict, must have at least "start" and "step"')
        th.tfce = True
        start = float(threshold["start"])
        step = float(threshold["step"])
        th.start = start
        th.step = step
        mx = -np.inf
        mn = np.inf
        nf = 0
        for v in x:
            if np.isfinite(v):
                nf += 1
                mx = max(mx, v)
                mn = min(mn, v)
        if nf == 0:
            raise RuntimeError("No finite values found in the observed statistic values")
        if tail == -1:
            if start > 0:
                raise ValueError('threshold["start"] must be <= 0 for tail == -1')
            if step >= 0:
                raise ValueError('threshold["step"] must be < 0 for tail == -1')
            stop = mn
        elif tail == 1:
            stop = mx
        else:
            stop = max(mx, -mn)
        ths = np.arange(start, stop, step, dtype=float)
        th.thresholds = [float(v) for v in ths]
        th.h_power = float(threshold.get("h_power", 2.0))
        th.e_power = float(threshold.get("e_power", 0.5))
    else:
        th.tfce = False
        th.start = 0.0
        th.step = 0.0
        th.thresholds = [float(threshold)]
        th.h_power = 2.0
        th.e_power = 0.5
    return th


def _find_clusters_flat(x: np.ndarray[float, 1], th: _Thresh, tail: int, g: _Graph,
                        include: Optional[np.ndarray[bool, 1]],
                        partitions: Optional[np.ndarray[int, 1]], t_power: float
                        ) -> Tuple[List[np.ndarray[int, 1]], np.ndarray[float, 1]]:
    n = x.shape[0]
    clusters = List[np.ndarray[int, 1]]()
    sums = List[float]()
    scores = np.zeros(n if th.tfce else 0)
    nth = len(th.thresholds)
    for ti in range(nth):
        thresh = th.thresholds[ti]
        clusters = List[np.ndarray[int, 1]]()
        n_dir = 2 if tail == 0 else 1
        for d in range(n_dir):
            x_in = np.empty(n, dtype=bool)
            for i in range(n):
                if tail == 0:
                    v = (x[i] > thresh) if d == 0 else (x[i] < -thresh)
                elif tail == -1:
                    v = x[i] < thresh
                else:
                    v = x[i] > thresh
                if include is not None:
                    v = v and include[i]
                x_in[i] = v
            if not np.any(x_in):
                continue
            # check_disjoint partitions do not change the result (components
            # never span disjoint partitions), so they are not materialized.
            comps = _components(x_in, g)
            for c in comps:
                clusters.append(c)
                if not th.tfce:
                    sums.append(_cluster_sum(x, c, t_power))
        if th.tfce:
            dh = abs(thresh) if ti == 0 else abs(thresh - th.thresholds[ti - 1])
            h = abs(thresh) ** th.h_power * dh
            for c in clusters:
                inc = h * float(len(c)) ** th.e_power
                for i in c:
                    scores[i] += inc
    if th.tfce:
        return List[np.ndarray[int, 1]](), scores
    return clusters, np.array(sums, dtype=float)


def _bin_perm_rep(ndim: int) -> np.ndarray[int, 2]:
    nperms = 2 ** ndim
    perms = np.zeros((nperms, ndim), dtype=int)
    half_point = nperms // 2
    perms[half_point:, 0] = 1
    for j in range(1, ndim):
        half_col = perms[::2, j - 1].copy()
        perms[:half_point, j] = half_col
        perms[half_point:, j] = half_col
    return perms


def _get_1samp_orders(n_samples: int, n_permutations, tail: int, rng):
    max_perms = 2 ** (n_samples - int(tail == 0)) - 1
    extra = ""
    if isinstance(n_permutations, str):
        if n_permutations != "all":
            raise ValueError('n_permutations as a string must be "all"')
        n_perm = max_perms
    else:
        n_perm = int(n_permutations)
    if max_perms < n_perm:
        extra = " (exact test)"
        orders = _bin_perm_rep(n_samples)[1:max_perms + 1].copy()
    elif n_samples <= 20:
        sel = rng.choice(max_perms, n_perm - 1, replace=False)
        orders = np.zeros((n_perm - 1, n_samples), dtype=int)
        for k in range(n_perm - 1):
            v = int(sel[k]) + 1
            for j in range(n_samples):
                orders[k, n_samples - 1 - j] = v & 1
                v >>= 1
    else:
        orders = np.zeros((n_perm - 1, n_samples), dtype=int)
        hashes = Set[str]()
        ii = 0
        use_samples = n_samples - int(tail == 0)
        while ii < n_perm - 1:
            u = rng.uniform(size=use_samples)
            key = "".join(["1" if u[j] < 0.5 else "0" for j in range(use_samples)])
            if key not in hashes:
                for j in range(use_samples):
                    orders[ii, j] = 1 if u[j] < 0.5 else 0
                if tail == 0 and rng.uniform() < 0.5:
                    for j in range(n_samples):
                        orders[ii, j] = 1 - orders[ii, j]
                hashes.add(key)
                ii += 1
    return orders, n_perm, extra


def _pval_from_histogram(T: np.ndarray[float, 1], H0: np.ndarray[float, 1], tail: int) -> np.ndarray[float, 1]:
    out = np.empty(T.shape[0])
    nh = float(H0.shape[0])
    for k in range(T.shape[0]):
        t = T[k]
        c = 0
        for h in H0:
            if tail == -1:
                c += int(h <= t)
            elif tail == 1:
                c += int(h >= t)
            else:
                c += int(abs(h) >= abs(t))
        out[k] = c / nh
    return out


def _ttest_from_signs(signs: np.ndarray[float, 1], X: np.ndarray[float, 2],
                      sum_sq: np.ndarray[float, 1], sqrt_n_nm1: float) -> np.ndarray[float, 1]:
    n = float(X.shape[0])
    col_sum = signs @ X
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        mean = col_sum[j] / n
        dsq = max(sum_sq[j] - n * mean * mean, 0.0)
        if dsq > 0:
            out[j] = mean / np.sqrt(dsq) * sqrt_n_nm1
    return out


def _flat2(x) -> np.ndarray[float, 2]:
    a = np.asarray(x, dtype=float)
    return np.ascontiguousarray(a.reshape((a.shape[0], -1)))


def _sample_shape(x):
    a = np.asarray(x)
    if static.len(a.shape) == 1:
        return (1,)
    else:
        return a.shape[1:]


def _as_tuple(Xs, lst: List[np.ndarray[float, 2]]):
    return tuple(lst[i] for i in static.range(static.len(Xs)))


def _stat_1samp(stat_fun, X: np.ndarray[float, 2]) -> np.ndarray[float, 1]:
    if isinstance(stat_fun, NoneType):
        return ttest_1samp_no_p(X)
    else:
        return np.asarray(stat_fun(X), dtype=float).ravel()


def _stat_between(stat_fun, Xs, lst: List[np.ndarray[float, 2]]) -> np.ndarray[float, 1]:
    if isinstance(stat_fun, NoneType):
        return _f_oneway_list(lst)
    elif isinstance(Xs, Tuple):
        return np.asarray(stat_fun(*_as_tuple(Xs, lst)), dtype=float).ravel()
    else:
        # list input with a custom stat_fun: the list of groups is passed as-is
        return np.asarray(stat_fun(lst), dtype=float).ravel()


def _max_sum(sums: np.ndarray[float, 1], one_sample: bool) -> float:
    if len(sums) == 0:
        return 0.0
    if one_sample:
        return float(sums[int(np.argmax(np.abs(sums)))])
    return float(np.max(sums))


def _check_args(threshold, tail: int, out_type: str):
    if out_type != "mask" and out_type != "indices":
        raise ValueError(f"Invalid value for the 'out_type' parameter. Allowed values are 'mask' and 'indices', but got '{out_type}' instead.")
    if tail != -1 and tail != 0 and tail != 1:
        raise ValueError(f"Invalid value for the 'tail' parameter. Allowed values are -1, 0, and 1, but got {tail} instead.")
    if not isinstance(threshold, Dict):
        thr = float(threshold)
        if (tail < 0 and thr > 0) or (tail > 0 and thr < 0) or (tail == 0 and thr < 0):
            raise ValueError(f"incompatible tail and threshold signs, got {tail} and {thr}")


def _make_graph(adjacency, sample_shape: List[int], n_tests: int, max_step: int) -> _Graph:
    if isinstance(adjacency, NoneType):
        return _grid_graph(sample_shape)
    elif isinstance(adjacency, bool):
        if adjacency:
            raise ValueError("adjacency must be None, False or a sparse array")
        return _Graph(np.zeros(1, dtype=int), np.zeros(0, dtype=int), True)
    else:
        return _adjacency_graph(adjacency, n_tests, sample_shape[0], max_step)


def _make_include(exclude, n_tests: int) -> Optional[np.ndarray[bool, 1]]:
    if isinstance(exclude, NoneType):
        return None
    if isinstance(exclude, Optional):
        if exclude is None:
            return None
        return _make_include(exclude.__val__(), n_tests)
    ex = np.asarray(exclude).ravel()
    if ex.size != n_tests:
        raise ValueError("exclude must be the same shape as X[0]")
    return np.logical_not(ex.astype(bool))


class _ClusterResult:
    t_obs: np.ndarray[float, 1]
    clusters: List[np.ndarray[int, 1]]
    cluster_pv: np.ndarray[float, 1]
    H0: np.ndarray[float, 1]

    def __init__(self):
        self.t_obs = np.zeros(0)
        self.clusters = List[np.ndarray[int, 1]]()
        self.cluster_pv = np.zeros(0)
        self.H0 = np.zeros(0)


def _run_permutations(Xs, Xlist: List[np.ndarray[float, 2]], one_sample: Literal[bool], th: _Thresh,
                      n_permutations, tail: int, stat_fun, g: _Graph,
                      include: Optional[np.ndarray[bool, 1]], rng, step_down_p: float,
                      t_power: float) -> _ClusterResult:
    n_samples = Xlist[0].shape[0]
    n_tests = Xlist[0].shape[1]
    if one_sample:
        t_obs = _stat_1samp(stat_fun, Xlist[0])
    else:
        t_obs = _stat_between(stat_fun, Xs, Xlist)
    if t_obs.size != n_tests:
        raise ValueError("t_obs provided by stat_fun is not compatible with the sample shape")
    th2 = th
    clusters, cluster_stats = _find_clusters_flat(t_obs, th2, tail, g, include, None, t_power)
    res = _ClusterResult()
    if th2.tfce:
        res.t_obs = cluster_stats * np.sign(t_obs)
        clusters = [np.array([c], dtype=int) for c in range(n_tests)]
    else:
        res.t_obs = t_obs
    res.clusters = clusters
    res.cluster_pv = np.zeros(0)
    res.H0 = np.zeros(0)
    # orders are drawn before the early exit, exactly as in mne
    orders = np.zeros((0, 0), dtype=int)
    perm_rows = List[np.ndarray[int, 1]]()
    splits = [0]
    if one_sample:
        orders, _, _ = _get_1samp_orders(n_samples, n_permutations, tail, rng)
        n_orders = orders.shape[0]
        X_full = Xlist[0]
    else:
        n_perm = int(n_permutations)
        X_full = np.concatenate(Xlist, axis=0)
        for x in Xlist:
            splits.append(splits[-1] + x.shape[0])
        for _ in range(n_perm - 1):
            perm_rows.append(rng.permutation(X_full.shape[0]))
        n_orders = len(perm_rows)
    if len(clusters) == 0:
        return res
    use_tt = one_sample and isinstance(stat_fun, NoneType)
    sum_sq = np.sum(X_full ** 2, axis=0)
    sqrt_n_nm1 = np.sqrt(float(n_samples * (n_samples - 1)))
    n_removed = 1
    total_removed = 0
    step_down_include: Optional[np.ndarray[bool, 1]] = None
    while n_removed > 0:
        this_include: Optional[np.ndarray[bool, 1]] = include
        if step_down_include is not None:
            sdi: np.ndarray[bool, 1] = step_down_include
            if include is not None:
                inc: np.ndarray[bool, 1] = include
                both: np.ndarray[bool, 1] = inc & sdi
                this_include = both
            else:
                this_include = sdi
        H0 = np.empty(n_orders + 1)
        for k in range(n_orders):
            if one_sample:
                if use_tt:
                    surr = _ttest_from_signs(2.0 * orders[k].astype(float) - 1.0, X_full, sum_sq, sqrt_n_nm1)
                else:
                    signs = (2 * orders[k] - 1).astype(float)
                    surr = _stat_1samp(stat_fun, X_full * signs[:, None])
            else:
                idx = perm_rows[k]
                parts = List[np.ndarray[float, 2]]()
                for gi in range(len(Xlist)):
                    parts.append(X_full[idx[splits[gi]:splits[gi + 1]], :])
                surr = _stat_between(stat_fun, Xs, parts)
            th_s = th2
            if th2.tfce:
                # mne recomputes the TFCE threshold range from each surrogate
                th_s = _make_thresh({"start": th2.start, "step": th2.step,
                                     "h_power": th2.h_power, "e_power": th2.e_power},
                                    surr, tail)
            _, ps = _find_clusters_flat(surr, th_s, tail, g, this_include, None, t_power)
            H0[k + 1] = _max_sum(ps, one_sample)
        if tail == -1:
            orig = float(cluster_stats.min())
        elif tail == 1:
            orig = float(cluster_stats.max())
        else:
            orig = float(np.abs(cluster_stats).max())
        H0[0] = orig
        cluster_pv = _pval_from_histogram(cluster_stats, H0, tail)
        n_rm = 0
        sd = np.ones(n_tests, dtype=bool)
        for ci in range(len(cluster_pv)):
            if cluster_pv[ci] < step_down_p:
                n_rm += 1
                for i in clusters[ci]:
                    sd[i] = False
        n_removed = n_rm - total_removed
        total_removed = n_rm
        step_down_include = sd
        res.H0 = H0
        res.cluster_pv = cluster_pv
    return res


def _format_clusters(clusters: List[np.ndarray[int, 1]], shp, out_type: str):
    """indices: tuple of index arrays per cluster (np.unravel_index order);
    mask: boolean array of the sample shape per cluster."""
    n = 1
    for s in shp:
        n *= s
    idx = [np.unravel_index(c, shp) for c in clusters]
    masks = List[np.ndarray[bool, 1]]()
    if out_type == "mask":
        for c in clusters:
            m = np.zeros(n, dtype=bool)
            for i in c:
                m[i] = True
            masks.append(m)
    return idx, [m.reshape(shp) for m in masks]


def _finish(res: _ClusterResult, shp, out_type: Literal[str]):
    t = res.t_obs.reshape(shp)
    idx, masks = _format_clusters(res.clusters, shp, out_type)
    if out_type == "mask":
        return t, masks, res.cluster_pv, res.H0
    else:
        return t, idx, res.cluster_pv, res.H0


def _check_fun_within(X, stat_fun, threshold, tail: int):
    if isinstance(threshold, NoneType):
        p_thresh = 0.05 / (1 + int(tail == 0))
        n_samples = len(X)
        thr = -float(_sst.t.ppf(p_thresh, n_samples - 1))
        if tail < 0:
            thr = -thr
        return thr
    else:
        return threshold


def _check_fun_between(Xs, stat_fun, threshold, tail: int):
    if isinstance(threshold, NoneType):
        p_thresh = 0.05
        n = 0
        k = 0
        for x in Xs:
            n += len(x)
            k += 1
        dfn = k - 1
        dfd = n - k
        return float(_sst.f.ppf(1.0 - p_thresh, dfn, dfd))
    else:
        return threshold


def _cluster_test_core(Xs, one_sample: Literal[bool], threshold, n_permutations, tail: int, stat_fun,
                       adjacency, rng, max_step: int, exclude, step_down_p: float,
                       t_power: float, out_type: str, shp) -> _ClusterResult:
    _check_args(threshold, tail, out_type)
    Xlist = List[np.ndarray[float, 2]]()
    if one_sample:
        Xlist.append(_flat2(Xs))
    elif isinstance(Xs, Tuple):
        for i in static.range(static.len(Xs)):
            Xlist.append(_flat2(Xs[i]))
    else:
        for x in Xs:
            Xlist.append(_flat2(x))
    n_tests = Xlist[0].shape[1]
    for x in Xlist:
        if x.shape[1] != n_tests:
            raise ValueError("All samples mush have the same size")
    sample_shape = [int(s) for s in shp]
    g = _make_graph(adjacency, sample_shape, n_tests, max_step)
    include = _make_include(exclude, n_tests)
    # threshold may depend on observed statistic (TFCE stop): compute lazily
    if one_sample:
        t0 = _stat_1samp(stat_fun, Xlist[0])
    else:
        t0 = _stat_between(stat_fun, Xs, Xlist)
    th = _make_thresh(threshold, t0, tail)
    return _run_permutations(Xs, Xlist, one_sample, th, n_permutations, tail, stat_fun, g,
                             include, rng, step_down_p, t_power)


def permutation_cluster_1samp_test(X, threshold = None, n_permutations = 1024, tail: int = 0,
                                   stat_fun = None, adjacency = None, n_jobs = None,
                                   max_step: int = 1, exclude = None, step_down_p: float = 0,
                                   t_power: float = 1, out_type: Literal[str] = "indices",
                                   check_disjoint: bool = False, buffer_size = 1000,
                                   verbose = None, *, rng = None, seed = None):
    """Non-parametric cluster-level paired t-test (mne).

    Returns (T_obs, clusters, cluster_p_values, H0)."""
    r = _check_rng(rng, seed)
    Xa = np.asarray(X, dtype=float)
    thr = _check_fun_within(Xa, stat_fun, threshold, tail)
    shp = _sample_shape(Xa)
    res = _cluster_test_core(Xa, True, thr, n_permutations, tail, stat_fun, adjacency, r,
                             max_step, exclude, float(step_down_p), float(t_power), out_type, shp)
    return _finish(res, shp, out_type)


def permutation_cluster_test(X, threshold = None, n_permutations = 1024, tail: int = 0,
                             stat_fun = None, adjacency = None, n_jobs = None,
                             max_step: int = 1, exclude = None, step_down_p: float = 0,
                             t_power: float = 1, out_type: Literal[str] = "indices",
                             check_disjoint: bool = False, buffer_size = 1000,
                             verbose = None, *, rng = None, seed = None):
    """Cluster-level statistical permutation test (F-test by default) (mne).

    X is a list or tuple of (n_observations, ...) arrays.
    Returns (T_obs, clusters, cluster_p_values, H0)."""
    r = _check_rng(rng, seed)
    thr = _check_fun_between(X, stat_fun, threshold, tail)
    shp = _sample_shape(X[0])
    res = _cluster_test_core(X, False, thr, n_permutations, tail, stat_fun, adjacency, r,
                             max_step, exclude, float(step_down_p), float(t_power), out_type, shp)
    return _finish(res, shp, out_type)


def _st_mask_from_s_inds(n_times: int, n_vertices: int, vertices, set_as: bool = True):
    mask = np.zeros((n_times, n_vertices), dtype=bool)
    for v in vertices:
        mask[:, int(v)] = True
    m = mask.ravel()
    if not set_as:
        m = np.logical_not(m)
    return m


def spatio_temporal_cluster_1samp_test(X, threshold = None, n_permutations = 1024, tail: int = 0,
                                       stat_fun = None, adjacency = None, n_jobs = None,
                                       max_step: int = 1, spatial_exclude = None,
                                       step_down_p: float = 0, t_power: float = 1,
                                       out_type: Literal[str] = "indices",
                                       check_disjoint: bool = False, buffer_size = 1000,
                                       verbose = None, *, rng = None, seed = None):
    """Non-parametric cluster-level paired t-test for spatio-temporal data (mne)."""
    Xa = np.asarray(X, dtype=float)
    exclude: Optional[np.ndarray[bool, 1]] = None
    if not isinstance(spatial_exclude, NoneType):
        nt = 1
        for k in range(1, Xa.ndim - 1):
            nt *= Xa.shape[k]
        exclude = _st_mask_from_s_inds(nt, Xa.shape[Xa.ndim - 1], spatial_exclude, True)
    return permutation_cluster_1samp_test(Xa, threshold=threshold, n_permutations=n_permutations,
                                          tail=tail, stat_fun=stat_fun, adjacency=adjacency,
                                          max_step=max_step, exclude=exclude,
                                          step_down_p=step_down_p, t_power=t_power,
                                          out_type=out_type, rng=rng, seed=seed)


def spatio_temporal_cluster_test(X, threshold = None, n_permutations = 1024, tail: int = 0,
                                 stat_fun = None, adjacency = None, n_jobs = None,
                                 max_step: int = 1, spatial_exclude = None,
                                 step_down_p: float = 0, t_power: float = 1,
                                 out_type: Literal[str] = "indices",
                                 check_disjoint: bool = False, buffer_size = 1000,
                                 verbose = None, *, rng = None, seed = None):
    """Non-parametric cluster-level test for spatio-temporal data (mne)."""
    exclude: Optional[np.ndarray[bool, 1]] = None
    if not isinstance(spatial_exclude, NoneType):
        X0 = X[0]
        nt = 1
        for k in range(1, X0.ndim - 1):
            nt *= X0.shape[k]
        exclude = _st_mask_from_s_inds(nt, X0.shape[X0.ndim - 1], spatial_exclude, True)
    return permutation_cluster_test(X, threshold=threshold, n_permutations=n_permutations,
                                    tail=tail, stat_fun=stat_fun, adjacency=adjacency,
                                    max_step=max_step, exclude=exclude,
                                    step_down_p=step_down_p, t_power=t_power,
                                    out_type=out_type, rng=rng, seed=seed)


def combine_adjacency(*structure):
    """Create a sparse binary adjacency/neighbors matrix (mne).

    Each entry is an int (lattice of that length) or a square array/sparse
    matrix of 0/1 neighbors.  Returns a scipy.sparse coo_array."""
    from scipy import sparse
    shape = List[int]()
    rows_l = List[List[int]]()
    cols_l = List[List[int]]()
    for di in static.range(static.len(structure)):
        dim = structure[di]
        rr = List[int]()
        cc = List[int]()
        if isinstance(dim, int):
            n = dim
            # dia_array((ones, [-1, 1])).tocoo(): row-major (row, col) order
            for i in range(n):
                if i > 0:
                    rr.append(i)
                    cc.append(i - 1)
                if i < n - 1:
                    rr.append(i)
                    cc.append(i + 1)
        else:
            if isinstance(dim, np.ndarray):
                dd = np.asarray(dim, dtype=float)
                n = dd.shape[0]
                if dd.shape[0] != dd.shape[1]:
                    raise ValueError(f"structure[{di}] must be square, got shape {dd.shape}")
                for i in range(n):
                    for j in range(n):
                        if i != j and dd[i, j] != 0:
                            if dd[i, j] != 1:
                                raise ValueError("All adjacency values must be 0 or 1")
                            rr.append(i)
                            cc.append(j)
            else:
                co = dim.tocoo()
                n = int(dim.shape[0])
                if dim.shape[0] != dim.shape[1]:
                    raise ValueError(f"structure[{di}] must be square, got shape {dim.shape}")
                for k in range(len(co.row)):
                    i = int(co.row[k])
                    j = int(co.col[k])
                    if i != j and co.data[k] != 0:
                        if co.data[k] != 1:
                            raise ValueError("All adjacency values must be 0 or 1")
                        rr.append(i)
                        cc.append(j)
        shape.append(n)
        rows_l.append(rr)
        cols_l.append(cc)
    nd = len(shape)
    n_diag = 1
    for s in shape:
        n_diag *= s
    strides = [1] * nd
    for d in range(nd - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]
    R = List[int]()
    C = List[int]()
    for di in range(nd):
        # vertices[s_l].ravel(): index di replaced by the edge list, others full,
        # C-order over the resulting array (edge axis at position di)
        other = [d for d in range(nd) if d != di]
        ne = len(rows_l[di])
        pre = 1
        for d in range(di):
            pre *= shape[d]
        post = 1
        for d in range(di + 1, nd):
            post *= shape[d]
        for a in range(pre):
            for e in range(ne):
                for b in range(post):
                    base = a * shape[di] * post + b
                    R.append(base + rows_l[di][e] * post)
                    C.append(base + cols_l[di][e] * post)
    for i in range(n_diag):
        R.append(i)
        C.append(i)
    data = np.ones(len(R))
    return sparse.coo_array((data, (np.array(R, dtype=int), np.array(C, dtype=int))),
                            shape=(n_diag, n_diag))
