# Ported from mne 1.13.2 (BSD-3-Clause)
#
# linear_regression_raw (rERP: regression-based evoked responses, solver="cholesky")
# and summarize_clusters_stc.

import math
import numpy as np
from .._info import Info, _picks_to_idx
from .._containers import Evoked
from ..epochs import _is_good, _reject_dict


def _event_ids(events: np.ndarray[int, 2], event_id) -> Tuple[List[str], List[List[int]]]:
    names = List[str]()
    ids = List[List[int]]()
    if isinstance(event_id, NoneType):
        u = List[int]()
        for i in range(events.shape[0]):
            v = int(events[i, 2])
            if v not in u:
                u.append(v)
        for v in u:
            names.append(str(v))
            ids.append([v])
    else:
        for k in event_id:
            names.append(str(k))
            val = event_id[k]
            if isinstance(val, int):
                ids.append([int(val)])
            else:
                lst = List[int]()
                for q in val:
                    lst.append(int(q))
                ids.append(lst)
    return names, ids


def _win(spec, name: str, default: float) -> float:
    if isinstance(spec, float) or isinstance(spec, int):
        return float(spec)
    else:
        if name in spec:
            return float(spec[name])
        return default


def linear_regression_raw(raw, events, event_id = None, tmin = -0.1, tmax = 1, covariates = None,
                          reject = None, flat = None, tstep: float = 1.0, decim: int = 1,
                          picks = None, solver = "cholesky"):
    """Estimate regression-based evoked responses (rERPs) by linear modeling (mne 1.13.2).

    Returns a dict mapping condition name to Evoked."""
    if not isinstance(solver, str):
        raise NotImplementedError("only solver='cholesky' is supported")
    if solver != "cholesky":
        raise ValueError(f"No such solver: {solver}")
    pk = _picks_to_idx(raw.info, picks, "all", "bads")
    info = raw.info._pick(pk)
    dec = int(decim)
    info.sfreq = info.sfreq / dec
    nt_all = raw._data.shape[1]
    nsm = (nt_all + dec - 1) // dec
    ev0 = np.asarray(events)
    for i in range(ev0.shape[0]):
        for j in range(i + 1, ev0.shape[0]):
            if ev0[i, 0] == ev0[j, 0]:
                raise ValueError("`events` contains duplicate time points. Make sure all entries in the first column of `events` are unique.")
    ev = np.empty((ev0.shape[0], 3), dtype = int)
    for i in range(ev0.shape[0]):
        ev[i, 0] = (int(ev0[i, 0]) - raw.first_samp) // dec
        ev[i, 1] = int(ev0[i, 1])
        ev[i, 2] = int(ev0[i, 2])
    for i in range(ev.shape[0]):
        for j in range(i + 1, ev.shape[0]):
            if ev[i, 0] == ev[j, 0]:
                raise ValueError("After decimating, `events` contains duplicate time points. This means some events are too closely spaced for the requested decimation factor. Choose different events, drop close events, or choose a different decimation factor.")
    data = np.empty((len(pk), nsm))
    for c in range(len(pk)):
        for t in range(nsm):
            data[c, t] = raw._data[pk[c], t * dec]
    names, ids = _event_ids(ev, event_id)
    cov_names = List[str]()
    cov_vals = List[np.ndarray[float, 1]]()
    if not isinstance(covariates, NoneType):
        for k in covariates:
            cov_names.append(str(k))
            cov_vals.append(np.asarray(covariates[k], dtype=float))
    sf = info.sfreq
    conds = names + cov_names
    nc = len(conds)
    tmin_s = [0 for _ in range(nc)]
    tmax_s = [0 for _ in range(nc)]
    for q in range(nc):
        tmin_s[q] = int(round(_win(tmin, conds[q], -0.1) * sf))
        tmax_s[q] = int(round(_win(tmax, conds[q], 1.0) * sf)) + 1
    ncol = 0
    for q in range(nc):
        ncol += tmax_s[q] - tmin_s[q]
    X = np.zeros((nsm, ncol))
    cond_length = [0 for _ in range(nc)]
    col0 = 0
    for q in range(nc):
        t0 = tmin_s[q]
        n_lags = tmax_s[q] - t0
        n_ev = 0
        for i in range(ev.shape[0]):
            val = 0.0
            if q < len(names):
                if int(ev[i, 2]) in ids[q]:
                    val = 1.0
            else:
                cv = cov_vals[q - len(names)]
                if cv.shape[0] != ev.shape[0]:
                    raise ValueError(f"Condition {conds[q]} from ``covariates`` is not the same length as ``events``")
                val = cv[i]
            if val == 0.0:
                continue
            n_ev += 1
            for j in range(n_lags):
                r = int(ev[i, 0]) + t0 + j
                if r >= 0 and r < nsm:
                    X[r, col0 + j] += val
        cond_length[q] = n_ev
        col0 += n_lags
    # rows with any nonzero predictor entry
    has = [False for _ in range(nsm)]
    for r in range(nsm):
        for c in range(ncol):
            if X[r, c] != 0.0:
                has[r] = True
                break
    rj = _reject_dict(reject)
    fl = _reject_dict(flat)
    if rj is not None or fl is not None:
        step = int(math.ceil(tstep * info.sfreq))
        step = int(math.ceil(float(step) / float(dec)))
        first = 0
        while first < nsm:
            last = first + step
            if last > nsm:
                break
            seg = np.ascontiguousarray(data[:, first:last])
            bad = _is_good(seg, info, rj, fl, 0, step)
            if len(bad) > 0:
                for r in range(first, last):
                    has[r] = False
            first += step
    rows = [r for r in range(nsm) if has[r]]
    nr = len(rows)
    Xs = np.empty((nr, ncol))
    Ys = np.empty((nr, len(pk)))
    for a in range(nr):
        Xs[a, :] = X[rows[a], :]
        for c in range(len(pk)):
            Ys[a, c] = data[c, rows[a]]
    G = np.dot(Xs.T, Xs)
    B = np.dot(Xs.T, Ys)
    coefs = np.linalg.solve(G, B).T
    out = Dict[str, Evoked]()
    cum = 0
    for q in range(nc):
        n_lags = tmax_s[q] - tmin_s[q]
        cd = np.empty((len(pk), n_lags))
        for c in range(len(pk)):
            for j in range(n_lags):
                cd[c, j] = coefs[c, cum + j]
        tm = np.empty(n_lags)
        for j in range(n_lags):
            tm[j] = float(tmin_s[q] + j) / sf
        ev_o = Evoked(cd, info.copy(), tm, cond_length[q], conds[q])
        out[conds[q]] = ev_o
        cum += n_lags
    return out


def summarize_clusters_stc(clu, p_thresh: float = 0.05, tstep: float = 1.0, tmin = 0,
                           subject: str = "fsaverage", vertices = None):
    """Assemble a summary SourceEstimate from spatiotemporal cluster results (mne 1.13.2).

    clu is (t_obs, clusters, cluster_pv, H0) with clusters as (t_inds, v_inds) tuples."""
    from ..source_estimate import SourceEstimate
    t_obs = clu[0]
    clusters = clu[1]
    clu_pvals = clu[2]
    n_times = t_obs.shape[0]
    n_vertices = t_obs.shape[1]
    verts = List[np.ndarray[int, 1]]()
    if isinstance(vertices, NoneType):
        verts.append(np.arange(10242))
        verts.append(np.arange(10242))
    else:
        for v in vertices:
            verts.append(np.asarray(v).astype(int))
    need = 0
    for v in verts:
        need += v.shape[0]
    if n_vertices != need:
        raise ValueError(f"Number of cluster vertices ({n_vertices}) did not match the provided vertices ({need})")
    good = List[int]()
    for i in range(len(clu_pvals)):
        if clu_pvals[i] < p_thresh:
            good.append(i)
    if len(good) == 0:
        raise RuntimeError("No significant clusters available. Please adjust your threshold or check your statistical analysis.")
    summ = np.zeros((n_vertices, len(good) + 1))
    for ii in range(len(good)):
        cl = clusters[good[ii]]
        t_inds = cl[0]
        v_inds = cl[1]
        dat = np.zeros((n_vertices, n_times))
        for k in range(len(t_inds)):
            tv = t_obs[int(t_inds[k]), int(v_inds[k])]
            dat[int(v_inds[k]), int(t_inds[k])] = tv
        for v in range(n_vertices):
            s = 0.0
            for t in range(n_times):
                d = dat[v, t]
                sg = 0.0
                if d > 0.0:
                    sg = 1.0
                elif d < 0.0:
                    sg = -1.0
                s += sg * tstep
            summ[v, ii + 1] = s
    for v in range(n_vertices):
        s = 0.0
        for c in range(len(good) + 1):
            s += summ[v, c]
        summ[v, 0] = s
    return SourceEstimate(summ, verts, float(tmin), tstep, subject, "surface")
