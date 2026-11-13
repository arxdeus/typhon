# Ported from mne 1.13.2 (BSD-3-Clause), mne/preprocessing/_fine_cal.py
#
# compute_fine_calibration. Typhon Info limits (see _hfc.py / _maxwell_filter.py):
# no dev_head_t / cross-talk (cross_talk must be None), no `calibration`
# input (must be None), origin must be explicit (no "auto"). The returned
# calibration is the tuple (ch_names, locs (n, 12), imb_cals) that
# read_fine_calibration / write_fine_calibration use, plus the segment count.

import math
import numpy as np
from scipy.optimize import minimize
from .._info import Info
from ._maxwell import _MFCoils, _prep_mf_coils, _basis, _col_norm_pinv

_MAG_SCALE = 100.0


def _unit(v: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return v / n


def _find_vector_rotation(a: np.ndarray[float, 1], b: np.ndarray[float, 1]) -> np.ndarray[float, 2]:
    # Rodrigues rotation mapping unit vector a onto unit vector b (mne.transforms)
    v = np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])
    s = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    c = b[0] * a[0] + b[1] * a[1] + b[2] * a[2]
    vx = np.zeros((3, 3))
    vx[0, 1] = -v[2]
    vx[0, 2] = v[1]
    vx[1, 0] = v[2]
    vx[1, 2] = -v[0]
    vx[2, 0] = -v[1]
    vx[2, 1] = v[0]
    degenerate = s < 2.2250738585072014e-308
    factor = (1.0 - c) / (1.0 if degenerate else s)
    R = np.eye(3) + vx + np.dot(vx, vx) * factor
    if degenerate and c < 0:
        ai = 0
        for k in range(3):
            if abs(a[k]) < abs(a[ai]):
                ai = k
        e = np.zeros(3)
        e[ai] = 1.0
        kk = np.array([a[1] * e[2] - a[2] * e[1], a[2] * e[0] - a[0] * e[2], a[0] * e[1] - a[1] * e[0]])
        kk = _unit(kk)
        R = 2.0 * np.outer(kk, kk) - np.eye(3)
    return R


def _loc_to_trans(loc: np.ndarray[float, 1]) -> np.ndarray[float, 2]:
    t = np.eye(4)
    for i in range(3):
        t[i, 3] = loc[i]
    for c in range(3):
        for i in range(3):
            t[i, c] = loc[3 + 3 * c + i]
    return t


def _trans_to_loc(t: np.ndarray[float, 2]) -> np.ndarray[float, 1]:
    loc = np.zeros(12)
    for i in range(3):
        loc[i] = t[i, 3]
    for c in range(3):
        for i in range(3):
            loc[3 + 3 * c + i] = t[i, c]
    return loc


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-8 + 1e-5 * abs(b)


def _matched_loc_idx(mag_loc: np.ndarray[float, 1], locs: np.ndarray[float, 2]) -> List[int]:
    out = List[int]()
    for k in range(locs.shape[0]):
        ok = True
        for j in range(3):
            if not _close(mag_loc[9 + j], locs[k, 9 + j]) or not _close(mag_loc[j], locs[k, j]):
                ok = False
        if ok:
            out.append(k)
    return out


def _rotate_locs(locs: np.ndarray[float, 2], idxs: List[int], new_z: np.ndarray[float, 1]):
    nz = _unit(new_z)
    oz = _unit(np.array([locs[idxs[0], 9], locs[idxs[0], 10], locs[idxs[0], 11]]))
    rot = _find_vector_rotation(oz, nz)
    for ci in idxs:
        t = _loc_to_trans(locs[ci].copy())
        r = np.dot(rot, t[:3, :3])
        for i in range(3):
            for j in range(3):
                t[i, j] = r[i, j]
        locs[ci] = _trans_to_loc(t)


def _angle_deg(x: np.ndarray[float, 1], y: np.ndarray[float, 1]) -> float:
    nx = math.sqrt(float(np.dot(x, x)))
    ny = math.sqrt(float(np.dot(y, y)))
    c = float(np.dot(x, y)) / (nx * ny)
    c = min(1.0, max(-1.0, c))
    return abs(math.degrees(math.acos(c)))


class _Coil:
    # coils with per-channel row ranges (mne slice_map)
    c: _MFCoils
    start: List[int]
    stop: List[int]

    def __init__(self, c: _MFCoils):
        self.c = c
        self.start = [0 for _ in range(c.n_coils)]
        self.stop = [0 for _ in range(c.n_coils)]
        seen = [False for _ in range(c.n_coils)]
        for p in range(c.bins.shape[0]):
            b = int(c.bins[p])
            if not seen[b]:
                seen[b] = True
                self.start[b] = p
            self.stop[b] = p + 1


def _basis_scaled(c: _MFCoils, origin: np.ndarray[float, 1], ext_order: int,
                  scale: np.ndarray[float, 1]) -> np.ndarray[float, 2]:
    S = _basis(c, origin, 0, ext_order)
    for i in range(S.shape[0]):
        S[i, :] *= scale[i]
    return S


def _rotated_single(co: _Coil, idx: int, new_z: np.ndarray[float, 1], old_z: np.ndarray[float, 1],
                    inplace: bool) -> _MFCoils:
    oz = _unit(old_z)
    nz = _unit(new_z)
    rot = _find_vector_rotation(oz, nz)
    a = co.start[idx]
    b = co.stop[idx]
    rm = np.zeros((b - a, 3))
    cm = np.zeros((b - a, 3))
    for k in range(b - a):
        for i in range(3):
            sr = 0.0
            sc = 0.0
            for j in range(3):
                sr += rot[i, j] * co.c.rmags[a + k, j]
                sc += rot[i, j] * co.c.cosmags[a + k, j]
            rm[k, i] = sr
            cm[k, i] = sc
    if inplace:
        for k in range(b - a):
            for i in range(3):
                co.c.rmags[a + k, i] = rm[k, i]
                co.c.cosmags[a + k, i] = cm[k, i]
    return _MFCoils(rm, cm, np.zeros(b - a, dtype=int), 1)


def _col_pinv(S: np.ndarray[float, 2]) -> np.ndarray[float, 2]:
    return _col_norm_pinv(S.copy())[0]


def _err_total(data: np.ndarray[float, 2], S_tot: np.ndarray[float, 2], cals: np.ndarray[float, 1]) -> float:
    S = S_tot.copy()
    for i in range(S.shape[0]):
        S[i, :] /= cals[i]
    model = np.dot(np.dot(S, _col_pinv(S)), data)
    d = model - data
    return 100.0 * math.sqrt(float(np.sum(d * d))) / math.sqrt(float(np.sum(data * data)))


def _err_rows(data: np.ndarray[float, 2], S_tot: np.ndarray[float, 2], cals: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
    S = S_tot.copy()
    for i in range(S.shape[0]):
        S[i, :] /= cals[i]
    model = np.dot(np.dot(S, _col_pinv(S)), data)
    out = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        a = 0.0
        b = 0.0
        for t in range(data.shape[1]):
            d = model[i, t] - data[i, t]
            a += d * d
            b += data[i, t] * data[i, t]
        out[i] = 100.0 * math.sqrt(a) / math.sqrt(b)
    return out


def _err_one(data: np.ndarray[float, 2], S_tot: np.ndarray[float, 2], cals: np.ndarray[float, 1],
             idx: int) -> float:
    S = S_tot.copy()
    for i in range(S.shape[0]):
        S[i, :] /= cals[i]
    row = np.dot(S[idx], _col_pinv(S))
    model = np.dot(row, data)
    a = 0.0
    b = 0.0
    for t in range(data.shape[1]):
        d = model[t] - data[idx, t]
        a += d * d
        b += data[idx, t] * data[idx, t]
    return 100.0 * math.sqrt(a) / math.sqrt(b)


class _CalTarget:
    old_z: np.ndarray[float, 1]
    co: _Coil
    cal_idx: int
    data: np.ndarray[float, 2]
    cals: np.ndarray[float, 1]
    match: List[int]
    S_tot: np.ndarray[float, 2]
    origin: np.ndarray[float, 1]
    ext_order: int

    def __init__(self, old_z, co: _Coil, cal_idx: int, data, cals, match: List[int], S_tot, origin, ext_order: int):
        self.old_z = old_z
        self.co = co
        self.cal_idx = cal_idx
        self.data = data
        self.cals = cals
        self.match = match
        self.S_tot = S_tot
        self.origin = origin
        self.ext_order = ext_order

    def __call__(self, new_z: np.ndarray[float, 1]) -> float:
        nrm = math.sqrt(float(np.dot(new_z, new_z)))
        self.cals[self.cal_idx] = 1.0 / nrm
        S = self.S_tot.copy()
        sc = np.full(1, _MAG_SCALE)
        for idx in self.match:
            sub = _rotated_single(self.co, idx, new_z, self.old_z, False)
            row = _basis_scaled(sub, self.origin, self.ext_order, sc)
            S[idx, :] = row[0, :]
        return _err_one(self.data, S, self.cals, self.cal_idx)


def _adjust_mag_normals(info: Info, data_all: np.ndarray[float, 2], locs_all: np.ndarray[float, 2],
                        origin: np.ndarray[float, 1], ext_order: int, angle_limit: float,
                        err_limit: float):
    # info/data_all/locs_all: MEG channels only. Returns (zs (n_mag, 3), imb_cals (n_meg,), good)
    n_meg = info.nchan
    mag = [i for i in range(n_meg) if info.ch_types[i] == "mag"]
    nm = len(mag)
    data = np.zeros((nm, data_all.shape[1]))
    for k in range(nm):
        data[k, :] = data_all[mag[k], :] * _MAG_SCALE
    cals = np.ones(nm)
    angles = np.zeros(nm)
    minfo = Info([info.ch_names[i] for i in mag], info.sfreq, ["mag" for _ in mag])
    for k in range(nm):
        minfo.locs[k] = [locs_all[mag[k], j] for j in range(12)]
        minfo.coil_types[k] = info.coil_types[mag[k]]
    co = _Coil(_prep_mf_coils(minfo, [i for i in range(nm)]))
    scale = np.full(nm, _MAG_SCALE)
    S_tot = _basis_scaled(co.c, origin, ext_order, scale)
    first_err = _err_total(data, S_tot, cals)
    zs = np.zeros((nm, 3))
    for k in range(nm):
        zs[k] = _unit(np.array([minfo.locs[k][9], minfo.locs[k][10], minfo.locs[k][11]]))
    orig_zs = zs.copy()
    mlocs = np.zeros((nm, 12))
    for k in range(nm):
        for j in range(12):
            mlocs[k, j] = minfo.locs[k][j]
    match = List[List[int]]()
    for k in range(nm):
        match.append(_matched_loc_idx(mlocs[k], mlocs))
    counts = [0 for _ in range(nm)]
    for ki in range(2):
        S_tot = _basis_scaled(co.c, origin, ext_order, scale)
        for pick in range(nm):
            cal_idx = pick
            if ki == 0:
                err = _err_rows(data, S_tot, cals)
                order = List[int]()
                for q in range(nm):
                    order.append(q)
                order.sort(key = lambda q: err[q])
                cal_idx = 0
                while len(order) > 0:
                    cal_idx = order.pop()
                    if counts[cal_idx] < 3:
                        break
                if err[cal_idx] < 2.5:
                    break
            counts[cal_idx] += 1
            old_z = zs[cal_idx].copy()
            obj = _CalTarget(old_z, co, cal_idx, data, cals, match[cal_idx], S_tot, origin, ext_order)
            bnds = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
            res = minimize(obj, old_z, bounds = bnds, method = "COBYLA", options = {"rhobeg": 1e-1})
            zs[cal_idx] = res.x
            nrm = math.sqrt(float(np.dot(zs[cal_idx], zs[cal_idx])))
            cals[cal_idx] = 1.0 / nrm
            zs[cal_idx] = zs[cal_idx] * cals[cal_idx]
            for idx in match[cal_idx]:
                _rotated_single(co, idx, zs[cal_idx], old_z, True)
            S_tot = _basis_scaled(co.c, origin, ext_order, scale)
            angles[cal_idx] = abs(_angle_deg(zs[cal_idx], orig_zs[cal_idx]))
    last_err = _err_total(data, S_tot, cals)
    max_angle = 0.0
    for q in range(nm):
        max_angle = max(max_angle, angles[q])
    good = max_angle < angle_limit
    each = _err_rows(data, S_tot, cals)
    for q in range(nm):
        if each[q] > err_limit:
            good = False
    imb = np.ones(n_meg)
    for k in range(nm):
        imb[mag[k]] = cals[k]
    return zs, imb, good


def _grad_point_coils(locs: np.ndarray[float, 2], grad: List[int], n_types: int) -> List[_MFCoils]:
    out = List[_MFCoils]()
    for r in range(n_types):
        rm = np.zeros((len(grad), 3))
        cm = np.zeros((len(grad), 3))
        for k in range(len(grad)):
            t = _loc_to_trans(locs[grad[k]].copy())
            for i in range(3):
                rm[k, i] = t[i, 3]
                cm[k, i] = t[i, r]
        out.append(_MFCoils(rm, cm, np.arange(len(grad)), len(grad)))
    return out


def _estimate_imbalance(info: Info, data_in: np.ndarray[float, 2], locs: np.ndarray[float, 2],
                        cals: np.ndarray[float, 1], n_imbalance: int, origin: np.ndarray[float, 1],
                        ext_order: int) -> np.ndarray[float, 2]:
    n_meg = info.nchan
    mag = [i for i in range(n_meg) if info.ch_types[i] == "mag"]
    grad = [i for i in range(n_meg) if info.ch_types[i] == "grad"]
    ng = len(grad)
    data = data_in.copy()
    for i in mag:
        data[i, :] *= _MAG_SCALE
    imb = np.zeros((ng, n_imbalance))
    minfo = info.copy()
    for k in range(n_meg):
        minfo.locs[k] = [locs[k, j] for j in range(12)]
    all_c = _prep_mf_coils(minfo, [i for i in range(n_meg)])
    csc = np.ones(n_meg)
    for i in mag:
        csc[i] = _MAG_SCALE
    S_orig = _basis_scaled(all_c, origin, ext_order, csc)
    for i in range(n_meg):
        S_orig[i, :] /= cals[i]
    pts = _grad_point_coils(locs, grad, n_imbalance)
    S_pt = List[np.ndarray[float, 2]]()
    for c in pts:
        S_pt.append(_basis_scaled(c, origin, ext_order, np.full(ng, _MAG_SCALE)))
    n_mom = S_orig.shape[1]
    nt = data.shape[1]
    for it in range(3):
        S_tot = S_orig.copy()
        S_recon = np.zeros((ng, n_mom))
        for j in range(ng):
            S_recon[j, :] = S_tot[grad[j], :]
        for j in range(ng):
            for i in range(n_imbalance):
                S_tot[grad[j], :] += imb[j, i] * S_pt[i][j, :]
        mm = np.dot(_col_pinv(S_tot), data)
        data_recon = np.dot(S_recon, mm)
        new = np.zeros((ng, n_imbalance))
        for j in range(ng):
            khi = np.zeros((nt, n_imbalance))
            for i in range(n_imbalance):
                khi[:, i] = np.dot(S_pt[i][j, :], mm)
            resid = data[grad[j], :] - data_recon[j, :]
            d = np.dot(resid, khi)
            A = np.dot(khi.T, khi)
            u, s, vh = np.linalg.svd(A)
            tol = 1e-6 * s[0]
            sinv = np.zeros(s.shape[0])
            for q in range(s.shape[0]):
                if s[q] > tol and s[q] > 0.0:
                    sinv[q] = 1.0 / s[q]
            dinv = np.dot(vh.T * sinv, u.T)
            new[j, :] = np.dot(d, dinv)
        imb = new
    out = np.zeros((n_meg, n_imbalance))
    for j in range(ng):
        out[grad[j], :] = imb[j, :]
    return out


def _clean_names(names: List[str]) -> List[str]:
    cleaned = List[str]()
    for name in names:
        n = name
        if " " in n:
            n = n.replace(" ", "")
        if "-" in n:
            n = n.split("-")[0]
        if n.endswith("_v"):
            n = n[:len(n) - 2]
        cleaned.append(n)
    uniq = Set[str]()
    for c in cleaned:
        uniq.add(c)
    if len(uniq) != len(names):
        return names
    return cleaned


def compute_fine_calibration(raw, n_imbalance: int = 3, t_window: float = 10.0, ext_order: int = 2,
                             origin = (0.0, 0.0, 0.0), cross_talk = None, calibration = None, *,
                             angle_limit: float = 5.0, err_limit: float = 5.0, verbose = None):
    """Compute fine calibration from empty-room data (mne 1.13.2).

    Returns ((ch_names, locs (n, 12), imb_cals), count)."""
    if n_imbalance != 1 and n_imbalance != 3:
        raise ValueError(f"Invalid value for the 'n_imbalance' parameter. Allowed values are 1 and 3, but got {n_imbalance} instead.")
    if cross_talk is not None:
        raise NotImplementedError("cross_talk is not supported in Typhon mne")
    if calibration is not None:
        raise NotImplementedError("calibration input is not supported in Typhon mne")
    if isinstance(origin, str):
        raise NotImplementedError("origin='auto' is not supported, pass explicit coordinates")
    org = np.array(origin, dtype=float)
    if org.shape[0] != 3:
        raise ValueError("origin must be a 3-element array")
    if len(raw.info.bads) != 0:
        raise ValueError("raw.info['bads'] must be empty")
    if angle_limit < 0:
        raise ValueError(f"angle_limit must be greater than or equal to 0, got {angle_limit}")
    if err_limit < 0:
        raise ValueError(f"err_limit must be greater than or equal to 0, got {err_limit}")
    picks = List[int]()
    for i in range(raw.info.nchan):
        t = raw.info.ch_types[i]
        if t == "mag" or t == "grad":
            picks.append(i)
    info = raw.info._pick(picks)
    n_meg = info.nchan
    mag = [i for i in range(n_meg) if info.ch_types[i] == "mag"]
    grad = [i for i in range(n_meg) if info.ch_types[i] == "grad"]
    full = raw._data
    n_samp = full.shape[1]
    cals = np.ones(n_meg)
    end = n_samp + 1
    step = int(round(t_window * raw.info.sfreq))
    tidx = List[int]()
    q = 0
    while q < end:
        tidx.append(q)
        q += step
    if len(tidx) == 1:
        tidx.append(end)
    if tidx[len(tidx) - 1] != end:
        tidx[len(tidx) - 1] = end
    count = 0
    locs = np.zeros((n_meg, 12))
    for k in range(n_meg):
        for j in range(12):
            locs[k, j] = info.locs[k][j]
    zs = np.zeros((len(mag), 3))
    for k in range(len(mag)):
        for j in range(3):
            zs[k, j] = locs[mag[k], 9 + j]
    segs = List[Tuple[int, int]]()
    for q in range(len(tidx) - 1):
        segs.append((tidx[q], tidx[q + 1]))

    def seg_data(a: int, b: int) -> np.ndarray[float, 2]:
        bb = min(b, n_samp)
        d = np.zeros((n_meg, bb - a))
        for k in range(n_meg):
            d[k, :] = full[picks[k], a:bb]
        return d

    if len(mag) > 0:
        cal_list = List[np.ndarray[float, 1]]()
        z_list = List[np.ndarray[float, 2]]()
        for sg in segs:
            data = seg_data(sg[0], sg[1])
            z, cal, good = _adjust_mag_normals(info, data, locs, org, ext_order, angle_limit, err_limit)
            if good:
                z_list.append(z)
                cal_list.append(cal)
        count = len(cal_list)
        if count == 0:
            raise RuntimeError("No usable segments found")
        cals = np.zeros(n_meg)
        zs = np.zeros((len(mag), 3))
        for c in cal_list:
            cals += c
        for z in z_list:
            zs += z
        cals = cals / float(count)
        zs = zs / float(count)
        for ii in range(len(mag)):
            idxs = _matched_loc_idx(locs[mag[ii]].copy(), locs)
            _rotate_locs(locs, idxs, zs[ii].copy())
    imb = np.zeros((n_meg, n_imbalance))
    if len(grad) > 0:
        acc = np.zeros((n_meg, n_imbalance))
        for sg in segs:
            data = seg_data(sg[0], sg[1])
            acc += _estimate_imbalance(info, data, locs, cals, n_imbalance, org, ext_order)
        imb = acc / float(len(segs))
    imb_cals = List[List[float]]()
    for ii in range(n_meg):
        if info.ch_types[ii] == "mag":
            imb_cals.append([cals[ii]])
        else:
            imb_cals.append([imb[ii, j] for j in range(n_imbalance)])
    names = _clean_names(info.ch_names)
    return (names, locs, imb_cals), count
