# Ported from mne 1.13.2 (BSD-3-Clause)
#
# annotate_movement (rotation / translation velocity limits) and
# compute_average_dev_head_t. The mean_distance_limit option needs cHPI
# digitization points that Typhon infos do not carry (raises).

import math
import numpy as np
from ..annotations import Annotations


def _quat_real(q0: float, q1: float, q2: float) -> float:
    return math.sqrt(max(1.0 - q0 * q0 - q1 * q1 - q2 * q2, 0.0))


def quat_to_rot(quat) -> np.ndarray[float, 2]:
    """Rotation matrix from a 3-element unit quaternion (mne.transforms.quat_to_rot)."""
    q = np.asarray(quat, dtype=float)
    b = q[0]
    c = q[1]
    d = q[2]
    bb = b * b
    cc = c * c
    dd = d * d
    aa = max(1.0 - bb - cc - dd, 0.0)
    a = math.sqrt(aa)
    ab_2 = 2 * a * b
    ac_2 = 2 * a * c
    ad_2 = 2 * a * d
    bc_2 = 2 * b * c
    bd_2 = 2 * b * d
    cd_2 = 2 * c * d
    r = np.empty((3, 3))
    r[0, 0] = aa + bb - cc - dd
    r[0, 1] = bc_2 - ad_2
    r[0, 2] = bd_2 + ac_2
    r[1, 0] = bc_2 + ad_2
    r[1, 1] = aa + cc - bb - dd
    r[1, 2] = cd_2 - ab_2
    r[2, 0] = bd_2 - ac_2
    r[2, 1] = cd_2 + ab_2
    r[2, 2] = aa + dd - bb - cc
    return r


def rot_to_quat(rot) -> np.ndarray[float, 1]:
    """3-element quaternion of a rotation matrix (mne.transforms.rot_to_quat)."""
    R = np.asarray(rot, dtype=float)
    r = [R[0, 0], R[0, 1], R[0, 2], R[1, 0], R[1, 1], R[1, 2], R[2, 0], R[2, 1], R[2, 2]]
    det = (r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6])
           + r[2] * (r[3] * r[7] - r[4] * r[6]))
    if abs(det - 1.0) > 1e-3:
        raise ValueError("Matrix is not a pure rotation, got determinant != 1")
    t = 1.0 + r[0] + r[4] + r[8]
    qx = 0.0
    qy = 0.0
    qz = 0.0
    if t > 2.220446049250313e-16:
        s = math.sqrt(t) * 2.0
        qx = (r[7] - r[5]) / s
        qy = (r[2] - r[6]) / s
        qz = (r[3] - r[1]) / s
    elif r[0] > r[4] and r[0] > r[8]:
        s = math.sqrt(1.0 + r[0] - r[4] - r[8]) * 2.0
        qx = 0.25 * s
        qy = (r[1] + r[3]) / s
        qz = (r[2] + r[6]) / s
    elif r[4] > r[8]:
        s = math.sqrt(1.0 - r[0] + r[4] - r[8]) * 2
        qx = (r[1] + r[3]) / s
        qy = 0.25 * s
        qz = (r[5] + r[7]) / s
    else:
        s = math.sqrt(1.0 - r[0] - r[4] + r[8]) * 2.0
        qx = (r[2] + r[6]) / s
        qy = (r[5] + r[7]) / s
        qz = 0.25 * s
    return np.array([qx, qy, qz])


def _angle_between(x0: float, x1: float, x2: float, y0: float, y1: float, y2: float) -> float:
    # z = conj(x) * y with 3-element quaternions
    a0 = -x0
    a1 = -x1
    a2 = -x2
    w1 = _quat_real(a0, a1, a2)
    w2 = _quat_real(y0, y1, y2)
    o0 = w1 * y0 + a0 * w2 + a1 * y2 - a2 * y1
    o1 = w1 * y1 - a0 * y2 + a1 * w2 + a2 * y0
    o2 = w1 * y2 + a0 * y1 - a1 * y0 + a2 * w2
    w = w1 * w2 - a0 * y0 - a1 * y1 - a2 * y2
    sg = 0.0
    if w > 0.0:
        sg = 1.0
    elif w < 0.0:
        sg = -1.0
    o0 *= sg
    o1 *= sg
    o2 *= sg
    z0 = _quat_real(o0, o1, o2)
    return 2.0 * math.atan2(math.sqrt(o0 * o0 + o1 * o1 + o2 * o2), z0)


def _mask_runs(mask: List[bool], times: np.ndarray[float, 1], name: str, orig_time: Optional[float]) -> Annotations:
    """mne annotations._annotations_from_mask for a mask shorter than times by one."""
    on = List[float]()
    du = List[float]()
    n = len(mask)
    nt = times.shape[0]
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            e = j
            if nt == n:
                if e >= nt:
                    e = nt - 1
            else:
                if e >= n:
                    e = n
            on.append(float(times[i]))
            du.append(float(times[e]) - float(times[i]))
            i = j
        else:
            i += 1
    return Annotations(on, du, [name for _ in range(len(on))], orig_time)


def annotate_movement(raw, pos, rotation_velocity_limit = None, translation_velocity_limit = None,
                      mean_distance_limit = None, use_dev_head_trans: str = "average"):
    """Detect segments with head movement (mne 1.13.2). Returns (annot, disp)."""
    if not isinstance(mean_distance_limit, NoneType):
        raise NotImplementedError("mean_distance_limit needs cHPI digitization, which is not available")
    sfreq = raw.info.sfreq
    P = np.asarray(pos, dtype=float)
    npos = P.shape[0]
    ft = raw.first_time
    hp_ts = np.empty(npos + 1)
    for i in range(npos):
        hp_ts[i] = P[i, 0] - ft
    hp_ts[npos] = hp_ts[npos - 1] + 1.0 / sfreq
    orig_time = raw.info.meas_date
    on = List[float]()
    du = List[float]()
    de = List[str]()
    if not isinstance(rotation_velocity_limit, NoneType):
        lim = float(rotation_velocity_limit)
        assert lim > 0.0
        lim_r = lim * math.pi / 180.0
        mask = List[bool]()
        for i in range(npos - 1):
            ang = _angle_between(P[i, 1], P[i, 2], P[i, 3], P[i + 1, 1], P[i + 1, 2], P[i + 1, 3])
            mask.append(ang / (P[i + 1, 0] - P[i, 0]) >= lim_r)
        a = _mask_runs(mask, hp_ts, "BAD_mov_rotat_vel", orig_time)
        for k in range(len(a)):
            on.append(a.onset[k])
            du.append(a.duration[k])
            de.append(a.description[k])
    if not isinstance(translation_velocity_limit, NoneType):
        lim = float(translation_velocity_limit)
        assert lim > 0.0
        mask = List[bool]()
        for i in range(npos - 1):
            dx = P[i + 1, 4] - P[i, 4]
            dy = P[i + 1, 5] - P[i, 5]
            dz = P[i + 1, 6] - P[i, 6]
            v = math.sqrt(dx * dx + dy * dy + dz * dz) / (P[i + 1, 0] - P[i, 0])
            mask.append(v >= lim)
        a = _mask_runs(mask, hp_ts, "BAD_mov_trans_vel", orig_time)
        for k in range(len(a)):
            on.append(a.onset[k])
            du.append(a.duration[k])
            de.append(a.description[k])
    if not isinstance(orig_time, NoneType):
        on = [o + ft for o in on]
    annot = Annotations(on, du, de, orig_time)
    return annot, np.zeros((0, 0))


def _average_quats(q: np.ndarray[float, 2], w: np.ndarray[float, 1]) -> np.ndarray[float, 1]:
    n = q.shape[0]
    tot = float(w.sum())
    if tot == 0.0:
        return np.zeros(3)
    A = np.zeros((4, 4))
    for i in range(n):
        wi = w[i] / tot
        v = [_quat_real(q[i, 0], q[i, 1], q[i, 2]) * wi, q[i, 0] * wi, q[i, 1] * wi, q[i, 2] * wi]
        for a in range(4):
            for b in range(4):
                A[a, b] += v[a] * v[b]
    ev, evec = np.linalg.eigh(A)
    best = np.empty(4)
    for k in range(4):
        best[k] = evec[k, 3]
    first = 0.0
    for k in range(4):
        if best[k] != 0.0:
            first = best[k]
            break
    sg = 1.0 if first > 0.0 else -1.0
    return np.array([best[1] * sg, best[2] * sg, best[3] * sg])


def _raw_hp_weights(raw, pos: np.ndarray[float, 2]):
    sfreq = raw.info.sfreq
    n = raw._data.shape[1]
    seg_good = np.ones(n)
    ft = raw.first_time
    keep = List[int]()
    for i in range(pos.shape[0]):
        t = pos[i, 0] - ft
        if i == 0 and t < 0.0:
            t = 0.0
        if t <= float(n - 1) / sfreq:
            keep.append(i)
    hp = np.empty((len(keep), pos.shape[1]))
    for a in range(len(keep)):
        hp[a, :] = pos[keep[a], :]
    idx = List[int]()
    for a in range(len(keep)):
        t = hp[a, 0] - raw.first_samp / sfreq
        v = int(round(t * sfreq))
        if a == 0 and v < 0:
            v = 0
        idx.append(v)
    idx.append(n)
    ann = raw.annotations
    for k in range(len(ann)):
        if ann.description[k].upper().startswith("BAD"):
            s = int(round((ann.onset[k] - ft) * sfreq))
            e = int(round((ann.onset[k] - ft + ann.duration[k]) * sfreq))
            for q in range(max(s, 0), min(e, n)):
                seg_good[q] = 0.0
    cs = np.zeros(n + 1)
    for q in range(n):
        cs[q + 1] = cs[q] + seg_good[q]
    dt = np.empty(len(keep))
    for a in range(len(keep)):
        dt[a] = (cs[idx[a + 1]] - cs[idx[a]]) / sfreq
    return hp, dt


def compute_average_dev_head_t(raw, pos, *, verbose = None):
    """Average device-to-head transform (4x4 array) over cHPI position fits (mne 1.13.2)."""
    raws = [raw] if not isinstance(raw, list) else raw
    poss = [pos] if not isinstance(pos, list) else pos
    if len(poss) != len(raws):
        raise ValueError(f"Number of head positions ({len(poss)}) must match the number of raw instances ({len(raws)})")
    hps = List[np.ndarray[float, 2]]()
    dts = List[np.ndarray[float, 1]]()
    tot = 0
    for k in range(len(raws)):
        hp, dt = _raw_hp_weights(raws[k], np.asarray(poss[k], dtype=float))
        hps.append(hp)
        dts.append(dt)
        tot += hp.shape[0]
    hp = np.empty((tot, hps[0].shape[1]))
    dt = np.empty(tot)
    o = 0
    for k in range(len(hps)):
        for a in range(hps[k].shape[0]):
            hp[o, :] = hps[k][a, :]
            dt[o] = dts[k][a]
            o += 1
    dt = dt / float(dt.sum())
    q = np.empty((tot, 3))
    for a in range(tot):
        q[a, :] = hp[a, 1:4]
    best_q = _average_quats(q, dt)
    trans = np.eye(4)
    trans[:3, :3] = quat_to_rot(best_q)
    for c in range(3):
        s = 0.0
        for a in range(tot):
            s += dt[a] * hp[a, 4 + c]
        trans[c, 3] = s
    return trans
