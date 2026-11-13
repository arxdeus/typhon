# Ported from mne 1.13.2 (BSD-3-Clause)
#
# interpolate_bridged_electrodes: interpolate groups of bridged EEG
# electrodes using a virtual channel at the group centroid.

import math
import numpy as np
from .._info import Info
from .._containers import RawArray, EpochsArray, Evoked
from ..channels import _make_interpolation_matrix, _auto_origin


def _centroid_sphere(pts: List[List[float]]) -> List[float]:
    n = len(pts)
    avg_r = 0.0
    cx = 0.0
    cy = 0.0
    cz = 0.0
    for p in pts:
        avg_r += math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])
        cx += p[0]
        cy += p[1]
        cz += p[2]
    avg_r /= n
    cx /= n
    cy /= n
    cz /= n
    r = math.sqrt(cx * cx + cy * cy + cz * cz)
    nrm = r if r > 0.0 else 1.0
    az = math.atan2(cy, cx)
    pol = math.acos(cz / nrm)
    if pol != pol:
        pol = 0.0
    return [avg_r * math.sin(pol) * math.cos(az), avg_r * math.sin(pol) * math.sin(az),
            avg_r * math.cos(pol)]


def _bridged_groups(bridged_idx) -> List[List[int]]:
    nodes = List[int]()
    for b in bridged_idx:
        for v in (int(b[0]), int(b[1])):
            if v not in nodes:
                nodes.append(v)
    nodes = sorted(nodes)
    parent = [i for i in range(len(nodes))]

    def find(x: int) -> int:
        while parent[x] != x:
            x = parent[x]
        return x

    for b in bridged_idx:
        i0 = nodes.index(int(b[0]))
        i1 = nodes.index(int(b[1]))
        r0 = find(i0)
        r1 = find(i1)
        if r0 != r1:
            parent[r1] = r0
    roots = List[int]()
    labels = List[int]()
    for i in range(len(nodes)):
        r = find(i)
        if r not in roots:
            roots.append(r)
        labels.append(roots.index(r))
    groups = List[List[int]]()
    for k in range(len(roots)):
        g = List[int]()
        for i in range(len(nodes)):
            if labels[i] == k:
                g.append(nodes[i])
        groups.append(g)
    return groups


def _interp_bridged_data(info: Info, data, bridged_idx, bad_limit: int):
    """In-place interpolation on data[..., ch, t] (2-D or 3-D)."""
    if bad_limit <= 0:
        raise ValueError(f"Argument 'bad_limit' should be a strictly positive integer. Provided {bad_limit} is invalid.")
    if info._montage is None:
        raise RuntimeError("No channel positions found in ``inst``")
    m = info._montage.__val__()
    if m.coord_frame != "head":
        raise RuntimeError(f"Montage channel positions must be in ``head`` got {m.coord_frame}")
    groups = _bridged_groups(bridged_idx)
    for g in groups:
        if len(g) > bad_limit:
            nm = ", ".join([info.ch_names[k] for k in g])
            raise RuntimeError(f"The channels {nm} are bridged together and form a large area of bridged electrodes. Interpolation might be inaccurate.")
    origin = _auto_origin(info)
    bad_set = List[int]()
    for g in groups:
        for k in g:
            bad_set.append(k)
    goods = List[int]()
    for i in range(info.nchan):
        if info.ch_types[i] == "eeg" and i not in bad_set:
            goods.append(i)
    vpos = List[List[float]]()
    for g in groups:
        pts = List[List[float]]()
        for k in g:
            j = m.ch_names.index(info.ch_names[k])
            pts.append(list(m.pos[j]))
        vpos.append(_centroid_sphere(pts))
    ng = len(goods)
    nv = len(groups)
    pg = np.empty((ng + nv, 3))
    for a in range(ng):
        for c in range(3):
            pg[a, c] = info.locs[goods[a]][c] - origin[c]
    for v in range(nv):
        for c in range(3):
            pg[ng + v, c] = vpos[v][c] - origin[c]
    bads = sorted(bad_set)
    pb = np.empty((len(bads), 3))
    for a in range(len(bads)):
        for c in range(3):
            pb[a, c] = info.locs[bads[a]][c] - origin[c]
    interp = _make_interpolation_matrix(pg, pb)
    if data.ndim == 2:
        src = np.empty((ng + nv, data.shape[1]))
        for a in range(ng):
            src[a, :] = data[goods[a], :]
        for v in range(nv):
            src[ng + v, :] = 0.0
            for k in groups[v]:
                src[ng + v, :] += data[k, :]
            src[ng + v, :] /= len(groups[v])
        out = interp @ src
        for a in range(len(bads)):
            data[bads[a], :] = out[a]
    else:
        for e in range(data.shape[0]):
            src = np.empty((ng + nv, data.shape[2]))
            for a in range(ng):
                src[a, :] = data[e, goods[a], :]
            for v in range(nv):
                src[ng + v, :] = 0.0
                for k in groups[v]:
                    src[ng + v, :] += data[e, k, :]
                src[ng + v, :] /= len(groups[v])
            out = interp @ src
            for a in range(len(bads)):
                data[e, bads[a], :] = out[a]


def interpolate_bridged_electrodes(inst, bridged_idx, bad_limit: int = 4):
    """Interpolate bridged electrode pairs (mne 1.13.2)."""
    if isinstance(inst, RawArray):
        _interp_bridged_data(inst.info, inst._data, bridged_idx, bad_limit)
    elif isinstance(inst, EpochsArray):
        _interp_bridged_data(inst.info, inst._data, bridged_idx, bad_limit)
    elif isinstance(inst, Evoked):
        _interp_bridged_data(inst.info, inst.data, bridged_idx, bad_limit)
    else:
        compile_error("Not a Raw, Epochs or Evoked")
    return inst
