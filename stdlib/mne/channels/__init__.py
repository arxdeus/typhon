# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# mne.channels subset: DigMontage, make_standard_montage (colin27/standard
# 1005/1020, biosemi*, easycap-M1/M10), make_dig_montage, set_montage /
# get_montage, spherical-spline interpolate_bads, find_ch_adjacency
# (Delaunay on topomap coordinates), read_ch_adjacency from neighbor lists,
# equalize_channels.

import numpy as np
from .._info import Info, _picks_to_idx, DigMontage, _apply, _ras_to_neuromag, create_info

HEAD_SIZE_DEFAULT = 0.095
_MNI_LPA = (-0.08061611652374268, -0.02908875234425068, -0.04131076857447624)
_MNI_NAS = (0.0014676303835585713, 0.08506715297698975, -0.03483611345291138)
_MNI_RPA = (0.08436284959316254, -0.02850276418030262, -0.041277434676885605)

_DEPRECATED = (("standard_1005", "colin27_1005"), ("standard_1020", "colin27_1020"))
_BUILTIN = ("colin27_1005", "colin27_1020", "biosemi16", "biosemi32", "biosemi64",
            "biosemi128", "easycap-M1", "easycap-M10")


def _data_dir() -> str:
    f = __file__
    return f[:f.rfind("/") + 1] + "data/montages/"


def make_dig_montage(ch_pos = None, nasion = None, lpa = None, rpa = None, hsp = None,
                     hpi = None, coord_frame: str = "unknown") -> DigMontage:
    names = List[str]()
    pos = List[List[float]]()
    if not isinstance(ch_pos, NoneType):
        for k, v in ch_pos.items():
            names.append(str(k))
            pos.append([float(v[0]), float(v[1]), float(v[2])])
    return DigMontage(names, pos, coord_frame, _opt3(nasion), _opt3(lpa), _opt3(rpa))


def _opt3(x) -> Optional[List[float]]:
    if isinstance(x, NoneType):
        return None
    else:
        return [float(x[0]), float(x[1]), float(x[2])]


def _sph_to_cart(r: float, az: float, pol: float) -> List[float]:
    z = r * np.cos(pol)
    xy = r * np.sin(pol)
    return [xy * np.cos(az), xy * np.sin(az), z]


def _read_theta_phi(fname: str, head_size: float, fid_names: List[str], add_fid: bool) -> DigMontage:
    names = List[str]()
    theta = List[float]()
    phi = List[float]()
    first = True
    with open(fname) as f:
        for line in f:
            if first:
                first = False
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            names.append(parts[0])
            # numpy loadtxt dtype 'i4' truncates fractional degrees
            theta.append(float(int(float(parts[1]))))
            phi.append(float(int(float(parts[2]))))
    fids = fid_names
    if add_fid:
        fids = ["Nasion", "LPA", "RPA"]
        names.extend(fids)
        theta.extend([115.0, -115.0, 115.0])
        phi.extend([90.0, 0.0, 0.0])
    pos = [_sph_to_cart(head_size, np.deg2rad(phi[i]), np.deg2rad(theta[i])) for i in range(len(names))]
    nas: Optional[List[float]] = None
    l: Optional[List[float]] = None
    r: Optional[List[float]] = None
    cn = List[str]()
    cp = List[List[float]]()
    for i in range(len(names)):
        if len(fids) == 3 and names[i] == fids[0]:
            nas = pos[i]
        elif len(fids) == 3 and names[i] == fids[1]:
            l = pos[i]
        elif len(fids) == 3 and names[i] == fids[2]:
            r = pos[i]
        else:
            cn.append(names[i])
            cp.append(pos[i])
    return DigMontage(cn, cp, "unknown", nas, l, r)


def _read_elc_mgh(fname: str) -> DigMontage:
    """_mgh_or_standard(..., coord_frame='mri', head_size=None)."""
    pos = List[List[float]]()
    names = List[str]()
    state = 0
    with open(fname) as f:
        for line in f:
            if state == 0:
                if "Positions\n" in line:
                    state = 1
            elif state == 1:
                if "Labels\n" in line:
                    state = 2
                    continue
                pos.append([float(x) / 1000.0 for x in line.split()])
            elif state == 2:
                if len(line.strip(" ")) == 0 or line.strip(" \n") == "":
                    break
                names.append(line.strip(" ").strip("\n"))
    cn = List[str]()
    cp = List[List[float]]()
    for i in range(len(names)):
        if names[i] in ("Nz", "LPA", "RPA"):
            continue
        if names[i] in cn:
            raise ValueError(f"Duplicate channel name {names[i]}")
        cn.append(names[i])
        cp.append(pos[i])
    return DigMontage(cn, cp, "mri", list(_MNI_NAS), list(_MNI_LPA), list(_MNI_RPA))


def make_standard_montage(kind: str, head_size = "auto") -> DigMontage:
    k = kind
    for p in _DEPRECATED:
        if k == p[0]:
            print(f"FutureWarning: Montage name '{k}' is deprecated and will be removed in MNE 1.14. Use '{p[1]}' instead.")
            k = p[1]
    if k not in _BUILTIN:
        raise ValueError(f"Invalid value for the 'kind' parameter. Allowed values are {list(_BUILTIN)}, but got '{kind}' instead (Typhon mne ships a subset of mne's montages).")
    hs = HEAD_SIZE_DEFAULT
    if not isinstance(head_size, str) and not isinstance(head_size, NoneType):
        hs = float(head_size)
    d = _data_dir()
    if k.startswith("colin27"):
        return _read_elc_mgh(d + k + ".elc")
    if k.startswith("biosemi"):
        return _read_theta_phi(d + k + ".txt", hs, ["Nz", "LPA", "RPA"], False)
    return _read_theta_phi(d + k + ".txt", hs, List[str](), True)


_EEG_LIKE = ("eeg", "seeg", "dbs", "ecog")


def _set_montage(info: Info, montage, match_case: bool, on_missing: str):
    if isinstance(montage, NoneType):
        for i in range(info.nchan):
            info.locs[i] = [np.nan for _ in range(12)]
        info._montage = None
        return
    m = make_standard_montage(montage) if isinstance(montage, str) else montage
    h = m._to_head()
    picks = [i for i in range(info.nchan) if info.ch_types[i] in _EEG_LIKE]
    keys = h.ch_names if match_case else [c.lower() for c in h.ch_names]
    missing = List[str]()
    for i in picks:
        nm = info.ch_names[i] if match_case else info.ch_names[i].lower()
        if nm in keys:
            p = h.pos[keys.index(nm)]
            loc = [p[0], p[1], p[2], 0.0, 0.0, 0.0]
            if np.isnan(p[0]) or np.isnan(p[1]) or np.isnan(p[2]):
                loc = [np.nan for _ in range(6)]
            for q in range(6):
                info.locs[i][q] = loc[q]
        else:
            missing.append(info.ch_names[i])
    if len(missing) > 0:
        msg = f"DigMontage is only a subset of info. There {'are' if len(missing) > 1 else 'is'} {len(missing)} channel position{'s' if len(missing) > 1 else ''} not present in the DigMontage. The channel{'s' if len(missing) > 1 else ''} missing from the montage {'are' if len(missing) > 1 else 'is'}:\n\n{missing}.\n\nConsider using inst.rename_channels to match the montage nomenclature, or inst.set_channel_types if {'these' if len(missing) > 1 else 'this'} {'are' if len(missing) > 1 else 'is'} not {'' if len(missing) > 1 else 'an '}EEG channel{'s' if len(missing) > 1 else ''}, or use the on_missing parameter if the channel position{'s' if len(missing) > 1 else ''} {'are' if len(missing) > 1 else 'is'} allowed to be unknown in your analyses."
        if on_missing == "raise":
            raise ValueError(msg)
        elif on_missing == "warn":
            print("RuntimeWarning:", msg)
        for i in picks:
            if info.ch_names[i] in missing:
                for q in range(6):
                    info.locs[i][q] = np.nan
    # keep dig: head-frame fiducials and the matched EEG points
    dn = List[str]()
    dp = List[List[float]]()
    for i in picks:
        if info.ch_names[i] not in missing:
            dn.append(info.ch_names[i])
            dp.append([info.locs[i][0], info.locs[i][1], info.locs[i][2]])
    info._montage = DigMontage(dn, dp, "head", h.nasion, h.lpa, h.rpa)


def _get_montage(info: Info) -> Optional[DigMontage]:
    if info._montage is None:
        return None
    return info._montage.__val__().copy()


@extend
class Info:
    def set_montage(self, montage, match_case: bool = True, match_alias: bool = False,
                    on_missing: str = "raise", verbose = None):
        _set_montage(self, montage, match_case, on_missing)
        return self

    def get_montage(self):
        return _get_montage(self)


def _legval(x: float, c: List[float]) -> float:
    """numpy.polynomial.legendre.legval (Clenshaw recursion)."""
    nd = len(c)
    c0 = c[-2]
    c1 = c[-1]
    for i in range(3, nd + 1):
        tmp = c0
        nd2 = nd - i + 2
        c0 = c[-i] - (c1 * (nd2 - 1)) / nd2
        c1 = tmp + (c1 * x * (2 * nd2 - 1)) / nd2
    return c0 + c1 * x


def _calc_g(cosang: np.ndarray[float, 2], stiffness: float = 4, n_legendre_terms: int = 50) -> np.ndarray[float, 2]:
    factors = [0.0] + [(2 * n + 1) / (n ** stiffness * (n + 1) ** stiffness * 4 * np.pi)
                       for n in range(1, n_legendre_terms + 1)]
    out = np.empty(cosang.shape)
    for i in range(cosang.shape[0]):
        for j in range(cosang.shape[1]):
            out[i, j] = _legval(cosang[i, j], factors)
    return out


def _make_interpolation_matrix(pos_from: np.ndarray[float, 2], pos_to: np.ndarray[float, 2],
                               alpha: float = 1e-5) -> np.ndarray[float, 2]:
    from scipy.linalg import pinv
    pf = pos_from.copy()
    pt = pos_to.copy()
    for a in (pf, pt):
        for i in range(a.shape[0]):
            nrm = np.linalg.norm(a[i])
            if nrm > 0:
                a[i] /= nrm
    n_from = pf.shape[0]
    n_to = pt.shape[0]
    G_from = _calc_g(pf.dot(pf.T))
    G_to_from = _calc_g(pt.dot(pf.T))
    for i in range(n_from):
        G_from[i, i] += alpha
    C = np.zeros((n_from + 1, n_from + 1))
    C[:n_from, :n_from] = G_from
    C[:n_from, n_from] = 1.0
    C[n_from, :n_from] = 1.0
    C_inv = pinv(C)
    A = np.ones((n_to, n_from + 1))
    A[:, :n_from] = G_to_from
    return A @ C_inv[:, :n_from]


def _fit_sphere(points: np.ndarray[float, 2]) -> Tuple[float, np.ndarray[float, 1]]:
    n = points.shape[0]
    A = np.ones((n, 4))
    A[:, :3] = 2 * points
    b = (points ** 2).sum(axis=1)
    x = np.linalg.lstsq(A, b, rcond=1e-6)[0]
    radius = np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3])
    return radius, x[:3].copy()


def _auto_origin(info: Info) -> np.ndarray[float, 1]:
    """fit_sphere_to_headshape(info, dig_kinds='auto') using EEG dig points
    (no 'extra' head-shape points exist in Typhon infos)."""
    if info._montage is None:
        raise RuntimeError('Cannot fit headshape without digitization, info["dig"] is None')
    m = info._montage.__val__()
    hsp = List[List[float]]()
    for p in m.pos:
        if not (p[2] < -1e-6 and p[1] > 1e-6):
            hsp.append(p)
    if len(hsp) < 4:
        raise ValueError(f"Only {len(hsp)} head digitization points of the specified kinds (\"extra\", \"eeg\",), at least 4 required")
    pts = np.array(hsp)
    r, o = _fit_sphere(pts)
    return o


def _interpolate_bads_eeg_data(info: Info, data, origin, exclude: List[str]):
    """Interpolate bad EEG channels in data[..., ch, t] in place (2-D or 3-D)."""
    picks = [i for i in range(info.nchan) if info.ch_types[i] == "eeg" and info.ch_names[i] not in exclude]
    bads = [i for i in picks if info.ch_names[i] in info.bads]
    goods = [i for i in picks if info.ch_names[i] not in info.bads]
    if len(picks) == 0 or len(bads) == 0:
        return
    o = np.zeros(3)
    if isinstance(origin, str):
        o = _auto_origin(info)
    else:
        o = np.array([float(origin[0]), float(origin[1]), float(origin[2])])
    pg = np.array([[info.locs[i][0] - o[0], info.locs[i][1] - o[1], info.locs[i][2] - o[2]] for i in goods])
    pb = np.array([[info.locs[i][0] - o[0], info.locs[i][1] - o[1], info.locs[i][2] - o[2]] for i in bads])
    interp = _make_interpolation_matrix(pg, pb)
    ga = np.array(goods, dtype=int)
    if data.ndim == 2:
        out = interp @ data[ga, :]
        for k in range(len(bads)):
            data[bads[k], :] = out[k]
    else:
        for e in range(data.shape[0]):
            out = interp @ data[e][ga, :]
            for k in range(len(bads)):
                data[e, bads[k], :] = out[k]


def _cart_to_sph_row(x: float, y: float, z: float) -> Tuple[float, float, float]:
    r = np.sqrt(x * x + y * y + z * z)
    nrm = r if r > 0 else 1.0
    az = np.arctan2(y, x)
    pol = np.arccos(z / nrm)
    if np.isnan(pol):
        pol = 0.0
    return r, az, pol


def _topomap_coords(info: Info, picks: List[int], sphere: float = HEAD_SIZE_DEFAULT) -> np.ndarray[float, 2]:
    out = np.zeros((len(picks), 2))
    for k in range(len(picks)):
        l = info.locs[picks[k]]
        x = l[0] if np.isfinite(l[0]) else 0.0
        y = l[1] if np.isfinite(l[1]) else 0.0
        z = l[2] if np.isfinite(l[2]) else 0.0
        r, az, pol = _cart_to_sph_row(x, y, z)
        # _pol_to_cart([[pol, az]]) * (r / (pi/2))  (sphere origin at 0)
        out[k, 0] = pol * np.cos(az) * r / (np.pi / 2.0)
        out[k, 1] = pol * np.sin(az) * r / (np.pi / 2.0)
    return out


def find_ch_adjacency(info: Info, ch_type: Optional[str]):
    """Channel adjacency from a Delaunay triangulation of the 2-D topomap
    positions (EEG). Returns (dense bool adjacency as csr_array, ch_names)."""
    from scipy.sparse import csr_array
    t = "eeg"
    if ch_type is None:
        types = set([info.ch_types[i] for i in range(info.nchan)])
        if len(types) != 1:
            raise ValueError("info must contain only one channel type if ch_type is None.")
        t = info.ch_types[0]
    else:
        t = ch_type.__val__()
    if t != "eeg":
        raise NotImplementedError("Typhon mne: find_ch_adjacency supports EEG only")
    picks = [i for i in range(info.nchan) if info.ch_types[i] == t]
    names = [info.ch_names[i] for i in picks]
    xy = _topomap_coords(info, picks)
    simp = _delaunay_2d(xy)
    n = len(picks)
    adj = np.eye(n)
    for tr in simp:
        for a in range(3):
            for b in range(3):
                adj[tr[a], tr[b]] = 1.0
    return csr_array(adj), names


def _circum_contains(ax: float, ay: float, bx: float, by: float, cx: float, cy: float,
                     px: float, py: float) -> bool:
    # in-circle determinant (counter-clockwise triangle a, b, c)
    adx = ax - px
    ady = ay - py
    bdx = bx - px
    bdy = by - py
    cdx = cx - px
    cdy = cy - py
    det = ((adx * adx + ady * ady) * (bdx * cdy - cdx * bdy)
           - (bdx * bdx + bdy * bdy) * (adx * cdy - cdx * ady)
           + (cdx * cdx + cdy * cdy) * (adx * bdy - bdx * ady))
    return det > 1e-30


def _delaunay_2d(xy: np.ndarray[float, 2]) -> List[List[int]]:
    """Bowyer-Watson Delaunay triangulation (stand-in for scipy.spatial.Delaunay;
    identical edge set for points in general position)."""
    n = xy.shape[0]
    px = [float(xy[i, 0]) for i in range(n)]
    py = [float(xy[i, 1]) for i in range(n)]
    mnx = min(px)
    mxx = max(px)
    mny = min(py)
    mxy = max(py)
    d = max(mxx - mnx, mxy - mny) * 20.0 + 1.0
    cx = (mnx + mxx) / 2.0
    cy = (mny + mxy) / 2.0
    px.extend([cx - d, cx + d, cx])
    py.extend([cy - d, cy - d, cy + d])
    tris = [[n, n + 1, n + 2]]
    for i in range(n):
        bad = List[int]()
        for k in range(len(tris)):
            t = tris[k]
            if _circum_contains(px[t[0]], py[t[0]], px[t[1]], py[t[1]], px[t[2]], py[t[2]], px[i], py[i]):
                bad.append(k)
        edges = List[Tuple[int, int]]()
        for k in bad:
            t = tris[k]
            for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                shared = False
                for k2 in bad:
                    if k2 == k:
                        continue
                    t2 = tris[k2]
                    if (e[0] in t2) and (e[1] in t2):
                        shared = True
                        break
                if not shared:
                    edges.append(e)
        tris = [tris[k] for k in range(len(tris)) if k not in bad]
        for e in edges:
            a = e[0]
            b = e[1]
            # keep counter-clockwise orientation
            cr = (px[b] - px[a]) * (py[i] - py[a]) - (py[b] - py[a]) * (px[i] - px[a])
            if cr > 0:
                tris.append([a, b, i])
            else:
                tris.append([b, a, i])
    return [t for t in tris if t[0] < n and t[1] < n and t[2] < n]


def _ch_neighbor_adjacency(ch_names: List[str], neighbors: List[List[str]]):
    from scipy.sparse import csr_array
    if len(ch_names) != len(neighbors):
        raise ValueError("`ch_names` and `neighbors` must have the same length")
    n = len(ch_names)
    adj = np.eye(n)
    for i in range(n):
        for c in neighbors[i]:
            if c not in ch_names:
                raise ValueError("Some of your neighbors are not present in the list of channel names")
            adj[i, ch_names.index(c)] = 1.0
    return csr_array(adj)


def read_ch_adjacency(fname, picks = None):
    """Read a FieldTrip neighbours .mat file (uncompressed MAT v5)."""
    from scipy.io import loadmat
    m = loadmat(str(fname))
    nb = m["neighbours"]
    names = List[str]()
    neigh = List[List[str]]()
    for i in range(nb.size):
        names.append(str(nb.get(i, "label")))
        cell = nb.get(i, "neighblabel")
        neigh.append([str(c) for c in cell.items])
    adj = _ch_neighbor_adjacency(names, neigh)
    return adj, names


def equalize_channels(instances, copy: bool = True, verbose = None):
    tmpl = list(instances[0].ch_names)
    common = Set[str](tmpl)
    for inst in instances:
        common = common & Set[str](list(inst.ch_names))
    order = [c for c in tmpl if c in common]
    out = [(inst.copy() if copy else inst) for inst in instances]
    for inst in out:
        if list(inst.ch_names) != order:
            inst._pick_keep([inst.info.ch_names.index(c) for c in order])
    return out


def _calc_h(cosang: np.ndarray[float, 2], stiffness: float = 4, n_legendre_terms: int = 50) -> np.ndarray[float, 2]:
    factors = [0.0] + [(2 * n + 1) / (n ** (stiffness - 1) * (n + 1) ** (stiffness - 1) * 4 * np.pi)
                       for n in range(1, n_legendre_terms + 1)]
    out = np.empty(cosang.shape)
    for i in range(cosang.shape[0]):
        for j in range(cosang.shape[1]):
            out[i, j] = _legval(cosang[i, j], factors)
    return out
