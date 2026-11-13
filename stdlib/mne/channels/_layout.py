# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# mne.channels.layout: Layout, read_layout, make_eeg_layout, make_grid_layout,
# generate_2d_layout, find_layout (Vectorview / EEG / automatic layouts).

import numpy as np
from .._info import Info
from . import _topomap_coords


def _layout_dir() -> str:
    f = __file__
    return f[:f.rfind("/") + 1] + "data/layouts/"


def _fmt82(v: float) -> str:
    s = f"{v:.2f}"
    while len(s) < 8:
        s = " " + s
    return s


class Layout:
    """Sensor layout: box, pos (n, 4: x, y, w, h), names, ids, kind."""
    box: Tuple[float, float, float, float]
    pos: np.ndarray[float, 2]
    names: List[str]
    ids: np.ndarray[int, 1]
    kind: str

    def __init__(self, box, pos, names, ids, kind):
        self.box = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        self.pos = np.asarray(pos, dtype=float)
        self.names = [str(n) for n in names]
        self.ids = np.array([int(i) for i in ids])
        self.kind = str(kind)

    def copy(self):
        return Layout(self.box, self.pos.copy(), list(self.names), self.ids.copy(), self.kind)

    def __repr__(self) -> str:
        k = len(self.names)
        if k > 3:
            k = 3
        return f"<Layout | {self.kind} - Channels: {', '.join(self.names[:k])} ...>"

    def save(self, fname, overwrite: bool = False):
        f = str(fname)
        out = ""
        if f.endswith(".lout"):
            out = (_fmt82(self.box[0]) + " " + _fmt82(self.box[1]) + " "
                   + _fmt82(self.box[2]) + " " + _fmt82(self.box[3]) + "\n")
        elif f.endswith(".lay"):
            out = ""
        else:
            raise ValueError("Unknown layout type. Should be of type .lout or .lay.")
        for i in range(self.pos.shape[0]):
            idn = f"{int(self.ids[i]):03d}"
            out += (idn + " " + _fmt82(self.pos[i, 0]) + " " + _fmt82(self.pos[i, 1]) + " "
                    + _fmt82(self.pos[i, 2]) + " " + _fmt82(self.pos[i, 3]) + " " + self.names[i] + "\n")
        with open(f, "w") as fh:
            fh.write(out)


def _read_layout_file(fname: str, is_lout: bool):
    box = (0.0, 0.0, 0.0, 0.0)
    names = List[str]()
    rows = List[List[float]]()
    ids = List[int]()
    first = True
    with open(fname) as f:
        for line in f:
            sp = line.split()
            if is_lout and first:
                first = False
                box = (float(sp[0]), float(sp[1]), float(sp[2]), float(sp[3]))
                continue
            if len(sp) == 0:
                continue
            if len(sp) == 7:
                name = sp[5] + " " + sp[6]
            else:
                name = sp[5]
            ids.append(int(sp[0]))
            rows.append([float(sp[1]), float(sp[2]), float(sp[3]), float(sp[4])])
            names.append(name)
    pos = np.zeros((len(rows), 4))
    for i in range(len(rows)):
        for j in range(4):
            pos[i, j] = rows[i][j]
    return box, pos, names, ids


def read_layout(fname = None, *, scale: bool = True) -> Layout:
    """mne.channels.read_layout (built-in names or .lout/.lay paths)."""
    if isinstance(fname, NoneType):
        raise ValueError("fname must be given")
    f = str(fname)
    path = f
    d = _layout_dir()
    import os
    for suffix in ("", ".lout", ".lay"):
        if os.path.exists(d + f + suffix) and not os.path.isdir(d + f + suffix):
            path = d + f + suffix
            break
    if not os.path.exists(path):
        raise FileNotFoundError(f"layout does not exist: {f}")
    is_lout = False
    if path.endswith(".lout"):
        is_lout = True
    elif not path.endswith(".lay"):
        raise ValueError("Invalid value for the 'fname extension' parameter. Allowed values are ['.lout', '.lay']")
    slash = path.rfind("/")
    base = path[slash + 1:]
    kind = base[:base.rfind(".")]
    box, pos, names, ids = _read_layout_file(path, is_lout)
    if scale:
        n = pos.shape[0]
        mx = pos[0, 0]
        my = pos[0, 1]
        for i in range(n):
            mx = min(mx, pos[i, 0])
            my = min(my, pos[i, 1])
        for i in range(n):
            pos[i, 0] -= mx
            pos[i, 1] -= my
        a = pos[0, 0]
        for i in range(n):
            a = max(a, pos[i, 0], pos[i, 1])
        scaling = a + pos[0, 2]
        for i in range(n):
            for j in range(4):
                pos[i, j] /= scaling
            pos[i, 0] += 0.03
            pos[i, 1] += 0.03
            pos[i, 0] *= 0.97 / 1.03
            pos[i, 1] *= 0.97 / 1.03
            pos[i, 2] *= 0.94
            pos[i, 3] *= 0.94
    return Layout(box, pos, names, ids, kind)


def _box_size(pts: np.ndarray[float, 2], width: Optional[float], height: Optional[float],
              padding: float = 0.0) -> Tuple[float, float]:
    n = pts.shape[0]
    w = 1.0
    h = 1.0
    have_w = width is not None
    have_h = height is not None
    if have_w:
        w = width.__val__()
    if have_h:
        h = height.__val__()
    if not have_w and not have_h:
        if n <= 1:
            w = 1.0
            h = 1.0
            have_w = True
            have_h = True
        else:
            best = 1e300
            bi = 0
            bj = 1
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pts[i, 0] - pts[j, 0]
                    dy = pts[i, 1] - pts[j, 1]
                    dd = np.sqrt(dx * dx + dy * dy)
                    if dd < best:
                        best = dd
                        bi = i
                        bj = j
            dxw = abs(pts[bi, 0] - pts[bj, 0])
            dyh = abs(pts[bi, 1] - pts[bj, 1])
            if dxw > dyh:
                w = dxw
                have_w = True
            else:
                h = dyh
                have_h = True
    if have_w and not have_h:
        found = False
        m = 1e300
        for i in range(n):
            for j in range(i + 1, n):
                if abs(pts[i, 0] - pts[j, 0]) < w:
                    found = True
                    m = min(m, abs(pts[i, 1] - pts[j, 1]))
        h = m if found else 1.0
    elif have_h and not have_w:
        found = False
        m = 1e300
        for i in range(n):
            for j in range(i + 1, n):
                if abs(pts[i, 1] - pts[j, 1]) < h:
                    found = True
                    m = min(m, abs(pts[i, 0] - pts[j, 0]))
        w = m if found else 1.0
    return w * (1.0 - padding), h * (1.0 - padding)


def _eeg_picks(info: Info, exclude, kind: str) -> List[int]:
    ex = List[str]()
    if isinstance(exclude, str):
        if exclude == "bads":
            ex = list(info.bads)
    else:
        for e in exclude:
            ex.append(str(e))
    out = List[int]()
    for i in range(len(info.ch_names)):
        if info.ch_types[i] == kind and info.ch_names[i] not in ex:
            out.append(i)
    return out


def make_eeg_layout(info: Info, radius: float = 0.5, width = None, height = None,
                    exclude = "bads", csd: bool = False) -> Layout:
    """mne.channels.make_eeg_layout."""
    w: Optional[float] = None
    h: Optional[float] = None
    if not isinstance(width, NoneType):
        w = float(width)
        if w.__val__() < 0.0 or w.__val__() > 1.0:
            raise ValueError("The width parameter should be between 0 and 1.")
    if not isinstance(height, NoneType):
        h = float(height)
        if h.__val__() < 0.0 or h.__val__() > 1.0:
            raise ValueError("The height parameter should be between 0 and 1.")
    picks = _eeg_picks(info, exclude, "csd" if csd else "eeg")
    loc2d = _topomap_coords(info, picks)
    names = [info.ch_names[i] for i in picks]
    n = len(picks)
    sc = 0.0
    for i in range(n):
        for j in range(2):
            sc = max(sc, -loc2d[i, j], loc2d[i, j])
    sc = sc * 2.0
    loc2d = loc2d / sc
    ww = 1.0
    hh = 1.0
    if w is None or h is None:
        ww, hh = _box_size(loc2d, w, h, 0.1)
    else:
        ww = w.__val__()
        hh = h.__val__()
    loc2d = loc2d * (2.0 * radius)
    scaling = min(1.0 / (1.0 + ww), 1.0 / (1.0 + hh))
    loc2d = loc2d * scaling
    ww *= scaling
    hh *= scaling
    loc2d = loc2d + 0.5
    pos = np.zeros((n, 4))
    ids = List[int]()
    for i in range(n):
        pos[i, 0] = loc2d[i, 0] - 0.5 * ww
        pos[i, 1] = loc2d[i, 1] - 0.5 * hh
        pos[i, 2] = ww
        pos[i, 3] = hh
        ids.append(i + 1)
    return Layout((0.0, 1.0, 0.0, 1.0), pos, names, ids, "EEG")


def make_grid_layout(info: Info, picks = None, n_col = None) -> Layout:
    """mne.channels.make_grid_layout."""
    from .._info import _picks_to_idx
    pk = _picks_to_idx(info, picks, "misc")
    names = [info.ch_names[k] for k in pk]
    size = len(pk)
    if size == 0:
        raise ValueError("No misc data channels found.")
    ids = List[int]()
    for i in range(size):
        ids.append(i)
    nc = 0
    nr = 0
    if isinstance(n_col, NoneType):
        r = np.sqrt(float(size))
        nc = int(r)
        nr = int(r)
        if r % 1.0 != 0.0:
            nc = int(r + 1.0)
            nr = int(r)
        if nc * nr < size:
            nr += 1
    else:
        nc = int(n_col)
        nr = int(np.ceil(size / float(nc)))
    xs = np.zeros(size)
    ys = np.zeros(size)
    k = 0
    for iy in range(nr):
        for ix in range(nc):
            if k < size:
                xs[k] = -0.5 + (1.0 * ix / (nc - 1) if nc > 1 else 0.0)
                ys[k] = -0.5 + (1.0 * iy / (nr - 1) if nr > 1 else 0.0)
            k += 1
    xy = np.zeros((size, 2))
    for i in range(size):
        xy[i, 0] = xs[i]
        xy[i, 1] = ys[i]
    width, height = _box_size(xy, None, None, 0.1)
    xsc = 1.0 / (1.0 + width + 0.01)
    ysc = 1.0 / (1.0 + height + 0.01)
    width *= xsc
    height *= ysc
    pos = np.zeros((size, 4))
    for i in range(size):
        pos[i, 0] = xs[i] * xsc + 0.5 - 0.5 * width
        pos[i, 1] = ys[i] * ysc + 0.5 - 0.5 * height
        pos[i, 2] = width
        pos[i, 3] = height
    return Layout((0.0, 1.0, 0.0, 1.0), pos, names, ids, "grid-misc")


def generate_2d_layout(xy, w: float = 0.07, h: float = 0.05, pad: float = 0.02,
                       ch_names = None, ch_indices = None, name: str = "ecog",
                       bg_image = None, normalize: bool = True) -> Layout:
    """mne.channels.generate_2d_layout (array bg_image not supported)."""
    a = np.asarray(xy, dtype=float)
    n = a.shape[0]
    idx = List[int]()
    if isinstance(ch_indices, NoneType):
        for i in range(n):
            idx.append(i)
    else:
        for v in ch_indices:
            idx.append(int(v))
    nm = List[str]()
    if isinstance(ch_names, NoneType):
        for v in idx:
            nm.append(str(v))
    else:
        for v in ch_names:
            nm.append(str(v))
    if len(nm) != len(idx):
        raise ValueError("# channel names and indices must be equal")
    if len(nm) != n:
        raise ValueError("# channel names and xy vals must be equal")
    pts = a.copy()
    if normalize:
        for d in range(2):
            mn = pts[0, d]
            mx = pts[0, d]
            for i in range(n):
                mn = min(mn, pts[i, d])
                mx = max(mx, pts[i, d])
            for i in range(n):
                pts[i, d] = (pts[i, d] - mn) / (mx - mn)
    bw, bh = _box_size(pts, None, None, pad)
    pos = np.zeros((n, 4))
    for i in range(n):
        pos[i, 0] = pts[i, 0]
        pos[i, 1] = pts[i, 1]
        pos[i, 2] = w
        pos[i, 3] = h
    return Layout((0.0, 0.0, bw, bh), pos, nm, idx, name)


def _clean_ws(names: List[str]) -> List[str]:
    return [n.replace(" ", "") for n in names]


def find_layout(info: Info, ch_type = None, exclude = "bads"):
    """mne.channels.find_layout (Vectorview, EEG and automatic layouts)."""
    ct = ""
    if not isinstance(ch_type, NoneType):
        ct = str(ch_type)
    n_mag = 0
    n_grad = 0
    n_eeg = 0
    for i in range(len(info.ch_names)):
        t = info.ch_types[i]
        if t == "mag":
            n_mag += 1
        elif t == "grad":
            n_grad += 1
        elif t == "eeg":
            n_eeg += 1
    has_vv_mag = n_mag > 0 and info.coil_types[info.ch_types.index("mag")] in (3024, 3022)
    has_vv_grad = n_grad > 0 and info.coil_types[info.ch_types.index("grad")] == 3012
    ex = List[str]()
    if isinstance(exclude, str):
        if exclude == "bads":
            ex = list(info.bads)
    else:
        for e in exclude:
            ex.append(str(e))
    name = ""
    if (has_vv_mag and has_vv_grad and ct == "") or ((has_vv_mag or has_vv_grad) and ct == "meg"):
        name = "Vectorview-all"
    elif (has_vv_mag and not has_vv_grad) or (has_vv_mag and has_vv_grad and ct == "mag"):
        name = "Vectorview-mag"
    elif (has_vv_grad and not has_vv_mag) or (has_vv_mag and has_vv_grad and ct == "grad"):
        if info.ch_names[0].endswith("X"):
            name = "Vectorview-grad_norm"
        else:
            name = "Vectorview-grad"
    elif n_eeg > 0 and n_mag == 0 and n_grad == 0 and (ct == "" or ct == "eeg"):
        return make_eeg_layout(info, 0.5, None, None, exclude)
    if name == "":
        pk = List[int]()
        for i in range(len(info.ch_names)):
            t = info.ch_types[i]
            if t in ("mag", "grad", "eeg", "seeg", "ecog", "dbs", "csd"):
                pk.append(i)
        xy = _topomap_coords(info, pk)
        return generate_2d_layout(xy, 0.07, 0.05, 0.02, [info.ch_names[i] for i in pk], None, "custom", None, True)
    lay = read_layout(name)
    lay.names = _clean_ws(lay.names)
    keep = List[int]()
    for i in range(len(lay.names)):
        if lay.names[i] not in ex:
            keep.append(i)
    npos = np.zeros((len(keep), 4))
    nids = List[int]()
    nn = List[str]()
    for k in range(len(keep)):
        for j in range(4):
            npos[k, j] = lay.pos[keep[k], j]
        nids.append(int(lay.ids[keep[k]]))
        nn.append(lay.names[keep[k]])
    lay.pos = npos
    lay.ids = np.array(nids)
    lay.names = nn
    return lay
