# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne.source_space subset: discrete and spherical volume source spaces
# (setup_volume_source_space without MRI), SourceSpaces container.

import numpy as np

FIFFV_COORD_MRI: int = 5
FIFFV_COORD_HEAD: int = 4


class SourceSpace:
    """One source space (dict entry of mne.SourceSpaces)."""
    type: str
    rr: np.ndarray[float, 2]
    nn: np.ndarray[float, 2]
    inuse: np.ndarray[bool, 1]
    vertno: np.ndarray[int, 1]
    nuse: int
    np: int
    coord_frame: int
    shape: Optional[Tuple[int, int, int]]

    def __init__(self, type: str, rr: np.ndarray[float, 2], nn: np.ndarray[float, 2],
                 coord_frame: int):
        self.type = type
        self.rr = rr
        self.nn = nn
        n = rr.shape[0]
        self.np = n
        self.inuse = np.ones(n, dtype=bool)
        self.vertno = np.arange(n)
        self.nuse = n
        self.coord_frame = coord_frame
        self.shape = None

    def copy(self):
        s = SourceSpace(self.type, self.rr.copy(), self.nn.copy(), self.coord_frame)
        s.inuse = self.inuse.copy()
        s.vertno = self.vertno.copy()
        s.nuse = self.nuse
        s.shape = self.shape
        return s

    def __getitem__(self, key: Literal[str]):
        if key == "type":
            return self.type
        elif key == "rr":
            return self.rr
        elif key == "nn":
            return self.nn
        elif key == "inuse":
            return self.inuse
        elif key == "vertno":
            return self.vertno
        elif key == "nuse":
            return self.nuse
        elif key == "np":
            return self.np
        elif key == "coord_frame":
            return self.coord_frame
        else:
            compile_error("unsupported source space key")


class SourceSpaces:
    """List of source spaces (mne.SourceSpaces)."""
    spaces: List[SourceSpace]

    def __init__(self, spaces: List[SourceSpace]):
        self.spaces = spaces

    def __len__(self) -> int:
        return len(self.spaces)

    def __getitem__(self, i: int) -> SourceSpace:
        return self.spaces[i]

    def __iter__(self):
        for s in self.spaces:
            yield s

    def copy(self):
        return SourceSpaces([s.copy() for s in self.spaces])

    @property
    def kind(self) -> str:
        t = self.spaces[0].type
        if t == "vol":
            return "volume"
        elif t == "surf":
            return "surface"
        return t

    def __repr__(self) -> str:
        parts = [f"{s.type}, n_used={s.nuse}" for s in self.spaces]
        return f"<SourceSpaces: [{', '.join(parts)}] {self.kind} coords>"


def _normalize_rows(nn):
    out = nn.copy()
    for i in range(out.shape[0]):
        s = np.sqrt(np.sum(out[i] * out[i]))
        if s > 0:
            out[i] /= s
    return out


def _make_discrete_source_space(rr_in, nn_in) -> SourceSpace:
    rr = np.array(rr_in, dtype=float)
    nn = np.array(nn_in, dtype=float)
    if rr.shape[0] != nn.shape[0] or rr.shape[1] != 3 or nn.shape[1] != 3 or \
            not np.isfinite(rr).all() or not np.isfinite(nn).all():
        raise RuntimeError('"rr" and "nn" must both be finite 2D arrays with the same number of rows and 3 columns')
    nn = _normalize_rows(nn)
    for i in range(nn.shape[0]):
        if np.sum(nn[i] * nn[i]) == 0:
            raise RuntimeError("sources have zero length normal")
    return SourceSpace("discrete", rr, nn, FIFFV_COORD_MRI)


def _grid_n(m: float, grid: float) -> int:
    if m > 0:
        return int(np.floor(abs(m) / grid)) + 1
    return -int(np.floor(abs(m) / grid)) - 1


def _filter_sphere(s: SourceSpace, center, inner_r: Optional[float], limit_mm: float):
    """_filter_source_spaces with a _CheckInsideSphere (rr in sphere frame)."""
    vert = [i for i in range(s.np) if s.inuse[i]]
    if inner_r is None:
        return
    r = inner_r.__val__()
    omit = 0
    for v in vert:
        d = s.rr[v] - center
        dist = np.sqrt(np.sum(d * d))
        inside = dist <= r
        if inside and limit_mm > 0.0:
            if dist >= r - limit_mm / 1000.0:
                if abs(dist - r) < limit_mm / 1000.0:
                    inside = False
        if not inside:
            s.inuse[v] = False
            omit += 1
    s.nuse -= omit
    s.vertno = np.array([i for i in range(s.np) if s.inuse[i]], dtype=int)


def setup_volume_source_space(subject = None, pos = 5.0, mri = None, sphere = None, bem = None,
                              surface = None, mindist: float = 5.0, exclude: float = 0.0,
                              subjects_dir = None, volume_label = None,
                              add_interpolator: bool = True, sphere_units: str = "m",
                              single_volume: bool = False, *, n_jobs = None,
                              verbose = None) -> SourceSpaces:
    """mne.setup_volume_source_space: discrete (pos=dict) or spherical grid."""
    from .bem import _check_sphere
    if not isinstance(mri, NoneType) or not isinstance(bem, NoneType) or \
            not isinstance(surface, NoneType) or not isinstance(volume_label, NoneType):
        compile_error("setup_volume_source_space: only sphere-bounded or discrete source spaces are supported in Typhon mne")
    if isinstance(pos, Dict):
        if "rr" not in pos or "nn" not in pos:
            raise KeyError('pos, if dict, must contain "rr" and "nn"')
        return SourceSpaces([_make_discrete_source_space(pos["rr"], pos["nn"])])
    sph = _check_sphere(sphere)
    if sphere_units == "mm":
        sph = sph / 1000.0
    grid = float(pos) / 1000.0
    excl = exclude / 1000.0
    cm = sph[:3].copy()
    radius = float(sph[3])
    mins = cm - radius
    maxs = cm + radius
    maxn = [_grid_n(float(maxs[i]), grid) for i in range(3)]
    minn = [_grid_n(float(mins[i]), grid) for i in range(3)]
    ns = (maxn[0] - minn[0] + 1, maxn[1] - minn[1] + 1, maxn[2] - minn[2] + 1)
    npts = ns[0] * ns[1] * ns[2]
    rr = np.empty((npts, 3))
    k = 0
    for z in range(minn[2], maxn[2] + 1):
        for y in range(minn[1], maxn[1] + 1):
            for x in range(minn[0], maxn[0] + 1):
                rr[k, 0] = x * grid
                rr[k, 1] = y * grid
                rr[k, 2] = z * grid
                k += 1
    nn = np.zeros((npts, 3))
    nn[:, 2] = 1.0
    sp = SourceSpace("vol", rr, nn, FIFFV_COORD_MRI)
    sp.shape = ns
    for i in range(npts):
        d = rr[i] - cm
        dist = np.sqrt(np.sum(d * d))
        if dist < excl or dist > radius:
            sp.inuse[i] = False
            sp.nuse -= 1
    _filter_sphere(sp, cm, radius, mindist)
    # no MRI -> 'discrete'
    sp.type = "discrete"
    sp.shape = None
    return SourceSpaces([sp])
