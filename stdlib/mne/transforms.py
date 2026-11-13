# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# mne.transforms subset: rotation/translation/scaling matrices, apply_trans,
# spherical conversions.

import numpy as np


def rotation3d(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray[float, 2]:
    cx = np.cos(x)
    cy = np.cos(y)
    cz = np.cos(z)
    sx = np.sin(x)
    sy = np.sin(y)
    sz = np.sin(z)
    r = np.zeros((3, 3))
    r[0, 0] = cy * cz
    r[0, 1] = -cx * sz + sx * sy * cz
    r[0, 2] = sx * sz + cx * sy * cz
    r[1, 0] = cy * sz
    r[1, 1] = cx * cz + sx * sy * sz
    r[1, 2] = -sx * cz + cx * sy * sz
    r[2, 0] = -sy
    r[2, 1] = sx * cy
    r[2, 2] = cx * cy
    return r


def rotation(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray[float, 2]:
    r = np.eye(4)
    r3 = rotation3d(x, y, z)
    for i in range(3):
        for j in range(3):
            r[i, j] = r3[i, j]
    return r


def translation(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray[float, 2]:
    m = np.eye(4)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def scaling(x: float = 1.0, y: float = 1.0, z: float = 1.0) -> np.ndarray[float, 2]:
    m = np.eye(4)
    m[0, 0] = x
    m[1, 1] = y
    m[2, 2] = z
    return m


def rotation_angles(m) -> Tuple[float, float, float]:
    x = np.arctan2(m[2, 1], m[2, 2])
    c2 = np.sqrt(m[0, 0] ** 2 + m[1, 0] ** 2)
    y = np.arctan2(-m[2, 0], c2)
    s1 = np.sin(x)
    c1 = np.cos(x)
    z = np.arctan2(s1 * m[0, 2] - c1 * m[0, 1], c1 * m[1, 1] - s1 * m[1, 2])
    return x, y, z


def apply_trans(trans, pts, move: bool = True):
    """Apply a 4x4 transform to a point (3,) or points (n, 3)."""
    t = np.asarray(trans)
    p = np.asarray(pts)
    if p.size == 0:
        return p.copy()
    out = np.dot(p, t[:3, :3].T)
    if move:
        out = out + t[:3, 3]
    return out


def _cart_to_sph(cart) -> np.ndarray[float, 2]:
    c = np.atleast_2d(np.asarray(cart))
    out = np.zeros((c.shape[0], 3))
    for i in range(c.shape[0]):
        r = np.sqrt(c[i, 0] ** 2 + c[i, 1] ** 2 + c[i, 2] ** 2)
        nrm = r if r > 0 else 1.0
        out[i, 0] = r
        out[i, 1] = np.arctan2(c[i, 1], c[i, 0])
        out[i, 2] = np.arccos(c[i, 2] / nrm)
    return np.nan_to_num(out)


def _sph_to_cart(sph) -> np.ndarray[float, 2]:
    s = np.atleast_2d(np.asarray(sph))
    out = np.zeros((s.shape[0], 3))
    for i in range(s.shape[0]):
        out[i, 2] = s[i, 0] * np.cos(s[i, 2])
        xy = s[i, 0] * np.sin(s[i, 2])
        out[i, 0] = xy * np.cos(s[i, 1])
        out[i, 1] = xy * np.sin(s[i, 1])
    return out


# ---------------------------------------------------------------------------
# Transform class and FIF trans io (mne.transforms.Transform / read_trans / write_trans)

_FRAME_NAMES: List[str] = ["unknown", "meg", "isotrak", "hpi", "head", "mri"]


def _frame_to_const(name: str) -> int:
    if name == "unknown":
        return 0
    if name == "meg":
        return 1
    if name == "isotrak":
        return 2
    if name == "hpi":
        return 3
    if name == "head":
        return 4
    if name == "mri":
        return 5
    if name == "mri_voxel":
        return 2001
    if name == "mni_tal":
        return 2002
    if name == "ras":
        return 2003
    if name == "fs_tal":
        return 2004
    if name == "ctf_head":
        return 2005
    if name == "ctf_meg":
        return 2006
    raise KeyError(name)


def _frame_to_name(c: int) -> str:
    names = {0: "unknown", 1: "meg", 2: "isotrak", 3: "hpi", 4: "head", 5: "mri", 2001: "mri_voxel",
             2002: "mni_tal", 2003: "ras", 2004: "fs_tal", 2005: "ctf_head", 2006: "ctf_meg"}
    return names[c]


class Transform:
    """mne.transforms.Transform: 4x4 matrix between two coordinate frames."""
    fro: int
    to: int
    trans: np.ndarray[float, 2]

    def __init__(self, fro, to, trans = None):
        if isinstance(fro, str):
            self.fro = _frame_to_const(fro)
        else:
            self.fro = int(fro)
        if isinstance(to, str):
            self.to = _frame_to_const(to)
        else:
            self.to = int(to)
        if isinstance(trans, NoneType):
            self.trans = np.eye(4)
        else:
            t = np.asarray(trans, dtype=float)
            if t.shape[0] != 4 or t.shape[1] != 4:
                raise ValueError(f"Transformation must be shape (4, 4) rather than {t.shape}")
            self.trans = t.copy()

    def __getitem__(self, key: Literal[str]):
        if key == "from":
            return self.fro
        elif key == "to":
            return self.to
        elif key == "trans":
            return self.trans
        else:
            compile_error("unsupported Transform key")

    def copy(self):
        return Transform(self.fro, self.to, self.trans)

    def __eq__(self, other: Transform) -> bool:
        return self.fro == other.fro and self.to == other.to and bool((self.trans == other.trans).all())

    def __repr__(self) -> str:
        return f"<Transform | {_frame_to_name(self.fro)}->{_frame_to_name(self.to)}>"


def invert_transform(trans: Transform) -> Transform:
    return Transform(trans.to, trans.fro, np.linalg.inv(trans.trans))


def combine_transforms(t_first: Transform, t_second: Transform, fro, to) -> Transform:
    """mne.transforms.combine_transforms (t_second @ t_first)."""
    f = _frame_to_const(fro) if isinstance(fro, str) else int(fro)
    t = _frame_to_const(to) if isinstance(to, str) else int(to)
    if t_first.fro != f:
        raise RuntimeError(f"From mismatch: {t_first.fro} != {f}")
    if t_first.to != t_second.fro:
        raise RuntimeError(f"Transform mismatch: t1[\"to\"] = {t_first.to}, t2[\"from\"] = {t_second.fro}")
    if t_second.to != t:
        raise RuntimeError(f"To mismatch: {t_second.to} != {t}")
    return Transform(f, t, np.dot(t_second.trans, t_first.trans))


def _read_transforms(fname: str) -> List[Transform]:
    from .io._fifext import _find_all_tags, FIFF_COORD_TRANS
    from .io._fiff import FifFile
    from .io._bytes import _i32_be, _f32_be
    f = FifFile(fname)
    out = List[Transform]()
    for t in _find_all_tags(f.tree, FIFF_COORD_TRANS):
        p = t.pos + 16
        fr = _i32_be(f.b, p)
        to = _i32_be(f.b, p + 4)
        m = np.eye(4)
        for i in range(3):
            for j in range(3):
                m[i, j] = _f32_be(f.b, p + 8 + 4 * (3 * i + j))
            m[i, 3] = _f32_be(f.b, p + 8 + 36 + 4 * i)
        out.append(Transform(fr, to, m))
    return out


def read_trans(fname, return_all: bool = False, verbose = None):
    """mne.read_trans: first transformation in a -trans.fif file (or all of them)."""
    ts = _read_transforms(str(fname))
    if len(ts) == 0:
        raise OSError("This does not seem to be a -trans.fif file.")
    return ts[0]


def write_trans(fname, trans: Transform, *, overwrite: bool = False, verbose = None):
    """mne.write_trans."""
    from .io._fiff import FifWriter
    w = FifWriter()
    w.write_coord_trans(trans.fro, trans.to, trans.trans)
    w.end_file(str(fname))
