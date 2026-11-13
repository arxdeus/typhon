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
