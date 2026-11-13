# Ported from scipy 1.18.1 (BSD-3-Clause): scipy/spatial/transform/
# _rotation.py and the numpy backend _rotation_cy.pyx.
#
# Rotation is generic over a static `S` flag (single rotation vs stack of
# rotations) so methods return scalars/1-D arrays for single rotations like
# scipy.  Only 0-D/1-D rotation shapes are supported (no N-D stacks).

import numpy as np
from math import sqrt, sin, cos, atan2, hypot, pi, isnan, fabs

def _norm3(a, i: int) -> float:
    return sqrt(a[i, 0] * a[i, 0] + a[i, 1] * a[i, 1] + a[i, 2] * a[i, 2])

def _canon1(q, i: int):
    if (q[i, 3] < 0 or (q[i, 3] == 0 and q[i, 0] < 0)
            or (q[i, 3] == 0 and q[i, 0] == 0 and q[i, 1] < 0)
            or (q[i, 3] == 0 and q[i, 0] == 0 and q[i, 1] == 0 and q[i, 2] < 0)):
        for k in range(4):
            q[i, k] = -q[i, k]

def _compose1(p, i: int, q, j: int, r, k: int):
    c0 = p[i, 1] * q[j, 2] - p[i, 2] * q[j, 1]
    c1 = p[i, 2] * q[j, 0] - p[i, 0] * q[j, 2]
    c2 = p[i, 0] * q[j, 1] - p[i, 1] * q[j, 0]
    r0 = p[i, 3] * q[j, 0] + q[j, 3] * p[i, 0] + c0
    r1 = p[i, 3] * q[j, 1] + q[j, 3] * p[i, 1] + c1
    r2 = p[i, 3] * q[j, 2] + q[j, 3] * p[i, 2] + c2
    r3 = p[i, 3] * q[j, 3] - p[i, 0] * q[j, 0] - p[i, 1] * q[j, 1] - p[i, 2] * q[j, 2]
    r[k, 0] = r0
    r[k, 1] = r1
    r[k, 2] = r2
    r[k, 3] = r3

def compose_quat(p, q):
    n = q.shape[0] if p.shape[0] == 1 else p.shape[0]
    if not (p.shape[0] == 1 or q.shape[0] == 1 or p.shape[0] == q.shape[0]):
        raise ValueError(f"Expected equal number of rotations in both or a single rotation in either object, got {p.shape[0]} rotations in first and {q.shape[0]} rotations in second object.")
    r = np.empty((n, 4))
    for k in range(n):
        _compose1(p, 0 if p.shape[0] == 1 else k, q, 0 if q.shape[0] == 1 else k, r, k)
    return r

def _normalize_rows(q):
    for i in range(q.shape[0]):
        nn = sqrt(q[i, 0] ** 2 + q[i, 1] ** 2 + q[i, 2] ** 2 + q[i, 3] ** 2)
        if nn == 0:
            raise ValueError("Found zero norm quaternions in `quat`.")
        for k in range(4):
            q[i, k] /= nn

def _axis_index(c: str) -> int:
    if c == 'x':
        return 0
    if c == 'y':
        return 1
    if c == 'z':
        return 2
    raise ValueError(f"Expected axis to be from ['x', 'y', 'z'], got {c}")

def _check_seq(seq: str, maxlen: int, exact: bool):
    n = len(seq)
    if exact:
        if n != 3:
            raise ValueError(f"Expected 3 axes, got {seq}.")
    elif n < 1 or n > maxlen:
        raise ValueError(f"Expected axis specification to be a non-empty string of upto 3 characters, got {seq}")
    up = True
    lo = True
    for ch in seq:
        if ch not in 'XYZ':
            up = False
        if ch not in 'xyz':
            lo = False
    if not (up or lo):
        raise ValueError(f"Expected axes from `seq` to be from ['x', 'y', 'z'] or ['X', 'Y', 'Z'], got {seq}")
    for i in range(n - 1):
        if seq[i] == seq[i + 1]:
            raise ValueError(f"Expected consecutive axes to be different, got {seq}")
    return up

def _from_matrix_orthogonal(m):
    n = m.shape[0]
    q = np.empty((n, 4))
    dec = [0.0, 0.0, 0.0, 0.0]
    for ind in range(n):
        dec[0] = m[ind, 0, 0]
        dec[1] = m[ind, 1, 1]
        dec[2] = m[ind, 2, 2]
        dec[3] = m[ind, 0, 0] + m[ind, 1, 1] + m[ind, 2, 2]
        choice = 0
        vmax = dec[0]
        for i in range(1, 4):
            if dec[i] > vmax:
                choice = i
                vmax = dec[i]
        if choice != 3:
            i = choice
            j = (i + 1) % 3
            k = (j + 1) % 3
            q[ind, i] = 1 - dec[3] + 2 * m[ind, i, i]
            q[ind, j] = m[ind, j, i] + m[ind, i, j]
            q[ind, k] = m[ind, k, i] + m[ind, i, k]
            q[ind, 3] = m[ind, k, j] - m[ind, j, k]
        else:
            q[ind, 0] = m[ind, 2, 1] - m[ind, 1, 2]
            q[ind, 1] = m[ind, 0, 2] - m[ind, 2, 0]
            q[ind, 2] = m[ind, 1, 0] - m[ind, 0, 1]
            q[ind, 3] = 1 + dec[3]
        nn = sqrt(q[ind, 0] ** 2 + q[ind, 1] ** 2 + q[ind, 2] ** 2 + q[ind, 3] ** 2)
        for c in range(4):
            q[ind, c] /= nn
    return q

def _from_rotvec(rv):
    n = rv.shape[0]
    q = np.empty((n, 4))
    for ind in range(n):
        angle = _norm3(rv, ind)
        scale = 0.0
        if angle <= 1e-3:
            a2 = angle * angle
            scale = 0.5 - a2 / 48 + a2 * a2 / 3840
        else:
            scale = sin(angle / 2) / angle
        q[ind, 0] = scale * rv[ind, 0]
        q[ind, 1] = scale * rv[ind, 1]
        q[ind, 2] = scale * rv[ind, 2]
        q[ind, 3] = cos(angle / 2)
    return q

def _as_rotvec(quat):
    n = quat.shape[0]
    out = np.empty((n, 3))
    qs = np.empty((1, 4))
    for ind in range(n):
        for k in range(4):
            qs[0, k] = quat[ind, k]
        _canon1(qs, 0)
        angle = 2 * atan2(_norm3(qs, 0), qs[0, 3])
        scale = 0.0
        if angle <= 1e-3:
            a2 = angle * angle
            scale = 2 + a2 / 12 + 7 * a2 * a2 / 2880
        else:
            scale = angle / sin(angle / 2)
        for k in range(3):
            out[ind, k] = scale * qs[0, k]
    return out

def _as_matrix(quat):
    n = quat.shape[0]
    m = np.empty((n, 3, 3))
    for ind in range(n):
        x = quat[ind, 0]
        y = quat[ind, 1]
        z = quat[ind, 2]
        w = quat[ind, 3]
        x2 = x * x
        y2 = y * y
        z2 = z * z
        w2 = w * w
        xy = x * y
        zw = z * w
        xz = x * z
        yw = y * w
        yz = y * z
        xw = x * w
        m[ind, 0, 0] = x2 - y2 - z2 + w2
        m[ind, 1, 0] = 2 * (xy + zw)
        m[ind, 2, 0] = 2 * (xz - yw)
        m[ind, 0, 1] = 2 * (xy - zw)
        m[ind, 1, 1] = -x2 + y2 - z2 + w2
        m[ind, 2, 1] = 2 * (yz + xw)
        m[ind, 0, 2] = 2 * (xz + yw)
        m[ind, 1, 2] = 2 * (yz - xw)
        m[ind, 2, 2] = -x2 - y2 + z2 + w2
    return m

def _get_angles(angles, ind: int, extrinsic: bool, symmetric: bool, sign: int,
                lamb: float, a: float, b: float, c: float, d: float) -> int:
    af = 0 if extrinsic else 2
    at = 2 if extrinsic else 0
    angles[ind, 1] = 2 * atan2(hypot(c, d), hypot(a, b))
    case = 0
    if abs(angles[ind, 1]) <= 1e-7:
        case = 1
    elif abs(angles[ind, 1] - pi) <= 1e-7:
        case = 2
    half_sum = atan2(b, a)
    half_diff = atan2(d, c)
    if case == 0:
        angles[ind, af] = half_sum - half_diff
        angles[ind, at] = half_sum + half_diff
    else:
        angles[ind, 2] = 0.0
        if case == 1:
            angles[ind, 0] = 2 * half_sum
        else:
            angles[ind, 0] = 2 * half_diff * (-1.0 if extrinsic else 1.0)
    if not symmetric:
        angles[ind, at] *= float(sign)
        angles[ind, 1] -= lamb
    for idx in range(3):
        if angles[ind, idx] < -pi:
            angles[ind, idx] += 2 * pi
        elif angles[ind, idx] > pi:
            angles[ind, idx] -= 2 * pi
    return 1 if case != 0 else 0

def _from_euler(seq: str, angles2):
    intrinsic = _check_seq(seq, 3, False)
    s = seq.lower()
    n = angles2.shape[0]
    q = np.zeros((n, 4))
    ax0 = _axis_index(s[0])
    for ind in range(n):
        q[ind, 3] = cos(angles2[ind, 0] / 2)
        q[ind, ax0] = sin(angles2[ind, 0] / 2)
    for idx in range(1, len(s)):
        e = np.zeros((n, 4))
        ax = _axis_index(s[idx])
        for ind in range(n):
            e[ind, 3] = cos(angles2[ind, idx] / 2)
            e[ind, ax] = sin(angles2[ind, idx] / 2)
        q = compose_quat(q, e) if intrinsic else compose_quat(e, q)
    return q

def _as_euler(quat, seq: str):
    intrinsic = _check_seq(seq, 3, True)
    extrinsic = not intrinsic
    s = seq.lower()
    if not extrinsic:
        s = s[::-1]
    i = _axis_index(s[0])
    j = _axis_index(s[1])
    k = _axis_index(s[2])
    symmetric = i == k
    if symmetric:
        k = 3 - i - j
    sign = (i - j) * (j - k) * (k - i) // 2
    n = quat.shape[0]
    angles = np.empty((n, 3))
    for ind in range(n):
        a = 0.0
        b = 0.0
        c = 0.0
        d = 0.0
        if symmetric:
            a = quat[ind, 3]
            b = quat[ind, i]
            c = quat[ind, j]
            d = quat[ind, k] * sign
        else:
            a = quat[ind, 3] - quat[ind, j]
            b = quat[ind, i] + quat[ind, k] * sign
            c = quat[ind, j] + quat[ind, 3]
            d = quat[ind, k] * sign - quat[ind, i]
        _get_angles(angles, ind, extrinsic, symmetric, sign, pi / 2, a, b, c, d)
    return angles

def _cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])

def _order_extrinsic(order: str) -> bool:
    if order == 'e' or order == 'extrinsic':
        return True
    if order == 'i' or order == 'intrinsic':
        return False
    raise ValueError(f"order should be 'e'/'extrinsic' for extrinsic sequences or 'i'/'intrinsic' for intrinsic sequences, got {order}")

def _from_davenport(axes, order: str, angles2):
    extrinsic = _order_extrinsic(order)
    ax = np.asarray(axes, dtype=float)
    A = ax.reshape((-1, 3)).copy()
    na = A.shape[0]
    if na < 1 or na > 3:
        raise ValueError(f"Expected up to 3 axes, got {na}")
    for i in range(na):
        nn = sqrt(A[i, 0] ** 2 + A[i, 1] ** 2 + A[i, 2] ** 2)
        for k in range(3):
            A[i, k] /= nn
    if (na > 1 and abs(float(np.dot(A[0], A[1]))) >= 1e-7) or (na > 2 and abs(float(np.dot(A[1], A[2]))) >= 1e-7):
        raise ValueError("Consecutive axes must be orthogonal.")
    if angles2.shape[1] != na:
        raise ValueError(f"Expected `angles` to match number of axes, got {angles2.shape[1]} angles and {na} axes.")
    n = angles2.shape[0]
    q = np.zeros((n, 4))
    for i in range(n):
        q[i, 3] = 1.0
    for i in range(na):
        rv = np.empty((n, 3))
        for r in range(n):
            for k in range(3):
                rv[r, k] = angles2[r, i] * A[i, k]
        qi = _from_rotvec(rv)
        q = compose_quat(qi, q) if extrinsic else compose_quat(q, qi)
    return q

def _as_davenport(quat, axes, order: str):
    extrinsic = _order_extrinsic(order)
    A = np.asarray(axes, dtype=float)
    if A.shape[0] != 3:
        raise ValueError(f"Expected 3 axes, got {A.shape[0]}.")
    n1 = A[0] / np.linalg.norm(A[0])
    n2 = A[1] / np.linalg.norm(A[1])
    n3 = A[2] / np.linalg.norm(A[2])
    if float(np.dot(n1, n2)) >= 1e-7 or float(np.dot(n2, n3)) >= 1e-7:
        raise ValueError("Consecutive axes must be orthogonal.")
    if not extrinsic:
        t = n1
        n1 = n3
        n3 = t
    nc = _cross(n1, n2)
    lamb = atan2(float(np.dot(n3, nc)), float(np.dot(n3, n1)))
    correct = False
    if lamb < 0:
        n2 = -n2
        lamb = -lamb
        nc = -nc
        correct = True
    ql = np.array([[sin(lamb / 2) * n2[0], sin(lamb / 2) * n2[1], sin(lamb / 2) * n2[2], cos(lamb / 2)]])
    n = quat.shape[0]
    angles = np.empty((n, 3))
    qt = np.empty((1, 4))
    for ind in range(n):
        _compose1(ql, 0, quat, ind, qt, 0)
        a = qt[0, 3]
        b = qt[0, 0] * n1[0] + qt[0, 1] * n1[1] + qt[0, 2] * n1[2]
        c = qt[0, 0] * n2[0] + qt[0, 1] * n2[1] + qt[0, 2] * n2[2]
        d = qt[0, 0] * nc[0] + qt[0, 1] * nc[1] + qt[0, 2] * nc[2]
        _get_angles(angles, ind, extrinsic, False, 1, lamb, a, b, c, d)
        if correct:
            angles[ind, 1] = -angles[ind, 1]
    return angles

def _as2(x, ncol: int):
    a = np.asarray(x, dtype=float)
    return np.ascontiguousarray(a).reshape((-1, ncol)).copy()

class Rotation[S: Literal[bool]]:
    _quat: np.ndarray[float, 2]

    def __init__(self, quat: np.ndarray[float, 2]):
        self._quat = quat

    @staticmethod
    def _wrap(q2, single: Literal[bool]):
        return Rotation[single](q2)

    # ---------------------------------------------------------------- ctors
    @staticmethod
    def from_quat(quat, normalize: bool = True, copy: bool = True, scalar_first: bool = False):
        a = np.asarray(quat, dtype=float)
        if a.shape[a.ndim - 1] != 4:
            raise ValueError(f"Expected `quat` to have shape (..., 4), got {a.shape}.")
        q = _as2(a, 4)
        if scalar_first:
            for i in range(q.shape[0]):
                w = q[i, 0]
                q[i, 0] = q[i, 1]
                q[i, 1] = q[i, 2]
                q[i, 2] = q[i, 3]
                q[i, 3] = w
        if normalize:
            _normalize_rows(q)
        if a.ndim == 1:
            return Rotation[True](q)
        else:
            return Rotation[False](q)

    @staticmethod
    def from_matrix(matrix, assume_valid: bool = False):
        a = np.asarray(matrix, dtype=float)
        m = np.ascontiguousarray(a).reshape((-1, 3, 3)).copy()
        if not assume_valid:
            for i in range(m.shape[0]):
                if np.linalg.det(m[i]) <= 0:
                    raise ValueError(f"Non-positive determinant (left-handed or null coordinate frame) in rotation matrix {i}: {m[i]}.")
                g = m[i] @ m[i].T
                ortho = True
                for r in range(3):
                    for c in range(3):
                        e = 1.0 if r == c else 0.0
                        if abs(g[r, c] - e) > 1e-12 + 1e-5 * abs(e):
                            ortho = False
                if not ortho:
                    U, sv, Vt = np.linalg.svd(m[i])
                    m[i] = U @ Vt
        q = _from_matrix_orthogonal(m)
        if a.ndim == 2:
            return Rotation[True](q)
        else:
            return Rotation[False](q)

    @staticmethod
    def from_rotvec(rotvec, degrees: bool = False):
        a = np.asarray(rotvec, dtype=float)
        rv = _as2(a, 3)
        if degrees:
            rv = rv * (pi / 180.0)
        q = _from_rotvec(rv)
        if a.ndim == 1:
            return Rotation[True](q)
        else:
            return Rotation[False](q)

    @staticmethod
    def from_mrp(mrp):
        a = np.asarray(mrp, dtype=float)
        m = _as2(a, 3)
        n = m.shape[0]
        q = np.empty((n, 4))
        for i in range(n):
            s = 1 + m[i, 0] ** 2 + m[i, 1] ** 2 + m[i, 2] ** 2
            q[i, 0] = 2 * m[i, 0] / s
            q[i, 1] = 2 * m[i, 1] / s
            q[i, 2] = 2 * m[i, 2] / s
            q[i, 3] = (2 - s) / s
        if a.ndim == 1:
            return Rotation[True](q)
        else:
            return Rotation[False](q)

    @staticmethod
    def from_euler(seq: str, angles, degrees: bool = False):
        a = np.asarray(angles, dtype=float)
        n = len(seq)
        if a.ndim == 0:
            ang = a.reshape((1, 1)).copy()
        else:
            ang = np.ascontiguousarray(a).reshape((-1, a.shape[a.ndim - 1])).copy()
        if degrees:
            ang = ang * (pi / 180.0)
        if ang.shape[1] != n:
            raise ValueError(f"Expected last dimension of `angles` to match number of sequence axes specified, got {ang.shape[1]}.")
        q = _from_euler(seq, ang)
        if a.ndim < 2:
            return Rotation[True](q)
        else:
            return Rotation[False](q)

    @staticmethod
    def from_davenport(axes, order: str, angles, degrees: bool = False):
        a = np.asarray(angles, dtype=float)
        if a.ndim == 0:
            ang = a.reshape((1, 1)).copy()
        else:
            ang = np.ascontiguousarray(a).reshape((-1, a.shape[a.ndim - 1])).copy()
        if degrees:
            ang = ang * (pi / 180.0)
        q = _from_davenport(axes, order, ang)
        if a.ndim < 2:
            return Rotation[True](q)
        else:
            return Rotation[False](q)

    @staticmethod
    def identity(num = None):
        if num is None:
            q = np.zeros((1, 4))
            q[0, 3] = 1.0
            return Rotation[True](q)
        else:
            q = np.zeros((num, 4))
            for i in range(num):
                q[i, 3] = 1.0
            return Rotation[False](q)

    @staticmethod
    def random(num = None, rng = None):
        g = np.random.default_rng(rng)
        if num is None:
            s = g.normal(size=4)
            q = s.reshape((1, 4)).copy()
            _normalize_rows(q)
            return Rotation[True](q)
        else:
            q = g.normal(size=(num, 4))
            _normalize_rows(q)
            return Rotation[False](q)

    @staticmethod
    def concatenate(rotations):
        qs = List[np.ndarray[float, 2]]()
        for r in rotations:
            qs.append(r._quat)
        return Rotation[False](np.concatenate(qs, axis=0))

    # --------------------------------------------------------------- output
    def _out(self, a):
        if S:
            return a[0].copy()
        else:
            return a

    def as_quat(self, canonical: bool = False, scalar_first: bool = False):
        q = self._quat.copy()
        if canonical:
            for i in range(q.shape[0]):
                _canon1(q, i)
        if scalar_first:
            for i in range(q.shape[0]):
                w = q[i, 3]
                q[i, 3] = q[i, 2]
                q[i, 2] = q[i, 1]
                q[i, 1] = q[i, 0]
                q[i, 0] = w
        return self._out(q)

    def as_matrix(self):
        return self._out(_as_matrix(self._quat))

    def as_rotvec(self, degrees: bool = False):
        r = _as_rotvec(self._quat)
        if degrees:
            r = r * (180.0 / pi)
        return self._out(r)

    def as_mrp(self):
        n = self._quat.shape[0]
        out = np.empty((n, 3))
        for i in range(n):
            sign = -1.0 if self._quat[i, 3] < 0 else 1.0
            den = 1 + sign * self._quat[i, 3]
            for k in range(3):
                out[i, k] = sign * self._quat[i, k] / den
        return self._out(out)

    def as_euler(self, seq: str, degrees: bool = False, suppress_warnings: bool = False):
        a = _as_euler(self._quat, seq)
        if degrees:
            a = a * (180.0 / pi)
        return self._out(a)

    def as_davenport(self, axes, order: str, degrees: bool = False, suppress_warnings: bool = False):
        a = _as_davenport(self._quat, axes, order)
        if degrees:
            a = a * (180.0 / pi)
        return self._out(a)

    # ------------------------------------------------------------ operators
    @property
    def single(self) -> bool:
        return S

    def __len__(self) -> int:
        if S:
            raise TypeError("Single rotation has no len().")
        return self._quat.shape[0]

    def __getitem__(self, idx):
        if S:
            compile_error("Single rotation is not subscriptable.")
        if isinstance(idx, int):
            n = self._quat.shape[0]
            i = idx + n if idx < 0 else idx
            if i < 0 or i >= n:
                raise IndexError(f"index {idx} is out of bounds for axis 0 with size {n}")
            return Rotation[True](self._quat[i:i + 1].copy())
        else:
            q = self._quat[idx]
            return Rotation[False](np.ascontiguousarray(q).reshape((-1, 4)).copy())

    def apply(self, vectors, inverse: bool = False):
        v = np.asarray(vectors, dtype=float)
        V = _as2(v, 3)
        m = _as_matrix(self._quat)
        nr = m.shape[0]
        nv = V.shape[0]
        if nv != 1 and nr != 1 and nv != nr:
            raise ValueError(f"Cannot broadcast {nr} rotations to {nv} vectors.")
        n = max(nr, nv)
        out = np.empty((n, 3))
        for i in range(n):
            ri = 0 if nr == 1 else i
            vi = 0 if nv == 1 else i
            for a in range(3):
                s = 0.0
                for b in range(3):
                    s += (m[ri, b, a] if inverse else m[ri, a, b]) * V[vi, b]
                out[i, a] = s
        if S and v.ndim == 1:
            return out[0].copy()
        else:
            return out

    def __mul__(self, other: Rotation):
        q = compose_quat(self._quat, other._quat)
        _normalize_rows(q)
        if S and other.S:
            return Rotation[True](q)
        else:
            return Rotation[False](q)

    def __pow__(self, n: float):
        q = self._quat
        r = np.zeros((1, 1))
        if n == 0:
            r = np.zeros((q.shape[0], 4))
            for i in range(q.shape[0]):
                r[i, 3] = 1.0
        elif n == -1:
            r = self.inv()._quat
        elif n == 1:
            r = q.copy()
        else:
            r = _from_rotvec(n * _as_rotvec(q))
        return Rotation[S](r)

    def inv(self):
        q = self._quat.copy()
        for i in range(q.shape[0]):
            for k in range(3):
                q[i, k] = -q[i, k]
        return Rotation[S](q)

    def magnitude(self):
        n = self._quat.shape[0]
        out = np.empty(n)
        for i in range(n):
            out[i] = 2 * atan2(_norm3(self._quat, i), abs(self._quat[i, 3]))
        if S:
            return out[0]
        else:
            return out

    def approx_equal(self, other: Rotation, atol: Optional[float] = None, degrees: bool = False):
        tol = 1e-8 if atol is None else (atol * pi / 180.0 if degrees else atol)
        r = other * self.inv()
        m = r.magnitude()
        if S and other.S:
            return m < tol
        else:
            out = np.empty(np.atleast_1d(m).size, dtype=bool)
            mm = np.atleast_1d(m)
            for i in range(out.size):
                out[i] = mm[i] < tol
            return out

    def mean(self, weights = None):
        q = self._quat
        n = q.shape[0]
        if n == 0:
            raise ValueError("Mean of an empty rotation set is undefined.")
        w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
        for i in range(w.size):
            if w[i] < 0:
                raise ValueError("`weights` must be non-negative.")
        K = np.dot(w * q.T, q)
        evals, v = np.linalg.eigh(K)
        r = v[:, 3].copy().reshape((1, 4))
        return Rotation[True](r)

    @staticmethod
    def align_vectors(a, b, weights = None, return_sensitivity: Literal[bool] = False):
        A = _as2(a, 3)
        B = _as2(b, 3)
        if A.shape[0] != B.shape[0]:
            raise ValueError(f"Expected inputs `a` and `b` to have same shapes, got {np.asarray(a).shape} and {np.asarray(b).shape} respectively.")
        N = A.shape[0]
        w = np.ones(N) if weights is None else np.asarray(weights, dtype=float).ravel()
        for i in range(w.size):
            if w[i] < 0:
                raise ValueError("`weights` may not contain negative values")
        ninf = 0
        iinf = -1
        if N == 1:
            ninf = 1
            iinf = 0
        else:
            for i in range(N):
                if w[i] == np.inf:
                    ninf += 1
                    iinf = i
        if ninf > 1:
            raise ValueError("Only one infinite weight is allowed")
        if ninf == 1:
            if return_sensitivity:
                raise ValueError("Cannot return sensitivity matrix with an infinite weight or one vector pair")
            ap = A[iinf].copy()
            bp = B[iinf].copy()
            an = float(np.linalg.norm(ap))
            bn = float(np.linalg.norm(bp))
            if an == 0 or bn == 0:
                raise ValueError("Cannot align zero length primary vectors")
            ap = ap / an
            bp = bp / bn
            cr = _cross(bp, ap)
            crn = float(np.linalg.norm(cr))
            theta = atan2(crn, float(np.dot(ap, bp)))
            tol = 1e-3
            qflip = np.array([[0.0, 0.0, 0.0, 1.0]])
            if (pi - theta) < tol:
                r = np.zeros(3)
                if crn == 0:
                    i = int(np.argmin(np.abs(ap)))
                    r[(i - 1) % 3] = ap[(i - 2) % 3]
                    r[(i - 2) % 3] = -ap[(i - 1) % 3]
                else:
                    r = cr
                qflip = _from_rotvec((r / np.linalg.norm(r) * pi).reshape((1, 3)))
                theta = pi - theta
                cr = -cr
            rr = np.zeros(3)
            if abs(theta) < tol:
                t2 = theta * theta
                rr = cr * (1 + t2 / 6 + t2 * t2 * 7 / 360)
            else:
                rr = cr * theta / sin(theta)
            qpri = compose_quat(_from_rotvec(rr.reshape((1, 3))), qflip)
            qopt = qpri
            if N > 1:
                ssum = 0.0
                csum = 0.0
                Rp = Rotation[True](qpri)
                for i in range(N):
                    if i == iinf:
                        continue
                    c = Rp.apply(B[i])
                    st = float(np.dot(_cross(c, A[i]), ap))
                    ct = float(np.dot(c, A[i])) - float(np.dot(c, ap)) * float(np.dot(A[i], ap))
                    ssum += w[i] * st
                    csum += w[i] * ct
                phi = atan2(ssum, csum)
                qsec = _from_rotvec((phi * ap).reshape((1, 3)))
                qopt = compose_quat(qsec, qpri)
            wz = w.copy()
            if N > 1 or w[0] == np.inf:
                wz[iinf] = 0.0
            R = Rotation[True](qopt)
            est = R.apply(B)
            tot = 0.0
            for i in range(N):
                for k in range(3):
                    tot += wz[i] * (A[i, k] - est[i, k]) ** 2
            if return_sensitivity:
                return R, sqrt(tot), np.zeros((3, 3))
            else:
                return R, sqrt(tot)
        Bm = np.zeros((3, 3))
        for j in range(N):
            for i in range(3):
                for k in range(3):
                    Bm[i, k] += w[j] * A[j, i] * B[j, k]
        u, s, vh = np.linalg.svd(Bm)
        if np.linalg.det(u @ vh) < 0:
            s[2] = -s[2]
            for i in range(3):
                u[i, 2] = -u[i, 2]
        C = u @ vh
        tot = 0.0
        for j in range(N):
            for k in range(3):
                tot += w[j] * (B[j, k] ** 2 + A[j, k] ** 2)
        rssd = sqrt(max(tot - 2 * (s[0] + s[1] + s[2]), 0.0))
        R = Rotation[True](_from_matrix_orthogonal(C.reshape((1, 3, 3))))
        if return_sensitivity:
            zeta = (s[0] + s[1]) * (s[1] + s[2]) * (s[2] + s[0])
            kappa = s[0] * s[1] + s[1] * s[2] + s[2] * s[0]
            sens = float(np.mean(w)) / zeta * (kappa * np.eye(3) + Bm @ Bm.T)
            return R, rssd, sens
        else:
            return R, rssd

    def __repr__(self) -> str:
        return f"Rotation.from_matrix({self.as_matrix()})"

class Slerp:
    def __init__(self, times, rotations: Rotation[False]):
        t = np.asarray(times, dtype=float)
        if t.ndim != 1:
            raise ValueError(f"Expected times to be specified in a 1 dimensional array, got {t.ndim} dimensions.")
        n = len(rotations)
        if n <= 1:
            raise ValueError("`rotations` must be a sequence of at least 2 rotations.")
        if t.shape[0] != n:
            raise ValueError(f"Expected number of rotations to be equal to number of timestamps given, got {n} rotations and {t.shape[0]} timestamps.")
        for i in range(n - 1):
            if t[i + 1] - t[i] <= 0:
                raise ValueError("Times must be in strictly increasing order.")
        self.times = t
        self.timedelta = np.diff(t)
        self.rotations = Rotation[False](rotations._quat[:n - 1].copy())
        nxt = Rotation[False](rotations._quat[1:].copy())
        self.rotvecs = (self.rotations.inv() * nxt).as_rotvec()

    def _q(self, times):
        ct = np.atleast_1d(np.asarray(times, dtype=float)).ravel()
        m = ct.shape[0]
        nr = self.rotations._quat.shape[0]
        q = np.empty((m, 4))
        for i in range(m):
            ind = int(np.searchsorted(self.times, ct[i])) - 1
            if ct[i] == self.times[0]:
                ind = 0
            if ind < 0 or ind > nr - 1:
                raise ValueError(f"Interpolation times must be within the range [{self.times[0]}, {self.times[self.times.size - 1]}], both inclusive.")
            alpha = (ct[i] - self.times[ind]) / self.timedelta[ind]
            rv = (self.rotvecs[ind] * alpha).reshape((1, 3))
            qi = compose_quat(self.rotations._quat[ind:ind + 1], _from_rotvec(rv))
            _normalize_rows(qi)
            for k in range(4):
                q[i, k] = qi[0, k]
        return q

    def __call__(self, times):
        if isinstance(times, float) or isinstance(times, int):
            return Rotation[True](self._q(times))
        else:
            return Rotation[False](self._q(times))
