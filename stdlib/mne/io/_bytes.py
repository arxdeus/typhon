# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# Typed byte decoding / encoding helpers used by the FIF, EDF, BrainVision
# and MATLAB readers (struct.unpack returns unions in Typhon).

import numpy as np


def _read_file(fname: str) -> bytes:
    with open(fname, "rb") as f:
        return f.read()


def _read_text(fname: str) -> str:
    b = _read_file(fname)
    return _ascii(b, 0, len(b))


def _u8(b: bytes, off: int) -> int:
    return int(b._ptr[off])


def _uint_be(b: bytes, off: int, n: int) -> int:
    u = 0
    for k in range(n):
        u = (u << 8) | int(b._ptr[off + k])
    return u


def _uint_le(b: bytes, off: int, n: int) -> int:
    u = 0
    for k in range(n):
        u = (u << 8) | int(b._ptr[off + n - 1 - k])
    return u


def _sint(u: int, n: int) -> int:
    if n >= 8:
        return u  # already two's complement in a 64-bit int (1 << 64 is UB)
    if u >= (1 << (8 * n - 1)):
        return u - (1 << (8 * n))
    return u


def _i32_be(b: bytes, off: int) -> int:
    return _sint(_uint_be(b, off, 4), 4)


def _i32_le(b: bytes, off: int) -> int:
    return _sint(_uint_le(b, off, 4), 4)


def _i16_le(b: bytes, off: int) -> int:
    return _sint(_uint_le(b, off, 2), 2)


def _i16_be(b: bytes, off: int) -> int:
    return _sint(_uint_be(b, off, 2), 2)


def _bits_to_f32(u: int) -> float:
    x = u32(u)
    return float(Ptr[float32](__ptr__(x).as_byte())[0])


def _bits_to_f64(u: int) -> float:
    x = u64(u)
    return Ptr[float](__ptr__(x).as_byte())[0]


def _f32_bits(v: float) -> int:
    x = float32(v)
    return int(Ptr[u32](__ptr__(x).as_byte())[0])


def _f64_bits(v: float) -> int:
    x = v
    return int(Ptr[u64](__ptr__(x).as_byte())[0])


def _f32_be(b: bytes, off: int) -> float:
    return _bits_to_f32(_uint_be(b, off, 4))


def _f64_be(b: bytes, off: int) -> float:
    return _bits_to_f64(_uint_be(b, off, 8))


def _f32_le(b: bytes, off: int) -> float:
    return _bits_to_f32(_uint_le(b, off, 4))


def _f64_le(b: bytes, off: int) -> float:
    return _bits_to_f64(_uint_le(b, off, 8))


def _ascii(b: bytes, off: int, n: int) -> str:
    return str(b._ptr + off, n)


def _arr_be(b: bytes, off: int, n: int, code: str) -> np.ndarray[float, 1]:
    """Decode n big-endian values ('f4','f8','i4','i2','u1') as float64."""
    out = np.empty(n, dtype=float)
    if code == "f4":
        for i in range(n):
            out[i] = _f32_be(b, off + 4 * i)
    elif code == "f8":
        for i in range(n):
            out[i] = _f64_be(b, off + 8 * i)
    elif code == "i4":
        for i in range(n):
            out[i] = float(_i32_be(b, off + 4 * i))
    elif code == "i2":
        for i in range(n):
            out[i] = float(_i16_be(b, off + 2 * i))
    else:
        for i in range(n):
            out[i] = float(_u8(b, off + i))
    return out


def _arr_le(b: bytes, off: int, n: int, code: str) -> np.ndarray[float, 1]:
    """Decode n little-endian values ('f4','f8','i4','i2','i3','u1') as float64."""
    out = np.empty(n, dtype=float)
    if code == "f4":
        for i in range(n):
            out[i] = _f32_le(b, off + 4 * i)
    elif code == "f8":
        for i in range(n):
            out[i] = _f64_le(b, off + 8 * i)
    elif code == "i4":
        for i in range(n):
            out[i] = float(_i32_le(b, off + 4 * i))
    elif code == "i2":
        for i in range(n):
            out[i] = float(_i16_le(b, off + 2 * i))
    elif code == "i3":
        for i in range(n):
            out[i] = float(_sint(_uint_le(b, off + 3 * i, 3), 3))
    else:
        for i in range(n):
            out[i] = float(_u8(b, off + i))
    return out


class _ByteWriter:
    """Growable big-endian byte buffer."""
    buf: List[int]

    def __init__(self):
        self.buf = List[int]()

    def __len__(self) -> int:
        return len(self.buf)

    def uint_be(self, u: int, n: int):
        for k in range(n):
            self.buf.append((u >> (8 * (n - 1 - k))) & 255)

    def i32(self, v: int):
        self.uint_be(v & 0xFFFFFFFF, 4)

    def f32(self, v: float):
        self.uint_be(_f32_bits(v), 4)

    def f64(self, v: float):
        self.uint_be(_f64_bits(v), 8)

    def raw(self, s: str):
        for k in range(len(s)):
            self.buf.append(int(s._ptr[k]))

    def zeros(self, n: int):
        for _ in range(n):
            self.buf.append(0)

    def to_bytes(self) -> bytes:
        return bytes(self.buf)

    def save(self, fname: str):
        with open(fname, "wb") as f:
            f.write(self.to_bytes())
