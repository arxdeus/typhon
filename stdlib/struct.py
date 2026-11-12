# Ported from CPython 3.14 (PSF-2.0)
# Supported codes: x c b B ? h H i I l L q Q n N P f d s, byte orders @ = < > !.
# Unsupported: e (half float), p (pascal string), unpack returns List[Union[...]].

class error(Exception):
    pass

def _fmt_info(fmt: str) -> Tuple[bool, bool]:
    # (little_endian, native)
    if len(fmt) > 0:
        c = fmt[0]
        if c == "<":
            return (True, False)
        if c == ">" or c == "!":
            return (False, False)
        if c == "=":
            return (True, False)
    return (True, True)

def _body(fmt: str) -> str:
    if len(fmt) > 0 and fmt[0] in "@=<>!":
        return fmt[1:]
    return fmt

def _size_of(code: str, native: bool) -> int:
    if code in "xcbB?s":
        return 1
    if code in "hH":
        return 2
    if code in "iIf":
        return 4
    if code in "lL":
        return 8 if native else 4
    if code in "qQdnNP":
        return 8
    raise error("bad char in struct format")

# parse into list of (code, count, offset) ; returns total size
def _layout(fmt: str) -> Tuple[List[str], List[int], List[int], int]:
    native = _fmt_info(fmt)[1]
    body = _body(fmt)
    codes: List[str] = []
    counts: List[int] = []
    offs: List[int] = []
    pos = 0
    i = 0
    n = len(body)
    while i < n:
        c = body[i]
        if c == " " or c == "\t" or c == "\n":
            i += 1
            continue
        cnt = -1
        if c.isdigit():
            cnt = 0
            while i < n and body[i].isdigit():
                cnt = cnt * 10 + (ord(body[i]) - 48)
                i += 1
            if i >= n:
                raise error("repeat count given without format specifier")
            c = body[i]
        i += 1
        sz = _size_of(c, native)
        if native and sz > 1 and c != "s":
            pos = (pos + sz - 1) // sz * sz
        if c == "s":
            codes.append(c)
            counts.append(1 if cnt < 0 else cnt)
            offs.append(pos)
            pos += 1 if cnt < 0 else cnt
        elif c == "x":
            pos += 1 if cnt < 0 else cnt
        else:
            reps = 1 if cnt < 0 else cnt
            for _ in range(reps):
                codes.append(c)
                counts.append(1)
                offs.append(pos)
                pos += sz
    return (codes, counts, offs, pos)

def calcsize(fmt: str) -> int:
    return _layout(fmt)[3]

def _put_int(buf: bytearray, off: int, size: int, v: int, little: bool, signed: bool, code: str):
    lo = 0
    hi = 0
    if size == 8:
        # int is 64-bit: every value fits (unsigned range limited to >= 0)
        if not signed and v < 0:
            raise error("'" + code + "' format requires 0 <= number <= 18446744073709551615")
        lo = v
        hi = v
    elif signed:
        lo = -(1 << (size * 8 - 1))
        hi = (1 << (size * 8 - 1)) - 1
    else:
        hi = (1 << (size * 8)) - 1
    if v < lo or v > hi:
        if signed:
            raise error("'" + code + "' format requires " + str(lo) + " <= number <= " + str(hi))
        raise error("'" + code + "' format requires 0 <= number <= " + str(hi))
    u = v if size == 8 else v & ((1 << (size * 8)) - 1)
    for k in range(size):
        b = (u >> (8 * k)) & 255
        if little:
            buf[off + k] = b
        else:
            buf[off + size - 1 - k] = b

def _get_uint(data: bytes, off: int, size: int, little: bool) -> int:
    u = 0
    for k in range(size):
        b = int(data[off + k])
        if little:
            u |= b << (8 * k)
        else:
            u = (u << 8) | b
    return u

def _pack_one(buf: bytearray, code: str, cnt: int, off: int, little: bool, native: bool, a):
    if code == "s":
        if isinstance(a, bytes):
            k = 0
            while k < cnt and k < len(a):
                buf[off + k] = a[k]
                k += 1
        else:
            raise error("argument for 's' must be a bytes object")
    elif code == "c":
        if isinstance(a, bytes) and len(a) == 1:
            buf[off] = a[0]
        else:
            raise error("char format requires a bytes object of length 1")
    elif code == "?":
        if isinstance(a, bool):
            buf[off] = 1 if a else 0
        elif isinstance(a, int):
            buf[off] = 1 if a != 0 else 0
        else:
            raise error("required argument is not an integer")
    elif code == "f" or code == "d":
        x = 0.0
        if isinstance(a, float):
            x = a
        elif isinstance(a, int):
            x = float(a)
        else:
            raise error("required argument is not a float")
        if code == "d":
            p = __ptr__(x)
            q = Ptr[u8](p.as_byte())
            for k in range(8):
                if little:
                    buf[off + k] = int(q[k])
                else:
                    buf[off + 7 - k] = int(q[k])
        else:
            y = f32(x)
            p = __ptr__(y)
            q = Ptr[u8](p.as_byte())
            for k in range(4):
                if little:
                    buf[off + k] = int(q[k])
                else:
                    buf[off + 3 - k] = int(q[k])
    else:
        v = 0
        if isinstance(a, int):
            v = a
        elif isinstance(a, bool):
            v = 1 if a else 0
        else:
            raise error("required argument is not an integer")
        size = _size_of(code, native)
        _put_int(buf, off, size, v, little, code in "bhilqn", code)

def pack(fmt: str, *args) -> bytes:
    info = _fmt_info(fmt)
    lay = _layout(fmt)
    if len(lay[0]) != len(args):
        raise error("pack expected " + str(len(lay[0])) + " items for packing (got " + str(len(args)) + ")")
    buf = bytearray(lay[3])
    idx = 0
    for a in args:
        _pack_one(buf, lay[0][idx], lay[1][idx], lay[2][idx], info[0], info[1], a)
        idx += 1
    return bytes(buf)

def _unpack_one(data: bytes, code: str, cnt: int, off: int, little: bool, native: bool) -> Union[int, float, bytes, bool]:
    if code == "s":
        return data[off:off + cnt]
    if code == "c":
        return data[off:off + 1]
    if code == "?":
        return int(data[off]) != 0
    if code == "d":
        u = _get_uint(data, off, 8, little)
        x = 0.0
        p = __ptr__(x)
        q = Ptr[u8](p.as_byte())
        for k in range(8):
            q[k] = u8((u >> (8 * k)) & 255)
        return x
    if code == "f":
        u = _get_uint(data, off, 4, little)
        y = f32(0.0)
        p = __ptr__(y)
        q = Ptr[u8](p.as_byte())
        for k in range(4):
            q[k] = u8((u >> (8 * k)) & 255)
        return float(y)
    size = _size_of(code, native)
    u = _get_uint(data, off, size, little)
    if size < 8 and code in "bhilqn" and u >= (1 << (size * 8 - 1)):
        u -= 1 << (size * 8)
    return u

def unpack(fmt: str, buffer: bytes) -> List[Union[int, float, bytes, bool]]:
    info = _fmt_info(fmt)
    lay = _layout(fmt)
    if len(buffer) != lay[3]:
        raise error("unpack requires a buffer of " + str(lay[3]) + " bytes")
    out: List[Union[int, float, bytes, bool]] = []
    for i in range(len(lay[0])):
        out.append(_unpack_one(buffer, lay[0][i], lay[1][i], lay[2][i], info[0], info[1]))
    return out

def unpack_from(fmt: str, buffer: bytes, offset: int = 0) -> List[Union[int, float, bytes, bool]]:
    return unpack(fmt, buffer[offset:offset + calcsize(fmt)])

def unpack_ints(fmt: str, buffer: bytes) -> List[int]:
    out: List[int] = []
    for v in unpack(fmt, buffer):
        if isinstance(v, int):
            out.append(v)
        elif isinstance(v, bool):
            out.append(1 if v else 0)
        else:
            raise error("non-integer item in format")
    return out

def unpack_floats(fmt: str, buffer: bytes) -> List[float]:
    out: List[float] = []
    for v in unpack(fmt, buffer):
        if isinstance(v, float):
            out.append(v)
        else:
            raise error("non-float item in format")
    return out
