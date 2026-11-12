# Ported from CPython 3.14 (PSF-2.0)

class Error(ValueError):
    pass

_HEX = "0123456789abcdef"

def hexlify(data: bytes, sep: Optional[str] = None, bytes_per_sep: int = 1) -> bytes:
    out = ""
    first = True
    for c in data:
        if not first and sep is not None:
            out += sep
        first = False
        out += _HEX[(int(c) >> 4) & 15]
        out += _HEX[int(c) & 15]
    return out.encode()

def b2a_hex(data: bytes) -> bytes:
    return hexlify(data)

def _hv(c: int) -> int:
    if 48 <= c <= 57:
        return c - 48
    if 97 <= c <= 102:
        return c - 87
    if 65 <= c <= 70:
        return c - 55
    raise Error("Non-hexadecimal digit found")

def unhexlify(hexstr: bytes) -> bytes:
    n = len(hexstr)
    if n % 2 != 0:
        raise Error("Odd-length string")
    out = bytearray()
    i = 0
    while i < n:
        out.append((_hv(int(hexstr[i])) << 4) | _hv(int(hexstr[i + 1])))
        i += 2
    return bytes(out)

def a2b_hex(hexstr: bytes) -> bytes:
    return unhexlify(hexstr)

def crc32(data: bytes, value: int = 0) -> int:
    crc = (value ^ 0xFFFFFFFF) & 0xFFFFFFFF
    for b in data:
        crc ^= int(b)
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xEDB88320
            else:
                crc >>= 1
    return crc ^ 0xFFFFFFFF
