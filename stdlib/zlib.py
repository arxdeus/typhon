# Ported from zlib/CPython 3.14 (checksums only; zlib license)

MAX_WBITS = 15
DEF_MEM_LEVEL = 8
Z_DEFAULT_COMPRESSION = -1

def _make_table() -> List[int]:
    t = List[int](256)
    for n in range(256):
        c = n
        for _ in range(8):
            if c & 1:
                c = 0xEDB88320 ^ (c >> 1)
            else:
                c = c >> 1
        t.append(c)
    return t

_CRC_TABLE = _make_table()

def crc32(data: bytes, value: int = 0, /) -> int:
    c = (value & 0xFFFFFFFF) ^ 0xFFFFFFFF
    for i in range(len(data)):
        c = _CRC_TABLE[(c ^ int(data[i])) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF

def adler32(data: bytes, value: int = 1, /) -> int:
    a = value & 0xFFFF
    b = (value >> 16) & 0xFFFF
    for i in range(len(data)):
        a = (a + int(data[i])) % 65521
        b = (b + a) % 65521
    return (b << 16) | a
