# Ported from CPython 3.14 (PSF-2.0)
import binascii

_B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
_B32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"

def _enc64(s: bytes, alpha: str) -> bytes:
    out = bytearray()
    n = len(s)
    i = 0
    while i < n:
        b0 = int(s[i])
        b1 = int(s[i + 1]) if i + 1 < n else 0
        b2 = int(s[i + 2]) if i + 2 < n else 0
        v = (b0 << 16) | (b1 << 8) | b2
        out.append(ord(alpha[(v >> 18) & 63]))
        out.append(ord(alpha[(v >> 12) & 63]))
        out.append(ord(alpha[(v >> 6) & 63]) if i + 1 < n else 61)
        out.append(ord(alpha[v & 63]) if i + 2 < n else 61)
        i += 3
    return bytes(out)

def _dec64(s: bytes, alpha: str, validate: bool) -> bytes:
    out = bytearray()
    acc = 0
    bits = 0
    cnt = 0
    pad = 0
    for ch in s:
        c = chr(int(ch))
        if c == "=":
            pad += 1
            continue
        k = alpha.find(c)
        if k < 0:
            if validate:
                raise binascii.Error("Only base64 data is allowed")
            continue
        if pad > 0 and validate:
            raise binascii.Error("Excess data after padding")
        acc = ((acc << 6) | k) & 0xFFFFFF
        bits += 6
        cnt += 1
        if bits >= 8:
            bits -= 8
            out.append((acc >> bits) & 255)
    if cnt % 4 == 1:
        raise binascii.Error("Invalid base64-encoded string: number of data characters (" + str(cnt) + ") cannot be 1 more than a multiple of 4")
    if (cnt + pad) % 4 != 0:
        raise binascii.Error("Incorrect padding")
    return bytes(out)

def b64encode(s: bytes, altchars: Optional[bytes] = None) -> bytes:
    r = _enc64(s, _B64)
    if altchars is not None:
        r = r.replace(b"+", altchars[0:1]).replace(b"/", altchars[1:2])
    return r

def b64decode(s: bytes, altchars: Optional[bytes] = None, validate: bool = False) -> bytes:
    if altchars is not None:
        s = s.replace(altchars[0:1], b"+").replace(altchars[1:2], b"/")
    return _dec64(s, _B64, validate)

def standard_b64encode(s: bytes) -> bytes:
    return b64encode(s)

def standard_b64decode(s: bytes) -> bytes:
    return b64decode(s)

def urlsafe_b64encode(s: bytes) -> bytes:
    return b64encode(s, b"-_")

def urlsafe_b64decode(s: bytes) -> bytes:
    return b64decode(s, b"-_")

def b32encode(s: bytes) -> bytes:
    out = bytearray()
    n = len(s)
    i = 0
    while i < n:
        chunk = 0
        m = 0
        while m < 5:
            chunk = (chunk << 8) | (int(s[i + m]) if i + m < n else 0)
            m += 1
        have = min(5, n - i)
        nch = (have * 8 + 4) // 5
        k = 0
        while k < 8:
            if k < nch:
                out.append(ord(_B32[(chunk >> (35 - 5 * k)) & 31]))
            else:
                out.append(61)
            k += 1
        i += 5
    return bytes(out)

def b32decode(s: bytes, casefold: bool = False) -> bytes:
    out = bytearray()
    acc = 0
    bits = 0
    cnt = 0
    for ch in s:
        c = chr(int(ch))
        if c == "=":
            continue
        if casefold:
            c = c.upper()
        k = _B32.find(c)
        if k < 0:
            raise binascii.Error("Non-base32 digit found")
        acc = ((acc << 5) | k) & 0xFFFFFFFF
        bits += 5
        cnt += 1
        if bits >= 8:
            bits -= 8
            out.append((acc >> bits) & 255)
    if len(s) % 8 != 0:
        raise binascii.Error("Incorrect padding")
    return bytes(out)

def b16encode(s: bytes) -> bytes:
    return binascii.hexlify(s).upper()

def b16decode(s: bytes, casefold: bool = False) -> bytes:
    for c in s:
        if 97 <= int(c) <= 102 and not casefold:
            raise binascii.Error("Non-base16 digit found")
    return binascii.unhexlify(s)

def encodebytes(s: bytes) -> bytes:
    out = bytearray()
    n = len(s)
    i = 0
    while i < n:
        out.extend(_enc64(s[i:i + 57], _B64))
        out.append(10)
        i += 57
    return bytes(out)

def decodebytes(s: bytes) -> bytes:
    return b64decode(s)
