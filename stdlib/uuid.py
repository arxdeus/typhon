# Ported from CPython 3.14 (PSF-2.0)
# uuid4 draws from the random module (not os.urandom, which is unavailable).
import hashlib
import random

class UUID:
    bytes: bytes

    def __init__(self, hex: str):
        h = hex.replace("urn:", "").replace("uuid:", "")
        h = h.strip("{}").replace("-", "")
        if len(h) != 32:
            raise ValueError("badly formed hexadecimal UUID string")
        out = bytearray()
        for i in range(16):
            out.append(int(h[2 * i:2 * i + 2], 16))
        self.bytes = bytes(out)

    def __init__(self, bytes: bytes):
        if len(bytes) != 16:
            raise ValueError("bytes is not a 16-char string")
        self.bytes = bytes

    @staticmethod
    def from_ints(hi: int, lo: int) -> UUID:
        out = bytearray()
        for i in range(8):
            out.append((hi >> (8 * (7 - i))) & 255)
        for i in range(8):
            out.append((lo >> (8 * (7 - i))) & 255)
        return UUID(bytes(out))

    @property
    def hex(self) -> str:
        return self.bytes.hex()

    @property
    def hi(self) -> int:
        v = 0
        for i in range(8):
            v = (v << 8) | int(self.bytes[i])
        return v

    @property
    def lo(self) -> int:
        v = 0
        for i in range(8, 16):
            v = (v << 8) | int(self.bytes[i])
        return v

    @property
    def version(self) -> int:
        return (int(self.bytes[6]) >> 4) & 15

    def __str__(self) -> str:
        h = self.hex
        return h[:8] + "-" + h[8:12] + "-" + h[12:16] + "-" + h[16:20] + "-" + h[20:]

    def __repr__(self) -> str:
        return "UUID('" + str(self) + "')"

    @property
    def urn(self) -> str:
        return "urn:uuid:" + str(self)

    def __eq__(self, o: UUID) -> bool:
        return self.bytes == o.bytes

    def __ne__(self, o: UUID) -> bool:
        return self.bytes != o.bytes

    def __lt__(self, o: UUID) -> bool:
        return self.bytes < o.bytes

    def __hash__(self) -> int:
        return hash(self.bytes)

def _stamp(b: bytes, version: int) -> UUID:
    o = bytearray(b)
    o[8] = (int(o[8]) & 0x3F) | 0x80
    o[6] = (int(o[6]) & 0x0F) | (version << 4)
    return UUID(bytes(o))

NAMESPACE_DNS = UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
NAMESPACE_URL = UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")
NAMESPACE_OID = UUID("6ba7b812-9dad-11d1-80b4-00c04fd430c8")
NAMESPACE_X500 = UUID("6ba7b814-9dad-11d1-80b4-00c04fd430c8")

def uuid4() -> UUID:
    out = bytearray()
    for _ in range(16):
        out.append(random.getrandbits(8))
    return _stamp(bytes(out), 4)

def uuid3(namespace: UUID, name: str) -> UUID:
    h = hashlib.md5(namespace.bytes + name.encode()).digest()
    return _stamp(h[:16], 3)

def uuid5(namespace: UUID, name: str) -> UUID:
    h = hashlib.sha1(namespace.bytes + name.encode()).digest()
    return _stamp(h[:16], 5)
