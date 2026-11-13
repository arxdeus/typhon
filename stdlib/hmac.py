# Ported from CPython 3.14 hmac (PSF license)
import hashlib

def compare_digest(a, b) -> bool:
    x = bytes(a)
    y = bytes(b)
    if len(x) != len(y):
        return False
    r = 0
    for i in range(len(x)):
        r |= int(x[i]) ^ int(y[i])
    return r == 0

class HMAC:
    _inner: hashlib._Hash
    _outer: hashlib._Hash
    _name: str

    def __init__(self, key: bytes, digestmod: str = "sha256"):
        self._name = digestmod
        h = hashlib.new(digestmod)
        bs = h.block_size
        k = key
        if len(k) > bs:
            k = hashlib.new(digestmod, k).digest()
        ik = List[int](bs)
        ok = List[int](bs)
        for i in range(bs):
            c = int(k[i]) if i < len(k) else 0
            ik.append(c ^ 0x36)
            ok.append(c ^ 0x5C)
        self._inner = hashlib.new(digestmod, bytes(ik))
        self._outer = hashlib.new(digestmod, bytes(ok))

    @property
    def name(self) -> str:
        return "hmac-" + self._name

    @property
    def digest_size(self) -> int:
        return self._inner.digest_size

    @property
    def block_size(self) -> int:
        return self._inner.block_size

    def update(self, msg: bytes):
        self._inner.update(msg)

    def copy(self) -> HMAC:
        h = HMAC(b"", self._name)
        h._inner = self._inner.copy()
        h._outer = self._outer.copy()
        return h

    def digest(self) -> bytes:
        o = self._outer.copy()
        o.update(self._inner.digest())
        return o.digest()

    def hexdigest(self) -> str:
        return self.digest().hex()

def new(key: bytes, msg: bytes = b"", digestmod: str = "sha256") -> HMAC:
    h = HMAC(key, digestmod)
    if len(msg) > 0:
        h.update(msg)
    return h

def digest(key: bytes, msg: bytes, digest: str) -> bytes:
    return new(key, msg, digest).digest()
