# Pure Typhon implementation of a subset of CPython's hashlib:
# md5, sha1, sha224, sha256, sha384, sha512.

algorithms_guaranteed = {"md5", "sha1", "sha224", "sha256", "sha384", "sha512"}
algorithms_available = algorithms_guaranteed

_MD5: Literal[int] = 0
_SHA1: Literal[int] = 1
_SHA224: Literal[int] = 2
_SHA256: Literal[int] = 3
_SHA384: Literal[int] = 4
_SHA512: Literal[int] = 5

_MD5_S = (7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
          5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
          4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
          6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21)

_MD5_K = (
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a,
    0xa8304613, 0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340,
    0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8,
    0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
    0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92,
    0xffeff47d, 0x85845dd1, 0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391)

_SHA256_K = (
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2)

_SHA512_K = (
    0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc,
    0x3956c25bf348b538, 0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118,
    0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694,
    0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,
    0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
    0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2, 0xd5a79147930aa725, 0x06ca6351e003826f, 0x142929670a0e6e70,
    0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
    0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,
    0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30,
    0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
    0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b,
    0xca273eceea26619c, 0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178,
    0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
    0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817)

def _rotl32(x: u32, n: int) -> u32:
    return (x << u32(n)) | (x >> u32(32 - n))

def _rotr32(x: u32, n: int) -> u32:
    return (x >> u32(n)) | (x << u32(32 - n))

def _rotr64(x: u64, n: int) -> u64:
    return (x >> u64(n)) | (x << u64(64 - n))

def _load_le32(p: Ptr[u8]) -> u32:
    return u32(p[0]) | (u32(p[1]) << u32(8)) | (u32(p[2]) << u32(16)) | (u32(p[3]) << u32(24))

def _load_be32(p: Ptr[u8]) -> u32:
    return (u32(p[0]) << u32(24)) | (u32(p[1]) << u32(16)) | (u32(p[2]) << u32(8)) | u32(p[3])

def _load_be64(p: Ptr[u8]) -> u64:
    r = u64(0)
    for i in range(8):
        r = (r << u64(8)) | u64(p[i])
    return r

def _md5_block(st: Ptr[u64], p: Ptr[u8]):
    m = __array__[u32](16)
    for i in range(16):
        m[i] = _load_le32(p + 4 * i)
    a0 = u32(st[0])
    b0 = u32(st[1])
    c0 = u32(st[2])
    d0 = u32(st[3])
    a, b, c, d = a0, b0, c0, d0
    for i in range(64):
        f = u32(0)
        g = 0
        if i < 16:
            f = (b & c) | (~b & d)
            g = i
        elif i < 32:
            f = (d & b) | (~d & c)
            g = (5 * i + 1) & 15
        elif i < 48:
            f = b ^ c ^ d
            g = (3 * i + 5) & 15
        else:
            f = c ^ (b | ~d)
            g = (7 * i) & 15
        f = f + a + u32(_MD5_K[i]) + m[g]
        a = d
        d = c
        c = b
        b = b + _rotl32(f, _MD5_S[i])
    st[0] = u64(a0 + a)
    st[1] = u64(b0 + b)
    st[2] = u64(c0 + c)
    st[3] = u64(d0 + d)

def _sha1_block(st: Ptr[u64], p: Ptr[u8]):
    w = __array__[u32](80)
    for i in range(16):
        w[i] = _load_be32(p + 4 * i)
    for i in range(16, 80):
        w[i] = _rotl32(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1)
    a = u32(st[0])
    b = u32(st[1])
    c = u32(st[2])
    d = u32(st[3])
    e = u32(st[4])
    for i in range(80):
        f = u32(0)
        k = u32(0)
        if i < 20:
            f = (b & c) | (~b & d)
            k = u32(0x5A827999)
        elif i < 40:
            f = b ^ c ^ d
            k = u32(0x6ED9EBA1)
        elif i < 60:
            f = (b & c) | (b & d) | (c & d)
            k = u32(0x8F1BBCDC)
        else:
            f = b ^ c ^ d
            k = u32(0xCA62C1D6)
        t = _rotl32(a, 5) + f + e + k + w[i]
        e = d
        d = c
        c = _rotl32(b, 30)
        b = a
        a = t
    st[0] = u64(u32(st[0]) + a)
    st[1] = u64(u32(st[1]) + b)
    st[2] = u64(u32(st[2]) + c)
    st[3] = u64(u32(st[3]) + d)
    st[4] = u64(u32(st[4]) + e)

def _sha256_block(st: Ptr[u64], p: Ptr[u8]):
    w = __array__[u32](64)
    for i in range(16):
        w[i] = _load_be32(p + 4 * i)
    for i in range(16, 64):
        s0 = _rotr32(w[i - 15], 7) ^ _rotr32(w[i - 15], 18) ^ (w[i - 15] >> u32(3))
        s1 = _rotr32(w[i - 2], 17) ^ _rotr32(w[i - 2], 19) ^ (w[i - 2] >> u32(10))
        w[i] = w[i - 16] + s0 + w[i - 7] + s1
    a = u32(st[0]); b = u32(st[1]); c = u32(st[2]); d = u32(st[3])
    e = u32(st[4]); f = u32(st[5]); g = u32(st[6]); h = u32(st[7])
    for i in range(64):
        S1 = _rotr32(e, 6) ^ _rotr32(e, 11) ^ _rotr32(e, 25)
        ch = (e & f) ^ (~e & g)
        t1 = h + S1 + ch + u32(_SHA256_K[i]) + w[i]
        S0 = _rotr32(a, 2) ^ _rotr32(a, 13) ^ _rotr32(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        t2 = S0 + maj
        h = g; g = f; f = e; e = d + t1
        d = c; c = b; b = a; a = t1 + t2
    st[0] = u64(u32(st[0]) + a); st[1] = u64(u32(st[1]) + b)
    st[2] = u64(u32(st[2]) + c); st[3] = u64(u32(st[3]) + d)
    st[4] = u64(u32(st[4]) + e); st[5] = u64(u32(st[5]) + f)
    st[6] = u64(u32(st[6]) + g); st[7] = u64(u32(st[7]) + h)

def _sha512_block(st: Ptr[u64], p: Ptr[u8]):
    w = __array__[u64](80)
    for i in range(16):
        w[i] = _load_be64(p + 8 * i)
    for i in range(16, 80):
        s0 = _rotr64(w[i - 15], 1) ^ _rotr64(w[i - 15], 8) ^ (w[i - 15] >> u64(7))
        s1 = _rotr64(w[i - 2], 19) ^ _rotr64(w[i - 2], 61) ^ (w[i - 2] >> u64(6))
        w[i] = w[i - 16] + s0 + w[i - 7] + s1
    a = st[0]; b = st[1]; c = st[2]; d = st[3]
    e = st[4]; f = st[5]; g = st[6]; h = st[7]
    for i in range(80):
        S1 = _rotr64(e, 14) ^ _rotr64(e, 18) ^ _rotr64(e, 41)
        ch = (e & f) ^ (~e & g)
        t1 = h + S1 + ch + u64(_SHA512_K[i]) + w[i]
        S0 = _rotr64(a, 28) ^ _rotr64(a, 34) ^ _rotr64(a, 39)
        maj = (a & b) ^ (a & c) ^ (b & c)
        t2 = S0 + maj
        h = g; g = f; f = e; e = d + t1
        d = c; c = b; b = a; a = t1 + t2
    st[0] += a; st[1] += b; st[2] += c; st[3] += d
    st[4] += e; st[5] += f; st[6] += g; st[7] += h

def _kind_of(name: str) -> int:
    n = name.lower().replace("-", "").replace("_", "")
    if n == "md5":
        return _MD5
    if n == "sha1":
        return _SHA1
    if n == "sha224":
        return _SHA224
    if n == "sha256":
        return _SHA256
    if n == "sha384":
        return _SHA384
    if n == "sha512":
        return _SHA512
    raise ValueError("unsupported hash type " + name)

class _Hash:
    _kind: int
    _state: Ptr[u64]
    _buf: Ptr[u8]
    _buflen: int
    _total: int

    def __init__(self, kind: int):
        self._kind = kind
        self._state = Ptr[u64](8)
        self._buf = Ptr[u8](128)
        self._buflen = 0
        self._total = 0
        st = self._state
        if kind == _MD5:
            st[0] = u64(0x67452301); st[1] = u64(0xefcdab89)
            st[2] = u64(0x98badcfe); st[3] = u64(0x10325476)
        elif kind == _SHA1:
            st[0] = u64(0x67452301); st[1] = u64(0xEFCDAB89)
            st[2] = u64(0x98BADCFE); st[3] = u64(0x10325476)
            st[4] = u64(0xC3D2E1F0)
        elif kind == _SHA224:
            iv = (0xc1059ed8, 0x367cd507, 0x3070dd17, 0xf70e5939,
                  0xffc00b31, 0x68581511, 0x64f98fa7, 0xbefa4fa4)
            for i in range(8):
                st[i] = u64(iv[i])
        elif kind == _SHA256:
            iv = (0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                  0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19)
            for i in range(8):
                st[i] = u64(iv[i])
        elif kind == _SHA384:
            iv = (0xcbbb9d5dc1059ed8, 0x629a292a367cd507, 0x9159015a3070dd17,
                  0x152fecd8f70e5939, 0x67332667ffc00b31, 0x8eb44a8768581511,
                  0xdb0c2e0d64f98fa7, 0x47b5481dbefa4fa4)
            for i in range(8):
                st[i] = u64(iv[i])
        else:
            iv = (0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b,
                  0xa54ff53a5f1d36f1, 0x510e527fade682d1, 0x9b05688c2b3e6c1f,
                  0x1f83d9abfb41bd6b, 0x5be0cd19137e2179)
            for i in range(8):
                st[i] = u64(iv[i])

    @property
    def name(self) -> str:
        k = self._kind
        if k == _MD5:
            return "md5"
        if k == _SHA1:
            return "sha1"
        if k == _SHA224:
            return "sha224"
        if k == _SHA256:
            return "sha256"
        if k == _SHA384:
            return "sha384"
        return "sha512"

    @property
    def digest_size(self) -> int:
        k = self._kind
        if k == _MD5:
            return 16
        if k == _SHA1:
            return 20
        if k == _SHA224:
            return 28
        if k == _SHA256:
            return 32
        if k == _SHA384:
            return 48
        return 64

    @property
    def block_size(self) -> int:
        return 128 if self._kind >= _SHA384 else 64

    def _compress(self, p: Ptr[u8]):
        k = self._kind
        if k == _MD5:
            _md5_block(self._state, p)
        elif k == _SHA1:
            _sha1_block(self._state, p)
        elif k <= _SHA256:
            _sha256_block(self._state, p)
        else:
            _sha512_block(self._state, p)

    def _update_ptr(self, p: Ptr[u8], n: int):
        bs = self.block_size
        self._total += n
        i = 0
        if self._buflen > 0:
            take = min(bs - self._buflen, n)
            str.memcpy(self._buf + self._buflen, p, take)
            self._buflen += take
            i = take
            if self._buflen == bs:
                self._compress(self._buf)
                self._buflen = 0
        while n - i >= bs:
            self._compress(p + i)
            i += bs
        if i < n:
            str.memcpy(self._buf + self._buflen, p + i, n - i)
            self._buflen += n - i

    def update(self, data: bytes):
        self._update_ptr(data._ptr, len(data))

    def update(self, data: str):
        self.update(data.encode())

    def update(self, data: bytearray):
        self.update(bytes(data))

    def copy(self) -> _Hash:
        h = _Hash(self._kind)
        str.memcpy(h._state.as_byte(), self._state.as_byte(), 64)
        str.memcpy(h._buf, self._buf, 128)
        h._buflen = self._buflen
        h._total = self._total
        return h

    def digest(self) -> bytes:
        h = self.copy()
        bs = h.block_size
        lenfield = 16 if bs == 128 else 8
        bits = u64(self._total) << u64(3)
        pad = Ptr[u8](256)
        npad = bs - ((self._total + lenfield) % bs)
        if npad == 0:
            npad = bs
        for i in range(256):
            pad[i] = u8(0)
        pad[0] = u8(0x80)
        total_pad = npad + lenfield
        lp = pad + npad
        if h._kind == _MD5:
            for i in range(8):
                lp[i] = u8(int(bits >> u64(8 * i)) & 0xff)
        else:
            for i in range(lenfield):
                sh = 8 * (lenfield - 1 - i)
                lp[i] = u8(int(bits >> u64(sh)) & 0xff) if sh < 64 else u8(0)
        h._update_ptr(pad, total_pad)
        ds = h.digest_size
        out = Ptr[u8](ds)
        st = h._state
        if h._kind == _MD5:
            for i in range(16):
                out[i] = u8(int(st[i // 4] >> u64(8 * (i % 4))) & 0xff)
        elif h._kind <= _SHA256:
            for i in range(ds):
                out[i] = u8(int(st[i // 4] >> u64(8 * (3 - i % 4))) & 0xff)
        else:
            for i in range(ds):
                out[i] = u8(int(st[i // 8] >> u64(8 * (7 - i % 8))) & 0xff)
        return bytes(out, ds)

    def hexdigest(self) -> str:
        return self.digest().hex()

    def __repr__(self) -> str:
        return f"<{self.name} _hashlib.HASH object>"

def _make(kind: int, data, usedforsecurity: bool) -> _Hash:
    h = _Hash(kind)
    if not isinstance(data, None):
        h.update(data)
    return h

def new(name: str, data = None, usedforsecurity: bool = True) -> _Hash:
    return _make(_kind_of(name), data, usedforsecurity)

def md5(data = None, usedforsecurity: bool = True) -> _Hash:
    return _make(_MD5, data, usedforsecurity)

def sha1(data = None, usedforsecurity: bool = True) -> _Hash:
    return _make(_SHA1, data, usedforsecurity)

def sha224(data = None, usedforsecurity: bool = True) -> _Hash:
    return _make(_SHA224, data, usedforsecurity)

def sha256(data = None, usedforsecurity: bool = True) -> _Hash:
    return _make(_SHA256, data, usedforsecurity)

def sha384(data = None, usedforsecurity: bool = True) -> _Hash:
    return _make(_SHA384, data, usedforsecurity)

def sha512(data = None, usedforsecurity: bool = True) -> _Hash:
    return _make(_SHA512, data, usedforsecurity)
