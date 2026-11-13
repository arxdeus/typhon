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

class error(Exception):
    pass

Z_NO_COMPRESSION = 0
Z_BEST_SPEED = 1
Z_BEST_COMPRESSION = 9

_LBASE = [3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258]
_LEXT = [0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0]
_DBASE = [1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577]
_DEXT = [0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13]

class _Bits:
    data: bytes
    pos: int
    bitbuf: int
    bitcnt: int

    def __init__(self, data: bytes, pos: int):
        self.data = data
        self.pos = pos
        self.bitbuf = 0
        self.bitcnt = 0

    def bits(self, n: int) -> int:
        while self.bitcnt < n:
            if self.pos >= len(self.data):
                raise error("Error -5 while decompressing data: incomplete or truncated stream")
            self.bitbuf |= int(self.data[self.pos]) << self.bitcnt
            self.pos += 1
            self.bitcnt += 8
        v = self.bitbuf & ((1 << n) - 1)
        self.bitbuf >>= n
        self.bitcnt -= n
        return v

class _Huff:
    count: List[int]
    symbol: List[int]

    def __init__(self, lengths: List[int], n: int):
        self.count = [0] * 16
        self.symbol = [0] * n
        for i in range(n):
            self.count[lengths[i]] += 1
        offs = [0] * 16
        for i in range(1, 15):
            offs[i + 1] = offs[i] + self.count[i]
        for i in range(n):
            if lengths[i] != 0:
                self.symbol[offs[lengths[i]]] = i
                offs[lengths[i]] += 1

    def decode(self, b: _Bits) -> int:
        code = 0
        first = 0
        index = 0
        for ln in range(1, 16):
            code |= b.bits(1)
            c = self.count[ln]
            if code - c < first:
                return self.symbol[index + (code - first)]
            index += c
            first += c
            first <<= 1
            code <<= 1
        raise error("Error -3 while decompressing data: invalid code lengths set")

def _fixed_lengths() -> List[int]:
    l = [8] * 288
    for i in range(144, 256):
        l[i] = 9
    for i in range(256, 280):
        l[i] = 7
    return l

def _codes(b: _Bits, out: List[int], lc: _Huff, dc: _Huff):
    while True:
        sym = lc.decode(b)
        if sym < 256:
            out.append(sym)
        elif sym == 256:
            return
        else:
            sym -= 257
            if sym >= 29:
                raise error("Error -3 while decompressing data: invalid literal/length code")
            ln = _LBASE[sym] + b.bits(_LEXT[sym])
            ds = dc.decode(b)
            if ds >= 30:
                raise error("Error -3 while decompressing data: invalid distance code")
            dist = _DBASE[ds] + b.bits(_DEXT[ds])
            if dist > len(out):
                raise error("Error -3 while decompressing data: invalid distance too far back")
            st = len(out) - dist
            for k in range(ln):
                out.append(out[st + k])

def _inflate(data: bytes, pos: int, out: List[int]) -> int:
    b = _Bits(data, pos)
    last = 0
    while last == 0:
        last = b.bits(1)
        typ = b.bits(2)
        if typ == 0:
            b.bitbuf = 0
            b.bitcnt = 0
            if b.pos + 4 > len(data):
                raise error("Error -5 while decompressing data: incomplete or truncated stream")
            ln = int(data[b.pos]) | (int(data[b.pos + 1]) << 8)
            nln = int(data[b.pos + 2]) | (int(data[b.pos + 3]) << 8)
            if ln != (~nln & 0xFFFF):
                raise error("Error -3 while decompressing data: invalid stored block lengths")
            b.pos += 4
            if b.pos + ln > len(data):
                raise error("Error -5 while decompressing data: incomplete or truncated stream")
            for k in range(ln):
                out.append(int(data[b.pos + k]))
            b.pos += ln
        elif typ == 1:
            lc = _Huff(_fixed_lengths(), 288)
            dc = _Huff([5] * 30, 30)
            _codes(b, out, lc, dc)
        elif typ == 2:
            nlen = b.bits(5) + 257
            ndist = b.bits(5) + 1
            ncode = b.bits(4) + 4
            order = [16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15]
            lens = [0] * 19
            for i in range(ncode):
                lens[order[i]] = b.bits(3)
            cl = _Huff(lens, 19)
            ll = [0] * 320
            idx = 0
            while idx < nlen + ndist:
                sym = cl.decode(b)
                if sym < 16:
                    ll[idx] = sym
                    idx += 1
                else:
                    ln = 0
                    rep = 0
                    if sym == 16:
                        if idx == 0:
                            raise error("Error -3 while decompressing data: invalid bit length repeat")
                        ln = ll[idx - 1]
                        rep = 3 + b.bits(2)
                    elif sym == 17:
                        rep = 3 + b.bits(3)
                    else:
                        rep = 11 + b.bits(7)
                    if idx + rep > nlen + ndist:
                        raise error("Error -3 while decompressing data: invalid bit length repeat")
                    for _ in range(rep):
                        ll[idx] = ln
                        idx += 1
            lc = _Huff(ll[:nlen], nlen)
            dl = List[int](ndist)
            for i in range(ndist):
                dl.append(ll[nlen + i])
            dc = _Huff(dl, ndist)
            _codes(b, out, lc, dc)
        else:
            raise error("Error -3 while decompressing data: invalid block type")
    return b.pos

def _to_bytes(out: List[int]) -> bytes:
    return bytes(out)

def decompress(data: bytes, /, wbits: int = 15, bufsize: int = 16384) -> bytes:
    out = List[int]()
    pos = 0
    trailer = 0
    if wbits < 0:
        _inflate(data, 0, out)
    elif wbits >= 16:
        # gzip
        if len(data) < 18 or int(data[0]) != 0x1f or int(data[1]) != 0x8b:
            raise error("Error -3 while decompressing data: incorrect header check")
        flg = int(data[3])
        p = 10
        if flg & 4:
            p += 2 + (int(data[p]) | (int(data[p + 1]) << 8))
        if flg & 8:
            while int(data[p]) != 0:
                p += 1
            p += 1
        if flg & 16:
            while int(data[p]) != 0:
                p += 1
            p += 1
        if flg & 2:
            p += 2
        e = _inflate(data, p, out)
        crc = int(data[e]) | (int(data[e + 1]) << 8) | (int(data[e + 2]) << 16) | (int(data[e + 3]) << 24)
        if crc != crc32(bytes(out)):
            raise error("Error -3 while decompressing data: incorrect data check")
    else:
        if len(data) < 6:
            raise error("Error -5 while decompressing data: incomplete or truncated stream")
        cmf = int(data[0])
        flg = int(data[1])
        if (cmf & 0x0F) != 8 or ((cmf << 8) | flg) % 31 != 0:
            raise error("Error -3 while decompressing data: incorrect header check")
        p = 2
        if flg & 0x20:
            p += 4
        e = _inflate(data, p, out)
        if e + 4 > len(data):
            raise error("Error -5 while decompressing data: incomplete or truncated stream")
        ad = (int(data[e]) << 24) | (int(data[e + 1]) << 16) | (int(data[e + 2]) << 8) | int(data[e + 3])
        if ad != adler32(bytes(out)):
            raise error("Error -3 while decompressing data: incorrect data check")
    return bytes(out)

class _BitW:
    out: List[int]
    buf: int
    cnt: int

    def __init__(self):
        self.out = []
        self.buf = 0
        self.cnt = 0

    def put(self, v: int, n: int):
        self.buf |= v << self.cnt
        self.cnt += n
        while self.cnt >= 8:
            self.out.append(self.buf & 255)
            self.buf >>= 8
            self.cnt -= 8

    def put_rev(self, code: int, n: int):
        r = 0
        for i in range(n):
            r = (r << 1) | ((code >> i) & 1)
        self.put(r, n)

    def flush(self):
        if self.cnt > 0:
            self.out.append(self.buf & 255)
            self.buf = 0
            self.cnt = 0

def _put_fixed_lit(w: _BitW, sym: int):
    if sym < 144:
        w.put_rev(0x30 + sym, 8)
    elif sym < 256:
        w.put_rev(0x190 + sym - 144, 9)
    elif sym < 280:
        w.put_rev(sym - 256, 7)
    else:
        w.put_rev(0xC0 + sym - 280, 8)

def _deflate_raw(data: bytes, level: int) -> List[int]:
    w = _BitW()
    n = len(data)
    if level == 0:
        pos = 0
        if n == 0:
            w.put(1, 1)
            w.put(0, 2)
            w.flush()
            w.out.append(0); w.out.append(0); w.out.append(255); w.out.append(255)
            return w.out
        while pos < n:
            ln = min(65535, n - pos)
            last = 1 if pos + ln >= n else 0
            w.put(last, 1)
            w.put(0, 2)
            w.flush()
            w.out.append(ln & 255); w.out.append(ln >> 8)
            w.out.append((~ln) & 255); w.out.append(((~ln) >> 8) & 255)
            for k in range(ln):
                w.out.append(int(data[pos + k]))
            pos += ln
        return w.out
    w.put(1, 1)
    w.put(1, 2)
    head = Dict[int, int]()
    i = 0
    while i < n:
        bl = 0
        bd = 0
        if i + 2 < n:
            key = (int(data[i]) << 16) | (int(data[i + 1]) << 8) | int(data[i + 2])
            if key in head:
                j = head[key]
                if i - j <= 32768:
                    ml = 0
                    while i + ml < n and ml < 258 and data[j + ml] == data[i + ml]:
                        ml += 1
                    if ml >= 3:
                        bl = ml
                        bd = i - j
            head[key] = i
        if bl >= 3:
            s = 28
            while _LBASE[s] > bl:
                s -= 1
            _put_fixed_lit(w, 257 + s)
            w.put(bl - _LBASE[s], _LEXT[s])
            ds = 29
            while _DBASE[ds] > bd:
                ds -= 1
            w.put_rev(ds, 5)
            w.put(bd - _DBASE[ds], _DEXT[ds])
            i += bl
        else:
            _put_fixed_lit(w, int(data[i]))
            i += 1
    _put_fixed_lit(w, 256)
    w.flush()
    return w.out

def compress(data: bytes, /, level: int = -1, wbits: int = 15) -> bytes:
    from _zdeflate import _Deflate
    if level == -1:
        level = 6
    if level < 0 or level > 9:
        raise error("Bad compression level")
    wb = wbits
    if wb < 0:
        wb = -wb
    elif wb >= 16:
        wb -= 16
    if wb < 9:
        wb = 9
    raw = List[int]()
    if level == 0:
        raw = _deflate_raw(data, 0)
    else:
        raw = _Deflate(data, level, wb).run()
    out = List[int]()
    if wbits < 0:
        return bytes(raw)
    if wbits >= 16:
        xfl = 2 if level == 9 else (4 if level < 2 else 0)
        for v in [0x1f, 0x8b, 8, 0, 0, 0, 0, 0, xfl, 3]:
            out.append(v)
        for v in raw:
            out.append(v)
        c = crc32(data)
        for k in range(4):
            out.append((c >> (8 * k)) & 255)
        n = len(data) & 0xFFFFFFFF
        for k in range(4):
            out.append((n >> (8 * k)) & 255)
        return bytes(out)
    lf = 0 if level < 2 else (1 if level < 6 else (2 if level == 6 else 3))
    hdr = ((8 + ((wb - 8) << 4)) << 8) | (lf << 6)
    hdr += 31 - hdr % 31
    out.append(hdr >> 8)
    out.append(hdr & 255)
    for v in raw:
        out.append(v)
    a = adler32(data)
    for k in range(4):
        out.append((a >> (24 - 8 * k)) & 255)
    return bytes(out)
