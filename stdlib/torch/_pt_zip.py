# Copyright (C) 2026 the Typhon authors
# Ported from torch 2.15.0 (BSD-3-Clause)
#
# Native PyTorch checkpoint format (torch.save default since 1.6):
# a stored (uncompressed) zip archive
#   <archive>/data.pkl              protocol-2 pickle of the object
#   <archive>/.format_version       "1"
#   <archive>/.storage_alignment    "64"
#   <archive>/byteorder             "little"
#   <archive>/data/<key>            raw storage bytes (64-byte aligned)
#   <archive>/version               "3\n"
#   <archive>/.data/serialization_id
# Tensors are pickled as torch._utils._rebuild_tensor_v2(
#   persistent_load(('storage', torch.<T>Storage, key, location, numel)),
#   storage_offset, size, stride, requires_grad, OrderedDict()).
# The reader is a restricted unpickler equivalent to weights_only=True for
# state_dict-like objects (dicts of tensors / primitive values).

from torch._tensor import Tensor, _Storage, _BOOL, _U8, _I8, _I16, _I32, _I64, _F32, _F64, _itemsize
from zlib import crc32

# ---------------------------------------------------------------- zip reader

def _u16(p: Ptr[u8], i: int) -> int:
    return int(p[i]) | (int(p[i + 1]) << 8)

def _u32(p: Ptr[u8], i: int) -> int:
    return _u16(p, i) | (_u16(p, i + 2) << 16)

def _u64(p: Ptr[u8], i: int) -> int:
    return _u32(p, i) | (_u32(p, i + 4) << 32)

@tuple
class _ZEntry:
    name: str
    start: int   # absolute offset of the member data
    size: int

def _zip_entries(data: bytes) -> List[_ZEntry]:
    p = data._ptr
    n = len(data)
    # end of central directory (scan back over a possible comment)
    e = n - 22
    lo = max(0, n - 22 - 65535)
    while e >= lo:
        if p[e] == u8(0x50) and p[e + 1] == u8(0x4b) and p[e + 2] == u8(5) and p[e + 3] == u8(6):
            break
        e -= 1
    if e < lo:
        raise RuntimeError("PytorchStreamReader failed locating file: end of central directory not found")
    count = _u16(p, e + 10)
    cd = _u32(p, e + 16)
    if count == 0xFFFF or cd == 0xFFFFFFFF:
        loc = e - 20
        if loc < 0 or _u32(p, loc) != 0x07064b50:
            raise RuntimeError("PytorchStreamReader: bad zip64 locator")
        z = _u64(p, loc + 8)
        if _u32(p, z) != 0x06064b50:
            raise RuntimeError("PytorchStreamReader: bad zip64 end of central directory")
        count = _u64(p, z + 32)
        cd = _u64(p, z + 48)
    out = List[_ZEntry](count)
    q = cd
    for _ in range(count):
        if q + 46 > n or _u32(p, q) != 0x02014b50:
            raise RuntimeError("PytorchStreamReader: bad central directory entry")
        method = _u16(p, q + 10)
        csize = _u32(p, q + 20)
        usize = _u32(p, q + 24)
        nl = _u16(p, q + 28)
        xl = _u16(p, q + 30)
        cl = _u16(p, q + 32)
        off = _u32(p, q + 42)
        name = str(p + q + 46, nl)
        # zip64 extended information extra field
        x = q + 46 + nl
        xe = x + xl
        while x + 4 <= xe:
            hid = _u16(p, x)
            hsz = _u16(p, x + 2)
            if hid == 1:
                k = x + 4
                if usize == 0xFFFFFFFF:
                    usize = _u64(p, k); k += 8
                if csize == 0xFFFFFFFF:
                    csize = _u64(p, k); k += 8
                if off == 0xFFFFFFFF:
                    off = _u64(p, k); k += 8
            x += 4 + hsz
        if method != 0 or csize != usize:
            raise RuntimeError("PytorchStreamReader: compressed record " + name + " is not supported")
        if _u32(p, off) != 0x04034b50:
            raise RuntimeError("PytorchStreamReader: bad local header for " + name)
        start = off + 30 + _u16(p, off + 26) + _u16(p, off + 28)
        if start + usize > n:
            raise RuntimeError("PytorchStreamReader: truncated record " + name)
        out.append(_ZEntry(name, start, usize))
        q += 46 + nl + xl + cl
    return out

# ---------------------------------------------------------------- restricted unpickler

_K_NONE: Literal[int] = 0
_K_INT: Literal[int] = 1
_K_STR: Literal[int] = 2
_K_BOOL: Literal[int] = 3
_K_TUPLE: Literal[int] = 4
_K_LIST: Literal[int] = 5
_K_DICT: Literal[int] = 6
_K_GLOBAL: Literal[int] = 7
_K_MARK: Literal[int] = 8
_K_STORAGE: Literal[int] = 9
_K_TENSOR: Literal[int] = 10
_K_FLOAT: Literal[int] = 11

class _V:
    kind: int
    i: int
    f: float
    s: str
    items: List[_V]
    vals: List[_V]
    t: Optional[Tensor]

    def __init__(self, kind: int):
        self.kind = kind
        self.i = 0
        self.f = 0.0
        self.s = ""
        self.items = List[_V]()
        self.vals = List[_V]()
        self.t = None

def _vi(k: int, i: int) -> _V:
    v = _V(k)
    v.i = i
    return v

def _vs(k: int, s: str) -> _V:
    v = _V(k)
    v.s = s
    return v

def _storage_dt(name: str) -> int:
    if name == "torch FloatStorage": return _F32
    if name == "torch DoubleStorage": return _F64
    if name == "torch LongStorage": return _I64
    if name == "torch IntStorage": return _I32
    if name == "torch ShortStorage": return _I16
    if name == "torch CharStorage": return _I8
    if name == "torch ByteStorage": return _U8
    if name == "torch BoolStorage": return _BOOL
    raise RuntimeError("Weights only load failed: unsupported global " + name)

def _int_list(v: _V) -> List[int]:
    return [x.i for x in v.items]

class _Unpickler:
    p: Ptr[u8]
    n: int
    pos: int
    stack: List[_V]
    marks: List[int]
    memo: Dict[int, _V]
    data: bytes
    base: str
    recs: Dict[str, _ZEntry]
    storages: Dict[str, _Storage]

    def __init__(self, data: bytes, pk: _ZEntry, base: str, recs: Dict[str, _ZEntry]):
        self.data = data
        self.p = data._ptr + pk.start
        self.n = pk.size
        self.pos = 0
        self.stack = List[_V]()
        self.marks = List[int]()
        self.memo = Dict[int, _V]()
        self.base = base
        self.recs = recs
        self.storages = Dict[str, _Storage]()

    def _need(self, k: int):
        if self.pos + k > self.n:
            raise RuntimeError("pickle data was truncated")

    def b1(self) -> int:
        self._need(1)
        r = int(self.p[self.pos]); self.pos += 1
        return r

    def b2(self) -> int:
        self._need(2)
        r = _u16(self.p, self.pos); self.pos += 2
        return r

    def b4(self) -> int:
        self._need(4)
        r = _u32(self.p, self.pos); self.pos += 4
        return r

    def text(self, k: int) -> str:
        self._need(k)
        r = str(self.p + self.pos, k); self.pos += k
        return r

    def line(self) -> str:
        s = self.pos
        while self.pos < self.n and self.p[self.pos] != u8(10):
            self.pos += 1
        r = str(self.p + s, self.pos - s)
        self.pos += 1
        return r

    def pop_mark(self) -> List[_V]:
        m = self.marks.pop()
        r = self.stack[m:]
        del self.stack[m:]
        return r

    def storage(self, key: str, dt: int, numel: int) -> _Storage:
        if key in self.storages:
            return self.storages[key]
        nm = self.base + "/data/" + key
        if nm not in self.recs:
            raise RuntimeError("PytorchStreamReader failed locating file " + nm)
        r = self.recs[nm]
        nb = numel * _itemsize(dt)
        if r.size < nb:
            raise RuntimeError("storage record too small: " + nm)
        st = _Storage(nb)
        str.memcpy(Ptr[u8](st.ptr), self.data._ptr + r.start, nb)
        self.storages[key] = st
        return st

    def reduce(self, fn: _V, args: _V) -> _V:
        if fn.kind != _K_GLOBAL:
            raise RuntimeError("Weights only load failed: REDUCE on non-global")
        g = fn.s
        if g == "collections OrderedDict" or g == "builtins dict":
            return _V(_K_DICT)
        if g == "torch._utils _rebuild_tensor_v2":
            a = args.items
            if len(a) < 4 or a[0].kind != _K_STORAGE:
                raise RuntimeError("Weights only load failed: bad _rebuild_tensor_v2 args")
            st = a[0]
            t = Tensor(st.t.__val__()._st, a[1].i, _int_list(a[2]), _int_list(a[3]), st.t.__val__()._dt)
            v = _V(_K_TENSOR)
            v.t = t
            return v
        raise RuntimeError("Weights only load failed: unsupported global " + g)

    def run(self) -> _V:
        while True:
            op = self.b1()
            if op == 0x80:      # PROTO
                self.b1()
            elif op == 0x95:    # FRAME
                self._need(8); self.pos += 8
            elif op == 0x2e:    # STOP
                return self.stack.pop()
            elif op == 0x28:    # MARK
                self.marks.append(len(self.stack))
            elif op == 0x71:    # BINPUT
                self.memo[self.b1()] = self.stack[-1]
            elif op == 0x72:    # LONG_BINPUT
                self.memo[self.b4()] = self.stack[-1]
            elif op == 0x94:    # MEMOIZE
                self.memo[len(self.memo)] = self.stack[-1]
            elif op == 0x68:    # BINGET
                self.stack.append(self.memo[self.b1()])
            elif op == 0x6a:    # LONG_BINGET
                self.stack.append(self.memo[self.b4()])
            elif op == 0x58:    # BINUNICODE
                self.stack.append(_vs(_K_STR, self.text(self.b4())))
            elif op == 0x8c:    # SHORT_BINUNICODE
                self.stack.append(_vs(_K_STR, self.text(self.b1())))
            elif op == 0x4b:    # BININT1
                self.stack.append(_vi(_K_INT, self.b1()))
            elif op == 0x4d:    # BININT2
                self.stack.append(_vi(_K_INT, self.b2()))
            elif op == 0x4a:    # BININT
                x = self.b4()
                if x >= 0x80000000: x -= 0x100000000
                self.stack.append(_vi(_K_INT, x))
            elif op == 0x8a:    # LONG1
                k = self.b1()
                if k > 8:
                    raise RuntimeError("pickle: LONG1 too large")
                x = 0
                for j in range(k):
                    x |= self.b1() << (8 * j)
                if k > 0 and k < 8 and (x >> (8 * k - 1)) & 1:
                    x -= 1 << (8 * k)
                self.stack.append(_vi(_K_INT, x))
            elif op == 0x47:    # BINFLOAT (big-endian double)
                x = 0
                for j in range(8):
                    x = (x << 8) | self.b1()
                v = _V(_K_FLOAT)
                v.f = Ptr[float](__ptr__(x).as_byte())[0]
                self.stack.append(v)
            elif op == 0x4e:    # NONE
                self.stack.append(_V(_K_NONE))
            elif op == 0x88:    # NEWTRUE
                self.stack.append(_vi(_K_BOOL, 1))
            elif op == 0x89:    # NEWFALSE
                self.stack.append(_vi(_K_BOOL, 0))
            elif op == 0x29:    # EMPTY_TUPLE
                self.stack.append(_V(_K_TUPLE))
            elif op == 0x74:    # TUPLE
                v = _V(_K_TUPLE); v.items = self.pop_mark()
                self.stack.append(v)
            elif op == 0x85 or op == 0x86 or op == 0x87:  # TUPLE1..3
                k = op - 0x84
                v = _V(_K_TUPLE); v.items = self.stack[len(self.stack) - k:]
                del self.stack[len(self.stack) - k:]
                self.stack.append(v)
            elif op == 0x5d:    # EMPTY_LIST
                self.stack.append(_V(_K_LIST))
            elif op == 0x61:    # APPEND
                x = self.stack.pop()
                self.stack[-1].items.append(x)
            elif op == 0x65:    # APPENDS
                xs = self.pop_mark()
                self.stack[-1].items.extend(xs)
            elif op == 0x7d:    # EMPTY_DICT
                self.stack.append(_V(_K_DICT))
            elif op == 0x73:    # SETITEM
                val = self.stack.pop()
                key = self.stack.pop()
                d = self.stack[-1]
                d.items.append(key); d.vals.append(val)
            elif op == 0x75:    # SETITEMS
                xs = self.pop_mark()
                d = self.stack[-1]
                for j in range(0, len(xs) - 1, 2):
                    d.items.append(xs[j]); d.vals.append(xs[j + 1])
            elif op == 0x63:    # GLOBAL
                mod = self.line()
                nm = self.line()
                self.stack.append(_vs(_K_GLOBAL, mod + " " + nm))
            elif op == 0x52:    # REDUCE
                args = self.stack.pop()
                fn = self.stack.pop()
                self.stack.append(self.reduce(fn, args))
            elif op == 0x62:    # BUILD (state_dict._metadata etc.: ignored)
                self.stack.pop()
            elif op == 0x51:    # BINPERSID
                pid = self.stack.pop()
                a = pid.items
                if len(a) != 5 or a[0].s != "storage" or a[1].kind != _K_GLOBAL:
                    raise RuntimeError("Weights only load failed: unsupported persistent id")
                dt = _storage_dt(a[1].s)
                st = self.storage(a[2].s, dt, a[4].i)
                v = _V(_K_STORAGE)
                v.t = Tensor(st, 0, [a[4].i], [1], dt)
                self.stack.append(v)
            else:
                raise RuntimeError("Weights only load failed: unsupported pickle opcode " + str(op))

def is_torch_zip(data: bytes) -> bool:
    return len(data) >= 4 and data[0] == byte(0x50) and data[1] == byte(0x4b) and data[2] == byte(3) and data[3] == byte(4)

def load_bytes(data: bytes) -> Dict[str, Tensor]:
    """Load a torch.save zip checkpoint holding a state_dict (str -> Tensor)."""
    ents = _zip_entries(data)
    recs = Dict[str, _ZEntry]()
    pk = _ZEntry("", -1, 0)
    base = ""
    for e in ents:
        recs[e.name] = e
        if pk.start < 0 and e.name.endswith("/data.pkl") and e.name.count("/") == 1:
            pk = e
            base = e.name[:len(e.name) - 9]
    if pk.start < 0:
        raise RuntimeError("PytorchStreamReader failed locating file data.pkl")
    bo = base + "/byteorder"
    if bo in recs:
        r = recs[bo]
        if str(data._ptr + r.start, r.size) != "little":
            raise RuntimeError("torch.load: big-endian checkpoints are not supported")
    up = _Unpickler(data, pk, base, recs)
    top = up.run()
    if top.kind != _K_DICT:
        raise RuntimeError("torch.load: checkpoint top-level object is not a dict")
    out = Dict[str, Tensor]()
    for j in range(len(top.items)):
        k = top.items[j]
        v = top.vals[j]
        if k.kind != _K_STR or v.kind != _K_TENSOR:
            raise RuntimeError("torch.load: only Dict[str, Tensor] checkpoints are supported")
        out[k.s] = v.t.__val__()
    return out

# ---------------------------------------------------------------- pickle writer

class _PW:
    buf: List[u8]
    memo: Dict[str, int]   # memoized strings/globals by identity-equivalent key
    n: int

    def __init__(self):
        self.buf = List[u8]()
        self.memo = Dict[str, int]()
        self.n = 0

    def op(self, c: int):
        self.buf.append(u8(c))

    def u32(self, x: int):
        for j in range(4):
            self.buf.append(u8((x >> (8 * j)) & 0xff))

    def put(self) -> int:
        i = self.n
        self.n += 1
        if i < 256:
            self.op(0x71); self.op(i)
        else:
            self.op(0x72); self.u32(i)
        return i

    def get(self, i: int):
        if i < 256:
            self.op(0x68); self.op(i)
        else:
            self.op(0x6a); self.u32(i)

    def raw(self, s: str):
        for c in s:
            self.buf.append(u8(ord(c)))

    def ustr(self, s: str):
        b = s.encode()
        self.op(0x58); self.u32(len(b))
        for j in range(len(b)):
            self.buf.append(u8(int(b[j])))
        self.put()

    def shared_str(self, s: str):
        k = "s:" + s
        if k in self.memo:
            self.get(self.memo[k])
        else:
            b = s.encode()
            self.op(0x58); self.u32(len(b))
            for j in range(len(b)):
                self.buf.append(u8(int(b[j])))
            self.memo[k] = self.put()

    def glob(self, mod: str, name: str):
        k = "g:" + mod + " " + name
        if k in self.memo:
            self.get(self.memo[k])
        else:
            self.op(0x63); self.raw(mod + "\n" + name + "\n")
            self.memo[k] = self.put()

    def int_(self, x: int):
        if 0 <= x < 256:
            self.op(0x4b); self.op(x)
        elif 0 <= x < 65536:
            self.op(0x4d); self.op(x & 0xff); self.op(x >> 8)
        elif -0x80000000 <= x < 0x80000000:
            self.op(0x4a); self.u32(x & 0xffffffff)
        else:
            self.op(0x8a); self.op(8)
            for j in range(8):
                self.op((x >> (8 * j)) & 0xff)

    def int_tuple(self, xs: List[int]):
        if len(xs) == 0:
            self.op(0x29)
            return
        if len(xs) > 3:
            self.op(0x28)
        for x in xs:
            self.int_(x)
        if len(xs) > 3:
            self.op(0x74)
        else:
            self.op(0x84 + len(xs))
        self.put()

def _storage_name(dt: int) -> str:
    if dt == _F32: return "FloatStorage"
    if dt == _F64: return "DoubleStorage"
    if dt == _I64: return "LongStorage"
    if dt == _I32: return "IntStorage"
    if dt == _I16: return "ShortStorage"
    if dt == _I8: return "CharStorage"
    if dt == _U8: return "ByteStorage"
    if dt == _BOOL: return "BoolStorage"
    raise ValueError("torch.save: unsupported dtype")

# ---------------------------------------------------------------- zip writer

class _ZW:
    out: List[u8]
    cd: List[u8]
    count: int

    def __init__(self):
        self.out = List[u8]()
        self.cd = List[u8]()
        self.count = 0

def _put16(b: List[u8], x: int):
    b.append(u8(x & 0xff)); b.append(u8((x >> 8) & 0xff))

def _put32(b: List[u8], x: int):
    _put16(b, x & 0xffff); _put16(b, (x >> 16) & 0xffff)

def _put64(b: List[u8], x: int):
    _put32(b, x & 0xffffffff); _put32(b, (x >> 32) & 0xffffffff)

def _putstr(b: List[u8], s: str):
    for c in s:
        b.append(u8(ord(c)))

def _putblk(b: List[u8], p: Ptr[u8], n: int):
    # append n bytes with one grow and one memcpy (tensor payloads are MBs)
    if n <= 0:
        return
    m = len(b)
    if b._cap < m + n:
        c = max(m + n, (3 * b._cap) >> 1)
        b._resize(c)
    str.memcpy((b._ptr + m).as_byte(), p.as_byte(), n)
    b._len = m + n

def _crc(p: Ptr[u8], n: int) -> int:
    return crc32(bytes(p, n))

def _zw_record(z: _ZW, name: str, p: Ptr[u8], n: int):
    o = z.out
    off = len(o)
    crc = _crc(p, n)
    # local header (flags 0x0808: data descriptor + utf-8, like torch)
    _put32(o, 0x04034b50); _put16(o, 0); _put16(o, 0x0808); _put16(o, 0)
    _put16(o, 0); _put16(o, 0); _put32(o, 0); _put32(o, 0); _put32(o, 0)
    nl = len(name)
    pad = 64 - (off + 30 + nl + 4) % 64
    if pad == 64:
        pad = 0
    _put16(o, nl); _put16(o, 4 + pad)
    _putstr(o, name)
    _put16(o, 0x4246)  # "FB" padding extra field
    _put16(o, pad)
    for _ in range(pad):
        o.append(u8(0x5a))
    _putblk(o, p, n)
    # data descriptor
    _put32(o, 0x08074b50); _put32(o, crc); _put32(o, n); _put32(o, n)
    c = z.cd
    _put32(c, 0x02014b50); _put16(c, 0); _put16(c, 0); _put16(c, 0x0808); _put16(c, 0)
    _put16(c, 0); _put16(c, 0); _put32(c, crc); _put32(c, n); _put32(c, n)
    _put16(c, nl); _put16(c, 0); _put16(c, 0); _put16(c, 0); _put16(c, 0)
    _put32(c, 0); _put32(c, off)
    _putstr(c, name)
    z.count += 1

def _zw_str(z: _ZW, name: str, s: str):
    b = s.encode()
    _zw_record(z, name, b._ptr, len(b))

def _zw_finish(z: _ZW) -> bytes:
    o = z.out
    cdoff = len(o)
    _putblk(o, z.cd._ptr, len(z.cd))
    cdsize = len(z.cd)
    z64 = len(o)
    _put32(o, 0x06064b50); _put64(o, 44); _put16(o, 0x31e); _put16(o, 45)
    _put32(o, 0); _put32(o, 0); _put64(o, z.count); _put64(o, z.count)
    _put64(o, cdsize); _put64(o, cdoff)
    _put32(o, 0x07064b50); _put32(o, 0); _put64(o, z64); _put32(o, 1)
    _put32(o, 0x06054b50); _put16(o, 0); _put16(o, 0)
    _put16(o, min(z.count, 0xffff)); _put16(o, min(z.count, 0xffff))
    _put32(o, min(cdsize, 0xffffffff)); _put32(o, min(cdoff, 0xffffffff)); _put16(o, 0)
    return bytes(Ptr[byte](o._ptr), len(o))

def save_bytes(sd: Dict[str, Tensor], archive: str = "archive") -> bytes:
    """torch.save(state_dict) in the native zip+pickle format (OrderedDict, protocol 2)."""
    w = _PW()
    w.op(0x80); w.op(2)
    w.glob("collections", "OrderedDict")
    od = w.memo["g:collections OrderedDict"]
    w.op(0x29); w.op(0x52); w.put()
    sts = List[Tensor]()        # one representative per storage, in key order
    skey = Dict[int, int]()     # storage ptr -> key
    if len(sd) > 0:
        w.op(0x28)
    for k, t0 in sd.items():
        t = t0.detach()
        w.ustr(k)
        w.glob("torch._utils", "_rebuild_tensor_v2")
        w.op(0x28)
        w.op(0x28)
        w.shared_str("storage")
        w.glob("torch", _storage_name(t._dt))
        sid = t._st.ptr.__int__()
        if sid not in skey:
            skey[sid] = len(sts)
            sts.append(t)
        w.ustr(str(skey[sid]))
        w.shared_str("cpu")
        w.int_(t._st.nbytes // _itemsize(t._dt))
        w.op(0x74); w.put()
        w.op(0x51)
        w.int_(t._off)
        w.int_tuple(t._shape)
        w.int_tuple(t._strides)
        w.op(0x89)
        w.get(od); w.op(0x29); w.op(0x52); w.put()
        w.op(0x74); w.put()
        w.op(0x52); w.put()
    if len(sd) > 0:
        w.op(0x75)
    w.op(0x2e)
    z = _ZW()
    pk = w.buf
    _zw_record(z, archive + "/data.pkl", pk._ptr, len(pk))
    _zw_str(z, archive + "/.format_version", "1")
    _zw_str(z, archive + "/.storage_alignment", "64")
    _zw_str(z, archive + "/byteorder", "little")
    for i in range(len(sts)):
        st = sts[i]._st
        _zw_record(z, archive + "/data/" + str(i), Ptr[u8](st.ptr), st.nbytes)
    _zw_str(z, archive + "/version", "3\n")
    _zw_str(z, archive + "/.data/serialization_id", "0" * 40)
    return _zw_finish(z)
