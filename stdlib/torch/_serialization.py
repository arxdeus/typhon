# Ported from torch 2.15.0 (BSD-3-Clause)
#
# Tensor serialization.
#
# - safetensors format (https://github.com/huggingface/safetensors): 8-byte
#   little-endian header length, JSON header
#   {"name": {"dtype": "F32", "shape": [...], "data_offsets": [b, e]}, ...,
#    "__metadata__": {...}}, then raw little-endian tensor bytes.
#   Supported dtypes: F64, F32, I64, I32, I16, I8, U8, BOOL.
# - torch.save / torch.load use the native torch zip+pickle checkpoint
#   format (see _pt_zip.py) for a Dict[str, Tensor] (state_dict); torch.load
#   also accepts safetensors files (detected by the zip magic).

from torch._tensor import Tensor, _new, _BOOL, _U8, _I8, _I16, _I32, _I64, _F32, _F64, _itemsize, _contig_as

def _st_name(dt: int) -> str:
    if dt == _F64: return "F64"
    if dt == _F32: return "F32"
    if dt == _I64: return "I64"
    if dt == _I32: return "I32"
    if dt == _I16: return "I16"
    if dt == _I8: return "I8"
    if dt == _U8: return "U8"
    if dt == _BOOL: return "BOOL"
    raise ValueError("safetensors: unsupported dtype")

def _st_code(name: str) -> int:
    if name == "F64": return _F64
    if name == "F32": return _F32
    if name == "I64": return _I64
    if name == "I32": return _I32
    if name == "I16": return _I16
    if name == "I8": return _I8
    if name == "U8": return _U8
    if name == "BOOL": return _BOOL
    raise ValueError("safetensors: unsupported dtype " + name)

# ---------------------------------------------------------------- tiny JSON

class _P:
    s: str
    i: int

    def __init__(self, s: str):
        self.s = s
        self.i = 0

    def ws(self):
        while self.i < len(self.s) and self.s[self.i] in " \t\r\n":
            self.i += 1

    def expect(self, c: str):
        self.ws()
        if self.i >= len(self.s) or self.s[self.i] != c:
            raise ValueError("safetensors: malformed header (expected '" + c + "' at " + str(self.i) + ")")
        self.i += 1

    def peek(self) -> str:
        self.ws()
        return self.s[self.i] if self.i < len(self.s) else ""

    def string(self) -> str:
        self.expect('"')
        out = List[str]()
        while self.s[self.i] != '"':
            c = self.s[self.i]
            if c == "\\":
                self.i += 1
                e = self.s[self.i]
                if e == "n":
                    out.append("\n")
                elif e == "t":
                    out.append("\t")
                elif e == "u":
                    code = int(self.s[self.i + 1:self.i + 5], 16)
                    out.append(chr(code))
                    self.i += 4
                else:
                    out.append(e)
            else:
                out.append(c)
            self.i += 1
        self.i += 1
        return "".join(out)

    def integer(self) -> int:
        self.ws()
        j = self.i
        while self.i < len(self.s) and (self.s[self.i].isdigit() or self.s[self.i] == "-"):
            self.i += 1
        return int(self.s[j:self.i])

    def int_list(self) -> List[int]:
        self.expect("[")
        out = List[int]()
        if self.peek() == "]":
            self.i += 1
            return out
        while True:
            out.append(self.integer())
            if self.peek() == ",":
                self.i += 1
            else:
                break
        self.expect("]")
        return out

    def skip_value(self):
        c = self.peek()
        if c == '"':
            self.string()
        elif c == "{" or c == "[":
            close = "}" if c == "{" else "]"
            depth = 0
            while True:
                ch = self.s[self.i]
                if ch == '"':
                    self.string()
                    continue
                if ch == "{" or ch == "[":
                    depth += 1
                elif ch == "}" or ch == "]":
                    depth -= 1
                self.i += 1
                if depth == 0:
                    break
        else:
            while self.i < len(self.s) and self.s[self.i] not in ",}]":
                self.i += 1

    def str_map(self) -> Dict[str, str]:
        out = Dict[str, str]()
        self.expect("{")
        if self.peek() == "}":
            self.i += 1
            return out
        while True:
            k = self.string()
            self.expect(":")
            if self.peek() == '"':
                out[k] = self.string()
            else:
                self.skip_value()
            if self.peek() == ",":
                self.i += 1
            else:
                break
        self.expect("}")
        return out

@tuple
class _Entry:
    name: str
    dtype: str
    shape: List[int]
    begin: int
    end: int

def _parse_header(h: str) -> Tuple[List[_Entry], Dict[str, str]]:
    p = _P(h)
    entries = List[_Entry]()
    meta = Dict[str, str]()
    p.expect("{")
    if p.peek() == "}":
        return (entries, meta)
    while True:
        name = p.string()
        p.expect(":")
        if name == "__metadata__":
            meta = p.str_map()
        else:
            p.expect("{")
            dtype = ""
            shape = List[int]()
            offs = List[int]()
            while True:
                k = p.string()
                p.expect(":")
                if k == "dtype":
                    dtype = p.string()
                elif k == "shape":
                    shape = p.int_list()
                elif k == "data_offsets":
                    offs = p.int_list()
                else:
                    p.skip_value()
                if p.peek() == ",":
                    p.i += 1
                else:
                    break
            p.expect("}")
            if len(offs) != 2:
                raise ValueError("safetensors: missing data_offsets for " + name)
            entries.append(_Entry(name, dtype, shape, offs[0], offs[1]))
        if p.peek() == ",":
            p.i += 1
        else:
            break
    p.expect("}")
    return (entries, meta)

def _json_str(s: str) -> str:
    out = List[str]()
    out.append('"')
    for c in s:
        if c == '"':
            out.append('\\"')
        elif c == "\\":
            out.append("\\\\")
        elif c == "\n":
            out.append("\\n")
        else:
            out.append(c)
    out.append('"')
    return "".join(out)

# ---------------------------------------------------------------- bytes <-> tensors

def load_bytes(data: bytes) -> Dict[str, Tensor]:
    """Parse a safetensors byte string into an ordered Dict[str, Tensor]."""
    if len(data) < 8:
        raise ValueError("safetensors: file too small")
    p = data._ptr
    n = 0
    for i in range(8):
        n |= int(p[i]) << (8 * i)
    if 8 + n > len(data):
        raise ValueError("safetensors: invalid header length")
    header = _ascii(p + 8, n)
    entries, meta = _parse_header(header)
    base = p + 8 + n
    avail = len(data) - 8 - n
    out = Dict[str, Tensor]()
    for e in entries:
        dt = _st_code(e.dtype)
        t = _new(e.shape.copy(), dt)
        nbytes = t.numel() * _itemsize(dt)
        if e.end - e.begin != nbytes or e.end > avail:
            raise ValueError("safetensors: bad data_offsets for " + e.name)
        dst = Ptr[u8](t._st.ptr)
        src = base + e.begin
        str.memcpy(dst.as_byte(), src.as_byte(), nbytes)
        out[e.name] = t
    return out

def save_bytes(tensors: Dict[str, Tensor], metadata: Optional[Dict[str, str]] = None) -> bytes:
    """Serialize tensors to safetensors bytes (keys sorted like safetensors)."""
    names = sorted(list(tensors.keys()))
    parts = List[str]()
    off = 0
    hdr = List[str]()
    if metadata is not None:
        md = metadata.__val__()
        items = List[str]()
        for k in sorted(list(md.keys())):
            items.append(_json_str(k) + ":" + _json_str(md[k]))
        hdr.append('"__metadata__":{' + ",".join(items) + "}")
    blobs = List[Tensor]()
    for k in names:
        t = _contig_as(tensors[k].detach(), tensors[k]._dt)
        nb = t.numel() * _itemsize(t._dt)
        shape = ",".join([str(s) for s in t._shape])
        hdr.append(_json_str(k) + ':{"dtype":"' + _st_name(t._dt) + '","shape":[' + shape + '],"data_offsets":[' + str(off) + "," + str(off + nb) + "]}")
        off += nb
        blobs.append(t)
    h = "{" + ",".join(hdr) + "}"
    # pad header to 8-byte alignment with spaces (as safetensors does)
    while len(h) % 8 != 0:
        h += " "
    hb = h.encode()
    L = len(hb)
    total = 8 + L + off
    buf = Ptr[u8](total)
    for i in range(8):
        buf[i] = u8((L >> (8 * i)) & 0xff)
    str.memcpy((buf + 8).as_byte(), hb._ptr, L)
    pos = 8 + L
    for t in blobs:
        nb = t.numel() * _itemsize(t._dt)
        src = Ptr[u8](t._st.ptr) + t._off * _itemsize(t._dt)
        str.memcpy((buf + pos).as_byte(), src.as_byte(), nb)
        pos += nb
    return bytes(buf, total)

def _read_file(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def _write_file(path: str, data: bytes):
    with open(path, "wb") as f:
        f.write(data)

def _ascii(p: Ptr[u8], n: int) -> str:
    return bytes(p, n).decode()

def load_file(filename: str, device: str = "cpu") -> Dict[str, Tensor]:
    return load_bytes(_read_file(filename))

def save_file(tensors: Dict[str, Tensor], filename: str, metadata: Optional[Dict[str, str]] = None):
    _write_file(filename, save_bytes(tensors, metadata))

def _archive_name(f: str) -> str:
    # torch uses the file's basename without extension as the zip root
    b = f
    i = b.rfind("/")
    if i >= 0:
        b = b[i + 1:]
    j = b.rfind(".")
    if j > 0:
        b = b[:j]
    return b if b != "" else "archive"

def save(obj, f):
    """torch.save for a state_dict (any str -> Tensor mapping) in the native
    torch zip+pickle format; f is a str or path-like."""
    from torch._pt_zip import save_bytes as _pt_save
    p = str(f)
    d = Dict[str, Tensor]()
    for k, v in obj.items():
        d[k] = v
    _write_file(p, _pt_save(d, _archive_name(p)))

def load(f, map_location = None, weights_only: bool = True) -> Dict[str, Tensor]:
    """torch.load of a native torch checkpoint (zip+pickle state_dict) or a
    safetensors file (sniffed by magic)."""
    from torch._pt_zip import is_torch_zip, load_bytes as _pt_load
    data = _read_file(str(f))
    if is_torch_zip(data):
        return _pt_load(data)
    return load_bytes(data)
