# Ported from SciPy 1.18.1 (BSD-3-Clause), scipy/io/matlab/_mio5.py
#
# Minimal MATLAB 5 (.mat v5/v6/v7 uncompressed) reader. Compressed
# (miCOMPRESSED, zlib) elements and v7.3 (HDF5) files are not supported.
#
# Because Typhon is statically typed, loadmat returns Dict[str, MatVar]
# where MatVar is a typed tree (numeric / char / struct / cell / logical)
# instead of scipy's object ndarrays. MatVar.array() gives the float64 data
# (shape as in MATLAB, column-major on disk converted to C order), str()
# the char data, MatVar[field] / .get(i, field) struct fields and
# .items cell / struct elements.

import numpy as np

miINT8 = 1
miUINT8 = 2
miINT16 = 3
miUINT16 = 4
miINT32 = 5
miUINT32 = 6
miSINGLE = 7
miDOUBLE = 9
miINT64 = 12
miUINT64 = 13
miMATRIX = 14
miCOMPRESSED = 15
miUTF8 = 16
miUTF16 = 17
miUTF32 = 18

mxCELL_CLASS = 1
mxSTRUCT_CLASS = 2
mxOBJECT_CLASS = 3
mxCHAR_CLASS = 4
mxSPARSE_CLASS = 5


def _ule(b: bytes, off: int, n: int) -> int:
    u = 0
    for k in range(n):
        u = (u << 8) | int(b._ptr[off + n - 1 - k])
    return u


def _sle(b: bytes, off: int, n: int) -> int:
    u = _ule(b, off, n)
    if n < 8 and u >= (1 << (8 * n - 1)):
        return u - (1 << (8 * n))
    return u


def _decode(b: bytes, off: int, typ: int, nbytes: int) -> np.ndarray[float, 1]:
    sz = 1
    if typ in (miINT16, miUINT16):
        sz = 2
    elif typ in (miINT32, miUINT32, miSINGLE, miUTF32):
        sz = 4
    elif typ in (miDOUBLE, miINT64, miUINT64):
        sz = 8
    n = nbytes // sz
    out = np.empty(n, dtype=float)
    for i in range(n):
        p = off + i * sz
        if typ == miDOUBLE:
            x = u64(_ule(b, p, 8))
            out[i] = Ptr[float](__ptr__(x).as_byte())[0]
        elif typ == miSINGLE:
            x = u32(_ule(b, p, 4))
            out[i] = float(Ptr[float32](__ptr__(x).as_byte())[0])
        elif typ in (miINT8, miINT16, miINT32, miINT64):
            out[i] = float(_sle(b, p, sz))
        else:
            out[i] = float(_ule(b, p, sz))
    return out


class MatVar:
    """A MATLAB value: numeric/logical (data), char (text), struct, cell."""
    cls: int
    dims: List[int]
    name: str
    data: np.ndarray[float, 1]
    text: str
    fields: List[str]
    items: List[MatVar]

    def __init__(self):
        self.cls = 0
        self.dims = List[int]()
        self.name = ""
        self.data = np.zeros(0)
        self.text = ""
        self.fields = List[str]()
        self.items = List[MatVar]()

    @property
    def size(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def shape(self) -> List[int]:
        return self.dims

    def is_char(self) -> bool:
        return self.cls == mxCHAR_CLASS

    def is_struct(self) -> bool:
        return self.cls == mxSTRUCT_CLASS or self.cls == mxOBJECT_CLASS

    def is_cell(self) -> bool:
        return self.cls == mxCELL_CLASS

    def is_numeric(self) -> bool:
        return not (self.is_char() or self.is_struct() or self.is_cell())

    def __str__(self) -> str:
        return self.text

    def scalar(self) -> float:
        if self.data.shape[0] == 0:
            return np.nan
        return float(self.data[0])

    def array(self) -> np.ndarray[float, 2]:
        """2-D float64 array (MATLAB rows x cols); column-major input -> C order."""
        r = self.dims[0] if len(self.dims) > 0 else 0
        c = self.size // r if r > 0 else 0
        out = np.zeros((r, c))
        for j in range(c):
            for i in range(r):
                out[i, j] = self.data[j * r + i]
        return out

    def get(self, idx: int, field: str) -> MatVar:
        """struct(idx).field (idx in column-major element order)."""
        if field not in self.fields:
            raise KeyError(field)
        return self.items[idx * len(self.fields) + self.fields.index(field)]

    def has(self, field: str) -> bool:
        return field in self.fields

    def __getitem__(self, field: str) -> MatVar:
        return self.get(0, field)


def _read_tag(b: bytes, pos: int) -> Tuple[int, int, int, int]:
    """(type, nbytes, data_offset, next_pos)."""
    w = _ule(b, pos, 4)
    if (w >> 16) != 0:
        return w & 0xFFFF, w >> 16, pos + 4, pos + 8
    nb = _ule(b, pos + 4, 4)
    nxt = pos + 8 + nb
    if nxt % 8 != 0:
        nxt += 8 - nxt % 8
    return w, nb, pos + 8, nxt


def _read_matrix(b: bytes, start: int, end: int, in_struct: bool) -> MatVar:
    v = MatVar()
    if end - start == 0:
        v.cls = 6
        v.dims = [0, 0]
        return v
    t, nb, off, pos = _read_tag(b, start)  # array flags
    flags = _ule(b, off, 4)
    v.cls = flags & 0xFF
    t, nb, off, pos = _read_tag(b, pos)  # dims
    v.dims = [_sle(b, off + 4 * i, 4) for i in range(nb // 4)]
    t, nb, off, pos = _read_tag(b, pos)  # name
    v.name = str(b._ptr + off, nb)
    if v.cls == mxSPARSE_CLASS:
        raise NotImplementedError("Typhon scipy.io.loadmat: sparse matrices are not supported")
    if v.cls == mxCELL_CLASS:
        for _ in range(v.size):
            t, nb, off, nxt = _read_tag(b, pos)
            v.items.append(_read_matrix(b, off, off + nb, False))
            pos = nxt
    elif v.cls == mxSTRUCT_CLASS or v.cls == mxOBJECT_CLASS:
        if v.cls == mxOBJECT_CLASS:
            t, nb, off, pos = _read_tag(b, pos)  # class name
        t, nb, off, pos = _read_tag(b, pos)
        flen = _ule(b, off, 4)
        t, nb, off, pos = _read_tag(b, pos)
        nf = nb // flen
        for k in range(nf):
            s = str(b._ptr + off + k * flen, flen)
            z = s.find("\0")
            v.fields.append(s[:z] if z >= 0 else s)
        for _ in range(v.size):
            for _f in range(nf):
                t, nb, off, nxt = _read_tag(b, pos)
                v.items.append(_read_matrix(b, off, off + nb, True))
                pos = nxt
    elif v.cls == mxCHAR_CLASS:
        t, nb, off, pos = _read_tag(b, pos)
        codes = List[int]()
        if t == miUTF8 or t == miINT8 or t == miUINT8:
            for i in range(nb):
                codes.append(int(b._ptr[off + i]))
            s = str(b._ptr + off, nb)
            v.text = _char_rows(s, v.dims)
        else:
            vals = _decode(b, off, t, nb)
            chars = "".join([chr(int(vals[i])) for i in range(vals.shape[0])])
            v.text = _char_rows(chars, v.dims)
    else:
        t, nb, off, pos = _read_tag(b, pos)
        v.data = _decode(b, off, t, nb)
    return v


def _char_rows(s: str, dims: List[int]) -> str:
    # MATLAB char arrays are column-major; join rows of a single-row array
    if len(dims) >= 2 and dims[0] > 1:
        r = dims[0]
        c = len(s) // r
        rows = ["".join([s[j * r + i] for j in range(c)]) for i in range(r)]
        return "\n".join(rows)
    return s


def loadmat(file_name, mdict = None, appendmat: bool = True, **kwargs) -> Dict[str, MatVar]:
    fname = str(file_name)
    with open(fname, "rb") as f:
        b = f.read()
    if len(b) < 128:
        raise ValueError("Mat file appears to be empty")
    if str(b._ptr, 6) != "MATLAB":
        raise NotImplementedError("Typhon scipy.io.loadmat: only MATLAB 5 files are supported")
    if str(b._ptr + 126, 2) != "IM":
        raise NotImplementedError("Typhon scipy.io.loadmat: big-endian MAT files are not supported")
    if _ule(b, 124, 2) == 0x0200:
        raise NotImplementedError("Please use HDF reader for matlab v7.3 files, e.g. h5py")
    out = Dict[str, MatVar]()
    pos = 128
    while pos + 8 <= len(b):
        t, nb, off, nxt = _read_tag(b, pos)
        if t == miCOMPRESSED:
            raise NotImplementedError("Typhon scipy.io.loadmat: compressed (zlib) MAT variables are not supported")
        if t == miMATRIX:
            v = _read_matrix(b, off, off + nb, False)
            out[v.name] = v
        pos = nxt
    return out
