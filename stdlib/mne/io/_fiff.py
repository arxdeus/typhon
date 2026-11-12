# Ported from MNE-Python 1.13.2 (BSD-3-Clause)
#
# FIF reading/writing: port of mne/_fiff/{constants,tag,tree,open,write}.py and
# the meas_info read/write path (subset: channel info, sfreq, filter bands,
# bads, line_freq, description, experimenter).

import numpy as np
from ._bytes import (_read_file, _i32_be, _f32_be, _f64_be, _ascii, _arr_be, _uint_be,
                     _ByteWriter)
from .._info import Info, _kind_to_type, _type_to_kind

# constants (mne/_fiff/constants.py)
FIFF_FILE_ID = 100
FIFF_DIR_POINTER = 101
FIFF_BLOCK_ID = 103
FIFF_BLOCK_START = 104
FIFF_BLOCK_END = 105
FIFF_FREE_LIST = 106
FIFF_NOP = 108
FIFF_NCHAN = 200
FIFF_SFREQ = 201
FIFF_DATA_PACK = 202
FIFF_CH_INFO = 203
FIFF_COMMENT = 206
FIFF_NAVE = 207
FIFF_FIRST_SAMPLE = 208
FIFF_LAST_SAMPLE = 209
FIFF_ASPECT_KIND = 210
FIFF_EXPERIMENTER = 212
FIFF_LOWPASS = 219
FIFF_HIGHPASS = 223
FIFF_NO_SAMPLES = 228
FIFF_FIRST_TIME = 229
FIFF_NAME = 233
FIFF_LINE_FREQ = 235
FIFF_DATA_BUFFER = 300
FIFF_DATA_SKIP = 301
FIFF_EPOCH = 302
FIFF_DATA_SKIP_SAMP = 303
FIFF_MNE_CH_NAME_LIST = 3507
FIFF_MNE_EVENT_LIST = 3561
FIFF_MNE_BASELINE_MIN = 3568
FIFF_MNE_BASELINE_MAX = 3569
FIFF_MNE_EPOCHS_SELECTION = 3800
FIFF_MNE_EPOCHS_DROP_LOG = 3801
FIFF_MNE_EPOCHS_RAW_SFREQ = 3803
FIFFB_ROOT = 999
FIFFB_MEAS = 100
FIFFB_MEAS_INFO = 101
FIFFB_RAW_DATA = 102
FIFFB_PROCESSED_DATA = 103
FIFFB_EVOKED = 104
FIFFB_ASPECT = 105
FIFFB_CONTINUOUS_DATA = 112
FIFFB_CH_INFO = 113
FIFFB_MNE_BAD_CHANNELS = 359
FIFFB_MNE_EVENTS = 361
FIFFB_MNE_EPOCHS = 373
FIFFB_MNE_ANNOTATIONS = 3810
FIFFT_BYTE = 1
FIFFT_SHORT = 2
FIFFT_INT = 3
FIFFT_FLOAT = 4
FIFFT_DOUBLE = 5
FIFFT_STRING = 10
FIFFT_DAU_PACK16 = 16
FIFFT_CH_INFO_STRUCT = 30
FIFFT_ID_STRUCT = 31
FIFFT_MATRIX = 0x40000000


class Tag:
    kind: int
    type: int
    size: int
    next: int
    pos: int

    def __init__(self, kind: int, type: int, size: int, next: int, pos: int):
        self.kind = kind
        self.type = type
        self.size = size
        self.next = next
        self.pos = pos


class Node:
    block: int
    entries: List[Tag]
    children: List[Node]

    def __init__(self, block: int):
        self.block = block
        self.entries = List[Tag]()
        self.children = List[Node]()

    def find(self, block: int) -> List[Node]:
        """dir_tree_find: all nodes (self included) with the given block kind."""
        out = List[Node]()
        if self.block == block:
            out.append(self)
        for c in self.children:
            for n in c.find(block):
                out.append(n)
        return out

    def tag(self, kind: int) -> Optional[Tag]:
        for t in self.entries:
            if t.kind == kind:
                return t
        return None


class FifFile:
    """An opened FIF file (fiff_open): raw bytes + directory tree."""
    b: bytes
    tree: Node

    def __init__(self, fname: str):
        self.b = _read_file(fname)
        b = self.b
        if len(b) < 16 or _i32_be(b, 0) != FIFF_FILE_ID:
            raise ValueError(f"file {fname} does not start with a file id tag")
        tags = List[Tag]()
        pos = 0
        n = len(b)
        while pos + 16 <= n:
            t = Tag(_i32_be(b, pos), _uint_be(b, pos + 4, 4), _i32_be(b, pos + 8),
                    _i32_be(b, pos + 12), pos)
            tags.append(t)
            if t.next == 0:
                pos += 16 + t.size
            elif t.next > 0:
                pos = t.next
            else:
                break
        # make_dir_tree
        root = Node(FIFFB_ROOT)
        stack = [root]
        for t in tags:
            if t.kind == FIFF_BLOCK_START:
                nd = Node(self.int(t))
                stack[-1].children.append(nd)
                stack.append(nd)
            elif t.kind == FIFF_BLOCK_END:
                if len(stack) > 1:
                    stack.pop()
            else:
                stack[-1].entries.append(t)
        self.tree = root

    # --- tag data decoding (read_tag) ---
    def int(self, t: Tag) -> int:
        return _i32_be(self.b, t.pos + 16)

    def ints(self, t: Tag) -> List[int]:
        return [_i32_be(self.b, t.pos + 16 + 4 * i) for i in range(t.size // 4)]

    def float(self, t: Tag) -> float:
        if t.type == FIFFT_DOUBLE:
            return _f64_be(self.b, t.pos + 16)
        return _f32_be(self.b, t.pos + 16)

    def floats(self, t: Tag) -> np.ndarray[float, 1]:
        if t.type == FIFFT_DOUBLE:
            return _arr_be(self.b, t.pos + 16, t.size // 8, "f8")
        if t.type == FIFFT_INT:
            return _arr_be(self.b, t.pos + 16, t.size // 4, "i4")
        return _arr_be(self.b, t.pos + 16, t.size // 4, "f4")

    def string(self, t: Tag) -> str:
        return _ascii(self.b, t.pos + 16, t.size)

    def matrix(self, t: Tag) -> Tuple[np.ndarray[float, 1], List[int]]:
        """Dense matrix tag: flat C-order data (float64) and dims (C order)."""
        if (t.type & FIFFT_MATRIX) == 0:
            raise ValueError("tag is not a matrix")
        dt = t.type & 0xFFFF
        end = t.pos + 16 + t.size
        ndim = _i32_be(self.b, end - 4)
        dims = [_i32_be(self.b, end - 4 * (ndim + 1) + 4 * i) for i in range(ndim)]
        dims.reverse()
        n = 1
        for d in dims:
            n *= d
        code = "f4"
        if dt == FIFFT_DOUBLE:
            code = "f8"
        elif dt == FIFFT_INT:
            code = "i4"
        elif dt != FIFFT_FLOAT:
            raise NotImplementedError(f"matrix data type {dt} not supported")
        return _arr_be(self.b, t.pos + 16, n, code), dims

    def buffer(self, t: Tag, nchan: int) -> np.ndarray[float, 2]:
        """FIFF_DATA_BUFFER as (nsamp, nchan) float64 (raw values, no cal)."""
        code = "f4"
        bs = 4
        if t.type == FIFFT_DOUBLE:
            code = "f8"
            bs = 8
        elif t.type == FIFFT_INT:
            code = "i4"
        elif t.type == FIFFT_SHORT or t.type == FIFFT_DAU_PACK16:
            code = "i2"
            bs = 2
        elif t.type != FIFFT_FLOAT:
            raise NotImplementedError(f"data buffer type {t.type} not supported")
        n = t.size // bs
        return _arr_be(self.b, t.pos + 16, n, code).reshape((n // nchan, nchan))


def _read_ch_info(f: FifFile, t: Tag, info: Info, i: int):
    b = f.b
    p = t.pos + 16
    kind = _i32_be(b, p + 8)
    rng = _f32_be(b, p + 12)
    cal = _f32_be(b, p + 16)
    coil = _i32_be(b, p + 20)
    loc = [_f32_be(b, p + 24 + 4 * k) for k in range(12)]
    unit = _i32_be(b, p + 72)
    name = _ascii(b, p + 80, 16)
    z = name.find("\0")
    if z >= 0:
        name = name[:z]
    info.ch_names[i] = name
    info.ch_types[i] = _kind_to_type(kind, unit)
    info.ranges[i] = rng
    info.cals[i] = cal
    info.coil_types[i] = coil
    info.locs[i] = loc
    info.units[i] = unit


def read_meas_info(f: FifFile) -> Info:
    meas = f.tree.find(FIFFB_MEAS)
    if len(meas) == 0:
        raise ValueError("Could not find measurement data")
    mi = meas[0].find(FIFFB_MEAS_INFO)
    if len(mi) == 0:
        raise ValueError("Could not find measurement info")
    node = mi[0]
    nchan = 0
    sfreq = 0.0
    chs = List[Tag]()
    for t in node.entries:
        if t.kind == FIFF_NCHAN:
            nchan = f.int(t)
        elif t.kind == FIFF_SFREQ:
            sfreq = f.float(t)
        elif t.kind == FIFF_CH_INFO:
            chs.append(t)
    if len(chs) != nchan:
        raise ValueError("Incorrect number of channel definitions found")
    info = Info([str(i) for i in range(nchan)], sfreq, ["misc" for _ in range(nchan)])
    for i in range(nchan):
        _read_ch_info(f, chs[i], info, i)
    info.lowpass = sfreq / 2.0
    for t in node.entries:
        if t.kind == FIFF_LOWPASS:
            v = f.float(t)
            if not np.isnan(v):
                info.lowpass = v
        elif t.kind == FIFF_HIGHPASS:
            v = f.float(t)
            if not np.isnan(v):
                info.highpass = v
        elif t.kind == FIFF_LINE_FREQ:
            info.line_freq = f.float(t)
        elif t.kind == FIFF_DESCRIPTION_KIND:
            info.description = f.string(t)
        elif t.kind == FIFF_EXPERIMENTER:
            info.experimenter = f.string(t)
    # extended channel info (long names)
    ext = node.find(FIFFB_CH_INFO)
    if len(ext) == nchan:
        for i in range(nchan):
            nt = ext[i].tag(FIFF_NAME)
            if nt is not None:
                info.ch_names[i] = f.string(nt)
    for bn in node.find(FIFFB_MNE_BAD_CHANNELS):
        bt = bn.tag(FIFF_MNE_CH_NAME_LIST)
        if bt is not None:
            s = f.string(bt)
            if len(s) > 0:
                info.bads = s.split(":")
    return info


FIFF_DESCRIPTION_KIND = 206


class FifWriter:
    """start_file/end_file + write_* tag helpers (mne/_fiff/write.py)."""
    w: _ByteWriter

    def __init__(self):
        self.w = _ByteWriter()
        self._head(FIFF_FILE_ID, FIFFT_ID_STRUCT, 20)
        self._id_body()
        self.write_int(FIFF_DIR_POINTER, -1)
        self.write_int(FIFF_FREE_LIST, -1)

    def _head(self, kind: int, typ: int, size: int, nxt: int = 0):
        self.w.i32(kind)
        self.w.i32(typ)
        self.w.i32(size)
        self.w.i32(nxt)

    def _id_body(self):
        self.w.i32(65540)
        for _ in range(4):
            self.w.i32(0)

    def write_id(self, kind: int):
        self._head(kind, FIFFT_ID_STRUCT, 20)
        self._id_body()

    def write_int(self, kind: int, v: int):
        self._head(kind, FIFFT_INT, 4)
        self.w.i32(v)

    def write_ints(self, kind: int, v: List[int]):
        self._head(kind, FIFFT_INT, 4 * len(v))
        for x in v:
            self.w.i32(x)

    def write_float(self, kind: int, v: float):
        self._head(kind, FIFFT_FLOAT, 4)
        self.w.f32(v)

    def write_floats(self, kind: int, v: List[float]):
        self._head(kind, FIFFT_FLOAT, 4 * len(v))
        for x in v:
            self.w.f32(x)

    def write_double(self, kind: int, v: float):
        self._head(kind, FIFFT_DOUBLE, 8)
        self.w.f64(v)

    def write_string(self, kind: int, s: str):
        self._head(kind, FIFFT_STRING, len(s))
        self.w.raw(s)

    def start_block(self, kind: int):
        self.write_int(FIFF_BLOCK_START, kind)

    def end_block(self, kind: int):
        self.write_int(FIFF_BLOCK_END, kind)

    def write_float_matrix(self, kind: int, flat: np.ndarray[float, 1], dims: List[int]):
        nd = len(dims)
        self._head(kind, FIFFT_FLOAT | FIFFT_MATRIX, 4 * flat.shape[0] + 4 * (nd + 1))
        for i in range(flat.shape[0]):
            self.w.f32(flat[i])
        for k in range(nd):
            self.w.i32(dims[nd - 1 - k])
        self.w.i32(nd)

    def write_buffer(self, data: np.ndarray[float, 2]):
        """FIFF_DATA_BUFFER (float) for data of shape (nsamp, nchan)."""
        n = data.shape[0] * data.shape[1]
        self._head(FIFF_DATA_BUFFER, FIFFT_FLOAT, 4 * n)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                self.w.f32(data[i, j])

    def write_ch_info(self, info: Info, k: int):
        self._head(FIFF_CH_INFO, FIFFT_CH_INFO_STRUCT, 4 * 13 + 4 * 7 + 16)
        self.w.i32(k + 1)
        self.w.i32(k + 1)
        self.w.i32(_type_to_kind(info.ch_types[k]))
        self.w.f32(1.0)  # reset_range
        self.w.f32(info.cals[k])
        self.w.i32(info.coil_types[k])
        for v in info.locs[k]:
            self.w.f32(v)
        self.w.i32(info.units[k])
        self.w.i32(0)
        name = info.ch_names[k]
        if len(name) > 15:
            raise ValueError(f"Typhon mne: channel names longer than 15 characters cannot be written to FIF ({name})")
        self.w.raw(name)
        self.w.zeros(16 - len(name))

    def write_meas_info(self, info: Info, data_type: int = 0):
        self.start_block(FIFFB_MEAS_INFO)
        if len(info.bads) > 0:
            self.start_block(FIFFB_MNE_BAD_CHANNELS)
            self.write_string(FIFF_MNE_CH_NAME_LIST, ":".join(info.bads))
            self.end_block(FIFFB_MNE_BAD_CHANNELS)
        if info.experimenter is not None:
            self.write_string(FIFF_EXPERIMENTER, info.experimenter.__val__())
        if info.description is not None:
            self.write_string(FIFF_COMMENT, info.description.__val__())
        self.write_int(FIFF_NCHAN, info.nchan)
        self.write_float(FIFF_SFREQ, info.sfreq)
        self.write_float(FIFF_LOWPASS, info.lowpass)
        self.write_float(FIFF_HIGHPASS, info.highpass)
        if info.line_freq is not None:
            self.write_float(FIFF_LINE_FREQ, info.line_freq.__val__())
        if data_type != 0:
            self.write_int(FIFF_DATA_PACK, data_type)
        for k in range(info.nchan):
            self.write_ch_info(info, k)
        self.end_block(FIFFB_MEAS_INFO)

    def write_annotations(self, onset: List[float], duration: List[float], desc: List[str]):
        self.start_block(FIFFB_MNE_ANNOTATIONS)
        self.write_floats(FIFF_MNE_BASELINE_MIN, onset)
        self.write_floats(FIFF_MNE_BASELINE_MAX, [onset[i] + duration[i] for i in range(len(onset))])
        self.write_string(FIFF_COMMENT, ":".join([d.replace(":", ";") for d in desc]))
        self.end_block(FIFFB_MNE_ANNOTATIONS)

    def end_file(self, fname: str):
        self._head(FIFF_NOP, 0, 0, -1)
        self.w.save(fname)


def read_annotations_node(f: FifFile, node: Node) -> Tuple[List[float], List[float], List[str], bool]:
    """_read_annotations_fif: (onset, duration, description, found)."""
    an = node.find(FIFFB_MNE_ANNOTATIONS)
    on = List[float]()
    du = List[float]()
    de = List[str]()
    if len(an) == 0:
        return on, du, de, False
    nd = an[0]
    ends = List[float]()
    for t in nd.entries:
        if t.kind == FIFF_MNE_BASELINE_MIN:
            on = list(f.floats(t))
        elif t.kind == FIFF_MNE_BASELINE_MAX:
            ends = list(f.floats(t))
        elif t.kind == FIFF_COMMENT:
            s = f.string(t)
            de = [x.replace(";", ":") for x in s.split(":")] if len(s) > 0 else List[str]()
    for i in range(len(on)):
        du.append(ends[i] - on[i])
    return on, du, de, True


def _parse_drop_log(s: str) -> List[List[str]]:
    """json.loads for a drop log: list of lists of strings."""
    out = List[List[str]]()
    i = 0
    n = len(s)
    depth = 0
    cur = List[str]()
    while i < n:
        c = s[i]
        if c == "[":
            depth += 1
            if depth == 2:
                cur = List[str]()
        elif c == "]":
            if depth == 2:
                out.append(cur)
            depth -= 1
        elif c == '"':
            j = i + 1
            buf = ""
            while s[j] != '"':
                if s[j] == "\\":
                    j += 1
                buf += s[j]
                j += 1
            cur.append(buf)
            i = j
        i += 1
    return out


def _dump_drop_log(log: List[List[str]]) -> str:
    parts = List[str]()
    for e in log:
        parts.append("[" + ", ".join(['"' + x.replace("\\", "\\\\").replace('"', '\\"') + '"' for x in e]) + "]")
    return "[" + ", ".join(parts) + "]"
