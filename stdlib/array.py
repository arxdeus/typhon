# Ported from CPython 3.14 array (PSF license), simplified: typecode is a compile-time-free
# runtime string; element storage is int64 for integer codes and float64 for 'f'/'d'.
# Integer codes wrap values to their C width on store.

typecodes = "bBuhHiIlLqQfd"

def _itemsize(tc: str) -> int:
    if tc == "b" or tc == "B":
        return 1
    if tc == "h" or tc == "H":
        return 2
    if tc == "i" or tc == "I" or tc == "f":
        return 4
    return 8

def _wrap(tc: str, v: int) -> int:
    sz = _itemsize(tc)
    if sz == 8:
        return v
    bits = sz * 8
    m = v & ((1 << bits) - 1)
    if tc == "b" or tc == "h" or tc == "i" or tc == "l" or tc == "q":
        if m >= (1 << (bits - 1)):
            m -= 1 << bits
    return m

def _check(tc: str, v: int):
    sz = _itemsize(tc)
    if sz == 8:
        return
    bits = sz * 8
    signed = tc == "b" or tc == "h" or tc == "i"
    lo = -(1 << (bits - 1)) if signed else 0
    hi = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
    if v < lo:
        raise OverflowError("signed integer is less than minimum" if signed else "unsigned byte integer is less than minimum")
    if v > hi:
        raise OverflowError("signed integer is greater than maximum" if signed else "unsigned integer is greater than maximum")

class array:
    typecode: str
    _data: List[T]
    T: type

    def __init__(self, typecode: str):
        if typecode not in typecodes or len(typecode) != 1:
            raise ValueError("bad typecode (must be b, B, u, h, H, i, I, l, L, q, Q, f or d)")
        self.typecode = typecode
        self._data = []

    def __init__(self, typecode: str, initializer: List[T]):
        self.__init__(typecode)
        for v in initializer:
            self.append(v)

    def __init__(self, typecode: str, initializer: bytes):
        self.__init__(typecode)
        self.frombytes(initializer)

    def _isfloat(self) -> bool:
        return self.typecode == "f" or self.typecode == "d"

    @property
    def itemsize(self) -> int:
        return 8 if self.typecode == "d" else _itemsize(self.typecode)

    def __len__(self) -> int:
        return len(self._data)

    def _conv(self, v):
        if T is float:
            if self.typecode == "f":
                return float(f32(float(v)))
            return float(v)
        else:
            if not isinstance(v, int):
                compile_error("an integer is required")
            _check(self.typecode, v)
            return v

    def append(self, v):
        self._data.append(self._conv(v))

    def extend(self, it):
        for v in it:
            self.append(v)

    def __getitem__(self, i: int) -> T:
        n = len(self._data)
        if i < 0:
            i += n
        if i < 0 or i >= n:
            raise IndexError("array index out of range")
        return self._data[i]

    def __setitem__(self, i: int, v):
        n = len(self._data)
        if i < 0:
            i += n
        if i < 0 or i >= n:
            raise IndexError("array assignment index out of range")
        self._data[i] = self._conv(v)

    def pop(self, i: int = -1) -> T:
        n = len(self._data)
        if n == 0:
            raise IndexError("pop from empty array")
        if i < 0:
            i += n
        if i < 0 or i >= n:
            raise IndexError("pop index out of range")
        return self._data.pop(i)

    def reverse(self):
        self._data.reverse()

    def tolist(self) -> List[T]:
        return self._data.copy()

    def buffer_info(self) -> Tuple[int, int]:
        return (0, len(self._data))

    def tobytes(self) -> bytes:
        out = List[int]()
        tc = self.typecode
        sz = self.itemsize
        if T is float:
            for f in self._data:
                if tc == "d":
                    x = f
                    p = Ptr[u8](__ptr__(x).as_byte())
                    for k in range(8):
                        out.append(int(p[k]))
                else:
                    y = f32(f)
                    p = Ptr[u8](__ptr__(y).as_byte())
                    for k in range(4):
                        out.append(int(p[k]))
        else:
            for v in self._data:
                u = v & ((1 << (sz * 8)) - 1) if sz < 8 else v
                for k in range(sz):
                    out.append((u >> (8 * k)) & 255)
        return bytes(out)

    def frombytes(self, b: bytes):
        sz = self.itemsize
        if len(b) % sz != 0:
            raise ValueError("bytes length not a multiple of item size")
        tc = self.typecode
        for i in range(0, len(b), sz):
            if T is float:
                if tc == "d":
                    x = 0.0
                    p = Ptr[u8](__ptr__(x).as_byte())
                    for k in range(8):
                        p[k] = u8(int(b[i + k]))
                    self._data.append(x)
                else:
                    y = f32(0.0)
                    p = Ptr[u8](__ptr__(y).as_byte())
                    for k in range(4):
                        p[k] = u8(int(b[i + k]))
                    self._data.append(float(y))
            else:
                u = 0
                for k in range(sz):
                    u |= int(b[i + k]) << (8 * k)
                self._data.append(_wrap(tc, u))

    def byteswap(self):
        sz = self.itemsize
        raw = self.tobytes()
        out = List[int]()
        for i in range(0, len(raw), sz):
            for k in range(sz - 1, -1, -1):
                out.append(int(raw[i + k]))
        self._data = []
        self.frombytes(bytes(out))

    def count(self, v: T) -> int:
        c = 0
        for x in self._data:
            if x == v:
                c += 1
        return c

    def index(self, v: T) -> int:
        for i in range(len(self._data)):
            if self._data[i] == v:
                return i
        raise ValueError("array.index(x): x not in array")

    def __iter__(self) -> Generator[T]:
        for x in self._data:
            yield x

    def __repr__(self) -> str:
        if len(self._data) == 0:
            return "array('" + self.typecode + "')"
        return "array('" + self.typecode + "', " + str(self._data) + ")"
