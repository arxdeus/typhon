SEEK_SET: Literal[int] = 0
SEEK_CUR: Literal[int] = 1
SEEK_END: Literal[int] = 2
DEFAULT_BUFFER_SIZE: Literal[int] = 131072

UnsupportedOperation = OSError

from internal.file import open


class StringIO:
    _buf: str
    _pos: int
    _closed: bool

    def __init__(self, initial_value: str = "", newline: Optional[str] = "\n"):
        self._buf = initial_value
        self._pos = 0
        self._closed = False

    def _check(self):
        if self._closed:
            raise ValueError("I/O operation on closed file")

    @property
    def closed(self) -> bool:
        return self._closed

    def getvalue(self) -> str:
        self._check()
        return self._buf

    def write(self, s: str) -> int:
        self._check()
        n = len(s)
        if n == 0:
            return 0
        if self._pos > len(self._buf):
            self._buf = self._buf + "\0" * (self._pos - len(self._buf))
        if self._pos == len(self._buf):
            self._buf = self._buf + s
        else:
            self._buf = self._buf[:self._pos] + s + self._buf[self._pos + n:]
        self._pos += n
        return n

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def read(self, size: int = -1) -> str:
        self._check()
        if self._pos >= len(self._buf):
            return ""
        end = len(self._buf) if size < 0 else min(len(self._buf), self._pos + size)
        r = self._buf[self._pos:end]
        self._pos = end
        return r

    def readline(self, size: int = -1) -> str:
        self._check()
        if self._pos >= len(self._buf):
            return ""
        i = self._buf.find("\n", self._pos)
        end = len(self._buf) if i < 0 else i + 1
        if size >= 0 and self._pos + size < end:
            end = self._pos + size
        r = self._buf[self._pos:end]
        self._pos = end
        return r

    def readlines(self, hint: int = -1) -> List[str]:
        out = List[str]()
        total = 0
        while True:
            line = self.readline()
            if not line:
                break
            out.append(line)
            total += len(line)
            if hint > 0 and total >= hint:
                break
        return out

    def __iter__(self) -> Generator[str]:
        while True:
            line = self.readline()
            if not line:
                break
            yield line

    def tell(self) -> int:
        self._check()
        return self._pos

    def seek(self, pos: int, whence: int = 0) -> int:
        self._check()
        if whence == 0:
            if pos < 0:
                raise ValueError(f"Negative seek position {pos}")
            self._pos = pos
        elif whence == 1:
            if pos != 0:
                raise OSError("Can't do nonzero cur-relative seeks")
        elif whence == 2:
            if pos != 0:
                raise OSError("Can't do nonzero end-relative seeks")
            self._pos = len(self._buf)
        else:
            raise ValueError(f"Invalid whence ({whence}, should be 0, 1 or 2)")
        return self._pos

    def truncate(self, size: Optional[int] = None) -> int:
        self._check()
        n = self._pos if size is None else size
        if n < 0:
            raise ValueError(f"Negative size value {n}")
        if n < len(self._buf):
            self._buf = self._buf[:n]
        return n

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def flush(self):
        pass

    def close(self):
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()


class BytesIO:
    _buf: List[int]
    _pos: int
    _closed: bool

    def __init__(self, initial_bytes: bytes = b""):
        self._buf = [int(b) for b in initial_bytes]
        self._pos = 0
        self._closed = False

    def _check(self):
        if self._closed:
            raise ValueError("I/O operation on closed file")

    def _slice(self, a: int, b: int) -> bytes:
        n = b - a
        p = Ptr[u8](n if n > 0 else 1)
        for i in range(n):
            p[i] = u8(self._buf[a + i])
        return bytes(p, n)

    @property
    def closed(self) -> bool:
        return self._closed

    def getvalue(self) -> bytes:
        self._check()
        return self._slice(0, len(self._buf))

    def write(self, b: bytes) -> int:
        self._check()
        n = len(b)
        while len(self._buf) < self._pos:
            self._buf.append(0)
        for i in range(n):
            j = self._pos + i
            if j < len(self._buf):
                self._buf[j] = int(b[i])
            else:
                self._buf.append(int(b[i]))
        self._pos += n
        return n

    def write(self, b: bytearray) -> int:
        return self.write(bytes(b))

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def read(self, size: int = -1) -> bytes:
        self._check()
        if self._pos >= len(self._buf):
            return b""
        end = len(self._buf) if size < 0 else min(len(self._buf), self._pos + size)
        r = self._slice(self._pos, end)
        self._pos = end
        return r

    def read1(self, size: int = -1) -> bytes:
        return self.read(size)

    def readline(self, size: int = -1) -> bytes:
        self._check()
        if self._pos >= len(self._buf):
            return b""
        end = self._pos
        while end < len(self._buf):
            end += 1
            if self._buf[end - 1] == 10:
                break
        if size >= 0 and self._pos + size < end:
            end = self._pos + size
        r = self._slice(self._pos, end)
        self._pos = end
        return r

    def readlines(self) -> List[bytes]:
        out = List[bytes]()
        while True:
            line = self.readline()
            if not line:
                break
            out.append(line)
        return out

    def __iter__(self) -> Generator[bytes]:
        while True:
            line = self.readline()
            if not line:
                break
            yield line

    def tell(self) -> int:
        self._check()
        return self._pos

    def seek(self, pos: int, whence: int = 0) -> int:
        self._check()
        if whence == 0:
            if pos < 0:
                raise ValueError(f"negative seek value {pos}")
            self._pos = pos
        elif whence == 1:
            self._pos = max(0, self._pos + pos)
        elif whence == 2:
            self._pos = max(0, len(self._buf) + pos)
        else:
            raise ValueError(f"invalid whence ({whence}, should be 0, 1 or 2)")
        return self._pos

    def truncate(self, size: Optional[int] = None) -> int:
        self._check()
        n = self._pos if size is None else size
        if n < 0:
            raise ValueError(f"negative size value {n}")
        while len(self._buf) > n:
            self._buf.pop()
        return n

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def flush(self):
        pass

    def close(self):
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()
