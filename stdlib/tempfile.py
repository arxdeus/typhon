import os
import shutil as _shutil
import random as _random
from internal.file import open

TMP_MAX: Literal[int] = 10000
template: Literal[str] = "tmp"

_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789_"
_tempdir: Optional[str] = None
_rng = _random.Random(None)


def _rand_name() -> str:
    return "".join([_CHARS[_rng.randrange(0, len(_CHARS))] for _ in range(8)])


def _candidate_tempdir_list() -> List[str]:
    dirlist = List[str]()
    for envname in ("TMPDIR", "TEMP", "TMP"):
        d = os.environ.get(envname, "")
        if d:
            dirlist.append(d)
    dirlist.extend(["/tmp", "/var/tmp", "/usr/tmp"])
    try:
        dirlist.append(os.getcwd())
    except OSError:
        dirlist.append(os.curdir)
    return dirlist


def _get_default_tempdir() -> str:
    for d in _candidate_tempdir_list():
        if d != os.curdir:
            d = os.path.abspath(d)
        if os.path.isdir(d) and os.access(d, os.W_OK | os.X_OK):
            return d
    raise FileNotFoundError(2, "No usable temporary directory found", "")


def gettempdir() -> str:
    global _tempdir
    t = _tempdir
    if t is None:
        d = _get_default_tempdir()
        _tempdir = d
        return d
    return t


def gettempdirb() -> bytes:
    return gettempdir().encode()


def gettempprefix() -> str:
    return template


def _sanitize(suffix: Optional[str], prefix: Optional[str], dir: Optional[str]) -> Tuple[str, str, str]:
    s = ""
    p = template
    d = ""
    if suffix is not None:
        s = suffix
    if prefix is not None:
        p = prefix
    if dir is not None:
        d = dir
    else:
        d = gettempdir()
    return (s, p, d)


def mkstemp(suffix: Optional[str] = None, prefix: Optional[str] = None,
            dir: Optional[str] = None, text: bool = False) -> Tuple[int, str]:
    s, p, d = _sanitize(suffix, prefix, dir)
    for _ in range(TMP_MAX):
        name = os.path.join(d, p + _rand_name() + s)
        try:
            fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            return (fd, os.path.abspath(name))
        except FileExistsError:
            continue
    raise FileExistsError(17, "No usable temporary file name found", "")


def mkdtemp(suffix: Optional[str] = None, prefix: Optional[str] = None,
            dir: Optional[str] = None) -> str:
    s, p, d = _sanitize(suffix, prefix, dir)
    for _ in range(TMP_MAX):
        name = os.path.join(d, p + _rand_name() + s)
        try:
            os.mkdir(name, 0o700)
            return os.path.abspath(name)
        except FileExistsError:
            continue
    raise FileExistsError(17, "No usable temporary directory name found", "")


def mktemp(suffix: str = "", prefix: str = template, dir: Optional[str] = None) -> str:
    d = dir if dir is not None else gettempdir()
    for _ in range(TMP_MAX):
        name = os.path.join(d, prefix + _rand_name() + suffix)
        if not os.path.exists(name):
            return name
    raise FileExistsError(17, "No usable temporary filename found", "")


class _TemporaryFileWrapper[F]:
    file: F
    name: str
    delete: bool
    _closed: bool

    def __init__(self, file: F, name: str, delete: bool):
        self.file = file
        self.name = name
        self.delete = delete
        self._closed = False

    def write(self, s):
        return self.file.write(s)

    def read(self, n: int = -1):
        return self.file.read(n)

    def readline(self):
        return self.file.readline()

    def readlines(self):
        return self.file.readlines()

    def __iter__(self):
        return self.file.__iter__()

    def seek(self, offset: int, whence: int = 0) -> int:
        return self.file.seek(offset, whence)

    def tell(self) -> int:
        return self.file.tell()

    def flush(self):
        self.file.flush()

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self):
        if not self._closed:
            self._closed = True
            self.file.close()
            if self.delete:
                try:
                    os.unlink(self.name)
                except FileNotFoundError:
                    pass

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()


def NamedTemporaryFile(mode: Literal[str] = "w+b", buffering: int = -1,
                       encoding: Optional[str] = None, newline: Optional[str] = None,
                       suffix: Optional[str] = None, prefix: Optional[str] = None,
                       dir: Optional[str] = None, delete: bool = True, *,
                       errors: Optional[str] = None, delete_on_close: bool = True):
    fd, name = mkstemp(suffix, prefix, dir)
    os.close(fd)
    f = open(name, mode, encoding=encoding, errors=errors, newline=newline)
    return _TemporaryFileWrapper(f, name, delete)


def TemporaryFile(mode: Literal[str] = "w+b", buffering: int = -1,
                  encoding: Optional[str] = None, newline: Optional[str] = None,
                  suffix: Optional[str] = None, prefix: Optional[str] = None,
                  dir: Optional[str] = None, *, errors: Optional[str] = None):
    return NamedTemporaryFile(mode, buffering, encoding, newline, suffix, prefix,
                              dir, True, errors=errors)


class TemporaryDirectory:
    name: str
    _ignore_errors: bool
    _delete: bool
    _done: bool

    def __init__(self, suffix: Optional[str] = None, prefix: Optional[str] = None,
                 dir: Optional[str] = None, ignore_cleanup_errors: bool = False,
                 *, delete: bool = True):
        self.name = mkdtemp(suffix, prefix, dir)
        self._ignore_errors = ignore_cleanup_errors
        self._delete = delete
        self._done = False

    def __repr__(self) -> str:
        return f"<TemporaryDirectory {self.name!r}>"

    def __enter__(self) -> str:
        return self.name

    def cleanup(self):
        if not self._done:
            self._done = True
            if os.path.exists(self.name):
                _shutil.rmtree(self.name, ignore_errors=self._ignore_errors)

    def __exit__(self):
        if self._delete:
            self.cleanup()
