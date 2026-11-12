import os
import fnmatch as _fnmatch
from internal.file import open

COPY_BUFSIZE: Literal[int] = 1048576


class Error(OSError):
    def __init__(self, message: str = ""):
        super().__init__(message)


class SameFileError(Error):
    def __init__(self, message: str = ""):
        super().__init__(message)


class SpecialFileError(OSError):
    def __init__(self, message: str = ""):
        super().__init__(message)


class ReadError(OSError):
    def __init__(self, message: str = ""):
        super().__init__(message)


def _samefile(src: str, dst: str) -> bool:
    try:
        return os.path.samefile(src, dst)
    except OSError:
        return False


def copyfileobj(fsrc, fdst, length: int = 0):
    n = length if length > 0 else COPY_BUFSIZE
    while True:
        buf = fsrc.read(n)
        if not buf:
            break
        fdst.write(buf)


def copyfile(src: str, dst: str, *, follow_symlinks: bool = True) -> str:
    if _samefile(src, dst):
        raise SameFileError(f"{src!r} and {dst!r} are the same file")
    if not follow_symlinks and os.path.islink(src):
        os.symlink(os.readlink(src), dst)
        return dst
    if os.path.isdir(src):
        raise IsADirectoryError(21, "Is a directory", src)
    with open(src, "rb") as fsrc:
        with open(dst, "wb") as fdst:
            copyfileobj(fsrc, fdst)
    return dst


def copymode(src: str, dst: str, *, follow_symlinks: bool = True):
    st = os.stat(src, follow_symlinks=follow_symlinks)
    os.chmod(dst, st.st_mode & 0o7777)


def _utime(path: str, atime_ns: int, mtime_ns: int):
    from C import utimes(cobj, cobj) -> i32
    tv = Ptr[int](4)
    tv[0] = atime_ns // 1000000000
    tv[1] = (atime_ns % 1000000000) // 1000
    tv[2] = mtime_ns // 1000000000
    tv[3] = (mtime_ns % 1000000000) // 1000
    if utimes(path.c_str(), tv.as_byte()) != i32(0):
        raise OSError(f"utime failed: {path!r}")


def copystat(src: str, dst: str, *, follow_symlinks: bool = True):
    st = os.stat(src, follow_symlinks=follow_symlinks)
    _utime(dst, st.st_atime_ns, st.st_mtime_ns)
    os.chmod(dst, st.st_mode & 0o7777)


def _dest(src: str, dst: str) -> str:
    if os.path.isdir(dst):
        return os.path.join(dst, os.path.basename(src))
    return dst


def copy(src: str, dst: str, *, follow_symlinks: bool = True) -> str:
    d = _dest(src, dst)
    copyfile(src, d, follow_symlinks=follow_symlinks)
    copymode(src, d)
    return d


def copy2(src: str, dst: str, *, follow_symlinks: bool = True) -> str:
    d = _dest(src, dst)
    copyfile(src, d, follow_symlinks=follow_symlinks)
    copystat(src, d)
    return d


def ignore_patterns(*patterns):
    pats = [p for p in patterns]
    def _ignore(path: str, names: List[str]) -> Set[str]:
        ignored = Set[str]()
        for p in pats:
            for n in _fnmatch.filter(names, p):
                ignored.add(n)
        return ignored
    return _ignore


def _copytree(src: str, dst: str, symlinks: bool, ignore, copy_function,
              dirs_exist_ok: bool) -> str:
    names = sorted(os.listdir(src))
    ignored = Set[str]()
    if not isinstance(ignore, NoneType):
        ignored = ignore(src, names)
    os.makedirs(dst, exist_ok=dirs_exist_ok)
    errors = List[str]()
    for name in names:
        if name in ignored:
            continue
        s = os.path.join(src, name)
        d = os.path.join(dst, name)
        try:
            if os.path.islink(s) and symlinks:
                os.symlink(os.readlink(s), d)
            elif os.path.isdir(s):
                _copytree(s, d, symlinks, ignore, copy_function, dirs_exist_ok)
            else:
                copy_function(s, d)
        except OSError as e:
            errors.append(f"{s}: {e}")
    try:
        copystat(src, dst)
    except OSError as e:
        errors.append(f"{src}: {e}")
    if errors:
        raise Error("; ".join(errors))
    return dst


def copytree(src: str, dst: str, symlinks: bool = False, ignore=None,
             copy_function=copy2, ignore_dangling_symlinks: bool = False,
             dirs_exist_ok: bool = False) -> str:
    return _copytree(src, dst, symlinks, ignore, copy_function, dirs_exist_ok)


def _rmtree(path: str):
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if os.path.isdir(full) and not os.path.islink(full):
            _rmtree(full)
        else:
            os.unlink(full)
    os.rmdir(path)


def rmtree(path: str, ignore_errors: bool = False, onerror=None, *, onexc=None,
           dir_fd=None):
    try:
        if os.path.islink(path):
            raise OSError("Cannot call rmtree on a symbolic link")
        _rmtree(path)
    except OSError as e:
        if not ignore_errors:
            raise e


def move(src: str, dst: str, copy_function=copy2) -> str:
    real_dst = dst
    if os.path.isdir(dst):
        if _samefile(src, dst) and not os.path.islink(src):
            os.rename(src, dst)
            return dst
        real_dst = os.path.join(dst, os.path.basename(src.rstrip(os.sep)))
        if os.path.exists(real_dst):
            raise Error(f"Destination path '{real_dst}' already exists")
    try:
        os.rename(src, real_dst)
    except OSError:
        if os.path.islink(src):
            os.symlink(os.readlink(src), real_dst)
            os.unlink(src)
        elif os.path.isdir(src):
            a = os.path.abspath(src)
            b = os.path.abspath(dst)
            if b.startswith(a + os.sep) or a == b:
                raise Error(f"Cannot move a directory '{src}' into itself '{dst}'.")
            copytree(src, real_dst, symlinks=True, copy_function=copy_function)
            rmtree(src)
        else:
            copy_function(src, real_dst)
            os.unlink(src)
    return real_dst


@tuple
class _ntuple_diskusage:
    total: int
    used: int
    free: int

    def __repr__(self) -> str:
        return f"usage(total={self.total}, used={self.used}, free={self.free})"


def disk_usage(path: str) -> _ntuple_diskusage:
    from C import statvfs(cobj, cobj) -> i32
    buf = Ptr[u64](16)
    if statvfs(path.c_str(), buf.as_byte()) != i32(0):
        raise FileNotFoundError(2, "No such file or directory", path)
    frsize = int(buf[1])
    blocks = 0
    bfree = 0
    bavail = 0
    if __apple__:
        # struct statvfs: two unsigned longs, then 32-bit block counts.
        p32 = Ptr[u32](buf.as_byte() + 16)
        blocks = int(p32[0])
        bfree = int(p32[1])
        bavail = int(p32[2])
    else:
        blocks = int(buf[2])
        bfree = int(buf[3])
        bavail = int(buf[4])
    total = blocks * frsize
    free = bavail * frsize
    used = (blocks - bfree) * frsize
    return _ntuple_diskusage(total, used, free)


def _access_check(fn: str, mode: int) -> bool:
    return os.path.exists(fn) and os.access(fn, mode) and not os.path.isdir(fn)


def which(cmd: str, mode: int = os.F_OK | os.X_OK, path: Optional[str] = None) -> Optional[str]:
    if os.path.dirname(cmd):
        if _access_check(cmd, mode):
            return cmd
        return None
    p = path if path is not None else os.environ.get("PATH", os.defpath)
    if not p:
        return None
    for d in p.split(os.pathsep):
        name = os.path.join(d, cmd)
        if _access_check(name, mode):
            return name
    return None


def get_terminal_size(fallback: Tuple[int, int] = (80, 24)) -> Tuple[int, int]:
    columns = 0
    lines = 0
    try:
        columns = int(os.environ.get("COLUMNS", "0"))
    except ValueError:
        columns = 0
    try:
        lines = int(os.environ.get("LINES", "0"))
    except ValueError:
        lines = 0
    if columns <= 0:
        columns = fallback[0]
    if lines <= 0:
        lines = fallback[1]
    return (columns, lines)
