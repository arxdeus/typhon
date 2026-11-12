import os
import fnmatch as _fnmatch


def has_magic(s: str) -> bool:
    for c in s:
        if c == "*" or c == "?" or c == "[":
            return True
    return False


def _ishidden(name: str) -> bool:
    return len(name) > 0 and name[0] == "."


def _isrecursive(pat: str) -> bool:
    return pat == "**"


def _listdir(dirname: str, dironly: bool) -> List[str]:
    out = List[str]()
    d = dirname if dirname else os.curdir
    try:
        names = os.listdir(d)
    except OSError:
        return out
    for n in names:
        if not dironly or os.path.isdir(os.path.join(d, n)):
            out.append(n)
    return out


def _lexists(pathname: str, dironly: bool) -> bool:
    if not dironly:
        return os.path.lexists(pathname)
    return os.path.isdir(pathname)


def _glob1(dirname: str, pattern: str, include_hidden: bool, dironly: bool) -> List[str]:
    names = _listdir(dirname, dironly)
    if not (include_hidden or _ishidden(pattern)):
        names = [x for x in names if not _ishidden(x)]
    return _fnmatch.filter(names, pattern)


def _glob0(dirname: str, basename: str, dironly: bool) -> List[str]:
    if basename:
        if _lexists(os.path.join(dirname, basename), dironly):
            return [basename]
    else:
        # `os.path.split()` returns an empty basename for paths ending with a
        # directory separator: 'q*x/' should match only directories.
        if os.path.isdir(dirname):
            return [basename]
    return List[str]()


def _rlistdir(dirname: str, dironly: bool, include_hidden: bool) -> List[str]:
    out = List[str]()
    for x in _listdir(dirname, dironly):
        if include_hidden or not _ishidden(x):
            out.append(x)
            path = os.path.join(dirname, x) if dirname else x
            if os.path.isdir(path) and not os.path.islink(path):
                for y in _rlistdir(path, dironly, include_hidden):
                    out.append(os.path.join(x, y))
    return out


def _glob2(dirname: str, pattern: str, include_hidden: bool, dironly: bool) -> List[str]:
    out = [pattern[:0]]
    out.extend(_rlistdir(dirname, dironly, include_hidden))
    return out


def _iglob(pathname: str, root_dir: Optional[str], recursive: bool,
           dironly: bool, include_hidden: bool) -> List[str]:
    dirname, basename = os.path.split(pathname)
    result = List[str]()
    if not has_magic(pathname):
        if basename:
            full = pathname if root_dir is None else os.path.join(root_dir, pathname)
            if _lexists(full, dironly):
                result.append(pathname)
        else:
            full = dirname if root_dir is None else os.path.join(root_dir, dirname)
            if os.path.isdir(full):
                result.append(pathname)
        return result
    if not dirname:
        base = "" if root_dir is None else root_dir
        if recursive and _isrecursive(basename):
            result.extend(_glob2(base, basename, include_hidden, dironly))
        else:
            result.extend(_glob1(base, basename, include_hidden, dironly))
        return result
    dirs = List[str]()
    if dirname != pathname and has_magic(dirname):
        dirs = _iglob(dirname, root_dir, recursive, True, include_hidden)
    else:
        dirs = [dirname]
    for d in dirs:
        full = d if root_dir is None else os.path.join(root_dir, d)
        names = List[str]()
        if has_magic(basename):
            if recursive and _isrecursive(basename):
                names = _glob2(full, basename, include_hidden, dironly)
            else:
                names = _glob1(full, basename, include_hidden, dironly)
        else:
            names = _glob0(full, basename, dironly)
        for n in names:
            result.append(os.path.join(d, n))
    return result


def glob(pathname: str, *, root_dir: Optional[str] = None, dir_fd=None,
         recursive: bool = False, include_hidden: bool = False) -> List[str]:
    res = _iglob(pathname, root_dir, recursive, False, include_hidden)
    if recursive and _isrecursive(pathname):
        # CPython drops the empty first match of a bare '**'.
        res = [x for x in res if x]
    return res


def iglob(pathname: str, *, root_dir: Optional[str] = None, dir_fd=None,
          recursive: bool = False, include_hidden: bool = False) -> Generator[str]:
    for x in glob(pathname, root_dir=root_dir, recursive=recursive,
                  include_hidden=include_hidden):
        yield x


def escape(pathname: str) -> str:
    drive, rest = os.path.splitdrive(pathname)
    out = List[str]()
    for c in rest:
        if c == "*" or c == "?" or c == "[":
            out.append("[" + c + "]")
        else:
            out.append(c)
    return drive + "".join(out)


