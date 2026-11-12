import sys as _sys


class Warning(Exception):
    def __init__(self, message: str = ""):
        super().__init__(message)


class UserWarning(Warning):
    def __init__(self, message: str = ""):
        super().__init__(message)


class DeprecationWarning(Warning):
    def __init__(self, message: str = ""):
        super().__init__(message)


class PendingDeprecationWarning(Warning):
    def __init__(self, message: str = ""):
        super().__init__(message)


class SyntaxWarning(Warning):
    def __init__(self, message: str = ""):
        super().__init__(message)


class RuntimeWarning(Warning):
    def __init__(self, message: str = ""):
        super().__init__(message)


class FutureWarning(Warning):
    def __init__(self, message: str = ""):
        super().__init__(message)


class ImportWarning(Warning):
    def __init__(self, message: str = ""):
        super().__init__(message)


class UnicodeWarning(Warning):
    def __init__(self, message: str = ""):
        super().__init__(message)


class BytesWarning(Warning):
    def __init__(self, message: str = ""):
        super().__init__(message)


class ResourceWarning(Warning):
    def __init__(self, message: str = ""):
        super().__init__(message)


class EncodingWarning(Warning):
    def __init__(self, message: str = ""):
        super().__init__(message)


_ACTIONS = ("error", "ignore", "always", "all", "default", "module", "once")


class _Filter:
    action: str
    message: str
    lineno: int

    def __init__(self, action: str, message: str, lineno: int):
        self.action = action
        self.message = message
        self.lineno = lineno

    def matches(self, w: Warning, text: str, lineno: int) -> bool:
        return False

    def __eq__(self, other: _Filter) -> bool:
        return self.action == other.action and self.message == other.message and self.lineno == other.lineno and self.key() == other.key()

    def key(self) -> str:
        return ""


class _CatFilter[C](_Filter):
    def __init__(self, action: str, message: str, lineno: int):
        super().__init__(action, message, lineno)

    def key(self) -> str:
        return C.__name__

    def matches(self, w: Warning, text: str, lineno: int) -> bool:
        if not isinstance(w, C):
            return False
        if self.lineno != 0 and self.lineno != lineno:
            return False
        if self.message:
            # CPython compiles the message as a case-insensitive regex matched
            # at the start; a case-insensitive prefix match covers literal use.
            if not text.lower().startswith(self.message.lower()):
                return False
        return True


class WarningMessage:
    message: Warning
    category_name: str
    filename: str
    lineno: int
    line: Optional[str]

    def __init__(self, message: Warning, category_name: str, filename: str,
                 lineno: int, line: Optional[str] = None):
        self.message = message
        self.category_name = category_name
        self.filename = filename
        self.lineno = lineno
        self.line = line

    def __str__(self) -> str:
        return (f"{{message : {str(self.message)!r}, category : {self.category_name!r}, "
                f"filename : {self.filename!r}, lineno : {self.lineno}, line : {self.line!r}}}")


filters: List[_Filter] = List[_Filter]()
filters.append(_CatFilter[DeprecationWarning]("ignore", "", 0))
filters.append(_CatFilter[PendingDeprecationWarning]("ignore", "", 0))
filters.append(_CatFilter[ImportWarning]("ignore", "", 0))
filters.append(_CatFilter[ResourceWarning]("ignore", "", 0))
_onceregistry: Set[str] = Set[str]()
_record_stack: List[Optional[List[WarningMessage]]] = []


def _check_action(action: str):
    if action not in _ACTIONS:
        raise ValueError(f"invalid action: {action!r}")


def _add_filter(f: _Filter, append: bool):
    global filters
    if not append:
        filters = [x for x in filters if not (x == f)]
        filters.insert(0, f)
    else:
        for x in filters:
            if x == f:
                return
        filters.append(f)
    _onceregistry.clear()


def filterwarnings(action: str, message: str = "", category: type = Warning,
                   module: str = "", lineno: int = 0, append: bool = False):
    _check_action(action)
    if lineno < 0:
        raise ValueError("lineno must be an int >= 0")
    _add_filter(_CatFilter[category](action, message, lineno), append)


def simplefilter(action: str, category: type = Warning, lineno: int = 0,
                 append: bool = False):
    _check_action(action)
    if lineno < 0:
        raise ValueError("lineno must be an int >= 0")
    _add_filter(_CatFilter[category](action, "", lineno), append)


def resetwarnings():
    global filters
    filters = []
    _onceregistry.clear()


def formatwarning(message: str, category_name: str, filename: str, lineno: int,
                  line: Optional[str] = None) -> str:
    s = f"{filename}:{lineno}: {category_name}: {message}\n"
    if line is not None:
        l = line.strip()
        if l:
            s += f"  {l}\n"
    return s


def showwarning(message: str, category_name: str, filename: str, lineno: int,
                line: Optional[str] = None):
    _sys.stderr.write(formatwarning(message, category_name, filename, lineno, line))
    _sys.stderr.flush()


def _warn_impl(w: Warning, filename: str, lineno: int):
    text = str(w)
    cat = w.typename
    action = "default"
    for f in filters:
        if f.matches(w, text, lineno):
            action = f.action
            break
    if action == "ignore":
        return
    if action == "error":
        raise w
    if action == "once" or action == "default" or action == "module":
        key = cat + "\0" + text
        if action == "default":
            key += "\0" + filename + "\0" + str(lineno)
        if key in _onceregistry:
            return
        _onceregistry.add(key)
    if _record_stack:
        rec = _record_stack[-1]
        if rec is not None:
            rec.append(WarningMessage(w, cat, filename, lineno))
            return
    showwarning(text, cat, filename, lineno)


def warn_explicit(message: str, category: type, filename: str, lineno: int,
                  module: Optional[str] = None, registry=None):
    w: Warning = category(message)
    _warn_impl(w, filename, lineno)


@overload
def warn_explicit(message: Warning, category: type, filename: str, lineno: int,
                  module: Optional[str] = None, registry=None):
    _warn_impl(message, filename, lineno)


def warn(message: str, category: type = UserWarning, stacklevel: int = 1,
         source=None, skip_file_prefixes=()):
    # The caller's frame is not available at runtime, so the location is
    # reported the way CPython does when the stack is exhausted: `sys:1`.
    w: Warning = category(message)
    _warn_impl(w, "sys", 1)


@overload
def warn(message: Warning, category=None, stacklevel: int = 1, source=None,
         skip_file_prefixes=()):
    _warn_impl(message, "sys", 1)


class catch_warnings:
    _record: bool
    _saved: List[_Filter]
    _saved_once: Set[str]
    _action: str
    _category_filter: Optional[_Filter]

    def __init__(self, *, record: bool = False, action: str = "",
                 category: type = Warning, lineno: int = 0, append: bool = False):
        self._record = record
        self._saved = []
        self._saved_once = Set[str]()
        self._action = action
        self._category_filter = None
        if action:
            _check_action(action)
            self._category_filter = _CatFilter[category](action, "", lineno)

    def __enter__(self) -> Optional[List[WarningMessage]]:
        self._saved = list(filters)
        self._saved_once = set(_onceregistry)
        f = self._category_filter
        if f is not None:
            _add_filter(f, False)
        if self._record:
            log = List[WarningMessage]()
            _record_stack.append(log)
            return log
        _record_stack.append(None)
        return None

    def __exit__(self):
        global filters, _onceregistry
        filters = self._saved
        _onceregistry = self._saved_once
        _record_stack.pop()
