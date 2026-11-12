import sys as _sys
import time as _time
import os as _os
import internal.static as _static

CRITICAL: Literal[int] = 50
FATAL: Literal[int] = 50
ERROR: Literal[int] = 40
WARNING: Literal[int] = 30
WARN: Literal[int] = 30
INFO: Literal[int] = 20
DEBUG: Literal[int] = 10
NOTSET: Literal[int] = 0

BASIC_FORMAT: Literal[str] = "%(levelname)s:%(name)s:%(message)s"

raiseExceptions = True

_levelToName: Dict[int, str] = {
    50: "CRITICAL", 40: "ERROR", 30: "WARNING", 20: "INFO", 10: "DEBUG", 0: "NOTSET",
}
_nameToLevel: Dict[str, int] = {
    "CRITICAL": 50, "FATAL": 50, "ERROR": 40, "WARN": 30, "WARNING": 30,
    "INFO": 20, "DEBUG": 10, "NOTSET": 0,
}

_startTime = _time.time()


def getLevelName(level: int) -> str:
    if level in _levelToName:
        return _levelToName[level]
    return f"Level {level}"


def getLevelNamesMapping() -> Dict[str, int]:
    return dict(_nameToLevel)


def addLevelName(level: int, levelName: str):
    _levelToName[level] = levelName
    _nameToLevel[levelName] = level


def _checkLevel(level: int) -> int:
    return level


@overload
def _checkLevel(level: str) -> int:
    if level not in _nameToLevel:
        raise ValueError(f"Unknown level: {level!r}")
    return _nameToLevel[level]


class LogRecord:
    name: str
    msg: str
    levelno: int
    levelname: str
    pathname: str
    filename: str
    module: str
    lineno: int
    funcName: str
    created: float
    msecs: float
    relativeCreated: float
    process: int
    processName: str
    threadName: str
    message: str
    exc_text: Optional[str]

    def __init__(self, name: str, level: int, pathname: str, lineno: int,
                 msg, args=(), exc_info=None, func: str = "(unknown function)"):
        ct = _time.time()
        self.name = name
        self.msg = _format_args(msg, args)
        self.levelno = level
        self.levelname = getLevelName(level)
        self.pathname = pathname
        self.filename = pathname
        self.module = "(unknown module)"
        self.lineno = lineno
        self.funcName = func
        self.created = ct
        self.msecs = float(int((ct - float(int(ct))) * 1000.0))
        self.relativeCreated = (ct - _startTime) * 1000.0
        self.process = _os.getpid()
        self.processName = "MainProcess"
        self.threadName = "MainThread"
        self.message = self.msg
        self.exc_text = _exc_text(exc_info)

    def getMessage(self) -> str:
        return self.msg

    def __repr__(self) -> str:
        return f'<LogRecord: {self.name}, {self.levelno}, {self.pathname}, {self.lineno}, "{self.msg}">'


_MONTHS = ("January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December")
_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def _pad(v: int, n: int) -> str:
    s = str(v)
    while len(s) < n:
        s = "0" + s
    return s


def _strftime(fmt: str, t: _time.struct_time) -> str:
    out = List[str]()
    i = 0
    n = len(fmt)
    while i < n:
        c = fmt[i]
        if c != "%" or i + 1 >= n:
            out.append(c)
            i += 1
            continue
        d = fmt[i + 1]
        i += 2
        if d == "Y":
            out.append(str(t.tm_year))
        elif d == "y":
            out.append(_pad(t.tm_year % 100, 2))
        elif d == "m":
            out.append(_pad(t.tm_mon, 2))
        elif d == "d":
            out.append(_pad(t.tm_mday, 2))
        elif d == "H":
            out.append(_pad(t.tm_hour, 2))
        elif d == "I":
            h = t.tm_hour % 12
            out.append(_pad(12 if h == 0 else h, 2))
        elif d == "p":
            out.append("AM" if t.tm_hour < 12 else "PM")
        elif d == "M":
            out.append(_pad(t.tm_min, 2))
        elif d == "S":
            out.append(_pad(t.tm_sec, 2))
        elif d == "j":
            out.append(_pad(t.tm_yday, 3))
        elif d == "b":
            out.append(_MONTHS[t.tm_mon - 1][:3])
        elif d == "B":
            out.append(_MONTHS[t.tm_mon - 1])
        elif d == "a":
            out.append(_DAYS[t.tm_wday][:3])
        elif d == "A":
            out.append(_DAYS[t.tm_wday])
        elif d == "%":
            out.append("%")
        else:
            out.append("%" + d)
    return "".join(out)


class Formatter:
    default_time_format: str
    default_msec_format: str
    _fmt: str
    datefmt: Optional[str]
    _parts: List[Tuple[str, str, str]]

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None,
                 style: str = "%", validate: bool = True):
        if style != "%":
            raise ValueError("only '%' style formats are supported")
        self._fmt = fmt if fmt is not None else "%(message)s"
        self.datefmt = datefmt
        self.default_time_format = "%Y-%m-%d %H:%M:%S"
        self.default_msec_format = "%s,%03d"
        self._parts = Formatter._parse(self._fmt)
        if validate:
            ok = False
            for p in self._parts:
                if p[0]:
                    ok = True
            if not ok:
                raise ValueError(f"Invalid format '{self._fmt}' for '%' style")

    def _parse(fmt: str) -> List[Tuple[str, str, str]]:
        # (key, spec, conversion); key == "" means literal text in spec.
        parts = List[Tuple[str, str, str]]()
        lit = List[str]()
        i = 0
        n = len(fmt)
        while i < n:
            if fmt[i] == "%" and i + 1 < n and fmt[i + 1] == "(":
                j = fmt.find(")", i + 2)
                if j < 0:
                    lit.append(fmt[i:])
                    break
                key = fmt[i + 2:j]
                k = j + 1
                while k < n and fmt[k] in "#0- +.0123456789":
                    k += 1
                if k >= n:
                    lit.append(fmt[i:])
                    break
                if lit:
                    parts.append(("", "".join(lit), ""))
                    lit = List[str]()
                parts.append((key, fmt[j + 1:k], fmt[k]))
                i = k + 1
            elif fmt[i] == "%" and i + 1 < n and fmt[i + 1] == "%":
                lit.append("%")
                i += 2
            else:
                lit.append(fmt[i])
                i += 1
        if lit:
            parts.append(("", "".join(lit), ""))
        return parts

    def usesTime(self) -> bool:
        for p in self._parts:
            if p[0] == "asctime":
                return True
        return False

    def formatTime(self, record: LogRecord, datefmt: Optional[str] = None) -> str:
        ct = _time.localtime(int(record.created))
        if datefmt is not None:
            return _strftime(datefmt, ct)
        s = _strftime(self.default_time_format, ct)
        return s + "," + _pad(int(record.msecs), 3)

    def formatMessage(self, record: LogRecord, asctime: str = "") -> str:
        out = List[str]()
        for key, spec, conv in self._parts:
            if not key:
                out.append(spec)
                continue
            f = "%" + spec + conv
            if key == "name":
                out.append(f % record.name)
            elif key == "message":
                out.append(f % record.message)
            elif key == "levelname":
                out.append(f % record.levelname)
            elif key == "levelno":
                out.append(f % record.levelno)
            elif key == "asctime":
                out.append(f % asctime)
            elif key == "msecs":
                out.append(f % record.msecs)
            elif key == "created":
                out.append(f % record.created)
            elif key == "relativeCreated":
                out.append(f % record.relativeCreated)
            elif key == "pathname":
                out.append(f % record.pathname)
            elif key == "filename":
                out.append(f % record.filename)
            elif key == "module":
                out.append(f % record.module)
            elif key == "lineno":
                out.append(f % record.lineno)
            elif key == "funcName":
                out.append(f % record.funcName)
            elif key == "process":
                out.append(f % record.process)
            elif key == "processName":
                out.append(f % record.processName)
            elif key == "threadName":
                out.append(f % record.threadName)
            elif key == "taskName":
                out.append(f % "None")
            else:
                raise KeyError(key)
        return "".join(out)

    def format(self, record: LogRecord) -> str:
        record.message = record.getMessage()
        asctime = ""
        if self.usesTime():
            asctime = self.formatTime(record, self.datefmt)
        s = self.formatMessage(record, asctime)
        et = record.exc_text
        if et is not None:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + et
        return s


class Filter:
    name: str

    def __init__(self, name: str = ""):
        self.name = name

    def filter(self, record: LogRecord) -> bool:
        n = len(self.name)
        if n == 0 or self.name == record.name:
            return True
        if not record.name.startswith(self.name):
            return False
        return record.name[n] == "."


class Filterer:
    filters: List[Filter]

    def __init__(self):
        self.filters = []

    def addFilter(self, f: Filter):
        for x in self.filters:
            if x is f:
                return
        self.filters.append(f)

    def removeFilter(self, f: Filter):
        self.filters = [x for x in self.filters if x is not f]

    def filter(self, record: LogRecord) -> bool:
        for f in self.filters:
            if not f.filter(record):
                return False
        return True


_defaultFormatter = Formatter()


class Handler(Filterer):
    level: int
    formatter: Optional[Formatter]
    name: Optional[str]

    def __init__(self, level: int = 0):
        super().__init__()
        self.level = level
        self.formatter = None
        self.name = None

    def setLevel(self, level):
        self.level = _checkLevel(level)

    def setFormatter(self, fmt: Optional[Formatter]):
        self.formatter = fmt

    def format(self, record: LogRecord) -> str:
        f = self.formatter
        if f is not None:
            return f.format(record)
        return _defaultFormatter.format(record)

    def emit(self, record: LogRecord):
        raise NotImplementedError("emit must be implemented by Handler subclasses")

    def handle(self, record: LogRecord) -> bool:
        rv = self.filter(record)
        if rv:
            self.emit(record)
        return rv

    def flush(self):
        pass

    def close(self):
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} ({getLevelName(self.level)})>"


class _StderrHandler(Handler):
    def __init__(self, level: int = 0):
        super().__init__(level)

    def emit(self, record: LogRecord):
        _sys.stderr.write(self.format(record) + "\n")
        _sys.stderr.flush()


class _StreamHandler[S](Handler):
    stream: S
    terminator: str

    def __init__(self, stream: S, level: int = 0):
        super().__init__(level)
        self.stream = stream
        self.terminator = "\n"

    def emit(self, record: LogRecord):
        self.stream.write(self.format(record) + self.terminator)
        self.flush()

    def flush(self):
        if hasattr(self.stream, "flush"):
            self.stream.flush()

    def setStream(self, stream: S) -> S:
        old = self.stream
        self.flush()
        self.stream = stream
        return old

    def __repr__(self) -> str:
        return f"<StreamHandler ({getLevelName(self.level)})>"


def StreamHandler(stream=None):
    if isinstance(stream, NoneType):
        return _StreamHandler(_sys.stderr)
    else:
        return _StreamHandler(stream)


class FileHandler(Handler):
    baseFilename: str
    mode: str
    encoding: Optional[str]
    terminator: str
    _file: Optional[File[str]]

    def __init__(self, filename: str, mode: str = "a", encoding: Optional[str] = None,
                 delay: bool = False):
        super().__init__()
        self.baseFilename = _os.path.abspath(filename)
        self.mode = mode
        self.encoding = encoding
        self.terminator = "\n"
        self._file = None
        if not delay:
            self._open()

    def _open(self):
        self._file = open(self.baseFilename, self.mode, encoding=self.encoding)

    def emit(self, record: LogRecord):
        if self._file is None:
            self._open()
        f = self._file
        if f is not None:
            f.write(self.format(record) + self.terminator)
            f.flush()

    def flush(self):
        f = self._file
        if f is not None:
            f.flush()

    def close(self):
        f = self._file
        if f is not None:
            f.close()
            self._file = None

    def __repr__(self) -> str:
        return f"<FileHandler {self.baseFilename} ({getLevelName(self.level)})>"


class NullHandler(Handler):
    def __init__(self):
        super().__init__()

    def emit(self, record: LogRecord):
        pass

    def handle(self, record: LogRecord) -> bool:
        return True


lastResort: Handler = _StderrHandler(WARNING)


def _format_args(msg, args) -> str:
    m = str(msg)
    if _static.len(args) == 0:
        return m
    elif _static.len(args) == 1:
        if isinstance(args[0], Dict):
            return m % args[0]
        else:
            return m % args
    else:
        return m % args


class Logger(Filterer):
    name: str
    level: int
    parent: Optional[Logger]
    propagate: bool
    handlers: List[Handler]
    disabled: bool

    def __init__(self, name: str, level: int = 0):
        super().__init__()
        self.name = name
        self.level = _checkLevel(level)
        self.parent = None
        self.propagate = True
        self.handlers = []
        self.disabled = False

    def setLevel(self, level):
        self.level = _checkLevel(level)

    def getEffectiveLevel(self) -> int:
        lg: Optional[Logger] = self
        while lg is not None:
            l = lg
            if l.level:
                return l.level
            lg = l.parent
        return NOTSET

    def isEnabledFor(self, level: int) -> bool:
        if self.disabled or _disable_level >= level:
            return False
        return level >= self.getEffectiveLevel()

    def addHandler(self, h: Handler):
        for x in self.handlers:
            if x is h:
                return
        self.handlers.append(h)

    def removeHandler(self, h: Handler):
        self.handlers = [x for x in self.handlers if x is not h]

    def hasHandlers(self) -> bool:
        lg: Optional[Logger] = self
        while lg is not None:
            l = lg
            if l.handlers:
                return True
            if not l.propagate:
                return False
            lg = l.parent
        return False

    def callHandlers(self, record: LogRecord):
        found = 0
        lg: Optional[Logger] = self
        while lg is not None:
            l = lg
            for h in l.handlers:
                found += 1
                if record.levelno >= h.level:
                    h.handle(record)
            if not l.propagate:
                lg = None
            else:
                lg = l.parent
        if found == 0:
            if record.levelno >= lastResort.level:
                lastResort.handle(record)

    def handle(self, record: LogRecord):
        if not self.disabled and self.filter(record):
            self.callHandlers(record)

    def makeRecord(self, name: str, level: int, fn: str, lno: int, msg: str) -> LogRecord:
        return LogRecord(name, level, fn, lno, msg, (), None)

    def _log(self, level: int, msg: str, exc_text: Optional[str] = None):
        record = self.makeRecord(self.name, level, "(unknown file)", 0, msg)
        record.exc_text = exc_text
        self.handle(record)

    def log(self, level: int, msg, *args, exc_info=None):
        if self.isEnabledFor(level):
            self._log(level, _format_args(msg, args), _exc_text(exc_info))

    def debug(self, msg, *args, exc_info=None):
        if self.isEnabledFor(DEBUG):
            self._log(DEBUG, _format_args(msg, args), _exc_text(exc_info))

    def info(self, msg, *args, exc_info=None):
        if self.isEnabledFor(INFO):
            self._log(INFO, _format_args(msg, args), _exc_text(exc_info))

    def warning(self, msg, *args, exc_info=None):
        if self.isEnabledFor(WARNING):
            self._log(WARNING, _format_args(msg, args), _exc_text(exc_info))

    def warn(self, msg, *args, exc_info=None):
        self.warning(msg, *args, exc_info=exc_info)

    def error(self, msg, *args, exc_info=None):
        if self.isEnabledFor(ERROR):
            self._log(ERROR, _format_args(msg, args), _exc_text(exc_info))

    def exception(self, msg, *args, exc_info=None):
        if self.isEnabledFor(ERROR):
            self._log(ERROR, _format_args(msg, args), _exc_text(exc_info))

    def critical(self, msg, *args, exc_info=None):
        if self.isEnabledFor(CRITICAL):
            self._log(CRITICAL, _format_args(msg, args), _exc_text(exc_info))

    def fatal(self, msg, *args, exc_info=None):
        self.critical(msg, *args, exc_info=exc_info)

    def getChild(self, suffix: str) -> Logger:
        if self.name == "root":
            return getLogger(suffix)
        return getLogger(self.name + "." + suffix)

    def __repr__(self) -> str:
        return f"<Logger {self.name} ({getLevelName(self.getEffectiveLevel())})>"


def _exc_text(exc_info) -> Optional[str]:
    # Tracebacks are not available at runtime: only the exception line is
    # rendered (CPython prints "Traceback ..." followed by this line).
    if isinstance(exc_info, NoneType):
        return None
    elif isinstance(exc_info, bool):
        return None
    elif isinstance(exc_info, BaseException):
        return f"{exc_info.typename}: {exc_info}"
    else:
        return None


_disable_level = 0


def disable(level: int = CRITICAL):
    global _disable_level
    _disable_level = level


root = Logger("root", WARNING)
_loggers: Dict[str, Logger] = {}


def _fixup_parents(lg: Logger):
    name = lg.name
    i = name.rfind(".")
    rv: Optional[Logger] = None
    while i > 0 and rv is None:
        sub = name[:i]
        if sub in _loggers:
            rv = _loggers[sub]
        i = sub.rfind(".")
    lg.parent = rv if rv is not None else root
    # re-parent existing children
    prefix = name + "."
    for k, c in _loggers.items():
        if k.startswith(prefix):
            p = c.parent
            if p is not None and (p is root or not p.name.startswith(prefix)):
                if p is root or len(p.name) < len(name):
                    c.parent = lg


def getLogger(name: Optional[str] = None) -> Logger:
    if name is None or name == "" or name == "root":
        return root
    n: str = name
    if n in _loggers:
        return _loggers[n]
    lg = Logger(n)
    _loggers[n] = lg
    _fixup_parents(lg)
    return lg


def getRootLogger() -> Logger:
    return root


def basicConfig(level=None, format: Optional[str] = None, datefmt: Optional[str] = None,
                filename: Optional[str] = None, filemode: str = "a",
                stream=None, force: bool = False, style: str = "%",
                encoding: Optional[str] = None):
    if force:
        for h in root.handlers:
            h.close()
        root.handlers = []
    if len(root.handlers) == 0:
        fmt = Formatter(format if format is not None else BASIC_FORMAT, datefmt, style)
        if filename is not None:
            if not isinstance(stream, NoneType):
                raise ValueError("'stream' and 'filename' should not be specified together")
            h: Handler = FileHandler(filename, filemode, encoding)
            h.setFormatter(fmt)
            root.addHandler(h)
        else:
            sh: Handler = StreamHandler(stream)
            sh.setFormatter(fmt)
            root.addHandler(sh)
    if not isinstance(level, NoneType):
        root.setLevel(level)


def _ensure_root():
    if len(root.handlers) == 0:
        basicConfig()


def log(level: int, msg, *args, exc_info=None):
    _ensure_root()
    root.log(level, msg, *args, exc_info=exc_info)


def debug(msg, *args, exc_info=None):
    _ensure_root()
    root.debug(msg, *args, exc_info=exc_info)


def info(msg, *args, exc_info=None):
    _ensure_root()
    root.info(msg, *args, exc_info=exc_info)


def warning(msg, *args, exc_info=None):
    _ensure_root()
    root.warning(msg, *args, exc_info=exc_info)


def warn(msg, *args, exc_info=None):
    warning(msg, *args, exc_info=exc_info)


def error(msg, *args, exc_info=None):
    _ensure_root()
    root.error(msg, *args, exc_info=exc_info)


def exception(msg, *args, exc_info=None):
    _ensure_root()
    root.exception(msg, *args, exc_info=exc_info)


def critical(msg, *args, exc_info=None):
    _ensure_root()
    root.critical(msg, *args, exc_info=exc_info)


def fatal(msg, *args, exc_info=None):
    critical(msg, *args, exc_info=exc_info)


def shutdown():
    for h in root.handlers:
        h.flush()
        h.close()
    for _, lg in _loggers.items():
        for h in lg.handlers:
            h.flush()
            h.close()
