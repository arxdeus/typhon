
# Warning(!): This type must be consistent with the exception
# header type defined in runtime/exc.cpp.
class BaseException(object):
    message: str
    func: str
    file: str
    line: int
    col: int
    cause: Optional[BaseException]

    def __init__(self, message: str = "", /):
        self.message = message
        self.func = ""
        self.file = ""
        self.line = 0
        self.col = 0
        self.cause = Optional._ref_new(T=BaseException)

    def __str__(self: BaseException, /) -> str:
        return self.message

    def __repr__(self: BaseException, /) -> str:
        return f'{self.typename}({self.message.__repr__()})'

    @property
    def typename(self):
        return get_typeinfo(self).nice_name

    @property
    def __cause__(self):
        return self.cause

    def _context_exit_is_legacy_none(exit_method, is_async: Literal[bool]) -> Literal[bool]:
        """Select plain finally cleanup only for exclusively legacy None exits."""
        import internal.static as static
        if static.function.can_call(exit_method, NoneType(), NoneType(), NoneType()):
            return False
        elif static.function.can_call(exit_method, get_typeinfo(BaseException()),
                                      BaseException(), NoneType()):
            return False
        elif not static.function.can_call(exit_method):
            return False
        elif is_async:
            if isinstance(exit_method(), Coroutine):
                return type(exit_method()).T is NoneType
            else:
                return False
        else:
            return type(exit_method()) is NoneType

    def _context_exit(exit_method, exc_type, exc_value):
        """Prefer the exception-aware protocol, retaining legacy zero-argument exits."""
        import internal.static as static
        if static.function.can_call(exit_method, exc_type, exc_value, NoneType()):
            return exit_method(exc_type, exc_value, NoneType())
        else:
            return exit_method()

    async def _context_aexit(exit_method, exc_type, exc_value):
        """Await the selected async exit protocol without coercing its result."""
        import internal.static as static
        if static.function.can_call(exit_method, exc_type, exc_value, NoneType()):
            return await exit_method(exc_type, exc_value, NoneType())
        else:
            return await exit_method()

    @__hidden__
    def _set_header(e, func, file, line, col, cause):
        if not isinstance(e, BaseException):
            compile_error("exceptions must derive from BaseException")

        e.func = func
        e.file = file
        e.line = line
        e.col = col
        if cause is not None:
            c: BaseException = cause
            e.cause = c
        return e


class Exception(BaseException):
    def __init__(self, message: str = "", /):
        super().__init__(message)

class NameError(Exception):
    def __init__(self, message: str = "", /):
        super().__init__(message)

def _errno() -> int:
    import internal.static as static
    if __apple__:
        from C import __error() -> Ptr[i32]
        return int(__error()[0])
    elif static.platform() == "win32":
        from C import _get_errno(Ptr[i32]) -> i32
        code = i32(0)
        _get_errno(__ptr__(code))
        return int(code)
    elif __android__:
        from C import __errno() -> Ptr[i32]
        return int(__errno()[0])
    else:
        from C import __errno_location() -> Ptr[i32]
        return int(__errno_location()[0])

class OSError(Exception):
    errno: Optional[int]
    strerror: Optional[str]
    filename: Optional[str]
    filename2: Optional[str]
    winerror: Optional[int]

    def _construct(message: str = "") -> OSError:
        result = OSError.__new__()
        result.__init__(message)
        return result

    def _construct(errno: Optional[int], strerror: str,
                   filename: Optional[str] = None, winerror: Optional[int] = None,
                   filename2: Optional[str] = None) -> OSError:
        import internal.static as static
        code = unwrap(errno) if errno is not None else 0
        darwin = __apple__
        windows = static.platform() == "win32"
        if code == 2:
            return FileNotFoundError(errno, strerror, filename, winerror, filename2)
        if code == 17:
            return FileExistsError(errno, strerror, filename, winerror, filename2)
        if code == 1 or code == 13:
            return PermissionError(errno, strerror, filename, winerror, filename2)
        if code == 20:
            return NotADirectoryError(errno, strerror, filename, winerror, filename2)
        if code == 21:
            return IsADirectoryError(errno, strerror, filename, winerror, filename2)
        if code == 3:
            return ProcessLookupError(errno, strerror, filename, winerror, filename2)
        if code == 4:
            return InterruptedError(errno, strerror, filename, winerror, filename2)
        if code == 10:
            return ChildProcessError(errno, strerror, filename, winerror, filename2)
        if code == 32 or (not windows and code == (58 if darwin else 108)):
            return BrokenPipeError(errno, strerror, filename, winerror, filename2)
        if code == (53 if darwin else 106 if windows else 103):
            return ConnectionAbortedError(errno, strerror, filename, winerror, filename2)
        if code == (61 if darwin else 107 if windows else 111):
            return ConnectionRefusedError(errno, strerror, filename, winerror, filename2)
        if code == (54 if darwin else 108 if windows else 104):
            return ConnectionResetError(errno, strerror, filename, winerror, filename2)
        if code == (60 if darwin else 138 if windows else 110):
            return TimeoutError(errno, strerror, filename, winerror, filename2)
        if code == (35 if darwin else 11) or code == (37 if darwin else 103 if windows else 114) or code == (36 if darwin else 112 if windows else 115):
            return BlockingIOError(errno, strerror, filename, winerror, filename2)
        result = OSError.__new__()
        result.__init__(errno, strerror, filename, winerror, filename2)
        return result

    def __init__(self, message: str = "", /):
        super().__init__(message)
        self.errno = None
        self.strerror = None
        self.filename = None
        self.filename2 = None
        self.winerror = None

    def __init__(self, errno: Optional[int], strerror: str,
                 filename: Optional[str] = None, winerror: Optional[int] = None,
                 filename2: Optional[str] = None, /):
        super().__init__()
        self.errno = errno
        self.strerror = strerror
        self.filename = filename
        self.filename2 = filename2
        self.winerror = winerror
        self.message = self._format_message()

    def _format_message(self) -> str:
        if self.strerror is None:
            return self.message
        result = str(self.strerror)
        if self.winerror is not None:
            result = f"[WinError {self.winerror}] {result}"
        elif self.errno is not None:
            result = f"[Errno {self.errno}] {result}"
        if self.filename is not None:
            result += ": " + repr(str(self.filename))
            if self.filename2 is not None:
                result += " -> " + repr(str(self.filename2))
        return result

IOError = OSError

class ValueError(Exception):
    def __init__(self, message: str = "", /):
        super().__init__(message)

class LookupError(Exception):
    def __init__(self, message: str = "", /):
        super().__init__(message)

class IndexError(LookupError):
    def __init__(self, message: str = "", /):
        super().__init__(message)

class KeyError(LookupError):
    def __init__(self, message: str = "", /):
        super().__init__(message)

class CError(Exception):
    def __init__(self, message: str = ""):
        super().__init__(message)

class TypeError(Exception):
    def __init__(self, message: str = "", /):
        super().__init__(message)

class ArithmeticError(Exception):
    def __init__(self, message: str = "", /):
        super().__init__(message)

class ZeroDivisionError(ArithmeticError):
    def __init__(self, message: str = "", /):
        super().__init__(message)

class OverflowError(ArithmeticError):
    def __init__(self, message: str = "", /):
        super().__init__(message)

class AttributeError(Exception):
    def __init__(self, message: str = "", /):
        super().__init__(message)

class RuntimeError(Exception):
    def __init__(self, message: str = "", /):
        super().__init__(message)

def _raise_no_active_exception():
    raise RuntimeError("No active exception to reraise")

def _missing_return_error(message: str, name: str, file: str, line: int, column: int) -> BaseException:
    error = RuntimeError(message)
    error.func = name
    error.file = file
    error.line = line
    error.col = column
    return error

class NotImplementedError(RuntimeError):
    def __init__(self, message: str = "", /):
        super().__init__(message)

class StopIteration(Exception):
    def __init__(self, message: str = "", /):
        super().__init__(message)

class AssertionError(Exception):
    def __init__(self, message: str = "", /):
        super().__init__(message)

class EOFError(Exception):
    def __init__(self, message: str = "", /):
        super().__init__(message)

class SystemExit(BaseException):
    _status: int
    _message_exit: bool

    def __init__(self, message: str, status: int = 1, /):
        super().__init__(message)
        self._status = status
        self._message_exit = True

    def __init__(self, status: int = 0, /):
        self.__init__("", status)
        self._message_exit = False

    @property
    def status(self):
        return self._status

class StaticCompileError(Exception):
    def __init__(self, message: str = ""):
        super().__init__(message)


EnvironmentError = OSError

class GeneratorExit(BaseException):
    def __init__(self, message: str = "", /):
        super().__init__(message)

class KeyboardInterrupt(BaseException):
    def __init__(self, message: str = "", /):
        super().__init__(message)

class MemoryError(Exception):
    def __init__(self, message: str = "", /):
        super().__init__(message)

class ReferenceError(Exception):
    def __init__(self, message: str = "", /):
        super().__init__(message)

class SystemError(Exception):
    def __init__(self, message: str = "", /):
        super().__init__(message)

class StopAsyncIteration(Exception):
    def __init__(self, message: str = "", /):
        super().__init__(message)

class FloatingPointError(ArithmeticError):
    def __init__(self, message: str = "", /):
        super().__init__(message)

class RecursionError(RuntimeError):
    def __init__(self, message: str = "", /):
        super().__init__(message)

class UnicodeError(ValueError):
    def __init__(self, message: str = "", /):
        super().__init__(message)

class UnicodeDecodeError(UnicodeError):
    encoding: str
    object: bytes
    start: int
    end: int
    reason: str
    _args: Tuple[str, bytes, int, int, str]

    def __init__(self, encoding: str, object: bytes, start: int, end: int, reason: str, /):
        super().__init__()
        self.encoding, self.object = encoding, object
        self.start, self.end, self.reason = start, end, reason
        self._args = encoding, object, start, end, reason
        self.message = self.__str__()

    @property
    def args(self):
        return self._args

    def __repr__(self, /):
        return self.typename + repr(self.args)

    def __str__(self, /):
        prefix = "'" + self.encoding + "' codec can't decode "
        if self.end == self.start + 1 and 0 <= self.start < len(self.object):
            return prefix + ("byte 0x%02x in position %d: " % (self.object[self.start], self.start)) + self.reason
        return prefix + ("bytes in position %d-%d: " % (self.start, self.end - 1)) + self.reason

def _unicode_error_character(character: str) -> str:
    code = ord(character)
    if code <= 0xFF:
        return "\\x%02x" % code
    if code <= 0xFFFF:
        return "\\u%04x" % code
    return "\\U%08x" % code

class UnicodeEncodeError(UnicodeError):
    encoding: str
    object: str
    start: int
    end: int
    reason: str
    _args: Tuple[str, str, int, int, str]

    def __init__(self, encoding: str, object: str, start: int, end: int, reason: str, /):
        super().__init__()
        self.encoding, self.object = encoding, object
        self.start, self.end, self.reason = start, end, reason
        self._args = encoding, object, start, end, reason
        self.message = self.__str__()

    @property
    def args(self):
        return self._args

    def __repr__(self, /):
        return self.typename + repr(self.args)

    def __str__(self, /):
        prefix = "'" + self.encoding + "' codec can't encode "
        if self.end == self.start + 1 and 0 <= self.start < len(self.object):
            return prefix + "character '" + _unicode_error_character(self.object[self.start]) + ("' in position %d: " % self.start) + self.reason
        return prefix + ("characters in position %d-%d: " % (self.start, self.end - 1)) + self.reason

class UnicodeTranslateError(UnicodeError):
    object: str
    start: int
    end: int
    reason: str
    _args: Tuple[str, int, int, str]

    def __init__(self, object: str, start: int, end: int, reason: str, /):
        super().__init__()
        self.object, self.start, self.end, self.reason = object, start, end, reason
        self._args = object, start, end, reason
        self.message = self.__str__()

    @property
    def encoding(self) -> Optional[str]:
        return None

    @property
    def args(self):
        return self._args

    def __repr__(self, /):
        return self.typename + repr(self.args)

    def __str__(self, /):
        if self.end == self.start + 1 and 0 <= self.start < len(self.object):
            return "can't translate character '" + _unicode_error_character(self.object[self.start]) + ("' in position %d: " % self.start) + self.reason
        return ("can't translate characters in position %d-%d: " % (self.start, self.end - 1)) + self.reason

class BlockingIOError(OSError):
    def __init__(self, /, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ChildProcessError(OSError):
    def __init__(self, /, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ConnectionError(OSError):
    def __init__(self, /, *args, **kwargs):
        super().__init__(*args, **kwargs)

class BrokenPipeError(ConnectionError):
    def __init__(self, /, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ConnectionAbortedError(ConnectionError):
    def __init__(self, /, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ConnectionRefusedError(ConnectionError):
    def __init__(self, /, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ConnectionResetError(ConnectionError):
    def __init__(self, /, *args, **kwargs):
        super().__init__(*args, **kwargs)

class FileExistsError(OSError):
    def __init__(self, /, *args, **kwargs):
        super().__init__(*args, **kwargs)

class FileNotFoundError(OSError):
    def __init__(self, /, *args, **kwargs):
        super().__init__(*args, **kwargs)

class InterruptedError(OSError):
    def __init__(self, /, *args, **kwargs):
        super().__init__(*args, **kwargs)

class IsADirectoryError(OSError):
    def __init__(self, /, *args, **kwargs):
        super().__init__(*args, **kwargs)

class NotADirectoryError(OSError):
    def __init__(self, /, *args, **kwargs):
        super().__init__(*args, **kwargs)

class PermissionError(OSError):
    def __init__(self, /, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ProcessLookupError(OSError):
    def __init__(self, /, *args, **kwargs):
        super().__init__(*args, **kwargs)

class TimeoutError(OSError):
    def __init__(self, /, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

# dataclasses.FrozenInstanceError (raised by compiler-generated frozen checks).
class FrozenInstanceError(AttributeError):
    def __init__(self, message: str = ""):
        super().__init__(message)
