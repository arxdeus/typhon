class AbstractContextManager:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None


class _GeneratorContextManager[T]:
    gen: Generator[T]

    def __init__(self, gen: Generator[T]):
        self.gen = gen

    def __enter__(self) -> T:
        if self.gen.__done__():
            raise RuntimeError("generator didn't yield")
        return next(self.gen)

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_value is None:
            if self.gen.__done__():
                return False
            self.gen.__resume__()
            if not self.gen.__done__():
                raise RuntimeError("generator didn't stop")
            return False
        else:
            exc: BaseException = exc_value
            if self.gen.__done__():
                return False
            try:
                self.gen.throw(exc)
            except StopIteration:
                # The generator swallowed the exception: suppress it.
                return True
            except BaseException as e:
                if e.__raw__() == exc.__raw__():
                    # Re-raised the same exception: let the with statement propagate it.
                    return False
                raise
            raise RuntimeError("generator didn't stop after throw()")


def contextmanager(func):
    def helper(*args, **kwargs):
        return _GeneratorContextManager(func(*args, **kwargs))
    return helper


class _suppress[E]:
    _dummy: int

    def __init__(self):
        self._dummy = 0

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return isinstance(exc_value, E)


def suppress(E: type):
    return _suppress[E]()


class _nullcontext[T]:
    enter_result: T

    def __init__(self, enter_result: T):
        self.enter_result = enter_result

    def __enter__(self) -> T:
        return self.enter_result

    def __exit__(self, exc_type, exc_value, traceback):
        return None


def nullcontext(enter_result=None):
    return _nullcontext(enter_result)


class closing[T]:
    thing: T

    def __init__(self, thing: T):
        self.thing = thing

    def __enter__(self) -> T:
        return self.thing

    def __exit__(self, exc_type, exc_value, traceback):
        self.thing.close()
        return None


class _Callback:
    def call(self):
        pass


class _FnCallback[F](_Callback):
    fn: F

    def __init__(self, fn: F):
        super().__init__()
        self.fn = fn

    def call(self):
        self.fn()


class _ExitCallback[C](_Callback):
    cm: C

    def __init__(self, cm: C):
        super().__init__()
        self.cm = cm

    def call(self):
        BaseException._context_exit(self.cm.__exit__, None, None)


class ExitStack:
    _callbacks: List[_Callback]

    def __init__(self):
        self._callbacks = []

    def __enter__(self):
        return self

    def callback(self, fn):
        self._callbacks.append(_FnCallback(fn))
        return fn

    def push(self, cm):
        self._callbacks.append(_ExitCallback(cm))
        return cm

    def enter_context(self, cm):
        r = cm.__enter__()
        self._callbacks.append(_ExitCallback(cm))
        return r

    def pop_all(self) -> ExitStack:
        s = ExitStack()
        s._callbacks = self._callbacks
        self._callbacks = []
        return s

    def close(self):
        while self._callbacks:
            self._callbacks.pop().call()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


from internal.builtin import _TextSink, _TextSinkOf, _swap_print_sink


class _redirect_stream[T]:
    target: T
    _is_stderr: bool
    _saved: Optional[_TextSink]

    def __init__(self, target: T, is_stderr: bool):
        self.target = target
        self._is_stderr = is_stderr
        self._saved = None

    def __enter__(self) -> T:
        sink: _TextSink = _TextSinkOf(self.target)
        self._saved = _swap_print_sink(self._is_stderr, sink)
        return self.target

    def __exit__(self, exc_type, exc_value, traceback):
        _swap_print_sink(self._is_stderr, self._saved)
        return False


def redirect_stdout(new_target):
    """Redirect print() output (default file / sys.stdout) to `new_target`.

    Only print() is redirected: sys.stdout itself keeps its static File type,
    so direct `sys.stdout.write()` calls still go to the process stdout."""
    return _redirect_stream(new_target, False)


def redirect_stderr(new_target):
    """Redirect print(..., file=sys.stderr) to `new_target` (see redirect_stdout)."""
    return _redirect_stream(new_target, True)
