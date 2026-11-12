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

    def __exit__(self, exc_type, exc_value, traceback):
        # The dialect cannot throw an exception into a suspended generator,
        # so the code after `yield` runs (like a `finally:` block) and the
        # original exception, if any, propagates.
        if not self.gen.__done__():
            self.gen.__resume__()
            if not self.gen.__done__():
                raise RuntimeError("generator didn't stop")
        return False


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
