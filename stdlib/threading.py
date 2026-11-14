# Thread, Event, Condition, Semaphore, excepthook ported from CPython 3.14 threading (PSF license)

from C import pthread_create(Ptr[cobj], cobj, cobj, cobj) -> i32
from C import pthread_join(cobj, Ptr[cobj]) -> i32
from C import pthread_detach(cobj) -> i32
from C import pthread_self() -> cobj
from C import seq_register_current_thread() -> None
from C import seq_allow_register_threads() -> None
from C import atexit(cobj) -> i32

# Thread-local variable indicator. Used internally by the compiler to mark ThreadLocal variables.
# Semantically identical to T.
@tuple
class ThreadLocal[T]:
    def __new__(T: type) -> ThreadLocal[T]:
        compile_error("this type cannot be instantiated")

@tuple
class Lock:
    p: cobj

    def __new__() -> Lock:
        return Lock(_C.seq_lock_new(),)

    def acquire(self, /, block: bool = True, timeout: float = -1.0) -> bool:
        if timeout >= 0.0 and not block:
            raise ValueError("can't specify a timeout for a non-blocking call")
        return _C.seq_lock_acquire(self.p, block, timeout)

    def release(self, /):
        _C.seq_lock_release(self.p)

    def __enter__(self, /):
        return self.acquire()

    def __exit__(self, /):
        self.release()

@tuple
class RLock:
    p: cobj

    def __new__() -> RLock:
        return RLock(_C.seq_rlock_new(),)

    def acquire(self, /, block: bool = True, timeout: float = -1.0) -> bool:
        if timeout >= 0.0 and not block:
            raise ValueError("can't specify a timeout for a non-blocking call")
        return _C.seq_rlock_acquire(self.p, block, timeout)

    def release(self, /):
        _C.seq_rlock_release(self.p)

    def __enter__(self, /):
        return self.acquire()

    def __exit__(self, /):
        self.release()

def _now() -> float:
    return _C.seq_time_monotonic() / 1e9

class Condition:
    """Condition variable bound to a (non-reentrant) Lock."""
    _lock: Lock
    _cv: cobj
    _waiters: int

    def __init__(self, lock: Optional[Lock] = None):
        self._lock = lock if lock is not None else Lock()
        self._cv = _C.seq_cond_new()
        self._waiters = 0

    def acquire(self, block: bool = True, timeout: float = -1.0) -> bool:
        return self._lock.acquire(block, timeout)

    def release(self):
        self._lock.release()

    def __enter__(self):
        return self._lock.acquire()

    def __exit__(self):
        self._lock.release()

    def _wait_raw(self, timeout: float):
        self._waiters += 1
        _C.seq_cond_wait(self._cv, self._lock.p, timeout)
        self._waiters -= 1

    def wait(self, timeout: Optional[float] = None) -> bool:
        # The lock must be held. Returns False if the timeout expired.
        # (Spurious wakeups are possible, as with CPython: use wait_for.)
        if timeout is None:
            self._wait_raw(-1.0)
            return True
        t = timeout
        if t < 0.0:
            t = 0.0
        end = _now() + t
        self._wait_raw(t)
        return _now() < end

    def wait_for(self, predicate, timeout: Optional[float] = None) -> bool:
        if timeout is None:
            result = predicate()
            while not result:
                self._wait_raw(-1.0)
                result = predicate()
            return bool(result)
        end = _now() + timeout
        result = predicate()
        while not result:
            left = end - _now()
            if left <= 0.0:
                break
            self._wait_raw(left)
            result = predicate()
        return bool(result)

    def notify(self, n: int = 1):
        k = n if n < self._waiters else self._waiters
        for _ in range(k):
            _C.seq_cond_notify(self._cv)

    def notify_all(self):
        self.notify(self._waiters)

class Semaphore:
    _cond: Condition
    _value: int

    def __init__(self, value: int = 1):
        if value < 0:
            raise ValueError("semaphore initial value must be >= 0")
        self._cond = Condition()
        self._value = value

    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        if not blocking and timeout is not None:
            raise ValueError("can't specify timeout for non-blocking acquire")
        rc = False
        with self._cond:
            if not blocking:
                if self._value > 0:
                    self._value -= 1
                    rc = True
            else:
                rc = self._cond.wait_for(lambda: self._value > 0, timeout)
                if rc:
                    self._value -= 1
        return rc

    def release(self, n: int = 1):
        with self._cond:
            self._value += n
            self._cond.notify(n)

    def __enter__(self):
        return self.acquire()

    def __exit__(self):
        self.release()

class Event:
    _cond: Condition
    _flag: bool

    def __init__(self):
        self._cond = Condition()
        self._flag = False

    def __repr__(self) -> str:
        return f"<threading.Event: {'set' if self._flag else 'unset'}>"

    def is_set(self) -> bool:
        return self._flag

    def set(self):
        with self._cond:
            self._flag = True
            self._cond.notify_all()

    def clear(self):
        with self._cond:
            self._flag = False

    def wait(self, timeout: Optional[float] = None) -> bool:
        with self._cond:
            if self._flag:
                return True
            if timeout is None:
                while not self._flag:
                    self._cond._wait_raw(-1.0)
                return True
            return self._cond.wait_for(lambda: self._flag, timeout)

_counter_lock = Lock()
_counter: int = 0
_live_lock = Lock()
_atexit_installed: bool = False

def _newname(suffix: str) -> str:
    global _counter
    with _counter_lock:
        _counter += 1
        n = _counter
    return f"Thread-{n} ({suffix})" if suffix else f"Thread-{n}"

def _thread_entry(data: cobj, F: type, A: type) -> cobj:
    seq_register_current_thread()
    p = Ptr[Tuple[F, A, Thread]](data)
    fn, args, th = p[0]
    th._ident = pthread_self().__int__()
    th._started.set()
    try:
        fn(*args)
    except SystemExit:
        pass
    except BaseException as e:
        excepthook(e, th)
    th._mark_finished()
    return cobj()

def excepthook(e, thread):
    # Like CPython's threading.excepthook (frames are not available).
    import sys
    sys.stderr.write(f"Exception in thread {thread.name}:\n")
    sys.stderr.write("Traceback (most recent call last):\n")
    msg = str(e)
    sys.stderr.write(f"{e.typename}: {msg}\n" if msg else f"{e.typename}\n")
    sys.stderr.flush()

def _join_non_daemon():
    while True:
        t: Optional[Thread] = None
        with _live_lock:
            for x in _non_daemon:
                if x.is_alive():
                    t = x
                    break
        if t is None:
            return
        t.join()

class Thread:
    """OS thread running `target(*args)`. Exceptions go to `excepthook`."""
    name: str
    daemon: bool
    _fn: cobj
    _data: cobj
    _tid: cobj
    _ident: int
    _started: Event
    _done: Event
    _start_called: bool

    def __init__(self, group = None, target = None, name: Optional[str] = None,
                 args = (), kwargs = None, daemon: Optional[bool] = None):
        if kwargs is not None:
            compile_error("Thread(kwargs=...) is not supported; use args or a lambda")
        self._start_called = False
        self._ident = 0
        self._tid = cobj()
        self._started = Event()
        self._done = Event()
        self.daemon = daemon if daemon is not None else False
        if target is None:
            self._fn = cobj()
            self._data = cobj()
            self.name = name if name is not None else _newname("")
        else:
            F = type(target)
            A = type(args)
            p = Ptr[Tuple[F, A, Thread]](1)
            p[0] = (target, args, self)
            self._data = p.as_byte()
            self._fn = _thread_entry(F=F, A=A, ...).F.T.__raw__()
            self.name = name if name is not None else _newname(target.__name__.split("[")[0].split("(")[0])

    def __repr__(self) -> str:
        status = "initial"
        if self._started.is_set():
            status = "stopped" if self._done.is_set() else "started"
        if self.daemon:
            status += " daemon"
        if self._ident:
            status += f" {self._ident}"
        return f"<Thread({self.name}, {status})>"

    @property
    def ident(self) -> Optional[int]:
        return self._ident if self._ident else None

    @property
    def native_id(self) -> Optional[int]:
        return self.ident

    def isDaemon(self) -> bool:
        return self.daemon

    def setDaemon(self, daemonic: bool):
        self.daemon = daemonic

    def getName(self) -> str:
        return self.name

    def setName(self, name: str):
        self.name = name

    def _mark_finished(self):
        self._done.set()

    def start(self):
        global _atexit_installed
        if self._start_called:
            raise RuntimeError("threads can only be started once")
        self._start_called = True
        if not self._fn:
            self._started.set()
            self._done.set()
            return
        if not self.daemon:
            with _live_lock:
                _non_daemon.append(self)
                if not _atexit_installed:
                    _atexit_installed = True
                    atexit(_join_non_daemon.__raw__())
        seq_allow_register_threads()
        tid = cobj()
        rc = pthread_create(__ptr__(tid), cobj(), self._fn, self._data)
        if rc != i32(0):
            raise RuntimeError("can't start new thread")
        self._tid = tid
        pthread_detach(tid)
        self._started.wait()

    def run(self):
        pass

    def join(self, timeout: Optional[float] = None):
        if not self._start_called:
            raise RuntimeError("cannot join thread before it is started")
        if self._ident != 0 and self._ident == pthread_self().__int__():
            raise RuntimeError("cannot join current thread")
        self._done.wait(timeout)

    def is_alive(self) -> bool:
        return self._started.is_set() and not self._done.is_set()

_non_daemon: List[Thread] = []

class _MainThread:
    name: str
    daemon: bool
    ident: int

    def __init__(self):
        self.name = "MainThread"
        self.daemon = False
        self.ident = pthread_self().__int__()

    def is_alive(self) -> bool:
        return True

_main_thread = _MainThread()

def main_thread() -> _MainThread:
    return _main_thread

def active_count() -> int:
    from openmp import get_num_threads
    return get_num_threads()

def get_native_id() -> int:
    from openmp import get_thread_num
    return get_thread_num()

def get_ident() -> int:
    return get_native_id() + 1
