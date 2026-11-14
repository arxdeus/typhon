# Ported from CPython 3.14 queue (PSF license)
# Thread-safe: one mutex and not_empty/not_full/all_tasks_done conditions, like CPython.
from heapq import heappush, heappop
from threading import Lock, Condition

class Empty(Exception):
    def __init__(self):
        super().__init__("")

class Full(Exception):
    def __init__(self):
        super().__init__("")

class ShutDown(Exception):
    def __init__(self):
        super().__init__("")

def _now() -> float:
    return _C.seq_time_monotonic() / 1e9

class Queue[T]:
    """Create a queue object with a given maximum size.

    If maxsize is <= 0, the queue size is infinite.
    """
    maxsize: int
    _q: List[T]
    _head: int
    unfinished_tasks: int
    is_shutdown: bool
    mutex: Lock
    not_empty: Condition
    not_full: Condition
    all_tasks_done: Condition

    def __init__(self, maxsize: int = 0):
        self.maxsize = maxsize
        self._q = []
        self._head = 0
        self.unfinished_tasks = 0
        self.is_shutdown = False
        self.mutex = Lock()
        self.not_empty = Condition(self.mutex)
        self.not_full = Condition(self.mutex)
        self.all_tasks_done = Condition(self.mutex)

    # Override these in subclasses (called with the mutex held).
    def _qsize(self) -> int:
        return len(self._q) - self._head

    def _put(self, item: T):
        self._q.append(item)

    def _get(self) -> T:
        v = self._q[self._head]
        self._head += 1
        if self._head > 32 and self._head * 2 > len(self._q):
            self._q = self._q[self._head:]
            self._head = 0
        return v

    def task_done(self):
        with self.all_tasks_done:
            unfinished = self.unfinished_tasks - 1
            if unfinished <= 0:
                if unfinished < 0:
                    raise ValueError("task_done() called too many times")
                self.all_tasks_done.notify_all()
            self.unfinished_tasks = unfinished

    def join(self):
        with self.all_tasks_done:
            while self.unfinished_tasks:
                self.all_tasks_done._wait_raw(-1.0)

    def qsize(self) -> int:
        with self.mutex:
            return self._qsize()

    def empty(self) -> bool:
        with self.mutex:
            return not self._qsize()

    def full(self) -> bool:
        with self.mutex:
            return 0 < self.maxsize <= self._qsize()

    def put(self, item: T, block: bool = True, timeout: Optional[float] = None):
        with self.not_full:
            if self.is_shutdown:
                raise ShutDown()
            if self.maxsize > 0:
                if not block:
                    if self._qsize() >= self.maxsize:
                        raise Full()
                elif timeout is None:
                    while self._qsize() >= self.maxsize:
                        self.not_full._wait_raw(-1.0)
                        if self.is_shutdown:
                            raise ShutDown()
                elif timeout < 0:
                    raise ValueError("'timeout' must be a non-negative number")
                else:
                    endtime = _now() + timeout
                    while self._qsize() >= self.maxsize:
                        remaining = endtime - _now()
                        if remaining <= 0.0:
                            raise Full()
                        self.not_full._wait_raw(remaining)
                        if self.is_shutdown:
                            raise ShutDown()
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()

    def get(self, block: bool = True, timeout: Optional[float] = None) -> T:
        with self.not_empty:
            if self.is_shutdown and not self._qsize():
                raise ShutDown()
            if not block:
                if not self._qsize():
                    raise Empty()
            elif timeout is None:
                while not self._qsize():
                    self.not_empty._wait_raw(-1.0)
                    if self.is_shutdown and not self._qsize():
                        raise ShutDown()
            elif timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                endtime = _now() + timeout
                while not self._qsize():
                    remaining = endtime - _now()
                    if remaining <= 0.0:
                        raise Empty()
                    self.not_empty._wait_raw(remaining)
                    if self.is_shutdown and not self._qsize():
                        raise ShutDown()
            item = self._get()
            self.not_full.notify()
            return item

    def put_nowait(self, item: T):
        self.put(item, block=False)

    def get_nowait(self) -> T:
        return self.get(block=False)

    def shutdown(self, immediate: bool = False):
        with self.mutex:
            self.is_shutdown = True
            if immediate:
                while self._qsize():
                    self._get()
                    if self.unfinished_tasks > 0:
                        self.unfinished_tasks -= 1
                self.all_tasks_done.notify_all()
            self.not_empty.notify_all()
            self.not_full.notify_all()

class LifoQueue[T](Queue[T]):
    def __init__(self, maxsize: int = 0):
        super().__init__(maxsize)

    def _qsize(self) -> int:
        return len(self._q)

    def _put(self, item: T):
        self._q.append(item)

    def _get(self) -> T:
        return self._q.pop()

class PriorityQueue[T](Queue[T]):
    def __init__(self, maxsize: int = 0):
        super().__init__(maxsize)

    def _qsize(self) -> int:
        return len(self._q)

    def _put(self, item: T):
        heappush(self._q, item)

    def _get(self) -> T:
        return heappop(self._q)

class SimpleQueue[T](Queue[T]):
    def __init__(self):
        super().__init__(0)
