# Ported from CPython 3.14 queue (PSF license), single-threaded
from heapq import heappush, heappop

class Empty(Exception):
    def __init__(self):
        super().__init__("")

class Full(Exception):
    def __init__(self):
        super().__init__("")

class ShutDown(Exception):
    def __init__(self):
        super().__init__("")

class Queue[T]:
    maxsize: int
    _q: List[T]
    _head: int
    unfinished_tasks: int

    def __init__(self, maxsize: int = 0):
        self.maxsize = maxsize
        self._q = []
        self._head = 0
        self.unfinished_tasks = 0

    def qsize(self) -> int:
        return len(self._q) - self._head

    def empty(self) -> bool:
        return self.qsize() == 0

    def full(self) -> bool:
        return self.maxsize > 0 and self.qsize() >= self.maxsize

    def put(self, item: T, block: bool = True, timeout: Optional[float] = None):
        if self.full():
            raise Full()
        self._q.append(item)
        self.unfinished_tasks += 1

    def put_nowait(self, item: T):
        self.put(item, False)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> T:
        if self.empty():
            raise Empty()
        v = self._q[self._head]
        self._head += 1
        if self._head > 32 and self._head * 2 > len(self._q):
            self._q = self._q[self._head:]
            self._head = 0
        return v

    def get_nowait(self) -> T:
        return self.get(False)

    def task_done(self):
        if self.unfinished_tasks <= 0:
            raise ValueError("task_done() called too many times")
        self.unfinished_tasks -= 1

    def join(self):
        pass

class LifoQueue[T]:
    maxsize: int
    _q: List[T]
    unfinished_tasks: int

    def __init__(self, maxsize: int = 0):
        self.maxsize = maxsize
        self._q = []
        self.unfinished_tasks = 0

    def qsize(self) -> int:
        return len(self._q)

    def empty(self) -> bool:
        return len(self._q) == 0

    def full(self) -> bool:
        return self.maxsize > 0 and len(self._q) >= self.maxsize

    def put(self, item: T, block: bool = True, timeout: Optional[float] = None):
        if self.full():
            raise Full()
        self._q.append(item)
        self.unfinished_tasks += 1

    def put_nowait(self, item: T):
        self.put(item, False)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> T:
        if self.empty():
            raise Empty()
        return self._q.pop()

    def get_nowait(self) -> T:
        return self.get(False)

    def task_done(self):
        if self.unfinished_tasks <= 0:
            raise ValueError("task_done() called too many times")
        self.unfinished_tasks -= 1

    def join(self):
        pass

class PriorityQueue[T]:
    maxsize: int
    _q: List[T]
    unfinished_tasks: int

    def __init__(self, maxsize: int = 0):
        self.maxsize = maxsize
        self._q = []
        self.unfinished_tasks = 0

    def qsize(self) -> int:
        return len(self._q)

    def empty(self) -> bool:
        return len(self._q) == 0

    def full(self) -> bool:
        return self.maxsize > 0 and len(self._q) >= self.maxsize

    def put(self, item: T, block: bool = True, timeout: Optional[float] = None):
        if self.full():
            raise Full()
        heappush(self._q, item)
        self.unfinished_tasks += 1

    def put_nowait(self, item: T):
        self.put(item, False)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> T:
        if self.empty():
            raise Empty()
        return heappop(self._q)

    def get_nowait(self) -> T:
        return self.get(False)

    def task_done(self):
        if self.unfinished_tasks <= 0:
            raise ValueError("task_done() called too many times")
        self.unfinished_tasks -= 1

    def join(self):
        pass

class SimpleQueue[T](Queue[T]):
    def __init__(self):
        super().__init__(0)
