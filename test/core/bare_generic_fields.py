# Copyright (C) 2026 the Typhon authors
# Fields annotated with a bare generic class deduce the element type from the constructor.
import queue
from dataclasses import dataclass

@dataclass
class B:
    dq: list
    n: int = 0

b = B([1])
print(b.dq, b.n)  #: [1] 0

class C:
    xs: list
    def __init__(self, xs):
        self.xs = xs

print(C(['a']).xs)  #: ['a']

@dataclass
class Backend:
    data_queue: queue.Queue
    threads: list
    extra: list | None = None

q = queue.Queue(maxsize=4)
q.put(2.5)
be = Backend(q, [1, 2])
print(be.data_queue.get(), be.threads, be.extra is None)  #: 2.5 [1, 2] True
