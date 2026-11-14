# Copyright (C) 2026 the Typhon authors
# Python semantics: static len(tuple) rebinding, narrowing after `if x is None: <exit>`,
# and first-assignment field order.
from typing import List, Optional

def rebind(t):
    if len(t) == 2:
        t = (t[0], t[1], 0)
    return t

print(rebind((1, 2)), rebind((1, 2, 3)))  #: (1, 2, 0) (1, 2, 3)

def g(x: int):
    return x * 2

def drain(q: List[Optional[int]]):
    s = 0
    while True:
        c = q.pop()
        if c is None:
            break
        s += g(c)
    return s

print(drain([None, 1, 2]))  #: 6

def h(c: Optional[int]):
    if c is None:
        return -1
    return g(c)

print(h(3), h(None))  #: 6 -1

class C:
    def __init__(self):
        self.z = 1
        self.a = 2
        self.m = 3

print([k for k, _ in static.vars(C())])  #: ['z', 'a', 'm']

def _sq(x):
    return x * x

def narrow_untyped_loop(q):
    s = 0
    while True:
        c = q.pop()
        if c is None:
            break
        s += _sq(c)
    return s
print(narrow_untyped_loop([None, 2, 3]))  #: 13

def narrow_not_none(c: Optional[int]):
    if c is not None:
        return _sq(c)
    return -1
print(narrow_not_none(4), narrow_not_none(None))  #: 16 -1

def narrow_rebound(c: Optional[int]):
    if c is not None:
        c = None
    return c is None
print(narrow_rebound(1))  #: True
