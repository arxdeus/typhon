# Copyright (C) 2026 the Typhon authors
# Python semantics: implicit Optional for `x: T = None`, bare generic params, sys.path[0] imports.
from typing import List
from implicit_pkg.sub.b import get

def f(t: list = None):
    if t is None:
        t = ['a', 'b']
    return len(t)

def g(x: int = None):
    return 0 if x is None else x

def h(x: List[str] = None):
    return 0 if x is None else len(x)

def s(x: str = None, d: dict = None):
    return ('-' if x is None else x, 0 if d is None else len(d))

def app(t: list):
    t.append('x')
    return len(t)

class A:
    def m(self, *, n: int = None, s: str = None):
        return (0 if n is None else n, s)

print(f(), f(['a', 'b', 'c']), f(None))  #: 2 3 2
print(g(), g(4), g(None))  #: 0 4 0
print(h(), h(['q']))  #: 0 1
print(s(), s('a', {'k': 1}))  #: ('-', 0) ('a', 1)
l = ['y']
print(app(l), l)  #: 2 ['y', 'x']
print(A().m(), A().m(n=3, s='q'))  #: (0, None) (3, 'q')
print(get())  #: 42
