import functools

calls = [0]

@functools.cache
def sq(n: int) -> int:
    calls[0] += 1
    return n * n

print(sq(3), sq(3), sq(4), calls[0])  #: 9 9 16 2

@functools.lru_cache(maxsize=None)
def add(a: int, b: float) -> float:
    calls[0] += 1
    return a + b

print(add(1, 2.5), add(1, 2.5), calls[0])  #: 3.5 3.5 3

def mk():
    d = {}
    def inner(n: int):
        d[n] = n * 2
        return d[n]
    return inner
print(mk()(3))  #: 6
