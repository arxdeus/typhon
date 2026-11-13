# Ported from CPython 3.14 functools (PSF)
from functools import reduce

def add(a: int, b: int) -> int:
    return a + b

@test
def test_reduce():
    assert reduce(add, [1, 2, 3], 10) == 16
    assert reduce(add, [1, 2, 3, 4]) == 10
    assert reduce(add, [5]) == 5
    assert reduce(lambda a, b: a * b, range(1, 6)) == 120

test_reduce()
