# Ported from CPython 3.14 timeit (PSF license), callable-based subset
from time import perf_counter

default_timer = perf_counter

def timeit(stmt, number: int = 1000000) -> float:
    t0 = default_timer()
    for _ in range(number):
        stmt()
    return default_timer() - t0

def repeat(stmt, repeat: int = 5, number: int = 1000000) -> List[float]:
    r = List[float]()
    for _ in range(repeat):
        r.append(timeit(stmt, number))
    return r
