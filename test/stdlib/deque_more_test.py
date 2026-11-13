# Ported from CPython 3.14 collections.deque (PSF)
from collections import deque, Counter

@test
def test_deque_ops():
    d = deque([1, 2, 3], 5)
    d.extend([4, 5, 6])
    assert list(d) == [2, 3, 4, 5, 6]
    assert repr(d) == "deque([2, 3, 4, 5, 6], maxlen=5)"
    d.rotate(2)
    assert list(d) == [5, 6, 2, 3, 4]
    d.rotate(-1)
    assert list(d) == [6, 2, 3, 4, 5]
    d.extendleft([9])
    assert list(d) == [9, 6, 2, 3, 4]
    assert d.count(9) == 1 and d.index(3) == 3
    d.reverse()
    assert list(d) == [4, 3, 2, 6, 9]
    d.remove(2)
    assert list(d) == [4, 3, 6, 9]
    d[1] = 10
    del d[0]
    assert list(d) == [10, 6, 9]
    assert list(reversed(d)) == [9, 6, 10]

@test
def test_counter():
    c = Counter("abracadabra")
    assert c.most_common(2) == [("a", 5), ("b", 2)]
    assert c.total() == 11
    assert len(list(c.elements())) == 11

test_deque_ops()
test_counter()
