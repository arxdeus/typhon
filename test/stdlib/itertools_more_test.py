# Ported from CPython 3.14 itertools (PSF)
from itertools import pairwise, batched

@test
def test_pairwise():
    assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(pairwise([1])) == []

@test
def test_batched():
    assert list(batched([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(batched(range(4), 2)) == [[0, 1], [2, 3]]

test_pairwise()
test_batched()
