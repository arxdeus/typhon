from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence, Mapping, Callable, Set
from collections import deque, Counter


class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        return 0.0

    def describe(self) -> str:
        return f"area={self.area()}"


class Sq(Shape):
    s: float

    def __init__(self, s: float):
        super().__init__()
        self.s = s

    def area(self) -> float:
        return self.s * self.s


def total(xs: Iterable[int]) -> int:
    return sum(xs)


def size(xs: Sequence[int]) -> int:
    return len(xs)


def apply(f: Callable[[int], int], v: int) -> int:
    return f(v)


@test
def test_abc():
    shapes = List[Shape]()
    shapes.append(Sq(2.0))
    assert shapes[0].describe() == "area=4.0"
    assert isinstance(Sq(1.0), Shape)
    assert total(i for i in range(4)) == 6
    assert size([1, 2, 3]) == 3
    assert apply(lambda x: x * 2, 4) == 8
    assert isinstance([1], Sequence)
    assert isinstance({1: 2}, Mapping)
    assert isinstance({1}, Set)
    d = deque([1, 2])
    assert len(d) == 2
    assert Counter("aab")["a"] == 2
test_abc()
