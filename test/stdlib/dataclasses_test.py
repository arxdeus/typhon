import dataclasses
from dataclasses import dataclass, field, asdict, astuple, replace, fields


@dataclass(eq=True, order=True)
class Point:
    x: int
    y: int = 0


@dataclass
class Item:
    name: str
    qty: int = field(default=1)
    tags: List[str] = field(default_factory=List[str])


@test
def test_basic():
    p = Point(1, 2)
    assert p.x == 1 and p.y == 2
    assert Point(3).y == 0
    assert p == Point(1, 2)
    assert p != Point(2, 1)
    assert p < Point(2, 0)
    assert Point(1, 1) < p
test_basic()


@test
def test_helpers():
    p = Point(1, 2)
    assert astuple(p) == (1, 2)
    assert asdict(p) == {"x": 1, "y": 2}
    assert [f.name for f in fields(p)] == ["x", "y"]
    q = replace(p, y=9)
    assert q.x == 1 and q.y == 9 and p.y == 2
    it = Item("a")
    assert it.qty == 1
    assert it.tags == List[str]()
    assert astuple(Item("b", 3)) == ("b", 3, List[str]())
test_helpers()
