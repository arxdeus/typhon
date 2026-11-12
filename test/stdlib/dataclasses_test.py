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


@dataclass
class Bag:
    items: List[int] = field(default_factory=List[int])
    n: int = 0

    def __post_init__(self):
        self.n = len(self.items) + 1


@dataclass(frozen=True)
class Frozen:
    x: int


@dataclass(kw_only=True)
class KW:
    a: int
    b: str = "b"


@dataclass(order=True)
class Ord:
    v: int


@test
def test_semantics():
    b1 = Bag()
    b2 = Bag()
    b1.items.append(1)
    assert b2.items == List[int]()  # fresh object per instance
    assert b1.n == 1 and Bag([1, 2]).n == 3
    assert repr(Bag([1], 5)) == "Bag(items=[1], n=2)"
    assert str(KW(a=1)) == "KW(a=1, b='b')"
    f = Frozen(1)
    caught = False
    try:
        f.x = 2
    except dataclasses.FrozenInstanceError as e:
        caught = str(e) == "cannot assign to field 'x'"
    assert caught and f.x == 1
    assert Ord(1) < Ord(2) and Ord(3) == Ord(3)
test_semantics()
