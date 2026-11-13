# Ported from CPython 3.14 collections (PSF); expected values from the reference interpreter
from collections import OrderedDict, Counter, ChainMap

@test
def test_ordered():
    o = OrderedDict[int, int]()
    o[1] = 2
    o[3] = 4
    o[5] = 6
    o.move_to_end(1)
    assert list(o.items()) == [(3, 4), (5, 6), (1, 2)]
    o.move_to_end(1, False)
    assert list(o.items()) == [(1, 2), (3, 4), (5, 6)]
    assert o.popitem(last=False) == (1, 2)
    assert o.popitem() == (5, 6)

@test
def test_counter():
    c = Counter("abracadabra")
    d = Counter("alacazam")
    assert c.most_common(3) == [("a", 5), ("b", 2), ("r", 2)]
    assert (c + d).most_common(2) == [("a", 9), ("b", 2)]
    assert (c - d).most_common(2) == [("b", 2), ("r", 2)]
    assert (c & d).most_common(2) == [("a", 4), ("c", 1)]
    assert (c | d).most_common(2) == [("a", 5), ("b", 2)]
    c.subtract(d)
    assert sorted(c.items()) == [("a", 1), ("b", 2), ("c", 0), ("d", 1), ("l", -1), ("m", -1), ("r", 2), ("z", -1)]
    c.update("zzz")
    assert c["z"] == 2 and c.total() == 6
    assert Counter("ab") <= Counter("abc")
    assert Counter("ab") < Counter("abc")
    assert not (Counter("abc") <= Counter("ab"))

@test
def test_chainmap():
    a = {"x": 1, "y": 2}
    b = {"y": 20, "z": 30}
    cm = ChainMap([a, b])
    assert cm["y"] == 2 and cm["z"] == 30
    assert len(cm) == 3
    assert "z" in cm and "q" not in cm
    cm["w"] = 5
    assert a["w"] == 5
    assert cm.keys() == ["y", "z", "x", "w"]
    ch = cm.new_child()
    ch["x"] = 100
    assert ch["x"] == 100 and cm["x"] == 1
    assert ch.parents["x"] == 1

test_ordered()
test_counter()
test_chainmap()
