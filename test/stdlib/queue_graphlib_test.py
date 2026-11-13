# Ported from CPython 3.14 queue/graphlib (PSF)
import queue
from graphlib import TopologicalSorter, CycleError

@test
def test_queue():
    q = queue.Queue[int]()
    q.put(1); q.put(2); q.put(3)
    assert q.qsize() == 3
    assert q.get() == 1 and q.get() == 2
    assert not q.empty()
    q.get()
    assert q.empty()
    try:
        q.get_nowait()
        assert False
    except queue.Empty:
        pass
    b = queue.Queue[int](1)
    b.put(1)
    assert b.full()
    try:
        b.put_nowait(2)
        assert False
    except queue.Full:
        pass

@test
def test_lifo_prio():
    l = queue.LifoQueue[int]()
    l.put(1); l.put(2)
    assert l.get() == 2 and l.get() == 1
    p = queue.PriorityQueue[int]()
    for x in [5, 1, 4, 2]:
        p.put(x)
    assert [p.get(), p.get(), p.get(), p.get()] == [1, 2, 4, 5]

@test
def test_graphlib():
    ts = TopologicalSorter[str]()
    ts.add("D", "B", "C")
    ts.add("B", "A")
    ts.add("C", "A")
    assert list(ts.static_order()) == ["A", "B", "C", "D"]
    ts2 = TopologicalSorter[int]()
    ts2.add(1, 2)
    ts2.add(2, 1)
    try:
        ts2.prepare()
        assert False
    except CycleError:
        pass
    ts3 = TopologicalSorter[int]()
    ts3.add(3, 1)
    ts3.prepare()
    r = ts3.get_ready()
    assert r == [1]
    assert ts3.is_active()
    ts3.done(1)
    assert ts3.get_ready() == [3]
    ts3.done(3)
    assert not ts3.is_active()

test_queue()
test_lifo_prio()
test_graphlib()
