# threading.Thread/Event/Condition/Semaphore and blocking thread-safe queue.Queue
# (behaviour as in CPython 3.14)
import threading
import queue
import time

def _add(xs: List[int], a: int, b: int):
    xs.append(a + b)

@test
def test_thread_basic():
    xs: List[int] = []
    t = threading.Thread(target=_add, args=(xs, 2, 3), daemon=True)
    assert not t.is_alive()
    assert t.ident is None
    assert t.daemon
    t.start()
    t.join()
    assert xs == [5]
    assert not t.is_alive()
    assert t.ident is not None
    assert t.name.startswith("Thread-") and t.name.endswith("(_add)")
    try:
        t.start()
        assert False
    except RuntimeError:
        pass
    n = threading.Thread(target=_add, args=(xs, 1, 1), name="worker")
    assert n.name == "worker" and not n.daemon
    n.start(); n.join()
    assert xs == [5, 2]

def _sleep_then_set(e: threading.Event, s: float):
    time.sleep(s)
    e.set()

@test
def test_event():
    e = threading.Event()
    assert not e.is_set()
    t0 = time.time()
    assert e.wait(0.1) == False
    assert time.time() - t0 >= 0.09
    t = threading.Thread(target=_sleep_then_set, args=(e, 0.05))
    t.start()
    assert e.wait(5.0) == True
    assert e.is_set()
    t.join()
    e.clear()
    assert not e.is_set()
    e.set()
    assert e.wait()

@test
def test_join_timeout():
    e = threading.Event()
    t = threading.Thread(target=e.wait, args=(10.0,), daemon=True)
    t.start()
    t0 = time.time()
    t.join(timeout=0.1)
    assert t.is_alive()
    assert time.time() - t0 < 2.0
    e.set()
    t.join()
    assert not t.is_alive()

def _boom(x: int):
    raise ValueError(f"bad {x}")

@test
def test_thread_exception():
    # Printed to stderr by excepthook; the process and other threads survive.
    t = threading.Thread(target=_boom, args=(7,))
    t.start()
    t.join()
    assert not t.is_alive()

@test
def test_queue_nowait_timeout():
    q = queue.Queue[str](maxsize=2)
    try:
        q.get_nowait()
        assert False
    except queue.Empty:
        pass
    t0 = time.time()
    try:
        q.get(timeout=0.1)
        assert False
    except queue.Empty:
        pass
    assert time.time() - t0 >= 0.09
    q.put_nowait("a")
    q.put("b")
    assert q.full() and q.qsize() == 2
    try:
        q.put_nowait("c")
        assert False
    except queue.Full:
        pass
    try:
        q.put("c", timeout=0.05)
        assert False
    except queue.Full:
        pass
    assert q.get() == "a" and q.get_nowait() == "b"
    assert q.empty()

def _producer(q: queue.Queue[Optional[int]], base: int, n: int):
    for i in range(n):
        q.put(base + i)

def _consumer(q: queue.Queue[Optional[int]], out: queue.Queue[int]):
    s = 0
    while True:
        v = q.get()
        if v is None:
            break
        s += v
        q.task_done()
    q.task_done()
    out.put(s)

@test
def test_queue_stress():
    q = queue.Queue[Optional[int]](maxsize=8)
    out = queue.Queue[int]()
    P, C, N = 6, 4, 5000
    cons = [threading.Thread(target=_consumer, args=(q, out), daemon=True) for _ in range(C)]
    prods = [threading.Thread(target=_producer, args=(q, p * N, N)) for p in range(P)]
    for t in cons:
        t.start()
    for t in prods:
        t.start()
    for t in prods:
        t.join()
    for _ in range(C):
        q.put(None)
    q.join()
    for t in cons:
        t.join(timeout=5.0)
        assert not t.is_alive()
    total = 0
    for _ in range(C):
        total += out.get(timeout=1.0)
    M = P * N
    assert total == M * (M - 1) // 2
    assert q.qsize() == 0

def _count(lock: threading.Lock, box: List[int], n: int):
    for _ in range(n):
        with lock:
            box[0] += 1

@test
def test_lock_and_semaphore():
    lock = threading.Lock()
    box = [0]
    ts = [threading.Thread(target=_count, args=(lock, box, 20000)) for _ in range(4)]
    for t in ts: t.start()
    for t in ts: t.join()
    assert box[0] == 80000
    s = threading.Semaphore(2)
    assert s.acquire() and s.acquire()
    assert not s.acquire(blocking=False)
    assert not s.acquire(timeout=0.05)
    s.release()
    assert s.acquire(blocking=False)

@test
def test_daemon_does_not_block_exit():
    d = threading.Thread(target=time.sleep, args=(100.0,), daemon=True)
    d.start()
    assert d.is_alive()

test_thread_basic()
test_event()
test_join_timeout()
test_thread_exception()
test_queue_nowait_timeout()
test_queue_stress()
test_lock_and_semaphore()
test_daemon_does_not_block_exit()
print("threading/queue OK")
