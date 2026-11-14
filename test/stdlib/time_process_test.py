import time

def burn(n: int) -> float:
    s = 0.0
    for i in range(n):
        s += float(i % 7) * 1e-9
    return s

p0 = time.process_time()
t0 = time.thread_time()
w0 = time.perf_counter()
x = burn(30000000)
p1 = time.process_time()
t1 = time.thread_time()
w1 = time.perf_counter()
assert x > 0.0
assert p1 > p0 and t1 > t0
# CPU time of a busy single-threaded loop is close to (never much above) wall time
assert p1 - p0 <= (w1 - w0) * 1.5 + 0.01
assert t1 - t0 <= p1 - p0 + 0.01
assert isinstance(time.process_time_ns(), int) and time.process_time_ns() > 0
assert time.thread_time_ns() > 0
# sleeping does not consume CPU time
c0 = time.process_time()
time.sleep(0.05)
assert time.process_time() - c0 < 0.04
print("process_time ok")
