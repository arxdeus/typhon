# Python int -> float widening of function locals: a variable bound to an int literal
# and later assigned (or in-place updated with) a float holds the float, as in Python.
# Int-only locals stay int.

def best_of(vals):
    best = 0
    for v in vals:
        if v > best:
            best = v
    return best

def acc(vals):
    c = 0
    for v in vals:
        c += v
    return c

def count(n):
    k = 0
    for i in range(n):
        k += i
    return k

def mixed():
    correct = 0
    total = 0
    for i in range(4):
        correct += (i % 2) * 0.5
        total += 1
    return correct / total, total

def nested():
    def inner(xs):
        b = 0
        a = 0
        for x in xs:
            if x > b:
                b = x
                a = x * 2.0
        return f"{b:.4f} {a:.4f}"
    return inner([0.3, 0.9])

@test
def t_widen():
    assert best_of([0.25, 0.75, 0.5]) == 0.75
    assert isinstance(best_of([0.25]), float)
    assert best_of([1, 3, 2]) == 3
    assert isinstance(best_of([1, 3, 2]), int)
    assert acc([1.5, 2.25]) == 3.75
    assert isinstance(acc([1.5]), float)
    assert count(5) == 10
    assert isinstance(count(5), int)
    r, t = mixed()
    assert r == 0.25 and t == 4 and isinstance(t, int)
    assert nested() == "0.9000 1.8000"
    # Static typing: once any float flows in, the local is float on every path (Python
    # would keep the int 0 when no float is assigned at run time); values compare equal.
    assert best_of([-1.0]) == 0
t_widen()
