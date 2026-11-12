import contextlib
from contextlib import contextmanager, suppress, nullcontext, closing, ExitStack

LOG = List[str]()


@contextmanager
def tag(name: str):
    LOG.append("<" + name + ">")
    yield name.upper()
    LOG.append("</" + name + ">")


class Res:
    n: int
    log: List[str]

    def __init__(self, n: int, log: List[str]):
        self.n = n
        self.log = log

    def close(self):
        self.log.append(f"close{self.n}")

    def __enter__(self):
        self.log.append(f"enter{self.n}")
        return self.n

    def __exit__(self, t, v, tb):
        self.log.append(f"exit{self.n}")
        return False


@test
def test_contextmanager():
    LOG.clear()
    with tag("a") as v:
        LOG.append(v)
    assert LOG == ["<a>", "A", "</a>"]
    LOG.clear()
    with tag("x"):
        with tag("y"):
            LOG.append("body")
    assert LOG == ["<x>", "<y>", "body", "</y>", "</x>"]
test_contextmanager()


@test
def test_suppress():
    hit = False
    with suppress(KeyError):
        d = {1: 2}
        d[3]
        hit = True
    assert not hit
    ok = False
    try:
        with suppress(KeyError):
            raise ValueError("v")
    except ValueError:
        ok = True
    assert ok
test_suppress()


@test
def test_nullcontext_closing():
    with nullcontext(5) as x:
        assert x == 5
    with nullcontext() as y:
        assert y is None
    log = List[str]()
    with closing(Res(1, log)) as r:
        assert r.n == 1
    assert log == ["close1"]
test_nullcontext_closing()


@test
def test_exitstack():
    log = List[str]()
    with ExitStack() as st:
        a = st.enter_context(Res(1, log))
        b = st.enter_context(Res(2, log))
        st.callback(lambda: log.append("cb"))
        assert a == 1 and b == 2
    assert log == ["enter1", "enter2", "cb", "exit2", "exit1"]
    log.clear()
    st2 = ExitStack()
    st2.enter_context(Res(3, log))
    moved = st2.pop_all()
    st2.close()
    assert log == ["enter3"]
    moved.close()
    assert log == ["enter3", "exit3"]
test_exitstack()


@contextmanager
def catcher(log: List[str]):
    try:
        yield 1
    except ValueError as e:
        log.append("caught " + str(e))
    finally:
        log.append("finally")


@contextmanager
def passthrough(log: List[str]):
    try:
        yield 1
    finally:
        log.append("cleanup")


@contextmanager
def stubborn():
    try:
        yield 1
    except ValueError:
        yield 2


@contextmanager
def translator():
    try:
        yield 1
    except ValueError as e:
        raise KeyError("translated " + str(e))


@test
def test_contextmanager_throw():
    log = List[str]()
    with catcher(log):
        raise ValueError("boom")
    assert log == ["caught boom", "finally"]
    log.clear()
    with catcher(log):
        log.append("body")
    assert log == ["body", "finally"]
    log.clear()
    ok = False
    try:
        with passthrough(log):
            raise ValueError("v")
    except ValueError as e:
        ok = str(e) == "v"
    assert ok and log == ["cleanup"]
    msg = ""
    try:
        with stubborn():
            raise ValueError("x")
    except RuntimeError as e:
        msg = str(e)
    assert msg == "generator didn't stop after throw()"
    msg = ""
    try:
        with translator():
            raise ValueError("y")
    except KeyError as e:
        msg = str(e)
    assert "translated y" in msg
test_contextmanager_throw()


@test
def test_generator_throw_close():
    def g(log: List[str]):
        try:
            yield 1
            yield 2
        except ValueError:
            log.append("ve")
            yield 10
        finally:
            log.append("fin")
    log = List[str]()
    it = g(log)
    assert next(it) == 1
    assert it.throw(ValueError("e")) == 10
    it.close()
    assert log == ["ve", "fin"]
    it2 = g(log)
    next(it2)
    hit = False
    try:
        it2.throw(KeyError("k"))
    except KeyError:
        hit = True
    assert hit
test_generator_throw_close()
