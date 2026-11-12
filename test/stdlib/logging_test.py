import logging
import io
import os


def _capture(name: str, fmt: str = "%(levelname)s:%(name)s:%(message)s"):
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(logging.Formatter(fmt))
    lg = logging.getLogger(name)
    lg.addHandler(h)
    lg.propagate = False
    return lg, buf


@test
def test_levels_and_format():
    lg, buf = _capture("t1")
    lg.setLevel(logging.INFO)
    lg.debug("no")
    lg.info("hello %s", "world")
    lg.warning("n=%d", 5)
    lg.error("plain")
    lg.critical("%(a)s!", {"a": "dict"})
    assert buf.getvalue() == "INFO:t1:hello world\nWARNING:t1:n=5\nERROR:t1:plain\nCRITICAL:t1:dict!\n"
    assert lg.isEnabledFor(logging.INFO)
    assert not lg.isEnabledFor(logging.DEBUG)
    lg.setLevel("ERROR")
    assert lg.level == 40
    assert logging.getLevelName(30) == "WARNING"
    assert logging.getLevelName(15) == "Level 15"
test_levels_and_format()


@test
def test_hierarchy():
    parent, buf = _capture("pkg", "%(name)s|%(levelname)-8s|%(message)s")
    child = logging.getLogger("pkg.mod.leaf")
    assert child.parent is parent
    mid = logging.getLogger("pkg.mod")
    assert child.parent is mid and mid.parent is parent
    assert parent.parent is logging.getLogger()
    parent.setLevel(logging.DEBUG)
    assert child.getEffectiveLevel() == logging.DEBUG
    child.debug("deep")
    assert buf.getvalue() == "pkg.mod.leaf|DEBUG   |deep\n"
    assert logging.getLogger("pkg") is parent
    assert parent.getChild("mod") is mid
test_hierarchy()


@test
def test_filters_and_formatting():
    lg, buf = _capture("filt", "[%(levelno)03d] %(message)s %% %(name)10s")
    lg.setLevel(logging.DEBUG)
    lg.addFilter(logging.Filter("other"))
    lg.info("dropped")
    assert buf.getvalue() == ""
    lg.filters.clear()
    lg.info("kept")
    assert buf.getvalue() == "[020] kept %       filt\n"
    f = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y")
    r = logging.LogRecord("x", logging.INFO, "p", 1, "%s", ("m",), None)
    s = f.format(r)
    assert len(s) == 6 and s.endswith(" m")
    f2 = logging.Formatter("%(asctime)s|%(message)s")
    s2 = f2.format(r)
    # YYYY-MM-DD HH:MM:SS,mmm
    assert len(s2) == 25 and s2[4] == "-" and s2[19] == "," and s2.endswith("|m")
test_filters_and_formatting()


@test
def test_file_handler():
    path = "build/logging-test-" + str(os.getpid()) + ".log"
    os.makedirs("build", exist_ok=True)
    fh = logging.FileHandler(path, mode="w")
    fh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    lg = logging.getLogger("filelog")
    lg.propagate = False
    lg.addHandler(fh)
    lg.warning("to file")
    lg.info("skipped")
    fh.close()
    with open(path) as f:
        assert f.read() == "WARNING to file\n"
    os.remove(path)
test_file_handler()


@test
def test_basic_config():
    buf = io.StringIO()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s", stream=buf, force=True)
    logging.info("a %s", "b")
    logging.debug("hidden")
    logging.getLogger("x.y").warning("w")
    assert buf.getvalue() == "INFO - a b\nWARNING - w\n"
test_basic_config()
