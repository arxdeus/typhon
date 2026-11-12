import os
import tempfile


@test
def test_dirs_and_files():
    d = tempfile.gettempdir()
    assert os.path.isdir(d) and os.path.isabs(d)
    assert tempfile.gettempprefix() == "tmp"

    fd, path = tempfile.mkstemp(suffix=".txt", prefix="tst")
    assert os.path.basename(path).startswith("tst") and path.endswith(".txt")
    assert os.path.dirname(path) == d
    os.write(fd, b"hello")
    os.close(fd)
    with open(path) as f:
        assert f.read() == "hello"
    assert os.stat(path).st_mode & 0o777 == 0o600
    os.remove(path)

    td = tempfile.mkdtemp(prefix="dd")
    assert os.path.isdir(td) and os.path.basename(td).startswith("dd")
    assert os.stat(td).st_mode & 0o777 == 0o700
    fd2, p2 = tempfile.mkstemp(dir=td)
    os.close(fd2)
    assert os.path.dirname(p2) == td
    os.remove(p2)
    os.rmdir(td)
    n1 = tempfile.mkstemp()
    n2 = tempfile.mkstemp()
    assert n1[1] != n2[1]
    os.close(n1[0]); os.close(n2[0]); os.remove(n1[1]); os.remove(n2[1])
test_dirs_and_files()


@test
def test_context_managers():
    keep = ""
    with tempfile.TemporaryDirectory(prefix="ctx") as name:
        keep = name
        assert os.path.isdir(name)
        os.makedirs(os.path.join(name, "a", "b"))
        with open(os.path.join(name, "a", "b", "f.txt"), "w") as f:
            f.write("x")
    assert not os.path.exists(keep)

    td = tempfile.TemporaryDirectory()
    assert os.path.isdir(td.name)
    td.cleanup()
    assert not os.path.exists(td.name)

    fname = ""
    with tempfile.NamedTemporaryFile("w+", suffix=".log") as tf:
        fname = tf.name
        assert fname.endswith(".log") and os.path.exists(fname)
        tf.write("line1\nline2\n")
        tf.flush()
        tf.seek(0)
        assert tf.read() == "line1\nline2\n"
    assert not os.path.exists(fname)

    with tempfile.NamedTemporaryFile(delete=False) as bf:
        fname = bf.name
        bf.write(b"\x00\x01")
    assert os.path.exists(fname)
    with open(fname, "rb") as f:
        assert f.read() == b"\x00\x01"
    os.remove(fname)
test_context_managers()
