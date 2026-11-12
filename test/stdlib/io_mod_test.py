import io


@test
def test_stringio():
    s = io.StringIO()
    assert s.write("hello\n") == 6
    print("world", 42, sep="-", file=s)
    assert s.getvalue() == "hello\nworld-42\n"
    s.seek(0)
    assert s.readline() == "hello\n"
    assert s.read() == "world-42\n"
    assert s.read() == ""
    s.seek(0)
    assert s.readlines() == ["hello\n", "world-42\n"]
    s.seek(0)
    assert [l for l in s] == ["hello\n", "world-42\n"]
    s.seek(2)
    s.write("XY")
    assert s.getvalue() == "heXYo\nworld-42\n"
    assert s.tell() == 4
    s.truncate(5)
    assert s.getvalue() == "heXYo"
    s.seek(0, io.SEEK_END)
    assert s.tell() == 5
    t = io.StringIO("a\nb")
    assert t.read(1) == "a"
    assert t.readline() == "\n"
    assert t.readline() == "b"
    with io.StringIO("x") as u:
        assert u.read() == "x"
    assert u.closed
    try:
        u.read()
        assert False
    except ValueError as e:
        assert str(e) == "I/O operation on closed file."
test_stringio()


@test
def test_bytesio():
    b = io.BytesIO(b"abc\ndef")
    assert b.readline() == b"abc\n"
    assert b.read(2) == b"de"
    assert b.read() == b"f"
    b.seek(-2, io.SEEK_END)
    assert b.read() == b"ef"
    b.write(b"gh")
    assert b.getvalue() == b"abc\ndefgh"
    b.seek(0)
    b.write(b"XY")
    assert b.getvalue() == b"XYc\ndefgh"
    b.truncate(3)
    assert b.getvalue() == b"XYc"
    assert io.BytesIO().getvalue() == b""
    assert io.SEEK_SET == 0 and io.SEEK_CUR == 1 and io.SEEK_END == 2
test_bytesio()
