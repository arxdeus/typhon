import struct

@test
def test_struct():
    assert struct.calcsize("<bhi") == 7
    assert struct.calcsize("@bhi") == 8
    assert struct.calcsize("@bq") == 16
    assert struct.calcsize("!2sH") == 4
    assert struct.pack("<hI", -2, 7) == b"\xfe\xff\x07\x00\x00\x00"
    assert struct.pack(">Hq", 258, -1) == b"\x01\x02" + b"\xff" * 8
    assert struct.pack("<d", 1.5) == b"\x00\x00\x00\x00\x00\x00\xf8?"
    assert struct.pack(">f", 1.5) == b"?\xc0\x00\x00"
    assert struct.pack("<3sc?x", b"ab", b"z", True) == b"ab\x00z\x01\x00"
    assert struct.pack("<2i", 1, 2) == b"\x01\x00\x00\x00\x02\x00\x00\x00"
    assert struct.unpack_ints("<hI", b"\xfe\xff\x07\x00\x00\x00") == [-2, 7]
    assert struct.unpack_ints(">Hq", b"\x01\x02" + b"\xff" * 8) == [258, -1]
    assert struct.unpack_floats("<d", b"\x00\x00\x00\x00\x00\x00\xf8?") == [1.5]
    assert struct.unpack_floats(">f", b"?\xc0\x00\x00") == [1.5]
    r = struct.unpack("<3sB", b"abc\x09")
    assert len(r) == 2
    b = r[0]
    assert isinstance(b, bytes) and b == b"abc"
    assert struct.unpack_from("<B", b"\x00\x05", 1)[0] == 5
    try:
        struct.pack("<B", 256)
        assert False
    except struct.error:
        pass
    try:
        struct.unpack("<i", b"\x00")
        assert False
    except struct.error:
        pass
test_struct()
